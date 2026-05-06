#!/usr/bin/env python
"""
Deduplicate player_id_mapping rows by yahoo_id then normalized_name.

This is a data-only migration with no downgrade path. It dedupes ~1,513 duplicate
rows identified in K-NEXT-2 analysis. The _pim_bdl_id_uc UNIQUE constraint on
bdl_id is already in place (migrate_v28_player_id_mapping_fix.py) and acts as
a backstop: if the dedupe logic leaves duplicate bdl_ids, PostgreSQL will raise
UniqueViolation and the transaction rolls back automatically.

Algorithm:
  1. Snapshot: count rows, distinct names, orphans before
  2. Dedupe by yahoo_id: keep richest row per yahoo_id, COALESCE-merge bdl_id/mlbam_id/yahoo_key
  3. Dedupe by normalized_name: skip groups with distinct non-null bdl_ids (real name conflicts)
  4. ANALYZE player_id_mapping
  5. Final snapshot: print summary

Safety:
  - merged_count > 2000 raises an error before commit
  - --dry-run prints planned SQL without executing DML
  - Idempotent: second run prints merged_count=0

Usage:
    python scripts/migration_dedupe_player_id_mapping.py           # run migration
    python scripts/migration_dedupe_player_id_mapping.py --dry-run # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SAFETY_CEILING = 2000

# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------

SQL_SNAPSHOT = """
SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT normalized_name) AS distinct_names,
    COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL AND bdl_id IS NULL) AS orphan_count
FROM player_id_mapping;
"""

# Step 2a: COALESCE-merge richest values into the keeper row (grouped by yahoo_id)
SQL_YAHOO_MERGE = """
UPDATE player_id_mapping AS keeper
SET
    bdl_id   = COALESCE(keeper.bdl_id,   losers.best_bdl_id),
    mlbam_id = COALESCE(keeper.mlbam_id, losers.best_mlbam_id),
    yahoo_key = COALESCE(keeper.yahoo_key, losers.best_yahoo_key)
FROM (
    SELECT
        first_value(id) OVER w AS keeper_id,
        first_value(bdl_id) OVER (
            PARTITION BY yahoo_id
            ORDER BY (bdl_id IS NOT NULL) DESC, id ASC
        ) AS best_bdl_id,
        first_value(mlbam_id) OVER (
            PARTITION BY yahoo_id
            ORDER BY (mlbam_id IS NOT NULL) DESC, id ASC
        ) AS best_mlbam_id,
        first_value(yahoo_key) OVER (
            PARTITION BY yahoo_id
            ORDER BY (yahoo_key IS NOT NULL) DESC, id ASC
        ) AS best_yahoo_key,
        ROW_NUMBER() OVER w AS rn
    FROM player_id_mapping
    WHERE yahoo_id IS NOT NULL
    WINDOW w AS (
        PARTITION BY yahoo_id
        ORDER BY
            (bdl_id IS NOT NULL) DESC,
            (mlbam_id IS NOT NULL) DESC,
            (yahoo_key IS NOT NULL) DESC,
            updated_at DESC NULLS LAST,
            id ASC
    )
) losers
WHERE keeper.id = losers.keeper_id
  AND losers.rn = 1;
"""

# Step 2b: Delete non-keeper rows grouped by yahoo_id
SQL_YAHOO_DELETE = """
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY yahoo_id
               ORDER BY
                   (bdl_id IS NOT NULL) DESC,
                   (mlbam_id IS NOT NULL) DESC,
                   (yahoo_key IS NOT NULL) DESC,
                   updated_at DESC NULLS LAST,
                   id ASC
           ) AS rn
    FROM player_id_mapping
    WHERE yahoo_id IS NOT NULL
)
DELETE FROM player_id_mapping
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
"""

# Step 3: Find normalized_name groups that are safe to merge
# (no distinct non-null bdl_ids -- i.e., all share the same bdl_id or bdl_id is null)
SQL_NAME_SAFE_GROUPS = """
SELECT normalized_name
FROM player_id_mapping
WHERE normalized_name IS NOT NULL
GROUP BY normalized_name
HAVING COUNT(*) > 1
  AND COUNT(DISTINCT bdl_id) FILTER (WHERE bdl_id IS NOT NULL) <= 1;
"""

# Count of skipped groups (real name conflicts with distinct bdl_ids)
SQL_NAME_CONFLICT_COUNT = """
SELECT COUNT(*) AS conflict_count
FROM (
    SELECT normalized_name
    FROM player_id_mapping
    WHERE normalized_name IS NOT NULL
    GROUP BY normalized_name
    HAVING COUNT(*) > 1
      AND COUNT(DISTINCT bdl_id) FILTER (WHERE bdl_id IS NOT NULL) > 1
) sub;
"""

# Step 3a: COALESCE-merge richest values into the keeper row (grouped by normalized_name)
# Parameterized via Python string format -- called per batch after filtering safe groups
SQL_NAME_MERGE_TEMPLATE = """
UPDATE player_id_mapping AS keeper
SET
    bdl_id    = COALESCE(keeper.bdl_id,    losers.best_bdl_id),
    mlbam_id  = COALESCE(keeper.mlbam_id,  losers.best_mlbam_id),
    yahoo_key = COALESCE(keeper.yahoo_key, losers.best_yahoo_key)
FROM (
    SELECT
        first_value(id) OVER w AS keeper_id,
        first_value(bdl_id) OVER (
            PARTITION BY normalized_name
            ORDER BY (bdl_id IS NOT NULL) DESC, id ASC
        ) AS best_bdl_id,
        first_value(mlbam_id) OVER (
            PARTITION BY normalized_name
            ORDER BY (mlbam_id IS NOT NULL) DESC, id ASC
        ) AS best_mlbam_id,
        first_value(yahoo_key) OVER (
            PARTITION BY normalized_name
            ORDER BY (yahoo_key IS NOT NULL) DESC, id ASC
        ) AS best_yahoo_key,
        ROW_NUMBER() OVER w AS rn
    FROM player_id_mapping
    WHERE normalized_name IN :safe_names
    WINDOW w AS (
        PARTITION BY normalized_name
        ORDER BY
            (bdl_id IS NOT NULL) DESC,
            (mlbam_id IS NOT NULL) DESC,
            (yahoo_key IS NOT NULL) DESC,
            updated_at DESC NULLS LAST,
            id ASC
    )
) losers
WHERE keeper.id = losers.keeper_id
  AND losers.rn = 1;
"""

# Step 3b: Delete non-keeper rows for safe normalized_name groups
SQL_NAME_DELETE_TEMPLATE = """
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY normalized_name
               ORDER BY
                   (bdl_id IS NOT NULL) DESC,
                   (mlbam_id IS NOT NULL) DESC,
                   (yahoo_key IS NOT NULL) DESC,
                   updated_at DESC NULLS LAST,
                   id ASC
           ) AS rn
    FROM player_id_mapping
    WHERE normalized_name IN :safe_names
)
DELETE FROM player_id_mapping
WHERE id IN (SELECT id FROM ranked WHERE rn > 1);
"""

SQL_ANALYZE = "ANALYZE player_id_mapping;"

SQL_COUNT = "SELECT COUNT(*) FROM player_id_mapping;"
SQL_ORPHAN_COUNT = """
SELECT COUNT(*) FROM player_id_mapping
WHERE yahoo_id IS NOT NULL AND bdl_id IS NULL;
"""


# ---------------------------------------------------------------------------
# Main run function
# ---------------------------------------------------------------------------

def run(engine, dry_run=False):
    print("=== DEDUPE: player_id_mapping by yahoo_id then normalized_name ===")

    # ------------------------------------------------------------------
    # Step 1 -- Snapshot (outside transaction, read-only)
    # ------------------------------------------------------------------
    with engine.connect() as conn:
        row = conn.execute(text(SQL_SNAPSHOT)).fetchone()
    rows_before = int(row.total_rows)
    distinct_names_before = int(row.distinct_names)
    orphans_before = int(row.orphan_count)
    print(f"rows_before={rows_before}, distinct_names={distinct_names_before}, orphans={orphans_before}")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following DML ---")
        print("\n[Step 2a] COALESCE-merge by yahoo_id:")
        print(SQL_YAHOO_MERGE.strip())
        print("\n[Step 2b] Delete losers by yahoo_id:")
        print(SQL_YAHOO_DELETE.strip())
        print("\n[Step 3] Find safe normalized_name groups (no distinct bdl_id conflicts)")
        print(SQL_NAME_SAFE_GROUPS.strip())
        print("\n[Step 3a] COALESCE-merge by normalized_name (safe groups only):")
        print(SQL_NAME_MERGE_TEMPLATE.strip())
        print("\n[Step 3b] Delete losers by normalized_name (safe groups only):")
        print(SQL_NAME_DELETE_TEMPLATE.strip())
        print("\n[Step 4] ANALYZE player_id_mapping")
        print("--- END DRY RUN ---\n")
        return

    # ------------------------------------------------------------------
    # Steps 2-4 -- atomic transaction (all-or-nothing)
    # ------------------------------------------------------------------
    with engine.begin() as conn:

        # Step 2a: COALESCE-merge keeper rows by yahoo_id
        print("[Step 2a] Merging fields into yahoo_id keepers...")
        conn.execute(text(SQL_YAHOO_MERGE))
        print("  OK: yahoo_id COALESCE merge complete")

        # Step 2b: Delete yahoo_id losers
        print("[Step 2b] Deleting yahoo_id duplicates...")
        result = conn.execute(text(SQL_YAHOO_DELETE))
        yahoo_deleted = result.rowcount
        print(f"  OK: deleted {yahoo_deleted} yahoo_id duplicate rows")

        # Step 3: Find safe normalized_name groups (no distinct bdl_id conflicts)
        print("[Step 3] Finding safe normalized_name groups...")
        safe_rows = conn.execute(text(SQL_NAME_SAFE_GROUPS)).fetchall()
        safe_names = [r[0] for r in safe_rows]
        print(f"  safe_to_merge groups: {len(safe_names)}")

        # Count skipped conflict groups
        name_conflicts_skipped = conn.execute(text(SQL_NAME_CONFLICT_COUNT)).scalar()
        if name_conflicts_skipped > 0:
            print(f"  name_conflicts_skipped: {name_conflicts_skipped} groups (distinct bdl_ids -- real different players)")

        name_deleted = 0
        if safe_names:
            # Step 3a: COALESCE-merge by normalized_name (safe groups only)
            print("[Step 3a] Merging fields into normalized_name keepers...")
            conn.execute(
                text(SQL_NAME_MERGE_TEMPLATE),
                {"safe_names": tuple(safe_names)},
            )
            print("  OK: normalized_name COALESCE merge complete")

            # Step 3b: Delete normalized_name losers
            print("[Step 3b] Deleting normalized_name duplicates...")
            result = conn.execute(
                text(SQL_NAME_DELETE_TEMPLATE),
                {"safe_names": tuple(safe_names)},
            )
            name_deleted = result.rowcount
            print(f"  OK: deleted {name_deleted} normalized_name duplicate rows")
        else:
            print("[Step 3a/3b] No safe normalized_name groups to merge -- skipping")

        total_deleted = yahoo_deleted + name_deleted

        # Safety ceiling check -- before commit
        if total_deleted > SAFETY_CEILING:
            raise RuntimeError(
                f"ERROR: merged_count={total_deleted} exceeds safety ceiling of {SAFETY_CEILING}. "
                f"Aborting. Re-run with Architect review."
            )

    # Transaction committed successfully
    print("  COMMIT: transaction committed")

    # Step 4: ANALYZE (outside transaction — valid on all PG versions)
    print("\nStep 4: Running ANALYZE...")
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as analyze_conn:
        analyze_conn.execute(text(SQL_ANALYZE))
    print("  ANALYZE done")

    # ------------------------------------------------------------------
    # Step 5 -- Final snapshot
    # ------------------------------------------------------------------
    with engine.connect() as conn:
        rows_after = int(conn.execute(text(SQL_COUNT)).scalar())
        orphans_remaining = int(conn.execute(text(SQL_ORPHAN_COUNT)).scalar())

    merged_count = rows_before - rows_after

    print("\n=== SUMMARY ===")
    print(f"rows_before={rows_before}")
    print(f"rows_after={rows_after}")
    print(f"merged_count={merged_count}")
    print(f"name_conflicts_skipped={name_conflicts_skipped}")
    print(f"orphans_remaining={orphans_remaining}  (yahoo_id non-null but bdl_id still null)")
    print("SUCCESS")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate player_id_mapping by yahoo_id then normalized_name (data-only, no downgrade)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned SQL without executing DML",
    )
    args = parser.parse_args()

    from backend.models import DATABASE_URL

    engine = create_engine(DATABASE_URL)
    run(engine, dry_run=args.dry_run)
