#!/usr/bin/env python
"""
M35 -- player_type backfill for player_projections

Problem solved
--------------
71% of player_projections rows have player_type = NULL.
New players added after M34 migration were never classified.

This migration re-runs the M34 classification logic on NULL rows only:
  - Contains SP/RP/P → 'pitcher'
  - Everything else → 'hitter'

Idempotent: safe to re-run. Uses WHERE player_type IS NULL guard.

Usage
-----
    railway run python scripts/migrate_v35_player_type_backfill.py
    python scripts/migrate_v35_player_type_backfill.py --dry-run
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- Step 1: Classify pitchers from positions JSON
UPDATE player_projections
SET player_type = 'pitcher'
WHERE player_type IS NULL
  AND (
      positions::text LIKE '%"SP"%'
   OR positions::text LIKE '%"RP"%'
   OR positions::text LIKE '% "P" %'
   OR positions::text LIKE '%["P"]%'
   OR positions::text LIKE '%,"P"]%'
   OR positions::text LIKE '%["P",%'
  );

-- Step 2: Classify all remaining as hitters
UPDATE player_projections
SET player_type = 'hitter'
WHERE player_type IS NULL;
"""

VERIFY_SQL = """
SELECT
    player_type,
    COUNT(*)                                          AS total
FROM player_projections
GROUP BY player_type
ORDER BY player_type NULLS LAST;
"""


def run(dry_run: bool = False) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    print("=== M35: player_type backfill migration ===")

    if dry_run:
        print("[DRY RUN] Would execute:\n")
        print(UPGRADE_SQL)
        return

    engine = create_engine(db_url)

    # Acquire advisory lock 100_016
    print("Acquiring advisory lock 100_016...")
    with engine.begin() as conn:
        lock_result = conn.execute(
            text("SELECT pg_try_advisory_lock(100016)")
        ).scalar()

        if not lock_result:
            print("ERROR: Advisory lock 100_016 held by another process")
            sys.exit(1)

        print("Lock acquired. Executing migration...")

        # Execute UPDATE statements
        for statement in UPGRADE_SQL.strip().split(";"):
            stmt = statement.strip()
            if not stmt:
                continue

            # Strip leading comment lines
            sql_lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            if not any(l.strip() for l in sql_lines):
                continue

            print(f"  Executing: {stmt[:80]}...")
            result = conn.execute(text(stmt))
            print(f"    Rows affected: {result.rowcount}")

            # Release lock after each major step
            conn.execute(text("SELECT pg_advisory_unlock(100016)"))

            # Re-acquire for next step
            lock_result = conn.execute(
                text("SELECT pg_try_advisory_lock(100016)")
            ).scalar()
            if not lock_result:
                print("ERROR: Failed to re-acquire advisory lock 100_016")
                sys.exit(1)

    # Final lock release
    conn.execute(text("SELECT pg_advisory_unlock(100016)"))
    print("Lock released.")

    print("\n=== Verification ===")
    with engine.connect() as conn:
        rows = conn.execute(text(VERIFY_SQL)).fetchall()
        print(f"{'player_type':<12} {'total':>7}")
        print("-" * 20)
        for row in rows:
            player_type_label = row[0] if row[0] else "NULL"
            print(f"{player_type_label:<12} {row[1]:>7}")

    # Correctness checks
    with engine.connect() as conn:
        null_count = conn.execute(
            text("SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL")
        ).scalar()
        total_count = conn.execute(
            text("SELECT COUNT(*) FROM player_projections")
        ).scalar()

    if null_count:
        print(f"\nFAIL: {null_count} rows still have player_type IS NULL (out of {total_count} total)")
        sys.exit(1)
    else:
        print(f"\nPASS: M35 migration complete — all {total_count} rows classified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M35 player_type backfill")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    run(dry_run=args.dry_run)
