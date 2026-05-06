#!/usr/bin/env python3
"""
Backfill player_projections.player_type for rows where it is NULL (~71% of rows).

Inference rules (matches values written by ros_projection_refresh):
  - positions has SP/RP/P  AND batter pos  → 'both'
  - positions has SP/RP/P  (only)          → 'pitcher'
  - positions has batter pos (only)        → 'hitter'
  - positions NULL/empty, era IS NOT NULL  → 'pitcher'  (stat-based fallback)
  - else                                   → 'hitter'   (default fallback)

Idempotent: only touches rows WHERE player_type IS NULL.

Usage:
    python scripts/backfill_player_type.py           # run
    python scripts/backfill_player_type.py --dry-run # estimate counts, no writes
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    sys.exit(1)

from sqlalchemy import create_engine, text

engine = create_engine(DATABASE_URL)

# ------------------------------------------------------------------
# SQL fragments
# ------------------------------------------------------------------

# Shared CASE expression for inference
_CASE = """
    CASE
        WHEN positions::jsonb ?| array['SP','RP','P']
             AND positions::jsonb ?| array['C','1B','2B','3B','SS','OF','LF','CF','RF','Util','DH']
            THEN 'both'
        WHEN positions::jsonb ?| array['SP','RP','P']
            THEN 'pitcher'
        WHEN positions IS NOT NULL
             AND jsonb_array_length(positions::jsonb) > 0
            THEN 'hitter'
        WHEN era IS NOT NULL
            THEN 'pitcher'
        ELSE 'hitter'
    END
""".strip()

SQL_COUNT_NULL = "SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL;"

SQL_ESTIMATE = f"""
SELECT {_CASE} AS inferred_type, COUNT(*) AS cnt
FROM player_projections
WHERE player_type IS NULL
GROUP BY 1
ORDER BY 1;
"""

SQL_UPDATE = f"""
UPDATE player_projections
SET player_type = {_CASE}
WHERE player_type IS NULL;
"""

SQL_DISTRIBUTION = """
SELECT player_type, COUNT(*) AS cnt
FROM player_projections
GROUP BY player_type
ORDER BY player_type NULLS LAST;
"""


def main(dry_run: bool = False) -> int:
    print("=== BACKFILL: player_projections.player_type ===")

    with engine.connect() as conn:
        null_count = conn.execute(text(SQL_COUNT_NULL)).scalar()
    print(f"rows with NULL player_type: {null_count}")

    if null_count == 0:
        print("Nothing to do — all rows already have player_type set.")
        return 0

    print("\nInference estimate:")
    with engine.connect() as conn:
        for row in conn.execute(text(SQL_ESTIMATE)).fetchall():
            print(f"  {row[0]}: {row[1]}")

    if dry_run:
        print("\n--- DRY RUN: no writes performed ---")
        return 0

    print("\nRunning UPDATE...")
    with engine.begin() as conn:
        result = conn.execute(text(SQL_UPDATE))
        updated = result.rowcount
    print(f"  updated: {updated} rows")

    print("\nFinal distribution:")
    with engine.connect() as conn:
        rows = conn.execute(text(SQL_DISTRIBUTION)).fetchall()
    null_remaining = 0
    for row in rows:
        label = row[0] if row[0] is not None else "NULL"
        print(f"  {label}: {row[1]}")
        if row[0] is None:
            null_remaining = row[1]

    if null_remaining > 0:
        print(f"\nWARNING: {null_remaining} rows still have NULL player_type")
        return 1

    print("\nSUCCESS: no NULL player_type rows remaining")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill player_projections.player_type from positions array"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show inference estimates without writing to DB",
    )
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
