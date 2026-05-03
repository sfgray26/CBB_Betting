#!/usr/bin/env python
"""
M34 -- player_type discriminator for player_projections (v34)

Problem solved
--------------
player_projections had no player_type column. Every row had numeric defaults
for BOTH batting and pitching stat columns, regardless of position:
  - Batters got:  era=4.00, whip=1.30, k_per_nine=8.5  (fake pitcher stats)
  - Pitchers got: hr=15, r=65, rbi=65, sb=5            (fake batter stats)

This poisoned z-score rankings: pitchers appeared in batter category rankings
and batters appeared in pitcher category rankings.

What this migration does
------------------------
1. Adds player_type VARCHAR(10) column.
2. Classifies all existing rows from their positions JSON:
   - Contains SP/RP/P → 'pitcher'
   - Everything else  → 'hitter'
3. Nulls out fake pitching stats on hitters (era, whip, k_per_nine, bb_per_nine).
4. Nulls out fake batting counting stats on pitchers (hr, r, rbi, sb).

Idempotent: safe to re-run. Uses IF NOT EXISTS and WHERE guards.

Usage
-----
    railway run python scripts/migrate_v34_player_type.py
    python scripts/migrate_v34_player_type.py --dry-run
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- Step 1: Add player_type column
ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS player_type VARCHAR(10);

COMMENT ON COLUMN player_projections.player_type IS
    'M34: hitter | pitcher. Primary classification signal for cat_scores and z-score engines.';

-- Step 2: Classify from positions JSON
--   SP, RP, or P (as a standalone element) → pitcher
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

-- Everything unclassified → hitter (batters, DH, util, unknown)
UPDATE player_projections
SET player_type = 'hitter'
WHERE player_type IS NULL;

-- Step 3: Null out fake pitching stats on hitters
UPDATE player_projections
SET era        = NULL,
    whip       = NULL,
    k_per_nine = NULL,
    bb_per_nine = NULL
WHERE player_type = 'hitter';

-- Step 4: Null out fake batting counting stats on pitchers
UPDATE player_projections
SET hr  = NULL,
    r   = NULL,
    rbi = NULL,
    sb  = NULL
WHERE player_type = 'pitcher';
"""

DOWNGRADE_SQL = """
ALTER TABLE player_projections DROP COLUMN IF EXISTS player_type;
"""

VERIFY_SQL = """
SELECT
    player_type,
    COUNT(*)                                          AS total,
    COUNT(*) FILTER (WHERE era   IS NOT NULL)         AS era_not_null,
    COUNT(*) FILTER (WHERE whip  IS NOT NULL)         AS whip_not_null,
    COUNT(*) FILTER (WHERE hr    IS NOT NULL)         AS hr_not_null,
    COUNT(*) FILTER (WHERE rbi   IS NOT NULL)         AS rbi_not_null
FROM player_projections
GROUP BY player_type
ORDER BY player_type;
"""


def run(dry_run: bool = False) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    print("=== M34: player_type discriminator migration ===")

    if dry_run:
        print("[DRY RUN] Would execute:\n")
        print(UPGRADE_SQL)
        return

    engine = create_engine(db_url)
    with engine.begin() as conn:
        for statement in UPGRADE_SQL.strip().split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            # Strip leading comment lines to detect whether real SQL remains
            sql_lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            if not any(l.strip() for l in sql_lines):
                continue  # pure-comment chunk, skip
            print(f"  Executing: {stmt[:80]}...")
            conn.execute(text(stmt))

    print("\n=== Verification ===")
    with engine.connect() as conn:
        rows = conn.execute(text(VERIFY_SQL)).fetchall()
        print(f"{'player_type':<12} {'total':>7} {'era≠NULL':>9} {'whip≠NULL':>10} {'hr≠NULL':>8} {'rbi≠NULL':>9}")
        print("-" * 60)
        for row in rows:
            print(f"{str(row[0]):<12} {row[1]:>7} {row[2]:>9} {row[3]:>10} {row[4]:>8} {row[5]:>9}")

    # Correctness checks
    with engine.connect() as conn:
        hitter_era_leak = conn.execute(
            text("SELECT COUNT(*) FROM player_projections WHERE player_type='hitter' AND era IS NOT NULL")
        ).scalar()
        pitcher_hr_leak = conn.execute(
            text("SELECT COUNT(*) FROM player_projections WHERE player_type='pitcher' AND hr IS NOT NULL")
        ).scalar()
        unclassified = conn.execute(
            text("SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL")
        ).scalar()

    ok = True
    if hitter_era_leak:
        print(f"FAIL: {hitter_era_leak} hitters still have ERA != NULL")
        ok = False
    if pitcher_hr_leak:
        print(f"FAIL: {pitcher_hr_leak} pitchers still have HR != NULL")
        ok = False
    if unclassified:
        print(f"FAIL: {unclassified} rows still have player_type IS NULL")
        ok = False
    if ok:
        print("\nPASS: M34 migration complete — no data contamination detected.")
    else:
        print("\nM34 migration completed with warnings above.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="M34 player_type migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    parser.add_argument("--downgrade", action="store_true", help="Remove player_type column")
    args = parser.parse_args()

    if args.downgrade:
        db_url = os.getenv("DATABASE_URL")
        engine = create_engine(db_url)
        with engine.begin() as conn:
            conn.execute(text(DOWNGRADE_SQL.strip()))
        print("Downgrade complete.")
    else:
        run(dry_run=args.dry_run)
