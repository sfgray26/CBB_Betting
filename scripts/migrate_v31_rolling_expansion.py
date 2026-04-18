#!/usr/bin/env python
"""
V31 -- Rolling Stats Expansion (R, TB, QS)

Phase 2 Workstream A1-A2: Adds columns required for 6 new Z-score categories:
  - player_rolling_stats.w_runs   FLOAT  (decay-weighted runs scored)
  - player_rolling_stats.w_tb     FLOAT  (decay-weighted total bases)
  - player_rolling_stats.w_qs     FLOAT  (decay-weighted quality starts)

Design notes
------------
w_runs: For the R (Runs) batting category. Source: mlb_player_stats.runs (BDL).
w_tb: For the TB (Total Bases) batting category. Computed as H + 2B*2 + 3B*3 + HR*4.
w_qs: For the QS (Quality Starts) pitching category. Derived per-game when IP≥6 AND ER≤3.

These are Phase 2 prerequisites. The Z-score columns (z_runs, z_tb, z_qs, z_h, z_k_b,
z_ops, z_k_p) will be added in a separate migration after scoring_engine.py is updated.

Idempotent: uses ADD COLUMN IF NOT EXISTS. Safe to run on any environment.
Nullable additive columns only -- zero downtime, no backfill required.

Usage
-----
    python scripts/migrate_v31_rolling_expansion.py              # run upgrade
    python scripts/migrate_v31_rolling_expansion.py --downgrade  # run downgrade
    python scripts/migrate_v31_rolling_expansion.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
ALTER TABLE player_rolling_stats
    ADD COLUMN IF NOT EXISTS w_runs DOUBLE PRECISION;

ALTER TABLE player_rolling_stats
    ADD COLUMN IF NOT EXISTS w_tb DOUBLE PRECISION;

ALTER TABLE player_rolling_stats
    ADD COLUMN IF NOT EXISTS w_qs DOUBLE PRECISION;

COMMENT ON COLUMN player_rolling_stats.w_runs IS
    'V31 Decay-weighted runs scored over the rolling window. '
    'Source: mlb_player_stats.runs (BDL). '
    'Drives z_runs for the R (Runs) batting category.';

COMMENT ON COLUMN player_rolling_stats.w_tb IS
    'V31 Decay-weighted total bases over the rolling window. '
    'Computed as singles + 2*doubles + 3*triples + 4*home_runs per game. '
    'Drives z_tb for the TB (Total Bases) batting category.';

COMMENT ON COLUMN player_rolling_stats.w_qs IS
    'V31 Decay-weighted quality starts over the rolling window. '
    'A quality start is IP ≥ 6.0 AND ER ≤ 3. '
    'Drives z_qs for the QS (Quality Starts) pitching category.';
"""

DOWNGRADE_SQL = """
ALTER TABLE player_rolling_stats     DROP COLUMN IF EXISTS w_qs;
ALTER TABLE player_rolling_stats     DROP COLUMN IF EXISTS w_tb;
ALTER TABLE player_rolling_stats     DROP COLUMN IF EXISTS w_runs;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Adding R/TB/QS columns (V31) ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in UPGRADE_SQL.split(";"):
            statement = statement.strip()
            if not statement or statement.startswith("--"):
                continue
            try:
                conn.execute(text(statement))
            except Exception as e:
                if "already exists" in str(e).lower():
                    print(f"  WARNING: Skipping (already exists): {str(e)[:100]}")
                else:
                    raise
        print("SUCCESS: R/TB/QS columns added")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing R/TB/QS columns (V31) ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(DOWNGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in DOWNGRADE_SQL.split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    print(f"  WARNING: {str(e)[:100]}")
        print("SUCCESS: R/TB/QS columns removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate V31 - R/TB/QS columns")
    parser.add_argument("--downgrade", action="store_true", help="Rollback migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    eng = create_engine(db_url)

    if args.downgrade:
        downgrade(eng, dry_run=args.dry_run)
    else:
        upgrade(eng, dry_run=args.dry_run)
