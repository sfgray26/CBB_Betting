#!/usr/bin/env python
"""
M3 Fix -- Add w_games column to player_rolling_stats (v24)

Adds a DOUBLE PRECISION column 'w_games' to player_rolling_stats.
This stores the sum of decay weights (lambda=0.95^days_back) used in
the rolling window computation.  Required because the simulation engine
divides decay-weighted stat sums by the decay-weighted game count (not
the raw game count) to derive per-game rates.

  Prior behaviour: rates used games_in_window (raw count), which over-
  estimated per-game rates for windows with many low-weight old games.

  New behaviour:   rates use w_games = sum(0.95^days_back), giving
  consistent decay-weighted per-game rates.

The column is nullable so existing rows (computed before this fix)
continue to work.  The simulation engine falls back to games_in_window
when w_games is NULL.

Usage:
    python scripts/migrate_v24_w_games.py              # run upgrade
    python scripts/migrate_v24_w_games.py --downgrade  # run downgrade
    python scripts/migrate_v24_w_games.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
ALTER TABLE player_rolling_stats
    ADD COLUMN IF NOT EXISTS w_games DOUBLE PRECISION;

COMMENT ON COLUMN player_rolling_stats.w_games IS
    'Sum of decay weights (0.95^days_back) for games in the window. Used by P16 simulation_engine to derive per-game rates from decay-weighted sums. NULL for legacy rows computed before M3 fix - simulation engine falls back to games_in_window.'
"""

DOWNGRADE_SQL = """
ALTER TABLE player_rolling_stats
    DROP COLUMN IF EXISTS w_games;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Adding w_games to player_rolling_stats ===")

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
                    print("  WARNING: Skipping (already exists): {}".format(str(e)[:100]))
                else:
                    raise
        print("SUCCESS: w_games column added to player_rolling_stats")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing w_games from player_rolling_stats ===")

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
                    print("  WARNING: {}".format(str(e)[:100]))
        print("SUCCESS: w_games column removed from player_rolling_stats")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v24 - Add w_games column")
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
