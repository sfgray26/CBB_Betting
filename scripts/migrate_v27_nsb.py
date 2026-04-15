#!/usr/bin/env python
"""
P27 -- Net Stolen Bases (NSB) columns (v27)

Adds the columns required to score the H2H-One-Win category NSB (SB - CS):
  - player_rolling_stats.w_caught_stealing    FLOAT  (decay-weighted CS)
  - player_rolling_stats.w_net_stolen_bases   FLOAT  (w_stolen_bases - w_caught_stealing)
  - player_scores.z_nsb                       FLOAT  (league Z of w_net_stolen_bases)

Design notes
------------
NSB is the canonical H2H One Win 5x5 basestealing category. Previously the
scoring engine used z_sb (based on w_stolen_bases), which ignores the negative
value of caught stealings. NSB correlates >0.95 with SB for most players (CS
events are rare), so z_nsb replaces z_sb in HITTER_CATEGORIES for composite
scoring. z_sb is kept for backward compatibility with explainability narratives
and UAT checks but is excluded from composite_z to avoid double-counting.

Idempotent: uses ADD COLUMN IF NOT EXISTS. Safe to run on any environment.
Nullable additive columns only -- zero downtime, no backfill required
(next scheduled rolling_windows + player_scores job populates them).

Usage
-----
    python scripts/migrate_v27_nsb.py              # run upgrade
    python scripts/migrate_v27_nsb.py --downgrade  # run downgrade
    python scripts/migrate_v27_nsb.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
ALTER TABLE player_rolling_stats
    ADD COLUMN IF NOT EXISTS w_caught_stealing DOUBLE PRECISION;

ALTER TABLE player_rolling_stats
    ADD COLUMN IF NOT EXISTS w_net_stolen_bases DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_nsb DOUBLE PRECISION;

COMMENT ON COLUMN player_rolling_stats.w_caught_stealing IS
    'P27 Decay-weighted caught stealing over the rolling window. '
    'Source: mlb_player_stats.caught_stealing (BDL). '
    'Null for pure pitchers (no at-bats in window).';

COMMENT ON COLUMN player_rolling_stats.w_net_stolen_bases IS
    'P27 w_stolen_bases - w_caught_stealing. '
    'Drives z_nsb in the H2H One Win NSB scoring category.';

COMMENT ON COLUMN player_scores.z_nsb IS
    'P27 League Z-score of w_net_stolen_bases (SB - CS). '
    'Replaces z_sb in HITTER_CATEGORIES composite_z. '
    'z_sb is still populated for backward compat but excluded from composite.';
"""

DOWNGRADE_SQL = """
ALTER TABLE player_scores            DROP COLUMN IF EXISTS z_nsb;
ALTER TABLE player_rolling_stats     DROP COLUMN IF EXISTS w_net_stolen_bases;
ALTER TABLE player_rolling_stats     DROP COLUMN IF EXISTS w_caught_stealing;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Adding NSB columns (v27) ===")

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
        print("SUCCESS: NSB columns added")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing NSB columns (v27) ===")

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
        print("SUCCESS: NSB columns removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v27 - NSB columns")
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
