#!/usr/bin/env python
"""
M33 -- Pitcher Counting Stats for PlayerProjection (v33)

Adds pitcher counting stat columns required by waiver/draft engines to
player_projections table:
  - w (wins)
  - l (losses)
  - hr_pit (home runs allowed as pitcher)
  - k_pit (strikeouts counting stat for pitchers)
  - qs (quality starts)
  - nsv (net saves = saves - blown saves)

Design notes
------------
The keeper_engine and draft_engine expect these pitcher counting stats
(p.w, p.l, p.hr_pit, p.k_pit, p.qs, p.nsv) but they were missing from
the PlayerProjection schema. This migration adds them so CSV ingestion
can populate real Steamer projection data.

Idempotent: uses ADD COLUMN IF NOT EXISTS. Safe to run on any environment.

Usage
-----
    railway run python scripts/migrate_v33_pitcher_counting_stats.py
    python scripts/migrate_v33_pitcher_counting_stats.py --dry-run  # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS w INTEGER DEFAULT 0;

ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS l INTEGER DEFAULT 0;

ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS hr_pit INTEGER DEFAULT 0;

ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS k_pit INTEGER DEFAULT 0;

ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS qs INTEGER DEFAULT 0;

ALTER TABLE player_projections
    ADD COLUMN IF NOT EXISTS nsv INTEGER DEFAULT 0;

COMMENT ON COLUMN player_projections.w IS
    'M33 Projected wins for pitchers. Populated by Steamer CSV ingestion.';

COMMENT ON COLUMN player_projections.l IS
    'M33 Projected losses for pitchers. Populated by Steamer CSV ingestion.';

COMMENT ON COLUMN player_projections.hr_pit IS
    'M33 Projected home runs allowed for pitchers. Populated by Steamer CSV ingestion.';

COMMENT ON COLUMN player_projections.k_pit IS
    'M33 Projected strikeouts counting stat for pitchers. Populated by Steamer CSV ingestion.';

COMMENT ON COLUMN player_projections.qs IS
    'M33 Projected quality starts for pitchers. Estimated as ~50% of GS in CSV ingestion.';

COMMENT ON COLUMN player_projections.nsv IS
    'M33 Projected net saves (saves - blown saves). Populated by Steamer CSV ingestion.';
"""

DOWNGRADE_SQL = """
ALTER TABLE player_projections DROP COLUMN IF EXISTS nsv;
ALTER TABLE player_projections DROP COLUMN IF EXISTS qs;
ALTER TABLE player_projections DROP COLUMN IF EXISTS k_pit;
ALTER TABLE player_projections DROP COLUMN IF EXISTS hr_pit;
ALTER TABLE player_projections DROP COLUMN IF EXISTS l;
ALTER TABLE player_projections DROP COLUMN IF EXISTS w;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Adding pitcher counting stats to player_projections (v33) ===")

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
        print("SUCCESS: Pitcher counting stats columns added")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing pitcher counting stats from player_projections (v33) ===")

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
        print("SUCCESS: Pitcher counting stats columns removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v33 - Pitcher Counting Stats")
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
