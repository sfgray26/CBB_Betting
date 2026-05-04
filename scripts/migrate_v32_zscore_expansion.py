#!/usr/bin/env python
"""
V32 -- Z-Score Expansion (Phase 2A)

Phase 2 Workstream A5-A7: Adds Z-score columns for 6 new categories to player_scores:
  - player_scores.z_r      FLOAT  (league Z of w_runs)
  - player_scores.z_h      FLOAT  (league Z of w_hits)
  - player_scores.z_tb     FLOAT  (league Z of w_tb)
  - player_scores.z_k_b    FLOAT  (league Z of w_strikeouts_bat, lower-is-better)
  - player_scores.z_ops    FLOAT  (league Z of w_ops)
  - player_scores.z_k_p    FLOAT  (league Z of w_strikeouts_pit)
  - player_scores.z_qs     FLOAT  (league Z of w_qs)

Design notes
------------
These Z-scores complete the Phase 2 expansion from 9 to 15 categories.
The 4 greenfield categories (W, L, HR_P, NSV) are NOT included here as
there is no upstream data source. They will be added in a future phase
after Yahoo/MLB Stats API ingestion is implemented.

Idempotent: uses ADD COLUMN IF NOT EXISTS. Safe to run on any environment.
Nullable additive columns only -- zero downtime, no backfill required.

Usage
-----
    python scripts/migrate_v32_zscore_expansion.py              # run upgrade
    python scripts/migrate_v32_zscore_expansion.py --downgrade  # run downgrade
    python scripts/migrate_v32_zscore_expansion.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_r DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_h DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_tb DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_k_b DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_ops DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_k_p DOUBLE PRECISION;

ALTER TABLE player_scores
    ADD COLUMN IF NOT EXISTS z_qs DOUBLE PRECISION;

COMMENT ON COLUMN player_scores.z_r IS
    'V31 League Z-score of w_runs (decay-weighted runs scored). '
    'For the R (Runs) batting category.';

COMMENT ON COLUMN player_scores.z_h IS
    'V31 League Z-score of w_hits (decay-weighted hits). '
    'For the H (Hits) batting category.';

COMMENT ON COLUMN player_scores.z_tb IS
    'V31 League Z-score of w_tb (decay-weighted total bases). '
    'For the TB (Total Bases) batting category.';

COMMENT ON COLUMN player_scores.z_k_b IS
    'V31 League Z-score of w_strikeouts_bat (decay-weighted batter K). '
    'For the K_B (Batting Strikeouts) category. Lower-is-better: Z is negated.';

COMMENT ON COLUMN player_scores.z_ops IS
    'V31 League Z-score of w_ops (decay-weighted OBP + SLG). '
    'For the OPS (On-Base Plus Slugging) batting category.';

COMMENT ON COLUMN player_scores.z_k_p IS
    'V31 League Z-score of w_strikeouts_pit (decay-weighted pitcher K). '
    'For the K_P (Pitching Strikeouts) category.';

COMMENT ON COLUMN player_scores.z_qs IS
    'V31 League Z-score of w_qs (decay-weighted quality starts). '
    'For the QS (Quality Starts) pitching category. QS = IP≥6 AND ER≤3.';
"""

DOWNGRADE_SQL = """
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_qs;
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_k_p;
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_ops;
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_k_b;
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_tb;
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_h;
ALTER TABLE player_scores  DROP COLUMN IF EXISTS z_r;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Adding Z-score columns (V32) ===")

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
        print("SUCCESS: Z-score columns added")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing Z-score columns (V32) ===")

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
        print("SUCCESS: Z-score columns removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate V32 - Z-score expansion")
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
