#!/usr/bin/env python
"""
P15 -- Player Momentum Table (v18)

Creates player_momentum for delta-Z momentum signals derived from 14d vs 30d
player_scores rows.

Key design decisions:
  - Natural key: (bdl_player_id, as_of_date) -- one momentum reading per player per day
  - delta_z = composite_z_14d - composite_z_30d
  - Signal thresholds (boundary-exact):
      delta_z >  0.5  -> SURGING
      delta_z >= 0.2  -> HOT
      delta_z >  -0.2 -> STABLE
      delta_z >= -0.5 -> COLD
      else            -> COLLAPSING
  - confidence = min(confidence_14d, confidence_30d)
  - computed_at uses TIMESTAMPTZ -- always timezone-aware
  - Downstream: P16 probabilistic layer, P7 decision engines

Usage:
    python scripts/migrate_v18_player_momentum.py              # run upgrade
    python scripts/migrate_v18_player_momentum.py --downgrade  # run downgrade
    python scripts/migrate_v18_player_momentum.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS player_momentum (
    id              BIGSERIAL        PRIMARY KEY,
    bdl_player_id   INTEGER          NOT NULL,
    as_of_date      DATE             NOT NULL,
    player_type     VARCHAR(10)      NOT NULL,
    delta_z         DOUBLE PRECISION NOT NULL,
    signal          VARCHAR(12)      NOT NULL,
    composite_z_14d DOUBLE PRECISION NOT NULL,
    composite_z_30d DOUBLE PRECISION NOT NULL,
    score_14d       DOUBLE PRECISION NOT NULL,
    score_30d       DOUBLE PRECISION NOT NULL,
    confidence_14d  DOUBLE PRECISION NOT NULL,
    confidence_30d  DOUBLE PRECISION NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    computed_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Natural key: one momentum row per player per date
-- Named to match SQLAlchemy ORM UniqueConstraint so ON CONFLICT targets work.
ALTER TABLE player_momentum
    ADD CONSTRAINT IF NOT EXISTS _pm_player_date_uc
    UNIQUE (bdl_player_id, as_of_date);

-- Primary access pattern: bulk reads by date + signal (momentum feed queries)
CREATE INDEX IF NOT EXISTS idx_pm_date_signal
    ON player_momentum (as_of_date, signal);

-- Player history lookup
CREATE INDEX IF NOT EXISTS idx_pm_player_date
    ON player_momentum (bdl_player_id, as_of_date);

COMMENT ON TABLE player_momentum IS
    'P15 delta-Z momentum signals per player per date. '
    'Computed daily by _compute_player_momentum() (lock 100_020, 5 AM ET). '
    'Input: player_scores (P14) -- requires both 14d and 30d rows for same player + date. '
    'delta_z = composite_z_14d - composite_z_30d. '
    'Signal thresholds: SURGING(>0.5), HOT(>=0.2), STABLE(>-0.2), COLD(>=-0.5), COLLAPSING(<-0.5). '
    'confidence = min(confidence_14d, confidence_30d). '
    'Downstream: P16 probabilistic layer, P7 lineup/waiver decision engines.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_pm_player_date;
DROP INDEX IF EXISTS idx_pm_date_signal;
DROP TABLE IF EXISTS player_momentum;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating player_momentum ===")

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
        print("SUCCESS: player_momentum created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing player_momentum ===")

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
        print("SUCCESS: player_momentum removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v18 - Player Momentum")
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
