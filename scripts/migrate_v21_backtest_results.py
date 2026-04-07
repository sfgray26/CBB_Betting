#!/usr/bin/env python
"""
P18 -- Backtesting Harness Results Table (v21)

Creates backtest_results for per-player forecast accuracy metrics derived
from comparing simulation_results (P16 projections) against actual
mlb_player_stats outcomes over a rolling 14-day window.

Key design decisions:
  - Natural key: (bdl_player_id, as_of_date) -- one row per player per date
  - Named constraint _br_player_date_uc matches SQLAlchemy ORM UniqueConstraint
    so ON CONFLICT targeting works in daily_ingestion._run_backtesting()
  - computed_at uses TIMESTAMPTZ -- always timezone-aware
  - Upstream: simulation_results (P16) vs mlb_player_stats (actuals)
  - Downstream: P19 Explainability Layer

Usage:
    python scripts/migrate_v21_backtest_results.py              # run upgrade
    python scripts/migrate_v21_backtest_results.py --downgrade  # run downgrade
    python scripts/migrate_v21_backtest_results.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS backtest_results (
    id              BIGSERIAL           PRIMARY KEY,
    bdl_player_id   INTEGER             NOT NULL,
    as_of_date      DATE                NOT NULL,
    player_type     VARCHAR(10)         NOT NULL,
    games_played    INTEGER             NOT NULL,
    mae_hr          DOUBLE PRECISION,
    rmse_hr         DOUBLE PRECISION,
    mae_rbi         DOUBLE PRECISION,
    rmse_rbi        DOUBLE PRECISION,
    mae_sb          DOUBLE PRECISION,
    rmse_sb         DOUBLE PRECISION,
    mae_avg         DOUBLE PRECISION,
    rmse_avg        DOUBLE PRECISION,
    mae_k           DOUBLE PRECISION,
    rmse_k          DOUBLE PRECISION,
    mae_era         DOUBLE PRECISION,
    rmse_era        DOUBLE PRECISION,
    mae_whip        DOUBLE PRECISION,
    rmse_whip       DOUBLE PRECISION,
    composite_mae   DOUBLE PRECISION,
    direction_correct BOOLEAN,
    computed_at     TIMESTAMPTZ         NOT NULL DEFAULT now()
);

ALTER TABLE backtest_results
    ADD CONSTRAINT _br_player_date_uc
    UNIQUE (bdl_player_id, as_of_date);

CREATE INDEX IF NOT EXISTS idx_br_date
    ON backtest_results (as_of_date);

CREATE INDEX IF NOT EXISTS idx_br_player_date
    ON backtest_results (bdl_player_id, as_of_date);

COMMENT ON TABLE backtest_results IS
    'P18 Backtesting Harness results -- per-player forecast accuracy metrics. '
    'Computed daily by _run_backtesting() (lock 100_023, 8 AM ET). '
    'Upstream: simulation_results (P16 projections) vs mlb_player_stats (actuals). '
    'Compares proj_p50 against actual stats over a rolling 14-day window. '
    'Natural key: (bdl_player_id, as_of_date). '
    'Downstream: P19 Explainability Layer.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_br_player_date;
DROP INDEX IF EXISTS idx_br_date;
DROP TABLE IF EXISTS backtest_results;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating backtest_results ===")

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
                    print("  WARNING: Skipping (already exists): " + str(e)[:100])
                else:
                    raise
        print("SUCCESS: backtest_results created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing backtest_results ===")

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
                    print("  WARNING: " + str(e)[:100])
        print("SUCCESS: backtest_results removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v21 - Backtesting Results")
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
