#!/usr/bin/env python
"""
P20 -- Daily Snapshots Table (v23)

Creates daily_snapshots for end-of-pipeline state capture. One row per day.
Stores counts from all 9 prior phases, pipeline health status, top player IDs,
regression flag, and a human-readable summary string.

Key design decisions:
  - Natural key: (as_of_date,) -- one snapshot per calendar day
  - Named constraint _ds_date_uc matches SQLAlchemy ORM unique=True
    so ON CONFLICT targeting works in daily_ingestion._run_snapshot()
  - JSONB columns for top_lineup_player_ids, top_waiver_player_ids,
    pipeline_jobs_run, health_reasons (richer querying than plain JSON)
  - summary capped at 500 chars; pipeline_health capped at 10 chars
  - computed_at uses TIMESTAMPTZ -- always timezone-aware
  - Upstream: all P14-P19 pipeline stages (player_scores, player_momentum,
              simulation_results, decision_results, backtest_results,
              decision_explanations)
  - Downstream: GET /admin/snapshot/latest, GET /admin/snapshot/{date}

Usage:
    python scripts/migrate_v23_daily_snapshots.py              # run upgrade
    python scripts/migrate_v23_daily_snapshots.py --downgrade  # run downgrade
    python scripts/migrate_v23_daily_snapshots.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS daily_snapshots (
    id                      BIGSERIAL       PRIMARY KEY,
    as_of_date              DATE            NOT NULL,

    n_players_scored        INTEGER         NOT NULL DEFAULT 0,
    n_momentum_records      INTEGER         NOT NULL DEFAULT 0,
    n_simulation_records    INTEGER         NOT NULL DEFAULT 0,
    n_decisions             INTEGER         NOT NULL DEFAULT 0,
    n_explanations          INTEGER         NOT NULL DEFAULT 0,
    n_backtest_records      INTEGER         NOT NULL DEFAULT 0,

    mean_composite_mae      FLOAT,
    regression_detected     BOOLEAN         NOT NULL DEFAULT FALSE,

    -- JSONB arrays for efficient querying
    top_lineup_player_ids   JSONB,
    top_waiver_player_ids   JSONB,
    pipeline_jobs_run       JSONB,

    pipeline_health         VARCHAR(10)     NOT NULL DEFAULT 'UNKNOWN',
    health_reasons          JSONB,
    summary                 VARCHAR(500),

    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Natural key: one snapshot per calendar day
-- Named to match SQLAlchemy ORM unique=True so ON CONFLICT targets work.
ALTER TABLE daily_snapshots
    ADD CONSTRAINT _ds_date_uc
    UNIQUE (as_of_date);

-- Primary access pattern: lookup by date
CREATE INDEX IF NOT EXISTS idx_ds_date
    ON daily_snapshots (as_of_date);

COMMENT ON TABLE daily_snapshots IS
    'P20 daily pipeline state capture. One row per day. '
    'Upstream: all P14-P19 pipeline stages (player_scores, player_momentum, '
    'simulation_results, decision_results, backtest_results, decision_explanations). '
    'Endpoint: GET /admin/snapshot/latest.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_ds_date;
DROP TABLE IF EXISTS daily_snapshots;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating daily_snapshots ===")

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
        print("SUCCESS: daily_snapshots created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing daily_snapshots ===")

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
        print("SUCCESS: daily_snapshots removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v23 - Daily Snapshots")
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
