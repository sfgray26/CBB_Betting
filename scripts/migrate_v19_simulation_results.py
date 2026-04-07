#!/usr/bin/env python
"""
P16 -- Rest-of-Season Monte Carlo Simulation Results Table (v19)

Creates simulation_results for 1000-run Monte Carlo ROS projections derived
from the 14-day decay-weighted rolling window (player_rolling_stats).

Key design decisions:
  - Natural key: (bdl_player_id, as_of_date) -- one simulation snapshot per player per day
  - Named constraint _sr_player_date_uc matches SQLAlchemy ORM UniqueConstraint
    so ON CONFLICT targeting works in daily_ingestion._run_ros_simulation()
  - CV=0.35 (35% game-to-game variation), N=1000 simulations, remaining_games=130
  - Hitter percentiles: proj_hr, proj_rbi, proj_sb, proj_avg (P10/25/50/75/90)
  - Pitcher percentiles: proj_k, proj_era, proj_whip (P10/25/50/75/90)
  - Two-way players: all fields populated (NULL only for pure pitcher/hitter fields)
  - Risk metrics (composite_variance, downside_p25, upside_p75, prob_above_median)
    require league_means/stds from player_scores; NULL if scores unavailable
  - computed_at uses TIMESTAMPTZ -- always timezone-aware
  - Downstream: P17 decision engines (lineup optimizer, waiver intelligence)

Usage:
    python scripts/migrate_v19_simulation_results.py              # run upgrade
    python scripts/migrate_v19_simulation_results.py --downgrade  # run downgrade
    python scripts/migrate_v19_simulation_results.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS simulation_results (
    id                  BIGSERIAL        PRIMARY KEY,
    bdl_player_id       INTEGER          NOT NULL,
    as_of_date          DATE             NOT NULL,
    window_days         INTEGER          NOT NULL DEFAULT 14,
    remaining_games     INTEGER          NOT NULL,
    n_simulations       INTEGER          NOT NULL,
    player_type         VARCHAR(10)      NOT NULL,

    -- Hitter projection percentiles (NULL for pure pitchers)
    proj_hr_p10         DOUBLE PRECISION,
    proj_hr_p25         DOUBLE PRECISION,
    proj_hr_p50         DOUBLE PRECISION,
    proj_hr_p75         DOUBLE PRECISION,
    proj_hr_p90         DOUBLE PRECISION,

    proj_rbi_p10        DOUBLE PRECISION,
    proj_rbi_p25        DOUBLE PRECISION,
    proj_rbi_p50        DOUBLE PRECISION,
    proj_rbi_p75        DOUBLE PRECISION,
    proj_rbi_p90        DOUBLE PRECISION,

    proj_sb_p10         DOUBLE PRECISION,
    proj_sb_p25         DOUBLE PRECISION,
    proj_sb_p50         DOUBLE PRECISION,
    proj_sb_p75         DOUBLE PRECISION,
    proj_sb_p90         DOUBLE PRECISION,

    proj_avg_p10        DOUBLE PRECISION,
    proj_avg_p25        DOUBLE PRECISION,
    proj_avg_p50        DOUBLE PRECISION,
    proj_avg_p75        DOUBLE PRECISION,
    proj_avg_p90        DOUBLE PRECISION,

    -- Pitcher projection percentiles (NULL for pure hitters)
    proj_k_p10          DOUBLE PRECISION,
    proj_k_p25          DOUBLE PRECISION,
    proj_k_p50          DOUBLE PRECISION,
    proj_k_p75          DOUBLE PRECISION,
    proj_k_p90          DOUBLE PRECISION,

    proj_era_p10        DOUBLE PRECISION,
    proj_era_p25        DOUBLE PRECISION,
    proj_era_p50        DOUBLE PRECISION,
    proj_era_p75        DOUBLE PRECISION,
    proj_era_p90        DOUBLE PRECISION,

    proj_whip_p10       DOUBLE PRECISION,
    proj_whip_p25       DOUBLE PRECISION,
    proj_whip_p50       DOUBLE PRECISION,
    proj_whip_p75       DOUBLE PRECISION,
    proj_whip_p90       DOUBLE PRECISION,

    -- Risk metrics (NULL when player_scores unavailable for this date)
    composite_variance  DOUBLE PRECISION,
    downside_p25        DOUBLE PRECISION,
    upside_p75          DOUBLE PRECISION,
    prob_above_median   DOUBLE PRECISION,

    computed_at         TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Natural key: one simulation row per player per date
-- Named to match SQLAlchemy ORM UniqueConstraint so ON CONFLICT targets work.
ALTER TABLE simulation_results
    ADD CONSTRAINT IF NOT EXISTS _sr_player_date_uc
    UNIQUE (bdl_player_id, as_of_date);

-- Primary access pattern: bulk reads by date (daily feed queries)
CREATE INDEX IF NOT EXISTS idx_sr_date
    ON simulation_results (as_of_date);

-- Player history lookup
CREATE INDEX IF NOT EXISTS idx_sr_player_date
    ON simulation_results (bdl_player_id, as_of_date);

COMMENT ON TABLE simulation_results IS
    'P16 Rest-of-Season Monte Carlo simulation results per player per date. '
    'Computed daily by _run_ros_simulation() (lock 100_021, 6 AM ET). '
    'Input: player_rolling_stats (14d window -- current form baseline). '
    'Algorithm: N=1000 simulations, CV=0.35 game-to-game variation, '
    'remaining_games=130 (mid-April 2026 default). '
    'Risk metrics (composite_variance, downside_p25, upside_p75, prob_above_median) '
    'require league_means/stds from player_scores -- NULL if unavailable. '
    'Downstream: P17 lineup/waiver decision engines.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_sr_player_date;
DROP INDEX IF EXISTS idx_sr_date;
DROP TABLE IF EXISTS simulation_results;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating simulation_results ===")

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
        print("SUCCESS: simulation_results created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing simulation_results ===")

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
        print("SUCCESS: simulation_results removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v19 - Simulation Results")
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
