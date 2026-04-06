#!/usr/bin/env python
"""
P13 -- Player Rolling Stats Table (v16)

Creates player_rolling_stats for decay-weighted rolling window metrics.

Key design decisions:
  - Natural key: (bdl_player_id, as_of_date, window_days)
  - Exponential decay: weight = 0.95 ** days_back per game
  - Window sizes: 7, 14, 30 days
  - Batting fields NULL for pure pitchers, pitching fields NULL for pure hitters
  - Two-way players (Ohtani etc.) have both batting + pitching fields populated
  - w_ip stored as DOUBLE PRECISION (decimal innings, e.g. 6.667 for "6.2" BDL)
  - Derived rates (w_avg, w_era, etc.) computed from weighted sums -- NOT copied
    from per-game rate stats which have no decay component
  - computed_at uses TIMESTAMPTZ -- always timezone-aware

Usage:
    python scripts/migrate_v16_player_rolling_stats.py              # run upgrade
    python scripts/migrate_v16_player_rolling_stats.py --downgrade  # run downgrade
    python scripts/migrate_v16_player_rolling_stats.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS player_rolling_stats (
    id                BIGSERIAL        PRIMARY KEY,
    bdl_player_id     INTEGER          NOT NULL,
    as_of_date        DATE             NOT NULL,
    window_days       INTEGER          NOT NULL,
    games_in_window   INTEGER          NOT NULL,

    -- Batting weighted sums (NULL for pure pitchers with no AB in window)
    w_ab              DOUBLE PRECISION,
    w_hits            DOUBLE PRECISION,
    w_doubles         DOUBLE PRECISION,
    w_triples         DOUBLE PRECISION,
    w_home_runs       DOUBLE PRECISION,
    w_rbi             DOUBLE PRECISION,
    w_walks           DOUBLE PRECISION,
    w_strikeouts_bat  DOUBLE PRECISION,
    w_stolen_bases    DOUBLE PRECISION,

    -- Batting derived rates (computed from weighted sums, not from per-game rate stats)
    w_avg             DOUBLE PRECISION,   -- w_hits / w_ab
    w_obp             DOUBLE PRECISION,   -- (w_hits + w_walks) / (w_ab + w_walks)
    w_slg             DOUBLE PRECISION,   -- weighted total bases / w_ab
    w_ops             DOUBLE PRECISION,   -- w_obp + w_slg

    -- Pitching weighted sums (NULL for pure hitters with no IP in window)
    w_ip              DOUBLE PRECISION,   -- decimal innings (e.g. 6.667 for "6.2" BDL string)
    w_earned_runs     DOUBLE PRECISION,
    w_hits_allowed    DOUBLE PRECISION,
    w_walks_allowed   DOUBLE PRECISION,
    w_strikeouts_pit  DOUBLE PRECISION,

    -- Pitching derived rates
    w_era             DOUBLE PRECISION,   -- 9 * w_earned_runs / w_ip
    w_whip            DOUBLE PRECISION,   -- (w_hits_allowed + w_walks_allowed) / w_ip
    w_k_per_9         DOUBLE PRECISION,   -- 9 * w_strikeouts_pit / w_ip

    computed_at       TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Natural key: one row per (player, date, window size)
-- Named to match SQLAlchemy ORM UniqueConstraint so ON CONFLICT targets work.
ALTER TABLE player_rolling_stats
    ADD CONSTRAINT IF NOT EXISTS _prs_player_date_window_uc
    UNIQUE (bdl_player_id, as_of_date, window_days);

-- Primary access pattern: look up all windows for a player on a date
CREATE INDEX IF NOT EXISTS idx_prs_player_date
    ON player_rolling_stats (bdl_player_id, as_of_date);

-- Secondary access pattern: bulk reads by date + window (for Z-score computation in P14)
CREATE INDEX IF NOT EXISTS idx_prs_date_window
    ON player_rolling_stats (as_of_date, window_days);

COMMENT ON TABLE player_rolling_stats IS
    'P13 decay-weighted rolling window metrics per player per date per window size. '
    'Computed daily by _compute_rolling_windows() (lock 100_018, 3 AM ET). '
    'Exponential decay: weight = 0.95 ** days_back where days_back = (as_of_date - game_date).days. '
    'Window sizes: 7, 14, 30 days. '
    'Batting fields NULL for pitchers with no AB; pitching fields NULL for hitters with no IP. '
    'w_ip is decimal innings (6.667 = 6 innings 2 outs), not the BDL string "6.2". '
    'Downstream: P14 Z-score engine reads this table to compute league/position Z-scores.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_prs_date_window;
DROP INDEX IF EXISTS idx_prs_player_date;
DROP TABLE IF EXISTS player_rolling_stats;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating player_rolling_stats ===")

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
        print("SUCCESS: player_rolling_stats created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing player_rolling_stats ===")

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
        print("SUCCESS: player_rolling_stats removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v16 - Player Rolling Stats")
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
