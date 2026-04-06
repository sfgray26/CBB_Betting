#!/usr/bin/env python
"""
P14 -- Player Scores Table (v17)

Creates player_scores for league Z-scores + composite 0-100 ranking per player.

Key design decisions:
  - Natural key: (bdl_player_id, as_of_date, window_days) -- same grain as player_rolling_stats
  - Z-score methodology: population std (ddof=0); MIN_SAMPLE=5 players required
  - Lower-is-better categories (ERA, WHIP): Z is negated so higher Z = better
  - Z capped at +/-3.0 to dampen outlier distortion
  - composite_z = mean of all applicable non-None per-category Z-scores
  - score_0_100 = percentile rank (0-100) within player_type cohort (hitter/pitcher/two_way)
  - confidence = min(1.0, games_in_window / window_days)
  - Hitter categories: z_hr, z_rbi, z_sb, z_avg, z_obp
  - Pitcher categories: z_era, z_whip, z_k_per_9
  - Two-way players get all categories (Ohtani-style)
  - Position Z-scores deferred -- position data not yet linked to bdl_player_id
  - computed_at uses TIMESTAMPTZ -- always timezone-aware

Usage:
    python scripts/migrate_v17_player_scores.py              # run upgrade
    python scripts/migrate_v17_player_scores.py --downgrade  # run downgrade
    python scripts/migrate_v17_player_scores.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS player_scores (
    id              BIGSERIAL        PRIMARY KEY,
    bdl_player_id   INTEGER          NOT NULL,
    as_of_date      DATE             NOT NULL,
    window_days     INTEGER          NOT NULL,
    player_type     VARCHAR(10)      NOT NULL,
    games_in_window INTEGER          NOT NULL,

    -- Per-category Z-scores (NULL if not applicable or below MIN_SAMPLE threshold)
    z_hr            DOUBLE PRECISION,
    z_rbi           DOUBLE PRECISION,
    z_sb            DOUBLE PRECISION,
    z_avg           DOUBLE PRECISION,
    z_obp           DOUBLE PRECISION,
    z_era           DOUBLE PRECISION,
    z_whip          DOUBLE PRECISION,
    z_k_per_9       DOUBLE PRECISION,

    -- Aggregate scores
    composite_z     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    score_0_100     DOUBLE PRECISION NOT NULL DEFAULT 50.0,
    confidence      DOUBLE PRECISION NOT NULL DEFAULT 0.0,

    computed_at     TIMESTAMPTZ      NOT NULL DEFAULT now()
);

-- Natural key: one row per (player, date, window size)
-- Named to match SQLAlchemy ORM UniqueConstraint so ON CONFLICT targets work.
ALTER TABLE player_scores
    ADD CONSTRAINT IF NOT EXISTS _ps_player_date_window_uc
    UNIQUE (bdl_player_id, as_of_date, window_days);

-- Primary access pattern: bulk reads by date + window (scoring queries)
CREATE INDEX IF NOT EXISTS idx_ps_date_window
    ON player_scores (as_of_date, window_days);

-- Player lookup across dates
CREATE INDEX IF NOT EXISTS idx_ps_player_date
    ON player_scores (bdl_player_id, as_of_date);

-- Ranking queries: top N players by score for a given date + window
CREATE INDEX IF NOT EXISTS idx_ps_score
    ON player_scores (as_of_date, window_days, score_0_100);

COMMENT ON TABLE player_scores IS
    'P14 league Z-scores + composite 0-100 ranking per player per date per window size. '
    'Computed daily by _compute_player_scores() (lock 100_019, 4 AM ET). '
    'Input: player_rolling_stats (P13). Window sizes: 7, 14, 30 days. '
    'Z-score methodology: population std (ddof=0); MIN_SAMPLE=5; Z capped at +/-3.0. '
    'Lower-is-better categories (ERA, WHIP): Z is negated so higher Z = better player. '
    'Hitter categories: z_hr, z_rbi, z_sb, z_avg, z_obp. '
    'Pitcher categories: z_era, z_whip, z_k_per_9. '
    'Two-way players (Ohtani etc.) receive all applicable categories. '
    'score_0_100 = percentile rank within player_type cohort (hitter/pitcher/two_way). '
    'confidence = min(1.0, games_in_window / window_days). '
    'Position Z-scores deferred -- position data not yet linked to bdl_player_id. '
    'Downstream: P15 momentum layer reads this table to compute delta Z-scores.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_ps_score;
DROP INDEX IF EXISTS idx_ps_player_date;
DROP INDEX IF EXISTS idx_ps_date_window;
DROP TABLE IF EXISTS player_scores;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating player_scores ===")

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
        print("SUCCESS: player_scores created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing player_scores ===")

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
        print("SUCCESS: player_scores removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v17 - Player Scores")
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
