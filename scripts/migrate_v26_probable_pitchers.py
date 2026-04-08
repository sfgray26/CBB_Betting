#!/usr/bin/env python
"""
P26 -- Probable Pitchers Table (v26)

Creates probable_pitchers for daily probable pitcher tracking from MLB Stats API.
The DailyLineupOptimizer already fetches probable pitchers but does not persist.
This table enables historical tracking and Two-Start Command Center UI consumption.

Natural key: (game_date, team) -- one probable pitcher per team per date.

Key design decisions:
  - game_date is the MLB game date (ET, not UTC)
  - team uses 3-letter abbreviation ("NYY", "LAA") matching MLBGameLog
  - opponent and is_home provide matchup context for frontend
  - bdl_player_id enables joins to player_id_mapping for full player data
  - mlbam_id is the canonical MLB Advanced Media ID (primary cross-reference)
  - is_confirmed distinguishes official announcements from MLB.com "probable"
  - park_factor and quality_score are precomputed for Two-Start Command Center
  - fetched_at/updated_at track data freshness (probable pitchers change frequently)

Usage:
    python scripts/migrate_v26_probable_pitchers.py              # run upgrade
    python scripts/migrate_v26_probable_pitchers.py --downgrade  # run downgrade
    python scripts/migrate_v26_probable_pitchers.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS probable_pitchers (
    id                BIGSERIAL       PRIMARY KEY,
    game_date         DATE            NOT NULL,
    team              VARCHAR(10)     NOT NULL,
    opponent          VARCHAR(10),
    is_home           BOOLEAN,

    pitcher_name      VARCHAR(100),
    bdl_player_id     INTEGER,
    mlbam_id          INTEGER,
    is_confirmed      BOOLEAN         NOT NULL DEFAULT FALSE,

    game_time_et      VARCHAR(10),
    park_factor       DOUBLE PRECISION,
    quality_score     DOUBLE PRECISION,

    fetched_at        TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Natural key: one probable pitcher per team per date
-- Named to match SQLAlchemy ORM UniqueConstraint name for ON CONFLICT targeting
ALTER TABLE probable_pitchers
    ADD CONSTRAINT _pp_date_team_uc
    UNIQUE (game_date, team);

-- Primary access pattern: lookup by date
CREATE INDEX IF NOT EXISTS idx_pp_date
    ON probable_pitchers (game_date);

-- Join pattern: lookup by player
CREATE INDEX IF NOT EXISTS idx_pp_pitcher
    ON probable_pitchers (bdl_player_id)
    WHERE bdl_player_id IS NOT NULL;

COMMENT ON TABLE probable_pitchers IS
    'P26 Daily probable pitchers from MLB Stats API. '
    'Natural key: (game_date, team). '
    'Source: MLB Stats API /api/v1/schedule/games (probablePitchers field). '
    'Refresh: Job 100_014 (6 AM ET) + game-day 12 PM ET updates. '
    'Downstream: Two-Start Command Center UI, GET /api/fantasy/two-starts.';

COMMENT ON COLUMN probable_pitchers.is_confirmed IS
    'True = team officially announced starter (lineup card released). '
    'False = probable per MLB.com (subject to change).';

COMMENT ON COLUMN probable_pitchers.quality_score IS
    'Precomputed matchup rating from -2.0 (terrible) to +2.0 (great). '
    'Factors: opponent quality, park factor, pitcher recent form.';

COMMENT ON COLUMN probable_pitchers.park_factor IS
    'Park factor for the ballpark (1.0 = neutral, >1.0 = hitter-friendly, <1.0 = pitcher-friendly).';
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_pp_pitcher;
DROP INDEX IF EXISTS idx_pp_date;
DROP TABLE IF EXISTS probable_pitchers;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating probable_pitchers ===")

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
        print("SUCCESS: probable_pitchers created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing probable_pitchers ===")

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
        print("SUCCESS: probable_pitchers removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v26 - Probable Pitchers")
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
