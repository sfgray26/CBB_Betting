#!/usr/bin/env python
"""
P7 — MLB Phase 2 Game Log Ingestion Schema (v13)

Creates three tables that form the Phase 2 data foundation:
  mlb_team          -- team dimension, seeded from BDL game responses
  mlb_game_log      -- game fact table, idempotent upsert on game_id
  mlb_odds_snapshot -- line-movement history, upsert on (game_id, vendor, snapshot_window)

Design decisions:
  - raw_payload JSONB on every fact table (dual-write principle)
  - mlb_team must exist before mlb_game_log (FK dependency)
  - mlb_game_log must exist before mlb_odds_snapshot (FK dependency)
  - spread/total columns are VARCHAR to match BDL contract (values arrive as strings)
  - game_date is DATE (ET), not the raw UTC ISO 8601 timestamp from BDL

Usage:
    python scripts/migrate_v13_mlb_game_log.py              # run upgrade
    python scripts/migrate_v13_mlb_game_log.py --downgrade  # run downgrade
    python scripts/migrate_v13_mlb_game_log.py --dry-run    # print SQL, no execute
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- Step 1: Team dimension table
CREATE TABLE IF NOT EXISTS mlb_team (
    team_id      INTEGER      PRIMARY KEY,
    abbreviation VARCHAR(10)  NOT NULL,
    name         VARCHAR(100) NOT NULL,
    display_name VARCHAR(150) NOT NULL,
    short_name   VARCHAR(50)  NOT NULL,
    location     VARCHAR(100) NOT NULL,
    slug         VARCHAR(50)  NOT NULL,
    league       VARCHAR(10)  NOT NULL,
    division     VARCHAR(10)  NOT NULL,
    ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
);

COMMENT ON TABLE mlb_team IS
    'MLB team dimension. Seeded from MLBTeam objects embedded in BDL game responses. '
    'Upserted on team_id before every mlb_game_log write.';

-- Step 2: Game fact table
CREATE TABLE IF NOT EXISTS mlb_game_log (
    game_id      INTEGER      PRIMARY KEY,
    game_date    DATE         NOT NULL,
    season       INTEGER      NOT NULL,
    season_type  VARCHAR(20)  NOT NULL,
    status       VARCHAR(30)  NOT NULL,
    home_team_id INTEGER      NOT NULL REFERENCES mlb_team(team_id),
    away_team_id INTEGER      NOT NULL REFERENCES mlb_team(team_id),
    home_runs    INTEGER,
    away_runs    INTEGER,
    home_hits    INTEGER,
    away_hits    INTEGER,
    home_errors  INTEGER,
    away_errors  INTEGER,
    venue        VARCHAR(200),
    attendance   INTEGER,
    period       INTEGER,
    raw_payload  JSONB        NOT NULL,
    ingested_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_mlb_game_log_date
    ON mlb_game_log (game_date);

CREATE INDEX IF NOT EXISTS idx_mlb_game_log_status
    ON mlb_game_log (status);

CREATE INDEX IF NOT EXISTS idx_mlb_game_log_season_date
    ON mlb_game_log (season, game_date);

COMMENT ON TABLE mlb_game_log IS
    'MLB game fact table. One row per BDL game_id. '
    'Upserted as game status progresses SCHEDULED -> IN_PROGRESS -> FINAL. '
    'game_date is ET (converted from BDL UTC timestamp). '
    'Scores stored as home_runs/away_runs from MLBTeamGameData.runs. '
    'raw_payload preserves the full BDL MLBGame dict for audit and replay.';

COMMENT ON COLUMN mlb_game_log.home_runs IS
    'From MLBTeamGameData.runs. NULL pre-game.';
COMMENT ON COLUMN mlb_game_log.game_date IS
    'ET date. BDL MLBGame.date is UTC ISO 8601 -- must convert with ZoneInfo America/New_York.';

-- Step 3: Odds snapshot table
CREATE TABLE IF NOT EXISTS mlb_odds_snapshot (
    id               BIGSERIAL    PRIMARY KEY,
    odds_id          INTEGER      NOT NULL,
    game_id          INTEGER      NOT NULL REFERENCES mlb_game_log(game_id),
    vendor           VARCHAR(50)  NOT NULL,
    snapshot_window  TIMESTAMPTZ  NOT NULL,
    spread_home      VARCHAR(10)  NOT NULL,
    spread_away      VARCHAR(10)  NOT NULL,
    spread_home_odds INTEGER      NOT NULL,
    spread_away_odds INTEGER      NOT NULL,
    ml_home_odds     INTEGER      NOT NULL,
    ml_away_odds     INTEGER      NOT NULL,
    total            VARCHAR(10)  NOT NULL,
    total_over_odds  INTEGER      NOT NULL,
    total_under_odds INTEGER      NOT NULL,
    raw_payload      JSONB        NOT NULL,
    UNIQUE (game_id, vendor, snapshot_window)
);

CREATE INDEX IF NOT EXISTS idx_mlb_odds_game
    ON mlb_odds_snapshot (game_id);

CREATE INDEX IF NOT EXISTS idx_mlb_odds_vendor_window
    ON mlb_odds_snapshot (vendor, snapshot_window);

COMMENT ON TABLE mlb_odds_snapshot IS
    'MLB odds line-movement history. One row per (game_id, vendor, snapshot_window). '
    'snapshot_window is the poll time rounded to the 30-minute bucket. '
    'spread/total columns are VARCHAR to match BDL contract (API sends strings). '
    'raw_payload preserves the full BDL MLBBettingOdd dict for audit and replay.';

COMMENT ON COLUMN mlb_odds_snapshot.snapshot_window IS
    'Poll timestamp rounded to 30-min bucket. Matches _poll_mlb_odds job cadence. '
    'Idempotent upsert key component.';

COMMENT ON COLUMN mlb_odds_snapshot.odds_id IS
    'BDL MLBBettingOdd.id. Stored for cross-reference; not the natural key '
    '(BDL may reissue IDs as lines shift).';
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_mlb_odds_vendor_window;
DROP INDEX IF EXISTS idx_mlb_odds_game;
DROP TABLE IF EXISTS mlb_odds_snapshot;

DROP INDEX IF EXISTS idx_mlb_game_log_season_date;
DROP INDEX IF EXISTS idx_mlb_game_log_status;
DROP INDEX IF EXISTS idx_mlb_game_log_date;
DROP TABLE IF EXISTS mlb_game_log;

DROP TABLE IF EXISTS mlb_team;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating mlb_team, mlb_game_log, mlb_odds_snapshot ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text(UPGRADE_SQL))
            print("SUCCESS: mlb_team, mlb_game_log, mlb_odds_snapshot created")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  WARNING: Skipping (already exists): {str(e)[:100]}")
            else:
                raise


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing mlb_odds_snapshot, mlb_game_log, mlb_team ===")

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

        print("SUCCESS: MLB game log tables removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v13 - MLB Game Log Schema")
    parser.add_argument("--downgrade", action="store_true", help="Rollback migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    engine = create_engine(db_url)

    if args.downgrade:
        downgrade(engine, dry_run=args.dry_run)
    else:
        upgrade(engine, dry_run=args.dry_run)
