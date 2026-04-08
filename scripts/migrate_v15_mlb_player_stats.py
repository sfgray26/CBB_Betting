#!/usr/bin/env python
"""
P11 -- MLB Player Box Stats Table (v15)

Creates mlb_player_stats for per-player per-game box stat storage.
Natural key: (bdl_player_id, game_id) -- enforced via unique constraint.
Dual-write: raw_payload column stores the full BDL API dict alongside
normalized columns so downstream analytics never re-parse the wire format.

Key design decisions:
  - Column names differ from API field names to avoid Python keyword conflicts
    (r -> runs, h -> hits, double -> doubles, etc.)
  - rate stats (avg, obp, slg, ops, whip, era) stored as DOUBLE PRECISION
    because the BDL probe (S19) confirmed they arrive as floats, NOT strings
  - innings_pitched (ip) stored as VARCHAR(10) because "6.2" is a string in the API
  - game_id FK -> mlb_game_log.game_id is nullable (stats may arrive before game row)
  - raw_payload is plain JSON (not JSONB) to match other ingestion tables in this schema
  - ingested_at uses TIMESTAMPTZ -- always store timezone-aware timestamps

Usage:
    python scripts/migrate_v15_mlb_player_stats.py              # run upgrade
    python scripts/migrate_v15_mlb_player_stats.py --downgrade  # run downgrade
    python scripts/migrate_v15_mlb_player_stats.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS mlb_player_stats (
    id                BIGSERIAL     PRIMARY KEY,
    bdl_stat_id       INTEGER,                        -- BDL stats record id (nullable -- may not always be present)
    bdl_player_id     INTEGER       NOT NULL,          -- player.id from BDL
    game_id           INTEGER       REFERENCES mlb_game_log(game_id),  -- nullable: stats may arrive before game row
    game_date         DATE          NOT NULL,
    season            INTEGER       NOT NULL DEFAULT 2026,

    -- Batting stats (null for pure pitchers)
    ab                INTEGER,
    runs              INTEGER,                        -- 'r' from API: renamed to avoid Python keyword
    hits              INTEGER,                        -- 'h' from API: renamed for clarity
    doubles           INTEGER,                        -- 'double' from API: renamed to avoid Python builtin
    triples           INTEGER,
    home_runs         INTEGER,                        -- 'hr' from API
    rbi               INTEGER,
    walks             INTEGER,                        -- 'bb' from API
    strikeouts_bat    INTEGER,                        -- 'so' from API
    stolen_bases      INTEGER,                        -- 'sb' from API
    caught_stealing   INTEGER,                        -- 'cs' from API
    avg               DOUBLE PRECISION,               -- float per S19 probe (NOT string)
    obp               DOUBLE PRECISION,
    slg               DOUBLE PRECISION,
    ops               DOUBLE PRECISION,

    -- Pitching stats (null for pure hitters)
    innings_pitched   VARCHAR(10),                    -- 'ip' e.g. "6.2": stored as string per API contract
    hits_allowed      INTEGER,                        -- 'h_allowed' from API
    runs_allowed      INTEGER,                        -- 'r_allowed' from API
    earned_runs       INTEGER,                        -- 'er' from API
    walks_allowed     INTEGER,                        -- 'bb_allowed' from API
    strikeouts_pit    INTEGER,                        -- 'k' from API
    whip              DOUBLE PRECISION,               -- float per S19 probe
    era               DOUBLE PRECISION,               -- float per S19 probe

    -- Audit
    raw_payload       JSON          NOT NULL,         -- full BDL dict (dual-write, plain JSON)
    ingested_at       TIMESTAMPTZ   NOT NULL DEFAULT now()
);

-- Unique constraint on (bdl_player_id, game_id) -- PostgreSQL NULL semantics mean
-- two rows with game_id=NULL do NOT conflict (NULL != NULL), which is correct behaviour
-- for the rare case of stats arriving without a game reference.
-- Named to match SQLAlchemy ORM UniqueConstraint name so ON CONFLICT targets work.
ALTER TABLE mlb_player_stats
    ADD CONSTRAINT _mps_player_game_uc
    UNIQUE (bdl_player_id, game_id);

CREATE INDEX IF NOT EXISTS idx_mps_player_date
    ON mlb_player_stats (bdl_player_id, game_date);

CREATE INDEX IF NOT EXISTS idx_mps_game
    ON mlb_player_stats (game_id)
    WHERE game_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_mps_date
    ON mlb_player_stats (game_date);

COMMENT ON TABLE mlb_player_stats IS
    'MLB per-player per-game box stats from BDL /mlb/v1/stats. '
    'Natural key: (bdl_player_id, game_id). '
    'Dual-write: raw_payload stores full BDL dict alongside normalized columns. '
    'Rate stats (avg, obp, slg, ops, whip, era) stored as DOUBLE PRECISION -- '
    'confirmed as floats in S19 live probe. '
    'innings_pitched stored as VARCHAR because "6.2" is a string in the API. '
    'game_id FK is nullable: stats may arrive before the game_log row is written.';

COMMENT ON COLUMN mlb_player_stats.runs IS
    'Batting runs scored (API field: r). Renamed to avoid Python keyword conflict.';

COMMENT ON COLUMN mlb_player_stats.hits IS
    'Batting hits (API field: h). Renamed for unambiguous column naming.';

COMMENT ON COLUMN mlb_player_stats.doubles IS
    'Doubles (API field: double). Renamed to avoid Python builtin shadowing.';

COMMENT ON COLUMN mlb_player_stats.innings_pitched IS
    'Innings pitched (API field: ip). Stored as VARCHAR e.g. "6.2" -- fractional '
    'innings in baseball notation: "6.2" means 6 innings + 2 outs, NOT 6.2 decimal innings.';

COMMENT ON COLUMN mlb_player_stats.raw_payload IS
    'Full BDL /mlb/v1/stats row dict. Plain JSON (not JSONB) to match ingestion table convention. '
    'Contains all API fields including nested player and team objects.';
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_mps_date;
DROP INDEX IF EXISTS idx_mps_game;
DROP INDEX IF EXISTS idx_mps_player_date;
DROP TABLE IF EXISTS mlb_player_stats;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating mlb_player_stats ===")

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
        print("SUCCESS: mlb_player_stats created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing mlb_player_stats ===")

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
        print("SUCCESS: mlb_player_stats removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v15 - MLB Player Box Stats")
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
