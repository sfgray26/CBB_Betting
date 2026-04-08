#!/usr/bin/env python
"""
P25 -- Position Eligibility Table (v25)

Creates position_eligibility for H2H One Win fantasy baseball format.
Tracks LF/CF/RF granularity (not generic OF) which is critical for position
scarcity calculations in the Yahoo H2H One Win format.

Natural key: (bdl_player_id,) -- one eligibility record per player.

Key design decisions:
  - Position-specific booleans for each Yahoo fantasy position (C, 1B, 2B, 3B,
    SS, LF, CF, RF, OF, DH, UTIL) enable multi-eligibility counting
  - primary_position stores the player's main position for scarcity calculations
  - player_type distinguishes batters (B) from pitchers (P) for filtering
  - scarcity_rank (1-100 within position) computed daily by ingestion job
  - league_rostered_pct for free agent/waiver decision context
  - multi_eligibility_count precomputed for fast CF scarcity queries
    (e.g., Bellinger CF/LF/RF counts as 3-eligible for ALL three OF slots)
  - fetched_at and updated_at use TIMESTAMPTZ for timezone-aware audit trail
  - NO foreign key to mlb_player_stats to avoid circular dependency --
    bdl_player_id is the natural cross-reference

Usage:
    python scripts/migrate_v25_position_eligibility.py              # run upgrade
    python scripts/migrate_v25_position_eligibility.py --downgrade  # run downgrade
    python scripts/migrate_v25_position_eligibility.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS position_eligibility (
    id                      BIGSERIAL       PRIMARY KEY,
    bdl_player_id           INTEGER         NOT NULL,

    -- Yahoo Fantasy position eligibility (H2H One Win format)
    can_play_c              BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_1b             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_2b             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_3b             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_ss             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_lf             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_cf             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_rf             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_of             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_dh             BOOLEAN         NOT NULL DEFAULT FALSE,
    can_play_util           BOOLEAN         NOT NULL DEFAULT FALSE,

    -- Primary position for scarcity calculations
    primary_position        VARCHAR(10),

    -- Player type for filtering (B=batter, P=pitcher)
    player_type             VARCHAR(10)     NOT NULL,

    -- Scarcity metrics (computed daily)
    scarcity_rank           INTEGER,
    league_rostered_pct     DOUBLE PRECISION,
    multi_eligibility_count INTEGER         NOT NULL DEFAULT 0,

    -- Audit trail
    fetched_at              TIMESTAMPTZ     NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Natural key: one eligibility record per player
-- Named to match SQLAlchemy ORM UniqueConstraint name for ON CONFLICT targeting
ALTER TABLE position_eligibility
    ADD CONSTRAINT _pe_player_uc
    UNIQUE (bdl_player_id);

-- Primary access pattern: lookup by player
CREATE INDEX IF NOT EXISTS idx_pe_player
    ON position_eligibility (bdl_player_id);

-- Scarcity computation: filter by position
CREATE INDEX IF NOT EXISTS idx_pe_primary_position
    ON position_eligibility (primary_position)
    WHERE primary_position IS NOT NULL;

-- Multi-eligibility queries for CF scarcity
CREATE INDEX IF NOT EXISTS idx_pe_multi_eligible
    ON position_eligibility (multi_eligibility_count)
    WHERE multi_eligibility_count > 1;

COMMENT ON TABLE position_eligibility IS
    'P25 H2H One Win position eligibility with LF/CF/RF granularity. '
    'Natural key: (bdl_player_id,). '
    'Multi-eligibility (e.g., Bellinger CF/LF/RF) counts for ALL positions. '
    'Computed by daily ingestion job from Yahoo Fantasy API + MLB Stats API. '
    'Upstream: Yahoo Fantasy League metadata, MLB Stats roster endpoints. '
    'Downstream: GET /api/fantasy/scarcity-index, Weekly Compass widget.';

COMMENT ON COLUMN position_eligibility.can_play_lf IS
    'True if player is eligible at Left Field in Yahoo Fantasy format.';

COMMENT ON COLUMN position_eligibility.can_play_cf IS
    'True if player is eligible at Center Field. CF is scarcest OF position (~45 MLB players qualify).';

COMMENT ON COLUMN position_eligibility.can_play_rf IS
    'True if player is eligible at Right Field.';

COMMENT ON COLUMN position_eligibility.can_play_of IS
    'Generic OF eligibility (Yahoo Fantasy combines LF/CF/RF into OF slot). '
    'Players with can_play_of=True may also have individual LF/CF/RF flags.';

COMMENT ON COLUMN position_eligibility.multi_eligibility_count IS
    'Precomputed count of position booleans set to TRUE. '
    'Used for fast scarcity queries: Bellinger (CF/LF/RF) has count=3, '
    'meaning he hedges scarcity for ALL three outfield slots.';

COMMENT ON COLUMN position_eligibility.scarcity_rank IS
    '1-100 rank within primary_position group. Lower = scarcer. '
    'Computed daily by ingestion job: rank() OVER (PARTITION BY primary_position ORDER BY league_rostered_pct DESC).';

COMMENT ON COLUMN position_eligibility.league_rostered_pct IS
    'Percentage of Yahoo Fantasy leagues where this player is rostered. '
    'Primary input for scarcity calculations.';

COMMENT ON COLUMN position_eligibility.player_type IS
    'Player type: B=batter, P=pitcher. Filters for position-specific queries.';
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_pe_multi_eligible;
DROP INDEX IF EXISTS idx_pe_primary_position;
DROP INDEX IF EXISTS idx_pe_player;
DROP TABLE IF EXISTS position_eligibility;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating position_eligibility ===")

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
        print("SUCCESS: position_eligibility created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing position_eligibility ===")

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
        print("SUCCESS: position_eligibility removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v25 - Position Eligibility")
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
