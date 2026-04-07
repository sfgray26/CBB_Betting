#!/usr/bin/env python
"""
P10 — Player Identity Mapping Table (v14)

Creates player_id_mapping for cross-system player identity resolution.
Maps Yahoo player keys, BDL player IDs, and mlbam IDs to canonical rows.

Key design decisions (from K-B spec):
  - yahoo_key "469.p.7590" format: {game_id}.p.{yahoo_id} — NOT mlb.p.{mlbam_id}
  - yahoo_id "7590" is a PROPRIETARY Yahoo ID — does NOT match mlbam_id
  - bdl_id is BDL internal integer — does NOT match mlbam_id
  - mlbam_id is the canonical cross-platform identifier (MLB Advanced Media)
  - normalized_name enables fuzzy matching (Unicode-normalized, lowercase)

Resolution approach:
  1. Cache hit on yahoo_key or bdl_id -> return mlbam_id
  2. pybaseball.playerid_lookup() -> cache with source='pybaseball'
  3. Manual override (source='manual') -> highest trust

Usage:
    python scripts/migrate_v14_player_id_mapping.py              # run upgrade
    python scripts/migrate_v14_player_id_mapping.py --downgrade  # run downgrade
    python scripts/migrate_v14_player_id_mapping.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS player_id_mapping (
    id                    SERIAL       PRIMARY KEY,
    yahoo_key             VARCHAR(50)  UNIQUE,           -- "469.p.7590" -- nullable until known
    yahoo_id              VARCHAR(20),                   -- "7590" -- proprietary Yahoo ID
    mlbam_id              INTEGER,                       -- MLB Advanced Media canonical ID
    bdl_id                INTEGER,                       -- BDL player.id internal
    full_name             VARCHAR(150) NOT NULL,
    normalized_name       VARCHAR(150) NOT NULL,         -- lowercase, no accents
    source                VARCHAR(20)  NOT NULL DEFAULT 'manual', -- pybaseball|manual|api
    resolution_confidence FLOAT,                         -- 0.0-1.0 for fuzzy matches
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_verified         DATE
);

CREATE INDEX IF NOT EXISTS idx_pim_mlbam
    ON player_id_mapping (mlbam_id)
    WHERE mlbam_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pim_bdl
    ON player_id_mapping (bdl_id)
    WHERE bdl_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pim_normalized
    ON player_id_mapping (normalized_name);

CREATE INDEX IF NOT EXISTS idx_pim_yahoo_id
    ON player_id_mapping (yahoo_id)
    WHERE yahoo_id IS NOT NULL;

COMMENT ON TABLE player_id_mapping IS
    'Cross-system player identity cache. One row per player. '
    'yahoo_key format: "469.p.7590" (game_id.p.yahoo_id). '
    'yahoo_id is PROPRIETARY -- does NOT equal mlbam_id. '
    'Seeded via pybaseball.playerid_lookup() + manual overrides. '
    'mlbam_id is the canonical cross-platform identifier.';

COMMENT ON COLUMN player_id_mapping.yahoo_key IS
    'Yahoo composite key: "{game_id}.p.{yahoo_id}" e.g. "469.p.7590". '
    'UNIQUE where present. 469 is the Yahoo MLB game_id, not an ID type prefix.';

COMMENT ON COLUMN player_id_mapping.yahoo_id IS
    'Proprietary Yahoo player ID e.g. "7590". '
    'NOT the same as mlbam_id. Cannot be used for cross-platform lookup.';

COMMENT ON COLUMN player_id_mapping.normalized_name IS
    'Unicode-normalized lowercase name (NFKD, combining chars stripped). '
    'Used for fuzzy name matching across BDL/Yahoo/pybaseball.';

COMMENT ON COLUMN player_id_mapping.resolution_confidence IS
    '1.0 = exact match or manual. <1.0 = fuzzy match score. '
    'NULL = unresolved. Used to flag low-confidence rows for manual review.';
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_pim_yahoo_id;
DROP INDEX IF EXISTS idx_pim_normalized;
DROP INDEX IF EXISTS idx_pim_bdl;
DROP INDEX IF EXISTS idx_pim_mlbam;
DROP TABLE IF EXISTS player_id_mapping;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating player_id_mapping ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text(UPGRADE_SQL))
            print("SUCCESS: player_id_mapping created")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  WARNING: Skipping (already exists): {str(e)[:100]}")
            else:
                raise


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing player_id_mapping ===")

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
        print("SUCCESS: player_id_mapping removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v14 - Player ID Mapping")
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
