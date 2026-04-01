#!/usr/bin/env python
"""
ARCH-001 Phase 2 — Player Valuation Cache v12

Adds player_valuation_cache table for pre-computed PlayerValuationReports.
Worker writes at 6 AM ET; API reads. Stale records soft-deleted via invalidated_at.

Usage:
    python scripts/migrate_v12_valuation_cache.py              # run upgrade
    python scripts/migrate_v12_valuation_cache.py --downgrade  # run downgrade
    python scripts/migrate_v12_valuation_cache.py --dry-run    # print SQL, no execute
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS player_valuation_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    target_date DATE NOT NULL,
    league_key VARCHAR(100) NOT NULL,
    report JSONB NOT NULL,
    computed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    invalidated_at TIMESTAMPTZ,
    data_as_of TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pvc_player_date_league
    ON player_valuation_cache (player_id, target_date, league_key)
    WHERE invalidated_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_pvc_league_date
    ON player_valuation_cache (league_key, target_date DESC)
    WHERE invalidated_at IS NULL;

COMMENT ON TABLE player_valuation_cache IS 'Pre-computed PlayerValuationReport per player per day. Worker writes at 6 AM ET; API reads. Stale records soft-deleted via invalidated_at.';
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_pvc_league_date;
DROP INDEX IF EXISTS idx_pvc_player_date_league;
DROP TABLE IF EXISTS player_valuation_cache;
"""


def upgrade(engine, dry_run=False):
    """Apply the migration."""
    print("=== UPGRADE: Creating player_valuation_cache table ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    # Use autocommit mode to avoid transaction block issues with CREATE INDEX
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text(UPGRADE_SQL))
            print("SUCCESS: player_valuation_cache table created successfully")
        except Exception as e:
            # Ignore "already exists" errors
            if "already exists" in str(e).lower():
                print(f"  WARNING: Skipping (already exists): {str(e)[:80]}")
            else:
                raise


def downgrade(engine, dry_run=False):
    """Rollback the migration."""
    print("=== DOWNGRADE: Removing player_valuation_cache table ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(DOWNGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    # Use autocommit mode to avoid transaction block issues
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in DOWNGRADE_SQL.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    print(f"  WARNING: {str(e)[:80]}")

        print("SUCCESS: player_valuation_cache table removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v12 - Player Valuation Cache")
    parser.add_argument("--downgrade", action="store_true", help="Rollback migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    # Get database URL from environment
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    engine = create_engine(db_url)

    if args.downgrade:
        downgrade(engine, dry_run=args.dry_run)
    else:
        upgrade(engine, dry_run=args.dry_run)
