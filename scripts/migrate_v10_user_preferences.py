#!/usr/bin/env python
"""
EMAC-090 Phase B — User Preferences Schema Migration v10

Adds user_preferences table for dashboard customization.

Usage:
    python scripts/migrate_v10_user_preferences.py              # run upgrade
    python scripts/migrate_v10_user_preferences.py --downgrade  # run downgrade
    python scripts/migrate_v10_user_preferences.py --dry-run    # print SQL, no execute
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- Create user_preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL UNIQUE,
    user_email VARCHAR(255),
    notifications JSONB NOT NULL DEFAULT '{
        "lineup_deadline": true,
        "injury_alerts": true,
        "waiver_suggestions": true,
        "trade_offers": false,
        "hot_streak_alerts": true,
        "channels": ["discord"],
        "discord_user_id": null,
        "email_enabled": false
    }'::jsonb,
    dashboard_layout JSONB NOT NULL DEFAULT '{
        "panels": [
            {"id": "lineup_gaps", "position": "top-left", "size": "medium", "enabled": true},
            {"id": "hot_cold_streaks", "position": "top-right", "size": "medium", "enabled": true},
            {"id": "waiver_targets", "position": "middle-left", "size": "medium", "enabled": true},
            {"id": "injury_flags", "position": "middle-right", "size": "small", "enabled": true},
            {"id": "matchup_preview", "position": "bottom-left", "size": "medium", "enabled": true},
            {"id": "probable_pitchers", "position": "bottom-right", "size": "small", "enabled": true}
        ],
        "refresh_interval_seconds": 300,
        "theme": "dark"
    }'::jsonb,
    projection_weights JSONB NOT NULL DEFAULT '{
        "steamer": 0.30,
        "zips": 0.25,
        "depth_charts": 0.20,
        "atc": 0.15,
        "the_bat": 0.10
    }'::jsonb,
    streak_settings JSONB NOT NULL DEFAULT '{
        "hot_threshold": 0.5,
        "cold_threshold": -0.5,
        "min_sample_days": 7,
        "rolling_windows": [7, 14, 30]
    }'::jsonb,
    waiver_preferences JSONB NOT NULL DEFAULT '{
        "min_percent_owned": 0,
        "max_percent_owned": 60,
        "positions_of_need": [],
        "priority_categories": [],
        "hide_injured": true,
        "streamer_threshold": 0.3
    }'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes (removed CONCURRENTLY to avoid transaction block issues)
CREATE INDEX IF NOT EXISTS idx_user_prefs_user_id 
    ON user_preferences (user_id);
CREATE INDEX IF NOT EXISTS idx_user_prefs_email 
    ON user_preferences (user_email) WHERE user_email IS NOT NULL;

-- Add comment for documentation
COMMENT ON TABLE user_preferences IS 'User-customizable settings for fantasy baseball dashboard (Phase B)';
"""

DOWNGRADE_SQL = """
-- Remove user_preferences table
DROP INDEX IF EXISTS idx_user_prefs_user_id;
DROP INDEX IF EXISTS idx_user_prefs_email;
DROP TABLE IF EXISTS user_preferences;
"""


def upgrade(engine, dry_run=False):
    """Apply the migration."""
    print("=== UPGRADE: Creating user_preferences table ===")
    
    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return
    
    # Use autocommit mode to avoid transaction block issues with CREATE INDEX
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        # Execute each statement separately for better error handling
        for statement in UPGRADE_SQL.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    # Ignore "already exists" errors
                    if "already exists" in str(e).lower():
                        print(f"  WARNING: Skipping (already exists): {str(e)[:80]}")
                    else:
                        raise
        
        print("SUCCESS: user_preferences table created successfully")


def downgrade(engine, dry_run=False):
    """Rollback the migration."""
    print("=== DOWNGRADE: Removing user_preferences table ===")
    
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
                    print(f"  ⚠️  Warning: {str(e)[:80]}")
        
        print("SUCCESS: user_preferences table removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v10 - User Preferences")
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
