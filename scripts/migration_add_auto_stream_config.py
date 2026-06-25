"""
Migration: Add auto_stream_config column to UserPreferences

Run via: railway run python scripts/migration_add_auto_stream_config.py
"""
import os
import sys
from sqlalchemy import text

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.models import SessionLocal

def migrate():
    """Add auto_stream_config column to user_preferences table."""
    db = SessionLocal()
    try:
        # Check if column already exists
        check_sql = text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'user_preferences'
            AND column_name = 'auto_stream_config'
        """)
        result = db.execute(check_sql).fetchone()

        if result:
            print("Column auto_stream_config already exists. Skipping migration.")
            return

        # Add the column
        alter_sql = text("""
            ALTER TABLE user_preferences
            ADD COLUMN auto_stream_config JSONB
            DEFAULT '{"enabled": false, "min_confidence": "HIGH", "min_recommendation": "EXCELLENT", "max_adds_per_week": 2, "drop_priority": [], "updated_at": null}'::jsonb
        """)
        db.execute(alter_sql)

        # Add comment
        comment_sql = text("""
            COMMENT ON COLUMN user_preferences.auto_stream_config
            IS 'Auto-Stream feature configuration: enabled, drop_priority, min_confidence, min_recommendation, max_adds_per_week'
        """)
        db.execute(comment_sql)

        db.commit()
        print("Migration completed successfully: Added auto_stream_config column to user_preferences")

    except Exception as e:
        db.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    migrate()
