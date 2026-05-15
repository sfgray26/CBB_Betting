#!/usr/bin/env python3
"""
Migration: Add handedness column to probable_pitchers table

This migration adds a VARCHAR(1) column to store pitcher throwing hand:
- "L" for left-handed pitchers
- "R" for right-handed pitchers
- NULL when handedness is not yet known

Usage:
    python scripts/migrations/add_probable_pitcher_handedness.py
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sqlalchemy import create_engine, text


def run_migration():
    """Add handedness column to probable_pitchers table."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    engine = create_engine(database_url)

    with engine.connect() as conn:
        with conn.begin():
            # Check if column already exists
            check_sql = """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'probable_pitchers'
                AND column_name = 'handedness';
            """
            result = conn.execute(text(check_sql))
            if result.fetchone():
                print("Column 'handedness' already exists in probable_pitchers table")
                return

            # Add the column
            add_sql = """
                ALTER TABLE probable_pitchers
                ADD COLUMN handedness VARCHAR(1);
            """
            conn.execute(text(add_sql))

            # Add comment for documentation
            comment_sql = """
                COMMENT ON COLUMN probable_pitchers.handedness IS
                'Pitcher throwing hand: L (left) or R (right)';
            """
            conn.execute(text(comment_sql))

            print("SUCCESS: Added 'handedness' column to probable_pitchers table")


if __name__ == "__main__":
    run_migration()
