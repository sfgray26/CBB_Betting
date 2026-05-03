#!/usr/bin/env python
"""Backfill player_type from positions JSONB."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

from sqlalchemy import text
from backend.models import SessionLocal

def main():
    db = SessionLocal()

    try:
        # Check current state
        print("Current player_type distribution:")
        rows = db.execute(text('''
          SELECT player_type, COUNT(*)
          FROM player_projections
          GROUP BY player_type
          ORDER BY player_type NULLS FIRST
        ''')).fetchall()

        for r in rows:
            print(f"  {r[0] if r[0] else 'NULL':10} {r[1]:5}")

        # Backfill player_type from positions
        print("\nBackfilling player_type from positions JSONB...")
        updated = db.execute(text('''
          UPDATE player_projections
          SET player_type = CASE
            WHEN positions ? ANY(array['SP','RP','P']) THEN 'pitcher'
            ELSE 'hitter'
          END
          WHERE player_type IS NULL
        ''')).rowcount

        db.commit()
        print(f"Updated {updated} rows")

        # Verify after backfill
        print("\nNew player_type distribution:")
        rows = db.execute(text('''
          SELECT player_type, COUNT(*)
          FROM player_projections
          GROUP BY player_type
          ORDER BY player_type
        ''')).fetchall()

        for r in rows:
            print(f"  {r[0]:10} {r[1]:5}")

        # Check for any remaining NULLs
        nulls = db.execute(text('''
          SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL
        ''')).scalar()

        if nulls == 0:
            print("\nSUCCESS: All NULLs backfilled")
            return 0
        else:
            print(f"\nWARNING: {nulls} NULLs remain (positions JSONB missing)")
            return 1

    except Exception as e:
        db.rollback()
        print(f"\nERROR: {e}")
        return 1
    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(main())
