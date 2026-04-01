"""
Migration: Add ensemble blend columns to player_daily_metrics.
Phase 2.2 — In-Season Projection Pipeline

Applies scripts/migrations/add_blend_columns.sql via SQLAlchemy.
Safe to run multiple times — uses ADD COLUMN IF NOT EXISTS.

Usage (Railway):
    railway run python scripts/migrate_v11_blend_columns.py
"""
import sys
import os

# Allow execution from repo root without modifying the installed package
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from backend.models import SessionLocal

MIGRATION_SQL = """
ALTER TABLE player_daily_metrics
    ADD COLUMN IF NOT EXISTS blend_hr FLOAT,
    ADD COLUMN IF NOT EXISTS blend_rbi FLOAT,
    ADD COLUMN IF NOT EXISTS blend_avg FLOAT,
    ADD COLUMN IF NOT EXISTS blend_era FLOAT,
    ADD COLUMN IF NOT EXISTS blend_whip FLOAT;
"""

VERIFY_SQL = """
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'player_daily_metrics'
  AND column_name IN ('blend_hr', 'blend_rbi', 'blend_avg', 'blend_era', 'blend_whip')
ORDER BY column_name;
"""


def run():
    db = SessionLocal()
    try:
        print("Applying migration: add blend columns to player_daily_metrics ...")
        db.execute(text(MIGRATION_SQL))
        db.commit()
        print("Migration applied.")

        result = db.execute(text(VERIFY_SQL))
        found = [row[0] for row in result]
        expected = {"blend_avg", "blend_era", "blend_hr", "blend_rbi", "blend_whip"}
        missing = expected - set(found)

        if missing:
            print(f"ERROR: columns still missing after migration: {sorted(missing)}", file=sys.stderr)
            sys.exit(1)

        print(f"Verified columns present: {sorted(found)}")
        print("Migration complete. Dashboard should recover on next request.")
    except Exception as exc:
        db.rollback()
        print(f"Migration FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    run()
