"""
Migration v7: Fantasy Baseball tables — draft sessions, draft picks, lineups.

=============================================================================
HOW TO RUN
=============================================================================

  python scripts/migrate_v7.py

Or on Railway:
  railway run python scripts/migrate_v7.py

=============================================================================
WHAT THIS MIGRATION DOES
=============================================================================

Creates three tables (idempotent — safe to run multiple times):

  fantasy_draft_sessions  — one row per draft session (league + snake config)
  fantasy_draft_picks     — one row per pick recorded during a session
  fantasy_lineups         — one row per daily saved lineup (season ops)

These tables are already defined in backend/models.py as ORM classes.
This script creates them in PostgreSQL using SQLAlchemy's metadata API,
which is identical to the model definitions — no drift risk.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text, inspect
from backend.models import engine, Base

# Import the models so their metadata is registered
from backend.models import FantasyDraftSession, FantasyDraftPick, FantasyLineup  # noqa: F401

FANTASY_TABLES = [
    "fantasy_draft_sessions",
    "fantasy_draft_picks",
    "fantasy_lineups",
]


def run_migration() -> None:
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())

    with engine.connect() as conn:
        for table_name in FANTASY_TABLES:
            if table_name in existing:
                print(f"[SKIP] Table {table_name} already exists")
            else:
                # Use SQLAlchemy metadata to create just this table
                table_obj = Base.metadata.tables.get(table_name)
                if table_obj is not None:
                    table_obj.create(bind=engine, checkfirst=True)
                    print(f"[ OK ] Created table {table_name}")
                else:
                    print(f"[WARN] Table {table_name} not found in metadata — skipping")

        # Verify
        print("\nVerification:")
        inspector2 = inspect(engine)
        current = set(inspector2.get_table_names())
        for t in FANTASY_TABLES:
            status = "EXISTS" if t in current else "MISSING"
            print(f"  {t}: {status}")

        conn.commit()

    print("""
=============================================================================
Migration v7 complete.
=============================================================================

NEXT STEPS:
  1. Restart the backend:
       uvicorn backend.main:app --reload

  2. Test the draft board endpoint:
       curl http://localhost:8000/api/fantasy/draft-board -H "X-API-Key: <key>"

  3. Create a draft session:
       curl -X POST "http://localhost:8000/api/fantasy/draft-session?my_draft_position=7" \\
            -H "X-API-Key: <key>"
=============================================================================
""")


if __name__ == "__main__":
    run_migration()
