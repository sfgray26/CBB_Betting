"""
Migration v4: add run_tier to predictions, update unique constraint.

Usage:
    python scripts/migrate_v4.py

This script is safe to run multiple times — it checks whether the column
already exists before attempting to add it, and whether the old constraint
exists before attempting to drop/replace it.

What this adds
--------------
predictions.run_tier  VARCHAR  NOT NULL  DEFAULT 'nightly'
    Distinguishes opener / nightly / closing analysis runs on the same day.
    Combined with (game_id, prediction_date) it forms the new unique key so
    that multiple analysis tiers can coexist without locking each other out.

Constraint change
-----------------
Old: _game_prediction_date_uc (game_id, prediction_date)
New: _game_prediction_date_tier_uc (game_id, prediction_date, run_tier)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from backend.models import engine


def column_exists(conn, table: str, column: str) -> bool:
    result = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name = :t AND column_name = :c"
    ), {"t": table, "c": column})
    return result.fetchone() is not None


def constraint_exists(conn, constraint_name: str) -> bool:
    result = conn.execute(text(
        "SELECT constraint_name FROM information_schema.table_constraints "
        "WHERE constraint_name = :c"
    ), {"c": constraint_name})
    return result.fetchone() is not None


def run_migration():
    with engine.connect() as conn:
        # Step 1: Add run_tier column
        if column_exists(conn, "predictions", "run_tier"):
            print("Column predictions.run_tier already exists — skipping.")
        else:
            conn.execute(text(
                "ALTER TABLE predictions "
                "ADD COLUMN run_tier VARCHAR NOT NULL DEFAULT 'nightly'"
            ))
            print("Added predictions.run_tier (default='nightly')")

        # Step 2: Drop old constraint, create new one
        old_constraint = "_game_prediction_date_uc"
        new_constraint = "_game_prediction_date_tier_uc"

        if constraint_exists(conn, old_constraint):
            conn.execute(text(
                f"ALTER TABLE predictions DROP CONSTRAINT {old_constraint}"
            ))
            print(f"Dropped old constraint {old_constraint}")

        if constraint_exists(conn, new_constraint):
            print(f"Constraint {new_constraint} already exists — skipping.")
        else:
            conn.execute(text(
                "ALTER TABLE predictions "
                "ADD CONSTRAINT _game_prediction_date_tier_uc "
                "UNIQUE (game_id, prediction_date, run_tier)"
            ))
            print(f"Created constraint {new_constraint}")

        conn.commit()
        print("Migration v4 complete.")


if __name__ == "__main__":
    run_migration()
