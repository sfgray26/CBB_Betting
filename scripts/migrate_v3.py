"""
Migration v3: add actual_margin to the predictions table.

Usage:
    python scripts/migrate_v3.py

This script is safe to run multiple times — it checks whether the column
already exists before attempting to add it.

What this adds
--------------
predictions.actual_margin  FLOAT  NULL
    Populated automatically by update_completed_games() (runs every 2 hours)
    when a game's score is ingested from The Odds API.  Enables:
      - Model margin-prediction MAE tracking
      - Probability calibration (predicted win prob vs actual outcome)
      - Per-rating-source accuracy metrics in the daily performance snapshot
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


def run_migration():
    with engine.connect() as conn:
        if column_exists(conn, "predictions", "actual_margin"):
            print("Column predictions.actual_margin already exists — nothing to do.")
        else:
            conn.execute(text(
                "ALTER TABLE predictions ADD COLUMN actual_margin FLOAT"
            ))
            conn.commit()
            print("Added predictions.actual_margin")

        print("Migration v3 complete.")


if __name__ == "__main__":
    run_migration()
