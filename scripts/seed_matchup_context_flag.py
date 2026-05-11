"""Seed feature_matchup_enabled into feature_flags (default: disabled)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import SessionLocal
from sqlalchemy import text


def main():
    db = SessionLocal()
    try:
        db.execute(
            text(
                "INSERT INTO feature_flags (flag_name, enabled) "
                "VALUES ('feature_matchup_enabled', false) "
                "ON CONFLICT (flag_name) DO NOTHING"
            )
        )
        db.commit()
        row = db.execute(
            text(
                "SELECT flag_name, enabled FROM feature_flags "
                "WHERE flag_name = 'feature_matchup_enabled'"
            )
        ).fetchone()
        print(f"Flag state: {row}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
