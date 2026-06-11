"""
Seed the predictive_stats_v1_enabled feature flag (default: disabled).

Run once:
    railway run python scripts/seed_predictive_stats_flag.py
"""

import os
import sys

from sqlalchemy import text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.models import SessionLocal


def main():
    db = SessionLocal()
    try:
        db.execute(text("""
            INSERT INTO feature_flags (flag_name, enabled, description)
            VALUES ('predictive_stats_v1_enabled', FALSE,
                    'Gate FIP/xFIP/SIERA (FanGraphs) and xwOBA/Hard-Hit%/wRC+ (Savant) predictive stats pipeline')
            ON CONFLICT (flag_name) DO NOTHING
        """))
        db.commit()
        print("Seeded feature flag: predictive_stats_v1_enabled=False")
    finally:
        db.close()


if __name__ == "__main__":
    main()
