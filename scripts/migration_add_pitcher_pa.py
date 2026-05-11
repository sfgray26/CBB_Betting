"""
Add pa (batters faced) column to statcast_pitcher_metrics.

Baseball Savant's /leaderboard/custom?type=pitcher endpoint returns pa for all
pitchers but ip is always empty. The savant_pitch_quality scoring engine uses
ip to compute sample_confidence; without it every pitcher scores exactly 100.0.

Fix: store pa from Savant and use it as IP proxy in _sample_confidence:
    ip_proxy = pa / 4.3  (~4.3 batters per inning)

Usage:
    railway run python scripts/migration_add_pitcher_pa.py
    DATABASE_URL=<url> python scripts/migration_add_pitcher_pa.py
"""
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_db_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return url


def run_migration() -> None:
    conn = psycopg2.connect(get_db_url())
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # Add pa column — idempotent
        cur.execute("""
            ALTER TABLE statcast_pitcher_metrics
            ADD COLUMN IF NOT EXISTS pa INTEGER;
        """)

        # Verify
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'statcast_pitcher_metrics'
              AND column_name = 'pa';
        """)
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Column pa not found after ALTER TABLE")

        conn.commit()
        logger.info("Migration complete: statcast_pitcher_metrics.pa column ready (%s)", row[1])

    except Exception as exc:
        conn.rollback()
        logger.error("Migration failed: %s", exc)
        raise
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
