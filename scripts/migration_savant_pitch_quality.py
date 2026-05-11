"""
Create persistence for Savant Pitch Quality scores.

Run:
    railway run python scripts/migration_savant_pitch_quality.py
or:
    DATABASE_URL=<url> python scripts/migration_savant_pitch_quality.py
"""
from __future__ import annotations

import logging
import os
import sys

import psycopg2

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("DATABASE_URL="):
                        url = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not url:
        logger.error("DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


def run_migration() -> None:
    conn = psycopg2.connect(get_db_url())
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS savant_pitch_quality_scores (
                id SERIAL PRIMARY KEY,
                player_id VARCHAR(50) NOT NULL,
                player_name VARCHAR(100) NOT NULL,
                team VARCHAR(10),
                season INTEGER NOT NULL,
                as_of_date DATE NOT NULL,
                savant_pitch_quality DOUBLE PRECISION NOT NULL,
                arsenal_quality DOUBLE PRECISION,
                bat_missing_skill DOUBLE PRECISION,
                contact_suppression DOUBLE PRECISION,
                command_stability DOUBLE PRECISION,
                trend_adjustment DOUBLE PRECISION,
                sample_confidence DOUBLE PRECISION,
                signals JSONB NOT NULL DEFAULT '[]'::jsonb,
                inputs JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                CONSTRAINT uq_savant_pitch_quality_player_season_date
                    UNIQUE (player_id, season, as_of_date)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_spq_score_date
            ON savant_pitch_quality_scores (as_of_date, savant_pitch_quality)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_spq_player_season
            ON savant_pitch_quality_scores (player_id, season)
            """
        )
        conn.commit()
        logger.info("savant_pitch_quality migration ready.")
    except Exception as exc:
        conn.rollback()
        logger.error("savant_pitch_quality migration failed: %s", exc)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    run_migration()
