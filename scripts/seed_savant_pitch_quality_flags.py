"""
Seed disabled feature flags for Savant Pitch Quality rollout.

Run:
    railway run python scripts/seed_savant_pitch_quality_flags.py
or:
    DATABASE_URL=<url> python scripts/seed_savant_pitch_quality_flags.py
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


def seed_flags() -> None:
    flags = [
        (
            "savant_pitch_quality_enabled",
            False,
            "Enable Savant Pitch Quality score availability after validation",
        ),
        (
            "savant_pitch_quality_waiver_signals_enabled",
            False,
            "Enable Savant Pitch Quality waiver/breakout signals",
        ),
        (
            "savant_pitch_quality_projection_adjustments_enabled",
            False,
            "Enable Bayesian projection adjustments from Savant Pitch Quality",
        ),
    ]

    conn = psycopg2.connect(get_db_url())
    conn.autocommit = False
    cur = conn.cursor()
    try:
        for flag_name, enabled, description in flags:
            cur.execute(
                """
                INSERT INTO feature_flags (flag_name, enabled, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (flag_name) DO UPDATE SET
                    description = EXCLUDED.description
                """,
                (flag_name, enabled, description),
            )
            logger.info("Seeded flag metadata: %s (default=%s)", flag_name, enabled)
        conn.commit()
        logger.info("Successfully seeded %d Savant Pitch Quality flags", len(flags))
    except Exception as exc:
        conn.rollback()
        logger.error("Failed to seed Savant Pitch Quality flags: %s", exc)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    seed_flags()
