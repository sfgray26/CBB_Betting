"""
PR 2.x — Seed feature flags for Statcast advanced metrics.

Adds feature flags for:
  - statcast_sprint_speed_enabled (already exists from PR 2.3)
  - statcast_stuff_plus_enabled (NEW)
  - statcast_location_plus_enabled (NEW)
  - statcast_advanced_enabled (NEW — master flag for all advanced metrics)

Run: railway run python scripts/seed_statcast_advanced_flags.py
Or:  DATABASE_URL=<url> python scripts/seed_statcast_advanced_flags.py
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path so backend imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if line.startswith("DATABASE_URL="):
                        url = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not url:
        logger.error("DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


def seed_flags() -> None:
    import psycopg2

    url = get_db_url()

    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        # Feature flags to seed
        flags = [
            (
                "statcast_sprint_speed_enabled",
                True,
                "Enable sprint_speed in scoring (PR 2.2-2.4)"
            ),
            (
                "statcast_stuff_plus_enabled",
                False,  # Disabled by default until backfill verified
                "Enable stuff_plus signals for pitchers (PR 2.x)"
            ),
            (
                "statcast_location_plus_enabled",
                False,  # Disabled by default until backfill verified
                "Enable location_plus signals for pitchers (PR 2.x)"
            ),
            (
                "statcast_advanced_enabled",
                False,  # Master flag - controls all advanced metrics
                "Enable all Statcast advanced metrics (sprint_speed, stuff_plus, location_plus)"
            ),
        ]

        seeded = 0
        for flag_name, enabled, description in flags:
            cur.execute(
                """
                INSERT INTO feature_flags (flag_name, enabled, description)
                VALUES (%s, %s, %s)
                ON CONFLICT (flag_name) DO UPDATE SET
                    description = EXCLUDED.description
                RETURNING flag_name
                """,
                (flag_name, enabled, description)
            )
            result = cur.fetchone()
            if result:
                seeded += 1
                logger.info("Seeded flag: %s (enabled=%s)", flag_name, enabled)

        conn.commit()
        logger.info("Successfully seeded %d feature flags", seeded)

    except Exception as exc:
        conn.rollback()
        logger.error("Failed to seed feature flags: %s", exc)
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    seed_flags()
