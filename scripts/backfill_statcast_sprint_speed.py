"""
PR 2.4 — Backfill sprint_speed for the 2026 season.

Fetches the Baseball Savant sprint speed leaderboard CSV and updates
statcast_batter_metrics.sprint_speed for every matched player.

Idempotent: running twice produces the same result (UPDATE, not INSERT).

Run: railway run python scripts/backfill_statcast_sprint_speed.py
Or:  DATABASE_URL=<url> python scripts/backfill_statcast_sprint_speed.py
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


def backfill(year: int = 2026) -> None:
    import psycopg2

    from backend.ingestion.savant_scraper import fetch_sprint_speed

    url = get_db_url()

    logger.info("Fetching sprint speed leaderboard for %d...", year)
    df = fetch_sprint_speed(year=year)

    if df.empty:
        logger.error("No sprint speed data returned — aborting backfill.")
        sys.exit(1)

    logger.info("Fetched %d players from Savant.", len(df))

    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()

    updated = 0
    skipped = 0

    try:
        for _, row in df.iterrows():
            mlbam_id_str = str(int(row["mlbam_id"]))
            cur.execute(
                """
                UPDATE statcast_batter_metrics
                SET sprint_speed = %s
                WHERE mlbam_id = %s AND season = %s
                """,
                (float(row["sprint_speed"]), mlbam_id_str, year),
            )
            if cur.rowcount:
                updated += 1
            else:
                skipped += 1

            if (updated + skipped) % 100 == 0:
                logger.info("Progress: %d updated, %d no DB row yet", updated, skipped)

        conn.commit()

    except Exception as exc:
        conn.rollback()
        logger.error("Backfill failed: %s", exc)
        sys.exit(1)
    finally:
        cur.close()

    logger.info(
        "Backfill complete: %d / %d updated, %d had no statcast_batter_metrics row.",
        updated, len(df), skipped,
    )

    # Post-backfill coverage check
    try:
        cur2 = conn.cursor()
        cur2.execute(
            "SELECT COUNT(*) FROM statcast_batter_metrics WHERE season = %s", (year,)
        )
        total = cur2.fetchone()[0]
        cur2.execute(
            "SELECT COUNT(*) FROM statcast_batter_metrics WHERE sprint_speed IS NOT NULL AND season = %s",
            (year,),
        )
        non_null = cur2.fetchone()[0]
        rate = non_null / total if total else 0.0
        logger.info(
            "Coverage: %d / %d rows have sprint_speed (%.1f%%)", non_null, total, rate * 100
        )
        if rate < 0.70:
            logger.warning("Coverage %.1f%% is below 70%% threshold.", rate * 100)
        cur2.close()
    except Exception as exc:
        logger.warning("Coverage check failed: %s", exc)
    finally:
        conn.close()


if __name__ == "__main__":
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    backfill(year=year)
