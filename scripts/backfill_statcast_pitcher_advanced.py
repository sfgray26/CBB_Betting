"""
PR 2.x — Backfill stuff_plus and location_plus for the 2026 season.

Fetches the Baseball Savant pitching leaderboard CSV and updates
statcast_pitcher_metrics.stuff_plus and statcast_pitcher_metrics.location_plus
for every matched pitcher.

Idempotent: running twice produces the same result (UPDATE, not INSERT).

Run: railway run python scripts/backfill_statcast_pitcher_advanced.py
Or:  DATABASE_URL=<url> python scripts/backfill_statcast_pitcher_advanced.py
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

    from backend.ingestion.savant_scraper import fetch_pitcher_advanced

    url = get_db_url()

    logger.info("Fetching pitching advanced leaderboard for %d...", year)
    df = fetch_pitcher_advanced(year=year)

    if df.empty:
        logger.error("No pitching advanced data returned — aborting backfill.")
        sys.exit(1)

    logger.info("Fetched %d pitchers from Savant.", len(df))

    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()

    updated_stuff = 0
    updated_location = 0
    skipped = 0

    try:
        for _, row in df.iterrows():
            mlbam_id_str = str(int(row["mlbam_id"]))
            stuff_plus = row.get("stuff_plus")
            location_plus = row.get("location_plus")

            # Update stuff_plus if not NULL
            if pd.notna(stuff_plus):
                cur.execute(
                    """
                    UPDATE statcast_pitcher_metrics
                    SET stuff_plus = %s
                    WHERE mlbam_id = %s AND season = %s
                    """,
                    (float(stuff_plus), mlbam_id_str, year),
                )
                if cur.rowcount:
                    updated_stuff += 1

            # Update location_plus if not NULL
            if pd.notna(location_plus):
                cur.execute(
                    """
                    UPDATE statcast_pitcher_metrics
                    SET location_plus = %s
                    WHERE mlbam_id = %s AND season = %s
                    """,
                    (float(location_plus), mlbam_id_str, year),
                )
                if cur.rowcount:
                    updated_location += 1

            if (updated_stuff + updated_location + skipped) % 100 == 0:
                logger.info(
                    "Progress: stuff_plus=%d, location_plus=%d, no DB row yet=%d",
                    updated_stuff, updated_location, skipped
                )

        conn.commit()

    except Exception as exc:
        conn.rollback()
        logger.error("Backfill failed: %s", exc)
        sys.exit(1)
    finally:
        cur.close()

    logger.info(
        "Backfill complete: stuff_plus=%d / %d, location_plus=%d / %d, %d had no statcast_pitcher_metrics row.",
        updated_stuff, len(df), updated_location, len(df), skipped
    )

    # Post-backfill coverage check
    try:
        cur2 = conn.cursor()
        cur2.execute(
            "SELECT COUNT(*) FROM statcast_pitcher_metrics WHERE season = %s", (year,)
        )
        total = cur2.fetchone()[0]

        cur2.execute(
            "SELECT COUNT(*) FROM statcast_pitcher_metrics WHERE stuff_plus IS NOT NULL AND season = %s",
            (year,),
        )
        non_null_stuff = cur2.fetchone()[0]

        cur2.execute(
            "SELECT COUNT(*) FROM statcast_pitcher_metrics WHERE location_plus IS NOT NULL AND season = %s",
            (year,),
        )
        non_null_location = cur2.fetchone()[0]

        stuff_rate = non_null_stuff / total if total else 0.0
        location_rate = non_null_location / total if total else 0.0

        logger.info(
            "Coverage: stuff_plus %d / %d (%.1f%%), location_plus %d / %d (%.1f%%)",
            non_null_stuff, total, stuff_rate * 100,
            non_null_location, total, location_rate * 100
        )

        if stuff_rate < 0.70:
            logger.warning("stuff_plus coverage %.1f%% is below 70%% threshold.", stuff_rate * 100)
        if location_rate < 0.70:
            logger.warning("location_plus coverage %.1f%% is below 70%% threshold.", location_rate * 100)

        cur2.close()
    except Exception as exc:
        logger.warning("Coverage check failed: %s", exc)
    finally:
        conn.close()


if __name__ == "__main__":
    import pandas as pd  # need pandas for pd.notna()
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2026
    backfill(year=year)
