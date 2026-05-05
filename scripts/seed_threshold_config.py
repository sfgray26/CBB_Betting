"""
PR 1.4 — Backfill threshold_config with default values.

Seeds all constants that were wired to get_threshold() in PR 1.3.
Idempotent: ON CONFLICT DO NOTHING — running twice is safe.

Run: railway run python scripts/seed_threshold_config.py
"""
import os
import sys
import psycopg2
import json


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
        print("ERROR: DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


SEEDS = [
    # scoring_engine constants (PR 1.3)
    ("scoring.z_cap",           3.0,  "global", "Winsorization cap for category Z-scores"),
    ("scoring.min_sample",      3,    "global", "Min players with non-null value before computing Z"),
    ("scoring.rate_stat_protect", 0.5, "global", "Rate-stat protection gate threshold (z-score units)"),

    # momentum_engine deprecated delta-z thresholds (legacy, not used in active logic)
    ("momentum.surging.delta_z",    0.5,  "global", "DEPRECATED delta-Z threshold for SURGING signal"),
    ("momentum.hot.delta_z",        0.2,  "global", "DEPRECATED delta-Z threshold for HOT signal"),
    ("momentum.cold.delta_z",      -0.2,  "global", "DEPRECATED delta-Z threshold for COLD signal"),
    ("momentum.collapsing.delta_z", -0.5, "global", "DEPRECATED delta-Z threshold for COLLAPSING signal"),

    # momentum_engine active percentile thresholds (PR 1.3)
    ("momentum.top_pct.surging",          0.90, "global", "Top percentile cutoff for SURGING (top 10%)"),
    ("momentum.top_pct.hot",              0.70, "global", "Top percentile cutoff for HOT (top 30%)"),
    ("momentum.bot_pct.cold",             0.30, "global", "Bottom percentile cutoff for COLD (bottom 30%)"),
    ("momentum.bot_pct.collapsing",       0.10, "global", "Bottom percentile cutoff for COLLAPSING (bottom 10%)"),
    ("momentum.level_gate.bottom_quartile", 25.0, "global", "Percentile rank below which SURGING is blocked"),

    # waiver streamer threshold (referenced in spec)
    ("waiver.streamer_threshold", 0.3, "global", "Z-score threshold for streamer suggestions"),
]


def seed():
    url = get_db_url()
    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()

    inserted = 0
    skipped = 0

    try:
        for key, value, scope, desc in SEEDS:
            cur.execute("""
                INSERT INTO threshold_config (config_key, config_value, scope, description)
                VALUES (%s, %s::jsonb, %s, %s)
                ON CONFLICT (config_key, scope) DO NOTHING
            """, (key, json.dumps(value), scope, desc))
            if cur.rowcount:
                inserted += 1
                print(f"  seeded: {key} = {value}")
            else:
                skipped += 1
                print(f"  exists: {key} (skipped)")

        conn.commit()
        print(f"\nSeed complete: {inserted} inserted, {skipped} already present.")

        cur.execute("SELECT COUNT(*) FROM threshold_config")
        total = cur.fetchone()[0]
        print(f"Total threshold_config rows: {total}")

    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    seed()
