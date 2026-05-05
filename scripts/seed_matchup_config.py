"""
PR 5.2 — Seed matchup context engine config into threshold_config.

Seeds 9 keys used by matchup_engine.py and daily_lineup_optimizer.py.
Idempotent: ON CONFLICT DO NOTHING — running twice is safe.
feature.matchup_enabled is seeded as false (opt-in activation after validation).

Run: railway run python scripts/seed_matchup_config.py
"""
import json
import os
import sys


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
    # Component weights (must sum to 1.0)
    ("matchup.weight.handedness", 0.35, "global", "Hitter platoon split weight in matchup score"),
    ("matchup.weight.pitcher",    0.25, "global", "Opponent starter ERA/WHIP weight in matchup score"),
    ("matchup.weight.park",       0.15, "global", "Park factor weight in matchup score"),
    ("matchup.weight.weather",    0.10, "global", "Weather bonus weight in matchup score"),
    ("matchup.weight.bullpen",    0.15, "global", "Opponent bullpen ERA weight in matchup score"),
    # Boost parameters (applied in lineup optimizer)
    ("matchup.boost.cap",         0.2,  "global", "Max fractional lineup score adjustment (+/-)"),
    ("matchup.boost.z_scale",     0.1,  "global", "Multiplier: matchup_z → fractional boost before cap"),
    # Confidence gate
    ("matchup.confidence_gate",   0.4,  "global", "Min confidence below which boost is zeroed out"),
    # Feature flag — false until validated in production
    ("feature.matchup_enabled",   False, "global", "Enable matchup context boost in lineup optimizer"),
]


def seed() -> None:
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    url = get_db_url()
    conn = psycopg2.connect(url)
    conn.autocommit = False
    cur = conn.cursor()

    inserted = 0
    skipped = 0

    try:
        for key, value, scope, desc in SEEDS:
            cur.execute(
                """
                INSERT INTO threshold_config (config_key, config_value, scope, description)
                VALUES (%s, %s::jsonb, %s, %s)
                ON CONFLICT (config_key, scope) DO NOTHING
                """,
                (key, json.dumps(value), scope, desc),
            )
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
        print(f"ERROR during seed: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    seed()
