"""
PR 5.1 — Create matchup_context Schema

Stores per-game matchup context for hitters and pitchers.
"""

import os
import sys
import psycopg2


def get_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("DATABASE_URL="):
                        url = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not url:
        print("ERROR: DATABASE_URL not found")
        sys.exit(1)
    return url


DDL = """
CREATE TABLE IF NOT EXISTS matchup_context (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    game_date DATE NOT NULL,
    opponent_team VARCHAR(10),
    opponent_starter_name VARCHAR(100),
    opponent_starter_hand VARCHAR(1),
    opponent_starter_era FLOAT,
    opponent_starter_whip FLOAT,
    opponent_starter_k_per_nine FLOAT,
    opponent_bullpen_era FLOAT,
    opponent_bullpen_whip FLOAT,
    home_team VARCHAR(10),
    park_factor_runs FLOAT,
    park_factor_hr FLOAT,
    weather_temp_f FLOAT,
    weather_wind_mph FLOAT,
    weather_wind_direction VARCHAR(10),
    hitter_woba_vs_hand FLOAT,
    hitter_k_pct_vs_hand FLOAT,
    hitter_iso_vs_hand FLOAT,
    matchup_score FLOAT,
    matchup_z FLOAT,
    matchup_confidence FLOAT,
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (bdl_player_id, game_date)
);

CREATE INDEX IF NOT EXISTS idx_matchup_context_player_date
    ON matchup_context(bdl_player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_matchup_context_date
    ON matchup_context(game_date);
"""


def migrate():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Applying PR 5.1 migration: matchup_context...")
        cur.execute(DDL)
        conn.commit()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'matchup_context'
        """)
        if cur.fetchone():
            print("  Verified: matchup_context exists.")
        print("PR 5.1 migration ready.")
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
