"""
PR 3.1 — Create player_opportunity Schema

Stores playing-time intelligence per player per day:
  - Lineup entropy, platoon risk, role certainty
  - Opportunity score, Z-score, confidence
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
CREATE TABLE IF NOT EXISTS player_opportunity (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    as_of_date DATE NOT NULL,

    pa_per_game FLOAT,
    ab_per_game FLOAT,
    games_played_14d INTEGER,
    games_started_14d INTEGER,
    games_started_pct FLOAT,

    lineup_slot_avg FLOAT,
    lineup_slot_mode INTEGER,
    lineup_slot_entropy FLOAT,

    pa_vs_lhp_14d INTEGER,
    pa_vs_rhp_14d INTEGER,
    platoon_ratio FLOAT,
    platoon_risk_score FLOAT,

    appearances_14d INTEGER,
    saves_14d INTEGER,
    holds_14d INTEGER,
    role_certainty_score FLOAT,

    days_since_last_game INTEGER,
    il_stint_flag BOOLEAN,

    opportunity_score FLOAT,
    opportunity_z FLOAT,
    opportunity_confidence FLOAT,

    fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    UNIQUE (bdl_player_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_player_opp_bdl_date
    ON player_opportunity(bdl_player_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_player_opp_date
    ON player_opportunity(as_of_date);
CREATE INDEX IF NOT EXISTS idx_player_opp_opportunity_z
    ON player_opportunity(as_of_date, opportunity_z);
"""


def migrate():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Applying PR 3.1 migration: player_opportunity...")
        cur.execute(DDL)
        conn.commit()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'player_opportunity'
        """)
        if cur.fetchone():
            print("  Verified: player_opportunity exists.")
        print("PR 3.1 migration ready.")
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
