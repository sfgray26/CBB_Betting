"""
PR 4.1 — Create player_market_signals Schema

Stores Yahoo ownership trends and market-derived signals.
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
CREATE TABLE IF NOT EXISTS player_market_signals (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER NOT NULL REFERENCES player_id_mapping(bdl_id),
    as_of_date DATE NOT NULL,

    yahoo_owned_pct FLOAT,
    yahoo_owned_pct_7d_ago FLOAT,
    yahoo_owned_pct_30d_ago FLOAT,

    ownership_delta_7d FLOAT,
    ownership_delta_30d FLOAT,
    ownership_velocity FLOAT,

    add_rate_7d FLOAT,
    drop_rate_7d FLOAT,
    add_drop_ratio FLOAT,

    market_score FLOAT,
    market_tag VARCHAR(20),
    market_urgency VARCHAR(20),

    fetched_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (bdl_player_id, as_of_date)
);

CREATE INDEX IF NOT EXISTS idx_market_signals_player_date
    ON player_market_signals(bdl_player_id, as_of_date);
CREATE INDEX IF NOT EXISTS idx_market_signals_date_score
    ON player_market_signals(as_of_date, market_score);
"""


def migrate():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Applying PR 4.1 migration: player_market_signals...")
        cur.execute(DDL)
        conn.commit()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'player_market_signals'
        """)
        if cur.fetchone():
            print("  Verified: player_market_signals exists.")
        print("PR 4.1 migration ready.")
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
