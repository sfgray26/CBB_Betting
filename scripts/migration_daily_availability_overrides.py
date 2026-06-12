"""
Create daily_availability_overrides table.

Table defined in backend/models.py (DailyAvailabilityOverride) but was missing
from the Railway PostgreSQL instance, causing 503 errors on waiver endpoints via
POST /api/admin/availability-override.

Idempotent: CREATE TABLE IF NOT EXISTS / CREATE INDEX IF NOT EXISTS.
Safe to run on a live database while the app is running.
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
CREATE TABLE IF NOT EXISTS daily_availability_overrides (
    id          SERIAL PRIMARY KEY,
    player_key  VARCHAR(64)  NOT NULL,
    player_name VARCHAR(128) NOT NULL,
    game_date   DATE         NOT NULL,
    status      VARCHAR(32)  NOT NULL,
    note        VARCHAR(256),
    source      VARCHAR(32)  DEFAULT 'admin',
    created_at  TIMESTAMP WITH TIME ZONE,
    CONSTRAINT uq_override_player_date UNIQUE (player_key, game_date)
);

CREATE INDEX IF NOT EXISTS idx_dao_game_date
    ON daily_availability_overrides (game_date);
"""


def migrate():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Applying migration: daily_availability_overrides...")
        cur.execute(DDL)
        conn.commit()

        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = 'daily_availability_overrides'
        """)
        if cur.fetchone():
            print("  Verified: daily_availability_overrides table exists.")
        else:
            print("  ERROR: table not found after migration.")
            sys.exit(1)
        print("Migration complete.")
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
