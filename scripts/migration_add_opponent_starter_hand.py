"""
PR 5.1b — Add opponent_starter_hand to mlb_player_stats

Required for hitter split computation in the matchup engine.
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
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'mlb_player_stats'
          AND column_name = 'opponent_starter_hand'
    ) THEN
        ALTER TABLE mlb_player_stats
            ADD COLUMN opponent_starter_hand VARCHAR(1);
        RAISE NOTICE 'Added opponent_starter_hand to mlb_player_stats';
    ELSE
        RAISE NOTICE 'opponent_starter_hand already exists';
    END IF;
END $$;
"""


def migrate():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Applying PR 5.1b migration: opponent_starter_hand column...")
        cur.execute(DDL)
        conn.commit()

        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'mlb_player_stats'
              AND column_name = 'opponent_starter_hand'
        """)
        if cur.fetchone():
            print("  Verified: opponent_starter_hand column exists.")
        print("PR 5.1b migration ready.")
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
