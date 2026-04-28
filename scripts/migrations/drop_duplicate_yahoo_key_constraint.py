"""
Drop the duplicate player_id_mapping_yahoo_key_key constraint.

The table has two unique constraints on yahoo_key:
  - player_id_mapping_yahoo_key_key  (auto-named, the duplicate)
  - uq_player_id_mapping_yahoo_key   (the canonical named constraint)

Run via: railway run python scripts/migrations/drop_duplicate_yahoo_key_constraint.py
"""
import os
import sys
import psycopg2


def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    conn = psycopg2.connect(database_url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # Check whether the duplicate constraint still exists
            cur.execute(
                """
                SELECT constraint_name
                FROM information_schema.table_constraints
                WHERE table_name = 'player_id_mapping'
                  AND constraint_name = 'player_id_mapping_yahoo_key_key'
                  AND constraint_type = 'UNIQUE'
                """
            )
            row = cur.fetchone()
            if row is None:
                print("Constraint player_id_mapping_yahoo_key_key not found — nothing to do.")
                return

            print("Dropping player_id_mapping_yahoo_key_key ...")
            cur.execute(
                "ALTER TABLE player_id_mapping DROP CONSTRAINT IF EXISTS player_id_mapping_yahoo_key_key"
            )
            conn.commit()
            print("Done.")
    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
