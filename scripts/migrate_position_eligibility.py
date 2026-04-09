"""
Migration: Restructure position_eligibility for multi-position eligibility.

Changes:
  1. Add yahoo_player_key (VARCHAR(50), NOT NULL, UNIQUE) as natural key
  2. Add player_name, first_name, last_name columns
  3. Add can_play_sp, can_play_rp boolean columns
  4. Make bdl_player_id nullable (populated later via player_id_mapping)
  5. Drop old _pe_player_uc constraint (bdl_player_id unique)
  6. Add new _pe_yahoo_uc constraint (yahoo_player_key unique)
  7. Add idx_pe_bdl_player_id index

Table is currently EMPTY (0 rows) so this is safe to run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: Set DATABASE_URL")
    sys.exit(1)


def migrate():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = False
    cur = conn.cursor()

    print("=== Migration: position_eligibility restructure ===")

    # Verify table is empty (safety check)
    cur.execute("SELECT COUNT(*) FROM position_eligibility")
    row_count = cur.fetchone()[0]
    if row_count > 0:
        print(f"WARNING: Table has {row_count} rows. Truncating...")
        cur.execute("TRUNCATE position_eligibility RESTART IDENTITY CASCADE")

    # Step 1: Drop old constraint and index
    print("  Dropping old constraints...")
    cur.execute("""
        DO $$
        BEGIN
            -- Drop old unique constraint on bdl_player_id
            IF EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '_pe_player_uc') THEN
                ALTER TABLE position_eligibility DROP CONSTRAINT _pe_player_uc;
            END IF;
        END $$
    """)

    # Step 2: Add new columns
    print("  Adding new columns...")

    # yahoo_player_key
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'position_eligibility' AND column_name = 'yahoo_player_key'
            ) THEN
                ALTER TABLE position_eligibility ADD COLUMN yahoo_player_key VARCHAR(50);
            END IF;
        END $$
    """)

    # player_name
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'position_eligibility' AND column_name = 'player_name'
            ) THEN
                ALTER TABLE position_eligibility ADD COLUMN player_name VARCHAR(100);
            END IF;
        END $$
    """)

    # first_name
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'position_eligibility' AND column_name = 'first_name'
            ) THEN
                ALTER TABLE position_eligibility ADD COLUMN first_name VARCHAR(50);
            END IF;
        END $$
    """)

    # last_name
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'position_eligibility' AND column_name = 'last_name'
            ) THEN
                ALTER TABLE position_eligibility ADD COLUMN last_name VARCHAR(50);
            END IF;
        END $$
    """)

    # can_play_sp
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'position_eligibility' AND column_name = 'can_play_sp'
            ) THEN
                ALTER TABLE position_eligibility ADD COLUMN can_play_sp BOOLEAN NOT NULL DEFAULT FALSE;
            END IF;
        END $$
    """)

    # can_play_rp
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'position_eligibility' AND column_name = 'can_play_rp'
            ) THEN
                ALTER TABLE position_eligibility ADD COLUMN can_play_rp BOOLEAN NOT NULL DEFAULT FALSE;
            END IF;
        END $$
    """)

    # Step 3: Make bdl_player_id nullable
    print("  Making bdl_player_id nullable...")
    cur.execute("ALTER TABLE position_eligibility ALTER COLUMN bdl_player_id DROP NOT NULL")

    # Make player_type have a default
    cur.execute("ALTER TABLE position_eligibility ALTER COLUMN player_type SET DEFAULT 'batter'")

    # Step 4: Set yahoo_player_key NOT NULL (table is empty, so safe)
    cur.execute("ALTER TABLE position_eligibility ALTER COLUMN yahoo_player_key SET NOT NULL")

    # Step 5: Add new unique constraint
    print("  Adding new constraints...")
    cur.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = '_pe_yahoo_uc') THEN
                ALTER TABLE position_eligibility ADD CONSTRAINT _pe_yahoo_uc UNIQUE (yahoo_player_key);
            END IF;
        END $$
    """)

    # Step 6: Add index on bdl_player_id
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_pe_bdl_player_id ON position_eligibility (bdl_player_id)
    """)

    conn.commit()

    # Verify
    cur.execute("""SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_name = 'position_eligibility'
        ORDER BY ordinal_position""")
    print("\n  Final schema:")
    for col in cur.fetchall():
        print(f"    {col[0]}: {col[1]} (nullable={col[2]})")

    cur.execute("""SELECT conname FROM pg_constraint
        WHERE conrelid = 'position_eligibility'::regclass""")
    print("\n  Constraints:")
    for c in cur.fetchall():
        print(f"    {c[0]}")

    cur.close()
    conn.close()
    print("\n=== Migration complete ===")


if __name__ == "__main__":
    migrate()
