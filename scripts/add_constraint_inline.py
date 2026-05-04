#!/usr/bin/env python3
import os
import psycopg2

DATABASE_URL = os.environ.get('DATABASE_URL')

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Check if constraint exists
cursor.execute("SELECT COUNT(*) FROM pg_constraint WHERE conname = '_pim_yahoo_key_uc'")
exists = cursor.fetchone()[0]
print(f"Constraint exists: {exists}")

if not exists:
    print("Adding constraint...")
    cursor.execute("ALTER TABLE player_id_mapping ADD CONSTRAINT _pim_yahoo_key_uc UNIQUE (yahoo_key)")
    conn.commit()
    print("Done: Added _pim_yahoo_key_uc constraint")
else:
    print("Constraint already exists, skipping")

cursor.close()
conn.close()
