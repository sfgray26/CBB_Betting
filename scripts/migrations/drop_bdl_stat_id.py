"""
Drop the bdl_stat_id column from mlb_player_stats.

This column is 100% null (12,297 rows) because BDL does not expose per-row
stat IDs in its /mlb/v1/stats response. The column was added speculatively
and has never been populated.

Run via: railway run python scripts/migrations/drop_bdl_stat_id.py
"""
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2

url = os.environ.get("DATABASE_URL")
if not url:
    print("ERROR: DATABASE_URL not set", file=sys.stderr)
    sys.exit(1)

conn = psycopg2.connect(url)
conn.autocommit = True
cur = conn.cursor()
cur.execute("ALTER TABLE mlb_player_stats DROP COLUMN IF EXISTS bdl_stat_id;")
print("bdl_stat_id dropped (or did not exist).")
cur.close()
conn.close()
