"""Kill any idle-in-transaction sessions on player_rolling_stats."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
conn.autocommit = True
cur = conn.cursor()

cur.execute("""
    SELECT pid, state, LEFT(query, 80)
    FROM pg_stat_activity
    WHERE state = 'idle in transaction'
      AND pid <> pg_backend_pid()
""")
rows = cur.fetchall()
print(f"Found {len(rows)} idle-in-transaction sessions")
for r in rows:
    print(" ", r)

if rows:
    pids = [r[0] for r in rows]
    cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid = ANY(%s)", (pids,))
    print(f"Terminated {cur.rowcount} sessions")

cur.close()
conn.close()
print("Done")
