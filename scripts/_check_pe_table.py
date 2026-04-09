"""Check position_eligibility table state."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""SELECT column_name, data_type FROM information_schema.columns
    WHERE table_name = 'position_eligibility' ORDER BY ordinal_position""")
cols = cur.fetchall()
if cols:
    print("=== position_eligibility columns ===")
    for c in cols:
        print(f"  {c[0]}: {c[1]}")
    cur.execute("SELECT COUNT(*) FROM position_eligibility")
    print(f"\nRow count: {cur.fetchone()[0]}")
else:
    print("Table does not exist yet")
conn.close()
