"""One-shot: apply PR 1.1 migration using PUBLIC_DB_URL env var."""
import os
import sys
import psycopg2

url = os.environ.get("PUBLIC_DB_URL") or os.environ.get("DATABASE_URL")
if not url:
    print("ERROR: PUBLIC_DB_URL not set")
    sys.exit(1)

conn = psycopg2.connect(url)
conn.autocommit = False
cur = conn.cursor()

sql = open("scripts/migration_threshold_config.sql", encoding="utf-8").read()
try:
    cur.execute(sql)
    conn.commit()
    print("Migration applied.")
except Exception as exc:
    conn.rollback()
    print("ERROR:", exc)
    sys.exit(1)

cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name IN ('threshold_config', 'threshold_audit', 'feature_flags')
    ORDER BY table_name
""")
print("Tables present:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'threshold_config'")
print("Indexes on threshold_config:", [r[0] for r in cur.fetchall()])

cur.execute("SELECT COUNT(*) FROM feature_flags")
print("feature_flags row count:", cur.fetchone()[0])

cur.close()
conn.close()
