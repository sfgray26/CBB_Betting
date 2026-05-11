import os, psycopg2
conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()
cur.execute("""
    SELECT table_name FROM information_schema.tables
    WHERE table_schema = 'public'
      AND table_name IN ('player_opportunity', 'player_market_signals', 'matchup_context')
    ORDER BY table_name
""")
print("Existing tables:", [r[0] for r in cur.fetchall()])
conn.close()
