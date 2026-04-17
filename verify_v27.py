import psycopg2, os
url = os.environ['DATABASE_URL'].replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
conn = psycopg2.connect(url)
cur = conn.cursor()

print("=== SCHEMA VERIFICATION (v27) ===")
for table in ['player_rolling_stats', 'player_scores']:
    cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}'")
    cols = [r[0] for r in cur.fetchall()]
    print(f"Table: {table}")
    for target in ['w_caught_stealing', 'w_net_stolen_bases', 'z_nsb']:
        if target in cols:
            print(f"  [PASS] {target}")
        elif (table == 'player_rolling_stats' and target.startswith('w_')) or (table == 'player_scores' and target == 'z_nsb'):
            print(f"  [FAIL] {target} MISSING")

conn.close()
