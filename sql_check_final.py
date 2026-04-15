import psycopg2, os
url = os.environ['DATABASE_URL'].replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
try:
    conn = psycopg2.connect(url, connect_timeout=10)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM statcast_performances")
    count = cur.fetchone()[0]
    print(f'Total count: {count}')
    cur.execute("SELECT player_name, game_date, exit_velocity_avg, xwoba FROM statcast_performances WHERE pa > 0 OR exit_velocity_avg > 0 LIMIT 5")
    rows = cur.fetchall()
    print("Sample rows:")
    for r in rows:
        print(r)
    conn.close()
except Exception as e:
    print(f"Error: {e}")
