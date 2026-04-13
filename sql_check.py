import psycopg2, os
url = os.environ['DATABASE_URL'].replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
conn = psycopg2.connect(url)
cur = conn.cursor()
cur.execute("SELECT player_name, game_date, exit_velocity_avg, launch_angle_avg, xwoba FROM statcast_performances WHERE exit_velocity_avg > 0 OR xwoba > 0 LIMIT 10")
rows = cur.fetchall()
print(f'Found {len(rows)} rows with non-zero stats:')
for r in rows:
    print(r)
conn.close()
