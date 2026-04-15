import psycopg2, os
# Use the public URL since internal resolution is failing locally
url = os.environ['DATABASE_URL'].replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')

try:
    # 1. Connect with a decent timeout
    conn = psycopg2.connect(url, connect_timeout=20)
    cur = conn.cursor()
    
    # 2. Query for the absolute latest 5 records that have non-zero Exit Velocity
    # This proves the column mapping is working and data is flowing.
    query = """
        SELECT player_name, game_date, exit_velocity_avg, xwoba, launch_angle_avg
        FROM statcast_performances 
        WHERE exit_velocity_avg > 0 
        ORDER BY game_date DESC, exit_velocity_avg DESC 
        LIMIT 10;
    """
    cur.execute(query)
    rows = cur.fetchall()
    
    print("=== CONCRETE DATA EVIDENCE ===")
    if not rows:
        print("No non-zero data found yet. This would indicate a failure.")
    else:
        for r in rows:
            print(f"PLAYER: {r[0]:25s} DATE: {r[1]} EV: {r[2]:5.1f} xWOBA: {r[3]:5.3f} LA: {r[4]:5.1f}")
    print("===============================")
    
    # 3. Check total row count of non-zero records
    cur.execute("SELECT count(*) FROM statcast_performances WHERE exit_velocity_avg > 0")
    nz_count = cur.fetchone()[0]
    print(f"Total rows with non-zero EV: {nz_count}")
    
    conn.close()
except Exception as e:
    print(f"ERROR: {e}")
