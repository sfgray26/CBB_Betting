import os, sys
import psycopg2

url = os.environ.get('DATABASE_URL')
if not url:
    print('No DATABASE_URL')
    sys.exit(1)

conn = psycopg2.connect(url)
cur = conn.cursor()

print("=== INGESTION LOGS LAST 24H ===")
cur.execute("""
SELECT job_name, status, started_at, completed_at, error_message
FROM data_ingestion_logs
WHERE started_at >= NOW() - INTERVAL '24 hours'
ORDER BY started_at DESC;
""")
rows = cur.fetchall()
print(f"Total jobs: {len(rows)}")
for r in rows[:20]:
    print(r)

print("\n=== PROJECTION_FRESHNESS JOBS ===")
cur.execute("""
SELECT status, started_at, completed_at, error_message, error_details
FROM data_ingestion_logs
WHERE job_name = 'projection_freshness'
ORDER BY started_at DESC
LIMIT 10;
""")
for r in cur.fetchall():
    print(r)

print("\n=== TABLE COUNTS ===")
tables = ['player_projections','player_id_mapping','player_rolling_stats','player_momentum','player_daily_metrics','probable_pitchers','ingested_injuries','fantasy_lineups','player_valuation_cache','statcast_performances']
for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    count = cur.fetchone()[0]
    print(f"{t}: {count}")

print("\n=== PLAYER_DAILY_METRICS z_score_total ===")
cur.execute("SELECT COUNT(*), COUNT(z_score_total) FROM player_daily_metrics")
total, non_null = cur.fetchone()
print(f"Total: {total}, Non-null z_score_total: {non_null}, Null rate: {100*(total-non_null)/total:.1f}%")

print("\n=== ORPHAN PROJECTIONS ===")
cur.execute("""
SELECT COUNT(*) FROM player_projections pp
LEFT JOIN player_id_mapping pim ON pp.bdl_id = pim.bdl_id
WHERE pim.bdl_id IS NULL AND pp.bdl_id IS NOT NULL
""")
print(f"player_projections with bdl_id not in mapping: {cur.fetchone()[0]}")

print("\n=== GHOST PLAYERS ===")
cur.execute("SELECT COUNT(*) FROM player_projections WHERE player_name IS NULL OR player_name = ''")
print(f"Null/empty player_name: {cur.fetchone()[0]}")

print("\n=== PROBABLE_PITCHERS is_confirmed ===")
cur.execute("SELECT COUNT(*), SUM(CASE WHEN is_confirmed THEN 1 ELSE 0 END) FROM probable_pitchers")
total, confirmed = cur.fetchone()
print(f"Total: {total}, Confirmed: {confirmed or 0}")

print("\n=== NULL RATES player_rolling_stats ===")
cols = ['rolling_7d','rolling_14d','rolling_15d','rolling_30d','ros_projection','row_projection']
for c in cols:
    cur.execute(f"SELECT COUNT(*), COUNT({c}) FROM player_rolling_stats")
    total, non_null = cur.fetchone()
    print(f"{c}: {100*(total-non_null)/total:.1f}% null")

print("\n=== NULL RATES player_momentum ===")
cur.execute("SELECT COUNT(*), COUNT(momentum_score) FROM player_momentum")
total, non_null = cur.fetchone()
print(f"momentum_score: {100*(total-non_null)/total:.1f}% null")

print("\n=== NEGATIVE PROJECTION VALUES ===")
cur.execute("""
SELECT player_name, hr, r, rbi, sb, ops, avg
FROM player_projections
WHERE (hr < 0 OR r < 0 OR rbi < 0 OR sb < 0 OR ops < 0 OR avg < 0)
AND player_name IS NOT NULL
LIMIT 10
""")
for r in cur.fetchall():
    print(r)

conn.close()
print("\n=== DONE ===")
