import psycopg2

conn = psycopg2.connect('postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway')
cur = conn.cursor()

cur.execute("SELECT COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL), COUNT(*) FROM player_id_mapping")
with_yahoo, total = cur.fetchone()
print(f'Yahoo coverage: {with_yahoo}/{total} ({with_yahoo/total*100:.1f}%)')

cur.execute("SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NULL AND bdl_id IS NOT NULL")
print(f'BDL-only rows (no Yahoo): {cur.fetchone()[0]}')

cur.execute("SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL AND bdl_id IS NULL")
print(f'Yahoo-only rows (no BDL): {cur.fetchone()[0]}')

cur.execute("SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL AND bdl_id IS NOT NULL")
print(f'Both Yahoo + BDL: {cur.fetchone()[0]}')

# Sample of rows missing Yahoo — are they real players or stale/inactive?
cur.execute("""
    SELECT full_name, bdl_id, normalized_name
    FROM player_id_mapping
    WHERE yahoo_id IS NULL
    ORDER BY full_name
    LIMIT 20
""")
print('\nSample BDL-only rows (no Yahoo):')
for row in cur.fetchall():
    print(f'  {row[0]} (bdl_id={row[1]})')

conn.close()
