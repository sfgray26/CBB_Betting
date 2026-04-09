import psycopg2, os
# Manually construct public URL from internal parts
internal_url = os.environ['DATABASE_URL']
# Replace internal host with public host
public_url = internal_url.replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')

conn = psycopg2.connect(public_url)
cur = conn.cursor()

# Column inventory
cur.execute("""
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'position_eligibility'
ORDER BY ordinal_position
""")
print('=== COLUMNS ===')
for r in cur.fetchall():
    print(f'  {r[0]:30s} {r[1]:20s} nullable={r[2]}')

# Constraints
cur.execute("""
SELECT conname, contype
FROM pg_constraint
WHERE conrelid = 'position_eligibility'::regclass
ORDER BY conname
""")
print('\n=== CONSTRAINTS ===')
for r in cur.fetchall():
    print(f'  {r[0]:40s} type={r[1]}')

conn.close()
