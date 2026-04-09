import psycopg2, os, re

# Get DATABASE_URL from Railway env
url = os.environ.get("DATABASE_URL")
if not url:
    print("ERROR: DATABASE_URL not found")
    exit(1)

# Map internal to public if we are running locally (which railway run does on Windows)
if "railway.internal" in url:
    # Use the public host found from railway variables earlier
    # postgres-ygnv-production.up.railway.app
    url = url.replace("postgres-ygnv.railway.internal", "postgres-ygnv-production.up.railway.app")

try:
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    # 1. Schema Check
    cur.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'position_eligibility'
    ORDER BY ordinal_position
    """)
    print('=== COLUMNS ===')
    for r in cur.fetchall():
        print(f'  {r[0]:30s} {r[1]:20s} nullable={r[2]}')

    # 2. Constraints Check
    cur.execute("""
    SELECT conname, contype
    FROM pg_constraint
    WHERE conrelid = 'position_eligibility'::regclass
    ORDER BY conname
    """)
    print('\n=== CONSTRAINTS ===')
    for r in cur.fetchall():
        print(f'  {r[0]:40s} type={r[1]}')

    # 3. Data Quality Check
    cur.execute('SELECT count(*) FROM position_eligibility')
    total = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM position_eligibility WHERE yahoo_player_key IS NULL')
    null_keys = cur.fetchone()[0]

    cur.execute('SELECT count(DISTINCT yahoo_player_key) FROM position_eligibility')
    unique_keys = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM position_eligibility WHERE multi_eligibility_count >= 3')
    multi = cur.fetchone()[0]

    cur.execute('SELECT player_name, primary_position, multi_eligibility_count FROM position_eligibility ORDER BY multi_eligibility_count DESC LIMIT 5')
    top = cur.fetchall()

    print('\n=== DATA QUALITY ===')
    print(f'Total rows:           {total}')
    print(f'NULL yahoo_keys:      {null_keys}')
    print(f'Unique yahoo_keys:    {unique_keys}')
    print(f'Duplicates:           {total - unique_keys}')
    print(f'Multi-eligible (>=3): {multi}')
    print(f'\nTop multi-eligible players:')
    for r in top:
        print(f'  {r[0]:30s} primary={r[1]:5s} count={r[2]}')

    conn.close()
except Exception as e:
    print(f"ERROR: {e}")
