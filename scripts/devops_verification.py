import psycopg2, os

# Handle Railway environment variable
url = os.environ.get("DATABASE_URL")
if not url:
    print("ERROR: DATABASE_URL not found")
    exit(1)

# Use public host for local execution if internal host is detected
if "railway.internal" in url:
    url = url.replace("postgres-ygnv.railway.internal", "postgres-ygnv-production.up.railway.app")

try:
    conn = psycopg2.connect(url)
    cur = conn.cursor()
    
    # 1. Column Verification
    cur.execute("""
    SELECT column_name, data_type, is_nullable
    FROM information_schema.columns
    WHERE table_name = 'position_eligibility'
    ORDER BY ordinal_position
    """)
    print('=== COLUMNS ===')
    cols = cur.fetchall()
    for r in cols:
        print(f'  {r[0]:30s} {r[1]:20s} nullable={r[2]}')

    # 2. Constraints Verification
    cur.execute("""
    SELECT conname, contype
    FROM pg_constraint
    WHERE conrelid = 'position_eligibility'::regclass
    ORDER BY conname
    """)
    print('\n=== CONSTRAINTS ===')
    constraints = cur.fetchall()
    for r in constraints:
        print(f'  {r[0]:40s} type={r[1]}')

    # 3. Data Quality Metrics
    cur.execute('SELECT count(*) FROM position_eligibility')
    total = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM position_eligibility WHERE yahoo_player_key IS NULL')
    null_keys = cur.fetchone()[0]

    cur.execute('SELECT count(DISTINCT yahoo_player_key) FROM position_eligibility')
    unique_keys = cur.fetchone()[0]

    cur.execute('SELECT count(*) FROM position_eligibility WHERE multi_eligibility_count >= 3')
    multi = cur.fetchone()[0]

    # Specific check for Josh Smith
    cur.execute("SELECT player_name, primary_position, multi_eligibility_count FROM position_eligibility WHERE player_name LIKE '%Josh Smith%' LIMIT 1")
    josh_smith = cur.fetchone()

    print('\n=== DATA QUALITY ===')
    print(f'Total rows:           {total}')
    print(f'NULL yahoo_keys:      {null_keys}')
    print(f'Unique yahoo_keys:    {unique_keys}')
    print(f'Duplicates:           {total - unique_keys}')
    print(f'Multi-eligible (>=3): {multi}')
    
    if josh_smith:
        print(f'\nTarget Player Check:')
        print(f'  {josh_smith[0]:30s} primary={josh_smith[1]:5s} count={josh_smith[2]}')
    else:
        # Get top player if Josh Smith not found
        cur.execute('SELECT player_name, primary_position, multi_eligibility_count FROM position_eligibility ORDER BY multi_eligibility_count DESC LIMIT 1')
        top = cur.fetchone()
        if top:
            print(f'\nTop multi-eligible player:')
            print(f'  {top[0]:30s} primary={top[1]:5s} count={top[2]}')

    conn.close()
except Exception as e:
    print(f"ERROR: {e}")
