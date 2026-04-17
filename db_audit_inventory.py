import psycopg2, os
url = os.environ['DATABASE_URL'].replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')

try:
    conn = psycopg2.connect(url, connect_timeout=10)
    cur = conn.cursor()
    
    # 1. Get all tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
    tables = [r[0] for r in cur.fetchall()]
    
    print("=== FANTASY APP DATABASE INVENTORY ===")
    print(f"{'TABLE NAME':40s} | {'ROWS':10s}")
    print("-" * 55)
    
    for table in tables:
        try:
            cur.execute(f'SELECT count(*) FROM "{table}"')
            count = cur.fetchone()[0]
            print(f"{table:40s} | {count:10d}")
        except Exception as e:
            print(f"{table:40s} | ERROR: {e}")
            conn.rollback()
            
    conn.close()
except Exception as e:
    print(f"CONNECTION ERROR: {e}")
