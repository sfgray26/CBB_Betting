import psycopg2, os, sys

url = os.environ["RAILWAY_DB_URL"]
try:
    conn = psycopg2.connect(url, sslmode='require', connect_timeout=15)
    cur = conn.cursor()
    cur.execute("SELECT 1")
    print("Connected OK")

    # Check if column exists
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name='probable_pitchers' AND column_name='handedness'
    """)
    if cur.fetchone():
        print("Column 'handedness' already exists — nothing to do")
    else:
        cur.execute("ALTER TABLE probable_pitchers ADD COLUMN handedness VARCHAR(1)")
        cur.execute("COMMENT ON COLUMN probable_pitchers.handedness IS 'Pitcher throwing hand: L or R'")
        conn.commit()
        print("SUCCESS: Added 'handedness' column to probable_pitchers")
    conn.close()
except Exception as e:
    print("Error:", e)
    sys.exit(1)
