
import sqlite3
import os

db_path = "backend/cbb.db"
if not os.path.exists(db_path):
    db_path = "cbb.db"

if not os.path.exists(db_path):
    # Try searching for any .db file
    import glob
    db_files = glob.glob("**/*.db", recursive=True)
    if db_files:
        db_path = db_files[0]
    else:
        print("Database not found")
        exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print(f"Searching database: {db_path}")

print("--- UNCG Games ---")
cursor.execute("SELECT * FROM games WHERE home_team LIKE '%Greensboro%' OR away_team LIKE '%Greensboro%'")
for row in cursor.fetchall():
    print(dict(row))

print("\n--- EWU Games ---")
cursor.execute("SELECT * FROM games WHERE home_team LIKE '%Eastern Washington%' OR away_team LIKE '%Eastern Washington%'")
for row in cursor.fetchall():
    print(dict(row))

print("\n--- UNCG Bets ---")
cursor.execute("SELECT * FROM bet_log WHERE pick LIKE '%Greensboro%'")
for row in cursor.fetchall():
    print(dict(row))

conn.close()
