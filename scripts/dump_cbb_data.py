
import sqlite3
import os

db_path = "backend/cbb.db"
if not os.path.exists(db_path):
    db_path = "cbb.db"

if not os.path.exists(db_path):
    print("Database not found")
    exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

print("--- Games (First 10) ---")
cursor.execute("SELECT id, home_team, away_team, home_score, away_score, completed FROM games LIMIT 10")
for row in cursor.fetchall():
    print(dict(row))

print("\n--- BetLog (First 10) ---")
cursor.execute("SELECT id, pick, outcome, game_id FROM bet_log LIMIT 10")
for row in cursor.fetchall():
    print(dict(row))

conn.close()
