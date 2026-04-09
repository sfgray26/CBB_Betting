"""Quick coverage check after backfill."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
import psycopg2

conn = psycopg2.connect(os.environ["DATABASE_URL"])
cur = conn.cursor()

cur.execute("""SELECT game_date, COUNT(*) as total,
    COUNT(ab) as has_ab, COUNT(hits) as has_hits, COUNT(strikeouts_bat) as has_k
    FROM mlb_player_stats WHERE season=2026
    GROUP BY game_date ORDER BY game_date""")
print("=== mlb_player_stats coverage ===")
print(f"{'Date':<12} {'Total':>6} {'w/AB':>6} {'w/H':>6} {'w/K':>6}")
for r in cur.fetchall():
    print(f"{str(r[0]):<12} {r[1]:>6} {r[2]:>6} {r[3]:>6} {r[4]:>6}")

cur.execute("""SELECT game_date, COUNT(*) FROM mlb_game_log
    WHERE game_date >= '2026-03-27' GROUP BY game_date ORDER BY game_date""")
print("\n=== mlb_game_log coverage ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]} games")

cur.execute("""SELECT as_of_date, COUNT(*) FROM player_rolling_stats
    WHERE as_of_date >= '2026-03-27' GROUP BY as_of_date ORDER BY as_of_date""")
print("\n=== player_rolling_stats coverage ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]} rows")

cur.execute("""SELECT as_of_date, COUNT(*) FROM player_scores
    WHERE as_of_date >= '2026-03-27' GROUP BY as_of_date ORDER BY as_of_date""")
print("\n=== player_scores coverage ===")
for r in cur.fetchall():
    print(f"  {r[0]}: {r[1]} rows")

conn.close()
print("\nDone.")
