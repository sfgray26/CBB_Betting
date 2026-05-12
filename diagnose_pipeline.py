"""Diagnostic queries for A-4 decision pipeline"""
import sqlalchemy
from sqlalchemy import create_engine, text

# Production DB connection
DB_URL = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

engine = create_engine(DB_URL)

queries = [
    ("mlb_game_log", "SELECT COUNT(*) FROM mlb_game_log"),
    ("mlb_player_stats", "SELECT COUNT(*) FROM mlb_player_stats"),
    ("player_rolling_stats", "SELECT COUNT(*) FROM player_rolling_stats"),
    ("player_scores", "SELECT COUNT(*) FROM player_scores"),
    ("decision_results", "SELECT COUNT(*) FROM decision_results"),
]

print("=" * 60)
print("A-4 Pipeline Diagnostic Results")
print("=" * 60)

with engine.connect() as conn:
    for table_name, query in queries:
        result = conn.execute(text(query)).fetchone()
        count = result[0] if result else 0
        print(f"{table_name:30s}: {count:>10,} rows")

print("=" * 60)
