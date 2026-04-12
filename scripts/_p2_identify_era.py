"""P-2 Step 1: Identify impossible ERA rows. Temporary - delete after use."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    rows = db.execute(text("""
        SELECT id, bdl_player_id, era, earned_runs, innings_pitched, game_id, game_date
        FROM mlb_player_stats
        WHERE era > 100 OR era < 0
        ORDER BY era DESC
    """)).fetchall()
    print(f"COUNT: {len(rows)}")
    for r in rows:
        print(dict(r._mapping))
finally:
    db.close()
