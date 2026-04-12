"""P-2 Step 2: NULL out impossible ERA values. Temporary - delete after use."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    r = db.execute(text("UPDATE mlb_player_stats SET era = NULL WHERE era > 100 OR era < 0"))
    print(f"Rows updated: {r.rowcount}")
    db.commit()
    # Verify immediately
    c = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 100 OR era < 0")).scalar()
    print(f"Impossible ERA rows remaining: {c}")
    assert c == 0, f"FAIL: {c} remaining"
    print("VERIFY OK")
finally:
    db.close()
