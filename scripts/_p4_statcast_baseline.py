"""P-4 Step 1 + Step 3: statcast_performances baseline + verify."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    total = db.execute(text("SELECT COUNT(*) FROM statcast_performances")).scalar()
    distinct_dates = db.execute(text("SELECT COUNT(DISTINCT game_date) FROM statcast_performances")).scalar()
    latest = db.execute(text("SELECT MAX(game_date) FROM statcast_performances")).scalar()
    earliest = db.execute(text("SELECT MIN(game_date) FROM statcast_performances")).scalar()
    print(f"total rows:        {total}")
    print(f"distinct game_dates: {distinct_dates}")
    print(f"earliest game_date: {earliest}")
    print(f"latest game_date:   {latest}")
finally:
    db.close()
