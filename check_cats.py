
from backend.models import SessionLocal, PlayerProjection
from sqlalchemy import text
import os

def check():
    db = SessionLocal()
    try:
        total = db.query(PlayerProjection).count()
        # Use a more robust check for empty JSONB
        res = db.execute(text("SELECT COUNT(*) FROM player_projections WHERE cat_scores IS NOT NULL AND cat_scores != '{}'::jsonb")).scalar()
        print(f"total={total}, with_cat_scores={res}, pct={100*res//max(total,1)}%")
    finally:
        db.close()

if __name__ == "__main__":
    check()
