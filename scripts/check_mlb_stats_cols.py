from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    cols = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'mlb_player_stats'")).fetchall()
    print("Columns in mlb_player_stats:")
    for c in cols:
        print(f" - {c[0]}")
