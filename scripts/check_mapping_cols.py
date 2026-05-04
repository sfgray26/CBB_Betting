from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    cols = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'player_id_mapping'")).fetchall()
    print("Columns in player_id_mapping:")
    for c in cols:
        print(f" - {c[0]}")
