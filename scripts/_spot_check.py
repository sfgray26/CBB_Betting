import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv; load_dotenv()
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
v31 = db.execute(text("SELECT COUNT(*) FROM player_rolling_stats WHERE w_runs IS NOT NULL")).scalar()
v32 = db.execute(text("SELECT COUNT(*) FROM player_scores WHERE z_r IS NOT NULL")).scalar()
bdl = db.execute(text("SELECT COUNT(*) FROM information_schema.columns WHERE table_name='mlb_player_stats' AND column_name='bdl_stat_id'")).scalar()
print(f"V31 w_runs non-null: {v31}")
print(f"V32 z_r non-null:    {v32}")
print(f"bdl_stat_id exists:  {bdl} (expect 0)")
db.close()
