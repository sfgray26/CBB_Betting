"""Quick check: does w_games column exist in player_rolling_stats?"""
import os
from sqlalchemy import create_engine, text

e = create_engine(os.environ["DATABASE_URL"])
with e.connect() as c:
    r = c.execute(text(
        "SELECT column_name, data_type "
        "FROM information_schema.columns "
        "WHERE table_name = :t AND column_name = :col"
    ), {"t": "player_rolling_stats", "col": "w_games"}).fetchone()
    if r:
        print(f"OK: w_games column exists (type={r.data_type})")
    else:
        print("NOT FOUND: w_games column does not exist")
