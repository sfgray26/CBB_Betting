import os
import sqlalchemy
from sqlalchemy import text

def verify():
    url = os.environ.get('DATABASE_URL')
    if not url:
        print("Error: DATABASE_URL not set")
        return
    
    # Force public URL if running from outside Railway
    url = url.replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
    
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        # 1. Yahoo Coverage
        res = conn.execute(text('SELECT COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL), COUNT(*) FROM player_id_mapping')).fetchone()
        with_yahoo, total_mapping = res
        print(f"Yahoo Coverage: {with_yahoo}/{total_mapping} ({with_yahoo/total_mapping*100:.2f}%)")
        
        # 2. Cat Scores
        res2 = conn.execute(text('SELECT COUNT(*) FILTER (WHERE cat_scores IS NOT NULL), COUNT(*) FROM player_projections')).fetchone()
        with_cats, total_projs = res2
        print(f"Cat Scores Coverage: {with_cats}/{total_projs} ({with_cats/total_projs*100:.2f}%)")
        
        # 3. Projection Freshness
        res3 = conn.execute(text('SELECT MAX(updated_at) FROM player_projections')).fetchone()
        print(f"Latest Projection Update: {res3[0]}")
        
        # 4. Statcast Freshness
        res4 = conn.execute(text('SELECT MAX(game_date) FROM statcast_performances')).fetchone()
        print(f"Latest Statcast Game: {res4[0]}")

if __name__ == "__main__":
    verify()
