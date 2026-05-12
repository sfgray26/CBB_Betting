import os
import sqlalchemy
from sqlalchemy import text

def specific_checks():
    url = os.environ.get('DATABASE_URL').replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        print('--- Specific Date Checks ---')
        latest_decision_date = conn.execute(text("SELECT MAX(as_of_date) FROM decision_results")).scalar()
        if latest_decision_date:
            count = conn.execute(text(f"SELECT COUNT(*) FROM decision_results WHERE as_of_date = '{latest_decision_date}'")).scalar()
            print(f'Latest decision_results date: {latest_decision_date} | Count: {count}')
            
            # Check scores for the same date
            scores_14d = conn.execute(text(f"SELECT COUNT(*) FROM player_scores WHERE as_of_date = '{latest_decision_date}' AND window_days = 14")).scalar()
            print(f'Player scores (14d) for {latest_decision_date}: {scores_14d}')
        else:
            print("No decision_results found.")

        # Overall mapping health
        mapped = conn.execute(text("SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL AND bdl_id IS NOT NULL")).scalar()
        total_mapping = conn.execute(text("SELECT COUNT(*) FROM player_id_mapping")).scalar()
        print(f'Mapped Roster Candidates: {mapped} / {total_mapping}')

if __name__ == "__main__":
    specific_checks()
