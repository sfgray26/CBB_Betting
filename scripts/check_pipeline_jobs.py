import os
import sqlalchemy
from sqlalchemy import text
from datetime import datetime

def check_jobs():
    url = os.environ.get('DATABASE_URL').replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        print('--- Pipeline Job Health ---')
        jobs = ['mlb_game_log', 'mlb_box_stats', 'rolling_windows', 'player_scores', 'vorp', 'player_momentum', 'ros_simulation', 'decision_optimization', 'snapshot']
        print(f"{'Job Name':25} | {'Status':10} | {'Last Run'}")
        print('-' * 60)
        for job in jobs:
            sql = f"SELECT status, started_at FROM data_ingestion_logs WHERE job_type = '{job}' ORDER BY started_at DESC LIMIT 1"
            res = conn.execute(text(sql)).fetchone()
            if res:
                print(f'{job:25} | {res[0]:10} | {res[1]}')
            else:
                print(f'{job:25} | {"MISSING":10} | N/A')

if __name__ == "__main__":
    check_jobs()
