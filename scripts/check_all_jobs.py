import os
import sqlalchemy
from sqlalchemy import text
from datetime import datetime

def check_all_jobs():
    url = os.environ.get('DATABASE_URL').replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        print('--- All Pipeline Job Health ---')
        sql = "SELECT job_type, status, started_at FROM data_ingestion_logs ORDER BY job_type, started_at DESC"
        rows = conn.execute(text(sql)).fetchall()
        
        # Group by job_type to get the latest run for each
        latest_jobs = {}
        for r in rows:
            if r.job_type not in latest_jobs:
                latest_jobs[r.job_type] = r
                
        print(f"{'Job Name':30} | {'Status':10} | {'Last Run'}")
        print('-' * 70)
        for job_name in sorted(latest_jobs.keys()):
            r = latest_jobs[job_name]
            print(f'{job_name:30} | {r.status:10} | {r.started_at}')

if __name__ == "__main__":
    check_all_jobs()
