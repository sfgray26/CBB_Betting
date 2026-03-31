
import os
import json
from sqlalchemy import create_engine, text

db_url = "postgresql://postgres:nrvjuGWnjwOttjEiesPTGJwVxSfzNDCV@shinkansen.proxy.rlwy.net:17252/railway"
engine = create_engine(db_url)

with engine.connect() as conn:
    print("--- job_queue schema ---")
    q = text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'job_queue'")
    for row in conn.execute(q):
        print(f"{row.column_name}: {row.data_type}")
    
    print("\n--- testing jq_submit logic ---")
    import uuid
    from datetime import datetime
    from zoneinfo import ZoneInfo
    
    def _now_et():
        return datetime.now(ZoneInfo("America/New_York"))

    job_id = str(uuid.uuid4())
    job_type = "lineup_optimization"
    payload = {"target_date": "2026-04-01", "risk_tolerance": "balanced"}
    priority = 3
    
    try:
        conn.execute(
            text(
                """
                INSERT INTO job_queue
                    (id, job_type, payload, status, priority, created_at,
                     retry_count, max_retries, league_key, team_key)
                VALUES
                    (:id, :job_type, CAST(:payload AS JSONB), 'pending', :priority,
                     :created_at, 0, 3, :league_key, :team_key)
                """
            ),
            {
                "id": job_id,
                "job_type": job_type,
                "payload": json.dumps(payload),
                "priority": priority,
                "created_at": _now_et(),
                "league_key": None,
                "team_key": None,
            },
        )
        conn.commit()
        print(f"Successfully inserted job {job_id}")
    except Exception as e:
        print(f"FAILED to insert: {e}")
