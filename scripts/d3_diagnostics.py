import os
import sqlalchemy
from sqlalchemy import text

def run_sql_diagnostics():
    url = os.environ.get('DATABASE_URL'); url = url.replace('postgres-ygnv.railway.internal', 'postgres-ygnv-production.up.railway.app')
    if not url:
        print("Error: DATABASE_URL not set")
        return
    
    # Replace internal with public if needed, but since I'm in DevOps Lead role,
    # I should try to make it work. railway run should use remote env.
    # If it fails, I'll try the replace trick again.
    
    engine = sqlalchemy.create_engine(url)
    with engine.connect() as conn:
        print("--- 1. TABLE COUNTS & FRESHNESS ---")
        sql_counts = """
        SELECT 'mlb_game_log' AS table_name, COUNT(*) AS n, MIN(game_date::text) as min_date, MAX(game_date::text) as max_date FROM mlb_game_log
        UNION ALL SELECT 'mlb_player_stats', COUNT(*), MIN(game_date::text), MAX(game_date::text) FROM mlb_player_stats
        UNION ALL SELECT 'player_rolling_stats', COUNT(*), MIN(as_of_date::text), MAX(as_of_date::text) FROM player_rolling_stats
        UNION ALL SELECT 'player_scores', COUNT(*), MIN(as_of_date::text), MAX(as_of_date::text) FROM player_scores
        UNION ALL SELECT 'player_momentum', COUNT(*), MIN(as_of_date::text), MAX(as_of_date::text) FROM player_momentum
        UNION ALL SELECT 'simulation_results', COUNT(*), MIN(as_of_date::text), MAX(as_of_date::text) FROM simulation_results
        UNION ALL SELECT 'decision_results', COUNT(*), MIN(as_of_date::text), MAX(as_of_date::text) FROM decision_results
        UNION ALL SELECT 'decision_explanations', COUNT(*), MIN(as_of_date::text), MAX(as_of_date::text) FROM decision_explanations;
        """
        results = conn.execute(text(sql_counts)).fetchall()
        print(f"{'Table':25} | {'Count':8} | {'Min Date':10} | {'Max Date':10}")
        print("-" * 65)
        for r in results:
            print(f"{r[0]:25} | {r[1]:8} | {r[2] or 'N/A':10} | {r[3] or 'N/A':10}")

        print("\n--- 2. RECENT JOB RUNS ---")
        sql_jobs = """
        SELECT job_name, status, records_processed, started_at
        FROM job_runs
        WHERE job_name IN (
          'mlb_game_log', 'mlb_box_stats', 'rolling_windows', 'player_scores',
          'player_momentum', 'ros_simulation', 'decision_optimization', 'explainability', 'yahoo_id_sync'
        )
        ORDER BY started_at DESC
        LIMIT 10;
        """
        jobs = conn.execute(text(sql_jobs)).fetchall()
        print(f"{'Job Name':25} | {'Status':10} | {'Records':8} | {'Started At'}")
        print("-" * 75)
        for j in jobs:
            print(f"{j[0]:25} | {j[1]:10} | {j[2] if j[2] is not None else 'N/A':8} | {j[3]}")

        print("\n--- 3. TODAY'S 14D SCORES ---")
        today_scores = conn.execute(text("SELECT COUNT(*) FROM player_scores WHERE as_of_date = CURRENT_DATE AND window_days = 14")).scalar()
        print(f"Today's 14d Scores: {today_scores}")

        print("\n--- 4. MAPPED ROSTER CANDIDATES ---")
        mapped_candidates = conn.execute(text("SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL AND bdl_id IS NOT NULL")).scalar()
        print(f"Mapped Roster Candidates: {mapped_candidates}")

if __name__ == "__main__":
    run_sql_diagnostics()

