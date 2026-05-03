
from datetime import date, timedelta
from sqlalchemy import text
from backend.models import SessionLocal, MLBPlayerStats, PlayerRollingStats
from backend.services.rolling_window_engine import compute_all_rolling_windows

def test_date(as_of_date_str):
    as_of_date = date.fromisoformat(as_of_date_str)
    db = SessionLocal()
    try:
        lookback_start = as_of_date - timedelta(days=30)
        stat_rows = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.game_date >= lookback_start,
            MLBPlayerStats.game_date <= as_of_date,
        ).all()
        print(f"Stats found: {len(stat_rows)}")
        
        results = compute_all_rolling_windows(stat_rows, as_of_date=as_of_date)
        print(f"Computed results: {len(results)}")
        result_map = {(r.bdl_player_id, r.window_days): r for r in results}
        
        null_rows = db.query(PlayerRollingStats.id, PlayerRollingStats.bdl_player_id, PlayerRollingStats.window_days).filter(
            PlayerRollingStats.as_of_date == as_of_date,
            PlayerRollingStats.w_runs.is_(None),
        ).all()
        print(f"Null rows found: {len(null_rows)}")
        
        updates = []
        for row in null_rows:
            computed = result_map.get((row.bdl_player_id, row.window_days))
            if computed:
                updates.append((row.id, computed.w_runs))
        
        print(f"Computable updates: {len(updates)}")
        if updates:
            print(f"First update: {updates[0]}")
            
    finally:
        db.close()

if __name__ == "__main__":
    test_date("2026-03-27")
