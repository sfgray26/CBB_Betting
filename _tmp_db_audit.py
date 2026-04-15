import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("audit_start", flush=True)

try:
    from backend.db import make_engine
    from sqlalchemy import text
    import os
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("NO DATABASE_URL", flush=True)
        sys.exit(1)
    engine = make_engine(db_url)
    print("imports_ok", flush=True)

    queries = [
        ("mlb_player_stats total", 'SELECT COUNT(*) FROM "mlb_player_stats"'),
        ("mlb_player_stats with ops", 'SELECT COUNT(*) FROM "mlb_player_stats" WHERE ops IS NOT NULL'),
        ("mlb_player_stats with whip", 'SELECT COUNT(*) FROM "mlb_player_stats" WHERE whip IS NOT NULL'),
        ("mlb_player_stats with cs", 'SELECT COUNT(*) FROM "mlb_player_stats" WHERE caught_stealing IS NOT NULL'),
        ("statcast_performances total", 'SELECT COUNT(*) FROM "statcast_performances"'),
        ("statcast non-zero launch_speed", 'SELECT COUNT(*) FROM "statcast_performances" WHERE launch_speed > 0'),
        ("statcast non-zero xwoba", 'SELECT COUNT(*) FROM "statcast_performances" WHERE estimated_woba_using_speedangle > 0'),
        ("statcast zero-metric", 'SELECT COUNT(*) FROM "statcast_performances" WHERE launch_speed = 0 AND estimated_woba_using_speedangle = 0 AND barrel = 0'),
        ("player_rolling_stats total", 'SELECT COUNT(*) FROM "player_rolling_stats"'),
        ("player_rolling_stats latest", 'SELECT MAX(as_of_date) FROM "player_rolling_stats"'),
        ("player_scores total", 'SELECT COUNT(*) FROM "player_scores"'),
        ("player_scores latest", 'SELECT MAX(as_of_date) FROM "player_scores"'),
        ("simulation_results total", 'SELECT COUNT(*) FROM "simulation_results"'),
        ("simulation_results latest", 'SELECT MAX(created_at)::date FROM "simulation_results"'),
        ("decision_results total", 'SELECT COUNT(*) FROM "decision_results"'),
        ("probable_pitchers total", 'SELECT COUNT(*) FROM "probable_pitchers"'),
        ("position_eligibility total", 'SELECT COUNT(*) FROM "position_eligibility"'),
        ("player_id_mapping total", 'SELECT COUNT(*) FROM "player_id_mapping"'),
    ]

    with engine.connect() as conn:
        print("connected", flush=True)
        for name, q in queries:
            try:
                r = conn.execute(text(q)).scalar()
                print(f"{name}: {r}", flush=True)
            except Exception as e:
                print(f"{name}: ERR {e}", flush=True)
except Exception as e:
    print(f"FATAL: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("audit_end", flush=True)
