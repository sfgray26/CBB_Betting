import sys
import os
sys.path.insert(0, '.')

from sqlalchemy import text
from backend.models import engine

with engine.connect() as conn:
    print("--- player_rolling_stats Null Audit ---")
    query = text("""
        SELECT 
            window_days,
            COUNT(*) as total_rows,
            COUNT(*) FILTER (WHERE w_runs IS NULL) as runs_null,
            COUNT(*) FILTER (WHERE w_hits IS NULL) as hits_null,
            COUNT(*) FILTER (WHERE w_strikeouts_pit IS NULL) as k_pit_null
        FROM player_rolling_stats
        GROUP BY window_days
        ORDER BY window_days
    """)
    result = conn.execute(query)
    for row in result.mappings():
        print(f"Window {row['window_days']}d: Total={row['total_rows']}, R_null={row['runs_null']}, H_null={row['hits_null']}, K_null={row['k_pit_null']}")

    print("\n--- mlb_player_stats Null Audit ---")
    query_mps = text("""
        SELECT 
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE ab IS NULL) as ab_null,
            COUNT(*) FILTER (WHERE ip IS NULL) as ip_null,
            COUNT(*) FILTER (WHERE ab IS NULL AND ip IS NULL) as both_null
        FROM mlb_player_stats
    """)
    mps = conn.execute(query_mps).mappings().first()
    print(f"Total: {mps['total']}, ab_null: {mps['ab_null']}, ip_null: {mps['ip_null']}, both_null: {mps['both_null']}")
