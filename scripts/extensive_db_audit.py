from sqlalchemy import create_engine, text
import os
import json
from datetime import datetime

db_url = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
engine = create_engine(db_url)

queries = {
    'Table Existence': """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_name IN ('player_id_mapping', 'position_eligibility', 'probable_pitchers')
        ORDER BY table_name;
    """,
    'Row Counts': """
        SELECT 
            (SELECT COUNT(*) FROM player_id_mapping) as player_id_mapping_count,
            (SELECT COUNT(*) FROM position_eligibility) as position_eligibility_count,
            (SELECT COUNT(*) FROM probable_pitchers) as probable_pitchers_count;
    """,
    'Recent Activity': """
        SELECT MAX(created_at) as latest_created, MAX(updated_at) as latest_updated 
        FROM player_id_mapping;
    """,
    'Sample: player_id_mapping': """
        SELECT bdl_id, mlbam_id, full_name, team_abbrev, source 
        FROM player_id_mapping 
        ORDER BY created_at DESC
        LIMIT 5;
    """,
    'Sample: position_eligibility': """
        SELECT bdl_player_id, primary_position, can_play_cf, can_play_lf, can_play_rf, multi_eligibility_count 
        FROM position_eligibility 
        ORDER BY updated_at DESC
        LIMIT 5;
    """,
    'Sample: probable_pitchers': """
        SELECT game_date, team, opponent, pitcher_name, is_confirmed 
        FROM probable_pitchers 
        ORDER BY game_date DESC, fetched_at DESC
        LIMIT 5;
    """
}

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError ("Type %s not serializable" % type(obj))

print("--- Postgres-ygnV (Fantasy DB) EXTENSIVE AUDIT ---")
with engine.connect() as conn:
    for title, sql in queries.items():
        print(f"\n=== {title} ===")
        try:
            result = conn.execute(text(sql))
            rows = result.all()
            if not rows:
                print("No rows found.")
            else:
                for row in rows:
                    # Convert to dict and handle datetimes for clean printing
                    d = dict(row._mapping)
                    print(json.dumps(d, indent=2, default=json_serial))
        except Exception as e:
            print(f"ERROR executing query: {e}")
print("\n--- AUDIT COMPLETE ---")
