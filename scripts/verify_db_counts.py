from sqlalchemy import create_engine, text
import os

db_url_fantasy = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
db_url_legacy = "postgresql://postgres:nrvjuGWnjwOttjEiesPTGJwVxSfzNDCV@shinkansen.proxy.rlwy.net:17252/railway"

tables = [
    'position_eligibility',
    'player_id_mapping',
    'probable_pitchers',
    'mlb_game_log',
    'mlb_player_stats',
    'statcast_performances'
]

def audit_db(url, name):
    print(f"--- {name} Audit ---")
    engine = create_engine(url)
    with engine.connect() as conn:
        for table in tables:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"{table:25} : {count}")
            except Exception as e:
                print(f"{table:25} : ERROR ({e})")
    print()

audit_db(db_url_fantasy, "Postgres-ygnV (Fantasy DB)")
audit_db(db_url_legacy, "Postgres (Legacy DB)")
