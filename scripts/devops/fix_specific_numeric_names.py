import sys
import os
sys.path.insert(0, '.')

from sqlalchemy import text
from backend.models import engine

ids_to_fix = ['669743', '657136', '669065', '641598', '642201', '608701']

with engine.connect() as conn:
    print(f"--- Fixing {len(ids_to_fix)} Numeric Names ---")
    for yid in ids_to_fix:
        # 1. Look up name in mapping table
        query = text("SELECT full_name FROM player_id_mapping WHERE yahoo_id = :yid")
        name = conn.execute(query, {"yid": yid}).scalar()
        
        if name:
            print(f"Found mapping: {yid} -> {name}")
            # 2. Update projection
            update = text("UPDATE player_projections SET player_name = :name WHERE player_id = :yid")
            conn.execute(update, {"name": name, "yid": yid})
            conn.commit()
            print(f"✅ Updated {yid}")
        else:
            print(f"❌ No mapping found for Yahoo ID {yid}")
