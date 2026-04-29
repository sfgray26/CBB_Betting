import sys
import os
sys.path.insert(0, '.')

from sqlalchemy import text
from backend.models import engine

with engine.connect() as conn:
    print("--- Numeric Name Investigation ---")
    query = text("""
        SELECT pp.player_name, pim.full_name, pim.mlbam_id, pim.bdl_id
        FROM player_projections pp
        LEFT JOIN player_id_mapping pim ON pp.player_name = pim.mlbam_id::text
        WHERE pp.player_name ~ '^[0-9]+$'
    """)
    result = conn.execute(query)
    rows = result.fetchall()
    print(f"Found {len(rows)} numeric-name rows")
    for r in rows:
        print(f"Name: {r[0]} | Mapping: {r[1]} | MLBAM: {r[2]} | BDL: {r[3]}")
