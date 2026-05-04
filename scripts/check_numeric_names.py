#!/usr/bin/env python
"""Check numeric player names in player_projections table."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    result = db.execute(text("""
        SELECT pp.id, pp.player_id, pp.player_name
        FROM player_projections pp
        WHERE pp.player_name ~ '^[0-9]+$'
        LIMIT 10
    """)).fetchall()
    print(f"Found {len(result)} numeric-name projections:")
    for r in result:
        print(f"  id={r[0]}, player_id={r[1]}, name={r[2]}")
finally:
    db.close()
