"""P-3 Investigation round 2: orphans by player_name."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    print("=== All Ohtani/Lorenzen rows in position_eligibility ===")
    rows = db.execute(text("""
        SELECT id, yahoo_player_key, player_name, first_name, last_name,
               bdl_player_id, primary_position, player_type
        FROM position_eligibility
        WHERE player_name ILIKE '%Ohtani%' OR player_name ILIKE '%Lorenzen%'
           OR last_name ILIKE '%Ohtani%' OR last_name ILIKE '%Lorenzen%'
        ORDER BY player_name, yahoo_player_key
    """)).fetchall()
    print(f"Total: {len(rows)} rows")
    for r in rows:
        print(dict(r._mapping))

    print("\n=== Orphan position_eligibility where any name col has null ===")
    rows = db.execute(text("""
        SELECT COUNT(*) FILTER (WHERE player_name IS NULL) AS null_player_name,
               COUNT(*) FILTER (WHERE first_name IS NULL) AS null_first_name,
               COUNT(*) FILTER (WHERE last_name IS NULL) AS null_last_name
        FROM position_eligibility
        WHERE bdl_player_id IS NULL
    """)).fetchone()
    print(dict(rows._mapping))

    print("\n=== Sample of 15 orphans with names ===")
    rows = db.execute(text("""
        SELECT id, yahoo_player_key, player_name, primary_position, player_type
        FROM position_eligibility
        WHERE bdl_player_id IS NULL
        ORDER BY player_name NULLS LAST
        LIMIT 15
    """)).fetchall()
    for r in rows:
        print(dict(r._mapping))
finally:
    db.close()
