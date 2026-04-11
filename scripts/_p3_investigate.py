"""P-3 Steps 1-3: Orphan baseline + find Ohtani/Lorenzen in mapping + orphans."""
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Step 1: current orphan count
    total_pe = db.execute(text("SELECT COUNT(*) FROM position_eligibility")).scalar()
    orphans_by_bdl = db.execute(text(
        "SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NULL"
    )).scalar()
    print(f"Total position_eligibility rows: {total_pe}")
    print(f"Orphans (bdl_player_id IS NULL): {orphans_by_bdl}")

    # Step 2: Ohtani + Lorenzen in player_id_mapping (actual columns: mlbam_id, bdl_id)
    print("\n=== player_id_mapping lookups ===")
    for pattern in ("Ohtani", "Lorenzen"):
        rows = db.execute(text("""
            SELECT id, full_name, bdl_id, mlbam_id, yahoo_key, yahoo_id, source
            FROM player_id_mapping
            WHERE full_name ILIKE :p
            ORDER BY id
        """), {"p": f"%{pattern}%"}).fetchall()
        print(f"\n--- {pattern} ({len(rows)} rows) ---")
        for r in rows:
            print(dict(r._mapping))

    # Step 3: Ohtani + Lorenzen orphans in position_eligibility
    print("\n=== position_eligibility orphans (Ohtani/Lorenzen) ===")
    rows = db.execute(text("""
        SELECT id, yahoo_player_key, bdl_player_id
        FROM position_eligibility
        WHERE bdl_player_id IS NULL
          AND (yahoo_player_key IN (
            SELECT yahoo_key FROM player_id_mapping
            WHERE full_name ILIKE '%Ohtani%' OR full_name ILIKE '%Lorenzen%'
          ))
        ORDER BY yahoo_player_key
    """)).fetchall()
    print(f"Found {len(rows)} orphaned position_eligibility rows matching Ohtani/Lorenzen yahoo_keys via player_id_mapping")
    for r in rows:
        print(dict(r._mapping))

    # Also check if position_eligibility has a player_name column at all
    print("\n=== position_eligibility schema check ===")
    cols = db.execute(text("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'position_eligibility'
        ORDER BY ordinal_position
    """)).fetchall()
    for c in cols:
        print(dict(c._mapping))
finally:
    db.close()
