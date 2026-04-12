"""P-3 Step 4: Apply manual BDL ID overrides for Ohtani and Lorenzen two-way splits."""
from backend.models import SessionLocal
from sqlalchemy import text

# Values from investigation:
OHTANI_BDL_ID = 208
LORENZEN_BDL_ID = 2293

OHTANI_IDS = [169, 198]      # Shohei Ohtani (Batter) + (Pitcher)
LORENZEN_IDS = [957, 1077]   # Michael Lorenzen (Batter) + (Pitcher)

db = SessionLocal()
try:
    # Pre-check: verify target rows are still orphaned
    print("=== Pre-check: target rows ===")
    rows = db.execute(text("""
        SELECT id, yahoo_player_key, player_name, bdl_player_id
        FROM position_eligibility
        WHERE id IN :ids
        ORDER BY id
    """), {"ids": tuple(OHTANI_IDS + LORENZEN_IDS)}).fetchall()
    for r in rows:
        print(dict(r._mapping))

    # Apply Ohtani override
    r1 = db.execute(text("""
        UPDATE position_eligibility
        SET bdl_player_id = :bid
        WHERE id IN :ids AND bdl_player_id IS NULL
    """), {"bid": OHTANI_BDL_ID, "ids": tuple(OHTANI_IDS)})
    print(f"\nOhtani rows updated: {r1.rowcount}")

    # Apply Lorenzen override
    r2 = db.execute(text("""
        UPDATE position_eligibility
        SET bdl_player_id = :bid
        WHERE id IN :ids AND bdl_player_id IS NULL
    """), {"bid": LORENZEN_BDL_ID, "ids": tuple(LORENZEN_IDS)})
    print(f"Lorenzen rows updated: {r2.rowcount}")

    db.commit()

    # Verify
    print("\n=== Post-check: target rows ===")
    rows = db.execute(text("""
        SELECT id, yahoo_player_key, player_name, bdl_player_id
        FROM position_eligibility
        WHERE id IN :ids
        ORDER BY id
    """), {"ids": tuple(OHTANI_IDS + LORENZEN_IDS)}).fetchall()
    for r in rows:
        print(dict(r._mapping))

    # Overall orphan count
    orphan_count = db.execute(text(
        "SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NULL"
    )).scalar()
    print(f"\nTotal orphans remaining: {orphan_count}")
    assert orphan_count == 362, f"Expected 362 orphans remaining, got {orphan_count}"
    print("VERIFY OK")
finally:
    db.close()
