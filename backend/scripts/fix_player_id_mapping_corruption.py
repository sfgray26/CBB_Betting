"""
Fix PlayerIDMapping data corruption - REVISED STRATEGY.

ROOT CAUSE: Duplicate rows where bdl_id contains mlbam_id values.
The optimizer finds the wrong row via yahoo_key and uses wrong bdl_id.

FIX STRATEGY (revised):
1. DELETE the wrong row (yahoo_key + wrong bdl_id)
2. UPDATE the correct row (no yahoo_key + correct bdl_id) to add the yahoo_key
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sqlalchemy import text
from backend.models import SessionLocal, PlayerIDMapping


def fix_corrupted_mapping(db, player_name: str, yahoo_key: str, correct_bdl_id: int, wrong_bdl_id: int):
    """
    Fix corrupted mapping by:
    1. DELETE wrong row (yahoo_key + wrong bdl_id)
    2. UPDATE correct row (no yahoo_key + correct bdl_id) to add yahoo_key
    """
    print(f"\n{'='*60}")
    print(f"Fixing: {player_name}")
    print(f"{'='*60}")
    print(f"Yahoo Key: {yahoo_key}")
    print(f"Wrong BDL ID: {wrong_bdl_id} (to be deleted)")
    print(f"Correct BDL ID: {correct_bdl_id} (to get yahoo_key)")

    # Step 1: Find and DELETE the wrong row
    wrong_row = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.yahoo_key == yahoo_key,
        PlayerIDMapping.bdl_id == wrong_bdl_id
    ).first()

    if wrong_row:
        print(f"Step 1: Deleting wrong row (id={wrong_row.id})")
        db.delete(wrong_row)
        db.commit()
        print(f"  DELETED row id={wrong_row.id}")
    else:
        print(f"Step 1: No wrong row found (may already be fixed)")

    # Step 2: Find the correct row and UPDATE it to add yahoo_key
    correct_row = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.bdl_id == correct_bdl_id
    ).first()

    if not correct_row:
        print(f"Step 2: ERROR - No row found with correct bdl_id={correct_bdl_id}")
        return False

    print(f"Step 2: Updating correct row (id={correct_row.id}) to add yahoo_key")
    print(f"  Before: yahoo_key={correct_row.yahoo_key}, bdl_id={correct_row.bdl_id}")

    correct_row.yahoo_key = yahoo_key
    # Also set yahoo_id if not set
    if not correct_row.yahoo_id and ".p." in yahoo_key:
        correct_row.yahoo_id = yahoo_key.split(".p.", 1)[-1]

    db.commit()
    print(f"  After: yahoo_key={correct_row.yahoo_key}, bdl_id={correct_row.bdl_id}")

    # Verify
    verify_row = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.yahoo_key == yahoo_key
    ).first()
    print(f"Verified: yahoo_key={yahoo_key} -> bdl_id={verify_row.bdl_id}")

    return True


def add_yahoo_key_to_bdl_id(db, player_name: str, yahoo_key: str, bdl_id: int, mlbam_id: int):
    """
    Add yahoo_key to a row that has bdl_id but no yahoo_key.
    """
    print(f"\n{'='*60}")
    print(f"Adding yahoo_key to: {player_name}")
    print(f"{'='*60}")
    print(f"BDL ID: {bdl_id}")
    print(f"Yahoo Key: {yahoo_key}")

    # Find row with bdl_id
    row = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.bdl_id == bdl_id
    ).first()

    if not row:
        print(f"ERROR: No row found with bdl_id={bdl_id}")
        return False

    print(f"Found row: id={row.id}, current yahoo_key={row.yahoo_key}")

    if row.yahoo_key:
        print(f"WARNING: Row already has yahoo_key={row.yahoo_key}")
        return False

    # Update to add yahoo_key
    row.yahoo_key = yahoo_key
    if not row.yahoo_id and ".p." in yahoo_key:
        row.yahoo_id = yahoo_key.split(".p.", 1)[-1]

    db.commit()
    print(f"UPDATED: Added yahoo_key={yahoo_key}")

    return True


def main():
    db = SessionLocal()

    try:
        print("="*60)
        print("PLAYERIDMAPPING CORRUPTION FIX - REVISED")
        print("="*60)

        fixes_applied = 0

        # FIX 1: Sam Antonacci - yahoo_key=469.p.64533 has wrong bdl_id
        # Delete row with (yahoo_key=469.p.64533, bdl_id=803011)
        # Update row with bdl_id=4839465 to add yahoo_key=469.p.64533
        if fix_corrupted_mapping(db, "Sam Antonacci", "469.p.64533", 4839465, 803011):
            fixes_applied += 1

        # FIX 2: Juan Soto - yahoo_key=469.p.10626 has wrong bdl_id
        # Delete row with (yahoo_key=469.p.10626, bdl_id=665742)
        # Update row with bdl_id=1106 to add yahoo_key=469.p.10626
        if fix_corrupted_mapping(db, "Juan Soto", "469.p.10626", 1106, 665742):
            fixes_applied += 1

        # FIX 3: Carson Benge - yahoo_key=469.p.64329 has wrong bdl_id
        # Delete row with (yahoo_key=469.p.64329, bdl_id=701807)
        # Update row with bdl_id=4839085 to add yahoo_key=469.p.64329
        if fix_corrupted_mapping(db, "Carson Benge", "469.p.64329", 4839085, 701807):
            fixes_applied += 1

        # FIX 4: Munetaka Murakami - yahoo_key=469.p.66369 has wrong bdl_id
        # Delete row with (yahoo_key=469.p.66369, bdl_id=808959)
        # Update row with bdl_id=4667586 to add yahoo_key=469.p.66369
        if fix_corrupted_mapping(db, "Munetaka Murakami", "469.p.66369", 4667586, 808959):
            fixes_applied += 1

        # FIX 5: Jordan Walker - need to add yahoo_key for bdl_id=539
        # Need to find his yahoo_key first - skip for now
        print(f"\n{'='*60}")
        print(f"SKIPPED: Jordan Walker (bdl_id=539) - needs yahoo_key discovery")
        print(f"{'='*60}")

        # FIX 6-7: Edwin Díaz and Cristopher Sánchez - need full mapping
        print(f"\n{'='*60}")
        print(f"SKIPPED: Edwin Díaz and Cristopher Sánchez - need Yahoo API lookup")
        print(f"{'='*60}")

        print(f"\n{'='*60}")
        print(f"SUMMARY: {fixes_applied} fixes applied")
        print(f"{'='*60}")

    except Exception as e:
        db.rollback()
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
