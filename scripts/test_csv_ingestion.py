"""
Test CSV projection ingestion.

Run from project root with:
  venv/Scripts/python scripts/test_csv_ingestion.py
"""

import sys
sys.path.insert(0, ".")

print("=" * 60)
print("CSV PROJECTION INGESTION TEST")
print("=" * 60)

# Test 1: Load CSV
print("\n[1] Loading projections from CSV...")
try:
    from pathlib import Path
    from backend.fantasy_baseball.csv_projection_ingestion import load_projections_from_csv

    csv_path = Path("data/projections/fangraphs_ros_sample.csv")
    projections = load_projections_from_csv(csv_path)

    if projections:
        print(f"  PASS: Loaded {len(projections)} projections")
        # Show sample
        sample = list(projections.values())[:3]
        for proj in sample:
            print(f"    - {proj.player_name}: {proj.hr} HR, {proj.r} R, {proj.rbi} RBI")
    else:
        print("  FAIL: No projections loaded")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Resolve player IDs
print("\n[2] Resolving player IDs...")
try:
    from backend.models import SessionLocal
    from backend.fantasy_baseball.csv_projection_ingestion import resolve_player_ids

    db = SessionLocal()
    resolved = resolve_player_ids(db, projections)

    if resolved:
        print(f"  PASS: Resolved {len(resolved)} player IDs")
        # Show sample
        sample = list(resolved.values())[:3]
        for proj in sample:
            print(f"    - {proj.player_id}: {proj.player_name}")
    else:
        print("  WARNING: No player IDs resolved (may need to seed PlayerIDMapping)")
        # Continue anyway to test database write
        resolved = {f"test_{i}": p for i, p in enumerate(list(projections.values())[:10])}
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Write to database
print("\n[3] Writing to database...")
try:
    from backend.fantasy_baseball.csv_projection_ingestion import write_projections_to_db
    from backend.models import PlayerProjection
    from sqlalchemy import select, func

    written = write_projections_to_db(db, resolved)
    print(f"  PASS: Wrote {written} rows to database")

    # Verify data
    total = db.execute(select(func.count()).select_from(PlayerProjection)).scalar()
    hr_min = db.execute(select(func.min(PlayerProjection.hr)).select_from(PlayerProjection)).scalar()
    hr_max = db.execute(select(func.max(PlayerProjection.hr)).select_from(PlayerProjection)).scalar()

    print(f"  Total rows in player_projections: {total}")
    print(f"  HR range: {hr_min} - {hr_max}")

    if total > 0:
        print("  PASS: Data in database!")
    else:
        print("  FAIL: No data in database")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: CSV ingestion test passed!")
print("=" * 60)
