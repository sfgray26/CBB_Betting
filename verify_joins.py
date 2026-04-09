#!/usr/bin/env python
"""
Quick verification of cross-system joins for Task 3.
Run this from the project root with: railway run python verify_joins.py
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from models import SessionLocal
    from sqlalchemy import text

    print("=" * 70)
    print("TASK 3: Cross-System Join Verification")
    print("=" * 70)

    db = SessionLocal()

    # Step 1: Test two-way join
    print("\nStep 1: Two-way join (position_eligibility -> mlb_player_stats)")
    print("-" * 70)

    result1 = db.execute(text('''
        SELECT
            pe.player_name,
            pe.bdl_player_id,
            COUNT(ms.id) as stat_count
        FROM position_eligibility pe
        LEFT JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
        WHERE pe.bdl_player_id IS NOT NULL
        GROUP BY pe.player_name, pe.bdl_player_id
        ORDER BY stat_count DESC
        LIMIT 10
    ''')).fetchall()

    print("Top 10 players with stat counts:")
    for row in result1:
        print(f"  {row.player_name:30s} | BDL: {row.bdl_player_id:8} | Stats: {row.stat_count:4} games")

    # Quality check: Count players with vs without stats
    with_stats = db.execute(text('''
        SELECT COUNT(DISTINCT pe.bdl_player_id)
        FROM position_eligibility pe
        INNER JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
    ''')).scalar()

    total_with_bdl = db.execute(text('''
        SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NOT NULL
    ''')).scalar()

    percentage = (with_stats / total_with_bdl * 100) if total_with_bdl > 0 else 0
    without_stats = total_with_bdl - with_stats

    print(f"\nPlayer stat coverage:")
    print(f"  WITH stats:    {with_stats:4} ({percentage:.1f}%)")
    print(f"  WITHOUT stats: {without_stats:4} ({100-percentage:.1f}%)")
    print(f"  Total:         {total_with_bdl:4}")

    # Step 2: Test three-way join
    print("\nStep 2: Three-way join (position_eligibility -> player_id_mapping -> mlb_player_stats)")
    print("-" * 70)

    result2 = db.execute(text('''
        SELECT
            pe.player_name,
            pim.yahoo_key,
            pim.bdl_id,
            COUNT(ms.id) as stat_count
        FROM position_eligibility pe
        INNER JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
        LEFT JOIN mlb_player_stats ms ON pim.bdl_id = ms.bdl_player_id
        WHERE pe.bdl_player_id IS NOT NULL
        GROUP BY pe.player_name, pim.yahoo_key, pim.bdl_id
        LIMIT 10
    ''')).fetchall()

    print("Cross-system join test (10 players):")
    for row in result2:
        print(f"  {row.player_name:30s} | Yahoo: {row.yahoo_key:12} | BDL: {row.bdl_id:8} | Stats: {row.stat_count:4} games")

    # Consistency check
    two_way = db.execute(text('''
        SELECT COUNT(DISTINCT pe.bdl_player_id)
        FROM position_eligibility pe
        LEFT JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
        WHERE pe.bdl_player_id IS NOT NULL AND ms.id IS NOT NULL
    ''')).scalar()

    three_way = db.execute(text('''
        SELECT COUNT(DISTINCT pim.bdl_id)
        FROM position_eligibility pe
        INNER JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
        LEFT JOIN mlb_player_stats ms ON pim.bdl_id = ms.bdl_player_id
        WHERE pe.bdl_player_id IS NOT NULL AND ms.id IS NOT NULL
    ''')).scalar()

    print(f"\nConsistency check:")
    print(f"  Two-way join:   {two_way} distinct BDL IDs")
    print(f"  Three-way join: {three_way} distinct BDL IDs")
    if two_way == three_way:
        print("  ✓ CONSISTENT")
    else:
        print(f"  ⚠ MISMATCH: {abs(two_way - three_way)} difference")

    # Sample successful joins
    print("\nStep 3: Sample successful cross-system joins")
    print("-" * 70)

    sample = db.execute(text('''
        SELECT
            pe.player_name,
            pe.positions,
            pim.yahoo_key,
            pe.bdl_player_id,
            COUNT(ms.id) as game_count,
            MIN(ms.game_date) as first_game,
            MAX(ms.game_date) as last_game
        FROM position_eligibility pe
        INNER JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
        INNER JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
        GROUP BY pe.player_name, pe.positions, pim.yahoo_key, pe.bdl_player_id
        ORDER BY game_count DESC
        LIMIT 5
    ''')).fetchall()

    print("Top 5 players with successful cross-system joins:")
    for row in sample:
        print(f"\n  {row.player_name} ({row.positions})")
        print(f"    Yahoo Key: {row.yahoo_key}")
        print(f"    BDL ID:    {row.bdl_player_id}")
        print(f"    Games:     {row.game_count}")
        print(f"    Date Range: {row.first_game} to {row.last_game}")

    db.close()

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE ✓")
    print("=" * 70)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)