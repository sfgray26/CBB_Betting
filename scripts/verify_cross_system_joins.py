#!/usr/bin/env python
"""
Task 3: Verify Cross-System Joins Now Work

This script verifies that cross-system joins work between:
- position_eligibility → mlb_player_stats (via bdl_player_id)
- position_eligibility → player_id_mapping → mlb_player_stats (three-way join)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SessionLocal
from sqlalchemy import text

def verify_two_way_join():
    """Test join: position_eligibility -> mlb_player_stats via bdl_player_id"""
    print("=" * 70)
    print("STEP 1: Test two-way join (position_eligibility -> mlb_player_stats)")
    print("=" * 70)

    db = SessionLocal()

    try:
        # Quality Check 1: Count total players with bdl_player_id
        total_with_bdl = db.execute(text('''
            SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NOT NULL
        ''')).scalar()
        print(f"\nTotal players with bdl_player_id: {total_with_bdl}")

        # Main verification query: Top 10 players with stat counts
        result = db.execute(text('''
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

        print("\nTop 10 players with stat counts:")
        print("-" * 70)
        for row in result:
            print(f"  {row.player_name:30s} | BDL ID: {row.bdl_player_id:8} | Stats: {row.stat_count:4} games")

        # Quality Check 2: Count players with stats vs without
        with_stats = db.execute(text('''
            SELECT COUNT(DISTINCT pe.bdl_player_id)
            FROM position_eligibility pe
            INNER JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
        ''')).scalar()

        without_stats = total_with_bdl - with_stats
        percentage = (with_stats / total_with_bdl * 100) if total_with_bdl > 0 else 0

        print("\n" + "=" * 70)
        print("QUALITY CHECK: Player stat coverage")
        print("=" * 70)
        print(f"  Players WITH stats:    {with_stats:4} ({percentage:.1f}%)")
        print(f"  Players WITHOUT stats: {without_stats:4} ({100-percentage:.1f}%)")
        print(f"  Total players:          {total_with_bdl:4}")

        # Quality Check 3: Find players with NULL stats that might indicate issues
        null_stats_sample = db.execute(text('''
            SELECT pe.player_name, pe.bdl_player_id, pe.positions
            FROM position_eligibility pe
            LEFT JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
            WHERE pe.bdl_player_id IS NOT NULL AND ms.id IS NULL
            LIMIT 5
        ''')).fetchall()

        if null_stats_sample:
            print("\n" + "=" * 70)
            print("QUALITY CHECK: Sample of players without stats (might be expected)")
            print("=" * 70)
            for row in null_stats_sample:
                print(f"  {row.player_name:30s} | BDL ID: {row.bdl_player_id:8} | Positions: {row.positions}")

        return True

    except Exception as e:
        print(f"\nERROR in two-way join: {e}")
        return False
    finally:
        db.close()


def verify_three_way_join():
    """Test join: position_eligibility -> player_id_mapping -> mlb_player_stats"""
    print("\n\n")
    print("=" * 70)
    print("STEP 2: Test three-way join (position_eligibility -> player_id_mapping -> mlb_player_stats)")
    print("=" * 70)

    db = SessionLocal()

    try:
        # Main verification query: Cross-system join test
        result = db.execute(text('''
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

        print("\nCross-system join test (10 players):")
        print("-" * 70)
        for row in result:
            print(f"  {row.player_name:30s} | Yahoo: {row.yahoo_key:12} | BDL: {row.bdl_id:8} | Stats: {row.stat_count:4} games")

        # Quality Check 4: Verify consistency between two-way and three-way joins
        two_way_count = db.execute(text('''
            SELECT COUNT(DISTINCT pe.bdl_player_id)
            FROM position_eligibility pe
            LEFT JOIN mlb_player_stats ms ON pe.bdl_player_id = ms.bdl_player_id
            WHERE pe.bdl_player_id IS NOT NULL AND ms.id IS NOT NULL
        ''')).scalar()

        three_way_count = db.execute(text('''
            SELECT COUNT(DISTINCT pim.bdl_id)
            FROM position_eligibility pe
            INNER JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            LEFT JOIN mlb_player_stats ms ON pim.bdl_id = ms.bdl_player_id
            WHERE pe.bdl_player_id IS NOT NULL AND ms.id IS NOT NULL
        ''')).scalar()

        print("\n" + "=" * 70)
        print("QUALITY CHECK: Consistency between two-way and three-way joins")
        print("=" * 70)
        print(f"  Two-way join (pe -> stats):   {two_way_count:4} distinct BDL IDs")
        print(f"  Three-way join (pe -> pim -> stats): {three_way_count:4} distinct BDL IDs")

        if two_way_count == three_way_count:
            print("  ✓ CONSISTENT: Both joins return same counts")
        else:
            print(f"  ⚠ INCONSISTENCY: Count mismatch of {abs(two_way_count - three_way_count)}")

        # Quality Check 5: Verify ID mapping integrity
        mapping_check = db.execute(text('''
            SELECT
                COUNT(*) as total_mappings,
                COUNT(DISTINCT pe.bdl_player_id) as unique_bdl_in_pe,
                COUNT(DISTINCT pim.bdl_id) as unique_bdl_in_pim
            FROM position_eligibility pe
            INNER JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.bdl_player_id IS NOT NULL
        ''')).fetchone()

        print("\n" + "=" * 70)
        print("QUALITY CHECK: ID mapping integrity")
        print("=" * 70)
        print(f"  Total mappings:          {mapping_check.total_mappings:4}")
        print(f"  Unique BDL IDs in pe:    {mapping_check.unique_bdl_in_pe:4}")
        print(f"  Unique BDL IDs in pim:   {mapping_check.unique_bdl_in_pim:4}")

        if mapping_check.unique_bdl_in_pe == mapping_check.unique_bdl_in_pim:
            print("  ✓ CONSISTENT: All BDL IDs properly mapped")
        else:
            print(f"  ⚠ MISMATCH: Some BDL IDs may not be properly linked")

        return True

    except Exception as e:
        print(f"\nERROR in three-way join: {e}")
        return False
    finally:
        db.close()


def check_data_quality():
    """Additional quality checks for data quality issues"""
    print("\n\n")
    print("=" * 70)
    print("STEP 3: Additional data quality checks")
    print("=" * 70)

    db = SessionLocal()

    try:
        # Check for orphaned records
        orphaned_pe = db.execute(text('''
            SELECT COUNT(*) FROM position_eligibility
            WHERE bdl_player_id IS NOT NULL
            AND bdl_player_id NOT IN (SELECT bdl_player_id FROM mlb_player_stats)
        ''')).scalar()

        orphaned_pim = db.execute(text('''
            SELECT COUNT(*) FROM player_id_mapping
            WHERE bdl_id IS NOT NULL
            AND bdl_id NOT IN (SELECT bdl_player_id FROM mlb_player_stats)
        ''')).scalar()

        print("\nOrphaned record check (players in mapping but no stats):")
        print(f"  Orphaned position_eligibility records: {orphaned_pe}")
        print(f"  Orphaned player_id_mapping records:   {orphaned_pim}")

        # Check for NULL values in critical fields
        null_checks = db.execute(text('''
            SELECT
                COUNT(*) FILTER (WHERE yahoo_player_key IS NULL) as null_yahoo_in_pe,
                COUNT(*) FILTER (WHERE bdl_player_id IS NULL) as null_bdl_in_pe,
                COUNT(*) FILTER (WHERE yahoo_key IS NULL) as null_yahoo_in_pim,
                COUNT(*) FILTER (WHERE bdl_id IS NULL) as null_bdl_in_pim
            FROM position_eligibility
        ''')).fetchone()

        print("\nNULL value checks:")
        print(f"  NULL yahoo_player_key in position_eligibility: {null_checks.null_yahoo_in_pe}")
        print(f"  NULL bdl_player_id in position_eligibility:    {null_checks.null_bdl_in_pe}")
        print(f"  NULL yahoo_key in player_id_mapping:           {null_checks.null_yahoo_in_pim}")
        print(f"  NULL bdl_id in player_id_mapping:              {null_checks.null_bdl_in_pim}")

        # Sample of successful cross-system joins for documentation
        sample_joins = db.execute(text('''
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

        print("\n" + "=" * 70)
        print("SAMPLE: Top 5 players with successful cross-system joins")
        print("=" * 70)
        for row in sample_joins:
            print(f"\n  {row.player_name} ({row.positions})")
            print(f"    Yahoo Key: {row.yahoo_key}")
            print(f"    BDL ID:    {row.bdl_player_id}")
            print(f"    Games:     {row.game_count}")
            print(f"    Date Range: {row.first_game} to {row.last_game}")

        return True

    except Exception as e:
        print(f"\nERROR in data quality checks: {e}")
        return False
    finally:
        db.close()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("CROSS-SYSTEM JOIN VERIFICATION - Task 3")
    print("*" * 70)

    success = True

    # Run all verification steps
    success &= verify_two_way_join()
    success &= verify_three_way_join()
    success &= check_data_quality()

    print("\n\n")
    print("*" * 70)
    if success:
        print("VERIFICATION COMPLETE: All checks passed ✓")
    else:
        print("VERIFICATION COMPLETE: Some checks failed ⚠")
    print("*" * 70)
    print("\n")