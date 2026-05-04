#!/usr/bin/env python
"""Verification script for P0 fixes Round 2."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Railway production database
os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

from sqlalchemy import text
from backend.models import SessionLocal

def check_player_type_backfill():
    """Verify player_type NULLs were backfilled."""
    print("=" * 60)
    print("CHECK 1: player_type Backfill")
    print("=" * 60)

    db = SessionLocal()
    try:
        # Check current distribution
        rows = db.execute(text('''
          SELECT player_type, COUNT(*)
          FROM player_projections
          GROUP BY player_type
          ORDER BY player_type NULLS FIRST
        ''')).fetchall()

        print("\nplayer_type distribution:")
        for r in rows:
            print(f"  {r[0] if r[0] else 'NULL':10} {r[1]:5}")

        # Check NULLs
        nulls = db.execute(text('''
          SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL
        ''')).scalar()

        print(f"\n[CHECK] Result: {nulls} NULL rows remaining")
        if nulls == 0:
            print("  STATUS: SUCCESS - All rows backfilled")
            return True
        else:
            print(f"  STATUS: FAILED - {nulls} NULLs remain")
            return False
    finally:
        db.close()

def check_yahoo_id_sync():
    """Verify Yahoo ID sync coverage."""
    print("\n" + "=" * 60)
    print("[CHECK] CHECK 2: Yahoo ID Sync Coverage")
    print("=" * 60)

    db = SessionLocal()
    try:
        total = db.execute(text('''
          SELECT COUNT(*) FROM player_id_mapping
        ''')).scalar()

        yahoo = db.execute(text('''
          SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL
        ''')).scalar()

        coverage = (yahoo / total * 100) if total > 0 else 0

        print(f"\nTotal players: {total:,}")
        print(f"Yahoo IDs mapped: {yahoo:,}")
        print(f"Coverage: {coverage:.1f}%")

        print(f"\n[CHECK] Result: {coverage:.1f}% coverage")
        if coverage > 50:
            print("  STATUS: SUCCESS - Above 50% target")
            return True
        elif coverage > 10:
            print("  STATUS: PARTIAL - Improved but below target")
            return False
        else:
            print("  STATUS: FAILED - Still critically low")
            return False
    finally:
        db.close()

def check_park_factors_loaded():
    """Verify park factors were loaded on startup."""
    print("\n" + "=" * 60)
    print("[CHECK] CHECK 3: Park Factors Cache")
    print("=" * 60)

    try:
        from backend.fantasy_baseball.ballpark_factors import _park_factor_cache

        cache_size = len(_park_factor_cache)
        print(f"\nCache entries: {cache_size}")

        if cache_size > 0:
            print("  Sample entries:")
            sample_keys = list(_park_factor_cache.keys())[:5]
            for key in sample_keys:
                print(f"    {key}: {_park_factor_cache[key]:.2f}")

        print(f"\n[CHECK] Result: {cache_size} entries loaded")
        if cache_size >= 30:  # Expected: ~30 parks × 3 factors
            print("  STATUS: SUCCESS - Cache populated")
            return True
        else:
            print("  STATUS: FAILED - Cache not loaded")
            return False
    except Exception as e:
        print(f"\n[CHECK] Result: Cache check failed with error: {e}")
        print("  STATUS: FAILED - Could not verify cache")
        return False

def main():
    """Run all verification checks."""
    print("\n" + "=" * 60)
    print("P0 FIXES VERIFICATION - Round 2")
    print("Date: 2026-05-03")
    print("Commit: c8d50a2 (runtime errors fixed)")
    print("=" * 60)

    results = []
    results.append(("player_type backfill", check_player_type_backfill()))
    results.append(("Yahoo ID sync", check_yahoo_id_sync()))
    results.append(("Park factors cache", check_park_factors_loaded()))

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All checks passed - P0 fixes verified!")
        print("=" * 60)
        return 0
    else:
        print("WARNING: Some checks failed - review above")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
