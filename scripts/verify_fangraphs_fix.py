"""
Verification script for FanGraphs 403 fix and RoS projection hydration.

Run locally to test:
  1. pybaseball User-Agent patch works (no 403 errors)
  2. RoS projections can be fetched
  3. Projections resolve to player IDs
  4. PlayerProjection table gets real data

Usage:
    python scripts/verify_fangraphs_fix.py

Expected output:
    SUCCESS: All checks passed
"""

import logging
import sys

# Configure logging to see pybaseball activity
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def verify_user_agent_patch() -> bool:
    """Verify pybaseball session has browser User-Agent."""
    print("\n[1/4] Verifying User-Agent patch...")

    try:
        from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent

        # Apply patch
        _patch_pybaseball_user_agent()

        # Check it worked
        import pybaseball
        if hasattr(pybaseball, "session"):
            ua = pybaseball.session.headers.get("User-Agent", "")
            if "Mozilla" in ua and "Chrome" in ua:
                print("  PASS: User-Agent patched to browser UA")
                print(f"  UA: {ua[:60]}...")
                return True
            else:
                print(f"  FAIL: Unexpected User-Agent: {ua}")
                return False
        else:
            print("  FAIL: pybaseball.session not found")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def verify_fangraphs_fetch() -> bool:
    """Verify we can fetch from FanGraphs without 403."""
    print("\n[2/4] Verifying FanGraphs fetch (no 403)...")

    try:
        import pybaseball

        # Try a small fetch (recent stats, not full RoS)
        print("  Fetching sample batting data...")
        df = pybaseball.batting_stats(2026, qual=100)

        if df is not None and len(df) > 0:
            print(f"  PASS: Fetched {len(df)} rows without 403")
            print(f"  Sample players: {', '.join(df['Name'].head(3).tolist())}")
            return True
        else:
            print("  FAIL: Empty response")
            return False
    except Exception as e:
        error_str = str(e)
        if "403" in error_str or "Forbidden" in error_str:
            print(f"  FAIL: Got 403 Forbidden - {e}")
        else:
            print(f"  FAIL: {e}")
        return False


def verify_ros_projection_fetch() -> bool:
    """Verify RoS projection fetch works."""
    print("\n[3/4] Verifying RoS projection fetch...")

    try:
        from backend.database import get_db
        from backend.fantasy_baseball.ros_projection_ingestion import fetch_ros_projections

        db = next(get_db())
        projections = fetch_ros_projections(db, year=2026)

        if projections:
            print(f"  PASS: Fetched {len(projections)} RoS projections")
            # Show sample
            sample = list(projections.values())[:3]
            for proj in sample:
                print(f"    - {proj.player_name}: {proj.hr} HR, {proj.r} R, {proj.rbi} RBI")
            return True
        else:
            print("  FAIL: No projections fetched")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def verify_db_write() -> bool:
    """Verify projections can be written to database."""
    print("\n[4/4] Verifying database write...")

    try:
        from backend.database import get_db
        from backend.fantasy_baseball.ros_projection_ingestion import run_ros_backfill
        from backend.models import PlayerProjection
        from sqlalchemy import select, func

        db = next(get_db())

        # Run backfill
        result = run_ros_backfill(db, year=2026)
        print(f"  Backfill result: {result['message']}")

        # Verify data in database
        total = db.execute(select(func.count()).select_from(PlayerProjection)).scalar()
        if total == 0:
            print("  FAIL: No rows in player_projections table")
            return False

        # Check for variance
        hr_variance = db.execute(
            select(func.variance(PlayerProjection.hr)).select_from(PlayerProjection)
        ).scalar()

        hr_min = db.execute(select(func.min(PlayerProjection.hr)).select_from(PlayerProjection)).scalar()
        hr_max = db.execute(select(func.max(PlayerProjection.hr)).select_from(PlayerProjection)).scalar()

        print(f"  Total rows: {total}")
        print(f"  HR range: {hr_min} - {hr_max}")
        print(f"  HR variance: {hr_variance:.2f}")

        if hr_variance is None or hr_variance <= 0:
            print("  FAIL: HR has zero variance (all players have same HR)")
            return False

        print("  PASS: Real projections with variance in database")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print("FANGRAPHS 403 FIX + RoS PROJECTION VERIFICATION")
    print("=" * 60)

    checks = [
        ("User-Agent patch", verify_user_agent_patch),
        ("FanGraphs fetch", verify_fangraphs_fetch),
        ("RoS projection fetch", verify_ros_projection_fetch),
        ("Database write", verify_db_write),
    ]

    results = []
    for name, check_fn in checks:
        try:
            passed = check_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n{name}: EXCEPTION - {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(passed for _, passed in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("SUCCESS: All checks passed!")
        return 0
    else:
        print("FAILURE: Some checks failed - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
