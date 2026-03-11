#!/usr/bin/env python3
"""
OpenClaw Health Check Script

Quick diagnostic tool to verify OpenClaw is functioning correctly.
Run this before deployments or when troubleshooting.

Usage:
    python .openclaw/health_check.py
    python .openclaw/health_check.py --verbose
    python .openclaw/health_check.py --test-escalation

Exit codes:
    0 = All checks passed
    1 = One or more checks failed
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports() -> tuple[bool, str]:
    """Check that all required modules can be imported."""
    try:
        from backend.services.openclaw_lite import (
            OpenClawLite,
            IntegrityResult,
            perform_sanity_check,
            async_perform_sanity_check,
            get_openclaw_lite,
            get_escalation_queue,
        )
        return True, "All imports successful"
    except ImportError as e:
        return False, f"Import failed: {e}"


def check_heuristic_performance() -> tuple[bool, str]:
    """Check that heuristic checks are fast."""
    try:
        from backend.services.openclaw_lite import get_openclaw_lite
        
        checker = get_openclaw_lite(enable_telemetry=False)
        
        # Run 100 checks
        start = time.time()
        for _ in range(100):
            result = checker._check_integrity_heuristic_sync(
                search_text="Team looking good, no injuries.",
                home_team="Duke",
                away_team="UNC",
                recommended_units=0.5
            )
        elapsed = time.time() - start
        
        avg_ms = (elapsed / 100) * 1000
        if avg_ms < 1.0:  # Should be under 1ms
            return True, f"Avg latency: {avg_ms:.3f}ms (100 checks)"
        else:
            return False, f"Slow: {avg_ms:.3f}ms average (expected <1ms)"
    except Exception as e:
        return False, f"Performance check failed: {e}"


def check_verdict_accuracy() -> tuple[bool, str]:
    """Check that verdicts are accurate on known test cases."""
    try:
        from backend.services.openclaw_lite import get_openclaw_lite, IntegrityVerdict
        
        checker = get_openclaw_lite(enable_telemetry=False)
        
        test_cases = [
            # (search_text, expected_verdict, description)
            ("Everything looks good.", IntegrityVerdict.CONFIRMED.value, "clean"),
            ("Star player out with injury.", IntegrityVerdict.ABORT.value, "critical"),
            ("Conflicting reports about status.", IntegrityVerdict.VOLATILE.value, "conflict"),
            ("Player is questionable.", IntegrityVerdict.CAUTION.value, "risk"),
        ]
        
        passed = 0
        for text, expected, desc in test_cases:
            result = checker._check_integrity_heuristic_sync(
                search_text=text,
                home_team="Team A",
                away_team="Team B",
                recommended_units=0.5
            )
            if expected in result.verdict:
                passed += 1
        
        if passed == len(test_cases):
            return True, f"All {len(test_cases)} test cases passed"
        else:
            return False, f"Only {passed}/{len(test_cases)} test cases passed"
    except Exception as e:
        return False, f"Accuracy check failed: {e}"


def check_escalation_queue() -> tuple[bool, str]:
    """Check that escalation queue is working."""
    try:
        from backend.services.openclaw_lite import get_escalation_queue
        
        queue = get_escalation_queue()
        
        # Test enqueue
        queue_id = queue.enqueue(
            game_key="Test@Game",
            home_team="Test Home",
            away_team="Test Away",
            recommended_units=2.0,
            integrity_verdict="CAUTION",
            reason="Health check test"
        )
        
        # Verify file exists
        queue_file = queue.queue_dir / f"{queue_id}.json"
        if not queue_file.exists():
            return False, "Escalation file not created"
        
        # Clean up
        queue_file.unlink()
        
        return True, "Escalation queue functional"
    except Exception as e:
        return False, f"Escalation check failed: {e}"


async def check_async_functionality() -> tuple[bool, str]:
    """Check that async operations work correctly."""
    try:
        from backend.services.openclaw_lite import async_perform_sanity_check
        
        # Run multiple checks concurrently
        tasks = [
            async_perform_sanity_check(
                home_team="Team A",
                away_team="Team B",
                verdict="Bet 1.0u Team A -4",
                search_results="No issues."
            )
            for _ in range(10)
        ]
        
        start = time.time()
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        
        # All should return valid verdicts
        valid_verdicts = {"CONFIRMED", "CAUTION", "VOLATILE", "ABORT", "RED FLAG"}
        all_valid = all(r in valid_verdicts for r in results)
        
        if all_valid and elapsed < 1.0:  # Should complete quickly
            return True, f"Async concurrent: {len(results)} checks in {elapsed:.3f}s"
        else:
            return False, f"Async issues: valid={all_valid}, time={elapsed:.3f}s"
    except Exception as e:
        return False, f"Async check failed: {e}"


async def check_telemetry() -> tuple[bool, str]:
    """Check that telemetry is working."""
    try:
        from backend.services.openclaw_lite import OpenClawLite
        
        # Create a fresh instance with telemetry enabled
        checker = OpenClawLite(enable_telemetry=True)
        
        # Run some checks via the async method (which records telemetry)
        for i in range(5):
            await checker.check_integrity(
                search_text="Test",
                home_team="A",
                away_team="B",
                recommended_units=0.5,
                game_key=f"test_{i}"
            )
        
        telemetry = checker.get_telemetry()
        if telemetry and telemetry.get("total_checks") == 5:
            return True, f"Telemetry tracking: {telemetry['total_checks']} checks"
        else:
            return False, f"Telemetry not tracking correctly (got {telemetry.get('total_checks', 0)} checks)"
    except Exception as e:
        return False, f"Telemetry check failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Health Check")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--test-escalation", action="store_true", help="Test escalation queue")
    args = parser.parse_args()
    
    print("=" * 60)
    print("OpenClaw Health Check v3.0")
    print("=" * 60)
    
    checks = [
        ("Imports", check_imports),
        ("Heuristic Performance", check_heuristic_performance),
        ("Verdict Accuracy", check_verdict_accuracy),
        ("Escalation Queue", check_escalation_queue),
    ]
    
    # Async checks
    async def run_async_checks():
        telemetry_result = await check_telemetry()
        async_result = await check_async_functionality()
        return [
            ("Telemetry", telemetry_result[0], telemetry_result[1]),
            ("Async Functionality", async_result[0], async_result[1]),
        ]
    
    results = []
    
    # Run synchronous checks
    for name, check_fn in checks:
        print(f"\n[CHECK] {name}...")
        try:
            passed, message = check_fn()
            status = "PASS" if passed else "FAIL"
            results.append((name, passed, message))
            print(f"   [{status}]: {message}")
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"   [FAIL]: {e}")
    
    # Run async checks
    print(f"\n[CHECK] Running async checks...")
    try:
        async_results = asyncio.run(run_async_checks())
        for name, passed, message in async_results:
            status = "PASS" if passed else "FAIL"
            results.append((name, passed, message))
            print(f"   [{status}] {name}: {message}")
    except Exception as e:
        results.append(("Async Checks", False, str(e)))
        print(f"   [FAIL] Async checks: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    
    for name, passed, message in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if args.verbose:
            print(f"   {message}")
    
    print("-" * 60)
    print(f"Result: {passed_count}/{total_count} checks passed")
    
    if passed_count == total_count:
        print("[OK] All systems operational")
        return 0
    else:
        print("[WARN] Some checks failed - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
