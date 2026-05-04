"""
Simple test for FanGraphs 403 fix.

Run from project root with:
  venv/Scripts/python scripts/test_fangraphs_403_fix.py
"""

import sys
sys.path.insert(0, ".")

print("=" * 60)
print("FANGRAPHS 403 FIX TEST")
print("=" * 60)

# Test 1: Apply User-Agent patch
print("\n[1] Applying User-Agent patch...")
try:
    from backend.fantasy_baseball.pybaseball_loader import _patch_pybaseball_user_agent
    _patch_pybaseball_user_agent()
    print("  PASS: Patch applied")
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 2: Verify requests.get is patched
print("\n[2] Verifying requests.get patch...")
try:
    import requests
    if hasattr(requests, "_pybaseball_patched"):
        print("  PASS: requests.get is patched")
    else:
        print("  FAIL: requests.get not patched")
        sys.exit(1)
except Exception as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# Test 3: Test actual FanGraphs fetch
print("\n[3] Testing FanGraphs fetch (this may take 30s)...")
try:
    import pybaseball
    print("  Fetching batting data (qual=100 to limit results)...")
    df = pybaseball.batting_stats(2026, qual=100)

    if df is not None and len(df) > 0:
        print(f"  PASS: Fetched {len(df)} rows without 403!")
        print(f"  Sample: {df['Name'].head(3).tolist()}")
    else:
        print("  FAIL: Empty response")
        sys.exit(1)
except Exception as e:
    error_str = str(e)
    if "403" in error_str or "Forbidden" in error_str:
        print(f"  FAIL: Still getting 403 - {e}")
        sys.exit(1)
    else:
        print(f"  FAIL: {e}")
        sys.exit(1)

print("\n" + "=" * 60)
print("SUCCESS: All tests passed!")
print("=" * 60)
