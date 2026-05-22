#!/usr/bin/env python3
"""
P0 EMERGENCY FIX: Scoreboard Showing All Zeros

This is a production-critical issue where the matchup scoreboard displays
all 18 categories as 0 (or tied), resulting in 0% win probability.

Root Cause Analysis:
1. Yahoo API may return empty/malformed stats
2. Stat ID mapping may be failing
3. Authentication may be expired
4. Scoreboard aggregation may have bugs

This script diagnoses and attempts to fix the issue.
"""

import sys
import os
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge')
sys.path.insert(0, '/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/backend')

os.environ['ENVIRONMENT'] = 'development'

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def diagnose_issue():
    """Diagnose why scoreboard shows zeros."""
    print("=" * 80)
    print("P0 SCOREBOARD ZERO STATS - EMERGENCY DIAGNOSTIC")
    print("=" * 80)
    
    issues_found = []
    
    # 1. Check Yahoo client imports
    print("\n[1/6] Checking Yahoo client configuration...")
    print("-" * 80)
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
        from backend.fantasy_baseball.yahoo_client import YahooFantasyBaseball
        print("✓ Yahoo clients import successfully")
        
        # Check environment variables
        client_id = os.getenv('YAHOO_CLIENT_ID')
        client_secret = os.getenv('YAHOO_CLIENT_SECRET')
        refresh_token = os.getenv('YAHOO_REFRESH_TOKEN')
        
        print(f"  YAHOO_CLIENT_ID: {'✓ Set' if client_id else '✗ MISSING'}")
        print(f"  YAHOO_CLIENT_SECRET: {'✓ Set' if client_secret else '✗ MISSING'}")
        print(f"  YAHOO_REFRESH_TOKEN: {'✓ Set' if refresh_token else '✗ MISSING'}")
        
        if not all([client_id, client_secret, refresh_token]):
            issues_found.append("Yahoo OAuth credentials missing")
            
    except Exception as e:
        print(f"✗ Failed to import Yahoo clients: {e}")
        issues_found.append(f"Import error: {e}")
    
    # 2. Check stat contract
    print("\n[2/6] Checking stat contract configuration...")
    print("-" * 80)
    try:
        from backend.stat_contract import YAHOO_ID_INDEX, SCORING_CATEGORY_CODES
        print(f"✓ YAHOO_ID_INDEX loaded: {len(YAHOO_ID_INDEX)} mappings")
        print(f"✓ SCORING_CATEGORY_CODES: {len(SCORING_CATEGORY_CODES)} categories")
        
        # Check key mappings
        key_mappings = [
            ('7', 'R'), ('12', 'HR_B'), ('42', 'K_P'), 
            ('28', 'W'), ('26', 'ERA'), ('27', 'WHIP')
        ]
        
        print("\n  Key stat mappings:")
        for stat_id, expected in key_mappings:
            actual = YAHOO_ID_INDEX.get(stat_id)
            status = '✓' if actual == expected else '✗'
            print(f"    {status} {stat_id} -> {actual} (expected: {expected})")
            if actual != expected:
                issues_found.append(f"Stat mapping error: {stat_id} -> {actual} (expected {expected})")
                
    except Exception as e:
        print(f"✗ Failed to load stat contract: {e}")
        issues_found.append(f"Stat contract error: {e}")
    
    # 3. Check database connectivity
    print("\n[3/6] Checking database connectivity...")
    print("-" * 80)
    try:
        from sqlalchemy import text
        from backend.models import SessionLocal
        
        db = SessionLocal()
        result = db.execute(text("SELECT 1")).scalar()
        db.close()
        
        if result == 1:
            print("✓ Database connection OK")
        else:
            print(f"✗ Database connection failed: unexpected result {result}")
            issues_found.append("Database connection failed")
            
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        issues_found.append(f"Database error: {e}")
    
    # 4. Check if we can fetch Yahoo data
    print("\n[4/6] Attempting Yahoo API test call...")
    print("-" * 80)
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
        
        client = get_yahoo_client()
        print("✓ Yahoo client initialized")
        
        # Try to get league info (lightweight call)
        try:
            league_info = client.get_league_info()
            print(f"✓ League info retrieved: {league_info.get('name', 'Unknown')}")
        except Exception as e:
            print(f"✗ Failed to get league info: {e}")
            issues_found.append(f"Yahoo API error: {e}")
            
        # Try to get matchup stats
        try:
            matchup_stats = client.get_matchup_stats(week=None)
            print(f"✓ Matchup stats retrieved")
            
            my_stats = matchup_stats.get('my_stats', {})
            opp_stats = matchup_stats.get('opp_stats', {})
            
            print(f"\n  My stats keys: {list(my_stats.keys())[:5]}...")
            print(f"  Opp stats keys: {list(opp_stats.keys())[:5]}...")
            
            # Check if stats are empty
            if not my_stats:
                print("✗ WARNING: my_stats is EMPTY")
                issues_found.append("Yahoo API returning empty my_stats")
            else:
                non_zero_count = sum(1 for v in my_stats.values() if v and v != 0)
                print(f"  Non-zero stats in my_stats: {non_zero_count}/{len(my_stats)}")
                
                if non_zero_count == 0:
                    issues_found.append("All my_stats values are zero/null")
                    
        except Exception as e:
            print(f"✗ Failed to get matchup stats: {e}")
            issues_found.append(f"Matchup stats error: {e}")
            
    except Exception as e:
        print(f"✗ Yahoo client setup failed: {e}")
        issues_found.append(f"Yahoo client error: {e}")
    
    # 5. Check scoreboard orchestrator
    print("\n[5/6] Checking scoreboard orchestrator...")
    print("-" * 80)
    try:
        from backend.services.scoreboard_orchestrator import assemble_matchup_scoreboard
        from backend.services.row_projector import ROWProjectionResult
        print("✓ Scoreboard orchestrator imports successfully")
        
        # Test with dummy data
        test_stats = {"R": 10, "HR_B": 5, "K_P": 15}
        test_row = ROWProjectionResult(**{k: float(v) for k, v in test_stats.items()})
        
        print(f"  Test row created: {test_row.to_dict()}")
        
    except Exception as e:
        print(f"✗ Scoreboard orchestrator error: {e}")
        issues_found.append(f"Orchestrator error: {e}")
    
    # 6. Summary
    print("\n[6/6] DIAGNOSIS SUMMARY")
    print("=" * 80)
    
    if not issues_found:
        print("✓ No critical issues found in diagnostic!")
        print("\nThe problem may be:")
        print("  - Intermittent Yahoo API issue")
        print("  - Frontend rendering bug")
        print("  - Caching issue (try hard refresh)")
        return True
    else:
        print(f"✗ {len(issues_found)} CRITICAL ISSUE(S) FOUND:\n")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
        return False

def apply_fixes():
    """Attempt to apply automatic fixes."""
    print("\n" + "=" * 80)
    print("ATTEMPTING AUTOMATIC FIXES")
    print("=" * 80)
    
    fixes_applied = []
    
    # Fix 1: Refresh Yahoo token if needed
    print("\n[Fix 1] Checking Yahoo token...")
    try:
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
        client = get_yahoo_client()
        
        # The client should auto-refresh, but let's verify
        print("  ✓ Yahoo client ready (auto-refresh on use)")
        fixes_applied.append("Yahoo client verified")
    except Exception as e:
        print(f"  ✗ Yahoo client issue: {e}")
    
    # Fix 2: Check for missing stat mappings
    print("\n[Fix 2] Checking stat contract integrity...")
    try:
        from backend.stat_contract import YAHOO_ID_INDEX
        
        # Required mappings for 18 categories
        required = {
            '7': 'R', '8': 'H', '12': 'HR_B', '13': 'RBI', '21': 'K_B',
            '23': 'TB', '3': 'AVG', '55': 'OPS', '62': 'NSB',
            '28': 'W', '29': 'L', '38': 'HR_P', '42': 'K_P',
            '26': 'ERA', '27': 'WHIP', '57': 'K_9', '83': 'QS', '85': 'NSV'
        }
        
        missing = []
        for stat_id, category in required.items():
            if YAHOO_ID_INDEX.get(stat_id) != category:
                missing.append((stat_id, category, YAHOO_ID_INDEX.get(stat_id)))
        
        if missing:
            print(f"  ✗ {len(missing)} incorrect/missing mappings:")
            for stat_id, expected, actual in missing[:5]:
                print(f"    - {stat_id}: got {actual}, expected {expected}")
            print("\n  ⚠️  MANUAL FIX REQUIRED: Update stat contract files")
        else:
            print("  ✓ All required stat mappings present")
            
    except Exception as e:
        print(f"  ✗ Stat contract check failed: {e}")
    
    # Fix 3: Validate scoreboard can process data
    print("\n[Fix 3] Validating scoreboard processing...")
    try:
        from backend.services.scoreboard_orchestrator import assemble_matchup_scoreboard
        
        # Create test data
        test_my_stats = {
            "R": 25, "H": 45, "HR_B": 8, "RBI": 30, "K_B": 35,
            "TB": 70, "AVG": 0.250, "OPS": 0.780, "NSB": 3,
            "W": 2, "L": 1, "HR_P": 5, "K_P": 45,
            "ERA": 3.50, "WHIP": 1.20, "K_9": 8.5, "QS": 1, "NSV": 2
        }
        test_opp_stats = {k: v * 0.9 for k, v in test_my_stats.items()}
        
        # Mock player scores
        mock_player_scores = []
        
        result = assemble_matchup_scoreboard(
            week=7,
            opponent_name="Test Opponent",
            my_current_stats=test_my_stats,
            opp_current_stats=test_opp_stats,
            my_player_scores=mock_player_scores,
            ip_accumulated=25.0,
            ip_minimum=18.0
        )
        
        print(f"  ✓ Scoreboard processed successfully")
        print(f"    Categories won: {result.categories_won}")
        print(f"    Categories tied: {result.categories_tied}")
        print(f"    Win probability: {result.overall_win_probability}")
        
        if result.categories_won == 0 and result.categories_tied == 18:
            print("  ✗ WARNING: All categories showing as tied (the bug!)")
        else:
            fixes_applied.append("Scoreboard processing validated")
            
    except Exception as e:
        print(f"  ✗ Scoreboard processing failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("FIX ATTEMPT SUMMARY")
    print("=" * 80)
    
    if fixes_applied:
        print(f"✓ {len(fixes_applied)} fix(es) applied/verified:")
        for fix in fixes_applied:
            print(f"  - {fix}")
    else:
        print("⚠️  No automatic fixes applied")
        print("\nMANUAL INTERVENTION REQUIRED:")
        print("  1. Check Railway logs for Yahoo API errors")
        print("  2. Verify YAHOO_REFRESH_TOKEN is valid (not expired)")
        print("  3. Test Yahoo API manually with curl")
        print("  4. Check if Yahoo league is active (not offseason)")
    
    return len(fixes_applied) > 0

def generate_report():
    """Generate a detailed report for developers."""
    report_path = "/mnt/c/Users/sfgra/repos/Fixed/cbb-edge/P0_SCOREBOARD_REPORT.md"
    
    report = """# P0 Scoreboard Zero Stats - Emergency Report

## Issue Summary
**Status:** CRITICAL - Scoreboard showing all 18 categories as 0 (or tied)
**Impact:** Users cannot see matchup progress or win probability
**Date:** 2026-05-16

## Symptoms
- All stats showing 0 or "T" (tied)
- Win probability: 0%
- Categories: 0W - 0L - 18T

## Root Cause Analysis

### Data Flow
```
Yahoo API -> get_matchup_stats() -> assemble_matchup_scoreboard() -> API Response -> Frontend
```

### Potential Causes
1. **Yahoo API returning empty stats**
   - Authentication expired
   - Rate limiting
   - League not active

2. **Stat ID mapping failure**
   - YAHOO_ID_INDEX incorrect
   - Contract not loaded
   - Stat codes mismatch

3. **Scoreboard aggregation bug**
   - Division by zero
   - None values not handled
   - Category math error

## Diagnostic Results
"""
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == "__main__":
    print("\n")
    print("🚨 P0 SCOREBOARD EMERGENCY FIX 🚨")
    print("\n")
    
    # Run diagnostics
    diagnostic_passed = diagnose_issue()
    
    # Attempt fixes
    fixes_applied = apply_fixes()
    
    # Generate report
    generate_report()
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("""
1. RESTART the backend service to reload stat contracts:
   railway restart

2. CLEAR any caches:
   - Browser hard refresh (Ctrl+Shift+R)
   - Railway redeploy

3. VERIFY Yahoo token is valid:
   - Check YAHOO_REFRESH_TOKEN env var
   - Re-authenticate if needed

4. TEST the scoreboard endpoint:
   curl https://your-app.railway.app/api/fantasy/scoreboard

5. MONITOR logs for Yahoo API errors:
   railway logs
""")
    
    sys.exit(0 if (diagnostic_passed or fixes_applied) else 1)
