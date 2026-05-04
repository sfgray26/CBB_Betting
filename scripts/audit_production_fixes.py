"""
Production System Audit - May 3, 2026

Comprehensive audit of Railway production deployment to verify all P0 fixes.
Tests database schema, data quality, and core functionality.
"""

import subprocess
import json
from datetime import datetime

def run_railway_command(cmd: str) -> tuple[bool, str]:
    """Run a command on Railway and return (success, output)."""
    full_cmd = f"railway run {cmd}"
    try:
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        success = result.returncode == 0
        return success, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)

def audit_database_schema():
    """Verify database schema is correct."""
    print("\n=== 1. DATABASE SCHEMA AUDIT ===")

    # Check player_id_mapping schema
    cmd = """
    from backend.models import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()

    # Check if bdl_id column exists in player_id_mapping
    try:
        result = db.execute(text('SELECT bdl_id FROM player_id_mapping LIMIT 1'))
        print("OK: player_id_mapping.bdl_id column exists")
    except Exception as e:
        print(f"FAILED: player_id_mapping.bdl_id - {e}")
        return False

    # Check if full_name column exists in player_id_mapping
    try:
        result = db.execute(text('SELECT full_name FROM player_id_mapping LIMIT 1'))
        print("OK: player_id_mapping.full_name column exists")
    except Exception as e:
        print(f"FAILED: player_id_mapping.full_name - {e}")
        return False

    # Check if normalized_name column exists in player_id_mapping
    try:
        result = db.execute(text('SELECT normalized_name FROM player_id_mapping LIMIT 1'))
        print("OK: player_id_mapping.normalized_name column exists")
    except Exception as e:
        print(f"FAILED: player_id_mapping.normalized_name - {e}")
        return False

    # Check if source column exists in player_id_mapping
    try:
        result = db.execute(text('SELECT source FROM player_id_mapping LIMIT 1'))
        print("OK: player_id_mapping.source column exists")
    except Exception as e:
        print(f"FAILED: player_id_mapping.source - {e}")
        return False

    db.close()
    return True

def audit_yahoo_id_sync():
    """Verify Yahoo ID sync is working."""
    print("\n=== 2. YAHOO ID SYNC AUDIT ===")

    # Check player_id_mapping table
    cmd = """
    from backend.models import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()

    # Total players
    total = db.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
    print(f"Total players in player_id_mapping: {total:,}")

    # Players with Yahoo IDs
    yahoo = db.execute(text('SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL')).scalar()
    coverage = yahoo / total * 100 if total > 0 else 0
    print(f"Players with Yahoo IDs: {yahoo:,}")
    print(f"Yahoo ID coverage: {coverage:.1f}%")

    # Check for recent entries (last 24 hours)
    recent = db.execute(text('''
        SELECT COUNT(*) FROM player_id_mapping
        WHERE created_at > NOW() - INTERVAL '24 hours'
    ''')).scalar()
    print(f"Players added in last 24h: {recent}")

    db.close()

    # Success criteria
    if coverage >= 10.0:
        print(f"OK: Yahoo ID coverage is {coverage:.1f}% (target: >=10%)")
        return True
    else:
        print(f"FAILED: Yahoo ID coverage is {coverage:.1f}% (target: >=10%)")
        return False

def audit_opponent_stats():
    """Verify opponent stats are populated."""
    print("\n=== 3. OPPONENT STATS AUDIT ===")

    cmd = """
    from backend.models import SessionLocal
    from sqlalchemy import text
    from backend.stat_contract import SCORING_CATEGORY_CODES, PITCHING_CODES

    db = SessionLocal()

    # Check a few sample players have stats
    sample_players = db.execute(text('''
        SELECT bdl_player_id, COUNT(*) as stat_count
        FROM mlb_player_stats
        GROUP BY bdl_player_id
        LIMIT 5
    ''')).fetchall()

    print(f"Sample players with stats: {len(sample_players)}")
    for row in sample_players:
        print(f"  Player {row[0]}: {row[1]} stat records")

    # Check we have pitching stats (ERA, WHIP)
    pitching_stats = db.execute(text('''
        SELECT COUNT(*) FROM mlb_player_stats
        WHERE era IS NOT NULL OR whip IS NOT NULL
    ''')).scalar()
    print(f"Players with pitching stats (ERA/WHIP): {pitching_stats:,}")

    db.close()

    if pitching_stats > 0:
        print("OK: Pitching stats are present in database")
        return True
    else:
        print("FAILED: No pitching stats found")
        return False

def audit_category_tracker():
    """Verify category tracker uses all categories."""
    print("\n=== 4. CATEGORY TRACKER AUDIT ===")

    cmd = """
    from backend.stat_contract import YAHOO_ID_INDEX, SCORING_CATEGORY_CODES, BATTING_CODES, PITCHING_CODES

    print(f"YAHOO_ID_INDEX covers {len(YAHOO_ID_INDEX)} stat_ids")
    print(f"SCORING_CATEGORY_CODES covers {len(SCORING_CATEGORY_CODES)} categories")
    print(f"  Batting: {len([c for c in SCORING_CATEGORY_CODES if c in BATTING_CODES])} categories")
    print(f"  Pitching: {len([c for c in SCORING_CATEGORY_CODES if c in PITCHING_CODES])} categories")

    # Verify pitching categories are included
    pitching_cats = ['W', 'K', 'SV', 'ERA', 'WHIP']
    all_present = all(c in SCORING_CATEGORY_CODES for c in pitching_cats)

    if all_present:
        print(f"OK: All pitching categories mapped in SCORING_CATEGORY_CODES")
        return True
    else:
        print(f"FAILED: Some pitching categories missing from SCORING_CATEGORY_CODES")
        return False
    """

    success, output = run_railway_command(f'python -c "{cmd}"')
    print(output)
    return success

def audit_statcast_loader():
    """Verify Statcast loader has no ROUND() warnings."""
    print("\n=== 5. STATCAST LOADER AUDIT ===")

    # Check if the fix is in place
    cmd = """
    import re
    with open('backend/fantasy_baseball/statcast_loader.py', 'r') as f:
        content = f.read()

    # Check for the fix pattern
    if 'ROUND((SUM(sp.er)::numeric / NULLIF(SUM(sp.ip), 0)) * 9, 2)' in content:
        print("OK: Statcast loader has explicit type casts for ROUND()")
        return True
    else:
        print("FAILED: Statcast loader missing ROUND() fix")
        return False
    """

    success, output = run_railway_command(f'python -c "{cmd}"')
    print(output)
    return success

def audit_park_factors():
    """Verify park factors are loaded."""
    print("\n=== 6. PARK FACTORS AUDIT ===")

    # Check logs for park factor loading
    cmd = "railway logs --tail 200 | grep -i 'park factor' | tail -3"
    success, output = run_railway_command(cmd)

    if "Loaded 81 park factors into memory" in output:
        print("OK: Park factors loaded successfully")
        return True
    else:
        print("PARTIAL: Park factor loading not confirmed in logs")
        return False

def main():
    """Run full system audit."""
    print("="*70)
    print("PRODUCTION SYSTEM AUDIT")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    results = {}

    # Run all audits
    results['database_schema'] = audit_database_schema()
    results['yahoo_id_sync'] = audit_yahoo_id_sync()
    results['opponent_stats'] = audit_opponent_stats()
    results['category_tracker'] = audit_category_tracker()
    results['statcast_loader'] = audit_statcast_loader()
    results['park_factors'] = audit_park_factors()

    # Summary
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)

    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed

    for check, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status}: {check}")

    print()
    print(f"Total checks: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nSUCCESS: All audits passed!")
        return 0
    else:
        print(f"\nFAILED: {failed} audit(s) failed - see details above")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
