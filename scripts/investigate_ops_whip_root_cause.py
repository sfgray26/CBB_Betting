#!/usr/bin/env python
"""
Root Cause Investigation for ops/whip NULL values in mlb_player_stats.

This script tests the three most likely hypotheses:
1. Conditional check failure
2. BDL API override (BDL returns null, overriding computed values)
3. Code path not executing

Run with: railway run python scripts/investigate_ops_whip_root_cause.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import SessionLocal, MLBPlayerStats
from sqlalchemy import text
import json


def investigate_conditional_checks():
    """Test if the conditional checks are failing."""
    print("\n" + "="*80)
    print("HYPOTHESIS 1: Conditional Check Failure")
    print("="*80)

    db = SessionLocal()
    try:
        result = db.execute(text('''
            SELECT
                COUNT(*) as total,
                COUNT(obp) as has_obp,
                COUNT(slg) as has_slg,
                COUNT(CASE WHEN obp IS NOT NULL AND slg IS NOT NULL THEN 1 END) as ops_computable,
                COUNT(walks_allowed) as has_bb,
                COUNT(hits_allowed) as has_h,
                COUNT(ip) as has_ip,
                COUNT(CASE WHEN walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL AND ip IS NOT NULL THEN 1 END) as whip_computable
            FROM mlb_player_stats
        ''')).fetchone()

        print(f"\nTotal records: {result.total}")
        print(f"Has OBP: {result.has_obp} ({result.has_obp/result.total*100:.1f}%)")
        print(f"Has SLG: {result.has_slg} ({result.has_slg/result.total*100:.1f}%)")
        print(f"OPS computable (has both OBP and SLG): {result.ops_computable} ({result.ops_computable/result.total*100:.1f}%)")
        print(f"Has walks_allowed: {result.has_bb} ({result.has_bb/result.total*100:.1f}%)")
        print(f"Has hits_allowed: {result.has_h} ({result.has_h/result.total*100:.1f}%)")
        print(f"Has IP: {result.has_ip} ({result.has_ip/result.total*100:.1f}%)")
        print(f"WHIP computable (has BB, H, and IP): {result.whip_computable} ({result.whip_computable/result.total*100:.1f}%)")

        if result.ops_computable == 0:
            print("\n❌ CONCLUSION: Conditional check FAILS for all records")
            print("   All records missing OBP and/or SLG")
            return False
        else:
            print(f"\n✅ CONCLUSION: Conditional check should SUCCEED for {result.ops_computable} records")
            return True

    finally:
        db.close()


def investigate_bdl_override():
    """Test if BDL API is overriding computed values."""
    print("\n" + "="*80)
    print("HYPOTHESIS 2: BDL API Override")
    print("="*80)

    db = SessionLocal()
    try:
        # Check raw payload for ops/whip values
        result = db.execute(text('''
            SELECT
                raw_payload,
                ops as ops_in_db,
                whip as whip_in_db,
                obp,
                slg
            FROM mlb_player_stats
            LIMIT 5
        ''')).fetchall()

        print(f"\nSample raw payloads (first 5 records):")

        bdl_provides_ops = False
        bdl_provides_whip = False

        for i, row in enumerate(result, 1):
            payload = json.loads(row.raw_payload) if isinstance(row.raw_payload, str) else row.raw_payload

            ops_from_bdl = payload.get('ops')
            whip_from_bdl = payload.get('whip')
            obp = payload.get('obp')
            slg = payload.get('slg')

            print(f"\nRecord {i}:")
            print(f"  BDL OBP: {obp}")
            print(f"  BDL SLG: {slg}")
            print(f"  BDL OPS: {ops_from_bdl}")
            print(f"  DB OPS: {row.ops_in_db}")
            print(f"  BDL WHIP: {whip_from_bdl}")
            print(f"  DB WHIP: {row.whip_in_db}")

            if ops_from_bdl is not None:
                bdl_provides_ops = True
            if whip_from_bdl is not None:
                bdl_provides_whip = True

        print(f"\n{'='*80}")
        print(f"BDL provides OPS: {bdl_provides_ops}")
        print(f"BDL provides WHIP: {bdl_provides_whip}")

        if bdl_provides_ops and bdl_provides_whip:
            print("\n❌ CONCLUSION: BDL API PROVIDES ops and whip")
            print("   If BDL provides null values, they may override our computations")
            print("   Check: Does raw_payload show ops=null or ops=<numeric value>?")
        elif not bdl_provides_ops and not bdl_provides_whip:
            print("\n✅ CONCLUSION: BDL API does NOT provide ops/whip (returns null/missing)")
            print("   Our computed values should be used")
        else:
            print(f"\n⚠️  MIXED: BDL provides ops={bdl_provides_ops}, whip={bdl_provides_whip}")

        return not (bdl_provides_ops or bdl_provides_whip)

    finally:
        db.close()


def investigate_code_execution():
    """Test if the ingestion code is actually running."""
    print("\n" + "="*80)
    print("HYPOTHESIS 3: Code Path Not Executing")
    print("="*80)

    db = SessionLocal()
    try:
        # Check latest ingestion timestamp
        result = db.execute(text('''
            SELECT
                MAX(ingested_at) as latest_ingestion,
                COUNT(*) as total_records,
                MIN(ingested_at) as earliest_ingestion
            FROM mlb_player_stats
        ''')).fetchone()

        print(f"\nLatest ingestion: {result.latest_ingestion}")
        print(f"Earliest ingestion: {result.earliest_ingestion}")
        print(f"Total records: {result.total_records}")

        if result.latest_ingestion is None:
            print("\n❌ CONCLUSION: No records exist - ingestion code has never run")
            return False
        else:
            print(f"\n✅ CONCLUSION: Ingestion code has run (last: {result.latest_ingestion})")
            return True

    finally:
        db.close()


def test_computation_manually():
    """Manually test the computation logic on actual data."""
    print("\n" + "="*80)
    print("MANUAL COMPUTATION TEST")
    print("="*80)

    db = SessionLocal()
    try:
        # Get a sample record and manually compute ops/whip
        result = db.execute(text('''
            SELECT
                obp,
                slg,
                walks_allowed,
                hits_allowed,
                ip,
                ops,
                whip
            FROM mlb_player_stats
            WHERE obp IS NOT NULL AND slg IS NOT NULL
            LIMIT 1
        ''')).fetchone()

        if result:
            obp = result.obp
            slg = result.slg
            ops_in_db = result.ops

            print(f"\nSample batting record:")
            print(f"  OBP: {obp}")
            print(f"  SLG: {slg}")
            print(f"  OPS in DB: {ops_in_db}")

            if obp is not None and slg is not None:
                computed_ops = obp + slg
                print(f"  Computed OPS: {computed_ops}")
                print(f"  Match: {computed_ops == ops_in_db}")
            else:
                print(f"  Cannot compute - missing source data")

        # Test WHIP computation
        result = db.execute(text('''
            SELECT
                walks_allowed,
                hits_allowed,
                ip,
                whip
            FROM mlb_player_stats
            WHERE walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL AND ip IS NOT NULL
            LIMIT 1
        ''')).fetchone()

        if result:
            bb = result.walks_allowed
            h = result.hits_allowed
            ip = result.ip
            whip_in_db = result.whip

            print(f"\nSample pitching record:")
            print(f"  BB allowed: {bb}")
            print(f"  H allowed: {h}")
            print(f"  IP: {ip}")
            print(f"  WHIP in DB: {whip_in_db}")

            # Simple IP parsing (assume string like "6.2" or integer)
            try:
                if isinstance(ip, str):
                    if '.' in ip:
                        parts = ip.split('.')
                        ip_decimal = int(parts[0]) + int(parts[1]) / 3.0
                    else:
                        ip_decimal = float(ip)
                else:
                    ip_decimal = float(ip)

                if bb is not None and h is not None and ip_decimal is not None and ip_decimal > 0:
                    computed_whip = (bb + h) / ip_decimal
                    print(f"  Parsed IP as decimal: {ip_decimal}")
                    print(f"  Computed WHIP: {computed_whip}")
                    print(f"  Match: {abs(computed_whip - (whip_in_db or 0)) < 0.01}")
            except Exception as e:
                print(f"  Error computing WHIP: {e}")

    finally:
        db.close()


def main():
    """Run all investigations."""
    print("\n" + "="*80)
    print("ROOT CAUSE INVESTIGATION: ops/whip NULL values")
    print("="*80)

    # Test each hypothesis
    conditional_ok = investigate_conditional_checks()
    bdl_ok = investigate_bdl_override()
    code_running = investigate_code_execution()

    # Manual computation test
    test_computation_manually()

    # Final diagnosis
    print("\n" + "="*80)
    print("FINAL DIAGNOSIS")
    print("="*80)

    print(f"\n1. Conditional checks: {'✅ PASS' if conditional_ok else '❌ FAIL'}")
    print(f"2. BDL override issue: {'✅ NO ISSUE' if bdl_ok else '❌ ISSUE DETECTED'}")
    print(f"3. Code execution: {'✅ RUNNING' if code_running else '❌ NOT RUNNING'}")

    if not conditional_ok:
        print("\n🎯 ROOT CAUSE: Conditional check failure")
        print("   All records are missing OBP/SLG (or BB/H/IP)")
        print("   FIX: None - data quality issue, not code bug")
    elif not bdl_ok:
        print("\n🎯 ROOT CAUSE: BDL API override")
        print("   BDL returns ops/whip (possibly null) and overrides our computations")
        print("   FIX: Ensure computed values take precedence over BDL null values")
    elif not code_running:
        print("\n🎯 ROOT CAUSE: Ingestion code not running")
        print("   mlb_box_stats job has never executed or failed")
        print("   FIX: Run the ingestion job manually")
    else:
        print("\n🔍 INCONCLUSIVE")
        print("   All basic checks pass - deeper investigation needed")
        print("   Possible issues:")
        print("   - Transaction rollback")
        print("   - Production code mismatch")
        print("   - Data type conversion issues")


if __name__ == "__main__":
    main()
