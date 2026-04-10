#!/usr/bin/env python3
"""
Direct Railway backfill using DATABASE_URL from Railway environment.
This bypasses the need for local psql installation.
"""

import subprocess
import sys
import os

def get_railway_db_url():
    """Get DATABASE_URL from Railway environment."""
    try:
        result = subprocess.run(
            ['railway', 'variables', '--json'],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        variables = json.loads(result.stdout)
        for var in variables:
            if var.get('name') == 'DATABASE_URL':
                return var.get('value')
        return None
    except Exception as e:
        print(f"Error getting DATABASE_URL: {e}")
        return None

def execute_backfill():
    """Execute backfill using direct SQL execution."""
    db_url = get_railway_db_url()
    if not db_url:
        print("Could not get DATABASE_URL from Railway")
        return False

    print(f"Got DATABASE_URL: {db_url[:30]}...")

    # Set environment variable for database connection
    os.environ['DATABASE_URL'] = db_url

    # Now import and run the backfill
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)

    with engine.connect() as conn:
        # Check initial NULL counts
        print("\n[STATS] Initial NULL counts:")
        initial_ops = conn.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
        )).scalar()
        initial_whip = conn.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
        )).scalar()
        total = conn.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats"
        )).scalar()

        print(f"  ops:    {initial_ops:,} / {total:,} ({initial_ops/total*100:.1f}%)")
        print(f"  whip:   {initial_whip:,} / {total:,} ({initial_whip/total*100:.1f}%)")

        # Backfill ops
        print("\n[EXEC] Backfilling ops...")
        result_ops = conn.execute(text("""
            UPDATE mlb_player_stats
            SET ops = obp + slg
            WHERE ops IS NULL
              AND obp IS NOT NULL
              AND slg IS NOT NULL
        """))
        ops_count = result_ops.rowcount
        print(f"[OK] Backfilled {ops_count:,} ops values")

        # Backfill whip
        print("\n[EXEC] Backfilling whip...")
        result_whip = conn.execute(text("""
            UPDATE mlb_player_stats
            SET whip = (bb_allowed + h_allowed)::numeric /
                      NULLIF(
                          CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
                          CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0,
                          0
                      )
            WHERE whip IS NULL
              AND bb_allowed IS NOT NULL
              AND h_allowed IS NOT NULL
              AND innings_pitched IS NOT NULL
              AND innings_pitched != ''
        """))
        whip_count = result_whip.rowcount
        print(f"[OK] Backfilled {whip_count:,} whip values")

        # Commit the transaction
        conn.commit()

        # Check final NULL counts
        print("\n[STATS] Final NULL counts:")
        final_ops = conn.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
        )).scalar()
        final_whip = conn.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
        )).scalar()

        print(f"  ops:    {final_ops:,} / {total:,} ({final_ops/total*100:.1f}%)")
        print(f"  whip:   {final_whip:,} / {total:,} ({final_whip/total*100:.1f}%)")

        print("\n" + "=" * 60)
        print("[SUCCESS] BACKFILL COMPLETE")
        print("=" * 60)
        print(f"ops backfilled:   {ops_count:,}")
        print(f"whip backfilled:  {whip_count:,}")
        print(f"remaining NULL ops:    {final_ops:,}")
        print(f"remaining NULL whip:   {final_whip:,}")
        print("=" * 60)

        return True

if __name__ == "__main__":
    try:
        success = execute_backfill()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
