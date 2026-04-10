#!/usr/bin/env python3
"""
Backfill ops and whip data for existing NULL values in mlb_player_stats.

Task 26: Populate computed fields that were NULL due to field name bugs.
- ops = obp + slg
- whip = (bb_allowed + h_allowed) / innings_pitched

This version uses the existing database connection from backend.models.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from backend.models import SessionLocal


def backfill_ops():
    """
    Backfill ops = obp + slg for NULL values where source data exists.
    Returns count of updated rows.
    """
    db = SessionLocal()

    sql = text("""
        UPDATE mlb_player_stats
        SET ops = obp + slg
        WHERE ops IS NULL
          AND obp IS NOT NULL
          AND slg IS NOT NULL
    """)

    result = db.execute(sql)
    db.commit()

    count = result.rowcount
    print(f"[OK] Backfilled {count} ops values")
    db.close()
    return count


def backfill_whip():
    """
    Backfill whip = (bb_allowed + h_allowed) / innings_pitched for NULL values.

    Handles innings_pitched string format "6.2" → 6.667 decimal.
    - First part (6) = full innings
    - Second part (2) = outs (0, 1, or 2)
    - Decimal = full_innings + (outs / 3.0)
    """
    db = SessionLocal()

    # SQL to convert "6.2" format to decimal: 6 + 2/3 = 6.667
    sql = text("""
        UPDATE mlb_player_stats
        SET whip = (walks_allowed + hits_allowed)::numeric /
                  NULLIF(
                      CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
                      CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0,
                      0
                  )
        WHERE whip IS NULL
          AND walks_allowed IS NOT NULL
          AND hits_allowed IS NOT NULL
          AND innings_pitched IS NOT NULL
          AND innings_pitched != ''
    """)

    result = db.execute(sql)
    db.commit()

    count = result.rowcount
    print(f"[OK] Backfilled {count} whip values")
    db.close()
    return count


def count_nulls():
    """Count remaining NULL values for ops and whip."""
    db = SessionLocal()

    ops_null = db.execute(text(
        "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
    )).scalar()

    whip_null = db.execute(text(
        "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
    )).scalar()

    total = db.execute(text(
        "SELECT COUNT(*) FROM mlb_player_stats"
    )).scalar()

    print(f"\n[STATS] NULL counts:")
    print(f"  ops:    {ops_null:,} / {total:,} ({ops_null/total*100:.1f}%)")
    print(f"  whip:   {whip_null:,} / {total:,} ({whip_null/total*100:.1f}%)")

    db.close()
    return ops_null, whip_null, total


def main():
    """Execute backfill operations."""
    print("=" * 60)
    print("Task 26: Backfill ops and whip Data")
    print("=" * 60)

    print("\n[STATS] Starting NULL counts:")
    ops_null, whip_null, total = count_nulls()

    print("\n[EXEC] Starting backfill...")
    ops_count = backfill_ops()
    whip_count = backfill_whip()

    print("\n[STATS] Final NULL counts:")
    ops_null, whip_null, total = count_nulls()

    print("\n" + "=" * 60)
    print("[SUCCESS] BACKFILL COMPLETE")
    print("=" * 60)
    print(f"ops backfilled:   {ops_count:,}")
    print(f"whip backfilled:  {whip_count:,}")
    print(f"remaining NULL ops:    {ops_null:,}")
    print(f"remaining NULL whip:   {whip_null:,}")
    print("=" * 60)

    # Exit with error if significant NULLs remain
    if ops_null > 100 or whip_null > 100:
        print("\n[WARNING] Many NULL values remain - investigate source data")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
