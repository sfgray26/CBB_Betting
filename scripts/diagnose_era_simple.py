"""
Simplified ERA diagnostic for Railway Python shell.

Usage:
    railway run --service Fantasy-App -- python
    >>> exec(open('scripts/diagnose_era_simple.py').read())
"""

import os
from sqlalchemy import create_engine, text

# Get DATABASE_URL from Railway environment
database_url = os.getenv('DATABASE_URL')
if not database_url:
    print("ERROR: DATABASE_URL not found in environment")
    exit(1)

# Create engine
engine = create_engine(database_url)

print("=" * 70)
print("ERA VALUE INVESTIGATION REPORT")
print("=" * 70)
print()

with engine.connect() as conn:
    # Query 1: Overall ERA distribution
    print("## QUERY 1: Overall ERA Distribution")
    print("-" * 70)

    result = conn.execute(text("""
        SELECT
            COUNT(*) as total_rows,
            COUNT(era) as rows_with_era,
            COUNT(*) - COUNT(era) as rows_null_era,
            MIN(era) as min_era,
            MAX(era) as max_era,
            ROUND(AVG(era), 3) as avg_era
        FROM mlb_player_stats
    """)).fetchone()

    print(f"Total rows:           {result.total_rows:,}")
    print(f"Rows with ERA:        {result.rows_with_era:,}")
    print(f"Rows with NULL ERA:   {result.rows_null_era:,}")
    print(f"Min ERA:              {result.min_era}")
    print(f"Max ERA:              {result.max_era}")
    print(f"Avg ERA:              {result.avg_era}")
    print()

    # Query 2: Find extremely high ERA values (> 50)
    print("## QUERY 2: Extremely High ERA Values (> 50)")
    print("-" * 70)

    high_era = conn.execute(text("""
        SELECT
            bdl_player_id,
            era,
            earned_runs,
            innings_pitched,
            game_id
        FROM mlb_player_stats
        WHERE era IS NOT NULL AND era > 50
        ORDER BY era DESC
        LIMIT 10
    """)).fetchall()

    if high_era:
        print(f"Found {len(high_era)} rows with ERA > 50:")
        print()
        for row in high_era:
            print(f"  bdl_id={row.bdl_player_id}, era={row.era:.3f}, "
                  f"er={row.earned_runs}, ip={row.innings_pitched}, "
                  f"game={row.game_id}")

            # Manually calculate ERA to check
            if row.innings_pitched and row.innings_pitched > 0:
                calculated_era = (row.earned_runs / row.innings_pitched) * 9
                match = "✓" if abs(calculated_era - row.era) < 0.01 else "✗ MISMATCH"
                print(f"    → Calculated ERA: {calculated_era:.3f} {match}")
            print()
    else:
        print("No rows with ERA > 50 found.")
        print()

    # Query 3: Find extremely low ERA values (< 1.0)
    print("## QUERY 3: Extremely Low ERA Values (< 1.0)")
    print("-" * 70)

    low_era = conn.execute(text("""
        SELECT
            bdl_player_id,
            era,
            earned_runs,
            innings_pitched,
            game_id
        FROM mlb_player_stats
        WHERE era IS NOT NULL AND era < 1.0
        ORDER BY era ASC
        LIMIT 10
    """)).fetchall()

    if low_era:
        print(f"Found {len(low_era)} rows with ERA < 1.0:")
        print()
        for row in low_era:
            print(f"  bdl_id={row.bdl_player_id}, era={row.era:.3f}, "
                  f"er={row.earned_runs}, ip={row.innings_pitched}, "
                  f"game={row.game_id}")
            print(f"    → Note: ERA < 1.0 is excellent (not impossible)")
            print()
    else:
        print("No rows with ERA < 1.0 found.")
        print()

    # Query 4: Check for the specific ERA value mentioned
    print("## QUERY 4: Check for ERA = 1.726 (mentioned in task summary)")
    print("-" * 70)

    specific_era = conn.execute(text("""
        SELECT
            bdl_player_id,
            era,
            earned_runs,
            innings_pitched,
            game_id
        FROM mlb_player_stats
        WHERE era = 1.726
    """)).fetchall()

    if specific_era:
        print(f"Found {len(specific_era)} rows with ERA = 1.726:")
        print()
        for row in specific_era:
            print(f"  bdl_id={row.bdl_player_id}, era={row.era:.3f}, "
                  f"er={row.earned_runs}, ip={row.innings_pitched}, "
                  f"game={row.game_id}")
            print(f"    → Note: ERA of 1.726 is excellent (NOT impossible)")
            print()
    else:
        print("No rows found with ERA = 1.726.")
        print()

    # Query 5: Summary
    print("=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    print()

    # Count rows in each category
    count_gt_100 = conn.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 100")).scalar()
    count_gt_50 = conn.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 50")).scalar()
    count_lt_1 = conn.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era < 1")).scalar()
    count_null = conn.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era IS NULL")).scalar()

    print(f"ERA > 100 (truly impossible):  {count_gt_100}")
    print(f"ERA > 50 (very high):           {count_gt_50}")
    print(f"ERA < 1 (excellent but rare):   {count_lt_1}")
    print(f"ERA NULL:                      {count_null}")
    print()

    # Recommendations
    if count_gt_100 > 0:
        print("🔴 ACTION REQUIRED:")
        print(f"   - {count_gt_100} rows have impossible ERA (> 100)")
        print("   - These should be investigated and fixed")
        print()
    elif count_gt_50 > 0:
        print("🟡 REVIEW RECOMMENDED:")
        print(f"   - {count_gt_50} rows have very high ERA (> 50)")
        print("   - Verify if these are legitimate or data errors")
        print()
    else:
        print("✅ NO IMPOSSIBLE ERA VALUES FOUND")
        print("   - All ERA values appear to be in valid range")
        print("   - Task 10 may need to be updated or closed")
        print()

print()
print("=" * 70)
print("END OF ERA INVESTIGATION REPORT")
print("=" * 70)
