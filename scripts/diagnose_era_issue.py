"""
Diagnostic Script: ERA Value Investigation

This script investigates the ERA values in mlb_player_stats table to find
any impossible or problematic values that need correction.

Context: Task 10 requires fixing an "impossible ERA value" but the actual
value needs to be verified (1.726 is excellent, not impossible).

Usage on Railway:
    railway run --service Fantasy-App -- python scripts/diagnose_era_issue.py

Output: Markdown report with ERA distribution and problematic rows
"""

import sys
import os
sys.path.insert(0, ".")

# Railway sets env vars automatically
from sqlalchemy import create_engine, text
from backend.models import SessionLocal

def main():
    db = SessionLocal()

    try:
        print("=" * 70)
        print("ERA VALUE INVESTIGATION REPORT")
        print("=" * 70)
        print()

        # Query 1: Overall ERA distribution
        print("## QUERY 1: Overall ERA Distribution")
        print("-" * 70)

        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(era) as rows_with_era,
                COUNT(*) - COUNT(era) as rows_null_era,
                MIN(era) as min_era,
                MAX(era) as max_era,
                ROUND(AVG(era), 3) as avg_era,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY era), 3) as median_era
            FROM mlb_player_stats
        """)).fetchone()

        print(f"Total rows:           {result.total_rows:,}")
        print(f"Rows with ERA:        {result.rows_with_era:,}")
        print(f"Rows with NULL ERA:   {result.rows_null_era:,}")
        print(f"Min ERA:              {result.min_era}")
        print(f"Max ERA:              {result.max_era}")
        print(f"Avg ERA:              {result.avg_era}")
        print(f"Median ERA:           {result.median_era}")
        print()

        # Query 2: Find extremely high ERA values (> 50)
        print("## QUERY 2: Extremely High ERA Values (> 50)")
        print("-" * 70)

        high_era = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                opponent_team
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
                      f"game={row.game_id}, opp={row.opponent_team}")

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

        low_era = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                opponent_team
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
                      f"game={row.game_id}, opp={row.opponent_team}")

                # Manually calculate ERA to check
                if row.innings_pitched and row.innings_pitched > 0:
                    calculated_era = (row.earned_runs / row.innings_pitched) * 9
                    match = "✓" if abs(calculated_era - row.era) < 0.01 else "✗ MISMATCH"
                    print(f"    → Calculated ERA: {calculated_era:.3f} {match}")
                print()
        else:
            print("No rows with ERA < 1.0 found.")
            print()

        # Query 4: Find rows with ERA calculation mismatches
        print("## QUERY 4: ERA Calculation Mismatches")
        print("-" * 70)

        mismatches = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                ((earned_runs::float / NULLIF(innings_pitched, 0)) * 9) as calculated_era
            FROM mlb_player_stats
            WHERE era IS NOT NULL
              AND innings_pitched IS NOT NULL
              AND innings_pitched > 0
              AND earned_runs IS NOT NULL
              AND ABS(era - ((earned_runs::float / innings_pitched) * 9)) > 0.01
            ORDER BY ABS(era - ((earned_runs::float / innings_pitched) * 9)) DESC
            LIMIT 10
        """)).fetchall()

        if mismatches:
            print(f"Found {len(mismatches)} rows with ERA calculation mismatches:")
            print()
            for row in mismatches:
                diff = abs(row.era - row.calculated_era)
                print(f"  bdl_id={row.bdl_player_id}:")
                print(f"    Stored ERA:      {row.era:.3f}")
                print(f"    Calculated ERA:  {row.calculated_era:.3f}")
                print(f"    Difference:      {diff:.3f}")
                print(f"    er={row.earned_runs}, ip={row.innings_pitched}, game={row.game_id}")
                print()
        else:
            print("No ERA calculation mismatches found.")
            print()

        # Query 5: Check for the specific ERA value mentioned in task (1.726)
        print("## QUERY 5: Check for ERA = 1.726 (mentioned in task summary)")
        print("-" * 70)

        specific_era = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                opponent_team
            FROM mlb_player_stats
            WHERE era = 1.726
        """)).fetchall()

        if specific_era:
            print(f"Found {len(specific_era)} rows with ERA = 1.726:")
            print()
            for row in specific_era:
                print(f"  bdl_id={row.bdl_player_id}, era={row.era:.3f}, "
                      f"er={row.earned_runs}, ip={row.innings_pitched}, "
                      f"game={row.game_id}, opp={row.opponent_team}")
                print(f"    → Note: ERA of 1.726 is excellent (not impossible)")
                print()
        else:
            print("No rows found with ERA = 1.726.")
            print()

        # Query 6: Summary and recommendations
        print("=" * 70)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 70)
        print()

        # Count rows in each category
        count_gt_100 = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 100")).scalar()
        count_gt_50 = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 50")).scalar()
        count_lt_1 = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era < 1")).scalar()
        count_null = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era IS NULL")).scalar()

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

        if count_lt_1 > 0:
            print("📊 NOTE:")
            print(f"   - {count_lt_1} rows have excellent ERA (< 1.0)")
            print("   - These are rare but mathematically possible")
            print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

    print()
    print("=" * 70)
    print("END OF ERA INVESTIGATION REPORT")
    print("=" * 70)


if __name__ == "__main__":
    main()
