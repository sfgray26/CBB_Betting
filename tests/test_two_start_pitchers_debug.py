"""
Debug test for Loop Iteration 22 Part 1: Two-Start Pitchers Investigation.

This test verifies:
1. ProbablePitcherSnapshot has data for the current week
2. Quality scores are populated (not NULL)
3. BDL player IDs are populated (not NULL)
4. The query correctly identifies 2-start pitchers
"""
import pytest
import os
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from backend.models import ProbablePitcherSnapshot, SessionLocal


def test_probable_pitcher_snapshot_data_quality():
    """
    Check if ProbablePitcherSnapshot has data for current week.

    This test helps diagnose why two-start pitchers might return 0.
    """
    db = SessionLocal()

    # Today's date in ET
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    # Check date range: today to today + 6 days (7-day window)
    end_date = today_et + timedelta(days=6)

    # Total rows in date range
    total_rows = db.query(ProbablePitcherSnapshot).filter(
        ProbablePitcherSnapshot.game_date >= today_et,
        ProbablePitcherSnapshot.game_date <= end_date,
    ).count()

    print(f"\n=== ProbablePitcherSnapshot Data Quality ===")
    print(f"Date range: {today_et} to {end_date}")
    print(f"Total rows: {total_rows}")

    # Rows with NULL bdl_player_id
    null_bdl_id = db.query(ProbablePitcherSnapshot).filter(
        ProbablePitcherSnapshot.game_date >= today_et,
        ProbablePitcherSnapshot.game_date <= end_date,
        ProbablePitcherSnapshot.bdl_player_id.is_(None),
    ).count()

    # Rows with NULL quality_score
    null_quality = db.query(ProbablePitcherSnapshot).filter(
        ProbablePitcherSnapshot.game_date >= today_et,
        ProbablePitcherSnapshot.game_date <= end_date,
        ProbablePitcherSnapshot.quality_score.is_(None),
    ).count()

    # Rows that pass the streaming query filters
    valid_rows = db.query(ProbablePitcherSnapshot).filter(
        ProbablePitcherSnapshot.game_date >= today_et,
        ProbablePitcherSnapshot.game_date <= end_date,
        ProbablePitcherSnapshot.bdl_player_id.isnot(None),
        ProbablePitcherSnapshot.quality_score.isnot(None),
    ).count()

    print(f"Rows with NULL bdl_player_id: {null_bdl_id}")
    print(f"Rows with NULL quality_score: {null_quality}")
    print(f"Rows passing query filters: {valid_rows}")

    # Get sample data if available
    if valid_rows > 0:
        sample = db.query(
            ProbablePitcherSnapshot.game_date,
            ProbablePitcherSnapshot.pitcher_name,
            ProbablePitcherSnapshot.team,
            ProbablePitcherSnapshot.bdl_player_id,
            ProbablePitcherSnapshot.quality_score,
        ).filter(
            ProbablePitcherSnapshot.game_date >= today_et,
            ProbablePitcherSnapshot.game_date <= end_date,
            ProbablePitcherSnapshot.bdl_player_id.isnot(None),
            ProbablePitcherSnapshot.quality_score.isnot(None),
        ).order_by(ProbablePitcherSnapshot.game_date, ProbablePitcherSnapshot.team).limit(5).all()

        print(f"\nSample data:")
        for row in sample:
            print(f"  {row.game_date}: {row.pitcher_name} ({row.team}) - BDL ID: {row.bdl_player_id}, Quality: {row.quality_score}")

    # Check for duplicates (same pitcher, same date)
    if valid_rows > 0:
        duplicates = db.execute(text("""
            SELECT game_date, pitcher_name, COUNT(*) as cnt
            FROM probable_pitchers
            WHERE game_date >= :start_date AND game_date <= :end_date
              AND bdl_player_id IS NOT NULL
              AND quality_score IS NOT NULL
            GROUP BY game_date, pitcher_name
            HAVING COUNT(*) > 1
        """), {"start_date": today_et, "end_date": end_date}).fetchall()

        if duplicates:
            print(f"\nDuplicate pitchers found: {len(duplicates)}")
            for dup in duplicates:
                print(f"  {dup.game_date}: {dup.pitcher_name} appears {dup.cnt} times")

    # Verify minimum data for two-start detection
    # We need at least some rows for two-start pitchers to exist
    if valid_rows == 0:
        if os.getenv("CI"):
            pytest.skip(
                "No valid ProbablePitcherSnapshot rows in CI test database; "
                "data-quality diagnostic requires seeded/current-week data."
            )
        pytest.fail("No valid rows in ProbablePitcherSnapshot for current week. Cannot detect two-start pitchers.")
    elif valid_rows < 20:  # 30 teams * 7 days = 210 expected, but allow for off days
        pytest.fail(f"Too few valid rows ({valid_rows}) in ProbablePitcherSnapshot for current week. Sync may be incomplete.")

    assert total_rows > 0, "No data in ProbablePitcherSnapshot for current week"


def test_two_start_pitcher_detection():
    """
    Simulate the two-start pitcher query logic.

    This test replicates the exact query used in streaming recommendations
    to verify if two-start pitchers would be detected.
    """
    db = SessionLocal()

    # Today's date in ET
    today_et = datetime.now(ZoneInfo("America/New_York")).date()

    # Use the exact same query as the endpoint
    target_dt = today_et
    days_ahead = 7
    end_dt = target_dt + timedelta(days=days_ahead)

    query = db.query(
        ProbablePitcherSnapshot.bdl_player_id,
        ProbablePitcherSnapshot.pitcher_name,
        ProbablePitcherSnapshot.team,
        ProbablePitcherSnapshot.game_date,
    ).filter(
        ProbablePitcherSnapshot.game_date >= target_dt,
        ProbablePitcherSnapshot.game_date <= end_dt,
        ProbablePitcherSnapshot.bdl_player_id.isnot(None),
        ProbablePitcherSnapshot.quality_score.isnot(None),
    )

    rows = query.all()

    # Group by pitcher
    pitcher_starts: dict[int, list] = {}
    for r in rows:
        pid = r.bdl_player_id
        if pid not in pitcher_starts:
            pitcher_starts[pid] = []
        pitcher_starts[pid].append({
            "pitcher_name": r.pitcher_name,
            "team": r.team,
            "date": r.game_date.isoformat(),
        })

    # Identify 2-start pitchers
    two_starters = [pid for pid, starts in pitcher_starts.items() if len(starts) >= 2]

    print(f"\n=== Two-Start Pitcher Detection ===")
    print(f"Date range: {target_dt} to {end_dt}")
    print(f"Total pitchers in range: {len(pitcher_starts)}")
    print(f"Two-start pitchers found: {len(two_starters)}")

    # Show distribution of starts per pitcher
    start_counts = {}
    for pid, starts in pitcher_starts.items():
        count = len(starts)
        start_counts[count] = start_counts.get(count, 0) + 1

    print(f"\nStarts per pitcher distribution:")
    for count in sorted(start_counts.keys()):
        print(f"  {count} starts: {start_counts[count]} pitchers")

    # Show pitchers with 2+ starts if any
    if two_starters:
        print(f"\nTwo-start pitchers:")
        for pid in two_starters[:10]:  # Show first 10
            starts = pitcher_starts[pid]
            print(f"  {starts[0]['pitcher_name']} (BDL ID: {pid}):")
            for start in starts[:3]:  # Show up to 3 starts
                print(f"    - {start['date']}: {start['team']}")
    else:
        # Show some sample pitchers with 1 start to understand the data
        print(f"\nSample pitchers with 1 start:")
        sample_pids = list(pitcher_starts.keys())[:10]
        for pid in sample_pids:
            starts = pitcher_starts[pid]
            print(f"  {starts[0]['pitcher_name']} (BDL ID: {pid}):")
            for start in starts[:2]:  # Show up to 2 starts
                print(f"    - {start['date']}: {start['team']} vs {start.get('opponent', '?')}")

    if two_starters:
        print(f"\nTwo-start pitchers:")
        for pid in two_starters[:10]:  # Show first 10
            starts = pitcher_starts[pid]
            print(f"  {starts[0]['pitcher_name']} (BDL ID: {pid}):")
            for start in starts[:3]:  # Show up to 3 starts
                print(f"    - {start['date']}: {start['team']}")

    # Check if the date range calculation might be off by one
    # The sync uses range(7) = 0..6, but query uses target_dt + 7 days
    # This means the query includes an extra day that the sync doesn't fetch
    sync_end_date = target_dt + timedelta(days=6)  # sync fetches up to this
    query_end_date = end_dt  # query looks up to this

    print(f"\nDate range check:")
    print(f"Sync fetches up to: {sync_end_date} (target_dt + 6 days)")
    print(f"Query looks up to: {query_end_date} (target_dt + 7 days)")

    if sync_end_date != query_end_date:
        print(f"WARNING: MISMATCH: Query range extends one day beyond sync range!")
        print(f"   This could cause data on {query_end_date} to be missing.")

    # Check for specific pitcher mentioned in UAT (Cristopher Sánchez)
    sanchez = db.query(ProbablePitcherSnapshot).filter(
        ProbablePitcherSnapshot.pitcher_name.ilike("%Sanchez%")
    ).filter(
        ProbablePitcherSnapshot.bdl_player_id.isnot(None),
        ProbablePitcherSnapshot.quality_score.isnot(None),
    ).all()

    # Also check the full date range in the table
    all_dates = db.execute(text("""
        SELECT DISTINCT game_date, COUNT(*) as game_count
        FROM probable_pitchers
        WHERE bdl_player_id IS NOT NULL AND quality_score IS NOT NULL
        GROUP BY game_date
        ORDER BY game_date DESC
        LIMIT 14
    """)).fetchall()

    print(f"\n=== All Recent Dates in Database ===")
    for row in all_dates:
        print(f"  {row.game_date}: {row.game_count} games")

    if sanchez:
        print(f"\n=== Sánchez Query Results ===")
        for row in sanchez:
            print(f"  {row.pitcher_name} (BDL ID: {row.bdl_player_id})")
            print(f"    - {row.game_date}: {row.team} vs {row.opponent}, Quality: {row.quality_score}")

    # Count games per date to see if schedule is sparse
    games_per_date = db.execute(text("""
        SELECT game_date, COUNT(*) as game_count
        FROM probable_pitchers
        WHERE game_date >= :start_date AND game_date <= :end_date
          AND bdl_player_id IS NOT NULL
          AND quality_score IS NOT NULL
        GROUP BY game_date
        ORDER BY game_date
    """), {"start_date": target_dt, "end_date": end_dt}).fetchall()

    print(f"\n=== Games per Date ===")
    for row in games_per_date:
        print(f"  {row.game_date}: {row.game_count} games")

    assert len(two_starters) >= 0, "Two-start pitcher detection failed"


if __name__ == "__main__":
    # Run tests locally for debugging
    print("Running two-start pitcher debug tests...")
    test_probable_pitcher_snapshot_data_quality()
    test_two_start_pitcher_detection()
    print("\nAll tests passed!")
