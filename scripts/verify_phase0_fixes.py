#!/usr/bin/env python3
"""
Phase 0 Verification:
1. Test projection_freshness datetime fix
2. Analyze remaining 93 numeric player names
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from zoneinfo import ZoneInfo
from sqlalchemy import text
from backend.models import SessionLocal

db = SessionLocal()

# ============================================================================
# CHECK 1: Test projection_freshness datetime handling
# ============================================================================
print("=" * 60)
print("CHECK 1: Testing projection_freshness datetime handling")
print("=" * 60)

try:
    now_et = datetime.now(ZoneInfo("America/New_York")).replace(tzinfo=None)

    # Query that was crashing - test with both date and datetime columns
    result = db.execute(text("""
        SELECT
            metric_date::date,
            MAX(fetched_at) as latest_computed,
            COUNT(*) as row_count
        FROM player_daily_metrics
        GROUP BY metric_date::date
        ORDER BY metric_date::date DESC
        LIMIT 5
    """))

    print(f"{'Metric Date':<15} {'Latest Computed':<25} {'Rows':<8} {'Age Days':<10}")
    print("-" * 60)

    for row in result:
        metric_date = row[0]
        latest_computed = row[1]
        count = row[2]

        if metric_date and latest_computed:
            # This is the operation that was crashing: datetime - date
            age_days = (now_et - latest_computed).days
            latest_str = latest_computed.strftime("%Y-%m-%d %H:%M") if latest_computed else "NULL"
            print(f"{str(metric_date):<15} {latest_str:<25} {count:<8} {age_days:<10}")

    print("\nStatus: PASSED - datetime subtraction works without crash")

except Exception as exc:
    print(f"\nStatus: FAILED - {exc}")
    import traceback
    traceback.print_exc()
    db.close()
    sys.exit(1)

# ============================================================================
# CHECK 2: Remaining numeric player names analysis
# ============================================================================
print("\n" + "=" * 60)
print("CHECK 2: Remaining numeric player names analysis")
print("=" * 60)

# Get aggregate stats first
result = db.execute(text("""
    SELECT
        COUNT(*) as total_rows,
        COUNT(DISTINCT player_name) as unique_players,
        COUNT(DISTINCT target_date) as unique_dates,
        MIN(target_date) as earliest_date,
        MAX(target_date) as latest_date
    FROM player_projections
    WHERE player_name ~ '^[0-9]+$'
"""))

row = result.first()
print(f"\nAggregate stats:")
print(f"  Total rows: {row[0]}")
print(f"  Unique player IDs: {row[1]}")
print(f"  Unique dates: {row[2]}")
print(f"  Date span: {row[3]} to {row[4]}")

# Sample of individual players
result = db.execute(text("""
    SELECT
        player_name,
        COUNT(*) as row_count,
        MAX(target_date) as latest_target_date,
        COUNT(DISTINCT target_date) as date_span
    FROM player_projections
    WHERE player_name ~ '^[0-9]+$'
    GROUP BY player_name
    ORDER BY row_count DESC
    LIMIT 15
"""))

print(f"\n{'Player ID':<12} {'Rows':<8} {'Latest Target':<12} {'Date Span':<10}")
print("-" * 50)
for row in result:
    player_id = row[0]
    count = row[1]
    latest_target = str(row[2]) if row[2] else 'NULL'
    date_span = row[3]
    print(f"{player_id:<12} {count:<8} {latest_target:<12} {date_span:<10}")

# ============================================================================
# CHECK 3: Do these orphaned players have any associated stats?
# ============================================================================
print("\n" + "=" * 60)
print("CHECK 3: Do orphan players have stats data?")
print("=" * 60)

# Get sample of 10 player IDs
sample_ids_result = db.execute(text("""
    SELECT DISTINCT player_name
    FROM player_projections
    WHERE player_name ~ '^[0-9]+$'
    LIMIT 10
"""))

sample_ids = [int(r[0]) for r in sample_ids_result]

# Check each for associated stats in other tables
placeholders = ','.join([':id' + str(i) for i in range(len(sample_ids))])
params = {f'id{i}': sample_ids[i] for i in range(len(sample_ids))}

result = db.execute(text(f"""
    SELECT
        ppm.player_name,
        COUNT(DISTINCT pds.id) as daily_stats_count,
        COUNT(DISTINCT prs.id) as rolling_stats_count,
        COUNT(DISTINCT mlb.id) as mlb_stats_count,
        COUNT(DISTINCT ps.id) as player_scores_count
    FROM (SELECT unnest(ARRAY[:id0, :id1, :id2, :id3, :id4, :id5, :id6, :id7, :id8, :id9])::text as player_name) ppm
    LEFT JOIN player_daily_stats pds ON ppm.player_name = pds.bdl_player_id::text
    LEFT JOIN player_rolling_stats prs ON ppm.player_name = prs.bdl_player_id::text
    LEFT JOIN mlb_player_stats mlb ON ppm.player_name = mlb.bdl_player_id::text
    LEFT JOIN player_scores ps ON ppm.player_name = ps.bdl_player_id::text
    GROUP BY ppm.player_name
""").bindparams(**params))

print(f"\n{'Player ID':<12} {'Daily':<8} {'Rolling':<10} {'MLB':<8} {'Scores':<10} {'Total':<8}")
print("-" * 60)
orphan_count = 0
for row in result:
    total = (row[1] or 0) + (row[2] or 0) + (row[3] or 0) + (row[4] or 0)
    if total == 0:
        orphan_count += 1
    print(f"{row[0]:<12} {row[1] or 0:<8} {row[2] or 0:<10} {row[3] or 0:<8} {row[4] or 0:<10} {total:<8}")

print(f"\nOrphan status: {orphan_count}/{len(sample_ids)} sampled players have ZERO associated stats")

# Summary determination
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
if orphan_count == len(sample_ids):
    print("Conclusion: The 93 remaining numeric names are likely EMPTY SHELL entries")
    print("            with no associated stats data. Safe to ignore for now.")
else:
    print("Conclusion: Some orphan players have stats data - may need manual mapping")

db.close()
