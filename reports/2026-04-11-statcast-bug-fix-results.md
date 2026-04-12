# Statcast Persistence Bug Fix Results

**Date:** 2026-04-11
**Bug:** FOLLOW-UP 1 from production deployment — statcast_performances table remains empty despite successful pybaseball fetches
**Root Cause:** Baseball Savant CSV schema mismatch (no player_id column)

## Before Fix
- pybaseball fetch: ✅ Working (10,562 pitcher + 6,079 batter rows per date)
- transform_to_performance(): ❌ Returns empty list (all rows skipped)
- statcast_performances table: 0 rows

## After Fix
- Added PlayerIdResolver: name→mlbam_id cache from player_id_mapping
- transform_to_performance(): Handles missing player_id column, uses player_name fallback
- backfill_statcast.py: Error logging instead of silent except...continue
- statcast_performances table: ~15,000 rows expected (pending Railway deployment)

## Files Changed
- `backend/fantasy_baseball/statcast_ingestion.py` — PlayerIdResolver class, transform logic
- `scripts/backfill_statcast.py` — error logging
- `tests/test_statcast_ingestion.py` — new test file (3 tests, all passing)
- `backend/admin_endpoints_validation.py` — updated Statcast check

## Tests Added
- test_transform_to_performance_handles_missing_player_id_column
- test_transform_to_performance_skips_rows_with_missing_player_name
- test_transform_to_performance_with_pitcher_rows

## Lessons Learned
1. CSV schema assumptions must be validated against real data before writing transforms
2. Silent error swallowing (except...continue) hides bugs for weeks
3. Always add diagnostic logging showing actual column names on first run
4. Baseball Savant CSV with group_by='name-date' does NOT include player_id — use player_name instead