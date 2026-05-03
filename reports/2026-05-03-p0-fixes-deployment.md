# P0 Fixes Deployment Report — May 3, 2026

## Deployment Summary
- **Commit:** a5a22601ca16988ddbebed4504a1a041525cf341
- **Deployed at:** 2026-05-03T01:45:00-04:00
- **Status:** PARTIAL (Deploy successful, but runtime errors in new code)

## Fix Results

### Fix #1: player_type Backfill
- **Before:** 441 NULL rows (estimated)
- **After:** 0 NULL rows remaining
- **Status:** SUCCESS

### Fix #2: Yahoo ID Sync
- **Before:** 3.7% coverage
- **After:** Sync failed (ImportError)
- **Status:** FAILED
- **Error:** `ImportError: cannot import name 'try_advisory_lock' from 'backend.services.daily_ingestion'`

### Fix #3: Park Factors Bulk-Load
- **Before:** Waiver 27s, Dashboard 19s
- **After:** Load failed on startup (SQL Error)
- **Status:** FAILED
- **Error:** `(psycopg2.errors.UndefinedColumn) column "team" does not exist` (Table uses `park_name`)

## Verification Tests
- [x] player_type backfill works (Remaining NULLs: 0)
- [ ] Yahoo ID sync (BLOCKED by ImportError)
- [ ] Performance improved (BLOCKED by startup SQL error)

## Issues Found
1. **Yahoo ID Sync:** `yahoo_id_sync.py` attempts to import `try_advisory_lock` and `release_advisory_lock` from `backend.services.daily_ingestion`, but these functions are not defined in that module (only an internal `_with_advisory_lock` helper exists).
2. **Park Factors:** `load_park_factors()` in `ballpark_factors.py` queries for a `team` column in the `park_factors` table, but the column name in the model is `park_name`.

## Recommendations
- Escalate to Claude Code to fix the logic errors in `backend/fantasy_baseball/yahoo_id_sync.py` and `backend/fantasy_baseball/ballpark_factors.py`.
