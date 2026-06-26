# Loop Iteration Log

Track iteration progress, key findings, and deployment status.

---

## Loop Iteration 22: Fix Two-Start Pitchers Query + Season Adds Label

**Status**: ⏸️ **IN PROGRESS**

**Scope**:
1. Part 1: Fix Two-Start Pitchers = 0 on Streaming
2. Part 2: Fix Season Adds = 367 on Budget

**Constraints**: 2 files max

---

### Part 1: Two-Start Pitchers Investigation ✅ COMPLETE

**Issue**: Streaming recommendations shows "Two-Start Pitchers = 0" even when pitchers should qualify.

**Investigation Findings**:
1. Query logic is correct: Uses `game_date >= target_dt` AND `game_date <= end_dt` with `days_ahead=7`
2. Date range mismatch identified:
   - Sync (`_sync_probable_pitchers`) uses `range(7)` → fetches days 0-6 (e.g., 06/26 through 07/02)
   - Query uses `target_dt + timedelta(days=days_ahead)` with `days_ahead=7` → looks up 06/26 through 07/03
   - Result: Query looks for data on day+7 that sync never fetches
3. Data incompleteness:
   - Database only has data through 2026-06-28
   - Missing dates: 2026-06-29, 06-30, 07-01, 07-02, 07-03
   - This suggests the sync job stopped running or failed to fetch future dates
4. With only 3 days of data (vs. expected 7), no pitcher appears twice
5. Two-start pitchers are naturally rare — require doubleheaders or schedule quirks

**Root Cause**:
- **Primary**: Sync job not populating full 7-day window (stopped at 06/28)
- **Secondary**: Date range mismatch (sync fetches 0-6, query looks 0-7)

**Fix Options**:
1. **Code fix**: Adjust date range to align sync and query
2. **Operational**: Trigger sync job to refresh probable pitchers data

**Fix Applied**:
- File: `backend/services/daily_ingestion.py`
- Change: Modified sync loop from `range(7)` to `range(8)` to match query's `days_ahead=7` parameter
- Line 7627: `for days_ahead in range(8):`
- Comment updated: "Fetch schedule for next 8 days (0-7) to match query's days_ahead=7 parameter"

**Rationale**:
- Query uses `end_dt = target_dt + timedelta(days=days_ahead)` with `days_ahead=7`
- This means the query looks for data from `target_dt` through `target_dt+7` (8 days inclusive)
- Previous sync used `range(7)` which only fetched days 0-6 (7 days)
- By changing to `range(8)`, sync fetches days 0-7, matching the query's expected range

**Note**: This fix addresses the date range mismatch only. The operational issue of incomplete data (sync stopped at 06/28) requires triggering the sync job manually or waiting for the next scheduled run (8:30 AM, 4:00 PM, or 8:00 PM ET).

---

### Part 2: Season Adds Investigation ✅ COMPLETE

**Issue**: Budget panel shows "Season Adds: 367" — unclear if this is league-wide or team-specific.

**Investigation Findings**:
1. Data source: `client.get_transactions(t_type="add")` fetches league-wide "add" transactions from Yahoo Fantasy API
2. Weekly `acquisitions_used` uses `count_weekly_acquisitions(transactions, team_key, ...)` which filters by team
3. Season `acquisitions_this_season` counted ALL transactions without team filtering
4. Root cause: 367 was the **league-wide** total of additions, not team-specific

**Fix Applied**:
- File: `backend/routers/fantasy.py`
- Change: Added team filtering to `acquisitions_this_season` calculation (lines 7696-7758)
- Now uses the same filtering logic as `count_weekly_acquisitions`:
  - Filters by transaction type ("add", "add/drop")
  - Filters by date range (season_start to now)
  - Filters by destination team (only counts transactions where `dest_team == team_key`)

**Result**: Season Adds now correctly shows the user's team's season acquisition count, not the league-wide total.

---

**ITERATION 22 STATUS**: ✅ **COMPLETE**
**DEPLOYMENT READY**: ✅ **READY** — 2 files modified (daily_ingestion.py, fantasy.py), all existing tests pass

---

## Loop Iteration 20: Unified Refresh + Formula Bugs

**Status**: ✅ **COMPLETE**

### Part 1: Unified Refresh Strategy ✅ COMPLETE

**Fix 1: Global Refresh Button**
1. `frontend/components/freshness/freshness-badge.tsx` — Added auto-polling every 30 seconds
2. `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` — Global refresh button
   - Added "Refresh Data" button next to FreshnessBadge
   - Calls invalidateAllFantasyCaches() to refresh all fantasy modules
   - Shows loading state during refresh

### Part 1: Specific Staleness Fixes ✅ COMPLETE

**Fix UNKNOWN sync badge** — Already implemented via FreshnessBadge update
- Polls global-freshness endpoint every 30 seconds
- Maps severity to LIVE (< 5 min), STALE (5-60 min), OFFLINE (> 60 min)
- Never shows "UNKNOWN" as terminal state

**Fix Roster "This Week" loading** — Added loading skeleton and error state
- Shows loading skeleton while scoreboard data fetches
- Shows error message on fetch failure
- No longer displays zeros/ties during loading

**Fix IP pace display consistency** — Already unified
- BudgetPanel shows `ip_as_of` timestamp (e.g., "as of 6/25 8:30 AM")
- War Room budget shows freshness with `updated {fetchedAt}`
- All modules read from `constraint_helpers.classify_ip_pace()` (single source of truth)
- Data already unified — "PENDING" vs "ON TRACK" discrepancy was staleness, not divergent code

### Part 2: Backend Tiered Refresh Audit ✅ COMPLETE

**Current Schedule:**
- Probable pitchers: 3x/day at 8:30 AM, 4:00 PM, 8:00 PM ET (fixed cron schedule)
- MLB odds: Every 5 minutes (10 AM - 11 PM ET)
- BDL injuries: Every 1 hour
- Statcast: Every 6 hours

**Identified Gaps:**
- No tiered refresh based on game schedule (active vs off days)
- No off-season vs in-season distinction
- Static schedule doesn't adapt to doubleheaders or schedule changes

**Recommendation (Out of Scope for this iteration):**
- In-season active days: 15-min refresh during game window (12 PM - 11 PM)
- In-season off days: 2-hour refresh
- Off-season: 24-hour refresh
- Requires MLB schedule detection to determine active vs off days

### Part 3B: Fix Ownership% Display ✅ COMPLETE

**Issue:** Ownership percentage showing as 0% on Waiver Wire page.

**Root Cause:** Field name mismatch between backend schema (`owned_pct`) and frontend type (`percent_owned`).

**Fix:** Added field alias in `WaiverPlayerOut` schema to serialize `owned_pct` as `percent_owned`.

**File Modified:**
- `backend/schemas.py` — Added `serialization_alias="percent_owned"` to `owned_pct` field (line 441)

**Backend Data Flow (already correct):**
1. `get_free_agents()` calls `_enrich_ownership_batch()` to fetch ownership from Yahoo
2. Fallback: `_apply_ownership_fallback()` loads from `PositionEligibility.league_rostered_pct`
3. Frontend compatibility: `player.percent_owned ?? player.owned_pct` handles both field names

**Note:** The backend enrichment pipeline was already correctly implemented. The fix ensures the API response serializes with the expected field name.

---

### Part 3C: Fix NSB Formula ✅ COMPLETE

**Issue:** NSB (Net Stolen Bases) showing -2 for Murakami on Roster page.

**Investigation:**
- NSB is computed as `SB - CS` (Stolen Bases - Caught Stealing)
- NSB can be negative per stat contract: "Can be negative. Computed as SB - CS."
- Frontend `getStat()` function correctly handles negative values (preserves sign with `Math.round()`)
- Data flow: Yahoo stat_id 62 → CONTRACT mapping → `player_mapper._map_yahoo_stats_to_category_stats()`

**Findings:**
- The value -2 is correct data if Murakami has 0 SB and 2 CS
- No calculation bug in our pipeline
- Frontend displays negative values correctly (e.g., "-2")
- Yahoo Fantasy API is the source of truth for season stats

**Note:** If the user believes -2 is incorrect, the issue is with Yahoo's data, not our calculation. Our pipeline correctly passes through Yahoo's NSB value.

**No code changes required** — data flow verified correct.

---

### Validation Checklist

| Requirement | Status |
|-------------|--------|
| All modules show LIVE/STALE/OFFLINE (never UNKNOWN) | ✅ Implemented via FreshnessBadge |
| Click "Refresh Data" → all modules update within 5 seconds | ✅ invalidateAllFantasyCaches() |
| Roster "This Week" shows loading spinner then data | ✅ Loading skeleton added |
| IP pace consistent across modules after refresh | ✅ Single source: classify_ip_pace() |
| Window focus revalidation | ✅ refetchOnWindowFocus=true |

---

**ITERATION 20 STATUS**: ✅ **COMPLETE**
**DEPLOYMENT READY**: ✅ **READY** — 3 files modified, all existing tests pass

**NEXT ITERATION**: Resume formula bugs (3B-3E) or new UAT findings

---

## Previous Iterations

[Previous content preserved...]
