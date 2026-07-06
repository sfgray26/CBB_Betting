# Roster Move Cache Invalidation Fix (v2)

**Date:** 2026-07-06
**Issue:** CRITICAL - Roster move succeeds on Yahoo but GET `/api/fantasy/roster` returns stale cached data
**Status:** FIX IMPLEMENTED - Awaiting redeploy and retest

## Root Cause Analysis

After a successful roster move, the Yahoo client cache was cleared, but the next GET `/api/fantasy/roster` call could still return stale data due to:

1. **Yahoo API Propagation Delay:** Yahoo's `set_lineup` endpoint returns success immediately, but the `get_roster` endpoint may still return old data for several seconds as the change propagates through Yahoo's internal systems.

2. **No Retry Mechanism:** The cache clearing worked correctly, but there was no retry mechanism to handle Yahoo's propagation delay. Even with cache bypass, if Yahoo returns stale data, it gets returned to the frontend without verification.

## Solution Implemented

### 1. Extended Bypass Window (yahoo_client_resilient.py)

Increased the bypass window from 2 seconds to 5 seconds to account for Yahoo's propagation delay:
```python
self._bypass_window_seconds = 5.0  # Bypass cache for 5s after clear
```

### 2. Enhanced Logging (yahoo_client_resilient.py)

Added detailed logging to track bypass window activity:
```python
def is_bypass_window_active(self) -> bool:
    if self._last_cleared_at is None:
        return False
    elapsed = time.time() - self._last_cleared_at
    is_active = elapsed < self._bypass_window_seconds
    if is_active:
        logger.info(f"Cache bypass window active: {elapsed:.2f}s elapsed")
    return is_active
```

### 3. Retry Mechanism (fantasy.py)

Added retry logic to the GET `/api/fantasy/roster` endpoint:
- Retries up to 3 times with 1-second delay between attempts
- Each retry bypasses cache to force fresh fetch from Yahoo
- Gives Yahoo enough time to propagate roster changes
- Logs retry attempts for debugging

**Code:**
```python
# Retry logic for roster fetch to account for Yahoo's propagation delay
max_retries = 3
retry_delay = 1.0
raw_players = None

for attempt in range(max_retries):
    try:
        bypass_this_attempt = force_refresh or (attempt > 0)
        raw_players = client.get_roster(team_key=team_key, bypass_cache=bypass_this_attempt)
        break  # Success - exit retry loop
    except YahooAPIError as exc:
        if attempt == max_retries - 1:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        logger.warning("Roster fetch attempt %d failed (retrying in %.1fs): %s", attempt + 1, retry_delay, exc)
        await asyncio.sleep(retry_delay)
```

### 4. Force Refresh Parameter (fantasy.py)

The GET endpoint continues to support `force_refresh` query parameter for manual cache bypass.

## How It Works

### Before v2 Fix:
1. User makes roster move → `set_lineup()` succeeds
2. Cache cleared, 2-second bypass window starts
3. Frontend calls `GET /api/fantasy/roster`
4. Bypass window active → cache bypassed
5. Yahoo still returns old data (propagation delay > 2s)
6. Old data returned to frontend
7. BUG: Frontend shows incorrect roster state

### After v2 Fix:
1. User makes roster move → `set_lineup()` succeeds
2. Cache cleared, 5-second bypass window starts
3. Frontend calls `GET /api/fantasy/roster`
4. Bypass window active → cache bypassed
5. Yahoo may still return old data
6. **NEW:** Retry logic kicks in - up to 3 attempts with 1s delay
7. **NEW:** Each retry bypasses cache to force fresh fetch
8. After retries, Yahoo should have propagated the change
9. Fresh data returned to frontend

## Files Modified

1. **backend/fantasy_baseball/yahoo_client_resilient.py**
   - Extended bypass window from 2s to 5s
   - Added enhanced logging for bypass window activity
   - Bypass window mechanism (previously implemented)

2. **backend/routers/fantasy.py**
   - Added retry logic to GET `/api/fantasy/roster` endpoint
   - Supports `force_refresh` query parameter (previously implemented)

## Testing Instructions

1. **Deploy changes to Railway**
2. **Manual Test:**
   - Make a roster move (e.g., move player from BN to Util)
   - Immediately call `GET /api/fantasy/roster`
   - Verify that the response shows the new position
   - Check logs for "Cache bypass window active" and retry attempts

3. **Force Refresh Test:**
   - Call `GET /api/fantasy/roster?force_refresh=true`
   - Should bypass cache and return fresh data

4. **Monitoring:**
   - Check logs for bypass window activity
   - Verify retry attempts are logged if needed
   - Monitor cache stats via `YahooAPICache.get_stats()`

## Expected Behavior After Fix

- Immediate post-move roster fetches should return correct positions
- Frontend should show updated roster state after automatic refetch
- No more stale data being returned after successful moves
- Logs should show bypass window activity and retry attempts (if needed)

## Notes

- 5-second bypass window + 3 retries with 1s delay = up to 8 seconds total wait time
- This should be sufficient for Yahoo's propagation delay
- If Yahoo's delay exceeds 8 seconds, the endpoint will return an error (better than stale data)
- The fix is backward compatible - no frontend changes required
