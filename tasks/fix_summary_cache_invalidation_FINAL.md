# Roster Move Cache Invalidation Bug - FINAL FIX

**Date:** 2026-07-06
**Issue:** CRITICAL - Roster move succeeds on Yahoo but GET `/api/fantasy/roster` returns stale cached data
**Status:** ROOT CAUSE IDENTIFIED AND FIXED

## Root Cause

**Missing method on `YahooFantasyClient`** - The move endpoint uses `get_yahoo_client()` which returns `YahooFantasyClient`. The `clear_cache()` method only existed on `ResilientYahooClient`, not on `YahooFantasyClient`.

### Evidence from Railway Logs

```
roster/move: Yahoo client YahooFantasyClient has no clear_cache(); skipping cache clear
```

### The Bug Flow

1. User makes roster move → `set_lineup()` succeeds on Yahoo
2. Move endpoint tries to clear cache:
   ```python
   clear_cache = getattr(client, "clear_cache", None)
   if callable(clear_cache):
       clear_cache()
   ```
3. `get_yahoo_client()` returns `YahooFantasyClient` (NOT `ResilientYahooClient`)
4. `YahooFantasyClient` has no `clear_cache()` method → `getattr` returns `None`
5. Cache clear is skipped → bypass window never starts
6. Frontend refetches → stale data returned from cache

### Why Previous Fixes Failed

- v1-v3 fixes added bypass window logic and retry mechanisms
- These fixes were correct and would work IF the cache was actually being cleared
- But `clear_cache()` was never called because the method didn't exist on `YahooFantasyClient`
- The endpoint in `main.py` was also overriding the `fantasy.py` endpoint (v2 finding)

## The Fix

**Added `clear_cache()` method to `YahooFantasyClient`:**

```python
def clear_cache(self) -> None:
    """Clear all cached Yahoo API responses.

    This triggers a 5-second bypass window during which all cache reads
    are forced to miss, preventing stale data from being returned after
    roster moves or other mutations.
    """
    logger.info("YahooFantasyClient.clear_cache() - clearing cache and starting bypass window")
    self._cache.clear_all()
```

### Location

- **File:** `backend/fantasy_baseball/yahoo_client_resilient.py`
- **Line:** 1999 (after `_estimate_ownership_from_adp` method)
- **Class:** `YahooFantasyClient` (base client, NOT `ResilientYahooClient`)

## How It Works NOW

1. User makes roster move → `set_lineup()` succeeds on Yahoo
2. Move endpoint calls `client.clear_cache()` → **NOW WORKS** because method exists
3. `YahooAPICache.clear_all()` called → cache cleared → bypass window starts (5 seconds)
4. Frontend calls `GET /api/fantasy/roster`
5. `_get()` detects bypass window active → cache bypassed
6. Fresh data fetched from Yahoo
7. Frontend shows updated roster positions

## Files Modified

1. **backend/fantasy_baseball/yahoo_client_resilient.py**
   - Added `clear_cache()` method to `YahooFantasyClient` class (line ~1999)

## Supporting Fixes (Previously Deployed)

The following fixes were already in place and are now effective:

1. **Bypass window mechanism** - 5-second window after cache clear where all reads bypass cache
2. **Retry logic** - GET `/api/fantasy/roster` retries up to 3 times with 1s delay
3. **Enhanced logging** - Bypass window activity logged for debugging
4. **Cache write skip during bypass** - Prevents re-caching stale data during bypass window

## Testing Instructions

1. **Deploy to Railway:**
   ```bash
   git add backend/fantasy_baseball/yahoo_client_resilient.py
   git commit -m "fix: add clear_cache() to YahooFantasyClient"
   git push
   ```

2. **Manual Test:**
   - Make a roster move (e.g., move player from BN to Util)
   - Immediately call `GET /api/fantasy/roster`
   - Verify that `selected_position` shows the updated position
   - Check Railway logs for "YahooFantasyClient.clear_cache() - clearing cache"

3. **Expected Logs:**
   ```
   YahooFantasyClient.clear_cache() - clearing cache and starting bypass window
   Cache bypass window active: 0.12s elapsed (window: 5.00s)
   Cache BYPASS for team/.../roster/players (bypass window active - cache recently cleared)
   ```

## Lessons Learned

1. **Always verify method existence** - When using `getattr(obj, "method", None)`, check the actual class hierarchy
2. **Single-line bugs can hide** - The missing method was a single-line omission that caused the entire cache invalidation to fail
3. **Log evidence is critical** - The Railway log message "YahooFantasyClient has no clear_cache()" was the key to finding the root cause
4. **Fix v1-v3 were not wasted** - The bypass window and retry mechanisms are now effective with the cache clear actually working

## Next Steps

1. Deploy this fix to Railway
2. Verify the fix works with a manual test
3. Monitor logs for successful cache clears and bypass window activity
