# Roster Move Cache Invalidation Fix

**Date:** 2026-07-06
**Issue:** CRITICAL - Roster move succeeds on Yahoo but GET `/api/fantasy/roster` returns stale cached data

## Root Cause Analysis

After a successful roster move, the Yahoo client cache was cleared, but the next GET `/api/fantasy/roster` call could still return stale data due to two issues:

1. **Yahoo API Propagation Delay:** Yahoo's `set_lineup` endpoint returns success immediately, but the `get_roster` endpoint may still return old data for a short period (typically <2 seconds) as the change propagates through Yahoo's internal systems.

2. **No Bypass Mechanism:** The cache clearing worked correctly, but there was no mechanism to ensure the next read would bypass the cache without re-caching Yahoo's transient stale response during the propagation window.

## Solution Implemented

### 1. Bypass Window Mechanism (yahoo_client_resilient.py)

Added a "bypass window" to the `YahooAPICache` class:
- When `clear_cache()` is called, a `_last_cleared_at` timestamp is recorded
- For 2 seconds after clearing, all cache reads are automatically bypassed
- This ensures post-move reads do not use or re-seed stale cache while Yahoo is propagating the change

**Changes:**
- `YahooAPICache.__init__`: Added `_last_cleared_at` and `_bypass_window_seconds = 2.0`
- `YahooAPICache.clear()`: Records timestamp when cache is cleared
- `YahooAPICache.is_bypass_window_active()`: Checks if bypass window is active
- `YahooAPICache.get()`: Added `bypass_window_active` parameter
- `YahooAPICache.get_stats()`: Returns bypass window status for monitoring

### 2. Force Refresh Parameter (yahoo_client_resilient.py + fantasy.py)

Added explicit cache bypass capability:
- `_get()` method now supports `bypass_cache` parameter
- `get_roster()` method now supports `bypass_cache` parameter
- GET `/api/fantasy/roster` endpoint now supports `force_refresh` query parameter

**Usage:**
```bash
# Force fresh fetch from Yahoo (bypasses cache)
GET /api/fantasy/roster?force_refresh=true
```

### 3. Automatic Bypass in _get() (yahoo_client_resilient.py)

The `_get()` method now automatically checks if the bypass window is active and bypasses the cache accordingly:
```python
# Check if bypass window is active (cache was recently cleared)
bypass_window_active = self._cache.is_bypass_window_active()
should_bypass_cache = bypass_cache or bypass_window_active
```

Bypassed reads also skip cache writes. This prevents a transient stale Yahoo roster
response from being cached again during the propagation window.

## How It Works

### Before the Fix:
1. User makes roster move → `set_lineup()` succeeds
2. Cache is cleared via `clear_cache()`
3. Frontend immediately calls `GET /api/fantasy/roster`
4. Cache miss, but Yahoo API still returns old data (propagation delay)
5. Old data gets cached again for 5 minutes
6. Frontend shows incorrect roster state

### After the Fix:
1. User makes roster move → `set_lineup()` succeeds
2. Cache is cleared, bypass window starts (2 seconds)
3. Frontend calls `GET /api/fantasy/roster` (with or without `force_refresh`)
4. Bypass window active → cache read is bypassed
5. Yahoo response is returned but not cached while bypass is active
6. After 2 seconds, normal caching resumes and the next Yahoo roster read can seed cache

## Files Modified

1. **backend/fantasy_baseball/yahoo_client_resilient.py**
   - Modified `YahooAPICache` class with bypass window mechanism
   - Modified `_get()` method to check bypass window
   - Modified `_get()` method to skip cache writes during bypass reads
   - Modified `get_roster()` to support `bypass_cache` parameter

2. **backend/routers/fantasy.py**
   - Added `force_refresh` query parameter to `GET /api/fantasy/roster`

## Testing Recommendations

1. **Manual Test:** Make a roster move and immediately call `GET /api/fantasy/roster`. Should return updated positions.
2. **Force Refresh Test:** Call `GET /api/fantasy/roster?force_refresh=true` to bypass cache manually.
3. **Monitoring Test:** Check cache stats via `YahooAPICache.get_stats()` to verify bypass window state.

## Future Enhancements

- Consider making bypass window duration configurable via environment variable
- Add metrics/monitoring for bypass window activations
- Consider adding retry logic if Yahoo API returns stale data after move

## Notes

- The 2-second bypass window is a conservative estimate for Yahoo's propagation delay
- Frontend can optionally use `force_refresh=true` for immediate post-move fetches
- The fix is backward compatible - existing calls work without changes
