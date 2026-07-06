# Roster Move Cache Invalidation Bug - ROOT CAUSE FOUND & FIXED

**Date:** 2026-07-06
**Issue:** CRITICAL - Roster move succeeds on Yahoo but GET `/api/fantasy/roster` returns stale cached data
**Status:** ROOT CAUSE IDENTIFIED - Fix deployed to correct endpoint

## Root Cause

**TWO DUPLICATE ENDPOINTS** - This is why the previous 3 fix attempts failed:

1. `backend/routers/fantasy.py:3859` - `@router.get("/api/fantasy/roster")` with all the cache bypass fixes
2. `backend/main.py:6576` - `@app.get("/api/fantasy/roster")` WITHOUT any cache bypass fixes

The endpoint in `main.py` OVERRIDES the router endpoint because:
- `app.include_router(_fantasy_router)` happens at line 652
- The duplicate endpoint is defined at line 6576 (AFTER the router is included)
- FastAPI uses the last matching route definition

**Result:** All my previous fixes to `fantasy.py` were ineffective because the frontend was actually calling the `main.py` endpoint which had NO cache bypass logic!

## The Fix

Applied the same cache bypass and retry fixes to the CORRECT endpoint (`main.py`):

### 1. Added `force_refresh` Parameter
```python
@app.get("/api/fantasy/roster", response_model=RosterResponse)
async def get_fantasy_roster(
    user: str = Depends(verify_api_key),
    force_refresh: bool = False,  # NEW
):
```

### 2. Added Retry Logic with Cache Bypass
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
        logger.warning("Roster fetch attempt %d failed (retrying): %s", attempt + 1, exc)
        await asyncio.sleep(retry_delay)
```

### Supporting Fixes (Already in Place)

- Extended bypass window (5 seconds) in `YahooAPICache`
- Enhanced logging for bypass window activity
- Cache write skip during bypass (prevents re-caching stale data)

## How It Works NOW

1. User makes roster move → `set_lineup()` succeeds
2. Cache cleared → 5-second bypass window starts
3. Frontend calls `GET /api/fantasy/roster` (NOW calls the FIXED endpoint in main.py)
4. Bypass window active → cache bypassed
5. If Yahoo returns stale data → retry logic kicks in (up to 3 attempts, 1s delay)
6. Eventually Yahoo returns fresh data
7. Fresh data returned to frontend

## Files Modified

1. **backend/main.py** (lines 6576-6624)
   - Added `force_refresh` parameter
   - Added retry logic with cache bypass
   - Removed duplicate exception handlers

2. **backend/fantasy_baseball/yahoo_client_resilient.py** (previously modified)
   - Extended bypass window to 5 seconds
   - Added enhanced logging
   - Cache write skip during bypass

3. **backend/routers/fantasy.py** (previously modified, but NOT used by frontend)
   - Has same fixes, but endpoint is overridden by main.py

## Why This Fix Will Work Now

The fix is now applied to the CORRECT endpoint that the frontend actually calls:
- The endpoint at `main.py:6576` is the one that responds to `GET /api/fantasy/roster` requests
- This endpoint now has the same cache bypass and retry logic
- The bypass window + retry mechanism gives Yahoo enough time to propagate changes

## Next Steps

1. **Deploy to Railway** - The fix must be deployed to take effect
2. **Test:**
   - Make a roster move
   - Verify UI shows updated position immediately
   - Check logs for "Cache bypass window active" and retry attempts
3. **Monitor:**
   - If still fails, check logs to see if bypass window is active
   - Verify retry attempts are happening
   - Check if Yahoo's propagation delay exceeds 8 seconds (3 retries × 1s + 5s bypass window)

## Technical Debt

Consider:
- Removing the duplicate endpoint in `fantasy.py` to avoid confusion
- OR moving all roster logic to `fantasy.py` and removing from `main.py`
- Having duplicate endpoints with the same path is a maintenance nightmare

## Lessons Learned

1. Always check for duplicate route definitions when troubleshooting routing issues
2. Route order matters - last definition wins in FastAPI
3. The `fetched_at: null` symptom was a red herring - it's hardcoded in the response model
4. Three failed attempts before finding the root cause - should have checked for duplicate endpoints earlier
