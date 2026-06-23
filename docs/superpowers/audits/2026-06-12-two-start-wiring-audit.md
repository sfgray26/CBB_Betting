# Task 1.3 Wiring Audit: Two-Start Pitcher Identifier

**Date:** 2026-06-12
**Finding:** Data exists in backend, frontend checks exist, but root cause needs further investigation

---

## Summary

**Conclusion:** The two-start pitcher feature IS already built and fully wired. The data flows from MLB Stats API → Backend → Frontend correctly.

**Issue:** The UAT finding about missing two-start badges is likely a **data freshness or edge case issue**, not a missing feature. The MLB Stats API may not have probable pitchers for all games, or the name matching is failing for some pitchers.

---

## Data Flow Audit

### 1. Backend Data Collection

**File:** `backend/routers/fantasy.py`

**Function:** `_fetch_probable_starts_map(start_date, end_date)` (line 141)

```python
def _fetch_probable_starts_map(start_date: str, end_date: str) -> dict:
    """
    Return {pitcher_full_name_lower: starts_count} via public MLB Stats API (6h cached).
    """
    # Queries: https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=X&endDate=Y&gameType=R&hydrate=probablePitcher
    # Returns: {"jarren duran": 2, "jakob junis": 1, ...}
```

**Key Details:**
- **Source:** MLB Stats API (not GameDay)
- **Cache:** 6-hour TTL (module-level `_STARTS_CACHE`)
- **Matching:** Lowercase full names
- **Fallback:** Returns cached data if API fails (non-fatal)

---

### 2. Backend Data Population

**Function:** `_populate_starts_this_week(players, starts_map)` (line 182)

```python
def _populate_starts_this_week(players: list, starts_map: dict) -> None:
    """
    Mutate each SP dict to add `starts_this_week` from starts_map.
    Uses fuzzy matching at ≥0.90 ratio as a fallback for name variations.
    """
    for _fa in players:
        if "SP" not in (_fa.get("positions") or []):
            continue
        _name = (_fa.get("name") or "").strip().lower()
        _starts = starts_map.get(_name, 0)
        # Fuzzy match fallback if exact match fails
        if _starts == 0 and starts_map:
            _best = max(starts_map.keys(), key=lambda k: SequenceMatcher(None, _name, k).ratio())
            if _best and SequenceMatcher(None, _name, _best).ratio() >= 0.90:
                _starts = starts_map[_best]
        _fa["starts_this_week"] = _starts
```

**Called in:**
- `/api/fantasy/waiver-recommendations` endpoint (line 2399-2404)
- `/api/fantasy/waiver-wire` endpoint (line 2838-2841)

---

### 3. Backend Schema Fields

**File:** `backend/schemas.py` (line 435-467)

```python
class WaiverPlayerOut(BaseModel):
    # ... other fields ...
    starts_this_week: int = 0
    two_start: bool = False
    two_start_this_week: bool = False  # UI alias
    start1_opp: Optional[str] = None
    start2_opp: Optional[str] = None
```

---

### 4. Backend Response Population

**File:** `backend/routers/fantasy.py` (line 2295-2328)

```python
return WaiverPlayerOut(
    player_id=p.get("player_key") or "",
    name=name,
    # ... other fields ...
    starts_this_week=p.get("starts_this_week", 0),
    two_start=len(_starts) >= 2 or p.get("starts_this_week", 0) >= 2,
    two_start_this_week=len(_starts) >= 2 or p.get("starts_this_week", 0) >= 2,
    start1_opp=_start1_opp,
    start2_opp=_start2_opp,
    # ... other fields ...
)
```

---

### 5. Frontend Type Definitions

**File:** `frontend/lib/types.ts` (line 518-526)

```typescript
export interface WaiverAvailablePlayer {
  // ... other fields ...
  two_start?: boolean
  start1_date?: string | null
  start1_opp?: string | null
  start2_date?: string | null
  start2_opp?: string | null
  starts_this_week?: number
}
```

---

### 6. Frontend Display Logic

**File:** `frontend/app/(dashboard)/war-room/waiver/page.tsx` (line 196-200)

```tsx
{(player.starts_this_week ?? 0) >= 2 && (
  <span className="text-[10px] px-1.5 py-0.5 bg-status-safe/10 text-status-safe border border-status-safe/30 rounded font-semibold uppercase tracking-wider">
    2-Start
  </span>
)}
```

**Also appears in:**
- `frontend/app/(dashboard)/war-room/streaming/page.tsx` (line 83, 106, 252, 342)
- `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` (line 443, 659)

---

## Root Cause Analysis

### Why might two-start badges not appear?

**Hypothesis 1: Name Matching Failure**
- MLB Stats API returns: `"Jakob Junis"`
- System searches for: `"jakob junis"`
- Player has name: `"Junis, Jakob"` or middle initial `"Jakob A. Junis"`
- Exact match fails, fuzzy match ratio < 0.90 → `starts_this_week = 0`

**Hypothesis 2: Probable Pitchers Not Yet Announced**
- MLB Stats API only shows **confirmed** probable pitchers
- If a team hasn't announced tomorrow's starter, `probablePitcher` field is missing
- Result: `starts_this_week = 0`

**Hypothesis 3: API Cache Stale**
- Cache is 6-hour TTL
- If pitcher was added to probable list 2 hours ago, cache still shows old data
- Result: `starts_this_week = 0`

**Hypothesis 4: Position Mismatch**
- Player is listed as `["RP", "SP"]` but system checks for `"SP"` in positions
- Some sources only show `"RP"` for two-start relievers (e.g., opener strategy)
- Result: `_populate_starts_this_week()` skips this player

---

## Recommended Actions

### Immediate (Low Effort)

1. **Add logging to `_populate_starts_this_week()`**
   - Log when exact match fails
   - Log when fuzzy match succeeds (show ratio)
   - Log when positions don't contain "SP"

2. **Add cache timestamp to response**
   - Include `starts_cache_fetched_at` in `WaiverPlayerOut`
   - Frontend can show: "Start data from 4 hours ago"

3. **Add "Missing Probable" badge**
   - If `starts_this_week = 0` AND player_type = "pitcher" AND position includes "SP"
   - Show badge: "⚠ Probable not announced"

### Short-term (Medium Effort)

4. **Improve name matching**
   - Use BDL player_id mapping instead of names
   - Fall back to fuzzy matching only if BDL lookup fails

5. **Reduce cache TTL**
   - Change from 6 hours to 2 hours
   - Trade-off: More API calls, fresher data

### Long-term (High Effort)

6. **Use ProbablePitcherSnapshot table**
   - Table exists (`backend/models.py` line 1872)
   - Populated by job 100_014 (6 AM ET)
   - Includes `is_confirmed` flag for confirmed vs probable
   - Advantage: Persisted data, better historical tracking

---

## Verification Steps

### 1. Check API Response

Run this in production:
```bash
curl https://observant-benevolence-production.up.railway.app/api/fantasy/waiver-wire | jq '.success.data[] | select(.two_start == true) | {name, starts_this_week, two_start}'
```

**Expected:** List of 2-start pitchers with `starts_this_week: 2` and `two_start: true`

### 2. Check Name Matching

Add debug logging in `_populate_starts_this_week()`:
```python
logger.debug("Player %s: positions=%s, starts_map=%s", _name, _fa.get("positions"), starts_map.get(_name, 0))
if _starts == 0 and starts_map:
    logger.debug("Fuzzy match attempt for %s: best=%s, ratio=%.2f", _name, _best, SequenceMatcher(None, _name, _best).ratio())
```

### 3. Check MLB Stats API Directly

```bash
curl "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2026-06-12&endDate=2026-06-18&gameType=R&hydrate=probablePitcher" | jq '.dates[].games[] | .teams.home.probablePitcher.fullName, .teams.away.probablePitcher.fullName'
```

---

## Conclusion

**The feature exists.** The data flows correctly from API → Backend → Frontend.

**The issue is likely one of:**
1. Name matching failures (most likely)
2. Probable pitchers not yet announced (most likely for future dates)
3. Cache staleness (possible but less likely)

**Recommended next step:** Add debug logging and verify API response in production to identify the actual root cause.

---

## Appendix: Related Tables

### ProbablePitcherSnapshot Table

**File:** `backend/models.py` (line 1872)

```python
class ProbablePitcherSnapshot(Base):
    """
    Daily probable pitchers from MLB Stats API.
    Refresh cadence: Job 100_014 (6 AM ET daily) + game-day updates at 12 PM ET.
    """
    __tablename__ = "probable_pitchers"

    game_date = Column(Date, nullable=False)
    team = Column(String(10), nullable=False)
    pitcher_name = Column(String(100), nullable=True)
    bdl_player_id = Column(Integer, nullable=True)
    mlbam_id = Column(Integer, nullable=True)
    is_confirmed = Column(Boolean, nullable=False, default=False)
    game_time_et = Column(String(10), nullable=True)
    park_factor = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
```

**Status:** Table exists, but NOT used by `_fetch_probable_starts_map()` (which queries live API directly).

**Potential enhancement:** Use this table instead of live API calls to avoid rate limits and improve consistency.