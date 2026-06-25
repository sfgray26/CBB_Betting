# Data Inconsistency Diagnostic Report

**Date**: 2026-06-23
**Iteration**: Loop 8
**Scope**: Diagnostic only — root cause identification

---

## Symptoms

1. **Matchup record inconsistency**: Same matchup shows different records
   - War Room: `0W-0L-18T`
   - Roster page: `4W-11L-3T`

2. **need_score inconsistency**: Different values for the same player/need
   - Source A: `9.38`
   - Source B: `19.34`

---

## Data Flow Analysis

### Endpoints Identified

| Module | Frontend Page | API Endpoint | Backend Location | Cache? |
|--------|---------------|--------------|------------------|--------|
| War Room | `/war-room/page.tsx` | `/api/fantasy/matchup` | `routers/fantasy.py:4923` | **YES (5-min TTL)** |
| Roster | `/war-room/roster/page.tsx` | `/api/fantasy/scoreboard` | `routers/fantasy.py:7087` | NO |
| Waiver need_score | waiver calculation | Inline in `main.py:6037` | **Separate get_scoreboard() call** | NO |

### Data Source Hierarchy

```
Yahoo API (single source of truth)
    │
    ├── client.get_scoreboard() ────────────────────────┐
    │                                                      │
    ├── /api/fantasy/matchup (cached 5-min) ─────────────┤
    │   └── Returns: MatchupResponse {my_team.stats, opponent.stats} │
    │                                                      │
    ├── /api/fantasy/scoreboard (live) ──────────────────┤
    │   └── Returns: MatchupScoreboardResponse {categories_won/lost/tied} │
    │                                                      │
    └── Waiver need_score (live) ─────────────────────────┘
        └── Computes: category_deficits from fresh get_scoreboard() call
```

---

## Root Cause Analysis

### Finding 1: Cache Staleness

**Location**: `routers/fantasy.py:118-119`

```python
_MATCHUP_CACHE: dict = {}
_MATCHUP_CACHE_TTL = 300  # seconds (5 minutes)
```

**Problem**: `/api/fantasy/matchup` serves cached data for 5 minutes after the first fetch. If:
1. The first fetch occurs during pre-season/incomplete data (0W-0L-18T)
2. The cache is not invalidated when real data arrives
3. Subsequent requests return the stale cached response

**Evidence**: The comment at line 7131-7133 confirms a previous issue with silent empty returns:
> "The previous approach (get_matchup_stats()) had fragile nested-struct team-key matching that silently returned {} on shape variations, producing 0W-0L-18T on the roster page."

### Finding 2: need_score Data Source

**Location**: `main.py:6037`

```python
# Re-fetch scoreboard to get per-category stats
matchups2 = client.get_scoreboard()
```

**Problem**: The waiver endpoint makes a **separate, uncached call** to `get_scoreboard()`. This means:
- Waiver need_scores use FRESH data
- War Room matchup display uses STALE cache (if cache is active)

**Result**: The same matchup shows different numbers because they're reading from different temporal snapshots of the Yahoo data.

### Finding 3: Response Shape Mismatch

**MatchupResponse** (`schemas.py:655`):
```python
class MatchupResponse(BaseModel):
    week: Optional[int] = None
    my_team: MatchupTeamOut      # {team_key, team_name, stats: dict}
    opponent: MatchupTeamOut
    is_playoffs: bool = False
    message: Optional[str] = None
```

**ScoreboardResponse** (`contracts.py:379`):
```python
class MatchupScoreboardResponse(BaseModel):
    week: int
    opponent_name: str
    categories_won: int
    categories_lost: int
    categories_tied: int
    # ... projection fields
```

**Impact**: These are fundamentally different response shapes:
- `MatchupResponse` returns raw stats (my_team.stats dict)
- `ScoreboardResponse` returns computed W/L/T counts

**Divergence Point**: The W/L/T computation happens differently in each endpoint.

---

## Data Flow Trace

### War Room (0W-0L-18T path)

```
Frontend: /war-room/page.tsx
    ↓ calls
endpoints.getMatchup()
    ↓ calls
GET /api/fantasy/matchup
    ↓ checks
_MATCHUP_CACHE.get(user) → HIT (stale)
    ↓ returns (stale data)
MatchupResponse with stats showing all zeros/ties
```

### Roster (4W-11L-3T path)

```
Frontend: /war-room/roster/page.tsx
    ↓ calls
endpoints.getScoreboard()
    ↓ calls
GET /api/fantasy/scoreboard
    ↓ fetches
client.get_scoreboard() (LIVE)
    ↓ computes
assemble_matchup_scoreboard() → scoreboard_orchestrator.py
    ↓ returns
MatchupScoreboardResponse with computed categories_won/lost/tied
```

### Waiver need_score (9.38 vs 19.34 path)

```
Waiver endpoint in main.py
    ↓
Separate client.get_scoreboard() call (LIVE, no cache)
    ↓
Computes category_deficits from fresh stats
    ↓
Weighted z-score calculation produces need_score
```

---

## Root Cause Summary

**PRIMARY CAUSE**: Cache staleness in `/api/fantasy/matchup`

- The 5-minute cache (`_MATCHUP_CACHE_TTL = 300`) can serve stale data
- If the cache is populated during incomplete data, it returns 0W-0L-18T
- Roster page bypasses this cache with live data → shows correct 4W-11L-3T

**SECONDARY CAUSE**: need_score uses separate live fetch

- Waiver calculations make an independent `get_scoreboard()` call
- This bypasses the matchup cache entirely
- Results in different need_score values (9.38 vs 19.34) depending on which temporal snapshot is used

---

## Recommended Fix Approach

### Option 1: Disable/Remove Cache (Simplest)

**Change**: Remove `_MATCHUP_CACHE` from `routers/fantasy.py:118-119`

**Pros**:
- Simplest fix (2 lines)
- Eliminates staleness entirely
- All endpoints use fresh Yahoo data

**Cons**:
- Increased Yahoo API load (multiple parallel requests hit Yahoo instead of cache)

### Option 2: Reduce Cache TTL + Add Invalidator

**Change**:
1. Reduce TTL from 300s to 60s
2. Add cache invalidation when data transitions from empty to populated

**Pros**:
- Reduces API load vs Option 1
- Still prevents long staleness

**Cons**:
- More complex (requires staleness detection logic)
- 60s still allows some stale data window

### Option 3: Single Source of Truth (Architectural)

**Change**: Make `/api/fantasy/scoreboard` the canonical endpoint

1. Deprecate `/api/fantasy/matchup`
2. Update frontend to use `/api/fantasy/scoreboard` everywhere
3. Add caching at the scoreboard level with proper invalidation

**Pros**:
- Single data path eliminates divergence
- Richer response format (W/L/T pre-computed)

**Cons**:
- Multiple frontend files to update (exceeds 3-file constraint for this iteration)

---

## Architectural Concern

**Comment from `routers/fantasy.py:7131-7134`**:
> "The previous approach (get_matchup_stats()) had fragile nested-struct team-key matching that silently returned {} on shape variations, producing 0W-0L-18T on the roster page. Replacing it with _iter_scoreboard_matchup_teams() fixes both pages to read from the same Yahoo data source."

This suggests the issue was THOUGHT to be fixed by unifying the Yahoo data source. However, the **cache layer was not addressed**, so the fix was incomplete.

---

## Files Changed (This Report)

1. `data_inconsistency_diagnostic.md` — This file (NEW)
2. `loop_log.md` — To be updated with Loop 8 completion

---

## Next Steps (Awaiting User Approval)

1. Review this diagnostic report
2. Approve fix approach (Option 1, 2, or 3)
3. Implement approved fix in Loop Iteration 9

**No code changes will be made until user approval.**
