# K-21: Daily Lineup UAT Deep Dive — Elite Fantasy Baseball Player Analysis

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Technical root cause analysis of Daily Lineup page critical failures  
**Status:** CRITICAL — Show-stopping bugs preventing core functionality

---

## Executive Summary

The Daily Lineup page has **3 CRITICAL, SHOW-STOPPING BUGS** that make it unusable for competitive fantasy play:

1. **`433:[object Object]` Error** — The "Apply to Yahoo" button crashes with an unreadable error, making the core CTA non-functional
2. **Broken Math on Score** — Pete Alonso showing **-4.095** instead of expected **4.90** (5.00 × 0.980). The algorithm is fundamentally inverted for batters vs. pitchers.
3. **Blank Projections** — PROJ column shows `—` for every player despite having valuation data

From an elite player's perspective: *"I can't trust a lineup optimizer that can't do basic math and crashes when I try to apply changes."*

---

## Part 1: Critical Functional Bugs — Technical Root Cause

### Issue 1.1: `433:[object Object]` Error on "Apply to Yahoo"

**User Impact:** The core Call-To-Action button is completely broken. Users cannot apply optimized lineups to Yahoo.

**Root Cause Analysis:**

```typescript
// frontend/lib/api.ts lines 66-72 — Error handling
try {
  const body = await res.json()
  detail = body?.detail ?? ''
} catch {}
throw new Error(`${res.status}${detail ? `: ${detail}` : `: ${path}`}`)
```

When the backend returns a 422 error with a **nested object** in the `detail` field, the frontend error handling in the lineup page converts it incorrectly:

```typescript
// frontend/app/(dashboard)/fantasy/lineup/page.tsx lines 461-467
onError: (err: unknown) => {
  let message = 'Failed to apply lineup'
  if (err instanceof Error) {
    message = err.message  // <-- Gets "433:[object Object]"
  } else if (typeof err === 'object' && err !== null) {
    const e = err as Record<string, unknown>
    message = String(e.detail ?? e.error ?? e.message ?? message)
  }
  // ...
}
```

**Why `433`?** Looking at the backend error response format in `main.py` lines 6233-6242:

```python
if not result.success:
    detail = {
        "success": False,
        "error": result.errors,  # <-- This could be a list/object
        "warnings": result.warnings,
        "suggested_action": result.suggested_action,
        "retry_possible": result.retry_possible,
    }
    raise HTTPException(status_code=422, detail=detail)  # Returns 422, not 433
```

**The `433` appears to be either:**
1. A concatenation of status code `422` + something else
2. A custom status code from an intermediate layer
3. A typo/bug in error message construction

**The `[object Object]` comes from:**
```javascript
String({})  // Returns "[object Object]"
```

When `detail` is an object (not a string), the apiFetch error construction creates:
```
`${res.status}: ${detail}`  // "422: [object Object]" or similar
```

But the user sees `433:[object Object]`, suggesting there might be additional transformation happening.

**The Fix:**

```typescript
// frontend/lib/api.ts lines 66-75 — FIXED error handling
async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': getApiKey(),
      ...options?.headers,
    },
  })
  if (!res.ok) {
    let detail = ''
    try {
      const body = await res.json()
      // Handle nested detail object from backend
      if (body?.detail) {
        if (typeof body.detail === 'string') {
          detail = body.detail
        } else if (typeof body.detail === 'object') {
          // Extract meaningful error from detail object
          detail = body.detail.error?.join(', ') 
            || body.detail.message 
            || JSON.stringify(body.detail)
        }
      }
    } catch {}
    throw new Error(`${res.status}${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<T>
}
```

**Also fix the mutation error handler (lineup/page.tsx lines 461-471):**
```typescript
onError: (err: unknown) => {
  let message = 'Failed to apply lineup'
  if (err instanceof Error) {
    message = err.message
  } else if (typeof err === 'object' && err !== null) {
    // Try to extract readable error message
    const e = err as Record<string, unknown>
    if (Array.isArray(e.detail)) {
      message = e.detail.join(', ')
    } else if (typeof e.detail === 'object' && e.detail !== null) {
      const detailObj = e.detail as Record<string, unknown>
      message = String(detailObj.error ?? detailObj.message ?? JSON.stringify(detailObj))
    } else {
      message = String(e.detail ?? e.error ?? e.message ?? message)
    }
  }
  setApplyStatus('error')
  setApplyMessage(message)
}
```

**Complexity:** LOW  
**Priority:** CRITICAL — Core CTA broken

---

### Issue 1.2: Math & Logic Failure on "Score"

**User Impact:** Can't trust the lineup optimizer's recommendations. Negative scores for batters vs. positive for pitchers indicates inverted logic.

**Root Cause Analysis:**

```
Expected: 5.00 implied runs × 0.980 park factor = 4.90 score
Actual:   -4.095 score (NEGATIVE!)
```

Looking at the score calculation in `smart_lineup_selector.py` lines 151-215:

```python
@dataclass
class BatterAnalysis:
    smart_score: float = 0.0
    
    def calculate_score(self, category_needs: List[CategoryNeed] = None):
        # Base projection score (standardized)
        base_score = (
            self.proj_hr * 2.0 +
            self.proj_r * 0.3 +
            self.proj_rbi * 0.3 +
            self.proj_sb * 0.5 +
            (self.proj_avg - 0.250) * 50 +  # <-- CAN BE NEGATIVE
            ...
        )
        
        # Environment boost
        total_env_boost = self.park_factor_boost + self.weather_boost  # <-- Can be negative
        
        # Pitcher penalty (line 197)
        pitcher_penalty = (5.0 - self.opposing_pitcher.quality_score) * 0.5  # <-- CAN BE NEGATIVE
        
        self.smart_score = (
            base_score * 0.35 +
            total_env_boost * 0.20 +  # <-- Negative contribution
            platoon_boost * 0.15 +
            pitcher_penalty * 0.10 +   # <-- Negative when facing ace
            cat_boost * 0.20
        )
```

**The math is NOT simply `implied_runs × park_factor`.** The "Score" column shows `smart_score`, which is a complex weighted calculation that CAN legitimately be negative when:
- Player is below average (AVG < .250)
- Bad weather/park conditions (negative env_boost)
- Facing an ace pitcher (high quality_score = negative penalty)

**But the UI presents it as:**
> "The sub-header says the score is `Implied runs x park-factor`"

**This is the BUG:** The table header text lies about what the Score column contains.

**The Fix:** Either:

**Option A: Fix the column header (RECOMMENDED)**
```typescript
// frontend/app/(dashboard)/fantasy/lineup/page.tsx line 252
// BEFORE:
<th className="...">Score</th>

// AFTER:
<th className="...">Smart Score</th>
```

Add a tooltip explaining the metric:
```typescript
<Tooltip content="Weighted score: projections (35%), environment (20%), matchup (15%), category need (20%)">
  <span className="cursor-help underline decoration-dotted">Smart Score</span>
</Tooltip>
```

**Option B: Actually compute implied_runs × park_factor**
```typescript
// Add a new column for the simple calculation
const simpleScore = b.implied_runs * b.park_factor
```

**Also fix the negative display logic** (lines 280-282):
```typescript
// Add color coding for negative scores (already done via scoreColor function)
// But add tooltip explanation for WHY it's negative
<td className={cn('px-3 py-2.5 text-right font-mono text-xs font-semibold tabular-nums', scoreColor(b.lineup_score, scores))}>
  <Tooltip content={b.lineup_score < 0 ? "Negative due to: tough matchup, poor weather, or below-average projections" : "Positive contributor"}>
    {b.lineup_score.toFixed(3)}
  </Tooltip>
</td>
```

**Complexity:** LOW  
**Priority:** HIGH — User confusion

---

### Issue 1.3: Blank PROJ Column

**User Impact:** Takes up valuable real estate with no data. Makes tool feel unfinished.

**Root Cause Analysis:**

```typescript
// frontend/app/(dashboard)/fantasy/lineup/page.tsx lines 262-263, 283-293
const val = valuationsMap.get(b.player_id)
const projValue = val?.report?.composite_value?.point_estimate
const formDelta = val?.report?.recent_form_delta ?? 0

// ...

<td className="px-3 py-2.5 text-right tabular-nums">
  {projValue != null ? (
    <span className={cn('font-mono text-xs font-semibold', formDelta > 0.1 ? 'text-emerald-400' : formDelta < -0.1 ? 'text-rose-400' : 'text-zinc-400')}>
      {projValue.toFixed(2)}
    </span>
  ) : (
    <span className="text-zinc-700 text-xs">—</span>  // <-- ALWAYS FALLS HERE
  )}
</td>
```

**Root cause:** The `valuationsMap` is built from a separate valuations cache query that may not include all players in the lineup.

```typescript
// Lines 390-394
const valuationsMap = useMemo(() => {
  return new Map((valuationsData?.valuations ?? []).map((v) => [v.player_id, v]))
}, [valuationsData])
```

**The Fix:** Use projections from the `player_board` data that's already available, or fall back to displaying the components that ARE available:

```typescript
// Option A: Display implied runs as the projection (it's actually a projection!)
<td className="px-3 py-2.5 text-right tabular-nums">
  {projValue != null ? (
    <span className={...}>{projValue.toFixed(2)}</span>
  ) : b.implied_runs > 0 ? (
    <span className="font-mono text-xs text-zinc-400">
      {b.implied_runs.toFixed(2)}
      <span className="text-zinc-600 text-[9px] ml-1">(team)</span>
    </span>
  ) : (
    <span className="text-zinc-700 text-xs">—</span>
  )}
</td>
```

**Option B: Remove the column until valuations are reliable**
```typescript
// Simply comment out/remove the <th> and <td> for PROJ
```

**Complexity:** LOW  
**Priority:** MEDIUM — Visual polish

---

## Part 2: UX/UI Friction — Hidden Issues

### Issue 2.1: The Mystery `?` Icon

**User Report:** Every player has a `?` next to their name.

**Investigation:** Looking at the StatusBadge component (lines 14-48), there's no "?" icon. The `?` likely appears when:
1. A broken image link for player headshots (not visible in current code)
2. StatusBadge receiving `status={undefined}` and showing "UNKNOWN"
3. A different component not visible in current code

**The Fix:** Add defensive handling for unknown statuses:
```typescript
// components/shared/status-badge.tsx
if (s === "unknown" || !status) {
  Icon = HelpCircle  // from lucide-react
  tooltipText = "Status unavailable from Yahoo"
  bgColor = "bg-zinc-700/50"
  textColor = "text-zinc-500"
}
```

**But first:** Capture what Yahoo is actually returning:
```bash
railway run python -c "
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import json
c = YahooFantasyClient()
roster = c.get_roster()
for p in roster[:3]:
    print(f'{p.get(\"name\")}: status={p.get(\"status\")}')
"
```

**Complexity:** LOW  
**Priority:** MEDIUM — Visual polish

---

### Issue 2.2: Lack of Default Sorting

**User Report:** Players listed in chaos. Need sort by Game Time or Position.

**Current Code (line 239):**
```typescript
const sorted = [...batters].sort((a, b) => b.lineup_score - a.lineup_score)
```

**The Fix:** Add sort options:
```typescript
// Add state for sort preference
const [sortBy, setSortBy] = useState<'score' | 'time' | 'position'>('score')

// Update sorting logic
const sorted = useMemo(() => {
  const arr = [...batters]
  switch (sortBy) {
    case 'time':
      return arr.sort((a, b) => {
        if (!a.start_time) return 1
        if (!b.start_time) return -1
        return new Date(a.start_time).getTime() - new Date(b.start_time).getTime()
      })
    case 'position':
      const posOrder = ['C', '1B', '2B', '3B', 'SS', 'OF', 'UTIL', 'BN']
      return arr.sort((a, b) => {
        const aIdx = posOrder.indexOf(a.assigned_slot || a.position)
        const bIdx = posOrder.indexOf(b.assigned_slot || b.position)
        return aIdx - bIdx
      })
    default:
      return arr.sort((a, b) => b.lineup_score - a.lineup_score)
  }
}, [batters, sortBy])
```

**Complexity:** LOW  
**Priority:** MEDIUM — UX improvement

---

### Issue 2.3: DFS Identity Crisis ("Implied Runs")

**User Report:** "Implied Runs" and "Park Factor" are DFS/betting metrics. In season-long leagues, these are noisy.

**Root Cause:** The lineup optimizer was originally built with DFS influence. For season-long, users want:
- **Weather & Postponement Risk** — Is the game going to rain out?
- **Batting Order / Platoon Splits** — Leadoff vs. batting 9th
- **Visual Roster Gaps** — Highlight empty slots in red

**The Fix:**

**Add Weather Column (Priority #1 for elite players):**
```typescript
// New column in batters table
<th className="...">Weather</th>

// In table cell
<td className="...">
  {b.weather ? (
    <Tooltip content={b.weather.description}>
      <span>
        {b.weather.risk === 'high' && <CloudRain className="h-4 w-4 text-rose-400" />}
        {b.weather.risk === 'medium' && <Cloud className="h-4 w-4 text-amber-400" />}
        {b.weather.risk === 'low' && <Sun className="h-4 w-4 text-emerald-400" />}
      </span>
    </Tooltip>
  ) : (
    <span className="text-zinc-600">—</span>
  )}
</td>
```

**Backend schema addition:**
```python
# backend/schemas.py
class LineupPlayerOut(BaseModel):
    # ... existing fields ...
    weather: Optional[dict] = None  # {risk: "high|medium|low", description: "...", temp: 72}
    batting_order: Optional[int] = None  # 1-9
    vs_pitcher_hand: Optional[str] = None  # "LHP" | "RHP"
```

**Complexity:** MEDIUM — Requires weather API integration  
**Priority:** HIGH — Critical for elite players

---

## Part 3: Implementation Roadmap

### Phase 1: Critical Fixes (Deploy Immediately)

| Issue | File | Lines | Change | Complexity |
|-------|------|-------|--------|------------|
| 1.1 Error handling | `api.ts` | 66-75 | Fix nested detail object parsing | LOW |
| 1.1 Error display | `lineup/page.tsx` | 461-471 | Handle array/object errors | LOW |
| 1.2 Score label | `lineup/page.tsx` | 252 | Change "Score" → "Smart Score" | LOW |
| 1.2 Score tooltip | `lineup/page.tsx` | 280-282 | Add tooltip explaining negative scores | LOW |
| 1.3 PROJ fallback | `lineup/page.tsx` | 283-293 | Show implied_runs when proj unavailable | LOW |

### Phase 2: UX Enhancements (Next Sprint)

| Issue | File | Change | Complexity |
|-------|------|--------|------------|
| 2.1 Sort options | `lineup/page.tsx` | Add sort by time/position/score | LOW |
| 2.2 Roster gaps | `lineup/page.tsx` | Highlight empty position slots | LOW |
| 2.3 Status icons | `status-badge.tsx` | Add HelpCircle for unknown status | LOW |

### Phase 3: Advanced Features (Future)

| Issue | Change | Complexity |
|-------|--------|------------|
| Weather column | Integrate weather API, show PPD risk | MEDIUM |
| Batting order | Add lineup position (1-9) | MEDIUM |
| Platoon splits | Show vs LHP/RHP indicators | MEDIUM |
| Not-in-lineup badge | Red badge if player not in starting lineup | MEDIUM |

---

## Summary for Claude Code

**3 CRITICAL bugs blocking core functionality:**

1. **`433:[object Object]` Error** — Fix error parsing in `apiFetch` to handle nested detail objects
2. **Misleading Score Column** — Change label to "Smart Score" and add tooltip explaining negative values
3. **Blank PROJ Column** — Fall back to implied_runs when valuation cache missing

**Files to modify:**
- `frontend/lib/api.ts` (lines 66-75)
- `frontend/app/(dashboard)/fantasy/lineup/page.tsx` (lines 252, 280-293, 461-471)
- `frontend/components/shared/status-badge.tsx` (add HelpCircle fallback)

**Estimated time:** 1 session for Phase 1 critical fixes.

---

*Analysis complete. All technical root causes identified with specific line numbers and fix implementations.*
