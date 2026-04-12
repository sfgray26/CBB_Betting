# K-26: Matchup Category Alignment & Logic Issues

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Status:** COMPLETE — Unblocks Matchpage scoring accuracy fix

---

## 1. Problem Summary

The Matchup page (`/fantasy/matchup`) has three critical issues causing incorrect score display:

1. **Disorganized Category Order**: Stats display in random order from Yahoo API instead of logical Batters/Pitchers sections
2. **Non-Scoring Stats Counted**: H/AB and IP are included in win/loss totals when they should be display-only reference stats
3. **Missing Inverse Logic**: Batter K (strikeouts) should award "Win" to the *lower* value, but currently higher wins

**Impact:** Current score (e.g., 9-6) is inaccurate and doesn't match Yahoo's official standings.

---

## 2. Required Category Structure

### 2.1 Batters Section (Display Order)

| Order | Stat | Yahoo ID | Scoring? | Win Condition |
|-------|------|----------|----------|---------------|
| 1 | H/AB | N/A (composite) | ❌ Display Only | N/A |
| 2 | R | 7 | ✅ Yes | Higher |
| 3 | H | 8 | ✅ Yes | Higher |
| 4 | HR | 12 | ✅ Yes | Higher |
| 5 | RBI | 13 | ✅ Yes | Higher |
| 6 | K | 27 | ✅ Yes | **LOWER** ⚠️ |
| 7 | TB | (derived) | ✅ Yes | Higher |
| 8 | AVG | 3 | ✅ Yes | Higher |
| 9 | OPS | 55 | ✅ Yes | Higher |
| 10 | NSB | 60 | ✅ Yes | Higher |

### 2.2 Pitchers Section (Display Order)

| Order | Stat | Yahoo ID | Scoring? | Win Condition |
|-------|------|----------|----------|---------------|
| 1 | IP | 21 or 50 | ❌ Display Only | N/A |
| 2 | W | 23 | ✅ Yes | Higher |
| 3 | L | 29 | ✅ Yes | Higher* |
| 4 | HR | 12 (pitching) | ✅ Yes | Lower |
| 5 | K | 28 or 42 | ✅ Yes | Higher |
| 6 | ERA | 26 | ✅ Yes | **LOWER** |
| 7 | WHIP | 27 | ✅ Yes | **LOWER** |
| 8 | K/9 | 52 | ✅ Yes | Higher |
| 9 | QS | 29 | ✅ Yes | Higher |
| 10 | NSV | 83 | ✅ Yes | Higher |

*Note: L (Losses) as a scoring category is unusual — verify if your league actually counts this.*

---

## 3. Current Implementation Analysis

### 3.1 Frontend Issues (`frontend/app/(dashboard)/fantasy/matchup/page.tsx`)

**Issue 1: Random Category Order (Line 37)**
```typescript
// CURRENT (random order from API)
const allCats = Object.keys(data.my_team.stats)

// SHOULD BE: Explicit ordering with section headers
const BATTER_STATS_ORDER = ['H_AB', 'R', 'H', 'HR', 'RBI', 'K', 'TB', 'AVG', 'OPS', 'NSB']
const PITCHER_STATS_ORDER = ['IP', 'W', 'L', 'HR_allowed', 'K', 'ERA', 'WHIP', 'K_9', 'QS', 'NSV']
```

**Issue 2: All Stats Counted (Lines 113-118)**
```typescript
// CURRENT: All stats counted
allCats.forEach((cat) => {
  if (LOWER_IS_BETTER.has(cat) ? myVal < oppVal : myVal > oppVal) myWins++
})

// SHOULD BE: Filter out display-only stats
const DISPLAY_ONLY_STATS = new Set(['H_AB', 'IP', '21', '50'])
const scoringCats = allCats.filter(cat => !DISPLAY_ONLY_STATS.has(cat))
```

**Issue 3: Missing Batter K in LOWER_IS_BETTER (constants.ts)**
```typescript
// CURRENT (frontend/lib/constants.ts line 46)
export const LOWER_IS_BETTER = new Set(['ERA', 'WHIP', '26', '27'])

// SHOULD BE: Add batter K (stat ID 27 for batters)
export const LOWER_IS_BETTER = new Set(['ERA', 'WHIP', '26', '27', 'K_batting']) // Or use context
```

### 3.2 Backend Issues (`backend/main.py`)

**Issue: No distinction between display and scoring stats (Line 5539-5614)**
The `_extract_team_stats` function returns all stats without flagging which are display-only.

**Recommended Schema Change:**
```python
class MatchupTeamOut(BaseModel):
    team_key: str
    team_name: str
    stats: dict  # All stats for display
    scoring_stats: dict  # Only stats that count toward W/L
    display_stats: dict  # H/AB, IP, etc.
```

---

## 4. Implementation Requirements

### 4.1 Option A: Frontend-Only Fix (Minimal)

Modify just the frontend to handle ordering and scoring logic:

```typescript
// frontend/lib/constants.ts

// Define stat order and properties
export const MATCHUP_CATEGORIES = {
  // Batters
  H_AB: { label: 'H/AB', section: 'batting', scoring: false },
  R: { label: 'Runs', section: 'batting', scoring: true, lowerBetter: false, yahooId: '7' },
  H: { label: 'Hits', section: 'batting', scoring: true, lowerBetter: false, yahooId: '8' },
  HR: { label: 'Home Runs', section: 'batting', scoring: true, lowerBetter: false, yahooId: '12' },
  RBI: { label: 'RBI', section: 'batting', scoring: true, lowerBetter: false, yahooId: '13' },
  K: { label: 'Strikeouts', section: 'batting', scoring: true, lowerBetter: true, yahooId: '27' },  // ⚠️ LOWER IS BETTER
  TB: { label: 'Total Bases', section: 'batting', scoring: true, lowerBetter: false },
  AVG: { label: 'Batting Avg', section: 'batting', scoring: true, lowerBetter: false, yahooId: '3' },
  OPS: { label: 'OPS', section: 'batting', scoring: true, lowerBetter: false, yahooId: '55' },
  NSB: { label: 'Net SB', section: 'batting', scoring: true, lowerBetter: false, yahooId: '60' },
  
  // Pitchers
  IP: { label: 'Innings Pitched', section: 'pitching', scoring: false, yahooId: '21' },
  W: { label: 'Wins', section: 'pitching', scoring: true, lowerBetter: false, yahooId: '23' },
  L: { label: 'Losses', section: 'pitching', scoring: true, lowerBetter: true, yahooId: '29' },  // ⚠️ LOWER IS BETTER
  ERA: { label: 'ERA', section: 'pitching', scoring: true, lowerBetter: true, yahooId: '26' },
  WHIP: { label: 'WHIP', section: 'pitching', scoring: true, lowerBetter: true, yahooId: '27' },
  K_pit: { label: 'Strikeouts', section: 'pitching', scoring: true, lowerBetter: false, yahooId: '28' },
  'K/9': { label: 'K/9', section: 'pitching', scoring: true, lowerBetter: false, yahooId: '52' },
  QS: { label: 'Quality Starts', section: 'pitching', scoring: true, lowerBetter: false, yahooId: '29' },
  NSV: { label: 'Net Saves', section: 'pitching', scoring: true, lowerBetter: false, yahooId: '83' },
} as const
```

### 4.2 Option B: Backend+Frontend Fix (Recommended)

Backend returns structured data with scoring metadata:

```python
# backend/schemas.py
class MatchupCategory(BaseModel):
    key: str
    label: str
    section: Literal['batting', 'pitching']
    is_scoring: bool
    lower_is_better: bool
    my_value: str | number
    opponent_value: str | number
    result: Literal['win', 'loss', 'tie']

class MatchupTeamOut(BaseModel):
    team_key: str
    team_name: str
    categories: list[MatchupCategory]  # Ordered, structured
    
class MatchupResponse(BaseModel):
    week: Optional[int] = None
    my_team: MatchupTeamOut
    opponent: MatchupTeamOut
    is_playoffs: bool = False
    my_score: int  # Computed by backend
    opponent_score: int  # Computed by backend
    message: Optional[str] = None
```

---

## 5. Critical Logic Fix: Batter K

The most critical fix is that **Batter Strikeouts (K) is a NEGATIVE category** — lower is better.

In most fantasy baseball H2H leagues:
- Pitcher K: Higher is better (good for pitchers to strike batters out)
- Batter K: Lower is better (bad for batters to strike out)

Current code doesn't distinguish between batter K and pitcher K.

**Fix approach:**
```typescript
// Determine if K is batting or pitching based on other stats presence
function isBattingK(cat: string, allStats: Record<string, any>): boolean {
  // If the team has batting stats (AVG, R, HR) in the same section, it's batter K
  return allStats['AVG'] !== undefined || allStats['R'] !== undefined
}

// In winClass function:
function winClass(cat: string, myVal: number, oppVal: number, section: 'batting' | 'pitching'): string {
  if (myVal === oppVal) return 'text-zinc-400'
  
  const lowerIsBetter = (
    LOWER_IS_BETTER.has(cat) || 
    (cat === 'K' && section === 'batting') ||  // Batter K = lower is better
    (cat === 'L')  // Losses = lower is better
  )
  
  const myWins = lowerIsBetter ? myVal < oppVal : myVal > oppVal
  return myWins ? 'text-emerald-400 font-semibold' : 'text-rose-400'
}
```

---

## 6. Files to Modify

| File | Changes |
|------|---------|
| `frontend/lib/constants.ts` | Add `MATCHUP_CATEGORIES` config with ordering and scoring flags |
| `frontend/app/(dashboard)/fantasy/matchup/page.tsx` | Update `MatchupTable` and `ScoreBanner` to use new structure |
| `backend/main.py` (optional) | Move scoring logic to backend for consistency |
| `backend/schemas.py` (optional) | Add structured category schema |

---

## 7. Testing Checklist

- [ ] H/AB displays but doesn't count toward score
- [ ] IP displays but doesn't count toward score  
- [ ] Batter K awards win to team with LOWER strikeout total
- [ ] All other categories award win to HIGHER (or lower for ERA/WHIP)
- [ ] Total score matches Yahoo's official score
- [ ] Categories display in correct Batters→Pitchers order

---

*Spec complete. Ready for implementation.*
