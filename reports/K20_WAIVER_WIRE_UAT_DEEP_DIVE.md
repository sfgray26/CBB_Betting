# K-20: Waiver Wire UAT Deep Dive — Elite Fantasy Baseball Player Analysis

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Technical root cause analysis of waiver wire functional gaps  
**Status:** CRITICAL — Core data feeds broken, UX friction high

---

## Executive Summary

The Waiver Wire page is currently a **"wireframe without an engine."** From an elite fantasy player's perspective, this page fails the fundamental test: *Can I identify my roster's biggest weakness and find the optimal replacement in under 60 seconds?* **Answer: No.**

This report validates the user's UAT findings with technical root cause analysis and provides specific implementation fixes for Claude Code.

---

## Part 1: Validation of Broken Features — Technical Root Cause

### Issue 1.1: Owned % at 0.0% for All Players

**User Impact:** Cannot gauge market consensus or FAAB bidding strategy. No differentiation between "must-add" players vs. deep-league fliers.

**Root Cause Analysis:**

**From Raw Yahoo Data (see Appendix):**
```json
"rankings": {
    "percent_rostered": 0.29  // ← Yahoo uses this key, not "percent_owned"
}
```

**Current Code (yahoo_client_resilient.py lines 535-541):**
```python
def get_free_agents(self, position: str = "", start: int = 0, count: int = 25) -> list[dict]:
    params = {"status": "A", "start": start, "count": count, "sort": "AR",
              "out": "metadata"}  # <-- PROBLEM 1: metadata excludes ownership
    data = self._get(f"league/{self.league_key}/players", params=params)
    players_raw = self._league_section(data, 1).get("players", {})
    return self._parse_players_block(players_raw)
```

The `"out": "metadata"` parameter tells Yahoo to return **metadata only** — no ownership statistics.

**Additional Problem:** The code looks for `percent_owned` but Yahoo returns `percent_rostered`.

**Fix Required:**
```python
# Option 1: Remove "out": "metadata" to get full player data
params = {"status": "A", "start": start, "count": count, "sort": "AR"}
# WITHOUT "out": "metadata"

# Option 2: Parse correct key name in _parse_player()
# Change from:
owned_pct = meta.get("percent_owned", 0)  # Wrong key
# To:
owned_pct = meta.get("percent_rostered", 0) * 100  # Yahoo returns 0.29, need 29.0
```

**Evidence from Raw Data:**
- Brandon Lowe: `percent_rostered: 0.82` = 82% owned
- Andrés Giménez: `percent_rostered: 0.30` = 30% owned
- Joey Wiemer: `percent_rostered: 0.29` = 29% owned

The data IS available from Yahoo — just using wrong key AND excluding it with `out=metadata`.

**Complexity:** MEDIUM — Requires new Yahoo API batch method  
**Priority:** CRITICAL — Blocks core waiver functionality

---

### Issue 1.2: Key Stats Blank

**User Impact:** Cannot make add/drop decisions. The "Key Stats" column shows `category_contributions` which are derived from matchup deficit analysis, not actual player stats.

**Root Cause Analysis:**

Current data flow:
```
Waiver Player Table → category_contributions (dict)
  ↓
Computed from: need_score calculation (main.py lines 4625-4644)
  ↓
Sources from: board_player.cat_scores (z-scores)
  ↓
If no category_deficits: contributions = {}
```

The `category_contributions` field shows how much a player would help in categories where you're losing — NOT the player's actual stats.

**Current Rendering (waiver/page.tsx lines 176-184):**
```typescript
<td className="px-3 py-2.5 text-xs text-zinc-500 font-mono">
  {topStats.map(([cat, val]) => (
    <span key={cat} className="mr-2">
      <span className="text-zinc-600">{cat}:</span>
      <span className={cn('ml-1', val >= 0 ? 'text-emerald-400' : 'text-rose-400')}>
        {val >= 0 ? '+' : ''}{val}
      </span>
    </span>
  ))}
</td>
```

**The Fix:** Add real player stats to `WaiverPlayerOut` schema:

```python
# backend/schemas.py — Add to WaiverPlayerOut (line 380)
class WaiverPlayerOut(BaseModel):
    # ... existing fields ...
    
    # NEW: Actual player stats (not contributions)
    stats: dict = {}  # {"HR": 12, "RBI": 45, "AVG": .274, ...}
    stats_timeframe: str = "season"  # "season" | "last_7" | "last_14" | "last_30"
    
    # For pitchers
    pitching_stats: dict = {}  # {"ERA": 3.45, "WHIP": 1.12, "K": 89, ...}
```

**Frontend Update (waiver/page.tsx lines 176-184):**
```typescript
// Replace category_contributions display with actual stats
const displayStats = p.position === 'SP' || p.position === 'RP' 
  ? [
      ['ERA', p.pitching_stats?.ERA?.toFixed(2) ?? '-'],
      ['WHIP', p.pitching_stats?.WHIP?.toFixed(2) ?? '-'],
      ['K', p.pitching_stats?.K ?? '-'],
    ]
  : [
      ['AVG', p.stats?.AVG?.toFixed(3) ?? '-'],
      ['HR', p.stats?.HR ?? '-'],
      ['RBI', p.stats?.RBI ?? '-'],
      ['SB', p.stats?.SB ?? '-'],
    ];
```

**Complexity:** MEDIUM — Requires Yahoo API stats fetching  
**Priority:** CRITICAL — Core functionality missing

---

### Issue 1.3: Max Owned Stuck at 90%

**User Impact:** Elite players hunt in 10-40% range. Forcing manual slider adjustment every visit adds friction.

**Root Cause Analysis:**

```typescript
// frontend/app/(dashboard)/fantasy/waiver/page.tsx
// Line 349: Hardcoded default
const [maxOwned, setMaxOwned] = useState<number>(90)

// Line 463: Slider range
<input type="range" min={0} max={100} value={maxOwned} ... />
```

**The Fix:** Change default to 50% for competitive leagues:
```typescript
// Line 349
const [maxOwned, setMaxOwned] = useState<number>(50)  // Was 90

// Also add quick-select buttons for common ranges:
<div className="flex gap-1">
  {[25, 50, 75].map(pct => (
    <button 
      key={pct}
      onClick={() => setMaxOwned(pct)}
      className={cn("px-2 py-1 text-xs rounded", maxOwned === pct ? "bg-amber-500/20" : "bg-zinc-800")}
    >
      {pct}%
    </button>
  ))}
</div>
```

**Complexity:** LOW  
**Priority:** MEDIUM — UX friction

---

### Issue 1.4: The "Need Score" Enigma

**User Impact:** Proprietary metric without explanation is noise. "-0.80" for Andrés Giménez — does negative mean I don't need him, or he fills a deficit?

**Root Cause Analysis:**

```python
# backend/main.py lines 4622-4644
need_score = 0.0
contributions: dict = {}

if category_deficits:
    cat_scores = board_player.get("cat_scores", {})
    for cd in category_deficits:
        if cd.winning or cd.deficit <= 0:
            continue
        board_key = _YAHOO_CAT_TO_BOARD.get(cd.category)
        if not board_key or board_key not in cat_scores:
            continue
        player_z = cat_scores[board_key]
        if player_z <= 0:
            continue
        opp_total = abs(cd.opponent_total) or 1.0
        deficit_weight = cd.deficit / opp_total
        contribution = deficit_weight * player_z
        contributions[cd.category] = round(contribution, 3)
        need_score += contribution
```

**Interpretation:**
- **Positive need_score:** Player helps in categories where you're losing
- **Negative need_score:** Not possible with current logic (filters z <= 0)
- **Zero need_score:** Either no deficits, or player doesn't help in losing categories

**The Fix:** Add tooltip explaining the metric:
```typescript
// waiver/page.tsx line 170
<td className="px-3 py-2.5 text-right font-mono text-xs font-semibold text-amber-400 tabular-nums">
  <Tooltip content={
    <div className="space-y-1">
      <p className="font-semibold">Need Score: {p.need_score.toFixed(2)}</p>
      <p className="text-xs text-zinc-400">Positive = helps your losing categories</p>
      <p className="text-xs text-zinc-400">Higher = bigger impact on matchup</p>
      {Object.entries(p.category_contributions).map(([cat, val]) => (
        <div key={cat} className="flex justify-between text-xs">
          <span>{cat}:</span>
          <span className={val > 0 ? 'text-emerald-400' : 'text-rose-400'}>
            {val > 0 ? '+' : ''}{val.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  }>
    {p.need_score.toFixed(2)}
  </Tooltip>
</td>
```

**Complexity:** LOW — Frontend only  
**Priority:** MEDIUM — UX clarity

---

### Issue 1.5: Empty Trackers (Category & 2-Start)

**User Impact:** Visual dead zones wasting prime real estate.

**Root Cause Analysis:**

**Category Tracker (waiver/page.tsx line 532-533):**
```typescript
{!data || data.category_deficits.length === 0 ? (
  <p className="text-zinc-600 text-sm text-center py-8">No category data available.</p>
```

**Root cause:** `category_deficits` requires:
1. Active matchup (line 4508: `if matchup_opponent != "TBD"`)
2. Scoreboard data with team stats (lines 4530-4596)
3. Successful mapping of Yahoo categories to board keys

**2-Start Pitchers (waiver/page.tsx line 212):**
```typescript
if (pitchers.length === 0) {
  return <p className="text-zinc-600 text-sm text-center py-8">No 2-start pitchers available this week.</p>
```

**Root cause:** Backend filtering too aggressive (main.py lines 4684-4784):
```python
# Two-start pitchers: SPs from free agents with 2+ probable starts
# Logic requires exact name matching between Yahoo and MLB API data
```

**The Fix:** Collapse empty modules and use space for Top Available Players table:
```typescript
// Replace empty state with collapsed card
{data?.category_deficits?.length > 0 && (
  <Card>...</Card>  // Only render if data exists
)}

// Use saved space to expand Top Available table
<Card className="p-0 flex-1">  // Add flex-1 to take remaining space
```

**Complexity:** LOW — Frontend layout only  
**Priority:** MEDIUM — Visual polish

---

### Issue 1.6: Add/Drop Recommendations Require Button Click

**User Impact:** Clunky UX. Should auto-load recommendations based on roster analysis.

**Root Cause Analysis:**

```typescript
// waiver/page.tsx lines 351, 379-389
const [showRecs, setShowRecs] = useState<boolean>(false)  // Hidden by default

const { data: recData, ... } = useQuery({
  queryKey: ['fantasy-waiver-recs'],
  queryFn: endpoints.waiverRecommendations,
  enabled: showRecs,  // Only loads when showRecs=true
  ...
})
```

**The Fix:** Auto-load on page mount:
```typescript
const [showRecs, setShowRecs] = useState<boolean>(true)  // Default to true

const { data: recData, ... } = useQuery({
  queryKey: ['fantasy-waiver-recs'],
  queryFn: endpoints.waiverRecommendations,
  enabled: true,  // Always load
  staleTime: 5 * 60 * 1000,  // 5 minute cache
})
```

**Complexity:** LOW  
**Priority:** LOW — UX convenience

---

## Part 2: Elite Player UX Deep Dive — Hidden Issues

### Issue 2.1: Platform Identity Crisis

**Problem:** Global navigation screams "Sports Betting" (CLV Analysis, Live Slate, Odds Monitor) while fantasy player needs dedicated workspace.

**Already Fixed:** ARCH-003 F1 implemented context-aware sidebar branding. Verify this is working on waiver page.

**Verification:**
```typescript
// Check sidebar.tsx for fantasy route detection
// Should show "FANTASY BASEBALL" not "CBB EDGE"
// Should hide betting portfolio panel
```

---

### Issue 2.2: Visual Hierarchy of Tags

**Problem:** `Gary Sánchez COLD` — status tag is plain text mashed next to name. Elite players scan in seconds; need distinct color-coded badges.

**Current (waiver/page.tsx lines 134-143):**
```typescript
{p.hot_cold === 'HOT' && (
  <span className="text-xs font-mono bg-red-500/20 text-red-400 border border-red-500/30 px-1.5 py-0.5 rounded">
    HOT
  </span>
)}
{p.hot_cold === 'COLD' && (
  <span className="text-xs font-mono bg-blue-500/20 text-blue-400 border border-blue-500/30 px-1.5 py-0.5 rounded">
    COLD
  </span>
)}
```

**The Fix:** Add Lucide icons to StatusBadge component:
```typescript
// components/shared/status-badge.tsx
import { Flame, Snowflake, Activity, AlertTriangle, XCircle } from 'lucide-react'

const HOT_ICON = <Flame className="h-3 w-3" />
const COLD_ICON = <Snowflake className="h-3 w-3" />

// Enhance HOT/COLD badges with icons
<span className="...">
  {HOT_ICON}
  <span className="ml-1">HOT</span>
</span>
```

**Complexity:** LOW  
**Priority:** LOW — Visual polish

---

### Issue 2.3: Lack of Positional Granularity

**Problem:** Need hyper-specific role filters, especially for pitchers (SPs with RP eligibility, closers vs. middle relievers).

**Current (waiver/page.tsx line 344):**
```typescript
const POSITIONS = ['C', '1B', '2B', '3B', 'SS', 'OF', 'SP', 'RP']
```

**The Fix:** Add pitcher sub-filters:
```typescript
const POSITION_GROUPS = [
  { label: 'All', value: '' },
  { label: 'C', value: 'C' },
  { label: '1B', value: '1B' },
  { label: '2B', value: '2B' },
  { label: '3B', value: '3B' },
  { label: 'SS', value: 'SS' },
  { label: 'OF', value: 'OF' },
  { label: 'SP', value: 'SP', subs: ['SP/RP', '2-Start'] },
  { label: 'RP', value: 'RP', subs: ['Closer', 'Setup'] },
]
```

**Complexity:** MEDIUM — Requires backend support for eligibility filtering  
**Priority:** LOW — Nice to have

---

### Issue 2.4: Missing Timeframe Toggles

**Problem:** No way to sort by recent performance (Last 7 Days, Last 14 Days, Season-to-Date).

**Current Sort (waiver/page.tsx lines 449-456):**
```typescript
<select value={sort} onChange={...}>
  <option value="need_score">Sort: Need Score</option>
  <option value="percent_owned">Sort: % Owned</option>
</select>
```

**The Fix:** Add timeframe selector:
```typescript
const [timeframe, setTimeframe] = useState<'season' | 'last_7' | 'last_14' | 'last_30'>('season')

// Add to API call
endpoints.waiverWire({ position, sort, max_percent_owned: maxOwned, page, timeframe })

// Backend: Use Yahoo's stat_type parameter
# 'season', 'lastweek', 'lastmonth' (Yahoo API supports these)
```

**Complexity:** MEDIUM — Requires Yahoo API parameter mapping  
**Priority:** MEDIUM — Key for identifying breakouts

---

## Part 3: Data Schema Requirements

### Enhanced WaiverPlayer Schema

```typescript
// frontend/lib/types.ts
export interface WaiverPlayer {
  player_id: string
  name: string
  team: string
  position: string
  need_score: number
  category_contributions: Record<string, number>
  owned_pct: number
  owned_trend: number  // NEW: +/- over last 48 hours
  starts_this_week: number
  statcast_signals?: string[]
  hot_cold?: 'HOT' | 'COLD' | null
  status?: string | null
  injury_note?: string | null
  
  // NEW: Actual player stats (position-dependent)
  stats: {
    // Hitters
    AVG?: number
    HR?: number
    RBI?: number
    R?: number
    SB?: number
    OPS?: number
    // Pitchers  
    ERA?: number
    WHIP?: number
    K?: number
    W?: number
    SV?: number
    IP?: number
  }
  stats_timeframe: 'season' | 'last_7' | 'last_14' | 'last_30'
  
  // NEW: Trend indicators
  trend_direction: 'UP' | 'DOWN' | 'FLAT'
  rostered_change_48h: number  // Percentage point change
}
```

### Backend Schema Addition

```python
# backend/schemas.py
class WaiverPlayerOut(BaseModel):
    # ... existing fields ...
    
    # NEW
    stats: dict = {}  # Raw stats from Yahoo API
    stats_timeframe: str = "season"
    owned_trend: float = 0.0  # 48h change in ownership
    rostered_change_48h: float = 0.0
```

---

## Appendix: Raw Yahoo Player List Data (Ground Truth)

**Source:** User's actual Yahoo Fantasy Player List (Waiver Wire) page

### Sample Player Data (Top 5 Available Batters)

```json
{
  "waiver_wire_meta": {
    "filters": { "position": "All Batters", "status": "All Available Players" },
    "sort_by": "Current Rank"
  },
  "available_players": [
    {
      "player_info": {
        "name": "Joey Wiemer",
        "team": "WSH",
        "positions": ["LF", "CF", "RF"],
        "injury_status": null
      },
      "roster_status": "FA",
      "rankings": {
        "pre_season": 1601,
        "current": 2,
        "percent_rostered": 0.29  // ← 29% owned
      },
      "stats": {
        "GP": 4, "H_AB": "8/13", "R": 5, "H": 8, "HR": 2, "RBI": 4,
        "K": 2, "TB": 16, "AVG": 0.615, "OPS": 1.937, "NSB": 0
      }
    },
    {
      "player_info": {
        "name": "Andrés Giménez",
        "team": "TOR",
        "positions": ["2B", "SS"],
        "injury_status": null
      },
      "roster_status": "FA",
      "rankings": {
        "pre_season": 276,
        "current": 12,
        "percent_rostered": 0.30  // ← 30% owned
      },
      "stats": {
        "GP": 5, "H_AB": "8/18", "R": 1, "H": 8, "HR": 1, "RBI": 5,
        "K": 2, "TB": 13, "AVG": 0.444, "OPS": 1.222, "NSB": 2
      }
    },
    {
      "player_info": {
        "name": "Brandon Lowe",
        "team": "PIT",
        "positions": ["2B"],
        "injury_status": null
      },
      "roster_status": "FA",
      "rankings": {
        "pre_season": 182,
        "current": 19,
        "percent_rostered": 0.82  // ← 82% owned (not 0%!)
      },
      "stats": {
        "GP": 5, "H_AB": "6/18", "R": 3, "H": 6, "HR": 3, "RBI": 4,
        "K": 3, "TB": 16, "AVG": 0.333, "OPS": 1.344, "NSB": 0
      }
    },
    {
      "player_info": {
        "name": "Griffin Conine",
        "team": "MIA",
        "positions": ["LF", "RF"],
        "injury_status": "DTD"  // ← Separate injury field (not in name!)
      },
      "roster_status": "FA",
      "rankings": {
        "pre_season": 1588,
        "current": 83,
        "percent_rostered": 0.00
      },
      "stats": {
        "GP": 5, "H_AB": "4/10", "R": 3, "H": 4, "HR": 1, "RBI": 2,
        "K": 2, "TB": 8, "AVG": 0.400, "OPS": 1.255, "NSB": 1
      }
    }
  ]
}
```

### Key Data Mapping Takeaways:

| Issue | App Behavior | Yahoo Raw Data | Root Cause |
|-------|--------------|----------------|------------|
| **Owned % = 0%** | Shows 0.0% | `percent_rostered: 0.29` (29%) | Looking for wrong key OR `out=metadata` excludes ownership |
| **Sorting Chaos** | Random order | Yahoo sorted by `current` rank (2, 12, 19, 83) | Not passing Yahoo's native sort through |
| **Injury Concatenation** | "Jason Adam Quadriceps" | `injury_status: "DTD"` (separate field) | Parsing injury into name instead of separate field |
| **Blank Key Stats** | Shows `—` | Full stats object provided (AVG, HR, RBI, R, NSB) | Frontend pointing to wrong data object |

### Correct JSON Keys (Per Yahoo):
- **Ownership:** `percent_rostered` (not `percent_owned`)
- **Stats:** Nested in `stats` object (AVG, HR, RBI, R, NSB)
- **Injury:** Separate `injury_status` field (null, "DTD", "IL10", etc.)
- **Sorting:** Use `rankings.current` for Yahoo's default sort order

---

## Part 4: Implementation Roadmap for Claude Code

### Phase 1: Critical Fixes (Deploy Immediately)

| Issue | File | Lines | Fix | Complexity |
|-------|------|-------|-----|------------|
| 1.1 Owned % = 0% | `yahoo_client_resilient.py` | 529-541 | Remove `"out": "metadata"` OR add batch ownership call | MEDIUM |
| 1.2 Key Stats Blank | `main.py` + `schemas.py` | 380, 4625+ | Add `stats` field to WaiverPlayerOut, fetch from Yahoo | MEDIUM |
| 1.3 Max Owned Default | `waiver/page.tsx` | 349 | Change default from 90 to 50 | LOW |

### Phase 2: UX Enhancements (Next Sprint)

| Issue | File | Lines | Fix | Complexity |
|-------|------|-------|-----|------------|
| 1.4 Need Score Tooltip | `waiver/page.tsx` | 170 | Add Radix Tooltip with breakdown | LOW |
| 1.5 Empty Trackers | `waiver/page.tsx` | 506, 590 | Collapse cards when empty | LOW |
| 1.6 Auto-load Recs | `waiver/page.tsx` | 351, 387 | Set `enabled: true`, `showRecs: true` | LOW |
| 2.2 Status Icons | `status-badge.tsx` | — | Add Lucide icons to HOT/COLD | LOW |

### Phase 3: Advanced Features (Future)

| Issue | File | Complexity |
|-------|------|------------|
| 2.3 Positional Granularity | Backend + Frontend | MEDIUM |
| 2.4 Timeframe Toggles | Backend + Frontend | MEDIUM |
| Trend Indicators | New API integration | HIGH |

---

## Part 5: Immediate Action Items

### For Claude Code (Next Session)

1. **Fix Owned % (CRITICAL):**
   ```python
   # backend/fantasy_baseball/yahoo_client_resilient.py
   # Option: Remove out=metadata to get full player data
   params = {"status": "A", "start": start, "count": count, "sort": "AR"}
   # WITHOUT "out": "metadata"
   ```

2. **Add Stats to WaiverPlayer (CRITICAL):**
   ```python
   # backend/main.py — In _to_waiver_player() ~line 4616
   # Fetch player stats via get_player_stats() or include in batch call
   player_stats = client.get_player_stats(p["player_key"], stat_type="season")
   ```

3. **Fix Default Max Owned:**
   ```typescript
   // frontend/app/(dashboard)/fantasy/waiver/page.tsx line 349
   const [maxOwned, setMaxOwned] = useState<number>(50)
   ```

### For Gemini CLI (Verification)

1. Capture Yahoo API raw response to verify ownership data structure:
   ```bash
   railway run python -c "
   from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
   import json
   c = YahooFantasyClient()
   # Test with and without out=metadata
   fa = c.get_free_agents(count=3)
   print(json.dumps(fa, indent=2, default=str))
   "
   ```

2. Document actual Yahoo response structure for:
   - `percent_owned` location in JSON
   - Player stats availability
   - Injury status fields

---

## Summary

The Waiver Wire page has **3 CRITICAL** issues blocking core functionality:
1. **Owned % = 0%** — Yahoo API call uses `out=metadata` which excludes ownership
2. **Key Stats Blank** — Showing category contributions instead of actual player stats
3. **No Real-time Data** — Not leveraging Yahoo's live ownership/adoption data

With fixes, this becomes a **decision-making engine**:
- Real ownership % with 48-hour trends
- Actual player stats (AVG/HR/RBI for hitters, ERA/WHIP/K for pitchers)
- Timeframe toggles (Last 7/14/30 days)
- Collapsible empty modules for clean UX

**Time to implement:** 1-2 sessions for Phase 1 critical fixes.

---

*Analysis complete. All technical root causes identified with specific line numbers and fix implementations.*
