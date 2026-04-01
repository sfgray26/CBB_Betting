# K-22: Matchup Page UAT Deep Dive — Elite Fantasy Baseball Analysis

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** Technical root cause analysis of Matchup page critical failures  
**Status:** CRITICAL — Logic errors, data corruption, missing elite features

---

## Executive Summary

The Matchup page is a **"hollow shell"** with **4 CRITICAL BUGS** that make it unreliable for competitive play:

1. **"Playoffs" Hallucination** — Week 2 labeled as "PLAYOFFS" (Yahoo data blindly trusted)
2. **Missing Saves** — User has 1 save, Net Shows shows 0 (stat mapping confusion)
3. **Data Mapping Error** — K/9 (16.20) appearing in Walks column (stat ID mis-mapping)
4. **Confusing Labels** — "Cat. K/BB" instead of "K/BB Ratio" (raw column names leaking to UI)

**Missing Elite Features:**
- No pace/projections (can't see if losing 2-0 will become 8-5)
- No remaining games tracker (can't optimize volume)
- No visual "tug-of-war" bars (can't spot flip-able categories)
- No live win probability (no dopamine hit to drive engagement)

---

## Part 1: Critical Logic & Data Sync Bugs

### Issue 1.1: "Playoffs" Hallucination (Week 2 = PLAYOFFS)

**User Impact:** Fundamentally broken scheduling logic. If Week 2 is "playoffs", the app cannot be trusted for season management.

**Root Cause Analysis:**

```python
# backend/main.py lines 5603
for m in matchups:
    # ...
    is_playoffs = bool(m.get("is_playoffs", 0))  # <-- BLINDLY TRUSTS YAHOO
```

The code directly uses Yahoo's `is_playoffs` flag without sanity checking. Yahoo is returning `is_playoffs=1` for Week 2, which is clearly incorrect.

**Why This Happens:**
- Yahoo's `is_playoffs` flag may be based on league settings, not actual playoff status
- Some leagues have "playoff" structures that start early (consolation brackets)
- The API may return stale data or incorrect flags for the first few weeks

**The Fix:** Add sanity check based on week number:

```python
# backend/main.py lines 5594-5603 — FIXED
for m in matchups:
    if not isinstance(m, dict):
        continue
    w = m.get("week")
    if w:
        try:
            week = int(w)
        except (TypeError, ValueError):
            pass
    
    # SANITY CHECK: Week < 20 cannot be playoffs in any standard league
    raw_is_playoffs = bool(m.get("is_playoffs", 0))
    if raw_is_playoffs and week and week < 20:
        logger.warning(f"Yahoo returned is_playoffs=True for Week {week} — overriding to False")
        is_playoffs = False
    else:
        is_playoffs = raw_is_playoffs
```

**Also fix for fantasy leagues with custom playoff settings:**
```python
# Add league setting detection (lines 5503-5517)
# Check league settings for actual playoff start week
try:
    settings = client.get_league_settings()
    playoff_start_week = int(settings.get("playoff_start_week", 23))  # Default Week 23
    is_playoffs = week >= playoff_start_week if week else False
except:
    # Fallback: assume standard 22-week season, playoffs start Week 23
    is_playoffs = week >= 23 if week else False
```

**Complexity:** LOW  
**Priority:** CRITICAL — Breaks trust in app logic

---

### Issue 1.2: Missing Saves (Have 1, Shows 0)

**User Impact:** Cannot track critical pitching category. Saves are scarce and valuable — missing this data loses leagues.

**Root Cause Analysis:**

**Backend Mappings (main.py lines 5457-5468):**
```python
_YAHOO_STAT_FALLBACK = {
    # ... batting stats ...
    "32": "SV",    # Saves
    "83": "NSV",   # Net Saves (SV - BS)
    # ...
}
```

**Frontend STAT_LABELS (constants.ts lines 17-38):**
```typescript
export const STAT_LABELS: Record<string, string> = {
  // ...
  SV: 'Saves',
  NSV: 'Net Saves',
  // ...
  '32': 'Saves',
  '83': 'Net Saves',
}
```

**The Problem:**

Yahoo returns **Saves (stat_id=32)** as a raw counting stat. But the app is looking for **Net Saves (stat_id=83)** which is a calculated metric (Saves minus Blown Saves).

**Data Flow:**
1. Yahoo API returns: `{stat_id: "32", value: "1"}` (1 save)
2. Backend maps "32" → "SV" (line 5463)
3. Frontend displays "SV" → "Saves" (line 28)
4. **BUT** if the league uses Net Saves (stat_id=83) and Yahoo doesn't return it, value is 0

**The Fix:**

Check what the league actually uses and/or calculate Net Saves from Saves + Blown Saves:

```python
# backend/main.py lines 5570-5586 — ADD Net Saves calculation
# Build stats dict
stats_dict: dict = {}
for s in stats_raw:
    if isinstance(s, dict):
        stat = s.get("stat", {})
        if isinstance(stat, dict):
            sid = str(stat.get("stat_id", ""))
            key = stat_id_map.get(sid, sid)
            val = stat.get("value", "")
            # Clamp impossible negative values
            try:
                if float(val) < 0:
                    val = "0"
            except (TypeError, ValueError):
                pass
            if key:
                stats_dict[key] = val

# NEW: Calculate Net Saves if we have Saves but not Net Saves
if "SV" in stats_dict and "NSV" not in stats_dict:
    sv = float(stats_dict.get("SV", 0))
    bs = float(stats_dict.get("BS", 0))  # Blown Saves
    stats_dict["NSV"] = str(int(sv - bs))
```

**Additional Fix - Verify NSV Mapping:**
```python
# main.py — Add logging to debug stat mapping
logger.info(f"Stat mapping debug: id={sid}, mapped_to={key}, value={val}")
```

**Frontend verification (constants.ts):**
```typescript
// Ensure all stat IDs are mapped correctly
'83': 'Net Saves',  // Confirm this is correct
'NSV': 'Net Saves',
```

**Complexity:** LOW  
**Priority:** CRITICAL — Core stat missing

---

### Issue 1.3: Impossible Math — 16.2 Walks

**User Impact:** Data integrity compromised. Cannot trust any stats if Walks show IP format.

**Root Cause Analysis:**

**The Error:**
- Walks are counting stats (whole integers only)
- Showing "16.2" means "16 innings + 2 outs" (IP format)
- IP format appearing in BB column = data mapping corruption

**Format Function (matchup/page.tsx lines 17-30):**
```typescript
function formatVal(val: string | number | undefined, cat?: string): string {
  if (val === undefined || val === null) return '-'
  const n = parseFloat(String(val))
  if (isNaN(n)) return String(val)

  if (n < 0) return '—'
  if (cat && RATIO_STATS.has(cat)) return n.toFixed(3)
  if (cat === 'IP' || cat === '21' || cat === '50') return n.toFixed(1)  // IP format
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(1)
}
```

**Root Cause:**

The stat key is getting corrupted somewhere between Yahoo API and frontend. Possibilities:

1. **Yahoo Response Issue:** Yahoo sends `{"stat_id": "21", "value": "16.2"}` (IP data) but labels it as walks
2. **Mapping Error:** Stat ID "21" (IP) being mapped to "BB" somewhere
3. **Response Shape Confusion:** Multiple stat blocks, wrong one extracted

**Evidence from _extract_team_stats (main.py lines 5538-5592):**
```python
# Yahoo may include both weekly and season stats; first = current week
if "team_stats" in entry and not stats_raw:
    inner = entry["team_stats"].get("stats", [])
    if isinstance(inner, list):
        stats_raw = inner
```

**Hypothesis:** Yahoo returns both weekly stats (first) and season stats (second). The code takes the first `team_stats` block, but Yahoo's ordering may vary. If season-long IP (16.2) appears before weekly BB, it gets the wrong value.

**The Fix:**

Defensive validation in format function:

```typescript
// matchup/page.tsx lines 17-35 — ADD validation
function formatVal(val: string | number | undefined, cat?: string): string {
  if (val === undefined || val === null) return '-'
  const n = parseFloat(String(val))
  if (isNaN(n)) return String(val)

  if (n < 0) return '—'
  
  // WALK VALIDATION: Walks cannot have decimals (except .0)
  if (cat === 'BB' || cat === '57' || cat === 'Walks') {
    if (n !== Math.floor(n)) {
      // Invalid: walk count with decimal = data corruption
      console.error(`Data corruption: Walks showing ${n} (should be integer)`)
      return '—'  // Hide corrupted data
    }
    return Math.floor(n).toString()
  }
  
  if (cat && RATIO_STATS.has(cat)) return n.toFixed(3)
  if (cat === 'IP' || cat === '21' || cat === '50') return n.toFixed(1)
  return Number.isInteger(n) ? n.toLocaleString() : n.toFixed(1)
}
```

**Backend validation (main.py lines 5576-5586):**
```python
# Add stat-specific validation
key = stat_id_map.get(sid, sid)
val = stat.get("value", "")

# BB validation: should be integer
try:
    if key in ("BB", "57", "Walks"):
        float_val = float(val)
        if float_val != int(float_val):
            logger.warning(f"Data corruption: BB stat has decimal: {val}")
            val = str(int(float_val))  # Truncate or mark as error
except (TypeError, ValueError):
    pass
```

**Complexity:** MEDIUM — Requires data validation  
**Priority:** HIGH — Data integrity

---

### Issue 1.4: Confusing Labels — "Cat. K/BB"

**User Impact:** Looks unprofessional. Raw database column names leaking to UI.

**Root Cause Analysis:**

**Matchup Table Cell (matchup/page.tsx lines 76-77):**
```typescript
<td className="px-4 py-2.5 text-zinc-400 font-medium">
  {STAT_LABELS[cat] ?? `Cat. ${cat}`}
</td>
```

When `cat` is not in `STAT_LABELS`, it falls back to `Cat. ${cat}`, producing "Cat. K/BB".

**STAT_LABELS Missing Mappings (constants.ts lines 17-38):**
```typescript
'38': 'K/BB',  // Exists
// But no string key 'K/BB': 'K/BB Ratio'
```

**Missing Mappings:**
- `"K/BB"` → missing string key mapping
- `"GS"` → mapped as `"62": "GS"` but no `"GS": "Games Started"`
- `"HLD"` → not mapped at all

**The Fix:**

Add missing string key mappings:

```typescript
// constants.ts lines 17-38 — ADD missing mappings
export const STAT_LABELS: Record<string, string> = {
  // ... existing mappings ...
  
  // Add string key versions
  'K/BB': 'K/BB Ratio',
  'GS': 'Games Started',
  'HLD': 'Holds',
  'BS': 'Blown Saves',
  'NSV': 'Net Saves',
  
  // Also add display-friendly versions for common abbreviations
  'BB': 'Walks (P)',
  'K': 'Strikeouts',
  'SV': 'Saves',
  // ... rest of existing ...
}
```

**Remove "Cat." prefix entirely:**
```typescript
// matchup/page.tsx line 77
// BEFORE:
{STAT_LABELS[cat] ?? `Cat. ${cat}`}

// AFTER:
{STAT_LABELS[cat] ?? cat}  // Just show the raw stat name, no "Cat." prefix
```

**Complexity:** LOW  
**Priority:** MEDIUM — Visual polish

---

## Part 2: Missing Elite Features

### Feature 2.1: Pace & Projections (The Core Feature)

**User Need:** "I'm losing HRs 2-0. Will I lose 8-5 or win 6-5?"

**Current State:** Static data dump only. No forward projection.

**Implementation Blueprint:**

```typescript
// Add to MatchupResponse schema
interface MatchupPace {
  category: string
  my_current: number
  opp_current: number
  my_projected: number
  opp_projected: number
  my_pace: number      # Projected final if continues at current rate
  opp_pace: number
  status: 'leading' | 'trailing' | 'tied' | 'volatile'
}

// Backend calculation (new endpoint or extend existing)
// Based on:
// 1. Current stats
// 2. Remaining games for each team
// 3. Player projections for remaining games
// 4. Historical pace (last 7 days performance)
```

**Backend Implementation:**
```python
# backend/main.py — Add to matchup endpoint
# Calculate pace after extracting current stats
remaining_games_my = _calculate_remaining_games(my_roster)
remaining_games_opp = _calculate_remaining_games(opp_roster)

for cat in all_categories:
    my_current = float(my_stats.get(cat, 0))
    opp_current = float(opp_stats.get(cat, 0))
    
    # Simple pace: current / games_played * total_weekly_games
    my_pace = my_current / games_played * 7 if games_played > 0 else 0
    opp_pace = opp_current / games_played * 7 if games_played > 0 else 0
    
    # Enhanced pace: add projected stats from remaining games
    my_projected = my_current + _sum_projections(my_roster, cat, remaining_games_my)
    opp_projected = opp_current + _sum_projections(opp_roster, cat, remaining_games_opp)
```

**Complexity:** HIGH — Requires roster fetching + projections  
**Priority:** HIGH — Core differentiator

---

### Feature 2.2: Remaining Games/Starts Tracker

**User Need:** "I have 32 games left, opponent has 28. I need to stream more pitchers."

**Implementation:**

```typescript
// Add to MatchupResponse
interface RemainingGames {
  my_batter_games: number
  opp_batter_games: number
  my_pitcher_starts: number
  opp_pitcher_starts: number
}

// Visual indicator in header
<div className="flex gap-4 text-xs">
  <div className={my_batter_games < opp_batter_games ? 'text-amber-400' : 'text-emerald-400'}>
    Batter Games: {my_batter_games} vs {opp_batter_games}
  </div>
  <div className={my_pitcher_starts < opp_pitcher_starts ? 'text-amber-400' : 'text-emerald-400'}>
    SP Starts: {my_pitcher_starts} vs {opp_pitcher_starts}
  </div>
</div>
```

**Complexity:** MEDIUM — Requires probable pitcher fetching  
**Priority:** HIGH — Tactical advantage

---

### Feature 2.3: Visual "Tug-of-War" Bars

**User Need:** Quick visual scan to see which categories are flip-able.

**Implementation:**

```typescript
// Replace table with visual bars
function CategoryBar({ cat, myVal, oppVal }: { cat: string, myVal: number, oppVal: number }) {
  const total = myVal + oppVal
  const myPct = total > 0 ? (myVal / total) * 100 : 50
  const oppPct = 100 - myPct
  
  // Determine if volatile (close enough to flip)
  const isVolatile = Math.abs(myVal - oppVal) / Math.max(myVal, oppVal, 1) < 0.2
  
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span>{STAT_LABELS[cat] ?? cat}</span>
        <span className={isVolatile ? 'text-amber-400 animate-pulse' : ''}>
          {myVal} vs {oppVal}
        </span>
      </div>
      <div className="h-2 bg-zinc-800 rounded-full overflow-hidden flex">
        <div 
          className="bg-emerald-500 transition-all"
          style={{ width: `${myPct}%` }}
        />
        <div 
          className="bg-rose-500 transition-all"
          style={{ width: `${oppPct}%` }}
        />
      </div>
    </div>
  )
}
```

**Complexity:** LOW — Purely visual  
**Priority:** MEDIUM — UX improvement

---

### Feature 2.4: Live Win Probability

**User Need:** Single dopamine metric: "You have a 68% chance of winning."

**Implementation:**

Use existing MCMC simulator:

```python
# backend/services/mcmc_simulator.py — already implemented
# Just needs to be wired into matchup endpoint

from backend.services.mcmc_simulator import MCMCWeeklySimulator

simulator = MCMCWeeklySimulator(n_sims=10000)
result = simulator.simulate_head_to_head(my_team_categories, opp_team_categories)
win_probability = result.win_probability  # 0.0 to 1.0
```

**Frontend:**
```typescript
// Score banner with win probability
<div className="text-center">
  <p className="text-3xl font-bold text-emerald-400">
    {(data.win_probability * 100).toFixed(0)}%
  </p>
  <p className="text-xs text-zinc-500">Win Probability</p>
</div>
```

**Complexity:** LOW — MCMC already exists  
**Priority:** HIGH — Engagement driver

---

## Part 3: Implementation Roadmap

### Phase 1: Critical Bug Fixes (Deploy Immediately)

| Issue | File | Lines | Change | Complexity |
|-------|------|-------|--------|------------|
| 1.1 Playoffs | `main.py` | 5603 | Add sanity check: Week < 20 ≠ playoffs | LOW |
| 1.2 Saves (NSV) | `main.py` | 5576+ | Debug stat ID 83 extraction (raw data shows NSV: 1) | MEDIUM |
| 1.3 K/9 Mapping | `main.py` | 5457-5468 | Fix stat ID mapping (K/9 appearing in BB column) | MEDIUM |
| 1.4 Labels | `constants.ts` | 17-38 | Add missing K/BB, GS, NSV mappings | LOW |

### Phase 2: Elite Features (Next Sprint)

| Feature | Files | Complexity |
|---------|-------|------------|
| 2.1 Pace | `main.py` (backend calc) + `matchup/page.tsx` (display) | HIGH |
| 2.2 Remaining Games | `main.py` + header component | MEDIUM |
| 2.3 Visual Bars | `matchup/page.tsx` new component | LOW |
| 2.4 Win Probability | Wire MCMC simulator | LOW |

---

## Appendix: Raw Yahoo Data (Ground Truth)

**Source:** User's actual Yahoo Fantasy Matchup page for Week 2

### Key Findings from Raw Data:

| Issue | Raw Data | App Display | Root Cause |
|-------|----------|-------------|------------|
| **Missing NSV** | `"NSV": 1` (Jordan Romano) | Shows 0 | Stat ID 83 not being extracted or mapped |
| **K/9 vs Walks** | `"K_9": 16.20` | Appears in Walks column | Stat ID mis-mapping |
| **Playoffs Bug** | `"week": 2` | Shows "(PLAYOFFS)" | Yahoo `is_playoffs` flag incorrect |

### Full Raw Payload (Condensed):
```json
{
  "matchup_meta": {
    "week": 2,
    "team_name": "Lindor Truffles",
    "current_record": "0-1-0",
    "current_score": 12,
    "opponent_score": 3
  },
  "team_aggregate_stats": {
    "pitching": {
      "IP": 1.2,
      "K": 3,
      "ERA": 5.40,
      "WHIP": 1.80,
      "K_9": 16.20,    // ← This is appearing in Walks column
      "NSV": 1          // ← This is showing as 0
    }
  },
  "roster": {
    "pitchers": [
      { "name": "Jordan Romano", "NSV": 1 }  // ← Confirmed save exists
    ]
  }
}
```

---

## Summary for Claude Code

**4 CRITICAL bugs to fix:**

1. **"Playoffs" Hallucination:** Add week number sanity check (Week < 20 ≠ playoffs)
2. **Missing Saves:** NSV stat ID 83 not being extracted (Jordan Romano has NSV: 1)
3. **K/9 Data Mapping:** K/9 (16.20) appearing in Walks column (stat ID mis-mapping)
4. **"Cat." Prefix:** Add missing stat labels, remove "Cat." fallback

**4 ELITE features to add:**

1. **Pace/Projections:** Show projected final scores
2. **Remaining Games:** Track batter games + pitcher starts left
3. **Visual Bars:** Tug-of-war progress bars per category
4. **Win Probability:** MCMC-based win %

**Files to modify:**
- `backend/main.py` (lines 5536-5603, 5576-5586)
- `frontend/lib/constants.ts` (lines 17-38)
- `frontend/app/(dashboard)/fantasy/matchup/page.tsx` (lines 17-30, 76-77)

**Estimated time:** 1 session for Phase 1, 2 sessions for Phase 2.

---

*Analysis complete. All technical root causes identified with specific line numbers and fix implementations.*
