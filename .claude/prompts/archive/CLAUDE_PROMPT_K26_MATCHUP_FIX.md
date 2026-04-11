# ⚠️ ARCHIVED — K-26 Matchup Category Alignment Fix

> **Status:** ARCHIVED (April 11, 2026)  
> **Original Location:** `CLAUDE_PROMPT_K26_MATCHUP_FIX.md` (root)  
> **Archive Reason:** Task completed (K-26 implemented)  
> 
> This document has been archived as part of repository documentation cleanup.
> The matchup category alignment fix has been implemented and deployed.
> 
> **See Also:** 
> - `CLAUDE_PROMPTS_INDEX.md` for all active prompts
> - `reports/2026-04-01-matchup-category-alignment.md` for implementation details
> 
> ---
> *Original content preserved below for historical reference.*

# Claude Handoff Prompt: K-26 Matchup Category Alignment Fix

**Priority:** CRITICAL — Matchup scores are currently wrong  
**Est. Time:** 45-60 minutes  
**Files:** 2 files (frontend only)

---

## The Problem

The Matchup page (`/fantasy/matchup`) shows incorrect scores (e.g., 9-6) that don't match Yahoo's official standings. Three issues:

1. **H/AB and IP are counted as scoring categories** — they're display-only reference stats
2. **Batter K (strikeouts) awards win to higher value** — should be LOWER is better (fewer strikeouts = good for batters)
3. **Categories display in random order** — should be Batters section then Pitchers section

---

## Your Task

### Step 1: Update `frontend/lib/constants.ts`

Add a new configuration object that defines category order, section, and win direction:

```typescript
// Add after LOWER_IS_BETTER (line 46)

/** 
 * Matchup category configuration.
 * - section: 'batting' or 'pitching' for UI grouping
 * - scoring: whether this category counts toward W/L total
 * - lowerIsBetter: true for ERA, WHIP, Batter K, Losses
 * - order: display order within section
 */
export const MATCHUP_CATEGORY_CONFIG: Record<string, {
  label: string
  section: 'batting' | 'pitching'
  scoring: boolean
  lowerIsBetter: boolean
  order: number
}> = {
  // Batters (order 1-10)
  H_AB: { label: 'H/AB', section: 'batting', scoring: false, lowerIsBetter: false, order: 1 },
  R: { label: 'Runs', section: 'batting', scoring: true, lowerIsBetter: false, order: 2 },
  H: { label: 'Hits', section: 'batting', scoring: true, lowerIsBetter: false, order: 3 },
  HR: { label: 'Home Runs', section: 'batting', scoring: true, lowerIsBetter: false, order: 4 },
  RBI: { label: 'RBI', section: 'batting', scoring: true, lowerIsBetter: false, order: 5 },
  K: { label: 'Strikeouts', section: 'batting', scoring: true, lowerIsBetter: true, order: 6 },  // ⚠️ LOWER IS BETTER
  TB: { label: 'Total Bases', section: 'batting', scoring: true, lowerIsBetter: false, order: 7 },
  AVG: { label: 'Batting Avg', section: 'batting', scoring: true, lowerIsBetter: false, order: 8 },
  OPS: { label: 'OPS', section: 'batting', scoring: true, lowerIsBetter: false, order: 9 },
  NSB: { label: 'Net SB', section: 'batting', scoring: true, lowerIsBetter: false, order: 10 },
  
  // Pitchers (order 11-20)
  IP: { label: 'Innings Pitched', section: 'pitching', scoring: false, lowerIsBetter: false, order: 11 },
  W: { label: 'Wins', section: 'pitching', scoring: true, lowerIsBetter: false, order: 12 },
  L: { label: 'Losses', section: 'pitching', scoring: true, lowerIsBetter: true, order: 13 },
  // Note: HR allowed uses same key as HR batting; differentiate by section if needed
  ERA: { label: 'ERA', section: 'pitching', scoring: true, lowerIsBetter: true, order: 15 },
  WHIP: { label: 'WHIP', section: 'pitching', scoring: true, lowerIsBetter: true, order: 16 },
  QS: { label: 'Quality Starts', section: 'pitching', scoring: true, lowerIsBetter: false, order: 18 },
  NSV: { label: 'Net Saves', section: 'pitching', scoring: true, lowerIsBetter: false, order: 19 },
}

// Helper to get ordered categories
export function getOrderedCategories(): string[] {
  return Object.entries(MATCHUP_CATEGORY_CONFIG)
    .sort(([, a], [, b]) => a.order - b.order)
    .map(([key]) => key)
}

// Stats that are display-only (not counted in W/L total)
export const DISPLAY_ONLY_STATS = new Set(['H_AB', 'IP'])
```

### Step 2: Update `frontend/app/(dashboard)/fantasy/matchup/page.tsx`

**Change 1: Import the new config**
```typescript
// Line 9: Update import
import { STAT_LABELS, RATIO_STATS, LOWER_IS_BETTER, MATCHUP_CATEGORY_CONFIG, DISPLAY_ONLY_STATS } from '@/lib/constants'
```

**Change 2: Update `winClass` function to use config**
```typescript
// Lines 11-15: Replace winClass function
function winClass(cat: string, myVal: number, oppVal: number): string {
  if (myVal === oppVal) return 'text-zinc-400'
  
  const config = MATCHUP_CATEGORY_CONFIG[cat]
  const lowerIsBetter = config?.lowerIsBetter ?? LOWER_IS_BETTER.has(cat)
  
  const myWins = lowerIsBetter ? myVal < oppVal : myVal > oppVal
  return myWins ? 'text-emerald-400 font-semibold' : 'text-rose-400'
}
```

**Change 3: Update `MatchupTable` to use ordered categories**
```typescript
// Line 36-44: Replace MatchupTable component start
function MatchupTable({ data }: { data: MatchupResponse }) {
  // Get categories in proper order, filtering to only those present in data
  const orderedCats = Object.entries(MATCHUP_CATEGORY_CONFIG)
    .sort(([, a], [, b]) => a.order - b.order)
    .map(([key]) => key)
    .filter(cat => data.my_team.stats[cat] !== undefined || data.opponent.stats[cat] !== undefined)
  
  // Fallback to API order if config doesn't cover all stats
  const allCats = Object.keys(data.my_team.stats)
  const displayCats = orderedCats.length > 0 ? orderedCats : allCats
  
  if (displayCats.length === 0) {
    return (
      <p className="text-zinc-500 text-sm text-center py-8">
        No stats yet — all zeros at the start of the week.
      </p>
    )
  }
  
  // ... rest of component uses displayCats instead of allCats
```

**Change 4: Update table rendering to show section headers**
```typescript
// In the table body (around line 65-97), group by section:
<tbody className="divide-y divide-zinc-800/60">
  {/* Batting Section */}
  {displayCats.some(cat => MATCHUP_CATEGORY_CONFIG[cat]?.section === 'batting') && (
    <tr className="bg-zinc-900/30">
      <td colSpan={4} className="px-4 py-2 text-xs font-semibold text-amber-400 uppercase tracking-wider">
        Batters
      </td>
    </tr>
  )}
  {displayCats
    .filter(cat => MATCHUP_CATEGORY_CONFIG[cat]?.section === 'batting')
    .map((cat) => { /* existing row render */ })}
  
  {/* Pitching Section */}
  {displayCats.some(cat => MATCHUP_CATEGORY_CONFIG[cat]?.section === 'pitching') && (
    <tr className="bg-zinc-900/30">
      <td colSpan={4} className="px-4 py-2 text-xs font-semibold text-sky-400 uppercase tracking-wider">
        Pitchers
      </td>
    </tr>
  )}
  {displayCats
    .filter(cat => MATCHUP_CATEGORY_CONFIG[cat]?.section === 'pitching')
    .map((cat) => { /* existing row render */ })}
</tbody>
```

**Change 5: Update `ScoreBanner` to exclude display-only stats**
```typescript
// Lines 107-119: Replace ScoreBanner scoring logic
function ScoreBanner({ data }: { data: MatchupResponse }) {
  const allCats = Object.keys(data.my_team.stats)
  
  // FILTER: Only count scoring categories (exclude H/AB, IP, etc.)
  const scoringCats = allCats.filter(cat => 
    !DISPLAY_ONLY_STATS.has(cat) && MATCHUP_CATEGORY_CONFIG[cat]?.scoring !== false
  )
  
  let myWins = 0
  let oppWins = 0
  let ties = 0

  scoringCats.forEach((cat) => {
    const myVal = parseFloat(String(data.my_team.stats[cat] ?? 0))
    const oppVal = parseFloat(String(data.opponent.stats[cat] ?? 0))
    if (myVal === oppVal) { ties++; return }
    
    const config = MATCHUP_CATEGORY_CONFIG[cat]
    const lowerIsBetter = config?.lowerIsBetter ?? LOWER_IS_BETTER.has(cat)
    
    if (lowerIsBetter ? myVal < oppVal : myVal > oppVal) myWins++
    else oppWins++
  })
  
  // ... rest of component unchanged
```

### Step 3: Test & Verify

Run TypeScript check:
```bash
cd frontend && npx tsc --noEmit
```

Verify logic with these test cases:
- My team: 50 K, Opponent: 40 K → Opponent should WIN (lower is better for batters)
- My team: 3.50 ERA, Opponent: 4.00 ERA → I should WIN (lower is better for ERA)
- My team: 25 HR, Opponent: 20 HR → I should WIN (higher is better for HR)
- H/AB row should show but NOT count in score total

---

## Acceptance Criteria

- [ ] Matchup score (e.g., "7-5") matches Yahoo's official score exactly
- [ ] H/AB and IP display but don't count toward W/L total
- [ ] Batter K awards win to team with FEWER strikeouts
- [ ] Categories display in order: Batters section (H/AB, R, H, HR, RBI, K, TB, AVG, OPS, NSB) then Pitchers section (IP, W, L, ERA, WHIP, QS, NSV)
- [ ] Section headers "Batters" and "Pitchers" visible in table
- [ ] TypeScript passes (`npx tsc --noEmit`)

---

## Research Reference

Full spec: `reports/K26_MATCHUP_CATEGORY_ALIGNMENT_SPEC.md`

Key insight: Batter K is a NEGATIVE category (you want fewer strikeouts), while Pitcher K is POSITIVE (you want more strikeouts). The current code doesn't distinguish them.
