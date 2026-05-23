# Dashboard & Waiver UI Update — Wave 2 Report

**Date:** 2026-05-22  
**Branch:** `agent/kimi/wave2-dashboard-waiver-ui`  
**Agent:** Kimi CLI  
**Scope:** Need Score tooltips, Pitcher card fallback, Visual tier grading

---

## TASK A — Need Score Contextual Tooltip

**Files:** `frontend/app/(dashboard)/dashboard/page.tsx`, `frontend/app/(dashboard)/war-room/waiver/page.tsx`, `frontend/lib/types.ts`

### Problem
The Need Score (e.g., 22.18) had no contextual anchor. Users couldn't tell if 25 was perfect, if it was a Z-score, or what the scale meant.

### Changes

**1. Extended `WaiverTarget` type** (dashboard waiver cards):
```ts
need_score?: number
category_contributions?: Record<string, number>
```

**2. Built reusable tooltip content component** (`NeedScoreTooltipContent`):
- Score value + tier label (Premium / Strong / Standard)
- Category breakdown: e.g., `HR fit: +4.2 | SB fit: +2.1 | K/9 fit: +3.8`
- Scale context: `Scores range 0-30. >20 = premium target`

**3. Dashboard `WaiverTargetsCard`:**
- Added `Need: 22.18` inline text with dotted underline (hover trigger)
- Wrapped with the existing `<Tooltip>` component from `@/components/shared/tooltip`
- Shows breakdown from `category_contributions` when available

**4. Waiver page `NeedBar`:**
- The numeric score value (`{score.toFixed(2)}`) is now wrapped in a `<Tooltip>`
- Passed `contributions` prop through to the tooltip content
- Cursor changes to `cursor-help` with dotted underline to indicate interactivity

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## TASK B — Pitcher Card Null Guard + Fallback

**File:** `frontend/app/(dashboard)/dashboard/page.tsx`

### Problem
`DashboardData` included `probable_pitchers: ProbablePitcherInfo[]` but it was never rendered in the UI. When the backend returned an empty array, the dashboard simply showed nothing — no indication that pitcher data was missing.

### Changes

**1. Added `ProbablePitchersCard` component:**
- Always renders (unlike Two-Start Pitchers which hides when empty)
- When data exists: shows a grid of probable pitchers with matchup quality badges, stream scores, and reasons
- When empty: shows a graceful fallback instead of hiding the section

**2. Empty-state fallback:**
```
⚠️ Pitcher data temporarily unavailable.
Your pitching slots are still active on Yahoo.
[Retry button]
```

**3. Retry button:**
- Calls `refetch()` from the dashboard `useQuery` hook
- Re-fetches the entire dashboard endpoint (same endpoint that serves pitcher data)

**4. No alarming counts:**
- The fallback message does NOT show "0 active pitcher slots filled" or any zero-count language
- It simply informs the user that data is temporarily unavailable

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## TASK C — Visual Tier Grading for Waiver Targets

**Files:** `frontend/app/(dashboard)/dashboard/page.tsx`, `frontend/app/(dashboard)/war-room/waiver/page.tsx`

### Problem
Josh Jung (premium, score ≥ 20) and Jake Bauers (fringe, score < 15) had identical visual weight. Users couldn't quickly distinguish premium targets from standard ones.

### Changes

**1. Added `NeedScoreTierBadge` component:**
| Score Range | Badge | Style |
|-------------|-------|-------|
| ≥ 20 | **PREMIUM** | Gold (`bg-accent-gold/10 text-accent-gold`) |
| 15–19 | **STRONG** | Silver/gray (`bg-text-muted/10 text-text-secondary`) |
| < 15 | (none) | Standard weight |

**2. Dashboard `WaiverTargetsCard`:**
- `PREMIUM` / `STRONG` badge appears inline next to the player name
- Existing tier badge (`must_add` / `strong_add` / `streamer`) is preserved on the right

**3. Waiver page `PlayerRow`:**
- Same tier badge logic added to the identity row
- Badge sits between the player name and the `HotColdBadge`

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## Build Summary

```
▲ Next.js 15.5.13
✓ Compiled successfully in 9.7s
✓ Generating static pages (25/25)
✓ Finalizing page optimization ...
```

Modified routes:
- `/dashboard` — 5.29 kB (↑ from 4.24 kB)
- `/war-room/waiver` — 8.74 kB (↑ from 8.41 kB)

Only pre-existing warnings remain (`<img>` tag usage in `preview/page.tsx` and `yahoo-roster-view.tsx`).

---

## Files Modified

1. `frontend/lib/types.ts`
   - Added `need_score` and `category_contributions` to `WaiverTarget`

2. `frontend/app/(dashboard)/dashboard/page.tsx`
   - Imported `Calendar`, `RefreshCw`, `cn`, `Tooltip`, `ProbablePitcherInfo`
   - Added `NeedScoreTooltipContent`, `NeedScoreTierBadge` helpers
   - Updated `WaiverTargetsCard` with need-score tooltip + tier badges
   - Added new `ProbablePitchersCard` with empty-state fallback and retry button
   - Rendered `ProbablePitchersCard` in the dashboard grid

3. `frontend/app/(dashboard)/war-room/waiver/page.tsx`
   - Imported `Tooltip`
   - Added `formatContributionKey`, `NeedScoreTooltipContent` helpers
   - Updated `NeedBar` to wrap score in interactive tooltip
   - Updated `PlayerRow` to render `PREMIUM` / `STRONG` tier badges

---

## Next Steps / Notes

- **Backend dependency:** The dashboard endpoint must begin populating `need_score` and `category_contributions` on `WaiverTarget` objects for the tooltip and tier badges to appear in the dashboard waiver card. The waiver page already receives these fields via `WaiverAvailablePlayer`.
- **Probable pitchers:** The `probable_pitchers` array in `DashboardData` is now surfaced in the UI. When empty, users see the fallback message instead of a missing section.
- **Tooltip pattern:** All tooltips use the existing `@/components/shared/tooltip` component (Radix UI based). The `Tooltip.Provider` is instantiated per-tooltip, following the existing codebase pattern.
