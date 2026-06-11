# Wave 6 Frontend Polish Report

**Date:** 2026-05-25  
**Branch:** `agent/kimi/wave6-frontend`  
**Agent:** Kimi CLI  
**Scope:** Four dashboard/waiver polish features

---

## TASK 1 — Waiver Priority Display in Budget Panel

**File:** `frontend/components/dashboard/budget-panel.tsx`, `frontend/lib/types.ts`

### Changes
- Extended `BudgetData` with:
  ```ts
  waiver_priority?: number
  waiver_total?: number
  waiver_recommendation?: string | null
  ```
- Added a **Waiver Priority** section to `BudgetPanel`:
  - Displays as `7 / 10` with a horizontal progress bar
  - Color-coded by priority tier:
    - **Green** (`status-safe`): priority 1–3 (high)
    - **Yellow** (`status-bubble`): priority 4–6 (moderate)
    - **Red** (`status-lost`): priority 7–10 (low)
  - Bar width is inverted — higher priority position = more bar filled
  - Shows `waiver_recommendation` text below the bar when present
- Section only renders when both `waiver_priority` and `waiver_total` are available

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## TASK 2 — Data Staleness Warning Banner

**File:** `frontend/app/(dashboard)/dashboard/page.tsx`, `frontend/lib/types.ts`

### Changes
- Extended `DashboardData` with:
  ```ts
  last_sync?: string | null
  stale_warning?: string | null
  has_mlb_games_today?: boolean
  ```
- Added a **dismissible yellow banner** at the top of the dashboard:
  - Message: `⚠️ Data may be stale — last updated [X hours ago]. Starting lineups may have changed.`
  - Uses `status-bubble` (amber/yellow) styling for consistency with warning patterns
  - Includes an **X dismiss button** that hides the banner for the session
- Added `formatRelativeTime` helper:
  - `just now` (< 1 min)
  - `45m ago` (< 1 hour)
  - `2h ago` (< 24 hours)
  - `1d ago` (≥ 24 hours)
- Banner only shows when:
  - `stale_warning` is present from the backend
  - `has_mlb_games_today` is not explicitly `false`
  - User has not dismissed it

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## TASK 3 — Two-Start Pitcher Badge

**Files:** `frontend/app/(dashboard)/dashboard/page.tsx`, `frontend/app/(dashboard)/war-room/waiver/page.tsx`, `frontend/lib/types.ts`

### Changes
- Extended `WaiverTarget` with `starts_this_week?: number` (dashboard waiver cards)
- `WaiverAvailablePlayer` already had `starts_this_week?: number`
- Added green **"2-Start"** badge:
  - **Dashboard** (`WaiverTargetsCard`): badge appears inline next to the player name, before team/positions
  - **Waiver page** (`PlayerRow`): badge appears in the identity row, between the tier badge and the Hot/Cold badge
  - Styled with `bg-status-safe/10 text-status-safe border border-status-safe/30`
  - Only renders when `starts_this_week >= 2`

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## TASK 4 — Lineup Optimization Gaps Display

**File:** `frontend/app/(dashboard)/dashboard/page.tsx`

### Changes
- Split `LineupGapsCard` into **two sections**:
  1. **Critical / Warning / Info gaps** — rendered first with existing red/amber/gray dot styling
  2. **Optimization gaps** — rendered below in a separate amber section
- Optimization section features:
  - Section header: `Sub-Optimal Placement` in `accent-gold`
  - Top border separator when regular gaps exist above
  - Gold dot (`bg-accent-gold`) for each optimization gap
  - `Optimize` badge on each gap row
  - Full `gap.message` text rendered (includes player names, scores, and slot recommendations)

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## Build Summary

```
▲ Next.js 15.5.13
✓ Compiled successfully in 5.6s
✓ Generating static pages (25/25)
✓ Finalizing page optimization ...
```

Modified routes:
- `/dashboard` — 5.84 kB (↑ from 5.33 kB)
- `/war-room/waiver` — 8.81 kB (↑ from 8.78 kB)

Only pre-existing `<img>` tag warnings remain.

---

## Files Modified

1. `frontend/lib/types.ts`
   - `BudgetData`: added `waiver_priority`, `waiver_total`, `waiver_recommendation`
   - `DashboardData`: added `last_sync`, `stale_warning`, `has_mlb_games_today`
   - `WaiverTarget`: added `starts_this_week`

2. `frontend/components/dashboard/budget-panel.tsx`
   - Added Waiver Priority section with color-coded bar and recommendation text

3. `frontend/app/(dashboard)/dashboard/page.tsx`
   - Added `useState` import + `X` icon import
   - Added `formatRelativeTime` helper
   - Added dismissible stale-data banner
   - Restructured `LineupGapsCard` into two sections (regular + optimization)
   - Added 2-Start badge to `WaiverTargetsCard`

4. `frontend/app/(dashboard)/war-room/waiver/page.tsx`
   - Added 2-Start badge to `PlayerRow`
