# Wave 5B Frontend — Need Score Tooltips, Pitcher Fallback, Tier Badges, Small Sample Warning

**Date:** 2026-05-22  
**Branch:** `agent/kimi/wave5b-frontend`  
**Agent:** Kimi CLI  
**Scope:** Complete Wave 5B frontend fixes (Tasks 1–4)

---

## Verification of Prior Work

Tasks 1–3 were already present on `stable/cbb-prod` from the previous Wave 2 session (`agent/kimi/wave2-dashboard-waiver-ui`). Verified via code inspection before implementing Task 4.

- ✅ **TASK 1 (Need Score Tooltip):** `NeedScoreTooltipContent` exists in both `dashboard/page.tsx` and `war-room/waiver/page.tsx`; `NeedBar` wraps score in `<Tooltip>`
- ✅ **TASK 2 (Pitcher Fallback):** `ProbablePitchersCard` renders on dashboard with empty-state message + retry button
- ✅ **TASK 3 (Tier Badges):** `NeedScoreTierBadge` renders `PREMIUM` (≥20) and `STRONG` (15–19) on both dashboard and waiver pages

---

## TASK 4 — Small Sample Warning Badge (NEW)

**Files:** `frontend/lib/types.ts`, `frontend/app/(dashboard)/dashboard/page.tsx`, `frontend/app/(dashboard)/war-room/waiver/page.tsx`

### Problem
The backend now returns `small_sample: true` for players with insufficient data (e.g., Jake Bauers false-positives). The UI had no way to surface this warning to users.

### Changes

**1. Extended types:**
```ts
// WaiverTarget (dashboard waiver card)
small_sample?: boolean

// WaiverAvailablePlayer (waiver page)
small_sample?: boolean
```

**2. Dashboard `WaiverTargetsCard`:**
- Added yellow `⚠️ Small Sample` badge inline with the Need Score display
- Styled with `bg-status-bubble/10 text-status-bubble border border-status-bubble/30` (project amber/yellow warning palette)

**3. Waiver page `PlayerRow`:**
- Added the same badge in the Match Score column header area, next to the "Match Score" label
- Maintains visual hierarchy — the warning sits above the `NeedBar` so users see it before interpreting the score

### Backward Compatibility
- When `small_sample` is absent or `false`, no badge renders (no visual regression)
- Badge only appears when the backend explicitly flags the player

---

## Build Summary

```
▲ Next.js 15.5.13
✓ Compiled successfully in 10.5s
✓ Generating static pages (25/25)
✓ Finalizing page optimization ...
```

- `/dashboard` — 5.33 kB
- `/war-room/waiver` — 8.78 kB

Only pre-existing `<img>` tag warnings remain.

---

## Files Modified

1. `frontend/lib/types.ts`
   - Added `small_sample?: boolean` to `WaiverTarget`
   - Added `small_sample?: boolean` to `WaiverAvailablePlayer`

2. `frontend/app/(dashboard)/dashboard/page.tsx`
   - Added small-sample badge render in `WaiverTargetsCard` need-score row

3. `frontend/app/(dashboard)/war-room/waiver/page.tsx`
   - Added small-sample badge render in `PlayerRow` match-score column header
