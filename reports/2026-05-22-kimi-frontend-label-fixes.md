# Frontend Label Fixes — Wave 1 Report

**Date:** 2026-05-22  
**Branch:** `agent/kimi/wave1-frontend-fixes`  
**Agent:** Kimi CLI  
**Scope:** 3 frontend bugs in war-room and dashboard pages

---

## FIX 1 — Preview page crash (P1, charAt TypeError)

**File:** `frontend/app/(dashboard)/war-room/preview/page.tsx`  
**Line:** 198

### Problem
`data.opponent_name.charAt(0)` crashed with a TypeError when `opponent_name` was `null` or `undefined` (e.g., when the matchup preview endpoint returned partial data or when no opponent was set for the upcoming week).

### Fix
Added a null-safe fallback:

```tsx
// BEFORE
<span className="text-lg font-bold text-text-muted">{data.opponent_name.charAt(0)}</span>

// AFTER
<span className="text-lg font-bold text-text-muted">{(data.opponent_name || '?').charAt(0)}</span>
```

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed
- ✅ No other unguarded `.charAt()` calls found in this file (only 1 occurrence).

---

## FIX 2 — '0% owned' label on roster players (P2)

**File:** `frontend/app/(dashboard)/war-room/roster/page.tsx`  
**Component:** `PlayerCard` (identity row)

### Problem
Every player on the user's roster displayed `0% owned` (or `— owned`). This value is the waiver-wire ownership percentage, which is meaningless for players already rostered and made star players like Juan Soto appear undrafted.

### Fix
Suppressed the ownership percentage field entirely on the roster page, since every player shown here is by definition on the user's team.

```tsx
// BEFORE
{player.ownership_pct != null ? (
  <span className="text-[10px] text-text-muted">{player.ownership_pct.toFixed(0)}% owned</span>
) : (
  <span className="text-[10px] text-text-muted">— owned</span>
)}

// AFTER
{/* Ownership % is meaningless for rostered players — suppressed on roster page */}
```

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed

---

## FIX 3 — H2H category W/L displayed as match record (P2)

**File:** `frontend/app/(dashboard)/war-room/roster/page.tsx`  
**Components:** `CategorySummary` and `MatchupStrip`

### Problem
The season-stats tile header and the matchup strip showed raw W/L counts (e.g., `4W·9L`) without clarifying that these are **per-category H2H totals**, not the user's actual Yahoo match record (which might be `1-6-1`). Users consistently misread the numbers as their season match record.

### Fix
Added explicit "Category W-L" labels so the distinction is clear. Match record is not yet available from the backend API, so the existing data is now correctly labeled rather than removed.

**CategorySummary (Season Stats header):**
```tsx
// BEFORE
<span className="text-[10px] text-text-muted">
  ·{' '}
  <span className="text-status-safe font-bold">{wCount}W</span>
  {' · '}
  <span className="text-status-lost font-bold">{lCount}L</span>
</span>

// AFTER
<span className="text-[10px] text-text-muted">
  {' · Category W-L: '}
  <span className="text-status-safe font-bold">{wCount}W</span>
  {' · '}
  <span className="text-status-lost font-bold">{lCount}L</span>
</span>
```

**MatchupStrip (This Week header):**
```tsx
// BEFORE
<div className="flex items-center gap-1.5">
  <span className="text-xs font-bold text-status-safe">{categories_won}W</span>
  ...
</div>

// AFTER
<div className="flex items-center gap-1.5">
  <span className="text-[10px] text-text-muted uppercase tracking-wider">Category W-L</span>
  <span className="text-xs font-bold text-status-safe">{categories_won}W</span>
  ...
</div>
```

### Verification
- ✅ `npx tsc --noEmit` — passed
- ✅ `npm run build` — passed
- ⚠️ Pre-existing `<img>` tag warnings in `preview/page.tsx` and `yahoo-roster-view.tsx` remain unchanged.

---

## Build Summary

```
▲ Next.js 15.5.13
✓ Compiled successfully in 8.2s
✓ Generating static pages (25/25)
✓ Finalizing page optimization ...
```

All modified routes built successfully:
- `/war-room/preview` — 4.79 kB
- `/war-room/roster` — 15.8 kB

---

## Files Modified

1. `frontend/app/(dashboard)/war-room/preview/page.tsx` — null guard for `opponent_name`
2. `frontend/app/(dashboard)/war-room/roster/page.tsx` — suppress ownership% + clarify Category W-L labels

---

## Next Steps / Notes

- **Match record API:** When the backend exposes an actual `match_record` field (e.g., `1-6-1`) on the scoreboard or dashboard response, the UI should be updated to show **both** labels side-by-side:  
  `Match Record: 1-6-1 | Category W-L: 4W-9L`
- **FIX 3 file-path note:** The original task spec cited `dashboard/page.tsx` and `budget/page.tsx` for the W/L label issue, but the actual W/L displays live in `roster/page.tsx` (`CategorySummary` and `MatchupStrip` components). The fix was applied in the correct location.
