# UAT Wave Bug Fixes

**Date:** 2026-05-21
**Source:** 5-persona UAT analysis report (Elite FM + Quant Sabermetrics criteria)

---

## Fixes Applied

### Task 3 — IP Accumulation Tracker: 0.0 on Day 4

**Root cause:** `GET /api/fantasy/budget` calls `client.get_matchup_stats()` to fetch IP. When Yahoo's API fails or returns empty `my_stats`, `ip_accumulated` silently defaulted to 0.0, causing a false "BEHIND" pace warning.

**Fix:**
- `backend/routers/fantasy.py`: Added `ip_data_available` bool — set `True` only when Yahoo actually returned an IP value, `False` on any failure.
- Added `ip_data_available` to the budget response dict.
- `frontend/lib/types.ts`: Added `ip_data_available?: boolean` to `BudgetData`.
- `frontend/components/dashboard/budget-panel.tsx`: When `ip_data_available === false`, show "PENDING" badge (muted, no red) and "Yahoo stats syncing…" sub-label instead of the false "BEHIND" alarm.

---

### Task 4 — "Fit for your gaps" specificity

**Root cause:** Static string "fit for your gaps" shown below the Match Score bar, regardless of which categories the player actually addresses.

**Fix:**
- `frontend/app/(dashboard)/war-room/waiver/page.tsx` (line 203): Replaced static string with dynamic label using the already-computed `needMatches` array: shows "Fits: K, RBI" (up to 3 categories) when matches exist, falls back to "fit for your gaps" only when `needMatches` is empty.

---

### Task 5 — Two-Start Pitcher flag + filter toggle

**Root cause:** `WaiverPlayerOut` lacked a `two_start` field — only `starts_this_week: int`. The streaming page's `player.two_start` check (line 270) never fired. No filter toggle existed.

**Fix:**
- `backend/schemas.py`: Added `two_start: bool = False`, `start1_opp: Optional[str] = None`, `start2_opp: Optional[str] = None` to `WaiverPlayerOut`.
- `backend/routers/fantasy.py`: Set `two_start=starts_this_week >= 2` when building `WaiverPlayerOut`.
- `frontend/app/(dashboard)/war-room/streaming/page.tsx`: Added `showTwoStartOnly` state + checkbox filter; applies to `passesFilters()`.

The "⚡ 2-START" badge was already implemented in `WaiverPlayerRow` at line 270 — it now fires correctly.

---

### Task 6 — HOT/COLD tag definition + suppression

**Root cause:** HOT/COLD showed on DTD/injured players and had no tooltip explaining the criteria.

**Fix:**
- `backend/routers/fantasy.py`: After computing `_hc`, suppress HOT when `injury_status` contains "DTD", "IL", or "DL". IL players are already excluded from `top_available`; this additionally catches DTD players.
- `frontend/components/hot-cold-badge.tsx`: Added `title` attribute (native browser tooltip): "HOT: avg category z-score > 0.75 across recent stats (7-day window)" / "COLD: avg < −0.5 …". Added `cursor-help` class.

---

### Bonus — DecisionAccuracyResponse field alignment

**Root cause:** Backend `Wave 5` endpoint used `override_better_count`, `override_worse_count`, `total_decisions`, `trend[i].accuracy` — the existing decisions page (`frontend/app/(dashboard)/decisions/page.tsx`) expected `better_count`, `worse_count`, `total_overrides`, `daily_trend[i].accuracy_pct`.

**Fix:**
- `backend/contracts.py`: Renamed `DecisionAccuracyTrendPoint.accuracy` → `accuracy_pct`; renamed `DecisionAccuracyResponse` fields to match frontend.
- `backend/routers/fantasy.py`: Updated endpoint return to use new field names.
- `tests/test_decision_tracker.py`: Updated 4 endpoint test assertions.

---

## Test Results

```
tests/test_matchup_preview.py    4/4 passed
tests/test_decision_tracker.py   8/8 passed
Full suite                       2963 passed, 3 skipped
```

## Files Changed

| File | Change |
|------|--------|
| `backend/routers/fantasy.py` | ip_data_available tracking; two_start field; HOT suppression for DTD; DecisionAccuracy field rename |
| `backend/schemas.py` | WaiverPlayerOut: two_start + start1/2_opp fields |
| `backend/contracts.py` | DecisionAccuracy field rename (accuracy_pct, better_count, etc.) |
| `frontend/lib/types.ts` | BudgetData.ip_data_available field |
| `frontend/components/dashboard/budget-panel.tsx` | PENDING state when ip_data_available=false |
| `frontend/components/hot-cold-badge.tsx` | Tooltip with criteria definition; cursor-help |
| `frontend/app/(dashboard)/war-room/waiver/page.tsx` | Dynamic "Fits: K, RBI" label |
| `frontend/app/(dashboard)/war-room/streaming/page.tsx` | showTwoStartOnly filter toggle |
| `tests/test_decision_tracker.py` | Updated field name assertions |
