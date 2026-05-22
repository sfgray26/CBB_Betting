# IL Filter + ETA Fix Report

**Branch:** `agent/copilot/wave1-il-filter-eta-fix`  
**Date:** 2026-05-22  
**Agent:** Copilot CLI (claude-sonnet-4.6)  
**Status:** ✅ COMPLETE — all targeted tests pass, no regressions introduced

---

## Summary

Two P1/P2 backend bugs fixed:

1. **FIX 1 (P1):** IL players (15-Day-IL, 10-Day-IL, 60-Day-IL) were appearing in active waiver recommendations. Added a hard-gate filter in `waiver_edge_detector.py` that excludes any IL or OUT player before scoring. DTD players are kept but score is reduced by 0.25 with a `dtd_warning` flag.

2. **FIX 2 (P2):** Blake Snell ETA was showing "Jun 30" (42 days) for a 15-Day-IL placement. The code blindly used BDL's `return_date` which applied the wrong IL duration formula. Rewrote ETA logic in `injury_overlay.py` to compute from `injury_date + IL-type-specific duration` and override BDL when its implied duration exceeds 1.5× the IL minimum.

---

## FIX 1 — IL Hard Gate in Waiver Recommendations

**File:** `backend/services/waiver_edge_detector.py`

### Root Cause
`_INACTIVE_STATUSES` (line 19) only gated roster drop-candidates. FA candidates had no IL filter before the scoring loop in `get_top_moves`, so IL players were scored and ranked alongside healthy players.

### Changes
- Added `_IL_STATUS_TOKENS` frozenset covering all IL variants: `il`, `il10`, `il15`, `il60`, `out`, `na`, `10dayil`, `15dayil`, `60dayil`
- Added `_normalize_status(s)` — strips spaces, hyphens, and lowercases
- Added `_is_il_player(player)` — checks both `injury_status` (BDL format: "15-Day-IL") and `status` (Yahoo short codes: "IL", "IL10")
- Added `_is_dtd_player(player)` — checks for "dtd" or "day-to-day"
- Added **IL hard gate** at top of `get_top_moves` loop: `if _is_il_player(fa): continue`
- Added **DTD penalty**: score reduced by `_DTD_PENALTY = 0.25` with `dtd_warning` field appended to move dict

### Behavior Change

| Player Status | Before | After |
|---|---|---|
| 15-Day-IL | Scored and ranked | Excluded (hard gate) |
| 10-Day-IL | Scored and ranked | Excluded (hard gate) |
| 60-Day-IL | Scored and ranked | Excluded (hard gate) |
| OUT | Scored and ranked | Excluded (hard gate) |
| DTD | Full score, no warning | Score −0.25 + `dtd_warning: true` |
| Healthy | Full score | Full score (unchanged) |

---

## FIX 2 — IL-Type-Aware ETA Calculation

**File:** `backend/services/injury_overlay.py`

### Root Cause
`build_injury_overlay()` accepted `return_date` from BDL and used it verbatim. BDL computed Blake Snell's return as Jun 30 (~42 days from May 19 placement) despite his 15-Day-IL designation. No IL-type-specific override existed.

### Changes
- Added `_IL_DURATIONS = {"15": 15, "10": 10, "60": 60}` + `_il_duration_days(status)` helper
- `_il_duration_days("15-Day-IL")` → 15; `"60-Day-IL"` → 60; generic `"IL"` → 10 (minimum)
- `build_injury_overlay()` now accepts `injury_date: Optional[datetime] = None`
- **ETA override logic:**
  - Use `injury_date` (retroactive IL start) as reference; fall back to `ingested_at` if absent
  - `computed_eta = ref_date + timedelta(days=il_days)`
  - If BDL's `return_date` is missing → use computed
  - If BDL's implied duration > `il_days × 1.5` → override with computed
  - If no IL type identified → use BDL's date as-is
- `load_injury_overlays()` updated to pass `injury_date=row.injury_date`

### Blake Snell Example
```
IL type:        15-Day-IL
injury_date:    2026-05-12 (retroactive)
BDL return:     2026-06-30 (49 days — wrong)
il_days:        15
threshold:      15 × 1.5 = 22.5 days
implied_days:   49 > 22.5 → OVERRIDE
computed_eta:   2026-05-12 + 15d = 2026-05-27
ETA shown:      "May 27"
```

---

## Tests Added

### `tests/test_waiver_edge.py`

| Test | Description |
|---|---|
| `TestILPlayerFilter` (16 tests) | Unit tests for `_is_il_player()` across all IL variants (IL, IL10, IL15, IL60, 10-Day-IL, etc.) and `_is_dtd_player()` |
| `test_il_players_excluded_from_waiver` | Integration: Jack Dreyer (15-Day-IL) absent from waiver moves |
| `test_il_player_with_yahoo_status_excluded` | Integration: Yahoo short-code "IL10" player excluded |
| `test_dtd_player_included_with_warning_and_reduced_score` | Integration: DTD player present with `dtd_warning=True`, score < healthy score |

### `tests/test_injury_overlay.py`

| Test | Description |
|---|---|
| `TestILDurationDays` (6 tests) | Unit tests for `_il_duration_days()` across all IL types, DTD (None), and unknown (None) |
| `test_blake_snell_15day_il_override` | BDL's Jun 30 overridden to May 27 for 15-Day-IL |
| `test_60day_il_within_range_keeps_bdl_date` | BDL date within 1.5× threshold is preserved |
| `test_no_injury_date_uses_bdl_return_date_as_fallback` | Falls back to BDL when injury_date is None |
| `test_no_return_date_computes_from_injury_date` | Computes ETA from injury_date when BDL has no return_date |
| `test_no_timetable_shows_eta_unknown` | Generic IL with no dates still shows ETA |
| `test_dtd_with_no_return_date_shows_no_eta` | DTD with no dates shows no ETA |

### `tests/test_roster_waiver_enrichment_contract.py`

Updated `test_waiver_overlays_bdl_injury_freshness_and_penalty`:
- **Before:** Asserted IL player appears in `top_available` (incorrect expectation)
- **After:** Asserts IL player is absent from `top_available` and present in `il_watch` with full overlay fields

---

## Verification

```
# py_compile
.\venv\Scripts\python -m py_compile backend/services/waiver_edge_detector.py  ✅
.\venv\Scripts\python -m py_compile backend/services/injury_overlay.py         ✅

# Targeted
.\venv\Scripts\python -m pytest tests/test_waiver_edge.py -v       → 19 passed
.\venv\Scripts\python -m pytest tests/test_injury_overlay.py -v    → 15 passed

# Full suite
.\venv\Scripts\python -m pytest tests/ -q    → 3007 passed, 3 skipped, 4 pre-existing failures
```

### Pre-existing Failures (not introduced by this branch)
`test_row_projector.py::test_blended_rate_rolling_and_season`,
`test_row_projector.py::test_custom_weights`,
`test_row_projector_fixes.py::TestP01DynamicSeasonDays::test_days_into_season_opening_day`,
`test_row_projector_fixes.py::TestP01DynamicSeasonDays::test_days_into_season_day_20`

These are date-arithmetic failures in `_days_into_season()` unrelated to this branch (confirmed by `git diff --name-only HEAD` showing zero changes to those files).

---

## Files Changed

| File | Change |
|---|---|
| `backend/services/waiver_edge_detector.py` | +70 lines: IL/OUT gate + DTD penalty helpers |
| `backend/services/injury_overlay.py` | +68 lines: IL-type-aware ETA + `_il_duration_days` |
| `tests/test_waiver_edge.py` | +~130 lines: 19 new tests |
| `tests/test_injury_overlay.py` | +~110 lines: 12 new tests |
| `tests/test_roster_waiver_enrichment_contract.py` | ~20 lines updated: test aligned with correct IL behavior |
