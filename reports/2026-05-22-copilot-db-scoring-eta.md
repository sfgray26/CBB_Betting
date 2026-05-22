# Wave 3: DB Fix + Scoring Penalty + ETA Polish
**Agent:** Copilot CLI (Claude Sonnet 4.6)
**Branch:** stable/cbb-prod
**Commit:** 586deb0
**Date:** 2026-05-22

---

## TASK A — roster_acquisitions Table Guard (P1: Production DB Error)

**File:** `backend/main.py` — `lifespan()` startup block

**Root cause:** `init_db()` in `models.py` (line 625) calls `Base.metadata.create_all()` but is
only invoked from `if __name__ == "__main__"`, NOT from the FastAPI `lifespan()` event. The
`RosterAcquisition` model was added after the initial Railway deploy, so the table was never
created in production.

**Fix:** Added a surgical `inspect(engine)` guard at the top of the `lifespan()` startup block:
```python
_inspector = _sa_inspect(_db_engine)
if "roster_acquisitions" not in _inspector.get_table_names():
    RosterAcquisition.__table__.create(_db_engine)
    logger.info("Lifespan: created roster_acquisitions table")
```
This is surgical — it only creates the one missing table rather than calling `create_all()` and
risking side effects on other tables. Wrapped in a `try/except` so a permissions failure at startup
never blocks the rest of the application.

---

## TASK B — Small Sample Size Penalty (P2: Jake Bauers "must add")

**Files:** `backend/services/waiver_edge_detector.py`, `backend/schemas.py`,
`backend/routers/fantasy.py`

**Root cause:** A player with a 3-game hot streak accumulates high `cat_scores` but those scores
are computed from very few plate appearances/innings. The scoring pipeline had no mechanism to
discount thin samples.

**Fix — `waiver_edge_detector.py` `get_top_moves()`:**
Added a penalty block after the depth-factor multiplier (line ~657), before building the `move` dict:
- **Pitchers:** read actual season-to-date IP from Yahoo stat ID `"50"`. If `0 < ip < 30`,
  apply `score *= max(0.5, ip / 30.0)` and flag `small_sample = True`.
- **Batters:** estimate PA from H (Yahoo stat ID `"8"`) using league-average BA ~0.265.
  If estimated PA < 100, apply `score *= max(0.5, pa_estimate / 100.0)` and flag `small_sample = True`.
- Floor of 0.5 prevents score dropping below half even for near-zero samples.
- `"small_sample"` key added to the `move` dict (feeds dashboard service).

**Fix — `schemas.py`:**
Added `small_sample: Optional[bool] = None` to `WaiverPlayerOut`.

**Fix — `routers/fantasy.py` `_to_waiver_player()`:**
Computes `_small_sample` flag from `_raw_stats` (same IP/H logic, flag only — no score penalty
in the router since scoring is separate via `category_aware_scorer`). Passes to `WaiverPlayerOut`.

**New tests:** `TestSmallSamplePenalty` (4 tests) — pitcher small IP, batter small PA,
empty stats dict (no penalty), minimum clamp at 0.5.

---

## TASK C — Blake Snell ETA TBD Display (P2: Fabricated Return Date)

**File:** `backend/services/injury_overlay.py`

**Root cause (pre-wave1):** BDL was returning `Jun 30` for a 15-Day IL player (applying the
60-day formula). Wave1 fixed dates > 1.5× the IL minimum but the Jun 2 case (21d) slipped
through (21 < 1.5×15=22.5).

**Wave1 fix limitation:** The 1.5× threshold treated "reasonably close" BDL dates as reliable
and showed them as specific ETAs. But Jun 2 for a player eligible May 27 is still uncertain —
FanGraphs lists "no timetable."

**Fix:** Changed threshold from `1.5×` to `1.2×` and changed the override behavior:
- **≤ 1.2×**: BDL date shown as specific ETA (doctor confirmed, within 20% of IL minimum)
- **> 1.2×**: Show `"ETA: TBD (eligible [computed_eta])"` — uncertain estimate, show earliest
  possible date as the qualifier instead of a potentially fabricated hard date
- **No BDL date**: Show `computed_eta` as specific date (IL minimum is reliable, no uncertainty)

```
Jun 2 case (21d): 21 > 15×1.2=18 → "ETA: TBD (eligible May 27)" ✓
Jun 30 case (49d): 49 > 18 → "ETA: TBD (eligible May 27)" ✓ (was: "May 27" without TBD)
May 27 case (15d): 15 ≤ 18 → "ETA May 27" ✓
```

**Updated docstring** to document the 1.2× threshold and TBD behavior.

**Updated/added tests:**
- Updated `test_blake_snell_15day_il_override`: now asserts `"TBD" in return_timeline`
- Updated `test_60day_il_within_range_keeps_bdl_date`: comment + added `assert "TBD" not in`
- Updated `test_no_injury_date_uses_bdl_return_date_as_fallback`: comment fix
- Added `TestETATBDDisplay` (5 tests): >1.2×, at-minimum, within 1.2×, no return_date (no TBD), 60-Day IL

---

## Verification

```
py_compile: all 5 modified backend files — PASS
pytest tests/test_waiver_edge.py tests/test_injury_overlay.py — 62 passed
pytest tests/ — 3042 passed, 3 skipped, 4 pre-existing failures (row_projector date arithmetic)
```

**Pre-existing failures (not introduced by this change):**
- `test_row_projector.py::test_blended_rate_rolling_and_season`
- `test_row_projector.py::test_custom_weights`
- `test_row_projector_fixes.py::TestP01DynamicSeasonDays::test_days_into_season_opening_day`
- `test_row_projector_fixes.py::TestP01DynamicSeasonDays::test_days_into_season_day_20`

Root cause: hard-coded opening day in `_days_into_season()`. Not in scope.

---

## Files Modified

| File | Change |
|------|--------|
| `backend/main.py` | Added `roster_acquisitions` table guard in `lifespan()` |
| `backend/services/waiver_edge_detector.py` | Small-sample penalty + `small_sample` key in `move` dict |
| `backend/services/injury_overlay.py` | 1.5→1.2× threshold + TBD display for uncertain ETAs |
| `backend/schemas.py` | `small_sample: Optional[bool]` added to `WaiverPlayerOut` |
| `backend/routers/fantasy.py` | `_small_sample` flag set in `_to_waiver_player()` |
| `tests/test_waiver_edge.py` | +4 tests (`TestSmallSamplePenalty`) |
| `tests/test_injury_overlay.py` | +5 tests (`TestETATBDDisplay`) + 3 updated assertions |
