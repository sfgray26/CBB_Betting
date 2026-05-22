# Week Boundary Fix — 2026-05-22

## Root Cause

`fantasy.py` used `date(2026, 3, 20)` (Opening Day, Thursday) as the epoch for the
Yahoo H2H week number formula:

```python
# BEFORE (wrong)
_MLB_OPENING_DATE_2026 = date(2026, 3, 20)
current_week = (days_since_opening // 7) + 1
# May 22 → 63 days → Week 10  ❌
```

Yahoo H2H weeks run **Monday–Sunday**. Week 1 = March 24–30, 2026.
The 4-day offset (Opening Day is Thursday; first Yahoo Monday is March 24) shifted
every week boundary, producing Week 10 when Yahoo shows Week 9.

## Three Bugs Fixed

| Bug | Root cause | Fix |
|-----|-----------|-----|
| IP tracker shows 0.0 IP | `get_matchup_stats(week=10)` queried a future (empty) week | Correct epoch → week=9 |
| Acquisition counter shows 0/8 | Week window still computed correctly from Monday (separate code), but the week number passed to Yahoo was wrong for matchup stats | Same epoch fix |
| Week label shows "Week 10" | Inline formula used wrong epoch | Replaced with helper |

## Changes

### `backend/routers/fantasy.py`
- Added module-level `_MLB_FIRST_MATCHUP_MONDAY = date(2026, 3, 24)` and `_FANTASY_TOTAL_WEEKS = 25`
- Extracted `_compute_mlb_current_week(today: date) -> int` helper (testable, with clamping)
- Replaced 5-line inline calculation in `get_constraint_budget` with single call to helper
- Added Yahoo sync guard: fetches `league_meta.current_week` and overrides with WARNING if drift detected
- Fixed `season_start` reference (was `_MLB_OPENING_DATE_2026`, now `_MLB_FIRST_MATCHUP_MONDAY`)

### `backend/services/row_projector.py`
- Changed `_MLB_OPENING_DAY` from `date(2026, 3, 27)` (unclear origin) to `date(2026, 3, 24)`
  to align with the canonical first-matchup-Monday constant

### `tests/test_fantasy_budget.py`
- Added 5 week-number unit tests covering: opening Monday, end of Week 1, May 22 regression,
  season-end clamp, pre-season clamp

## Commits

```
163d9ff fix(budget): use first Yahoo matchup Monday (Mar 24) as week epoch
c31090b fix(budget): add Yahoo current_week sync guard with WARNING on mismatch
ea0d821 fix(projector): align _MLB_OPENING_DAY to first Yahoo matchup Monday (Mar 24, 2026)
```

## Verification

- `venv\Scripts\python -m py_compile` → both files clean
- `pytest tests/test_fantasy_budget.py` → 19 passed (5 new week tests + 14 pre-existing)
- `pytest tests/test_fantasy_budget.py tests/test_constraint_helpers.py tests/test_roster_move_api.py tests/test_player_board_fusion.py` → 85 passed, 0 failed
- Live curl expected: `week_label: Week 9`, `acquisitions_used: 1`, `ip_accumulated: 19.2`
