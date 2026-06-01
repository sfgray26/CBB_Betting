# Pitcher Fix + Scoreboard Week Fix — 2026-05-25

## FIX 1 — Pitchers Empty: Type Mismatch in `_is_probable_starter`

### Root cause (confirmed via Railway logs)

`load_probable_pitchers_from_snapshot()` in `probable_pitcher_fallback.py` returns
`dict[str, dict]` — values are `{"name": "kyle harrison", "handedness": "L"}`.

`_is_probable_starter()` at L1210 does:

```python
if player_lower in probable[team] or probable[team] in player_lower:
```

`{"name": ...} in "kyle harrison"` → `TypeError: 'in <string>' requires string as left operand, not dict`

This exception propagated out of `flag_pitcher_starts()`, catching in the DIAG try/except wrapper
and returning `[]` — causing the endpoint to return 0 pitchers despite 11 SP/RP/P-eligible players.

### Fix

In `_fetch_probable_pitchers_for_date()` (`daily_lineup_optimizer.py ~L1140`), after loading
persisted snapshot data, normalize dict values to strings before returning:

```python
return {
    team: v.get("name", "") if isinstance(v, dict) else v
    for team, v in persisted.items()
}
```

All callers of `_fetch_probable_pitchers_for_date()` now receive `dict[str, str]` regardless of
which data source (snapshot DB, MLB Stats API, inferred) populated the map.

---

## FIX 2 — Scoreboard Week Calculation Uses Wrong Opening Day

### Root cause

`backend/routers/fantasy.py` L6236–6240 had a duplicate inline week calculation:

```python
days_since_opening = (now_et - datetime(now_et.year, 3, 28, ...)).days
week = max(1, min(25, (days_since_opening // 7) + 1))
```

`March 28` is a Saturday, not the first Yahoo matchup Monday (March 24). This produced a
week number 4 days off from `_compute_mlb_current_week()`, causing the scoreboard endpoint
to request the wrong Yahoo matchup week.

### Fix

Replaced with the already-correct helper used by the budget endpoint:

```python
if week is None:
    week = _compute_mlb_current_week(datetime.now(ZoneInfo("America/New_York")).date())
```

---

## Files Modified

| File | Change |
|------|--------|
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Normalize `dict[str, dict]` → `dict[str, str]` when returning persisted snapshot data |
| `backend/routers/fantasy.py` | Replace March 28 inline calc with `_compute_mlb_current_week()` |

## Verification

- `py_compile` both files → clean
- `pytest tests/test_lineup_optimizer.py tests/test_dashboard_service*.py` → 29 passed, 1 skipped
