# Decision Pipeline Fix — 2026-05-11

## Problem

GET `/api/fantasy/decisions` returns empty results (0 decisions) even when historical player data exists.

## Root Cause

`_run_decision_optimization()` in `backend/services/daily_ingestion.py` used a **strict date filter**: it queried `player_scores` for `as_of_date == today` (ET) with `window_days == 14`.

When the player_scores pipeline had not yet run for the current day (e.g., off-day, early-morning call before the 4 AM job, or pipeline failure), `score_rows` was empty. The function returned `"status": "no_input"` with **zero** `DecisionResult` rows written. The API then had no rows to return.

This pattern was already recognized elsewhere in the codebase. `_update_market_signals()` (line ~3653) explicitly falls back:

```python
# Use latest available date in case today's scoring job hasn't run yet.
```

But `_run_decision_optimization` did **not** have this fallback.

## Fix Applied

Modified `_run_decision_optimization` to determine an `effective_date`:

1. Start with `effective_date = today`
2. Query `MAX(player_scores.as_of_date)` where `window_days = 14`
3. If the latest date **is earlier than today**, fall back to that date
4. Use `effective_date` for **all** downstream queries and writes:
   - `player_scores`
   - `player_momentum`
   - `simulation_results`
   - `DecisionResult` / `DecisionExplanation` delete filters
   - `optimize_lineup()` and `optimize_waivers()` calls
   - Return value (`as_of_date` field)

This keeps the pipeline resilient on off-days or when the overnight jobs haven't completed yet, while still writing decisions with the **actual data date** so the API `MAX(as_of_date)` default behavior works correctly.

## Files Changed

- `backend/services/daily_ingestion.py` — `_run_decision_optimization()` only

## Verification

- `py_compile` passes with no errors.
- `test_decision_engine.py` — **40 passed** (pure-function tests unaffected).
- `test_decisions_api.py` — 17 errors pre-existing (TestClient `Client.__init__()` compatibility issue, unrelated to this change).

## Not Changed

- `models.py` — not modified per acceptance criteria.
- No timezone changes needed; `as_of_date` was already consistently ET.
- No roster mapping fallback added; the "fail closed" behavior for empty `roster_bdl_ids` was intentional (ADR-004). The root cause was the date filter, not roster resolution.

## Decision

Close as **fixed** — the date filter was too strict and now falls back to the most recent available `player_scores` date, matching the existing pattern in `_update_market_signals`.
