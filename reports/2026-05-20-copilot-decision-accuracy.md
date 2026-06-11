# Wave 5 — Decision Tracker Override Comparison Fix

**Date:** 2026-05-20
**Severity:** P2 BUGFIX — decisions page showed fake override accuracy stats (always 0/0)

---

## Problem

`get_daily_accuracy()` in `decision_tracker.py` line 220 hardcoded:
```python
override_better_count=0,  # TODO: Compare user vs system
override_worse_count=0,
```
Every call returned 0/0 override comparison regardless of actual history. Users had no way
to evaluate whether their manual overrides were improving their lineup performance.

---

## Root Cause

The comparison logic was omitted at implementation time. The data needed was already
present in each resolved `PlayerDecision`:
- `user_action is not None` → user overrode the system recommendation
- `outcome == "failure"` → system's recommendation turned out wrong → user was right
- `outcome == "success"` → system's recommendation turned out correct → user was wrong

---

## Fix

### `backend/fantasy_baseball/decision_tracker.py` — `get_daily_accuracy()`

Replaced the two hardcoded zeros with:

```python
override_better_count=sum(
    # System was wrong (failure) → user who overrode was right
    1 for d in resolved
    if d.user_action is not None and d.outcome == "failure"
),
override_worse_count=sum(
    # System was right (success) → user who overrode was wrong
    1 for d in resolved
    if d.user_action is not None and d.outcome == "success"
),
```

**Logic:** Since `recommended_action` captures what the system said and `user_action == "OVERRIDE"` means the user took the opposite action, the correctness of the override is the *inverse* of the system outcome.

### New: `GET /api/fantasy/decisions/accuracy`

**File:** `backend/routers/fantasy.py` (inserted after `get_decisions_status`)
**Schema:** `backend/contracts.py` (`DecisionAccuracyTrendPoint`, `DecisionAccuracyResponse`)

```
GET /api/fantasy/decisions/accuracy
→ 200: {
    "date": "2026-05-20",
    "total_decisions": 10,
    "override_better_count": 2,
    "override_worse_count": 1,
    "override_accuracy_pct": 0.6667,
    "trend": [
      {"date": "2026-05-07", "accuracy": 0.72},
      {"date": "2026-05-08", "accuracy": -1.0},   // -1.0 = no data that day
      ...
      {"date": "2026-05-20", "accuracy": 0.70}
    ]
  }
```

- `trend` is always exactly 14 points, oldest first
- `accuracy == -1.0` for days with no decision data
- `override_accuracy_pct == 0.0` when no overrides exist (no division error)

---

## Files Changed

| File | Change |
|------|--------|
| `backend/fantasy_baseball/decision_tracker.py` | Replaced hardcoded 0/0 with real override comparison logic |
| `backend/contracts.py` | Added `DecisionAccuracyTrendPoint`, `DecisionAccuracyResponse` |
| `backend/routers/fantasy.py` | New `get_decisions_accuracy` endpoint + 2 import additions |
| `tests/test_decision_tracker.py` | 8 new tests (4 unit + 4 endpoint) |

**Test result:** 8/8 passing.
**py_compile:** zero errors on all 3 modified backend files.
