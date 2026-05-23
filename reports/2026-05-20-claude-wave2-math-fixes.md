# Wave 2 Math Engine Fixes

**Date:** 2026-05-20
**Severity:** P2 HIGH — wrong win/loss status on multiple categories, wrong IP pace badge

---

## Bug 1 — Category Deficit Logic (`dashboard_service.py`)

**Root cause:** `deficit = opp_f - my_f` and `winning = deficit <= 0` always treated all categories as higher-is-better. Ignored `LOWER_IS_BETTER` from stat_contract.

**Observed failures:**
- L (losses): user 0, opp 2 → showed Bubble (incorrect). Should be SAFE/winning.
- HR_P: user 3, opp 2 → showed Bubble (incorrect). Should be BEHIND/losing.
- K_B: user 16, opp 14 → showed winning (incorrect). Should be behind.

**Fix:** `backend/services/dashboard_service.py` lines 525–542 — added `LOWER_IS_BETTER` check per canonical code. For lower_is_better cats: `deficit = my - opp`, `winning = my < opp`. For higher_is_better: `deficit = opp - my`, `winning = my > opp`.

The canonical code comes from `_CONTRACT.yahoo_id_index` so it's consistent with `LOWER_IS_BETTER` keys.

---

## Bug 2 — IP Pace AHEAD with 10.2 IP (`scoreboard_orchestrator.py`)

**Root cause:** `compute_budget_state` passed `days_elapsed=season_days_elapsed` (~61 days since Opening Day) and `days_total=182` (full season) to `classify_ip_pace`. This projected weekly IP as a season-long rate: `10.2 / 61 * 182 = 30.4 IP` → well above the 19.8 upper bound → AHEAD.

**Secondary issue:** Baseball IP notation is base-3: `10.2` means 10 complete innings + 2 outs = 10.667 true innings. Yahoo returns IP in this notation; was parsed as a linear float.

**Fix:** `backend/services/constraint_helpers.py` — added `ip_baseball_to_float(ip)` converter.

**Fix:** `backend/services/scoreboard_orchestrator.py compute_budget_state` — convert IP before pace check; compute `days_elapsed_weekly = max(1, 7 - days_remaining)` and use `days_total=7`. The pace is now weekly-scoped.

**Example (Thursday, day 4):** 10.667 IP / 4 days * 7 = 18.67 projected → ON_TRACK (not AHEAD). On day 5: 14.9 → BEHIND. Correct behavior.

---

## Bug 3 — Int Cast on Split Counts (`matchup_engine.py`)

**Root cause:** Split stat rows from the DB were cast to `int()` before wOBA computation. If the DB stores these as `Decimal` (NUMERIC column), truncation could cause precision loss (e.g., `Decimal("4.99")` → 4).

**Fix:** `backend/services/matchup_engine.py` lines 334–353 — changed `int()` to `float()` for all split count reads. wOBA weights are floats so the result precision is unchanged; the fix prevents silent truncation from Decimal columns.

---

## Files Changed

| File | Change |
|------|--------|
| `backend/services/constraint_helpers.py` | Added `ip_baseball_to_float()` |
| `backend/services/scoreboard_orchestrator.py` | Weekly IP pace params + base-3 conversion |
| `backend/services/dashboard_service.py` | Lower-is-better category deficit logic |
| `backend/services/matchup_engine.py` | float() instead of int() for split counts |
| `tests/test_fantasy_budget.py` | 7 new IP notation + pace tests |
| `tests/test_category_math.py` | 9 new lower-is-better deficit tests |

**Test result:** 97/97 passing across 3 targeted suites.
