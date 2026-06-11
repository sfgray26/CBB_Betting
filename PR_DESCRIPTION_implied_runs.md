# PR: Fix _implied_runs Sign Bug and Add Unit Tests

## Summary

Fixes the `_implied_runs()` sign bug in `daily_lineup_optimizer.py` and adds comprehensive unit tests to prevent regression.

## Bug Details

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`
**Function:** `_implied_runs()` (lines 432-448)

### The Problem

The original buggy code used:
```python
home_runs = (total + spread_home) / 2.0
```

When `spread_home` is negative (home team favored, e.g., -1.5), this incorrectly SUBTRACTED from the home team's implied runs, giving them FEWER runs instead of more.

**Example of bug:**
- Total: 9 runs
- Spread: -1.5 (home favored by 1.5 runs)
- Bug result: home_runs = (9 + (-1.5)) / 2 = 3.75 (WRONG - underdog level)
- Correct result: home_runs = (9 - (-1.5)) / 2 = 5.25 (CORRECT - favorite gets more runs)

### The Fix

Changed the formula to:
```python
home_runs = (total - spread_home) / 2.0
```

This correctly gives the home team MORE runs when they are favored (negative spread).

## Test Coverage

Added 14 comprehensive unit tests in `tests/test_lineup_optimizer.py`:

1. `test_home_favorite_negative_spread` - Home fav (-1.5) gets more runs
2. `test_away_favorite_positive_spread` - Away fav (+1.5) gets more runs  
3. `test_pickem_zero_spread` - Equal runs when spread is 0
4. `test_larger_spread_home_heavy_favorite` - Heavy home favorite case
5. `test_larger_spread_away_heavy_favorite` - Heavy away favorite case
6. `test_total_runs_sum_equals_input_total` - Verify total conservation
7. `test_low_total_clamping` - Minimum runs clamping (1.0)
8. `test_high_total_clamping` - Maximum runs clamping (12.0)
9. `test_negative_spread_gives_home_more_runs` - Regression test for negative spreads
10. `test_positive_spread_gives_away_more_runs` - Verify positive spreads
11. `test_decimal_totals_and_spreads` - Common betting lines (e.g., 8.5 total)
12. `test_small_spread_negligible_difference` - Small spreads work correctly
13. `test_edge_case_extreme_negative_spread` - Edge case handling
14. `test_rounding_to_two_decimal_places` - Precision validation

## Verification

All tests pass:
```
pytest tests/test_lineup_optimizer.py -v
======================== 14 passed in 1.42s =========================
```

## Impact

This fix ensures:
- Batter rankings correctly value players facing weak pitching (low opponent implied runs)
- Streaming pitcher recommendations properly account for favorable matchups
- DFS stack recommendations accurately identify high-scoring game environments

## Backward Compatibility

The fix maintains backward compatibility:
- Function signature unchanged
- Return values still rounded to 2 decimal places
- Clamping behavior unchanged (1.0-12.0 range)
- Only the sign logic is corrected
