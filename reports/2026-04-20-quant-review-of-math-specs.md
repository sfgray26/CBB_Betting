# Quantitative Review: Math Error Catalog & Fix Implementation Plan
**Reviewer**: Kimi CLI (Quantitative Analysis & Fantasy Baseball Domain Expert)
**Date**: April 20, 2026
**Scope**: `docs/superpowers/specs/2026-04-20-math-error-catalog.md` and `2026-04-20-math-fix-implementation-plan.md`
**Verdict**: Conditionally Approved — 3 critical corrections required before implementation

---

## Overall Assessment

Both documents are **well-structured, clearly written, and directionally correct**. The error taxonomy is sound, the phased implementation approach is sensible, and the test cases are thorough. However, **3 critical issues** must be corrected before implementation begins:

1. **P0-3 "Before" code is fabricated** — does not match actual `decision_engine.py`
2. **P0-2 impact magnitude is overstated by ~7×** (~2% not 15%)
3. **P1-5 fix is convoluted** — goes through 4 iterations without converging on a clean solution

Additionally, **4 moderate issues** should be addressed to ensure the fixes are robust and maintainable.

---

## 🔴 Critical Issue 1: P0-3 Spec Code Does Not Match Actual Codebase

**Location**: Math Error Catalog, P0-3 section; Implementation Plan, Phase 2

**Problem**: The spec presents this as the "Before" code:

```python
def _composite_value(self, player: CanonicalPlayerRow) -> float:
    projection = player.projection_78 or {}
    proj_era = projection.get("era") or 4.50
    ...
```

**Actual code in `backend/services/decision_engine.py` (lines 233–274)**:

```python
def _composite_value(player: PlayerDecisionInput) -> float:
    pt = (player.player_type or "unknown").lower()
    score_anchor = (player.score_0_100 / 100.0) * 1.5
    z_anchor = max(-1.0, min(player.composite_z, 2.0)) + 1.0
    baseline = max(score_anchor, z_anchor / 2.0)
    ...
```

**The function is a standalone function, not a method. It takes `PlayerDecisionInput`, not `CanonicalPlayerRow`. It has no `projection_78` attribute.**

**Impact**: If implemented as written, the fix will fail with `AttributeError: 'PlayerDecisionInput' object has no attribute 'projection_78'`.

**Correct Fix for Actual Codebase**:

The simplest and most robust fix is to **eliminate the arbitrary formula entirely** and use the already-computed `composite_z`:

```python
def _composite_value(player: PlayerDecisionInput) -> float:
    pt = (player.player_type or "unknown").lower()
    score_anchor = (player.score_0_100 / 100.0) * 1.5
    z_anchor = max(-1.0, min(player.composite_z, 2.0)) + 1.0
    baseline = max(score_anchor, z_anchor / 2.0)

    # P0-3 FIX: Use composite_z directly instead of arbitrary formula
    # composite_z is already computed from proper z-scores in scoring_engine.py
    if pt == "pitcher":
        return max(player.composite_z, baseline)
    if pt == "hitter":
        hr  = min((player.proj_hr_p50  or 0.0) / _HR_NORM,  1.0)
        rbi = min((player.proj_rbi_p50 or 0.0) / _RBI_NORM, 1.0)
        sb  = min((player.proj_sb_p50  or 0.0) / 50.0,      1.0)
        projection_total = hr + rbi + sb
        return max(projection_total, baseline)
    if pt == "two_way":
        # Use composite_z for both components
        return max(player.composite_z, baseline)

    return baseline
```

**Rationale**: `player.composite_z` already contains the properly normalized, LOWER_IS_BETTER-aware z-scores from `scoring_engine.py`. Using it directly avoids duplicating the category logic and eliminates the arbitrary formula.

**Trade-off**: `composite_z` for pitchers is computed from only 5 categories (ERA, WHIP, K/9, K_P, QS) because W, L, HR_P, NSV are greenfield in `scoring_engine.py`. This is acceptable because:
1. Those 4 categories have no upstream data anyway
2. The arbitrary formula also ignored W, L, NSV, HR_P
3. Using `composite_z` is strictly better than using `K/9 - ERA/9 + (1 - WHIP/2)`

---

## 🔴 Critical Issue 2: P0-2 Impact Overstated

**Location**: Math Error Catalog, P0-2 section; User summary table

**Claimed Impact**: "~15% OBP undervaluation" (from user summary table)
**Actual Impact**: **~2.1% OBP undervaluation** (from catalog's own table)

**The catalog's own math contradicts the summary**:

| Formula | OBP | Error |
|---------|-----|-------|
| Correct | 0.381 | — |
| Buggy | 0.373 | **-2.1%** |

The "~15%" figure in the summary table appears to have been computed as `(0.381 - 0.373) / 0.373 × 100 = 2.1%` — which is indeed ~2%, not 15%. The 15% figure may have come from an earlier draft or a different calculation.

**At team level**: The catalog claims "Team OBP undervalued by ~18 points, OPS undervalued by ~18 points." This is also incorrect. With 9 hitters each having OBP off by ~0.008, the **team OBP** (which is `(sum H + sum BB) / (sum AB)`) is off by approximately the same ~0.008, not 9× that. OBP is a ratio, not a sum — errors don't accumulate linearly across players.

**Recommended Correction**:
- Change summary table: "~15% OBP undervaluation" → "~2% OBP undervaluation"
- Change team-level text: "~18 points" → "~8 points of OBP"
- **Severity should be downgraded from P0 to P1** — a 2% bias in one category is not system-wide critical

**However**, keep the fix in Phase 1 because it's low-complexity and improves correctness.

---

## 🔴 Critical Issue 3: P1-5 Fix Is Convoluted and Unsettled

**Location**: Implementation Plan, Phase 3, P1-5 section

**Problem**: The spec presents **4 different approaches** for P1-5:
1. Option A: Normalize to 11-cat basis
2. Option B: Use only primary position categories
3. "Better approach": Don't normalize by count at all (weighted sum only)
4. "Final approach": Normalize to 10-category baseline

Then it says "Actually, I think the simplest fix is to just not divide by count at all" and provides yet another version.

**This is unacceptable for an implementation spec.** An implementation plan must have **one correct approach**, not a stream-of-consciousness exploration.

**Recommended Single Approach**:

Use **weighted mean with category count normalization**:

```python
# P1-4 + P1-5 combined fix
kv_pairs = [
    (k, getattr(result, k))
    for k in applicable_keys
    if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
]

weighted_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) * v for k, v in kv_pairs)
weight_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k, _ in kv_pairs)

# Normalize to standard 10-category basis for cross-player-type comparability
# 10 ≈ average of 11 (hitter) and 9 (pitcher) and 16 (two-way effective)
BASELINE_CAT_COUNT = 10.0
raw_mean = weighted_sum / weight_sum if weight_sum > 0 else 0.0

# Scale mean to baseline: preserves meaning while making comparable
# Hitter with 11 cats: mean × (11/10) = 1.1× → slight boost for breadth
# Pitcher with 7 cats: mean × (7/10) = 0.7× → slight penalty for fewer cats
# Two-way with 16 cats: mean × (16/10) = 1.6× → but their mean is already diluted
#    Example: two-way with all z=1.0: (16×1.0/16) × (16/10) = 1.6
#    Hitter with all z=1.0: (11×1.0/11) × (11/10) = 1.1
#    This correctly values two-way players more for contributing in more cats
result.composite_z = raw_mean * (len(kv_pairs) / BASELINE_CAT_COUNT)
```

Wait — this still has issues. Let me think more carefully.

**Actually, the cleanest solution is:**

```python
# Compute weighted mean (P1-4 fix)
weighted_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) * v for k, v in kv_pairs)
weight_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k, _ in kv_pairs)
result.composite_z = weighted_sum / weight_sum if weight_sum > 0 else 0.0

# P1-5: Two-way players get the SAME treatment — no special handling needed
# The weighted mean naturally handles different category counts:
# - Ohtani with 16 z-scores of +1.0: composite = 1.0
# - Judge with 11 z-scores of +1.0: composite = 1.0
# They are directly comparable!
```

The key insight: **weighted mean is already comparable across different category counts**. A mean of 1.0 over 16 categories is the same as a mean of 1.0 over 11 categories. The variance difference (σ²/16 vs σ²/11) is a real statistical phenomenon, not a bug — two-way players DO have lower composite variance because they diversify across more categories.

**For `score_0_100`**: The percentile rank is computed **within player type cohorts**. Two-way players are ranked against other two-way players. With only 1–3 two-way players in the league, this percentile is noisy, but that's a **sample size problem**, not a normalization problem. The fix for that is to either:
- Merge two-way players into hitter cohort for percentile ranking, OR
- Don't rank two-way players by percentile (use raw composite_z instead)

**Recommended spec change**: For P1-5, state:
> "No code change needed for composite_z computation. Weighted mean is inherently comparable across category counts. The real issue is `score_0_100` percentile ranking within tiny two-way cohorts. Fix: merge two-way players into hitter cohort for percentile computation, or exclude two-way from percentile and use raw composite_z."

---

## 🟡 Moderate Issue 4: P1-6 Confuses Two Different Metrics

**Location**: Math Error Catalog, P1-6 section

**Problem**: The spec conflates `MIN_SAMPLE` (number of PLAYERS with data) with `confidence` (games played by a single player).

- `MIN_SAMPLE = 5` means: "Need 5+ players before computing z-scores for a category"
- `result.confidence = games_in_window / window_days` means: "This player played X of 14 possible games"

These are **orthogonal**:
- You can have 500 players with data (n=500 >> MIN_SAMPLE) but a specific player only played 2 games (confidence=0.14)
- You can have 3 players with data (n=3 < MIN_SAMPLE) but each played 14 games (confidence=1.0)

The spec's proposed fix lowers MIN_SAMPLE to 3 and says "consumers should use confidence field to filter." But `confidence` doesn't reflect the sample-size uncertainty! A player with confidence=1.0 in a category computed from n=3 players has exactly the same sample-size problem as a player with confidence=0.2.

**Correct Fix for P1-6**:

Add a **category-level reliability metric**:

```python
# In scoring_engine.py, after computing z-scores for a category:
n_players_for_category = len(pairs)  # actual sample size

# Store sample size in result (new field needed)
# Or: scale z-score by reliability
reliability = min(1.0, n_players_for_category / 30.0)
# Apply reliability weighting to z-score
# z_weighted = z * reliability + 0 * (1 - reliability)  # shrink toward mean
```

**Simpler alternative** (no schema change):

```python
# Option B: Lower MIN_SAMPLE to 3 but add a comment warning
MIN_SAMPLE: int = 3
# WARNING: Z-scores computed with n<30 have high uncertainty.
# Consumers should not rely on early-season z-scores for high-stakes decisions.
```

**Recommended spec change**: Replace P1-6 with a **sample-size reliability field** or simply document the limitation. Don't pretend that `confidence` (games played) solves the `MIN_SAMPLE` (player count) problem.

---

## 🟡 Moderate Issue 5: `_days_into_season` Should Not Default to `datetime.now()`

**Location**: Implementation Plan, Phase 1, Step 1

**Problem**: The proposed `_days_into_season` function defaults to `datetime.now().date()`:

```python
def _days_into_season(as_of_date: Optional[date] = None) -> int:
    if as_of_date is None:
        as_of_date = datetime.now().date()
    ...
```

This makes the function **non-deterministic** and hard to test. The monkeypatch in the test case is a workaround, but it's brittle.

**Fix**: Make `as_of_date` **required** in the public API. The `compute_row_projection` function already receives an `as_of_date` parameter in the spec. Internal helpers should not default to "now."

```python
def _days_into_season(as_of_date: date) -> int:
    """Return days since MLB Opening Day (minimum 1)."""
    delta = (as_of_date - _MLB_OPENING_DAY).days + 1
    return max(1, delta)
```

Callers must always pass the date explicitly.

---

## 🟡 Moderate Issue 6: `_MLB_OPENING_DAY` Date Needs Verification

**Location**: Implementation Plan, Phase 1, Step 1

**Problem**: The spec hardcodes `_MLB_OPENING_DAY = date(2026, 3, 27)`.

**MLB Opening Day 2026 was March 26, 2026** (not March 27). Using March 27 would introduce a 1-day offset in all projections.

**Fix**: Verify the correct date and document the source.

```python
# MLB Opening Day 2026 — verified from MLB.com schedule
# https://www.mlb.com/schedule/2026-03-26
_MLB_OPENING_DAY = date(2026, 3, 26)
```

Alternatively, make this a configurable environment variable:

```python
_MLB_OPENING_DAY = date.fromisoformat(
    os.getenv("MLB_OPENING_DAY", "2026-03-26")
)
```

---

## 🟡 Moderate Issue 7: P1-4 Category Weights Are Heuristics, Not Data-Driven

**Location**: Implementation Plan, Phase 3, Fix P1-4

**Problem**: The proposed `_CATEGORY_WEIGHTS` are intuitive but not empirically derived:

```python
"z_hr": 1.2, "z_nsb": 1.3, "z_avg": 0.8, ...
```

**Questions**:
- Why is HR weighted 1.2× but RBI only 1.1×?
- Why is NSB 1.3× — is it actually the scarcest category?
- These weights will affect all player rankings. Who validates they're correct?

**Recommendation**: Either:
1. **Compute weights dynamically** from the actual data:
   ```python
   # Weight = 1 / std_dev_of_category_z_scores
   # Categories with higher variance (more differentiation) get higher weight
   ```
2. **Or start with equal weights (1.0 for all)** and add weighting as a future enhancement with A/B testing

For an implementation plan, I recommend **equal weights** to avoid introducing a second source of bias. The primary fix (switching from mean to weighted mean with equal weights) still solves the specialist dilution problem by allowing categories to accumulate rather than average.

If unequal weights are desired, they should be justified with data from the actual `player_scores` table.

---

## 🟢 Minor Issues

### Issue 8: Test Mock Objects May Not Reflect Actual ORM Structure

The test cases use `MagicMock` extensively:
```python
p = MagicMock()
p.w_ab = 50
p.w_home_runs = 10
```

In the actual `scoring_engine.py`, `compute_league_zscores` iterates over `PlayerRollingStats` ORM objects. The function accesses attributes like `row.w_ab`, `row.w_home_runs`, etc. MagicMock with attribute assignment works for this, but tests should verify that the actual ORM model has these fields.

### Issue 9: `_blended_daily_rate` Has Two Code Paths

In `row_projector.py`, `_blended_daily_rate` is defined at line 363, but the inline season rate computation at line 232 bypasses it:

```python
# Line 232 (inline)
season_rate = season_val / _DEFAULT_SEASON_DAYS if season_val else 0.0

# Line 380 (in _blended_daily_rate)
season_rate = season_total / _DEFAULT_SEASON_DAYS if season_total > 0 else 0.0
```

The spec only fixes line 232/406 but doesn't mention line 380. Both must be fixed.

### Issue 10: OPS Test Case Has Incorrect Expected Value

In the P0-2 test case:
```python
# OBP (no HBP/SF) = (155+50) / 500 = 0.410
```

This is wrong. With 155 H, 50 BB, and 500 AB, OBP without HBP/SF is:
```
OBP = (155 + 50) / (500 + 50) = 205 / 550 = 0.373
```

The test says 0.410 because it forgot to include BB in the denominator. This doesn't affect the fix correctness, but the test comment should be corrected.

---

## Revised Priority Table

| Error | Spec Claim | Quant Assessment | Recommended Severity | Recommended Phase |
|-------|-----------|------------------|---------------------|-------------------|
| P0-1 Hardcoded 100 days | 5× bias | **Confirmed** — severe, affects all counting stats | P0 | Phase 1 |
| P0-2 OPS missing HBP/SF | ~15% OBP error | **Overstated** — ~2% OBP error, not 15% | P1 | Phase 1 (low complexity) |
| P0-3 Pitcher composite | Arbitrary formula | **Confirmed** — but spec code is wrong | P0 | Phase 2 |
| P1-4 Mean dilution | Specialists undervalued | **Confirmed** — use weighted mean | P1 | Phase 3 |
| P1-5 Two-way counting | 2× inflated | **Partial** — mean is already comparable; issue is percentile ranking | P2 | Phase 3 |
| P1-6 MIN_SAMPLE=5 | 45% SE | **Confirmed** — but fix conflates metrics | P1 | Phase 3 |

---

## Recommended Spec Changes

### Math Error Catalog
1. **P0-2**: Change "~15% OBP undervaluation" → "~2% OBP undervaluation"
2. **P0-2**: Change "Team OBP undervalued by ~18 points" → "~8 points"
3. **P0-3**: Replace "Before" code with actual `decision_engine.py` code
4. **P1-5**: Simplify analysis — weighted mean is already comparable; issue is `score_0_100` cohort size
5. **P1-6**: Clarify distinction between `MIN_SAMPLE` and `confidence`; propose reliability field

### Fix Implementation Plan
1. **P0-1**: Make `_days_into_season(as_of_date)` required (no default to `now()`)
2. **P0-1**: Verify `_MLB_OPENING_DAY` is March 26, not 27
3. **P0-1**: Fix both line 232 AND line 380 (`_blended_daily_rate`)
4. **P0-2**: Remove redundant `sum_obp_denom` accumulation fix (Step 2) since Step 3 recalculates anyway
5. **P0-3**: Rewrite entire section with actual function signature; use `composite_z` directly
6. **P1-4**: Use equal weights (1.0) unless empirically justified; justify if not
7. **P1-5**: Replace 4-iteration exploration with single clean approach
8. **P1-6**: Replace with sample-size reliability field OR simple documentation

---

## Final Verdict

**Status**: Conditionally Approved

The specifications are **90% ready** for implementation. The 3 critical issues (P0-3 code mismatch, P0-2 impact overstatement, P1-5 convoluted fix) must be resolved before coding begins. The 4 moderate issues should be addressed to ensure robustness.

Once corrected, this is a solid implementation plan with good test coverage and clear verification steps.

---

**Reviewer**: Kimi CLI
**Date**: April 20, 2026
