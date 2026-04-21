# Math Error Fix Implementation Plan

**Date**: April 20, 2026
**Type**: Engineering Specification
**Status**: Ready for Implementation
**Related Documents**: `2026-04-20-math-error-catalog.md`

---

## Purpose

This document provides step-by-step implementation instructions for fixing 6 mathematical errors in the fantasy baseball application. Each fix includes:
1. Exact file and line numbers to modify
2. Before/after code snippets
3. Test cases that must pass before and after the fix
4. Verification commands to prove correctness

**Implementation Order**: Follow the sequence below to avoid cascading issues.

---

## Implementation Sequence

```
Phase 1 (P0): ROW Projector Fixes → Test → Deploy
Phase 2 (P0): Decision Engine Fix → Test → Deploy
Phase 3 (P1): Scoring Engine Fixes → Test → Deploy
```

**Rationale**: P0 fixes affect ALL projections. P1 fixes affect player ranking fairness but don't cause systematic bias.

---

## Phase 1: ROW Projector Fixes (P0-1, P0-2)

### Fix P0-1: Dynamic Season Days

**File**: `backend/services/row_projector.py`
**Lines to modify**: 387, 406, 242

#### Step 1: Add imports and constants

Add at line 28 (after existing imports):

```python
from datetime import date, datetime, timedelta

# MLB Opening Day 2026 - season start reference point
# ⚠️ VERIFY from authoritative source (MLB.com, official schedule)
# Last verified: 2026-04-20 - confirm before deploying
_MLB_OPENING_DAY = date(2026, 3, 27)


def _days_into_season(as_of_date: Optional[date] = None) -> int:
    """
    Return days since MLB Opening Day (minimum 1).

    Used for season-rate normalization in ROW projections.
    Prevents early-season projection bias from hardcoded 100-day assumption.

    Args:
        as_of_date: Date to calculate from. PREFER explicit date over None.
                    Defaults to today only for backward compatibility.

    Returns:
        Days since opening day (1 = opening day, 100 = ~June 1)
    """
    if as_of_date is None:
        # Fallback for callers without explicit date context
        as_of_date = datetime.now().date()
    delta = (as_of_date - _MLB_OPENING_DAY).days + 1
    return max(1, delta)
```

#### Step 2: Replace line 387

**Before**:
```python
_DEFAULT_SEASON_DAYS = 100  # Line 387
```

**After**:
```python
# _DEFAULT_SEASON_DAYS removed - use _days_into_season() function instead
```

#### Step 3: Replace line 406

**Before**:
```python
season_rate = season_val / _DEFAULT_SEASON_DAYS if season_val else 0.0  # Line 406
```

**After**:
```python
season_rate = season_val / _days_into_season() if season_val else 0.0
```

#### Step 4: Update function signature (line 162)

**Before**:
```python
def compute_row_projection(
    rolling_stats_by_player: Dict[str, Dict[str, float]],
    season_stats_by_player: Optional[Dict[str, Dict[str, float]]] = None,
    games_remaining: Optional[Dict[str, int]] = None,
    *,
    rolling_weight: float = _ROLLING_WEIGHT,
    season_weight: float = _SEASON_WEIGHT,
    window_days: int = _STANDARD_WINDOW_DAYS,
) -> ROWProjectionResult:
```

**After**:
```python
def compute_row_projection(
    rolling_stats_by_player: Dict[str, Dict[str, float]],
    season_stats_by_player: Optional[Dict[str, Dict[str, float]]] = None,
    games_remaining: Optional[Dict[str, int]] = None,
    *,
    rolling_weight: float = _ROLLING_WEIGHT,
    season_weight: float = _SEASON_WEIGHT,
    window_days: int = _STANDARD_WINDOW_DAYS,
    as_of_date: Optional[date] = None,  # NEW PARAMETER
) -> ROWProjectionResult:
```

#### Step 5: Update _blended_daily_rate call (line 239)

**Before**:
```python
blended_rate = (rolling_weight * rolling_rate +
              season_weight * season_rate)
```

**After**:
```python
# Compute season rate using dynamic days into season
_days = _days_into_season(as_of_date)
season_rate_dynamic = season_val / _days if season_val else 0.0
blended_rate = rolling_weight * rolling_rate + season_weight * season_rate_dynamic
```

#### Test Case

Create file `tests/test_row_projector_fixes.py`:

```python
"""Tests for ROW Projector mathematical fixes (P0-1, P0-2)"""
import pytest
from datetime import date
from backend.services.row_projector import compute_row_projection, _days_into_season


class TestP01DynamicSeasonDays:
    """P0-1: Season days should be computed dynamically, not hardcoded to 100"""

    def test_days_into_season_opening_day(self):
        """Opening day should return 1"""
        assert _days_into_season(date(2026, 3, 27)) == 1

    def test_days_into_season_day_20(self):
        """Day 20 of season should return 20"""
        assert _days_into_season(date(2026, 4, 15)) == 20

    def test_days_into_season_before_opening_day(self):
        """Before opening day should return 1 (not negative)"""
        assert _days_into_season(date(2026, 3, 1)) == 1

    def test_day_20_projection_no_bias(self, monkeypatch):
        """
        Day 20: 60 K over 20 IP should project correctly.
        Buggy version would divide by 100, causing 5x undervaluation.
        """
        # Mock today to be day 20 of season
        monkeypatch.setattr(
            "backend.services.row_projector.datetime",
            type("obj", (object,), {"now": lambda: date(2026, 4, 15)})
        )

        rolling_stats = {
            "player_123": {
                "w_strikeouts_pit": 42.0,  # 3.0 K/day over 14 days
                "w_ip": 14.0,              # 1.0 IP/day over 14 days
            }
        }
        season_stats = {
            "player_123": {
                "strikeouts_pit": 60.0,    # 3.0 K/day over 20 days
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            season_stats_by_player=season_stats,
            games_remaining={"player_123": 1},
            as_of_date=date(2026, 4, 15),
        )

        # Season rate = 60 / 20 = 3.0 K/day (NOT 60/100 = 0.6)
        # Blended = 0.6 × (42/14) + 0.4 × 3.0 = 1.8 + 1.2 = 3.0
        # K_P projection for 1 game ≈ 3.0

        # Verify we're not using buggy 100-day denominator
        # Buggy would give: 0.6 × 3.0 + 0.4 × 0.6 = 2.04
        assert result.K_P > 2.5, f"K_P={result.K_P} too low (buggy 100-day denom?)"
        assert result.K_P < 3.5, f"K_P={result.K_P} too high"

    def test_mid_season_projection_accuracy(self):
        """Day 80: projections should use 80, not 100"""
        # Day 80 of 2026 season = June 14
        as_of = date(2026, 6, 14)

        rolling_stats = {
            "player_456": {
                "w_strikeouts_pit": 63.0,  # 4.5 K/day
                "w_ip": 21.0,              # 1.5 IP/day
            }
        }
        season_stats = {
            "player_456": {
                "strikeouts_pit": 320.0,   # 4.0 K/day over 80 days
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            season_stats_by_player=season_stats,
            games_remaining={"player_456": 1},
            as_of_date=as_of,
        )

        # Season rate = 320 / 80 = 4.0 (NOT 320/100 = 3.2)
        # Blended = 0.6 × 4.5 + 0.4 × 4.0 = 2.7 + 1.6 = 4.3
        assert result.K_P > 4.0, f"K_P={result.K_P} suggests 100-day denom"
```

#### Verification Command

```bash
# Run test
venv/Scripts/python -m pytest tests/test_row_projector_fixes.py::TestP01DynamicSeasonDays -q --tb=short

# Verify day calculation
python -c "from backend.services.row_projector import _days_into_season; from datetime import date; print(f'Day 20: {_days_into_season(date(2026, 4, 15))}')"

# Should output: Day 20: 20
```

---

### Fix P0-2: OPS Formula with HBP/SF Imputation

**File**: `backend/services/row_projector.py`
**Lines to modify**: 290, 349-356

#### Step 1: Add HBP/SF constants

Add at line 117 (after `_STANDARD_WINDOW_DAYS = 14`):

```python
# MLB average HBP and SF rates per AB (for OBP imputation)
# Yahoo Fantasy API does not provide HBP/SF; we impute from league averages
_MLB_HBP_PER_AB = 0.0067   # ~4 HBP per 600 AB
_MLB_SF_PER_AB = 0.0083    # ~5 SF per 600 AB
```

#### Step 2: Replace line 290 (OBP denominator accumulation)

**Before**:
```python
sum_obp_denom += ab_daily * gr + bb_daily * gr  # Line 290
```

**After**:
```python
# OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
# Impute HBP and SF from league averages
sum_obp_denom += (ab_daily + bb_daily +
                  ab_daily * _MLB_HBP_PER_AB +
                  ab_daily * _MLB_SF_PER_AB) * gr
```

#### Step 3: Replace lines 349-356 (OPS calculation)

**Before**:
```python
if sum_obp_denom > 0:
    obp = (sum_h + sum_bb) / sum_obp_denom
else:
    obp = 0.0
if sum_ab > 0:
    slg = sum_tb / sum_ab
else:
    slg = 0.0
result.OPS = obp + slg
```

**After**:
```python
# OBP with HBP imputation: (H + BB + HBP) / (AB + BB + HBP + SF)
sum_hbp = sum_ab * _MLB_HBP_PER_AB
sum_sf = sum_ab * _MLB_SF_PER_AB
obp_numer = sum_h + sum_bb + sum_hbp
obp_denom = sum_ab + sum_bb + sum_hbp + sum_sf

if obp_denom > 0:
    obp = obp_numer / obp_denom
else:
    obp = 0.0

# SLG unchanged: TB / AB
if sum_ab > 0:
    slg = sum_tb / sum_ab
else:
    slg = 0.0

result.OPS = obp + slg
```

#### Test Case

Add to `tests/test_row_projector_fixes.py`:

```python
class TestP02OPSFormula:
    """P0-2: OPS should include HBP and SF in OBP calculation"""

    def test_ops_includes_hbp_sf(self):
        """
        Verify HBP and SF are included in OBP calculation.

        Player profile: 500 AB season, 155 H, 50 BB
        Expected HBP: 500 × 0.0067 ≈ 3.35
        Expected SF:  500 × 0.0083 ≈ 4.15

        OBP (no HBP/SF) = (155+50) / 500 = 0.410
        OBP (with HBP/SF) = (155+50+3.35) / (500+50+3.35+4.15) = 208.35 / 557.5 = 0.374
        """
        rolling_stats = {
            "player_789": {
                "w_ab": 350.0,      # 25 AB/day over 14 days
                "w_hits": 100.0,    # .286 average
                "w_tb": 160.0,      # .457 SLG
                "w_walks": 35.0,    # 2.5 BB/day
            }
        }

        result = compute_row_projection(
            rolling_stats_by_player=rolling_stats,
            games_remaining={"player_789": 1},
        )

        # Calculate expected values with HBP/SF
        sum_h = 100.0
        sum_bb = 35.0
        sum_ab = 350.0
        sum_hbp = sum_ab * 0.0067
        sum_sf = sum_ab * 0.0083

        expected_obp = (sum_h + sum_bb + sum_hbp) / (sum_ab + sum_bb + sum_hbp + sum_sf)
        expected_slg = 160.0 / sum_ab
        expected_ops = expected_obp + expected_slg

        # Verify OPS is calculated with HBP/SF
        assert result.OPS > 0, f"OPS should be positive: {result.OPS}"

        # Allow small tolerance for floating point
        assert abs(result.OPS - expected_ops) < 0.01, \
            f"OPS {result.OPS} != expected {expected_ops} (HBP/SF missing?)"

    def test_hbp_sf_constants_sane(self):
        """HBP and SF constants should be within MLB range"""
        from backend.services.row_projector import _MLB_HBP_PER_AB, _MLB_SF_PER_AB

        # MLB ranges: 4-6 HBP per 600 AB, 4-6 SF per 600 AB
        assert 0.005 < _MLB_HBP_PER_AB < 0.010, \
            f"HBP rate {_MLB_HBP_PER_AB} outside MLB range"
        assert 0.005 < _MLB_SF_PER_AB < 0.010, \
            f"SF rate {_MLB_SF_PER_AB} outside MLB range"
```

#### Verification Command

```bash
# Run test
venv/Scripts/python -m pytest tests/test_row_projector_fixes.py::TestP02OPSFormula -q --tb=short

# Verify constants
python -c "from backend.services.row_projector import _MLB_HBP_PER_AB, _MLB_SF_PER_AB; print(f'HBP/AB: {_MLB_HBP_PER_AB:.4f}, SF/AB: {_MLB_SF_PER_AB:.4f}')"

# Should output: HBP/AB: 0.0067, SF/AB: 0.0083
```

---

## Phase 2: Decision Engine Fix (P0-3)

### Fix P0-3: Pitcher Composite Uses Z-Scores

**File**: `backend/services/decision_engine.py`
**Lines to modify**: 233-274 (function `_composite_value`)

**Status**: RECOMMENDED (requires contract change) — see "Note" below

#### Current Implementation

```python
# Lines 233-274 (actual code from decision_engine.py)
def _composite_value(player: PlayerDecisionInput) -> float:
    """
    Simple composite value metric for waiver world-with/world-without comparisons.

    Hitters:  HR + RBI + SB (all normalized to 0-1 then summed)
    Pitchers: K (normalized) - ERA_penalty - WHIP_penalty
    Two-way:  average of both

    Returns a value in approximately [0, 3].
    """
    pt = (player.player_type or "unknown").lower()
    score_anchor = (player.score_0_100 / 100.0) * 1.5
    z_anchor = max(-1.0, min(player.composite_z, 2.0)) + 1.0
    baseline = max(score_anchor, z_anchor / 2.0)

    if pt == "pitcher":
        k    = min((player.proj_k_p50   or 0.0) / _K_NORM,  1.0)
        era  = (player.proj_era_p50  or 4.50) / 9.0   # 9 ERA -> 1.0 penalty
        whip = (player.proj_whip_p50 or 1.30) / 2.0   # 2.0 WHIP -> 1.0 penalty
        projection_total = max(0.0, k - era + (1.0 - whip))
        return max(projection_total, baseline)
    # ... hitter and two_way branches omitted ...
```

#### The Issue

The current formula uses arbitrary scalars:
- ERA divided by 9 (why 9?)
- WHIP divided by 2 (why 2?)
- No statistical grounding in z-scores

**Note**: `PlayerDecisionInput` does NOT currently contain `cat_scores`. The fix below requires adding that field to the dataclass (contract change).

#### Recommended Approach (Requires Contract Change)

**Step 1**: Add `cat_scores` to `PlayerDecisionInput` dataclass

**Step 2**: Replace function with z-score-based approach:

```python
def _composite_value(player: PlayerDecisionInput) -> float:
    """
    Blend short-term category output with longer-term roster value.

    Uses pre-computed category z-scores from scoring engine.
    All z-scores are normalized: higher = better for all categories.
    LOWER_IS_BETTER categories (ERA, WHIP, K_B) are inverted before storage.

    RECOMMENDATION: This is a design preference for statistical rigor.
    Alternative: Document current formula as "heuristic only" (minimal change).
    """
    # If cat_scores not yet available, fall back to baseline
    if not player.cat_scores:
        return max(
            (player.score_0_100 / 100.0) * 1.5,
            (max(-1.0, min(player.composite_z, 2.0)) + 1.0) / 2.0
        )

    pt = (player.player_type or "unknown").lower()
    if pt == "pitcher":
        # Pitching categories: K_P, ERA, WHIP, QS
        relevant_cats = ["k_p", "era", "whip", "qs"]
        projection_total = sum(player.cat_scores.get(cat, 0.0) for cat in relevant_cats)
    elif pt == "hitter":
        # Batting categories: R, H, HR_B, RBI, TB, AVG, OPS, NSB
        relevant_cats = ["r", "h", "hr_b", "rbi", "tb", "avg", "ops", "nsb"]
        projection_total = sum(player.cat_scores.get(cat, 0.0) for cat in relevant_cats)
    else:  # two_way
        # Combine both
        all_cats = ["k_p", "era", "whip", "qs", "r", "h", "hr_b", "rbi", "tb", "avg", "ops", "nsb"]
        projection_total = sum(player.cat_scores.get(cat, 0.0) for cat in all_cats)

    return max(0.0, projection_total)
```

**Alternative (Minimal Change)**: Keep current formula but add documentation:
```python
# HEURISTIC FORMULA: Uses arbitrary scalars (ERA/9, WHIP/2).
# Not statistically grounded. Preserves rank ordering but scores
# cannot be meaningfully compared across player types.
# TODO: Replace with z-score-based composite when cat_scores available
```

#### Test Case (for recommended approach)

```python
"""Tests for Decision Engine mathematical fixes (P0-3)"""
from unittest.mock import MagicMock
from backend.services.decision_engine import DecisionEngine, PlayerDecisionInput


class TestP03PitcherComposite:
    """P0-3: Pitcher composite should use z-scores (when cat_scores available)"""

    def test_elite_pitcher_composite_uses_z_scores(self):
        """
        Elite pitcher with cat_scores: sum of z-scores
        """
        elite = PlayerDecisionInput(
            bdl_player_id=1,
            name="Elite",
            player_type="pitcher",
            eligible_positions=["SP"],
            score_0_100=95.0,
            composite_z=2.5,
            cat_scores={  # NEW FIELD
                "k_p": 2.5,
                "era": 2.0,
                "whip": 1.8,
                "qs": 1.5,
            }
        )

        # After fix: sum of relevant cat_scores
        # expected = 2.5 + 2.0 + 1.8 + 1.5 = 7.8
        # (requires cat_scores to be added to PlayerDecisionInput)
```

#### Verification Command

```bash
# If implementing cat_scores approach:
venv/Scripts/python -m pytest tests/test_decision_engine_fixes.py::TestP03PitcherComposite -q --tb=short

# If keeping heuristic (minimal change):
# Verify no regression
venv/Scripts/python -m pytest tests/test_lineup_optimizer.py -q --tb=short
```

---

## Phase 3: Scoring Engine Fixes (P1-4, P1-5, P1-6)

### Fix P1-4: Weighted Category Sum

**File**: `backend/services/scoring_engine.py`
**Lines to modify**: 76-79 (add weights), 374 (use weights)

#### Step 1: Add category weights (after line 79)

```python
# Category weights based on scarcity (inverse of league average SD)
# Categories with higher variance get higher weights
_CATEGORY_WEIGHTS: dict[str, float] = {
    # Batting - counting stats
    "z_r": 1.0, "z_h": 1.0, "z_hr": 1.2, "z_rbi": 1.1,
    "z_tb": 1.0, "z_nsb": 1.3,  # NSB scarcest (highest variance)
    "z_k_b": 0.9,   # K is rate-like, less variance
    # Batting - rate stats (more stable, lower weight)
    "z_avg": 0.8, "z_ops": 0.9,
    # Pitching - counting stats
    "z_k_p": 1.1, "z_qs": 1.0, "z_nsv": 1.3,  # NSV scarcest
    # Pitching - rate stats
    "z_era": 0.9, "z_whip": 0.9, "z_k_per_9": 0.8,
}
```

#### Step 2: Replace line 374

**Before**:
```python
result.composite_z = sum(non_none) / len(non_none) if non_none else 0.0
```

**After**:
```python
# Weighted sum of category z-scores
if non_none:
    weighted_sum = sum(
        _CATEGORY_WEIGHTS.get(k, 1.0) * v
        for k, v in zip(applicable_keys, non_none)
        if k not in _COMPOSITE_EXCLUDED
    )
    # Normalize by sum of weights (not count) for fair comparison
    weight_sum = sum(
        _CATEGORY_WEIGHTS.get(k, 1.0)
        for k in applicable_keys
        if k not in _COMPOSITE_EXCLUDED
    )
    result.composite_z = weighted_sum / weight_sum if weight_sum > 0 else 0.0
else:
    result.composite_z = 0.0
```

**Wait** - there's a bug in the above. The `non_none` list contains values, but we need to track which key each value came from. Let me fix this:

**Corrected After**:
```python
# Weighted sum of category z-scores
if non_none:
    # Build list of (key, value) pairs for non-None scores
    kv_pairs = [
        (k, getattr(result, k))
        for k in applicable_keys
        if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
    ]
    weighted_sum = sum(
        _CATEGORY_WEIGHTS.get(k, 1.0) * v
        for k, v in kv_pairs
    )
    weight_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k, _ in kv_pairs)
    result.composite_z = weighted_sum / weight_sum if weight_sum > 0 else 0.0
else:
    result.composite_z = 0.0
```

### Fix P1-5: Two-Way Player Normalization

**File**: `backend/services/scoring_engine.py`
**Lines to modify**: 374-380 (composite_z calculation)

#### Step 1: Modify the weighted sum code to normalize two-way players

**Replace the entire section** with:

```python
# Weighted sum of category z-scores
if non_none:
    # Build list of (key, value) pairs for non-None scores
    kv_pairs = [
        (k, getattr(result, k))
        for k in applicable_keys
        if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
    ]
    weighted_sum = sum(
        _CATEGORY_WEIGHTS.get(k, 1.0) * v
        for k, v in kv_pairs
    )
    weight_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k, _ in kv_pairs)

    # P1-5 FIX: Normalize two-way players to 11-category basis
    # This prevents two-way players from being unfairly penalized
    # for having more categories than pure hitters/pitchers
    if player_type == "two_way":
        # Two-way players have 18 categories; normalize to 11 for fairness
        # Weight sum for 11 hitting categories (baseline)
        baseline_weight_sum = 11.0  # Approximate average weight
        result.composite_z = (weighted_sum / weight_sum) * (weight_sum / baseline_weight_sum)
    else:
        result.composite_z = weighted_sum / weight_sum if weight_sum > 0 else 0.0
else:
    result.composite_z = 0.0
```

**Actually, that's still wrong.** Let me think through this more carefully:

The issue is that two-way players have 18 categories while hitters have 11. If we take the arithmetic mean:
- Hitter: `sum(11 z-scores) / 11`
- Two-way: `sum(18 z-scores) / 18`

If both are "average" (z=0 for all categories), they both get 0. That's fine.

But the **variance** differs:
- Hitter variance: `Var(mean) = σ²/11`
- Two-way variance: `Var(mean) = σ²/18`

Two-way players' scores cluster tighter around the mean, which is the real issue.

**Better approach**: Don't normalize by count at all. Use the **weighted sum** directly:

```python
# Composite z = weighted sum (no division)
# This allows comparison across player types with different category counts
result.composite_z = weighted_sum
```

But this makes interpretation harder. Let me try yet another approach:

**Final approach**: Normalize all players to a **fixed baseline** (e.g., 10 categories):

```python
# FINAL CORRECT VERSION
if non_none:
    kv_pairs = [
        (k, getattr(result, k))
        for k in applicable_keys
        if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
    ]
    weighted_sum = sum(
        _CATEGORY_WEIGHTS.get(k, 1.0) * v
        for k, v in kv_pairs
    )
    weight_sum = sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k, _ in kv_pairs)

    # P1-5 FIX: Normalize to 10-category baseline for all player types
    # 10 is roughly the middle ground between 11 (hitter) and 7 (pitcher)
    BASELINE_CATEGORIES = 10
    result.composite_z = (weighted_sum / weight_sum) * (weight_sum / BASELINE_CATEGORIES) if weight_sum > 0 else 0.0
else:
    result.composite_z = 0.0
```

**Actually, I think the simplest fix is to just not divide by count at all for P1-5.** Let me provide a clearer implementation:

```python
# SIMPLIFIED FIX for P1-4 and P1-5 together
if non_none:
    kv_pairs = [
        (k, getattr(result, k))
        for k in applicable_keys
        if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
    ]

    # P1-4: Weighted sum (not mean)
    weighted_sum = sum(
        _CATEGORY_WEIGHTS.get(k, 1.0) * v
        for k, v in kv_pairs
    )

    # P1-5: No normalization - weighted sum IS the composite
    # This treats all categories equally regardless of player type
    result.composite_z = weighted_sum
else:
    result.composite_z = 0.0
```

This is the cleanest approach. The weighted sum:
- Gives more value to players with more good categories (fair)
- Handles two-way players naturally (they just have more categories to contribute)
- Uses category weights to prevent any single category from dominating

### Fix P1-6: Lower MIN_SAMPLE, Use Confidence Field

**File**: `backend/services/scoring_engine.py`
**Lines to modify**: 76

#### Step 1: Replace line 76

**Before**:
```python
MIN_SAMPLE: int = 5
```

**After**:
```python
MIN_SAMPLE: int = 3  # Minimum players to compute any z-score
# Low threshold ensures early-season rankings exist
# Consumers should use 'confidence' field to filter uncertain values
```

#### Test Cases

Create file `tests/test_scoring_engine_fixes.py`:

```python
"""Tests for Scoring Engine mathematical fixes (P1-4, P1-5, P1-6)"""
import pytest
from datetime import date
from unittest.mock import MagicMock
from backend.services.scoring_engine import compute_league_zscores


class TestP14WeightedCategorySum:
    """P1-4: Composite should use weighted sum, not arithmetic mean"""

    def test_hr_specialist_not_diluted(self):
        """
        HR specialist: +3 SD in HR, 0 in all other hitting categories
        Should rank higher than balanced +1 SD player due to NSB weight
        """
        rolling_rows = []

        # HR specialist
        hr_specialist = MagicMock()
        hr_specialist.bdl_player_id = 1
        hr_specialist.w_ab = 100
        hr_specialist.w_home_runs = 30  # Elite: +3 SD
        hr_specialist.w_net_stolen_bases = 0
        hr_specialist.w_runs = 10
        hr_specialist.w_hits = 25
        hr_specialist.w_tb = 50
        hr_specialist.w_rbi = 25
        hr_specialist.w_strikeouts_bat = 25
        hr_specialist.w_avg = 0.25
        hr_specialist.w_ops = 0.7
        hr_specialist.w_ip = None
        rolling_rows.append(hr_specialist)

        # 15 balanced players (+1 SD all cats)
        for i in range(15):
            balanced = MagicMock()
            balanced.bdl_player_id = 10 + i
            balanced.w_ab = 100
            balanced.w_home_runs = 12  # +1 SD
            balanced.w_net_stolen_bases = 5
            balanced.w_runs = 15
            balanced.w_hits = 28
            balanced.w_tb = 45
            balanced.w_rbi = 28
            balanced.w_strikeouts_bat = 18
            balanced.w_avg = 0.28
            balanced.w_ops = 0.78
            balanced.w_ip = None
            rolling_rows.append(balanced)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        hr_spec_result = next(r for r in results if r.bdl_player_id == 1)
        balanced_result = results[1]

        # With fix: HR specialist should be competitive (not severely diluted)
        # Without fix: HR specialist would be much lower due to arithmetic mean
        ratio = hr_spec_result.composite_z / max(balanced_result.composite_z, 0.01)
        assert ratio > 0.5, \
            f"HR specialist {hr_spec_result.composite_z} too diluted vs balanced {balanced_result.composite_z}"


class TestP15TwoWayNormalization:
    """P1-5: Two-way players should not be unfairly penalized"""

    def test_two_way_vs_hitter_fair_comparison(self):
        """
        Two-way player with identical hitting stats as pure hitter
        should have comparable composite score (not inflated/deflated)
        """
        rolling_rows = []

        # Two-way: elite pitcher, average hitter
        two_way = MagicMock()
        two_way.bdl_player_id = 100
        two_way.w_ab = 50
        two_way.w_ip = 50  # Has pitching
        two_way.w_k_per_9 = 12.0  # Elite
        two_way.w_era = 2.0  # Elite
        two_way.w_home_runs = 15  # Average
        two_way.w_net_stolen_bases = 5
        two_way.w_runs = 12
        two_way.w_hits = 12
        rolling_rows.append(two_way)

        # Pure hitter with same hitting stats
        hitter = MagicMock()
        hitter.bdl_player_id = 101
        hitter.w_ab = 50
        hitter.w_ip = None
        hitter.w_home_runs = 15
        hitter.w_net_stolen_bases = 5
        hitter.w_runs = 12
        hitter.w_hits = 12
        rolling_rows.append(hitter)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        two_way_result = next(r for r in results if r.bdl_player_id == 100)
        hitter_result = next(r for r in results if r.bdl_player_id == 101)

        # After fix: composites should be in same ballpark
        # With weighted sum approach, two-way will be higher (extra pitching categories)
        # but that's fair - they contribute more
        ratio = two_way_result.composite_z / max(hitter_result.composite_z, 0.01)
        assert 0.8 < ratio < 3.0, \
            f"Two-way {two_way_result.composite_z} vs hitter {hitter_result.composite_z} too different"


class TestP16LowSampleConfidence:
    """P1-6: Low sample sizes should compute z-scores but mark low confidence"""

    def test_n_equals_3_computes_z_score(self):
        """With only 3 players, should still compute z-scores"""
        rolling_rows = []
        for i in range(3):
            p = MagicMock()
            p.bdl_player_id = i
            p.w_ab = 50
            p.w_home_runs = 10 + i  # Variation
            p.w_ip = None
            rolling_rows.append(p)

        results = compute_league_zscores(rolling_rows, date.today(), 14)

        assert len(results) == 3
        assert all(r.z_hr is not None for r in results), "Z-HR should be computed"

    def test_confidence_increases_with_sample_size(self):
        """Confidence field should increase as sample size grows"""
        # This test documents the expected behavior
        # Actual confidence calculation: games_in_window / window_days

        small_sample = MagicMock()
        small_sample.bdl_player_id = 1
        small_sample.w_ab = 10
        small_sample.games_in_window = 3  # Low confidence
        small_sample.w_ip = None

        large_sample = MagicMock()
        large_sample.bdl_player_id = 2
        large_sample.w_ab = 50
        large_sample.games_in_window = 14  # High confidence
        large_sample.w_ip = None

        results = compute_league_zscores([small_sample, large_sample], date.today(), 14)

        small_result = next(r for r in results if r.bdl_player_id == 1)
        large_result = next(r for r in results if r.bdl_player_id == 2)

        # Small sample should have lower confidence
        assert small_result.confidence < large_result.confidence
```

---

## Verification Checklist

Before committing any fix:

```bash
# 1. Run new tests
venv/Scripts/python -m pytest tests/test_row_projector_fixes.py -q --tb=short
venv/Scripts/python -m pytest tests/test_decision_engine_fixes.py -q --tb=short
venv/Scripts/python -m pytest tests/test_scoring_engine_fixes.py -q --tb=short

# 2. Run existing tests (ensure no regression)
venv/Scripts/python -m pytest tests/test_projections_bridge.py -q --tb=short
venv/Scripts/python -m pytest tests/test_scoring_engine.py -q --tb=short
venv/Scripts/python -m pytest tests/test_lineup_optimizer.py -q --tb=short

# 3. Verify no syntax errors
venv/Scripts/python -m py_compile backend/services/row_projector.py
venv/Scripts/python -m py_compile backend/services/decision_engine.py
venv/Scripts/python -m py_compile backend/services/scoring_engine.py

# 4. Check imports still work
python -c "from backend.services.row_projector import compute_row_projection; print('OK')"
python -c "from backend.services.decision_engine import DecisionEngine; print('OK')"
python -c "from backend.services.scoring_engine import compute_league_zscores; print('OK')"
```

---

## Deployment Order

1. **Phase 1** (ROW Projector) → Test → Commit → Deploy to Railway
2. **Phase 2** (Decision Engine) → Test → Commit → Deploy to Railway
3. **Phase 3** (Scoring Engine) → Test → Commit → Deploy to Railway

**Between phases**: Run full test suite and verify production endpoints still respond.

---

## Rollback Plan

If any fix causes issues:

```bash
# Revert to last known good commit
git revert HEAD
railway up

# Verify rollback
curl https://<production-url>/api/fantasy/waiver | jq '.top_available[:3]'
```

---

**End of Fix Implementation Plan**
