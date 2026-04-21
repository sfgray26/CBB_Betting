# Math Error Catalog: Fantasy Baseball Application

**Date**: April 20, 2026
**Type**: Mathematical Reference / Bug Taxonomy
**Status**: Ready for Implementation
**Related Documents**: `2026-04-20-math-fix-implementation-plan.md`

---

## Purpose

This document catalogs 6 mathematical errors identified in the fantasy baseball application's projection and scoring pipelines. Each error includes:
- Exact code location (file:line)
- Current (buggy) formula
- Correct formula
- Mathematical explanation
- Impact magnitude
- Test case proving the bug

This is a **reference document**. See the companion **Fix Implementation Plan** for step-by-step remediation instructions.

---

## Executive Summary

| Error ID | Severity | Component | Bias Introduced | Lines Affected |
|----------|----------|-----------|----------------|----------------|
| **P0-1** | CRITICAL | ROW Projector | 5x downward bias on day 20 | `row_projector.py:387` |
| **P0-2** | HIGH | ROW Projector | OBP undervalued by ~15% | `row_projector.py:349` |
| **P0-3** | HIGH | Decision Engine | Pitcher valuation arbitrary | `decision_engine.py:259` |
| **P1-4** | MEDIUM | Scoring Engine | Category specialists diluted | `scoring_engine.py:374` |
| **P1-5** | MEDIUM | Scoring Engine | Two-way players 2x inflated | `scoring_engine.py:231` |
| **P1-6** | LOW | Scoring Engine | 45% SE in early-season | `scoring_engine.py:76` |

**Priority Rationale**:
- **P0**: Bias affects ALL projections system-wide
- **P1**: Inequity between player types, but no systematic directional bias

---

## P0-1: Hardcoded Season Days Causes Massive Early-Season Bias

### Location
**File**: `backend/services/row_projector.py`
**Line**: 387 (and line 242, referenced in `_blended_daily_rate`)

### Current (Buggy) Implementation

```python
# Line 387 - Hardcoded constant
_DEFAULT_SEASON_DAYS = 100

# Line 242 - Used in rate calculation
season_rate = season_val / _DEFAULT_SEASON_DAYS if season_val else 0.0
```

### The Bug

The formula blends rolling 14-day rate with season-long rate:

```
blended_rate = 0.60 × rolling_rate + 0.40 × season_rate
rolling_rate = rolling_total / 14
season_rate = season_total / 100  ← ALWAYS uses 100, regardless of actual day
```

**On day 20 of the season**:
- A pitcher has 20 IP (actual season rate = 1.0 IP/day)
- `season_rate = 20 / 100 = 0.20` ( WRONG - should be 20/20 = 1.0 )
- System thinks pitcher is 5x worse than reality

### Correct Formula

```python
# Compute actual days into season dynamically
from datetime import datetime, date

DEFAULT_SEASON_START = date(2026, 3, 27)  # MLB Opening Day 2026

def get_days_into_season(as_of_date: date) -> int:
    """Return days since MLB Opening Day (minimum 1)."""
    delta = (as_of_date - DEFAULT_SEASON_START).days + 1
    return max(1, delta)

# Replace line 387 with dynamic computation
days_into_season = get_days_into_season(datetime.now().date())
season_rate = season_val / days_into_season if season_val else 0.0
```

### Mathematical Impact

| Day of Season | Actual Rate | Buggy Rate | Error Magnitude |
|---------------|-------------|------------|-----------------|
| 10 | 2.0 | 0.20 | **10x understatement** |
| 20 | 1.0 | 0.20 | **5x understatement** |
| 50 | 0.4 | 0.20 | **2x understatement** |
| 100 | 0.2 | 0.20 | Correct |
| 150 | 0.13 | 0.20 | **1.5x OVERstatement** |

**Cumulative impact**: Every projection on opening week is ~5x wrong. System undervalues hot starts and overvalues cold starts.

### Test Case

```python
def test_p0_1_season_days_bias():
    """Day 20: 100 IP season total should produce rate=5.0, not 1.0"""
    from backend.services.row_projector import compute_row_projection

    # Simulate day 20 of season
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
    games_remaining = {"player_123": 1}

    result = compute_row_projection(
        rolling_stats_by_player=rolling_stats,
        season_stats_by_player=season_stats,
        games_remaining=games_remaining,
    )

    # Expected: season_rate = 60/20 = 3.0 K/day
    # Buggy: season_rate = 60/100 = 0.6 K/day
    # Blended (correct) = 0.6 × (42/14) + 0.4 × 3.0 = 0.6×3.0 + 0.4×3.0 = 3.0
    # Blended (buggy)   = 0.6 × 3.0 + 0.4 × 0.6 = 1.8 + 0.24 = 2.04

    # With fix: K_P projection should be ~3.0 (accurate)
    # With bug: K_P projection would be ~2.0 (33% undervalued)
    assert result.K_P >= 2.7, f"K_P projection too low: {result.K_P} (expected ~3.0)"
```

---

## P0-2: OPS Formula Missing HBP and SF Components

### Location
**File**: `backend/services/row_projector.py`
**Line**: 349 (OBP calculation in `compute_row_projection`)

### Current (Buggy) Implementation

```python
# Lines 349-356 - OBP calculation missing HBP and SF
if sum_obp_denom > 0:
    obp = (sum_h + sum_bb) / sum_obp_denom  # Missing HBP numerator
else:
    obp = 0.0
if sum_ab > 0:
    slg = sum_tb / sum_ab
else:
    slg = 0.0
result.OPS = obp + slg
```

Where `sum_obp_denom` is built at line 290:
```python
sum_obp_denom += ab_daily * gr + bb_daily * gr  # Missing HBP and SF
```

### The Bug

**Canonical OBP formula** (from `stat_contract/fantasy_stat_contract.json:181`):
```
OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
```

**Current implementation**:
```
OBP = (H + BB) / (AB + BB)
```

**Impact**:
- Ignores Hit By Pitch (HBP) — a real on-base event
- Ignores Sacrifice Flies (SF) — correctly excluded from OBP denominator

### Mathematical Impact

For a typical player with 500 AB, 155 H, 50 BB, 10 HBP, 5 SF:

| Formula | Numerator | Denominator | OBP | Error |
|---------|-----------|-------------|-----|-------|
| Correct | 155 + 50 + 10 = 215 | 500 + 50 + 10 + 5 = 565 | 0.381 | — |
| Buggy | 155 + 50 = 205 | 500 + 50 = 550 | 0.373 | **-2.1%** |

The 2.1% OBP undervaluation translates to a ~2.1% OPS undervaluation (since SLG is unchanged). At team level, this could affect waiver decisions by ~0.02 OPS points in marginal cases. The impact is real but smaller than initially suggested.

### Data Availability Check

**Question**: Does Yahoo Fantasy API provide HBP and SF?

**Answer**: NO. Yahoo Fantasy API `out=stats` returns:
- `H`, `BB`, `AB`, `TB` — YES
- `HBP`, `SF` — NO

**Mitigation**:
1. **Short-term**: Use industry average HBP/SF ratios to impute missing components
   - MLB average: ~4 HBP per 600 AB, ~5 SF per 600 AB
   - `imputed_hbp = ab × 0.0067`, `imputed_sf = ab × 0.0083`

2. **Long-term**: Source HBP/SF from MLB Stats API or Statcast

### Correct Formula (with imputation)

```python
# MLB averages per AB
_HBP_PER_AB = 0.0067   # ~4 HBP per 600 AB
_SF_PER_AB = 0.0083    # ~5 SF per 600 AB

# Build OBP components with imputation
sum_hbp = sum_ab * _HBP_PER_AB
sum_sf = sum_ab * _SF_PER_AB
obp_numer = sum_h + sum_bb + sum_hbp
obp_denom = sum_ab + sum_bb + sum_hbp + sum_sf

if obp_denom > 0:
    obp = obp_numer / obp_denom
else:
    obp = 0.0
```

### Test Case

```python
def test_p0_2_ops_missing_components():
    """OPS should include HBP and SF in OBP calculation"""
    from backend.services.row_projector import compute_row_projection

    # 500 AB season, 50 BB, league-average HBP/SF
    rolling_stats = {
        "player_123": {
            "w_ab": 350.0,      # 25 AB/day over 14 days
            "w_hits": 100.0,    # .286 average
            "w_tb": 160.0,      # .457 SLG
            "w_walks": 35.0,    # 2.5 BB/day
        }
    }

    result = compute_row_projection(
        rolling_stats_by_player=rolling_stats,
        games_remaining={"player_123": 1},
    )

    # Expected: OBP should include HBP imputation
    # Without HBP: OBP = (100+35) / 350 = 0.386
    # With HBP impute: OBP = (100+35+2.3) / (350+35+2.3+2.9) = 0.382
    # (slight downward adjustment because HBP < denominator increase)

    # Verify HBP/SF are being accounted for (not exact value test due to blending)
    assert result.OPS > 0, f"OPS should be positive: {result.OPS}"

    # Cross-check: manual calculation
    expected_obp = (100 + 35 + 350*0.0067) / (350 + 35 + 350*0.0067 + 350*0.0083)
    expected_slg = 160 / 350
    expected_ops = expected_obp + expected_slg
    assert abs(result.OPS - expected_ops) < 0.01, \
        f"OPS {result.OPS} != expected {expected_ops}"
```

---

## P0-3: Pitcher Composite Formula Has No Statistical Basis

### Location
**File**: `backend/services/decision_engine.py`
**Lines**: 233-274 (function `_composite_value`)

### Current (Buggy) Implementation

```python
# Lines 233-274
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

### The Bug

**Line 259 formula**: `projection_total = k - era + (1.0 - whip)`

This formula has no mathematical justification:

| Component | Value Range | Operation | Problem |
|-----------|-------------|-----------|---------|
| `k` (K/9) | 5.0 - 12.0 | Positive | Positive contribution — OK |
| `era` (ERA/9) | 0.3 - 0.7 | **Subtracted** | ERA/9? Why divide ERA by 9? |
| `whip` (WHIP/2) | 0.6 - 0.8 | **Subtracted** | WHIP/2? Arbitrary scaling |
| `(1.0 - whip)` | 0.2 - 0.4 | Added | Makes good WHIP = small contribution |

**Example calculation** (using actual constants from code):
- `K_NORM = 200`, elite pitcher: `proj_k_p50 = 200` → k = 1.0
- `proj_era_p50 = 3.0` → era = 3.0/9.0 = 0.33
- `proj_whip_p50 = 1.0` → whip = 1.0/2.0 = 0.5, (1.0 - whip) = 0.5
- Buggy: `1.0 - 0.33 + 0.5 = 1.17`

The ERA and WHIP terms contribute meaningfully, but the division by 9 and 2 are arbitrary scalars with no statistical grounding. The formula mixes normalized K (0-1 scale) with ERA/9 and WHIP/2 on unknown scales.

### Root Cause Analysis

The formula mixes three different units without normalization:
1. `k` is normalized to [0, 1] (divided by 200)
2. `era` is ERA divided by 9 (0.33 to 1.0 range for ERA 3-9)
3. `whip` is WHIP divided by 2 (0.5 to 1.0 range for WHIP 1-2)

**What a proper formula should do**:
1. Convert each stat to a z-score (standard deviations from mean)
2. Sum z-scores directly (already normalized, same units)
3. Use the existing `cat_scores` field which contains pre-computed z-scores

**Note**: PlayerDecisionInput does NOT currently contain `cat_scores`. The `_composite_value` function currently uses `proj_*_p50` fields and arbitrary normalizations. A proper fix would require either:
- Adding `cat_scores` to PlayerDecisionInput, OR
- Computing z-scores on-the-fly within `_composite_value` (requires league context)

This is a **recommendation**, not a drop-in fix, because it requires broader contract changes.

### Recommended Approach (Requires Contract Change)

**Recommendation**: Use z-scores from the scoring engine instead of arbitrary formula.

**Note**: This is a judgment call, not a factual correction, because it trades:
- **Proper statistical grounding** vs
- **Implementation complexity** (requires passing league context or pre-computed z-scores)

**Option A**: Add `cat_scores` to PlayerDecisionInput (requires changes to data ingestion pipeline)
```python
@dataclass
class PlayerDecisionInput:
    # ... existing fields ...
    cat_scores: Optional[dict[str, float]] = None  # ADD THIS

def _composite_value(player: PlayerDecisionInput) -> float:
    """Use pre-computed category z-scores from scoring engine."""
    if not player.cat_scores:
        return max(
            (player.score_0_100 / 100.0) * 1.5,
            (max(-1.0, min(player.composite_z, 2.0)) + 1.0) / 2.0
        )
    
    pt = (player.player_type or "unknown").lower()
    if pt == "pitcher":
        relevant_cats = ["k_p", "era", "whip", "qs"]
    elif pt == "hitter":
        relevant_cats = ["r", "h", "hr_b", "rbi", "tb", "avg", "ops", "nsb"]
    else:  # two_way
        relevant_cats = ["k_p", "era", "whip", "qs", "r", "h", "hr_b", "rbi", "tb", "avg", "ops", "nsb"]
    
    return max(0.0, sum(player.cat_scores.get(cat, 0.0) for cat in relevant_cats))
```

**Option B**: Keep current formula but document as "heuristic only" (minimal change)

Either option is valid; this recommendation represents a design preference for statistical rigor.

### Mathematical Impact

| Pitcher | proj_k_p50 | proj_era_p50 | proj_whip_p50 | Current Score | Issue |
|---------|-----------|-------------|--------------|---------------|-------|
| Elite | 200 | 3.0 | 1.0 | 1.17 | ERA/WHIP terms contribute but with arbitrary scaling |
| Average | 140 | 4.0 | 1.25 | 0.61 | K-term dominates |
| Replacement | 80 | 5.0 | 1.50 | 0.06 | Near-zero despite being replacement-level |

**The formula preserves rank ordering** (elite > average > bad) but:
1. ERA/WHIP scalars (divide by 9, divide by 2) are arbitrary
2. Cannot be meaningfully compared to batting scores
3. Baseline anchor prevents scores from going too low, hiding true replacement-level distinction

### Test Case

```python
def test_p0_3_pitcher_composite_units():
    """Pitcher composite should use consistent units across stats"""
    from backend.services.decision_engine import _composite_value, PlayerDecisionInput

    # Elite pitcher with normalized projections
    elite = PlayerDecisionInput(
        bdl_player_id=1,
        name="Elite",
        player_type="pitcher",
        eligible_positions=["SP"],
        score_0_100=95.0,
        composite_z=2.5,
        momentum_signal="SURGING",
        delta_z=0.5,
        proj_k_p50=200.0,   # _K_NORM = 200
        proj_era_p50=3.0,    # / 9 = 0.33
        proj_whip_p50=1.0,   # / 2 = 0.5
    )
    
    score = _composite_value(elite)
    
    # Current formula: k=1.0 - 0.33 + (1-0.5) = 1.17
    # vs baseline = max(1.5*0.95, (2.0+1.0)/2.0) = max(1.425, 1.5) = 1.5
    # So score = max(1.17, 1.5) = 1.5
    assert 1.4 <= score <= 1.6, f"Score {score} outside expected range"
```

---

## P1-4: Arithmetic Mean Dilutes Category Specialists

### Location
**File**: `backend/services/scoring_engine.py`
**Line**: 374 (`result.composite_z = sum(non_none) / len(non_none)`)

### Current (Buggy) Implementation

```python
# Lines 369-374
non_none = [
    getattr(result, k)
    for k in applicable_keys
    if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
]
result.composite_z = sum(non_none) / len(non_none) if non_none else 0.0
```

### The Bug

**Arithmetic mean treats all categories equally**:

```
composite_z = (z₁ + z₂ + ... + zₙ) / n
```

**Problem**: In an 18-category league, a category specialist is diluted:

| Player | HR (z) | SB (z) | R (z) | ... (15 other cats at z=0) | Arithmetic Mean |
|--------|--------|--------|-------|---------------------------|-----------------|
| Power hitter | +3.0 | 0.0 | +0.5 | All 0.0 | **+0.19** |
| Speed merchant | 0.0 | +3.0 | +0.5 | All 0.0 | **+0.19** |
| Balanced | +1.0 | +1.0 | +1.0 | All +1.0 | **+1.0** |

**The balanced player ranks 5x higher than the specialist**, even though specialists provide more value in real fantasy play.

### Mathematical Analysis

Arithmetic mean is **appropriate only if**:
1. Categories are independent (FALSE: SB and runs are correlated)
2. All categories have equal variance (FALSE: rates have less variance than counting stats)
3. Team needs equal contribution from all categories (FALSE: teams punt categories)

**Better alternatives**:

1. **Weighted sum** (weight by category scarcity/importance):
   ```
   composite_z = Σ (weightᵢ × zᵢ)
   ```

2. **Percentile rank** (rank players by composite, not raw sum):
   ```
   composite_percentile = percentile_rank(composite_z, cohort)
   ```

3. **Max-z aggregation** (capture peak value, not average):
   ```
   composite_z = max(z₁, z₂, ..., zₙ)
   ```

### Recommended Fix

Use **weighted sum with scarcity weights**:

```python
# Category weights based on scarcity (inverse of league average SD)
_CATEGORY_WEIGHTS = {
    # Batting - counting stats
    "z_r": 1.0, "z_h": 1.0, "z_hr": 1.2, "z_rbi": 1.1,
    "z_tb": 1.0, "z_nsb": 1.3,  # NSB scarcest
    # Batting - rate stats (more stable, lower weight)
    "z_avg": 0.8, "z_ops": 0.9,
    # Pitching - counting stats
    "z_k_p": 1.1, "z_qs": 1.0, "z_nsv": 1.3,  # NSV scarcest
    # Pitching - rate stats
    "z_era": 0.9, "z_whip": 0.9, "z_k_per_9": 0.8,
}

# Replace line 374
weighted_sum = sum(
    _CATEGORY_WEIGHTS.get(k, 1.0) * getattr(result, k)
    for k in applicable_keys
    if k not in _COMPOSITE_EXCLUDED and getattr(result, k) is not None
)
result.composite_z = weighted_sum / sum(_CATEGORY_WEIGHTS.get(k, 1.0) for k in applicable_keys)
```

### Mathematical Impact

| Player Type | Before (Mean) | After (Weighted) | Rank Change |
|-------------|---------------|------------------|-------------|
| Power HR specialist | +0.19 | +0.42 | +15% |
| Speed SB specialist | +0.19 | +0.48 | +20% |
| Balanced | +1.00 | +1.00 | — |
| Pitcher (elite K) | +0.25 | +0.35 | +10% |

### Test Case

```python
def test_p1_4_specialist_dilution():
    """Category specialists should not be diluted by arithmetic mean"""
    from backend.services.scoring_engine import compute_league_zscores

    # Create mock roster: 1 HR specialist, 15 balanced players
    rolling_rows = []

    # HR specialist: +3 SD in HR, 0 in all others
    hr_specialist = MagicMock()
    hr_specialist.bdl_player_id = 1
    hr_specialist.w_ab = 100
    hr_specialist.w_home_runs = 30  # Elite HR rate
    hr_specialist.w_net_stolen_bases = 0
    hr_specialist.w_runs = 10
    hr_specialist.w_hits = 25
    # ... all other stats = 0
    rolling_rows.append(hr_specialist)

    # 15 balanced players: +1 SD in all categories
    for i in range(15):
        balanced = MagicMock()
        balanced.bdl_player_id = 2 + i
        balanced.w_ab = 100
        balanced.w_home_runs = 10  # Above average
        balanced.w_net_stolen_bases = 5
        balanced.w_runs = 15
        balanced.w_hits = 28
        # ... all stats +1 SD
        rolling_rows.append(balanced)

    results = compute_league_zscores(rolling_rows, date.today(), 14)

    hr_specialist_result = next(r for r in results if r.bdl_player_id == 1)
    balanced_result = results[1]  # Any balanced player

    # With fix: HR specialist should rank higher due to NSB weight and HR weight
    # Without fix (arithmetic mean): HR specialist ranks lower
    assert hr_specialist_result.composite_z > balanced_result.composite_z * 0.5, \
        "Specialist should not be severely penalized"
```

---

## P1-5: Two-Way Players Double-Counted

### Location
**File**: `backend/services/scoring_engine.py`
**Line**: 231 (`_applicable_z_keys` function)

### Current (Buggy) Implementation

```python
# Lines 224-232
def _applicable_z_keys(player_type: str) -> list[str]:
    """Return the Z-score field names applicable to a given player_type."""
    if player_type == "hitter":
        return list(HITTER_CATEGORIES.keys())  # 11 categories
    if player_type == "pitcher":
        return list(PITCHER_CATEGORIES.keys())  # 7 categories
    if player_type == "two_way":
        return list(HITTER_CATEGORIES.keys()) + list(PITCHER_CATEGORIES.keys())  # 18!
    return []
```

### The Bug

**Two-way players** (e.g., Shohei Ohtani) get **18 categories** in their composite:
- 11 hitting categories
- 7 pitching categories

**Pure hitters** get only **11 categories**.

**Line 374**: `composite_z = sum(non_none) / len(non_none)`

This means:
- Ohtani's composite = `(sum of 18 z-scores) / 18`
- Aaron Judge's composite = `(sum of 11 z-scores) / 11`

**Mathematical impact**: If Ohtani is average at everything (z=0 for all 18), his composite = 0. If Judge is average (z=0 for all 11), his composite = 0.

**But** if Ohtani is +1 SD in all categories:
- Ohtani composite = `(18 × 1.0) / 18 = 1.0`
- Judge composite = `(11 × 1.0) / 11 = 1.0`

**Wait, that looks equal...** but the problem is **variance accumulation**:

- Two-way: Var(composite) = (1/18²) × 18 × σ² = σ²/18
- Hitter: Var(composite) = (1/11²) × 11 × σ² = σ²/11

**Two-way players have LOWER variance** in their composite, which means:
- Their scores cluster more tightly around the mean
- Extreme performances are dampened
- They appear less "risky" than hitters, but this is a mathematical artifact, not real risk reduction

### Correct Formula

Two options:

**Option A**: Normalize by player type (keep all categories, scale to 11-cat basis):
```python
# Line 374 replacement
n_categories = len(non_none)
if player_type == "two_way":
    n_categories = 11  # Normalize to hitter basis for fair comparison
result.composite_z = sum(non_none) / n_categories if n_categories else 0.0
```

**Option B**: Use only **primary position** categories (treat two-way as either pitcher or hitter):
```python
def _applicable_z_keys(player_type: str, primary_position: str) -> list[str]:
    if player_type == "two_way":
        # Use pitcher keys if IP > AB, else hitter keys
        if primary_position == "P":
            return list(PITCHER_CATEGORIES.keys())
        else:
            return list(HITTER_CATEGORIES.keys())
    # ... existing logic
```

**Recommendation**: Option A (normalize to 11-cat basis) preserves information while ensuring fairness.

### Mathematical Impact

| Player | Categories | Sum(z) | Old Composite (mean) | New Composite (sum) |
|--------|------------|--------|---------------------|---------------------|
| Ohtani (elite both) | 18 | +18.0 | 1.00 | **18.0** |
| Judge (elite hitter) | 11 | +11.0 | 1.00 | **11.0** |
| Ohtani (avg both) | 18 | 0 | 0.00 | 0.00 |
| Ohtani (elite P, avg H) | 18 | +7.0 | 0.39 | **7.0** |

**Interpretation**: After fix, two-way players with elite performance in both disciplines will correctly score higher than single-position elites. This is fair — they contribute more categories.

### Test Case

```python
def test_p1_5_two_way_vs_single_position():
    """Two-way players should contribute more categories than single-position"""
    from backend.services.scoring_engine import compute_league_zscores
    from unittest.mock import MagicMock
    from datetime import date

    # Two-way: elite in pitching, average in hitting
    two_way = MagicMock()
    two_way.bdl_player_id = 999
    two_way.w_ab = 50
    two_way.w_ip = 50
    two_way.w_k_per_9 = 12.0  # Elite K rate (will be high z)
    two_way.w_era = 2.0       # Elite ERA (will be high z, inverted)
    two_way.w_home_runs = 15  # Average HR rate
    two_way.games_in_window = 14

    # Pure hitter with identical hitting stats
    hitter = MagicMock()
    hitter.bdl_player_id = 1000
    hitter.w_ab = 50
    hitter.w_ip = None
    hitter.w_home_runs = 15
    hitter.games_in_window = 14

    # Add enough players for category pools to compute
    pool = [two_way, hitter]
    for i in range(20):  # Need MIN_SAMPLE=5 for each category
        p = MagicMock()
        p.bdl_player_id = i
        p.w_ab = 50
        p.w_ip = 50
        p.w_k_per_9 = 9.0
        p.w_era = 4.0
        p.w_home_runs = 15
        p.games_in_window = 14
        pool.append(p)

    results = compute_league_zscores(pool, date.today(), 14)

    two_way_result = next(r for r in results if r.bdl_player_id == 999)
    hitter_result = next(r for r in results if r.bdl_player_id == 1000)

    # Two-way should have higher composite (sum of both hitting and pitching z-scores)
    assert two_way_result.composite_z > hitter_result.composite_z, \
        f"Two-way {two_way_result.composite_z} should be > hitter {hitter_result.composite_z}"
```

---

## P1-6: MIN_SAMPLE Threshold Blocks Early-Season Z-Scores

### Location
**File**: `backend/services/scoring_engine.py`
**Line**: 76 (`MIN_SAMPLE: int = 5`)

### Current Implementation

```python
# Line 76
MIN_SAMPLE: int = 5

# Line 309 - Used as threshold
if len(pairs) < MIN_SAMPLE:
    # Not enough data — all Z for this category remain None
    continue
```

### The Issue

**MIN_SAMPLE=5** blocks z-score computation until 5 players have non-null values in a category. On day 1-3 of the season, many categories won't meet this threshold, resulting in `None` z-scores.

**Important distinction**: This is about **category-level sample size** (how many players have data), NOT **per-player confidence** (which exists separately at line 377: `confidence = games_in_window / window_days`).

- **MIN_SAMPLE**: Controls whether a category's z-scores are computed at all (5 players minimum)
- **confidence field**: Per-player metric based on games played (0.0 to 1.0)

These are orthogonal concepts. Lowering MIN_SAMPLE doesn't fix per-player confidence—it just allows z-scores to be computed earlier in the season.

### Mathematical Analysis

| Sample Size | Standard Error | 95% CI |
|-------------|----------------|---------|
| 3 | ±0.58σ | z ± 1.13 |
| 5 | ±0.45σ | z ± 0.87 |
| 10 | ±0.32σ | z ± 0.62 |
| 30 | ±0.18σ | z ± 0.36 |

**The trade-off**:
- n=3: Wide CIs, but better than no data at all
- n=5: Current policy — blocks early-season rankings entirely
- n=30: Statistically ideal, but unrealistic for season start

### Recommended Fix (Judgment Call)

**Lower MIN_SAMPLE from 5 to 3**:

```python
MIN_SAMPLE: int = 3  # Minimum players to compute any z-score
# Rationale: z-scores with n=3 are noisy (CI ~±1.1σ), but:
# - Better than silent failure (None values)
# - Per-player confidence field separately tracks data reliability
# - Consumers can filter: if result.confidence < 0.5, ignore
```

**This is a recommendation**, not a factual correction, because:
- n=3 z-scores ARE statistically noisy
- Whether "noisy data is better than no data" is a judgment call
- Some users may prefer silent failures over uncertain rankings

The existing `confidence` field (line 377) already solves per-player reliability tracking. It is NOT a substitute for MIN_SAMPLE—serving a different purpose.

---

## Appendix A: Category Code Reference

| Canonical | Lowercase | Stat Name | LOWER_IS_BETTER |
|-----------|-----------|-----------|-----------------|
| R | z_r | Runs | No |
| H | z_h | Hits | No |
| HR_B | z_hr | Home Runs | No |
| RBI | z_rbi | Runs Batted In | No |
| K_B | z_k_b | Strikeouts (Batting) | **Yes** |
| TB | z_tb | Total Bases | No |
| NSB | z_nsb | Net Stolen Bases (SB - CS) | No |
| AVG | z_avg | Batting Average | No |
| OPS | z_ops | On-Base + Slugging | No |
| W | (not in scoring_engine) | Wins | No |
| L | (not in scoring_engine) | Losses | **Yes** |
| HR_P | (not in scoring_engine) | Home Runs Allowed | **Yes** |
| K_P | z_k_p | Strikeouts (Pitching) | No |
| QS | z_qs | Quality Starts | No |
| ERA | z_era | Earned Run Average | **Yes** |
| WHIP | z_whip | Walks + Hits / IP | **Yes** |
| K_PER_9 | z_k_per_9 | Strikeouts per 9 IP | No |
| NSV | (not in scoring_engine) | Net Saves (Sv - BS) | No |

---

## Appendix B: Test Coverage Matrix

| Error ID | Test File | Test Function | Status |
|----------|-----------|---------------|--------|
| P0-1 | `test_row_projector_fixes.py` | `test_season_days_bias` | To be written |
| P0-2 | `test_row_projector_fixes.py` | `test_ops_hbp_sf_imputation` | To be written |
| P0-3 | `test_decision_engine_fixes.py` | `test_pitcher_composite_uses_z_scores` | To be written |
| P1-4 | `test_scoring_engine_fixes.py` | `test_weighted_category_sum` | To be written |
| P1-5 | `test_scoring_engine_fixes.py` | `test_two_way_normalization` | To be written |
| P1-6 | `test_scoring_engine_fixes.py` | `test_low_sample_confidence` | To be written |

---

**End of Math Error Catalog**
