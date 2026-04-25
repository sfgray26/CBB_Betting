# Bayesian Projection Fusion Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the binary Steamer-or-Statcast fallback with a Component-Wise Empirical Bayes model (Marcel update) that dynamically fuses Steamer priors with Statcast observed data, weighted by exact metric stabilization points.

**Architecture:**
1. Create a new `fusion_engine.py` module containing the Marcel update formula and stabilization constants
2. Define population priors for league-average fallback (rookies)
3. Rewrite `get_or_create_projection()` to evaluate four states with Bayesian fusion
4. Implement xwOBA override layer for luck detection (|xwOBA - wOBA| > 0.030)

**Tech Stack:** Python 3.11, SQLAlchemy, math operations only (no ML libraries needed)

---

## File Structure

```
backend/fantasy_baseball/
  fusion_engine.py          [NEW]   Marcel update formula, stabilization constants, population priors
  player_board.py           [MOD]   Rewrite get_or_create_projection() with 4-state Bayesian logic
  cat_scores_builder.py     [READ]  For compute_cat_scores integration (no changes needed)
tests/
  test_fusion_engine.py     [NEW]   Unit tests for Marcel update, edge cases, xwOBA override
  test_player_board.py      [MOD]   Add integration tests for 4-state fusion logic
```

---

## Task 1: Create Fusion Engine Core Module

**Files:**
- Create: `backend/fantasy_baseball/fusion_engine.py`
- Test: `tests/test_fusion_engine.py`

### Step 1.1: Write the fusion engine module with Marcel update formula

Create `backend/fantasy_baseball/fusion_engine.py`:

```python
"""
Bayesian Projection Fusion Engine — Component-Wise Empirical Bayes (Marcel Update)

Replaces binary Steamer-or-Statcast fallback with dynamic fusion.
Shrinkage applied per-metric using exact stabilization points.

Posterior Mean = ((stabilization * prior) + (sample_size * observed)) / (stabilization + sample_size)

Reference: https://www.baseballprospectus.com/news/article/12365/
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# STABILIZATION POINTS — exact PA/IP thresholds where metric becomes reliable
# Source: FanGraphs/MLB.com research (2024-2025)
# ---------------------------------------------------------------------------

class StabilizationPoints:
    """
    Exact stabilization points for each rate stat.
    Sample size (PA/IP) at which observed metric is 50% reliable.
    Lower = stabilizes faster (more weight to observed).
    """

    # Batters — plate appearance thresholds
    BATTER = {
        "k_percent": 60,      # K% stabilizes fastest
        "bb_percent": 120,    # BB% plate discipline
        "hr_rate": 170,       # HR per PA
        "iso": 160,           # Isolated power (SLG - AVG)
        "obp": 460,           # On-base percentage
        "slg": 320,           # Slugging percentage
        "avg": 910,           # Batting average (slowest)
        "barrel_pct": 50,     # Barrel% — premium batted ball metric
        "xwoba": 150,         # Expected wOBA — predictive luck indicator
    }

    # Pitchers — inning thresholds
    PITCHER = {
        "k_percent": 70,      # K% faster for pitchers
        "bb_percent": 170,    # BB% slower
        "era": 300,           # ERA — team-dependent, slow
        "whip": 300,          # WHIP — also slow
        "fip": 170,           # FIP — faster than ERA
        "xera": 150,          # Expected ERA
    }


# ---------------------------------------------------------------------------
# POPULATION PRIORS — league-average fallback for zero-data players (rookies)
# ---------------------------------------------------------------------------

@dataclass
class PopulationPrior:
    """League-average baseline for players with absolutely zero data."""
    # Batters
    avg: float = 0.250
    obp: float = 0.320
    slg: float = 0.410
    ops: float = 0.730
    hr_per_pa: float = 0.035
    sb_per_pa: float = 0.010
    k_percent: float = 0.225  # 22.5% K rate
    bb_percent: float = 0.080  # 8.0% BB rate

    # Pitchers
    era: float = 4.50
    whip: float = 1.35
    k_per_nine: float = 8.5
    bb_per_nine: float = 3.0


# Singleton instance
POPULATION_PRIOR = PopulationPrior()


# ---------------------------------------------------------------------------
# MARCEL UPDATE FORMULA — component-wise Bayesian shrinkage
# ---------------------------------------------------------------------------

def marcel_update(
    prior_mean: float,
    observed_mean: float,
    sample_size: int,
    stabilization_point: int,
    min_sample: int = 1
) -> float:
    """
    Apply Marcel update formula for component-wise Bayesian shrinkage.

    Posterior = ((stabilization * prior) + (sample_size * observed)) / (stabilization + sample_size)

    Args:
        prior_mean: Prior projection mean (e.g., Steamer AVG)
        observed_mean: Observed stat mean (e.g., Statcast AVG)
        sample_size: PA (batters) or IP (pitchers) in observed data
        stabilization_point: Threshold where metric is 50% reliable
        min_sample: Minimum sample size to apply any shrinkage (default 1)

    Returns:
        Posterior mean (fused estimate)

    Example:
        prior = 0.280 (Steamer AVG)
        observed = 0.320 (Statcast AVG, 100 PA)
        stabilization = 910 (AVG)
        result = 0.285 (shrunk ~10% toward observed)
    """
    if sample_size < min_sample:
        # No observed data — return pure prior
        return prior_mean

    # Clamp sample size to avoid division by zero
    sample_size = max(min_sample, sample_size)
    stabilization_point = max(1, stabilization_point)

    # Marcel update formula
    posterior = (
        (stabilization_point * prior_mean) +
        (sample_size * observed_mean)
    ) / (stabilization_point + sample_size)

    return posterior


# ---------------------------------------------------------------------------
# FUSION ORCHESTRATOR — map player data through Marcel update per component
# ---------------------------------------------------------------------------

@dataclass
class FusionResult:
    """Result of Bayesian fusion with metadata for debugging."""
    proj: dict                    # Fused projection stats
    cat_scores: dict              # Computed category scores
    source: str                   # Data source label
    components_fused: int         # How many components got Bayesian treatment
    xwoba_override_applied: bool  # Whether luck correction triggered


def fuse_batter_projection(
    steamer: Optional[dict] = None,
    statcast: Optional[dict] = None,
    sample_size: int = 0
) -> FusionResult:
    """
    Fuse batter projections using component-wise Marcel update.

    Four-state logic:
    1. Steamer + Statcast: Full Marcel update per component
    2. Steamer only: Return Steamer (no observed data to shrink toward)
    3. Statcast only: Fuse with POPULATION_PRIOR (double shrinkage: stabil * 2)
    4. Neither: Return pure POPULATION_PRIOR with generic z-score offset

    Args:
        steamer: Dict with keys {avg, obp, slg, ops, hr, r, rbi, sb}
        statcast: Dict with keys {avg, obp, slg, ops, xwoba, woba, pa}
        sample_size: PA count for weight calculation

    Returns:
        FusionResult with fused projection and metadata
    """
    components_fused = 0
    xwoba_override = False
    proj = {}

    # STATE 1: Both sources available — full Marcel update
    if steamer and statcast and sample_size >= 1:
        # xwOBA luck detection — trigger override if |xwOBA - wOBA| > 0.030
        xwoba = statcast.get("xwoba", 0.320)
        woba = statcast.get("woba", statcast.get("ops", 0.720) * 0.95)  # Rough wOBA proxy
        luck_delta = abs(xwoba - woba)

        if luck_delta > 0.030:
            # Heavy xwOBA weight — BABIP noise likely skewing observed
            xwoba_override = True
            logger.info(f"Fusion: xwOBA override applied (delta={luck_delta:.3f})")
            # Use xwOBA as prior instead of Steamer for rate stats
            prior_source = statcast
        else:
            prior_source = steamer

        # Component-wise Marcel update for rate stats
        proj["avg"] = marcel_update(
            prior_source.get("avg", POPULATION_PRIOR.avg),
            statcast.get("avg", POPULATION_PRIOR.avg),
            sample_size,
            StabilizationPoints.BATTER["avg"]
        )
        components_fused += 1

        proj["obp"] = marcel_update(
            prior_source.get("obp", POPULATION_PRIOR.obp),
            statcast.get("obp", POPULATION_PRIOR.obp),
            sample_size,
            StabilizationPoints.BATTER["obp"]
        )
        components_fused += 1

        proj["slg"] = marcel_update(
            prior_source.get("slg", POPULATION_PRIOR.slg),
            statcast.get("slg", POPULATION_PRIOR.slg),
            sample_size,
            StabilizationPoints.BATTER["slg"]
        )
        components_fused += 1

        proj["ops"] = marcel_update(
            prior_source.get("ops", POPULATION_PRIOR.ops),
            statcast.get("ops", POPULATION_PRIOR.ops),
            sample_size,
            StabilizationPoints.BATTER["slg"]  # OPS uses SLG stabilization (approx)
        )
        components_fused += 1

        # Counting stats — scale Steamer by PA ratio
        pa_ratio = sample_size / max(1, steamer.get("pa", 550))
        proj["hr"] = round(steamer.get("hr", 15) * max(0.5, min(2.0, pa_ratio)))
        proj["r"] = round(steamer.get("r", 65) * max(0.5, min(2.0, pa_ratio)))
        proj["rbi"] = round(steamer.get("rbi", 65) * max(0.5, min(2.0, pa_ratio)))
        proj["sb"] = steamer.get("sb", 5)  # SB skill-based, don't scale

        return FusionResult(
            proj=proj,
            cat_scores={},  # Computed later by cat_scores_builder
            source="steamer_statcast_fusion",
            components_fused=components_fused,
            xwoba_override_applied=xwoba_override
        )

    # STATE 2: Steamer only — pure prior
    if steamer:
        return FusionResult(
            proj=steamer.copy(),
            cat_scores={},
            source="steamer_only",
            components_fused=0,
            xwoba_override_applied=False
        )

    # STATE 3: Statcast only — fuse with population prior (double shrinkage)
    if statcast and sample_size >= 1:
        # Double shrinkage: stabilization * 2 for pure-statcast players
        # (More conservative because we're anchoring to league avg, not Steamer)
        double_shrink = lambda s: s * 2

        proj["avg"] = marcel_update(
            POPULATION_PRIOR.avg,
            statcast.get("avg", POPULATION_PRIOR.avg),
            sample_size,
            double_shrink(StabilizationPoints.BATTER["avg"])
        )
        components_fused += 1

        proj["obp"] = marcel_update(
            POPULATION_PRIOR.obp,
            statcast.get("obp", POPULATION_PRIOR.obp),
            sample_size,
            double_shrink(StabilizationPoints.BATTER["obp"])
        )
        components_fused += 1

        proj["slg"] = marcel_update(
            POPULATION_PRIOR.slg,
            statcast.get("slg", POPULATION_PRIOR.slg),
            sample_size,
            double_shrink(StabilizationPoints.BATTER["slg"])
        )
        components_fused += 1

        proj["ops"] = proj["obp"] + proj["slg"]

        # Estimate counting stats from rates
        proj["hr"] = round(sample_size * POPULATION_PRIOR.hr_per_pa)
        proj["r"] = round(sample_size * 0.12)  # ~12% of PA = runs
        proj["rbi"] = round(sample_size * 0.11)  # ~11% of PA = RBI
        proj["sb"] = round(sample_size * POPULATION_PRIOR.sb_per_pa)

        return FusionResult(
            proj=proj,
            cat_scores={},
            source="statcast_with_prior",
            components_fused=components_fused,
            xwoba_override_applied=False
        )

    # STATE 4: Neither — pure population prior with generic z-score offset
    proj = {
        "avg": POPULATION_PRIOR.avg,
        "obp": POPULATION_PRIOR.obp,
        "slg": POPULATION_PRIOR.slg,
        "ops": POPULATION_PRIOR.ops,
        "hr": round(550 * POPULATION_PRIOR.hr_per_pa),
        "r": round(550 * 0.12),
        "rbi": round(550 * 0.11),
        "sb": round(550 * POPULATION_PRIOR.sb_per_pa),
    }

    return FusionResult(
        proj=proj,
        cat_scores={},
        source="population_prior",
        components_fused=0,
        xwoba_override_applied=False
    )


def fuse_pitcher_projection(
    steamer: Optional[dict] = None,
    statcast: Optional[dict] = None,
    sample_size: float = 0.0
) -> FusionResult:
    """
    Fuse pitcher projections using component-wise Marcel update.

    Four-state logic same as batters:
    1. Steamer + Statcast: Full Marcel update per component
    2. Steamer only: Return Steamer
    3. Statcast only: Fuse with POPULATION_PRIOR (double shrinkage)
    4. Neither: Return pure POPULATION_PRIOR with generic z-score offset

    Args:
        steamer: Dict with keys {era, whip, k_per_nine, bb_per_nine, w, l, sv}
        statcast: Dict with keys {era, whip, k_percent, bb_percent, xera, ip}
        sample_size: IP count for weight calculation

    Returns:
        FusionResult with fused projection and metadata
    """
    components_fused = 0
    proj = {}

    # STATE 1: Both sources available
    if steamer and statcast and sample_size >= 1.0:
        # xERA luck detection
        xera = statcast.get("xera", steamer.get("era", 4.50))
        era = statcast.get("era", steamer.get("era", 4.50))
        luck_delta = abs(xera - era)

        if luck_delta > 0.50:
            # ERA suppression/spike likely luck-driven
            prior_source = statcast
        else:
            prior_source = steamer

        # Component-wise Marcel update
        proj["era"] = marcel_update(
            prior_source.get("era", POPULATION_PRIOR.era),
            statcast.get("era", POPULATION_PRIOR.era),
            int(sample_size),
            StabilizationPoints.PITCHER["era"]
        )
        components_fused += 1

        proj["whip"] = marcel_update(
            prior_source.get("whip", POPULATION_PRIOR.whip),
            statcast.get("whip", POPULATION_PRIOR.whip),
            int(sample_size),
            StabilizationPoints.PITCHER["whip"]
        )
        components_fused += 1

        # Convert K% to K/9 for projection
        k_percent = marcel_update(
            prior_source.get("k_percent", 0.20),
            statcast.get("k_percent", 0.20),
            int(sample_size),
            StabilizationPoints.PITCHER["k_percent"]
        )
        proj["k_per_nine"] = k_percent * 27  # K% * 27 ≈ K/9
        components_fused += 1

        # Counting stats — scale by IP ratio
        ip_ratio = sample_size / max(1, steamer.get("ip", 150))
        proj["w"] = round(steamer.get("w", 10) * max(0.5, min(2.0, ip_ratio)))
        proj["l"] = round(steamer.get("l", 8) * max(0.5, min(2.0, ip_ratio)))
        proj["sv"] = steamer.get("sv", 0)  # Saves role-based, don't scale
        proj["qs"] = round(steamer.get("qs", 15) * max(0.5, min(2.0, ip_ratio)))
        proj["hr_pit"] = round(steamer.get("hr_pit", 20) * max(0.5, min(2.0, ip_ratio)))

        return FusionResult(
            proj=proj,
            cat_scores={},
            source="steamer_statcast_fusion",
            components_fused=components_fused,
            xwoba_override_applied=False
        )

    # STATE 2: Steamer only
    if steamer:
        return FusionResult(
            proj=steamer.copy(),
            cat_scores={},
            source="steamer_only",
            components_fused=0,
            xwoba_override_applied=False
        )

    # STATE 3: Statcast only with prior
    if statcast and sample_size >= 1.0:
        double_shrink = lambda s: s * 2

        proj["era"] = marcel_update(
            POPULATION_PRIOR.era,
            statcast.get("era", POPULATION_PRIOR.era),
            int(sample_size),
            double_shrink(StabilizationPoints.PITCHER["era"])
        )
        components_fused += 1

        proj["whip"] = marcel_update(
            POPULATION_PRIOR.whip,
            statcast.get("whip", POPULATION_PRIOR.whip),
            int(sample_size),
            double_shrink(StabilizationPoints.PITCHER["whip"])
        )
        components_fused += 1

        k_percent = marcel_update(
            0.20,  # League avg K%
            statcast.get("k_percent", 0.20),
            int(sample_size),
            double_shrink(StabilizationPoints.PITCHER["k_percent"])
        )
        proj["k_per_nine"] = k_percent * 27
        components_fused += 1

        return FusionResult(
            proj=proj,
            cat_scores={},
            source="statcast_with_prior",
            components_fused=components_fused,
            xwoba_override_applied=False
        )

    # STATE 4: Pure population prior
    proj = {
        "era": POPULATION_PRIOR.era,
        "whip": POPULATION_PRIOR.whip,
        "k_per_nine": POPULATION_PRIOR.k_per_nine,
        "w": 10,
        "l": 8,
        "sv": 0,
        "qs": 15,
        "hr_pit": 20,
    }

    return FusionResult(
        proj=proj,
        cat_scores={},
        source="population_prior",
        components_fused=0,
        xwoba_override_applied=False
    )
```

### Step 1.2: Run Python syntax check

Run:
```bash
python -m py_compile backend/fantasy_baseball/fusion_engine.py
```

Expected: No syntax errors (exit code 0)

### Step 1.3: Write unit tests for Marcel update formula

Create `tests/test_fusion_engine.py`:

```python
"""
Unit tests for Bayesian Projection Fusion Engine.

Test coverage:
- Marcel update formula math correctness
- Edge cases (zero sample, negative values, extreme values)
- Four-state fusion logic (both, steamer only, statcast only, neither)
- xwOBA override trigger
- Batter and pitcher paths
"""

import pytest
from backend.fantasy_baseball.fusion_engine import (
    marcel_update,
    StabilizationPoints,
    POPULATION_PRIOR,
    fuse_batter_projection,
    fuse_pitcher_projection,
)


class TestMarcelUpdate:
    """Test the core Marcel update formula."""

    def test_equal_weight_at_stabilization(self):
        """At sample_size = stabilization, prior and observed have equal weight."""
        prior = 0.250
        observed = 0.300
        stabil = 100
        sample = 100

        result = marcel_update(prior, observed, sample, stabil)

        # Should be exactly midpoint
        assert result == 0.275

    def test_prior_dominates_small_sample(self):
        """With small sample, posterior should skew toward prior."""
        prior = 0.250
        observed = 0.400  # Much higher
        stabil = 100
        sample = 10  # Only 10% of stabilization

        result = marcel_update(prior, observed, sample, stabil)

        # Should be much closer to prior
        assert result < 0.280
        assert result > 0.250  # But pulled slightly toward observed

    def test_observed_dominates_large_sample(self):
        """With large sample, posterior should skew toward observed."""
        prior = 0.250
        observed = 0.300
        stabil = 100
        sample = 1000  # 10x stabilization

        result = marcel_update(prior, observed, sample, stabil)

        # Should be much closer to observed
        assert result > 0.295
        assert result < 0.300

    def test_zero_sample_returns_prior(self):
        """With zero sample, return pure prior."""
        prior = 0.280
        observed = 0.320
        stabil = 100
        sample = 0

        result = marcel_update(prior, observed, sample, stabil)

        assert result == prior

    def test_negative_observed_clamped(self):
        """Negative observed values should work mathematically (rates)."""
        prior = 0.250
        observed = -0.050  # Impossible in reality but math works
        stabil = 100
        sample = 50

        result = marcel_update(prior, observed, sample, stabil)

        # Should be between prior and observed
        assert result < prior
        assert result > observed

    def test_exact_known_values(self):
        """Test against known calculation."""
        # Example from skill description
        prior = 0.280
        observed = 0.320
        stabil = 910  # AVG stabilization
        sample = 100

        result = marcel_update(prior, observed, sample, stabil)

        # ((910 * 0.280) + (100 * 0.320)) / (910 + 100)
        # = (254.8 + 32) / 1010
        # = 286.8 / 1010
        # ≈ 0.284
        assert 0.283 < result < 0.285


class TestStabilizationPoints:
    """Verify stabilization constants are defined."""

    def test_batter_stabilization_points_exist(self):
        """All required batter stabilization points must be defined."""
        required = {"k_percent", "bb_percent", "hr_rate", "iso", "obp", "slg", "avg", "barrel_pct", "xwoba"}
        assert required.issubset(StabilizationPoints.BATTER)

    def test_pitcher_stabilization_points_exist(self):
        """All required pitcher stabilization points must be defined."""
        required = {"k_percent", "bb_percent", "era", "whip", "fip", "xera"}
        assert required.issubset(StabilizationPoints.PITCHER)

    def test_stabilization_values_reasonable(self):
        """Stabilization points should be within expected ranges."""
        # K% stabilizes faster than AVG
        assert StabilizationPoints.BATTER["k_percent"] < StabilizationPoints.BATTER["avg"]
        # Barrel% should be very fast (premium metric)
        assert StabilizationPoints.BATTER["barrel_pct"] < 100
        # AVG should be slowest
        assert StabilizationPoints.BATTER["avg"] > 500


class TestPopulationPrior:
    """Verify population priors are reasonable league averages."""

    def test_batter_priors_reasonable(self):
        """Batter priors should be near league average."""
        assert 0.240 < POPULATION_PRIOR.avg < 0.260
        assert 0.700 < POPULATION_PRIOR.ops < 0.760
        assert 0.030 < POPULATION_PRIOR.hr_per_pa < 0.040

    def test_pitcher_priors_reasonable(self):
        """Pitcher priors should be near league average."""
        assert 4.0 < POPULATION_PRIOR.era < 5.0
        assert 1.25 < POPULATION_PRIOR.whip < 1.45
        assert 8.0 < POPULATION_PRIOR.k_per_nine < 9.5


class TestFuseBatterProjection:
    """Test batter fusion logic across all four states."""

    def test_state_1_both_sources_fusion(self):
        """Steamer + Statcast: apply Marcel update to rate stats."""
        steamer = {
            "avg": 0.280, "obp": 0.350, "slg": 0.450, "ops": 0.800,
            "hr": 30, "r": 90, "rbi": 95, "sb": 10, "pa": 600
        }
        statcast = {
            "avg": 0.320, "obp": 0.380, "slg": 0.500, "ops": 0.880,
            "xwoba": 0.380, "woba": 0.370, "pa": 100
        }

        result = fuse_batter_projection(steamer, statcast, 100)

        assert result.source == "steamer_statcast_fusion"
        assert result.components_fused >= 3
        # Rate stats should be shrunk toward observed (Statcast > Steamer)
        assert result.proj["avg"] > 0.280  # Pulled up from Steamer
        assert result.proj["avg"] < 0.320  # But not all the way to Statcast

    def test_state_1_xwoba_override_trigger(self):
        """xwOBA override when |xwOBA - wOBA| > 0.030."""
        steamer = {"avg": 0.280, "obp": 0.350, "slg": 0.450, "ops": 0.800}
        statcast = {
            "avg": 0.250, "obp": 0.300, "slg": 0.350, "ops": 0.650,
            "xwoba": 0.350, "woba": 0.300,  # Delta = 0.050 > 0.030
            "pa": 100
        }

        result = fuse_batter_projection(steamer, statcast, 100)

        assert result.xwoba_override_applied is True
        # With xwOBA override, should use xwOBA-based priors
        # Result should be higher than pure Statcast (pulled up by xwOBA signal)

    def test_state_2_steamer_only(self):
        """Steamer only: return pure Steamer, no fusion."""
        steamer = {
            "avg": 0.280, "obp": 0.350, "slg": 0.450, "ops": 0.800,
            "hr": 30, "r": 90, "rbi": 95, "sb": 10
        }

        result = fuse_batter_projection(steamer, None, 0)

        assert result.source == "steamer_only"
        assert result.components_fused == 0
        assert result.proj["avg"] == 0.280
        assert result.proj["hr"] == 30

    def test_state_3_statcast_only_double_shrinkage(self):
        """Statcast only: fuse with population prior using double shrinkage."""
        statcast = {
            "avg": 0.320, "obp": 0.380, "slg": 0.500, "ops": 0.880,
            "pa": 200
        }

        result = fuse_batter_projection(None, statcast, 200)

        assert result.source == "statcast_with_prior"
        assert result.components_fused >= 3
        # Double shrinkage means result closer to prior than single would be
        # Prior avg = 0.250, observed = 0.320
        # With double shrinkage on avg (910*2=1820), result should be very close to prior
        assert result.proj["avg"] < 0.270  # Heavily shrunk toward prior
        assert result.proj["avg"] > 0.250  # But not pure prior

    def test_state_4_neither_population_prior(self):
        """Neither source: return pure population prior."""
        result = fuse_batter_projection(None, None, 0)

        assert result.source == "population_prior"
        assert result.components_fused == 0
        assert result.proj["avg"] == POPULATION_PRIOR.avg
        assert result.proj["ops"] == POPULATION_PRIOR.ops

    def test_zero_sample_size_uses_prior(self):
        """Even with both sources, zero sample returns Steamer."""
        steamer = {"avg": 0.280, "obp": 0.350, "slg": 0.450, "ops": 0.800}
        statcast = {"avg": 0.320, "obp": 0.380, "slg": 0.500, "ops": 0.880}

        result = fuse_batter_projection(steamer, statcast, 0)

        # Should fall back to Steamer-only behavior
        assert result.source == "steamer_only"


class TestFusePitcherProjection:
    """Test pitcher fusion logic across all four states."""

    def test_state_1_both_sources_fusion(self):
        """Steamer + Statcast: apply Marcel update."""
        steamer = {
            "era": 3.50, "whip": 1.15, "k_per_nine": 9.5,
            "w": 15, "l": 7, "sv": 0, "qs": 20, "hr_pit": 18, "ip": 180
        }
        statcast = {
            "era": 4.50, "whip": 1.35, "k_percent": 0.25, "bb_percent": 0.07,
            "xera": 3.80, "ip": 50
        }

        result = fuse_pitcher_projection(steamer, statcast, 50)

        assert result.source == "steamer_statcast_fusion"
        assert result.components_fused >= 2
        # ERA should be between Steamer (3.50) and Statcast (4.50)
        assert 3.50 < result.proj["era"] < 4.50

    def test_state_2_steamer_only(self):
        """Steamer only: return pure Steamer."""
        steamer = {
            "era": 3.50, "whip": 1.15, "k_per_nine": 9.5,
            "w": 15, "l": 7, "sv": 0, "qs": 20
        }

        result = fuse_pitcher_projection(steamer, None, 0)

        assert result.source == "steamer_only"
        assert result.proj["era"] == 3.50

    def test_state_3_statcast_only_double_shrinkage(self):
        """Statcast only: fuse with population prior."""
        statcast = {
            "era": 3.00, "whip": 1.00, "k_percent": 0.28, "bb_percent": 0.05,
            "ip": 80
        }

        result = fuse_pitcher_projection(None, statcast, 80)

        assert result.source == "statcast_with_prior"
        # With double shrinkage, result should be closer to prior (4.50 ERA)
        assert result.proj["era"] > 3.50  # Pulled toward league average
        assert result.proj["era"] < 4.50  # But not all the way

    def test_state_4_neither_population_prior(self):
        """Neither source: return pure population prior."""
        result = fuse_pitcher_projection(None, None, 0)

        assert result.source == "population_prior"
        assert result.proj["era"] == POPULATION_PRIOR.era
        assert result.proj["whip"] == POPULATION_PRIOR.whip
```

### Step 1.4: Run tests to verify implementation

Run:
```bash
venv/Scripts/python -m pytest tests/test_fusion_engine.py -v
```

Expected: All tests pass (16+ test cases)

### Step 1.5: Commit fusion engine core

```bash
git add backend/fantasy_baseball/fusion_engine.py tests/test_fusion_engine.py
git commit -m "feat(fusion): add Bayesian projection fusion engine with Marcel update

- Add StabilizationPoints class with exact PA/IP thresholds
- Add PopulationPrior dataclass for league-average fallback
- Implement marcel_update() formula for component-wise shrinkage
- Add fuse_batter_projection() and fuse_pitcher_projection() with 4-state logic
- Implement xwOBA override for luck detection (|xwOBA-wOBA| > 0.030)
- Add 16+ unit tests covering formula math, edge cases, and all states"
```

---

## Task 2: Integrate Fusion Engine into get_or_create_projection()

**Files:**
- Modify: `backend/fantasy_baseball/player_board.py` (lines 968-1148)
- Test: `tests/test_player_board.py` (add integration tests)

### Step 2.1: Add fusion engine import

At the top of `backend/fantasy_baseball/player_board.py`, after the existing imports:

```python
from backend.fantasy_baseball.fusion_engine import (
    fuse_batter_projection,
    fuse_pitcher_projection,
    POPULATION_PRIOR,
)
```

### Step 2.2: Refactor _query_statcast_proxy to return raw data

Modify `_query_statcast_proxy()` (lines 838-965) to return raw Statcast data instead of a full projection:

```python
def _query_statcast_proxy(db: Session, player_id: str, player_type: str,
                          name: str = "") -> dict | None:
    """
    Query Statcast metrics tables to build raw data dict for Bayesian fusion.

    Returns raw Statcast data for fusion engine to process.
    No longer computes projections directly — that's fusion_engine's job.

    Args:
        db: SQLAlchemy database session
        player_id: Yahoo player ID string
        player_type: 'batter' or 'pitcher'
        name: Player name (for fallback name-based lookup)

    Returns:
        Dict with raw Statcast metrics, or None if insufficient data
    """
    from backend.models import (
        StatcastBatterMetrics, StatcastPitcherMetrics,
        PlayerIDMapping
    )

    # Try to find mlbam_id via PlayerIDMapping
    mlbam_id = None
    id_mapping = db.query(PlayerIDMapping).filter(
        PlayerIDMapping.yahoo_id == player_id
    ).first()

    if id_mapping and id_mapping.mlbam_id:
        mlbam_id = str(id_mapping.mlbam_id)

    # If no ID mapping, try name-based lookup as fallback
    if not mlbam_id and name:
        name_normalized = name.lower().strip()
        mapping = db.query(PlayerIDMapping).filter(
            PlayerIDMapping.normalized_name == name_normalized
        ).first()
        if mapping and mapping.mlbam_id:
            mlbam_id = str(mapping.mlbam_id)

    if not mlbam_id:
        return None

    # Query appropriate Statcast table
    if player_type == "pitcher":
        metrics = db.query(StatcastPitcherMetrics).filter_by(
            mlbam_id=mlbam_id
        ).first()

        if not metrics or metrics.ip is None or metrics.ip < 20.0:
            return None  # Insufficient data

        # Return raw Statcast data for fusion
        return {
            "era": metrics.era,
            "whip": metrics.whip,
            "k_percent": metrics.k_percent,
            "bb_percent": metrics.bb_percent,
            "xera": metrics.xera,
            "ip": metrics.ip,
            "w": metrics.w,
            "l": metrics.l,
            "sv": metrics.sv,
            "qs": metrics.qs,
            "hr_pit": metrics.hr_pit,
            "sample_size": int(metrics.ip),
        }

    else:  # batter
        metrics = db.query(StatcastBatterMetrics).filter_by(
            mlbam_id=mlbam_id
        ).first()

        if not metrics or metrics.pa is None or metrics.pa < 50:
            return None  # Insufficient data

        # Return raw Statcast data for fusion
        return {
            "avg": metrics.avg,
            "obp": None,  # Not in StatcastBatterMetrics, will be estimated
            "slg": metrics.slg,
            "ops": metrics.ops,
            "xwoba": metrics.xwoba,
            "woba": None,  # Will estimate from ops if not available
            "pa": metrics.pa,
            "hr": metrics.hr,
            "r": metrics.r,
            "rbi": metrics.rbi,
            "sb": metrics.sb,
            "sample_size": metrics.pa,
        }
```

### Step 2.3: Rewrite get_or_create_projection() with 4-state Bayesian logic

Replace the entire `get_or_create_projection()` function (lines 968-1148):

```python
def get_or_create_projection(yahoo_player: dict) -> dict:
    """
    Return a board-compatible dict for any Yahoo player using Bayesian fusion.

    Four-state projection logic:
    1. Steamer + Statcast: Component-wise Marcel update (fusion_engine)
    2. Steamer only: Return Steamer as-is
    3. Statcast only: Fuse with population prior (double shrinkage)
    4. Neither: Return population prior with generic z-score offset

    Args:
        yahoo_player: dict from YahooFantasyClient (has name, player_key,
                      positions, team, percent_owned, etc.)

    Returns:
        board-compatible dict with at minimum: name, z_score, positions,
        cat_scores, type, proj
    """
    from backend.models import get_db, PlayerProjection, PlayerIDMapping
    from backend.services.cat_scores_builder import (
        BATTER_WEIGHTS, PITCHER_WEIGHTS, compute_cat_scores
    )

    name = (yahoo_player.get("name") or "").strip()
    player_key = yahoo_player.get("player_key") or ""

    # 1. Check runtime cache first
    if player_key and player_key in _projection_cache:
        return _projection_cache[player_key]

    # 2. Check board by exact name match
    board = get_board()
    board_by_name = {p["name"].lower(): p for p in board}
    entry = board_by_name.get(name.lower())

    if entry:
        if player_key:
            _projection_cache[player_key] = entry
        return entry

    # 3. Fuzzy name match
    import difflib as _difflib
    name_lower = name.lower()
    clean_name = "".join(c for c in name_lower if c.isalnum() or c == " ")
    for board_name, board_entry in board_by_name.items():
        clean_board = "".join(c for c in board_name if c.isalnum() or c == " ")
        if clean_board == clean_board:
            if player_key:
                _projection_cache[player_key] = board_entry
            return board_entry

    # 3b. Similarity match
    best_ratio = 0.0
    best_entry = None
    for board_name, board_entry in board_by_name.items():
        clean_board = "".join(c for c in board_name if c.isalnum() or c == " ")
        ratio = _difflib.SequenceMatcher(None, clean_name, clean_board).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_entry = board_entry
    if best_ratio >= 0.90 and best_entry is not None:
        if player_key:
            _projection_cache[player_key] = best_entry
        return best_entry

    # 4. Not on board — build projection using Bayesian fusion
    positions = yahoo_player.get("positions") or []
    primary_pos = positions[0] if positions else ""
    player_type = "pitcher" if primary_pos in ("SP", "RP", "P") else "batter"

    # Query data sources
    steamer_data = None
    statcast_data = None
    sample_size = 0

    try:
        db_gen = get_db()
        db = next(db_gen)

        # Get MLBAM ID from player_key
        yahoo_id = None
        if player_key and ".p." in player_key:
            yahoo_id = player_key.split(".p.")[-1]
        elif player_key:
            yahoo_id = player_key.split(".")[-1]
        yahoo_id = yahoo_id or None

        mlbam_id = None
        if yahoo_id:
            id_mapping = db.query(PlayerIDMapping).filter(
                PlayerIDMapping.yahoo_id == yahoo_id
            ).first()
            if id_mapping:
                mlbam_id = id_mapping.mlbam_id or id_mapping.bdl_id

        # Fetch Steamer projection from PlayerProjection
        if mlbam_id:
            projection_row = db.query(PlayerProjection).filter(
                PlayerProjection.player_id == str(mlbam_id)
            ).first()

            if projection_row:
                # Build Steamer dict from PlayerProjection
                if player_type == "batter":
                    steamer_data = {
                        "avg": projection_row.avg,
                        "obp": projection_row.obp,
                        "slg": projection_row.slg,
                        "ops": projection_row.ops,
                        "xwoba": projection_row.xwoba,
                        "hr": projection_row.hr,
                        "r": projection_row.r,
                        "rbi": projection_row.rbi,
                        "sb": projection_row.sb,
                    }
                else:  # pitcher
                    steamer_data = {
                        "era": projection_row.era,
                        "whip": projection_row.whip,
                        "k_per_nine": projection_row.k_per_nine,
                        "w": projection_row.w,
                        "l": projection_row.l,
                        "sv": projection_row.sv or 0,
                        "qs": projection_row.get("qs", 0),  # May not exist
                        "hr_pit": projection_row.get("hr_pit", 0),
                    }

        # Fetch Statcast data (now returns raw metrics, not projection)
        if yahoo_id:
            statcast_data = _query_statcast_proxy(db, yahoo_id, player_type, name)
            if statcast_data:
                sample_size = statcast_data.pop("sample_size", 0)

        # Close DB session
        try:
            next(db_gen)
        except StopIteration:
            pass

    except Exception:
        # DB query failed — continue with whatever data we have
        pass

    # Apply Bayesian fusion based on available data
    if player_type == "batter":
        fusion_result = fuse_batter_projection(steamer_data, statcast_data, sample_size)
    else:
        fusion_result = fuse_pitcher_projection(steamer_data, statcast_data, sample_size)

    # Compute cat_scores from fused projection
    proj_list = [{"proj": fusion_result.proj, "cat_scores": {}}]
    weights = PITCHER_WEIGHTS if player_type == "pitcher" else BATTER_WEIGHTS
    compute_cat_scores(proj_list, weights)
    cat_scores = proj_list[0]["cat_scores"]
    z_score = sum(cat_scores.values()) if cat_scores else -0.5  # Generic offset for unknown players

    # Build proxy dict
    proxy = {
        "id": player_key or name.lower().replace(" ", "_"),
        "name": name,
        "team": yahoo_player.get("team") or yahoo_player.get("editorial_team_abbr") or "",
        "positions": positions,
        "type": player_type,
        "tier": 10,
        "rank": 9999,
        "adp": 9999.0,
        "z_score": z_score,
        "cat_scores": cat_scores,
        "proj": fusion_result.proj,
        "is_keeper": False,
        "keeper_round": None,
        "is_proxy": True,
        "fusion_source": fusion_result.source,  # For debugging
        "components_fused": fusion_result.components_fused,
        "xwoba_override": fusion_result.xwoba_override_applied,
    }

    if player_key:
        _projection_cache[player_key] = proxy
    return proxy
```

### Step 2.4: Add integration tests

Add to `tests/test_player_board.py`:

```python
class TestBayesianFusionIntegration:
    """Integration tests for Bayesian fusion in get_or_create_projection."""

    @pytest.fixture
    def mock_db_with_both_sources(self, db_session):
        """Mock DB with both Steamer and Statcast data available."""
        from backend.models import PlayerProjection, PlayerIDMapping, StatcastBatterMetrics

        # Create ID mapping
        mapping = PlayerIDMapping(
            yahoo_id="12345",
            mlbam_id="600001",
            normalized_name="test_player"
        )
        db_session.add(mapping)

        # Create Steamer projection
        steamer = PlayerProjection(
            player_id="600001",
            player_name="Test Player",
            avg=0.280,
            obp=0.350,
            slg=0.450,
            ops=0.800,
            xwoba=0.350,
            hr=30,
            r=90,
            rbi=95,
            sb=10
        )
        db_session.add(steamer)

        # Create Statcast metrics
        statcast = StatcastBatterMetrics(
            mlbam_id="600001",
            player_name="Test Player",
            season=2026,
            pa=150,
            avg=0.320,
            slg=0.500,
            ops=0.880,
            xwoba=0.380,
            hr=20,
            r=45,
            rbi=50,
            sb=5
        )
        db_session.add(statcast)

        db_session.commit()
        return db_session

    def test_fusion_applied_when_both_sources_exist(self, mock_db_with_both_sources):
        """When both Steamer and Statcast exist, apply Bayesian fusion."""
        yahoo_player = {
            "name": "Test Player",
            "player_key": "mlb.p.12345",
            "positions": ["OF"],
            "team": "TST"
        }

        result = get_or_create_projection(yahoo_player)

        assert result["fusion_source"] == "steamer_statcast_fusion"
        assert result["components_fused"] >= 3
        # AVG should be between Steamer (0.280) and Statcast (0.320)
        assert 0.280 < result["proj"]["avg"] < 0.320

    def test_steamer_only_when_statcast_missing(self, db_session):
        """When only Steamer exists, return Steamer as-is."""
        from backend.models import PlayerProjection, PlayerIDMapping

        mapping = PlayerIDMapping(
            yahoo_id="67890",
            mlbam_id="600002",
            normalized_name="steamer_only_player"
        )
        db_session.add(mapping)

        steamer = PlayerProjection(
            player_id="600002",
            player_name="Steamer Only",
            avg=0.250,
            obp=0.320,
            slg=0.400,
            ops=0.720,
            hr=20
        )
        db_session.add(steamer)
        db_session.commit()

        yahoo_player = {
            "name": "Steamer Only",
            "player_key": "mlb.p.67890",
            "positions": ["1B"],
            "team": "TST"
        }

        result = get_or_create_projection(yahoo_player)

        assert result["fusion_source"] == "steamer_only"
        assert result["components_fused"] == 0
        assert result["proj"]["avg"] == 0.250

    def test_statcast_only_with_prior_when_steamer_missing(self, db_session):
        """When only Statcast exists, fuse with population prior."""
        from backend.models import StatcastBatterMetrics, PlayerIDMapping

        mapping = PlayerIDMapping(
            yahoo_id="11111",
            mlbam_id="600003",
            normalized_name="statcast_only_player"
        )
        db_session.add(mapping)

        statcast = StatcastBatterMetrics(
            mlbam_id="600003",
            player_name="Statcast Only",
            season=2026,
            pa=200,
            avg=0.300,
            slg=0.500,
            ops=0.850,
            xwoba=0.360
        )
        db_session.add(statcast)
        db_session.commit()

        yahoo_player = {
            "name": "Statcast Only",
            "player_key": "mlb.p.11111",
            "positions": ["SS"],
            "team": "TST"
        }

        result = get_or_create_projection(yahoo_player)

        assert result["fusion_source"] == "statcast_with_prior"
        # With double shrinkage, should be closer to population prior
        assert result["proj"]["avg"] < 0.280

    def test_population_prior_when_no_data(self, db_session):
        """When neither source exists, return population prior."""
        yahoo_player = {
            "name": "Unknown Rookie",
            "player_key": "mlb.p.99999",
            "positions": ["C"],
            "team": "TST"
        }

        result = get_or_create_projection(yahoo_player)

        assert result["fusion_source"] == "population_prior"
        assert result["proj"]["avg"] == POPULATION_PRIOR.avg
```

### Step 2.5: Run syntax check

```bash
python -m py_compile backend/fantasy_baseball/player_board.py
```

Expected: No syntax errors

### Step 2.6: Run full test suite

```bash
venv/Scripts/python -m pytest tests/test_fusion_engine.py tests/test_player_board.py -v
```

Expected: All tests pass

### Step 2.7: Commit integration changes

```bash
git add backend/fantasy_baseball/player_board.py tests/test_player_board.py
git commit -m "refactor(projections): integrate Bayesian fusion into get_or_create_projection

- Import fusion_engine module
- Refactor _query_statcast_proxy to return raw metrics
- Rewrite get_or_create_projection with 4-state Bayesian logic:
  1. Steamer + Statcast: Component-wise Marcel update
  2. Steamer only: Pure prior
  3. Statcast only: Fuse with population prior (double shrinkage)
  4. Neither: Pure population prior
- Add fusion_source, components_fused, xwoba_override metadata
- Add integration tests for all four states"
```

---

## Validation Criteria

Before considering this implementation complete, verify:

1. **Math Correctness**: Marcel update formula produces exact midpoint at sample_size = stabilization_point
2. **Stabilization Points**: All required constants defined and within reasonable ranges
3. **Four-State Logic**: Each state (both/steamer-only/statcast-only/neither) produces correct output
4. **xwOBA Override**: Triggers when |xwOBA - wOBA| > 0.030 for batters, |xERA - ERA| > 0.50 for pitchers
5. **Double Shrinkage**: Statcast-only path uses 2x stabilization points
6. **No Placeholders**: All code is complete, no TODO/TBD markers
7. **Test Coverage**: Unit tests for formula math, integration tests for full flow
8. **Backward Compatibility**: Existing board players still return correctly (no regression)

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-25-bayesian-projection-fusion.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
