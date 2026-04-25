"""
Bayesian Projection Fusion Engine for Fantasy Baseball.

Implements Component-Wise Empirical Bayes model using Marcel update formula.
Replaces binary Steamer-or-Statcast fallback with principled Bayesian shrinkage.

Mathematical Foundation:
    Posterior Mean = ((stabilization * prior) + (sample_size * observed)) / (stabilization + sample_size)

At sample_size = stabilization_point: posterior is exactly midpoint (50% prior, 50% observed)
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math


class StabilizationPoints:
    """
    Exact PA/IP thresholds where each metric becomes 50% reliable.

    Sources: MLB Statcast research, FanGraphs, Baseball Prospectus
    A stabilization point means: at this sample size, the metric is 50% predictive of future performance.
    """

    # BATTER STABILIZATION (Plate Appearances)
    # Strikeout rate stabilizes quickly
    BATTER_K_PERCENT: int = 60
    # Walk rate takes more PA
    BATTER_BB_PERCENT: int = 120
    # Home run rate
    BATTER_HR_RATE: int = 170
    # Isolated power (SLG - AVG)
    BATTER_ISO: int = 160
    # On-base percentage
    BATTER_OBP: int = 460
    # Slugging percentage
    BATTER_SLG: int = 320
    # Batting average (takes longest)
    BATTER_AVG: int = 910
    # Barrel percentage (batted ball quality)
    BATTER_BARREL_PCT: int = 50
    # Expected wOBA (Statcast)
    BATTER_XWOBA: int = 150

    # PITCHER STABILIZATION (Innings Pitched)
    # Strikeout rate
    PITCHER_K_PERCENT: int = 70
    # Walk rate (takes longer)
    PITCHER_BB_PERCENT: int = 170
    # ERA (full season needed)
    PITCHER_ERA: int = 300
    # WHIP (walks + hits per inning)
    PITCHER_WHIP: int = 300
    # Fielding Independent Pitching
    PITCHER_FIP: int = 170
    # Expected ERA (Statcast)
    PITCHER_XERA: int = 150


class PopulationPrior:
    """
    League-average baselines for rookies/unknown players.

    These represent the population mean for each metric - the expected value
    when we have no player-specific information.
    """

    # BATTER POPULATION PRIORS
    BATTER_AVG: float = 0.250
    BATTER_OBP: float = 0.320
    BATTER_SLG: float = 0.410
    BATTER_OPS: float = 0.730  # OBP + SLG
    BATTER_HR_PER_PA: float = 0.035  # ~25 HR over 700 PA
    BATTER_SB_PER_PA: float = 0.010  # ~7 SB over 700 PA
    BATTER_K_PERCENT: float = 0.225  # 22.5% K rate
    BATTER_BB_PERCENT: float = 0.080  # 8% BB rate

    # PITCHER POPULATION PRIORS
    PITCHER_ERA: float = 4.50
    PITCHER_WHIP: float = 1.35
    PITCHER_K_PER_NINE: float = 8.5
    PITCHER_BB_PER_NINE: float = 3.0


@dataclass
class FusionResult:
    """
    Result of projection fusion with metadata for auditing.

    Attributes:
        proj: Fused projection dictionary with all rate stats
        source: Data source label ('fusion', 'steamer', 'statcast_shrunk', 'population_prior')
        components_fused: Number of components that underwent Marcel update
        xwoba_override_detected: Whether xwOBA/xERA divergence was detected (metadata only)

    Note: cat_scores are NOT computed here. Use pre-computed DB z-scores from
    PlayerProjection.cat_scores when available; otherwise accept z_score=0.0.
    Computing cat_scores against a full player pool requires cat_scores_builder.py.
    """
    proj: Dict[str, Any]
    source: str
    components_fused: int
    xwoba_override_detected: bool


def marcel_update(
    prior_mean: float,
    observed_mean: float,
    sample_size: int,
    stabilization_point: int,
    min_sample: int = 1
) -> float:
    """
    Apply Marcel update formula for component-wise Bayesian shrinkage.

    The Marcel method (named after Marcel the Monkey, sabermetric pioneer) uses
    Empirical Bayes to shrink observed statistics toward a prior mean based on
    sample reliability.

    Formula:
        Posterior = ((stabilization * prior) + (sample_size * observed)) / (stabilization + sample_size)

    Key Properties:
        - At sample_size = stabilization_point: Posterior = (prior + observed) / 2 (midpoint)
        - Small sample (< stabilization): Weighted toward prior (regression to mean)
        - Large sample (> stabilization): Weighted toward observed (trust the data)

    Args:
        prior_mean: Population prior (league average for rookies)
        observed_mean: Observed player statistic
        sample_size: Actual PA/IP for this player
        stabilization_point: PA/IP where metric becomes 50% reliable
        min_sample: Minimum sample size to prevent division by zero (default: 1)

    Returns:
        Posterior mean (shrunk projection)

    Examples:
        >>> # Midpoint: at stabilization, exactly halfway
        >>> marcel_update(0.250, 0.300, 100, 100)
        0.275

        >>> # Prior dominance: small sample
        >>> marcel_update(0.250, 0.350, 10, 100)  # ~0.259 (91% toward prior)

        >>> # Observed dominance: large sample
        >>> marcel_update(0.250, 0.350, 1000, 100)  # ~0.341 (91% toward observed)
    """
    # Enforce minimum sample size
    effective_sample = max(sample_size, min_sample)

    # Marcel update formula
    numerator = (stabilization_point * prior_mean) + (effective_sample * observed_mean)
    denominator = stabilization_point + effective_sample

    return numerator / denominator


def _safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """
    Safely retrieve a value from a dict-like or object-like source.

    Handles None, dicts, and arbitrary objects (e.g. SQLAlchemy rows or test mocks).
    Returns the default if the key is absent or the value is not a real number.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        val = obj.get(key, default)
    else:
        val = getattr(obj, key, default)
    return val


def _safe_num(obj: Any, key: str) -> Optional[float]:
    """Return a numeric value from obj[key], or None if missing/non-numeric."""
    val = _safe_get(obj, key, None)
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    return None


def _should_apply_xwoba_override(statcast: Dict[str, Any]) -> bool:
    """
    Detect if xwOBA override should be applied (metadata only).

    When xwOBA differs significantly from actual wOBA, it suggests the player's
    performance was unsustainable (lucky or unlucky). This function detects
    that condition for metadata reporting; it does NOT swap the prior source.

    Future enhancement: use xwOBA as prior source instead of Steamer when True.

    Args:
        statcast: Statcast data dict containing 'xwoba' and 'woba'

    Returns:
        True if |xwOBA - wOBA| > 0.030
    """
    if statcast is None:
        return False

    xwoba = _safe_num(statcast, 'xwoba')
    woba = _safe_num(statcast, 'woba')

    if xwoba is None or woba is None:
        return False

    return abs(xwoba - woba) > 0.030


def _should_apply_xera_override(statcast: Dict[str, Any]) -> bool:
    """
    Detect if xERA override should be applied for pitchers (metadata only).

    When xERA differs significantly from actual ERA, it suggests the pitcher's
    performance was unsustainable. This function detects that condition for
    metadata reporting; it does NOT swap the prior source.

    Future enhancement: use xERA as prior source instead of Steamer when True.

    Args:
        statcast: Statcast data dict containing 'xera' and 'era'

    Returns:
        True if |xERA - ERA| > 0.50
    """
    if statcast is None:
        return False

    xera = _safe_num(statcast, 'xera')
    era = _safe_num(statcast, 'era')

    if xera is None or era is None:
        return False

    return abs(xera - era) > 0.50


def fuse_batter_projection(
    steamer: Optional[Dict[str, Any]],
    statcast: Optional[Dict[str, Any]],
    sample_size: int
) -> FusionResult:
    """
    Fuse batter projections using Component-Wise Empirical Bayes.

    Four-State Logic:
        1. Steamer + Statcast: Full Marcel update per component
        2. Steamer only: Return Steamer unchanged (trust the projection system)
        3. Statcast only: Fuse with POPULATION_PRIOR using double shrinkage
        4. Neither: Return pure POPULATION_PRIOR (rookie baseline)

    xwOBA Override Detection:
        If |xwOBA - wOBA| > 0.030, the override flag is set in metadata.
        Future enhancement: swap prior source to xwOBA when override detected.

    Args:
        steamer: Steamer projection dict with rate stats (avg, obp, slg, k_percent, bb_percent, hr_per_pa, sb_per_pa)
        statcast: Statcast data dict with observed stats and xwOBA
        sample_size: Plate appearances for this player

    Returns:
        FusionResult with projected stats and metadata
    """
    prior = PopulationPrior()
    stabil = StabilizationPoints()

    # Determine data availability
    has_steamer = steamer is not None
    has_statcast = statcast is not None

    # Check xwOBA override
    xwoba_override = _should_apply_xwoba_override(statcast)

    # Initialize result
    proj: Dict[str, Any] = {}
    components_fused = 0

    if has_steamer and has_statcast:
        # STATE 1: Full fusion - Steamer as prior, Statcast as observed
        source = 'fusion'

        def fuse_component(steam_key: str, stat_key: str, stabil_point: int, fallback: float) -> float:
            nonlocal components_fused
            steamer_val = _safe_num(steamer, steam_key)
            obs_val = _safe_num(statcast, stat_key)
            prior_val = steamer_val if steamer_val is not None else fallback
            if obs_val is None:
                # No observed data — return prior unchanged
                return prior_val
            components_fused += 1
            return marcel_update(
                prior_mean=prior_val,
                observed_mean=obs_val,
                sample_size=sample_size,
                stabilization_point=stabil_point
            )

        proj['avg'] = fuse_component('avg', 'avg', stabil.BATTER_AVG, prior.BATTER_AVG)
        proj['obp'] = fuse_component('obp', 'obp', stabil.BATTER_OBP, prior.BATTER_OBP)
        proj['slg'] = fuse_component('slg', 'slg', stabil.BATTER_SLG, prior.BATTER_SLG)
        proj['k_percent'] = fuse_component('k_percent', 'k_percent', stabil.BATTER_K_PERCENT, prior.BATTER_K_PERCENT)
        proj['bb_percent'] = fuse_component('bb_percent', 'bb_percent', stabil.BATTER_BB_PERCENT, prior.BATTER_BB_PERCENT)

        # Use Steamer for counting stats (per-PA rates)
        hr_rate = _safe_num(steamer, 'hr_per_pa')
        sb_rate = _safe_num(steamer, 'sb_per_pa')
        proj['hr_per_pa'] = hr_rate if hr_rate is not None else prior.BATTER_HR_PER_PA
        proj['sb_per_pa'] = sb_rate if sb_rate is not None else prior.BATTER_SB_PER_PA

    elif has_steamer:
        # STATE 2: Steamer only - return unchanged
        source = 'steamer'
        avg_val = _safe_num(steamer, 'avg')
        obp_val = _safe_num(steamer, 'obp')
        slg_val = _safe_num(steamer, 'slg')
        k_val = _safe_num(steamer, 'k_percent')
        bb_val = _safe_num(steamer, 'bb_percent')
        hr_val = _safe_num(steamer, 'hr_per_pa')
        sb_val = _safe_num(steamer, 'sb_per_pa')
        proj['avg'] = avg_val if avg_val is not None else prior.BATTER_AVG
        proj['obp'] = obp_val if obp_val is not None else prior.BATTER_OBP
        proj['slg'] = slg_val if slg_val is not None else prior.BATTER_SLG
        proj['k_percent'] = k_val if k_val is not None else prior.BATTER_K_PERCENT
        proj['bb_percent'] = bb_val if bb_val is not None else prior.BATTER_BB_PERCENT
        proj['hr_per_pa'] = hr_val if hr_val is not None else prior.BATTER_HR_PER_PA
        proj['sb_per_pa'] = sb_val if sb_val is not None else prior.BATTER_SB_PER_PA

    elif has_statcast:
        # STATE 3: Statcast only - fuse with population prior, double shrinkage
        source = 'statcast_shrunk'

        def shrink_to_population(stat_key: str, stabil_point: int, prior_val: float) -> float:
            nonlocal components_fused
            obs_val = _safe_num(statcast, stat_key)
            if obs_val is None:
                # No observed data for this component — fall back to prior
                return prior_val
            components_fused += 1
            return marcel_update(
                prior_mean=prior_val,
                observed_mean=obs_val,
                sample_size=sample_size,
                stabilization_point=stabil_point * 2  # Double shrinkage
            )

        proj['avg'] = shrink_to_population('avg', stabil.BATTER_AVG, prior.BATTER_AVG)
        proj['obp'] = shrink_to_population('obp', stabil.BATTER_OBP, prior.BATTER_OBP)
        proj['slg'] = shrink_to_population('slg', stabil.BATTER_SLG, prior.BATTER_SLG)
        proj['k_percent'] = shrink_to_population('k_percent', stabil.BATTER_K_PERCENT, prior.BATTER_K_PERCENT)
        proj['bb_percent'] = shrink_to_population('bb_percent', stabil.BATTER_BB_PERCENT, prior.BATTER_BB_PERCENT)

        # Use population priors for rate stats
        proj['hr_per_pa'] = prior.BATTER_HR_PER_PA
        proj['sb_per_pa'] = prior.BATTER_SB_PER_PA

    else:
        # STATE 4: No data - pure population prior
        source = 'population_prior'
        proj['avg'] = prior.BATTER_AVG
        proj['obp'] = prior.BATTER_OBP
        proj['slg'] = prior.BATTER_SLG
        proj['ops'] = prior.BATTER_OPS
        proj['k_percent'] = prior.BATTER_K_PERCENT
        proj['bb_percent'] = prior.BATTER_BB_PERCENT
        proj['hr_per_pa'] = prior.BATTER_HR_PER_PA
        proj['sb_per_pa'] = prior.BATTER_SB_PER_PA

    # Calculate OPS if not present
    if 'ops' not in proj and 'obp' in proj and 'slg' in proj:
        proj['ops'] = proj['obp'] + proj['slg']

    return FusionResult(
        proj=proj,
        source=source,
        components_fused=components_fused,
        xwoba_override_detected=xwoba_override
    )


def fuse_pitcher_projection(
    steamer: Optional[Dict[str, Any]],
    statcast: Optional[Dict[str, Any]],
    sample_size: int
) -> FusionResult:
    """
    Fuse pitcher projections using Component-Wise Empirical Bayes.

    Four-State Logic:
        1. Steamer + Statcast: Full Marcel update per component
        2. Steamer only: Return Steamer unchanged
        3. Statcast only: Fuse with POPULATION_PRIOR using double shrinkage
        4. Neither: Return pure POPULATION_PRIOR

    xERA Override Detection:
        If |xERA - ERA| > 0.50, the override flag is set in metadata.
        Future enhancement: swap prior source to xERA when override detected.

    Args:
        steamer: Steamer projection dict with era, whip, k_percent, bb_percent, k_per_nine, bb_per_nine
        statcast: Statcast data dict with observed stats and xera
        sample_size: Innings pitched for this player

    Returns:
        FusionResult with projected stats and metadata
    """
    prior = PopulationPrior()
    stabil = StabilizationPoints()

    # Determine data availability
    has_steamer = steamer is not None
    has_statcast = statcast is not None

    # Check xERA override
    xera_override = _should_apply_xera_override(statcast)

    # Initialize result
    proj: Dict[str, Any] = {}
    components_fused = 0

    if has_steamer and has_statcast:
        # STATE 1: Full fusion
        source = 'fusion'

        def fuse_component(steam_key: str, stat_key: str, stabil_point: int, fallback: float) -> float:
            nonlocal components_fused
            steamer_val = _safe_num(steamer, steam_key)
            obs_val = _safe_num(statcast, stat_key)
            prior_val = steamer_val if steamer_val is not None else fallback
            if obs_val is None:
                return prior_val
            components_fused += 1
            return marcel_update(
                prior_mean=prior_val,
                observed_mean=obs_val,
                sample_size=sample_size,
                stabilization_point=stabil_point
            )

        proj['era'] = fuse_component('era', 'era', stabil.PITCHER_ERA, prior.PITCHER_ERA)
        proj['whip'] = fuse_component('whip', 'whip', stabil.PITCHER_WHIP, prior.PITCHER_WHIP)
        proj['k_percent'] = fuse_component('k_percent', 'k_percent', stabil.PITCHER_K_PERCENT, 0.22)
        proj['bb_percent'] = fuse_component('bb_percent', 'bb_percent', stabil.PITCHER_BB_PERCENT, 0.07)

        # Use Steamer for rate stats
        k9_val = _safe_num(steamer, 'k_per_nine')
        bb9_val = _safe_num(steamer, 'bb_per_nine')
        proj['k_per_nine'] = k9_val if k9_val is not None else prior.PITCHER_K_PER_NINE
        proj['bb_per_nine'] = bb9_val if bb9_val is not None else prior.PITCHER_BB_PER_NINE

    elif has_steamer:
        # STATE 2: Steamer only
        source = 'steamer'
        era_val = _safe_num(steamer, 'era')
        whip_val = _safe_num(steamer, 'whip')
        k_val = _safe_num(steamer, 'k_percent')
        bb_val = _safe_num(steamer, 'bb_percent')
        k9_val = _safe_num(steamer, 'k_per_nine')
        bb9_val = _safe_num(steamer, 'bb_per_nine')
        proj['era'] = era_val if era_val is not None else prior.PITCHER_ERA
        proj['whip'] = whip_val if whip_val is not None else prior.PITCHER_WHIP
        proj['k_percent'] = k_val if k_val is not None else 0.22
        proj['bb_percent'] = bb_val if bb_val is not None else 0.07
        proj['k_per_nine'] = k9_val if k9_val is not None else prior.PITCHER_K_PER_NINE
        proj['bb_per_nine'] = bb9_val if bb9_val is not None else prior.PITCHER_BB_PER_NINE

    elif has_statcast:
        # STATE 3: Statcast only - double shrinkage
        source = 'statcast_shrunk'

        def shrink_to_population(stat_key: str, stabil_point: int, prior_val: float) -> float:
            nonlocal components_fused
            obs_val = _safe_num(statcast, stat_key)
            if obs_val is None:
                return prior_val
            components_fused += 1
            return marcel_update(
                prior_mean=prior_val,
                observed_mean=obs_val,
                sample_size=sample_size,
                stabilization_point=stabil_point * 2  # Double shrinkage
            )

        proj['era'] = shrink_to_population('era', stabil.PITCHER_ERA, prior.PITCHER_ERA)
        proj['whip'] = shrink_to_population('whip', stabil.PITCHER_WHIP, prior.PITCHER_WHIP)
        proj['k_percent'] = shrink_to_population('k_percent', stabil.PITCHER_K_PERCENT, 0.22)
        proj['bb_percent'] = shrink_to_population('bb_percent', stabil.PITCHER_BB_PERCENT, 0.07)

        # Use population priors
        proj['k_per_nine'] = prior.PITCHER_K_PER_NINE
        proj['bb_per_nine'] = prior.PITCHER_BB_PER_NINE

    else:
        # STATE 4: No data - pure population prior
        source = 'population_prior'
        proj['era'] = prior.PITCHER_ERA
        proj['whip'] = prior.PITCHER_WHIP
        proj['k_percent'] = 0.22
        proj['bb_percent'] = 0.07
        proj['k_per_nine'] = prior.PITCHER_K_PER_NINE
        proj['bb_per_nine'] = prior.PITCHER_BB_PER_NINE

    return FusionResult(
        proj=proj,
        source=source,
        components_fused=components_fused,
        xwoba_override_detected=xera_override
    )


# Public API for external modules
__all__ = [
    'StabilizationPoints',
    'PopulationPrior',
    'marcel_update',
    'fuse_batter_projection',
    'fuse_pitcher_projection',
    'FusionResult',
]
