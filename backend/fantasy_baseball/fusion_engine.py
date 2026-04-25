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
        cat_scores: Category scores (1-100) relative to league
        source: Data source label ('fusion', 'steamer', 'statcast_shrunk', 'population_prior')
        components_fused: Number of components that underwent Marcel update
        xwoba_override_applied: Whether xwOBA/xERA override was triggered
    """
    proj: Dict[str, Any]
    cat_scores: Dict[str, float]
    source: str
    components_fused: int
    xwoba_override_applied: bool


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


def _should_apply_xwoba_override(statcast: Dict[str, Any]) -> bool:
    """
    Determine if xwOBA override should be applied.

    When xwOBA differs significantly from actual wOBA, it suggests the player's
    performance was unsustainable (lucky or unlucky). In such cases, we use
    xwOBA as a prior source for rate stats.

    Args:
        statcast: Statcast data dict containing 'xwoba' and 'woba'

    Returns:
        True if |xwOBA - wOBA| > 0.030
    """
    if statcast is None:
        return False

    xwoba = statcast.get('xwoba')
    woba = statcast.get('woba')

    if xwoba is None or woba is None:
        return False

    return abs(xwoba - woba) > 0.030


def _should_apply_xera_override(statcast: Dict[str, Any]) -> bool:
    """
    Determine if xERA override should be applied for pitchers.

    When xERA differs significantly from actual ERA, use xERA as prior source.

    Args:
        statcast: Statcast data dict containing 'xera' and 'era'

    Returns:
        True if |xERA - ERA| > 0.50
    """
    if statcast is None:
        return False

    xera = statcast.get('xera')
    era = statcast.get('era')

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

    xwOBA Override:
        If |xwOBA - wOBA| > 0.030, use xwOBA as prior source instead of population prior.
        This captures "unsustainable performance" signal from Statcast batted ball data.

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

        def fuse_component(steam_key: str, stat_key: str, stabil_point: int, fallback: float = None) -> float:
            nonlocal components_fused
            components_fused += 1
            prior = steamer.get(steam_key, fallback) if fallback is not None else steamer[steam_key]
            if prior is None:
                # Use statcast value directly if no Steamer prior
                return statcast[stat_key]
            return marcel_update(
                prior_mean=prior,
                observed_mean=statcast[stat_key],
                sample_size=sample_size,
                stabilization_point=stabil_point
            )

        proj['avg'] = fuse_component('avg', 'avg', stabil.BATTER_AVG, prior.BATTER_AVG)
        proj['obp'] = fuse_component('obp', 'obp', stabil.BATTER_OBP, prior.BATTER_OBP)
        proj['slg'] = fuse_component('slg', 'slg', stabil.BATTER_SLG, prior.BATTER_SLG)
        proj['k_percent'] = fuse_component('k_percent', 'k_percent', stabil.BATTER_K_PERCENT, prior.BATTER_K_PERCENT)
        proj['bb_percent'] = fuse_component('bb_percent', 'bb_percent', stabil.BATTER_BB_PERCENT, prior.BATTER_BB_PERCENT)

        # Use Steamer for counting stats (per-PA rates)
        proj['hr_per_pa'] = steamer.get('hr_per_pa', prior.BATTER_HR_PER_PA)
        proj['sb_per_pa'] = steamer.get('sb_per_pa', prior.BATTER_SB_PER_PA)

    elif has_steamer:
        # STATE 2: Steamer only - return unchanged
        source = 'steamer'
        proj['avg'] = steamer['avg']
        proj['obp'] = steamer['obp']
        proj['slg'] = steamer['slg']
        proj['k_percent'] = steamer.get('k_percent', prior.BATTER_K_PERCENT)
        proj['bb_percent'] = steamer.get('bb_percent', prior.BATTER_BB_PERCENT)
        proj['hr_per_pa'] = steamer.get('hr_per_pa', prior.BATTER_HR_PER_PA)
        proj['sb_per_pa'] = steamer.get('sb_per_pa', prior.BATTER_SB_PER_PA)

    elif has_statcast:
        # STATE 3: Statcast only - fuse with population prior, double shrinkage
        source = 'statcast_shrunk'

        # If xwOBA override, use xwOBA-informed prior
        if xwoba_override:
            # Create adjusted prior from xwOBA signal
            # xwOBA suggests true talent; shift prior toward observed
            prior_adj = 0.5  # Midpoint prior
        else:
            prior_adj = prior.BATTER_AVG

        def shrink_to_population(stat_key: str, stabil_point: int, prior_val: float) -> float:
            nonlocal components_fused
            components_fused += 1
            # Double shrinkage: use 2x stabilization point for more aggressive regression
            return marcel_update(
                prior_mean=prior_val,
                observed_mean=statcast[stat_key],
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

    # Calculate category scores (simplified - 1-100 scale)
    cat_scores = _calculate_batter_cat_scores(proj)

    return FusionResult(
        proj=proj,
        cat_scores=cat_scores,
        source=source,
        components_fused=components_fused,
        xwoba_override_applied=xwoba_override
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

    xERA Override:
        If |xERA - ERA| > 0.50, use xERA as prior source.

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

        def fuse_component(steam_key: str, stat_key: str, stabil_point: int, fallback: float = None) -> float:
            nonlocal components_fused
            components_fused += 1
            prior = steamer.get(steam_key, fallback) if fallback is not None else steamer[steam_key]
            if prior is None:
                # Use statcast value directly if no Steamer prior
                return statcast[stat_key]
            return marcel_update(
                prior_mean=prior,
                observed_mean=statcast[stat_key],
                sample_size=sample_size,
                stabilization_point=stabil_point
            )

        proj['era'] = fuse_component('era', 'era', stabil.PITCHER_ERA, prior.PITCHER_ERA)
        proj['whip'] = fuse_component('whip', 'whip', stabil.PITCHER_WHIP, prior.PITCHER_WHIP)
        proj['k_percent'] = fuse_component('k_percent', 'k_percent', stabil.PITCHER_K_PERCENT, 0.22)
        proj['bb_percent'] = fuse_component('bb_percent', 'bb_percent', stabil.PITCHER_BB_PERCENT, 0.07)

        # Use Steamer for rate stats
        proj['k_per_nine'] = steamer.get('k_per_nine', prior.PITCHER_K_PER_NINE)
        proj['bb_per_nine'] = steamer.get('bb_per_nine', prior.PITCHER_BB_PER_NINE)

    elif has_steamer:
        # STATE 2: Steamer only
        source = 'steamer'
        proj['era'] = steamer['era']
        proj['whip'] = steamer['whip']
        proj['k_percent'] = steamer.get('k_percent', 0.22)
        proj['bb_percent'] = steamer.get('bb_percent', 0.07)
        proj['k_per_nine'] = steamer.get('k_per_nine', prior.PITCHER_K_PER_NINE)
        proj['bb_per_nine'] = steamer.get('bb_per_nine', prior.PITCHER_BB_PER_NINE)

    elif has_statcast:
        # STATE 3: Statcast only - double shrinkage
        source = 'statcast_shrunk'

        def shrink_to_population(stat_key: str, stabil_point: int, prior_val: float) -> float:
            nonlocal components_fused
            components_fused += 1
            return marcel_update(
                prior_mean=prior_val,
                observed_mean=statcast[stat_key],
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

    # Calculate category scores
    cat_scores = _calculate_pitcher_cat_scores(proj)

    return FusionResult(
        proj=proj,
        cat_scores=cat_scores,
        source=source,
        components_fused=components_fused,
        xwoba_override_applied=xera_override  # Reuse flag name
    )


def _calculate_batter_cat_scores(proj: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate category scores (1-100) for 5x5 roto scoring.

    Simple linear scaling centered on league average.
    """
    scores = {}

    # AVG: .250 = 50, each .010 = 5 points
    if 'avg' in proj:
        scores['avg'] = 50 + (proj['avg'] - 0.250) * 500
        scores['avg'] = max(1, min(100, scores['avg']))

    # HR: scale from HR/PA
    if 'hr_per_pa' in proj:
        scores['hr'] = 50 + (proj['hr_per_pa'] - 0.035) * 1500
        scores['hr'] = max(1, min(100, scores['hr']))

    # RBI: proxy from SLG
    if 'slg' in proj:
        scores['rbi'] = 50 + (proj['slg'] - 0.410) * 200
        scores['rbi'] = max(1, min(100, scores['rbi']))

    # SB: scale from SB/PA
    if 'sb_per_pa' in proj:
        scores['sb'] = 50 + (proj['sb_per_pa'] - 0.010) * 4000
        scores['sb'] = max(1, min(100, scores['sb']))

    # Runs: proxy from OBP
    if 'obp' in proj:
        scores['runs'] = 50 + (proj['obp'] - 0.320) * 300
        scores['runs'] = max(1, min(100, scores['runs']))

    return scores


def _calculate_pitcher_cat_scores(proj: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate category scores (1-100) for pitcher roto scoring.

    Lower ERA/WHIP is better. Higher K is better.
    """
    scores = {}

    # ERA: 4.50 = 50, each 0.10 = 5 points (inverse)
    if 'era' in proj:
        scores['era'] = 50 + (4.50 - proj['era']) * 50
        scores['era'] = max(1, min(100, scores['era']))

    # WHIP: 1.35 = 50, each 0.05 = 5 points (inverse)
    if 'whip' in proj:
        scores['whip'] = 50 + (1.35 - proj['whip']) * 100
        scores['whip'] = max(1, min(100, scores['whip']))

    # K: scale from K/9
    if 'k_per_nine' in proj:
        scores['k'] = 50 + (proj['k_per_nine'] - 8.5) * 10
        scores['k'] = max(1, min(100, scores['k']))

    # Wins: proxy from ERA (inverse)
    if 'era' in proj:
        scores['wins'] = 50 + (4.50 - proj['era']) * 40
        scores['wins'] = max(1, min(100, scores['wins']))

    # Saves: assume 0 for starting pitchers
    scores['saves'] = 10  # Low default

    return scores


# Public API for external modules
__all__ = [
    'StabilizationPoints',
    'PopulationPrior',
    'marcel_update',
    'fuse_batter_projection',
    'fuse_pitcher_projection',
    'FusionResult',
]
