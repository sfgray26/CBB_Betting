"""
ROW → Simulation Bridge Adapter

Phase 3 Workstream C: Connects Phase 2 ROW projections to Phase 4 simulation engines.

This module provides adapter functions that convert ROWProjectionResult outputs
into the input formats expected by:
- H2HOneWinSimulator (h2h_monte_carlo.py)
- MCMC simulator (mcmc_simulator.py)

Key Design Decisions:
1. ROW projections are TEAM-LEVEL totals, not player-level
2. Ratio stats (AVG, OPS, ERA, WHIP, K_9) require special handling
3. Simulation engines expect variance around projections
4. Lower-is-better categories must be inverted for z-score based simulators
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from backend.services.row_projector import ROWProjectionResult
from backend.services.category_math import (
    CategoryMathResult,
    compute_all_category_math,
)
from backend.stat_contract import (
    SCORING_CATEGORY_CODES,
    LOWER_IS_BETTER,
    BATTING_CODES,
    PITCHING_CODES,
)


@dataclass
class SimulationInputBundle:
    """
    Complete input package for both simulation engines.

    Contains all the data needed to run H2H Monte Carlo or MCMC simulations
    from ROW projections.
    """
    # My team and opponent team ROW projections
    my_row_finals: Dict[str, float]
    opp_row_finals: Dict[str, float]

    # Category math results (margin, delta_to_flip, is_winning)
    category_math: Dict[str, CategoryMathResult]

    # Variance estimates for simulation (optional, uses defaults if None)
    my_variance: Optional[Dict[str, float]] = None
    opp_variance: Optional[Dict[str, float]] = None

    # NumPy RNG seed for reproducibility
    seed: Optional[int] = None


def _derive_ratio_components(
    row_finals: Dict[str, float],
    # Assumptions for deriving components from ratio stats
    avg_assumed_ab: float = 450.0,
    ops_assumed_pa: float = 450.0,
    era_assumed_ip: float = 90.0,
    whip_assumed_ip: float = 90.0,
    k9_assumed_ip: float = 90.0,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Derive synthetic numerators/denominators for ratio stats from ROW projections.

    Uses reasonable assumptions for weekly fantasy baseball:
    - AVG: assumes 450 AB → H = AVG * AB
    - OPS: assumes 450 PA → (H + BB + TB) / PA (simplified to H + TB)
    - ERA: assumes 90 IP → ER = ERA * IP / 9
    - WHIP: assumes 90 IP → (H + BB) = WHIP * IP (simplified to H)
    - K_9: assumes 90 IP → K = K_9 * IP / 9
    """
    numerators = {}
    denominators = {}

    # AVG: H = AVG * AB
    if "AVG" in row_finals:
        numerators["AVG"] = row_finals["AVG"] * avg_assumed_ab
        denominators["AVG"] = avg_assumed_ab

    # OPS: simplified to H + TB (ignores BB for numerator)
    if "OPS" in row_finals:
        # Use TB as proxy numerator
        numerators["OPS"] = row_finals.get("TB", 0.0)
        denominators["OPS"] = ops_assumed_pa

    # ERA: ER = ERA * IP / 9
    if "ERA" in row_finals:
        ip_outs = era_assumed_ip * 3  # Convert IP to outs
        er = row_finals["ERA"] * era_assumed_ip / 9.0
        numerators["ERA"] = er
        denominators["ERA"] = ip_outs

    # WHIP: (H + BB) ≈ H for simplified derivation
    if "WHIP" in row_finals:
        ip_outs = whip_assumed_ip * 3
        walks_plus_hits = row_finals["WHIP"] * whip_assumed_ip
        numerators["WHIP"] = walks_plus_hits
        denominators["WHIP"] = ip_outs

    # K_9: K = K_9 * IP / 9
    if "K_9" in row_finals:
        k = row_finals["K_9"] * k9_assumed_ip / 9.0
        numerators["K_9"] = k
        denominators["K_9"] = k9_assumed_ip * 3  # IP in outs

    return numerators, denominators


def prepare_simulation_inputs(
    my_row_projection: ROWProjectionResult,
    opp_row_projection: ROWProjectionResult,
    my_numerators: Optional[Dict[str, float]] = None,
    my_denominators: Optional[Dict[str, float]] = None,
) -> SimulationInputBundle:
    """
    Prepare simulation inputs from two ROWProjectionResult objects.

    This is the main entry point for the ROW→Simulation bridge.

    Parameters
    ----------
    my_row_projection / opp_row_projection:
        ROWProjectionResult objects for both teams.
    my_numerators / my_denominators:
        Optional ratio stat components for my team (for category_math).
        If not provided, will derive from ROW projections using assumptions.

    Returns
    -------
    SimulationInputBundle ready for both simulation engines.
    """
    # Convert to dicts
    my_finals = my_row_projection.to_dict()
    opp_finals = opp_row_projection.to_dict()

    # Validate all 18 categories present
    my_set = set(my_finals.keys())
    opp_set = set(opp_finals.keys())
    if my_set != SCORING_CATEGORY_CODES:
        raise ValueError(f"my_row_projection missing categories: {SCORING_CATEGORY_CODES - my_set}")
    if opp_set != SCORING_CATEGORY_CODES:
        raise ValueError(f"opp_row_projection missing categories: {SCORING_CATEGORY_CODES - opp_set}")

    # Derive ratio stat components (always derive, user can override)
    derived_nums, derived_denoms = _derive_ratio_components(my_finals)
    my_numerators = my_numerators or {}
    my_denominators = my_denominators or {}
    # User-provided values override derived defaults
    my_numerators = {**derived_nums, **my_numerators}
    my_denominators = {**derived_denoms, **my_denominators}

    # Compute category math (needed for flip probability analysis)
    # Note: opp_numerators/opp_denominators not supported by category_math module
    category_math = compute_all_category_math(
        my_finals=my_finals,
        opp_finals=opp_finals,
        my_numerators=my_numerators,
        my_denominators=my_denominators,
    )

    return SimulationInputBundle(
        my_row_finals=my_finals,
        opp_row_finals=opp_finals,
        category_math=category_math,
        seed=None,
    )


def prepare_h2h_monte_carlo_inputs(
    bundle: SimulationInputBundle,
    n_sims: int = 10000,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Convert SimulationInputBundle to H2H Monte Carlo inputs.

    H2H Monte Carlo's simulate_week_from_projections() expects:
    - my_finals: Dict[canonical_code, float]
    - opp_finals: Dict[canonical_code, float]

    This is a simple pass-through, but the function exists as a clear
    integration point for future enhancements (e.g., variance inflation).

    Parameters
    ----------
    bundle: SimulationInputBundle
    n_sims: Number of simulations (not used here, but kept for API consistency)

    Returns
    -------
    Tuple of (my_finals, opp_finals) ready for H2HOneWinSimulator.simulate_week_from_projections()
    """
    return bundle.my_row_finals, bundle.opp_row_finals


def prepare_mcmc_simulation_inputs(
    bundle: SimulationInputBundle,
    my_roster_z_scores: Optional[List[Dict]] = None,
    opp_roster_z_scores: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Convert SimulationInputBundle to MCMC simulator inputs.

    MCMC expects player-level rosters with z-score cat_scores.
    Since ROW projections are team-level, we need to create synthetic
    "super-player" entries that represent the team totals.

    Z-score conversion:
    - Counting stats: z = (my - opp) / std_dev_of_difference
    - Ratio stats: Use the category math margin directly

    Parameters
    ----------
    bundle: SimulationInputBundle
    my_roster_z_scores: Optional pre-computed player z-scores (overrides default)
    opp_roster_z_scores: Optional pre-computed opponent z-scores

    Returns
    -------
    Tuple of (my_roster, opp_roster) ready for simulate_weekly_matchup()
    """
    if my_roster_z_scores is not None and opp_roster_z_scores is not None:
        return my_roster_z_scores, opp_roster_z_scores

    # Default: create synthetic rosters with z-scores derived from projections
    # Z-score formula: z = (my_final - opp_final) / (sqrt(2) * typical_std)
    # This represents the advantage over opponent in z-score units

    # Typical standard deviations for weekly category totals
    _CATEGORY_STD = {
        # Batting counting
        "R": 8.0, "H": 15.0, "HR_B": 4.0, "RBI": 10.0, "K_B": 12.0, "TB": 20.0, "NSV": 3.0,
        # Batting ratio (handle separately via margin)
        "AVG": 0.020, "OPS": 0.050,
        # Pitching counting
        "W": 2.0, "L": 2.0, "HR_P": 3.0, "K_P": 15.0, "QS": 2.0,
        # Pitching ratio
        "ERA": 0.50, "WHIP": 0.15, "K_9": 1.0,
    }

    my_z_scores = {}
    for cat in SCORING_CATEGORY_CODES:
        my_val = bundle.my_row_finals.get(cat, 0.0)
        opp_val = bundle.opp_row_finals.get(cat, 0.0)

        # For ratio stats, use the margin directly scaled to z-units
        if cat in ("AVG", "OPS", "ERA", "WHIP", "K_9"):
            # Margin is already in the stat's units
            margin = my_val - opp_val
            std = _CATEGORY_STD.get(cat, 1.0)
            # For lower-is-better, flip the sign so higher z = better
            if cat in LOWER_IS_BETTER:
                margin = -margin
            z = margin / std if std > 0 else 0.0
        else:
            # Counting stats: z-score based on typical weekly std
            std = _CATEGORY_STD.get(cat, 5.0)
            diff = my_val - opp_val
            z = diff / std if std > 0 else 0.0

        my_z_scores[cat.lower()] = z

    # Create synthetic rosters (single "super player" per team)
    my_synthetic = [{
        "name": "My Team",
        "positions": ["DH"],
        "starts_this_week": 1,
        "cat_scores": my_z_scores,
    }]

    # Opponent z-scores are the negation of mine (zero-sum)
    opp_z_scores = {k: -v for k, v in my_z_scores.items()}
    opp_synthetic = [{
        "name": "Opponent",
        "positions": ["DH"],
        "starts_this_week": 1,
        "cat_scores": opp_z_scores,
    }]

    return my_synthetic, opp_synthetic


def calculate_variance_inflation(
    row_projection: ROWProjectionResult,
    days_remaining: int,
    games_remaining_by_player: Optional[Dict[str, int]] = None,
) -> Dict[str, float]:
    """
    Calculate variance inflation factors for ROW projections.

    ROW projections have uncertainty that increases with:
    - Fewer games remaining (higher variance)
    - More volatile stat categories

    Returns a dict of canonical_code -> variance_multiplier (1.0 = baseline)
    """
    # Base variance multipliers by category
    _BASE_VARIANCE = {
        # High volatility
        "NSV": 2.0, "HR_P": 1.8, "W": 1.6, "L": 1.6,
        # Medium volatility
        "QS": 1.4, "K_P": 1.3, "R": 1.2, "RBI": 1.2, "HR_B": 1.3,
        # Lower volatility
        "H": 1.1, "TB": 1.1, "K_B": 1.1,
        # Ratio stats (relatively stable)
        "AVG": 1.05, "OPS": 1.08, "ERA": 1.1, "WHIP": 1.1, "K_9": 1.15,
        "NSB": 1.5,
    }

    # Days remaining adjustment: fewer days = more variance
    if days_remaining <= 0:
        days_factor = 2.0  # High variance if no games left
    elif days_remaining >= 7:
        days_factor = 1.0  # Full week = baseline
    else:
        # Linear interpolation: 1 day = 1.5x, 7 days = 1.0x
        days_factor = 1.0 + (7 - days_remaining) * 0.5 / 6

    return {
        cat: mult * days_factor
        for cat, mult in _BASE_VARIANCE.items()
    }


def summarize_simulation_bundles(
    bundle: SimulationInputBundle,
) -> Dict[str, any]:
    """
    Create a human-readable summary of the simulation input bundle.

    Useful for debugging and UI display.
    """
    return {
        "my_projected_total": sum(bundle.my_row_finals.values()),
        "opp_projected_total": sum(bundle.opp_row_finals.values()),
        "categories_winning": sum(
            1 for r in bundle.category_math.values() if r.is_winning
        ),
        "categories_losing": sum(
            1 for r in bundle.category_math.values() if not r.is_winning and r.margin != 0
        ),
        "categories_tied": sum(
            1 for r in bundle.category_math.values() if r.margin == 0
        ),
        "biggest_lead": max(
            (r.margin for r in bundle.category_math.values() if r.is_winning),
            default=0.0
        ),
        "biggest_deficit": min(
            (r.margin for r in bundle.category_math.values() if not r.is_winning),
            default=0.0
        ),
        "swing_categories": [
            cat for cat, result in bundle.category_math.items()
            if abs(result.margin) < 1.0  # Within 1 unit
        ],
    }
