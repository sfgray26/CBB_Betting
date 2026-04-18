"""
P13 Category Math -- Margin and delta-to-flip calculations for H2H categories.

Pure computation module. Zero I/O. Stateless functions only.
Input:  current stats, ROW projections, opponent projections.
Output: margin (positive = winning), delta-to-flip (what's needed to win).

Core concepts
------------
- margin: Positive means I'm winning, negative means I'm losing
- delta_to_flip: The change needed to reverse who is winning the category
- Counting stats: delta is a simple unit count
- Ratio stats: delta is numerator units (hits, K, ER, etc.)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from backend.stat_contract import (
    BATTING_CODES,
    LOWER_IS_BETTER,
    SCORING_CATEGORY_CODES,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CategoryMathResult:
    """
    Margin and delta-to-flip for a single scoring category.

    Attributes
    ----------
    canonical_code : str
        The v2 canonical category code (e.g., "R", "ERA", "AVG").
    margin : float
        Positive = I'm winning, negative = I'm losing.
        For higher_is_better: my_final - opp_final.
        For lower_is_better: opp_final - my_final.
    delta_to_flip : float
        The change needed to reverse the winner.
        For counting stats: unit count needed.
        For ratio stats: numerator units needed (or allowed, for lower_is_better).
        Positive means I need this many to win; negative/zero means I've already won.
    is_winning : bool
        True if margin > 0 (I'm currently winning this category).
    display_delta : str
        Human-readable delta string for UI display.
    """
    canonical_code: str
    margin: float
    delta_to_flip: float
    is_winning: bool

    @property
    def display_delta(self) -> str:
        """Human-readable delta string."""
        if self.delta_to_flip <= 0:
            return "Lead safe"

        code = self.canonical_code
        is_lower_better = code in LOWER_IS_BETTER

        if is_lower_better:
            # Lower is better: display "Keep X ≤ Y" or "Need -Z units"
            if code == "ERA":
                return f"Allow ≤{self.delta_to_flip:.0f} more ER"
            elif code == "WHIP":
                return f"Allow ≤{self.delta_to_flip:.0f} more H+BB"
            elif code == "K_B":
                return f"Need {self.delta_to_flip:.0f} fewer K"
            elif code == "L":
                return f"Need {self.delta_to_flip:.0f} fewer L"
            elif code == "HR_P":
                return f"Need {self.delta_to_flip:.0f} fewer HRA"
            else:
                return f"Need {self.delta_to_flip:.0f} fewer"
        else:
            # Higher is better: display "Need +X units"
            if code == "AVG":
                return f"Need ~{self.delta_to_flip:.0f} more H"
            elif code == "OPS":
                return f"Need +{self.delta_to_flip:.0f} TB-equivalent"
            elif code == "K_9":
                return f"Need +{self.delta_to_flip:.0f} K"
            elif code == "NSB":
                return f"Need +{self.delta_to_flip:.0f} NSB"
            else:
                return f"Need +{self.delta_to_flip:.0f} {code}"


# ---------------------------------------------------------------------------
# Counting stats: simple margin and delta
# ---------------------------------------------------------------------------

def compute_counting_margin(
    my_final: float,
    opp_final: float,
    is_lower_better: bool,
) -> float:
    """
    Compute margin for a counting stat category.

    Margin > 0 means I'm winning.
    For higher_is_better: my_final - opp_final.
    For lower_is_better: opp_final - my_final.

    Examples
    --------
    >>> compute_counting_margin(10, 5, False)  # Higher is better, I'm winning
    5.0
    >>> compute_counting_margin(5, 10, False)  # Higher is better, I'm losing
    -5.0
    >>> compute_counting_margin(30, 40, True)   # Lower is better (K), I'm winning
    10.0
    """
    if is_lower_better:
        return opp_final - my_final
    return my_final - opp_final


def compute_counting_delta_to_flip(
    my_final: float,
    opp_final: float,
    is_lower_better: bool,
) -> float:
    """
    Compute delta-to-flip for a counting stat.

    Returns the unit change needed to reverse the winner.
    Positive value means I need this many units to win.
    Zero or negative means I've already won (or tied, in which case 1 unit flips it).

    Examples
    --------
    >>> compute_counting_delta_to_flip(10, 15, False)  # Need +6 runs
    6.0
    >>> compute_counting_delta_to_flip(15, 10, False)  # Already winning
    -4.0
    >>> compute_counting_delta_to_flip(50, 40, True)   # Need -11 K (fewer)
    -9.0
    """
    if is_lower_better:
        # Lower is better: I need to reduce my count
        return my_final - opp_final - 1
    else:
        # Higher is better: I need to increase my count
        return opp_final - my_final + 1


# ---------------------------------------------------------------------------
# Ratio stats: margin and delta require numerator/denominator
# ---------------------------------------------------------------------------

def compute_ratio_margin(
    my_final: float,
    opp_final: float,
    is_lower_better: bool,
) -> float:
    """
    Compute margin for a ratio stat category.

    Uses the same sign convention as counting stats:
    margin > 0 means I'm winning.

    Examples
    --------
    >>> compute_ratio_margin(0.280, 0.250, False)  # AVG, I'm winning
    0.03
    >>> compute_ratio_margin(3.50, 4.00, True)    # ERA, I'm winning
    0.5
    """
    if is_lower_better:
        return opp_final - my_final
    return my_final - opp_final


def compute_ratio_delta_to_flip(
    my_numerator: float,
    my_denominator: float,
    opp_ratio: float,
    is_lower_better: bool,
    *,
    precision: int = 0,
) -> float:
    """
    Compute delta-to-flip for a ratio stat.

    Returns the numerator change needed to reverse the winner.
    Assumes denominator stays fixed (new AB/IP are known from ROW).

    Parameters
    ----------
    my_numerator : float
        My current + projected numerator (H for AVG, ER for ERA, etc.).
    my_denominator : float
        My current + projected denominator (AB for AVG, IP*27 for ERA, etc.).
        Must be > 0.
    opp_ratio : float
        Opponent's projected final ratio.
    is_lower_better : bool
        True for ERA/WHIP (lower is better), False for AVG/OPS/K_9.
    precision : int
        Decimal precision for rounding (default 0 = integer units).

    Returns
    -------
    Delta in numerator units. Positive means I need this many to win.
    Zero or negative means I've already won.

    Examples
    --------
    >>> # AVG: I have 25/100 (.250), opp has .267. Need +3 H.
    >>> compute_ratio_delta_to_flip(25, 100, 0.267, False)
    3.0

    >>> # ERA: I have 35 ER / 105 IP_outs (3.00 ERA), opp has 3.50.
    >>> # Allow up to 6 more ER before losing.
    >>> compute_ratio_delta_to_flip(35, 105, 3.50, True)
    7.0
    """
    if my_denominator <= 0:
        raise ValueError(f"Denominator must be positive, got {my_denominator}")

    if is_lower_better:
        # Lower is better: max allowed numerator to stay below opp_ratio
        # my_ratio < opp_ratio  ->  my_num / den < opp_ratio
        # my_num < opp_ratio * den
        max_allowed = opp_ratio * my_denominator - my_numerator
        # Round down (conservative) and add 1 for safety margin
        return math.floor(max_allowed)
    else:
        # Higher is better: min needed numerator to exceed opp_ratio
        # my_ratio > opp_ratio  ->  my_num / den > opp_ratio
        # my_num > opp_ratio * den
        min_needed = opp_ratio * my_denominator - my_numerator
        # Round up (need full unit) and add 1 to guarantee win
        return math.ceil(min_needed + 1)


# ---------------------------------------------------------------------------
# Unified category math function
# ---------------------------------------------------------------------------

def compute_category_math(
    canonical_code: str,
    my_final: Optional[float] = None,
    opp_final: Optional[float] = None,
    *,
    my_numerator: Optional[float] = None,
    my_denominator: Optional[float] = None,
    opp_numerator: Optional[float] = None,
    opp_denominator: Optional[float] = None,
) -> CategoryMathResult:
    """
    Compute margin and delta-to-flip for any scoring category.

    Parameters
    ----------
    canonical_code : str
        The v2 canonical category code (e.g., "R", "ERA", "AVG").
    my_final : float, optional
        My projected final value. Required for counting stats.
    opp_final : float, optional
        Opponent's projected final value. Required for counting stats.
    my_numerator : float, optional
        My numerator for ratio stats (e.g., H for AVG, ER for ERA).
        Required for ratio stats to compute delta-to-flip.
    my_denominator : float, optional
        My denominator for ratio stats (e.g., AB for AVG, IP_outs for ERA).
        Required for ratio stats.
    opp_numerator : float, optional
        Opponent's numerator (not currently used, reserved for future).
    opp_denominator : float, optional
        Opponent's denominator (not currently used, reserved for future).

    Returns
    -------
    CategoryMathResult with margin, delta_to_flip, and display string.

    Raises
    ------
    ValueError if required inputs are missing for the category type.
    """
    is_lower_better = canonical_code in LOWER_IS_BETTER
    is_ratio = canonical_code in {"AVG", "OPS", "ERA", "WHIP", "K_9"}

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if is_ratio:
        if my_numerator is None or my_denominator is None:
            raise ValueError(
                f"Ratio stat {canonical_code} requires my_numerator and my_denominator"
            )
        if my_denominator <= 0:
            raise ValueError(
                f"Denominator must be positive for {canonical_code}, got {my_denominator}"
            )
        # Compute my_final from numerator/denominator if not provided
        if my_final is None:
            if canonical_code == "ERA":
                my_final = 27.0 * my_numerator / my_denominator
            elif canonical_code == "WHIP":
                # WHIP numerator is H + BB
                my_final = 3.0 * my_numerator / my_denominator
            elif canonical_code == "K_9":
                my_final = 27.0 * my_numerator / my_denominator
            else:
                # AVG, OPS
                my_final = my_numerator / my_denominator
    else:
        if my_final is None or opp_final is None:
            raise ValueError(
                f"Counting stat {canonical_code} requires my_final and opp_final"
            )

    # ------------------------------------------------------------------
    # Compute margin
    # ------------------------------------------------------------------
    if is_ratio:
        margin = compute_ratio_margin(my_final, opp_final or 0.0, is_lower_better)
    else:
        margin = compute_counting_margin(my_final or 0.0, opp_final or 0.0, is_lower_better)

    # ------------------------------------------------------------------
    # Compute delta-to-flip
    # ------------------------------------------------------------------
    if is_ratio:
        # For ratio stats, we need opponent's ratio
        opp_ratio = opp_final or 0.0
        delta = compute_ratio_delta_to_flip(
            my_numerator or 0.0,
            my_denominator or 1.0,
            opp_ratio,
            is_lower_better,
        )
    else:
        delta = compute_counting_delta_to_flip(
            my_final or 0.0,
            opp_final or 0.0,
            is_lower_better,
        )

    is_winning = margin > 0

    return CategoryMathResult(
        canonical_code=canonical_code,
        margin=margin,
        delta_to_flip=delta,
        is_winning=is_winning,
    )


# ---------------------------------------------------------------------------
# Batch computation for all categories
# ---------------------------------------------------------------------------

def compute_all_category_math(
    my_finals: Dict[str, float],
    opp_finals: Dict[str, float],
    my_numerators: Optional[Dict[str, float]] = None,
    my_denominators: Optional[Dict[str, float]] = None,
) -> Dict[str, CategoryMathResult]:
    """
    Compute margin and delta-to-flip for all 18 scoring categories.

    Parameters
    ----------
    my_finals : dict
        {canonical_code: my_projected_final}
    opp_finals : dict
        {canonical_code: opp_projected_final}
    my_numerators : dict, optional
        {canonical_code: my_numerator} for ratio stats.
    my_denominators : dict, optional
        {canonical_code: my_denominator} for ratio stats.

    Returns
    -------
    dict of {canonical_code: CategoryMathResult}
    """
    my_numerators = my_numerators or {}
    my_denominators = my_denominators or {}

    results = {}
    for code in SCORING_CATEGORY_CODES:
        results[code] = compute_category_math(
            canonical_code=code,
            my_final=my_finals.get(code),
            opp_final=opp_finals.get(code),
            my_numerator=my_numerators.get(code),
            my_denominator=my_denominators.get(code),
        )
    return results


# ---------------------------------------------------------------------------
# Helper functions for deriving numerators/denominators
# ---------------------------------------------------------------------------

def derive_avg_components(
    current_h: float,
    row_h: float,
    current_ab: float,
    row_ab: float,
) -> Tuple[float, float]:
    """
    Derive AVG numerator/denominator from current + ROW.

    Returns (total_h, total_ab).
    """
    return (current_h + row_h, current_ab + row_ab)


def derive_era_components(
    current_er: float,
    row_er: float,
    current_ip: float,
    row_ip: float,
) -> Tuple[float, float]:
    """
    Derive ERA numerator/denominator from current + ROW.

    Note: ERA uses IP_outs (IP * 3) as denominator to avoid fractional IP issues.

    Returns (total_er, total_ip_outs).
    """
    total_er = current_er + row_er
    total_ip_outs = (current_ip + row_ip) * 3  # Convert IP to outs
    return (total_er, total_ip_outs)


def derive_whip_components(
    current_h_allowed: float,
    row_h_allowed: float,
    current_bb_allowed: float,
    row_bb_allowed: float,
    current_ip: float,
    row_ip: float,
) -> Tuple[float, float]:
    """
    Derive WHIP numerator/denominator from current + ROW.

    WHIP numerator is H + BB. Denominator uses IP_outs.

    Returns (total_h_plus_bb, total_ip_outs).
    """
    total_h_plus_bb = (current_h_allowed + row_h_allowed +
                      current_bb_allowed + row_bb_allowed)
    total_ip_outs = (current_ip + row_ip) * 3
    return (total_h_plus_bb, total_ip_outs)


def derive_k9_components(
    current_k: float,
    row_k: float,
    current_ip: float,
    row_ip: float,
) -> Tuple[float, float]:
    """
    Derive K/9 numerator/denominator from current + ROW.

    Returns (total_k, total_ip_outs).
    """
    total_k = current_k + row_k
    total_ip_outs = (current_ip + row_ip) * 3
    return (total_k, total_ip_outs)


def derive_ops_components(
    current_h: float,
    row_h: float,
    current_bb: float,
    row_bb: float,
    current_tb: float,
    row_tb: float,
    current_ab: float,
    row_ab: float,
) -> Tuple[float, float, float]:
    """
    Derive OPS numerator/denominator from current + ROW.

    OPS = OBP + SLG = (H+BB)/(AB+BB) + TB/AB.

    Returns (total_h_plus_bb, total_obp_denom, total_tb, total_slg_denom).
    """
    total_h_plus_bb = (current_h + row_h) + (current_bb + row_bb)
    total_obp_denom = (current_ab + row_ab) + (current_bb + row_bb)
    total_tb = current_tb + row_tb
    total_slg_denom = current_ab + row_ab
    return (total_h_plus_bb, total_obp_denom, total_tb, total_slg_denom)
