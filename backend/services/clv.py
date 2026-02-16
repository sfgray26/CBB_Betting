"""
Closing Line Value (CLV) calculation service.

CLV is the primary edge-validation metric in sports betting.
Positive CLV means we obtained better odds than where the market
ultimately settled (the closing line), which is correlated with
long-term profitability independent of win/loss outcomes.

Two calculation modes are supported:

    1. Spread CLV  — the spread itself moved (e.g., -4.5 → -6).
       clv_points  = opening_spread - closing_spread  (positive = good)
       clv_prob    = P(cover | closing_spread as true center) - 0.50

    2. Juice CLV   — spread held but juice moved (e.g., -110 → -115).
       Uses no-vig probability on both the opening and closing juice.
       clv_prob    = closing_novig_prob - opening_novig_prob

In practice a full bet may have both a spread move AND a juice move.
`calculate_clv_full` handles both simultaneously.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class CLVResult:
    """All CLV metrics for a single bet."""

    # Primary metrics (used for paper trade evaluation)
    clv_points: float       # Spread points better than closing line (positive = good)
    clv_prob: float         # Probability edge vs closing market (positive = good)

    # Derived / diagnostic
    cover_prob: float       # Win probability of our bet given closing spread as truth
    opening_novig: float    # No-vig implied prob of our side at bet-placement
    closing_novig: float    # No-vig implied prob of our side at close

    opening_spread: Optional[float]
    closing_spread: Optional[float]
    opening_odds: float
    closing_odds: float

    def is_positive(self) -> bool:
        """True when CLV is positive (we beat the closing line)."""
        return self.clv_prob > 0

    def grade(self) -> str:
        """Human-readable CLV grade for display."""
        if self.clv_prob >= 0.03:
            return "STRONG+"
        elif self.clv_prob >= 0.01:
            return "POSITIVE"
        elif self.clv_prob >= -0.01:
            return "NEUTRAL"
        elif self.clv_prob >= -0.03:
            return "NEGATIVE"
        return "STRONG-"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_american_odds(odds: float, name: str = "odds") -> None:
    """Raise ValueError for obviously invalid American odds."""
    if odds == 0:
        raise ValueError(f"{name} cannot be 0")
    # American odds are either >= +100 or <= -100
    if -99 < odds < 100 and odds != 0:
        raise ValueError(
            f"{name}={odds} is not a valid American odds value. "
            "Must be >= +100 (underdog) or <= -100 (favourite)."
        )


def _implied_prob(american_odds: float) -> float:
    """Convert American odds to raw implied probability (vig included)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def _remove_vig(odds_a: float, odds_b: float) -> Tuple[float, float]:
    """
    Remove vig from two sides of a market.

    Returns (no_vig_prob_a, no_vig_prob_b) that sum to 1.0.
    Raises ValueError if either side's implied probability is zero.
    """
    p_a = _implied_prob(odds_a)
    p_b = _implied_prob(odds_b)
    total = p_a + p_b
    if total <= 0:
        raise ValueError(f"Sum of implied probabilities is {total}; check odds inputs.")
    return p_a / total, p_b / total


def _spread_cover_prob(
    opening_spread: float,
    closing_spread: float,
    base_sd: float,
) -> float:
    """
    P(cover opening_spread | true expected margin = -closing_spread, SD = base_sd).

    Key identity:
        clv_points = opening_spread - closing_spread
        cover_prob = norm.cdf(clv_points / base_sd)

    Verified for both favourites and underdogs:
        Favourite: bet -4.5, closes -6  → clv_points=1.5  → norm.cdf(0.136) ≈ 0.554  (good)
        Favourite: bet -4.5, closes -3  → clv_points=-1.5 → norm.cdf(-0.136) ≈ 0.446 (bad)
        Underdog:  bet +4.5, closes +3  → clv_points=1.5  → norm.cdf(0.136) ≈ 0.554  (good)
        Underdog:  bet +4.5, closes +6  → clv_points=-1.5 → norm.cdf(-0.136) ≈ 0.446 (bad)
    """
    clv_points = opening_spread - closing_spread
    return float(norm.cdf(clv_points / base_sd))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_clv_spread(
    opening_spread: float,
    closing_spread: float,
    opening_odds: float = -110,
    closing_odds: float = -110,
    other_side_opening_odds: Optional[float] = None,
    other_side_closing_odds: Optional[float] = None,
    base_sd: float = 11.0,
) -> CLVResult:
    """
    Full CLV for a spread bet where the line itself moved.

    This is the primary CLV calculation for paper trading. It captures
    both the spread movement and any juice change.

    Args:
        opening_spread:           Spread on our side when we placed the bet
                                  (negative = favourite, e.g. -4.5).
        closing_spread:           Spread on our side at game-time close
                                  (e.g. -6.0 means market moved toward us).
        opening_odds:             Juice on our side at placement (e.g. -110).
        closing_odds:             Juice on our side at close (e.g. -115).
        other_side_opening_odds:  Other side's juice at placement. Defaults to -110.
        other_side_closing_odds:  Other side's juice at close. Defaults to -110.
        base_sd:                  Model SD for point→probability conversion (default 11).

    Returns:
        CLVResult with clv_points and clv_prob as primary metrics.
    """
    _validate_american_odds(opening_odds, "opening_odds")
    _validate_american_odds(closing_odds, "closing_odds")

    other_open = other_side_opening_odds or -110
    other_close = other_side_closing_odds or -110

    if other_open != -110:
        _validate_american_odds(other_open, "other_side_opening_odds")
    if other_close != -110:
        _validate_american_odds(other_close, "other_side_closing_odds")

    # Points CLV (the most interpretable metric)
    clv_points = opening_spread - closing_spread

    # Probability CLV from spread movement
    cover_prob = _spread_cover_prob(opening_spread, closing_spread, base_sd)
    clv_prob_spread = cover_prob - 0.5  # Edge vs 50%

    # No-vig probs for diagnostic / juice-side CLV
    opening_novig, _ = _remove_vig(opening_odds, other_open)
    closing_novig, _ = _remove_vig(closing_odds, other_close)

    # Blend: spread move is the primary signal; juice move is secondary
    # clv_prob uses spread-derived cover_prob (most reliable for spread bets)
    return CLVResult(
        clv_points=clv_points,
        clv_prob=clv_prob_spread,
        cover_prob=cover_prob,
        opening_novig=opening_novig,
        closing_novig=closing_novig,
        opening_spread=opening_spread,
        closing_spread=closing_spread,
        opening_odds=opening_odds,
        closing_odds=closing_odds,
    )


def calculate_clv_juice_only(
    opening_odds: float,
    closing_odds: float,
    other_side_opening_odds: Optional[float] = None,
    other_side_closing_odds: Optional[float] = None,
) -> CLVResult:
    """
    CLV when only juice moved (spread held constant).

    Use this when you have the closing juice but not the closing spread,
    or when tracking moneyline bets.

    Args:
        opening_odds:             American odds at bet placement.
        closing_odds:             American odds at close.
        other_side_opening_odds:  Other side at open. Defaults to -110.
        other_side_closing_odds:  Other side at close. Defaults to -110.

    Returns:
        CLVResult. clv_points is 0.0 (no spread movement).
        clv_prob = closing_novig - opening_novig.
    """
    _validate_american_odds(opening_odds, "opening_odds")
    _validate_american_odds(closing_odds, "closing_odds")

    other_open = other_side_opening_odds or -110
    other_close = other_side_closing_odds or -110

    opening_novig, _ = _remove_vig(opening_odds, other_open)
    closing_novig, _ = _remove_vig(closing_odds, other_close)

    # Positive when closing juice implies higher win prob (market moved toward us)
    clv_prob = closing_novig - opening_novig

    return CLVResult(
        clv_points=0.0,
        clv_prob=clv_prob,
        cover_prob=closing_novig,
        opening_novig=opening_novig,
        closing_novig=closing_novig,
        opening_spread=None,
        closing_spread=None,
        opening_odds=opening_odds,
        closing_odds=closing_odds,
    )


def calculate_clv_full(
    opening_odds: float,
    closing_odds: float,
    opening_spread: Optional[float] = None,
    closing_spread: Optional[float] = None,
    other_side_opening_odds: Optional[float] = None,
    other_side_closing_odds: Optional[float] = None,
    base_sd: float = 11.0,
) -> CLVResult:
    """
    Unified CLV calculation. Dispatches to the appropriate method.

    If both opening_spread and closing_spread are provided, uses
    calculate_clv_spread (preferred). Otherwise falls back to juice-only.

    This is the function to call from the API endpoint.
    """
    if opening_spread is not None and closing_spread is not None:
        return calculate_clv_spread(
            opening_spread=opening_spread,
            closing_spread=closing_spread,
            opening_odds=opening_odds,
            closing_odds=closing_odds,
            other_side_opening_odds=other_side_opening_odds,
            other_side_closing_odds=other_side_closing_odds,
            base_sd=base_sd,
        )
    return calculate_clv_juice_only(
        opening_odds=opening_odds,
        closing_odds=closing_odds,
        other_side_opening_odds=other_side_opening_odds,
        other_side_closing_odds=other_side_closing_odds,
    )
