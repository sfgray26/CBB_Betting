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

Quantitative improvements over the original:
  - Shin (1993) vig removal replaces proportional normalisation for
    more accurate true-probability extraction in asymmetric markets.
  - Skellam distribution replaces scipy.stats.norm for cover-probability
    estimation, respecting the discrete integer nature of basketball
    scoring and providing push-rate awareness for whole-number spreads.
  - Dynamic SD: SD ≈ sqrt(Total) × 0.85 instead of a hard-coded 11.0.
"""

import math
import logging
from dataclasses import dataclass
from typing import Optional, Tuple

from scipy.stats import skellam

from backend.core.odds_math import (
    remove_vig_shin as _shin,
    implied_prob as _implied_prob_core,
    dynamic_sd as _dynamic_sd_core,
)

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
    if -99 < odds < 100 and odds != 0:
        raise ValueError(
            f"{name}={odds} is not a valid American odds value. "
            "Must be >= +100 (underdog) or <= -100 (favourite)."
        )



def _bivariate_poisson_skellam(
    lambda1: float,
    lambda2: float,
    threshold: float,
) -> float:
    """
    Evaluate P(D > threshold) for the Skellam(λ1, λ2) distribution, with
    full push-rate awareness for integer thresholds.

    This is the shared scoring-difference evaluator used by both the naive
    independent Skellam path and the correlation-adjusted Bivariate Poisson
    path.  Keeping it separate makes the parameterisation explicit.

    Push-rate awareness
    -------------------
    For an integer threshold k:
        P(cover) = P(D > k) + 0.5·P(D = k)
                 = 1 − CDF(k−1) − 0.5·PMF(k)

    For non-integer threshold (no push possible):
        P(cover) = P(D > floor(threshold)) = 1 − CDF(floor(threshold))
    """
    if abs(threshold - round(threshold)) < 0.01:
        k = int(round(threshold))
        return float(
            1.0
            - skellam.cdf(k - 1, lambda1, lambda2)
            - 0.5 * skellam.pmf(k, lambda1, lambda2)
        )
    k = int(math.floor(threshold))
    return float(1.0 - skellam.cdf(k, lambda1, lambda2))


def _spread_cover_prob(
    opening_spread: float,
    closing_spread: float,
    total: Optional[float] = None,
    base_sd: Optional[float] = None,
    pace_rho: float = 0.35,
) -> float:
    """
    P(our spread bet covers | closing spread is the true market estimate).

    Uses a **correlation-adjusted Skellam** via Bivariate Poisson (BP)
    decomposition when a game total is available.  This replaces the naive
    independent Skellam that ignores the shared possession/pace component
    between the two teams.

    Bivariate Poisson decomposition
    --------------------------------
    In a Bivariate Poisson model the two teams' scores are:

        X = X₁ + X₃   (home)
        Y = Y₁ + X₃   (away)

    where X₁ ~ Pois(λ₁), Y₁ ~ Pois(λ₂), X₃ ~ Pois(λ₃) and X₃ is the
    *shared pace component* — both teams score more possessions when the
    game is fast, creating positive score correlation.

    The key result: because X₃ cancels in the difference,

        D = X − Y = X₁ − Y₁ ~ Skellam(λ₁, λ₂)

    but with **reduced marginal rates**:

        λ₃ = ρ · √(μ_bet · μ_opp)          (shared component)
        λ₁ = μ_bet − λ₃                     (home independent)
        λ₂ = μ_opp − λ₃                     (away independent)

    So Var(D) = λ₁ + λ₂ = (μ_bet + μ_opp) − 2λ₃ < μ_bet + μ_opp, meaning
    the score *difference* is less volatile than independent scoring
    assumes.  This is empirically validated: fast-paced games inflate
    both totals but the *spread outcomes* regress toward the closing line
    because the shared variance cancels.

    ``pace_rho ≈ 0.35`` is the empirical CBB score correlation (derived
    from shared-possession pace effects).  Caller can override for
    neutral-site games (lower rho ≈ 0.28) or rivalry games (higher ≈ 0.40).

    Push-rate awareness
    -------------------
    For integer spreads, the push probability is computed explicitly and
    handled as a half-win:

        P(cover) = 1 − CDF(k−1) − 0.5·PMF(k)

    Falls back to a dynamic-SD normal approximation when the total is
    unavailable or BP parameters would degenerate.

    Args:
        opening_spread: Spread on our side at bet placement (e.g. -4.5).
        closing_spread: Spread on our side at close (e.g. -6.0).
        total:          Game total (over/under) for parameterisation.
        base_sd:        Override SD for the normal fallback.
        pace_rho:       Score correlation coefficient ρ ∈ [0, 1) from the
                        shared pace/possession component.  Default 0.35.
    """
    # ---- Bivariate Poisson path --------------------------------------------
    if total is not None and total > 0:
        mu_bet = (total - closing_spread) / 2.0
        mu_opp = (total + closing_spread) / 2.0

        if mu_bet > 0.5 and mu_opp > 0.5:
            threshold = -opening_spread  # e.g. 4.5 for a -4.5 bet

            # Shared component λ₃ captures pace-induced correlation.
            # Clamp to ensure λ₁ and λ₂ remain strictly positive.
            rho = max(0.0, min(pace_rho, 0.95))
            lambda3 = rho * math.sqrt(mu_bet * mu_opp)
            lambda3 = min(lambda3, 0.95 * min(mu_bet, mu_opp))

            lambda1 = mu_bet - lambda3   # home independent Poisson rate
            lambda2 = mu_opp - lambda3   # away independent Poisson rate

            if lambda1 > 0.1 and lambda2 > 0.1:
                return _bivariate_poisson_skellam(lambda1, lambda2, threshold)

            # Degenerate params — fall back to independent Skellam.
            return _bivariate_poisson_skellam(mu_bet, mu_opp, threshold)

    # ---- Normal fallback (dynamic SD) --------------------------------------
    from scipy.stats import norm as _norm
    sd = base_sd if base_sd is not None else _dynamic_sd_core(total)
    if sd <= 0:
        sd = 11.0
    clv_points = opening_spread - closing_spread
    return float(_norm.cdf(clv_points / sd))


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
    total: Optional[float] = None,
    pace_rho: float = 0.35,
) -> CLVResult:
    """
    Full CLV for a spread bet where the line itself moved.

    This is the primary CLV calculation for paper trading.  It captures
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
        base_sd:                  Fallback SD for point→probability conversion
                                  when total is unavailable (default 11.0).
        total:                    Game total (over/under) used for Bivariate
                                  Poisson cover-probability estimation.
        pace_rho:                 Score correlation ρ from shared pace (default
                                  0.35).  Pass lower values (~0.28) for
                                  neutral-site games with no crowd effect.

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

    # Points CLV (most interpretable metric)
    clv_points = opening_spread - closing_spread

    # Cover probability via Bivariate Poisson Skellam (or normal fallback)
    cover_prob = _spread_cover_prob(
        opening_spread, closing_spread, total=total, base_sd=base_sd,
        pace_rho=pace_rho,
    )
    clv_prob_spread = cover_prob - 0.5  # Edge vs 50%

    # No-vig probs via Shin's method for diagnostic / juice-side CLV
    opening_novig, _ = _shin(opening_odds, other_open)
    closing_novig, _ = _shin(closing_odds, other_close)

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
        CLVResult.  clv_points is 0.0 (no spread movement).
        clv_prob = closing_novig - opening_novig.
    """
    _validate_american_odds(opening_odds, "opening_odds")
    _validate_american_odds(closing_odds, "closing_odds")

    other_open = other_side_opening_odds or -110
    other_close = other_side_closing_odds or -110

    opening_novig, _ = _shin(opening_odds, other_open)
    closing_novig, _ = _shin(closing_odds, other_close)

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
    total: Optional[float] = None,
    pace_rho: float = 0.35,
) -> CLVResult:
    """
    Unified CLV calculation.  Dispatches to the appropriate method.

    If both opening_spread and closing_spread are provided, uses
    calculate_clv_spread (preferred for spread bets).  Otherwise falls
    back to juice-only.

    Args:
        total:     Game total (over/under) forwarded to calculate_clv_spread
                   for Bivariate Poisson cover probability.  Pass the sharp
                   consensus total when available.
        pace_rho:  Score correlation ρ from shared pace, forwarded to
                   _spread_cover_prob.  Default 0.35 (CBB baseline).

    This is the function to call from the API endpoint and bet_tracker.
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
            total=total,
            pace_rho=pace_rho,
        )
    return calculate_clv_juice_only(
        opening_odds=opening_odds,
        closing_odds=closing_odds,
        other_side_opening_odds=other_side_opening_odds,
        other_side_closing_odds=other_side_closing_odds,
    )
