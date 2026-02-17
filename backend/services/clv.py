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


def _implied_prob(american_odds: float) -> float:
    """Convert American odds to raw implied probability (vig included)."""
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return abs(american_odds) / (abs(american_odds) + 100.0)


def _dynamic_sd(total: Optional[float]) -> float:
    """
    Game-total-based standard deviation: SD ≈ sqrt(Total) × 0.85.

    Replaces the hard-coded base_sd=11.0 with a value that scales with
    the expected combined score. At T=145 (typical CBB), SD ≈ 10.2 pts.
    Falls back to 11.0 when total is unavailable or non-positive.
    """
    if total is None or total <= 0:
        return 11.0
    return math.sqrt(total) * 0.85


def _remove_vig_shin(odds_a: float, odds_b: float) -> Tuple[float, float]:
    """
    Remove vig using Shin (1993) method for true probability extraction.

    Unlike proportional normalisation, Shin's method accounts for insider
    information embedded in the overround, giving more accurate true
    probabilities for asymmetric markets (e.g. moneylines).

    For symmetric spread markets (-110/-110) the result equals proportional
    normalisation and the function short-circuits immediately.

    Algorithm
    ---------
    1. Estimate insider fraction z from overround using the closed-form
       symmetric approximation (Shin 1992): k ≈ 1/(1 − z/2) ⟹ z ≈ 2(k−1)/k.
    2. Solve the Shin equation for true prob q_a via bisection:
           ω_a/K = (1−z)·q_a + z·q_a²/(q_a²+(1−q_a)²)
       where ω_a/K is the normalised implied probability of side A.

    Returns
    -------
    (true_prob_a, true_prob_b) summing to 1.0.
    """
    p_a = _implied_prob(odds_a)
    p_b = _implied_prob(odds_b)
    k = p_a + p_b  # total overround (> 1 due to vig)

    if k <= 0:
        raise ValueError(f"Sum of implied probabilities is {k}; check odds inputs.")

    # Symmetric market: proportional == Shin (short-circuit)
    if abs(p_a - p_b) < 1e-9:
        return p_a / k, p_b / k

    # Insider fraction z estimated from overround (Shin 1992)
    z = 2.0 * (k - 1.0) / k
    z = max(0.0, min(z, 0.999))

    # Normalised implied prob for side A
    w_a = p_a / k

    # Shin equation residual: f(q) = (1−z)·q + z·q²/(q²+(1−q)²) − w_a
    def residual(q: float) -> float:
        D = q * q + (1.0 - q) * (1.0 - q)
        return (1.0 - z) * q + z * q * q / D - w_a

    lo, hi = 1e-9, 1.0 - 1e-9
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if residual(mid) < 0.0:
            lo = mid
        else:
            hi = mid

    q_a = 0.5 * (lo + hi)
    q_b = 1.0 - q_a
    return q_a, q_b


def _spread_cover_prob(
    opening_spread: float,
    closing_spread: float,
    total: Optional[float] = None,
    base_sd: Optional[float] = None,
) -> float:
    """
    P(our spread bet covers | closing spread is the true market estimate).

    Uses a Skellam distribution when a game total is available.  The
    Skellam(μ_bet, μ_opp) distribution models the score differential as
    the difference of two independent Poisson RVs, which respects the
    discrete integer nature of basketball scoring.

    Parameterisation
    ----------------
    Given closing_spread s (negative = favourite) and total T:
        μ_bet = (T − s) / 2    (expected score of the team we bet on)
        μ_opp = (T + s) / 2    (expected opponent score)

    Cover condition for a bet placed at opening_spread:
        We win if score_diff > −opening_spread (threshold).

    Push-rate awareness
    -------------------
    For integer-valued spreads (e.g. -4.0) a tie at exactly the spread is
    a push (stake returned, $0 P&L).  The cover probability accounts for
    this half-win:
        P(cover) = P(D > k) + 0.5·P(D = k)
                 = 1 − CDF(k−1) − 0.5·PMF(k)
    where k = threshold = −opening_spread (integer).

    Falls back to a dynamic-SD normal approximation when the total is
    unavailable or Skellam parameters would be non-positive.

    Args:
        opening_spread: Spread on our side at bet placement (e.g. -4.5).
        closing_spread: Spread on our side at close (e.g. -6.0).
        total:          Game total (over/under) for Skellam parameterisation.
        base_sd:        Override SD for the normal fallback.  If None, uses
                        _dynamic_sd(total).
    """
    # ---- Skellam path -------------------------------------------------------
    if total is not None and total > 0:
        # Poisson rate for each team
        mu_bet = (total - closing_spread) / 2.0
        mu_opp = (total + closing_spread) / 2.0

        if mu_bet > 0.5 and mu_opp > 0.5:
            threshold = -opening_spread  # e.g. 4.5 for a -4.5 bet

            if abs(threshold - round(threshold)) < 0.01:
                # Integer spread — push-rate aware
                k = int(round(threshold))
                return float(
                    1.0
                    - skellam.cdf(k - 1, mu_bet, mu_opp)
                    - 0.5 * skellam.pmf(k, mu_bet, mu_opp)
                )
            else:
                # Non-integer spread — no push possible
                k = int(math.floor(threshold))
                return float(1.0 - skellam.cdf(k, mu_bet, mu_opp))

    # ---- Normal fallback (dynamic SD) ---------------------------------------
    from scipy.stats import norm as _norm
    sd = base_sd if base_sd is not None else _dynamic_sd(total)
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
        total:                    Game total (over/under) used for Skellam
                                  cover-probability estimation.

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

    # Cover probability via Skellam (or normal fallback)
    cover_prob = _spread_cover_prob(
        opening_spread, closing_spread, total=total, base_sd=base_sd
    )
    clv_prob_spread = cover_prob - 0.5  # Edge vs 50%

    # No-vig probs via Shin's method for diagnostic / juice-side CLV
    opening_novig, _ = _remove_vig_shin(opening_odds, other_open)
    closing_novig, _ = _remove_vig_shin(closing_odds, other_close)

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

    opening_novig, _ = _remove_vig_shin(opening_odds, other_open)
    closing_novig, _ = _remove_vig_shin(closing_odds, other_close)

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
) -> CLVResult:
    """
    Unified CLV calculation.  Dispatches to the appropriate method.

    If both opening_spread and closing_spread are provided, uses
    calculate_clv_spread (preferred for spread bets).  Otherwise falls
    back to juice-only.

    Args:
        total:  Game total (over/under) forwarded to calculate_clv_spread
                for Skellam-based cover probability.  Pass the sharp
                consensus total when available.

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
        )
    return calculate_clv_juice_only(
        opening_odds=opening_odds,
        closing_odds=closing_odds,
        other_side_opening_odds=other_side_opening_odds,
        other_side_closing_odds=other_side_closing_odds,
    )
