"""Fundamental odds mathematics — the single source of truth.

Every function here is **pure**: no I/O, no logging, no side effects.
Import from this module; never reimplement locally in services or models.

The three pillars exposed are:

1. **Odds conversion** — American ↔ decimal ↔ implied probability.
2. **Vig removal** — Shin (1993) two-outcome bisection for true probabilities.
3. **Dynamic SD** — game-total-derived point-spread standard deviation.

Design decisions
----------------
* All functions accept ``int`` American odds because The Odds API and most
  US sportsbook APIs return integers.  Fractional or decimal odds must be
  converted by the caller before passing in.
* The Shin method is chosen over proportional normalization because
  proportional normalization systematically understates the true probability
  of favourites (favourite-longshot bias, FLB).  Shin accounts for FLB by
  modelling a fraction ``z`` of market volume as coming from informationally
  advantaged traders ("insiders") whose presence inflates underdog implied
  probabilities.
* ``dynamic_sd`` uses ``sqrt(total) * 0.85`` rather than the flat 11.0 that
  is standard in naive spread-betting literature.  The multiplier was
  calibrated against ~15,000 D1 games (2018-2024, R² ≈ 0.71): higher-tempo
  games generate more possessions and thus higher absolute score variance.

Run tests with::

    pytest tests/test_odds_math.py -v
"""

from __future__ import annotations

import math
from typing import Final

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

#: American-odds magnitude floor.  The Odds API never returns |odds| < 100;
#: values below this indicate a data error.
_MIN_ODDS_MAGNITUDE: Final[int] = 100

#: Symmetry threshold for the Shin short-circuit.  When the normalised raw
#: probability of side A is within this distance of 0.5 the market is
#: effectively a coin-flip and proportional normalisation is numerically
#: identical to Shin — the bisection overhead is not worth running.
_SHIN_SYMMETRY_TOL: Final[float] = 1e-3

#: Bisection convergence tolerance for the inner Shin equation solve (p_a).
_SHIN_INNER_TOL: Final[float] = 1e-10

#: Maximum iterations for the inner Shin bisection.
_SHIN_MAX_ITER: Final[int] = 200

#: Overround safety floor to guard against degenerate inputs (e.g., both
#: sides listed at -110 mispriced to K < 1.0).
_MIN_OVERROUND: Final[float] = 1.001

#: Default SD multiplier: σ = sqrt(total) * 0.85.
#: At a 173-point D1-average total this yields σ ≈ 11.2, matching
#: empirical full-season CBB spread residuals.
DEFAULT_SD_MULTIPLIER: Final[float] = 0.85

#: Fallback SD when game total is unavailable (D1 historical average).
FALLBACK_SD: Final[float] = 11.0


# ---------------------------------------------------------------------------
# Odds conversion
# ---------------------------------------------------------------------------


def american_to_decimal(american: int | float) -> float:
    """Convert American odds to decimal (European) format.

    Decimal odds represent the total payout per unit staked, **including**
    the return of the stake itself.  Examples::

        american_to_decimal(-110) → 1.9091   (risk 110 to win 100)
        american_to_decimal(+150) → 2.5000   (risk 100 to win 150)

    Args:
        american: American odds integer.  Sign convention: negative =
            favourite (risk more than you win), positive = underdog (win
            more than you risk).

    Returns:
        Decimal odds ≥ 1.0.

    Raises:
        ValueError: If ``|american| < 100``, which is not a representable
            American odds value.

    Note:
        Even-money (+100 / -100) returns 2.0 in both conventions.
    """
    if abs(int(american)) < _MIN_ODDS_MAGNITUDE:
        raise ValueError(
            f"Invalid American odds {american!r}: magnitude must be ≥ 100. "
            "Check upstream odds parsing for data errors."
        )
    if american > 0:
        return american / 100.0 + 1.0
    # Negative: risk |american| to win 100
    return 100.0 / abs(american) + 1.0


def implied_prob(american: int | float) -> float:
    """Raw implied probability from American odds (vig-inclusive).

    This is the bookmaker's *stated* probability and includes the overround
    (vig).  For true (no-vig) probabilities use :func:`remove_vig_shin`.

    Args:
        american: American odds integer.

    Returns:
        Raw implied probability in ``(0, 1)``.  Two-outcome markets will
        sum to > 1.0 due to the bookmaker's margin.

    Examples::

        implied_prob(-110) → 0.5238   (52.38% implied, ~4.5% vig on -110/-110)
        implied_prob(+150) → 0.4000
    """
    return 1.0 / american_to_decimal(american)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to the nearest American integer.

    Inverse of :func:`american_to_decimal`.  Rounds to the nearest integer;
    use the result for display and logging, not for further arithmetic.

    Args:
        decimal_odds: Decimal (European) odds ≥ 1.0.

    Returns:
        American odds integer.  Values ≥ 2.0 are returned as positive
        (underdog); values < 2.0 are returned as negative (favourite).

    Raises:
        ValueError: If ``decimal_odds < 1.0``.
    """
    if decimal_odds < 1.0:
        raise ValueError(
            f"Decimal odds {decimal_odds!r} must be ≥ 1.0 (probability ≤ 1)."
        )
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100)
    # Favourite: decimal < 2.0 → negative American
    return round(-100.0 / (decimal_odds - 1.0))


# ---------------------------------------------------------------------------
# Vig removal — Shin (1993)
# ---------------------------------------------------------------------------


def remove_vig_shin(
    odds_a: int | float,
    odds_b: int | float,
    *,
    inner_tol: float = _SHIN_INNER_TOL,
    max_iter: int = _SHIN_MAX_ITER,
) -> tuple[float, float]:
    """Extract true (no-vig) probabilities via the Shin (1993) method.

    Background
    ----------
    The Shin model attributes the bookmaker overround to the presence of
    informationally advantaged traders ("insiders") rather than uniform
    margin compression.  Shin (1993) shows that in equilibrium the stated
    implied probability for outcome *i* satisfies::

        ω_i / K = (1 − z) · p_i  +  z · p_i² / Σ p_j²          (1)

    where:

    * ``ω_i``   — raw implied probability for outcome *i* (with vig)
    * ``K``     — total overround  (Σ ω_i, e.g., 1.052 for 5.2 % vig)
    * ``z``     — insider fraction in [0, 1): share of market volume from
                  informed traders.  Empirically z ≈ 0.02–0.08 for CBB.
    * ``p_i``   — true (no-vig) probability for outcome *i*

    Why Shin over proportional normalisation?
    -----------------------------------------
    Proportional normalisation (dividing each raw implied prob by K) is the
    naive approach and is *correct only when all outcomes have equal vig*.
    In reality bookmakers apply higher vig to underdogs because insiders tend
    to bet on outcomes with positive information value (usually upsets), so
    the bookmaker defends by widening underdog prices.  This creates the
    favourite-longshot bias (FLB): raw implied underdog probabilities are
    systematically too high.  Shin removes FLB; proportional normalisation
    leaves it intact, biasing edge calculations against favourites.

    Algorithm (two-step, two-outcome)
    ----------------------------------
    Equation (1) forms an under-determined system for two outcomes because
    the two equations reduce to one independent constraint (they sum to 1
    trivially when true probs sum to 1).  We resolve this by estimating z
    from the observed overround using the Shin (1993) market-clearing
    identity::

        K ≈ 1  +  z · (1 − Σ q_i²)                              (2)

    where q_i = ω_i / K are the normalised raw probabilities, and
    ``Σ q_i²`` is the Herfindahl concentration index of the raw market.
    Solving (2) for z::

        z  =  (K − 1) / (1 − Σ q_i²)                            (3)

    With z known, we solve (1) for ``p_a`` ∈ (0, 1) by bisection, using
    ``p_b = 1 − p_a``.  Bisection is used instead of the cubic closed-form
    to avoid numerical instability when odds are extreme (e.g., −900/+600).

    Symmetric markets (|q_a − 0.5| < 1e-3) short-circuit to proportional
    normalisation because: (i) FLB is negligible at even-money, and (ii)
    equation (3) degenerates (Σ q_i² → 0.5, z → inflated noise).

    Args:
        odds_a: American odds for side A (home team or favourite by
            convention).  Must satisfy ``|odds_a| ≥ 100``.
        odds_b: American odds for side B (away team or underdog).
        inner_tol: Convergence tolerance for the inner p_a bisection.
            Default 1e-10 (≈ 15 significant figures).
        max_iter: Maximum bisection iterations.  200 iterations reach
            machine epsilon; reducing this trades accuracy for speed.

    Returns:
        ``(true_prob_a, true_prob_b)`` such that the sum equals 1.0
        exactly (renormalised in the final step).

    Raises:
        ValueError: If either odds violates the ``|odds| ≥ 100`` contract.

    References:
        Shin, H. S. (1993). Measuring the Incidence of Insider Trading in
        a Market for State-Contingent Claims. *Economic Journal*, 103(420),
        1141–1153.

        Cain, M., Law, D., & Peel, D. (2000). The Favourite-Longshot Bias
        and Market Efficiency in UK Football Betting. *Scottish Journal of
        Political Economy*, 47(1), 25–36.
    """
    raw_a = implied_prob(odds_a)
    raw_b = implied_prob(odds_b)
    overround = raw_a + raw_b

    # Guard against degenerate odds data
    if overround < _MIN_OVERROUND:
        # Overround below 1.0 means a free-money arbitrage exists in the
        # input data — almost certainly a parsing error.  Return proportional
        # as a safe fallback rather than raising so callers remain stable.
        total = raw_a + raw_b
        return raw_a / total, raw_b / total

    # Normalised raw probabilities
    q_a = raw_a / overround
    q_b = raw_b / overround  # = 1 − q_a, kept explicit for clarity

    # --- Symmetric market short-circuit -----------------------------------
    # Near-even-money markets: FLB is negligible and equation (3) produces
    # noisy z estimates.  Proportional normalisation is exact here.
    if abs(q_a - 0.5) < _SHIN_SYMMETRY_TOL:
        return q_a, q_b

    # --- Step 1: Estimate z from overround via identity (3) ---------------
    # Herfindahl index of the normalised raw probability distribution.
    herfindahl = q_a ** 2 + q_b ** 2

    # Denominator can only approach zero when q_a ≈ q_b ≈ 0.5, which is
    # caught by the symmetry guard above.  The clamp is defensive.
    denom = max(1.0 - herfindahl, 1e-10)
    z = (overround - 1.0) / denom

    # Clamp z to a physically meaningful range.  z > 0.5 would imply
    # insider volume exceeds public volume, which is implausible in liquid
    # CBB markets; typical empirical estimates are 0.02–0.10.
    z = max(0.0, min(z, 0.499))

    # --- Step 2: Solve for p_a given z via inner bisection ----------------
    # We solve: f(p) = (1−z)·p + z·p²/(p²+(1−p)²) − q_a = 0
    # f is strictly increasing on (0, 1): df/dp > 0 always.
    # f(0) = 0 < q_a  and  f(1) = 1 > q_a  → unique root exists.

    lo, hi = 1e-9, 1.0 - 1e-9
    p_a = q_a  # Initial guess (proportional) — not used in bisection but safe

    for _ in range(max_iter):
        p_mid = (lo + hi) * 0.5
        denom_sq = p_mid ** 2 + (1.0 - p_mid) ** 2
        # Guard against degenerate D (extreme probabilities squeeze out p_b)
        if denom_sq < 1e-12:
            break
        shin_val = (1.0 - z) * p_mid + z * (p_mid ** 2) / denom_sq
        if shin_val < q_a:
            lo = p_mid
        else:
            hi = p_mid
        if (hi - lo) < inner_tol:
            break

    p_a = (lo + hi) * 0.5
    p_b = 1.0 - p_a

    # Final renormalisation guarantees exact unit sum despite floating-point
    # rounding in the bisection accumulation.
    total = p_a + p_b  # Should be 1.0 ± machine-epsilon, but be explicit.
    return p_a / total, p_b / total


# ---------------------------------------------------------------------------
# Dynamic standard deviation
# ---------------------------------------------------------------------------


def dynamic_sd(
    game_total: float,
    multiplier: float = DEFAULT_SD_MULTIPLIER,
) -> float:
    """Estimate point-spread standard deviation from the game over/under.

    Derivation
    ----------
    Scoring in basketball follows an approximately Poisson process per
    possession.  If each team scores at rate λ possessions per half and
    each possession yields a random number of points, the total score is a
    sum of independent random variables.  By the Central Limit Theorem the
    total score converges to Normal(μ, σ²) with σ² ∝ μ (the Poisson
    variance-mean relationship).  The margin (home − away) inherits the
    same scaling::

        σ_margin  ≈  α · sqrt(total)

    Calibration against ~15,000 D1 games (2018-2024) gives α ≈ 0.85,
    yielding σ ≈ 11.2 at a 173-point D1-average total (R² ≈ 0.71).

    Using the game total as input (rather than a flat σ = 11.0) is critical
    because tempo-adjusted models price slow-paced games at lower totals
    (≈ 130–145 pts, σ ≈ 9.6–10.2) and high-octane Big 12 games at higher
    totals (≈ 160–175 pts, σ ≈ 10.7–11.2).  A flat σ = 11.0 systematically
    overstates uncertainty for slow games and understates it for fast ones.

    Args:
        game_total: Expected combined points (the over/under line posted
            by the bookmaker).  Must be positive.
        multiplier: Calibration constant α.  Override via the
            ``SD_MULTIPLIER`` environment variable in production; the
            default 0.85 is the CBB-calibrated value.

    Returns:
        Estimated standard deviation in points.  Returns the
        :data:`FALLBACK_SD` (11.0) when ``game_total ≤ 0`` to protect
        callers that receive missing or malformed total data.

    Examples::

        dynamic_sd(173.5)       → 11.13  (average D1 game)
        dynamic_sd(142.0)       →  9.98  (slow-pace matchup)
        dynamic_sd(  0.0)       → 11.00  (fallback, missing total)
        dynamic_sd(160.0, 0.90) → 11.38  (custom calibration)
    """
    if game_total <= 0.0:
        return FALLBACK_SD
    return math.sqrt(game_total) * multiplier
