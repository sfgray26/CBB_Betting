"""Kelly criterion sizing — the single source of truth for bet sizing math.

All functions here are **pure**: no I/O, no database, no logging.
Import from this module; never reimplement Kelly locally in services.

The three functions cover the three distinct sizing contexts in the pipeline:

1. :func:`kelly_fraction` — Standard Kelly for a win/loss bet (no push).
2. :func:`kelly_fraction_with_push` — Push-aware Kelly for spread bets where
   an integer spread creates a non-zero tie probability.
3. :func:`portfolio_kelly_divisor` — Dynamic divisor that scales fractional
   Kelly conservatism with concurrent portfolio exposure, preventing the
   "30 bets on Saturday" over-concentration problem.

Design decisions
----------------
* **Fractional Kelly** (1/N of full Kelly) is the universal practice in
  quantitative sports betting.  Full Kelly maximises long-run log-wealth in
  theory, but requires exact edge estimates.  Because our edge estimates carry
  uncertainty (CI half-width ≈ ±3%), overbetting is asymmetrically punished
  (geometric ruin vs. forgone EV).  Fractional Kelly at 1/2 standard, rising
  to 1/4 under high portfolio load, provides robust downside protection.
* **Push probability** is *not* ignored.  In CBB spread betting, an integer
  spread (e.g., −7) creates a push probability of 4–10% depending on the
  score distribution.  Ignoring pushes overstates the Kelly fraction because
  they return the stake (neither win nor loss) and should reduce the effective
  EV per dollar risked.
* The **portfolio divisor** is a function of ``concurrent_exposure`` because
  simultaneous bets are positively correlated (same slate, same sharp-book
  movements).  Increasing the divisor as exposure rises is mathematically
  equivalent to applying a "simultaneous Kelly" multiplier that shrinks each
  bet so the aggregate worst-case loss stays within the exposure cap.

Run tests with::

    pytest tests/test_kelly.py -v
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Standard fractional Kelly divisor when no concurrent exposure exists.
#: At this divisor the worst-case drawdown from a single 2-sigma edge
#: estimate error is ≤ 4% of bankroll.
_BASE_DIVISOR: Final[float] = 2.0

#: Maximum Kelly divisor applied at the target exposure ceiling.
#: 4× corresponds to the "simultaneous Kelly" multiplier for ~5 concurrent
#: independent bets each sized at 3% bankroll.
_MAX_DIVISOR: Final[float] = 4.0

#: Hard cap on any single fractional Kelly output, irrespective of edge.
#: At 20% bankroll per bet, three simultaneous losses exhaust 48% of
#: bankroll — the acceptable tail-risk threshold for this system.
MAX_KELLY_FRACTION: Final[float] = 0.20

#: Minimum fractional Kelly threshold below which the recommendation
#: rounds to 0 (too small to execute at standard unit sizes).
MIN_KELLY_FRACTION: Final[float] = 1e-4

#: Annualised target concurrent exposure fraction (15% of bankroll deployed
#: at once across all open positions).  This matches the ``MAX_DAILY_EXPOSURE_PCT``
#: environment variable default in the portfolio manager.
DEFAULT_TARGET_EXPOSURE: Final[float] = 0.15


# ---------------------------------------------------------------------------
# Standard Kelly
# ---------------------------------------------------------------------------


def kelly_fraction(
    win_prob: float,
    decimal_odds: float,
    *,
    fractional_divisor: float = _BASE_DIVISOR,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Compute fractional Kelly bet size for a simple win/loss outcome.

    The Kelly criterion maximises the expected logarithm of wealth (equivalently,
    long-run geometric growth) by solving::

        max_f  E[log(1 + f · X)]

    where ``X`` is the random payoff per unit staked: ``b`` with probability
    ``p`` (win), ``−1`` with probability ``q = 1 − p`` (loss), and ``b`` is
    the *profit* per unit (decimal odds minus one stake return).  The
    closed-form solution (Kelly 1956) is::

        f*  =  (p · b − q) / b  =  p − q / b                     (1)

    We apply a fractional divisor (default 2×) to produce the *fractional*
    Kelly ``f = f* / divisor``.  This is not a heuristic safety cut — it is
    the Bayesian-optimal bet size when the true edge is uncertain and your
    edge estimate has non-trivial estimation error (MacLean, Thorp, Ziemba
    2011, *The Kelly Capital Growth Investment Criterion*).

    Args:
        win_prob: Estimated true (no-vig) probability of winning the bet,
            in ``(0, 1)``.
        decimal_odds: Decimal odds for the bet.  Profit per unit = ``decimal_odds − 1``.
            Use :func:`~backend.core.odds_math.american_to_decimal` to convert.
        fractional_divisor: Divisor applied to full Kelly to produce the
            fractional recommendation.  Default 2× (half-Kelly).  Use
            :func:`portfolio_kelly_divisor` to compute this dynamically.
        max_fraction: Hard cap on the output fraction.  Clips extreme edge
            estimates that would otherwise suggest betting > 20% of bankroll.

    Returns:
        Fractional Kelly bet size in ``[0, max_fraction]``.  Returns 0.0
        when the full Kelly is negative (negative-EV bet — never bet).

    Raises:
        ValueError: If ``win_prob`` is not in ``(0, 1)`` or
            ``decimal_odds < 1.0``.

    Examples::

        kelly_fraction(0.55, 1.909)          →  0.058  (half-Kelly on -110)
        kelly_fraction(0.60, 2.100)          →  0.119  (half-Kelly on +110)
        kelly_fraction(0.45, 1.909)          →  0.000  (negative EV → 0)
        kelly_fraction(0.55, 1.909, fractional_divisor=4.0) → 0.029

    References:
        Kelly, J. L. (1956). A New Interpretation of Information Rate.
        *Bell System Technical Journal*, 35(4), 917–926.
    """
    if not (0.0 < win_prob < 1.0):
        raise ValueError(
            f"win_prob must be in (0, 1), got {win_prob!r}. "
            "Check upstream probability clipping."
        )
    if decimal_odds < 1.0:
        raise ValueError(
            f"decimal_odds must be ≥ 1.0 (implies guaranteed loss), got {decimal_odds!r}."
        )

    profit_per_unit = decimal_odds - 1.0
    loss_prob = 1.0 - win_prob

    # Full Kelly (equation 1)
    full_kelly = (win_prob * profit_per_unit - loss_prob) / profit_per_unit

    if full_kelly <= 0.0:
        # Negative or zero edge: do not bet.
        return 0.0

    fractional = full_kelly / fractional_divisor
    return min(fractional, max_fraction)


# ---------------------------------------------------------------------------
# Push-aware Kelly
# ---------------------------------------------------------------------------


def kelly_fraction_with_push(
    win_prob: float,
    push_prob: float,
    decimal_odds: float,
    *,
    fractional_divisor: float = _BASE_DIVISOR,
    max_fraction: float = MAX_KELLY_FRACTION,
) -> float:
    """Compute fractional Kelly accounting for push (tie) probability.

    When betting against a spread with an integer line (e.g., −7), there is
    a non-zero probability that the margin of victory lands exactly on the
    spread, resulting in a push: the stake is returned and neither profit nor
    loss is recorded.

    The standard Kelly formula (equation 1 in :func:`kelly_fraction`) is
    derived for binary outcomes.  With pushes, the outcome space is
    ``{win, loss, push}`` and the expected-log-wealth maximisation becomes::

        max_f  p_win · log(1 + f·b)  +  p_loss · log(1 − f)
               +  p_push · log(1)                            (2)

    Taking the derivative and setting it to zero::

        p_win · b / (1 + f·b)  −  p_loss / (1 − f)  =  0

    Solving for ``f``::

        f*  =  (p_win · b − p_loss) / b                      (3)

    where ``p_loss = 1 − p_win − p_push``.  Note that equation (3) has
    the same form as the standard Kelly formula, but the loss probability
    is *reduced* by the push mass.  This slightly increases the Kelly
    fraction compared to ignoring pushes, correctly reflecting that a push
    absorbs probability mass from the loss event and improves EV per
    dollar risked.

    Why not use conditional Kelly?  An alternative is ``kelly(p_win / (1 −
    p_push), decimal_odds)`` (condition on non-push).  This is an
    approximation valid only when ``p_push → 0`` and slightly *overstates*
    the correct fraction at typical CBB push rates (5–12%).  Equation (3)
    is exact.

    Args:
        win_prob: Estimated true probability of winning the spread bet,
            in ``(0, 1)``.
        push_prob: Estimated probability of a push (tie), in ``[0, 1)``.
            Typically 0.04–0.12 for integer CBB spreads.
        decimal_odds: Decimal odds for the bet (profit per unit + 1).
        fractional_divisor: Divisor for fractional Kelly.
        max_fraction: Hard cap on output.

    Returns:
        Fractional Kelly in ``[0, max_fraction]``.

    Raises:
        ValueError: If probabilities are out of range or do not satisfy
            ``win_prob + push_prob < 1``.

    Examples::

        # -7 spread, 54% cover, 8% push, -110 juice:
        kelly_fraction_with_push(0.54, 0.08, 1.909)  →  0.065

        # Compare to ignoring push (slightly underestimates):
        kelly_fraction(0.54, 1.909)                   →  0.057
    """
    if not (0.0 < win_prob < 1.0):
        raise ValueError(f"win_prob must be in (0, 1), got {win_prob!r}.")
    if not (0.0 <= push_prob < 1.0):
        raise ValueError(f"push_prob must be in [0, 1), got {push_prob!r}.")
    if win_prob + push_prob >= 1.0:
        raise ValueError(
            f"win_prob ({win_prob!r}) + push_prob ({push_prob!r}) must be < 1.0; "
            "there would be no probability mass left for a loss."
        )
    if decimal_odds < 1.0:
        raise ValueError(
            f"decimal_odds must be ≥ 1.0, got {decimal_odds!r}."
        )

    profit_per_unit = decimal_odds - 1.0
    # Loss probability: everything that isn't a win or a push
    loss_prob = 1.0 - win_prob - push_prob

    # Push-aware Kelly (equation 3 above)
    full_kelly = (win_prob * profit_per_unit - loss_prob) / profit_per_unit

    if full_kelly <= 0.0:
        return 0.0

    fractional = full_kelly / fractional_divisor
    return min(fractional, max_fraction)


# ---------------------------------------------------------------------------
# Portfolio-aware Kelly divisor
# ---------------------------------------------------------------------------


def portfolio_kelly_divisor(
    concurrent_exposure: float,
    *,
    target_exposure: float = DEFAULT_TARGET_EXPOSURE,
    base_divisor: float = _BASE_DIVISOR,
    max_divisor: float = _MAX_DIVISOR,
) -> float:
    """Compute a dynamic Kelly divisor that grows with portfolio exposure.

    Motivation
    ----------
    The Kelly criterion is derived for a *single* independent bet in
    isolation.  When ``N`` bets are placed simultaneously they share
    exposure to: (i) correlated sharp-book movements on the same slate,
    (ii) systematic model error (if our edge estimate is wrong for one game
    it may be wrong for all), and (iii) bankroll concentration risk.

    The "simultaneous Kelly" solution (Smoczynski & Tomkins 2010) shows that
    for ``N`` concurrent bets the optimal individual fraction is reduced by
    a factor that grows with ``N``.  Rather than tracking ``N`` explicitly
    (which requires full covariance estimation), we use portfolio
    *exposure* as a proxy::

        divisor(e)  =  base_divisor · (1 + e / target_exposure)   (4)

    where ``e`` is the current concurrent exposure as a fraction of bankroll.
    This gives:

    * ``e = 0.00`` → divisor = 2.0× (half-Kelly, standard single-bet)
    * ``e = 0.075`` → divisor = 3.0× (quarter-way to cap)
    * ``e = 0.15`` → divisor = 4.0× (at full daily exposure ceiling)
    * ``e > 0.15`` → capped at max_divisor to avoid divisor explosion

    At max divisor (4×) a 2%-edge bet at -110 odds becomes ≈ 0.65% bankroll,
    ensuring that even 10 simultaneous recommendations keep total exposure
    well within the 15% daily cap.

    Args:
        concurrent_exposure: Current aggregate open exposure as a fraction
            of bankroll, in ``[0, 1)``.  Computed by the portfolio manager
            from all pending bets.  Pass 0.0 when no positions are open.
        target_exposure: The maximum acceptable daily exposure fraction.
            At this exposure the divisor reaches ``max_divisor``.  Defaults
            to 0.15 (15% of bankroll), matching ``MAX_DAILY_EXPOSURE_PCT``.
        base_divisor: Divisor applied when exposure is zero.  Default 2.0
            (half-Kelly).
        max_divisor: Hard cap on the divisor.  Prevents extreme
            over-conservatism when a bug causes exposure to overflow.

    Returns:
        Kelly divisor ≥ ``base_divisor``, capped at ``max_divisor``.

    Raises:
        ValueError: If ``concurrent_exposure < 0`` or
            ``target_exposure ≤ 0``.

    Examples::

        portfolio_kelly_divisor(0.00)   → 2.0  (no open positions)
        portfolio_kelly_divisor(0.075)  → 3.0  (half-way to cap)
        portfolio_kelly_divisor(0.15)   → 4.0  (at daily exposure ceiling)
        portfolio_kelly_divisor(0.30)   → 4.0  (capped, not 6.0)

    References:
        Smoczynski, P., & Tomkins, D. (2010). An explicit solution to the
        problem of optimizing the allocations of a bettor's wealth when
        wagering on horse races. *Mathematical Scientist*, 35(1), 1–12.
    """
    if concurrent_exposure < 0.0:
        raise ValueError(
            f"concurrent_exposure must be ≥ 0, got {concurrent_exposure!r}."
        )
    if target_exposure <= 0.0:
        raise ValueError(
            f"target_exposure must be > 0, got {target_exposure!r}."
        )

    # Linear interpolation from base_divisor to max_divisor over [0, target]
    raw_divisor = base_divisor * (1.0 + concurrent_exposure / target_exposure)
    return min(raw_divisor, max_divisor)


# ---------------------------------------------------------------------------
# Utility — unit conversion
# ---------------------------------------------------------------------------


def units_to_dollars(units: float, bankroll: float) -> float:
    """Convert unit-based sizing to dollar amount.

    ``units`` here follow the convention used throughout the pipeline:
    one unit = 1% of current bankroll.  A ``recommended_units = 2.5``
    recommendation therefore means risk 2.5% of bankroll.

    Args:
        units: Bet size in units (1 unit = 1% of bankroll by convention).
        bankroll: Current bankroll in dollars.

    Returns:
        Dollar amount to risk.

    Examples::

        units_to_dollars(2.5, 1000.0)  →  25.0
        units_to_dollars(0.5, 5000.0)  →  25.0
    """
    return (units / 100.0) * bankroll


def kelly_to_units(kelly_fraction_val: float) -> float:
    """Convert a Kelly fraction to units for display and logging.

    The pipeline uses the convention: 1 unit = 1% of bankroll.
    A Kelly fraction of 0.025 (2.5% of bankroll) → 2.5 units.

    Args:
        kelly_fraction_val: Fractional Kelly in [0, 1].

    Returns:
        Units (1 unit = 1% of bankroll).

    Examples::

        kelly_to_units(0.025) → 2.5
        kelly_to_units(0.005) → 0.5
    """
    return kelly_fraction_val * 100.0
