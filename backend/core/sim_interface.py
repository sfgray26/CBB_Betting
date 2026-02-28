"""Dependency-injection interfaces for swappable pricing engines.

This module defines the contracts that **every** pricing engine must satisfy.
``CBBEdgeModel`` (and any future sport models) accept a ``BasePricingEngine``
at construction time rather than importing ``PossessionSimulator`` directly.
This enables:

* **Unit testing** — inject a ``MockPricingEngine`` that returns fixed
  ``PricingResult`` values without running thousands of Monte Carlo draws.
* **Sport extension** — swap in an ``NBASimulator`` or ``GaussianEngine``
  without modifying the model file.
* **A/B testing** — run two engines on the same game and compare outputs
  before committing to a new simulator.

Design choices
--------------
* :class:`BasePricingEngine` is an abstract base class (ABC) rather than a
  ``typing.Protocol`` because we want ``isinstance`` checks at runtime
  (e.g., in the model's constructor guard) and explicit inheritance to force
  engine authors to read the contract.
* :class:`TeamSimInputs` is the canonical data-transfer object that flows
  from ``analysis.py`` into the pricing engine.  It carries **all** fields
  any engine might need.  Engines ignore fields they do not use; they do not
  raise on unknown fields.
* :class:`PricingResult` is frozen and slotted so it can be cached safely
  and passed across thread boundaries.

Run tests with::

    pytest tests/test_sim_interface.py -v
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.core.sport_config import SportConfig


# ---------------------------------------------------------------------------
# Data transfer objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TeamSimInputs:
    """All per-team inputs that a pricing engine may consume.

    Fields are grouped by category.  Each field carries a D1-average default
    so callers can set only the fields they have data for, and engines receive
    a fully-populated object without ``None`` guards scattered throughout
    simulation logic.

    Attributes:
        team_name: Team identifier for logging.  Not used in simulation math.

        --- Offensive four-factor rates ---
        efg_pct: Effective field goal percentage (offensive).
            Formula: ``(FGM + 0.5 × 3PM) / FGA``.  D1 avg ≈ 0.505.
        to_pct: Turnover rate (turnovers per offensive possession).
            D1 avg ≈ 0.175.
        ft_rate: Free-throw attempt rate (FTA / FGA).  D1 avg ≈ 0.280.
        three_par: Three-point attempt rate (3PA / FGA).  D1 avg ≈ 0.360.
        orb_pct: Offensive rebound percentage.  D1 avg ≈ 0.280.

        --- Defensive four-factor rates ---
        def_efg_pct: Defensive eFG% allowed.  D1 avg ≈ 0.505.
        def_to_pct: Defensive turnover-forcing rate.  D1 avg ≈ 0.175.
        def_ft_rate: Defensive FTA/FGA allowed.  D1 avg ≈ 0.280.
        def_orb_pct: Defensive offensive-rebound rate allowed (= 1 − DRB%).
            D1 avg ≈ 0.280 (same as orb_pct in equilibrium).

        --- Shooting splits (offensive) ---
        rim_rate: Fraction of FGA taken at the rim.  D1 avg ≈ 0.35.
        mid_rate: Fraction of FGA taken from mid-range.  D1 avg ≈ 0.25.
            Note: ``rim_rate + mid_rate + three_par ≈ 1.0``.
        ft_pct: Free-throw make percentage.  D1 avg ≈ 0.720.

        --- Tempo ---
        pace: Adjusted possessions per game (both teams combined).
            D1 avg ≈ 68.0 (BartTorvik Adj.T.).

        --- Style flags (used by matchup engine) ---
        transition_rate: Fraction of possessions that are fast-break.
            0.0 = pure half-court team.  D1 avg ≈ 0.12.
        zone_pct: Fraction of possessions defended in a zone scheme.
            0.0 = man-to-man exclusively.  D1 avg ≈ 0.20.
        drop_pct: Fraction of pick-and-roll possessions defended in drop
            coverage (important for 3PT vs. drop matchup factor).

        --- Metadata ---
        is_heuristic: True when four-factor rates were *estimated* from AdjEM
            rather than taken from real BartTorvik / KenPom data.  Pricing
            engines may widen their uncertainty when this flag is set.
        data_staleness_days: Days since the underlying stats were last
            refreshed.  Used by the model to apply data-staleness penalties.
    """

    # Identity
    team_name: str = ""

    # Offensive four-factors
    efg_pct: float = 0.505
    to_pct: float = 0.175
    ft_rate: float = 0.280
    three_par: float = 0.360
    orb_pct: float = 0.280

    # Defensive four-factors
    def_efg_pct: float = 0.505
    def_to_pct: float = 0.175
    def_ft_rate: float = 0.280
    def_orb_pct: float = 0.280

    # Shooting splits
    rim_rate: float = 0.35
    mid_rate: float = 0.25
    ft_pct: float = 0.720

    # Tempo
    pace: float = 68.0

    # Style flags
    transition_rate: float = 0.12
    zone_pct: float = 0.20
    drop_pct: float = 0.30

    # Metadata
    is_heuristic: bool = False
    data_staleness_days: int = 0


@dataclass(slots=True, frozen=True)
class PricingResult:
    """Immutable output from a pricing engine for a single side of a bet.

    A ``PricingResult`` represents the engine's estimate of the probability
    distribution over outcomes for **one side** of a spread or total bet.
    The three probabilities must sum to 1.0 (enforced by the
    :meth:`validate` method).

    Attributes:
        win_prob: Estimated probability that the bet wins (covers the spread
            or clears the total).  In ``(0, 1)``.
        loss_prob: Estimated probability that the bet loses.  In ``(0, 1)``.
        push_prob: Estimated probability of a push (stake returned).  For
            non-integer spreads this is 0.0.  In ``[0, 1)``.
        lower_ci: Lower bound of the 95% confidence interval on
            ``win_prob``.  Derived from the Monte Carlo distribution over
            simulated outcomes, not from analytical CI formulas.
        upper_ci: Upper bound of the 95% confidence interval on
            ``win_prob``.
        n_simulations: Number of Monte Carlo draws (or equivalent) used
            to produce this estimate.  Reported in audit logs.
        engine_name: Identifier of the engine that produced this result
            (e.g., ``"PossessionSimulator"``, ``"GaussianEngine"``).
            Used in the ``full_analysis`` JSON for model versioning.

        --- Diagnostics (optional) ---
        mean_home_score: Simulated mean home team score.  Useful for
            sanity-checking that the engine is in the right scoring range.
        mean_away_score: Simulated mean away team score.
        margin_std: Standard deviation of the simulated score differential.
            Should be close to the ``adj_sd`` passed to the betting model.
    """

    # Core probabilities — MUST sum to 1.0
    win_prob: float
    loss_prob: float
    push_prob: float

    # Confidence interval on win_prob
    lower_ci: float
    upper_ci: float

    # Audit
    n_simulations: int
    engine_name: str

    # Optional diagnostics
    mean_home_score: float | None = None
    mean_away_score: float | None = None
    margin_std: float | None = None

    def validate(self, tol: float = 1e-4) -> None:
        """Assert that the probability triple sums to 1.0 within tolerance.

        Args:
            tol: Acceptable deviation from 1.0 before raising.

        Raises:
            ValueError: If ``|win + loss + push − 1| > tol``, or if any
                probability is outside ``[0, 1]``, or if ``lower_ci >
                win_prob`` or ``upper_ci < win_prob``.
        """
        total = self.win_prob + self.loss_prob + self.push_prob
        if abs(total - 1.0) > tol:
            raise ValueError(
                f"PricingResult probabilities must sum to 1.0 "
                f"(got {total:.6f}, engine={self.engine_name!r})."
            )
        for name, val in [
            ("win_prob", self.win_prob),
            ("loss_prob", self.loss_prob),
            ("push_prob", self.push_prob),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"PricingResult.{name} must be in [0, 1], got {val!r}."
                )
        if self.lower_ci > self.win_prob + tol:
            raise ValueError(
                f"lower_ci ({self.lower_ci!r}) > win_prob ({self.win_prob!r})."
            )
        if self.upper_ci < self.win_prob - tol:
            raise ValueError(
                f"upper_ci ({self.upper_ci!r}) < win_prob ({self.win_prob!r})."
            )

    @property
    def ci_half_width(self) -> float:
        """Symmetric 95% CI half-width around ``win_prob``."""
        return (self.upper_ci - self.lower_ci) / 2.0

    def __repr__(self) -> str:
        return (
            f"PricingResult(win={self.win_prob:.3f}, "
            f"push={self.push_prob:.3f}, "
            f"loss={self.loss_prob:.3f}, "
            f"CI=[{self.lower_ci:.3f}, {self.upper_ci:.3f}], "
            f"engine={self.engine_name!r})"
        )


# ---------------------------------------------------------------------------
# Abstract pricing engine
# ---------------------------------------------------------------------------


class BasePricingEngine(ABC):
    """Contract that every pricing engine must satisfy.

    Subclasses must implement :meth:`price_spread` and :meth:`price_total`.
    The model layer calls only these two methods; it does not import or
    instantiate any concrete engine directly.

    Threading
    ---------
    Engines are called from the nightly analysis loop which may run
    concurrent per-game tasks.  Each call to ``price_spread`` or
    ``price_total`` must be **stateless** with respect to previous calls —
    i.e., results must not depend on call order.  Shared ``numpy.random``
    state is not safe; use ``numpy.random.default_rng(seed)`` per call or
    per-engine instance.

    Example implementation::

        class GaussianEngine(BasePricingEngine):
            engine_name = "GaussianEngine"

            def price_spread(self, home, away, spread, *, config, ...):
                adj_sd = dynamic_sd(config.d1_avg_pace)
                win_prob = 1 - norm.cdf(spread / adj_sd)
                return PricingResult(
                    win_prob=win_prob,
                    loss_prob=1 - win_prob,
                    push_prob=0.0,
                    lower_ci=win_prob - 0.05,
                    upper_ci=win_prob + 0.05,
                    n_simulations=0,
                    engine_name=self.engine_name,
                )
    """

    #: Short identifier included in every ``PricingResult.engine_name``.
    #: Must be overridden by subclasses.
    engine_name: str = "BasePricingEngine"

    @abstractmethod
    def price_spread(
        self,
        home: TeamSimInputs,
        away: TeamSimInputs,
        spread: float,
        *,
        config: SportConfig,
        matchup_margin_adj: float = 0.0,
        n_simulations: int = 10_000,
        seed: int | None = None,
    ) -> PricingResult:
        """Price a spread bet: probability that the home team covers.

        The spread is defined from the **home team's perspective**.
        A spread of ``-4.5`` means the home team is favoured by 4.5 and
        must win by ≥ 5 for a cover.  A spread of ``+3.5`` means the home
        team is a 3.5-point underdog and must lose by ≤ 3 or win outright.

        Args:
            home: Sim inputs for the home team.
            away: Sim inputs for the away team.
            spread: The spread from the home team's perspective (negative =
                home favoured).  E.g., ``-4.5`` means home is -4.5.
            config: Sport-level constants injected by the model.
            matchup_margin_adj: Additional margin adjustment (in points)
                from the matchup engine.  Added to the home team's effective
                scoring expectation before pricing.
            n_simulations: Number of Monte Carlo draws.  Lower for speed,
                higher for tighter confidence intervals.
            seed: Optional RNG seed for reproducibility.  Pass ``None``
                for production runs (non-deterministic).

        Returns:
            :class:`PricingResult` with ``win_prob`` = P(home covers).

        Note:
            ``win_prob`` is always from the **home team's perspective**.
            The model layer is responsible for flipping to the away side
            when ``edge_away > edge_home``.
        """

    @abstractmethod
    def price_total(
        self,
        home: TeamSimInputs,
        away: TeamSimInputs,
        total: float,
        *,
        config: SportConfig,
        direction: str = "over",
        n_simulations: int = 10_000,
        seed: int | None = None,
    ) -> PricingResult:
        """Price an over/under total bet.

        Args:
            home: Sim inputs for the home team.
            away: Sim inputs for the away team.
            total: The bookmaker's posted over/under line.
            config: Sport-level constants.
            direction: ``"over"`` (default) or ``"under"``.  Returns
                ``win_prob`` = P(combined score > total) for ``"over"``,
                or P(combined score < total) for ``"under"``.
            n_simulations: Monte Carlo draws.
            seed: Optional RNG seed.

        Returns:
            :class:`PricingResult` with ``win_prob`` = P(direction covers).
        """

    def supports_total_pricing(self) -> bool:
        """Return True if this engine has a meaningful total model.

        Some engines (e.g., a simple Gaussian spread model) cannot
        accurately price totals because they do not model team scoring
        separately.  Override to return ``False`` to signal the model
        to fall back to a spread-inferred total probability.

        The default implementation returns ``True``; override in engines
        that cannot provide total estimates.
        """
        return True

    def warm_up(self, n_warmup: int = 100) -> None:
        """Optional pre-computation hook called once at model init.

        Use to pre-compile Numba JIT kernels, allocate CUDA memory, or
        run initial Monte Carlo draws to warm the RNG chain.  Engines that
        do not need warm-up should leave this as a no-op.

        Args:
            n_warmup: Hint for how many warm-up iterations to run.
        """


# ---------------------------------------------------------------------------
# Null engine — safe fallback / testing stub
# ---------------------------------------------------------------------------


class _NullPricingEngine(BasePricingEngine):
    """A pricing engine that always returns a 50/50 result with zero CI.

    Used as a sentinel when no real engine is provided, or in tests that
    need to isolate the Kelly / edge-calculation layer from simulation
    noise.  **Never use in production.**
    """

    engine_name = "NullEngine"

    def price_spread(
        self,
        home: TeamSimInputs,
        away: TeamSimInputs,
        spread: float,
        *,
        config: SportConfig,
        matchup_margin_adj: float = 0.0,
        n_simulations: int = 10_000,
        seed: int | None = None,
    ) -> PricingResult:
        return PricingResult(
            win_prob=0.50,
            loss_prob=0.50,
            push_prob=0.00,
            lower_ci=0.50,
            upper_ci=0.50,
            n_simulations=0,
            engine_name=self.engine_name,
        )

    def price_total(
        self,
        home: TeamSimInputs,
        away: TeamSimInputs,
        total: float,
        *,
        config: SportConfig,
        direction: str = "over",
        n_simulations: int = 10_000,
        seed: int | None = None,
    ) -> PricingResult:
        return PricingResult(
            win_prob=0.50,
            loss_prob=0.50,
            push_prob=0.00,
            lower_ci=0.50,
            upper_ci=0.50,
            n_simulations=0,
            engine_name=self.engine_name,
        )

    def supports_total_pricing(self) -> bool:
        return False


#: Singleton null engine for use in tests and safe-fallback scenarios.
NULL_ENGINE: BasePricingEngine = _NullPricingEngine()
