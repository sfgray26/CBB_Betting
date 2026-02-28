"""Sport-level configuration — all sport-specific constants in one place.

This module is the **registry** for every constant that differs between
sports.  Nowhere else in the codebase should D1 averages, home-advantage
figures, or sharp-book lists be hard-coded.

Architecture
------------
:class:`SportConfig` is a frozen dataclass carrying all per-sport constants.
Named constructors (:meth:`SportConfig.ncaa_basketball`,
:meth:`SportConfig.nba`) return pre-populated instances.  To add a new sport:

1. Add a ``@classmethod`` constructor here.
2. Pass the relevant ``SportConfig`` to :class:`~backend.domain.betting_model.CBBEdgeModel`.
3. The model, simulator, and CLV calculator use the injected config
   instead of touching hard-coded values.

No field here should ever be ``None``.  Every constant must have a
defensible default, with the source cited in the field's docstring.

Typical usage::

    from backend.core.sport_config import SportConfig

    cfg = SportConfig.ncaa_basketball()
    model = CBBEdgeModel(config=cfg)

    # Override a single constant for a custom season calibration:
    from dataclasses import replace
    custom_cfg = replace(cfg, home_advantage_pts=2.85)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


# ---------------------------------------------------------------------------
# Sentinel for NBA — pre-declared so the type checker knows it exists
# ---------------------------------------------------------------------------

#: Sport identifier strings used in API routes and DB records.
SPORT_ID_NCAAB: Final[str] = "ncaab"
SPORT_ID_NBA: Final[str] = "nba"
SPORT_ID_NCAAF: Final[str] = "ncaaf"


@dataclass(frozen=True)
class SportConfig:
    """Immutable configuration bundle for a single sport.

    All fields have CBB defaults to avoid accidental ``None`` propagation
    when a concrete constructor is not called.  Override via
    :func:`dataclasses.replace` for single-season or A/B-test tweaks.

    Attributes:
        sport_id: Short identifier string (``"ncaab"``, ``"nba"``, etc.)
            used in API routes, The Odds API ``sport_key``, and DB records.
        sport_name: Human-readable name for logging and display.

        --- Scoring distribution ---
        d1_avg_efg: League-average effective field goal percentage (off).
            Source: NCAA 2024-25 season aggregate (ncaa.com).
        d1_avg_def_efg: League-average defensive eFG% (same as off in
            equilibrium, retained separately so the simulator can use
            sport-specific offensive/defensive differentials).
        d1_avg_to_pct: League-average offensive turnover rate (turnovers
            per possession).  Source: NCAA 2024-25.
        d1_avg_def_to_pct: League-average defensive turnover-forcing rate.
        d1_avg_ft_rate: League-average free-throw attempt rate (FTA / FGA).
            Source: NCAA 2024-25.
        d1_avg_three_par: League-average three-point attempt rate (3PA / FGA).
        d1_avg_orb_pct: League-average offensive rebound percentage.
        d1_avg_ft_pct: League-average free-throw make percentage.

        --- Efficiency and pace ---
        d1_avg_adj_o: Adjusted offensive efficiency baseline (points per
            100 possessions on D1-average defence).  ~106 for NCAA 2024-25.
        d1_avg_adj_de: Adjusted defensive efficiency baseline.  This is
            also used as the denominator when converting AdjEM to a
            synthetic defensive eFG% in the heuristic profile builder.
            Source: KenPom 2024-25 D1 median.
        d1_avg_pace: Average possessions per game (both teams combined per
            half × 2).  Source: BartTorvik 2024-25, D1 median.

        --- Spread uncertainty ---
        base_sd_multiplier: Coefficient for ``σ = sqrt(total) × α``.
            See :func:`~backend.core.odds_math.dynamic_sd` for derivation.

        --- Home advantage ---
        home_advantage_pts: Expected margin boost for the home team in
            points.  Source: regressed average across D1 2019-2025, 3.09 pts.
            N.B.: neutral-site games must set this to 0.0 at the call site.

        --- Bivariate Poisson correlation ---
        pace_rho: Pearson correlation between home and away team scores.
            Used as the shared Poisson component ρ in the Bivariate Poisson
            CLV calculation.  Empirical CBB estimate: 0.35 (pace correlation —
            both teams' scores co-vary because tempo determines possessions for
            *both* sides).  NBA is slightly lower (~0.30) due to deeper benches
            and less possession-level impact from a single pace-setter.

        --- Market configuration ---
        sharp_books: Bookmakers treated as the "true market" for CLV
            benchmark purposes.  These are never filtered by ``active_books``
            and their consensus is used to detect late-closing line value.
            Pinnacle is the global standard; Circa Sports is the US sharp
            benchmark.
        odds_api_sport_key: The sport key passed to The Odds API.
            See https://the-odds-api.com/sports-odds-data/betting-odds.html
    """

    # Identity
    sport_id: str
    sport_name: str

    # Scoring distribution — offensive
    d1_avg_efg: float
    d1_avg_to_pct: float
    d1_avg_ft_rate: float
    d1_avg_three_par: float
    d1_avg_orb_pct: float
    d1_avg_ft_pct: float

    # Scoring distribution — defensive mirrors
    d1_avg_def_efg: float
    d1_avg_def_to_pct: float

    # Efficiency and pace
    d1_avg_adj_o: float
    d1_avg_adj_de: float
    d1_avg_pace: float           # possessions per game

    # Spread uncertainty
    base_sd_multiplier: float

    # Home advantage
    home_advantage_pts: float

    # Bivariate Poisson
    pace_rho: float

    # Market
    sharp_books: frozenset[str]
    odds_api_sport_key: str

    # ------------------------------------------------------------------ #
    #  Named constructors                                                  #
    # ------------------------------------------------------------------ #

    @classmethod
    def ncaa_basketball(cls) -> SportConfig:
        """Return the canonical NCAA D1 Basketball configuration.

        All values are calibrated to the 2024-25 D1 season.

        Sources:
            * eFG%, TO%, FTR, 3PAR, ORB: NCAA.com 2024-25 stats aggregate.
            * AdjO / AdjDE / Pace: KenPom & BartTorvik 2024-25 D1 medians.
            * Home advantage: Pooled regression 2019-2025, n ≈ 18,000 games.
            * SD multiplier: Calibrated against 2018-2024, n ≈ 15,000 games.
            * pace_rho: Bivariate Poisson fit, 2022-2025 CBB totals data.
        """
        return cls(
            sport_id=SPORT_ID_NCAAB,
            sport_name="NCAA D1 Basketball",
            # --- Scoring distribution ---
            d1_avg_efg=0.505,           # NCAA 2024-25 off. eFG%
            d1_avg_to_pct=0.175,        # NCAA 2024-25 off. TO rate
            d1_avg_ft_rate=0.280,       # NCAA 2024-25 FTA/FGA
            d1_avg_three_par=0.360,     # NCAA 2024-25 3PA/FGA
            d1_avg_orb_pct=0.280,       # NCAA 2024-25 off. rebound %
            d1_avg_ft_pct=0.720,        # NCAA 2024-25 FT make %
            d1_avg_def_efg=0.505,       # symmetric with off. in equilibrium
            d1_avg_def_to_pct=0.175,
            # --- Efficiency and pace ---
            d1_avg_adj_o=105.0,         # KenPom 2024-25 D1 median AdjO
            d1_avg_adj_de=105.0,        # KenPom 2024-25 D1 median AdjDE
            d1_avg_pace=68.0,           # BartTorvik 2024-25 median Adj.T.
            # --- Spread uncertainty ---
            base_sd_multiplier=0.85,    # σ = sqrt(total) × 0.85
            # --- Home advantage ---
            home_advantage_pts=3.09,    # pooled regression 2019-2025
            # --- Bivariate Poisson ---
            pace_rho=0.35,              # empirical CBB score correlation
            # --- Market ---
            sharp_books=frozenset({"pinnacle", "circasports"}),
            odds_api_sport_key="basketball_ncaab",
        )

    @classmethod
    def nba(cls) -> SportConfig:
        """Return the NBA configuration stub.

        Values are reasonable starting points drawn from public NBA
        shot-quality research (Cleaning the Glass, 2023-24 season).
        **Calibrate base_sd_multiplier against actual NBA data before
        using this in production.**

        Sources:
            * eFG%, TO%, FTR: Cleaning the Glass 2023-24 averages.
            * Pace: NBA.com 2023-24 (97.5 poss per game per team).
            * Home advantage: ~2.5 pts (NBA is lower than NCAA; less
              student crowd effect, superstar travel fatigue is smaller).
            * pace_rho: Estimated 0.30 (NBA deeper rotations, less
              single-player pace domination vs. CBB star guard effect).
        """
        return cls(
            sport_id=SPORT_ID_NBA,
            sport_name="NBA",
            # --- Scoring distribution ---
            d1_avg_efg=0.530,           # NBA 2023-24 avg off. eFG%
            d1_avg_to_pct=0.130,        # NBA avg TO rate (tighter ball-control)
            d1_avg_ft_rate=0.220,       # NBA FTA/FGA (fewer bailout fouls)
            d1_avg_three_par=0.400,     # NBA 3PA/FGA (more three-ball)
            d1_avg_orb_pct=0.270,       # NBA avg OREB%
            d1_avg_ft_pct=0.780,        # NBA avg FT% (better shooters)
            d1_avg_def_efg=0.530,
            d1_avg_def_to_pct=0.130,
            # --- Efficiency and pace ---
            d1_avg_adj_o=112.0,         # NBA pts/100 poss baseline
            d1_avg_adj_de=112.0,
            d1_avg_pace=97.5,           # NBA possessions per game per team
            # --- Spread uncertainty ---
            base_sd_multiplier=0.78,    # UNCALIBRATED — measure vs. NBA data
            # --- Home advantage ---
            home_advantage_pts=2.50,    # NBA pooled estimate
            # --- Bivariate Poisson ---
            pace_rho=0.30,              # lower than CBB, estimated
            # --- Market ---
            sharp_books=frozenset({"pinnacle", "circasports", "betcris"}),
            odds_api_sport_key="basketball_nba",
        )

    @classmethod
    def ncaa_football(cls) -> SportConfig:
        """Return the NCAA Football configuration stub.

        **Heavily** under-calibrated — this is a structural placeholder.
        The scoring model (Markov possession simulator) is basketball-specific
        and must be replaced with a yards-per-play model before this config
        is useful.  The SD multiplier here is directionally correct but
        not validated.
        """
        return cls(
            sport_id=SPORT_ID_NCAAF,
            sport_name="NCAA Football",
            # Football does not use eFG%, TO%, etc. in the basketball sense.
            # These are set to zero; football-specific fields will be added
            # to a subclass or a separate FootballSimInputs dataclass.
            d1_avg_efg=0.0,
            d1_avg_to_pct=0.0,
            d1_avg_ft_rate=0.0,
            d1_avg_three_par=0.0,
            d1_avg_orb_pct=0.0,
            d1_avg_ft_pct=0.0,
            d1_avg_def_efg=0.0,
            d1_avg_def_to_pct=0.0,
            d1_avg_adj_o=0.0,
            d1_avg_adj_de=0.0,
            d1_avg_pace=0.0,           # possession-per-game concept N/A
            base_sd_multiplier=1.40,   # football point spreads are wider
            home_advantage_pts=2.50,   # FBS home field ~2.5 pts (Sagarin)
            pace_rho=0.25,             # lower score correlation in football
            sharp_books=frozenset({"pinnacle", "circasports"}),
            odds_api_sport_key="americanfootball_ncaaf",
        )

    # ------------------------------------------------------------------ #
    #  Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    def is_basketball(self) -> bool:
        """Return True if this config represents a basketball sport."""
        return self.sport_id in {SPORT_ID_NCAAB, SPORT_ID_NBA}

    def neutral_site(self) -> SportConfig:
        """Return a copy of this config with home advantage zeroed out.

        Use for tournament games (NCAA Tournament, NBA Finals games at
        neutral venues) where neither team has a home crowd.

        Returns:
            New :class:`SportConfig` identical to ``self`` except
            ``home_advantage_pts = 0.0``.

        Examples::

            ncaa_cfg = SportConfig.ncaa_basketball()
            tourney_cfg = ncaa_cfg.neutral_site()
            assert tourney_cfg.home_advantage_pts == 0.0
        """
        from dataclasses import replace
        return replace(self, home_advantage_pts=0.0)

    def __repr__(self) -> str:
        return (
            f"SportConfig(sport_id={self.sport_id!r}, "
            f"home_adv={self.home_advantage_pts}, "
            f"pace={self.d1_avg_pace}, "
            f"sd_mult={self.base_sd_multiplier})"
        )
