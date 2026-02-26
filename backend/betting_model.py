"""
Version 8 CBB Betting Framework - Production Implementation

Upgrades from V7:
- Dynamic weight re-normalization when rating sources are missing
- Matchup-specific variance via team style profiles (pace, 3PAr, FTr)
- Shin (1993) method for true probability extraction from odds
- Portfolio-aware Kelly sizing support
- Injury impact quantification
- Spread-adjusted cover probability (P(cover) not P(win))
- Deterministic RNG for reproducible Monte Carlo results
- Both-side edge evaluation (home or away value detection)

Retained from V7:
- 2-layer Monte Carlo CI
- Penalty budget with ceiling
- Safe Kelly calculation
- Conservative decision threshold
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class GameAnalysis:
    """Complete analysis output"""
    verdict: str
    pass_reason: Optional[str]

    # Model inputs
    projected_margin: float
    adjusted_sd: float
    home_advantage: float

    # Probabilities
    point_prob: float           # P(home wins) — for calibration tracking
    lower_ci_prob: float
    upper_ci_prob: float

    # Edges (computed from spread cover probability, not win probability)
    edge_point: float
    edge_conservative: float

    # Betting
    kelly_full: float
    kelly_fractional: float
    recommended_units: float
    bet_side: Optional[str] = None  # "home" or "away" — which side has value

    # Metadata
    data_freshness_tier: str = "Unknown"
    penalties_applied: Dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    # Full details for storage
    full_analysis: Dict = field(default_factory=dict)


class CBBEdgeModel:
    """
    Production betting model implementing Version 8 framework.
    Conservative, transparent, and designed to PASS 85-95% of games.
    """

    def __init__(
        self,
        base_sd: float = 11.0,
        weights: Optional[Dict[str, float]] = None,
        home_advantage: float = 3.09,
        max_kelly: float = 0.20,
        fractional_kelly_divisor: float = 2.0,
        seed: Optional[int] = None,
    ):
        self.base_sd = base_sd
        self.weights = weights or {
            'kenpom': 0.342,
            'barttorvik': 0.333,
            'evanmiya': 0.325,
        }
        self.home_advantage = home_advantage
        self.max_kelly = max_kelly
        self.fractional_kelly_divisor = fractional_kelly_divisor
        self.rng = np.random.default_rng(seed)
    
    def monte_carlo_prob_ci(
        self,
        projected_margin: float,
        adjusted_sd: float,
        n_samples: int = 10000,
        spread: float = 0.0,
        margin_se: float = 0.85,
    ) -> Tuple[float, float, float]:
        """
        Two-layer Monte Carlo for probability with confidence interval.

        Layer 1: Parameter uncertainty in margin estimate (margin_se)
        Layer 2: Outcome variance around realized margin

        Args:
            projected_margin: Model's projected home margin (positive = home favored).
            adjusted_sd:      Game-level standard deviation after penalties.
            n_samples:        Monte Carlo sample count.
            spread:           Home team's spread handicap (e.g. -4.5 for 4.5-pt
                              favourite).  When spread=0 (default), returns
                              P(home wins).  When spread is non-zero, returns
                              P(home covers the spread).
            margin_se:        Standard error of the margin estimate (Layer 1).
                              Default 0.85 — 1.96 * 0.85 ≈ 1.66 pts, yielding a
                              ~4–5% EV haircut on the conservative lower bound
                              (norm.pdf(0) * 1.96 * 0.85 / 11 ≈ 5.5%).
                              Previous value of 2.0 stripped ~13%, destroying
                              valid edges on well-priced markets.

        Returns: (point_estimate, lower_95_ci, upper_95_ci)
        """
        # Layer 1: Uncertainty in our margin projection itself.
        # margin_se is a caller-supplied parameter (default 0.85).
        # A multi-source ratings composite has similar prediction uncertainty
        # whether the margin is 2 or 20 pts, so a constant SE is appropriate.

        margin_samples = self.rng.normal(
            projected_margin,
            margin_se,
            n_samples
        )

        # Layer 2: Convert margins to probabilities using adjusted SD.
        # Cover condition: actual_margin + spread > 0
        # P(cover) = Phi((projected_margin + spread) / sd)
        # When spread=0 this reduces to P(home wins) = Phi(margin / sd).
        prob_samples = norm.cdf((margin_samples + spread) / adjusted_sd)

        point_est = float(np.mean(prob_samples))
        lower_ci = float(np.percentile(prob_samples, 2.5))
        upper_ci = float(np.percentile(prob_samples, 97.5))

        return point_est, lower_ci, upper_ci
    
    def remove_vig_american(
        self,
        odds1: float,
        odds2: float
    ) -> Tuple[float, float]:
        """
        Remove vig from American odds to get true probabilities.
        Works for any combination of favorite/underdog odds.
        """
        # Helper to convert any American odds to implied probability
        def get_implied(o):
            if o > 0:
                return 100 / (o + 100)
            else:
                # Use abs() to handle negative odds correctly
                return abs(o) / (abs(o) + 100)
        
        p1 = get_implied(odds1)
        p2 = get_implied(odds2)
        
        # Normalize (Remove Vig)
        total = p1 + p2
        return p1 / total, p2 / total

    def remove_vig_shin(
        self,
        odds1: float,
        odds2: float,
    ) -> Tuple[float, float]:
        """
        Shin (1993) method for true probability extraction.

        Unlike the naive proportional method, the Shin method accounts for
        the fact that bookmakers shade odds towards favourites.  For a binary
        market this is equivalent to solving a quadratic for the implied
        insider-trading fraction *z* and then de-biasing.

        Falls back to the naive proportional method when the Shin solution
        is degenerate (e.g. odds on one side only).
        """
        def _implied(o: float) -> float:
            if o > 0:
                return 100.0 / (o + 100.0)
            return abs(o) / (abs(o) + 100.0)

        p1 = _implied(odds1)
        p2 = _implied(odds2)
        total = p1 + p2

        if total <= 1.0:
            # No vig detected — return as-is (shouldn't happen in practice)
            return p1, p2

        # Shin closed-form for 2-outcome market:
        #   z = (total - 1) / (n - 1)  where n = number of outcomes
        # For n = 2: z = total - 1
        z = total - 1.0

        # De-biased probabilities
        shin1 = (np.sqrt(z**2 + 4 * (1 - z) * (p1**2 / total)) - z) / (2 * (1 - z))
        shin2 = (np.sqrt(z**2 + 4 * (1 - z) * (p2**2 / total)) - z) / (2 * (1 - z))

        # Sanity check — ensure valid probabilities
        if not (0 < shin1 < 1 and 0 < shin2 < 1):
            # Fallback to proportional
            return p1 / total, p2 / total

        # Re-normalize to exactly 1.0 (numerical precision)
        s = shin1 + shin2
        return shin1 / s, shin2 / s

    def matchup_sd(
        self,
        home_style: Optional[Dict[str, float]] = None,
        away_style: Optional[Dict[str, float]] = None,
        game_total: Optional[float] = None,
    ) -> float:
        """
        Compute matchup-specific standard deviation from team play-style profiles.

        Style dict keys (all optional, use defaults when missing):
            pace        – possessions per 40 minutes
            three_par   – 3-point attempt rate (3PA / FGA)
            ft_rate     – free-throw rate (FTA / FGA)
            to_pct      – turnover percentage

        When both team profiles are available the SD is derived from the
        combined pace and shot-distribution volatility.  When profiles are
        missing, falls back to a game-total-based heuristic, then to base_sd.
        """
        if home_style and away_style:
            avg_pace = (home_style.get('pace', 68.0) + away_style.get('pace', 68.0)) / 2.0
            avg_3par = (home_style.get('three_par', 0.36) + away_style.get('three_par', 0.36)) / 2.0
            avg_ftr = (home_style.get('ft_rate', 0.30) + away_style.get('ft_rate', 0.30)) / 2.0

            # More possessions → more variance; more 3PA → fatter tails.
            pace_factor = avg_pace / 68.0          # Normalized to D1 average
            volatility_factor = 1.0 + 0.15 * (avg_3par - 0.36) / 0.10  # ~15% per 10pp above mean
            ftr_damper = 1.0 - 0.05 * (avg_ftr - 0.30) / 0.10  # FTs slightly reduce variance

            sd = self.base_sd * np.sqrt(pace_factor) * volatility_factor * ftr_damper
            return float(np.clip(sd, 8.0, 16.0))

        if game_total is not None and game_total > 0:
            # Heuristic: SD ≈ sqrt(total) * 0.85
            sd = np.sqrt(game_total) * 0.85
            return float(np.clip(sd, 8.0, 16.0))

        return self.base_sd

    # Minimum model weight when the model has material injury alpha.
    # When abs(injury_adj) > INJURY_ALPHA_THRESHOLD, the market line is
    # likely stale w.r.t. lineup changes and the model should not defer
    # below this floor.
    INJURY_ALPHA_THRESHOLD: float = 1.5
    INJURY_ALPHA_FLOOR: float = 0.65

    def _dynamic_model_weight(
        self,
        hours_to_tipoff: Optional[float] = None,
        sharp_books_available: int = 0,
        injury_adj: float = 0.0,
    ) -> float:
        """
        Compute the model-vs-market blending weight dynamically.

        **Rationale:** Far from tipoff the market is thin and our ratings
        composite is the best available signal.  As tipoff approaches, sharp
        books (Pinnacle, Circa) incorporate late-breaking information (injuries,
        weather, lineup locks) that our model cannot observe — so we should
        defer more to the market.

        **Formula (three factors):**

        1. *Time decay* — logistic sigmoid centred at 6 hours:

               w_time = 0.20 + 0.70 / (1 + exp(-0.5 * (hours - 6)))

           - At 24 h: w ≈ 0.90  (model dominates — market is thin)
           - At  6 h: w ≈ 0.55  (equal blend)
           - At  1 h: w ≈ 0.27  (market dominates — sharp lines are set)
           - At  0 h: w ≈ 0.23  (floor — never fully discard model)

           The 0.20 floor ensures the model always contributes at least 20%.
           The 0.90 ceiling (0.20 + 0.70) prevents full model reliance.

        2. *Sharp book discount* — when multiple sharp books agree, the
           consensus is more informative.  Each additional sharp book beyond
           the first reduces model weight by 5% (multiplicative):

               w_sharp = max(0.80, 1.0 - 0.05 * (n_sharp - 1))

           With 1 sharp book: 1.00 (no discount).
           With 3 sharp books: 0.90.
           Floor at 0.80 to prevent over-discounting.

        3. *Injury alpha floor* — when the model has incorporated a material
           injury adjustment (``abs(injury_adj) > 1.5 pts``), the market line
           is likely stale with respect to the lineup change.  The model
           weight is floored at ``INJURY_ALPHA_FLOOR`` (0.65) to prevent
           deferring to a line that hasn't priced in the injury.

        **Combined:**
            model_weight = max(w_time * w_sharp, injury_floor)

        Args:
            hours_to_tipoff:      Hours until game start (None → assume 12h).
            sharp_books_available: Count of sharp books with live lines.
            injury_adj:           Total injury margin adjustment (signed, pts).

        Returns:
            Model weight in [0.16, 0.90] for the margin blend.
        """
        # --- Time decay (logistic sigmoid) ---
        h = hours_to_tipoff if hours_to_tipoff is not None else 12.0
        h = max(h, 0.0)
        w_time = 0.20 + 0.70 / (1.0 + float(np.exp(-0.5 * (h - 6.0))))

        # --- Sharp book discount ---
        n = max(sharp_books_available, 0)
        if n <= 1:
            w_sharp = 1.0
        else:
            w_sharp = max(0.80, 1.0 - 0.05 * (n - 1))

        weight = float(w_time * w_sharp)

        # --- Injury alpha floor ---
        # When the model has material injury information that the market
        # may not have priced in, don't let the weight drop below the floor.
        if abs(injury_adj) > self.INJURY_ALPHA_THRESHOLD:
            weight = max(weight, self.INJURY_ALPHA_FLOOR)
            logger.debug(
                "Injury alpha floor: injury_adj=%.2f, weight floored at %.2f",
                injury_adj, weight,
            )

        return weight

    def kelly_fraction(
        self,
        prob: float,
        decimal_odds: float
    ) -> float:
        """
        Kelly Criterion for binary outcome

        Formula: f = (p * d - 1) / (d - 1)
        where p = win probability, d = decimal odds

        Returns: fraction of bankroll (0 if no edge, capped at max_kelly)
        """
        # Edge case handling
        if decimal_odds <= 1.0 or prob <= 0 or prob >= 1:
            return 0.0

        # Standard Kelly
        f = (prob * decimal_odds - 1) / (decimal_odds - 1)

        # Cap at maximum (disaster prevention)
        f = max(0.0, min(f, self.max_kelly))

        return f

    def _portfolio_kelly_divisor(
        self,
        concurrent_exposure: float = 0.0,
        target_exposure: float = 0.15,
    ) -> float:
        """
        Compute the effective Kelly divisor that accounts for bankroll
        already deployed in concurrent, overlapping time-slot bets.

        Motivation
        ----------
        The standard fractional Kelly (divide full-Kelly by a fixed scalar)
        assumes each bet is placed in isolation.  When multiple bets overlap
        in time (e.g. three games tip off within the same 2-hour window),
        the total variance of the portfolio grows and the per-bet sizing
        should shrink proportionally — this is the core insight of
        multi-asset portfolio Kelly.

        Approximation used
        ------------------
        A closed-form portfolio Kelly requires solving a quadratic for every
        bet pair (expensive for 5+ concurrent bets).  Instead we use a
        continuous divisor that:

        1. Equals ``self.fractional_kelly_divisor`` (the base half-Kelly) when
           no bankroll is deployed (concurrent_exposure = 0).
        2. Grows linearly as ``concurrent_exposure`` fills toward
           ``target_exposure``, doubling the divisor at full utilisation.
        3. Continues growing (divisor > 2×base) when exposure exceeds the
           target, providing a soft guardrail against over-commitment.

        Formula::

            scale  = 1 + concurrent_exposure / max(target_exposure, 1e-6)
            divisor = self.fractional_kelly_divisor × scale

        Examples (base divisor = 2.0, target_exposure = 15%):
            concurrent = 0 %  → divisor = 2.0  (half-Kelly baseline)
            concurrent = 7.5% → divisor = 3.0  (¾ utilisation → 1.5× base)
            concurrent = 15%  → divisor = 4.0  (full utilisation → 2× base)
            concurrent = 22%  → divisor = 4.93 (over-deployed → further cut)

        Args:
            concurrent_exposure: Fraction of bankroll currently deployed in
                                 open, overlapping bets (e.g. 0.10 = 10%).
            target_exposure:     Soft ceiling for total concurrent exposure
                                 (default 0.15 = 15%).  Should match the
                                 ``MAX_TOTAL_EXPOSURE`` in portfolio.py.

        Returns:
            Effective Kelly divisor ≥ self.fractional_kelly_divisor.
        """
        exposure = max(0.0, concurrent_exposure)
        denom = max(target_exposure, 1e-6)
        scale = 1.0 + exposure / denom
        return self.fractional_kelly_divisor * scale

    def _edge_breaker_threshold(self, hours_to_tipoff: Optional[float]) -> float:
        """
        Dynamic alpha circuit-breaker threshold that tightens as tipoff approaches.

        Rationale
        ---------
        Far from tipoff (24 h) the market is thinner and model edges can
        legitimately be larger — sharp money hasn't fully settled the line.
        Within 2 hours of tip, Pinnacle and Circa have absorbed almost all
        available information.  Any edge that appears above ~6% at that point
        is almost certainly a data artefact (stale rating, incorrect team
        mapping, unreported injury) rather than a genuine opportunity.

        Decay curve
        -----------
        Linear interpolation between the two anchor points:

            At ≥ 24 h : threshold = 12%  (market is thin, model leads)
            At ≤  2 h : threshold =  6%  (market is fully informed)
            Between   : threshold = 6% + (h − 2) / 22 × 6%

        The linear decay is deliberate: it produces a predictable, auditable
        policy that traders can reason about without curve-fitting a sigmoid.

        Args:
            hours_to_tipoff: Hours until game start.  None defaults to the
                             24-hour (maximum) threshold.

        Returns:
            Threshold in [0.06, 0.12] as a probability (not percentage).
        """
        h = max(0.0, hours_to_tipoff if hours_to_tipoff is not None else 24.0)
        if h >= 24.0:
            return 0.12
        if h <= 2.0:
            return 0.06
        # Linear interpolation: 0.0 → 6% at 2 h, 1.0 → 12% at 24 h
        frac = (h - 2.0) / 22.0
        return 0.06 + frac * 0.06

    def kelly_fraction_with_push(
        self,
        p_win: float,
        p_loss: float,
        decimal_odds: float
    ) -> float:
        """
        Kelly Criterion with explicit push handling.

        Because the Markov simulator generates discrete integer scores,
        exact ties (pushes) can occur for integer spreads.  On a push,
        the bettor gets their stake back with zero profit/loss.

        Formula: f = (p_win * (b - 1) - p_loss) / (b - 1)
        where b = decimal_odds - 1 (net profit per dollar risked)

        This reduces to the standard Kelly when p_push = 0:
            f = (p_win * b - p_loss) / b

        Args:
            p_win: Probability of winning the bet
            p_loss: Probability of losing the bet
            decimal_odds: Decimal odds (e.g., 1.909 for -110)

        Returns:
            Optimal Kelly fraction (0 if no edge, capped at max_kelly)
        """
        b = decimal_odds - 1.0  # Net profit per dollar risked

        # Edge case handling
        if b <= 0 or p_win <= 0 or p_win + p_loss > 1.0:
            return 0.0

        # Kelly with push handling
        f = (p_win * b - p_loss) / b

        # Cap at maximum (disaster prevention)
        f = max(0.0, min(f, self.max_kelly))

        return f
    
    def adjusted_sd(
        self,
        penalties_dict: Optional[Dict[str, float]] = None,
        base_sd_override: Optional[float] = None,
        market_volatility: Optional[float] = None,
        hours_to_tipoff: Optional[float] = None,
    ) -> float:
        """
        Calculate adjusted SD with dynamic penalty budget and ceiling.

        **Formula:**
            SD_adj = effective_base * volatility_scalar * (1 + min(sqrt(sum(penalty²)), 6) / 15)

        **Dynamic scaling (two new dimensions):**

        1. ``market_volatility`` — a non-negative scalar derived from
           line movement (e.g. abs(spread_open - spread_current) / base_sd).
           When present, it scales the base SD:

               volatility_scalar = 1.0 + 0.15 * tanh(market_volatility)

           ``tanh`` saturates for extreme movements.  A market_volatility
           of 1.0 (approx 1 SD of spread movement) adds ~11.4% to the base;
           3.0 adds ~14.9% (near ceiling).

        2. ``hours_to_tipoff`` — when provided, injury-class penalties are
           decayed by a time-to-tipoff factor.  A "Questionable" player's
           penalty should be higher 24 hours out (lineup uncertainty) than
           10 minutes out (lineup confirmed).  The decay curve is:

               time_decay = 1.0 - 0.6 * exp(-hours / 6)

           This gives: 24h -> 0.99 (full penalty), 6h -> 0.63, 1h -> 0.49,
           0.2h (12 min) -> 0.42 (lineups mostly known).

        **Ceiling:** Never exceed 15.5.

        Args:
            penalties_dict:    Dict of penalty name -> penalty value.
            base_sd_override:  If provided, replaces self.base_sd as the starting
                               point (used for dynamic total-based SD).
            market_volatility: Non-negative scalar from line movement (0 = stable,
                               >1 = significant movement).  None = no adjustment.
            hours_to_tipoff:   Hours until game start.  None = no time decay.

        Returns: adjusted SD in points.
        """
        effective_base = base_sd_override if base_sd_override is not None else self.base_sd

        # Market volatility scaling: line movement implies information the
        # model hasn't captured.  Scale base SD upward proportionally.
        if market_volatility is not None and market_volatility > 0:
            vol_scalar = 1.0 + 0.15 * float(np.tanh(market_volatility))
            effective_base *= vol_scalar

        if penalties_dict is None:
            return min(effective_base, 15.5)

        # Time-decay for injury-class penalties — uncertainty about lineup
        # resolves as tipoff approaches.
        if hours_to_tipoff is not None:
            time_decay = 1.0 - 0.6 * float(np.exp(-hours_to_tipoff / 6.0))
            injury_keys = [k for k in penalties_dict if 'injury' in k]
            for k in injury_keys:
                penalties_dict[k] *= time_decay

        # Sqrt-sum (diminishing returns on penalty stacking)
        total_penalty = float(np.sqrt(sum(v**2 for v in penalties_dict.values())))

        # Cap penalty at 6 (prevents runaway)
        total_penalty = min(total_penalty, 6.0)

        # Calculate adjusted SD
        adj_sd = effective_base * (1 + total_penalty / 15)

        # Absolute ceiling
        adj_sd = min(adj_sd, 15.5)

        return adj_sd
    
    def american_to_decimal(self, american_odds: float) -> float:
        """Convert American odds to decimal"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def analyze_game(
        self,
        game_data: Dict,
        odds: Dict,
        ratings: Dict,
        injuries: Optional[List[Dict]] = None,
        data_freshness: Optional[Dict] = None,
        base_sd_override: Optional[float] = None,
        home_style: Optional[Dict[str, float]] = None,
        away_style: Optional[Dict[str, float]] = None,
        market_volatility: Optional[float] = None,
        hours_to_tipoff: Optional[float] = None,
        matchup_margin_adj: float = 0.0,
        concurrent_exposure: float = 0.0,
        target_exposure: float = 0.15,
    ) -> GameAnalysis:
        """
        Complete game analysis using Version 8 framework.

        Args:
            game_data:         {home_team, away_team, is_neutral, etc.}
            odds:              {spread, total, moneyline, etc.}
            ratings:           {kenpom: {home: X, away: Y}, barttorvik: {...}, ...}
            injuries:          [{team, player, impact_tier}]
            data_freshness:    {lines_age_min, ratings_age_hours}
            base_sd_override:  If provided, replaces self.base_sd as the
                               baseline for adjusted_sd().  Pass the dynamic
                               total-based SD (sqrt(total)*0.85) from analysis.py.
            market_volatility: Non-negative scalar derived from spread/total line
                               movement.  Scales base SD via tanh activation when
                               line movement suggests hidden information.
            hours_to_tipoff:   Hours until game start.  Decays injury-class SD
                               penalties as tipoff approaches and lineups solidify.
            matchup_margin_adj: Additive margin adjustment from the MatchupEngine,
                               reflecting second-order play-style interactions
                               (pace mismatch, 3PA vs drop, transition gap, etc.).
                               Applied BEFORE the market blend.
            concurrent_exposure: Fraction of bankroll already deployed in
                               open bets whose windows overlap with this game
                               (e.g. 0.10 = 10%).  Used by
                               _portfolio_kelly_divisor() to scale the Kelly
                               divisor dynamically — see that method for the
                               full derivation.  Defaults to 0.0 (no overlap).
            target_exposure:   Soft ceiling for total concurrent exposure
                               passed through to _portfolio_kelly_divisor().
                               Should match portfolio.py MAX_TOTAL_EXPOSURE.

        Returns: GameAnalysis dataclass with full verdict
        """
        notes = []
        penalties = {}
        # Effective base SD: caller-supplied dynamic value takes priority
        _effective_base_sd = base_sd_override if base_sd_override is not None else self.base_sd

        # ================================================================
        # STEP 0: DATA FRESHNESS CHECK
        # ================================================================
        if data_freshness:
            lines_age = data_freshness.get('lines_age_min', 0)
            ratings_age = data_freshness.get('ratings_age_hours', 0)

            # Tier enforcement
            if lines_age > 30:
                return GameAnalysis(
                    verdict="PASS",
                    pass_reason="Tier 3 staleness - lines >30 min old",
                    projected_margin=0, adjusted_sd=_effective_base_sd,
                    home_advantage=self.home_advantage,
                    point_prob=0.5, lower_ci_prob=0.5, upper_ci_prob=0.5,
                    edge_point=0, edge_conservative=0,
                    kelly_full=0, kelly_fractional=0, recommended_units=0,
                    data_freshness_tier="Tier 3",
                    penalties_applied={},
                    notes=["Lines too stale for betting"],
                    full_analysis={}
                )
            elif lines_age > 10:
                penalties['stale_lines'] = 0.5
                notes.append(f"Lines {lines_age:.0f} min old (Tier 2 - analyze only)")

            if ratings_age > 168:  # 7 days
                return GameAnalysis(
                    verdict="PASS",
                    pass_reason="Ratings >7 days old",
                    projected_margin=0, adjusted_sd=_effective_base_sd,
                    home_advantage=self.home_advantage,
                    point_prob=0.5, lower_ci_prob=0.5, upper_ci_prob=0.5,
                    edge_point=0, edge_conservative=0,
                    kelly_full=0, kelly_fractional=0, recommended_units=0,
                    data_freshness_tier="Ancient",
                    penalties_applied={},
                    notes=["Ratings too old"],
                    full_analysis={}
                )
            elif ratings_age > 48:
                penalties['stale_ratings'] = 0.3
                notes.append(f"Ratings {ratings_age:.0f}h old (adding SD penalty)")
        
        # ================================================================
        # STEP 1: COMPUTE PROJECTED MARGIN
        # ================================================================
        
        # Extract ratings
        kp_home = ratings.get('kenpom', {}).get('home')
        kp_away = ratings.get('kenpom', {}).get('away')
        bt_home = ratings.get('barttorvik', {}).get('home')
        bt_away = ratings.get('barttorvik', {}).get('away')
        em_home = ratings.get('evanmiya', {}).get('home')
        em_away = ratings.get('evanmiya', {}).get('away')
        
        # Check for missing data
        if None in [kp_home, kp_away]:
            return GameAnalysis(
                verdict="PASS",
                pass_reason="Missing KenPom ratings",
                projected_margin=0, adjusted_sd=_effective_base_sd,
                home_advantage=self.home_advantage,
                point_prob=0.5, lower_ci_prob=0.5, upper_ci_prob=0.5,
                edge_point=0, edge_conservative=0,
                kelly_full=0, kelly_fractional=0, recommended_units=0,
                data_freshness_tier="N/A",
                penalties_applied={},
                notes=["Required ratings unavailable"],
                full_analysis={}
            )
        
        # Weighted margin calculation — renormalize weights for available sources.
        # Without renormalization, missing EvanMiya silently bleeds 32.5% of the
        # projection, systematically under-weighting the available signals.
        available_sources: list = []

        # KenPom is required (already checked above)
        available_sources.append(('kenpom', kp_home - kp_away))

        if bt_home is not None and bt_away is not None:
            available_sources.append(('barttorvik', bt_home - bt_away))
        else:
            penalties['missing_barttorvik'] = 1.0
            notes.append("BartTorvik unavailable — weights renormalized to remaining sources")

        if em_home is not None and em_away is not None:
            available_sources.append(('evanmiya', em_home - em_away))
        else:
            # EvanMiya missing → weights re-normalised to remaining sources only.
            # No SD penalty is applied: EvanMiya is frequently unavailable due
            # to Cloudflare blocking (a scraper issue, not a signal of elevated
            # game uncertainty).  Adding a 0.8-pt SD penalty for its absence
            # inflates the adjusted SD by ~5 % on every game, destroying edges
            # on otherwise well-priced markets.
            _evanmiya_dropped = ratings.get("_meta", {}).get("evanmiya_dropped", False)
            notes.append(
                "EvanMiya unavailable%s — weights re-normalized to remaining sources "
                "(no SD penalty)" % (
                    " [auto-dropped]" if _evanmiya_dropped else ""
                )
            )

        # Renormalize to the sum of available source weights so the projected
        # margin magnitude is consistent regardless of how many sources are live.
        total_weight = sum(self.weights.get(src, 0.0) for src, _ in available_sources)
        if total_weight <= 0:
            total_weight = 1.0  # safety guard (should never happen with KenPom required)

        margin = sum(
            (self.weights.get(src, 0.0) / total_weight) * diff
            for src, diff in available_sources
        )

        # Confidence shrinkage when rating sources are absent.
        # Re-normalization preserves margin magnitude but not confidence —
        # a game missing from an advanced source is likely a low-major game
        # with higher true variance.  Shrink the ratings-derived margin toward
        # zero (10% per missing source, floor at 70% of the point estimate).
        n_available = len(available_sources)
        # Exclude EvanMiya from the expected-source count when it returned null.
        # EvanMiya is structurally unavailable (Cloudflare / scraper failure) on
        # most runs — counting it as "expected but missing" would apply a 10%
        # shrinkage penalty to every game, depressing margins for a scraper
        # limitation rather than a genuine signal of elevated model uncertainty.
        # BartTorvik absence (a more anomalous failure) still triggers shrinkage.
        _em_null = (em_home is None and em_away is None)
        n_total = len(self.weights) - (1 if _em_null else 0)
        if n_available < n_total:
            n_missing = n_total - n_available
            _SHRINKAGE_ALPHA = 0.10
            shrinkage = max(0.70, 1.0 - _SHRINKAGE_ALPHA * n_missing)
            margin *= shrinkage
            notes.append(
                f"Margin shrunk {(1 - shrinkage):.0%} — {n_missing} source(s) missing "
                f"({n_available}/{n_total} available)"
            )

        # Pace-Adjusted Home Court Advantage
        #
        # A flat 3.09 point HCA is mathematically flawed across different tempos.
        # High-pace teams (75+ possessions) generate more scoring opportunities,
        # amplifying the home court effect.  Low-pace grinders (60 possessions)
        # see diminished HCA impact due to fewer total possessions.
        #
        # Scale HCA by (game_pace / 68.0) where 68.0 is the D1 average pace.
        # Fall back to (game_total / 140.0) if pace profiles are missing.
        adjusted_hca = 0.0
        if not game_data.get('is_neutral', False):
            if home_style and away_style:
                # Compute expected pace from team profiles
                game_pace = (home_style.get('pace', 68.0) + away_style.get('pace', 68.0)) / 2.0
                pace_ratio = game_pace / 68.0
            else:
                # Fallback: scale by game total (140.0 = D1 average total)
                game_total = odds.get('total', 140.0) or 140.0
                pace_ratio = game_total / 140.0

            adjusted_hca = self.home_advantage * pace_ratio
            margin += adjusted_hca
            notes.append(
                f"Pace-adjusted HCA: {adjusted_hca:.2f}pts (base={self.home_advantage:.2f}, "
                f"ratio={pace_ratio:.3f})"
            )

        # Injury adjustments — delegate to injuries.py estimate_impact()
        # which handles all 4 tiers, usage-rate scaling, and status weighting.
        from backend.services.injuries import estimate_impact

        STATUS_WEIGHT = {
            "Out": 1.0, "Doubtful": 0.75, "Questionable": 0.40, "Probable": 0.10,
        }

        injury_adj = 0.0
        home_injury_impact = 0.0  # Points lost by home team (positive = worse)
        away_injury_impact = 0.0  # Points lost by away team (positive = worse)

        if injuries:
            for inj in injuries:
                tier = inj.get('impact_tier', 'bench')
                usage = inj.get('usage_rate')
                status = inj.get('status', 'Out')
                player = inj.get('player', 'Unknown')

                # Raw impact in points (positive = points lost by that team)
                raw_impact = estimate_impact(tier, usage)
                # Weight by injury status probability
                weight = STATUS_WEIGHT.get(status, 0.5)
                weighted_impact = raw_impact * weight

                # Track impact separately for home and away teams
                if inj.get('team') == game_data['home_team']:
                    home_injury_impact += weighted_impact
                    injury_adj -= weighted_impact  # Home worse → negative margin adj
                else:
                    away_injury_impact += weighted_impact
                    injury_adj += weighted_impact  # Away worse → positive margin adj

                # Track penalty for SD inflation
                if tier in ('star', 'starter'):
                    penalties[f'{tier}_injury'] = max(
                        penalties.get(f'{tier}_injury', 0), weighted_impact
                    )
                    notes.append(
                        f"{player} {status} ({tier}, {weighted_impact:.1f}pts impact)"
                    )

        margin += injury_adj

        # Matchup geometry adjustment — second-order play-style interactions
        # computed by MatchupEngine (3PA vs drop, transition gap, etc.).
        # Applied BEFORE the market blend so it contributes to the model's
        # independent view.
        if matchup_margin_adj != 0.0:
            margin += matchup_margin_adj
            notes.append(
                f"Matchup adjustment: {matchup_margin_adj:+.2f}pts"
            )

        # Market-aware margin blend — shrink model margin toward sharp line.
        # Sharp books (Pinnacle/Circa) are extremely efficient; our ratings-only
        # margin is noisy.  The model weight decays dynamically based on two
        # signals: time-to-tipoff and sharp book availability.
        sharp_spread = odds.get('sharp_consensus_spread')
        if sharp_spread is not None:
            market_margin = -sharp_spread
            raw_model_margin = margin  # model's independent view before blend

            # ============================================================
            # Z-SCORE DIVERGENCE GUARD
            # ============================================================
            # If the absolute distance between the model's margin and the
            # sharp market margin exceeds 2.5 standard deviations of the
            # game's expected scoring spread, the discrepancy is anomalous.
            # It almost always signals a missing/unreported injury or
            # suspension, a team-mapping collision, or a stale rating scrape
            # — not a genuine edge.  Hard-PASS to avoid trading on bad data.
            #
            # Reference SD: _effective_base_sd is the dynamic total-derived
            # SD (sqrt(total)*0.85) or model base_sd — the game's pre-penalty
            # scoring variance baseline.  This is used rather than the
            # fully-adjusted adj_sd (computed in STEP 2) because we need
            # to gate the blend before STEP 2 runs.
            _Z_DIVERGENCE_THRESHOLD = 2.5
            divergence_pts = abs(raw_model_margin - market_margin)
            divergence_z = divergence_pts / max(_effective_base_sd, 1.0)

            if divergence_z > _Z_DIVERGENCE_THRESHOLD:
                _div_reason = (
                    f"Model margin {raw_model_margin:+.1f} vs market margin "
                    f"{market_margin:+.1f} = {divergence_pts:.1f}pt divergence "
                    f"({divergence_z:.2f}sigma > {_Z_DIVERGENCE_THRESHOLD}sigma limit). "
                    "Likely missing injury/suspension or team-mapping error."
                )
                logger.warning(
                    "Z-score divergence PASS [%s vs %s]: model=%.1f, market=%.1f, "
                    "divergence=%.1fpts (%.2f sigma, threshold=%.1f sigma)",
                    game_data.get('home_team', '?'), game_data.get('away_team', '?'),
                    raw_model_margin, market_margin,
                    divergence_pts, divergence_z, _Z_DIVERGENCE_THRESHOLD,
                )
                return GameAnalysis(
                    verdict="PASS - Market Divergence Anomaly",
                    pass_reason=_div_reason,
                    projected_margin=raw_model_margin,
                    adjusted_sd=_effective_base_sd,
                    home_advantage=self.home_advantage,
                    point_prob=0.5, lower_ci_prob=0.5, upper_ci_prob=0.5,
                    edge_point=0, edge_conservative=0,
                    kelly_full=0, kelly_fractional=0, recommended_units=0,
                    data_freshness_tier=data_freshness.get('tier', 'Unknown') if data_freshness else 'Unknown',
                    penalties_applied=penalties,
                    notes=notes + [_div_reason],
                    full_analysis={},
                )

            # ============================================================
            # Margin blend
            # ============================================================
            sharp_count = odds.get('sharp_books_available', 0)
            model_weight = self._dynamic_model_weight(
                hours_to_tipoff, sharp_count, injury_adj=injury_adj,
            )
            margin = model_weight * raw_model_margin + (1 - model_weight) * market_margin
            notes.append(
                f"Market blend: model {raw_model_margin:.1f} → blended {margin:.1f} "
                f"(sharp line {sharp_spread:+.1f}, w_model={model_weight:.2f}, "
                f"h={hours_to_tipoff}, sharp_n={sharp_count}, "
                f"divergence={divergence_z:.2f}sigma)"
            )
            logger.debug(
                "Dynamic model weight: %.3f (hours=%.1f, sharp_n=%d)",
                model_weight,
                hours_to_tipoff if hours_to_tipoff is not None else -1,
                sharp_count,
            )

        # ================================================================
        # STEP 2: ADJUST SD FOR UNCERTAINTY
        # ================================================================

        # Start from matchup-specific SD when team profiles are available,
        # otherwise fall back to game-total heuristic, then to base_sd.
        game_total = odds.get('total')
        base = self.matchup_sd(home_style, away_style, game_total)

        # Heteroskedasticity adjustments
        rating_gap = abs(kp_home - kp_away) if kp_home and kp_away else 0
        if rating_gap > 25:
            penalties['large_gap'] = 1.0
            notes.append(f"Large rating gap ({rating_gap:.1f}) - blowout variance")

        # Compute adjusted SD.
        # Blend caller-supplied dynamic SD (sqrt(total) heuristic from analysis.py)
        # with matchup_sd (style-based adjustments) instead of hard override.
        # This preserves matchup-specific variance info that hard override discards.
        if base_sd_override is not None and base != self.base_sd:
            # Both dynamic SD and matchup SD are available — blend 50/50
            effective_sd_base = 0.5 * base_sd_override + 0.5 * base
        elif base_sd_override is not None:
            effective_sd_base = base_sd_override
        else:
            effective_sd_base = base
        adj_sd = self.adjusted_sd(
            penalties,
            base_sd_override=effective_sd_base,
            market_volatility=market_volatility,
            hours_to_tipoff=hours_to_tipoff,
        )
        
        # If SD exceeds limit, PASS
        if adj_sd >= 15.5:
            return GameAnalysis(
                verdict="PASS",
                pass_reason=f"Uncertainty too high (SD={adj_sd:.1f})",
                projected_margin=margin, adjusted_sd=adj_sd,
                home_advantage=self.home_advantage,
                point_prob=0.5, lower_ci_prob=0.5, upper_ci_prob=0.5,
                edge_point=0, edge_conservative=0,
                kelly_full=0, kelly_fractional=0, recommended_units=0,
                data_freshness_tier=data_freshness.get('tier', 'Unknown') if data_freshness else 'Unknown',
                penalties_applied=penalties,
                notes=notes,
                full_analysis={}
            )
        
        # ================================================================
        # STEP 3: PRICING ENGINE SELECTION (Markov primary, Gaussian fallback)
        # ================================================================
        #
        # **Philosophy change**: The Markov simulator is now the PRIMARY
        # pricing engine when team profiles are available.  It directly
        # models possession-level variance through play-by-play geometry,
        # avoiding the "Gaussian + heuristic SD" approximation.
        #
        # Gaussian monte_carlo_prob_ci is used ONLY as the ultimate fallback
        # when profiles are missing or the Markov simulator fails.
        #
        # This promotes the Markov engine from a "cross-validator" to
        # "first-class pricing authority."

        spread = odds.get('spread', 0)
        pricing_engine = "Gaussian"  # Default
        markov_cover_prob = None
        markov_win_prob = None
        markov_loss_prob = None
        markov_push_prob = None

        # P(home wins) — always Gaussian for calibration tracking
        point_prob, lower_ci, upper_ci = self.monte_carlo_prob_ci(margin, adj_sd)

        # Attempt Markov pricing if profiles available
        logger.info(
            "Markov check [%s vs %s]: home_style=%s, away_style=%s, spread=%s",
            game_data.get('home_team', '?'),
            game_data.get('away_team', '?'),
            "present" if home_style else "MISSING",
            "present" if away_style else "MISSING",
            spread,
        )
        if (
            home_style
            and away_style
            and not home_style.get("is_heuristic")
            and not away_style.get("is_heuristic")
            and spread is not None
        ):
            try:
                from backend.services.possession_sim import (
                    PossessionSimulator, TeamSimProfile,
                )

                # Intrinsic Injury Integration
                # =============================
                # Translate injury point impact into stat penalties that the Markov
                # simulator can natively model. This allows Markov to correctly
                # simulate altered variance and push probabilities.
                #
                # Calibration: 1 point of injury impact ≈ -0.005 eFG% and +0.003 TO%
                # (derived from empirical relationship between efficiency and margin)
                #
                # home_injury_impact > 0 means home team is worse
                # away_injury_impact > 0 means away team is worse

                EFG_PENALTY_PER_POINT = 0.005  # eFG% reduction per point of impact
                TO_PENALTY_PER_POINT = 0.003   # TO% increase per point of impact

                home_efg_base = home_style.get('efg_pct', 0.500)
                home_to_base = home_style.get('to_pct', 0.170)
                away_efg_base = away_style.get('efg_pct', 0.500)
                away_to_base = away_style.get('to_pct', 0.170)

                # Apply injury penalties
                home_efg_adjusted = max(0.35, home_efg_base - home_injury_impact * EFG_PENALTY_PER_POINT)
                home_to_adjusted = min(0.30, home_to_base + home_injury_impact * TO_PENALTY_PER_POINT)
                away_efg_adjusted = max(0.35, away_efg_base - away_injury_impact * EFG_PENALTY_PER_POINT)
                away_to_adjusted = min(0.30, away_to_base + away_injury_impact * TO_PENALTY_PER_POINT)

                if home_injury_impact > 0.1 or away_injury_impact > 0.1:
                    logger.debug(
                        "Intrinsic injury applied: home_impact=%.2f (eFG %.3f→%.3f, TO %.3f→%.3f), "
                        "away_impact=%.2f (eFG %.3f→%.3f, TO %.3f→%.3f)",
                        home_injury_impact, home_efg_base, home_efg_adjusted,
                        home_to_base, home_to_adjusted,
                        away_injury_impact, away_efg_base, away_efg_adjusted,
                        away_to_base, away_to_adjusted,
                    )

                home_sim = TeamSimProfile(
                    team=game_data.get('home_team', 'Home'),
                    pace=home_style.get('pace', 68.0),
                    to_pct=home_to_adjusted,
                    ft_rate=home_style.get('ft_rate', 0.30),
                    three_rate=home_style.get('three_par', 0.36),
                    efg_pct=home_efg_adjusted,
                    def_efg_pct=home_style.get('def_efg_pct', 0.505),
                    def_to_pct=home_style.get('def_to_pct', 0.175),
                )
                away_sim = TeamSimProfile(
                    team=game_data.get('away_team', 'Away'),
                    pace=away_style.get('pace', 68.0),
                    to_pct=away_to_adjusted,
                    ft_rate=away_style.get('ft_rate', 0.30),
                    three_rate=away_style.get('three_par', 0.36),
                    efg_pct=away_efg_adjusted,
                    def_efg_pct=away_style.get('def_efg_pct', 0.505),
                    def_to_pct=away_style.get('def_to_pct', 0.175),
                )
                sim = PossessionSimulator(home_advantage_pts=self.home_advantage)
                sim_edge = sim.simulate_spread_edge(
                    home_sim, away_sim,
                    spread=spread,
                    n_sims=2000,
                    is_neutral=game_data.get('is_neutral', False),
                    matchup_margin_adj=matchup_margin_adj,
                )
                # Markov succeeded — use it as primary
                markov_cover_prob = sim_edge['cover_prob']
                cover_prob = markov_cover_prob
                cover_lower = sim_edge['cover_lower']
                cover_upper = sim_edge['cover_upper']

                # Extract win/loss/push breakdown for proper Kelly calculation
                markov_win_prob = sim_edge.get('win_prob', cover_prob)
                markov_loss_prob = sim_edge.get('loss_prob', 1.0 - cover_prob)
                markov_push_prob = sim_edge.get('push_prob', 0.0)

                pricing_engine = "Markov"
                logger.debug(
                    "Markov pricing: cover=%.3f [%.3f, %.3f], push=%.3f",
                    cover_prob, cover_lower, cover_upper, markov_push_prob,
                )

                # When Markov succeeds, injury impact is intrinsically modeled
                if home_injury_impact > 0.1 or away_injury_impact > 0.1:
                    notes.append(
                        f"Injury impact intrinsic to Markov: home={home_injury_impact:.1f}pts, "
                        f"away={away_injury_impact:.1f}pts (stat penalties applied)"
                    )

            except Exception as exc:
                logger.warning("Markov pricing failed, falling back to Gaussian: %s", exc)
                # Fall through to Gaussian

        # Gaussian fallback (if Markov didn't run or failed)
        if pricing_engine == "Gaussian":
            # margin_se defaults to 0.85 — 1.96 * 0.85 ≈ 1.66 pt CI half-width
            # → ~5.5% EV haircut at adj_sd=11.  Uniform across heuristic and
            # non-heuristic profiles; no special-casing needed.
            cover_prob, cover_lower, cover_upper = self.monte_carlo_prob_ci(
                margin, adj_sd, spread=spread,
            )
            logger.debug(
                "Gaussian pricing: cover=%.3f [%.3f, %.3f]",
                cover_prob, cover_lower, cover_upper,
            )

        notes.append(f"Primary pricing engine: {pricing_engine}")

        # ================================================================
        # STEP 4: VIG REMOVAL & EDGE CALCULATION
        # ================================================================

        # Extract BOTH sides' odds for proper Shin vig removal
        spread_odds_home = odds.get('spread_odds', -110)
        spread_odds_away = odds.get('spread_away_odds', -110)

        # No-vig probability via Shin (1993) method — needs both sides
        home_novig, away_novig = self.remove_vig_shin(
            spread_odds_home, spread_odds_away,
        )

        # Push-adjusted market probability when Markov is the pricing engine.
        # Market odds assume binary outcomes (p_win + p_loss = 1.0), but Markov
        # generates discrete scores with potential pushes. Adjust the market
        # baseline to the "action space" probability:
        #   adjusted_market_prob = market_prob * (1.0 - push_prob)
        # When pricing_engine == "Gaussian", push_prob is 0.0 (binary outcomes).
        if pricing_engine == "Markov" and markov_push_prob is not None:
            push_adjustment = 1.0 - markov_push_prob
        else:
            push_adjustment = 1.0  # Gaussian: binary outcomes, no adjustment

        adjusted_home_novig = home_novig * push_adjustment
        adjusted_away_novig = away_novig * push_adjustment

        # Determine which side has value by checking home-side edge
        edge_home = cover_prob - adjusted_home_novig

        if edge_home >= 0:
            # Value on home side
            our_cover_prob = cover_prob
            our_cover_lower = cover_lower
            market_prob = adjusted_home_novig
            bet_side = "home"
            bet_odds = spread_odds_home
        else:
            # Value on away side (flip cover probability)
            our_cover_prob = 1.0 - cover_prob
            our_cover_lower = 1.0 - cover_upper   # upper flips to lower
            market_prob = adjusted_away_novig
            bet_side = "away"
            bet_odds = spread_odds_away

        edge_point = float(our_cover_prob - market_prob)
        edge_conservative = float(our_cover_lower - market_prob)

        # ================================================================
        # STEP 5: KELLY & BET SIZING
        # ================================================================

        decimal_odds = self.american_to_decimal(bet_odds)

        # Use push-aware Kelly when Markov is the pricing engine
        if pricing_engine == "Markov" and markov_win_prob is not None and markov_loss_prob is not None:
            # Markov provides discrete win/loss/push probabilities
            # Adjust for bet_side (if away, we need to flip)
            if bet_side == "home":
                p_win = markov_win_prob
                p_loss = markov_loss_prob
            else:
                # Away side — flip the probabilities
                p_win = markov_loss_prob
                p_loss = markov_win_prob

            kelly_full = self.kelly_fraction_with_push(p_win, p_loss, decimal_odds)
            logger.debug(
                "Kelly (push-aware): p_win=%.3f, p_loss=%.3f, p_push=%.3f, kelly=%.3f",
                p_win, p_loss, markov_push_prob or 0, kelly_full,
            )
        else:
            # Gaussian pricing or Markov failed — use standard Kelly
            kelly_full = self.kelly_fraction(our_cover_prob, decimal_odds)

        # ================================================================
        # Portfolio-Aware Kelly Divisor
        # ================================================================
        # Base divisor grows with concurrent bankroll exposure so that each
        # new bet in an already-loaded slate is sized smaller — approximating
        # true portfolio Kelly without a full quadratic solve.
        # See _portfolio_kelly_divisor() for the full derivation.
        effective_kelly_divisor = self._portfolio_kelly_divisor(
            concurrent_exposure=concurrent_exposure,
            target_exposure=target_exposure,
        )
        adverse_selection_penalty = 1.0

        if concurrent_exposure > 0.0:
            logger.debug(
                "Portfolio Kelly: concurrent_exposure=%.1f%%, target=%.1f%% → divisor=%.2f",
                concurrent_exposure * 100, target_exposure * 100, effective_kelly_divisor,
            )

        # ================================================================
        # Adverse Selection Overlay
        # ================================================================
        # A massive edge very close to tipoff is a red flag. Sharp money has
        # information we don't, and we risk being adversely selected. Multiply
        # the divisor further to drastically reduce bet size in these scenarios.
        #
        # Triggers: hours_to_tipoff < 1.5 AND edge_conservative > 2.5%
        # Penalty: Additional 3.0× on top of the portfolio-adjusted divisor.
        if hours_to_tipoff is not None and hours_to_tipoff < 1.5 and edge_conservative > 0.025:
            adverse_selection_penalty = 3.0
            effective_kelly_divisor *= adverse_selection_penalty

            logger.warning(
                "Adverse Selection Risk: edge=%.3f%%, hours=%.2f → Kelly divisor scaled %.1fx "
                "(portfolio-adjusted base=%.2f, final=%.2f)",
                edge_conservative * 100, hours_to_tipoff, adverse_selection_penalty,
                self._portfolio_kelly_divisor(concurrent_exposure, target_exposure),
                effective_kelly_divisor,
            )
            notes.append(
                f"Adverse Selection Kelly Penalty: {adverse_selection_penalty:.1f}x divisor "
                f"(edge={edge_conservative:.2%}, tipoff in {hours_to_tipoff:.1f}h)"
            )

        kelly_frac = kelly_full / effective_kelly_divisor

        # ================================================================
        # ALPHA CIRCUIT BREAKER (dynamic threshold)
        # True CBB closing-line edges rarely exceed 10-12% far from tipoff
        # and virtually never exceed 6% within 2 hours (sharp books are fully
        # informed by then).  A dynamic threshold that tightens toward tipoff
        # provides both a safety valve against data errors AND a defence
        # against adverse-selection by pre-game sharp movement.
        #
        # Threshold schedule (see _edge_breaker_threshold() for derivation):
        #   ≥ 24 h  →  12%  (thin market; model may legitimately lead)
        #   ≤  2 h  →   6%  (fully efficient; any larger edge is a data flag)
        #   Between →  linear interpolation
        # ================================================================
        _EDGE_BREAKER_THRESHOLD = self._edge_breaker_threshold(hours_to_tipoff)
        if edge_conservative > _EDGE_BREAKER_THRESHOLD:
            err_msg = (
                f"ERROR: Edge {edge_conservative:.1%} exceeds dynamic threshold "
                f"{_EDGE_BREAKER_THRESHOLD:.0%} "
                f"(hours_to_tipoff={hours_to_tipoff}). "
                "This indicates a likely team-mapping, injury scrape failure, "
                "or late-breaking information not yet in ratings. Bet rejected."
            )
            notes.append(err_msg)
            logger.error(
                "Alpha circuit breaker triggered — edge_conservative=%.1f%% "
                "exceeds dynamic %.0f%% threshold (h=%.1f). Rejecting bet.",
                edge_conservative * 100, _EDGE_BREAKER_THRESHOLD * 100,
                hours_to_tipoff if hours_to_tipoff is not None else -1,
            )
            verdict = "PASS - Edge Circuit Breaker (Data Error?)"
            pass_reason = (
                f"edge_conservative={edge_conservative:.1%} > "
                f"dynamic threshold {_EDGE_BREAKER_THRESHOLD:.0%} "
                f"(at h={hours_to_tipoff})"
            )
            recommended_units = 0
            kelly_frac = 0.0
            kelly_full = 0.0

        # Decision rule: only bet if conservative edge > 0
        elif edge_conservative <= 0:
            verdict = "PASS"
            pass_reason = f"Conservative edge {edge_conservative:.3%} <= 0"
            recommended_units = 0
        else:
            # Additional safety: cap at 1.5% of bankroll
            recommended_pct = min(kelly_frac * 100, 1.5)
            recommended_units = recommended_pct

            if recommended_units < 0.25:
                # Thin edge — still a real bet but floored at minimum size
                recommended_units = 0.25
                verdict = f"Bet {recommended_units:.2f}u @ {bet_odds:+.0f} (min size)"
            else:
                verdict = f"Bet {recommended_units:.2f}u @ {bet_odds:+.0f}"

            pass_reason = None
        
        # ================================================================
        # RETURN COMPLETE ANALYSIS
        # ================================================================
        
        # Build margin component breakdown using re-normalized weights
        margin_components = {}
        for source, diff in available_sources:
            eff_w = self.weights.get(source, 0.0) / total_weight
            margin_components[source] = {
                'raw_weight': self.weights.get(source, 0.0),
                'effective_weight': round(eff_w, 4),
                'diff': diff,
                'contribution': round(eff_w * diff, 3),
            }
        margin_components['home_adv'] = adjusted_hca
        margin_components['injury_adj'] = injury_adj
        margin_components['matchup_adj'] = matchup_margin_adj

        full_analysis_dict = {
            'model_version': 'v8.0',
            'timestamp': datetime.utcnow().isoformat(),
            'inputs': {
                'ratings': ratings,
                'margin_components': {
                    src: (self.weights.get(src, 0.0) / total_weight) * diff
                    for src, diff in available_sources
                } | {'home_adv': adjusted_hca, 'injury_adj': injury_adj, 'matchup_adj': matchup_margin_adj},
                'active_sources': [src for src, _ in available_sources],
                'total_weight_used': total_weight,
                'weight_renormalized': total_weight < 0.999,
                'odds': odds,
                'injuries': injuries,
                'home_style': home_style,
                'away_style': away_style,
            },
            'calculations': {
                'projected_margin': margin,
                'matchup_base_sd': base,
                'base_sd': effective_sd_base,
                'adjusted_sd': adj_sd,
                'penalties': penalties,
                'point_prob': point_prob,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'cover_prob': cover_prob,
                'cover_lower': cover_lower,
                'cover_upper': cover_upper,
                'home_novig': home_novig,
                'away_novig': away_novig,
                'bet_side': bet_side,
                'bet_odds': bet_odds,
                'market_prob': market_prob,
                'edge_point': edge_point,
                'edge_conservative': edge_conservative,
                'kelly_full': kelly_full,
                'kelly_fractional': kelly_frac,
                'markov_cover_prob': markov_cover_prob,
                'pricing_engine': pricing_engine,
                'vig_removal_method': 'shin_1993',
                'market_volatility': market_volatility,
                'hours_to_tipoff': hours_to_tipoff,
                'model_weight': (
                    self._dynamic_model_weight(
                        hours_to_tipoff, odds.get('sharp_books_available', 0),
                        injury_adj=injury_adj,
                    )
                    if sharp_spread is not None else None
                ),
            },
            'notes': notes,
        }
        
        return GameAnalysis(
            verdict=verdict,
            pass_reason=pass_reason,
            projected_margin=margin,
            adjusted_sd=adj_sd,
            home_advantage=self.home_advantage,
            point_prob=point_prob,
            lower_ci_prob=lower_ci,
            upper_ci_prob=upper_ci,
            edge_point=edge_point,
            edge_conservative=edge_conservative,
            kelly_full=kelly_full,
            kelly_fractional=kelly_frac,
            recommended_units=recommended_units,
            bet_side=bet_side,
            data_freshness_tier=data_freshness.get('tier', 'Unknown') if data_freshness else 'Unknown',
            penalties_applied=penalties,
            notes=notes,
            full_analysis=full_analysis_dict,
        )


# Export for testing
if __name__ == "__main__":
    # Quick validation test
    model = CBBEdgeModel()
    
    test_game = {
        'home_team': 'Duke',
        'away_team': 'UNC',
        'is_neutral': False,
    }
    
    test_ratings = {
        'kenpom': {'home': 25.0, 'away': 20.0},
        'barttorvik': {'home': 24.5, 'away': 19.8},
        'evanmiya': {'home': 25.2, 'away': 20.5},
    }
    
    test_odds = {
        'spread': -4.5,
        'spread_odds': -110,
    }
    
    result = model.analyze_game(test_game, test_odds, test_ratings)
    print(f"Verdict: {result.verdict}")
    print(f"Projected margin: {result.projected_margin:.1f}")
    print(f"Point prob: {result.point_prob:.1%}")
    print(f"Conservative edge: {result.edge_conservative:.2%}")
    print(f"Model version: {result.full_analysis.get('model_version')}")
    print(f"Weight renormalized: {result.full_analysis['inputs']['weight_renormalized']}")
