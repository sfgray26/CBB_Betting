"""
Possession-based Markov simulation engine.

Instead of taking ``Team A AdjEM - Team B AdjEM`` and wrapping it in a
normal distribution, this engine simulates individual possessions as a
Markov chain.  Each possession resolves through a state machine:

    BALL → {3PA, 2PA_RIM, 2PA_MID, FT_TRIP, TURNOVER, OREB_CYCLE}

The transition probabilities come from each team's Four Factors profile
(eFG%, TO%, ORB%, FTR) conditioned on the opponent's defensive profile.

By running 10,000 simulations the engine produces:
    - Full score distributions (not just mean + SD)
    - Style-emergent variance (no heuristic SD needed)
    - Direct win probabilities with confidence intervals
    - 1st-half and team-total sub-distributions for derivative markets

Usage::

    sim = PossessionSimulator()
    result = sim.simulate_game(home_profile, away_profile, n_sims=10000)
    print(result.home_win_prob, result.score_distribution)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FT trip-type distribution (NCAA baseline, 2024-25 season)
# ---------------------------------------------------------------------------
# Empirical breakdown of the *type* of FT trip drawn from play-by-play data:
#   AND1        – shooter made the FG and earns 1 bonus FT  (~12%)
#   TWO_SHOT    – standard shooting foul (double-bonus) → 2 FTs  (~65%)
#   THREE_SHOT  – fouled on a 3-point attempt → 3 FTs  (~8%)
#   ONE_AND_ONE – single-bonus situation (7th–9th foul, college only)  (~15%)
#
# The weights are used in two places:
#   1. To compute the *trip-implied ft_prob multiplier* that maps the
#      blended FTA/FGA rate onto a per-possession FT-trip probability.
#   2. To simulate the actual FT sequence in _simulate_possession().
#
# Math:  ft_prob = blended_ftr × (FGA_PER_POSS / E[FTs per trip])
#        E[FTs | 1-and-1] ≈ 1 + ft_make_prob  (first hit triggers second)

_FT_TRIP_AND1        = 0  # index into arrays below
_FT_TRIP_TWO_SHOT    = 1
_FT_TRIP_THREE_SHOT  = 2
_FT_TRIP_ONE_AND_ONE = 3

_FT_TRIP_WEIGHTS: np.ndarray = np.array([0.12, 0.65, 0.08, 0.15])

# Average FGA generated per possession (accounts for ~17% TO rate reducing
# the fraction of possessions that reach a shot attempt).
_FGA_PER_POSS: float = 0.65

# D1-average FT make probability used for 1-and-1 EV pre-computation.
_D1_FT_MAKE: float = 0.72

def _compute_ft_multiplier() -> float:
    """
    Derive the scalar that converts a blended FTA/FGA rate to a
    per-possession FT-trip probability.

    Derivation
    ----------
    FT trips / possession = (FTA/FGA) × (FGA/poss) / E[FTs per trip]

    For 1-and-1, E[FTs] = 1 + ft_make_prob (only if first is made).
    """
    e_fts_and1       = 1.0
    e_fts_two        = 2.0
    e_fts_three      = 3.0
    e_fts_one_and_one = 1.0 + _D1_FT_MAKE   # ≈ 1.72

    e_fts_per_trip = (
        _FT_TRIP_WEIGHTS[_FT_TRIP_AND1]        * e_fts_and1 +
        _FT_TRIP_WEIGHTS[_FT_TRIP_TWO_SHOT]    * e_fts_two +
        _FT_TRIP_WEIGHTS[_FT_TRIP_THREE_SHOT]  * e_fts_three +
        _FT_TRIP_WEIGHTS[_FT_TRIP_ONE_AND_ONE] * e_fts_one_and_one
    )
    return _FGA_PER_POSS / e_fts_per_trip  # ≈ 0.339

# Cached at import time — pure function, no I/O.
_FT_PROB_MULTIPLIER: float = _compute_ft_multiplier()


# ---------------------------------------------------------------------------
# Team profile for simulation
# ---------------------------------------------------------------------------

@dataclass
class TeamSimProfile:
    """
    Team-level profile derived from play-by-play or box-score data.

    Four Factors (tempo-free):
        efg_pct     – Effective FG% = (FGM + 0.5 * 3PM) / FGA
        to_pct      – Turnover rate  = TO / possessions
        orb_pct     – Offensive rebound rate
        ft_rate     – Free throw rate = FTA / FGA

    Shot distribution (fractions summing to ~1.0):
        rim_rate    – % of FGA at the rim
        mid_rate    – % of FGA from mid-range
        three_rate  – % of FGA from 3 (= 3PA / FGA)

    Shot accuracy:
        rim_fg_pct  – FG% at the rim
        mid_fg_pct  – FG% from mid-range
        three_fg_pct – FG% from 3

    Tempo:
        pace        – Possessions per 40 minutes
    """

    team: str

    # Four Factors
    efg_pct: float = 0.500
    to_pct: float = 0.170
    orb_pct: float = 0.280
    ft_rate: float = 0.300

    # Shot distribution
    rim_rate: float = 0.320
    mid_rate: float = 0.150
    three_rate: float = 0.360
    # Remainder (~0.17) is FT possessions / turnovers — handled implicitly

    # Shot accuracy
    rim_fg_pct: float = 0.620
    mid_fg_pct: float = 0.380
    three_fg_pct: float = 0.340

    # Tempo
    pace: float = 68.0

    # Defence (opponent-adjusted rates, used when defending)
    def_efg_pct: float = 0.505        # Opponent eFG% allowed (D1 avg baseline)
    def_to_pct: float = 0.175         # Opponent TO rate forced (D1 avg baseline)
    def_to_forced_pct: float = 0.190  # Legacy alias — def_to_pct is now canonical
    def_drb_pct: float = 0.720        # Defensive rebound rate (= 1 - opponent ORB)
    def_ft_rate_allowed: float = 0.280


@dataclass
class SimulationResult:
    """Output of a game simulation."""

    n_sims: int
    home_scores: np.ndarray = field(repr=False)
    away_scores: np.ndarray = field(repr=False)

    # First-half distributions (for derivative markets)
    home_1h_scores: Optional[np.ndarray] = field(default=None, repr=False)
    away_1h_scores: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def home_win_prob(self) -> float:
        return float(np.mean(self.home_scores > self.away_scores))

    @property
    def away_win_prob(self) -> float:
        return float(np.mean(self.away_scores > self.home_scores))

    @property
    def push_prob(self) -> float:
        return float(np.mean(self.home_scores == self.away_scores))

    @property
    def projected_margin(self) -> float:
        return float(np.mean(self.home_scores - self.away_scores))

    @property
    def margin_sd(self) -> float:
        return float(np.std(self.home_scores - self.away_scores))

    @property
    def projected_total(self) -> float:
        return float(np.mean(self.home_scores + self.away_scores))

    @property
    def total_sd(self) -> float:
        return float(np.std(self.home_scores + self.away_scores))

    def spread_cover_prob(self, spread: float) -> float:
        """
        Probability home team covers the given spread (e.g. -4.5).

        Does NOT include pushes in the cover probability.
        For integer spreads, use spread_cover_probs() to get win/loss/push breakdown.
        """
        margins = self.home_scores - self.away_scores
        return float(np.mean(margins + spread > 0))

    def spread_cover_probs(self, spread: float) -> Dict[str, float]:
        """
        Return win/loss/push probabilities for a spread bet.

        Because Markov generates discrete integer scores, exact ties
        (pushes) can occur for integer spreads. This method explicitly
        calculates all three outcomes.

        Returns:
            {'win': P(cover), 'loss': P(don't cover), 'push': P(tie)}
        """
        margins = self.home_scores - self.away_scores
        adjusted = margins + spread

        win_prob = float(np.mean(adjusted > 0))
        push_prob = float(np.mean(adjusted == 0))
        loss_prob = float(np.mean(adjusted < 0))

        return {
            'win': win_prob,
            'loss': loss_prob,
            'push': push_prob,
        }

    def total_over_prob(self, total: float) -> float:
        """Probability the game goes over the given total."""
        totals = self.home_scores + self.away_scores
        return float(np.mean(totals > total))

    def home_team_total_over_prob(self, line: float) -> float:
        """Probability home team scores over a team total line."""
        return float(np.mean(self.home_scores > line))

    def away_team_total_over_prob(self, line: float) -> float:
        """Probability away team scores over a team total line."""
        return float(np.mean(self.away_scores > line))

    def first_half_spread_prob(self, spread: float) -> float:
        """Probability home team covers the 1H spread."""
        if self.home_1h_scores is None or self.away_1h_scores is None:
            return 0.5
        margins = self.home_1h_scores - self.away_1h_scores
        return float(np.mean(margins + spread > 0))

    def percentile_margin(self, pct: float) -> float:
        return float(np.percentile(self.home_scores - self.away_scores, pct))

    def ci_win_prob(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Bootstrap CI on win probability."""
        n = len(self.home_scores)
        wins = (self.home_scores > self.away_scores).astype(float)
        bootstrap = np.random.choice(wins, size=(1000, n), replace=True)
        probs = np.mean(bootstrap, axis=1)
        alpha = (1 - confidence) / 2
        return float(np.percentile(probs, alpha * 100)), float(np.percentile(probs, (1 - alpha) * 100))

    def ci_cover_prob(self, spread: float, confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap CI on spread cover probability.

        Returns (point_estimate, lower_bound, upper_bound).

        Note: This returns the WIN probability only (not including pushes).
        For full win/loss/push breakdown, use ci_cover_probs_full().
        """
        n = len(self.home_scores)
        margins = self.home_scores - self.away_scores
        covers = (margins + spread > 0).astype(float)
        point_est = float(np.mean(covers))

        # Bootstrap CI
        bootstrap = np.random.choice(covers, size=(1000, n), replace=True)
        probs = np.mean(bootstrap, axis=1)
        alpha = (1 - confidence) / 2
        lower = float(np.percentile(probs, alpha * 100))
        upper = float(np.percentile(probs, (1 - alpha) * 100))

        return point_est, lower, upper

    def ci_cover_probs_full(self, spread: float, confidence: float = 0.95) -> Dict:
        """
        Bootstrap CI on win/loss/push probabilities with full breakdown.

        Returns dict with:
            'win': point estimate
            'win_lower': lower CI bound
            'win_upper': upper CI bound
            'loss': point estimate
            'push': point estimate
        """
        n = len(self.home_scores)
        margins = self.home_scores - self.away_scores
        adjusted = margins + spread

        # Point estimates
        wins = (adjusted > 0).astype(float)
        pushes = (adjusted == 0).astype(float)
        losses = (adjusted < 0).astype(float)

        win_prob = float(np.mean(wins))
        push_prob = float(np.mean(pushes))
        loss_prob = float(np.mean(losses))

        # Bootstrap CI for win probability
        bootstrap = np.random.choice(wins, size=(1000, n), replace=True)
        probs = np.mean(bootstrap, axis=1)
        alpha = (1 - confidence) / 2
        win_lower = float(np.percentile(probs, alpha * 100))
        win_upper = float(np.percentile(probs, (1 - alpha) * 100))

        return {
            'win': win_prob,
            'win_lower': win_lower,
            'win_upper': win_upper,
            'loss': loss_prob,
            'push': push_prob,
        }

    def to_dict(self) -> Dict:
        return {
            "n_sims": self.n_sims,
            "home_win_prob": round(self.home_win_prob, 4),
            "away_win_prob": round(self.away_win_prob, 4),
            "projected_margin": round(self.projected_margin, 2),
            "margin_sd": round(self.margin_sd, 2),
            "projected_total": round(self.projected_total, 2),
            "total_sd": round(self.total_sd, 2),
            "margin_5th": round(self.percentile_margin(5), 1),
            "margin_95th": round(self.percentile_margin(95), 1),
        }


# ---------------------------------------------------------------------------
# Possession simulator
# ---------------------------------------------------------------------------

class PossessionSimulator:
    """
    Markov-chain possession simulator for college basketball.

    Each possession resolves through these states::

        START → TURNOVER (end, 0 points)
              → SHOT_ATTEMPT → {MAKE_2, MAKE_3, MISS}
              → FREE_THROW_TRIP → {0, 1, 2, 3 points}
              → MISS → OFFENSIVE_REBOUND → (re-enter SHOT_ATTEMPT)
                     → DEFENSIVE_REBOUND → (end, 0 points)

    Transition probabilities are computed by blending the offence's
    tendency rates with the defence's resistance rates.
    """

    def __init__(self, home_advantage_pts: float = 3.09):
        self.home_advantage_pts = home_advantage_pts

    def _blend_rate(self, off_rate: float, def_rate: float, d1_avg: float) -> float:
        """
        Blend offensive tendency with defensive resistance using the
        multiplicative baseline method (Dean Oliver, *Basketball on Paper*, 2004).

        **Formula:**

            blended = (off_rate × def_rate) / d1_avg

        **Mathematical justification:**
        Each input is expressed as a rate relative to league-average performance.
        Multiplying them and dividing by the baseline yields a combined rate that:

          - Scales linearly in each factor (doubling offensive tendency doubles
            the blended rate, holding defence constant).
          - Preserves the identity: when both inputs equal d1_avg, the output
            equals d1_avg.
          - Amplifies extremes: a historically strong offence matched against a
            historically weak defence produces a rate above d1_avg, and vice
            versa — consistent with how basketball efficiency metrics compound.

        This differs from the previous Log5 (Bill James) implementation:
          - Log5 is logistic / odds-ratio based and saturates strongly at
            extremes, which is appropriate for win-probability estimation but
            over-compresses at the tails for *rate* estimation of box-score stats.
          - The multiplicative baseline is the standard for efficiency metrics
            (eFG%, TO%, ORB%) in basketball analytics because those rates can
            legitimately exceed 100% in extreme matchups (e.g. an elite 3PT
            offense attacking a historically bad 3PT defense).  The hard clip
            to [0.01, 0.99] prevents numerically degenerate possession chains.

        **Orientation of call sites** (all four pass rates in the same
        "event-success probability" orientation):
            - TO rate:  off.to_pct, def.def_to_pct          → "P(turnover)"
            - FT rate:  off.ft_rate, def.def_ft_rate_allowed → "P(FT trip)"
            - eFG:      off.efg_pct, def.def_efg_pct         → "P(eFG success)"
            - ORB:      off.orb_pct, (1 - def.def_drb_pct)   → "P(off rebound)"

        Args:
            off_rate: Offensive team's event probability vs average defence.
            def_rate: Defence's event probability allowed vs average offence
                      (higher = worse defence).
            d1_avg:   D1 baseline rate for the event.  Must be > 0.

        Returns:
            Blended probability clipped to [0.01, 0.99].
        """
        # Clip inputs to defensible probability bounds before multiplying.
        # Unlike Log5, the multiplicative formula can produce values > 1.0 when
        # both teams are extreme (e.g. eFG 62% × 60% / 50% = 74.4%), which is
        # numerically valid but must be clamped to keep downstream code stable.
        a = float(np.clip(off_rate, 0.01, 0.99))
        b = float(np.clip(def_rate, 0.01, 0.99))
        c = float(np.clip(d1_avg,   1e-6, 1.0 - 1e-6))

        raw = (a * b) / c
        return float(np.clip(raw, 0.01, 0.99))

    def _compute_possession_probs(
        self,
        offence: TeamSimProfile,
        defence: TeamSimProfile,
    ) -> Dict[str, float]:
        """
        Compute per-possession transition probabilities.

        Returns dict with keys:
            to_prob, ft_prob, shot_prob,
            rim_prob, mid_prob, three_prob,
            rim_make, mid_make, three_make,
            orb_prob, ft_make_prob
        """
        # D1 averages for blending (2024-25 season)
        D1_TO = 0.170
        D1_ORB = 0.280
        D1_EFG = 0.500
        D1_FTR = 0.300

        # Turnover probability — blend offence TO rate with defence's canonical
        # def_to_pct (opponent-TO rate forced, D1 avg 0.175).  def_to_forced_pct
        # is kept for legacy callers but def_to_pct is now the live field.
        to_prob = self._blend_rate(offence.to_pct, defence.def_to_pct, D1_TO)

        # Free throw trip probability
        # Convert blended FTA/FGA rate → per-possession FT-trip probability
        # using the trip-type-distribution-derived multiplier instead of a
        # hardcoded 0.35 scalar.
        ft_prob = self._blend_rate(offence.ft_rate, defence.def_ft_rate_allowed, D1_FTR)
        ft_prob = ft_prob * _FT_PROB_MULTIPLIER  # ≈ 0.339 (derived from trip distribution)

        # Shot attempt probability = 1 - TO - FT trips
        shot_prob = max(0.01, 1.0 - to_prob - ft_prob)

        # Shot type distribution (from offence, not blended — offence chooses shots)
        total_shot = offence.rim_rate + offence.mid_rate + offence.three_rate
        if total_shot > 0:
            rim_prob = offence.rim_rate / total_shot
            mid_prob = offence.mid_rate / total_shot
            three_prob = offence.three_rate / total_shot
        else:
            rim_prob, mid_prob, three_prob = 0.38, 0.18, 0.44

        # Shot accuracy (blend offence accuracy with defence eFG allowed)
        efg_blend = self._blend_rate(offence.efg_pct, defence.def_efg_pct, D1_EFG)
        # Scale individual shot types by the eFG blend ratio
        efg_ratio = efg_blend / max(offence.efg_pct, 0.01)

        rim_make = min(0.95, offence.rim_fg_pct * efg_ratio)
        mid_make = min(0.95, offence.mid_fg_pct * efg_ratio)
        three_make = min(0.95, offence.three_fg_pct * efg_ratio)

        # Offensive rebound probability after a miss
        orb_prob = self._blend_rate(offence.orb_pct, 1.0 - defence.def_drb_pct, D1_ORB)

        return {
            "to_prob": to_prob,
            "ft_prob": ft_prob,
            "shot_prob": shot_prob,
            "rim_prob": rim_prob,
            "mid_prob": mid_prob,
            "three_prob": three_prob,
            "rim_make": rim_make,
            "mid_make": mid_make,
            "three_make": three_make,
            "orb_prob": orb_prob,
            "ft_make_prob": 0.72,  # D1 average FT%
        }

    def _simulate_ft_trip(self, probs: Dict[str, float], rng: np.random.Generator) -> int:
        """
        Simulate a free-throw trip using the empirical NCAA trip-type distribution.

        Trip types and their rules:
            AND1        (12%) — FG already made; 1 bonus FT → 2 or 3 pts total.
            TWO_SHOT    (65%) — Standard shooting foul; 2 independent FTs.
            THREE_SHOT  ( 8%) — Fouled on 3PA; 3 independent FTs.
            ONE_AND_ONE (15%) — 1-and-1 bonus; second FT only if first is made.

        Returns points scored (1, 2, or 3 for non-AND1; 0–3 for AND1 and others).
        """
        ft_p = probs["ft_make_prob"]
        trip = int(rng.choice(4, p=_FT_TRIP_WEIGHTS))

        if trip == _FT_TRIP_AND1:
            # Field goal was already made (2 pts); add 1 bonus FT.
            return 2 + (1 if rng.random() < ft_p else 0)

        if trip == _FT_TRIP_TWO_SHOT:
            return (1 if rng.random() < ft_p else 0) + (1 if rng.random() < ft_p else 0)

        if trip == _FT_TRIP_THREE_SHOT:
            return sum(1 for _ in range(3) if rng.random() < ft_p)

        # ONE_AND_ONE: first FT is the gate; second only if first is made.
        if rng.random() < ft_p:
            return 1 + (1 if rng.random() < ft_p else 0)
        return 0

    def _attempt_shot(self, probs: Dict[str, float], rng: np.random.Generator) -> Optional[int]:
        """
        Simulate a single shot attempt.

        Returns:
            2 or 3 if the shot is made, None if missed.
        """
        shot_r = rng.random()
        if shot_r < probs["rim_prob"]:
            return 2 if rng.random() < probs["rim_make"] else None
        elif shot_r < probs["rim_prob"] + probs["mid_prob"]:
            return 2 if rng.random() < probs["mid_make"] else None
        else:
            return 3 if rng.random() < probs["three_make"] else None

    def _simulate_possession(self, probs: Dict[str, float], rng: np.random.Generator) -> int:
        """
        Simulate a single possession using the Markov chain.

        Returns points scored (0, 1, 2, or 3).

        OREB fix
        --------
        The original code returned 0 after exhausting all OREB cycles, which
        silently discards the expected value of the possession at that point.
        After ``max_rebounds`` consecutive offensive rebounds, the team still
        has the ball and must shoot.  The fix: the inner loop exits via
        ``continue`` on every OREB; if the range is fully exhausted, we fall
        through to a mandatory final shot evaluation.

        This prevents a downward bias of approximately:
            ORB^max_rebounds × shot_make_prob × pts_per_made_shot
        per simulation (negligible per possession, material across 680 k calls).
        """
        max_rebounds = 3  # Cap OREB cycles to prevent infinite loops

        # Pre-compute once per possession; avoids redundant division inside loop.
        ft_relative = probs["ft_prob"] / (probs["ft_prob"] + probs["shot_prob"])

        for _ in range(max_rebounds + 1):
            # Branch 1: turnover — possession ends with 0 pts.
            if rng.random() < probs["to_prob"]:
                return 0

            # Branch 2: free throw trip — weighted trip-type distribution.
            if rng.random() < ft_relative:
                return self._simulate_ft_trip(probs, rng)

            # Branch 3: shot attempt.
            result = self._attempt_shot(probs, rng)
            if result is not None:
                return result

            # Miss → check for offensive rebound.
            if rng.random() < probs["orb_prob"]:
                continue  # OREB: re-enter possession (loop advances)
            return 0     # Defensive rebound: possession ends.

        # ---------------------------------------------------------------
        # Exhausted OREB budget (max_rebounds consecutive OREBs, all with
        # missed shots).  The team still has the ball — force a final shot
        # to preserve the EV of the possession.
        # ---------------------------------------------------------------
        result = self._attempt_shot(probs, rng)
        return result if result is not None else 0

    def simulate_game(
        self,
        home: TeamSimProfile,
        away: TeamSimProfile,
        n_sims: int = 10000,
        is_neutral: bool = False,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Simulate a full game ``n_sims`` times.

        Each simulation plays ~68 possessions per team (based on blended
        pace) and accumulates points through the Markov chain.

        Home advantage is modelled as a small boost to home eFG% and
        a small penalty to away TO%.
        """
        rng = np.random.default_rng(seed)

        # Determine pace (average of both teams, tempo-free neutral)
        game_pace = (home.pace + away.pace) / 2.0
        # Possessions per half = pace / 2 (since pace is per 40 min)
        poss_per_half = int(round(game_pace / 2.0))

        # Apply home-court advantage (if not neutral)
        home_off = TeamSimProfile(**{
            k: getattr(home, k) for k in home.__dataclass_fields__
        })
        away_off = TeamSimProfile(**{
            k: getattr(away, k) for k in away.__dataclass_fields__
        })

        if not is_neutral:
            # Pace-adjusted home-court advantage:
            # Scale HCA by (game_pace / 68.0) to account for tempo variance.
            # High-pace games have more possessions → larger HCA impact.
            pace_ratio = game_pace / 68.0
            adjusted_hca_pts = self.home_advantage_pts * pace_ratio
            ha_factor = adjusted_hca_pts / 100.0  # Convert to % boost

            # Home team gets slight offensive boost, away gets slight penalty
            home_off.efg_pct = min(0.65, home.efg_pct + ha_factor)
            home_off.to_pct = max(0.08, home.to_pct - ha_factor * 0.3)
            away_off.efg_pct = max(0.35, away.efg_pct - ha_factor * 0.5)
            away_off.to_pct = min(0.30, away.to_pct + ha_factor * 0.3)

        # Compute possession probabilities
        home_probs = self._compute_possession_probs(home_off, away)
        away_probs = self._compute_possession_probs(away_off, home)

        home_scores = np.zeros(n_sims, dtype=np.int32)
        away_scores = np.zeros(n_sims, dtype=np.int32)
        home_1h = np.zeros(n_sims, dtype=np.int32)
        away_1h = np.zeros(n_sims, dtype=np.int32)

        for sim in range(n_sims):
            h_pts = 0
            a_pts = 0
            h_1h = 0
            a_1h = 0

            for half in range(2):
                # Add slight randomness to possession count (+/- 3)
                n_poss = poss_per_half + rng.integers(-3, 4)

                for _ in range(n_poss):
                    h_pts += self._simulate_possession(home_probs, rng)
                    a_pts += self._simulate_possession(away_probs, rng)

                if half == 0:
                    h_1h = h_pts
                    a_1h = a_pts

            home_scores[sim] = h_pts
            away_scores[sim] = a_pts
            home_1h[sim] = h_1h
            away_1h[sim] = a_1h

        return SimulationResult(
            n_sims=n_sims,
            home_scores=home_scores,
            away_scores=away_scores,
            home_1h_scores=home_1h,
            away_1h_scores=away_1h,
        )

    def simulate_spread_edge(
        self,
        home: TeamSimProfile,
        away: TeamSimProfile,
        spread: float,
        spread_odds: float = -110,
        n_sims: int = 10000,
        is_neutral: bool = False,
        matchup_margin_adj: float = 0.0,
    ) -> Dict:
        """
        Run simulation and compute edge against a given spread.

        Returns a dict with win/loss/push probabilities (with CI bounds),
        projected margin, edge, and Kelly fraction.

        **Push handling**: Because Markov generates discrete integer scores,
        exact ties can occur for integer spreads. The Kelly calculation
        accounts for pushes: f = (p_win * (b - 1) - p_loss) / (b - 1).

        **matchup_margin_adj**: Additive margin shift from MatchupEngine
        (positive = favours home). Applied by shifting the effective spread:
        P(margin + adj + spread > 0) ≡ P(margin + effective_spread > 0)
        where effective_spread = spread + matchup_margin_adj.
        This is mathematically equivalent to shifting every simulated margin
        by +matchup_margin_adj before computing cover probabilities.
        """
        result = self.simulate_game(home, away, n_sims, is_neutral)

        # Apply matchup geometry: shifting margins by +adj is equivalent to
        # using an effective spread = spread + adj for all cover-prob checks.
        effective_spread = spread + matchup_margin_adj

        # Get full win/loss/push breakdown with CI bounds
        full_probs = result.ci_cover_probs_full(effective_spread, confidence=0.95)
        win_prob = full_probs['win']
        loss_prob = full_probs['loss']
        push_prob = full_probs['push']

        # Also get simple cover prob + CI for backward compatibility
        cover_prob, cover_lower, cover_upper = result.ci_cover_prob(effective_spread, confidence=0.95)

        # Convert spread odds to implied prob (no-vig)
        if spread_odds > 0:
            implied = 100.0 / (spread_odds + 100.0)
        else:
            implied = abs(spread_odds) / (abs(spread_odds) + 100.0)

        # Push-adjusted edge calculation
        # The market-implied probability assumes a binary outcome (win or loss),
        # but Markov generates discrete scores with possible pushes. We adjust
        # the implied probability to account for the action space:
        #   adjusted_implied = implied * (1.0 - push_prob)
        # This prevents false negatives on integer spreads where push probability
        # is material (e.g., 8-12% for common NBA/CBB spreads).
        adjusted_implied = implied * (1.0 - push_prob)
        edge = win_prob - adjusted_implied

        # Kelly with push handling
        # Standard formula: f = (p * b - q) / b  where p=win, q=loss, b=decimal_odds-1
        # With pushes: f = (p_win * (b - 1) - p_loss) / (b - 1)
        # This is equivalent to f = (p_win * b - 1) / b  when pushes are 0
        if spread_odds > 0:
            decimal_odds = (spread_odds / 100.0) + 1.0
        else:
            decimal_odds = (100.0 / abs(spread_odds)) + 1.0

        b = decimal_odds - 1.0  # Net profit per dollar risked
        if b > 0 and edge > 0:
            # Kelly with explicit push handling
            kelly = (win_prob * b - loss_prob) / b
            kelly = max(0.0, min(kelly, 0.20))
        else:
            kelly = 0.0

        return {
            "sim_result": result.to_dict(),
            "spread": spread,
            "cover_prob": round(cover_prob, 4),
            "cover_lower": round(cover_lower, 4),
            "cover_upper": round(cover_upper, 4),
            "win_prob": round(win_prob, 4),
            "loss_prob": round(loss_prob, 4),
            "push_prob": round(push_prob, 4),
            "implied_prob": round(implied, 4),
            "edge": round(edge, 4),
            "kelly_full": round(kelly, 4),
            "total_over_prob": round(result.total_over_prob(result.projected_total), 4),
        }


# ---------------------------------------------------------------------------
# Four Factors calculator
# ---------------------------------------------------------------------------

def estimate_possessions(fga: int, oreb: int, to: int, fta: int) -> float:
    """
    Estimate possessions from box-score stats.

    Formula: Poss ≈ FGA - OREB + TO + 0.475 * FTA
    """
    return fga - oreb + to + 0.475 * fta


def calculate_four_factors(
    fgm: int, fga: int, fgm3: int, fga3: int,
    fta: int, ftm: int, oreb: int, to: int,
) -> Dict[str, float]:
    """
    Calculate tempo-free Four Factors from box-score totals.

    Returns dict with efg_pct, to_pct, orb_pct, ft_rate.
    """
    poss = estimate_possessions(fga, oreb, to, fta)
    poss = max(poss, 1.0)

    efg = (fgm + 0.5 * fgm3) / max(fga, 1) if fga > 0 else 0.0
    to_rate = to / poss
    ft_rate = fta / max(fga, 1)

    return {
        "efg_pct": round(efg, 4),
        "to_pct": round(to_rate, 4),
        "orb_pct": 0.28,  # ORB% requires opponent DRB — use default if not available
        "ft_rate": round(ft_rate, 4),
        "possessions": round(poss, 1),
    }


# ---------------------------------------------------------------------------
# Profile builder (from CBBpy or box-score data)
# ---------------------------------------------------------------------------

def build_sim_profile(
    team: str,
    four_factors: Dict[str, float],
    shot_dist: Optional[Dict[str, float]] = None,
    pace: float = 68.0,
) -> TeamSimProfile:
    """
    Build a TeamSimProfile from Four Factors and optional shot distribution.

    This is the bridge between raw data (CBBpy / box scores) and the
    simulation engine.
    """
    profile = TeamSimProfile(team=team)

    profile.efg_pct = four_factors.get("efg_pct", 0.500)
    profile.to_pct = four_factors.get("to_pct", 0.170)
    profile.orb_pct = four_factors.get("orb_pct", 0.280)
    profile.ft_rate = four_factors.get("ft_rate", 0.300)
    profile.pace = pace

    if shot_dist:
        profile.rim_rate = shot_dist.get("rim_rate", 0.32)
        profile.mid_rate = shot_dist.get("mid_rate", 0.15)
        profile.three_rate = shot_dist.get("three_rate", 0.36)
        profile.rim_fg_pct = shot_dist.get("rim_fg_pct", 0.62)
        profile.mid_fg_pct = shot_dist.get("mid_fg_pct", 0.38)
        profile.three_fg_pct = shot_dist.get("three_fg_pct", 0.34)

    return profile
