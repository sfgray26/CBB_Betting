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
    def_efg_pct: float = 0.480       # Opponent eFG% allowed
    def_to_forced_pct: float = 0.190  # Turnovers forced rate
    def_drb_pct: float = 0.720       # Defensive rebound rate (= 1 - opponent ORB)
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
        """Probability home team covers the given spread (e.g. -4.5)."""
        margins = self.home_scores - self.away_scores
        return float(np.mean(margins + spread > 0))

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
        Blend offensive tendency with defensive resistance.

        Uses the "opponent-adjusted" formula common in tempo-free analytics:
            adjusted = off_rate + def_rate - d1_avg

        Clamped to [0.01, 0.99] for safety.
        """
        blended = off_rate + def_rate - d1_avg
        return max(0.01, min(0.99, blended))

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

        # Turnover probability (blend offence TO rate with defence forced-TO rate)
        to_prob = self._blend_rate(offence.to_pct, defence.def_to_forced_pct, D1_TO)

        # Free throw trip probability
        ft_prob = self._blend_rate(offence.ft_rate, defence.def_ft_rate_allowed, D1_FTR)
        # FT trips are a fraction of non-TO possessions, scale down
        ft_prob = ft_prob * 0.35  # ~35% of FTA/FGA converts to actual FT possessions

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

    def _simulate_possession(self, probs: Dict[str, float], rng: np.random.Generator) -> int:
        """
        Simulate a single possession using the Markov chain.

        Returns points scored (0, 1, 2, or 3).
        """
        max_rebounds = 3  # Cap OREB cycles to prevent infinite loops

        for _ in range(max_rebounds + 1):
            # First branch: turnover?
            r = rng.random()
            if r < probs["to_prob"]:
                return 0

            r = rng.random()
            if r < probs["ft_prob"] / (probs["ft_prob"] + probs["shot_prob"]):
                # Free throw trip (assume 2 FTs, each with ft_make_prob)
                pts = 0
                for _ in range(2):
                    if rng.random() < probs["ft_make_prob"]:
                        pts += 1
                return pts

            # Shot attempt — determine type
            shot_r = rng.random()
            if shot_r < probs["rim_prob"]:
                if rng.random() < probs["rim_make"]:
                    return 2
            elif shot_r < probs["rim_prob"] + probs["mid_prob"]:
                if rng.random() < probs["mid_make"]:
                    return 2
            else:
                if rng.random() < probs["three_make"]:
                    return 3

            # Miss — check for offensive rebound
            if rng.random() < probs["orb_prob"]:
                continue  # Re-enter possession
            else:
                return 0  # Defensive rebound

        return 0  # Exhausted OREB cycles

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
            # Home team gets slight offensive boost, away gets slight penalty
            ha_factor = self.home_advantage_pts / 100.0  # ~3% boost
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
    ) -> Dict:
        """
        Run simulation and compute edge against a given spread.

        Returns a dict with win/cover probabilities, projected margin,
        edge, and Kelly fraction.
        """
        result = self.simulate_game(home, away, n_sims, is_neutral)

        cover_prob = result.spread_cover_prob(spread)

        # Convert spread odds to implied prob (no-vig)
        if spread_odds > 0:
            implied = 100.0 / (spread_odds + 100.0)
        else:
            implied = abs(spread_odds) / (abs(spread_odds) + 100.0)

        edge = cover_prob - implied

        # Kelly
        if spread_odds > 0:
            decimal_odds = (spread_odds / 100.0) + 1.0
        else:
            decimal_odds = (100.0 / abs(spread_odds)) + 1.0

        if decimal_odds > 1.0 and edge > 0:
            kelly = (cover_prob * decimal_odds - 1.0) / (decimal_odds - 1.0)
            kelly = max(0.0, min(kelly, 0.20))
        else:
            kelly = 0.0

        return {
            "sim_result": result.to_dict(),
            "spread": spread,
            "cover_prob": round(cover_prob, 4),
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
