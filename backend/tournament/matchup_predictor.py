"""
Tournament game-level prediction engine.

Uses V9.1 composite ratings with tournament-specific adjustments:
- Round-specific SD multipliers (R64: 1.12x, champion: 1.0x) per K-1/BRACKET-001 research
- Historical seed-based priors blended 70-80% model / 20-30% historical
- Tournament experience adjustment (returning player minutes %)

Does NOT modify existing betting_model.py (GUARDIAN active until Apr 7).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Round-specific SD multipliers from K-1 Tournament Intelligence + BRACKET-001
# Single-elimination variance is 15-25% higher than regular season
ROUND_SD_MULTIPLIERS: Dict[int, float] = {
    0: 1.15,  # First Four (play-in chaos)
    1: 1.12,  # Round of 64 (highest upset rate)
    2: 1.08,  # Round of 32
    3: 1.05,  # Sweet 16
    4: 1.02,  # Elite 8
    5: 1.00,  # Final Four
    6: 1.00,  # Championship
}

# Historical upset rates by (higher_seed, lower_seed) for R64 — 2000-2024
# Source: K-1 report aggregating Action Network, VSiN, OddsShark data
SEED_UPSET_RATES: Dict[Tuple[int, int], float] = {
    (1, 16): 0.013,   # 1.3% upset rate
    (2, 15): 0.067,   # 6.7%
    (3, 14): 0.153,   # 15.3%
    (4, 13): 0.216,   # 21.6%
    (5, 12): 0.352,   # 35.2% — famous 12-5 upset zone
    (6, 11): 0.389,   # 38.9%
    (7, 10): 0.394,   # 39.4%
    (8, 9):  0.487,   # 48.7% — essentially a coin flip
}

# V9.1 base SD (matches betting_model.py BASE_SD)
BASE_SD = 11.0


@dataclass
class TournamentTeam:
    """Team data for tournament bracket simulation."""
    name: str
    seed: int
    region: str
    # V9.1 composite rating: weighted KenPom + BartTorvik (AdjEM scale)
    composite_rating: float
    # Raw source ratings (optional — used for diagnostics)
    kp_adj_em: Optional[float] = None
    bt_adj_em: Optional[float] = None
    # Style profile (from BartTorvik / TeamProfile DB table)
    pace: float = 68.0          # Possessions per 40 min
    three_pt_rate: float = 0.35  # 3PA / FGA
    def_efg_pct: float = 0.50   # Opponent eFG% allowed
    conference: str = ""
    # Fraction of minutes played by players who were on the team last tournament
    tournament_exp: float = 0.70


def predict_game(
    team_a: TournamentTeam,
    team_b: TournamentTeam,
    round_num: int,
    is_neutral: bool = True,
) -> Tuple[float, float, float]:
    """
    Predict outcome of a single tournament game.

    Args:
        team_a: First team (perspective for win_prob and margin)
        team_b: Second team
        round_num: Tournament round (0=First Four, 1=R64, ..., 6=Championship)
        is_neutral: True for all tournament games (all are neutral site)

    Returns:
        Tuple of (win_prob_a, margin_pred, effective_sd) where:
        - win_prob_a: probability that team_a wins (0.0 to 1.0)
        - margin_pred: expected scoring margin from team_a's perspective (positive = team_a wins)
        - effective_sd: standard deviation used in the logistic calculation
    """
    # Base margin from V9.1 composite ratings
    margin = team_a.composite_rating - team_b.composite_rating

    # Tournament experience adjustment — capped at +/-1.5 pts
    # Small effect: returning players reduce first-game nerves
    exp_adj = (team_a.tournament_exp - team_b.tournament_exp) * 1.5
    exp_adj = max(-1.5, min(1.5, exp_adj))
    margin += exp_adj

    # Round-specific SD: higher in early rounds (more chaos)
    sd_mult = ROUND_SD_MULTIPLIERS.get(round_num, 1.0)
    effective_sd = BASE_SD * sd_mult

    # Logistic win probability (same formula as CBBEdgeModel)
    model_prob = 1.0 / (1.0 + math.exp(-margin / effective_sd))

    # For R64 and R32: blend with historical seed-based priors
    if round_num <= 2 and team_a.seed is not None and team_b.seed is not None:
        model_prob = _blend_with_seed_history(model_prob, team_a.seed, team_b.seed)

    return model_prob, margin, effective_sd


def _blend_with_seed_history(
    model_prob: float, seed_a: int, seed_b: int
) -> float:
    """
    Blend V9.1 win probability with historical seed-matchup upset rates.

    Weights: 60-80% model, 20-40% historical — higher history weight for extreme mismatches.
    This prevents the model from ever assigning <1% to a 12-seed beating a 5-seed when
    the historical rate is 35%.

    Args:
        model_prob: V9.1 probability that team with seed_a wins
        seed_a: seed of team_a (1 = top seed)
        seed_b: seed of team_b

    Returns:
        Blended probability that team_a wins
    """
    higher_seed = min(seed_a, seed_b)
    lower_seed = max(seed_a, seed_b)
    seed_diff = lower_seed - higher_seed

    hist_upset_rate = SEED_UPSET_RATES.get((higher_seed, lower_seed))
    if hist_upset_rate is None:
        return model_prob

    # Historical probability that team_a wins
    if seed_a < seed_b:
        # team_a is the favorite
        hist_prob = 1.0 - hist_upset_rate
    else:
        # team_a is the underdog
        hist_prob = hist_upset_rate

    # Weight more toward history for extreme mismatches (1v16, 2v15)
    # These matchups have decades of reliable data dwarfing a single season sample
    if seed_diff >= 13:
        weight_model = 0.60
    elif seed_diff >= 10:
        weight_model = 0.70
    else:
        weight_model = 0.80

    blended = weight_model * model_prob + (1.0 - weight_model) * hist_prob
    logger.debug(
        "Seed blend %d vs %d: model=%.3f hist=%.3f blended=%.3f (w=%.2f)",
        seed_a, seed_b, model_prob, hist_prob, blended, weight_model
    )
    return blended
