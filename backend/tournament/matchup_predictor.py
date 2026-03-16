"""
Tournament game-level prediction engine.

Uses V9.1 composite ratings with tournament-specific adjustments:
1. Per-round SD multipliers (R64: 1.12x → Championship: 1.0x)
2. Seed-matchup-aware historical blend — 5-year rolling rates (35% R64, 15% R32)
3. Style-based variance (pace mismatch, high 3PT rate adds chaos)
4. Tournament experience adjustment (returning player minutes %)
5. Recent form factor (March form over last 10 games, capped ±2 pts)
6. Composite rating (55% KenPom + 45% BartTorvik blend)

Does NOT modify existing betting_model.py (GUARDIAN active until Apr 7).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Round-specific SD multipliers — R64 chaos fades to championship clarity
ROUND_SD_MULTIPLIERS: Dict[int, float] = {
    0: 1.15,  # First Four (play-in chaos)
    1: 1.12,  # Round of 64 (highest upset rate)
    2: 1.08,  # Round of 32
    3: 1.05,  # Sweet 16
    4: 1.02,  # Elite 8
    5: 1.00,  # Final Four
    6: 1.00,  # Championship
}

# Historical upset rates by (higher_seed, lower_seed) for R64.
# Updated to 5-year rolling window (2020-2024) to reflect modern 3-point era.
# Key changes: 5v12 lower (was 35.2%), 6v11 lower (play-in adds quality 11s),
# 7v10 higher (mid-majors more competitive), 1v16 higher (post-UMBC era).
SEED_UPSET_RATES: Dict[Tuple[int, int], float] = {
    (1, 16): 0.010,   # 1.0% (5yr: UMBC 2018 lone loss; recent 100%)
    (2, 15): 0.070,   # 7.0% (5yr avg; 2 losses in last 5 years)
    (3, 14): 0.160,   # 16.0% (5yr: Furman/Princeton 2023 both won)
    (4, 13): 0.220,   # 22.0% (5yr avg; slightly above all-time)
    (5, 12): 0.370,   # 37.0% (5yr: down from all-time 35.2%; 3-pt era)
    (6, 11): 0.370,   # 37.0% (5yr: play-in elevates 11-seed quality)
    (7, 10): 0.420,   # 42.0% (5yr: mid-majors closing the gap)
    (8, 9):  0.490,   # 49.0% (true coin flip; nearly 50/50 all-time)
}

# Upset boost — additional variance for Cinderella potential
# Lower seeds get a small rating boost in simulations to allow for chaos
CINDERELLA_BOOST: Dict[int, float] = {
    12: 1.5,   # 12-seeds play like 10.5 seeds
    11: 1.0,   # 11-seeds play like 10 seeds  
    10: 0.5,   # 10-seeds play like 9.5 seeds
    13: 2.0,   # 13-seeds play like 11 seeds
    14: 2.5,   # 14-seeds play like 11.5 seeds
    15: 3.0,   # 15-seeds play like 12 seeds
}

# Style-based variance multipliers
PACE_MISMATCH_THRESHOLD = 10.0   # 10+ possession difference = chaos
HIGH_3PT_THRESHOLD = 0.40        # 40%+ 3PT rate = high variance
STYLE_VARIANCE_BOOST = 0.03      # +3% SD for style mismatches

# Recent form (March performance) adjustment cap
RECENT_FORM_CAP = 2.0            # ±2.0 points max

# V9.1 base SD
BASE_SD = 11.0


@dataclass
class TournamentTeam:
    """Team data for tournament bracket simulation."""
    name: str
    seed: int
    region: str
    
    # Composite rating: use provided value OR auto-calculate from KenPom + BartTorvik
    composite_rating: Optional[float] = None  # If None, will auto-calculate
    
    # Raw source ratings
    kp_adj_em: Optional[float] = None      # KenPom AdjEM
    bt_adj_em: Optional[float] = None      # BartTorvik AdjEM
    
    # Style profile (from BartTorvik / TeamProfile DB)
    pace: float = 68.0                     # Possessions per 40 min
    three_pt_rate: float = 0.35            # 3PA / FGA (0-1 scale)
    def_efg_pct: float = 0.50              # Opponent eFG% allowed
    conference: str = ""
    
    # Tournament experience (returning player minutes % from last tournament)
    tournament_exp: float = 0.70
    
    # Recent form (last 10 games AdjEM performance vs season average)
    # Positive = hot team, Negative = cold team
    recent_form: float = 0.0               # Capped at ±2.0 pts in calculations
    
    # Post-init to compute composite rating if not provided
    def __post_init__(self):
        # If composite_rating not provided, calculate from KP + BT
        if self.composite_rating is None:
            if self.kp_adj_em is not None and self.bt_adj_em is not None:
                self.composite_rating = 0.55 * self.kp_adj_em + 0.45 * self.bt_adj_em
            elif self.kp_adj_em is not None:
                self.composite_rating = self.kp_adj_em
            elif self.bt_adj_em is not None:
                self.composite_rating = self.bt_adj_em
            else:
                # Fallback: estimate from seed (1-seed ≈ +25, 16-seed ≈ -15)
                self.composite_rating = 25.0 - (self.seed - 1) * 2.5
                logger.warning(
                    f"{self.name}: No ratings provided, estimating from seed {self.seed}"
                )


def predict_game(
    team_a: TournamentTeam,
    team_b: TournamentTeam,
    round_num: int,
    is_neutral: bool = True,
) -> Tuple[float, float, float]:
    """
    Predict outcome of a single tournament game with all intelligence upgrades.

    Args:
        team_a: First team (perspective for win_prob and margin)
        team_b: Second team
        round_num: Tournament round (0=First Four, 1=R64, ..., 6=Championship)
        is_neutral: True for all tournament games (neutral site)

    Returns:
        Tuple of (win_prob_a, margin_pred, effective_sd)
    """
    # ============================================================
    # 1. BASE MARGIN from composite ratings
    # ============================================================
    margin = team_a.composite_rating - team_b.composite_rating
    
    # ============================================================
    # 2. CINDERELLA BOOST — lower seeds get upset potential
    # ============================================================
    # This adds "March Magic" — lower seeds play above their rating
    cinderella_a = CINDERELLA_BOOST.get(team_a.seed, 0.0)
    cinderella_b = CINDERELLA_BOOST.get(team_b.seed, 0.0)
    margin += (cinderella_a - cinderella_b)
    
    # ============================================================
    # 3. RECENT FORM ADJUSTMENT (March performance, capped ±2 pts)
    # ============================================================
    form_adj = (team_a.recent_form - team_b.recent_form) * 0.6
    form_adj = max(-RECENT_FORM_CAP, min(RECENT_FORM_CAP, form_adj))
    margin += form_adj
    
    # ============================================================
    # 4. TOURNAMENT EXPERIENCE ADJUSTMENT (capped ±1.5 pts)
    # ============================================================
    exp_adj = (team_a.tournament_exp - team_b.tournament_exp) * 1.5
    exp_adj = max(-1.5, min(1.5, exp_adj))
    margin += exp_adj
    
    # ============================================================
    # 4. ROUND-SPECIFIC SD MULTIPLIER
    # ============================================================
    sd_mult = ROUND_SD_MULTIPLIERS.get(round_num, 1.0)
    effective_sd = BASE_SD * sd_mult
    
    # ============================================================
    # 5. STYLE-BASED VARIANCE ADJUSTMENT
    # ============================================================
    # Pace mismatch adds chaos (fast vs slow = unpredictable)
    pace_diff = abs(team_a.pace - team_b.pace)
    if pace_diff >= PACE_MISMATCH_THRESHOLD:
        # High pace mismatch = higher variance
        effective_sd *= (1.0 + STYLE_VARIANCE_BOOST)
    
    # High 3PT rate teams = higher variance (live by the three, die by the three)
    high_3pt_count = sum([
        1 if team_a.three_pt_rate >= HIGH_3PT_THRESHOLD else 0,
        1 if team_b.three_pt_rate >= HIGH_3PT_THRESHOLD else 0
    ])
    if high_3pt_count >= 1:
        # Each high-3PT team adds variance
        effective_sd *= (1.0 + STYLE_VARIANCE_BOOST * high_3pt_count)
    
    # ============================================================
    # 6. BASE WIN PROBABILITY (logistic)
    # ============================================================
    model_prob = 1.0 / (1.0 + math.exp(-margin / effective_sd))
    
    # ============================================================
    # 7. SEED-MATCHUP HISTORICAL BLEND (R64 and R32 only)
    # ============================================================
    if round_num <= 2 and team_a.seed is not None and team_b.seed is not None:
        model_prob = _blend_with_seed_history_v2(
            model_prob, team_a.seed, team_b.seed, round_num
        )
    
    return model_prob, margin, effective_sd


def _blend_with_seed_history_v2(
    model_prob: float, seed_a: int, seed_b: int, round_num: int
) -> float:
    """
    Blend V9.1 win probability with historical seed-matchup upset rates.

    Weight schedule (aligned with backend/services/bracket_simulator.py
    recalibration — 5-year rolling historical data, survivor-bias aware):

    - R64: 35% history — AdjEM model is the primary signal
    - R32: 15% history — survivor bias begins; seeds less predictive
    - S16:  5% history — minimal; model drives
    - E8+:  0% history — pure model (survivor bias dominates)

    Old "MAXIMUM CHAOS" weights (75%/50%) caused the simulator to almost
    ignore AdjEM entirely, making every 5v12 game look like a coin flip
    regardless of the actual talent gap.
    """
    higher_seed = min(seed_a, seed_b)
    lower_seed = max(seed_a, seed_b)

    hist_upset_rate = SEED_UPSET_RATES.get((higher_seed, lower_seed))
    if hist_upset_rate is None:
        return model_prob

    # Historical probability that team_a wins
    if seed_a < seed_b:
        hist_prob = 1.0 - hist_upset_rate  # team_a is favorite
    else:
        hist_prob = hist_upset_rate  # team_a is underdog

    if round_num == 1:       # R64: 35% history, 65% model
        history_weight = 0.35
    elif round_num == 2:     # R32: 15% history, 85% model
        history_weight = 0.15
    elif round_num == 3:     # S16: 5% history, 95% model
        history_weight = 0.05
    else:                    # E8, F4, Championship: pure model
        history_weight = 0.00

    blended = (1.0 - history_weight) * model_prob + history_weight * hist_prob

    logger.debug(
        "Seed blend %d vs %d (R%d): model=%.3f hist=%.3f blended=%.3f (w_hist=%.2f)",
        seed_a, seed_b, round_num, model_prob, hist_prob, blended, history_weight
    )
    return blended
