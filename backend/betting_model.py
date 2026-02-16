"""
Version 7 CBB Betting Framework - Production Implementation
Implements all fixes from the final audit:
- 2-layer Monte Carlo CI
- Penalty budget with ceiling
- Safe Kelly calculation
- Proper vig removal
- Conservative decision threshold
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass, asdict
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
    point_prob: float
    lower_ci_prob: float
    upper_ci_prob: float
    
    # Edges
    edge_point: float
    edge_conservative: float
    
    # Betting
    kelly_full: float
    kelly_fractional: float
    recommended_units: float
    
    # Metadata
    data_freshness_tier: str
    penalties_applied: Dict
    notes: List[str]
    
    # Full details for storage
    full_analysis: Dict


class CBBEdgeModel:
    """
    Production betting model implementing Version 7 framework
    Conservative, transparent, and designed to PASS 85-95% of games
    """
    
    def __init__(
        self,
        base_sd: float = 11.0,
        weights: Optional[Dict[str, float]] = None,
        home_advantage: float = 3.09,
        max_kelly: float = 0.20,
        fractional_kelly_divisor: float = 2.0,
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
    
    def monte_carlo_prob_ci(
        self,
        projected_margin: float,
        adjusted_sd: float,
        n_samples: int = 10000
    ) -> Tuple[float, float, float]:
        """
        Two-layer Monte Carlo for win probability with confidence interval
        
        Layer 1: Parameter uncertainty in margin estimate (±10%)
        Layer 2: Outcome variance around realized margin
        
        Returns: (point_estimate, lower_95_ci, upper_95_ci)
        """
        # Layer 1: Uncertainty in our margin projection itself
        margin_se = abs(projected_margin) * 0.10  # 10% parameter uncertainty
        margin_se = max(margin_se, 1.5)  # Floor at 1.5 points
        
        margin_samples = np.random.normal(
            projected_margin,
            margin_se,
            n_samples
        )
        
        # Layer 2: Convert margins to probabilities using adjusted SD
        # This represents outcome variance for each projected margin
        prob_samples = norm.cdf(margin_samples / adjusted_sd)
        
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

    # def remove_vig_american(
    #     self,
    #     fav_odds: float,
    #     dog_odds: float
    # ) -> Tuple[float, float]:
    #     """
    #     Remove vig from American odds to get true probabilities
        
    #     Returns: (fav_true_prob, dog_true_prob)
    #     """
    #     # Convert to implied probabilities (include vig)
    #     fav_implied = abs(fav_odds) / (abs(fav_odds) + 100)
    #     dog_implied = 100 / (dog_odds + 100)
        
    #     # Total (should be >1, the vig)
    #     total = fav_implied + dog_implied
        
    #     # Normalize to 100%
    #     fav_true = fav_implied / total
    #     dog_true = dog_implied / total
        
    #     return fav_true, dog_true
    
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
    
    def adjusted_sd(
        self,
        penalties_dict: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate adjusted SD with penalty budget and ceiling
        
        Formula: SD_adj = base_SD * (1 + min(sqrt(sum(penalty²)), 6) / 15)
        Ceiling: Never exceed 15.5
        
        Returns: adjusted SD in points
        """
        if penalties_dict is None:
            return self.base_sd
        
        # Sqrt-sum (diminishing returns on penalty stacking)
        total_penalty = np.sqrt(sum(v**2 for v in penalties_dict.values()))
        
        # Cap penalty at 6 (prevents runaway)
        total_penalty = min(total_penalty, 6.0)
        
        # Calculate adjusted SD
        adj_sd = self.base_sd * (1 + total_penalty / 15)
        
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
        data_freshness: Optional[Dict] = None
    ) -> GameAnalysis:
        """
        Complete game analysis using Version 7 framework
        
        Args:
            game_data: {home_team, away_team, is_neutral, etc.}
            odds: {spread, total, moneyline, etc.}
            ratings: {kenpom: {home: X, away: Y}, barttorvik: {...}, ...}
            injuries: [  {team, player, impact_tier}]
            data_freshness: {lines_age_min, ratings_age_hours}
        
        Returns: GameAnalysis dataclass with full verdict
        """
        notes = []
        penalties = {}
        
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
                    projected_margin=0, adjusted_sd=self.base_sd,
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
                    projected_margin=0, adjusted_sd=self.base_sd,
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
                projected_margin=0, adjusted_sd=self.base_sd,
                home_advantage=self.home_advantage,
                point_prob=0.5, lower_ci_prob=0.5, upper_ci_prob=0.5,
                edge_point=0, edge_conservative=0,
                kelly_full=0, kelly_fractional=0, recommended_units=0,
                data_freshness_tier="N/A",
                penalties_applied={},
                notes=["Required ratings unavailable"],
                full_analysis={}
            )
        
        # Weighted margin calculation
        margin = 0
        if kp_home and kp_away:
            margin += self.weights['kenpom'] * (kp_home - kp_away)
        if bt_home and bt_away:
            margin += self.weights['barttorvik'] * (bt_home - bt_away)
        else:
            penalties['missing_barttorvik'] = 1.0
            notes.append("BartTorvik unavailable - using KenPom only")
        
        if em_home and em_away:
            margin += self.weights['evanmiya'] * (em_home - em_away)
        else:
            penalties['missing_evanmiya'] = 0.8
        
        # Home advantage
        if not game_data.get('is_neutral', False):
            margin += self.home_advantage
        
        # Injury adjustments (simplified tier system)
        injury_adj = 0
        if injuries:
            for inj in injuries:
                if inj.get('team') == game_data['home_team']:
                    multiplier = 1
                else:
                    multiplier = -1
                
                tier = inj.get('impact_tier', 'bench')
                if tier == 'star':
                    injury_adj += multiplier * -2.5  # Star out hurts
                    penalties['star_injury'] = 2.5
                    notes.append(f"{inj.get('player')} out (star) - high uncertainty")
                elif tier == 'role':
                    injury_adj += multiplier * -1.0
                    penalties['role_injury'] = 1.5
        
        margin += injury_adj
        
        # ================================================================
        # STEP 2: ADJUST SD FOR UNCERTAINTY
        # ================================================================
        
        # Heteroskedasticity adjustments
        rating_gap = abs(kp_home - kp_away) if kp_home and kp_away else 0
        if rating_gap > 25:
            penalties['large_gap'] = 1.0
            notes.append(f"Large rating gap ({rating_gap:.1f}) - blowout variance")
        
        # Compute adjusted SD
        adj_sd = self.adjusted_sd(penalties)
        
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
        # STEP 3: MONTE CARLO PROBABILITY + CI
        # ================================================================
        
        point_prob, lower_ci, upper_ci = self.monte_carlo_prob_ci(margin, adj_sd)
        
        # ================================================================
        # STEP 4: VIG REMOVAL & EDGE CALCULATION
        # ================================================================
        
        # Get market odds (assuming we're analyzing spread bet)
        spread = odds.get('spread', 0)
        spread_odds = odds.get('spread_odds', -110)
        
        # No-vig probability
        fav_novig, dog_novig = self.remove_vig_american(spread_odds, spread_odds)
        market_prob = fav_novig if margin > 0 else dog_novig
        
        # Edge calculations
        edge_point = point_prob - market_prob
        edge_conservative = lower_ci - market_prob  # Decision threshold
        
        # ================================================================
        # STEP 5: KELLY & BET SIZING
        # ================================================================
        
        decimal_odds = self.american_to_decimal(spread_odds)
        kelly_full = self.kelly_fraction(point_prob, decimal_odds)
        kelly_frac = kelly_full / self.fractional_kelly_divisor
        
        # Decision rule: only bet if conservative edge > 0
        if edge_conservative <= 0:
            verdict = "PASS"
            pass_reason = f"Conservative edge {edge_conservative:.3%} ≤ 0"
            recommended_units = 0
        else:
            # Additional safety: cap at 1.5% of bankroll
            recommended_pct = min(kelly_frac * 100, 1.5)
            recommended_units = recommended_pct
            
            if recommended_units < 0.25:
                verdict = f"Thin edge - max 0.25u only"
            else:
                verdict = f"Bet {recommended_units:.2f}u @ {spread_odds:+.0f}"
            
            pass_reason = None
        
        # ================================================================
        # RETURN COMPLETE ANALYSIS
        # ================================================================
        
        full_analysis_dict = {
            'model_version': 'v7.0',
            'timestamp': datetime.utcnow().isoformat(),
            'inputs': {
                'ratings': ratings,
                'margin_components': {
                    'kenpom': self.weights['kenpom'] * (kp_home - kp_away) if kp_home and kp_away else None,
                    'barttorvik': self.weights['barttorvik'] * (bt_home - bt_away) if bt_home and bt_away else None,
                    'evanmiya': self.weights['evanmiya'] * (em_home - em_away) if em_home and em_away else None,
                    'home_adv': self.home_advantage,
                    'injury_adj': injury_adj,
                },
                'odds': odds,
                'injuries': injuries,
            },
            'calculations': {
                'projected_margin': margin,
                'base_sd': self.base_sd,
                'adjusted_sd': adj_sd,
                'penalties': penalties,
                'point_prob': point_prob,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci,
                'market_prob': market_prob,
                'edge_point': edge_point,
                'edge_conservative': edge_conservative,
                'kelly_full': kelly_full,
                'kelly_fractional': kelly_frac,
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
