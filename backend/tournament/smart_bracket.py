"""
Smart Bracket Generator — Sophisticated upset prediction using all V9.1 data.

Combines:
1. Monte Carlo simulation results (actual win probabilities)
2. Style-based matchup analysis (pace, 3PT, defense)
3. Cinderella potential factors (tourney exp, recent form, seed momentum)
4. Historical seed upset rates (as baseline)
"""

import json
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.tournament.matchup_predictor import TournamentTeam, predict_game


@dataclass
class UpsetFactors:
    """Factors that increase/decrease upset likelihood."""
    base_upset_prob: float  # From historical seed rates
    model_upset_prob: float  # From V9.1 composite ratings
    sim_upset_prob: float  # From Monte Carlo results
    
    # Style factors
    pace_mismatch_boost: float  # High pace diff = chaos
    three_pt_boost: float  # High 3PT teams = variance
    defensive_matchup_boost: float  # Defensive mismatches
    
    # Cinderella factors
    tourney_exp_boost: float  # Underdog has more tourney experience
    recent_form_boost: float  # Underdog is "hot"
    momentum_boost: float  # Lower seed playing above rating
    
    # Composite
    final_upset_prob: float


class SmartBracketGenerator:
    """
    Generate brackets using sophisticated upset prediction.
    """
    
    # Historical seed upset rates (baseline)
    SEED_UPSET_RATES = {
        (1, 16): 0.013, (2, 15): 0.067, (3, 14): 0.153, (4, 13): 0.216,
        (5, 12): 0.352, (6, 11): 0.389, (7, 10): 0.394, (8, 9): 0.487,
    }
    
    # Style factor weights
    PACE_MISMATCH_THRESHOLD = 10.0
    HIGH_3PT_THRESHOLD = 0.40
    DEF_EFG_MISMATCH_THRESHOLD = 0.04
    
    def __init__(self, sim_results_path: Optional[str] = None, chaos_level: float = 0.5):
        """
        Args:
            sim_results_path: Path to sim_results.json from Monte Carlo
            chaos_level: 0.0 = chalk, 1.0 = maximum chaos, 0.5 = balanced
        """
        self.chaos_level = chaos_level
        self.sim_results = self._load_sim_results(sim_results_path)
    
    def _load_sim_results(self, path: Optional[str]) -> Dict:
        """Load Monte Carlo simulation results."""
        if not path:
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    
    def calculate_upset_factors(
        self,
        favorite: TournamentTeam,
        underdog: TournamentTeam,
        round_num: int,
        region: str = ""
    ) -> UpsetFactors:
        """
        Calculate all factors that influence upset probability.
        """
        # 1. Base historical rate
        higher_seed = min(favorite.seed, underdog.seed)
        lower_seed = max(favorite.seed, underdog.seed)
        base_upset = self.SEED_UPSET_RATES.get((higher_seed, lower_seed), 0.25)
        
        # 2. Model probability from V9.1 ratings
        fav_prob, margin, sd = predict_game(favorite, underdog, round_num)
        model_upset = 1.0 - fav_prob
        
        # 3. Simulation probability (if available)
        sim_upset = self._get_sim_upset_prob(favorite.name, underdog.name, round_num)
        
        # 4. Style-based factors
        pace_boost = self._calculate_pace_boost(favorite, underdog)
        three_pt_boost = self._calculate_three_pt_boost(favorite, underdog)
        def_boost = self._calculate_defensive_boost(favorite, underdog)
        
        # 5. Cinderella factors
        exp_boost = self._calculate_exp_boost(favorite, underdog)
        form_boost = self._calculate_form_boost(favorite, underdog)
        momentum_boost = self._calculate_momentum_boost(favorite, underdog)
        
        # 6. Combine all factors with weights
        # Base: 20% historical, 40% model, 40% simulation (if available)
        if sim_upset > 0:
            combined_base = 0.20 * base_upset + 0.40 * model_upset + 0.40 * sim_upset
        else:
            combined_base = 0.30 * base_upset + 0.70 * model_upset
        
        # Apply style and Cinderella boosts (scaled by chaos level)
        style_adjustment = (pace_boost + three_pt_boost + def_boost) * self.chaos_level
        cinderella_adjustment = (exp_boost + form_boost + momentum_boost) * self.chaos_level
        
        final_upset = combined_base + style_adjustment + cinderella_adjustment
        
        # Clamp to valid probability range
        final_upset = max(0.05, min(0.95, final_upset))
        
        return UpsetFactors(
            base_upset_prob=base_upset,
            model_upset_prob=model_upset,
            sim_upset_prob=sim_upset,
            pace_mismatch_boost=pace_boost,
            three_pt_boost=three_pt_boost,
            defensive_matchup_boost=def_boost,
            tourney_exp_boost=exp_boost,
            recent_form_boost=form_boost,
            momentum_boost=momentum_boost,
            final_upset_prob=final_upset
        )
    
    def _get_sim_upset_prob(
        self,
        fav_name: str,
        underdog_name: str,
        round_num: int
    ) -> float:
        """Get upset probability from simulation results."""
        if not self.sim_results:
            return 0.0
        
        # Map round_num to advancement key
        round_keys = {
            1: "r64", 2: "r32", 3: "s16", 4: "e8", 5: "f4", 6: "championship"
        }
        key = round_keys.get(round_num)
        if not key:
            return 0.0
        
        # Look for team advancement probabilities
        adv_data = self.sim_results.get("advancement", {})
        fav_adv = adv_data.get(fav_name, {}).get(key, 0.0)
        dog_adv = adv_data.get(underdog_name, {}).get(key, 0.0)
        
        if fav_adv + dog_adv > 0:
            return dog_adv / (fav_adv + dog_adv)
        return 0.0
    
    def _calculate_pace_boost(self, fav: TournamentTeam, dog: TournamentTeam) -> float:
        """
        High pace mismatch increases upset likelihood.
        Fast vs slow = unpredictable game = underdog has better chance.
        """
        pace_diff = abs(fav.pace - dog.pace)
        if pace_diff >= self.PACE_MISMATCH_THRESHOLD:
            # Underdog benefits more if they're the fast team
            if dog.pace > fav.pace:
                return 0.08  # +8% upset chance
            else:
                return 0.04  # +4% upset chance
        return 0.0
    
    def _calculate_three_pt_boost(self, fav: TournamentTeam, dog: TournamentTeam) -> float:
        """
        High 3PT rate teams = high variance = more upsets.
        """
        fav_high = fav.three_pt_rate >= self.HIGH_3PT_THRESHOLD
        dog_high = dog.three_pt_rate >= self.HIGH_3PT_THRESHOLD
        
        if fav_high and dog_high:
            # Both teams chuck threes = coin flip game
            return 0.12
        elif dog_high:
            # Only underdog shoots threes = live by the three, die by the three
            return 0.08
        elif fav_high:
            # Only favorite shoots threes = variance helps underdog
            return 0.05
        return 0.0
    
    def _calculate_defensive_boost(self, fav: TournamentTeam, dog: TournamentTeam) -> float:
        """
        Defensive mismatches can favor underdogs.
        """
        def_diff = fav.def_efg_pct - dog.def_efg_pct
        if def_diff > self.DEF_EFG_MISMATCH_THRESHOLD:
            # Underdog has better defense
            return 0.06
        elif def_diff < -self.DEF_EFG_MISMATCH_THRESHOLD:
            # Favorite has much better defense = chalk
            return -0.03
        return 0.0
    
    def _calculate_exp_boost(self, fav: TournamentTeam, dog: TournamentTeam) -> float:
        """
        Tournament experience matters in March.
        """
        exp_diff = dog.tournament_exp - fav.tournament_exp
        if exp_diff > 0.20:  # Underdog has significantly more tourney experience
            return 0.05 * exp_diff  # Up to +5%
        return 0.0
    
    def _calculate_form_boost(self, fav: TournamentTeam, dog: TournamentTeam) -> float:
        """
        Recent form (last 10 games) momentum.
        """
        form_diff = dog.recent_form - fav.recent_form
        if form_diff > 1.0:  # Underdog is hot
            return min(0.10, 0.05 * form_diff)  # Up to +10%
        return 0.0
    
    def _calculate_momentum_boost(self, fav: TournamentTeam, dog: TournamentTeam) -> float:
        """
        Seed momentum: lower seeds playing above their rating.
        """
        # If underdog has composite rating closer to favorite than expected
        expected_diff = (favorite.seed - underdog.seed) * 2.5  # ~2.5 pts per seed
        actual_diff = favorite.composite_rating - underdog.composite_rating
        
        if actual_diff < expected_diff * 0.7:  # Underdog is closer than expected
            return 0.06
        return 0.0
    
    def predict_winner(
        self,
        team_a: TournamentTeam,
        team_b: TournamentTeam,
        round_num: int,
        region: str = "",
        deterministic: bool = False
    ) -> Tuple[TournamentTeam, TournamentTeam, float, UpsetFactors]:
        """
        Predict winner using all factors.
        
        Args:
            deterministic: If True, always pick based on final_upset_prob threshold
                          If False, use probability-based random selection
        """
        # Identify favorite and underdog by seed
        if team_a.seed < team_b.seed:
            favorite, underdog = team_a, team_b
        elif team_b.seed < team_a.seed:
            favorite, underdog = team_b, team_a
        else:
            # Same seed — use composite rating
            if team_a.composite_rating >= team_b.composite_rating:
                favorite, underdog = team_a, team_b
            else:
                favorite, underdog = team_b, team_a
        
        factors = self.calculate_upset_factors(favorite, underdog, round_num, region)
        
        if deterministic:
            # Predict upset if final prob > 50%
            upset_happens = factors.final_upset_prob > 0.50
        else:
            # Probability-weighted random selection
            upset_happens = random.random() < factors.final_upset_prob
        
        if upset_happens:
            winner, loser = underdog, favorite
            win_prob = factors.final_upset_prob
        else:
            winner, loser = favorite, underdog
            win_prob = 1.0 - factors.final_upset_prob
        
        return winner, loser, win_prob, factors
    
    def generate_bracket_with_explanations(
        self,
        bracket: Dict[str, List[TournamentTeam]]
    ) -> Dict:
        """
        Generate full bracket with detailed upset explanations.
        """
        from backend.tournament.bracket_simulator import R64_SEED_ORDER
        
        results = {
            "regions": {},
            "upsets": [],
            "explanations": []
        }
        
        for region, teams in bracket.items():
            seed_to_team = {t.seed: t for t in teams}
            slots = [seed_to_team[s] for s in R64_SEED_ORDER if s in seed_to_team]
            
            region_results = {"rounds": {0: list(zip(slots[::2], slots[1::2]))}}
            current = slots
            
            for round_num in [1, 2, 3, 4]:
                round_matchups = []
                next_round = []
                
                for i in range(0, len(current), 2):
                    ta, tb = current[i], current[i + 1]
                    winner, loser, prob, factors = self.predict_winner(
                        ta, tb, round_num, region, deterministic=True
                    )
                    
                    is_upset = winner.seed > loser.seed
                    matchup_data = {
                        "ta": ta, "tb": tb,
                        "winner": winner, "loser": loser,
                        "prob": prob, "is_upset": is_upset,
                        "factors": factors
                    }
                    
                    if is_upset:
                        results["upsets"].append({
                            "region": region,
                            "round": round_num,
                            "winner": winner.name,
                            "winner_seed": winner.seed,
                            "loser": loser.name,
                            "loser_seed": loser.seed,
                            "upset_prob": factors.final_upset_prob,
                            "explanation": self._generate_explanation(factors, winner, loser)
                        })
                    
                    round_matchups.append(matchup_data)
                    next_round.append(winner)
                
                region_results["rounds"][round_num] = round_matchups
                current = next_round
            
            region_results["winner"] = current[0] if current else None
            results["regions"][region] = region_results
        
        return results
    
    def _generate_explanation(
        self,
        factors: UpsetFactors,
        winner: TournamentTeam,
        loser: TournamentTeam
    ) -> str:
        """Generate human-readable explanation for why upset was predicted."""
        reasons = []
        
        if factors.pace_mismatch_boost > 0.05:
            reasons.append(f"pace mismatch favors {winner.name}")
        if factors.three_pt_boost > 0.08:
            reasons.append(f"high-variance 3PT game")
        if factors.tourney_exp_boost > 0.03:
            reasons.append(f"{winner.name} has more tournament experience")
        if factors.recent_form_boost > 0.05:
            reasons.append(f"{winner.name} is hot (strong March form)")
        if factors.defensive_matchup_boost > 0.03:
            reasons.append(f"{winner.name} has defensive edge")
        
        if not reasons:
            reasons.append(f"historical {winner.seed}-{loser.seed} upset rate ({factors.base_upset_prob:.1%})")
        
        return "; ".join(reasons)


def generate_smart_bracket(
    bracket: Dict[str, List[TournamentTeam]],
    sim_results_path: Optional[str] = None,
    chaos_level: float = 0.5,
    return_explanations: bool = True
) -> Dict:
    """
    Convenience function to generate a smart bracket.
    
    Args:
        bracket: {region: [TournamentTeam]}
        sim_results_path: Path to Monte Carlo results
        chaos_level: 0.0 = chalk, 1.0 = max chaos
        return_explanations: Include upset explanations
    
    Returns:
        Dict with bracket structure and upset explanations
    """
    generator = SmartBracketGenerator(sim_results_path, chaos_level)
    return generator.generate_bracket_with_explanations(bracket)


if __name__ == "__main__":
    # Test with sample matchup
    from backend.tournament.matchup_predictor import TournamentTeam
    
    vanderbilt = TournamentTeam(
        name="Vanderbilt", seed=5, region="south",
        composite_rating=13.0, kp_adj_em=13.3, bt_adj_em=12.6,
        pace=69.0, three_pt_rate=0.37, def_efg_pct=0.50,
        conference="SEC", tournament_exp=0.60, recent_form=0.5
    )
    
    mcneese = TournamentTeam(
        name="McNeese", seed=12, region="south",
        composite_rating=1.0, kp_adj_em=1.3, bt_adj_em=0.6,
        pace=71.0, three_pt_rate=0.42, def_efg_pct=0.48,
        conference="Southland", tournament_exp=0.45, recent_form=1.5
    )
    
    generator = SmartBracketGenerator(chaos_level=0.7)
    factors = generator.calculate_upset_factors(vanderbilt, mcneese, round_num=1)
    
    print(f"Vanderbilt vs McNeese Upset Analysis:")
    print(f"  Base upset rate: {factors.base_upset_prob:.1%}")
    print(f"  Model upset prob: {factors.model_upset_prob:.1%}")
    print(f"  Pace boost: +{factors.pace_mismatch_boost:.1%}")
    print(f"  3PT boost: +{factors.three_pt_boost:.1%}")
    print(f"  Tourney exp boost: +{factors.tourney_exp_boost:.1%}")
    print(f"  Recent form boost: +{factors.recent_form_boost:.1%}")
    print(f"  FINAL UPSET PROB: {factors.final_upset_prob:.1%}")
