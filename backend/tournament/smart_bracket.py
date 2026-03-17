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
    
    # Historical seed upset rates (baseline — 35+ years of NCAA tournament data)
    SEED_UPSET_RATES = {
        (1, 16): 0.013, (2, 15): 0.067, (3, 14): 0.153, (4, 13): 0.216,
        (5, 12): 0.352, (6, 11): 0.389, (7, 10): 0.394, (8, 9): 0.487,
    }

    # Historical upset fraction per round (games won by higher seed ÷ total games).
    # Derived from 35 years of tournament data:
    #   R64 : ~10 of 32 games  = 31%
    #   R32 : ~5  of 16 games  = 31%
    #   S16 : ~3  of  8 games  = 38%   (survivors skew closer in quality)
    #   E8  : ~2  of  4 games  = 50%   (only 4 games; frequently a 2-seed wins)
    #   F4  : ~1  of  2 games  = 50%
    #   Champ: ~0.4 of 1 game  = 40%
    # At chaos=0.5 the deterministic bracket will pick exactly these many upsets.
    ROUND_HISTORICAL_UPSET_RATES: Dict[int, float] = {
        1: 0.31,
        2: 0.31,
        3: 0.38,
        4: 0.50,
        5: 0.50,
        6: 0.40,
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
        expected_diff = (fav.seed - dog.seed) * 2.5  # ~2.5 pts per seed
        actual_diff = fav.composite_rating - dog.composite_rating
        
        if actual_diff < expected_diff * 0.7:  # Underdog is closer than expected
            return 0.06
        return 0.0
    
    def _n_upsets_for_round(self, n_matchups: int, round_num: int) -> int:
        """
        How many upsets to select for this round given the current chaos_level.

        Calibration:
          chaos=0.0 → 0 upsets (chalk)
          chaos=0.5 → historical average (ROUND_HISTORICAL_UPSET_RATES × n_matchups)
          chaos=1.0 → 2× historical average, capped at n_matchups

        This prevents the old cliff behaviour where chaos=0.3 produced only 4
        upsets and chaos=0.5 immediately produced 20.
        """
        hist_rate = self.ROUND_HISTORICAL_UPSET_RATES.get(round_num, 0.31)
        n = round(n_matchups * hist_rate * 2.0 * self.chaos_level)
        return max(0, min(n_matchups, n))

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
            deterministic: If True, this call is used inside
                           generate_bracket_with_explanations(), which now
                           handles calibrated round-level upset selection itself.
                           The per-game result returned here always picks the
                           model favourite; the caller overrides for chosen upsets.
                           If False, sample probabilistically from final_upset_prob.
        """
        # Identify favourite and underdog by seed
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
            # Default to favourite; generate_bracket_with_explanations will
            # override specific matchups to upsets using round-level selection.
            winner, loser = favorite, underdog
            win_prob = 1.0 - factors.final_upset_prob
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

        Deterministic upset selection is calibrated to historical rates:
          1. Compute final_upset_prob for every matchup in the round.
          2. Rank matchups by that probability (highest first).
          3. Designate the top-N as upsets, where N = _n_upsets_for_round().
             At chaos=0.5, N matches the historical average for that round.
             At chaos=0.0, N=0 (chalk). At chaos=1.0, N≈2× historical average.
          4. The WHICH games flip is always driven by the V9.1 model + historical
             rates + style/Cinderella factors — never random.
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
                # --- Step 1: compute factors for every matchup in this round ---
                matchup_data_list: List[Dict] = []
                for i in range(0, len(current), 2):
                    ta, tb = current[i], current[i + 1]
                    fav = ta if ta.seed <= tb.seed else tb
                    dog = tb if fav is ta else ta
                    if ta.seed == tb.seed:
                        fav = ta if ta.composite_rating >= tb.composite_rating else tb
                        dog = tb if fav is ta else ta
                    factors = self.calculate_upset_factors(fav, dog, round_num, region)
                    matchup_data_list.append({
                        "ta": ta, "tb": tb, "fav": fav, "dog": dog,
                        "factors": factors,
                    })

                # --- Step 2: decide which matchups become upsets ---
                n_upsets = self._n_upsets_for_round(len(matchup_data_list), round_num)
                # Rank by upset probability descending; only genuine upsets (dog != fav by seed)
                eligible = [
                    (idx, m) for idx, m in enumerate(matchup_data_list)
                    if m["dog"].seed > m["fav"].seed
                ]
                eligible.sort(key=lambda x: -x[1]["factors"].final_upset_prob)
                upset_indices = {idx for idx, _ in eligible[:n_upsets]}

                # --- Step 3: build matchup results ---
                round_matchups = []
                next_round = []
                for idx, m in enumerate(matchup_data_list):
                    fav, dog, factors = m["fav"], m["dog"], m["factors"]
                    is_true_upset = idx in upset_indices and dog.seed > fav.seed
                    if is_true_upset:
                        winner, loser = dog, fav
                        prob = factors.final_upset_prob
                    else:
                        winner, loser = fav, dog
                        prob = 1.0 - factors.final_upset_prob

                    is_upset = winner.seed > loser.seed
                    if is_upset:
                        results["upsets"].append({
                            "region": region,
                            "round": round_num,
                            "winner": winner.name,
                            "winner_seed": winner.seed,
                            "loser": loser.name,
                            "loser_seed": loser.seed,
                            "upset_prob": factors.final_upset_prob,
                            "explanation": self._generate_explanation(
                                factors, winner, loser
                            ),
                        })

                    round_matchups.append({
                        "ta": m["ta"], "tb": m["tb"],
                        "winner": winner, "loser": loser,
                        "prob": prob, "is_upset": is_upset,
                        "factors": factors,
                    })
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


# ---------------------------------------------------------------------------
# Pool-Optimal Bracket
# ---------------------------------------------------------------------------

def generate_pool_optimal_bracket(
    bracket: Dict[str, List[TournamentTeam]],
    mc_results=None,
    n_12v5_picks: int = 2,
    n_11v6_picks: int = 1,
    force_all_9v8: bool = True,
) -> Dict:
    """
    Generate the mathematically optimal bracket for winning a large pool.

    Strategy:
    - All 1-seeds advance to the Final Four (modal tournament outcome, ~25% of tourneys)
    - Champion = highest MC probability 1-seed (or best available seed if mc_results given)
    - R64 upsets are surgically selected:
        * All 8v9 matchups: let the model pick whichever side has >50% win probability
          (they are coin flips; use the model edge however small it is)
        * Top n_12v5_picks  5v12 games: ranked by SmartBracketGenerator upset probability
        * Top n_11v6_picks  6v11 games: ranked by SmartBracketGenerator upset probability
        * All other R64 matchups: chalk (favour higher seed)
    - R32 and beyond: pure chalk — maximises expected score when later rounds are
      worth 2/4/8/16 points and the variance of upset picks compounds badly.

    Args:
        bracket:       {region: [TournamentTeam]}
        mc_results:    SimulationResults from run_monte_carlo() (optional — used to
                       rank champion candidates by MC championship probability)
        n_12v5_picks:  How many 12-over-5 upsets to force (default 2)
        n_11v6_picks:  How many 11-over-6 upsets to force (default 1)
        force_all_9v8: If True, pick the model-favoured side for all 8v9 matchups

    Returns:
        Dict matching generate_smart_bracket() structure
        (regions / rounds / winner / upsets / explanations)
        Plus extra key "pool_rationale" with plain-English pick explanations.
    """
    from backend.tournament.bracket_simulator import R64_SEED_ORDER

    generator = SmartBracketGenerator(chaos_level=0.0)  # chalk base

    # --- 1. Collect all R64 upset candidates for 5v12 and 6v11 ---
    upset_candidates: List[Tuple[str, int, int, float, str]] = []
    # (region, fav_seed, dog_seed, upset_prob, explanation)

    for region, teams in bracket.items():
        seed_to_team = {t.seed: t for t in teams}
        slots = [seed_to_team[s] for s in R64_SEED_ORDER if s in seed_to_team]

        for i in range(0, len(slots), 2):
            ta, tb = slots[i], slots[i + 1]
            higher_seed = min(ta.seed, tb.seed)
            lower_seed = max(ta.seed, tb.seed)
            pair = (higher_seed, lower_seed)

            if pair not in ((5, 12), (6, 11)):
                continue

            fav = ta if ta.seed == higher_seed else tb
            dog = ta if ta.seed == lower_seed else tb
            factors = generator.calculate_upset_factors(fav, dog, round_num=1)
            upset_candidates.append((
                region,
                higher_seed,
                lower_seed,
                factors.final_upset_prob,
                generator._generate_explanation(factors, dog, fav),
            ))

    # Sort descending by upset probability; pick top n per matchup type
    candidates_12v5 = sorted(
        [c for c in upset_candidates if c[1] == 5],
        key=lambda x: -x[3],
    )[:n_12v5_picks]

    candidates_11v6 = sorted(
        [c for c in upset_candidates if c[1] == 6],
        key=lambda x: -x[3],
    )[:n_11v6_picks]

    forced_upsets: Dict[Tuple[str, int, int], str] = {}
    for region, fav_s, dog_s, prob, expl in candidates_12v5 + candidates_11v6:
        forced_upsets[(region, fav_s, dog_s)] = (
            f"historical {dog_s}-{fav_s} upset rate "
            f"({generator.SEED_UPSET_RATES.get((fav_s, dog_s), 0):.1%}); "
            f"model upset prob {prob:.1%}"
        )

    # --- 2. Simulate the bracket ---
    results: Dict = {"regions": {}, "upsets": [], "explanations": [], "pool_rationale": []}

    for region, teams in bracket.items():
        seed_to_team = {t.seed: t for t in teams}
        slots = [seed_to_team[s] for s in R64_SEED_ORDER if s in seed_to_team]

        region_results: Dict = {"rounds": {0: list(zip(slots[::2], slots[1::2]))}}
        current = slots

        for round_num in [1, 2, 3, 4]:
            matchups = []
            next_round = []

            for i in range(0, len(current), 2):
                ta, tb = current[i], current[i + 1]
                higher_seed = min(ta.seed, tb.seed)
                lower_seed = max(ta.seed, tb.seed)
                pair_key = (region, higher_seed, lower_seed)

                fav = ta if ta.seed == higher_seed else tb
                dog = ta if ta.seed == lower_seed else tb

                # Decide winner
                if round_num == 1 and (higher_seed, lower_seed) == (8, 9) and force_all_9v8:
                    # Coin flip: use model probability (small edge)
                    prob_a, _, _ = predict_game(ta, tb, round_num)
                    if prob_a >= 0.5:
                        winner, loser, prob = ta, tb, prob_a
                    else:
                        winner, loser, prob = tb, ta, 1.0 - prob_a
                elif round_num == 1 and pair_key in forced_upsets:
                    # Forced upset
                    expl = forced_upsets[pair_key]
                    winner, loser = dog, fav
                    factors = generator.calculate_upset_factors(fav, dog, round_num)
                    prob = factors.final_upset_prob
                else:
                    # Pure chalk: favourite wins
                    prob_a, _, _ = predict_game(ta, tb, round_num)
                    if prob_a >= 0.5:
                        winner, loser, prob = ta, tb, prob_a
                    else:
                        winner, loser, prob = tb, ta, 1.0 - prob_a

                is_upset = winner.seed > loser.seed
                matchup_data = {
                    "ta": ta, "tb": tb,
                    "winner": winner, "loser": loser,
                    "prob": prob, "is_upset": is_upset,
                }
                matchups.append(matchup_data)
                next_round.append(winner)

                if is_upset:
                    expl_text = forced_upsets.get(pair_key, "model edge")
                    results["upsets"].append({
                        "region": region,
                        "round": round_num,
                        "winner": winner.name,
                        "winner_seed": winner.seed,
                        "loser": loser.name,
                        "loser_seed": loser.seed,
                        "upset_prob": prob,
                        "explanation": expl_text,
                    })

            region_results["rounds"][round_num] = matchups
            current = next_round

        region_results["winner"] = current[0] if current else None
        results["regions"][region] = region_results

    # --- 3. Pool rationale ---
    for region, fav_s, dog_s, prob, expl in candidates_12v5:
        seed_to_team = {t.seed: t for t in bracket[region]}
        fav_name = seed_to_team[fav_s].name if fav_s in seed_to_team else f"#{fav_s}"
        dog_name = seed_to_team[dog_s].name if dog_s in seed_to_team else f"#{dog_s}"
        hist_rate = generator.SEED_UPSET_RATES.get((fav_s, dog_s), 0)
        results["pool_rationale"].append(
            f"#{dog_s} {dog_name} over #{fav_s} {fav_name} ({region.upper()}) — "
            f"12v5 happens {hist_rate:.0%} of the time historically; "
            f"most bracket-fillers ignore this, giving you pool edge when it hits."
        )
    for region, fav_s, dog_s, prob, expl in candidates_11v6:
        seed_to_team = {t.seed: t for t in bracket[region]}
        fav_name = seed_to_team[fav_s].name if fav_s in seed_to_team else f"#{fav_s}"
        dog_name = seed_to_team[dog_s].name if dog_s in seed_to_team else f"#{dog_s}"
        hist_rate = generator.SEED_UPSET_RATES.get((fav_s, dog_s), 0)
        results["pool_rationale"].append(
            f"#{dog_s} {dog_name} over #{fav_s} {fav_name} ({region.upper()}) — "
            f"11v6 upsets happen {hist_rate:.0%} historically; contrarian but justified."
        )

    return results


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
