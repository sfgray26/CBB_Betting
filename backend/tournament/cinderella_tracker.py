"""
Cinderella and upset detection for tournament simulations.

Provides:
- cinderella_rankings(): Rank double-digit seeds by probability of deep runs
- upset_heat_map(): R64 upset probability for every first-round matchup
- format_cinderella_table(): Console/Discord output
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from backend.tournament.bracket_simulator import SimulationResults
from backend.tournament.matchup_predictor import TournamentTeam, predict_game
from backend.tournament.bracket_simulator import R64_SEED_ORDER

logger = logging.getLogger(__name__)


@dataclass
class CinderellaCandidate:
    """A double-digit seed with meaningful deep-run probability."""
    team: str
    seed: int
    region: str
    p_round_of_32: float
    p_sweet_sixteen: float
    p_elite_eight: float
    p_final_four: float
    # Composite score: higher seed + deeper run = better story
    cinderella_score: float


@dataclass
class UpsetMatchup:
    """First-round matchup with upset probability."""
    region: str
    favorite: str
    favorite_seed: int
    underdog: str
    underdog_seed: int
    upset_probability: float
    margin_estimate: float       # Expected margin if favorite wins (positive)
    risk_level: str              # "HIGH" (>35%), "MED" (>20%), "LOW"


def cinderella_rankings(
    sim_results: SimulationResults,
    all_teams: List[Dict],
    min_seed: int = 10,
    min_s16_prob: float = 0.01,
) -> List[CinderellaCandidate]:
    """
    Rank double-digit seeds by probability of reaching Sweet 16 or beyond.

    Args:
        sim_results: Aggregated Monte Carlo results
        all_teams: List of {name, seed, region} dicts for all 68 teams
        min_seed: Minimum seed number to consider Cinderella (default: 10)
        min_s16_prob: Minimum Sweet 16 probability to appear in rankings (default: 1%)

    Returns:
        List of CinderellaCandidate sorted by cinderella_score descending
    """
    candidates: List[CinderellaCandidate] = []

    for team_info in all_teams:
        team_name = team_info["name"]
        seed = team_info.get("seed", 0)
        region = team_info.get("region", "")

        if seed < min_seed:
            continue

        p_r32 = sim_results.round_of_32.get(team_name, 0.0)
        p_s16 = sim_results.sweet_sixteen.get(team_name, 0.0)
        p_e8 = sim_results.elite_eight.get(team_name, 0.0)
        p_f4 = sim_results.final_four.get(team_name, 0.0)

        if p_s16 < min_s16_prob:
            continue

        # Cinderella score: higher seed + deeper run = bigger story
        # Normalized so a 12-seed at 25% S16 and a 15-seed at 10% S16 compare meaningfully
        score = seed * p_s16 * 10

        candidates.append(CinderellaCandidate(
            team=team_name,
            seed=seed,
            region=region,
            p_round_of_32=p_r32,
            p_sweet_sixteen=p_s16,
            p_elite_eight=p_e8,
            p_final_four=p_f4,
            cinderella_score=score,
        ))

    candidates.sort(key=lambda x: x.cinderella_score, reverse=True)
    logger.info("Cinderella tracker: %d candidates (min seed %d, min P(S16) %.0f%%)",
                len(candidates), min_seed, min_s16_prob * 100)
    return candidates


def upset_heat_map(
    bracket: Dict[str, List[TournamentTeam]],
) -> List[UpsetMatchup]:
    """
    Calculate R64 upset probability for every first-round matchup.

    Uses the matchup predictor directly (no simulation needed — deterministic).

    Args:
        bracket: {region_name: [TournamentTeam, ...]}

    Returns:
        List of UpsetMatchup sorted by upset_probability descending (most dangerous first)
    """
    matchups: List[UpsetMatchup] = []

    for region, teams in bracket.items():
        seed_to_team = {t.seed: t for t in teams}
        seeds_present = [s for s in R64_SEED_ORDER if s in seed_to_team]

        for i in range(0, len(seeds_present), 2):
            if i + 1 >= len(seeds_present):
                continue

            sa = seeds_present[i]      # lower seed number = higher-ranked (favorite)
            sb = seeds_present[i + 1]  # higher seed number = lower-ranked (underdog)
            ta = seed_to_team[sa]
            tb = seed_to_team[sb]

            # predict_game with round_num=1 (R64)
            win_prob_fav, margin, _ = predict_game(ta, tb, round_num=1)
            upset_prob = 1.0 - win_prob_fav

            matchups.append(UpsetMatchup(
                region=region,
                favorite=ta.name,
                favorite_seed=ta.seed,
                underdog=tb.name,
                underdog_seed=tb.seed,
                upset_probability=round(upset_prob, 3),
                margin_estimate=round(margin, 1),
                risk_level=(
                    "HIGH" if upset_prob > 0.35
                    else "MED" if upset_prob > 0.20
                    else "LOW"
                ),
            ))

    matchups.sort(key=lambda x: x.upset_probability, reverse=True)
    n_high = sum(1 for m in matchups if m.risk_level == "HIGH")
    logger.info(
        "Upset heat map: %d R64 matchups, %d HIGH risk (>35%% upset prob)", len(matchups), n_high
    )
    return matchups


def format_cinderella_table(candidates: List[CinderellaCandidate], top_n: int = 10) -> str:
    """Format cinderella rankings as ASCII table for Discord/console."""
    if not candidates:
        return "No Cinderella candidates found."

    header = (
        f"{'#':<3} {'Team':<25} {'Seed':>4} {'P(R32)':>7} "
        f"{'P(S16)':>7} {'P(E8)':>7} {'P(F4)':>7} {'Score':>7}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for i, c in enumerate(candidates[:top_n], 1):
        rows.append(
            f"{i:<3} {c.team:<25} {c.seed:>4}  "
            f"{c.p_round_of_32*100:>5.1f}%  "
            f"{c.p_sweet_sixteen*100:>5.1f}%  "
            f"{c.p_elite_eight*100:>5.1f}%  "
            f"{c.p_final_four*100:>5.1f}%  "
            f"{c.cinderella_score:>6.2f}"
        )

    return "\n".join(rows)


def format_upset_heat_map(matchups: List[UpsetMatchup]) -> str:
    """Format R64 upset heat map as ASCII table."""
    if not matchups:
        return "No matchup data."

    header = (
        f"{'Region':<10} {'Matchup':<45} {'Upset%':>7} {'Risk':>5}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for m in matchups:
        matchup_str = f"#{m.favorite_seed} {m.favorite} vs #{m.underdog_seed} {m.underdog}"
        rows.append(
            f"{m.region:<10} {matchup_str:<45} "
            f"{m.upset_probability*100:>6.1f}%  {m.risk_level:>5}"
        )

    return "\n".join(rows)
