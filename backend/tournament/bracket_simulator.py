"""
Monte Carlo bracket simulation engine.

Simulates 50,000+ NCAA tournaments to produce:
- Championship / Final Four / Elite 8 / Sweet 16 win probabilities for all 68 teams
- Average championship margin
- Average upsets per tournament

Design:
- Uses Python multiprocessing to run parallel batches (ProcessPoolExecutor)
- Each simulation is deterministic given a random seed (reproducible)
- Bracket structure: 4 regions of 16 teams, standard seed pairing (1v16, 8v9, ...)
- Final Four cross-region pairings: South vs East, West vs Midwest

Performance targets (from BRACKET-001 spec):
- 10k sims: < 2 min (single core)
- 50k sims: < 5 min (4 workers)
"""

import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import concurrent.futures

from backend.tournament.matchup_predictor import TournamentTeam, predict_game

logger = logging.getLogger(__name__)

# Standard R64 seed pairing order within each region (position in bracket slots)
R64_SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

# Final Four cross-region pairings (standard NCAA bracket structure)
FF_PAIRINGS = [("south", "east"), ("west", "midwest")]


@dataclass
class TournamentResult:
    """Result of one complete tournament simulation."""
    champion: str
    runner_up: str
    final_four: List[str]
    elite_eight: List[str]
    sweet_sixteen: List[str]
    round_of_32: List[str]
    championship_margin: float
    total_upsets: int


@dataclass
class SimulationResults:
    """Aggregated probabilities from n tournament simulations."""
    n_sims: int
    championship: Dict[str, float] = field(default_factory=dict)
    runner_up: Dict[str, float] = field(default_factory=dict)
    final_four: Dict[str, float] = field(default_factory=dict)
    elite_eight: Dict[str, float] = field(default_factory=dict)
    sweet_sixteen: Dict[str, float] = field(default_factory=dict)
    round_of_32: Dict[str, float] = field(default_factory=dict)
    avg_championship_margin: float = 0.0
    avg_upsets_per_tournament: float = 0.0


def _simulate_region(
    teams: List[TournamentTeam],
    rng: random.Random,
) -> Tuple[TournamentTeam, List[str], List[str], List[str], int]:
    """
    Simulate one region through four rounds (R64, R32, S16, E8).

    Args:
        teams: All 16 teams in the region
        rng: Random number generator (seeded for reproducibility)

    Returns:
        Tuple of (region_winner, elite_eight_names, sweet16_names, r32_names, upsets_count)
    """
    seed_to_team = {t.seed: t for t in teams}
    # Arrange teams in standard bracket slot order
    slots = [seed_to_team[s] for s in R64_SEED_ORDER if s in seed_to_team]

    upsets = 0
    r32_names = []
    s16_names = []
    e8_names = []

    # Simulate each round — pair adjacent slots, winner advances
    for round_num in [1, 2, 3, 4]:  # R64=1, R32=2, S16=3, E8=4
        next_slots = []
        for i in range(0, len(slots), 2):
            if i + 1 >= len(slots):
                next_slots.append(slots[i])
                continue

            ta, tb = slots[i], slots[i + 1]
            win_prob_a, _, _ = predict_game(ta, tb, round_num)

            if rng.random() < win_prob_a:
                winner, loser = ta, tb
            else:
                winner, loser = tb, ta

            # Count upsets: lower seed (higher number) beating higher seed
            if winner.seed > loser.seed:
                upsets += 1

            next_slots.append(winner)

        slots = next_slots

        # Record survivors at each milestone
        if round_num == 2:
            r32_names = [t.name for t in slots]
        elif round_num == 3:
            s16_names = [t.name for t in slots]
        elif round_num == 4:
            e8_names = [t.name for t in slots]

    region_winner = slots[0]
    return region_winner, e8_names, s16_names, r32_names, upsets


def simulate_single_tournament(
    bracket: Dict[str, List[TournamentTeam]],
    seed: Optional[int] = None,
) -> TournamentResult:
    """
    Simulate one complete NCAA tournament.

    Args:
        bracket: {region_name: [TournamentTeam, ...]} with 4 regions of 16 teams
        seed: Random seed for reproducibility (None = non-deterministic)

    Returns:
        TournamentResult with champion, runner-up, and all milestone data
    """
    rng = random.Random(seed)

    region_winners: Dict[str, TournamentTeam] = {}
    all_e8: List[str] = []
    all_s16: List[str] = []
    all_r32: List[str] = []
    total_upsets = 0

    # Simulate all four regions
    for region, teams in bracket.items():
        winner, e8, s16, r32, upsets = _simulate_region(teams, rng)
        region_winners[region] = winner
        all_e8.extend(e8)
        all_s16.extend(s16)
        all_r32.extend(r32)
        total_upsets += upsets

    # Final Four — two semifinal matchups
    ff_names: List[str] = []
    finalists: List[TournamentTeam] = []

    for reg_a, reg_b in FF_PAIRINGS:
        ta = region_winners.get(reg_a)
        tb = region_winners.get(reg_b)

        # Fallback: if region names don't match exactly, take any two remaining winners
        if ta is None or tb is None:
            remaining = [t for r, t in region_winners.items() if r not in (reg_a, reg_b)]
            if ta is None and remaining:
                ta = remaining.pop(0)
            if tb is None and remaining:
                tb = remaining.pop(0)

        if ta is None or tb is None:
            logger.warning("Could not pair Final Four regions %s/%s", reg_a, reg_b)
            continue

        win_prob_a, _, _ = predict_game(ta, tb, round_num=5)
        ff_winner = ta if rng.random() < win_prob_a else tb
        ff_names.extend([ta.name, tb.name])
        finalists.append(ff_winner)

        if ff_winner.seed > (tb.seed if ff_winner is ta else ta.seed):
            total_upsets += 1

    # Championship game
    if len(finalists) < 2:
        # Degenerate case: use whatever region winners remain
        finalists = list(region_winners.values())[:2]

    ta, tb = finalists[0], finalists[1]
    win_prob_a, champ_margin, _ = predict_game(ta, tb, round_num=6)

    if rng.random() < win_prob_a:
        champion, runner_up = ta, tb
    else:
        champion, runner_up = tb, ta
        champ_margin = -champ_margin

    if champion.seed > runner_up.seed:
        total_upsets += 1

    return TournamentResult(
        champion=champion.name,
        runner_up=runner_up.name,
        final_four=ff_names,
        elite_eight=all_e8,
        sweet_sixteen=all_s16,
        round_of_32=all_r32,
        championship_margin=abs(champ_margin),
        total_upsets=total_upsets,
    )


# ---------------------------------------------------------------------------
# Multiprocessing batch function (must be module-level for pickling)
# ---------------------------------------------------------------------------

def _run_batch(args: Tuple) -> Dict:
    """
    Run a batch of tournament simulations. Module-level for multiprocessing pickling.

    Args:
        args: (bracket, batch_size, base_seed)

    Returns:
        Dict of aggregated counts for this batch
    """
    bracket, batch_size, base_seed = args

    counts: Dict = {
        "championship": defaultdict(int),
        "runner_up": defaultdict(int),
        "final_four": defaultdict(int),
        "elite_eight": defaultdict(int),
        "sweet_sixteen": defaultdict(int),
        "round_of_32": defaultdict(int),
        "championship_margins": [],
        "upsets": [],
    }

    for i in range(batch_size):
        sim_seed = (base_seed + i) if base_seed is not None else None
        result = simulate_single_tournament(bracket, seed=sim_seed)

        counts["championship"][result.champion] += 1
        counts["runner_up"][result.runner_up] += 1
        for t in result.final_four:
            counts["final_four"][t] += 1
        for t in result.elite_eight:
            counts["elite_eight"][t] += 1
        for t in result.sweet_sixteen:
            counts["sweet_sixteen"][t] += 1
        for t in result.round_of_32:
            counts["round_of_32"][t] += 1
        counts["championship_margins"].append(result.championship_margin)
        counts["upsets"].append(result.total_upsets)

    return counts


def run_monte_carlo(
    bracket: Dict[str, List[TournamentTeam]],
    n_sims: int = 50000,
    n_workers: int = 4,
    base_seed: Optional[int] = 42,
) -> SimulationResults:
    """
    Run full Monte Carlo tournament simulation.

    Args:
        bracket: {region_name: [TournamentTeam, ...]} — 4 regions of 16 teams each
        n_sims: Total number of simulations to run (default: 50,000)
        n_workers: Number of parallel processes (default: 4)
        base_seed: Base random seed for reproducibility (None = non-deterministic)

    Returns:
        SimulationResults with probability estimates for all teams at all milestones
    """
    logger.info(
        "Monte Carlo: %d sims across %d workers (seed=%s)", n_sims, n_workers, base_seed
    )

    # Distribute sims evenly across workers
    batch_size = n_sims // n_workers
    remainder = n_sims % n_workers

    args_list = []
    for i in range(n_workers):
        b = batch_size + (1 if i < remainder else 0)
        seed = (base_seed + i * batch_size) if base_seed is not None else None
        args_list.append((bracket, b, seed))

    # Aggregate counts across all batches
    aggregated: Dict = {
        "championship": defaultdict(int),
        "runner_up": defaultdict(int),
        "final_four": defaultdict(int),
        "elite_eight": defaultdict(int),
        "sweet_sixteen": defaultdict(int),
        "round_of_32": defaultdict(int),
        "championship_margins": [],
        "upsets": [],
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures_map = {executor.submit(_run_batch, args): i for i, args in enumerate(args_list)}
        for future in concurrent.futures.as_completed(futures_map):
            batch = future.result()
            for key in ("championship", "runner_up", "final_four", "elite_eight",
                        "sweet_sixteen", "round_of_32"):
                for team, count in batch[key].items():
                    aggregated[key][team] += count
            aggregated["championship_margins"].extend(batch["championship_margins"])
            aggregated["upsets"].extend(batch["upsets"])

    # Convert raw counts to probabilities
    results = SimulationResults(n_sims=n_sims)
    for key in ("championship", "runner_up", "final_four", "elite_eight",
                "sweet_sixteen", "round_of_32"):
        setattr(results, key, {
            team: count / n_sims
            for team, count in aggregated[key].items()
        })

    margins = aggregated["championship_margins"]
    if margins:
        results.avg_championship_margin = sum(margins) / len(margins)

    upsets = aggregated["upsets"]
    if upsets:
        results.avg_upsets_per_tournament = sum(upsets) / len(upsets)

    top5 = sorted(results.championship.items(), key=lambda x: -x[1])[:5]
    logger.info(
        "Simulation complete. Top 5 championship: %s",
        ", ".join(f"{t}={p*100:.1f}%" for t, p in top5),
    )
    return results
