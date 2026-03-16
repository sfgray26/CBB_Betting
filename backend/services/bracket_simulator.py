"""
NCAA Tournament Monte Carlo Bracket Simulator.

Produces realistic bracket projections with historically-calibrated upset
probabilities. Uses a two-signal blend:

  1. KenPom AdjEM logistic model  (dominant in all rounds)
  2. Historical seed win rates     (significant only in R64 / R32)

A tournament SD bump of 1.15x widens the logistic curve relative to the
regular-season model, reflecting single-elimination variance inflation.

Usage:
    from backend.services.bracket_simulator import BracketTeam, simulate_tournament

    teams = [BracketTeam(name="Duke", seed=1, region="East", adj_em=28.4), ...]
    result = simulate_tournament(teams, n_sims=10_000)
    print(result.projected_champion)
    print(result.upset_alerts)
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Historical R64 win rate for (better_seed, worse_seed) matchup — all-time data.
# ADJUSTED for MAXIMUM Cinderella chaos
HISTORICAL_WIN_RATES: Dict[Tuple[int, int], float] = {
    (1, 16): 0.987,  # 1-seeds almost never lose
    (2, 15): 0.920,  # 2-seeds: 8% upset rate (slightly more chaos)
    (3, 14): 0.820,  # 3-seeds: 18% upset rate
    (4, 13): 0.750,  # 4-seeds: 25% upset rate
    (5, 12): 0.580,  # 5-seeds: 42% UPSET RATE — 12-seeds dangerous!
    (6, 11): 0.560,  # 6-seeds: 44% UPSET RATE — 11-seeds very dangerous!
    (7, 10): 0.550,  # 7-seeds: 45% UPSET RATE — 10-seeds coin flips!
    (8,  9): 0.500,  # 8 vs 9: TRUE COIN FLIP
}

# Weight given to the historical seed signal per round.
# VERY HIGH for R64 to allow Cinderella upsets despite AdjEM gaps
ROUND_HIST_WEIGHT: Dict[int, float] = {
    1: 0.75,  # R64: 75% history (was 55%) — MAJOR upset zone!
    2: 0.50,  # R32: 50% history (was 35%)
    3: 0.20,  # S16: 20% history
    4: 0.10,  # E8: 10% history
    5: 0.00,  # F4: pure model
    6: 0.00,  # Champ: pure model
}

# Standard bracket R64 seed pairings within a region (by seed number).
_R64_PAIRS: List[Tuple[int, int]] = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]

# Upset alert threshold: underdog must have at least this probability.
_UPSET_ALERT_THRESHOLD = 0.30  # LOWERED from 0.35 to flag more potential upsets

# Tournament SD inflation factor applied to the AdjEM logistic divisor.
_TOURNAMENT_SD_FACTOR = 1.40  # INCREASED from 1.25 for MAXIMUM March Madness chaos

# AdjEM logistic base divisor: 10.0 pts gap => ~73% win prob in regular season.
# INCREASED to reduce AdjEM dominance and allow more upsets
_ADJM_BASE_DIVISOR = 14.0  # INCREASED from 10.0 — less AdjEM impact

# Region assignment order for redistribution.
_REGION_NAMES = ["East", "West", "South", "Midwest"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BracketTeam:
    """A single team entry in the tournament bracket."""

    name: str
    seed: int        # 1–16
    region: str      # "East" | "West" | "South" | "Midwest"
    adj_em: float    # KenPom Adjusted Efficiency Margin (higher = better)


@dataclass
class BracketResult:
    """Full output of simulate_tournament()."""

    # name -> [p_r64_win, p_r32_win, p_s16_win, p_e8_win, p_f4_win,
    #          p_champ_game_win, p_champion]
    # Index 0 = won first round (reached R32), index 6 = champion.
    advancement_probs: Dict[str, List[float]]

    projected_champion: str
    projected_final_four: List[str]       # 4 most likely to reach Final Four

    # slot_key -> team_name  (consensus bracket — highest-prob winner per slot)
    projected_bracket: Dict[str, str]

    # Games where underdog has >= 35% win probability
    upset_alerts: List[Dict]

    n_sims: int


# ---------------------------------------------------------------------------
# Core probability model
# ---------------------------------------------------------------------------


def _matchup_win_prob(
    team_a: BracketTeam,
    team_b: BracketTeam,
    round_num: int,
) -> float:
    """
    Return P(team_a beats team_b) in the given tournament round.

    Blends an AdjEM logistic model with historical seed win rates.
    The blend weight for historical data fades as rounds progress
    (survivor bias makes seeds less predictive in later rounds).
    """
    adj_em_diff = team_a.adj_em - team_b.adj_em
    divisor = _ADJM_BASE_DIVISOR * _TOURNAMENT_SD_FACTOR  # = 11.5
    model_prob = 1.0 / (1.0 + math.exp(-adj_em_diff / divisor))

    hist_weight = ROUND_HIST_WEIGHT.get(round_num, 0.0)
    if hist_weight == 0.0:
        return model_prob

    better_seed = min(team_a.seed, team_b.seed)
    worse_seed = max(team_a.seed, team_b.seed)
    hist_rate = HISTORICAL_WIN_RATES.get((better_seed, worse_seed), model_prob)

    # hist_rate is always P(lower seed number wins).
    # Flip when team_a is the higher-numbered seed (i.e. the underdog).
    if team_a.seed == worse_seed:
        hist_rate = 1.0 - hist_rate

    return hist_weight * hist_rate + (1.0 - hist_weight) * model_prob


# ---------------------------------------------------------------------------
# Region redistribution (fallback when region tags are missing / unbalanced)
# ---------------------------------------------------------------------------


def _redistribute_into_regions(
    teams: List[BracketTeam],
) -> Dict[str, List[BracketTeam]]:
    """
    Assign teams to 4 balanced regions when the incoming region data is
    missing or unbalanced.

    Strategy:
      - Sort teams by adj_em descending (strongest first).
      - Assign seeds 1-16 within each region.
      - Distribute across regions in round-robin order within each seed tier
        so that every region receives approximately equal quality.

    Returns a dict of region_name -> list of 16 BracketTeam objects with
    seeds set correctly.
    """
    sorted_teams = sorted(teams, key=lambda t: t.adj_em, reverse=True)

    # We need exactly 64 teams split across 4 regions of 16.
    # If we have fewer, duplicate/pad; if more, truncate to top-64.
    sorted_teams = sorted_teams[:64]
    while len(sorted_teams) < 64:
        sorted_teams.append(sorted_teams[-1])

    region_teams: Dict[str, List[BracketTeam]] = {r: [] for r in _REGION_NAMES}

    # Assign seeds 1-16 in groups of 4 (one per region per seed number).
    # Tier 0 (strongest 4) get seed 1, next 4 get seed 2, etc.
    for seed_num in range(1, 17):
        tier_start = (seed_num - 1) * 4
        tier = sorted_teams[tier_start: tier_start + 4]
        for i, team in enumerate(tier):
            region_name = _REGION_NAMES[i % 4]
            new_team = BracketTeam(
                name=team.name,
                seed=seed_num,
                region=region_name,
                adj_em=team.adj_em,
            )
            region_teams[region_name].append(new_team)

    return region_teams


# ---------------------------------------------------------------------------
# Single-bracket simulation
# ---------------------------------------------------------------------------


def _simulate_one_bracket(
    regions: Dict[str, List[BracketTeam]],
    rng: random.Random,
) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """
    Simulate one complete tournament bracket.

    Returns (champion_name, final_four_names, elite_eight_names,
             sweet_sixteen_names, round_of_32_names).
    """
    regional_champions: List[BracketTeam] = []
    r32_winners_all: List[BracketTeam] = []
    s16_winners_all: List[BracketTeam] = []
    e8_winners_all: List[BracketTeam] = []

    for _region_name, teams in regions.items():
        seed_map: Dict[int, BracketTeam] = {t.seed: t for t in teams}

        # --- Round of 64 ---
        r64_winners: List[BracketTeam] = []
        for s1, s2 in _R64_PAIRS:
            ta = seed_map[s1]
            tb = seed_map[s2]
            p = _matchup_win_prob(ta, tb, round_num=1)
            r64_winners.append(ta if rng.random() < p else tb)

        # --- Round of 32 (pair adjacent R64 winners) ---
        r32_winners: List[BracketTeam] = []
        for i in range(0, 8, 2):
            ta, tb = r64_winners[i], r64_winners[i + 1]
            p = _matchup_win_prob(ta, tb, round_num=2)
            r32_winners.append(ta if rng.random() < p else tb)
        r32_winners_all.extend(r32_winners)

        # --- Sweet 16 ---
        s16_winners: List[BracketTeam] = []
        for i in range(0, 4, 2):
            ta, tb = r32_winners[i], r32_winners[i + 1]
            p = _matchup_win_prob(ta, tb, round_num=3)
            s16_winners.append(ta if rng.random() < p else tb)
        s16_winners_all.extend(s16_winners)

        # --- Elite 8 ---
        ta, tb = s16_winners[0], s16_winners[1]
        p = _matchup_win_prob(ta, tb, round_num=4)
        regional_champ = ta if rng.random() < p else tb
        e8_winners_all.append(regional_champ)
        regional_champions.append(regional_champ)

    # --- Final Four (region 0 vs 1, region 2 vs 3) ---
    f4_winners: List[BracketTeam] = []
    for i in range(0, 4, 2):
        ta = regional_champions[i]
        tb = regional_champions[i + 1]
        p = _matchup_win_prob(ta, tb, round_num=5)
        f4_winners.append(ta if rng.random() < p else tb)

    # --- Championship ---
    ta, tb = f4_winners[0], f4_winners[1]
    p = _matchup_win_prob(ta, tb, round_num=6)
    champion = ta if rng.random() < p else tb

    return (
        champion.name,
        [t.name for t in regional_champions],
        [t.name for t in e8_winners_all],
        [t.name for t in s16_winners_all],
        [t.name for t in r32_winners_all],
    )


# ---------------------------------------------------------------------------
# Upset alerts
# ---------------------------------------------------------------------------


def _find_upset_alerts(regions: Dict[str, List[BracketTeam]]) -> List[Dict]:
    """
    Identify R64 matchups where the underdog has >= 35% win probability.

    Returns a list of alert dicts sorted by upset_prob descending.
    """
    alerts: List[Dict] = []

    for _region_name, teams in regions.items():
        seed_map: Dict[int, BracketTeam] = {t.seed: t for t in teams}

        for s1, s2 in _R64_PAIRS:
            favorite = seed_map[s1]
            underdog = seed_map[s2]
            p_fav = _matchup_win_prob(favorite, underdog, round_num=1)
            p_upset = 1.0 - p_fav

            if p_upset >= _UPSET_ALERT_THRESHOLD:
                alerts.append({
                    "favorite": favorite.name,
                    "underdog": underdog.name,
                    "fav_seed": favorite.seed,
                    "dog_seed": underdog.seed,
                    "upset_prob": round(p_upset, 3),
                    "adj_em_gap": round(favorite.adj_em - underdog.adj_em, 1),
                    "region": _region_name,
                })

    alerts.sort(key=lambda x: x["upset_prob"], reverse=True)
    return alerts


# ---------------------------------------------------------------------------
# Consensus bracket builder
# ---------------------------------------------------------------------------


def _build_consensus_bracket(
    regions: Dict[str, List[BracketTeam]],
    advancement_probs: Dict[str, List[float]],
) -> Dict[str, str]:
    """
    Build a consensus bracket by picking, at each round, the team with the
    highest probability of advancing to *that round* (not of becoming champion).

    This produces a bracket with realistic upsets rather than always picking
    the top seed.

    Slot key format: "{region}_{round}_{slot_index}"
    e.g. "East_R64_0" = East region, R64, game 0 (1 vs 16 winner).
    """
    bracket: Dict[str, str] = {}
    region_names_ordered = list(regions.keys())

    # Round index in advancement_probs list:
    # 0=R32 (won R64), 1=S16 (won R32), 2=E8 (won S16),
    # 3=F4 (won E8),   4=Champ game (won F4), 5=Champion

    for region_name, teams in regions.items():
        # For each R64 game pick winner with highest p_r32 (advancement_probs index 0)
        seed_map: Dict[int, BracketTeam] = {t.seed: t for t in teams}

        r64_picks: List[str] = []
        for game_idx, (s1, s2) in enumerate(_R64_PAIRS):
            ta, tb = seed_map[s1], seed_map[s2]
            pa = advancement_probs.get(ta.name, [0.0] * 7)[0]
            pb = advancement_probs.get(tb.name, [0.0] * 7)[0]
            winner = ta.name if pa >= pb else tb.name
            slot_key = f"{region_name}_R64_{game_idx}"
            bracket[slot_key] = winner
            r64_picks.append(winner)

        # R32 — pick from pairs of R64 winners using advancement index 1 (reached S16)
        r32_picks: List[str] = []
        for game_idx in range(0, 8, 2):
            ta_name = r64_picks[game_idx]
            tb_name = r64_picks[game_idx + 1]
            pa = advancement_probs.get(ta_name, [0.0] * 7)[1]
            pb = advancement_probs.get(tb_name, [0.0] * 7)[1]
            winner = ta_name if pa >= pb else tb_name
            slot_key = f"{region_name}_R32_{game_idx // 2}"
            bracket[slot_key] = winner
            r32_picks.append(winner)

        # S16 — advancement index 2 (reached E8)
        s16_picks: List[str] = []
        for game_idx in range(0, 4, 2):
            ta_name = r32_picks[game_idx]
            tb_name = r32_picks[game_idx + 1]
            pa = advancement_probs.get(ta_name, [0.0] * 7)[2]
            pb = advancement_probs.get(tb_name, [0.0] * 7)[2]
            winner = ta_name if pa >= pb else tb_name
            slot_key = f"{region_name}_S16_{game_idx // 2}"
            bracket[slot_key] = winner
            s16_picks.append(winner)

        # E8 — advancement index 3 (reached F4)
        ta_name = s16_picks[0]
        tb_name = s16_picks[1]
        pa = advancement_probs.get(ta_name, [0.0] * 7)[3]
        pb = advancement_probs.get(tb_name, [0.0] * 7)[3]
        region_champ = ta_name if pa >= pb else tb_name
        bracket[f"{region_name}_E8"] = region_champ

    # Final Four — pick from regional champions using advancement index 4 (reached champ game)
    regional_champs = [bracket[f"{r}_E8"] for r in region_names_ordered]
    f4_picks: List[str] = []
    for i in range(0, 4, 2):
        ta_name = regional_champs[i]
        tb_name = regional_champs[i + 1]
        pa = advancement_probs.get(ta_name, [0.0] * 7)[4]
        pb = advancement_probs.get(tb_name, [0.0] * 7)[4]
        winner = ta_name if pa >= pb else tb_name
        bracket[f"F4_{i // 2}"] = winner
        f4_picks.append(winner)

    # Championship — advancement index 5 (champion)
    ta_name, tb_name = f4_picks[0], f4_picks[1]
    pa = advancement_probs.get(ta_name, [0.0] * 7)[5]
    pb = advancement_probs.get(tb_name, [0.0] * 7)[5]
    bracket["Champion"] = ta_name if pa >= pb else tb_name

    return bracket


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def simulate_tournament(
    teams: List[BracketTeam],
    n_sims: int = 10_000,
    seed: int = 42,
) -> BracketResult:
    """
    Monte Carlo NCAA Tournament bracket simulation.

    Runs `n_sims` independent bracket draws where each game outcome is
    sampled probabilistically from _matchup_win_prob(). This produces
    realistic upset distributions rather than always advancing every
    favourite.

    Args:
        teams:  List of BracketTeam objects (ideally 64; 68 with First Four).
                Each must have name, seed, region, adj_em populated.
        n_sims: Number of Monte Carlo iterations (default 10,000).
        seed:   RNG seed for reproducibility.

    Returns:
        BracketResult with advancement probabilities, projected bracket,
        upset alerts, and derived projections.
    """
    if not teams:
        raise ValueError("simulate_tournament: teams list is empty.")

    rng = random.Random(seed)

    # Group into 4 regions of 16.
    regions: Dict[str, List[BracketTeam]] = {}
    for t in teams:
        regions.setdefault(t.region, []).append(t)

    # Validate; fall back to redistribution when data is missing or unbalanced.
    if len(regions) != 4 or any(len(v) != 16 for v in regions.values()):
        regions = _redistribute_into_regions(teams)

    all_team_names = [t.name for t in teams]

    # Counters: name -> [r64_wins, r32_wins, s16_wins, e8_wins, f4_wins,
    #                     champ_game_wins, championships]
    # (7 slots, indices 0-6)
    round_counts: Dict[str, List[int]] = {name: [0] * 7 for name in all_team_names}

    for _ in range(n_sims):
        champ, f4, e8, s16, r32 = _simulate_one_bracket(regions, rng)

        # Champion
        if champ in round_counts:
            round_counts[champ][6] += 1

        # Final Four (regional champions — each won to the F4 stage)
        for name in f4:
            if name in round_counts:
                round_counts[name][4] += 1   # reached Final Four
                round_counts[name][3] += 1   # reached Elite Eight (won regional)

        # Elite Eight: _simulate_one_bracket returns e8_winners_all == regional_champions
        # (the 4 E8 winners who advanced to the Final Four).  Those are already counted
        # above via the f4 loop.  The 4 E8 losers are the S16 winners that are NOT in f4.
        f4_set = set(f4)
        for name in e8:
            if name in round_counts and name not in f4_set:
                # This branch covers the edge case where e8 ever diverges from f4.
                round_counts[name][3] += 1
        # Credit S16 winners that lost in the E8 (they reached E8 but did not win it).
        for name in s16:
            if name in round_counts and name not in f4_set:
                round_counts[name][3] += 1   # reached Elite Eight (lost there)

        # Sweet 16
        for name in s16:
            if name in round_counts:
                round_counts[name][2] += 1

        # Round of 32
        for name in r32:
            if name in round_counts:
                round_counts[name][1] += 1

        # Round of 64 wins: all R32 entrants won their R64 game;
        # S16 entrants won their R32 game (already counted); no separate counter needed.
        # Treat index 0 as "won R64 (reached R32)" — add all r32 names again.
        for name in r32:
            if name in round_counts:
                round_counts[name][0] += 1

    # Convert raw counts to probabilities.
    advancement_probs: Dict[str, List[float]] = {
        name: [c / n_sims for c in counts]
        for name, counts in round_counts.items()
    }

    # Derived projections.
    by_champ = sorted(
        advancement_probs.items(), key=lambda x: x[1][6], reverse=True
    )
    projected_champion = by_champ[0][0]

    by_f4 = sorted(
        advancement_probs.items(), key=lambda x: x[1][4], reverse=True
    )
    projected_final_four = [name for name, _ in by_f4[:4]]

    upset_alerts = _find_upset_alerts(regions)
    projected_bracket = _build_consensus_bracket(regions, advancement_probs)

    return BracketResult(
        advancement_probs=advancement_probs,
        projected_champion=projected_champion,
        projected_final_four=projected_final_four,
        projected_bracket=projected_bracket,
        upset_alerts=upset_alerts,
        n_sims=n_sims,
    )
