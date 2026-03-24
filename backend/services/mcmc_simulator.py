"""
OOP wrapper around backend/fantasy_baseball/mcmc_simulator.py.
Provides MCMCWeeklySimulator class with dataclass-typed inputs/outputs.
"""
from __future__ import annotations
import dataclasses
from typing import Optional
from backend.fantasy_baseball.mcmc_simulator import (
    simulate_weekly_matchup,
    simulate_roster_move,
    _MCMC_DISABLED,
)


@dataclasses.dataclass
class PlayerCategoryDistribution:
    name: str
    positions: list[str]
    cat_scores: dict[str, float]
    starts_this_week: int = 1

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MatchupSimulationResult:
    win_prob: float
    category_win_probs: dict[str, float]
    expected_cats_won: float
    n_sims: int
    elapsed_ms: float
    categories_simulated: list[str]


@dataclasses.dataclass
class RosterMoveSimulationResult:
    win_prob_before: float
    win_prob_after: float
    win_prob_gain: float
    category_win_probs_before: dict[str, float]
    category_win_probs_after: dict[str, float]
    mcmc_enabled: bool
    n_sims: int
    elapsed_ms: float


class MCMCWeeklySimulator:
    """
    Stateless wrapper providing a consistent OOP interface for H2H MCMC simulation.
    All computation delegated to the functional mcmc_simulator module.
    """

    def __init__(self, n_sims: int = 1000, seed: Optional[int] = None):
        self.n_sims = n_sims
        self.seed = seed

    def simulate_matchup(
        self,
        my_roster: list[dict],
        opponent_roster: list[dict],
    ) -> MatchupSimulationResult:
        raw = simulate_weekly_matchup(
            my_roster, opponent_roster,
            n_sims=self.n_sims, seed=self.seed,
        )
        return MatchupSimulationResult(**raw)

    def simulate_roster_move(
        self,
        my_roster: list[dict],
        opponent_roster: list[dict],
        add_player: dict,
        drop_player_name: str,
    ) -> RosterMoveSimulationResult:
        raw = simulate_roster_move(
            my_roster, opponent_roster, add_player, drop_player_name,
            n_sims=self.n_sims, seed=self.seed,
        )
        return RosterMoveSimulationResult(
            win_prob_before=raw["win_prob_before"],
            win_prob_after=raw["win_prob_after"],
            win_prob_gain=raw["win_prob_gain"],
            category_win_probs_before=raw["category_win_probs_before"],
            category_win_probs_after=raw["category_win_probs_after"],
            mcmc_enabled=raw["mcmc_enabled"],
            n_sims=raw["n_sims"],
            elapsed_ms=raw["elapsed_ms"],
        )

    @staticmethod
    def disabled_result() -> RosterMoveSimulationResult:
        return RosterMoveSimulationResult(**{
            k: v for k, v in _MCMC_DISABLED.items()
            if k in RosterMoveSimulationResult.__dataclass_fields__
        })
