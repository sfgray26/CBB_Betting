"""
H2H One Win Monte Carlo Simulator

Category-by-category win probability simulation for H2H One Win fantasy format.
Returns P(win 10+ categories) instead of projected points.

Performance target: 10,000 sims <200ms via NumPy vectorization.

v2 Alignment:
- Uses canonical codes from stat_contract.SCORING_CATEGORY_CODES
- 18 categories: 9 batting, 9 pitching
- Win threshold: 10 (majority of 18)
- LOWER_IS_BETTER from stat_contract for ERA, WHIP, K_B, L, HR_P

Usage:
    sim = H2HOneWinSimulator()
    result = sim.simulate_week(my_roster, opponent_roster, n_sims=10000)
    # OR from pre-aggregated projections:
    result = sim.simulate_week_from_projections(my_finals, opp_finals, n_sims=10000)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import date

from backend.stat_contract import SCORING_CATEGORY_CODES, BATTING_CODES, PITCHING_CODES, LOWER_IS_BETTER


@dataclass
class H2HWinResult:
    """Result of H2H One Win Monte Carlo simulation."""

    win_probability: float  # P(win 10+ categories) [0.0, 1.0]

    locked_categories: List[str]  # >85% win probability
    swing_categories: List[str]  # 40-60% win probability (key matchups)
    vulnerable_categories: List[str]  # <30% win probability (risk zones)

    category_win_probs: Dict[str, float]  # Full breakdown: e.g. {"R": 0.72, "HR_B": 0.45}

    mean_categories_won: float  # Expected categories won (e.g., 9.2 / 18)
    std_categories_won: float  # Std dev (measure of volatility)

    n_simulations: int
    as_of_date: date

    # v2: Original input projections for audit trail
    my_input_projections: Optional[Dict[str, float]] = None
    opp_input_projections: Optional[Dict[str, float]] = None


@dataclass
class SwingCategory:
    """A category with close win probability (40-60%) — key matchup decision point."""

    category: str
    my_win_prob: float
    opponent_win_prob: float
    recommendation: str  # "STREAM_HITTER", "STREAM_PITCHER", "RIDE_OR_DIE"


class H2HOneWinSimulator:
    """
    Monte Carlo simulation for H2H One Win fantasy format.

    Simulates N iterations of weekly stats for both rosters, comparing
    category-by-category to determine win probability (10+ = win).

    v2 Categories (18 total):
    Batting (9): R, H, HR_B, RBI, K_B, TB, AVG, OPS, NSB
    Pitching (9): W, L, HR_P, K_P, ERA, WHIP, K_9, QS, NSV

    Performance: NumPy vectorization targets <200ms for 10,000 sims.
    """

    # Win threshold: majority of 18 categories
    WIN_THRESHOLD = 10

    # v2 canonical codes - reference from stat_contract
    HITTING_CATS = sorted(list(BATTING_CODES))  # ["AVG", "H", "HR_B", "K_B", "NSB", "OPS", "R", "RBI", "TB"]
    PITCHING_CATS = sorted(list(PITCHING_CODES))  # ["ERA", "HR_P", "K_9", "K_P", "L", "NSV", "QS", "W", "WHIP"]

    # Standard deviation for stat projection (game-to-game variance)
    # Conservative: CV=0.35 for counting stats, 0.15 for rate stats
    STAT_CV = {
        # Batting counting stats
        "R": 0.35,
        "H": 0.30,
        "HR_B": 0.40,
        "RBI": 0.35,
        "K_B": 0.25,
        "TB": 0.35,
        "NSB": 0.50,
        # Pitching counting stats
        "W": 0.30,
        "L": 0.30,
        "HR_P": 0.40,
        "K_P": 0.25,
        "QS": 0.25,
        "NSV": 0.45,
        # Rate stats (lower variance)
        "AVG": 0.08,
        "OPS": 0.10,
        "ERA": 0.15,
        "WHIP": 0.12,
        "K_9": 0.12,
    }

    def simulate_week(
        self,
        my_roster: List[Dict[str, Any]],
        opponent_roster: List[Dict[str, Any]],
        n_sims: int = 10000,
        as_of_date: date = None,
    ) -> H2HWinResult:
        """
        Run N Monte Carlo simulations of the weekly matchup.

        Args:
            my_roster: List of player dicts with projected stats
                Example: [{"name": "Ohtani", "R": 15, "HR_B": 4, ...}, ...]
            opponent_roster: List of player dicts with projected stats
            n_sims: Number of simulations (default: 10000)
            as_of_date: Date for the simulation week

        Returns:
            H2HWinResult with win probability and category breakdown
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Aggregate projected stats for both rosters
        my_proj = self._aggregate_roster(my_roster)
        opp_proj = self._aggregate_roster(opponent_roster)

        # Run Monte Carlo simulation (returns both total wins and per-category matrix)
        categories_won, category_win_matrix = self._run_simulation(my_proj, opp_proj, n_sims)

        # Analyze results
        win_prob = np.mean(categories_won >= self.WIN_THRESHOLD)
        locked, swing, vulnerable = self._classify_categories(category_win_matrix)

        return H2HWinResult(
            win_probability=float(win_prob),
            locked_categories=locked,
            swing_categories=swing,
            vulnerable_categories=vulnerable,
            category_win_probs=self._compute_category_probs(category_win_matrix),
            mean_categories_won=float(np.mean(categories_won)),
            std_categories_won=float(np.std(categories_won)),
            n_simulations=n_sims,
            as_of_date=as_of_date,
            my_input_projections=my_proj,
            opp_input_projections=opp_proj,
        )

    def simulate_week_from_projections(
        self,
        my_finals: Dict[str, float],
        opp_finals: Dict[str, float],
        n_sims: int = 10000,
        as_of_date: date = None,
    ) -> H2HWinResult:
        """
        Run Monte Carlo simulation from pre-aggregated team projections.

        This is the v2 entry point for ROW → Simulation bridge.
        Accepts final projected totals (e.g., from ROWProjectionResult.to_dict()).

        Args:
            my_finals: Dict of canonical_code -> projected total for my team
            opp_finals: Dict of canonical_code -> projected total for opponent
            n_sims: Number of simulations (default: 10000)
            as_of_date: Date for the simulation week

        Returns:
            H2HWinResult with win probability and category breakdown
        """
        if as_of_date is None:
            as_of_date = date.today()

        # Validate inputs contain all 18 categories
        my_set = set(my_finals.keys())
        opp_set = set(opp_finals.keys())
        if my_set != SCORING_CATEGORY_CODES:
            raise ValueError(f"my_finals missing categories: {SCORING_CATEGORY_CODES - my_set}")
        if opp_set != SCORING_CATEGORY_CODES:
            raise ValueError(f"opp_finals missing categories: {SCORING_CATEGORY_CODES - opp_set}")

        # Run Monte Carlo simulation directly from projections
        categories_won, category_win_matrix = self._run_simulation(my_finals, opp_finals, n_sims)

        # Analyze results
        win_prob = np.mean(categories_won >= self.WIN_THRESHOLD)
        locked, swing, vulnerable = self._classify_categories(category_win_matrix)

        return H2HWinResult(
            win_probability=float(win_prob),
            locked_categories=locked,
            swing_categories=swing,
            vulnerable_categories=vulnerable,
            category_win_probs=self._compute_category_probs(category_win_matrix),
            mean_categories_won=float(np.mean(categories_won)),
            std_categories_won=float(np.std(categories_won)),
            n_simulations=n_sims,
            as_of_date=as_of_date,
            my_input_projections=my_finals,
            opp_input_projections=opp_finals,
        )

    def _aggregate_roster(self, roster: List[Dict[str, Any]]) -> Dict[str, float]:
        """Sum projected stats across all players in roster."""
        aggregates = {cat: 0.0 for cat in self.HITTING_CATS + self.PITCHING_CATS}

        for player in roster:
            for cat in aggregates:
                aggregates[cat] += player.get(cat, 0.0)

        return aggregates

    def _run_simulation(
        self,
        my_proj: Dict[str, float],
        opp_proj: Dict[str, float],
        n_sims: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Monte Carlo simulation using NumPy vectorization.

        For each category:
          1. Sample stats from normal distribution (mean=proj, std=CV*mean)
          2. Compare my vs opponent (respects LOWER_IS_BETTER from stat_contract)
          3. Count categories won per simulation

        Returns:
            categories_won: Array of shape (n_sims,) with categories won per sim
            category_win_matrix: Matrix of shape (n_sims, n_categories) with per-category results
        """
        categories = self.HITTING_CATS + self.PITCHING_CATS
        n_categories = len(categories)

        # Pre-allocate results matrix: n_sims x n_categories
        category_win_matrix = np.zeros((n_sims, n_categories))

        for i, cat in enumerate(categories):
            # Get projections
            my_mean = my_proj.get(cat, 0.0)
            opp_mean = opp_proj.get(cat, 0.0)

            # Skip if both zeros (no data)
            if my_mean == 0 and opp_mean == 0:
                category_win_matrix[:, i] = 0.5  # Tie = half win
                continue

            # Compute std from CV
            cv = self.STAT_CV.get(cat, 0.35)
            my_std = max(my_mean * cv, 0.1)  # Minimum std to avoid zero variance
            opp_std = max(opp_mean * cv, 0.1)

            # Sample from normal distribution (vectorized)
            my_samples = np.random.normal(my_mean, my_std, n_sims)
            opp_samples = np.random.normal(opp_mean, opp_std, n_sims)

            # v2: Use LOWER_IS_BETTER from stat_contract
            if cat in LOWER_IS_BETTER:
                # Lower is better (ERA, WHIP, K_B, L, HR_P)
                category_win_matrix[:, i] = (my_samples < opp_samples).astype(float)
            else:
                # Higher is better (all other categories)
                category_win_matrix[:, i] = (my_samples > opp_samples).astype(float)

        # Sum categories won per simulation
        categories_won = np.sum(category_win_matrix, axis=1)

        return categories_won, category_win_matrix

    def _classify_categories(
        self, category_win_matrix: np.ndarray
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Classify categories by win probability.

        Locked: >85% win (safe)
        Swing: 40-60% win (key matchups)
        Vulnerable: <30% win (risk zones)
        """
        locked = []
        swing = []
        vulnerable = []

        # Compute category probabilities from simulation matrix
        category_probs = self._compute_category_probs(category_win_matrix)

        for cat, prob in category_probs.items():
            if prob > 0.85:
                locked.append(cat)
            elif prob < 0.30:
                vulnerable.append(cat)
            elif 0.40 <= prob <= 0.60:
                swing.append(cat)

        return locked, swing, vulnerable

    def _compute_category_probs(
        self, category_win_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute win probability per category from simulation matrix.

        Args:
            category_win_matrix: Matrix of shape (n_sims, n_categories)

        Returns:
            Dict mapping category name to win probability [0.0, 1.0]
        """
        categories = self.HITTING_CATS + self.PITCHING_CATS
        category_probs = {}

        for i, cat in enumerate(categories):
            # Mean win rate for this category across all simulations
            win_rate = np.mean(category_win_matrix[:, i])
            category_probs[cat] = float(win_rate)

        return category_probs
