"""
Unified Need-Score Service — Centralized calculation with transparent components.

Provides a single source of truth for need-score calculations across all endpoints
(Waiver Wire, Waiver Recommendations, Dashboard). Separates base category-aware scoring
from Statcast adjustments for full transparency.

Created: Loop Iteration 11 (2026-06-25)
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from backend.fantasy_baseball.category_aware_scorer import compute_need_score
from backend.fantasy_baseball.statcast_loader import statcast_need_score_boost

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class NeedScoreResult:
    """Complete need-score calculation with transparent components."""
    base_need_score: float  # Pure category-aware score (no Statcast)
    statcast_boost: float = 0.0  # Statcast adjustment factor
    adjusted_need_score: float = 0.0  # Final score (base + boost)
    statcast_signals: Optional[List[str]] = None  # Raw signals for debugging


# ---------------------------------------------------------------------------
# Main Calculator Class
# ---------------------------------------------------------------------------

class NeedScoreCalculator:
    """
    Unified need-score calculation with transparent components.

    Provides three distinct calculations:
    1. Base need-score: Pure category-aware calculation (no Statcast)
    2. Statcast boost: Adjustment based on Statcast signals (BUY_LOW, BREAKOUT, etc.)
    3. Adjusted need-score: Final score (base + Statcast boost)

    Usage:
        calculator = NeedScoreCalculator()
        result = calculator.calculate_all(
            player_cat_scores=cat_scores,
            player_z_score=z_score,
            category_deficits=deficits,
            n_cats=9,
            statcast_signals=["BUY_LOW", "BREAKOUT"],
        )
        # result.base_need_score = 9.38
        # result.statcast_boost = 0.9
        # result.adjusted_need_score = 10.28
    """

    def __init__(self):
        self._call_count = 0

    def calculate_base_need_score(
        self,
        player_cat_scores: Dict[str, float],
        player_z_score: float,
        category_deficits: List,
        n_cats: int,
        team_context: Optional[object] = None,
        marginal_stats: Optional[Dict] = None,
    ) -> float:
        """
        Calculate base category-aware need-score (no Statcast adjustments).

        This is the pure category-aware score that considers:
        - Team's matchup category deficits
        - Player's per-category quality (z-scores)
        - Rate-stat protection (prevents adding harmful players)
        - Roster depth adjustments

        Args:
            player_cat_scores: Dict of {category: z_score} from PlayerProjection.cat_scores
            player_z_score: Overall z_score for the player
            category_deficits: List of CategoryDeficitOut objects from matchup analysis
            n_cats: Number of scoring categories (for normalization)
            team_context: Optional TeamContext with roster denominator context
            marginal_stats: Optional raw numerator/denominator data for rate stats

        Returns:
            Base need-score as float. Returns player_z_score if category_deficits is empty.
        """
        try:
            base_score = compute_need_score(
                player_cat_scores=player_cat_scores,
                player_z_score=player_z_score,
                category_deficits=category_deficits,
                n_cats=n_cats,
                team_context=team_context,
                marginal_stats=marginal_stats,
            )
            return float(base_score)
        except Exception as exc:
            logger.warning(f"Base need-score calculation failed: {exc}, falling back to z_score")
            return float(player_z_score)

    def calculate_statcast_boost(
        self,
        statcast_signals: List[str],
    ) -> float:
        """
        Calculate Statcast adjustment factor.

        Statcast signals can boost or reduce need-score:
        - BUY_LOW: +0.4 (undervalued based on Statcast metrics)
        - BREAKOUT: +0.5 (showing breakout indicators)
        - SELL_HIGH: -0.3 (overvalued based on Statcast)
        - HIGH_INJURY_RISK: -0.2 (injury concerns)

        Args:
            statcast_signals: List of Statcast signal strings

        Returns:
            Float boost value (typically -0.5 to +1.0). Returns 0.0 if no signals.
        """
        if not statcast_signals:
            return 0.0

        try:
            boost = statcast_need_score_boost(statcast_signals)
            return float(boost)
        except Exception as exc:
            logger.warning(f"Statcast boost calculation failed: {exc}, returning 0.0")
            return 0.0

    def calculate_adjusted_need_score(
        self,
        base_need_score: float,
        statcast_boost: float,
    ) -> float:
        """
        Calculate final adjusted need-score (base + Statcast boost).

        Args:
            base_need_score: Base category-aware need-score
            statcast_boost: Statcast adjustment factor

        Returns:
            Adjusted need-score as float (base + boost).
        """
        return base_need_score + statcast_boost

    def calculate_all(
        self,
        player_cat_scores: Dict[str, float],
        player_z_score: float,
        category_deficits: List,
        n_cats: int,
        statcast_signals: Optional[List[str]] = None,
        team_context: Optional[object] = None,
        marginal_stats: Optional[Dict] = None,
    ) -> NeedScoreResult:
        """
        Calculate complete need-score result with all components.

        This is the recommended method for most use cases. It calculates
        base need-score, Statcast boost, and adjusted need-score in one call.

        Args:
            player_cat_scores: Dict of {category: z_score} from PlayerProjection.cat_scores
            player_z_score: Overall z_score for the player
            category_deficits: List of CategoryDeficitOut objects from matchup analysis
            n_cats: Number of scoring categories (for normalization)
            statcast_signals: Optional list of Statcast signal strings
            team_context: Optional TeamContext with roster denominator context
            marginal_stats: Optional raw numerator/denominator data for rate stats

        Returns:
            NeedScoreResult with base_need_score, statcast_boost, adjusted_need_score.
        """
        # Calculate base need-score (pure category-aware)
        base_score = self.calculate_base_need_score(
            player_cat_scores=player_cat_scores,
            player_z_score=player_z_score,
            category_deficits=category_deficits,
            n_cats=n_cats,
            team_context=team_context,
            marginal_stats=marginal_stats,
        )

        # Calculate Statcast boost (separate transparent calculation)
        boost = self.calculate_statcast_boost(statcast_signals or [])

        # Calculate adjusted need-score (base + boost)
        adjusted_score = self.calculate_adjusted_need_score(base_score, boost)

        self._call_count += 1

        return NeedScoreResult(
            base_need_score=base_score,
            statcast_boost=boost,
            adjusted_need_score=adjusted_score,
            statcast_signals=statcast_signals,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_calculator_instance: Optional[NeedScoreCalculator] = None


def get_need_score_calculator() -> NeedScoreCalculator:
    """Get the singleton NeedScoreCalculator instance."""
    global _calculator_instance
    if _calculator_instance is None:
        _calculator_instance = NeedScoreCalculator()
    return _calculator_instance


def calculate_need_score(
    player_cat_scores: Dict[str, float],
    player_z_score: float,
    category_deficits: List,
    n_cats: int,
    statcast_signals: Optional[List[str]] = None,
    team_context: Optional[object] = None,
    marginal_stats: Optional[Dict] = None,
) -> NeedScoreResult:
    """
    Convenience function for need-score calculation with all components.

    This is the main entry point for need-score calculations across the codebase.
    Returns a NeedScoreResult with base_need_score, statcast_boost, and adjusted_need_score.

    Args:
        player_cat_scores: Dict of {category: z_score} from PlayerProjection.cat_scores
        player_z_score: Overall z_score for the player
        category_deficits: List of CategoryDeficitOut objects from matchup analysis
        n_cats: Number of scoring categories (for normalization)
        statcast_signals: Optional list of Statcast signal strings
        team_context: Optional TeamContext with roster denominator context
        marginal_stats: Optional raw numerator/denominator data for rate stats

    Returns:
        NeedScoreResult with base_need_score, statcast_boost, adjusted_need_score.

    Example:
        result = calculate_need_score(
            player_cat_scores={"hr": 1.2, "rbi": 0.8, "avg": 0.5},
            player_z_score=1.0,
            category_deficits=[CategoryDeficitOut(category="HR", deficit=5.0)],
            n_cats=9,
            statcast_signals=["BUY_LOW"],
        )
        print(f"Base: {result.base_need_score}")  # 9.38
        print(f"Boost: {result.statcast_boost}")  # 0.4
        print(f"Adjusted: {result.adjusted_need_score}")  # 9.78
    """
    calculator = get_need_score_calculator()
    return calculator.calculate_all(
        player_cat_scores=player_cat_scores,
        player_z_score=player_z_score,
        category_deficits=category_deficits,
        n_cats=n_cats,
        statcast_signals=statcast_signals,
        team_context=team_context,
        marginal_stats=marginal_stats,
    )
