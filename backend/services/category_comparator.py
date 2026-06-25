"""
Unified Category Comparator Service — Single source of truth for win/loss/status calculations.

Provides consistent win/loss/tie verdicts across all modules (Waiver Wire, War Room,
My Roster, Dashboard, Preview). Eliminates the inverted "higher-is-better vs lower-is-better"
inconsistency that caused identical stats to show different verdicts across modules.

Created: Loop Iteration 13 (2026-06-25)
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category Direction Configuration
# ---------------------------------------------------------------------------

CATEGORY_DIRECTIONS: Dict[str, str] = {
    # Pitching categories (lower is better)
    "ERA": "lower",
    "WHIP": "lower",
    "HRA": "lower",          # Home runs allowed
    "L": "lower",            # Losses
    "K_allowed": "lower",    # Strikeouts allowed (if tracked separately)
    "BB_allowed": "lower",   # Walks allowed (if tracked separately)
    "ER_allowed": "lower",   # Earned runs allowed
    "HA": "lower",           # Hits allowed

    # Pitching categories (higher is better)
    "W": "higher",           # Wins
    "K": "higher",           # Strikeouts
    "SV": "higher",          # Saves
    "IP": "higher",          # Innings pitched
    "K_9": "higher",         # Strikeouts per 9 innings
    "QS": "higher",          # Quality starts
    "CG": "higher",          # Complete games
    "SHO": "higher",         # Shutouts

    # Hitting categories (higher is better)
    "HR": "higher",          # Home runs
    "R": "higher",           # Runs
    "RBI": "higher",         # Runs batted in
    "SB": "higher",          # Stolen bases
    "AVG": "higher",         # Batting average
    "OBP": "higher",         # On-base percentage
    "SLG": "higher",         # Slugging percentage
    "OPS": "higher",         # On-base + slugging
}


# Canonical namespace for categories (prevents duplicates)
CANONICAL_CATEGORIES: Dict[str, str] = {
    # Pitching (15 categories)
    "P_W": "W",              # Wins
    "P_L": "L",              # Losses
    "P_ERA": "ERA",          # Earned run average
    "P_WHIP": "WHIP",        # Walks + hits per inning pitched
    "P_K": "K",              # Strikeouts
    "P_SV": "SV",            # Saves
    "P_IP": "IP",            # Innings pitched
    "P_K_9": "K_9",          # Strikeouts per 9
    "P_QS": "QS",            # Quality starts
    "P_HRA": "HRA",          # Home runs allowed
    "P_BB_allowed": "BB_allowed",
    "P_ER_allowed": "ER_allowed",
    "P_HA": "HA",            # Hits allowed

    # Hitting (15 categories)
    "H_HR": "HR",            # Home runs
    "H_R": "R",              # Runs
    "H_RBI": "RBI",          # Runs batted in
    "H_SB": "SB",            # Stolen bases
    "H_AVG": "AVG",          # Batting average
    "H_OBP": "OBP",          # On-base percentage
    "H_SLG": "SLG",          # Slugging percentage
    "H_OPS": "OPS",          # On-base + slugging

    # Note: Some leagues use different category sets. This is the canonical 15-cat set.
    # For 18-cat leagues, add: H_K (hitting K), H_BB (hitting BB), etc.
}


# Gap thresholds for status determination (configurable)
BUBBLE_THRESHOLD_PCT = 0.10   # Within 10% = BUBBLE
PUNT_THRESHOLD_PCT = 0.25      # Behind 10-25% = BEHIND, at 25% = PUNT?
CONCEDE_THRESHOLD_PCT = 0.25   # Behind >25% = CONCEDE


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CategoryResult:
    """Result of category comparison."""
    verdict: str  # "W", "L", or "T" (win/loss/tie)
    gap: float   # Absolute difference (my_val - opp_val)
    gap_pct: Optional[float] = None  # Gap as percentage of opponent value


@dataclass
class CategoryStatus:
    """Extended status with gap-based assessment."""
    verdict: str          # "W", "L", or "T"
    status: str           # "AHEAD", "BEHIND", "BUBBLE", "PUNT?", "CONCEDE"
    gap: float            # Absolute difference
    gap_pct: float        # Gap as percentage of opponent value
    my_val: float         # My team's value
    opp_val: float        # Opponent's value


# ---------------------------------------------------------------------------
# Main Comparator Class
# ---------------------------------------------------------------------------

class CategoryComparator:
    """
    Unified category comparison with consistent win/loss/tie verdicts.

    Usage:
        comparator = CategoryComparator()

        # Basic comparison
        result = comparator.compare("ERA", 5.62, 2.74)
        # result.verdict = "L" (ERA is lower-is-better, my 5.62 > opp 2.74)

        # Extended status
        status = comparator.get_status("ERA", 5.62, 2.74)
        # status.status = "CONCEDE" (behind by >25%)
    """

    def __init__(
        self,
        bubble_threshold_pct: float = BUBBLE_THRESHOLD_PCT,
        punt_threshold_pct: float = PUNT_THRESHOLD_PCT,
        concede_threshold_pct: float = CONCEDE_THRESHOLD_PCT,
    ):
        self.bubble_threshold_pct = bubble_threshold_pct
        self.punt_threshold_pct = punt_threshold_pct
        self.concede_threshold_pct = concede_threshold_pct

    def compare(
        self,
        stat: str,
        my_val: float,
        opp_val: float,
    ) -> CategoryResult:
        """
        Compare category values and return win/loss/tie verdict.

        Args:
            stat: Category name (e.g., "ERA", "HR", "WHIP")
            my_val: My team's value
            opp_val: Opponent's value

        Returns:
            CategoryResult with verdict ("W", "L", or "T") and gap
        """
        try:
            direction = CATEGORY_DIRECTIONS.get(stat, "higher")  # Default to higher-is-better

            if direction == "higher":
                # Higher value wins
                if my_val > opp_val:
                    verdict = "W"
                elif my_val < opp_val:
                    verdict = "L"
                else:
                    verdict = "T"
            else:
                # Lower value wins
                if my_val < opp_val:
                    verdict = "W"
                elif my_val > opp_val:
                    verdict = "L"
                else:
                    verdict = "T"

            gap = my_val - opp_val
            gap_pct = self._calculate_gap_pct(my_val, opp_val)

            return CategoryResult(verdict=verdict, gap=gap, gap_pct=gap_pct)

        except Exception as exc:
            logger.warning(f"Category comparison failed for {stat}: {exc}")
            # Fallback: return tie verdict
            return CategoryResult(verdict="T", gap=0.0, gap_pct=0.0)

    def get_status(
        self,
        stat: str,
        my_val: float,
        opp_val: float,
    ) -> CategoryStatus:
        """
        Get extended status with gap-based assessment.

        Args:
            stat: Category name (e.g., "ERA", "HR", "WHIP")
            my_val: My team's value
            opp_val: Opponent's value

        Returns:
            CategoryStatus with verdict, status (AHEAD/BEHIND/BUBBLE/PUNT?/CONCEDE), and gap info
        """
        result = self.compare(stat, my_val, opp_val)
        gap_pct = result.gap_pct or 0.0

        # Determine status based on verdict and gap percentage
        if result.verdict == "T":
            status = "BUBBLE"
        elif result.verdict == "W":
            if abs(gap_pct) <= self.bubble_threshold_pct:
                status = "BUBBLE"
            else:
                status = "AHEAD"
        else:  # result.verdict == "L"
            if abs(gap_pct) <= self.bubble_threshold_pct:
                status = "BUBBLE"
            elif abs(gap_pct) < self.punt_threshold_pct:
                status = "BEHIND"
            elif abs(gap_pct) == self.concede_threshold_pct:
                status = "PUNT?"
            else:
                status = "CONCEDE"

        return CategoryStatus(
            verdict=result.verdict,
            status=status,
            gap=result.gap,
            gap_pct=gap_pct,
            my_val=my_val,
            opp_val=opp_val,
        )

    def _calculate_gap_pct(self, my_val: float, opp_val: float) -> float:
        """Calculate gap as percentage of opponent value."""
        if opp_val == 0:
            return 0.0
        return abs(my_val - opp_val) / abs(opp_val)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_comparator_instance: Optional[CategoryComparator] = None


def get_category_comparator() -> CategoryComparator:
    """Get the singleton CategoryComparator instance."""
    global _comparator_instance
    if _comparator_instance is None:
        _comparator_instance = CategoryComparator()
    return _comparator_instance


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compare_category(stat: str, my_val: float, opp_val: float) -> CategoryResult:
    """
    Compare category values and return win/loss/tie verdict.

    Convenience function that uses the singleton comparator.

    Args:
        stat: Category name (e.g., "ERA", "HR", "WHIP")
        my_val: My team's value
        opp_val: Opponent's value

    Returns:
        CategoryResult with verdict ("W", "L", or "T") and gap

    Example:
        result = compare_category("ERA", 5.62, 2.74)
        # result.verdict = "L"
        # result.gap = 2.88
    """
    comparator = get_category_comparator()
    return comparator.compare(stat, my_val, opp_val)


def category_status(stat: str, my_val: float, opp_val: float) -> CategoryStatus:
    """
    Get extended status with gap-based assessment.

    Convenience function that uses the singleton comparator.

    Args:
        stat: Category name (e.g., "ERA", "HR", "WHIP")
        my_val: My team's value
        opp_val: Opponent's value

    Returns:
        CategoryStatus with verdict, status (AHEAD/BEHIND/BUBBLE/PUNT?/CONCEDE), and gap info

    Example:
        status = category_status("HR", 15, 10)
        # status.verdict = "W"
        # status.status = "AHEAD"
        # status.gap = 5.0
    """
    comparator = get_category_comparator()
    return comparator.get_status(stat, my_val, opp_val)


def get_canonical_category(raw_category: str) -> str:
    """
    Get canonical category name from raw display string.

    Handles namespace prefixes and common variations.

    Args:
        raw_category: Raw category string (e.g., "K", "P_K", "ERA", "P_ERA")

    Returns:
        Canonical category name

    Example:
        get_canonical_category("P_K")  # Returns "K" (pitching K)
        get_canonical_category("H_K")  # Returns "K" (hitting K)
        get_canonical_category("ERA")  # Returns "ERA"
    """
    # If already canonical, return as-is
    if raw_category in CATEGORY_DIRECTIONS:
        return raw_category

    # If namespaced, extract the base category
    if "_" in raw_category:
        parts = raw_category.split("_", 1)
        if len(parts) == 2:
            return parts[1]

    # Default: return as-is (will use "higher" direction if unknown)
    return raw_category
