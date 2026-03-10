"""
Late-Season Recency Weighting — P3

Weights recent games more heavily in March when form is more predictive.
Tournament seeding is heavily influenced by late-season performance.

Usage:
    from backend.services.recency_weight import get_recency_weight, is_late_season
    
    # Check if we're in late season (March 1+ or tournament)
    if is_late_season(date.today()):
        weight = get_recency_weight(days_ago=10)  # Last 10 days
        # Apply 2x weight to recent games
"""

import logging
from datetime import date, datetime
from typing import Dict, Optional, Tuple

from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)


# Late season configuration
LATE_SEASON_START_MONTH = 3  # March
LATE_SEASON_START_DAY = 1
TOURNAMENT_MODE_THRESHOLD = 15  # March 15 = Selection Sunday week

# Recency weight tiers (days ago -> weight multiplier)
# Used during late season to weight recent games more heavily
DEFAULT_RECENCY_WEIGHTS: Dict[int, float] = {
    0: 2.0,    # Today
    1: 2.0,    # Yesterday
    2: 2.0,    # 2 days ago
    3: 1.9,
    4: 1.8,
    5: 1.7,
    6: 1.6,
    7: 1.6,    # Last week
    8: 1.5,
    9: 1.5,
    10: 1.4,
    11: 1.4,
    12: 1.3,
    13: 1.3,
    14: 1.2,   # Two weeks ago
    15: 1.2,
    16: 1.1,
    17: 1.1,
    18: 1.1,
    19: 1.0,
    20: 1.0,
    21: 1.0,   # Three weeks ago
}

# Flat weight outside late season
REGULAR_SEASON_WEIGHT = 1.0


def is_late_season(
    check_date: Optional[date] = None,
    tournament_mode_threshold: int = TOURNAMENT_MODE_THRESHOLD,
) -> bool:
    """
    Check if we're in late season (March 1 or later).
    
    Args:
        check_date: Date to check (default: today)
        tournament_mode_threshold: Day in March for tournament mode
        
    Returns:
        True if in late season (March 1+)
    """
    if check_date is None:
        check_date = date.today()
    
    # Late season = March 1 or later in the calendar year
    # (Assumes college basketball season spans parts of two calendar years)
    if check_date.month >= LATE_SEASON_START_MONTH:
        return True
    
    return False


def is_tournament_mode(
    check_date: Optional[date] = None,
    threshold_day: int = TOURNAMENT_MODE_THRESHOLD,
) -> bool:
    """
    Check if we're in tournament mode (March 15+ or neutral site tournament).
    
    Args:
        check_date: Date to check (default: today)
        threshold_day: Day of March that triggers tournament mode
        
    Returns:
        True if in tournament mode
    """
    if check_date is None:
        check_date = date.today()
    
    if check_date.month == LATE_SEASON_START_MONTH and check_date.day >= threshold_day:
        return True
    
    # Also tournament mode for any April games (tournament finals)
    if check_date.month > LATE_SEASON_START_MONTH:
        return True
    
    return False


def get_recency_weight(
    days_ago: int,
    is_late_season_flag: Optional[bool] = None,
    custom_weights: Optional[Dict[int, float]] = None,
) -> float:
    """
    Get weight for a game based on how many days ago it was played.
    
    Args:
        days_ago: Number of days since the game (0 = today)
        is_late_season_flag: Override late season detection
        custom_weights: Override default weight table
        
    Returns:
        Weight multiplier (1.0 = normal, 2.0 = 2x weight)
    """
    if is_late_season_flag is None:
        is_late_season_flag = is_late_season()
    
    # Outside late season: flat weight
    if not is_late_season_flag:
        return REGULAR_SEASON_WEIGHT
    
    # Use custom weights if provided
    weights = custom_weights or DEFAULT_RECENCY_WEIGHTS
    
    # Get weight for this day (default to 1.0 for old games)
    weight = weights.get(days_ago, 1.0)
    
    # Clamp to reasonable range
    return max(0.5, min(3.0, weight))


def compute_weighted_rating(
    games: list[Dict],
    rating_key: str = "adj_em",
    date_key: str = "game_date",
    is_late_season_flag: Optional[bool] = None,
) -> Tuple[float, Dict]:
    """
    Compute a weighted rating from recent games.
    
    Args:
        games: List of game dicts with rating and date
        rating_key: Key for rating value in game dict
        date_key: Key for date in game dict
        is_late_season_flag: Override late season detection
        
    Returns:
        Tuple of (weighted_rating, metadata)
    """
    if not games:
        return 0.0, {"error": "no_games"}
    
    if is_late_season_flag is None:
        is_late_season_flag = is_late_season()
    
    today = date.today()
    weighted_sum = 0.0
    total_weight = 0.0
    
    game_weights = []
    
    for game in games:
        rating = game.get(rating_key)
        game_date = game.get(date_key)
        
        if rating is None or game_date is None:
            continue
        
        # Calculate days ago
        if isinstance(game_date, str):
            try:
                game_date = datetime.fromisoformat(game_date.replace('Z', '+00:00')).date()
            except:
                continue
        
        days_ago = (today - game_date).days
        if days_ago < 0:
            days_ago = 0  # Future game (shouldn't happen)
        
        # Get weight
        weight = get_recency_weight(days_ago, is_late_season_flag)
        
        weighted_sum += rating * weight
        total_weight += weight
        
        game_weights.append({
            "date": game_date.isoformat() if hasattr(game_date, 'isoformat') else str(game_date),
            "days_ago": days_ago,
            "rating": rating,
            "weight": weight,
        })
    
    if total_weight == 0:
        return 0.0, {"error": "no_valid_games"}
    
    weighted_rating = weighted_sum / total_weight
    
    metadata = {
        "games_count": len(games),
        "valid_games": len(game_weights),
        "total_weight": total_weight,
        "is_late_season": is_late_season_flag,
        "game_weights": game_weights[:10],  # Last 10 for debugging
    }
    
    return weighted_rating, metadata


def get_tournament_adjustments(
    check_date: Optional[date] = None,
    is_neutral: bool = False,
) -> Dict:
    """
    Get tournament-specific adjustments.
    
    Args:
        check_date: Date to check
        is_neutral: Whether game is at neutral site
        
    Returns:
        Dict of adjustments to apply
    """
    if check_date is None:
        check_date = date.today()
    
    adjustments = {
        "neutral_site_override": False,
        "margin_se_inflation": 0.0,
        "recency_weight_active": False,
        "form_window_days": 30,  # Look at last 30 days for form
    }
    
    # Tournament mode adjustments
    if is_tournament_mode(check_date):
        adjustments["neutral_site_override"] = True
        adjustments["margin_se_inflation"] = 0.20  # Wider CI for upsets
        adjustments["recency_weight_active"] = True
        adjustments["form_window_days"] = 14  # Only last 2 weeks matter
    
    # Neutral site = always no HCA, higher variance
    if is_neutral:
        adjustments["margin_se_inflation"] = max(
            adjustments["margin_se_inflation"], 0.15
        )
    
    return adjustments


class RecencyWeightEngine:
    """
    Engine for applying recency weights to team ratings.
    """
    
    def __init__(self):
        self.weights_table = DEFAULT_RECENCY_WEIGHTS.copy()
        self.late_season_start = (LATE_SEASON_START_MONTH, LATE_SEASON_START_DAY)
    
    def is_active(self, check_date: Optional[date] = None) -> bool:
        """Check if recency weighting is active."""
        return is_late_season(check_date)
    
    def get_weight(self, days_ago: int) -> float:
        """Get weight for a specific day."""
        return get_recency_weight(days_ago, self.is_active())
    
    def apply_to_team_ratings(
        self,
        team_games: list[Dict],
    ) -> Tuple[float, Dict]:
        """
        Apply recency weights to a team's game ratings.
        
        Args:
            team_games: List of game performance dicts
            
        Returns:
            Tuple of (weighted_rating, metadata)
        """
        return compute_weighted_rating(
            team_games,
            is_late_season_flag=self.is_active(),
        )


# Singleton instance
_recency_engine: Optional[RecencyWeightEngine] = None

def get_recency_engine() -> RecencyWeightEngine:
    """Get singleton recency weight engine."""
    global _recency_engine
    if _recency_engine is None:
        _recency_engine = RecencyWeightEngine()
    return _recency_engine
