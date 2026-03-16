"""
Fatigue and rest-day adjustment service.

Quantifies the impact of schedule density, travel burden, and circadian
disruption on team performance. These factors create market inefficiencies
that the ratings-based model misses.

Sources of fatigue penalty:
    1. Rest days (B2B, 1-day rest, 2-day, 3+ days)
    2. Travel distance (miles flown/driven to game site)
    3. Time zone changes (East Coast → West Coast, etc.)
    4. Altitude changes (sea level → 5,000+ ft)
    5. Cumulative load (games in last 7/14 days)

Research basis:
    - NBA studies: B2B teams underperform by ~2.0 points
    - CBB tournament data: 3-in-5 days shows decline
    - Altitude effects: +1.5 pts home advantage boost in Denver/Laramie
    - Time zone: Each zone crossed = ~0.5 pt penalty (first day only)
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    """Minimal game record for schedule analysis."""
    game_date: datetime
    is_home: bool
    opponent: str
    travel_distance_miles: Optional[float] = None
    altitude_ft: Optional[float] = None


@dataclass
class TeamSchedule:
    """Team's recent schedule context."""
    team: str
    games: List[GameRecord] = field(default_factory=list)
    home_arena_altitude_ft: float = 0.0  # Sea level default
    timezone_offset: int = 0  # Hours from ET (ET=0, CT=-1, MT=-2, PT=-3)


@dataclass
class FatigueAdjustment:
    """Output: complete fatigue breakdown for a team."""
    team: str
    
    # Individual components
    rest_days: int = 3  # Days since last game
    rest_penalty: float = 0.0  # Points (negative = worse)
    
    travel_distance: float = 0.0  # Miles
    travel_penalty: float = 0.0
    
    timezone_shift: int = 0  # Hours changed
    timezone_penalty: float = 0.0
    
    altitude_shift: float = 0.0  # Feet changed
    altitude_penalty: float = 0.0
    
    cumulative_load_penalty: float = 0.0  # Games in last 7/14 days
    
    # Aggregate
    total_penalty: float = 0.0
    confidence: str = "medium"  # "high" if full data, "low" if estimated
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

# Rest day impact (points relative to fully rested = 0)
# Research: NBA B2B ≈ -2.0 pts. CBB slightly less due to younger athletes.
REST_PENALTIES = {
    0: get_float_env("FATIGUE_B2B_PENALTY", "1.8"),      # Back-to-back
    1: get_float_env("FATIGUE_1DAY_PENALTY", "0.7"),    # 1 day rest
    2: get_float_env("FATIGUE_2DAY_PENALTY", "0.2"),    # 2 days
    3: 0.0,  # 3+ days = fully rested baseline
}

# Travel distance tiers (miles → penalty)
def _travel_penalty(distance_miles: float) -> float:
    """
    Travel fatigue increases with distance but plateaus.
    
    - < 100 miles (bus trip): negligible
    - 100-500 miles: mild
    - 500-1500 miles: moderate  
    - 1500+ miles (cross-country): significant
    """
    if distance_miles <= 100:
        return 0.0
    elif distance_miles < 500:
        return 0.15
    elif distance_miles < 1000:
        return 0.35
    elif distance_miles < 1500:
        return 0.60
    else:
        # Long-haul flights have disproportionate impact
        return min(1.0, 0.60 + (distance_miles - 1500) / 1000 * 0.50)


# Time zone change (hours crossed → penalty, applies on Day 1 only)
def _timezone_penalty(hours_shifted: int, rest_days: int) -> float:
    """
    Circadian disruption from time zone changes.
    
    Only applies if the team hasn't had 2+ days to acclimate.
    Eastward travel (losing hours) is worse than westward.
    """
    if rest_days >= 2:
        return 0.0  # Acclimated
    
    abs_shift = abs(hours_shifted)
    if abs_shift <= 1:
        return 0.0
    
    base = 0.25 * (abs_shift - 1)  # 0.5 pts per zone after first
    
    # Eastward is worse (harder to wake up earlier)
    if hours_shifted < 0:  # West → East (e.g., LA to NY)
        base *= 1.3
    
    return min(base, 1.5)  # Cap at 1.5 pts


# Altitude change (feet gained → penalty, or bonus if acclimated home team)
def _altitude_penalty(altitude_change_ft: float, is_home_team: bool, rest_days: int) -> float:
    """
    Altitude effects on performance.
    
    - Teams traveling TO altitude suffer initially
    - Teams acclimated to altitude get home advantage boost
    - Effect requires ~24-48 hours to adapt
    
    Altitude thresholds:
        - 0-3000 ft: minimal effect
        - 3000-5000 ft: mild (Denver, Albuquerque)
        - 5000-7000 ft: significant (Laramie, Colorado Springs)
        - 7000+ ft: severe (very few CBB venues)
    """
    if is_home_team:
        # Home team at altitude = advantage (negative penalty)
        if altitude_change_ft >= 5000:
            return -1.0  # 1 pt advantage for high-altitude home teams
        elif altitude_change_ft >= 3000:
            return -0.5
        return 0.0
    
    # Visiting team arriving at altitude
    if rest_days >= 2:
        # Partially acclimated
        multiplier = 0.5
    else:
        multiplier = 1.0
    
    if altitude_change_ft >= 5000:
        return 1.0 * multiplier
    elif altitude_change_ft >= 3000:
        return 0.5 * multiplier
    
    return 0.0


# Cumulative load (games in recent window → penalty)
def _cumulative_load_penalty(games_last_7: int, games_last_14: int) -> float:
    """
    Fatigue from dense scheduling independent of rest before this game.
    
    - 3+ games in 7 days = heavy load
    - 5+ games in 14 days = concerning
    """
    penalty = 0.0
    
    if games_last_7 >= 4:
        penalty += 0.4
    elif games_last_7 >= 3:
        penalty += 0.2
    
    if games_last_14 >= 7:
        penalty += 0.3
    elif games_last_14 >= 5:
        penalty += 0.15
    
    return penalty


# ---------------------------------------------------------------------------
# Core calculation
# ---------------------------------------------------------------------------

def calculate_fatigue(
    team: str,
    last_game_date: Optional[datetime],
    current_game_date: datetime,
    is_home: bool,
    travel_distance_miles: Optional[float] = None,
    home_arena_altitude_ft: float = 0.0,
    game_arena_altitude_ft: float = 0.0,
    home_timezone_offset: int = 0,
    game_timezone_offset: int = 0,
    recent_games: Optional[List[GameRecord]] = None,
) -> FatigueAdjustment:
    """
    Calculate complete fatigue adjustment for a team.
    
    Args:
        team: Team name
        last_game_date: When they last played (None = opening night)
        current_game_date: When the current game is
        is_home: Whether this is a home game for the team
        travel_distance_miles: Distance traveled to this game
        home_arena_altitude_ft: Team's normal home altitude
        game_arena_altitude_ft: This game's arena altitude
        home_timezone_offset: Team's home timezone (hours from ET)
        game_timezone_offset: This game's timezone (hours from ET)
        recent_games: List of recent games for cumulative load calc
    
    Returns:
        FatigueAdjustment with all penalties broken out
    """
    adj = FatigueAdjustment(team=team)
    notes = []
    
    # 1. Rest days calculation
    if last_game_date is None:
        adj.rest_days = 7  # Opening night assumption
        notes.append("Opening night / long rest assumed")
    else:
        delta = current_game_date - last_game_date
        adj.rest_days = delta.days
        if adj.rest_days < 0:
            adj.rest_days = 7  # Data error fallback
            notes.append("Invalid date sequence, assuming rested")
    
    # Cap rest days at 3 (no benefit beyond that)
    rest_category = min(adj.rest_days, 3)
    adj.rest_penalty = REST_PENALTIES.get(rest_category, 0.0)
    
    if adj.rest_days == 0:
        notes.append(f"BACK-TO-BACK: severe fatigue penalty ({adj.rest_penalty:.1f} pts)")
    elif adj.rest_days == 1:
        notes.append(f"1-day rest: moderate fatigue ({adj.rest_penalty:.1f} pts)")
    elif adj.rest_days >= 3:
        notes.append("Fully rested (3+ days)")
    
    # 2. Travel penalty
    if travel_distance_miles is not None:
        adj.travel_distance = travel_distance_miles
        adj.travel_penalty = _travel_penalty(travel_distance_miles)
        if adj.travel_penalty > 0:
            notes.append(f"Travel: {travel_distance_miles:.0f} miles ({adj.travel_penalty:.1f} pts)")
    else:
        notes.append("Travel distance unknown — using estimate")
        # Estimate based on home/away
        if not is_home:
            # Average conference road trip ~350 miles
            adj.travel_distance = 350.0
            adj.travel_penalty = _travel_penalty(350.0)
        adj.confidence = "low"
    
    # 3. Time zone penalty
    adj.timezone_shift = game_timezone_offset - home_timezone_offset
    adj.timezone_penalty = _timezone_penalty(adj.timezone_shift, adj.rest_days)
    if abs(adj.timezone_shift) > 1:
        notes.append(
            f"Time zone shift: {adj.timezone_shift:+d} hrs "
            f"({adj.timezone_penalty:.1f} pts)"
        )
    
    # 4. Altitude penalty
    if is_home:
        adj.altitude_shift = home_arena_altitude_ft
    else:
        adj.altitude_shift = game_arena_altitude_ft - home_arena_altitude_ft
    
    adj.altitude_penalty = _altitude_penalty(
        adj.altitude_shift, is_home, adj.rest_days
    )
    
    if adj.altitude_shift >= 3000:
        if is_home:
            notes.append(
                f"Altitude advantage: {home_arena_altitude_ft:.0f} ft "
                f"({abs(adj.altitude_penalty):.1f} pts boost)"
            )
        else:
            notes.append(
                f"Altitude exposure: +{adj.altitude_shift:.0f} ft "
                f"({adj.altitude_penalty:.1f} pts)"
            )
    
    # 5. Cumulative load
    if recent_games:
        cutoff_7 = current_game_date - timedelta(days=7)
        cutoff_14 = current_game_date - timedelta(days=14)
        
        games_7 = sum(1 for g in recent_games if g.game_date >= cutoff_7)
        games_14 = sum(1 for g in recent_games if g.game_date >= cutoff_14)
        
        adj.cumulative_load_penalty = _cumulative_load_penalty(games_7, games_14)
        if adj.cumulative_load_penalty > 0:
            notes.append(
                f"Cumulative load: {games_7} games in 7 days, "
                f"{games_14} in 14 days ({adj.cumulative_load_penalty:.1f} pts)"
            )
    
    # 6. Total penalty
    adj.total_penalty = (
        adj.rest_penalty +
        adj.travel_penalty +
        adj.timezone_penalty +
        adj.altitude_penalty +
        adj.cumulative_load_penalty
    )
    
    adj.notes = notes
    return adj


# ---------------------------------------------------------------------------
# Game-level API
# ---------------------------------------------------------------------------

def calculate_game_fatigue(
    home_team: str,
    away_team: str,
    game_date: datetime,
    home_last_game: Optional[datetime] = None,
    away_last_game: Optional[datetime] = None,
    home_arena_altitude_ft: float = 0.0,
    game_arena_altitude_ft: Optional[float] = None,
    home_timezone_offset: int = 0,
    game_timezone_offset: int = 0,
    travel_distance_miles: Optional[float] = None,
    home_recent_games: Optional[List[GameRecord]] = None,
    away_recent_games: Optional[List[GameRecord]] = None,
) -> Tuple[FatigueAdjustment, FatigueAdjustment]:
    """
    Calculate fatigue for both teams in a game.
    
    Returns (home_adjustment, away_adjustment).
    
    The net fatigue effect is: away.total_penalty - home.total_penalty
    (positive = advantage to home team due to fatigue mismatch)
    """
    # Default game altitude to home arena if not specified
    if game_arena_altitude_ft is None:
        game_arena_altitude_ft = home_arena_altitude_ft
    
    home_adj = calculate_fatigue(
        team=home_team,
        last_game_date=home_last_game,
        current_game_date=game_date,
        is_home=True,
        travel_distance_miles=0.0,  # Home team doesn't travel
        home_arena_altitude_ft=home_arena_altitude_ft,
        game_arena_altitude_ft=game_arena_altitude_ft,
        home_timezone_offset=home_timezone_offset,
        game_timezone_offset=home_timezone_offset,  # Same as home
        recent_games=home_recent_games,
    )
    
    away_adj = calculate_fatigue(
        team=away_team,
        last_game_date=away_last_game,
        current_game_date=game_date,
        is_home=False,
        travel_distance_miles=travel_distance_miles,
        home_arena_altitude_ft=0.0,  # Unknown, assume sea level
        game_arena_altitude_ft=game_arena_altitude_ft,
        home_timezone_offset=0,  # Unknown, assume ET
        game_timezone_offset=game_timezone_offset,
        recent_games=away_recent_games,
    )
    
    return home_adj, away_adj


def get_fatigue_margin_adjustment(
    home_adj: FatigueAdjustment,
    away_adj: FatigueAdjustment,
) -> Tuple[float, Dict]:
    """
    Convert fatigue adjustments to a margin adjustment.
    
    Returns: (margin_adjustment, metadata_dict)
    
    margin_adjustment is added to the home team's projected margin.
    Positive = home team benefits from fatigue mismatch.
    """
    # Net effect: away team's penalties minus home team's penalties
    # If away is more fatigued, home margin gets a boost
    net_penalty = away_adj.total_penalty - home_adj.total_penalty
    
    metadata = {
        "home_rest_days": home_adj.rest_days,
        "away_rest_days": away_adj.rest_days,
        "home_total_penalty": home_adj.total_penalty,
        "away_total_penalty": away_adj.total_penalty,
        "net_fatigue_effect": net_penalty,
        "home_notes": home_adj.notes,
        "away_notes": away_adj.notes,
        "confidence": min(home_adj.confidence, away_adj.confidence),
    }
    
    return net_penalty, metadata


# ---------------------------------------------------------------------------
# Database integration helpers
# ---------------------------------------------------------------------------

def fetch_team_recent_games(
    db_session,
    team: str,
    before_date: datetime,
    days_back: int = 14,
) -> List[GameRecord]:
    """
    Fetch recent games for a team from database.
    
    Returns list of GameRecord for fatigue calculation.
    """
    from backend.models import Game
    
    cutoff = before_date - timedelta(days=days_back)
    
    games = (
        db_session.query(Game)
        .filter(
            ((Game.home_team == team) | (Game.away_team == team)),
            Game.game_date < before_date,
            Game.game_date >= cutoff,
            Game.home_score.isnot(None),  # Completed games only
        )
        .order_by(Game.game_date.desc())
        .all()
    )
    
    records = []
    for g in games:
        is_home = g.home_team == team
        records.append(GameRecord(
            game_date=g.game_date,
            is_home=is_home,
            opponent=g.away_team if is_home else g.home_team,
        ))
    
    return records


# ---------------------------------------------------------------------------
# Arena data (static lookup for common venues)
# ---------------------------------------------------------------------------

ARENA_DATA = {
    # High altitude venues
    "New Mexico": {"altitude_ft": 5312, "timezone": -2},
    "Colorado": {"altitude_ft": 5430, "timezone": -2},
    "Air Force": {"altitude_ft": 7258, "timezone": -2},
    "Denver": {"altitude_ft": 5430, "timezone": -2},
    "Utah": {"altitude_ft": 4800, "timezone": -2},
    "Wyoming": {"altitude_ft": 7220, "timezone": -2},
    "Colorado State": {"altitude_ft": 5003, "timezone": -2},
    "Northern Colorado": {"altitude_ft": 4653, "timezone": -2},
    "Montana": {"altitude_ft": 3200, "timezone": -2},
    "Montana State": {"altitude_ft": 4920, "timezone": -2},
    "Idaho State": {"altitude_ft": 5870, "timezone": -2},
    "Weber State": {"altitude_ft": 4700, "timezone": -2},
    "Southern Utah": {"altitude_ft": 5700, "timezone": -2},
    "Nevada": {"altitude_ft": 4505, "timezone": -3},
    "UNLV": {"altitude_ft": 2000, "timezone": -3},
    "BYU": {"altitude_ft": 4555, "timezone": -2},
    "Utah State": {"altitude_ft": 4558, "timezone": -2},
    "UTEP": {"altitude_ft": 3740, "timezone": -2},
    "New Mexico State": {"altitude_ft": 3950, "timezone": -2},
    "Santa Clara": {"altitude_ft": 102, "timezone": -3},
    "Loyola Marymount": {"altitude_ft": 102, "timezone": -3},
    "Pepperdine": {"altitude_ft": 830, "timezone": -3},
    "San Diego": {"altitude_ft": 102, "timezone": -3},
    "Saint Mary's": {"altitude_ft": 100, "timezone": -3},
    "Gonzaga": {"altitude_ft": 1890, "timezone": -3},
    "Portland": {"altitude_ft": 100, "timezone": -3},
    "San Francisco": {"altitude_ft": 102, "timezone": -3},
}


def get_arena_data(team_name: str) -> Dict:
    """
    Get altitude and timezone for a team's home arena.
    
    Returns dict with altitude_ft and timezone, or defaults.
    """
    # Try exact match
    if team_name in ARENA_DATA:
        return ARENA_DATA[team_name]
    
    # Try case-insensitive partial match
    team_lower = team_name.lower()
    for name, data in ARENA_DATA.items():
        if name.lower() in team_lower or team_lower in name.lower():
            return data
    
    # Default to sea level, ET
    return {"altitude_ft": 0.0, "timezone": 0}


# ---------------------------------------------------------------------------
# Public API class
# ---------------------------------------------------------------------------

class FatigueService:
    """
    Service for computing fatigue adjustments in games.
    
    Usage:
        service = FatigueService()
        home_adj, away_adj = service.get_game_adjustments(
            home_team="Duke",
            away_team="UNC",
            game_date=datetime(2025, 3, 15, 19, 0),
        )
        margin_adj, meta = service.get_margin_adjustment(home_adj, away_adj)
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
    
    def get_game_adjustments(
        self,
        home_team: str,
        away_team: str,
        game_date: datetime,
        db_session=None,
    ) -> Tuple[FatigueAdjustment, FatigueAdjustment]:
        """
        Get fatigue adjustments for both teams.
        
        If db_session is provided, will fetch actual recent game data.
        Otherwise uses defaults/estimates.
        """
        # Get arena data
        home_arena = get_arena_data(home_team)
        away_arena = get_arena_data(away_team)
        
        # Fetch recent games if DB available
        home_recent = None
        away_recent = None
        home_last = None
        away_last = None
        
        if db_session:
            try:
                home_recent = fetch_team_recent_games(
                    db_session, home_team, game_date
                )
                away_recent = fetch_team_recent_games(
                    db_session, away_team, game_date
                )
                
                # Last game date is most recent
                if home_recent:
                    home_last = home_recent[0].game_date
                if away_recent:
                    away_last = away_recent[0].game_date
            except Exception as e:
                logger.warning(f"Failed to fetch recent games: {e}")
        
        # Calculate
        home_adj, away_adj = calculate_game_fatigue(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            home_last_game=home_last,
            away_last_game=away_last,
            home_arena_altitude_ft=home_arena["altitude_ft"],
            game_arena_altitude_ft=home_arena["altitude_ft"],
            home_timezone_offset=home_arena["timezone"],
            game_timezone_offset=home_arena["timezone"],
            travel_distance_miles=None,  # Could add distance calculation
            home_recent_games=home_recent,
            away_recent_games=away_recent,
        )
        
        return home_adj, away_adj
    
    def get_margin_adjustment(
        self,
        home_adj: FatigueAdjustment,
        away_adj: FatigueAdjustment,
    ) -> Tuple[float, Dict]:
        """Get margin adjustment from fatigue adjustments."""
        return get_fatigue_margin_adjustment(home_adj, away_adj)


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_fatigue_service: Optional[FatigueService] = None


def get_fatigue_service() -> FatigueService:
    global _fatigue_service
    if _fatigue_service is None:
        _fatigue_service = FatigueService()
    return _fatigue_service
