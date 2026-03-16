"""
Team-to-Conference Lookup Service — Mission K-10

Provides lightweight CSV-backed lookup from team name to conference.
Used by analysis pipeline to apply conference-specific HCA.

Usage:
    from backend.services.team_conference_lookup import get_team_conference, get_conference_hca_for_team
    
    # Get conference for a team
    conf = get_team_conference("Duke")  # Returns "ACC"
    
    # Get HCA for a team (with conference lookup)
    hca = get_conference_hca_for_team("Duke", is_neutral=False)  # Returns 3.0
"""

import csv
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from backend.services.conference_hca import get_conference_hca, DEFAULT_HCA

logger = logging.getLogger(__name__)

# Path to the team-conference mapping CSV
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TEAM_CONFERENCE_CSV = DATA_DIR / "team_conference_map.csv"

# Baseline HCA used by the model (from identity.yaml base_hca: 3.09)
BASELINE_HCA = 3.09


@lru_cache(maxsize=1)
def _load_conference_map() -> Dict[str, str]:
    """
    Load team-to-conference mapping from CSV.
    
    Returns:
        Dictionary mapping team_name -> conference_name
    """
    conference_map = {}
    
    if not TEAM_CONFERENCE_CSV.exists():
        logger.warning(
            "Team conference map not found: %s. All teams will use default HCA.",
            TEAM_CONFERENCE_CSV
        )
        return conference_map
    
    try:
        with open(TEAM_CONFERENCE_CSV, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                team_name = row.get('team_name', '').strip()
                conference = row.get('conference', '').strip()
                if team_name and conference:
                    conference_map[team_name] = conference
        
        logger.info(
            "Loaded conference map: %d teams from %s",
            len(conference_map), TEAM_CONFERENCE_CSV
        )
    except Exception as e:
        logger.error("Failed to load conference map: %s", e)
    
    return conference_map


def get_team_conference(team_name: str) -> Optional[str]:
    """
    Return conference name for a team, or None if unknown.
    
    Args:
        team_name: Team name (e.g., "Duke", "UNC", "Gonzaga")
        
    Returns:
        Conference name (e.g., "ACC", "WCC") or None if not found
        
    Examples:
        >>> get_team_conference("Duke")
        "ACC"
        >>> get_team_conference("Gonzaga")
        "WCC"
        >>> get_team_conference("Unknown Team")
        None
    """
    if not team_name:
        return None
    
    conference_map = _load_conference_map()
    
    # Try exact match first
    if team_name in conference_map:
        return conference_map[team_name]
    
    # Try case-insensitive match
    team_name_lower = team_name.lower()
    for name, conf in conference_map.items():
        if name.lower() == team_name_lower:
            return conf
    
    return None


def get_conference_hca_for_team(team_name: str, is_neutral: bool = False) -> float:
    """
    Return HCA points for a team's home conference, or DEFAULT_HCA if unknown.
    
    This function returns the actual conference HCA value (not the delta).
    For neutral sites, always returns 0.0.
    
    Args:
        team_name: Team name (e.g., "Duke", "Michigan State")
        is_neutral: Whether game is at neutral site
        
    Returns:
        HCA points for the team's conference (e.g., 3.6 for Big Ten)
        or DEFAULT_HCA if team/conference unknown
        
    Examples:
        >>> get_conference_hca_for_team("Duke", is_neutral=False)
        3.0  # ACC HCA
        >>> get_conference_hca_for_team("Michigan State", is_neutral=False)
        3.6  # Big Ten HCA
        >>> get_conference_hca_for_team("Duke", is_neutral=True)
        0.0  # Neutral site
        >>> get_conference_hca_for_team("Unknown Team")
        2.5  # DEFAULT_HCA
    """
    if is_neutral:
        return 0.0
    
    conference = get_team_conference(team_name)
    
    if conference is None:
        logger.debug("Unknown team for HCA lookup: %s", team_name)
        return DEFAULT_HCA
    
    return get_conference_hca(conference, is_neutral=False)


def get_conference_hca_delta(team_name: str, is_neutral: bool = False) -> float:
    """
    Return the HCA delta (conference HCA - baseline HCA) for margin adjustment.
    
    This is the value that should be ADDED to matchup_margin_adj in analyze_game().
    For neutral sites, returns 0.0.
    For unknown teams, returns 0.0 (uses baseline HCA).
    
    Args:
        team_name: Team name (e.g., "Duke", "Michigan State")
        is_neutral: Whether game is at neutral site
        
    Returns:
        Delta to apply to matchup_margin_adj (e.g., +0.51 for Big Ten at home)
        
    Examples:
        >>> get_conference_hca_delta("Michigan State", is_neutral=False)
        0.51  # 3.6 - 3.09 = +0.51 for home team in Big Ten
        >>> get_conference_hca_delta("Duke", is_neutral=False)
        -0.09  # 3.0 - 3.09 = -0.09 for home team in ACC
        >>> get_conference_hca_delta("Duke", is_neutral=True)
        0.0   # Neutral site
    """
    if is_neutral:
        return 0.0
    
    conf_hca = get_conference_hca_for_team(team_name, is_neutral=False)
    
    # Delta is the difference from baseline
    # This delta should be ADDED to the existing matchup_margin_adj
    return conf_hca - BASELINE_HCA


def reload_conference_map() -> None:
    """Clear the cache and reload the conference map from disk."""
    _load_conference_map.cache_clear()
    _load_conference_map()
    logger.info("Conference map reloaded")


def get_teams_in_conference(conference: str) -> list:
    """
    Return all teams in a given conference.
    
    Args:
        conference: Conference name (e.g., "ACC", "Big Ten")
        
    Returns:
        List of team names in that conference
    """
    conference_map = _load_conference_map()
    conference_lower = conference.lower()
    
    return [
        team for team, conf in conference_map.items()
        if conf.lower() == conference_lower
    ]


# Legacy alias for backward compatibility
get_team_hca = get_conference_hca_for_team


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Team Conference Lookup Test")
    print("=" * 50)
    
    test_teams = ["Duke", "Gonzaga", "Michigan State", "Auburn", "Unknown Team"]
    
    for team in test_teams:
        conf = get_team_conference(team)
        hca = get_conference_hca_for_team(team, is_neutral=False)
        delta = get_conference_hca_delta(team, is_neutral=False)
        hca_neutral = get_conference_hca_for_team(team, is_neutral=True)
        
        print(f"\n{team}:")
        print(f"  Conference: {conf or 'UNKNOWN'}")
        print(f"  HCA (home): {hca:.2f}")
        print(f"  HCA (neutral): {hca_neutral:.2f}")
        print(f"  Delta (vs {BASELINE_HCA} baseline): {delta:+.2f}")
    
    print("\n" + "=" * 50)
    print(f"Total teams in map: {len(_load_conference_map())}")
