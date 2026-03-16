"""
Conference-Specific Home Court Advantage — P2

Provides conference-specific HCA multipliers to replace flat 3.09 pts.
Big Ten road games are significantly harder; SWAC home games are worth less.

Usage:
    from backend.services.conference_hca import get_conference_hca, apply_conference_hca
    
    # Get base HCA for conference
    hca = get_conference_hca("big_ten", is_neutral=False)
    
    # Apply with pace adjustment
    adjusted_hca = apply_conference_hca(
        conference="big_ten",
        is_neutral=False,
        pace_ratio=1.05,
        base_hca=3.09
    )
"""

import logging
from typing import Dict, Optional

from backend.utils.env_utils import get_float_env

logger = logging.getLogger(__name__)


# Conference HCA table (points)
# Based on historical home/away performance differentials
# Neutral site = 0.0 regardless of conference
CONFERENCE_HCA: Dict[str, float] = {
    # High-major power conferences
    "big_ten": 3.6,
    "big_12": 3.4,
    "sec": 3.2,
    "acc": 3.0,
    "pac_12": 2.8,
    
    # High-major non-power
    "big_east": 2.9,
    "wcc": 2.7,  # West Coast Conference (Gonzaga, St. Mary's)
    "aac": 2.6,  # American Athletic
    "a10": 2.5,  # Atlantic 10
    
    # Mid-majors with strong home courts
    "mwc": 2.7,   # Mountain West (altitude effect built in)
    "mvc": 2.4,   # Missouri Valley
    "wac": 2.3,   # WAC
    
    # Mid-majors
    "cusa": 2.2,
    "mac": 2.1,
    "sun_belt": 2.0,
    "caa": 1.9,
    
    # Low-majors
    "swac": 1.5,
    "meac": 1.5,
    "southland": 1.6,
    "big_sky": 1.7,
    "big_west": 1.7,
    "ivy": 1.8,
    "patriot": 1.7,
    "southern": 1.8,
    "summit": 1.9,
    "horizon": 1.9,
    "atlantic_sun": 1.5,
    "ohio_valley": 1.8,
    "metro_atlantic": 1.6,
    "northeast": 1.5,
    "america_east": 1.5,
    "big_south": 1.5,
}

# Default HCA for unknown conferences
DEFAULT_HCA = 2.5


def normalize_conference_name(name: Optional[str]) -> str:
    """
    Normalize conference name to standard key.
    
    Args:
        name: Raw conference name from data source
        
    Returns:
        Normalized conference key
    """
    if not name:
        return "mid_major"
    
    name_lower = name.lower().strip()
    
    # Common variations mapping
    variations = {
        # Big Ten
        "big ten": "big_ten",
        "big 10": "big_ten",
        "bigten": "big_ten",
        "b1g": "big_ten",
        
        # Big 12
        "big 12": "big_12",
        "big12": "big_12",
        
        # SEC
        "southeastern conference": "sec",
        "southeastern": "sec",
        
        # ACC
        "atlantic coast": "acc",
        
        # Pac-12
        "pac 12": "pac_12",
        "pac12": "pac_12",
        "pacific 12": "pac_12",
        
        # Big East
        "big east": "big_east",
        "bigeast": "big_east",
        
        # West Coast
        "west coast": "wcc",
        "wcc": "wcc",
        
        # American
        "american athletic": "aac",
        "the american": "aac",
        
        # Atlantic 10
        "atlantic 10": "a10",
        "atlantic10": "a10",
        
        # Mountain West
        "mountain west": "mwc",
        "mw": "mwc",
        
        # Missouri Valley
        "missouri valley": "mvc",
        "the valley": "mvc",
        
        # C-USA
        "conference usa": "cusa",
        "cusa": "cusa",
        "c-usa": "cusa",
        
        # MAC
        "mid-american": "mac",
        "mid american": "mac",
        
        # Sun Belt
        "sun belt": "sun_belt",
        "sunbelt": "sun_belt",
        
        # CAA
        "colonial": "caa",
        "colonial athletic": "caa",
        
        # SWAC
        "southwestern athletic": "swac",
        
        # MEAC
        "mid-eastern athletic": "meac",
        "mid eastern athletic": "meac",
        
        # Ivy
        "ivy league": "ivy",
    }
    
    return variations.get(name_lower, name_lower.replace(" ", "_"))


def get_conference_hca(
    conference: Optional[str],
    is_neutral: bool = False,
) -> float:
    """
    Get HCA for a conference.
    
    Args:
        conference: Conference name
        is_neutral: If True, returns 0.0 (neutral site)
        
    Returns:
        Home court advantage in points
    """
    if is_neutral:
        return 0.0
    
    normalized = normalize_conference_name(conference)
    hca = CONFERENCE_HCA.get(normalized, DEFAULT_HCA)
    
    return hca


def apply_conference_hca(
    conference: Optional[str],
    is_neutral: bool = False,
    pace_ratio: float = 1.0,
    base_hca: Optional[float] = None,
) -> tuple[float, dict]:
    """
    Apply conference-specific HCA with pace adjustment.
    
    Args:
        conference: Conference name
        is_neutral: Whether game is at neutral site
        pace_ratio: Game pace / D1 average pace (typically ~1.0)
        base_hca: Optional override base HCA
        
    Returns:
        Tuple of (adjusted_hca, metadata_dict)
    """
    # Get conference HCA
    conf_hca = get_conference_hca(conference, is_neutral)
    
    # Apply pace adjustment
    adjusted_hca = conf_hca * pace_ratio
    
    metadata = {
        "conference": conference,
        "conference_normalized": normalize_conference_name(conference) if conference else None,
        "base_hca": conf_hca,
        "pace_ratio": pace_ratio,
        "adjusted_hca": adjusted_hca,
        "is_neutral": is_neutral,
    }
    
    return adjusted_hca, metadata


def get_conference_difficulty_rating(conference: Optional[str]) -> str:
    """
    Get qualitative difficulty rating for conference road games.
    
    Args:
        conference: Conference name
        
    Returns:
        Difficulty rating string
    """
    hca = get_conference_hca(conference, is_neutral=False)
    
    if hca >= 3.4:
        return "EXTREME"  # Big Ten, Big 12
    elif hca >= 3.0:
        return "HIGH"     # SEC, ACC
    elif hca >= 2.5:
        return "MODERATE" # Big East, WCC, AAC
    elif hca >= 2.0:
        return "STANDARD" # Most mid-majors
    else:
        return "LOW"      # Low-majors


# Legacy conference mapping for backward compatibility
# Maps old rating source names to conference keys
LEGACY_CONFERENCE_MAP = {
    "barttorvik_conf": None,  # Would need to be extracted from team data
    "kenpom_conf": None,      # Would need to be extracted from team data
}
