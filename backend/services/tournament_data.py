"""
Tournament seed data service -- A-26 Task 2

Fetches NCAA March Madness bracket data including team seeds.
Primary source: BallDontLie API (paid, reliable)
Fallback: None (log warning and continue without seeds)

Seed-spread Kelly scalars are applied in betting_model.py based on
the seed data attached to game_dict by analysis.py.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import requests

from backend.services.team_mapping import normalize_team_name

logger = logging.getLogger(__name__)

_BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/ncaab/v1"
_CACHE: Dict = {}
_CACHE_TIMESTAMP: Optional[datetime] = None
_CACHE_TTL_HOURS = 6


class TournamentDataClient:
    """
    Client for fetching and caching NCAA tournament bracket data.

    Follows the same pattern as RatingsService in ratings.py:
    - Caches results for 6 hours (bracket doesn't change during tournament)
    - Fuzzy team name matching via normalize_team_name()
    - Graceful fallback to empty dict on API failure
    """

    def __init__(self):
        self.api_key = os.getenv("BALLDONTLIE_API_KEY")
        self._bracket_cache: Dict[str, int] = {}
        self._cache_timestamp: Optional[datetime] = None

    def fetch_bracket_data(self, season_year: Optional[int] = None) -> Dict[str, int]:
        """
        Fetch tournament bracket with seed data from BallDontLie API.

        Returns a dict mapping team_name -> seed (1-16).
        Returns empty dict if API key not set or API call fails.

        Args:
            season_year: Tournament season (e.g., 2026 for March 2026 tournament).
                        Defaults to current season from SEASON_YEAR env var.

        Returns:
            Dict[str, int]: {team_name: seed_number}
        """
        # Return cached data before checking API key (key may have changed since cache was populated)
        if self._bracket_cache and self._cache_timestamp:
            age_hours = (datetime.utcnow() - self._cache_timestamp).total_seconds() / 3600
            if age_hours < _CACHE_TTL_HOURS:
                logger.debug("Using cached bracket data (%d teams)", len(self._bracket_cache))
                return self._bracket_cache

        # Read API key fresh each call (supports runtime env var changes and test patching)
        api_key = os.getenv("BALLDONTLIE_API_KEY") or self.api_key
        if not api_key:
            logger.debug("BALLDONTLIE_API_KEY not set -- skipping seed fetch")
            return {}

        year = season_year or int(os.getenv("SEASON_YEAR", datetime.utcnow().year))

        try:
            url = f"{_BALLDONTLIE_BASE_URL}/bracket"
            headers = {"Authorization": api_key}
            params = {"season": year - 1}
            logger.debug(
                "BallDontLie bracket request: season=%d (tournament year %d)",
                year - 1,
                year,
            )

            resp = requests.get(url, headers=headers, params=params, timeout=15)
            resp.raise_for_status()

            data = resp.json()
            seed_map = {}

            for game in data.get("data", []):
                home = game.get("home_team", {})
                away = game.get("away_team", {})

                home_name = home.get("name", "").strip()
                away_name = away.get("name", "").strip()

                if home_name and home.get("seed"):
                    try:
                        seed_map[home_name] = int(home["seed"])
                    except (ValueError, TypeError):
                        pass

                if away_name and away.get("seed"):
                    try:
                        seed_map[away_name] = int(away["seed"])
                    except (ValueError, TypeError):
                        pass

            self._bracket_cache = seed_map
            self._cache_timestamp = datetime.utcnow()

            logger.info("TournamentDataClient: loaded %d teams from BallDontLie", len(seed_map))
            return seed_map

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning("BallDontLie bracket not available yet (404)")
            else:
                logger.warning("BallDontLie API error: %s", e)
            return {}
        except Exception as e:
            logger.warning("Failed to fetch tournament bracket: %s", e)
            return {}

    def get_team_seed(
        self,
        team_name: str,
        bracket_data: Optional[Dict[str, int]] = None
    ) -> Optional[int]:
        """
        Look up seed for a team using fuzzy name matching.

        Args:
            team_name: Team name from Odds API (e.g., "Duke Blue Devils")
            bracket_data: Pre-fetched bracket dict. If None, fetches fresh.

        Returns:
            int: Seed number (1-16) or None if not found
        """
        if bracket_data is None:
            bracket_data = self.fetch_bracket_data()

        if not bracket_data:
            return None

        # Try exact match first
        if team_name in bracket_data:
            return bracket_data[team_name]

        # Try fuzzy matching via normalize_team_name
        normalized = normalize_team_name(team_name, list(bracket_data.keys()))
        if normalized:
            return bracket_data.get(normalized)

        # Substring fallback (e.g., "Duke" matches "Duke Blue Devils")
        team_lower = team_name.lower()
        for full_name, seed in bracket_data.items():
            if team_lower in full_name.lower() or full_name.lower() in team_lower:
                return seed

        return None

    def get_game_seeds(
        self,
        home_team: str,
        away_team: str,
        bracket_data: Optional[Dict[str, int]] = None
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Get seeds for both teams in a game.

        Args:
            home_team: Home team name
            away_team: Away team name
            bracket_data: Pre-fetched bracket dict

        Returns:
            Tuple[Optional[int], Optional[int]]: (home_seed, away_seed)
        """
        if bracket_data is None:
            bracket_data = self.fetch_bracket_data()

        home_seed = self.get_team_seed(home_team, bracket_data)
        away_seed = self.get_team_seed(away_team, bracket_data)

        return home_seed, away_seed


# ---------------------------------------------------------------------------
# Singleton instance (follows ratings.py pattern)
# ---------------------------------------------------------------------------
_tournament_client: Optional[TournamentDataClient] = None


def get_tournament_client() -> TournamentDataClient:
    """Return singleton TournamentDataClient instance."""
    global _tournament_client
    if _tournament_client is None:
        _tournament_client = TournamentDataClient()
    return _tournament_client


def fetch_tournament_bracket(season_year: Optional[int] = None) -> Dict[str, int]:
    """Convenience function -- fetch bracket via singleton client."""
    return get_tournament_client().fetch_bracket_data(season_year)


def get_team_seed(team_name: str, bracket_data: Optional[Dict[str, int]] = None) -> Optional[int]:
    """Convenience function -- get seed via singleton client."""
    return get_tournament_client().get_team_seed(team_name, bracket_data)
