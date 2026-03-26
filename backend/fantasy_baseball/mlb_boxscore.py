"""
MLB Stats API integration for fetching box scores and resolving decisions.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MLBBoxScoreFetcher:
    """Fetch player stats from MLB Stats API."""
    
    BASE_URL = "https://statsapi.mlb.com/api/v1"
    
    def __init__(self):
        self.session = requests.Session()
    
    def get_player_stats_for_date(
        self, 
        player_name: str, 
        team_abbr: str, 
        game_date: str
    ) -> Optional[Dict[str, float]]:
        """
        Get batting stats for a player on a specific date.
        
        Args:
            player_name: Full player name (e.g., "Pete Alonso")
            team_abbr: Team abbreviation (e.g., "NYM")
            game_date: YYYY-MM-DD
            
        Returns:
            Dict with hr, r, rbi, sb, avg or None if no game/no stats
        """
        try:
            # Find the game for this team on this date
            game_pk = self._find_game_pk(team_abbr, game_date)
            if not game_pk:
                logger.debug(f"No game found for {team_abbr} on {game_date}")
                return None
            
            # Fetch box score
            box_score = self._fetch_box_score(game_pk)
            if not box_score:
                return None
            
            # Find player stats
            stats = self._extract_player_stats(box_score, player_name, team_abbr)
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to fetch stats for {player_name}: {e}")
            return None
    
    def get_all_stats_for_date(self, game_date: str) -> Dict[str, Dict[str, float]]:
        """
        Get all player stats for all games on a date.
        
        Returns:
            Dict mapping "Player Name" -> stats dict
        """
        all_stats = {}
        
        try:
            # Get all games for the date
            games = self._get_games_for_date(game_date)
            
            for game in games:
                game_pk = game.get("gamePk")
                if not game_pk:
                    continue
                
                # Check if game is final
                status = game.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    logger.debug(f"Game {game_pk} not final yet")
                    continue
                
                # Fetch box score
                box_score = self._fetch_box_score(game_pk)
                if not box_score:
                    continue
                
                # Extract all player stats from this game
                game_stats = self._extract_all_players_stats(box_score)
                all_stats.update(game_stats)
                
        except Exception as e:
            logger.error(f"Failed to fetch all stats for {game_date}: {e}")
        
        return all_stats
    
    def _find_game_pk(self, team_abbr: str, game_date: str) -> Optional[int]:
        """Find game PK for a team on a specific date."""
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
            "teamId": self._get_team_id(team_abbr),
        }
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            for date_info in data.get("dates", []):
                for game in date_info.get("games", []):
                    return game.get("gamePk")
                    
        except Exception as e:
            logger.warning(f"Failed to find game for {team_abbr}: {e}")
        
        return None
    
    def _get_games_for_date(self, game_date: str) -> List[Dict]:
        """Get all games for a date."""
        url = f"{self.BASE_URL}/schedule"
        params = {
            "sportId": 1,
            "date": game_date,
        }
        
        try:
            resp = self.session.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            games = []
            for date_info in data.get("dates", []):
                games.extend(date_info.get("games", []))
            return games
            
        except Exception as e:
            logger.error(f"Failed to fetch games for {game_date}: {e}")
            return []
    
    def _fetch_box_score(self, game_pk: int) -> Optional[Dict]:
        """Fetch box score for a game."""
        url = f"{self.BASE_URL}/game/{game_pk}/boxscore"
        
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch box score for {game_pk}: {e}")
            return None
    
    def _extract_player_stats(
        self, 
        box_score: Dict, 
        player_name: str, 
        team_abbr: str
    ) -> Optional[Dict[str, float]]:
        """Extract stats for a specific player from box score."""
        # Try both teams
        for team_type in ["home", "away"]:
            team_data = box_score.get("teams", {}).get(team_type, {})
            players = team_data.get("players", {})
            
            for player_id, player_data in players.items():
                info = player_data.get("person", {})
                name = info.get("fullName", "")
                
                # Match player name (case insensitive)
                if name.lower() == player_name.lower():
                    return self._parse_hitting_stats(player_data)
        
        return None
    
    def _extract_all_players_stats(self, box_score: Dict) -> Dict[str, Dict[str, float]]:
        """Extract all player stats from a box score."""
        all_stats = {}
        
        for team_type in ["home", "away"]:
            team_data = box_score.get("teams", {}).get(team_type, {})
            players = team_data.get("players", {})
            
            for player_id, player_data in players.items():
                info = player_data.get("person", {})
                name = info.get("fullName", "")
                
                stats = self._parse_hitting_stats(player_data)
                if stats:  # Only include if they have hitting stats
                    all_stats[name] = stats
        
        return all_stats
    
    def _parse_hitting_stats(self, player_data: Dict) -> Optional[Dict[str, float]]:
        """Parse hitting stats from player data."""
        stats = player_data.get("stats", {}).get("batting", {})
        
        if not stats:
            return None
        
        at_bats = stats.get("atBats", 0)
        hits = stats.get("hits", 0)
        
        return {
            "hr": float(stats.get("homeRuns", 0)),
            "r": float(stats.get("runs", 0)),
            "rbi": float(stats.get("rbi", 0)),
            "sb": float(stats.get("stolenBases", 0)),
            "avg": round(hits / at_bats, 3) if at_bats > 0 else 0.0,
            "h": float(hits),
            "ab": float(at_bats),
            "2b": float(stats.get("doubles", 0)),
            "3b": float(stats.get("triples", 0)),
            "bb": float(stats.get("baseOnBalls", 0)),
        }
    
    def _get_team_id(self, team_abbr: str) -> Optional[int]:
        """Map team abbreviation to MLB team ID."""
        # Common abbreviations to MLB team IDs
        team_ids = {
            "ARI": 109, "ATL": 144, "BAL": 110, "BOS": 111,
            "CHC": 112, "CWS": 145, "CIN": 113, "CLE": 114,
            "COL": 115, "DET": 116, "HOU": 117, "KC": 118,
            "LAA": 108, "LAD": 119, "MIA": 146, "MIL": 158,
            "MIN": 142, "NYM": 121, "NYY": 147, "OAK": 133,
            "PHI": 143, "PIT": 134, "SD": 135, "SF": 137,
            "SEA": 136, "STL": 138, "TB": 139, "TEX": 140,
            "TOR": 141, "WAS": 120,
            # Full names
            "DIAMONDBACKS": 109, "BRAVES": 144, "ORIOLES": 110, "RED SOX": 111,
            "CUBS": 112, "WHITE SOX": 145, "REDS": 113, "GUARDIANS": 114,
            "ROCKIES": 115, "TIGERS": 116, "ASTROS": 117, "ROYALS": 118,
            "ANGELS": 108, "DODGERS": 119, "MARLINS": 146, "BREWERS": 158,
            "TWINS": 142, "METS": 121, "YANKEES": 147, "ATHLETICS": 133,
            "PHILLIES": 143, "PIRATES": 134, "PADRES": 135, "GIANTS": 137,
            "MARINERS": 136, "CARDINALS": 138, "RAYS": 139, "RANGERS": 140,
            "BLUE JAYS": 141, "NATIONALS": 120,
        }
        return team_ids.get(team_abbr.upper())


# Singleton instance
_mlb_fetcher: Optional[MLBBoxScoreFetcher] = None


def get_mlb_fetcher() -> MLBBoxScoreFetcher:
    """Get singleton MLB fetcher instance."""
    global _mlb_fetcher
    if _mlb_fetcher is None:
        _mlb_fetcher = MLBBoxScoreFetcher()
    return _mlb_fetcher
