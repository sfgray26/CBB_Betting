"""
The Odds API integration for real-time CBB odds
https://the-odds-api.com/
"""

import requests
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"


class OddsAPIClient:
    """Client for The Odds API"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("THE_ODDS_API_KEY not set in environment")
    
    def get_cbb_odds(
        self,
        markets: str = "h2h,spreads,totals",
        regions: str = "us",
        odds_format: str = "american",
    ) -> List[Dict]:
        """
        Fetch current CBB odds

        Returns list of games with odds from multiple bookmakers
        """
        url = f"{BASE_URL}/sports/basketball_ncaab/odds"

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Log API quota usage
            remaining = response.headers.get("x-requests-remaining")
            used = response.headers.get("x-requests-used")
            logger.info(f"Odds API: {len(data)} games fetched. Quota: {used} used, {remaining} remaining")

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Odds API error: {e}")
            return []

    def get_cbb_derivative_odds(
        self,
        markets: str = "alternate_spreads,alternate_totals",
        regions: str = "us",
        odds_format: str = "american",
    ) -> List[Dict]:
        """
        Fetch derivative market odds (1st half, team totals, alt lines).

        Many retail books derive 1H lines by halving the full-game spread.
        A possession simulator can identify structural mispricing here.
        """
        url = f"{BASE_URL}/sports/basketball_ncaab/odds"

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            remaining = response.headers.get("x-requests-remaining")
            logger.info(
                "Derivative odds: %d games fetched. Remaining: %s",
                len(data), remaining,
            )
            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Derivative odds API error: {e}")
            return []
    
    def parse_odds_for_game(self, game_data: Dict) -> Dict:
        """
        Parse raw odds data into standardized format

        Returns best available odds across all bookmakers,
        including derivative markets when available.
        """
        result = {
            "game_id": game_data.get("id"),
            "commence_time": game_data.get("commence_time"),
            "home_team": game_data.get("home_team"),
            "away_team": game_data.get("away_team"),
            "bookmakers": [],
            "best_spread": None,
            "best_spread_odds": None,
            "best_total": None,
            "best_moneyline_home": None,
            "best_moneyline_away": None,
            # Derivative markets
            "best_1h_spread": None,
            "best_1h_spread_odds": None,
            "best_1h_total": None,
            "best_team_total_home": None,
            "best_team_total_away": None,
        }
        
        # Extract best odds from each bookmaker
        for bookmaker in game_data.get("bookmakers", []):
            book_name = bookmaker.get("key")
            markets = bookmaker.get("markets", [])
            
            book_odds = {"name": book_name}
            
            for market in markets:
                market_key = market.get("key")
                
                if market_key == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == result["home_team"]:
                            book_odds["spread_home"] = outcome.get("point")
                            book_odds["spread_home_odds"] = outcome.get("price")
                        else:
                            book_odds["spread_away"] = outcome.get("point")
                            book_odds["spread_away_odds"] = outcome.get("price")
                
                elif market_key == "totals":
                    for outcome in market.get("outcomes", []):
                        book_odds["total"] = outcome.get("point")
                        if outcome.get("name") == "Over":
                            book_odds["total_over_odds"] = outcome.get("price")
                        else:
                            book_odds["total_under_odds"] = outcome.get("price")
                
                elif market_key == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == result["home_team"]:
                            book_odds["moneyline_home"] = outcome.get("price")
                        else:
                            book_odds["moneyline_away"] = outcome.get("price")

                # --- Derivative markets ---
                elif market_key in ("spreads_h1", "h2h_h1"):
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == result["home_team"]:
                            book_odds["spread_1h_home"] = outcome.get("point")
                            book_odds["spread_1h_home_odds"] = outcome.get("price")

                elif market_key in ("totals_h1",):
                    for outcome in market.get("outcomes", []):
                        book_odds["total_1h"] = outcome.get("point")

                elif market_key in ("team_totals",):
                    for outcome in market.get("outcomes", []):
                        if outcome.get("description", "").lower().find("over") >= 0:
                            team_name = outcome.get("name", "")
                            if team_name == result["home_team"]:
                                book_odds["team_total_home"] = outcome.get("point")
                            elif team_name == result["away_team"]:
                                book_odds["team_total_away"] = outcome.get("point")

            result["bookmakers"].append(book_odds)
        
        # Find best available lines (line shopping)
        if result["bookmakers"]:
            # Best spread (most favorable for bettor)
            spreads = [(b.get("spread_home"), b.get("spread_home_odds")) 
                      for b in result["bookmakers"] if b.get("spread_home")]
            if spreads:
                # Best = smallest spread with best odds
                result["best_spread"] = spreads[0][0]
                result["best_spread_odds"] = max(s[1] for s in spreads)
            
            # Best total
            totals = [b.get("total") for b in result["bookmakers"] if b.get("total")]
            if totals:
                result["best_total"] = totals[0]
            
            # Best moneylines
            ml_home = [b.get("moneyline_home") for b in result["bookmakers"] if b.get("moneyline_home")]
            ml_away = [b.get("moneyline_away") for b in result["bookmakers"] if b.get("moneyline_away")]

            if ml_home:
                result["best_moneyline_home"] = max(ml_home)  # Best for home team
            if ml_away:
                result["best_moneyline_away"] = max(ml_away)  # Best for away team

            # Derivative markets
            spreads_1h = [
                (b.get("spread_1h_home"), b.get("spread_1h_home_odds"))
                for b in result["bookmakers"]
                if b.get("spread_1h_home") is not None
            ]
            if spreads_1h:
                result["best_1h_spread"] = spreads_1h[0][0]
                result["best_1h_spread_odds"] = max(
                    s[1] for s in spreads_1h if s[1] is not None
                ) if any(s[1] for s in spreads_1h) else -110

            totals_1h = [b.get("total_1h") for b in result["bookmakers"] if b.get("total_1h")]
            if totals_1h:
                result["best_1h_total"] = totals_1h[0]

            tt_home = [b.get("team_total_home") for b in result["bookmakers"] if b.get("team_total_home")]
            tt_away = [b.get("team_total_away") for b in result["bookmakers"] if b.get("team_total_away")]
            if tt_home:
                result["best_team_total_home"] = tt_home[0]
            if tt_away:
                result["best_team_total_away"] = tt_away[0]

        return result
    
    def get_todays_games(self) -> List[Dict]:
        """Get all CBB games for today with best odds"""
        raw_odds = self.get_cbb_odds()
        
        games = []
        for game in raw_odds:
            parsed = self.parse_odds_for_game(game)
            games.append(parsed)
        
        logger.info(f"📊 Parsed odds for {len(games)} games")
        return games


def get_data_freshness(fetched_at: datetime) -> Dict:
    """Calculate data freshness tier"""
    now = datetime.utcnow()
    age_minutes = (now - fetched_at).total_seconds() / 60
    age_hours = age_minutes / 60
    
    # Lines freshness
    if age_minutes < 10:
        tier = "Tier 1"
    elif age_minutes < 30:
        tier = "Tier 2"
    else:
        tier = "Tier 3"
    
    return {
        "fetched_at": fetched_at.isoformat(),
        "age_minutes": age_minutes,
        "age_hours": age_hours,
        "tier": tier,
        "lines_age_min": age_minutes,
    }


# Convenience function for direct import
def fetch_current_odds() -> tuple[List[Dict], Dict]:
    """Fetch odds and return with freshness metadata"""
    client = OddsAPIClient()
    fetched_at = datetime.utcnow()
    games = client.get_todays_games()
    freshness = get_data_freshness(fetched_at)
    
    return games, freshness


if __name__ == "__main__":
    # Test
    client = OddsAPIClient()
    games = client.get_todays_games()
    
    print(f"Found {len(games)} games:")
    for game in games[:3]:
        print(f"  {game['away_team']} @ {game['home_team']}")
        print(f"    Spread: {game['best_spread']} ({game['best_spread_odds']})")
        print(f"    Total: {game['best_total']}")
