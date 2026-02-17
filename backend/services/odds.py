"""
The Odds API integration for real-time CBB odds.
https://the-odds-api.com/

Sharp Market Isolation
----------------------
SHARP_BOOKS (Pinnacle, Circa) represent true market consensus and are the
correct benchmark for CLV measurement.  Retail books are used only for
line-shopping best available prices.

The parse_odds_for_game method now returns two distinct sets of lines:

  sharp_consensus_spread / sharp_consensus_total:
      Average spread / total across sharp books only.  This is the
      authoritative "true line" against which CLV is measured.

  best_spread / best_total / best_moneyline_*:
      Best available price across ALL bookmakers (line shopping).
      Use these for bet placement, not for CLV benchmarking.
"""

import requests
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

API_KEY = os.getenv("THE_ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"

# Books that represent the sharpest, most accurate market consensus.
# CLV should always be measured against these lines, not retail.
SHARP_BOOKS: frozenset = frozenset({"pinnacle", "circasports"})


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
        Fetch current CBB odds.

        Returns list of games with odds from multiple bookmakers.
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
            used = response.headers.get("x-requests-used")
            logger.info(
                "📡 Odds API: %d games fetched. Quota: %s used, %s remaining",
                len(data), used, remaining,
            )

            return data

        except requests.exceptions.RequestException as e:
            logger.error("❌ Odds API error: %s", e)
            return []

    def parse_odds_for_game(self, game_data: Dict) -> Dict:
        """
        Parse raw odds data into standardised format.

        Produces two independent line sets:
          - Sharp consensus (Pinnacle / Circa) — use for CLV measurement.
          - Best available across all books    — use for bet placement.

        Returns
        -------
        Dict with keys:
            game_id, commence_time, home_team, away_team, bookmakers
            sharp_consensus_spread, sharp_consensus_total, sharp_spread_odds
            sharp_books_available
            best_spread, best_spread_odds, best_total
            best_moneyline_home, best_moneyline_away
        """
        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")

        result: Dict = {
            "game_id": game_data.get("id"),
            "commence_time": game_data.get("commence_time"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [],
            # --- Sharp consensus (CLV benchmark) ---
            "sharp_consensus_spread": None,
            "sharp_consensus_total": None,
            "sharp_spread_odds": None,
            "sharp_books_available": 0,
            # --- Best available (line shopping) ---
            "best_spread": None,
            "best_spread_odds": None,
            "best_total": None,
            "best_moneyline_home": None,
            "best_moneyline_away": None,
        }

        sharp_parsed: List[Dict] = []

        for bookmaker in game_data.get("bookmakers", []):
            book_key = bookmaker.get("key", "").lower()
            book_odds: Dict = {"name": book_key}

            for market in bookmaker.get("markets", []):
                market_key = market.get("key")

                if market_key == "spreads":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home_team:
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
                        if outcome.get("name") == home_team:
                            book_odds["moneyline_home"] = outcome.get("price")
                        else:
                            book_odds["moneyline_away"] = outcome.get("price")

            result["bookmakers"].append(book_odds)

            if book_key in SHARP_BOOKS:
                sharp_parsed.append(book_odds)

        # ------------------------------------------------------------------
        # Sharp consensus — average across sharp books only
        # ------------------------------------------------------------------
        result["sharp_books_available"] = len(sharp_parsed)

        if sharp_parsed:
            sharp_spreads = [
                b["spread_home"]
                for b in sharp_parsed
                if b.get("spread_home") is not None
            ]
            sharp_totals = [
                b["total"] for b in sharp_parsed if b.get("total") is not None
            ]
            sharp_odds = [
                b["spread_home_odds"]
                for b in sharp_parsed
                if b.get("spread_home_odds") is not None
            ]

            if sharp_spreads:
                result["sharp_consensus_spread"] = sum(sharp_spreads) / len(
                    sharp_spreads
                )
            if sharp_totals:
                result["sharp_consensus_total"] = sum(sharp_totals) / len(
                    sharp_totals
                )
            if sharp_odds:
                # Integer average (American odds are always integers)
                result["sharp_spread_odds"] = int(
                    round(sum(sharp_odds) / len(sharp_odds))
                )

        # ------------------------------------------------------------------
        # Best available — line shopping across all bookmakers
        # ------------------------------------------------------------------
        all_books = result["bookmakers"]

        spreads = [
            (b.get("spread_home"), b.get("spread_home_odds"))
            for b in all_books
            if b.get("spread_home") is not None
        ]
        if spreads:
            result["best_spread"] = spreads[0][0]
            result["best_spread_odds"] = max(s[1] for s in spreads if s[1] is not None)

        totals = [b.get("total") for b in all_books if b.get("total") is not None]
        if totals:
            result["best_total"] = totals[0]

        ml_home = [
            b.get("moneyline_home")
            for b in all_books
            if b.get("moneyline_home") is not None
        ]
        ml_away = [
            b.get("moneyline_away")
            for b in all_books
            if b.get("moneyline_away") is not None
        ]
        if ml_home:
            result["best_moneyline_home"] = max(ml_home)
        if ml_away:
            result["best_moneyline_away"] = max(ml_away)

        return result

    def get_todays_games(self) -> List[Dict]:
        """Get all CBB games for today with best odds and sharp consensus."""
        raw_odds = self.get_cbb_odds()

        games = []
        for game in raw_odds:
            parsed = self.parse_odds_for_game(game)
            games.append(parsed)

        sharp_count = sum(1 for g in games if g["sharp_books_available"] > 0)
        logger.info(
            "📊 Parsed odds for %d games (%d with sharp lines)",
            len(games), sharp_count,
        )
        return games


def get_data_freshness(fetched_at: datetime) -> Dict:
    """Calculate data freshness tier."""
    now = datetime.utcnow()
    age_minutes = (now - fetched_at).total_seconds() / 60
    age_hours = age_minutes / 60

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


def fetch_current_odds() -> tuple:
    """Fetch odds and return with freshness metadata."""
    client = OddsAPIClient()
    fetched_at = datetime.utcnow()
    games = client.get_todays_games()
    freshness = get_data_freshness(fetched_at)
    return games, freshness


if __name__ == "__main__":
    client = OddsAPIClient()
    games = client.get_todays_games()

    print(f"Found {len(games)} games:")
    for game in games[:3]:
        print(f"  {game['away_team']} @ {game['home_team']}")
        print(f"    Sharp spread: {game['sharp_consensus_spread']} "
              f"({game['sharp_books_available']} sharp books)")
        print(f"    Best spread:  {game['best_spread']} ({game['best_spread_odds']})")
        print(f"    Total:        {game['best_total']}")
