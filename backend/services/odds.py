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

  Derivative markets (best_1h_spread, best_1h_total, best_team_total_*):
      Parsed when the API returns alternate_spreads/totals markets.
      Useful for structural mispricing analysis (1H lines vs half of full).
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
# Pinnacle is an "eu" region book; CircaSports is US but limited NCAAB coverage.
SHARP_BOOKS: frozenset = frozenset({"pinnacle", "circasports"})

# High-volume retail books used as a soft-sharp proxy when SHARP_BOOKS are
# unavailable.  Both must agree within PROXY_SPREAD_AGREEMENT_PT for the
# proxy to activate.  Never added to SHARP_BOOKS — proxy consensus carries
# higher uncertainty (SE inflation via SOFT_PROXY_SE_ADDEND in betting_model).
SOFT_SHARP_BOOKS: frozenset = frozenset({"draftkings", "fanduel"})
PROXY_SPREAD_AGREEMENT_PT: float = float(
    os.getenv("PROXY_AGREEMENT_PT", "0.5")
)


class OddsAPIClient:
    """Client for The Odds API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or API_KEY
        if not self.api_key:
            raise ValueError("THE_ODDS_API_KEY not set in environment")

    def get_cbb_odds(
        self,
        markets: str = "h2h,spreads,totals",
        regions: str = os.getenv("ODDS_API_REGIONS", "us,eu"),
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
                "Odds API: %d games fetched. Quota: %s used, %s remaining",
                len(data), used, remaining,
            )

            return data

        except requests.exceptions.RequestException as e:
            logger.error("Odds API error: %s", e)
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
            logger.error("Derivative odds API error: %s", e)
            return []

    def parse_odds_for_game(
        self,
        game_data: Dict,
        active_books: Optional[List[str]] = None,
    ) -> Dict:
        """
        Parse raw odds data into standardised format.

        Produces two independent line sets:
          - Sharp consensus (Pinnacle / Circa) — use for CLV measurement.
            **Never filtered by** ``active_books``.
          - Best available (line shopping)     — use for bet placement.
            Restricted to ``active_books`` when the list is non-empty.

        Also captures derivative markets (1H spreads/totals, team totals)
        when present in the API response.

        Args:
            game_data:    Raw game dict from The Odds API.
            active_books: Optional allowlist of bookmaker keys for the
                          ``best_*`` line-shopping fields (e.g.
                          ``["draftkings", "fanduel"]``).  Keys are
                          normalised to lowercase before comparison.
                          Pass ``None`` or an empty list to include all
                          available bookmakers (default behaviour).

        Returns
        -------
        Dict with keys:
            game_id, commence_time, home_team, away_team, bookmakers
            sharp_consensus_spread, sharp_consensus_total, sharp_spread_odds
            sharp_books_available
            best_spread, best_spread_odds, best_total
            best_moneyline_home, best_moneyline_away
            best_1h_spread, best_1h_spread_odds, best_1h_total
            best_team_total_home, best_team_total_away
            active_books_used  — normalised list that was actually applied
        """
        home_team = game_data.get("home_team")
        away_team = game_data.get("away_team")

        # Normalise active_books to a frozenset of lowercase strings.
        # An empty or None list means "no filter — use all books."
        _active_filter: Optional[frozenset] = None
        if active_books:
            _active_filter = frozenset(b.strip().lower() for b in active_books if b)

        result: Dict = {
            "game_id": game_data.get("id"),
            "commence_time": game_data.get("commence_time"),
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [],
            # --- Sharp consensus (CLV benchmark) — unaffected by active_books ---
            "sharp_consensus_spread": None,
            "sharp_consensus_total": None,
            "sharp_spread_odds": None,
            "sharp_books_available": 0,
            # --- Best available (line shopping) — filtered by active_books ---
            "best_spread": None,
            "best_spread_odds": None,
            "best_spread_away_odds": None,
            "best_total": None,
            "best_moneyline_home": None,
            "best_moneyline_away": None,
            # Derivative markets
            "best_1h_spread": None,
            "best_1h_spread_odds": None,
            "best_1h_total": None,
            "best_team_total_home": None,
            "best_team_total_away": None,
            # Audit field: which books were eligible for line shopping
            "active_books_used": sorted(_active_filter) if _active_filter else None,
            # Soft-sharp proxy fields (populated below when SHARP_BOOKS absent)
            "sharp_proxy_used": False,
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

                # --- Derivative markets ---
                elif market_key in ("spreads_h1", "h2h_h1"):
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == home_team:
                            book_odds["spread_1h_home"] = outcome.get("point")
                            book_odds["spread_1h_home_odds"] = outcome.get("price")

                elif market_key in ("totals_h1",):
                    for outcome in market.get("outcomes", []):
                        book_odds["total_1h"] = outcome.get("point")

                elif market_key in ("team_totals",):
                    for outcome in market.get("outcomes", []):
                        if outcome.get("description", "").lower().find("over") >= 0:
                            team_name = outcome.get("name", "")
                            if team_name == home_team:
                                book_odds["team_total_home"] = outcome.get("point")
                            elif team_name == away_team:
                                book_odds["team_total_away"] = outcome.get("point")

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
                result["sharp_consensus_spread"] = sum(sharp_spreads) / len(sharp_spreads)
            if sharp_totals:
                result["sharp_consensus_total"] = sum(sharp_totals) / len(sharp_totals)
            if sharp_odds:
                # Integer average (American odds are always integers)
                result["sharp_spread_odds"] = int(round(sum(sharp_odds) / len(sharp_odds)))

        # ------------------------------------------------------------------
        # Soft-sharp proxy — activates when SHARP_BOOKS are absent.
        # DraftKings + FanDuel must both post a spread AND agree within
        # PROXY_SPREAD_AGREEMENT_PT.  Produces a weaker consensus than true
        # sharp lines; the SE addend in betting_model is halved (0.15 vs 0.30).
        # ------------------------------------------------------------------
        if result["sharp_books_available"] == 0:
            proxy_parsed = [
                b for b in result["bookmakers"]
                if b.get("name") in SOFT_SHARP_BOOKS
                and b.get("spread_home") is not None
            ]
            if len(proxy_parsed) == 2:
                s0 = proxy_parsed[0]["spread_home"]
                s1 = proxy_parsed[1]["spread_home"]
                if abs(s0 - s1) <= PROXY_SPREAD_AGREEMENT_PT:
                    result["sharp_consensus_spread"] = (s0 + s1) / 2.0
                    t0 = proxy_parsed[0].get("total")
                    t1 = proxy_parsed[1].get("total")
                    if t0 is not None and t1 is not None:
                        result["sharp_consensus_total"] = (t0 + t1) / 2.0
                    result["sharp_proxy_used"] = True
                    logger.info(
                        "Sharp proxy activated for %s@%s: DK=%s FD=%s → consensus=%s",
                        result["away_team"], result["home_team"],
                        s0, s1, result["sharp_consensus_spread"],
                    )
                else:
                    logger.debug(
                        "Sharp proxy suppressed for %s@%s: DK=%s FD=%s disagree by %.1fpt",
                        result["away_team"], result["home_team"],
                        s0, s1, abs(s0 - s1),
                    )

        # ------------------------------------------------------------------
        # Best available — line shopping across eligible bookmakers
        # ------------------------------------------------------------------
        # When active_books is specified, restrict shopping to that subset.
        # The full result["bookmakers"] list is preserved unchanged so callers
        # that inspect raw book data still see the complete picture.
        if _active_filter is not None:
            shop_books = [b for b in result["bookmakers"] if b.get("name", "").lower() in _active_filter]
            if not shop_books:
                logger.warning(
                    "active_books filter %s matched 0 bookmakers for game %s — "
                    "best_* fields will remain None",
                    sorted(_active_filter), result.get("game_id"),
                )
        else:
            shop_books = result["bookmakers"]
        all_books = shop_books

        # Find bookmaker with best home spread juice — take spread + away odds
        # from that SAME book to avoid phantom lines (spread from A, juice from B)
        best_spread_juice = -9999
        for b in all_books:
            home_spread = b.get("spread_home")
            home_odds = b.get("spread_home_odds")
            if home_spread is not None and home_odds is not None:
                if home_odds > best_spread_juice:
                    best_spread_juice = home_odds
                    result["best_spread"] = home_spread
                    result["best_spread_odds"] = home_odds
                    result["best_spread_away_odds"] = b.get("spread_away_odds", -110)

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

        # Derivative markets
        spreads_1h = [
            (b.get("spread_1h_home"), b.get("spread_1h_home_odds"))
            for b in all_books
            if b.get("spread_1h_home") is not None
        ]
        if spreads_1h:
            result["best_1h_spread"] = spreads_1h[0][0]
            result["best_1h_spread_odds"] = (
                max(s[1] for s in spreads_1h if s[1] is not None)
                if any(s[1] for s in spreads_1h)
                else -110
            )

        totals_1h = [b.get("total_1h") for b in all_books if b.get("total_1h")]
        if totals_1h:
            result["best_1h_total"] = totals_1h[0]

        tt_home = [b.get("team_total_home") for b in all_books if b.get("team_total_home")]
        tt_away = [b.get("team_total_away") for b in all_books if b.get("team_total_away")]
        if tt_home:
            result["best_team_total_home"] = tt_home[0]
        if tt_away:
            result["best_team_total_away"] = tt_away[0]

        return result

    def get_todays_games(
        self,
        active_books: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Get all CBB games for today with best odds and sharp consensus.

        Args:
            active_books: Optional allowlist of bookmaker keys to use for
                          the ``best_*`` line-shopping fields.  Forwarded
                          directly to ``parse_odds_for_game``.  Pass None
                          (default) to include all available bookmakers.
        """
        raw_odds = self.get_cbb_odds()

        games = []
        for game in raw_odds:
            parsed = self.parse_odds_for_game(game, active_books=active_books)
            games.append(parsed)

        sharp_count = sum(1 for g in games if g["sharp_books_available"] > 0)
        proxy_count = sum(1 for g in games if g.get("sharp_proxy_used"))
        logger.info(
            "Parsed odds for %d games (%d true sharp, %d proxy, active_books=%s)",
            len(games), sharp_count, proxy_count,
            sorted(active_books) if active_books else "all",
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
