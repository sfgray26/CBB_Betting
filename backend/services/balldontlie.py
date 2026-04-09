"""
BallDontLie GOAT API client for NCAAB and MLB data.

Base URL: https://api.balldontlie.io
Auth: Authorization header with bare API key (env: BALLDONTLIE_API_KEY).
      No "Bearer" prefix — raw key only.

NCAAB endpoints (CBB season closed — archive/read only):
  /ncaab/v1/bracket         — Tournament bracket structure (GOAT tier)
  /ncaab/v1/odds            — Live ML/spread/total per sportsbook
  /ncaab/v1/games           — Game results (live scores, completed)
  /ncaab/v1/team_season_stats — Season pace/3PT stats for TournamentTeam enrichment

MLB endpoints (active — 2026 season):
  /mlb/v1/games             — Schedule + scores (validated via MLBGame contract)
  /mlb/v1/odds              — Lines per sportsbook (validated via MLBBettingOdd contract)
  /mlb/v1/player_injuries   — IL + DTD list (validated via MLBInjury contract)
  /mlb/v1/players           — Player lookup (validated via MLBPlayer contract)

Build sequence: get_mlb_games → get_mlb_odds → get_mlb_injuries → get_mlb_player
Each method returns validated Pydantic objects. Never raw dicts.
"""

import os
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from backend.data_contracts import MLBBettingOdd, MLBGame, MLBInjury, MLBPlayer, MLBPlayerStats, BDLResponse

logger = logging.getLogger(__name__)

BASE_URL = "https://api.balldontlie.io"
NCAAB_PREFIX = "/ncaab/v1"
MLB_PREFIX = "/mlb/v1"
TOURNAMENT_SEASON = 2025   # 2025-26 season (API accepts 2025, returns 2026 bracket)

# Sportsbooks to prefer for market_ml extraction (in priority order)
PREFERRED_BOOKS = ["pinnacle", "draftkings", "fanduel", "betmgm", "caesars"]


class BallDontLieClient:
    """
    Thin wrapper around the BallDontLie NCAAB REST API.

    Usage:
        client = BallDontLieClient()                   # reads BALLDONTLIE_API_KEY from env
        client = BallDontLieClient(api_key="abc123")   # explicit key
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("BALLDONTLIE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "BALLDONTLIE_API_KEY not set. "
                "Add it to .env or pass api_key= explicitly."
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": self.api_key})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        url = BASE_URL + NCAAB_PREFIX + path
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _paginate(
        self,
        path: str,
        params: Optional[Dict] = None,
        max_pages: int = 20,
    ) -> List[Dict]:
        """Fetch all pages using cursor-based pagination."""
        params = dict(params or {})
        params.setdefault("per_page", 100)
        results: List[Dict] = []
        page = 0
        while page < max_pages:
            data = self._get(path, params)
            results.extend(data.get("data", []))
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor
            page += 1
            time.sleep(0.1)   # be polite
        return results

    # ------------------------------------------------------------------
    # Bracket (GOAT tier)
    # ------------------------------------------------------------------

    def get_bracket(
        self,
        season: int = TOURNAMENT_SEASON,
        round_id: Optional[int] = None,
        region_id: Optional[int] = None,
    ) -> List[Dict]:
        """
        Fetch NCAA Tournament bracket entries.

        Args:
            season: e.g. 2025 for 2025-26 season
            round_id: 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=Championship, 7=First Four
            region_id: filter to a specific region

        Returns:
            List of bracket game objects with team names, seeds, scores, round info.
        """
        params: Dict[str, Any] = {"season": season}
        if round_id is not None:
            params["round_id"] = round_id
        if region_id is not None:
            params["region_id"] = region_id
        return self._paginate("/bracket", params)

    def get_full_bracket(self, season: int = TOURNAMENT_SEASON) -> Dict[str, List]:
        """
        Fetch the entire bracket and organize by round.

        Returns dict keyed by round name:
            {"first_four": [...], "r64": [...], "r32": [...], ...}
        """
        round_map = {
            7: "first_four",
            1: "r64",
            2: "r32",
            3: "s16",
            4: "e8",
            5: "f4",
            6: "championship",
        }
        result: Dict[str, List] = {v: [] for v in round_map.values()}
        for round_id, key in round_map.items():
            try:
                games = self.get_bracket(season=season, round_id=round_id)
                result[key] = games
                logger.info("Bracket R%d: %d games", round_id, len(games))
            except Exception as exc:
                logger.warning("Failed to fetch bracket round %d: %s", round_id, exc)
        return result

    # ------------------------------------------------------------------
    # Live odds
    # ------------------------------------------------------------------

    def get_odds_by_date(self, date: str) -> List[Dict]:
        """
        Fetch odds for all games on a given date (YYYY-MM-DD).

        Returns list of NCAABBettingOdd objects with spread, ML, total per vendor.
        """
        return self._paginate("/odds", {"dates[]": [date]})

    def get_odds_by_game(self, game_ids: List[int]) -> List[Dict]:
        """Fetch odds for specific game IDs."""
        if not game_ids:
            return []
        # API takes game_ids[] param; requests will serialize list correctly
        return self._paginate("/odds", {"game_ids[]": game_ids})

    def extract_market_ml(
        self,
        odds_records: List[Dict],
        home_team_name: str,
        away_team_name: str,
    ) -> Dict[str, Optional[int]]:
        """
        Extract best-available moneyline from odds records for a matchup.

        Prefers sharp books (Pinnacle first). Returns American odds integers.

        Returns:
            {"home_ml": -350, "away_ml": +280}  or  {"home_ml": None, "away_ml": None}
        """
        best: Optional[Dict] = None
        best_priority = 99

        for rec in odds_records:
            vendor = (rec.get("vendor") or "").lower()
            for i, book in enumerate(PREFERRED_BOOKS):
                if book in vendor and i < best_priority:
                    best = rec
                    best_priority = i
                    break

        if best is None and odds_records:
            best = odds_records[0]   # fallback to first available

        if best is None:
            return {"home_ml": None, "away_ml": None}

        return {
            "home_ml": best.get("moneyline_home_odds"),
            "away_ml": best.get("moneyline_away_odds"),
        }

    # ------------------------------------------------------------------
    # Games
    # ------------------------------------------------------------------

    def get_games(
        self,
        dates: Optional[List[str]] = None,
        team_ids: Optional[List[int]] = None,
        seasons: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """Fetch games with flexible filters."""
        params: Dict[str, Any] = {}
        if dates:
            params["dates[]"] = dates
        if team_ids:
            params["team_ids[]"] = team_ids
        if seasons:
            params["seasons[]"] = seasons
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date
        return self._paginate("/games", params)

    def get_live_tournament_games(self, date: str) -> List[Dict]:
        """Convenience: fetch today's tournament games."""
        return self.get_games(dates=[date], seasons=[TOURNAMENT_SEASON])

    # ------------------------------------------------------------------
    # Team season stats (for TournamentTeam enrichment)
    # ------------------------------------------------------------------

    def get_team_season_stats(
        self,
        season: int = TOURNAMENT_SEASON,
        team_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """
        Fetch season-aggregate stats per team.

        Useful fields: pts, fg3_pct, pace (if available), opp_fg_pct.
        """
        params: Dict[str, Any] = {"season": season}
        if team_ids:
            params["team_ids[]"] = team_ids
        return self._paginate("/team_season_stats", params)

    # ------------------------------------------------------------------
    # Teams lookup
    # ------------------------------------------------------------------

    def get_teams(self, conference_id: Optional[int] = None) -> List[Dict]:
        """Fetch all NCAAB teams. Optional conference filter."""
        params: Dict[str, Any] = {}
        if conference_id:
            params["conference_id"] = conference_id
        return self._paginate("/teams", params)

    def find_team_id(self, name: str) -> Optional[int]:
        """
        Fuzzy-search teams by name. Returns the best-match team ID or None.

        Used to map bracket team names to BDL team IDs for stats lookups.
        """
        teams = self._paginate("/teams", {"search": name})
        if not teams:
            return None
        # Prefer exact match, then first result
        for t in teams:
            if t.get("name", "").lower() == name.lower():
                return t["id"]
        return teams[0]["id"]

    # ------------------------------------------------------------------
    # MLB internal helpers
    # ------------------------------------------------------------------

    def _mlb_get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Single GET against the BDL MLB API. Raises on non-2xx."""
        url = BASE_URL + MLB_PREFIX + path
        resp = self.session.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # MLB Games — Priority 3a ✅
    # ------------------------------------------------------------------

    def get_mlb_games(self, date: str) -> List[MLBGame]:
        """
        Fetch all MLB games for a given date (YYYY-MM-DD).

        Handles cursor pagination. Returns empty list on any API error
        (logged at ERROR level — never silent, never raises).

        Returns:
            list[MLBGame] — Pydantic-validated. Never raw dicts.
        """
        games: List[MLBGame] = []
        cursor: Optional[int] = None
        page = 0
        max_pages = 20
        while page < max_pages:
            params: Dict[str, Any] = {"dates[]": date}
            if cursor is not None:
                params["cursor"] = cursor
            try:
                raw = self._mlb_get("/games", params=params)
                resp = BDLResponse[MLBGame].model_validate(raw)
                games.extend(resp.data)
                cursor = resp.meta.next_cursor
                if cursor is None:
                    break
                page += 1
                time.sleep(0.1)
            except Exception as exc:
                logger.error("get_mlb_games(%s) page=%d failed: %s", date, page, exc)
                break
        return games

    # ------------------------------------------------------------------
    # MLB Odds — Priority 3b
    # ------------------------------------------------------------------

    def get_mlb_odds(self, game_id: int) -> List[MLBBettingOdd]:
        """
        Fetch all sportsbook lines for a specific MLB game.

        BDL returns one record per vendor (fanduel, draftkings, betmgm, etc.).
        Spread and total values arrive as strings — use .spread_home_float etc.
        Handles pagination (unlikely for single-game odds, but correct regardless).
        Returns empty list on any API error (logged, never raises).

        Returns:
            list[MLBBettingOdd] — Pydantic-validated. Never raw dicts.
        """
        odds: List[MLBBettingOdd] = []
        cursor: Optional[int] = None
        page = 0
        max_pages = 10
        while page < max_pages:
            params: Dict[str, Any] = {"game_ids[]": game_id}
            if cursor is not None:
                params["cursor"] = cursor
            try:
                raw = self._mlb_get("/odds", params=params)
                resp = BDLResponse[MLBBettingOdd].model_validate(raw)
                odds.extend(resp.data)
                cursor = resp.meta.next_cursor
                if cursor is None:
                    break
                page += 1
                time.sleep(0.1)
            except Exception as exc:
                logger.error("get_mlb_odds(game_id=%d) page=%d failed: %s", game_id, page, exc)
                break
        return odds

    # ------------------------------------------------------------------
    # MLB Injuries — Priority 3c
    # ------------------------------------------------------------------

    def get_mlb_injuries(self) -> List[MLBInjury]:
        """
        Fetch the full current MLB injury list (IL + DTD).

        Uses cursor pagination — the live endpoint returns 25 items per page
        with next_cursor set. Fetches all pages to return the complete list.
        Returns empty list on any API error (logged, never raises).

        Returns:
            list[MLBInjury] — Pydantic-validated. Never raw dicts.
        """
        injuries: List[MLBInjury] = []
        cursor: Optional[int] = None
        page = 0
        max_pages = 50  # ~1250 players max — generous ceiling
        while page < max_pages:
            params: Dict[str, Any] = {}
            if cursor is not None:
                params["cursor"] = cursor
            try:
                raw = self._mlb_get("/player_injuries", params=params or None)
                resp = BDLResponse[MLBInjury].model_validate(raw)
                injuries.extend(resp.data)
                cursor = resp.meta.next_cursor
                if cursor is None:
                    break
                page += 1
                time.sleep(0.1)
            except Exception as exc:
                logger.error("get_mlb_injuries() page=%d failed: %s", page, exc)
                break
        return injuries

    # ------------------------------------------------------------------
    # MLB Players — Priority 3d
    # ------------------------------------------------------------------

    def get_mlb_player(self, player_id: int) -> Optional[MLBPlayer]:
        """
        Fetch a single MLB player by BDL player ID.

        NOTE: /players/{id} response envelope is unverified. BDL may return
        the object directly or wrapped in {"data": {...}}. If this method fails
        in production, capture the raw response and update accordingly.
        Use search_mlb_players() as the verified alternative.

        Returns None on any error (logged, never raises).
        """
        try:
            raw = self._mlb_get(f"/players/{player_id}")
            # BDL may wrap in {"data": {...}} or return object directly
            if "data" in raw and isinstance(raw["data"], dict):
                return MLBPlayer.model_validate(raw["data"])
            return MLBPlayer.model_validate(raw)
        except Exception as exc:
            logger.error("get_mlb_player(player_id=%d) failed: %s", player_id, exc)
            return None

    def get_all_mlb_players(self) -> List[MLBPlayer]:
        """
        Fetch ALL MLB players using cursor pagination.
        Returns list of Pydantic-validated MLBPlayer objects.
        """
        players: List[MLBPlayer] = []
        cursor: Optional[int] = None
        page = 0
        max_pages = 100  # generous ceiling for all MLB players

        while page < max_pages:
            params: Dict[str, Any] = {"per_page": 100}
            if cursor is not None:
                params["cursor"] = cursor
            try:
                raw = self._mlb_get("/players", params=params)
                resp = BDLResponse[MLBPlayer].model_validate(raw)
                players.extend(resp.data)
                cursor = resp.meta.next_cursor
                if cursor is None:
                    break
                page += 1
                time.sleep(0.1)
            except Exception as exc:
                logger.error("get_all_mlb_players() page=%d failed: %s", page, exc)
                break
        return players

    def search_mlb_players(self, query: str) -> List[MLBPlayer]:
        """
        Search MLB players by name fragment.

        Returns list (may be empty). Handles pagination — a common name
        like "Smith" may span multiple pages.

        Returns:
            list[MLBPlayer] — Pydantic-validated. Never raw dicts.
        """
        players: List[MLBPlayer] = []
        cursor: Optional[int] = None
        page = 0
        max_pages = 10
        while page < max_pages:
            params: Dict[str, Any] = {"search": query}
            if cursor is not None:
                params["cursor"] = cursor
            try:
                raw = self._mlb_get("/players", params=params)
                resp = BDLResponse[MLBPlayer].model_validate(raw)
                players.extend(resp.data)
                cursor = resp.meta.next_cursor
                if cursor is None:
                    break
                page += 1
                time.sleep(0.1)
            except Exception as exc:
                logger.error("search_mlb_players(%r) page=%d failed: %s", query, page, exc)
                break
        return players

    # ------------------------------------------------------------------
    # MLB Player Box Stats -- Priority P11
    # ------------------------------------------------------------------

    def get_mlb_stats(
        self,
        dates: Optional[List[str]] = None,
        player_ids: Optional[List[int]] = None,
        game_ids: Optional[List[int]] = None,
        per_page: int = 100,
    ) -> List[MLBPlayerStats]:
        """
        Fetch per-player per-game box stats from BDL /mlb/v1/stats.

        Natural key: (player.id, game_id).
        Rate stats (avg, obp, slg, era, whip) are floats per live probe.
        Pitching fields are null for hitters; batting fields are null for pitchers.

        Args:
            dates:      List of YYYY-MM-DD strings to filter by game date.
            player_ids: BDL player IDs to filter to specific players.
            per_page:   Page size (max 100 per BDL convention).

        Returns:
            list[MLBPlayerStats] -- Pydantic-validated. Rows that fail validation
            are logged at WARNING and skipped (never raises).
        """
        params: Dict[str, Any] = {"per_page": per_page}
        if dates:
            params["dates[]"] = dates
        if player_ids:
            params["player_ids[]"] = player_ids
        if game_ids:
            params["game_ids[]"] = game_ids

        results: List[MLBPlayerStats] = []
        cursor: Optional[int] = None
        page = 0
        max_pages = 50  # generous ceiling -- daily pull is bounded

        while page < max_pages:
            if cursor is not None:
                params["cursor"] = cursor
            try:
                raw = self._mlb_get("/stats", params=params)
            except Exception as exc:
                logger.error("get_mlb_stats() page=%d HTTP error: %s", page, exc)
                break

            rows = raw.get("data", [])
            meta = raw.get("meta", {})

            for raw_row in rows:
                try:
                    stat = MLBPlayerStats.model_validate(raw_row)
                    results.append(stat)
                except Exception as exc:
                    logger.warning(
                        "get_mlb_stats(): validation failed for row player_id=%s game_id=%s -- %s",
                        raw_row.get("player", {}).get("id") if isinstance(raw_row, dict) else "?",
                        raw_row.get("game_id") if isinstance(raw_row, dict) else "?",
                        exc,
                    )

            next_cursor = meta.get("next_cursor") if isinstance(meta, dict) else None
            if not next_cursor:
                break
            cursor = next_cursor
            page += 1
            time.sleep(0.1)

        return results

    # ------------------------------------------------------------------
    # Player season stats (for tournament_exp field)
    # ------------------------------------------------------------------

    def get_player_season_stats(
        self,
        season: int = TOURNAMENT_SEASON,
        team_ids: Optional[List[int]] = None,
        player_ids: Optional[List[int]] = None,
    ) -> List[Dict]:
        """Fetch per-player season averages. min filter: 5+ mpg recommended."""
        params: Dict[str, Any] = {"season": season}
        if team_ids:
            params["team_ids[]"] = team_ids
        if player_ids:
            params["player_ids[]"] = player_ids
        return self._paginate("/player_season_stats", params)


# ---------------------------------------------------------------------------
# Convenience factory (mirrors singleton pattern used elsewhere in the project)
# ---------------------------------------------------------------------------

_client: Optional[BallDontLieClient] = None


def get_bdl_client() -> BallDontLieClient:
    """Return the shared BallDontLie client (lazy-init)."""
    global _client
    if _client is None:
        _client = BallDontLieClient()
    return _client
