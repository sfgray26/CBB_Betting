"""
Yahoo Fantasy Sports API client — OAuth 2.0 with auto-refresh.

Authentication flow (one-time setup):
    python -m backend.fantasy_baseball.yahoo_client --auth

This opens a browser, you authorize, paste the redirect URL back,
and the refresh token is saved to .env automatically.

Subsequent calls use the stored refresh token to obtain fresh access tokens.

Yahoo Fantasy API base: https://fantasysports.yahooapis.com/fantasy/v2/
League key format:      mlb.l.{YAHOO_LEAGUE_ID}
"""

import json
import logging
import os
import re
import time
import webbrowser
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse, parse_qs

import requests
from dotenv import load_dotenv, set_key

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
YAHOO_AUTH_URL = "https://api.login.yahoo.com/oauth2/request_auth"
YAHOO_TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
YAHOO_API_BASE = "https://fantasysports.yahooapis.com/fantasy/v2"
YAHOO_SPORT = "mlb"

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"


class YahooAuthError(Exception):
    pass


class YahooAPIError(Exception):
    def __init__(self, message: str, status_code: int = 0):
        super().__init__(message)
        self.status_code = status_code


class YahooFantasyClient:
    """
    Thin wrapper around Yahoo Fantasy Sports API v2.

    Usage:
        client = YahooFantasyClient()
        league = client.get_league()
        roster = client.get_my_roster()
    """

    def __init__(self):
        self.client_id = os.getenv("YAHOO_CLIENT_ID", "")
        self.client_secret = os.getenv("YAHOO_CLIENT_SECRET", "")
        self.league_id = os.getenv("YAHOO_LEAGUE_ID", "72586")
        self.league_key = f"{YAHOO_SPORT}.l.{self.league_id}"
        self._refresh_token = os.getenv("YAHOO_REFRESH_TOKEN", "")
        self._access_token = os.getenv("YAHOO_ACCESS_TOKEN", "")
        self._token_expiry: float = 0.0
        self._session = requests.Session()

        if not self.client_id or not self.client_secret:
            raise YahooAuthError(
                "YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET must be set in .env"
            )

    # ------------------------------------------------------------------
    # OAuth 2.0 — Authorization Code Flow
    # ------------------------------------------------------------------

    def get_authorization_url(self) -> str:
        params = {
            "client_id": self.client_id,
            "redirect_uri": "oob",
            "response_type": "code",
            "language": "en-us",
        }
        return f"{YAHOO_AUTH_URL}?{urlencode(params)}"

    def exchange_code_for_tokens(self, auth_code: str) -> dict:
        """Exchange authorization code for access + refresh tokens."""
        response = requests.post(
            YAHOO_TOKEN_URL,
            data={
                "grant_type": "authorization_code",
                "code": auth_code.strip(),
                "redirect_uri": "oob",
            },
            auth=(self.client_id, self.client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            raise YahooAuthError(
                f"Token exchange failed: {response.status_code} — {response.text}"
            )
        tokens = response.json()
        self._store_tokens(tokens)
        return tokens

    def _refresh_access_token(self) -> None:
        """Use refresh token to get a new access token."""
        if not self._refresh_token:
            raise YahooAuthError(
                "No refresh token stored. Run: python -m backend.fantasy_baseball.yahoo_client --auth"
            )
        response = requests.post(
            YAHOO_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": self._refresh_token,
            },
            auth=(self.client_id, self.client_secret),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code != 200:
            raise YahooAuthError(
                f"Token refresh failed: {response.status_code} — {response.text}"
            )
        tokens = response.json()
        self._store_tokens(tokens)

    def _store_tokens(self, tokens: dict) -> None:
        """Persist tokens to .env and update in-memory state.

        On Railway (no writable .env), the write fails silently —
        tokens are still live in-memory for the process lifetime.
        Set YAHOO_REFRESH_TOKEN in Railway env vars directly after
        completing the one-time auth flow locally.
        """
        self._access_token = tokens["access_token"]
        self._refresh_token = tokens.get("refresh_token", self._refresh_token)
        self._token_expiry = time.time() + tokens.get("expires_in", 3600) - 60
        # Write back to .env — best-effort; fails silently on Railway
        try:
            set_key(str(ENV_PATH), "YAHOO_ACCESS_TOKEN", self._access_token)
            set_key(str(ENV_PATH), "YAHOO_REFRESH_TOKEN", self._refresh_token)
            logger.info("Yahoo tokens refreshed and persisted to .env")
        except Exception as exc:
            logger.info("Yahoo tokens refreshed (in-memory only — .env not writable: %s)", exc)

    def _ensure_token(self) -> None:
        if time.time() >= self._token_expiry or not self._access_token:
            self._refresh_access_token()

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET from Yahoo API, returning parsed JSON."""
        self._ensure_token()
        url = f"{YAHOO_API_BASE}/{path.lstrip('/')}"
        default_params = {"format": "json"}
        if params:
            default_params.update(params)

        for attempt in range(3):
            resp = self._session.get(
                url,
                params=default_params,
                headers={"Authorization": f"Bearer {self._access_token}"},
            )
            if resp.status_code == 401:
                # Token may have just expired mid-request
                self._refresh_access_token()
                continue
            if resp.status_code == 999:
                wait = 2 ** attempt
                logger.warning(f"Yahoo rate limit hit, waiting {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                raise YahooAPIError(
                    f"Yahoo API error {resp.status_code}: {resp.text[:300]}",
                    resp.status_code,
                )
            return resp.json()

        raise YahooAPIError("Yahoo API failed after 3 attempts")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flatten_league_section(raw) -> dict:
        """
        Yahoo returns league[N] as either:
          (a) a merged dict  {"name": "...", "teams": {...}, ...}
          (b) a list of single-key dicts  [{"name": "..."}, {"teams": {...}}, ...]

        This helper normalises both shapes to a plain dict.
        """
        if isinstance(raw, list):
            out = {}
            for item in raw:
                if isinstance(item, dict):
                    out.update(item)
            return out
        return raw if isinstance(raw, dict) else {}

    def _league_section(self, data: dict, index: int) -> dict:
        """Extract and flatten league[index] from a fantasy_content response."""
        return self._flatten_league_section(
            data["fantasy_content"]["league"][index]
        )

    def _team_section(self, data: dict) -> dict:
        """
        Flatten the entire team array from a fantasy_content response.

        Yahoo returns team as either:
          (a) [[meta_dict, ...], {"roster": {...}}]   — 2-element list
          (b) [{"team_key": ...}, {"name": ...}, ..., {"roster": {...}}]  — flat list

        We flatten ALL dict items across the outer list, skipping nested lists,
        so the result always contains keys like "roster", "name", etc.
        """
        return self._flatten_league_section(
            data["fantasy_content"]["team"]
        )

    # ------------------------------------------------------------------
    # League endpoints
    # ------------------------------------------------------------------

    def get_league(self) -> dict:
        """League metadata: name, scoring type, settings."""
        data = self._get(f"league/{self.league_key}")
        return self._league_section(data, 0)

    def get_league_settings(self) -> dict:
        data = self._get(f"league/{self.league_key}/settings")
        return self._league_section(data, 0)

    def get_standings(self) -> list[dict]:
        data = self._get(f"league/{self.league_key}/standings")
        sec = self._league_section(data, 1)
        teams_raw = sec.get("standings", [{}])[0].get("teams", {})
        teams = []
        count = int(teams_raw.get("count", 0))
        for i in range(count):
            team_data = teams_raw[str(i)]["team"]
            teams.append(self._parse_team(team_data))
        return teams

    def get_all_teams(self) -> list[dict]:
        data = self._get(f"league/{self.league_key}/teams")
        teams_raw = self._league_section(data, 1).get("teams", {})
        teams = []
        count = int(teams_raw.get("count", 0))
        for i in range(count):
            team_data = teams_raw[str(i)]["team"]
            teams.append(self._parse_team(team_data))
        return teams

    def get_my_team_key(self) -> str:
        """Return the team key for the authenticated user's team."""
        data = self._get(f"league/{self.league_key}/teams")
        teams_raw = self._league_section(data, 1).get("teams", {})
        count = int(teams_raw.get("count", 0))
        for i in range(count):
            team_list = teams_raw[str(i)]["team"]
            # team_list[0] is a mixed list of dicts and strings — guard with isinstance
            meta = {}
            if isinstance(team_list[0], list):
                for d in team_list[0]:
                    if isinstance(d, dict):
                        meta.update(d)
            if meta.get("is_owned_by_current_login"):
                return meta["team_key"]
        raise YahooAPIError("Could not find your team — are you authenticated?")

    # ------------------------------------------------------------------
    # Roster endpoints
    # ------------------------------------------------------------------

    def get_roster(self, team_key: Optional[str] = None) -> list[dict]:
        """Return full roster for team_key (defaults to authenticated user's team)."""
        if team_key is None:
            team_key = self.get_my_team_key()
        data = self._get(f"team/{team_key}/roster/players")
        players_raw = (
            self._team_section(data)
            .get("roster", {})
            .get("0", {})
            .get("players", {})
        )
        players = []
        count = int(players_raw.get("count", 0))
        for i in range(count):
            player_data = players_raw[str(i)]["player"]
            players.append(self._parse_player(player_data))
        return players

    def get_all_rosters(self) -> dict[str, list[dict]]:
        """All rosters keyed by team_key."""
        teams = self.get_all_teams()
        rosters = {}
        for team in teams:
            try:
                rosters[team["team_key"]] = self.get_roster(team["team_key"])
            except YahooAPIError as e:
                logger.warning(f"Failed to fetch roster for {team['name']}: {e}")
        return rosters

    # ------------------------------------------------------------------
    # Player endpoints
    # ------------------------------------------------------------------

    def get_player(self, player_key: str) -> dict:
        data = self._get(f"player/{player_key}")
        return self._parse_player(data["fantasy_content"]["player"][0])

    def search_players(self, name: str, status: str = "A") -> list[dict]:
        """
        Search available players by name.
        status: A=available, T=taken, W=on waivers, FA=free agent
        """
        data = self._get(
            f"league/{self.league_key}/players",
            params={"search": name, "status": status},
        )
        players_raw = self._league_section(data, 1).get("players", {})
        return self._parse_players_block(players_raw)

    def get_free_agents(self, position: str = "", start: int = 0, count: int = 25) -> list[dict]:
        """Paginated free agent list, optionally filtered by position."""
        params = {"status": "FA", "start": start, "count": count}
        if position:
            params["position"] = position
        data = self._get(f"league/{self.league_key}/players", params=params)
        players_raw = self._league_section(data, 1).get("players", {})
        return self._parse_players_block(players_raw)

    def get_player_stats(self, player_key: str, stat_type: str = "season") -> dict:
        """
        stat_type: 'season', 'average', 'projected_season'
        """
        data = self._get(f"player/{player_key}/stats;type={stat_type}")
        player = data["fantasy_content"]["player"]
        return self._parse_player_with_stats(player)

    def get_waiver_players(self, start: int = 0, count: int = 25) -> list[dict]:
        params = {"status": "W", "start": start, "count": count}
        data = self._get(f"league/{self.league_key}/players", params=params)
        players_raw = self._league_section(data, 1).get("players", {})
        return self._parse_players_block(players_raw)

    # ------------------------------------------------------------------
    # Draft endpoints
    # ------------------------------------------------------------------

    def get_draft_results(self) -> list[dict]:
        """Return completed draft picks (empty until draft runs)."""
        data = self._get(f"league/{self.league_key}/draftresults")
        picks_raw = (
            self._league_section(data, 1)
            .get("draft_results", {})
            .get("0", {})
            .get("draft_results", {})
        )
        picks = []
        count = int(picks_raw.get("count", 0))
        for i in range(count):
            pick = picks_raw[str(i)]["draft_result"][0]
            picks.append({
                "pick": pick.get("pick"),
                "round": pick.get("round"),
                "team_key": pick.get("team_key"),
                "player_key": pick.get("player_key"),
            })
        return picks

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Lineup management
    # ------------------------------------------------------------------

    def get_lineup(self, team_key: Optional[str] = None, date: Optional[str] = None) -> list[dict]:
        """Fetch current lineup for a date (YYYY-MM-DD). Defaults to today."""
        if team_key is None:
            team_key = self.get_my_team_key()
        if date is None:
            from datetime import datetime
            date = datetime.utcnow().strftime("%Y-%m-%d")
        data = self._get(f"team/{team_key}/roster/players", params={"date": date})
        players_raw = (
            self._team_section(data)
            .get("roster", {})
            .get("0", {})
            .get("players", {})
        )
        players = []
        count = int(players_raw.get("count", 0))
        for i in range(count):
            player_data = players_raw[str(i)]["player"]
            p = self._parse_player(player_data)
            # selected_position may be in the player data
            if isinstance(player_data, list):
                for item in player_data:
                    if isinstance(item, list):
                        for sub in item:
                            if isinstance(sub, dict) and "selected_position" in sub:
                                sp = sub["selected_position"]
                                if isinstance(sp, list):
                                    for spd in sp:
                                        if isinstance(spd, dict) and "position" in spd:
                                            p["selected_position"] = spd["position"]
            players.append(p)
        return players

    def set_lineup(self, team_key: Optional[str] = None, date: Optional[str] = None,
                   lineup: Optional[list[dict]] = None) -> bool:
        """
        Set lineup for a given date.

        lineup: list of {player_key: str, position: str}
            position: 'C','1B','2B','3B','SS','OF','Util','SP','RP','P','BN','DL'

        Returns True on success.
        """
        if team_key is None:
            team_key = self.get_my_team_key()
        if date is None:
            from datetime import datetime
            date = datetime.utcnow().strftime("%Y-%m-%d")
        if not lineup:
            return False

        self._ensure_token()
        # Build XML payload (Yahoo lineup PUT requires XML)
        player_xml = "\n".join(
            f'<player><player_key>{p["player_key"]}</player_key>'
            f'<position>{p["position"]}</position></player>'
            for p in lineup
        )
        xml_body = (
            f'<?xml version="1.0"?>'
            f'<fantasy_content><roster><coverage_type>date</coverage_type>'
            f'<date>{date}</date><players>{player_xml}</players></roster></fantasy_content>'
        )
        url = f"{YAHOO_API_BASE}/team/{team_key}/roster"
        resp = self._session.put(
            url,
            data=xml_body.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/xml",
            },
        )
        if resp.status_code not in (200, 204):
            raise YahooAPIError(f"set_lineup failed: {resp.status_code} — {resp.text[:300]}", resp.status_code)
        return True

    def get_scoreboard(self, week: Optional[int] = None) -> list[dict]:
        """Fetch matchup scoreboard for a week (defaults to current)."""
        path = f"league/{self.league_key}/scoreboard"
        params = {}
        if week:
            params["week"] = week
        data = self._get(path, params=params if params else None)
        matchups_raw = (
            self._league_section(data, 1)
            .get("scoreboard", {})
            .get("0", {})
            .get("matchups", {})
        )
        matchups = []
        count = int(matchups_raw.get("count", 0))
        for i in range(count):
            matchups.append(matchups_raw.get(str(i), {}).get("matchup", {}))
        return matchups

    def add_drop_player(self, add_player_key: str, drop_player_key: Optional[str] = None,
                        team_key: Optional[str] = None) -> bool:
        """Add a free agent (and optionally drop a player)."""
        if team_key is None:
            team_key = self.get_my_team_key()
        self._ensure_token()
        drop_xml = (
            f'<player><player_key>{drop_player_key}</player_key>'
            f'<transaction_data><type>drop</type>'
            f'<destination_team_key>LW</destination_team_key></transaction_data></player>'
        ) if drop_player_key else ""
        xml_body = (
            f'<?xml version="1.0"?><fantasy_content><transaction>'
            f'<type>{"add/drop" if drop_player_key else "add"}</type>'
            f'<trader_team_key>{team_key}</trader_team_key>'
            f'<players>'
            f'<player><player_key>{add_player_key}</player_key>'
            f'<transaction_data><type>add</type>'
            f'<destination_team_key>{team_key}</destination_team_key></transaction_data></player>'
            f'{drop_xml}</players></transaction></fantasy_content>'
        )
        url = f"{YAHOO_API_BASE}/league/{self.league_key}/transactions"
        resp = self._session.post(
            url,
            data=xml_body.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/xml",
            },
        )
        if resp.status_code not in (200, 201):
            raise YahooAPIError(f"add/drop failed: {resp.status_code} — {resp.text[:300]}", resp.status_code)
        return True

    def get_transactions(self, t_type: str = "add,drop,trade") -> list[dict]:
        """Recent transactions for the league."""
        data = self._get(
            f"league/{self.league_key}/transactions",
            params={"type": t_type},
        )
        txns_raw = self._league_section(data, 1).get("transactions", {})
        txns = []
        count = int(txns_raw.get("count", 0))
        for i in range(count):
            txns.append(txns_raw[str(i)].get("transaction", {}))
        return txns

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_team(team_list: list) -> dict:
        """Flatten Yahoo's nested team structure."""
        meta = {}
        if isinstance(team_list[0], list):
            for item in team_list[0]:
                if isinstance(item, dict):
                    meta.update(item)
        return {
            "team_key": meta.get("team_key"),
            "team_id": meta.get("team_id"),
            "name": meta.get("name"),
            "manager": meta.get("managers", [{}])[0].get("manager", {}).get("nickname"),
        }

    @staticmethod
    def _parse_player(player_list: list) -> dict:
        """Flatten Yahoo's nested player structure."""
        meta = {}
        if isinstance(player_list, list) and isinstance(player_list[0], list):
            for item in player_list[0]:
                if isinstance(item, dict):
                    meta.update(item)
        elif isinstance(player_list, list):
            for item in player_list:
                if isinstance(item, dict):
                    meta.update(item)

        # Extract eligible positions
        positions_raw = meta.get("eligible_positions", [])
        positions = []
        if isinstance(positions_raw, list):
            positions = [p.get("position") for p in positions_raw if isinstance(p, dict)]
        elif isinstance(positions_raw, dict):
            pos = positions_raw.get("position")
            positions = [pos] if pos else []

        return {
            "player_key": meta.get("player_key"),
            "player_id": meta.get("player_id"),
            "name": meta.get("full_name") or meta.get("name", {}).get("full"),
            "team": meta.get("editorial_team_abbr"),
            "positions": [p for p in positions if p],
            "status": meta.get("status"),
            "injury_note": meta.get("injury_note"),
            "is_undroppable": meta.get("is_undroppable", 0),
        }

    def _parse_player_with_stats(self, player: list) -> dict:
        parsed = self._parse_player(player[0] if isinstance(player[0], list) else player)
        stats_raw = {}
        for item in player:
            if isinstance(item, dict) and "player_stats" in item:
                stats_list = item["player_stats"].get("stats", [])
                for stat_entry in stats_list:
                    if isinstance(stat_entry, dict):
                        s = stat_entry.get("stat", {})
                        stats_raw[s.get("stat_id")] = s.get("value")
        parsed["stats"] = stats_raw
        return parsed

    def _parse_players_block(self, players_raw: dict) -> list[dict]:
        players = []
        count = int(players_raw.get("count", 0))
        for i in range(count):
            player_data = players_raw[str(i)]["player"]
            players.append(self._parse_player(player_data))
        return players


# ---------------------------------------------------------------------------
# CLI: one-time auth setup
# ---------------------------------------------------------------------------

def run_auth_flow():
    """Interactive OAuth setup — run once to get refresh token."""
    load_dotenv()
    client = YahooFantasyClient()

    auth_url = client.get_authorization_url()
    print("\n" + "=" * 60)
    print("YAHOO FANTASY — ONE-TIME AUTHORIZATION")
    print("=" * 60)
    print(f"\nStep 1: Open this URL in your browser:\n\n  {auth_url}\n")
    try:
        webbrowser.open(auth_url)
        print("(Browser opened automatically)")
    except Exception:
        pass

    print("\nStep 2: Authorize the app")
    print("Step 3: Yahoo will show you a 6-digit code (or redirect to oob://)")
    code = input("\nEnter the authorization code: ").strip()

    tokens = client.exchange_code_for_tokens(code)
    print("\n✓ Authorization successful!")
    print(f"  Access token expires in: {tokens.get('expires_in', '?')}s")
    print("  Refresh token saved to .env")

    # Quick test
    try:
        league = client.get_league()
        print(f"\n✓ Connected to league: {league.get('name')}")
        my_key = client.get_my_team_key()
        print(f"  Your team key: {my_key}")
    except Exception as e:
        print(f"\n⚠ Auth succeeded but test call failed: {e}")
        print("  Tokens are saved — try again after retrying.")


if __name__ == "__main__":
    import sys
    if "--auth" in sys.argv:
        run_auth_flow()
    else:
        print("Usage: python -m backend.fantasy_baseball.yahoo_client --auth")
        print("  Runs one-time OAuth setup to get your refresh token.")
