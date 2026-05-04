# yahoo_league_client.py
"""Minimal Yahoo Fantasy Sports API client — league settings only."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class YahooLeagueClientError(RuntimeError):
    pass


class YahooLeagueClient:
    BASE_URL = "https://fantasysports.yahooapis.com/fantasy/v2"

    def __init__(self, *, access_token: str, timeout_seconds: float = 15.0):
        self._access_token = access_token
        self._timeout = timeout_seconds

    @classmethod
    def from_env(cls) -> "YahooLeagueClient":
        token = os.environ.get("YAHOO_ACCESS_TOKEN")
        if not token:
            raise YahooLeagueClientError("YAHOO_ACCESS_TOKEN not set")
        return cls(access_token=token)

    def fetch_league_settings(self, league_key: str) -> dict[str, Any]:
        url = f"{self.BASE_URL}/league/{league_key}/settings"
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Accept": "application/json",
        }
        params = {"format": "json"}
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.get(url, headers=headers, params=params)
        except httpx.HTTPError as exc:
            raise YahooLeagueClientError(f"HTTP error: {exc}") from exc
        if resp.status_code != 200:
            raise YahooLeagueClientError(
                f"Unexpected status {resp.status_code}: {resp.text[:200]}"
            )
        return self._unwrap_settings(resp.json())

    @staticmethod
    def _unwrap_settings(raw: dict[str, Any]) -> dict[str, Any]:
        """
        Yahoo wraps responses in fantasy_content.league[1].settings[0]. Tolerate
        minor shape variations.
        """
        try:
            league = raw["fantasy_content"]["league"]
            if isinstance(league, list):
                league_meta = league[0]
                settings_wrapper = league[1].get("settings")
            else:
                league_meta = league
                settings_wrapper = league.get("settings")
            settings = (
                settings_wrapper[0] if isinstance(settings_wrapper, list)
                else settings_wrapper
            )
            # Merge top-level league fields we'll need (league_key).
            if "league_key" in league_meta and "league_key" not in settings:
                settings["league_key"] = league_meta["league_key"]
            return settings
        except (KeyError, IndexError, TypeError) as exc:
            raise YahooLeagueClientError(
                f"Unexpected Yahoo response shape: {exc}"
            ) from exc