"""
ProbablePitcherFallback — fetches confirmed same-day MLB lineups and resolves TBD opponents.

Uses MLB Stats API /api/v1/schedule with lineup hydration.
SLA: Data must be fetched and cached by 10:00 AM ET daily.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
_ET = ZoneInfo("America/New_York")

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
_TEAM_ALIASES = {
    "TBR": "TB",
    "KCR": "KC",
    "SFG": "SF",
    "SDP": "SD",
    "WSN": "WSH",
    "AZ": "ARI",
    "CHW": "CWS",
}

_LINEUP_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TIMESTAMPS: Dict[str, datetime] = {}
_CACHE_TTL_MINUTES = 30


def _default_target_date() -> str:
    return datetime.now(_ET).date().strftime("%Y-%m-%d")


def _is_cache_fresh(date_str: str) -> bool:
    ts = _CACHE_TIMESTAMPS.get(date_str)
    if ts is None:
        return False
    age_minutes = (datetime.now(_ET) - ts).total_seconds() / 60.0
    return age_minutes < _CACHE_TTL_MINUTES


def _normalize_team_abbr(abbr: Optional[str]) -> str:
    if not abbr:
        return ""
    return _TEAM_ALIASES.get(abbr.upper(), abbr.upper())


def fetch_daily_lineups(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch confirmed MLB starting lineups for target_date from MLB Stats API.

    Args:
        target_date: YYYY-MM-DD string; defaults to today ET.

    Returns:
        Dict mapping game_pk (str) → lineup data dict with keys:
            home_team, away_team, game_time, home_lineup, away_lineup,
            home_probable_pitcher, away_probable_pitcher, lineup_confirmed

    SLA: Should be called by 10:00 AM ET for confirmed lineup data.
    """
    import json
    import urllib.request

    target_date = target_date or _default_target_date()

    if _is_cache_fresh(target_date):
        logger.debug("lineup_cache: HIT for %s", target_date)
        return _LINEUP_CACHE.get(target_date, {})

    url = (
        f"{_MLB_API_BASE}/schedule"
        f"?sportId=1&date={target_date}&gameType=R&hydrate=lineup,probablePitcher"
    )
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "cbb-edge/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("fetch_daily_lineups: MLB API error for %s: %s", target_date, exc)
        return {}

    now_et = datetime.now(_ET)
    result: Dict[str, Any] = {}
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            game_pk = str(game.get("gamePk", ""))
            if not game_pk:
                continue

            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})

            home_lineup = _extract_lineup(home)
            away_lineup = _extract_lineup(away)
            home_probable = _extract_probable_pitcher(home)
            away_probable = _extract_probable_pitcher(away)
            lineup_confirmed = bool(home_lineup or away_lineup)

            result[game_pk] = {
                "game_pk": game_pk,
                "game_date": target_date,
                "game_time": game.get("gameDate"),
                "home_team": _normalize_team_abbr(home.get("team", {}).get("abbreviation", "UNK")),
                "away_team": _normalize_team_abbr(away.get("team", {}).get("abbreviation", "UNK")),
                "home_team_name": home.get("team", {}).get("name", "Unknown"),
                "away_team_name": away.get("team", {}).get("name", "Unknown"),
                "home_lineup": home_lineup,
                "away_lineup": away_lineup,
                "home_probable_pitcher": home_probable,
                "away_probable_pitcher": away_probable,
                "lineup_confirmed": lineup_confirmed,
                "last_updated": now_et.isoformat(),
            }

    _LINEUP_CACHE[target_date] = result
    _CACHE_TIMESTAMPS[target_date] = now_et
    logger.info("fetch_daily_lineups: cached %d games for %s", len(result), target_date)
    return result


def _extract_lineup(team_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract batting lineup from team data."""
    lineup: List[Dict[str, Any]] = []
    for player_entry in team_data.get("lineup", []):
        player = player_entry if isinstance(player_entry, dict) else {}
        person = player.get("person", {})
        if person:
            lineup.append(
                {
                    "player_id": person.get("id"),
                    "full_name": person.get("fullName", "Unknown"),
                    "batting_order": player.get("battingOrder"),
                    "position": player.get("position", {}).get("abbreviation"),
                }
            )
    return lineup


def _extract_probable_pitcher(team_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract probable pitcher from team data."""
    pitcher = team_data.get("probablePitcher") or {}
    if not pitcher:
        return None
    return {
        "player_id": pitcher.get("id"),
        "full_name": pitcher.get("fullName", "TBD"),
    }


def resolve_tbd_opponent(team_abbreviation: str, target_date: Optional[str] = None) -> Optional[str]:
    """
    Resolve 'TBD' opponent for a given team on target_date.

    Args:
        team_abbreviation: 3-letter MLB team abbreviation (e.g. "NYY", "BOS")
        target_date: YYYY-MM-DD; defaults to today ET

    Returns:
        Opponent team abbreviation if found, or None if truly no game / TBD.
    """
    lineups = fetch_daily_lineups(target_date)
    team_upper = _normalize_team_abbr(team_abbreviation)
    for game_data in lineups.values():
        if game_data.get("home_team") == team_upper:
            return game_data.get("away_team")
        if game_data.get("away_team") == team_upper:
            return game_data.get("home_team")
    return None


def get_lineup_card_freshness(target_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Return freshness / SLA status for lineup cards on target_date.

    SLA: lineup cards must be fetched by 10:00 AM ET.
    """
    target_date = target_date or _default_target_date()

    ts = _CACHE_TIMESTAMPS.get(target_date)
    now = datetime.now(_ET)

    try:
        sla_date = date.fromisoformat(target_date)
    except ValueError:
        sla_date = now.date()
    sla_deadline = datetime(
        sla_date.year,
        sla_date.month,
        sla_date.day,
        10,
        0,
        0,
        0,
        tzinfo=_ET,
    )
    sla_due = now >= sla_deadline

    if ts is None:
        return {
            "date": target_date,
            "last_updated": None,
            "age_minutes": None,
            "is_fresh": False,
            "sla_met": not sla_due,
            "game_count": 0,
        }

    age_minutes = (now - ts).total_seconds() / 60.0
    return {
        "date": target_date,
        "last_updated": ts.isoformat(),
        "age_minutes": round(age_minutes, 1),
        "is_fresh": age_minutes < _CACHE_TTL_MINUTES,
        "sla_met": ts <= sla_deadline or not sla_due,
        "game_count": len(_LINEUP_CACHE.get(target_date, {})),
    }
