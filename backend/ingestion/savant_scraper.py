"""
PR 2.x — Baseball Savant scrapers for advanced metrics.

Fetches leaderboards from baseballsavant.mlb.com for:
  - Sprint speed (batter running speed)

Pitcher quality metrics (Stuff+, Location+, Pitching+) are sourced from
FanGraphs via pybaseball — see backend/ingestion/fangraphs_scraper.py.

Failure contract: any HTTP or parse error returns an empty DataFrame and
logs a warning — callers must not crash if this returns empty.
"""
from __future__ import annotations

import io
import logging

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_SPRINT_SPEED_URL = (
    "https://baseballsavant.mlb.com/leaderboard/sprint_speed"
    "?year={year}&position=&team=&min=0&csv=true"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/html,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Omit Accept-Encoding to avoid Brotli (br) responses that requests can't decode
    "Connection": "keep-alive",
    "Referer": "https://baseballsavant.mlb.com/",
}

_TIMEOUT_SECONDS = 30


def fetch_sprint_speed(year: int = 2026) -> pd.DataFrame:
    """
    Fetch the Baseball Savant sprint speed leaderboard for `year`.

    Returns a DataFrame with columns:
        mlbam_id      int   — MLB Advanced Media player ID
        player_name   str   — "Last, First" format from Savant
        sprint_speed  float — ft/s

    Returns an empty DataFrame (same columns, zero rows) on any failure.
    """
    url = _SPRINT_SPEED_URL.format(year=year)
    try:
        resp = requests.get(url, headers=_BROWSER_HEADERS, timeout=_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return _parse_sprint_speed_csv(resp.text)
    except requests.RequestException as exc:
        logger.warning("savant_scraper: HTTP error fetching sprint speed: %s", exc)
    except Exception as exc:
        logger.warning("savant_scraper: unexpected error fetching sprint speed: %s", exc)
    return _empty_df(columns=["mlbam_id", "player_name", "sprint_speed"])


def _parse_sprint_speed_csv(csv_text: str) -> pd.DataFrame:
    """Parse Savant sprint speed CSV into the canonical three-column DataFrame."""
    # on_bad_lines='skip' tolerates player names with embedded commas or extra fields
    df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")

    if df.empty:
        return _empty_df(columns=["mlbam_id", "player_name", "sprint_speed"])

    # Savant CSV columns vary slightly by year; try common names
    id_col = _find_column(df, ("player_id", "mlbam_id", "batter", "pitcher", "id"))
    speed_col = _find_column(df, ("sprint_speed", "speed"))
    name_col = _find_column(df, ("player_name", "name", "last_name,first_name"))

    if id_col is None or speed_col is None:
        raise ValueError(
            f"Could not locate required columns in Savant CSV. "
            f"Found: {list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "mlbam_id": pd.to_numeric(df[id_col], errors="coerce"),
            "player_name": df[name_col].astype(str) if name_col else "",
            "sprint_speed": pd.to_numeric(df[speed_col], errors="coerce"),
        }
    )

    out = out.dropna(subset=["mlbam_id", "sprint_speed"])
    out["mlbam_id"] = out["mlbam_id"].astype(int)
    return out.reset_index(drop=True)


def _find_column(df: pd.DataFrame, candidates: tuple) -> str | None:
    """Return the first column name from candidates that exists in df."""
    lower_map = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _empty_df(columns: list) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)
