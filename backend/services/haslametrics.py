"""
Haslametrics ratings scraper for NCAA Division 1 basketball.

Source: https://haslametrics.com/ratings.php

Haslametrics (Erik Haslam) uses play-by-play data to produce
garbage-time-filtered efficiency numbers that are AdjEM-compatible
(Net Efficiency = AdjOE - AdjDE, scale approximately -30 to +30).
It is widely considered part of the "Big Three" of CBB analytics
alongside KenPom and BartTorvik, and is the designated replacement
for EvanMiya in the V9.2 three-source composite.

Integration note (post-GUARDIAN, Apr 7+):
    Wire into backend/services/ratings.py as the third source:
        from backend.services.haslametrics import get_haslametrics_ratings
    Replace EvanMiya's 32.5% weight with Haslametrics.

Failure policy:
    Any exception during fetch or parse returns {} so the model
    falls back gracefully to KenPom+BartTorvik with renormalized
    weights (same behaviour as a missing EvanMiya).
"""

import io
import logging
import os
from typing import Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

from backend.core.circuit_breaker import CircuitBreaker
from backend.services.team_mapping import normalize_team_name

load_dotenv()

logger = logging.getLogger(__name__)

_HASLAMETRICS_URL: str = os.getenv(
    "HASLAMETRICS_URL", "https://haslametrics.com/ratings.php"
)
_CURRENT_YEAR: int = int(os.getenv("SEASON_YEAR", "2026"))

# Module-level circuit breaker — mirrors the pattern in ratings.py.
_haslametrics_cb: CircuitBreaker = CircuitBreaker(
    failure_threshold=3, recovery_timeout=300
)

# Browser-like headers so the server does not reject the scraper.
_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://haslametrics.com/",
}

# Column name candidates for the efficiency margin, tried in order.
# The actual heading on haslametrics.com is "Net" but we defend
# against minor redesigns.
_NET_COLUMN_CANDIDATES = ("Net", "Margin", "AdjEM", "Eff", "EffMargin")

# Column name candidates for the team name column.
_TEAM_COLUMN_CANDIDATES = ("Team", "School", "Name")


def _find_column(columns: pd.Index, candidates: tuple) -> Optional[str]:
    """
    Return the first candidate that appears (case-insensitive) in *columns*,
    or None if none match.
    """
    lower_map = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        match = lower_map.get(cand.lower())
        if match is not None:
            return match
    return None


def fetch_haslametrics_ratings(year: int = None) -> Dict[str, float]:
    """
    Fetch Net Efficiency ratings from haslametrics.com/ratings.php.

    Parses the HTML table on the page with pandas.read_html().  The
    "Net" column (AdjOE - AdjDE) is extracted as the efficiency margin.

    Args:
        year: Season year (default: _CURRENT_YEAR from SEASON_YEAR env var).
              Reserved for future use if Haslametrics adds year-specific URLs.

    Returns:
        Dict mapping normalized team name (KenPom-standard) to Net Efficiency
        float.  Returns {} on any error so callers degrade gracefully.
    """
    _ = year or _CURRENT_YEAR  # reserved; URL is currently year-agnostic

    if not _haslametrics_cb.should_allow_request():
        logger.warning("Haslametrics circuit breaker OPEN — skipping ratings fetch")
        return {}

    try:
        response = requests.get(
            _HASLAMETRICS_URL, headers=_HEADERS, timeout=30
        )
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        _haslametrics_cb.record_failure()
        logger.warning("Haslametrics HTTP error: %s", exc)
        return {}
    except Exception as exc:
        _haslametrics_cb.record_failure()
        logger.warning("Haslametrics request failed: %s", exc)
        return {}

    try:
        tables = pd.read_html(io.StringIO(response.text))
    except Exception as exc:
        _haslametrics_cb.record_failure()
        logger.warning("Haslametrics HTML parse failed (no tables found): %s", exc)
        return {}

    if not tables:
        _haslametrics_cb.record_failure()
        logger.warning("Haslametrics: read_html returned no tables")
        return {}

    # Find the first table that contains both a team column and a net-
    # efficiency column.  Defend against ancillary tables (nav, footer, etc.).
    target_df: Optional[pd.DataFrame] = None
    team_col: Optional[str] = None
    net_col: Optional[str] = None

    for df in tables:
        t = _find_column(df.columns, _TEAM_COLUMN_CANDIDATES)
        n = _find_column(df.columns, _NET_COLUMN_CANDIDATES)
        if t is not None and n is not None:
            target_df = df
            team_col = t
            net_col = n
            break

    if target_df is None:
        _haslametrics_cb.record_failure()
        logger.warning(
            "Haslametrics: could not locate efficiency table. "
            "Column candidates tried: team=%s net=%s. "
            "Tables found: %d. First table columns: %s",
            _TEAM_COLUMN_CANDIDATES,
            _NET_COLUMN_CANDIDATES,
            len(tables),
            list(tables[0].columns) if tables else [],
        )
        return {}

    raw_results: Dict[str, float] = {}
    for _, row in target_df.iterrows():
        raw_name = row.get(team_col)
        raw_net = row.get(net_col)

        if pd.isna(raw_name) or pd.isna(raw_net):
            continue

        name = str(raw_name).strip()
        if not name:
            continue

        try:
            value = float(raw_net)
        except (ValueError, TypeError):
            logger.debug("Haslametrics: non-numeric Net for '%s': %r", name, raw_net)
            continue

        raw_results[name] = value

    if not raw_results:
        _haslametrics_cb.record_failure()
        logger.warning(
            "Haslametrics: table found but parsed 0 valid rows "
            "(team_col=%s, net_col=%s)", team_col, net_col
        )
        return {}

    # Normalize team names to KenPom-standard via team_mapping.
    valid_names = list(raw_results.keys())
    normalized: Dict[str, float] = {}
    for raw_name, value in raw_results.items():
        canon = normalize_team_name(raw_name, valid_names)
        if canon is not None:
            normalized[canon] = value
        else:
            # Keep the raw name rather than silently drop — callers
            # can still fuzzy-match against it.
            normalized[raw_name] = value

    _haslametrics_cb.record_success()
    logger.info(
        "Haslametrics: loaded %d teams (net_col='%s')", len(normalized), net_col
    )
    return normalized


def get_haslametrics_ratings() -> Dict[str, float]:
    """
    Public wrapper for fetch_haslametrics_ratings().

    Mirrors the thin public-wrapper pattern used in ratings.py
    (e.g., get_kenpom_ratings, get_barttorvik_ratings).  Call this
    from ratings.py after the GUARDIAN window lifts.
    """
    return fetch_haslametrics_ratings()
