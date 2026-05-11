"""
FanGraphs Rest-of-Season (RoS) Projection Loader

Fetches daily RoS projections from FanGraphs for four systems:
  ATC (30%), THE BAT (30%), Steamer (20%), ZiPS DC (20%)

Uses the FanGraphs JSON API endpoint (not HTML scraping):
  GET https://www.fangraphs.com/api/projections?type={system}&stats={bat|pit}&pos=all&team=0&lg=all&playerid=0

Each row includes:
  - PlayerName: "First Last" format (no "Last, First" conversion needed)
  - playerid:   FanGraphs player ID string
  - xMLBAMID:   MLBAM integer ID (direct bridge to player_identities.mlbam_id)
  - All standard stat columns (HR, R, RBI, SB, AVG, OBP, SLG for batters;
    W, SV, SO, ERA, WHIP, K/9, IP for pitchers)

Lock ID: 100_012 (reserved in daily_ingestion.py)
Cadence: Daily 3 AM ET

See reports/K25_FANGRAPHS_COLUMN_MAP.md for column spec.
"""

import logging
import time
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Projection system definitions
# ---------------------------------------------------------------------------

SYSTEMS = {
    # All use Rest-of-Season (RoS) type codes.
    # Critical: Steamer RoS is "steamerr" (double-r); all others use "r" prefix.
    "steamer":  {"weight": 0.30, "type_param": "steamerr"},
    "atc":      {"weight": 0.30, "type_param": "ratcdc"},
    "thebat":   {"weight": 0.25, "type_param": "rthebat"},
    "zips":     {"weight": 0.15, "type_param": "rzipsdc"},
}

_API_URL = "https://www.fangraphs.com/api/projections"

_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.fangraphs.com/projections",
}

# Batting columns we keep from the API response
_BAT_COLS = {"PlayerName", "playerid", "xMLBAMID", "Team", "PA", "HR", "R", "RBI", "SB", "SO", "AVG", "OBP", "SLG", "OPS"}
# Pitching columns we keep
_PIT_COLS = {"PlayerName", "playerid", "xMLBAMID", "Team", "IP", "W", "SV", "SO", "ERA", "WHIP", "K/9", "GS", "BB"}


def _make_player_id(name: str) -> str:
    """Normalize player name to stable ASCII key — mirrors projections_loader._make_player_id.

    Used as a secondary merge key. Primary key is now mlbam_id via xMLBAMID.
    """
    import re
    if not name:
        return ""
    # Strip generational suffixes
    name = re.sub(r'\b(jr|sr|ii|iii|iv)\.?\s*$', '', name, flags=re.IGNORECASE).strip()
    # Normalize accented characters
    name = (name
            .replace("\xe9", "e").replace("\xe8", "e").replace("\xea", "e")
            .replace("\xe1", "a").replace("\xe0", "a").replace("\xe2", "a")
            .replace("\xf3", "o").replace("\xf2", "o").replace("\xf4", "o")
            .replace("\xfa", "u").replace("\xf9", "u").replace("\xfb", "u").replace("\xfc", "u")
            .replace("\xed", "i").replace("\xec", "i").replace("\xee", "i").replace("\xef", "i")
            .replace("\xf1", "n").replace("\xe7", "c"))
    return (name.lower()
            .replace(" ", "_").replace(".", "").replace("'", "")
            .replace(",", "").replace("-", "_"))


def _fetch_projection_json(system: str, stat_type: str) -> Optional[list]:
    """Fetch projection data from FanGraphs JSON API.

    Args:
        system: type param value (e.g. 'atc', 'thebat', 'steamerr', 'zipsdc')
        stat_type: 'bat' or 'pit'

    Returns:
        List of dicts (one per player), or None on failure.
    """
    params = {
        "type": system,
        "stats": stat_type,
        "pos": "all",
        "team": "0",
        "lg": "all",
        "playerid": "0",
    }
    try:
        resp = requests.get(
            _API_URL,
            params=params,
            headers=_REQUEST_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("FanGraphs API returned empty/unexpected response for %s/%s", system, stat_type)
            return None
        return data
    except requests.exceptions.Timeout:
        logger.error("FanGraphs API timeout for %s/%s", system, stat_type)
        return None
    except Exception as e:
        logger.error("FanGraphs API fetch failed for %s/%s: %s", system, stat_type, e)
        return None


def fetch_system_projections(
    system_key: str,
    stat_type: str = "bat",
) -> Optional[pd.DataFrame]:
    """Fetch and parse RoS projections for one system + stat type.

    Args:
        system_key: one of 'atc', 'thebat', 'steamer', 'zips'
        stat_type: 'bat' or 'pit'

    Returns:
        DataFrame with PlayerName, player_id (name-derived key), mlbam_id (from xMLBAMID),
        and all standard projection stat columns. None on failure.
    """
    cfg = SYSTEMS.get(system_key)
    if not cfg:
        logger.error("Unknown projection system: %s", system_key)
        return None

    rows = _fetch_projection_json(cfg["type_param"], stat_type)
    if not rows:
        return None

    df = pd.DataFrame(rows)

    if "PlayerName" not in df.columns:
        logger.warning("No 'PlayerName' column in %s/%s response", system_key, stat_type)
        return None

    # Expose mlbam_id as a first-class column (primary identity bridge)
    df["mlbam_id"] = pd.to_numeric(df.get("xMLBAMID"), errors="coerce").astype("Int64")

    # Derive name-based player_id for secondary merge compatibility
    df["Name"] = df["PlayerName"]
    df["player_id"] = df["PlayerName"].apply(_make_player_id)
    df["system"] = system_key

    # Keep only the columns we use to avoid memory bloat
    expected = _BAT_COLS if stat_type == "bat" else _PIT_COLS
    available = set(df.columns)
    missing = expected - available - {"PlayerName", "player_id", "mlbam_id", "Name", "xMLBAMID"}
    if missing:
        logger.warning(
            "%s/%s missing columns %s — affected stats will be NaN",
            system_key, stat_type, sorted(missing),
        )

    logger.info(
        "Fetched %d %s projections from %s RoS (JSON API)",
        len(df), stat_type, system_key,
    )
    return df


def fetch_all_ros(
    stat_type: str = "bat",
    delay_seconds: float = 3.0,
) -> dict[str, pd.DataFrame]:
    """Fetch RoS projections from all four systems for one stat category.

    Args:
        stat_type: 'bat' or 'pit'
        delay_seconds: polite delay between requests to FanGraphs

    Returns:
        Dict mapping system_key -> DataFrame. Missing systems are omitted.
    """
    results: dict[str, pd.DataFrame] = {}
    for i, system_key in enumerate(SYSTEMS):
        if i > 0 and delay_seconds > 0:
            time.sleep(delay_seconds)
        df = fetch_system_projections(system_key, stat_type)
        if df is not None and not df.empty:
            results[system_key] = df
    logger.info(
        "RoS %s fetch complete: %d/%d systems succeeded",
        stat_type, len(results), len(SYSTEMS),
    )
    return results


def compute_ensemble_blend(
    projections: dict[str, pd.DataFrame],
    stat_columns: list[str],
) -> Optional[pd.DataFrame]:
    """Compute weighted ensemble blend across available systems.

    Args:
        projections: dict from fetch_all_ros() — system_key -> DataFrame
        stat_columns: list of numeric column names to blend (e.g. ['HR', 'RBI', 'AVG'])

    Returns:
        DataFrame with player_id + mlbam_id + blended stat columns, or None if no data.
        mlbam_id is the primary identity key (from FanGraphs xMLBAMID field).
        player_id is a secondary name-derived key for backward compatibility.
    """
    if not projections:
        return None

    # Key by mlbam_id when available, fall back to player_id (name-derived)
    all_players: dict = {}  # key -> {name, mlbam_id, player_id, stats per system}

    for system_key, df in projections.items():
        weight = SYSTEMS[system_key]["weight"]
        for _, row in df.iterrows():
            # Primary key: MLBAM ID (reliable); fallback: name-derived player_id
            mlbam_raw = row.get("mlbam_id")
            try:
                mlbam_int = int(mlbam_raw) if mlbam_raw is not None and not pd.isna(mlbam_raw) else None
            except (TypeError, ValueError):
                mlbam_int = None

            pid_name = row.get("player_id", "")
            merge_key = f"mlbam:{mlbam_int}" if mlbam_int else f"name:{pid_name}"
            if not merge_key or merge_key in ("mlbam:None", "name:"):
                continue

            if merge_key not in all_players:
                all_players[merge_key] = {
                    "player_id": pid_name,
                    "mlbam_id": mlbam_int,
                    "name": row.get("Name", row.get("PlayerName", "")),
                    "team": row.get("Team", ""),
                    "_weights": 0.0,
                }
                for col in stat_columns:
                    all_players[merge_key][col] = 0.0

            entry = all_players[merge_key]
            for col in stat_columns:
                val = pd.to_numeric(row.get(col, 0), errors="coerce")
                if pd.isna(val):
                    val = 0.0
                entry[col] += val * weight
            entry["_weights"] += weight

    if not all_players:
        return None

    # Normalize by actual weight sum (handles missing systems gracefully)
    rows = []
    for entry in all_players.values():
        w = entry.pop("_weights", 1.0) or 1.0
        for col in stat_columns:
            entry[col] = round(entry[col] / w, 4)
        rows.append(entry)

    blend_df = pd.DataFrame(rows)
    logger.info(
        "Ensemble blend computed for %d players across %d stat columns",
        len(blend_df), len(stat_columns),
    )
    return blend_df


# ---------------------------------------------------------------------------
# Convenience: full daily pipeline
# ---------------------------------------------------------------------------

def run_daily_ros_pipeline() -> dict:
    """Execute the full daily RoS fetch + blend pipeline.

    Returns summary dict with counts and status.
    """
    summary = {"batting": {}, "pitching": {}, "status": "ok"}

    # Batting
    bat_raw = fetch_all_ros("bat", delay_seconds=3.0)
    if bat_raw:
        bat_blend = compute_ensemble_blend(
            bat_raw,
            stat_columns=["HR", "R", "RBI", "SB", "AVG", "OPS"],
        )
        summary["batting"] = {
            "systems_fetched": list(bat_raw.keys()),
            "blend_players": len(bat_blend) if bat_blend is not None else 0,
        }
    else:
        summary["batting"] = {"systems_fetched": [], "blend_players": 0}
        summary["status"] = "partial"

    # Pitching
    pit_raw = fetch_all_ros("pit", delay_seconds=3.0)
    if pit_raw:
        pit_blend = compute_ensemble_blend(
            pit_raw,
            stat_columns=["W", "SV", "SO", "ERA", "WHIP"],
        )
        summary["pitching"] = {
            "systems_fetched": list(pit_raw.keys()),
            "blend_players": len(pit_blend) if pit_blend is not None else 0,
        }
    else:
        summary["pitching"] = {"systems_fetched": [], "blend_players": 0}
        summary["status"] = "partial"

    if not bat_raw and not pit_raw:
        summary["status"] = "failed"

    return summary
