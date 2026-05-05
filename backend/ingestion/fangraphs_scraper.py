"""
FanGraphs pitcher quality metrics scraper.

Fetches Stuff+, Location+, and Pitching+ via pybaseball.fg_pitching_data().
ID mapping: FanGraphs IDfg -> MLBAM ID via pybaseball Chadwick bureau lookup.

Failure contract: any error returns an empty DataFrame and logs a warning.
"""
from __future__ import annotations

import logging
import pandas as pd

logger = logging.getLogger(__name__)

_COLS = ["mlbam_id", "player_name", "stuff_plus", "location_plus", "pitching_plus"]


def fetch_pitcher_quality(season: int = 2026) -> pd.DataFrame:
    """
    Fetch Stuff+, Location+, Pitching+ from FanGraphs for `season`.

    Returns DataFrame with columns:
        mlbam_id      str   - MLBAM player ID (str, matches statcast_pitcher_metrics PK)
        player_name   str   - Player name from FanGraphs
        stuff_plus    float - Stuff+ metric (100 = league avg)
        location_plus float - Location+ / command metric
        pitching_plus float - Pitching+ composite

    All metric columns are nullable (None/NaN when FanGraphs doesn't publish the value).
    Returns empty DataFrame (same columns, zero rows) on any error.
    """
    try:
        from pybaseball import fg_pitching_data, playerid_reverse_lookup

        raw = fg_pitching_data(season, season, qual=0)
        if raw is None or raw.empty:
            logger.warning(
                "fangraphs_scraper: fg_pitching_data returned empty for season=%d", season
            )
            return _empty_df()

        fg_ids = raw["IDfg"].dropna().astype(int).tolist()
        if not fg_ids:
            logger.warning("fangraphs_scraper: no IDfg values in FanGraphs pitcher data")
            return _empty_df()

        id_map_df = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
        if id_map_df is None or id_map_df.empty:
            logger.warning(
                "fangraphs_scraper: playerid_reverse_lookup returned empty — ID mapping failed"
            )
            return _empty_df()

        id_map: dict[int, str] = {
            int(r["key_fangraphs"]): str(int(r["key_mlbam"]))
            for _, r in id_map_df.iterrows()
            if pd.notna(r.get("key_fangraphs")) and pd.notna(r.get("key_mlbam"))
        }

        rows = []
        for _, row in raw.iterrows():
            fg_id_raw = row.get("IDfg")
            if pd.isna(fg_id_raw):
                continue
            mlbam_id = id_map.get(int(fg_id_raw))
            if mlbam_id is None:
                continue  # partial coverage expected -- skip silently

            rows.append({
                "mlbam_id": mlbam_id,
                "player_name": str(row.get("Name") or ""),
                "stuff_plus": _to_float(row.get("Stuff+")),
                "location_plus": _to_float(row.get("Location+")),
                "pitching_plus": _to_float(row.get("Pitching+")),
            })

        if not rows:
            logger.warning(
                "fangraphs_scraper: 0 rows after IDfg->mlbam_id mapping. "
                "fg_ids fetched=%d, mapped=%d",
                len(fg_ids), len(id_map),
            )
            return _empty_df()

        out = pd.DataFrame(rows, columns=_COLS)
        logger.info(
            "fangraphs_scraper: fetched %d / %d pitchers with quality metrics for %d",
            len(out), len(fg_ids), season,
        )
        return out

    except Exception as exc:
        logger.warning(
            "fangraphs_scraper: unexpected error fetching pitcher quality: %s", exc
        )
        return _empty_df()


def _to_float(val) -> float | None:
    try:
        if val is None or (hasattr(val, '__class__') and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_COLS)
