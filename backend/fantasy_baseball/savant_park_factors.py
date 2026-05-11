"""
Baseball Savant Statcast park factor snapshot utilities.

Savant publishes park factors as 100-centered indexes. The rest of this app
uses 1.00-centered factors, so the loader normalizes every index field.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "park_factors"
    / "savant_park_factors_2025_3yr.json"
)

INDEX_TO_FACTOR_FIELD = {
    "index_runs": "run_factor",
    "index_hr": "hr_factor",
    "index_hits": "hits_factor",
    "index_woba": "woba_factor",
    "index_wobacon": "wobacon_factor",
    "index_xwobacon": "xwobacon_factor",
    "index_obp": "obp_factor",
    "index_bb": "bb_factor",
    "index_so": "so_factor",
    "index_bacon": "bacon_factor",
    "index_1b": "singles_factor",
    "index_2b": "doubles_factor",
    "index_3b": "triples_factor",
    "index_hardhit": "hardhit_factor",
}


def savant_index_to_factor(value: Any) -> float:
    """Convert Savant's 100-centered index to app's 1.00-centered factor."""
    if value in (None, ""):
        return 1.0

    try:
        return round(float(value) / 100.0, 3)
    except (TypeError, ValueError):
        return 1.0


def load_savant_park_factor_snapshot(
    path: str | Path = DEFAULT_SNAPSHOT_PATH,
) -> list[dict[str, Any]]:
    """
    Load the versioned Savant park factor snapshot.

    Returned rows are normalized for DB upsert and runtime lookup.
    """
    snapshot_path = Path(path)
    with snapshot_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    source = payload.get("source", "baseball_savant_statcast_park_factors")
    source_url = payload.get("source_url")
    season = int(payload["season"])
    rolling_years = int(payload["rolling_years"])
    bat_side = payload.get("bat_side", "All")
    condition = payload.get("condition", "All")
    year_range = payload.get("year_range")

    rows: list[dict[str, Any]] = []
    for record in payload.get("records", []):
        row: dict[str, Any] = {
            "team": record["team"],
            "venue_id": int(record["venue_id"]),
            "park_name": record["park_name"],
            "venue_name": record["park_name"],
            "club": record.get("club"),
            "season": season,
            "year_range": year_range,
            "rolling_years": rolling_years,
            "bat_side": bat_side,
            "condition": condition,
            "n_pa": int(record.get("n_pa") or 0),
            "data_source": source,
            "source_url": source_url,
        }

        for index_field, factor_field in INDEX_TO_FACTOR_FIELD.items():
            row[factor_field] = savant_index_to_factor(record.get(index_field))

        # For pitchers, the environment run factor is the clearest ERA proxy.
        row["era_factor"] = row["run_factor"]
        row["whip_factor"] = row["hits_factor"]
        rows.append(row)

    return rows
