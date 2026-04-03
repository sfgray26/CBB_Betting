"""Shared fantasy stat contract loader.

Canonical source lives in frontend/lib/fantasy-stat-contract.json so both the
Next.js UI and backend services read the same stat metadata.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "frontend" / "lib" / "fantasy-stat-contract.json"


@lru_cache(maxsize=1)
def get_fantasy_stat_contract() -> dict[str, Any]:
    with _CONTRACT_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_FANTASY_STAT_CONTRACT = get_fantasy_stat_contract()

YAHOO_STAT_ID_FALLBACK: dict[str, str] = dict(_FANTASY_STAT_CONTRACT["fallbackStatIds"])
STAT_LABELS: dict[str, str] = dict(_FANTASY_STAT_CONTRACT["statLabels"])
LOWER_IS_BETTER: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["lowerIsBetter"])
MATCHUP_DISPLAY_ONLY: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["matchupDisplayOnly"])
MATCHUP_STAT_ORDER: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["matchupStatOrder"])
CATEGORY_NEED_STAT_MAP: dict[str, str] = dict(_FANTASY_STAT_CONTRACT["categoryNeedStatMap"])
BATTING_CATEGORIES: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["battingCategories"])
