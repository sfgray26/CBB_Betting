"""Shared fantasy stat contract loader.

Canonical source lives in frontend/lib/fantasy-stat-contract.json so both the
Next.js UI and backend services read the same stat metadata.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

_ROOT_DIR = Path(__file__).resolve().parents[2]
_CONTRACT_CANDIDATE_PATHS = (
    _ROOT_DIR / "frontend" / "lib" / "fantasy-stat-contract.json",
    _ROOT_DIR / "backend" / "utils" / "fantasy_stat_contract.json",
)


@lru_cache(maxsize=1)
def get_fantasy_stat_contract() -> dict[str, Any]:
    for path in _CONTRACT_CANDIDATE_PATHS:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)

    searched = ", ".join(str(path) for path in _CONTRACT_CANDIDATE_PATHS)
    raise FileNotFoundError(f"Fantasy stat contract not found. Checked: {searched}")


_FANTASY_STAT_CONTRACT = get_fantasy_stat_contract()

YAHOO_STAT_ID_FALLBACK: dict[str, str] = dict(_FANTASY_STAT_CONTRACT["fallbackStatIds"])
STAT_LABELS: dict[str, str] = dict(_FANTASY_STAT_CONTRACT["statLabels"])
LOWER_IS_BETTER: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["lowerIsBetter"])
MATCHUP_DISPLAY_ONLY: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["matchupDisplayOnly"])
MATCHUP_STAT_ORDER: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["matchupStatOrder"])
CATEGORY_NEED_STAT_MAP: dict[str, str] = dict(_FANTASY_STAT_CONTRACT["categoryNeedStatMap"])
BATTING_CATEGORIES: tuple[str, ...] = tuple(_FANTASY_STAT_CONTRACT["battingCategories"])
# Hard-coded league scoring categories (League 72586): 9 batting + 9 pitching = 18 total.
# Used as fallback when get_league_settings() is unavailable.
LEAGUE_SCORING_CATEGORIES: frozenset[str] = frozenset(
    _FANTASY_STAT_CONTRACT.get("leagueScoringCategories", [])
)
