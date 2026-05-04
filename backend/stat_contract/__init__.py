"""Stat contract package — the single source of truth for all stat semantics."""

from .loader import load_contract
from .schema import (
    FantasyStatContract,
    StatEntry,
    SupportingStatEntry,
    ScoringCategories,
    WeeklyRules,
    Aggregation,
    ExternalIds,
    Provenance,
)

# Validated contract singleton — loaded once at import time
CONTRACT: FantasyStatContract = load_contract()

# Convenience constants derived from the loaded contract
SCORING_CATEGORY_CODES: frozenset[str] = frozenset(
    CONTRACT.scoring_categories.batting + CONTRACT.scoring_categories.pitching
)
BATTING_CODES: frozenset[str] = frozenset(CONTRACT.scoring_categories.batting)
PITCHING_CODES: frozenset[str] = frozenset(CONTRACT.scoring_categories.pitching)
LOWER_IS_BETTER: frozenset[str] = frozenset(
    code
    for code, entry in CONTRACT.stats.items()
    if entry.is_scoring_category and entry.direction == "lower_is_better"
)
YAHOO_ID_INDEX: dict[str, str] = dict(CONTRACT.yahoo_id_index)
MATCHUP_DISPLAY_ORDER: list[str] = list(CONTRACT.matchup_display_order)

__all__ = [
    "CONTRACT",
    "SCORING_CATEGORY_CODES",
    "BATTING_CODES",
    "PITCHING_CODES",
    "LOWER_IS_BETTER",
    "YAHOO_ID_INDEX",
    "MATCHUP_DISPLAY_ORDER",
    "load_contract",
    "FantasyStatContract",
    "StatEntry",
    "SupportingStatEntry",
    "ScoringCategories",
    "WeeklyRules",
]
