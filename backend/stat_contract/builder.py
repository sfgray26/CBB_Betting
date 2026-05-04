"""Pure contract construction from Yahoo league settings."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from .schema import (
    Aggregation,
    ExternalIds,
    FantasyStatContract,
    Provenance,
    ScoringCategories,
    StatEntry,
    SupportingStatEntry,
    WeeklyRules,
)
from .registry import (
    ALL_STATS,
    BATTING_STATS,
    DISPLAY_STATS,
    PITCHING_STATS,
    SUPPORTING_STATS,
    YAHOO_ID_TO_CANONICAL,
    StatDefinition,
)


GENERATOR_VERSION = "fantasy_stat_contract_v2.0.0"
SCHEMA_URL = "https://fantasy.app/schemas/fantasy_stat_contract/v2.json"


class UnknownYahooStatError(ValueError):
    """Raised when Yahoo returns a stat ID we have no canonical mapping for."""


class ContractBuildError(ValueError):
    """Raised when the Yahoo settings response is malformed or ambiguous."""


def build_contract(
    *,
    yahoo_settings: dict[str, Any],
    league_id: str,
    season: int,
    now: datetime | None = None,
) -> FantasyStatContract:
    """
    Build a validated FantasyStatContract from a Yahoo league settings response.

    Args:
        yahoo_settings: The parsed JSON body of Yahoo's league/{key}/settings.
        league_id: Internal league identifier (e.g. "yahoo_2026_league_12345").
        season: Four-digit season year.
        now: Injectable clock for deterministic tests.

    Raises:
        UnknownYahooStatError: Yahoo returned a stat_id missing from the registry.
        ContractBuildError: The response is missing required fields or is ambiguous.
    """
    now = now or datetime.now(timezone.utc)

    scored_yahoo_ids = _extract_scored_stat_ids(yahoo_settings)
    source_league_key = _extract_league_key(yahoo_settings)

    scored_canonical_codes: list[str] = []
    for yid in scored_yahoo_ids:
        if yid not in YAHOO_ID_TO_CANONICAL:
            raise UnknownYahooStatError(
                f"Yahoo stat_id {yid} is scored in this league but is not in the "
                f"master stat registry. Add it to master_stat_registry.py."
            )
        code = YAHOO_ID_TO_CANONICAL[yid]
        if code in scored_canonical_codes:
            # Duplicate can occur legitimately (e.g. TB stat_id 4 and 6) — skip.
            continue
        stat = ALL_STATS[code]
        if not stat.scorable:
            raise ContractBuildError(
                f"Yahoo reports stat_id {yid} ({code}) as scored but the master "
                f"registry marks it non-scorable."
            )
        scored_canonical_codes.append(code)

    batting_codes = [
        c for c in scored_canonical_codes if ALL_STATS[c].scoring_role == "batting"
    ]
    pitching_codes = [
        c for c in scored_canonical_codes if ALL_STATS[c].scoring_role == "pitching"
    ]

    # Which display-only stats we ALWAYS emit alongside scored stats.
    display_codes = [
        code for code, stat in DISPLAY_STATS.items()
        if any(scope in ("hitter", "pitcher") for scope in stat.scope)
    ]

    stats_section: dict[str, StatEntry] = {}
    for code in scored_canonical_codes + display_codes:
        stat_def = ALL_STATS[code]
        stats_section[code] = _to_entry(
            stat_def,
            is_scoring_category=(code in scored_canonical_codes),
        )

    # Supporting stats are only emitted if something in stats_section references them
    # via aggregation.  This keeps the contract minimal and honest.
    supporting_section: dict[str, SupportingStatEntry] = {}
    referenced = _collect_referenced_supporting_stats(stats_section)
    for code in sorted(referenced):
        if code not in SUPPORTING_STATS:
            raise ContractBuildError(
                f"Stat '{code}' is referenced by an aggregation but is not in "
                f"SUPPORTING_STATS."
            )
        supporting_section[code] = _to_supporting_entry(SUPPORTING_STATS[code])

    scoring_categories = ScoringCategories(
        batting=batting_codes,
        pitching=pitching_codes,
        total_count=len(scored_canonical_codes),
        win_threshold=_extract_win_threshold(yahoo_settings, len(scored_canonical_codes)),
    )

    weekly_rules = _extract_weekly_rules(yahoo_settings)

    yahoo_id_index = _build_yahoo_id_index(stats_section, supporting_section)

    matchup_display_order = _build_matchup_display_order(
        batting_codes, pitching_codes, display_codes
    )

    provenance = Provenance(
        derived_from="yahoo_league_settings",
        source_league_key=source_league_key,
        generated_at=now,
        generator_version=GENERATOR_VERSION,
    )

    contract = FantasyStatContract(
        **{"$schema": SCHEMA_URL},
        version="2.0.0",
        league_id=league_id,
        format=_extract_format(yahoo_settings),
        season=season,
        stats=stats_section,
        supporting_stats=supporting_section,
        scoring_categories=scoring_categories,
        weekly_rules=weekly_rules,
        yahoo_id_index=yahoo_id_index,
        matchup_display_order=matchup_display_order,
        provenance=provenance,
    )
    return contract


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _extract_scored_stat_ids(settings: dict[str, Any]) -> list[int]:
    """
    Yahoo's response nests stat_categories deeply. Walk it defensively and return
    the ordered list of Yahoo stat IDs that are actually scored this season.

    Expected shape (abbreviated, real Yahoo responses vary):
        settings["stat_categories"]["stats"]["stat"] -> list of dicts each with
            {"stat_id": int, "enabled": "1", "is_only_display_stat": "0" | "1"}
    """
    try:
        stats_list = settings["stat_categories"]["stats"]["stat"]
    except (KeyError, TypeError) as exc:
        raise ContractBuildError(
            "Yahoo settings missing stat_categories.stats.stat"
        ) from exc

    if not isinstance(stats_list, list):
        stats_list = [stats_list]

    scored_ids: list[int] = []
    for entry in stats_list:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("enabled", "0")) != "1":
            continue
        if str(entry.get("is_only_display_stat", "0")) == "1":
            continue
        stat_id_raw = entry.get("stat_id")
        if stat_id_raw is None:
            continue
        try:
            scored_ids.append(int(stat_id_raw))
        except (TypeError, ValueError) as exc:
            raise ContractBuildError(
                f"Non-integer stat_id in Yahoo response: {stat_id_raw!r}"
            ) from exc
    if not scored_ids:
        raise ContractBuildError("Yahoo reported no scored stats for this league")
    return scored_ids


def _extract_league_key(settings: dict[str, Any]) -> str:
    key = settings.get("league_key")
    if not key:
        raise ContractBuildError("Yahoo settings missing league_key")
    return str(key)


def _extract_format(settings: dict[str, Any]) -> str:
    scoring_type = settings.get("scoring_type", "")
    if scoring_type == "head":
        return "h2h_one_win"
    if scoring_type == "headpoint":
        return "h2h_points"
    if scoring_type == "roto":
        return "rotisserie"
    return scoring_type or "unknown"


def _extract_win_threshold(settings: dict[str, Any], total_count: int) -> int:
    # Yahoo does not expose a literal "categories_to_win" — in H2H One Win
    # leagues the convention is majority.  For even counts it's total/2 + 1,
    # for odd counts it's ceil(total/2).
    if total_count % 2 == 0:
        return total_count // 2 + 1
    return (total_count + 1) // 2


def _extract_weekly_rules(settings: dict[str, Any]) -> WeeklyRules:
    # These fields vary by Yahoo API version. Defaults reflect the current league.
    ip_min = float(settings.get("weekly_innings_min", 18.0))
    acquisitions_max = int(settings.get("max_weekly_adds", 8))
    waiver_days = int(settings.get("waiver_time", 1))
    win_threshold = _extract_win_threshold(
        settings, total_count=18  # Will be validated by model_validator.
    )
    return WeeklyRules(
        pitcher_ip_minimum=ip_min,
        pitcher_ip_minimum_outs=int(round(ip_min * 3)),
        acquisitions_max=acquisitions_max,
        waiver_window_days=waiver_days,
        categories_to_win_matchup=win_threshold,
    )


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _to_entry(stat: StatDefinition, *, is_scoring_category: bool) -> StatEntry:
    return StatEntry(
        canonical_code=stat.canonical_code,
        display_label=stat.display_label,
        short_label=stat.short_label,
        scope=list(stat.scope),
        is_scoring_category=is_scoring_category,
        scoring_role=stat.scoring_role,
        aggregation=Aggregation(
            method=stat.aggregation.method,
            numerator_stat=stat.aggregation.numerator_stat,
            denominator_stat=stat.aggregation.denominator_stat,
            numerator_formula=stat.aggregation.numerator_formula,
            multiplier=stat.aggregation.multiplier,
            formula=stat.aggregation.formula,
            components=list(stat.aggregation.components),
            source_stat=stat.aggregation.source_stat,
        ),
        direction=stat.direction,
        data_type=stat.data_type,
        precision=stat.precision,
        valid_range=list(stat.valid_range),
        external_ids=ExternalIds(
            yahoo_stat_id=stat.external_ids.yahoo_stat_id,
            yahoo_stat_id_alt=stat.external_ids.yahoo_stat_id_alt,
            mlb_stats_api=stat.external_ids.mlb_stats_api,
            mlb_stats_api_numerator=stat.external_ids.mlb_stats_api_numerator,
            mlb_stats_api_denominator=stat.external_ids.mlb_stats_api_denominator,
            balldontlie=stat.external_ids.balldontlie,
            pybaseball=stat.external_ids.pybaseball,
        ),
        display_contexts=list(stat.display_contexts),
        notes=stat.notes,
    )


def _to_supporting_entry(stat: StatDefinition) -> SupportingStatEntry:
    return SupportingStatEntry(
        display_label=stat.display_label,
        scope=list(stat.scope),
        data_type=stat.data_type,
        precision=stat.precision,
        external_ids=ExternalIds(
            yahoo_stat_id=stat.external_ids.yahoo_stat_id,
            yahoo_stat_id_alt=stat.external_ids.yahoo_stat_id_alt,
            mlb_stats_api=stat.external_ids.mlb_stats_api,
            mlb_stats_api_numerator=stat.external_ids.mlb_stats_api_numerator,
            mlb_stats_api_denominator=stat.external_ids.mlb_stats_api_denominator,
            balldontlie=stat.external_ids.balldontlie,
            pybaseball=stat.external_ids.pybaseball,
        ),
        notes=stat.notes,
    )


def _collect_referenced_supporting_stats(
    stats_section: dict[str, StatEntry],
) -> set[str]:
    referenced: set[str] = set()
    for entry in stats_section.values():
        agg = entry.aggregation
        for candidate in (agg.numerator_stat, agg.denominator_stat, agg.source_stat):
            if candidate and candidate in SUPPORTING_STATS:
                referenced.add(candidate)
        for comp in agg.components:
            if comp in SUPPORTING_STATS:
                referenced.add(comp)
        # Formula strings may also reference supporting stats; do a light scan.
        for formula in (agg.numerator_formula, agg.formula):
            if not formula:
                continue
            for token in _tokenize_formula(formula):
                if token in SUPPORTING_STATS:
                    referenced.add(token)
    # OBP and SLG each pull in their own dependencies (AB, BB, HBP, etc.).
    frontier = set(referenced)
    while frontier:
        code = frontier.pop()
        stat = SUPPORTING_STATS.get(code)
        if stat is None:
            continue
        agg = stat.aggregation
        for candidate in (agg.numerator_stat, agg.denominator_stat, agg.source_stat):
            if candidate and candidate in SUPPORTING_STATS and candidate not in referenced:
                referenced.add(candidate)
                frontier.add(candidate)
        for comp in agg.components:
            if comp in SUPPORTING_STATS and comp not in referenced:
                referenced.add(comp)
                frontier.add(comp)
    return referenced


def _tokenize_formula(formula: str) -> list[str]:
    # Minimal tokenizer: split on whitespace and arithmetic operators.
    return [t for t in re.split(r"[+\-*/() ]", formula) if t]


def _build_yahoo_id_index(
    stats: dict[str, StatEntry],
    supporting: dict[str, SupportingStatEntry],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for code, entry in {**stats, **supporting}.items():
        primary = entry.external_ids.yahoo_stat_id
        alt = entry.external_ids.yahoo_stat_id_alt
        for yid in (primary, alt):
            if yid is None:
                continue
            key = str(yid)
            if key in index and index[key] != code:
                raise ContractBuildError(
                    f"Yahoo stat_id {key} mapped to both {index[key]} and {code}"
                )
            index[key] = code
    # Deterministic ordering for stable JSON output.
    return dict(sorted(index.items(), key=lambda kv: int(kv[0])))


def _build_matchup_display_order(
    batting: list[str],
    pitching: list[str],
    display: list[str],
) -> list[str]:
    # Batting scored first, then pitching scored, then display-only.
    return batting + pitching + display
