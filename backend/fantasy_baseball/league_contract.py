from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


LEAGUE_ID = "72586"

ACTIVE_ROSTER_SLOTS: tuple[str, ...] = (
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "LF",
    "CF",
    "RF",
    "Util",
    "SP",
    "SP",
    "RP",
    "RP",
    "P",
    "P",
    "P",
)

VALID_ROSTER_SLOT_SET = frozenset(ACTIVE_ROSTER_SLOTS)

BATTING_SCORING_CATEGORIES: tuple[str, ...] = (
    "R",
    "H",
    "HR",
    "RBI",
    "K(B)",
    "TB",
    "AVG",
    "OPS",
    "NSB",
)

PITCHING_SCORING_CATEGORIES: tuple[str, ...] = (
    "W",
    "L",
    "HRA",
    "K",
    "ERA",
    "WHIP",
    "K/9",
    "QS",
    "NSV",
)

ACTIVE_SCORING_CATEGORIES: tuple[str, ...] = (
    *BATTING_SCORING_CATEGORIES,
    *PITCHING_SCORING_CATEGORIES,
)

ACTIVE_SCORING_CATEGORY_SET = frozenset(ACTIVE_SCORING_CATEGORIES)
RATIO_SCORING_CATEGORIES = frozenset({"AVG", "OPS", "ERA", "WHIP", "K/9"})

CANONICAL_SCORING_STAT_ID_MAP: dict[str, str] = {
    "7": "R",
    "8": "H",
    "12": "HR",
    "13": "RBI",
    "42": "K(B)",
    "6": "TB",
    "3": "AVG",
    "55": "OPS",
    "60": "NSB",
    "23": "W",
    "24": "L",
    "35": "HRA",
    "28": "K",
    "26": "ERA",
    "27": "WHIP",
    "57": "K/9",
    "29": "QS",
    "83": "NSV",
}


@dataclass(frozen=True)
class ExpectedScoringStat:
    abbreviation: str
    position_type: str
    preferred_id: str
    accepted_ids: tuple[str, ...]
    aliases: tuple[str, ...]


EXPECTED_SCORING_STATS: tuple[ExpectedScoringStat, ...] = (
    ExpectedScoringStat("R", "B", "7", ("7",), ("R", "RUNS")),
    ExpectedScoringStat("H", "B", "8", ("8",), ("H", "HITS")),
    ExpectedScoringStat("HR", "B", "12", ("12",), ("HR", "HOME RUNS", "HOMERUNS")),
    ExpectedScoringStat("RBI", "B", "13", ("13",), ("RBI", "RUNS BATTED IN")),
    ExpectedScoringStat("K(B)", "B", "42", ("42",), ("K", "SO", "STRIKEOUTS")),
    ExpectedScoringStat("TB", "B", "6", ("6", "4"), ("TB", "TOTAL BASES", "TOTALBASES")),
    ExpectedScoringStat("AVG", "B", "3", ("3",), ("AVG", "AVERAGE", "BATTING AVERAGE", "BATTINGAVG")),
    ExpectedScoringStat("OPS", "B", "55", ("55",), ("OPS",)),
    ExpectedScoringStat("NSB", "B", "60", ("60",), ("NSB", "NET STOLEN BASES", "NETSB")),
    ExpectedScoringStat("W", "P", "23", ("23",), ("W", "WINS")),
    ExpectedScoringStat("L", "P", "24", ("24",), ("L", "LOSSES")),
    ExpectedScoringStat("HRA", "P", "35", ("35",), ("HR", "HRA", "HOME RUNS ALLOWED", "HRALLOWED")),
    ExpectedScoringStat("K", "P", "28", ("28",), ("K", "SO", "STRIKEOUTS")),
    ExpectedScoringStat("ERA", "P", "26", ("26",), ("ERA",)),
    ExpectedScoringStat("WHIP", "P", "27", ("27",), ("WHIP",)),
    ExpectedScoringStat("K/9", "P", "57", ("57",), ("K/9", "K9")),
    ExpectedScoringStat("QS", "P", "29", ("29",), ("QS", "QUALITY STARTS", "QUALITYSTARTS")),
    ExpectedScoringStat("NSV", "P", "83", ("83",), ("NSV", "NET SAVES", "NETSAVES")),
)


def _normalize_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().upper()
    return "".join(ch for ch in text if ch.isalnum() or ch in {"/", "(" , ")"})


def _normalize_position_type(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"B", "BATTER", "HITTER"}:
        return "B"
    if text in {"P", "PITCHER"}:
        return "P"
    return ""


def iter_stat_nodes(payload: Any):
    seen: set[tuple[str, str, str]] = set()

    def _walk(node: Any):
        if isinstance(node, dict):
            stat = node.get("stat")
            if isinstance(stat, dict) and stat.get("stat_id") is not None:
                sid = str(stat.get("stat_id"))
                key = (sid, _normalize_position_type(stat.get("position_type")), _normalize_text(stat.get("display_name") or stat.get("abbreviation") or stat.get("name")))
                if key not in seen:
                    seen.add(key)
                    yield stat
            elif node.get("stat_id") is not None:
                sid = str(node.get("stat_id"))
                key = (sid, _normalize_position_type(node.get("position_type")), _normalize_text(node.get("display_name") or node.get("abbreviation") or node.get("name")))
                if key not in seen:
                    seen.add(key)
                    yield node

            for value in node.values():
                yield from _walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from _walk(item)

    yield from _walk(payload)


def _is_scoring_stat_node(stat: Mapping[str, Any]) -> bool:
    for key in ("is_scoring_category", "is_scoring_stat", "is_scoring"):
        if key in stat:
            return _normalize_flag(stat.get(key))
    if "is_only_display_stat" in stat:
        return not _normalize_flag(stat.get("is_only_display_stat"))
    return False


def _match_expected_scoring_stat(stat: Mapping[str, Any]) -> ExpectedScoringStat | None:
    stat_id = str(stat.get("stat_id") or "")
    position_type = _normalize_position_type(stat.get("position_type"))
    display_name = _normalize_text(stat.get("display_name") or stat.get("abbreviation") or stat.get("name"))

    for expected in EXPECTED_SCORING_STATS:
        if position_type and position_type != expected.position_type:
            continue
        if stat_id in expected.accepted_ids:
            return expected
        if display_name and display_name in {_normalize_text(alias) for alias in expected.aliases}:
            return expected
    return None


def build_scoring_stat_map_from_settings(settings_payload: Mapping[str, Any]) -> dict[str, str]:
    stat_nodes = list(iter_stat_nodes(settings_payload))
    if not stat_nodes:
        raise ValueError("League settings payload did not contain any stat nodes")

    resolved_by_abbreviation: dict[str, tuple[str, ExpectedScoringStat]] = {}

    def _resolve(scoring_only: bool) -> None:
        for stat in stat_nodes:
            expected = _match_expected_scoring_stat(stat)
            if expected is None:
                continue
            if scoring_only and not _is_scoring_stat_node(stat):
                continue

            stat_id = str(stat.get("stat_id"))
            existing = resolved_by_abbreviation.get(expected.abbreviation)
            if existing is None or stat_id == expected.preferred_id:
                resolved_by_abbreviation[expected.abbreviation] = (stat_id, expected)

    _resolve(scoring_only=True)
    if len(resolved_by_abbreviation) != len(ACTIVE_SCORING_CATEGORIES):
        _resolve(scoring_only=False)

    if len(resolved_by_abbreviation) != len(ACTIVE_SCORING_CATEGORIES):
        raise ValueError(
            f"League settings resolved {len(resolved_by_abbreviation)} active stats, expected {len(ACTIVE_SCORING_CATEGORIES)}"
        )

    return {
        stat_id: abbreviation
        for abbreviation in ACTIVE_SCORING_CATEGORIES
        for stat_id, expected in [resolved_by_abbreviation[abbreviation]]
    }


def ordered_scoring_stats(stats: Mapping[str, Any], fill_missing: bool = False) -> dict[str, Any]:
    ordered: dict[str, Any] = {}
    for category in ACTIVE_SCORING_CATEGORIES:
        if category in stats:
            ordered[category] = stats[category]
        elif fill_missing:
            ordered[category] = "0.000" if category in RATIO_SCORING_CATEGORIES else "0"
    return ordered
