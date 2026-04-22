"""
Canonical registry of baseball stats understood by the platform.

This file is the authoritative source for stat semantics. Yahoo league settings
tell us which of these stats a specific league scores; everything else (aggregation,
direction, scope, external IDs) comes from here.

ADDING A NEW STAT: create a new StatDefinition entry. The contract builder will
fail if Yahoo returns a stat ID not mapped here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


AggregationMethod = Literal[
    "sum",
    "weighted_ratio",
    "composite",
    "composite_display",
    "derived_from_outs",
]

Direction = Literal["higher_is_better", "lower_is_better", "display_only"]
Scope = Literal["hitter", "pitcher"]
DataType = Literal["integer", "decimal", "string"]


@dataclass(frozen=True)
class Aggregation:
    method: AggregationMethod
    numerator_stat: Optional[str] = None
    denominator_stat: Optional[str] = None
    numerator_formula: Optional[str] = None
    multiplier: Optional[float] = None
    formula: Optional[str] = None
    components: tuple[str, ...] = field(default_factory=tuple)
    source_stat: Optional[str] = None


@dataclass(frozen=True)
class ExternalIds:
    yahoo_stat_id: Optional[int] = None
    yahoo_stat_id_alt: Optional[int] = None
    mlb_stats_api: Optional[str] = None
    mlb_stats_api_numerator: Optional[str] = None
    mlb_stats_api_denominator: Optional[str] = None
    balldontlie: Optional[str] = None
    pybaseball: Optional[str] = None


@dataclass(frozen=True)
class StatDefinition:
    canonical_code: str
    display_label: str
    short_label: str
    scope: tuple[Scope, ...]
    scoring_role: Optional[Literal["batting", "pitching"]]
    aggregation: Aggregation
    direction: Direction
    data_type: DataType
    precision: Optional[int]
    valid_range: tuple[Optional[float], Optional[float]]
    external_ids: ExternalIds
    notes: Optional[str] = None
    display_contexts: tuple[str, ...] = field(default_factory=tuple)
    # Whether this stat CAN be a scoring category. Whether it IS one in a specific
    # league is determined by the Yahoo settings response.
    scorable: bool = True


# ---------------------------------------------------------------------------
# Scoring-eligible stats
# ---------------------------------------------------------------------------

BATTING_STATS: dict[str, StatDefinition] = {
    "R": StatDefinition(
        canonical_code="R",
        display_label="Runs",
        short_label="R",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=7,
            mlb_stats_api="runs",
            balldontlie="runs",
            pybaseball="R",
        ),
    ),
    "H": StatDefinition(
        canonical_code="H",
        display_label="Hits",
        short_label="H",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=8,
            mlb_stats_api="hits",
            balldontlie="hits",
            pybaseball="H",
        ),
    ),
    "HR_B": StatDefinition(
        canonical_code="HR_B",
        display_label="Home Runs",
        short_label="HR",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=12,
            mlb_stats_api="homeRuns",
            balldontlie="home_runs",
            pybaseball="HR",
        ),
        notes="Hitter home runs; distinct from HR_P (home runs allowed).",
    ),
    "RBI": StatDefinition(
        canonical_code="RBI",
        display_label="Runs Batted In",
        short_label="RBI",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=13,
            mlb_stats_api="rbi",
            balldontlie="rbi",
            pybaseball="RBI",
        ),
    ),
    "K_B": StatDefinition(
        canonical_code="K_B",
        display_label="Strikeouts (Batter)",
        short_label="K",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=23,
            mlb_stats_api="strikeOuts",
            balldontlie="strikeouts",
            pybaseball="SO",
        ),
        notes="Batter strikeouts. LOWER is better.",
    ),
    "TB": StatDefinition(
        canonical_code="TB",
        display_label="Total Bases",
        short_label="TB",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=21,
            mlb_stats_api="totalBases",
            balldontlie="total_bases",
            pybaseball="TB",
        ),
        notes="Yahoo emits as stat_id 4 or 6 depending on context.",
    ),
    "AVG": StatDefinition(
        canonical_code="AVG",
        display_label="Batting Average",
        short_label="AVG",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(
            method="weighted_ratio",
            numerator_stat="H",
            denominator_stat="AB",
        ),
        direction="higher_is_better",
        data_type="decimal",
        precision=3,
        valid_range=(0.0, 1.0),
        external_ids=ExternalIds(
            yahoo_stat_id=3,
            mlb_stats_api="avg",
            balldontlie="batting_average",
            pybaseball="AVG",
        ),
        notes="Aggregate as sum(H)/sum(AB), NOT average of averages.",
    ),
    "OPS": StatDefinition(
        canonical_code="OPS",
        display_label="On-Base Plus Slugging",
        short_label="OPS",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(
            method="composite",
            formula="OBP + SLG",
            components=("OBP", "SLG"),
        ),
        direction="higher_is_better",
        data_type="decimal",
        precision=3,
        valid_range=(0.0, 5.0),
        external_ids=ExternalIds(
            yahoo_stat_id=55,
            mlb_stats_api="ops",
            balldontlie="ops",
            pybaseball="OPS",
        ),
        notes="Aggregate components first, then sum.",
    ),
    "NSB": StatDefinition(
        canonical_code="NSB",
        display_label="Net Stolen Bases",
        short_label="NSB",
        scope=("hitter",),
        scoring_role="batting",
        aggregation=Aggregation(
            method="composite",
            formula="SB - CS",
            components=("SB", "CS"),
        ),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(None, None),
        external_ids=ExternalIds(
            yahoo_stat_id=62,
            mlb_stats_api_numerator="stolenBases",
            mlb_stats_api_denominator="caughtStealing",
        ),
        notes="Can be negative.",
    ),
}


PITCHING_STATS: dict[str, StatDefinition] = {
    "W": StatDefinition(
        canonical_code="W",
        display_label="Wins",
        short_label="W",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=28,
            mlb_stats_api="wins",
            pybaseball="W",
        ),
    ),
    "L": StatDefinition(
        canonical_code="L",
        display_label="Losses",
        short_label="L",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=29,
            mlb_stats_api="losses",
            pybaseball="L",
        ),
    ),
    "HR_P": StatDefinition(
        canonical_code="HR_P",
        display_label="Home Runs Allowed",
        short_label="HRA",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=38,
            mlb_stats_api="homeRunsAllowed",
            pybaseball="HR",
        ),
        notes="Home runs allowed by pitcher. Distinct from HR_B.",
    ),
    "K_P": StatDefinition(
        canonical_code="K_P",
        display_label="Strikeouts (Pitcher)",
        short_label="K",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=42,
            mlb_stats_api="strikeOuts",
            pybaseball="SO",
        ),
        notes="Pitcher strikeouts. Distinct from K_B.",
    ),
    "ERA": StatDefinition(
        canonical_code="ERA",
        display_label="Earned Run Average",
        short_label="ERA",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(
            method="weighted_ratio",
            numerator_stat="ER",
            denominator_stat="IP_OUTS",
            multiplier=27.0,
        ),
        direction="lower_is_better",
        data_type="decimal",
        precision=2,
        valid_range=(0.0, 99.99),
        external_ids=ExternalIds(
            yahoo_stat_id=26,
            mlb_stats_api="era",
            pybaseball="ERA",
        ),
        notes="Aggregate as (sum(ER)*27)/sum(outs). Using outs avoids 0.1/0.2 fractional trap.",
    ),
    "WHIP": StatDefinition(
        canonical_code="WHIP",
        display_label="Walks + Hits per IP",
        short_label="WHIP",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(
            method="weighted_ratio",
            numerator_formula="BB_P + H_P",
            denominator_stat="IP_OUTS",
            multiplier=3.0,
        ),
        direction="lower_is_better",
        data_type="decimal",
        precision=3,
        valid_range=(0.0, 99.999),
        external_ids=ExternalIds(
            yahoo_stat_id=27,
            mlb_stats_api="whip",
            pybaseball="WHIP",
        ),
    ),
    "K_9": StatDefinition(
        canonical_code="K_9",
        display_label="Strikeouts per Nine",
        short_label="K/9",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(
            method="weighted_ratio",
            numerator_stat="K_P",
            denominator_stat="IP_OUTS",
            multiplier=27.0,
        ),
        direction="higher_is_better",
        data_type="decimal",
        precision=2,
        valid_range=(0.0, 30.0),
        external_ids=ExternalIds(
            yahoo_stat_id=57,
            mlb_stats_api="strikeoutsPer9Inn",
            pybaseball="K/9",
        ),
    ),
    "QS": StatDefinition(
        canonical_code="QS",
        display_label="Quality Starts",
        short_label="QS",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=85,
            mlb_stats_api="qualityStarts",
            pybaseball="QS",
        ),
        notes="A start of 6+ IP with 3 or fewer ER.",
    ),
    "NSV": StatDefinition(
        canonical_code="NSV",
        display_label="Net Saves",
        short_label="NSV",
        scope=("pitcher",),
        scoring_role="pitching",
        aggregation=Aggregation(
            method="composite",
            formula="SV - BS",
            components=("SV", "BS"),
        ),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(None, None),
        external_ids=ExternalIds(
            yahoo_stat_id=83,
            mlb_stats_api_numerator="saves",
            mlb_stats_api_denominator="blownSaves",
        ),
        notes="Can be negative.",
    ),
}


# ---------------------------------------------------------------------------
# Display-only stats (may appear in matchup views but never as scoring categories)
# ---------------------------------------------------------------------------

DISPLAY_STATS: dict[str, StatDefinition] = {
    "IP": StatDefinition(
        canonical_code="IP",
        display_label="Innings Pitched",
        short_label="IP",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(
            method="derived_from_outs",
            source_stat="IP_OUTS",
            formula="outs / 3",
        ),
        direction="higher_is_better",
        data_type="decimal",
        precision=1,
        valid_range=(0.0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=50,
            mlb_stats_api="inningsPitched",
            pybaseball="IP",
        ),
        notes="Not a scoring category but gates weekly eligibility (18 IP min).",
        display_contexts=("matchup", "player_row", "streaming"),
        scorable=False,
    ),
    "GS": StatDefinition(
        canonical_code="GS",
        display_label="Games Started",
        short_label="GS",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(
            yahoo_stat_id=None,
            mlb_stats_api="gamesStarted",
            pybaseball="GS",
        ),
        display_contexts=("matchup", "player_row"),
        scorable=False,
    ),
    "H_AB": StatDefinition(
        canonical_code="H_AB",
        display_label="Hits / At Bats",
        short_label="H/AB",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(
            method="composite_display",
            formula="sum(H) / sum(AB)",
            components=("H", "AB"),
        ),
        direction="display_only",
        data_type="string",
        precision=None,
        valid_range=(None, None),
        external_ids=ExternalIds(
            yahoo_stat_id=60
        ),
        display_contexts=("matchup",),
        scorable=False,
    ),
}


# ---------------------------------------------------------------------------
# Supporting stats — atomic inputs needed by aggregation/derivation
# ---------------------------------------------------------------------------

SUPPORTING_STATS: dict[str, StatDefinition] = {
    "AB": StatDefinition(
        canonical_code="AB",
        display_label="At Bats",
        short_label="AB",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="atBats"),
        scorable=False,
    ),
    "BB": StatDefinition(
        canonical_code="BB",
        display_label="Walks (Batter)",
        short_label="BB",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="baseOnBalls"),
        scorable=False,
    ),
    "HBP": StatDefinition(
        canonical_code="HBP",
        display_label="Hit By Pitch",
        short_label="HBP",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="hitByPitch"),
        scorable=False,
    ),
    "SF": StatDefinition(
        canonical_code="SF",
        display_label="Sacrifice Flies",
        short_label="SF",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="sacFlies"),
        scorable=False,
    ),
    "SB": StatDefinition(
        canonical_code="SB",
        display_label="Stolen Bases",
        short_label="SB",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(yahoo_stat_id=16, mlb_stats_api="stolenBases"),
        scorable=False,
    ),
    "CS": StatDefinition(
        canonical_code="CS",
        display_label="Caught Stealing",
        short_label="CS",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="caughtStealing"),
        scorable=False,
    ),
    "OBP": StatDefinition(
        canonical_code="OBP",
        display_label="On-Base %",
        short_label="OBP",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(
            method="weighted_ratio",
            numerator_formula="H + BB + HBP",
            denominator_stat="OBP_DENOM",
        ),
        direction="higher_is_better",
        data_type="decimal",
        precision=3,
        valid_range=(0.0, 1.0),
        external_ids=ExternalIds(yahoo_stat_id=85, mlb_stats_api="obp"),
        scorable=False,
    ),
    "SLG": StatDefinition(
        canonical_code="SLG",
        display_label="Slugging %",
        short_label="SLG",
        scope=("hitter",),
        scoring_role=None,
        aggregation=Aggregation(
            method="weighted_ratio",
            numerator_stat="TB",
            denominator_stat="AB",
        ),
        direction="higher_is_better",
        data_type="decimal",
        precision=3,
        valid_range=(0.0, 4.0),
        external_ids=ExternalIds(mlb_stats_api="slg"),
        scorable=False,
    ),
    "ER": StatDefinition(
        canonical_code="ER",
        display_label="Earned Runs",
        short_label="ER",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="earnedRuns"),
        scorable=False,
    ),
    "IP_OUTS": StatDefinition(
        canonical_code="IP_OUTS",
        display_label="Outs Recorded",
        short_label="Outs",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="outs"),
        notes="Canonical internal representation of IP. 1 IP = 3 outs.",
        scorable=False,
    ),
    "BB_P": StatDefinition(
        canonical_code="BB_P",
        display_label="Walks Allowed",
        short_label="BB",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="baseOnBalls"),
        scorable=False,
    ),
    "H_P": StatDefinition(
        canonical_code="H_P",
        display_label="Hits Allowed",
        short_label="H",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="hits"),
        scorable=False,
    ),
    "SV": StatDefinition(
        canonical_code="SV",
        display_label="Saves",
        short_label="SV",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(yahoo_stat_id=32, mlb_stats_api="saves"),
        scorable=False,
    ),
    "BS": StatDefinition(
        canonical_code="BS",
        display_label="Blown Saves",
        short_label="BS",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="lower_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(mlb_stats_api="blownSaves"),
        scorable=False,
    ),
    "HLD": StatDefinition(
        canonical_code="HLD",
        display_label="Holds",
        short_label="HLD",
        scope=("pitcher",),
        scoring_role=None,
        aggregation=Aggregation(method="sum"),
        direction="higher_is_better",
        data_type="integer",
        precision=0,
        valid_range=(0, None),
        external_ids=ExternalIds(yahoo_stat_id=48, mlb_stats_api="holds"),
        scorable=False,
    ),
}


ALL_STATS: dict[str, StatDefinition] = {
    **BATTING_STATS,
    **PITCHING_STATS,
    **DISPLAY_STATS,
    **SUPPORTING_STATS,
}


def build_yahoo_id_to_canonical() -> dict[int, str]:
    """Return a Yahoo stat_id -> canonical_code map covering primary and alt IDs."""
    mapping: dict[int, str] = {}
    for code, stat in ALL_STATS.items():
        primary = stat.external_ids.yahoo_stat_id
        alt = stat.external_ids.yahoo_stat_id_alt
        if primary is not None:
            if primary in mapping and mapping[primary] != code:
                raise ValueError(
                    f"Yahoo stat_id {primary} mapped to both "
                    f"{mapping[primary]} and {code}"
                )
            mapping[primary] = code
        if alt is not None:
            if alt in mapping and mapping[alt] != code:
                raise ValueError(
                    f"Yahoo stat_id {alt} mapped to both "
                    f"{mapping[alt]} and {code}"
                )
            mapping[alt] = code
    return mapping


YAHOO_ID_TO_CANONICAL: dict[int, str] = build_yahoo_id_to_canonical()
