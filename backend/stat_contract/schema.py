"""Pydantic models for fantasy_stat_contract.json."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Aggregation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal[
        "sum",
        "weighted_ratio",
        "composite",
        "composite_display",
        "derived_from_outs",
    ]
    numerator_stat: Optional[str] = None
    denominator_stat: Optional[str] = None
    numerator_formula: Optional[str] = None
    multiplier: Optional[float] = None
    formula: Optional[str] = None
    components: list[str] = Field(default_factory=list)
    source_stat: Optional[str] = None


class ExternalIds(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yahoo_stat_id: Optional[int] = None
    yahoo_stat_id_alt: Optional[int] = None
    mlb_stats_api: Optional[str] = None
    mlb_stats_api_numerator: Optional[str] = None
    mlb_stats_api_denominator: Optional[str] = None
    balldontlie: Optional[str] = None
    pybaseball: Optional[str] = None


class StatEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_code: str
    display_label: str
    short_label: str
    scope: list[Literal["hitter", "pitcher"]]
    is_scoring_category: bool
    scoring_role: Optional[Literal["batting", "pitching"]] = None
    aggregation: Aggregation
    direction: Literal["higher_is_better", "lower_is_better", "display_only"]
    data_type: Literal["integer", "decimal", "string"]
    precision: Optional[int]
    valid_range: Optional[list[Optional[float]]] = None
    external_ids: ExternalIds
    display_contexts: list[str] = Field(default_factory=list)
    notes: Optional[str] = None

    @field_validator("valid_range")
    @classmethod
    def _validate_valid_range(cls, v: Optional[list[Optional[float]]]) -> Optional[list[Optional[float]]]:
        if v is not None and len(v) != 2:
            raise ValueError("valid_range must be length 2")
        return v


class SupportingStatEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    display_label: str
    scope: list[Literal["hitter", "pitcher"]]
    data_type: Literal["integer", "decimal", "string"]
    precision: Optional[int]
    external_ids: ExternalIds
    notes: Optional[str] = None


class ScoringCategories(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batting: list[str]
    pitching: list[str]
    total_count: int
    win_threshold: int

    @field_validator("win_threshold")
    @classmethod
    def _validate_win_threshold(cls, v: int, info) -> int:
        if "total_count" in info.data and v > info.data["total_count"]:
            raise ValueError("win_threshold exceeds total_count")
        return v

    @field_validator("total_count")
    @classmethod
    def _validate_counts(cls, v: int, info) -> int:
        batting = info.data.get("batting", [])
        pitching = info.data.get("pitching", [])
        if v != len(batting) + len(pitching):
            raise ValueError("total_count does not match batting+pitching length")
        return v


class WeeklyRules(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pitcher_ip_minimum: float
    pitcher_ip_minimum_outs: int
    acquisitions_max: int
    waiver_window_days: int
    categories_to_win_matchup: int

    @field_validator("pitcher_ip_minimum_outs")
    @classmethod
    def _validate_ip_outs(cls, v: int, info) -> int:
        if "pitcher_ip_minimum" in info.data:
            expected = int(round(info.data["pitcher_ip_minimum"] * 3))
            if v != expected:
                raise ValueError(
                    f"pitcher_ip_minimum_outs ({v}) "
                    f"does not equal pitcher_ip_minimum * 3 ({expected})"
                )
        return v


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    derived_from: str
    source_league_key: str
    generated_at: datetime
    generator_version: str


class FantasyStatContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_: str = Field(alias="$schema")
    version: str
    league_id: str
    format: str
    season: int
    stats: dict[str, StatEntry]
    supporting_stats: dict[str, SupportingStatEntry]
    scoring_categories: ScoringCategories
    weekly_rules: WeeklyRules
    yahoo_id_index: dict[str, str]
    matchup_display_order: list[str]
    provenance: Provenance

    @model_validator(mode="after")
    def _validate_scoring_category_consistency(self) -> "FantasyStatContract":
        declared = set(self.scoring_categories.batting) | set(self.scoring_categories.pitching)
        marked = {
            code for code, entry in self.stats.items() if entry.is_scoring_category
        }
        if declared != marked:
            raise ValueError(
                f"scoring_categories section disagrees with is_scoring_category flags.\n"
                f"Declared in scoring_categories: {sorted(declared)}\n"
                f"Marked as scoring in stats:     {sorted(marked)}"
            )
        return self

    @model_validator(mode="after")
    def _validate_yahoo_id_index_matches_stats(self) -> "FantasyStatContract":
        for yahoo_id_str, canonical in self.yahoo_id_index.items():
            if canonical not in self.stats and canonical not in self.supporting_stats:
                raise ValueError(
                    f"yahoo_id_index references unknown canonical code '{canonical}' "
                    f"for Yahoo stat_id {yahoo_id_str}"
                )
        return self
