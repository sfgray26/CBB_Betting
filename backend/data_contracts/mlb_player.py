"""
MLBPlayer — shared sub-model used by games, injuries, and players endpoints.

Ground truth: reports/SCHEMA_DISCOVERY.md
Verified nullable fields from 25-item live sample:
    debut_year  null in 3/25
    college     null in 18/25
    dob         null in 1/25
    age         null in 1/25
    draft       null in 16/25

CRITICAL: dob uses "D/M/YYYY" format, not ISO 8601. Custom validator required.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from backend.data_contracts.mlb_team import MLBTeam


class MLBPlayer(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    first_name: str
    last_name: str
    full_name: str
    debut_year: Optional[int] = None
    jersey: Optional[str] = None
    college: Optional[str] = None
    position: str
    active: bool
    birth_place: Optional[str] = None
    dob: Optional[str] = None           # "D/M/YYYY" format — NOT ISO 8601
    age: Optional[int] = None
    height: Optional[str] = None
    weight: Optional[str] = None
    draft: Optional[str] = None         # null for international/undrafted
    bats_throws: Optional[str] = None
    team: Optional[MLBTeam] = None

    @field_validator("dob", mode="before")
    @classmethod
    def validate_dob_format(cls, v: object) -> Optional[str]:
        """
        dob comes as "D/M/YYYY" (e.g. "5/7/1994" = July 5, not May 7).
        We store as-is (str) — callers must use parse_dob() to get a date.
        This validator just rejects obviously wrong types.
        """
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError(f"dob must be str or None, got {type(v).__name__}: {v!r}")
        return v
