"""
MLBTeam — nested team object in games, injuries, and player endpoints.

Ground truth: reports/SCHEMA_DISCOVERY.md
All fields observed non-null across 19 game sample.
NOTE: league and division can be empty for free agents/retired players.
"""

from typing import Optional, Literal, Any

from pydantic import BaseModel, ConfigDict, field_validator


class MLBTeam(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    slug: str
    abbreviation: str
    display_name: str
    short_display_name: str
    name: str
    location: str
    league: Optional[Literal["National", "American"]] = None
    division: Optional[Literal["East", "Central", "West"]] = None

    @field_validator("league", "division", mode="before")
    @classmethod
    def empty_string_to_none(cls, v: Any) -> Optional[str]:
        """Convert empty strings to None for league and division fields."""
        if v == "" or v is None:
            return None
        return v


# Rebuild model to handle forward references correctly
MLBTeam.model_rebuild()
