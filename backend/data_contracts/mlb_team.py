"""
MLBTeam — nested team object in games, injuries, and player endpoints.

Ground truth: reports/SCHEMA_DISCOVERY.md
All fields observed non-null across 19 game sample.
NOTE: league and division can be empty for free agents/retired players.
"""

from typing import Optional, Literal, Union, Any

from pydantic import BaseModel, ConfigDict, field_validator
from typing_extensions import Annotated
from pydantic.functional_validators import BeforeValidator


def _empty_to_none(value: Any) -> Any:
    """Convert empty strings to None before validation."""
    if value == "" or value is None:
        return None
    return value


class MLBTeam(BaseModel):
    model_config = ConfigDict(strict=False)

    id: int
    slug: str
    abbreviation: str
    display_name: str
    short_display_name: str
    name: str
    location: str
    league: Annotated[Optional[Literal["National", "American"]], BeforeValidator(_empty_to_none)] = None
    division: Annotated[Optional[Literal["East", "Central", "West"]], BeforeValidator(_empty_to_none)] = None


# Rebuild model to handle forward references correctly
MLBTeam.model_rebuild()
