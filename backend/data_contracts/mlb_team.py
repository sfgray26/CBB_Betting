"""
MLBTeam — nested team object in games, injuries, and player endpoints.

Ground truth: reports/SCHEMA_DISCOVERY.md
All fields observed non-null across 19 game sample.
NOTE: league and division can be empty for free agents/retired players.
"""

from typing import Optional, Literal

from pydantic import BaseModel, ConfigDict


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


# Rebuild model to handle forward references correctly
MLBTeam.model_rebuild()
