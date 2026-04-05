"""
MLBTeam — nested team object in games, injuries, and player endpoints.

Ground truth: reports/SCHEMA_DISCOVERY.md
All fields observed non-null across 19 game sample.
"""

from __future__ import annotations

from typing import Literal

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
    league: Literal["National", "American"]
    division: Literal["East", "Central", "West"]
