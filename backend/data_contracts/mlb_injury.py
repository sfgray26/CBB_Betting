"""
MLBInjury — BDL /mlb/v1/player_injuries data item contract.

Ground truth: reports/SCHEMA_DISCOVERY.md
Nullable fields verified from 25-item live sample:
    detail   null in 4/25
    side     null in 2/25

Pagination: uses next_cursor — must page to get full IL list.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

from backend.data_contracts.mlb_player import MLBPlayer


class MLBInjury(BaseModel):
    model_config = ConfigDict(strict=True)

    player: MLBPlayer
    date: str                           # ISO 8601 UTC — injury report date
    return_date: Optional[str] = None   # ISO 8601 UTC — estimated return; None if unknown
    type: str                           # Body part e.g. "Triceps", "Hamstring"
    detail: Optional[str] = None        # Sub-type e.g. "Strain" — null in 4/25
    side: Optional[str] = None          # "Right", "Left" — null in 2/25 (bilateral/unknown)
    status: str                         # "15-Day-IL", "60-Day-IL", "10-Day-IL", "DTD"
    long_comment: str                   # Paragraph-length note
    short_comment: str                  # One-sentence note
