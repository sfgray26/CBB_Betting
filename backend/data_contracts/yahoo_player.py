"""
YahooPlayer -- shared base model for all Yahoo Fantasy API player responses.

Ground truth: reports/SCHEMA_DISCOVERY.md Yahoo section
Verified from live capture 2026-04-05:
    status is Optional[bool] (True=injured, None=healthy) -- NOT a string
    injury_note independent of status (can have note with status=None)
    percent_owned always float, never null
    positions may include "IL" or "NA" as Yahoo status tokens alongside eligible positions
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class YahooPlayer(BaseModel):
    model_config = ConfigDict(strict=True)

    player_key: str
    player_id: str
    name: str
    team: str
    positions: list[str]
    status: Optional[bool] = None       # True=injured, None=healthy. NOT a string.
    injury_note: Optional[str] = None   # Body part only e.g. "Hip". Independent of status.
    is_undroppable: bool
    percent_owned: float                # Always present, 0.0-100.0. Never null.

    @property
    def is_injured(self) -> bool:
        """True if any injury signal is present.

        Three independent signals from live capture:
          1. status=True (Yahoo injury flag)
          2. injury_note is not None (body part string present)
          3. "IL" in positions (IL roster slot marker)

        All three must be checked -- none is a superset of the others.
        Example: Verlander has status=None, injury_note="Hip", positions includes "IL".
        Example: Crochet has status=True, injury_note=None, no "IL" in positions.
        """
        return bool(self.status) or self.injury_note is not None or "IL" in self.positions
