"""
MLBPlayerStats -- BDL /mlb/v1/stats per-game box stat row.

Ground truth: reports/K_A_BDL_STATS_SPEC.md (Kimi spec -- treat as best-guess)
Probe findings (S19, confirmed live):
  - Natural key: (player.id, game_id)
  - Rate stats (avg, obp, slg, era, whip) are floats, NOT strings
  - Pitching fields are null for hitters; batting fields are null for pitchers
  - No mlbam_id in stat response -- BDL player.id only
  - ip is a string like "6.2" (not a float)
  - game_id may be a flat field or extracted from a nested game object

All stat fields are Optional -- never assume the API returns complete rows.
ConfigDict(populate_by_name=True) -- NOT strict (nullable fields, graceful handling).
"""

from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict

from backend.data_contracts.mlb_player import MLBPlayer
from backend.data_contracts.mlb_team import MLBTeam


class MLBPlayerStats(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    # Top-level identifiers
    id: Optional[int] = None          # BDL stats record id (may or may not be present)
    player: Optional[MLBPlayer] = None # Nested player object
    team: Optional[MLBTeam] = None     # Nested team object

    # game_id: flat field OR extracted from nested game object
    # The BDL /mlb/v1/stats endpoint may return either {"game_id": 123} or
    # {"game": {"id": 123, ...}}. Accept both. Callers should prefer bdl_player_id
    # + game_id as the natural key.
    game_id: Optional[int] = None

    season: Optional[int] = None

    # ------------------------------------------------------------------
    # Batting stats (null for pure pitchers)
    # ------------------------------------------------------------------
    ab: Optional[int] = None
    r: Optional[int] = None           # runs
    h: Optional[int] = None           # hits
    double: Optional[int] = None      # doubles (API field name is "double")
    triple: Optional[int] = None
    hr: Optional[int] = None
    rbi: Optional[int] = None
    bb: Optional[int] = None          # walks
    so: Optional[int] = None          # strikeouts (batting)
    sb: Optional[int] = None
    cs: Optional[int] = None          # caught stealing
    avg: Optional[float] = None       # probe: floats, NOT strings
    obp: Optional[float] = None
    slg: Optional[float] = None
    ops: Optional[float] = None

    # ------------------------------------------------------------------
    # Pitching stats (null for pure hitters)
    # ------------------------------------------------------------------
    # ip: observed as "6.2" (str) or 7 (int) or 0.2 (float) in live responses.
    # Accept Union[str, float] to prevent validation crashes.
    ip: Optional[Union[str, float]] = None 
    h_allowed: Optional[int] = None
    r_allowed: Optional[int] = None
    er: Optional[int] = None
    bb_allowed: Optional[int] = None
    k: Optional[int] = None           # strikeouts (pitching)
    whip: Optional[float] = None
    era: Optional[float] = None

    # ------------------------------------------------------------------
    # Convenience property
    # ------------------------------------------------------------------

    @property
    def bdl_player_id(self) -> int:
        """BDL player.id for use as the ingestion natural key."""
        return self.player.id
