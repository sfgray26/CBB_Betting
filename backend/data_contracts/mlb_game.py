"""
MLBGame — BDL /mlb/v1/games data item contract.

Ground truth: reports/SCHEMA_DISCOVERY.md
Verified against 19-game live sample (2026-04-05).

Status strings observed: STATUS_FINAL, STATUS_IN_PROGRESS, STATUS_SCHEDULED
season_type observed: "regular" — enum guards against future mismatches.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from backend.data_contracts.mlb_team import MLBTeam


class MLBTeamGameData(BaseModel):
    model_config = ConfigDict(strict=True)

    hits: int
    runs: int
    errors: int
    inning_scores: list[int]            # [] pre-game; 9+ items post-game


class MLBScoringPlay(BaseModel):
    model_config = ConfigDict(strict=True)

    play: str
    inning: Literal["top", "bottom"]
    period: str                         # "1st", "2nd", ... — not worth enum
    away_score: int
    home_score: int


class MLBGame(BaseModel):
    model_config = ConfigDict(strict=True)

    id: int
    home_team_name: str
    away_team_name: str
    home_team: MLBTeam
    away_team: MLBTeam
    season: int
    postseason: bool
    season_type: Literal["regular", "postseason", "preseason"]
    date: str                           # ISO 8601 UTC e.g. "2026-04-05T00:10:00.000Z"
    home_team_data: MLBTeamGameData
    away_team_data: MLBTeamGameData
    venue: str
    attendance: Optional[int] = None    # 0 for pre-game; None guard for edge cases
    conference_play: bool
    status: str                         # STATUS_FINAL / STATUS_IN_PROGRESS / STATUS_SCHEDULED
    period: int                         # Current/final inning
    clock: int                          # Always 0 for MLB
    display_clock: str                  # Always "0:00" for MLB
    scoring_summary: list[MLBScoringPlay]   # [] pre-game
