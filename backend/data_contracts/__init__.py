"""
Pydantic V2 data contracts for all external API sources.

Layer 0 — Decision Contracts.

These models are the IMMUTABLE TRUTH of what valid data looks like.
Every downstream consumer (analysis, ingestion, optimization) imports
from here. No raw dicts pass the contract boundary.

Models:
    mlb_game    -- BDL /mlb/v1/games
    mlb_odds    -- BDL /mlb/v1/odds
    mlb_injury  -- BDL /mlb/v1/player_injuries
    mlb_player  -- BDL /mlb/v1/players (shared MLBPlayer sub-model)
    pagination  -- BDLMeta / BDLResponse generic wrapper
"""

from backend.data_contracts.mlb_team import MLBTeam
from backend.data_contracts.mlb_player import MLBPlayer
from backend.data_contracts.mlb_game import (
    MLBTeamGameData,
    MLBScoringPlay,
    MLBGame,
)
from backend.data_contracts.mlb_odds import MLBBettingOdd
from backend.data_contracts.mlb_injury import MLBInjury
from backend.data_contracts.pagination import BDLMeta, BDLResponse

__all__ = [
    "MLBTeam",
    "MLBTeamGameData",
    "MLBScoringPlay",
    "MLBGame",
    "MLBBettingOdd",
    "MLBInjury",
    "MLBPlayer",
    "BDLMeta",
    "BDLResponse",
]
