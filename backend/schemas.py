"""
Pydantic request/response schemas for the CBB Edge API.

Using explicit schemas instead of raw dicts prevents mass-assignment
vulnerabilities on ORM models and generates accurate OpenAPI docs.
"""

from __future__ import annotations

from typing import Literal, Optional
from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Bet logging
# ---------------------------------------------------------------------------

class BetLogCreate(BaseModel):
    """
    Payload for POST /api/bets/log.

    Only fields that a user should be able to set when manually logging
    a paper trade or real bet are exposed here. Outcome, P&L, and CLV
    are written separately via PUT /api/bets/{bet_id}/outcome.
    """

    game_id: int = Field(..., description="FK to games.id")
    prediction_id: Optional[int] = Field(
        None, description="FK to predictions.id (links this bet to a model prediction)"
    )

    # What we bet
    pick: str = Field(..., min_length=2, max_length=120, description='e.g. "Duke -4.5"')
    bet_type: Literal["spread", "total", "moneyline"] = Field(..., description="Market type")
    odds_taken: float = Field(..., description="American odds at bet placement")

    # Sizing
    bet_size_units: float = Field(..., gt=0, le=10, description="Units risked (max 10)")
    bet_size_dollars: float = Field(..., gt=0, description="Dollar amount risked")
    bankroll_at_bet: Optional[float] = Field(None, gt=0)
    bet_size_pct: Optional[float] = Field(None, ge=0, le=100)

    # Model context at time of bet (optional — populated automatically for paper trades)
    model_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    lower_ci_prob: Optional[float] = Field(None, ge=0.0, le=1.0)
    kelly_full: Optional[float] = Field(None, ge=0.0, le=1.0)
    kelly_fractional: Optional[float] = Field(None, ge=0.0, le=1.0)
    point_edge: Optional[float] = Field(None)
    conservative_edge: Optional[float] = Field(None)

    # Flags
    is_paper_trade: bool = Field(False, description="True = simulated, no real money")
    is_backfill: bool = Field(False, description="True = historical simulation entry")

    notes: Optional[str] = Field(None, max_length=1000)

    @field_validator("odds_taken")
    @classmethod
    def validate_american_odds(cls, v: float) -> float:
        if v == 0:
            raise ValueError("odds_taken cannot be 0")
        if -99 < v < 100:
            raise ValueError(
                f"odds_taken={v} is not valid American odds. "
                "Must be >= +100 or <= -100."
            )
        return v

    @field_validator("bet_size_units")
    @classmethod
    def validate_units(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("bet_size_units must be positive")
        return round(v, 4)

    model_config = {
        "json_schema_extra": {
            "example": {
                "game_id": 42,
                "prediction_id": 17,
                "pick": "Duke -4.5",
                "bet_type": "spread",
                "odds_taken": -110,
                "bet_size_units": 1.0,
                "bet_size_dollars": 10.0,
                "bankroll_at_bet": 1000.0,
                "model_prob": 0.573,
                "lower_ci_prob": 0.511,
                "is_paper_trade": True,
            }
        }
    }


class BetLogResponse(BaseModel):
    """Response schema for a logged bet."""
    message: str
    bet_id: int
    pick: str
    bet_size_units: float
    is_paper_trade: bool


# ---------------------------------------------------------------------------
# Outcome update
# ---------------------------------------------------------------------------

class OutcomeUpdate(BaseModel):
    """
    Payload for PUT /api/bets/{bet_id}/outcome.

    Supplying closing_spread enables full CLV calculation (preferred).
    If only closing_odds is provided, juice-only CLV is computed.
    At minimum, outcome is required.
    """

    outcome: Literal[0, 1] = Field(..., description="1 = win, 0 = loss")

    # Closing line data — provide as much as possible
    closing_spread: Optional[float] = Field(
        None, description="Closing spread for our side (e.g. -6.0)"
    )
    closing_odds: Optional[float] = Field(
        None, description="Closing American odds for our side (e.g. -115)"
    )
    closing_odds_other_side: Optional[float] = Field(
        None, description="Closing American odds for the other side (for vig removal)"
    )

    notes: Optional[str] = Field(None, max_length=500)

    @field_validator("closing_odds", "closing_odds_other_side", mode="before")
    @classmethod
    def validate_closing_odds(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v == 0:
            raise ValueError("Closing odds cannot be 0")
        if -99 < v < 100:
            raise ValueError(
                f"closing_odds={v} is not valid American odds. "
                "Must be >= +100 or <= -100."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "outcome": 1,
                "closing_spread": -6.0,
                "closing_odds": -110,
                "closing_odds_other_side": -110,
            }
        }
    }


class OutcomeResponse(BaseModel):
    """Response schema after settling a bet."""
    message: str
    bet_id: int
    outcome: int
    profit_loss_dollars: Optional[float]
    profit_loss_units: Optional[float]
    clv_points: Optional[float]
    clv_prob: Optional[float]
    clv_grade: Optional[str]


# ---------------------------------------------------------------------------
# Analysis trigger
# ---------------------------------------------------------------------------

class AnalysisTriggerResponse(BaseModel):
    """Response from /admin/run-analysis."""
    message: str
    status: str
    games_analyzed: int
    bets_recommended: int
    paper_trades_created: int
    errors: list[str]
    duration_seconds: float


# ---------------------------------------------------------------------------
# Prediction fetching
# ---------------------------------------------------------------------------

class GameResponse(BaseModel):
    """Basic details of a game."""
    id: int
    game_date: datetime
    home_team: str
    away_team: str
    is_neutral: bool

    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    """Full details of a single prediction, including the game."""
    id: int
    game_id: int
    model_version: str
    prediction_date: date
    projected_margin: float | None
    edge_conservative: float | None
    recommended_units: float | None
    verdict: str
    pass_reason: str | None
    full_analysis: dict | None
    game: GameResponse

    class Config:
        from_attributes = True


class TodaysPredictionsResponse(BaseModel):
    """Structure for the /api/predictions/today endpoint."""
    date: date
    total_games: int
    bets_recommended: int
    predictions: list[PredictionResponse]

