"""
Pydantic request/response schemas for the CBB Edge API.

Using explicit schemas instead of raw dicts prevents mass-assignment
vulnerabilities on ORM models and generates accurate OpenAPI docs.
"""

from __future__ import annotations

from typing import List, Literal, Optional
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


# ---------------------------------------------------------------------------
# Fantasy Baseball
# ---------------------------------------------------------------------------

class FantasyPlayerResponse(BaseModel):
    """A single player from the draft board."""
    id: str
    name: str
    team: str
    positions: list[str]
    type: str            # "batter" | "pitcher"
    tier: int
    rank: int
    adp: float
    z_score: float
    proj: dict
    cat_scores: Optional[dict] = None


class FantasyDraftBoardResponse(BaseModel):
    """Response for GET /api/fantasy/draft-board."""
    count: int
    players: list[FantasyPlayerResponse]


class DraftPickCreate(BaseModel):
    """Body for POST /api/fantasy/draft-session/{key}/pick."""
    player_id: str
    drafter_position: int = Field(..., ge=1, le=20)
    is_my_pick: bool = False


class DraftPickResponse(BaseModel):
    """Confirmation after recording a pick."""
    message: str
    pick_number: int
    player_name: str
    is_my_pick: bool
    next_recommendations: list[FantasyPlayerResponse]


class LineupSaveRequest(BaseModel):
    """Body for POST /api/fantasy/lineup."""
    lineup_date: date
    platform: str = "yahoo"
    positions: dict   # {"C": "player_id", ...}
    projected_points: Optional[float] = None
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# EMAC-075: Fantasy Season Ops
# ---------------------------------------------------------------------------

class LineupPlayerOut(BaseModel):
    """Daily batter recommendation."""
    player_id: str
    name: str
    team: str
    position: str
    implied_runs: float
    park_factor: float
    lineup_score: float
    start_time: Optional[datetime] = None
    opponent: Optional[str] = None
    status: str = "UNKNOWN"       # "START" | "BENCH" | "UNKNOWN"
    assigned_slot: Optional[str] = None  # "C", "1B", "2B", "3B", "SS", "OF", "Util", "BN"
    has_game: bool = True

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class StartingPitcherOut(BaseModel):
    """Daily pitcher recommendation with SP/RP delineation."""
    player_id: str
    name: str
    team: str
    pitcher_type: str = "SP"  # "SP" | "RP" | "P" (ambiguous)
    opponent: str = ""  # Opposing team
    opponent_implied_runs: float
    park_factor: float
    sp_score: float
    start_time: Optional[datetime] = None
    status: str = "UNKNOWN"  # "START" | "NO_START" | "RP"

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}


class DailyLineupResponse(BaseModel):
    """Response for GET /api/fantasy/lineup/{date}."""
    date: date
    batters: List[LineupPlayerOut]
    pitchers: List[StartingPitcherOut]
    games_count: int
    no_games_today: bool = False
    lineup_warnings: List[str] = []


class CategoryDeficitOut(BaseModel):
    """Matchup category status."""
    category: str
    my_total: float
    opponent_total: float
    deficit: float
    winning: bool


class WaiverPlayerOut(BaseModel):
    """Waiver wire recommendation."""
    player_id: str
    name: str
    team: str
    position: str
    need_score: float
    category_contributions: dict
    owned_pct: float
    starts_this_week: int
    statcast_signals: List[str] = []
    projected_saves: float = 0.0
    hot_cold: Optional[str] = None        # "HOT" | "COLD" | None
    status: Optional[str] = None          # Yahoo status: Active, DTD, IL, etc.
    injury_note: Optional[str] = None     # Yahoo injury note text


class PaginationOut(BaseModel):
    """Pagination metadata for list endpoints."""
    page: int
    per_page: int
    has_next: bool


class WaiverWireResponse(BaseModel):
    """Response for GET /api/fantasy/waiver."""
    week_end: date
    matchup_opponent: str
    category_deficits: List[CategoryDeficitOut]
    top_available: List[WaiverPlayerOut]
    two_start_pitchers: List[WaiverPlayerOut]
    pagination: Optional[PaginationOut] = None
    urgent_alert: Optional[dict] = None
    closer_alert: Optional[str] = None      # "NO_CLOSERS" | "LOW_CLOSERS" | None
    il_slots_used: int = 0
    il_slots_available: int = 0
    faab_balance: Optional[float] = None    # Remaining FAAB budget (None if not FAAB league)


class RosterMoveRecommendation(BaseModel):
    """A specific ADD, DROP, or ADD/DROP recommendation."""
    action: str                              # "ADD", "DROP", "ADD_DROP", "HOLD"
    add_player: Optional[WaiverPlayerOut]    # Player to add (null for DROP-only)
    drop_player_name: Optional[str]          # Name of player to drop (null for ADD-only)
    drop_player_position: Optional[str]      # Position of drop candidate
    rationale: str                           # Human-readable explanation
    category_targets: List[str]              # Which cats this move helps
    need_score: float                        # Composite value of this move
    confidence: float                        # 0.0-1.0 based on data completeness
    statcast_signals: List[str] = []         # ["BUY_LOW", "BREAKOUT", "SELL_HIGH", "HIGH_INJURY_RISK"]
    regression_delta: float = 0.0           # xwOBA-wOBA (batters) or xERA-ERA (pitchers)
    # MCMC win probability fields
    win_prob_before: float = 0.0            # Win prob before the move (0-1)
    win_prob_after: float = 0.0             # Win prob after the move (0-1)
    win_prob_gain: float = 0.0              # Absolute gain (after - before)
    category_win_probs: dict = {}           # Per-category win probability after move
    mcmc_enabled: bool = False              # True if MCMC simulation ran successfully


class WaiverRecommendationsResponse(BaseModel):
    """Response for GET /api/fantasy/waiver/recommendations."""
    week_end: date
    matchup_opponent: str
    recommendations: List[RosterMoveRecommendation]
    category_deficits: List[CategoryDeficitOut]


# ---------------------------------------------------------------------------
# MCMC Matchup Simulation
# ---------------------------------------------------------------------------

class MatchupSimulateRequest(BaseModel):
    """Request body for POST /api/fantasy/matchup/simulate."""
    my_roster: List[dict]
    opponent_roster: List[dict]
    n_sims: int = 1000
    week: Optional[str] = None  # ISO week label (informational only)


# ---------------------------------------------------------------------------
# EMAC-076: Yahoo Roster, Matchup, Lineup Apply
# ---------------------------------------------------------------------------

class RosterPlayerOut(BaseModel):
    player_key: str
    name: str
    team: Optional[str] = None
    positions: List[str] = []
    status: Optional[str] = None
    injury_note: Optional[str] = None
    z_score: Optional[float] = None
    is_undroppable: bool = False
    is_proxy: bool = False
    cat_scores: dict = {}
    selected_position: Optional[str] = None  # Yahoo lineup slot: IL, BN, C, 1B, OF, etc.


class RosterResponse(BaseModel):
    team_key: str
    players: List[RosterPlayerOut]
    count: int


class MatchupTeamOut(BaseModel):
    team_key: str
    team_name: str
    stats: dict


class MatchupResponse(BaseModel):
    week: Optional[int] = None
    my_team: MatchupTeamOut
    opponent: MatchupTeamOut
    is_playoffs: bool = False
    message: Optional[str] = None


class LineupApplyPlayer(BaseModel):
    player_key: str
    position: str


class LineupApplyRequest(BaseModel):
    date: Optional[str] = None
    players: List[LineupApplyPlayer]


# ---------------------------------------------------------------------------
# K-15: Oracle Validation
# ---------------------------------------------------------------------------

class OraclePredictionDetail(BaseModel):
    """One flagged prediction returned by GET /admin/oracle/flagged."""
    prediction_id: int
    game_date: datetime
    home_team: str
    away_team: str
    verdict: str
    projected_margin: Optional[float] = None
    oracle_spread: Optional[float] = None
    divergence_points: Optional[float] = None
    divergence_z: Optional[float] = None
    threshold_z: Optional[float] = None
    sources: List[str] = []
    run_tier: Optional[str] = None
    prediction_date: Optional[date] = None

    class Config:
        from_attributes = True


class OracleFlaggedResponse(BaseModel):
    """Response from GET /admin/oracle/flagged."""
    flagged_count: int
    predictions: List[OraclePredictionDetail]

