"""
Decision Contracts — immutable data boundaries between the API layer and the Worker layer.

Rules:
- All contracts are frozen (Config: frozen = True)
- All timestamps use Eastern Time via ZoneInfo
- Nothing crosses the API/Worker boundary except instances of these classes
- Once created, contracts are never mutated
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo


def _now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))


def _new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------

class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"


class DataSource(str, Enum):
    YAHOO = "yahoo"
    STATCAST = "statcast"
    PLAYER_BOARD = "player_board"
    MCMC = "mcmc"


class UncertaintyRange(BaseModel):
    """Probabilistic range — every projection must carry uncertainty."""
    point_estimate: float
    lower_80: float
    upper_80: float
    lower_95: float
    upper_95: float
    std_dev: float
    sample_size: int

    class Config:
        frozen = True


class AuditTrail(BaseModel):
    """Immutable provenance record attached to every contract."""
    created_at: datetime = Field(default_factory=_now_et)
    model_version: str
    data_sources: List[DataSource]
    data_as_of: datetime
    computation_ms: int
    warnings: List[str] = Field(default_factory=list)

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Contract 1: LineupOptimizationRequest
# ---------------------------------------------------------------------------

class PlayerSlot(BaseModel):
    player_id: str
    name: str
    eligible_positions: List[str]
    projected_value: Optional[UncertaintyRange] = None
    opponent: Optional[str] = None
    game_time: Optional[datetime] = None
    is_probable_starter: bool = False
    injury_status: Optional[str] = None

    class Config:
        frozen = True


class LineupOptimizationRequest(BaseModel):
    """Immutable input contract. The Worker receives exactly this."""
    request_id: str = Field(default_factory=_new_id)
    submitted_at: datetime = Field(default_factory=_now_et)
    league_key: str
    team_key: str
    scoring_categories: List[str]
    roster_positions: List[str]
    available_players: List[PlayerSlot]
    risk_tolerance: RiskTolerance = RiskTolerance.BALANCED
    target_date: str  # YYYY-MM-DD — always explicit, never default to "today"
    max_bench_sp: int = 2
    locked_starters: List[str] = Field(default_factory=list)
    locked_bench: List[str] = Field(default_factory=list)

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Contract 2: PlayerValuationReport
# ---------------------------------------------------------------------------

class CategoryProjection(BaseModel):
    category: str
    projection: UncertaintyRange
    z_score: float
    rank_in_pool: Optional[int] = None

    class Config:
        frozen = True


class PlayerValuationReport(BaseModel):
    """Pre-computed per player per day. Worker produces; API reads from cache."""
    report_id: str = Field(default_factory=_new_id)
    player_id: str
    player_name: str
    target_date: str
    category_projections: List[CategoryProjection]
    composite_value: UncertaintyRange
    matchup_quality: float
    start_probability: float
    recent_form_delta: float
    platoon_flag: Optional[str] = None
    park_factor: Optional[float] = None
    audit: AuditTrail

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Contract 3: ExecutionDecision
# ---------------------------------------------------------------------------

class SafetyCheck(BaseModel):
    check_type: str  # "no_game" | "injury" | "weather" | "pitcher_quality"
    status: str      # "pass" | "warning" | "block"
    affected_player: Optional[str] = None
    message: str

    class Config:
        frozen = True


class LineupAlternative(BaseModel):
    description: str
    swap: Tuple[str, str]  # (player_out, player_in)
    expected_value_delta: float
    risk_profile: str  # "safer" | "higher_ceiling" | "punts_category"

    class Config:
        frozen = True


class ExecutionDecision(BaseModel):
    """Immutable recommendation. Stored permanently — forms backtesting corpus."""
    decision_id: str = Field(default_factory=_new_id)
    request_id: str
    decided_at: datetime = Field(default_factory=_now_et)
    starters: List[str]
    bench: List[str]
    slot_assignments: Dict[str, str]
    primary_reasoning: List[str]
    category_impact: Dict[str, float]
    confidence_score: float
    win_probability: Optional[UncertaintyRange] = None
    expected_outcome_range: Tuple[float, float]
    alternatives: List[LineupAlternative]
    safety_checks: List[SafetyCheck]
    audit: AuditTrail

    class Config:
        frozen = True
