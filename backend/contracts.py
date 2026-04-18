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

from pydantic import BaseModel, Field, field_validator
from zoneinfo import ZoneInfo

from backend.stat_contract import SCORING_CATEGORY_CODES, LOWER_IS_BETTER, BATTING_CODES


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


# ---------------------------------------------------------------------------
# UI Contracts — Pydantic models for frontend-bound data
# These contracts define the authoritative shapes for all UI-bound data.
# All category-keyed dicts use canonical codes from the loaded stat contract.
# ---------------------------------------------------------------------------

# P0-1: CategoryStatusTag
class CategoryStatusTag(str, Enum):
    """Classification of a scoring category's matchup status.

    Threshold definitions (for L3/L4 classification logic):
    - LOCKED_WIN:   Monte Carlo win probability > 90%
    - LOCKED_LOSS:  Monte Carlo win probability < 10%
    - LEANING_WIN:  65% < win probability <= 90%
    - LEANING_LOSS: 10% <= win probability < 35%
    - BUBBLE:       35% <= win probability <= 65%
    """
    LOCKED_WIN = "locked_win"
    LOCKED_LOSS = "locked_loss"
    LEANING_WIN = "leaning_win"
    LEANING_LOSS = "leaning_loss"
    BUBBLE = "bubble"


# P0-2: IPPaceFlag + ConstraintBudget
class IPPaceFlag(str, Enum):
    """Weekly innings pitched pace relative to league minimum."""
    BEHIND = "behind"
    ON_TRACK = "on_track"
    AHEAD = "ahead"


class ConstraintBudget(BaseModel):
    """Current constraint state for the global header. Fields map to GH-6 through GH-14."""
    acquisitions_used: int
    acquisitions_remaining: int
    acquisition_limit: int
    acquisition_warning: bool
    il_used: int
    il_total: int
    ip_accumulated: float
    ip_minimum: float
    ip_pace: IPPaceFlag
    as_of: datetime

    class Config:
        frozen = True


# P0-3: FreshnessMetadata
class FreshnessMetadata(BaseModel):
    """Per-response freshness annotation. Every API response must include this."""
    primary_source: str
    fetched_at: Optional[datetime]
    computed_at: datetime
    staleness_threshold_minutes: int
    is_stale: bool

    class Config:
        frozen = True


# P0-4: CategoryStats
class CategoryStats(BaseModel):
    """Stats for a single time window across all scoring categories."""
    values: Dict[str, Optional[float]]

    @field_validator("values")
    @classmethod
    def validate_category_keys(cls, v: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        actual = set(v.keys())
        missing = SCORING_CATEGORY_CODES - actual
        if missing:
            raise ValueError(f"Missing scoring categories: {missing}")
        extra = actual - SCORING_CATEGORY_CODES
        if extra:
            raise ValueError(f"Unexpected category keys: {extra}")
        return v

    class Config:
        frozen = True


# P0-5: MatchupScoreboardRow + MatchupScoreboardResponse
class MatchupScoreboardRow(BaseModel):
    """One row per scoring category on the matchup scoreboard. Fields map to MS-1 through MS-12."""
    category: str                        # MS-1: canonical code (e.g. "HR_B", not "HR")
    category_label: str                  # MS-1: short_label from contract (e.g. "HR")
    is_lower_better: bool                # From contract direction field
    is_batting: bool                     # True if batting, False if pitching
    my_current: float                    # MS-2
    opp_current: float                   # MS-3
    current_margin: float                # MS-4: signed (positive = winning, respects direction)
    # --- Phase 2/3 fields: Optional until ROW projections + Monte Carlo are wired ---
    my_projected_final: Optional[float] = None  # MS-5
    opp_projected_final: Optional[float] = None # MS-6
    projected_margin: Optional[float] = None    # MS-7
    status: Optional[CategoryStatusTag] = None  # MS-8
    flip_probability: Optional[float] = None    # MS-9
    delta_to_flip: Optional[str] = None         # MS-10
    games_remaining: Optional[int] = None       # MS-11
    ip_context: Optional[str] = None            # MS-12

    class Config:
        frozen = True


class MatchupScoreboardResponse(BaseModel):
    """Full matchup scoreboard. Returned by GET /api/fantasy/scoreboard."""
    week: int
    opponent_name: str
    categories_won: int                         # MS-13
    categories_lost: int                        # MS-13
    categories_tied: int                        # MS-13
    projected_won: Optional[int] = None         # MS-14
    projected_lost: Optional[int] = None        # MS-14
    projected_tied: Optional[int] = None        # MS-14
    overall_win_probability: Optional[float] = None  # MS-15
    rows: List[MatchupScoreboardRow]            # 18 rows, one per scoring category
    budget: ConstraintBudget
    freshness: FreshnessMetadata

    class Config:
        frozen = True


# P0-6: CanonicalPlayerRow + PlayerGameContext
class PlayerGameContext(BaseModel):
    """Today's game context for a player."""
    opponent: str
    home_away: str                               # "home" or "away"
    game_time: Optional[datetime] = None
    # Pitcher-specific
    projected_k: Optional[float] = None          # PR-7
    projected_era_impact: Optional[float] = None # PR-8
    # Hitter-specific
    opposing_sp_name: Optional[str] = None       # PR-11
    opposing_sp_handedness: Optional[str] = None # PR-11: "L" or "R"
    projected_impact: Optional[float] = None     # PR-12

    class Config:
        frozen = True


class CanonicalPlayerRow(BaseModel):
    """Universal player representation. Fields map to PR-1 through PR-22."""
    # Identity
    player_name: str                             # PR-1
    team: str                                    # PR-2
    eligible_positions: List[str]                # PR-3
    # Status
    status: str                                  # PR-4: playing|not_playing|probable|IL|minors
    game_context: Optional[PlayerGameContext] = None  # PR-5 through PR-12
    # Stats by window (keyed by canonical codes via CategoryStats validator)
    season_stats: Optional[CategoryStats] = None    # PR-13
    rolling_7d: Optional[CategoryStats] = None      # PR-14
    rolling_15d: Optional[CategoryStats] = None     # PR-15
    rolling_30d: Optional[CategoryStats] = None     # PR-16
    ros_projection: Optional[CategoryStats] = None  # PR-17
    row_projection: Optional[CategoryStats] = None  # PR-18 (Phase 2 deliverable)
    # Metadata
    ownership_pct: Optional[float] = None            # PR-19
    injury_status: Optional[str] = None              # PR-20
    injury_return_timeline: Optional[str] = None     # PR-21
    freshness: FreshnessMetadata                     # PR-22
    # Internal IDs (not displayed)
    yahoo_player_key: Optional[str] = None
    bdl_player_id: Optional[int] = None
    mlbam_id: Optional[int] = None

    class Config:
        frozen = True

