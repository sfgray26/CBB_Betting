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
from typing import Any, Dict, List, Optional, Tuple, Literal

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
    BEHIND = "BEHIND"
    ON_TRACK = "ON_TRACK"
    AHEAD = "AHEAD"


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


class FreshnessSeverity(str, Enum):
    """Normalized severity classification for data freshness across all services."""
    FRESH = "fresh"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class FreshnessState(BaseModel):
    """Canonical freshness state attached to any service that returns time-sensitive data."""
    source_name: str
    last_updated: Optional[datetime]
    age_minutes: Optional[float]
    severity: FreshnessSeverity
    is_stale: bool
    message: Optional[str] = None

    class Config:
        frozen = True


def compute_freshness(
    source_name: str,
    last_updated: Optional[datetime],
    warning_minutes: int = 60,
    critical_minutes: int = 120,
) -> FreshnessState:
    """Compute FreshnessState for any data source."""
    if last_updated is None:
        return FreshnessState(
            source_name=source_name,
            last_updated=None,
            age_minutes=None,
            severity=FreshnessSeverity.UNKNOWN,
            is_stale=True,
            message=f"{source_name}: no timestamp available",
        )

    now = _now_et()
    if last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=ZoneInfo("America/New_York"))

    age = (now - last_updated).total_seconds() / 60.0
    if age < warning_minutes:
        severity = FreshnessSeverity.FRESH
        is_stale = False
        message = None
    elif age < critical_minutes:
        severity = FreshnessSeverity.WARNING
        is_stale = True
        message = (
            f"{source_name}: data is {age:.0f} minutes old "
            f"(warning threshold: {warning_minutes}m)"
        )
    else:
        severity = FreshnessSeverity.CRITICAL
        is_stale = True
        message = (
            f"{source_name}: data is {age:.0f} minutes old "
            f"(critical threshold: {critical_minutes}m)"
        )

    return FreshnessState(
        source_name=source_name,
        last_updated=last_updated,
        age_minutes=round(age, 1),
        severity=severity,
        is_stale=is_stale,
        message=message,
    )


class FreshnessReport(BaseModel):
    """Aggregated freshness report for all data sources. Returned by GET /api/fantasy/freshness."""
    computed_at: datetime = Field(default_factory=_now_et)
    sources: List[FreshnessState]
    overall_severity: FreshnessSeverity
    stale_count: int
    fresh_count: int

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
    # Live scoring (completed games so far)
    categories_won: int                         # MS-13
    categories_lost: int                        # MS-13
    categories_tied: int                        # MS-13
    # Projected scoring (rest-of-season projections)
    projected_won: Optional[int] = None         # MS-14
    projected_lost: Optional[int] = None        # MS-14
    projected_tied: Optional[int] = None        # MS-14
    # Win probability based on projections, NOT live score
    overall_win_probability: Optional[float] = None  # MS-15
    # Explanation of discrepancy between live and projected scores
    score_explanation: Optional[str] = None     # P0-4: Clarify win prob vs live
    rows: List[MatchupScoreboardRow]            # 18 rows, one per scoring category
    budget: ConstraintBudget
    freshness: FreshnessMetadata

    class Config:
        frozen = True


class BudgetResponse(BaseModel):
    """Budget constraint state. Returned by GET /api/fantasy/budget."""
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
    weather: Optional[Dict] = None               # PR-6: temp_f, wind_mph, wind_direction, precip_chance
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
    rolling_14d: Optional[CategoryStats] = None     # 14-day rolling (ROW projection input)
    rolling_15d: Optional[CategoryStats] = None     # PR-15
    rolling_30d: Optional[CategoryStats] = None     # PR-16
    ros_projection: Optional[CategoryStats] = None  # PR-17
    row_projection: Optional[CategoryStats] = None  # PR-18 (Phase 2 deliverable)
    # Metadata
    ownership_pct: Optional[float] = None            # PR-19
    injury_status: Optional[str] = None              # PR-20
    injury_return_timeline: Optional[str] = None     # PR-21
    freshness: FreshnessMetadata                     # PR-22
    # Current roster slot (Yahoo selected_position: BN, SP, RP, C, 1B, OF, IL, etc.)
    current_slot: Optional[str] = None
    # Internal IDs (not displayed)
    yahoo_player_key: Optional[str] = None
    bdl_player_id: Optional[int] = None
    mlbam_id: Optional[int] = None

    @field_validator("injury_status", "injury_return_timeline", mode="before")
    @classmethod
    def coerce_injury_fields_to_string(cls, v):
        """Yahoo API returns boolean injury flags; coerce to canonical strings.

        bool True  → "IL"  (Yahoo flag means player is on injured list)
        bool False → None  (no injury; frontend distinguishes null from "Active")
        """
        if isinstance(v, bool):
            return "IL" if v else None
        return v

    class Config:
        frozen = True


# P0-7: CategoryMathResult + CategoryMathSummary
class CategoryMathResult(BaseModel):
    """Margin and delta-to-flip for a single scoring category."""
    canonical_code: str                 # v2 canonical code (e.g., "R", "ERA", "AVG")
    margin: float                       # Positive = winning, negative = losing
    delta_to_flip: float                # Change needed to reverse winner
    is_winning: bool                    # True if margin > 0
    display_delta: Optional[str] = None  # Human-readable delta string (computed property)

    class Config:
        frozen = True


class CategoryMathSummary(BaseModel):
    """Batch category math results for all 18 scoring categories."""
    results: Dict[str, CategoryMathResult]  # Keyed by canonical_code
    categories_won: int                     # Count of categories with margin > 0
    categories_lost: int                    # Count of categories with margin < 0
    categories_tied: int                    # Count of categories with margin == 0

    class Config:
        frozen = True


# P0-8: RosterResponse (CanonicalPlayerRow version)
class CanonicalRosterResponse(BaseModel):
    """Full roster response with CanonicalPlayerRow format. Returned by GET /api/fantasy/roster."""
    team_key: str
    team_name: Optional[str] = None
    players: List[CanonicalPlayerRow]
    count: int
    freshness: FreshnessMetadata

    class Config:
        frozen = True


# P0-9: RosterMoveRequest + RosterMoveResponse
class RosterMoveRequest(BaseModel):
    """Request to move a player to a new roster slot."""
    player_key: str
    target_position: str  # 'C','1B','2B','3B','SS','LF','CF','RF','OF','Util','SP','RP','P','BN','IL','IL60'


class RosterMoveResponse(BaseModel):
    """Response from roster move operation."""
    success: bool
    player_key: str
    from_position: Optional[str] = None
    to_position: str
    message: str
    warnings: List[str] = Field(default_factory=list)
    freshness: FreshnessMetadata

    class Config:
        frozen = True


# BulkRosterMove schemas
class BulkRosterMove(BaseModel):
    """A single move within a bulk operation."""
    player_key: str
    target_position: str


class BulkRosterMoveRequest(BaseModel):
    """Request to apply multiple roster moves atomically."""
    moves: List[BulkRosterMove]


class BulkRosterMoveResponse(BaseModel):
    """Response from bulk roster move operation."""
    applied_count: int
    failed_count: int
    errors: List[str] = Field(default_factory=list)

    class Config:
        frozen = True


# MatchupPreview schemas — field names must match frontend MatchupPreviewResponse in types.ts
class MatchupPreviewCategoryProjection(BaseModel):
    """Per-category projection row for the weekly preview table."""
    category: str            # UPPERCASE v2 code, e.g. "HR_B", "ERA" (frontend filters on uppercase)
    win_prob: float          # probability my team wins this category (0–1)
    my_proj: Optional[float] = None   # projected stat value (None when simulation-only)
    opp_proj: Optional[float] = None  # opponent projected stat value

    class Config:
        frozen = True


class WeakCategory(BaseModel):
    """Category where we're projected to lose — drives streaming recommendations."""
    category: str            # lowercase v2 code
    label: str               # human-readable e.g. "ERA"
    win_prob: float
    my_proj: Optional[float] = None
    opp_proj: Optional[float] = None
    reason: str

    class Config:
        frozen = True


class ScheduleAdvantage(BaseModel):
    """Games scheduled for each team during the preview week."""
    my_games: int
    opponent_games: int

    class Config:
        frozen = True


class MatchupPreviewResponse(BaseModel):
    """Full next-week matchup preview — shape must match frontend MatchupPreviewResponse."""
    week_number: int
    opponent_name: str
    opponent_logo: Optional[str] = None
    overall_win_prob: Optional[float] = None
    category_projections: List[MatchupPreviewCategoryProjection]
    weak_categories: List[WeakCategory]
    schedule_advantage: ScheduleAdvantage
    message: Optional[str] = None

    class Config:
        frozen = True


# DecisionAccuracy schemas
class DecisionAccuracyTrendPoint(BaseModel):
    """Accuracy for one day in the trend window."""
    date: str         # YYYY-MM-DD
    accuracy_pct: float  # correct_predictions / total_resolved (0–1), -1 when no data

    class Config:
        frozen = True


class DecisionAccuracyResponse(BaseModel):
    """14-day override accuracy summary for GET /api/fantasy/decisions/accuracy."""
    date: str               # latest date in the window
    total_overrides: int    # total override decisions (better + worse)
    better_count: int       # times user override beat the system
    worse_count: int        # times system was right and user overrode
    override_accuracy_pct: float  # better / (better + worse), 0.0 when no overrides
    daily_trend: List[DecisionAccuracyTrendPoint]  # last 14 days, oldest first

    class Config:
        frozen = True


# P0-10: RosterOptimizeRequest + RosterOptimizeResponse

class ScoreBreakdown(BaseModel):
    """P28: Transparent breakdown of a player's blended lineup_score.

    Exposes the three signals (talent / form / matchup) that combine into the
    final score, so a user can see WHY a player ranks where they do rather than
    trusting an opaque number. Populated only when the blended-score path is
    active; absent (None) on the legacy 14-day-only path.
    """
    final_z: float
    talent_z: Optional[float] = None
    form_z: Optional[float] = None
    form_z_shrunk: Optional[float] = None
    matchup_z: Optional[float] = None
    confidence: float = 0.0
    talent_source: str = "none"
    low_confidence: bool = False

    class Config:
        frozen = True


class PlayerSlotAssignment(BaseModel):
    """Slot assignment for a single player."""
    player_key: str
    player_name: str
    assigned_slot: str  # 'C','1B','2B','3B','SS','OF','Util','SP','RP','P','BN'
    lineup_score: float
    reasoning: str
    # P28: optional blended-score transparency. Absent on the legacy path.
    score_breakdown: Optional[ScoreBreakdown] = None
    matchup_note: Optional[str] = None
    low_confidence: bool = False

    class Config:
        frozen = True


class RosterOptimizeRequest(BaseModel):
    """Request to optimize roster lineup."""
    target_date: Optional[str] = None  # YYYY-MM-DD, defaults to today

    class Config:
        frozen = True


class LineupMove(BaseModel):
    """A single player slot change in the proposed lineup diff."""
    player_key: str
    player_name: str
    from_slot: str
    to_slot: str
    lineup_score: float

    class Config:
        frozen = True


class ProposedLineupDiff(BaseModel):
    """Diff showing what changes from current to proposed lineup."""
    bench_to_start: List[LineupMove] = []
    start_to_bench: List[LineupMove] = []
    net_score_impact: float = 0.0


class RosterOptimizeResponse(BaseModel):
    """Response from roster optimization."""
    success: bool
    message: str
    target_date: str
    starters: List[PlayerSlotAssignment]
    bench: List[PlayerSlotAssignment]
    unrostered: List[str]  # player_keys that couldn't fit
    total_lineup_score: float
    freshness: FreshnessMetadata
    proposed_diff: Optional["ProposedLineupDiff"] = None
    schedule_available: bool = True  # False when no MLB games found for target_date

    class Config:
        frozen = True



# ---------------------------------------------------------------------------
# Trade Analyzer schemas
# ---------------------------------------------------------------------------

class TradePlayerInput(BaseModel):
    """A single player in a trade request identified by Yahoo player_key."""
    player_key: str
    player_name: Optional[str] = None  # Optional; used to help projection lookup


class TradeAnalyzeRequest(BaseModel):
    """Request body for POST /api/fantasy/trade/analyze."""
    give: List[TradePlayerInput]     # Players you are giving away
    receive: List[TradePlayerInput]  # Players you are receiving
    league_id: Optional[str] = None  # Reserved; defaults to env YAHOO_LEAGUE_ID


class TradeCategoryDelta(BaseModel):
    """Projected impact on a single H2H scoring category from this trade."""
    category: str    # lowercase board key, e.g. "hr", "era", "avg"
    give_z: float    # Sum of z-scores for that category across give-side players
    receive_z: float # Sum of z-scores for that category across receive-side players
    delta: float     # receive_z - give_z  (positive = receiving side helps more)
    direction: str   # "gain" | "loss" | "neutral"

    class Config:
        frozen = True


class TradeAnalysis(BaseModel):
    """Full trade analysis result returned by analyze_trade()."""
    give_players: List[Dict[str, Any]]       # Compact projection summaries for give side
    receive_players: List[Dict[str, Any]]    # Compact projection summaries for receive side
    category_deltas: List[TradeCategoryDelta]
    total_z_delta: float   # Weighted sum of all per-category deltas
    recommendation: str    # 'strong_accept'|'accept'|'neutral'|'reject'|'strong_reject'
    summary: str           # Human-readable explanation

    class Config:
        frozen = True


# ---------------------------------------------------------------------------
# Auto-Stream Configuration Contracts
# ---------------------------------------------------------------------------

class AutoStreamConfigureRequest(BaseModel):
    """Request body for POST /api/fantasy/auto-stream/configure."""

    enabled: bool = Field(description="Whether Auto-Stream is enabled")
    drop_priority: List[str] = Field(default_factory=list, description="Ordered list of player IDs to drop first")
    min_confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(default="HIGH", description="Minimum confidence level")
    min_recommendation: Literal["EXCELLENT", "GOOD", "AVERAGE"] = Field(default="EXCELLENT", description="Minimum recommendation tier")
    max_adds_per_week: int = Field(default=2, ge=1, le=10, description="Maximum adds per scoring week")

    class Config:
        frozen = True


class PredictiveStatsMeta(BaseModel):
    """Metadata contract for predictive stats pipeline state."""

    enabled: bool
    pitcher_stats_available: bool = False
    batter_stats_available: bool = False
    last_refresh: Optional[datetime] = None
    error: Optional[str] = None

    class Config:
        frozen = True
