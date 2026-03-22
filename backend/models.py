"""
Database models for CBB Edge Analyzer
SQLAlchemy ORM with PostgreSQL
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    JSON,
    Text,
    ForeignKey,
    Date,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, date
import os
import time

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Railway provides env vars directly

# 2. Sync URL — used by background scripts, migrations, and legacy sync code.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/cbb_edge")

# 3. Async URL — swaps psycopg2 driver for asyncpg.
#    Falls back gracefully if DATABASE_URL is not set (e.g. test environments).
_ASYNC_DATABASE_URL = DATABASE_URL.replace(
    "postgresql://", "postgresql+asyncpg://"
).replace(
    "postgresql+psycopg2://", "postgresql+asyncpg://"
)

# ── Sync engine (keep for all existing sync paths) ──────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# ── Async engine (used by nightly analysis hot path and APScheduler coroutines)
# Wrapped in try/except so the server still starts if asyncpg is not installed.
# Install asyncpg to enable the async hot path: pip install asyncpg==0.29.0
try:
    async_engine = create_async_engine(
        _ASYNC_DATABASE_URL,
        pool_pre_ping=False,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )
    AsyncSessionLocal = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
except Exception as _async_engine_exc:  # noqa: BLE001
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "asyncpg not available — async DB engine disabled (%s). "
        "Install asyncpg to enable the async hot path.",
        _async_engine_exc,
    )
    async_engine = None  # type: ignore[assignment]
    AsyncSessionLocal = None  # type: ignore[assignment]

Base = declarative_base()


# ── Session dependencies ─────────────────────────────────────────────────────

def get_db():
    """Sync session dependency with retry on transient connection failures."""
    db = None
    for attempt in range(3):
        try:
            db = SessionLocal()
            break
        except Exception as e:
            if attempt == 2:
                raise
            error_str = str(e).lower()
            if any(k in error_str for k in ("connection", "timeout", "ssl")):
                time.sleep(0.1 * (2 ** attempt))
            else:
                raise
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def get_async_db():
    """Async session dependency for FastAPI async routes."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise

# ... (rest of your classes like Game, Prediction, etc. remain exactly the same)

class Game(Base):
    """CBB game with teams and basic info"""

    __tablename__ = "games"

    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(String, unique=True, index=True)  # From odds API
    game_date = Column(DateTime, nullable=False, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    venue = Column(String)
    is_neutral = Column(Boolean, default=False)
    
    # Actual results (filled after game)
    home_score = Column(Integer)
    away_score = Column(Integer)
    completed = Column(Boolean, default=False)
    
    # Relationships
    predictions = relationship("Prediction", back_populates="game")
    bet_logs = relationship("BetLog", back_populates="game")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Prediction(Base):
    """Model predictions for each game"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    
    # Model metadata
    model_version = Column(String, default="v7.0")
    prediction_date = Column(Date, nullable=False, index=True, default=date.today)
    run_tier = Column(String, default="nightly", nullable=False)  # "opener" | "nightly" | "closing"
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Ratings used (for auditing)
    kenpom_home = Column(Float)
    kenpom_away = Column(Float)
    barttorvik_home = Column(Float)
    barttorvik_away = Column(Float)
    evanmiya_home = Column(Float)
    evanmiya_away = Column(Float)
    
    # Model outputs
    projected_margin = Column(Float)  # Positive = home favored
    adjusted_sd = Column(Float)
    point_prob = Column(Float)  # Point estimate probability
    lower_ci_prob = Column(Float)  # Lower 95% CI
    upper_ci_prob = Column(Float)  # Upper 95% CI

    # Actual outcome — populated by update_completed_games() after game finishes.
    # NULL while game is pending.  home_score - away_score (positive = home won).
    actual_margin = Column(Float)

    # Edge calculations
    edge_point = Column(Float)  # Point estimate edge
    edge_conservative = Column(Float)  # Lower CI edge (decision threshold)
    kelly_full = Column(Float)
    kelly_fractional = Column(Float)
    recommended_units = Column(Float)

    # V9 SNR & Integrity
    snr = Column(Float)
    snr_kelly_scalar = Column(Float)
    integrity_verdict = Column(String)
    
    # Verdict
    verdict = Column(String, nullable=False, index=True)  # "PASS" or "Bet X units..."
    pass_reason = Column(String)  # If PASS, why?
    
    # Full analysis (for debugging)
    full_analysis = Column(JSON)
    
    # Data quality
    data_freshness_tier = Column(String)  # Tier 1/2/3
    penalties_applied = Column(JSON)  # Dict of penalty types & values

    # K-14: which simulation engine produced this prediction
    pricing_engine = Column(String(20))  # 'markov' | 'gaussian' | None

    # Relationship
    game = relationship("Game", back_populates="predictions")

    __table_args__ = (UniqueConstraint('game_id', 'prediction_date', 'run_tier', name='_game_prediction_date_tier_uc'),)


class BetLog(Base):
    """Actual bets placed (manual entry or tracking)"""

    __tablename__ = "bet_logs"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    
    # Bet details
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    pick = Column(String, nullable=False)  # "Duke -4.5" or "UNC/Duke U145.5"
    bet_type = Column(String)  # "spread", "total", "moneyline"
    odds_taken = Column(Float, nullable=False)  # American odds
    
    # Sizing
    bankroll_at_bet = Column(Float)
    kelly_full = Column(Float)
    kelly_fractional = Column(Float)
    bet_size_pct = Column(Float)  # % of bankroll
    bet_size_units = Column(Float)  # In "units" (1 unit = 1% starting bankroll)
    bet_size_dollars = Column(Float)  # Actual $ amount
    
    # Model at time of bet
    model_prob = Column(Float)
    lower_ci_prob = Column(Float)
    point_edge = Column(Float)
    conservative_edge = Column(Float)
    
    # Outcome (filled after game)
    outcome = Column(Integer)  # 1=win, 0=loss, null=pending
    profit_loss_units = Column(Float)
    profit_loss_dollars = Column(Float)
    
    # CLV tracking
    closing_line = Column(Float)  # American odds at close
    clv_points = Column(Float)  # Points gained vs close
    clv_prob = Column(Float)  # Probability edge vs close
    
    # Flags
    is_backfill = Column(Boolean, default=False)  # Historical simulation
    is_paper_trade = Column(Boolean, default=False)  # Not real money
    executed = Column(Boolean, default=False)  # Actually placed
    
    # Notes
    notes = Column(Text)
    
    # Relationship
    game = relationship("Game", back_populates="bet_logs")


class ModelParameter(Base):
    """Tracking of model parameter changes over time"""

    __tablename__ = "model_parameters"

    id = Column(Integer, primary_key=True, index=True)
    effective_date = Column(DateTime, default=datetime.utcnow, index=True)
    parameter_name = Column(String, nullable=False)
    parameter_value = Column(Float)
    parameter_value_json = Column(JSON)  # For complex params like weights
    reason = Column(String)  # "quarterly_recalibration", "manual_adjustment", etc.
    changed_by = Column(String)  # "auto" or user identifier
    
    created_at = Column(DateTime, default=datetime.utcnow)


class PerformanceSnapshot(Base):
    """Daily/weekly/monthly performance summaries"""

    __tablename__ = "performance_snapshots"

    id = Column(Integer, primary_key=True, index=True)
    snapshot_date = Column(DateTime, default=datetime.utcnow, index=True)
    period_type = Column(String)  # "daily", "weekly", "monthly", "quarterly"
    period_start = Column(DateTime)
    period_end = Column(DateTime)
    
    # Aggregate stats
    total_bets = Column(Integer)
    total_wins = Column(Integer)
    total_losses = Column(Integer)
    win_rate = Column(Float)
    
    # Financial
    total_risked = Column(Float)
    total_profit_loss = Column(Float)
    roi = Column(Float)
    
    # CLV
    mean_clv = Column(Float)
    median_clv = Column(Float)
    
    # Calibration
    calibration_error = Column(Float)  # MAE between predicted prob and actual
    calibration_bins = Column(JSON)  # {bin: {predicted: X, actual: Y, count: N}}
    
    # Model performance
    mean_edge = Column(Float)
    bets_recommended = Column(Integer)
    pass_rate = Column(Float)
    
    # By system
    kenpom_mae = Column(Float)
    barttorvik_mae = Column(Float)
    evanmiya_mae = Column(Float)


class DataFetch(Base):
    """Track data fetches for monitoring scraper health"""

    __tablename__ = "data_fetches"

    id = Column(Integer, primary_key=True, index=True)
    fetch_time = Column(DateTime, default=datetime.utcnow, index=True)
    data_source = Column(String, nullable=False, index=True)  # "kenpom", "odds_api", etc.
    success = Column(Boolean, nullable=False)
    records_fetched = Column(Integer)
    error_message = Column(Text)
    response_time_ms = Column(Integer)

    created_at = Column(DateTime, default=datetime.utcnow)


class ClosingLine(Base):
    """Closing lines captured near game time for CLV calculation."""

    __tablename__ = "closing_lines"

    id = Column(Integer, primary_key=True, index=True)
    game_id = Column(Integer, ForeignKey("games.id"), nullable=False, index=True)
    captured_at = Column(DateTime, default=datetime.utcnow, index=True)

    spread = Column(Float)        # Home team spread (negative = home favourite)
    spread_odds = Column(Integer)  # American odds for the home spread
    total = Column(Float)
    total_odds = Column(Integer)
    moneyline_home = Column(Integer)
    moneyline_away = Column(Integer)

    game = relationship("Game")


class TeamProfile(Base):
    """
    Per-team offensive and defensive four-factor stats persisted from BartTorvik.

    Columns mirror the TeamSimProfile / TeamPlayStyle dataclasses so that the
    Markov simulator and matchup engine can load real per-team defensive data
    from the database instead of D1-average defaults.
    """

    __tablename__ = "team_profiles"

    id = Column(Integer, primary_key=True, index=True)
    team_name = Column(String, nullable=False, index=True)
    season_year = Column(Integer, nullable=False, index=True)
    source = Column(String, nullable=False, default="barttorvik")  # "barttorvik" | "kenpom"

    # Efficiency margins (KenPom / BartTorvik AdjEM scale, ≈ -30 to +30)
    adj_oe = Column(Float)
    adj_de = Column(Float)
    adj_em = Column(Float)

    # Offensive four factors
    pace = Column(Float)          # Possessions per 40 min
    efg_pct = Column(Float)       # Effective FG% (offensive)
    to_pct = Column(Float)        # Turnover rate (offensive, lower is better)
    ft_rate = Column(Float)       # FT attempts / FGA (offensive)
    three_par = Column(Float)     # 3PA / FGA (offensive)

    # Defensive four factors — the data the Markov engine was previously blind to
    def_efg_pct = Column(Float)   # Opponent eFG% allowed
    def_to_pct = Column(Float)    # Opponent TO rate forced
    def_ft_rate = Column(Float)   # Opponent FT rate allowed
    def_three_par = Column(Float) # Opponent 3PA rate allowed

    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint(
            "team_name", "season_year", "source",
            name="_team_season_source_uc",
        ),
    )


class DBAlert(Base):
    """Persisted performance alerts surfaced in the dashboard."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    alert_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)   # INFO | WARNING | CRITICAL
    message = Column(Text, nullable=False)
    threshold = Column(Float)
    current_value = Column(Float)
    acknowledged = Column(Boolean, default=False, index=True)
    acknowledged_at = Column(DateTime)


class FantasyDraftSession(Base):
    """Tracks a single fantasy draft session state."""

    __tablename__ = "fantasy_draft_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_key = Column(String(50), unique=True, nullable=False, index=True)
    my_draft_position = Column(Integer, nullable=False)
    num_teams = Column(Integer, nullable=False, default=12)
    num_rounds = Column(Integer, nullable=False, default=23)
    current_pick = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    picks = relationship("FantasyDraftPick", back_populates="session",
                         cascade="all, delete-orphan")


class FantasyDraftPick(Base):
    """Records each pick made during a fantasy draft."""

    __tablename__ = "fantasy_draft_picks"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("fantasy_draft_sessions.id"),
                        nullable=False, index=True)
    pick_number = Column(Integer, nullable=False)
    round_number = Column(Integer, nullable=False)
    drafter_position = Column(Integer, nullable=False)  # 1-12
    is_my_pick = Column(Boolean, nullable=False, default=False)
    player_id = Column(String(100), nullable=False)
    player_name = Column(String(100), nullable=False)
    player_team = Column(String(10))
    player_positions = Column(JSON)  # list of position strings
    player_tier = Column(Integer)
    player_adp = Column(Float)
    player_z_score = Column(Float)
    picked_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("FantasyDraftSession", back_populates="picks")

    __table_args__ = (
        UniqueConstraint("session_id", "player_id", name="_session_player_uc"),
    )


class FantasyLineup(Base):
    """Saved daily lineup for fantasy baseball."""

    __tablename__ = "fantasy_lineups"

    id = Column(Integer, primary_key=True, index=True)
    lineup_date = Column(Date, nullable=False, index=True)
    platform = Column(String(30), nullable=False, default="yahoo")
    positions = Column(JSON, nullable=False)   # {"C": "player_id", "1B": "player_id", ...}
    projected_points = Column(Float)
    actual_points = Column(Float)
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("lineup_date", "platform", name="_lineup_date_platform_uc"),
    )


class PlayerDailyMetric(Base):
    """
    Sparse time-series of per-player analytics (EMAC-077 EPIC-1).
    One row per (player_id, metric_date, sport). NULL fields are not computed yet.
    """

    __tablename__ = "player_daily_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    metric_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)  # 'mlb' | 'cbb'

    # Core value metrics
    vorp_7d = Column(Float)
    vorp_30d = Column(Float)
    z_score_total = Column(Float)
    z_score_recent = Column(Float)

    # Statcast 2.0 (MLB only — always NULL for CBB rows)
    blast_pct = Column(Float)
    bat_speed = Column(Float)
    squared_up_pct = Column(Float)
    swing_length = Column(Float)
    stuff_plus = Column(Float)
    plv = Column(Float)

    # Flexible rolling windows: {"7d": {"avg": 0.310, ...}, "30d": {...}}
    rolling_window = Column(JSONB, nullable=False, default=dict)

    data_source = Column(String(50))
    fetched_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "metric_date", "sport",
                         name="_pdm_player_date_sport_uc"),
    )


class ProjectionSnapshot(Base):
    """
    Delta-compressed audit trail of projection changes (EMAC-077 EPIC-1).
    One row per (snapshot_date, sport). Only stores changed projections.
    """

    __tablename__ = "projection_snapshots"

    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)  # 'mlb' | 'cbb'

    # {player_id: {"old": {...}, "new": {...}, "delta_reason": "..."}}
    player_changes = Column(JSONB, nullable=False, default=dict)

    total_players = Column(Integer)
    significant_changes = Column(Integer)   # rows where |delta| > threshold
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


if __name__ == "__main__":
    init_db()
