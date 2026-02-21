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
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, date
import os
from dotenv import load_dotenv  # Add this import

# 1. Force load the .env file BEFORE defining DATABASE_URL
load_dotenv()

# 2. Get the URL from environment, or use the explicit postgres user as fallback
# This prevents it from using your OS username 'sfgra'
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:5432/cbb_edge")

# 3. pool_pre_ping=True helps keep the connection alive with the Docker container
engine = create_engine(DATABASE_URL, pool_pre_ping=True, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency for FastAPI
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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
    
    # Verdict
    verdict = Column(String, nullable=False, index=True)  # "PASS" or "Bet X units..."
    pass_reason = Column(String)  # If PASS, why?
    
    # Full analysis (for debugging)
    full_analysis = Column(JSON)
    
    # Data quality
    data_freshness_tier = Column(String)  # Tier 1/2/3
    penalties_applied = Column(JSON)  # Dict of penalty types & values
    
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


# Create all tables
def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


if __name__ == "__main__":
    init_db()
