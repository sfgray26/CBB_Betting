"""
FastAPI application for CBB Edge Analyzer
Includes REST API, scheduled jobs, and monitoring
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text, func
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import os
from zoneinfo import ZoneInfo

from apscheduler.triggers.interval import IntervalTrigger

from backend.models import (
    get_db,
    Game,
    Prediction,
    BetLog,
    ClosingLine,
    PerformanceSnapshot,
    ModelParameter,
    DBAlert,
    SessionLocal,
)
from backend.auth import verify_api_key, verify_admin_api_key
from backend.betting_model import CBBEdgeModel
from backend.services.analysis import run_nightly_analysis
from backend.services.clv import calculate_clv_full
from backend.services.bet_tracker import update_completed_games, capture_closing_lines
from backend.services.performance import (
    calculate_summary_stats,
    calculate_clv_analysis,
    calculate_calibration,
    calculate_model_accuracy,
    calculate_timeline,
    generate_daily_snapshot,
)
from backend.services.alerts import check_performance_alerts, persist_alerts, run_alert_check
from backend.services.discord_notifier import send_todays_bets
from backend.services.dk_import import (
    parse_dk_csv, preview_import, apply_import,
    preview_direct_import, apply_direct_import,
)
from backend.services.odds_monitor import get_odds_monitor
from backend.services.portfolio import get_portfolio_manager
from backend.services.ratings import get_ratings_service
from backend.schemas import (
    BetLogCreate,
    BetLogResponse,
    OutcomeUpdate,
    OutcomeResponse,
    AnalysisTriggerResponse,
    TodaysPredictionsResponse,
    PredictionResponse,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Scheduler instance — AsyncIOScheduler runs jobs inside FastAPI's event loop,
# allowing async job handlers (nightly_job, _opener_attack_job) to await coroutines.
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting CBB Edge Analyzer")
    
    # Start scheduler
    nightly_hour = int(os.getenv("NIGHTLY_CRON_HOUR", "3"))
    timezone = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")
    
    scheduler.add_job(
        nightly_job,
        CronTrigger(hour=nightly_hour, minute=0, timezone=timezone),
        id="nightly_analysis",
        name="Nightly Game Analysis",
        replace_existing=True,
    )

    # Settle outcomes every 2 hours
    scheduler.add_job(
        _update_outcomes_job,
        IntervalTrigger(hours=2),
        id="update_outcomes",
        name="Update Completed Game Outcomes",
        replace_existing=True,
    )

    # Capture closing lines every 30 minutes
    scheduler.add_job(
        _capture_lines_job,
        IntervalTrigger(minutes=30),
        id="capture_closing_lines",
        name="Capture Closing Lines",
        replace_existing=True,
    )

    # Daily performance snapshot + alert check at 4:30 AM (after settle job)
    scheduler.add_job(
        _daily_snapshot_job,
        CronTrigger(hour=4, minute=30, timezone=timezone),
        id="daily_snapshot",
        name="Daily Performance Snapshot",
        replace_existing=True,
    )

    # Settle outcomes once daily at 4 AM (in addition to every-2h interval job)
    scheduler.add_job(
        _update_outcomes_job,
        CronTrigger(hour=4, minute=0, timezone=timezone),
        id="settle_games_daily",
        name="Daily Settle Completed Games",
        replace_existing=True,
    )

    # Pre-warm ratings cache at 8 AM so nightly analysis uses fresh data
    ratings_prewarm_hour = int(os.getenv("RATINGS_PREWARM_HOUR", "8"))
    scheduler.add_job(
        _fetch_ratings_job,
        CronTrigger(hour=ratings_prewarm_hour, minute=0, timezone=timezone),
        id="fetch_ratings",
        name="Pre-warm Ratings Cache",
        replace_existing=True,
    )

    # Odds monitor — poll every 5 minutes for line movements
    odds_monitor_interval = int(os.getenv("ODDS_MONITOR_INTERVAL_MIN", "5"))
    scheduler.add_job(
        _odds_monitor_job,
        IntervalTrigger(minutes=odds_monitor_interval),
        id="odds_monitor",
        name="Odds Line Movement Monitor",
        replace_existing=True,
    )

    # Opening line attack — run when overnight lines are posted.
    # Books typically hang openers between 10 PM and midnight ET.
    # We run analysis at 10:30 PM and 12:30 AM to catch early value.
    opener_enabled = os.getenv("OPENER_ATTACK_ENABLED", "false").lower() == "true"
    if opener_enabled:
        scheduler.add_job(
            _opener_attack_job,
            CronTrigger(hour=22, minute=30, timezone=timezone),
            id="opener_attack_2230",
            name="Opening Line Attack (10:30 PM)",
            replace_existing=True,
        )
        scheduler.add_job(
            _opener_attack_job,
            CronTrigger(hour=0, minute=30, timezone=timezone),
            id="opener_attack_0030",
            name="Opening Line Attack (12:30 AM)",
            replace_existing=True,
        )
        logger.info("Opening line attack scheduler enabled (22:30, 00:30 %s)", timezone)

    scheduler.start()
    logger.info(
        "Scheduler started: nightly@%02d:00, outcomes every 2h + daily@04:00, "
        "lines every 30min, odds monitor every %dmin, snapshot@04:30, "
        "ratings prewarm@%02d:00 %s",
        nightly_hour, odds_monitor_interval, ratings_prewarm_hour, timezone,
    )
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down CBB Edge Analyzer")
    scheduler.shutdown()


app = FastAPI(
    title="CBB Edge Analyzer",
    description="College Basketball Betting Framework - Version 8",
    version="8.0",
    lifespan=lifespan,
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],  # Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# SCHEDULED JOB
# ============================================================================

async def nightly_job():
    """Main nightly analysis job — runs at 3 AM ET by default."""
    logger.info("Starting nightly analysis job")
    try:
        results = await run_nightly_analysis()
        logger.info("Nightly analysis complete: %s", results)
        try:
            send_todays_bets(results.get("bet_details"), results)
        except Exception as disc_exc:
            logger.warning("Discord notification failed: %s", disc_exc)
    except Exception as exc:
        logger.error("Nightly job failed: %s", exc, exc_info=True)


def _update_outcomes_job():
    """Settle completed game bets — runs every 2 hours."""
    try:
        results = update_completed_games()
        logger.info("Outcome update: %s", results)
    except Exception as exc:
        logger.error("Outcome update job failed: %s", exc, exc_info=True)


def _capture_lines_job():
    """Capture closing lines for imminent games — runs every 30 min."""
    try:
        results = capture_closing_lines()
        logger.info("Closing lines captured: %s", results)
    except Exception as exc:
        logger.error("Closing lines job failed: %s", exc, exc_info=True)


def _daily_snapshot_job():
    """Generate daily performance snapshot and run alert checks — runs at 4 AM."""
    db = SessionLocal()
    try:
        generate_daily_snapshot(db)
        run_alert_check()
    except Exception as exc:
        logger.error("Daily snapshot job failed: %s", exc, exc_info=True)
    finally:
        db.close()


def _odds_monitor_job():
    """Poll odds API for line movements — runs every 5 min (configurable).

    Only active during the configured operating window (default 12–23 ET)
    to avoid burning API quota when no games are scheduled.
    """
    _tz_name = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")
    _start_h = int(os.getenv("ODDS_MONITOR_START_HOUR", "12"))
    _end_h   = int(os.getenv("ODDS_MONITOR_END_HOUR",   "23"))
    try:
        _local_hour = datetime.now(ZoneInfo(_tz_name)).hour
    except Exception:
        _local_hour = datetime.utcnow().hour  # fallback if tzdata missing

    if not (_start_h <= _local_hour < _end_h):
        logger.debug(
            "Odds monitor: outside window [%d, %d) %s (current=%d) — skipping",
            _start_h, _end_h, _tz_name, _local_hour,
        )
        return

    try:
        monitor = get_odds_monitor()
        result = monitor.poll()
        if result.get("significant_movements", 0) > 0:
            logger.info(
                "Odds monitor: %d significant movements detected",
                result["significant_movements"],
            )
    except Exception as exc:
        logger.error("Odds monitor job failed: %s", exc, exc_info=True)


async def _fetch_ratings_job():
    """Pre-warm ratings cache at 8 AM — fetches all sources concurrently.

    Runs get_ratings_service().async_get_all_ratings(use_cache=False) so
    the nightly analysis (3 AM next day) hits a warm 6-hour cache.
    DB profile save is attempted but non-fatal on failure.
    """
    logger.info("Ratings pre-warm job starting")
    try:
        ratings_service = get_ratings_service()
        await ratings_service.async_get_all_ratings(use_cache=False)
        logger.info("Ratings pre-warm: cache refreshed successfully")

        # Also persist team profiles to DB (non-fatal)
        try:
            db = SessionLocal()
            ratings_service.save_team_profiles(db)
            db.commit()
        except Exception as db_exc:
            logger.warning("Ratings pre-warm: DB profile save failed: %s", db_exc)
        finally:
            try:
                db.close()
            except Exception:
                pass

    except Exception as exc:
        logger.warning("Ratings pre-warm job failed (non-fatal): %s", exc)


async def _opener_attack_job():
    """
    Run analysis when overnight opening lines are posted.

    Bookmakers hang openers with lower limits because their models are
    vulnerable — they rely on sharp action to shape the line.  Running
    analysis immediately catches early value before the line moves.
    """
    logger.info("Opening line attack triggered — running analysis on fresh openers")
    try:
        results = await run_nightly_analysis()
        bets = results.get("bets_recommended", 0)
        if bets > 0:
            logger.info(
                "Opener attack: %d bets found in %d games (%.1fs)",
                bets, results.get("games_analyzed", 0),
                results.get("duration_seconds", 0),
            )
        else:
            logger.info("Opener attack: no value found in current openers")
    except Exception as exc:
        logger.error("Opener attack job failed: %s", exc, exc_info=True)


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "app": "CBB Edge Analyzer",
        "version": "8.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    health = {"status": "healthy", "database": "connected", "scheduler": "running"}
    
    try:
        # CHANGE THIS LINE: Wrap the string in text()
        db.execute(text("SELECT 1")) 
    except Exception as e:
        logger.error(f"Health check database error: {e}")
        health["status"] = "degraded"
        health["database"] = f"error: {str(e)}"
    
    if not scheduler.running:
        health["status"] = "degraded"
        health["scheduler"] = "stopped"
        
    return health


# ============================================================================
# AUTHENTICATED ENDPOINTS - PREDICTIONS
# ============================================================================

@app.get("/api/predictions/today", response_model=TodaysPredictionsResponse)
async def get_todays_predictions(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all UPCOMING predictions from the latest analysis batch."""
    today_utc = datetime.utcnow().date()
    now_utc = datetime.utcnow()
    
    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)  # From the latest batch
        .filter(Game.game_date > now_utc)  # Only games that haven't started
        .order_by(Game.game_date.asc())  # Show chronological upcoming games
        .options(joinedload(Prediction.game))
        .all()
    )
    
    return TodaysPredictionsResponse(
        date=today_utc,
        total_games=len(predictions),
        bets_recommended=len([p for p in predictions if p.verdict.startswith("Bet")]),
        predictions=predictions
    )


@app.get("/api/predictions/bets")
async def get_recommended_bets(
    days: int = Query(default=7, ge=1, le=30),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all recommended bets from the last N days"""
    cutoff = datetime.utcnow() - timedelta(days=days)

    bets = (
        db.query(Prediction)
        .join(Game)
        .filter(
            Prediction.created_at >= cutoff,
            Prediction.verdict.like("Bet%")
        )
        .order_by(Prediction.created_at.desc())
        .all()
    )

    return {
        "period_days": days,
        "total_bets": len(bets),
        "bets": [
            {
                "game_id": b.game_id,
                "date": b.game.game_date.isoformat(),
                "matchup": f"{b.game.away_team} @ {b.game.home_team}",
                "verdict": b.verdict,
                "edge_point": b.edge_point,
                "edge_conservative": b.edge_conservative,
                "recommended_units": b.recommended_units,
            }
            for b in bets
        ]
    }


@app.get("/api/predictions/game/{game_id}")
async def get_game_prediction(
    game_id: int,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get the most recent prediction for a specific game.
    Returns the latest prediction regardless of run_tier.
    """
    prediction = (
        db.query(Prediction)
        .filter(Prediction.game_id == game_id)
        .order_by(Prediction.created_at.desc())
        .first()
    )

    if not prediction:
        return {"message": "No prediction found for this game", "game_id": game_id}

    # Parse verdict to extract bet details
    import re
    bet_details = {
        "has_bet": prediction.verdict.startswith("Bet"),
        "pick": None,
        "bet_type": None,
        "odds": None,
        "units": prediction.recommended_units,
    }

    if bet_details["has_bet"]:
        # Extract pick and odds from verdict
        # Example verdicts:
        # "Bet 1.2u Duke -4.5 @ -110 (edge: 3.2%, kelly: 2.4%)"
        # "Bet 0.8u UNC/Duke U145.5 @ -110 (edge: 2.8%, kelly: 1.6%)"
        match = re.search(r'Bet\s+[\d.]+u\s+([^@]+)\s+@\s+([-+]?\d+)', prediction.verdict)
        if match:
            pick_str = match.group(1).strip()
            odds_str = match.group(2).strip()

            bet_details["pick"] = pick_str
            bet_details["odds"] = int(odds_str)

            # Determine bet type from pick format
            if "/" in pick_str and ("U" in pick_str or "O" in pick_str):
                bet_details["bet_type"] = "total"
            elif "-" in pick_str or "+" in pick_str:
                bet_details["bet_type"] = "spread"
            else:
                bet_details["bet_type"] = "moneyline"
        else:
            # Fallback if regex doesn't match - provide safe defaults
            bet_details["pick"] = ""
            bet_details["bet_type"] = "spread"
            bet_details["odds"] = -110

    return {
        "game_id": game_id,
        "prediction_id": prediction.id,
        "verdict": prediction.verdict,
        "projected_margin": prediction.projected_margin,
        "point_prob": prediction.point_prob,
        "edge_point": prediction.edge_point,
        "edge_conservative": prediction.edge_conservative,
        "recommended_units": prediction.recommended_units,
        "bet_details": bet_details,
    }


@app.get("/api/predictions/parlays")
async def get_optimal_parlays(
    max_legs: int = Query(default=3, ge=2, le=4),
    max_parlays: int = Query(default=10, ge=1, le=50),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Build optimal cross-game parlays from today's +EV straight bets.

    Uses conservative probability estimates (lower CI) and applies
    a 4x Kelly divisor to respect parlay variance.

    Args:
        max_legs: Maximum number of legs per parlay (2-4)
        max_parlays: Maximum number of parlay tickets to return

    Returns:
        List of parlay tickets sorted by expected value
    """
    from backend.services.parlay_engine import build_optimal_parlays

    # ── Portfolio capacity ──────────────────────────────────────────────────
    # Parlay Kelly sizing must respect what straight bets have already consumed
    # from the daily exposure budget.  Query today's paper-trade BetLogs to
    # compute capital already allocated, then derive the true remaining dollars.
    starting_bankroll = float(os.getenv("STARTING_BANKROLL", "1000"))
    max_daily_pct     = float(os.getenv("MAX_DAILY_EXPOSURE_PCT", "5.0"))
    max_daily_dollars = starting_bankroll * max_daily_pct / 100.0

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    already_allocated: float = (
        db.query(func.sum(BetLog.bet_size_dollars))
        .filter(BetLog.timestamp >= today_start)   # BetLog uses 'timestamp', not 'created_at'
        .filter(BetLog.is_paper_trade.is_(True))
        .scalar()
        or 0.0
    )

    true_remaining_capacity = max(0.0, max_daily_dollars - already_allocated)

    if true_remaining_capacity <= 0.0:
        return {
            "date": datetime.utcnow().date().isoformat(),
            "message": "Portfolio capacity exhausted — no room for parlay sizing",
            "capital_allocated_dollars": round(already_allocated, 2),
            "max_daily_dollars": round(max_daily_dollars, 2),
            "parlays": [],
        }

    # ── Today's +EV straight bets ───────────────────────────────────────────
    today_utc = datetime.utcnow().date()
    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)
        .filter(Prediction.verdict.like("Bet%"))
        .options(joinedload(Prediction.game))
        .all()
    )

    if not predictions:
        return {
            "message": "No +EV bets available today for parlay construction",
            "parlays": [],
        }

    # Format predictions into slate_bets for parlay engine.
    # Derive a clean pick string from bet_side + spread stored in full_analysis.
    slate_bets = []
    for pred in predictions:
        game  = pred.game
        calcs = (pred.full_analysis or {}).get("calculations", {})
        bet_side = calcs.get("bet_side", "home")
        spread   = calcs.get("spread") or (
            (pred.full_analysis or {}).get("inputs", {}).get("odds", {}).get("spread")
        ) or 0.0
        if bet_side == "away":
            away_spread = -spread
            sign = "+" if away_spread > 0 else ""
            pick = f"{game.away_team} {sign}{away_spread:.1f}"
        else:
            sign = "+" if spread > 0 else ""
            pick = f"{game.home_team} {sign}{spread:.1f}"

        slate_bets.append({
            "game_id":           pred.game_id,
            "pick":              pick,
            "edge_conservative": pred.edge_conservative,
            "lower_ci_prob":     pred.lower_ci_prob,
            "full_analysis":     pred.full_analysis or {},
        })

    # ── Build parlays ────────────────────────────────────────────────────────
    parlays = build_optimal_parlays(
        slate_bets,
        max_legs=max_legs,
        max_parlays=max_parlays,
        remaining_capacity_dollars=true_remaining_capacity,
        bankroll=starting_bankroll,
    )

    return {
        "date":                        today_utc.isoformat(),
        "capital_allocated_dollars":   round(already_allocated, 2),
        "remaining_capacity_dollars":  round(true_remaining_capacity, 2),
        "max_daily_dollars":           round(max_daily_dollars, 2),
        "straight_bets_available":     len(slate_bets),
        "parlays_generated":           len(parlays),
        "max_legs":                    max_legs,
        "parlays":                     parlays,
    }


# ============================================================================
# AUTHENTICATED ENDPOINTS - PERFORMANCE
# ============================================================================

@app.get("/api/performance/summary")
async def get_performance_summary(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Full performance summary: overall metrics, by-type, by-edge-bucket,
    and rolling windows (last 10/50/100 bets).
    """
    return calculate_summary_stats(db)


@app.get("/api/performance/clv-analysis")
async def get_clv_analysis(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Detailed CLV breakdown: distribution, by-confidence, top/bottom 10."""
    return calculate_clv_analysis(db)


@app.get("/api/performance/calibration")
async def get_calibration_data(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Model calibration: predicted probability vs actual win rate + Brier score."""
    return calculate_calibration(db)


@app.get("/api/performance/model-accuracy")
async def get_model_accuracy(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Model accuracy metrics over resolved predictions.

    Returns margin MAE (mean absolute error between projected and actual margin),
    per-rating-source MAE, probability calibration bins, and Brier score.
    Only includes predictions where the game has completed and actual_margin
    has been populated by the automated outcome-settlement job.
    """
    return calculate_model_accuracy(db, days=days)


@app.get("/api/performance/timeline")
async def get_performance_timeline(
    days: int = Query(default=30, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Daily performance timeline with cumulative P&L and ROI series."""
    return calculate_timeline(db, days=days)


@app.get("/api/performance/alerts")
async def get_performance_alerts(
    include_acknowledged: bool = Query(default=False),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return active system health alerts from the database."""
    query = db.query(DBAlert).order_by(DBAlert.created_at.desc())
    if not include_acknowledged:
        query = query.filter(DBAlert.acknowledged == False)

    db_alerts = query.limit(50).all()

    # Also run a live check to return the most current state
    live_alerts = check_performance_alerts(db)

    overall_severity = "OK"
    for a in live_alerts:
        if a.severity == "CRITICAL":
            overall_severity = "CRITICAL"
            break
        if a.severity == "WARNING":
            overall_severity = "WARNING"

    return {
        "alerts": [
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "message": a.message,
                "threshold": a.threshold,
                "current_value": a.current_value,
                "acknowledged": a.acknowledged,
                "created_at": a.created_at.isoformat(),
            }
            for a in db_alerts
        ],
        "live_alerts": [a.to_dict() for a in live_alerts],
        "status": overall_severity,
    }


# ============================================================================
# AUTHENTICATED ENDPOINTS - BET LOGS
# ============================================================================

@app.post("/api/bets/log", response_model=BetLogResponse)
async def log_bet(
    bet_data: BetLogCreate,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Manually log a paper trade or real bet."""
    bet = BetLog(
        game_id=bet_data.game_id,
        prediction_id=bet_data.prediction_id,
        pick=bet_data.pick,
        bet_type=bet_data.bet_type,
        odds_taken=bet_data.odds_taken,
        bankroll_at_bet=bet_data.bankroll_at_bet,
        kelly_full=bet_data.kelly_full,
        kelly_fractional=bet_data.kelly_fractional,
        bet_size_pct=bet_data.bet_size_pct,
        bet_size_units=bet_data.bet_size_units,
        bet_size_dollars=bet_data.bet_size_dollars,
        model_prob=bet_data.model_prob,
        lower_ci_prob=bet_data.lower_ci_prob,
        point_edge=bet_data.point_edge,
        conservative_edge=bet_data.conservative_edge,
        is_paper_trade=bet_data.is_paper_trade,
        is_backfill=bet_data.is_backfill,
        notes=bet_data.notes,
    )
    db.add(bet)
    db.commit()
    db.refresh(bet)

    logger.info("Bet logged: %s %.2fu by %s", bet.pick, bet.bet_size_units, user)

    return BetLogResponse(
        message="Bet logged successfully",
        bet_id=bet.id,
        pick=bet.pick,
        bet_size_units=bet.bet_size_units,
        is_paper_trade=bet.is_paper_trade,
    )


@app.put("/api/bets/{bet_id}/outcome", response_model=OutcomeResponse)
async def update_bet_outcome(
    bet_id: int,
    payload: OutcomeUpdate,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Settle a bet: record outcome, compute P&L, and calculate CLV.

    Provide closing_spread (preferred) or closing_odds for CLV tracking.
    """
    bet = db.query(BetLog).filter(BetLog.id == bet_id).first()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")

    bet.outcome = payload.outcome

    # ----------------------------------------------------------------
    # P&L calculation
    # ----------------------------------------------------------------
    if payload.outcome == 1:  # Win
        if bet.odds_taken > 0:
            profit = bet.bet_size_dollars * (bet.odds_taken / 100.0)
        else:
            profit = bet.bet_size_dollars * (100.0 / abs(bet.odds_taken))
        bet.profit_loss_dollars = round(profit, 2)
    else:  # Loss
        bet.profit_loss_dollars = round(-bet.bet_size_dollars, 2)

    # 1 unit = 1% of starting bankroll; derive units from dollar P&L
    if bet.bankroll_at_bet and bet.bankroll_at_bet > 0:
        unit_value = bet.bankroll_at_bet / 100.0
        bet.profit_loss_units = round(bet.profit_loss_dollars / unit_value, 4)
    else:
        bet.profit_loss_units = None

    # ----------------------------------------------------------------
    # CLV calculation  (requires at least closing_odds)
    # ----------------------------------------------------------------
    clv_grade: Optional[str] = None

    if payload.closing_odds is not None:
        try:
            # Extract opening spread from BetLog if available
            # (BetLog doesn't store opening spread directly, but pick contains it;
            #  fall back to None so juice-only CLV is used when no spread is known)
            opening_spread: Optional[float] = None
            if bet.prediction_id:
                pred = db.query(Prediction).filter(Prediction.id == bet.prediction_id).first()
                if pred and pred.full_analysis:
                    opening_spread = pred.full_analysis.get("inputs", {}).get("odds", {}).get("spread")

            base_sd = float(os.getenv("BASE_SD", "11.0"))

            clv = calculate_clv_full(
                opening_odds=bet.odds_taken,
                closing_odds=payload.closing_odds,
                opening_spread=opening_spread,
                closing_spread=payload.closing_spread,
                other_side_closing_odds=payload.closing_odds_other_side,
                base_sd=base_sd,
            )

            bet.closing_line = payload.closing_odds
            bet.clv_points = round(clv.clv_points, 3)
            bet.clv_prob = round(clv.clv_prob, 4)
            clv_grade = clv.grade()

            logger.info(
                "CLV for bet %d: %.3f pts / %.2f%% (%s)",
                bet_id,
                clv.clv_points,
                clv.clv_prob * 100,
                clv_grade,
            )
        except Exception as exc:
            logger.warning("CLV calculation failed for bet %d: %s", bet_id, exc)

    db.commit()

    logger.info(
        "Bet %d settled: %s, P&L $%.2f",
        bet_id,
        "WIN" if payload.outcome else "LOSS",
        bet.profit_loss_dollars,
    )

    return OutcomeResponse(
        message="Outcome updated",
        bet_id=bet.id,
        outcome=bet.outcome,
        profit_loss_dollars=bet.profit_loss_dollars,
        profit_loss_units=bet.profit_loss_units,
        clv_points=bet.clv_points,
        clv_prob=bet.clv_prob,
        clv_grade=clv_grade,
    )


# ============================================================================
# AUTHENTICATED ENDPOINTS - GAMES
# ============================================================================

@app.get("/api/games/recent")
async def get_recent_games(
    days_back: int = Query(default=7, ge=1, le=30),
    days_ahead: int = Query(default=3, ge=1, le=7),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return recent and upcoming games, used to populate the bet-entry game selector."""
    start = datetime.utcnow() - timedelta(days=days_back)
    end = datetime.utcnow() + timedelta(days=days_ahead)

    games = (
        db.query(Game)
        .filter(Game.game_date >= start, Game.game_date <= end)
        .order_by(Game.game_date.desc())
        .all()
    )

    return {
        "games": [
            {
                "id": g.id,
                "matchup": f"{g.away_team} @ {g.home_team}",
                "home_team": g.home_team,
                "away_team": g.away_team,
                "game_date": g.game_date.isoformat(),
                "completed": g.completed,
                "home_score": g.home_score,
                "away_score": g.away_score,
            }
            for g in games
        ]
    }


# ============================================================================
# AUTHENTICATED ENDPOINTS - BET LOG QUERIES
# ============================================================================

@app.get("/api/bets")
async def get_bet_logs(
    status: str = Query(default="all", description="all | pending | settled"),
    days: int = Query(default=60, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return bet logs with optional status filter and date window."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    query = (
        db.query(BetLog)
        .join(Game)
        .filter(BetLog.timestamp >= cutoff)
        .options(joinedload(BetLog.game))
    )

    if status == "pending":
        query = query.filter(BetLog.outcome.is_(None))
    elif status == "settled":
        query = query.filter(BetLog.outcome.isnot(None))

    bets = query.order_by(BetLog.timestamp.desc()).all()

    return {
        "total": len(bets),
        "bets": [
            {
                "id": b.id,
                "game_id": b.game_id,
                "matchup": f"{b.game.away_team} @ {b.game.home_team}",
                "game_date": b.game.game_date.isoformat(),
                "pick": b.pick,
                "bet_type": b.bet_type,
                "odds_taken": b.odds_taken,
                "bet_size_units": b.bet_size_units,
                "bet_size_dollars": b.bet_size_dollars,
                "model_prob": b.model_prob,
                "outcome": b.outcome,
                "profit_loss_dollars": b.profit_loss_dollars,
                "profit_loss_units": b.profit_loss_units,
                "clv_points": b.clv_points,
                "clv_prob": b.clv_prob,
                "is_paper_trade": b.is_paper_trade,
                "timestamp": b.timestamp.isoformat() if b.timestamp else None,
                "notes": b.notes,
            }
            for b in bets
        ],
    }


@app.get("/api/closing-lines/{game_id}")
async def get_closing_lines(
    game_id: int,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return the most recent closing line capture for a game."""
    cl = (
        db.query(ClosingLine)
        .filter(ClosingLine.game_id == game_id)
        .order_by(ClosingLine.captured_at.desc())
        .first()
    )
    if not cl:
        raise HTTPException(status_code=404, detail="No closing line found for this game")
    return {
        "game_id": cl.game_id,
        "captured_at": cl.captured_at.isoformat() if cl.captured_at else None,
        "spread": cl.spread,
        "spread_odds": cl.spread_odds,
        "total": cl.total,
        "total_odds": cl.total_odds,
        "moneyline_home": cl.moneyline_home,
        "moneyline_away": cl.moneyline_away,
    }


@app.get("/api/performance/history")
async def get_performance_history(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return time-series data for cumulative P&L and rolling win-rate charts."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    bets = (
        db.query(BetLog)
        .join(Game)
        .filter(
            BetLog.outcome.isnot(None),
            BetLog.timestamp >= cutoff,
        )
        .options(joinedload(BetLog.game))
        .order_by(Game.game_date.asc())
        .all()
    )

    if not bets:
        return {"data_points": []}

    cumulative_pl = 0.0
    cumulative_units = 0.0
    wins = 0

    data_points = []
    for i, b in enumerate(bets, start=1):
        if b.outcome == 1:
            wins += 1
        cumulative_pl += b.profit_loss_dollars or 0.0
        cumulative_units += b.profit_loss_units or 0.0

        data_points.append(
            {
                "bet_number": i,
                "date": b.game.game_date.date().isoformat(),
                "bet_id": b.id,
                "pick": b.pick,
                "outcome": b.outcome,
                "pl_dollars": b.profit_loss_dollars,
                "cumulative_pl_dollars": round(cumulative_pl, 2),
                "cumulative_pl_units": round(cumulative_units, 4),
                "win_rate": round(wins / i, 4),
                "clv_prob": b.clv_prob,
            }
        )

    return {"data_points": data_points}


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@app.post("/admin/run-analysis", response_model=AnalysisTriggerResponse)
async def trigger_analysis_manually(
    notify_discord: bool = False,
    user: str = Depends(verify_admin_api_key),
):
    """Manually trigger nightly analysis (admin only). Runs synchronously and returns results.

    Pass ?notify_discord=true to also fire a Discord notification after analysis.
    """
    logger.info("Manual analysis triggered by %s (discord=%s)", user, notify_discord)
    try:
        results = await run_nightly_analysis()
        if notify_discord:
            try:
                send_todays_bets(results.get("bet_details"), results)
            except Exception as disc_exc:
                logger.warning("Discord notification failed: %s", disc_exc)
        return AnalysisTriggerResponse(
            message="Analysis complete",
            status=results.get("status", "ok"),
            games_analyzed=results.get("games_analyzed", 0),
            bets_recommended=results.get("bets_recommended", 0),
            paper_trades_created=results.get("paper_trades_created", 0),
            errors=results.get("errors", []),
            duration_seconds=results.get("duration_seconds", 0.0),
        )
    except Exception as exc:
        logger.error("Manual analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/discord/test")
async def discord_test(user: str = Depends(verify_admin_api_key)):
    """Send a test Discord message to verify bot token and channel ID (admin only)."""
    from backend.services.discord_notifier import _bot_token, _channel_id, _post
    token = _bot_token()
    if not token:
        raise HTTPException(status_code=400, detail="DISCORD_BOT_TOKEN not configured")
    payload = {
        "embeds": [{
            "title": "CBB Edge — Discord Test",
            "description": "Bot connected successfully. Notifications are working.",
            "color": 0x2ECC71,
            "footer": {"text": f"Triggered by {user}"},
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    ok = _post(payload)
    if ok:
        return {"status": "ok", "channel_id": _channel_id()}
    raise HTTPException(status_code=502, detail="Discord API call failed — check logs")


@app.post("/admin/discord/send-todays-bets")
async def discord_send_todays_bets(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Send today's BET predictions to Discord from the database (admin only).
    Use this to push notifications without re-running the full analysis.
    """
    from backend.services.discord_notifier import send_todays_bets as _send, _channel_id

    today_utc = datetime.utcnow().date()
    now_utc = datetime.utcnow()

    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(
            Prediction.prediction_date == today_utc,
            Prediction.verdict.like("Bet%"),
            Game.game_date > now_utc,          # upcoming games only
            Game.external_id.isnot(None),      # skip orphan records with no Odds API ID
        )
        .options(joinedload(Prediction.game))
        .all()
    )

    all_today = (
        db.query(Prediction)
        .join(Game)
        .filter(
            Prediction.prediction_date == today_utc,
            Game.game_date > now_utc,
            Game.external_id.isnot(None),
        )
        .count()
    )

    # Deduplicate — keep the highest-edge prediction per game
    seen_games: set = set()
    bet_details = []
    for p in sorted(predictions, key=lambda x: x.edge_conservative or 0.0, reverse=True):
        if p.game_id in seen_games:
            continue
        seen_games.add(p.game_id)
        fa = p.full_analysis or {}
        calcs = fa.get("calculations", {})
        inputs = fa.get("inputs", {})
        odds = inputs.get("odds", {})
        bet_details.append({
            "home_team": p.game.home_team,
            "away_team": p.game.away_team,
            "spread": odds.get("spread"),
            "bet_side": calcs.get("bet_side", "home"),
            "edge_conservative": p.edge_conservative,
            "recommended_units": p.recommended_units,
            "bet_odds": calcs.get("bet_odds"),
            "kelly_fractional": p.kelly_fractional,
            "projected_margin": p.projected_margin,
            "verdict": p.verdict,
        })

    n_bets = len(predictions)
    n_considered = (
        db.query(Prediction)
        .join(Game)
        .filter(
            Prediction.prediction_date == today_utc,
            Prediction.verdict.like("CONSIDER%"),
            Game.game_date > now_utc,
            Game.external_id.isnot(None),
        )
        .count()
    )

    summary = {
        "games_analyzed": all_today,
        "bets_recommended": n_bets,
        "games_considered": n_considered,
        "duration_seconds": 0,
    }

    _send(bet_details, summary)
    return {
        "status": "ok",
        "bets_sent": len(bet_details),
        "channel_id": _channel_id(),
    }


@app.post("/admin/recalibrate")
async def manual_recalibration(
    dry_run: bool = False,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Manually trigger model parameter recalibration (admin only).

    Analyses settled bets vs. predictions and adjusts:
      - home_advantage  (corrects systematic home-team margin bias)
      - sd_multiplier   (corrects probability over/under-confidence)

    All changes are persisted to model_parameters and take effect on the
    next nightly analysis run.

    Query params:
        dry_run=true  — compute and return diagnostics without writing changes.
    """
    from backend.services.recalibration import run_recalibration

    logger.info(
        "Recalibration triggered by %s (dry_run=%s)", user, dry_run
    )
    try:
        result = run_recalibration(db, changed_by=user, apply_changes=not dry_run)
        return result
    except Exception as exc:
        logger.error("Recalibration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/force-update-outcomes")
async def force_update_outcomes(user: str = Depends(verify_admin_api_key)):
    """Manually trigger the outcome-update job (admin only)."""
    logger.info("Manual outcome update triggered by %s", user)
    try:
        results = update_completed_games()
        return {"message": "Outcome update complete", **results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/force-capture-lines")
async def force_capture_lines(user: str = Depends(verify_admin_api_key)):
    """Manually trigger the closing-line capture job (admin only)."""
    logger.info("Manual line capture triggered by %s", user)
    try:
        results = capture_closing_lines()
        return {"message": "Line capture complete", **results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: int,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Mark an alert as acknowledged (admin only)."""
    alert = db.query(DBAlert).filter(DBAlert.id == alert_id).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    alert.acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    db.commit()
    return {"message": "Alert acknowledged", "alert_id": alert_id}


@app.get("/admin/scheduler/status")
async def get_scheduler_status(user: str = Depends(verify_admin_api_key)):
    """Get scheduler job status"""
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return {
        "running": scheduler.running,
        "jobs": jobs,
    }


@app.get("/admin/portfolio/status")
async def get_portfolio_status(user: str = Depends(verify_admin_api_key)):
    """Return current portfolio state: exposure, drawdown, pending positions."""
    pm = get_portfolio_manager()
    state = pm.get_state()
    return {
        "current_bankroll": state.current_bankroll,
        "starting_bankroll": state.starting_bankroll,
        "drawdown_pct": round(state.drawdown_pct, 2),
        "total_exposure_pct": round(state.total_exposure_pct, 2),
        "is_halted": state.is_halted,
        "halt_reason": state.halt_reason,
        "pending_positions": len(state.positions),
    }


@app.get("/admin/odds-monitor/status")
async def get_odds_monitor_status(user: str = Depends(verify_admin_api_key)):
    """Return odds monitor status: tracked games, last poll time."""
    monitor = get_odds_monitor()
    return monitor.get_status()


# ============================================================================
# BANKROLL OVERRIDE
# ============================================================================

def _get_model_param(db: Session, name: str) -> Optional[ModelParameter]:
    return (
        db.query(ModelParameter)
        .filter(ModelParameter.parameter_name == name)
        .order_by(ModelParameter.effective_date.desc())
        .first()
    )


def get_effective_bankroll(db: Session) -> float:
    """Return the active bankroll: DB override if set, else STARTING_BANKROLL env var."""
    row = _get_model_param(db, "current_bankroll")
    if row and row.parameter_value and row.parameter_value > 0:
        return row.parameter_value
    return float(os.getenv("STARTING_BANKROLL", "1000"))


@app.get("/admin/bankroll")
async def get_bankroll(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Return current effective bankroll and its source."""
    row = _get_model_param(db, "current_bankroll")
    effective = get_effective_bankroll(db)
    return {
        "effective_bankroll": effective,
        "source": "db_override" if (row and row.parameter_value) else "env_var",
        "env_starting_bankroll": float(os.getenv("STARTING_BANKROLL", "1000")),
        "last_set": row.effective_date.isoformat() if row else None,
        "set_by": row.changed_by if row else None,
    }


@app.post("/admin/bankroll")
async def set_bankroll(
    amount: float = Query(..., gt=0, description="New bankroll in dollars"),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Override the effective bankroll used for Kelly sizing (admin only)."""
    db.add(ModelParameter(
        parameter_name="current_bankroll",
        parameter_value=round(amount, 2),
        reason="manual_override",
        changed_by=user,
    ))
    db.commit()
    logger.info("Bankroll overridden to $%.2f by %s", amount, user)
    return {"status": "ok", "bankroll_set": round(amount, 2)}


# ============================================================================
# PARLAY FORCE OVERRIDE
# ============================================================================

@app.get("/admin/parlay/override")
async def get_parlay_override(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Return current parlay force-sizing override status."""
    row = _get_model_param(db, "force_parlay_sizing")
    active = bool(row and row.parameter_value == 1.0)
    return {
        "force_parlay_sizing": active,
        "last_set": row.effective_date.isoformat() if row else None,
    }


@app.post("/admin/parlay/override")
async def set_parlay_override(
    active: bool = Query(..., description="True to force parlay sizing regardless of capacity"),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Toggle force-parlay sizing. When active, parlays are recommended even when
    the daily straight-bet budget is fully consumed (admin only)."""
    db.add(ModelParameter(
        parameter_name="force_parlay_sizing",
        parameter_value=1.0 if active else 0.0,
        reason="manual_override",
        changed_by=user,
    ))
    db.commit()
    logger.info("Force parlay sizing set to %s by %s", active, user)
    return {"status": "ok", "force_parlay_sizing": active}


# ============================================================================
# DRAFTKINGS CSV IMPORT
# ============================================================================

@app.post("/admin/dk/preview")
async def dk_import_preview(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Parse a DraftKings CSV and return proposed BetLog matches for review.

    Request body: {"csv_content": "<raw csv text>"}

    Returns a list of proposed matches with confidence scores.
    No database writes occur — call /admin/dk/confirm to apply.
    """
    csv_content = payload.get("csv_content", "")
    if not csv_content:
        raise HTTPException(status_code=400, detail="csv_content is required")

    data = parse_dk_csv(csv_content)
    matches = preview_import(db, data)

    return {
        "wagers_found": len(data.wagers),
        "payouts_found": len(data.payouts),
        "skipped_rows": data.skipped_rows,
        "matches": [
            {
                "bet_log_id": m.bet_log_id,
                "pick": m.pick,
                "bet_log_timestamp": m.bet_log_timestamp.isoformat(),
                "bet_log_dollars": m.bet_log_dollars,
                "dk_wager_id": m.dk_wager_id,
                "dk_wager_amount": m.dk_wager_amount,
                "dk_wager_timestamp": m.dk_wager_timestamp.isoformat(),
                "outcome": m.outcome,
                "profit_dollars": m.profit_dollars,
                "payout_amount": m.payout_amount,
                "confidence": m.confidence,
            }
            for m in matches
        ],
        "unmatched_wagers": len(data.wagers) - len(matches),
    }


@app.post("/admin/dk/confirm")
async def dk_import_confirm(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Apply confirmed DraftKings import matches to the database.

    Request body: {"matches": [...list from /admin/dk/preview...]}

    Each match in the list may have ``outcome`` overridden before confirming.
    Matches with outcome=null are skipped (left pending).
    """
    confirmed = payload.get("matches", [])
    if not confirmed:
        raise HTTPException(status_code=400, detail="matches list is required")

    summary = apply_import(db, confirmed)
    return {
        "status": "ok",
        "applied": summary.applied,
        "wins": summary.wins,
        "losses": summary.losses,
        "pending_skipped": summary.pending,
        "total_profit_dollars": round(summary.total_profit, 2),
        "errors": summary.errors,
    }


@app.post("/admin/dk/direct-preview")
async def dk_direct_import_preview(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Preview DraftKings wagers for direct creation as real BetLog entries.

    No paper trades required — creates brand-new BetLog rows for each wager,
    matched to games by calendar date.

    Request body: {"csv_content": "<raw csv text>"}
    """
    csv_content = payload.get("csv_content", "")
    if not csv_content:
        raise HTTPException(status_code=400, detail="csv_content is required")

    data = parse_dk_csv(csv_content)
    items = preview_direct_import(db, data)

    return {
        "wagers_found": len(data.wagers),
        "payouts_found": len(data.payouts),
        "items_with_game": sum(1 for i in items if i.suggested_game_id),
        "items_no_game": sum(1 for i in items if not i.suggested_game_id),
        "items": [
            {
                "dk_wager_id": i.dk_wager_id,
                "dk_amount": i.dk_amount,
                "dk_timestamp": i.dk_timestamp.isoformat(),
                "outcome": i.outcome,
                "profit_dollars": i.profit_dollars,
                "payout_amount": i.payout_amount,
                "candidate_games": i.candidate_games,
                "suggested_game_id": i.suggested_game_id,
                "suggested_game_label": getattr(i, "_suggested_game_label", ""),
            }
            for i in items
        ],
    }


@app.post("/admin/dk/direct-confirm")
async def dk_direct_import_confirm(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Create real BetLog entries from confirmed DK direct-import items.

    Request body: {"items": [...list from /admin/dk/direct-preview...]}
    Items with no game_id are skipped.
    """
    items = payload.get("items", [])
    if not items:
        raise HTTPException(status_code=400, detail="items list is required")

    summary = apply_direct_import(db, items)
    return {
        "status": "ok",
        "applied": summary.applied,
        "wins": summary.wins,
        "losses": summary.losses,
        "pending_added": summary.pending,
        "total_profit_dollars": round(summary.total_profit, 2),
        "errors": summary.errors,
    }


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
