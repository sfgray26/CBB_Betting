"""
FastAPI application for CBB Edge Analyzer
Includes REST API, scheduled jobs, and monitoring
"""

from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import os

from apscheduler.triggers.interval import IntervalTrigger

from backend.models import (
    get_db,
    Game,
    Prediction,
    BetLog,
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
    calculate_timeline,
    generate_daily_snapshot,
)
from backend.services.alerts import check_performance_alerts, persist_alerts, run_alert_check
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

# Scheduler instance
scheduler = BackgroundScheduler()


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

    # Daily performance snapshot + alert check at 4 AM
    scheduler.add_job(
        _daily_snapshot_job,
        CronTrigger(hour=4, minute=0, timezone=timezone),
        id="daily_snapshot",
        name="Daily Performance Snapshot",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started: nightly@%02d:00, outcomes every 2h, lines every 30min, snapshot@04:00 %s",
        nightly_hour, timezone,
    )
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down CBB Edge Analyzer")
    scheduler.shutdown()


app = FastAPI(
    title="CBB Edge Analyzer",
    description="College Basketball Betting Framework - Version 7",
    version="7.0",
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

def nightly_job():
    """Main nightly analysis job — runs at 3 AM ET by default."""
    logger.info("Starting nightly analysis job")
    try:
        results = run_nightly_analysis()
        logger.info("Nightly analysis complete: %s", results)
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


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "app": "CBB Edge Analyzer",
        "version": "7.0",
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
    user: str = Depends(verify_admin_api_key),
):
    """Manually trigger nightly analysis (admin only). Runs synchronously and returns results."""
    logger.info("Manual analysis triggered by %s", user)
    try:
        results = run_nightly_analysis()
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


@app.post("/admin/recalibrate")
async def manual_recalibration(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Manually trigger model recalibration (admin only)"""
    logger.info("Manual recalibration triggered by %s", user)
    return {"message": "Recalibration started (TODO: implement)"}


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
