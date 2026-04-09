"""
Admin router — all /admin/* routes and the public root endpoints / and /health.

Strangler-fig extraction from backend/main.py.
Do NOT import from other backend.routers modules here.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text, func, inspect
from typing import Optional
import logging
import os
from datetime import datetime, timedelta

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
from backend.services.analysis import run_nightly_analysis
from backend.services.bet_tracker import update_completed_games, capture_closing_lines
from backend.services.performance import (
    calculate_summary_stats,
    calculate_clv_analysis,
    calculate_calibration,
    calculate_model_accuracy,
    calculate_timeline,
    generate_daily_snapshot,
    calculate_financial_metrics,
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
from backend.utils.env_utils import get_float_env
from backend.schemas import (
    AnalysisTriggerResponse,
    OracleFlaggedResponse,
    OraclePredictionDetail,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================================
# HELPERS (shared with edge router)
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
    return get_float_env("STARTING_BANKROLL", "1000")


# ============================================================================
# PUBLIC ENDPOINTS (no auth required)
# ============================================================================

@router.get("/")
async def root():
    """Health check"""
    return {
        "app": "CBB Edge Analyzer",
        "version": "9.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health")
async def health_check():
    """
    Health check endpoint.

    DB check is performed inline so this returns 200 even when the DB is down
    (degraded status). Does NOT use Depends(get_db) so TestClient works without
    a live database.
    """
    health = {"status": "healthy", "database": "connected", "scheduler": "running"}

    try:
        db = SessionLocal()
        try:
            db.execute(text("SELECT 1"))
        finally:
            db.close()
    except Exception as e:
        logger.error("Health check database error: %s", e)
        health["status"] = "degraded"
        health["database"] = f"error: {str(e)}"

    # Lazy-import scheduler to avoid circular imports at module load time
    try:
        from backend.main import scheduler as _scheduler
        if not _scheduler.running:
            health["status"] = "degraded"
            health["scheduler"] = "stopped"
    except Exception:
        pass

    return health


# ============================================================================
# ADMIN ENDPOINTS
# ============================================================================

@router.post("/admin/run-analysis", response_model=AnalysisTriggerResponse)
async def trigger_analysis_manually(
    notify_discord: bool = False,
    user: str = Depends(verify_admin_api_key),
):
    """Manually trigger nightly analysis (admin only). Runs synchronously and returns results.

    Pass ?notify_discord=true to also fire a Discord notification after analysis.
    """
    logger.info("Manual analysis triggered by %s (discord=%s)", user, notify_discord)
    try:
        results, cache = await run_nightly_analysis()

        try:
            get_odds_monitor().set_reanalysis_cache(cache)
        except Exception as cache_exc:
            logger.warning("Failed to update OddsMonitor cache: %s", cache_exc)

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


@router.post("/admin/discord/test")
async def discord_test(user: str = Depends(verify_admin_api_key)):
    """Send a test Discord message to verify bot token and channel ID (admin only)."""
    from backend.services.discord_notifier import _bot_token, _channel_id, _post
    token = _bot_token()
    if not token:
        raise HTTPException(status_code=400, detail="DISCORD_BOT_TOKEN not configured")
    payload = {
        "embeds": [{
            "title": "CBB Edge - Discord Test",
            "description": "Bot connected successfully. Notifications are working.",
            "color": 0x2ECC71,
            "footer": {"text": f"Triggered by {user}"},
            "timestamp": datetime.utcnow().isoformat(),
        }]
    }
    ok = _post(payload)
    if ok:
        return {"status": "ok", "channel_id": _channel_id()}
    raise HTTPException(status_code=502, detail="Discord API call failed - check logs")


@router.post("/admin/discord/test-simple")
async def discord_test_simple(user: str = Depends(verify_admin_api_key)):
    """Test the simplified Discord notification system (admin only)."""
    from backend.services.discord_simple import send_test_message

    success = send_test_message()
    if success:
        return {"status": "ok", "message": "Discord test messages sent to all configured channels"}
    raise HTTPException(
        status_code=502,
        detail="Discord test failed - check DISCORD_BOT_TOKEN and channel IDs"
    )


@router.post("/admin/discord/send-todays-bets")
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
            Game.game_date > now_utc,
            Game.external_id.isnot(None),
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


@router.post("/admin/recalibrate")
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
        dry_run=true  - compute and return diagnostics without writing changes.
    """
    from backend.services.recalibration import run_recalibration

    logger.info("Recalibration triggered by %s (dry_run=%s)", user, dry_run)
    try:
        result = run_recalibration(db, changed_by=user, apply_changes=not dry_run)
        return result
    except Exception as exc:
        logger.error("Recalibration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/admin/recalibration/audit")
async def recalibration_audit(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Get recalibration audit data (admin only)."""
    settled_with_pred = (
        db.query(BetLog)
        .filter(BetLog.outcome.isnot(None))
        .filter(BetLog.prediction_id.isnot(None))
        .count()
    )

    current_ha = (
        db.query(ModelParameter)
        .filter(ModelParameter.parameter_name == 'home_advantage')
        .order_by(ModelParameter.effective_date.desc())
        .first()
    )

    current_sd = (
        db.query(ModelParameter)
        .filter(ModelParameter.parameter_name == 'sd_multiplier')
        .order_by(ModelParameter.effective_date.desc())
        .first()
    )

    ha_value = current_ha.parameter_value if current_ha else 3.09
    sd_value = current_sd.parameter_value if current_sd else 0.85

    baseline_ha = 3.09
    baseline_sd = 0.85

    ha_drift = abs(ha_value - baseline_ha) / baseline_ha * 100
    sd_drift = abs(sd_value - baseline_sd) / baseline_sd * 100

    last_recal = current_ha.effective_date if current_ha else None
    days_since = (datetime.utcnow() - last_recal).days if last_recal else None

    return {
        "settled_bets": settled_with_pred,
        "sufficient_data": settled_with_pred >= 30,
        "home_advantage": round(ha_value, 4),
        "sd_multiplier": round(sd_value, 4),
        "ha_drift_pct": round(ha_drift, 1),
        "sd_drift_pct": round(sd_drift, 1),
        "drift_alert": ha_drift > 15 or sd_drift > 15,
        "last_recalibration": last_recal.isoformat() if last_recal else None,
        "days_since_recalibration": days_since,
        "recommendations": {
            "needs_more_data": settled_with_pred < 30,
            "stale_recalibration": days_since > 7 if days_since else True,
            "parameter_drift": ha_drift > 15 or sd_drift > 15,
        }
    }


@router.get("/admin/debug/duplicate-bets")
async def debug_duplicate_bets(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Find BetLog entries where multiple paper trades exist for the same game on the same calendar day."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    bets = (
        db.query(BetLog)
        .join(Game)
        .options(joinedload(BetLog.game))
        .filter(
            BetLog.is_paper_trade.is_(True),
            BetLog.timestamp >= cutoff,
        )
        .order_by(BetLog.game_id, BetLog.timestamp)
        .all()
    )

    groups: dict = {}
    for b in bets:
        day_key = b.timestamp.date().isoformat() if b.timestamp else "unknown"
        key = (b.game_id, day_key)
        groups.setdefault(key, []).append(b)

    duplicates = []
    for (game_id, day), group in groups.items():
        if len(group) < 2:
            continue
        game = group[0].game
        duplicates.append({
            "game_id": game_id,
            "date": day,
            "matchup": f"{game.away_team} @ {game.home_team}" if game else "Unknown",
            "count": len(group),
            "bet_ids": [b.id for b in group],
            "picks": [b.pick for b in group],
            "outcomes": [b.outcome for b in group],
            "notes": [b.notes for b in group],
        })

    duplicates.sort(key=lambda x: x["date"], reverse=True)

    return {
        "duplicate_groups": duplicates,
        "total_duplicate_groups": len(duplicates),
        "total_extra_bets": sum(d["count"] - 1 for d in duplicates),
        "days_searched": days,
        "message": (
            f"Found {len(duplicates)} games with duplicate paper trade entries. "
            f"These inflate bet counts by {sum(d['count'] - 1 for d in duplicates)} extra rows."
            if duplicates else "No duplicate paper trades found."
        ),
    }


@router.get("/admin/debug/bets-last-24h")
async def debug_bets_last_24h(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Debug endpoint: Get all bets from last 24 hours."""
    since = datetime.utcnow() - timedelta(hours=24)

    predictions = (
        db.query(Prediction, Game)
        .join(Game, Prediction.game_id == Game.id)
        .filter(Game.game_date >= since)
        .all()
    )

    bets = [(p, g) for p, g in predictions if p.verdict.startswith("Bet")]

    return {
        "total_predictions": len(predictions),
        "bet_count": len(bets),
        "since": since.isoformat(),
        "bets": [
            {
                "game_id": g.id,
                "home_team": g.home_team,
                "away_team": g.away_team,
                "game_date": g.game_date.isoformat() if g.game_date else None,
                "verdict": p.verdict,
                "edge": p.edge_conservative,
                "units": p.recommended_units,
            }
            for p, g in bets
        ]
    }


@router.post("/admin/cleanup/duplicate-bets")
async def cleanup_duplicate_bets(
    dry_run: bool = Query(default=True, description="If true, only report what would be deleted without deleting"),
    days: int = Query(default=365, ge=1, le=730, description="How far back to look for duplicates"),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Delete duplicate paper trade BetLog entries."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    bets = (
        db.query(BetLog)
        .join(Game)
        .filter(
            BetLog.is_paper_trade.is_(True),
            BetLog.timestamp >= cutoff,
        )
        .order_by(BetLog.game_id, BetLog.timestamp)
        .all()
    )

    groups: dict = {}
    for b in bets:
        day_key = b.timestamp.date().isoformat() if b.timestamp else "unknown"
        key = (b.game_id, day_key)
        groups.setdefault(key, []).append(b)

    to_delete = []
    kept = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        sorted_group = sorted(group, key=lambda x: x.id)
        kept.append(sorted_group[0].id)
        to_delete.extend(sorted_group[1:])

    if not to_delete:
        return {
            "status": "ok",
            "dry_run": dry_run,
            "duplicates_found": 0,
            "deleted": 0,
            "message": "No duplicate paper trades found.",
        }

    deleted_info = [
        {
            "id": b.id,
            "game_id": b.game_id,
            "pick": b.pick,
            "timestamp": b.timestamp.isoformat() if b.timestamp else None,
            "outcome": b.outcome,
        }
        for b in to_delete
    ]

    deleted_count = 0
    if not dry_run:
        for b in to_delete:
            db.delete(b)
        db.commit()
        deleted_count = len(to_delete)
        logger.info(
            "Duplicate bet cleanup: deleted %d paper trade BetLog entries (kept %d)",
            deleted_count, len(kept),
        )

    return {
        "status": "ok",
        "dry_run": dry_run,
        "duplicates_found": len(to_delete),
        "deleted": deleted_count,
        "kept_ids": kept,
        "deleted_entries": deleted_info,
        "message": (
            f"{'Would delete' if dry_run else 'Deleted'} {len(to_delete)} duplicate paper trade "
            f"entries across {len([g for g in groups.values() if len(g) >= 2])} games."
        ),
    }


@router.post("/admin/force-update-outcomes")
async def force_update_outcomes(
    days_from: int = Query(default=2, ge=1, le=30, description="How many days back to fetch scores (max 30)"),
    user: str = Depends(verify_admin_api_key),
):
    """Manually trigger the outcome-update job (admin only). Use days_from>2 to settle historical bets."""
    logger.info("Manual outcome update triggered by %s (days_from=%d)", user, days_from)
    try:
        results = update_completed_games(days_from=days_from)
        return {"message": "Outcome update complete", **results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/admin/force-capture-lines")
async def force_capture_lines(user: str = Depends(verify_admin_api_key)):
    """Manually trigger the closing-line capture job (admin only)."""
    logger.info("Manual line capture triggered by %s", user)
    try:
        results = capture_closing_lines()
        return {"message": "Line capture complete", **results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete("/admin/bets/{bet_id}")
async def delete_bet_log(
    bet_id: int,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Delete a single BetLog entry by ID (admin only). Use to remove orphaned or bogus paper trades."""
    bet = db.query(BetLog).filter(BetLog.id == bet_id).first()
    if not bet:
        raise HTTPException(status_code=404, detail=f"BetLog {bet_id} not found")
    db.delete(bet)
    db.commit()
    logger.warning("Admin %s deleted BetLog #%d (%s, $%.2f)", user, bet_id, bet.pick or "?", bet.bet_size_dollars or 0)
    return {"deleted": True, "bet_id": bet_id, "pick": bet.pick, "dollars": bet.bet_size_dollars}


@router.delete("/admin/bets/orphaned/cleanup")
async def cleanup_orphaned_bets(
    dry_run: bool = Query(default=True),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Delete BetLog entries whose game_id no longer exists in the games table."""
    orphans = db.execute(text(
        "SELECT b.id, b.pick, b.bet_size_dollars, b.game_id "
        "FROM bet_logs b LEFT JOIN games g ON b.game_id = g.id "
        "WHERE g.id IS NULL"
    )).fetchall()
    if dry_run:
        return {"dry_run": True, "orphans_found": len(orphans),
                "orphans": [{"id": r[0], "pick": r[1], "dollars": r[2], "game_id": r[3]} for r in orphans]}
    ids = [r[0] for r in orphans]
    if ids:
        db.query(BetLog).filter(BetLog.id.in_(ids)).delete(synchronize_session=False)
        db.commit()
    logger.warning("Admin %s deleted %d orphaned BetLog entries", user, len(ids))
    return {"dry_run": False, "deleted": len(ids), "ids": ids}


@router.delete("/admin/games/{game_id}")
async def delete_game(
    game_id: int,
    force: bool = Query(default=False, description="Delete even if BetLogs exist"),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Delete a game and all its predictions/closing lines (admin only).
    Blocked if real BetLogs exist unless ?force=true is passed."""
    game = db.query(Game).filter(Game.id == game_id).first()
    if not game:
        raise HTTPException(status_code=404, detail=f"Game {game_id} not found")

    bet_logs = db.query(BetLog).filter(BetLog.game_id == game_id).all()
    if bet_logs and not force:
        raise HTTPException(
            status_code=409,
            detail=f"Game {game_id} has {len(bet_logs)} bet log(s) - use ?force=true to delete anyway"
        )

    bet_logs_deleted = 0
    if bet_logs and force:
        bet_logs_deleted = db.query(BetLog).filter(BetLog.game_id == game_id).delete()
        logger.warning("Admin %s force-deleting %d bet log(s) for game %d", user, bet_logs_deleted, game_id)

    predictions_deleted = db.query(Prediction).filter(Prediction.game_id == game_id).delete()
    closing_deleted = db.query(ClosingLine).filter(ClosingLine.game_id == game_id).delete()
    db.delete(game)
    db.commit()
    logger.info("Admin %s deleted game %d (%s @ %s) - %d predictions, %d closing lines, %d bet logs removed",
                user, game_id, game.away_team, game.home_team, predictions_deleted, closing_deleted, bet_logs_deleted)
    return {
        "deleted": True,
        "game_id": game_id,
        "matchup": f"{game.away_team} @ {game.home_team}",
        "predictions_deleted": predictions_deleted,
        "closing_lines_deleted": closing_deleted,
        "bet_logs_deleted": bet_logs_deleted,
    }


@router.post("/admin/alerts/{alert_id}/acknowledge")
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


@router.get("/admin/scheduler/status")
async def get_scheduler_status(user: str = Depends(verify_admin_api_key)):
    """Get scheduler job status"""
    from backend.main import scheduler as _scheduler
    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })
    return {
        "running": _scheduler.running,
        "jobs": jobs,
    }


@router.get("/admin/ingestion/status")
async def ingestion_status(user: str = Depends(verify_api_key)):
    """Return per-job status for the DailyIngestionOrchestrator, or disabled signal."""
    from backend.main import _ingestion_orchestrator
    if _ingestion_orchestrator is None:
        return {"enabled": False, "jobs": {}}
    return {"enabled": True, "jobs": _ingestion_orchestrator.get_status()}


@router.get("/admin/portfolio/status")
async def get_portfolio_status(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return current portfolio state: exposure, drawdown, pending positions."""
    pm = get_portfolio_manager()
    pm.load_from_db(db)
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


@router.get("/admin/audit-tables")
async def audit_tables(user: str = Depends(verify_api_key)):
    """Audit all database tables - returns row counts for each table."""
    from sqlalchemy import inspect
    from backend.models import engine

    inspector = inspect(engine)
    tables = sorted(inspector.get_table_names())

    results = []
    db = SessionLocal()
    try:
        for table in tables:
            try:
                result = db.execute(text(f'SELECT COUNT(*) FROM "{table}"'))
                count = result.scalar()
                results.append({
                    'table': table,
                    'count': count,
                    'empty': count == 0
                })
            except Exception as e:
                results.append({
                    'table': table,
                    'count': 0,
                    'empty': False,
                    'error': str(e)[:100]
                })
    finally:
        db.close()

    # Categorize results
    empty = [r for r in results if r.get('empty') and not r.get('error')]
    populated = [r for r in results if not r.get('empty')]
    errors = [r for r in results if r.get('error')]

    return {
        'summary': {
            'total_tables': len(tables),
            'empty_tables': len(empty),
            'populated_tables': len(populated),
            'errors': len(errors)
        },
        'empty_tables': empty,
        'populated_tables': populated,
        'errors': errors
    }


@router.get("/admin/odds-monitor/status")
async def get_odds_monitor_status(user: str = Depends(verify_api_key)):
    """Return odds monitor status: tracked games, last poll time."""
    monitor = get_odds_monitor()
    return monitor.get_status()


@router.get("/admin/oracle/flagged", response_model=OracleFlaggedResponse)
async def get_oracle_flagged(
    days_back: int = Query(7, ge=1, le=90, description="Look-back window in days"),
    run_tier: Optional[str] = Query(None, description="Filter by run tier: opener|nightly|closing"),
    db: Session = Depends(get_db),
    user: str = Depends(verify_admin_api_key),
):
    """Return all predictions where the model diverged significantly from the KenPom + BartTorvik consensus."""
    from datetime import date, timedelta

    cutoff = date.today() - timedelta(days=days_back)
    query = (
        db.query(Prediction, Game)
        .join(Game, Prediction.game_id == Game.id)
        .filter(
            Prediction.oracle_flag.is_(True),
            Prediction.prediction_date >= cutoff,
        )
    )
    if run_tier:
        query = query.filter(Prediction.run_tier == run_tier)

    rows = query.order_by(Prediction.prediction_date.desc()).all()

    details = []
    for pred, game in rows:
        oracle = pred.oracle_result or {}
        details.append(
            OraclePredictionDetail(
                prediction_id=pred.id,
                game_date=game.game_date,
                home_team=game.home_team,
                away_team=game.away_team,
                verdict=pred.verdict,
                projected_margin=pred.projected_margin,
                oracle_spread=oracle.get("oracle_spread"),
                divergence_points=oracle.get("divergence_points"),
                divergence_z=oracle.get("divergence_z"),
                threshold_z=oracle.get("threshold_z"),
                sources=oracle.get("sources", []),
                run_tier=pred.run_tier,
                prediction_date=pred.prediction_date,
            )
        )

    return OracleFlaggedResponse(flagged_count=len(details), predictions=details)


@router.get("/admin/ratings/status")
async def get_ratings_status(user: str = Depends(verify_admin_api_key)):
    """Return live rating source coverage."""
    service = get_ratings_service()
    ratings = service.get_all_ratings(use_cache=True)

    kenpom_teams      = len(ratings.get("kenpom", {}))
    barttorvik_teams  = len(ratings.get("barttorvik", {}))
    evanmiya_teams    = len(ratings.get("evanmiya", {}))
    meta              = ratings.get("_meta", {})
    evanmiya_dropped  = meta.get("evanmiya_dropped", False)
    kenpom_ff_teams   = meta.get("kenpom_ff_teams", 0)

    active_sources = [
        s for s, n in [
            ("kenpom", kenpom_teams),
            ("barttorvik", barttorvik_teams),
            ("evanmiya", evanmiya_teams if not evanmiya_dropped else 0),
        ] if n > 0
    ]

    return {
        "sources": {
            "kenpom":     {"teams": kenpom_teams, "status": "UP" if kenpom_teams > 0 else "DOWN"},
            "barttorvik": {"teams": barttorvik_teams, "status": "UP" if barttorvik_teams > 0 else "DOWN"},
            "evanmiya":   {
                "teams": evanmiya_teams,
                "status": "DROPPED" if evanmiya_dropped else ("UP" if evanmiya_teams > 0 else "DOWN"),
            },
            "kenpom_four_factors": {"teams": kenpom_ff_teams},
        },
        "active_count": len(active_sources),
        "active_sources": active_sources,
        "model_health": (
            "CRITICAL" if len(active_sources) < 2
            else "DEGRADED" if len(active_sources) < 3
            else "OK"
        ),
        "cache_age_hours": round(
            (
                (__import__("datetime").datetime.utcnow() - service.cache_timestamp).total_seconds() / 3600
                if service.cache_timestamp else 0
            ),
            2,
        ),
    }


# ============================================================================
# BANKROLL OVERRIDE
# ============================================================================

@router.get("/admin/bankroll")
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
        "env_starting_bankroll": get_float_env("STARTING_BANKROLL", "1000"),
        "last_set": row.effective_date.isoformat() if row else None,
        "set_by": row.changed_by if row else None,
    }


@router.post("/admin/bankroll")
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

@router.get("/admin/parlay/override")
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


@router.post("/admin/parlay/override")
async def set_parlay_override(
    active: bool = Query(..., description="True to force parlay sizing regardless of capacity"),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Toggle force-parlay sizing (admin only)."""
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
# FEATURE FLAGS (admin side)
# ============================================================================

_ALLOWED_FLAGS = {"draft_board_enabled"}


@router.post("/admin/feature-flags/{flag_name}")
async def set_feature_flag(
    flag_name: str,
    enabled: bool,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Toggle a feature flag (admin only)."""
    if flag_name not in _ALLOWED_FLAGS:
        raise HTTPException(status_code=400, detail=f"Unknown flag: {flag_name}. Allowed: {sorted(_ALLOWED_FLAGS)}")
    db.add(ModelParameter(
        parameter_name=flag_name,
        parameter_value_json=enabled,
        reason="admin_toggle",
        changed_by=user,
    ))
    db.commit()
    logger.info("Feature flag %s set to %s by %s", flag_name, enabled, user)
    return {"flag": flag_name, "enabled": enabled}


# ============================================================================
# DRAFTKINGS CSV IMPORT
# ============================================================================

@router.post("/admin/dk/preview")
async def dk_import_preview(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Parse a DraftKings CSV and return proposed BetLog matches for review.

    Request body: {"csv_content": "<raw csv text>"}
    No database writes occur - call /admin/dk/confirm to apply.
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


@router.post("/admin/dk/confirm")
async def dk_import_confirm(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Apply confirmed DraftKings import matches to the database."""
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


@router.post("/admin/dk/direct-preview")
async def dk_direct_import_preview(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Preview DraftKings wagers for direct creation as real BetLog entries."""
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


@router.post("/admin/dk/direct-confirm")
async def dk_direct_import_confirm(
    payload: dict,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Create real BetLog entries from confirmed DK direct-import items."""
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
# YAHOO ADMIN ENDPOINTS
# ============================================================================

@router.get("/admin/yahoo/test")
async def yahoo_test(user: str = Depends(verify_admin_api_key)):
    """Test Yahoo API connectivity."""
    from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client, YahooAuthError, YahooAPIError
    try:
        client = get_yahoo_client()
        league = client.get_league()
        team_key = client.get_my_team_key()
        return {
            "status": "ok",
            "league_name": league.get("name"),
            "league_key": client.league_key,
            "my_team_key": team_key,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/admin/yahoo/roster-raw")
async def yahoo_roster_raw(user: str = Depends(verify_admin_api_key)):
    """Return the raw fantasy_content structure from Yahoo for your roster."""
    from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
    try:
        client = get_yahoo_client()
        raw = client.get_roster_raw()
        return {"fantasy_content": raw}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/admin/yahoo/roster")
async def yahoo_roster(user: str = Depends(verify_admin_api_key)):
    """Return your parsed Yahoo Fantasy roster."""
    from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
    try:
        client = get_yahoo_client()
        roster = client.get_roster()
        return {"count": len(roster), "players": roster}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# FANTASY ADMIN ENDPOINTS
# ============================================================================

@router.post("/admin/fantasy/reload-board", dependencies=[Depends(verify_admin_api_key)])
async def admin_reload_fantasy_board():
    """Force a fresh read of projection CSVs (data/projections/*.csv)."""
    from backend.fantasy_baseball.projections_loader import load_full_board
    from backend.fantasy_baseball import player_board

    load_full_board.cache_clear()
    player_board._BOARD = None

    board = player_board.get_board()
    return {"status": "ok", "players_loaded": len(board) if board else 0}


@router.post("/admin/pybaseball/refresh")
async def admin_refresh_pybaseball(year: int = 2025, user: str = Depends(verify_admin_api_key)):
    """Force-refresh pybaseball Statcast cache and invalidate in-memory statcast_loader cache."""
    from backend.fantasy_baseball.pybaseball_loader import fetch_all_statcast_leaderboards
    import backend.fantasy_baseball.statcast_loader as _sc
    fetch_all_statcast_leaderboards(year=year, force_refresh=True)
    _sc._batter_cache.clear()
    _sc._pitcher_cache.clear()
    _sc._loaded_at = 0.0
    return {"status": "ok", "year": year}
