"""
Admin router — all /admin/* routes and the public root endpoints / and /health.

Strangler-fig extraction from backend/main.py.
Do NOT import from other backend.routers modules here.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text, func, inspect, create_engine
from typing import Optional
import logging
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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


@router.post("/admin/sync-yahoo-ids")
async def manual_yahoo_id_sync(user: str = Depends(verify_admin_api_key)):
    """
    Manually trigger Yahoo ID sync job (admin only).

    Syncs yahoo_id/yahoo_key from Yahoo Fantasy API to player_id_mapping
    by matching normalized names against BDL player index. Returns
    match statistics and any errors encountered.

    Runs synchronously and returns results immediately.
    """
    from backend.main import _ingestion_orchestrator
    if _ingestion_orchestrator is None:
        raise HTTPException(status_code=503, detail="Ingestion orchestrator not initialized")

    logger.info("Manual Yahoo ID sync triggered by %s", user)
    try:
        result = await _ingestion_orchestrator._sync_yahoo_id_mapping()
        return {
            "triggered_by": user,
            "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "result": result,
        }
    except Exception as exc:
        logger.error("Manual Yahoo ID sync failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/admin/player-id-mapping/conflicts")
async def get_player_id_mapping_conflicts(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Check player_id_mapping for unique constraint violations.

    Returns:
    - bdl_id_conflicts: bdl_ids appearing in multiple rows
    - yahoo_key_conflicts: yahoo_keys appearing in multiple rows
    - sample_conflict_rows: detailed rows for a specific conflict
    - overall_counts: total counts for each column
    """
    from backend.models import PlayerIDMapping

    # Check for duplicate bdl_ids
    bdl_conflicts = db.execute(text("""
        SELECT bdl_id, COUNT(*) as cnt
        FROM player_id_mapping
        WHERE bdl_id IS NOT NULL
        GROUP BY bdl_id
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
        LIMIT 20
    """)).fetchall()

    bdl_conflict_list = [{"bdl_id": row[0], "count": row[1]} for row in bdl_conflicts]

    # Check for duplicate yahoo_keys
    yahoo_conflicts = db.execute(text("""
        SELECT yahoo_key, COUNT(*) as cnt
        FROM player_id_mapping
        WHERE yahoo_key IS NOT NULL
        GROUP BY yahoo_key
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
        LIMIT 20
    """)).fetchall()

    yahoo_conflict_list = [{"yahoo_key": row[0], "count": row[1]} for row in yahoo_conflicts]

    # Get sample conflict rows if bdl_id conflicts exist
    sample_rows = []
    if bdl_conflict_list:
        sample_bdl_id = bdl_conflict_list[0]["bdl_id"]
        sample_rows = db.execute(text("""
            SELECT id, yahoo_key, yahoo_id, bdl_id, full_name, source
            FROM player_id_mapping
            WHERE bdl_id = :bdl_id
            ORDER BY id
        """), {"bdl_id": sample_bdl_id}).fetchall()
        sample_rows = [
            {
                "id": row[0],
                "yahoo_key": row[1],
                "yahoo_id": row[2],
                "bdl_id": row[3],
                "full_name": row[4],
                "source": row[5],
            }
            for row in sample_rows
        ]

    # Overall counts
    counts = db.execute(text("""
        SELECT
            COUNT(*) as total,
            COUNT(yahoo_key) as with_yahoo_key,
            COUNT(bdl_id) as with_bdl_id,
            COUNT(yahoo_id) as with_yahoo_id
        FROM player_id_mapping
    """)).fetchone()

    overall = {
        "total": counts[0],
        "with_yahoo_key": counts[1],
        "with_bdl_id": counts[2],
        "with_yahoo_id": counts[3],
    }

    return {
        "bdl_id_conflicts": bdl_conflict_list,
        "yahoo_key_conflicts": yahoo_conflict_list,
        "sample_conflict_rows": sample_rows,
        "overall_counts": overall,
    }


@router.post("/admin/backfill-numeric-names")
async def backfill_numeric_player_names(
    limit: int = Query(100, ge=1, le=500),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Backfill numeric player names in player_projections table.

    Finds projections with numeric names (e.g., "695578") and resolves them
    using BDL player search by player_id.

    Returns statistics on resolved/failed/skipped players.
    """
    from backend.services.balldontlie import BallDontLieClient
    from backend.models import PlayerProjection, PlayerIDMapping

    logger.info("Numeric name backfill triggered by %s, limit=%d", user, limit)

    # Find numeric-name projections
    numeric_players = db.execute(text("""
        SELECT DISTINCT pp.id, pp.player_id, pp.player_name
        FROM player_projections pp
        WHERE pp.player_name ~ '^[0-9]+$'
        LIMIT :limit
    """), {"limit": limit}).fetchall()

    total = len(numeric_players)
    resolved = 0
    failed = 0
    skipped = 0

    if total == 0:
        return {
            "status": "complete",
            "total": 0,
            "resolved": 0,
            "failed": 0,
            "skipped": 0,
            "message": "No numeric names found",
        }

    bdl = BallDontLieClient()

    for pp_id, player_id, numeric_name in numeric_players:
        try:
            # BDL lookup by player_id
            bdl_player = bdl.get_mlb_player(player_id)

            if bdl_player and bdl_player.full_name:
                real_name = " ".join(bdl_player.full_name.strip().split())

                # Update projection
                db.execute(text("""
                    UPDATE player_projections
                    SET player_name = :name, updated_at = NOW()
                    WHERE id = :pp_id
                """), {"name": real_name, "pp_id": pp_id})

                # Upsert player_id_mapping
                db.execute(text("""
                    INSERT INTO player_id_mapping (yahoo_id, mlb_id, bdl_id, player_name)
                    VALUES (:yahoo_id, :mlb_id, :bdl_id, :name)
                    ON CONFLICT (yahoo_id) DO UPDATE SET
                        mlb_id = EXCLUDED.mlb_id,
                        bdl_id = EXCLUDED.bdl_id,
                        player_name = EXCLUDED.player_name
                """), {
                    "yahoo_id": player_id,
                    "mlb_id": getattr(bdl_player, "mlb_id", None),
                    "bdl_id": bdl_player.id,
                    "name": real_name,
                })

                resolved += 1
                logger.info("Resolved %s: %s -> %s", player_id, numeric_name, real_name)
            else:
                logger.warning("BDL lookup failed for %s", player_id)
                failed += 1
        except Exception as exc:
            logger.error("Failed to resolve %s: %s", player_id, exc)
            failed += 1

    db.commit()

    # Check remaining numeric names
    remaining = db.execute(text("""
        SELECT COUNT(*) FILTER (WHERE player_name ~ '^[0-9]+$') AS numeric_names
        FROM player_projections
    """)).fetchone()[0]

    return {
        "status": "complete",
        "total": total,
        "resolved": resolved,
        "failed": failed,
        "skipped": skipped,
        "remaining_numeric_names": remaining,
    }


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


@router.get("/admin/check-databases")
async def check_databases(user: str = Depends(verify_api_key)):
    """Check for multiple databases in PostgreSQL and locate migration tables."""
    from urllib.parse import urlparse

    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        return {"error": "DATABASE_URL not found"}

    parsed = urlparse(db_url)

    # Connect to postgres database to list all databases
    admin_url = f"postgresql+psycopg2://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/postgres"
    admin_engine = create_engine(admin_url)

    databases_info = []

    with admin_engine.connect() as conn:
        # List all databases
        result = conn.execute(text("SELECT datname FROM pg_database WHERE datistemplate = false ORDER BY datname"))
        databases = [row[0] for row in result]

        for db_name in databases:
            if db_name in ['postgres', 'template0', 'template1']:
                continue

            try:
                db_url_specific = f"postgresql+psycopg2://{parsed.username}:{parsed.password}@{parsed.hostname}:{parsed.port}/{db_name}"
                engine_db = create_engine(db_url_specific)

                with engine_db.connect() as conn_db:
                    # Get table count
                    result = conn_db.execute(text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public'"))
                    table_count = result.scalar()

                    # Check for migration tables
                    target_tables = ['position_eligibility', 'probable_pitchers', 'ingested_injuries', 'player_valuation_cache']
                    found_tables = []
                    table_details = []

                    for target in target_tables:
                        try:
                            result = conn_db.execute(text(f"SELECT COUNT(*) FROM {target}"))
                            count = result.scalar()
                            found_tables.append(target)
                            table_details.append({
                                'table': target,
                                'count': count,
                                'exists': True
                            })
                        except Exception:
                            table_details.append({
                                'table': target,
                                'count': 0,
                                'exists': False
                            })

                    databases_info.append({
                        'database': db_name,
                        'table_count': table_count,
                        'has_target_tables': len(found_tables) > 0,
                        'target_tables': table_details
                    })

            except Exception as e:
                databases_info.append({
                    'database': db_name,
                    'error': str(e)[:100]
                })

    return {
        'current_database': parsed.path[1:],
        'current_host': parsed.hostname,
        'databases': databases_info
    }


@router.get("/admin/probable-pitchers/status")
async def get_probable_pitchers_status(
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Return probable_pitchers table status: total rows, today/tomorrow counts, is_confirmed counts."""
    from datetime import date, timedelta

    result = {}

    # Total rows
    total = db.execute(text("SELECT COUNT(*) FROM probable_pitchers")).scalar()
    result["total_rows"] = total

    # Today and tomorrow
    today = date.today()
    tomorrow = today + timedelta(days=1)

    today_count = db.execute(
        text("SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :today"),
        {"today": today}
    ).scalar()
    tomorrow_count = db.execute(
        text("SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :tomorrow"),
        {"tomorrow": tomorrow}
    ).scalar()

    result["today"] = {
        "date": str(today),
        "total": today_count,
    }
    result["tomorrow"] = {
        "date": str(tomorrow),
        "total": tomorrow_count,
    }

    # is_confirmed counts
    confirmed_today = db.execute(
        text("SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :today AND is_confirmed = true"),
        {"today": today}
    ).scalar()
    confirmed_tomorrow = db.execute(
        text("SELECT COUNT(*) FROM probable_pitchers WHERE game_date = :tomorrow AND is_confirmed = true"),
        {"tomorrow": tomorrow}
    ).scalar()

    result["today"]["confirmed"] = confirmed_today
    result["tomorrow"]["confirmed"] = confirmed_tomorrow

    # Recent sample rows
    if total > 0:
        recent = db.execute(text("""
            SELECT game_date, team, pitcher_name, is_confirmed, opponent
            FROM probable_pitchers
            WHERE game_date >= :today
            ORDER BY game_date, team
            LIMIT 5
        """), {"today": today}).fetchall()
        result["sample_rows"] = [
            {
                "game_date": str(r[0]),
                "team": r[1],
                "pitcher_name": r[2],
                "is_confirmed": bool(r[3]),
                "opponent": r[4]
            }
            for r in recent
        ]

    return result


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
async def admin_refresh_pybaseball(year: int = 2026, user: str = Depends(verify_admin_api_key)):
    """Force-refresh pybaseball Statcast cache and invalidate in-memory statcast_loader cache."""
    from backend.fantasy_baseball.pybaseball_loader import fetch_all_statcast_leaderboards
    import backend.fantasy_baseball.statcast_loader as _sc
    fetch_all_statcast_leaderboards(year=year, force_refresh=True)
    _sc._batter_cache.clear()
    _sc._pitcher_cache.clear()
    _sc._loaded_at = 0.0
    return {"status": "ok", "year": year}


@router.post("/admin/orphan-link")
async def admin_orphan_link(user: str = Depends(verify_admin_api_key), db: Session = Depends(get_db)):
    """Execute orphan linking for position_eligibility records and return results."""
    from backend.fantasy_baseball.orphan_linker import link_orphaned_records

    # Count before
    before = db.execute(text('''
        SELECT COUNT(*) FROM position_eligibility pe
        LEFT JOIN player_id_mapping pim ON pe.bdl_player_id = pim.id
        WHERE pe.yahoo_player_key IS NOT NULL AND pe.bdl_player_id IS NULL
    ''')).scalar()

    # Execute linking
    result = link_orphaned_records(db, dry_run=False, verbose=False)

    # Sample linked records
    sample = db.execute(text('''
        SELECT pe.player_name, pim.full_name, pe.bdl_player_id
        FROM position_eligibility pe
        JOIN player_id_mapping pim ON pe.bdl_player_id = pim.id
        WHERE pe.bdl_player_id IS NOT NULL
        ORDER BY pe.id DESC
        LIMIT 5
    ''')).fetchall()

    return {
        "timestamp": datetime.now().isoformat(),
        "before_count": before,
        "linked_count": result["linked_count"],
        "remaining_count": result["remaining_count"],
        "success_rate": result["success_rate"],
        "sample_records": [
            {"pe_name": row.player_name, "pim_name": row.full_name, "bdl_id": row.bdl_player_id}
            for row in sample
        ]
    }


@router.post("/admin/migrate/v28")
async def run_migration_v28(user: str = Depends(verify_admin_api_key), db: Session = Depends(get_db)):
    """
    Run Layer 2 Gap Closure migration (v28).

    Creates:
    - weather_forecasts table
    - park_factors table (with default data)
    - deployment_version table
    - Adds constraint _pp_date_team_uc to probable_pitchers
    """
    from sqlalchemy import text
    import subprocess

    results = {"steps": []}

    # 1. Add constraint to probable_pitchers
    try:
        check_constraint = text("""
            SELECT constraint_name
            FROM information_schema.table_constraints
            WHERE table_name = 'probable_pitchers'
            AND constraint_name = '_pp_date_team_uc'
        """)
        result = db.execute(check_constraint).fetchone()

        if result and result[0]:
            results["steps"].append({"name": "constraint _pp_date_team_uc", "status": "already exists"})
        else:
            # Remove duplicates if any exist
            db.execute(text("""
                DELETE FROM probable_pitchers p1
                USING probable_pitchers p2
                WHERE p1.id < p2.id
                AND p1.game_date = p2.game_date
                AND p1.team = p2.team
            """))
            # Add constraint
            db.execute(text("""
                ALTER TABLE probable_pitchers
                ADD CONSTRAINT _pp_date_team_uc
                UNIQUE (game_date, team)
            """))
            db.commit()
            results["steps"].append({"name": "constraint _pp_date_team_uc", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "constraint _pp_date_team_uc", "status": f"error: {e}"})

    # 2. Create weather_forecasts table
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_forecasts (
                id SERIAL PRIMARY KEY,
                game_date DATE NOT NULL,
                park_name VARCHAR(100) NOT NULL,
                forecast_date DATE NOT NULL DEFAULT CURRENT_DATE,
                temperature_high FLOAT,
                temperature_low FLOAT,
                humidity INTEGER,
                wind_speed FLOAT,
                wind_direction VARCHAR(10),
                precipitation_probability INTEGER,
                conditions VARCHAR(100),
                fetched_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (game_date, park_name, forecast_date)
            )
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_weather_game_date
            ON weather_forecasts (game_date)
        """))
        db.commit()
        results["steps"].append({"name": "weather_forecasts table", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "weather_forecasts table", "status": f"error: {e}"})

    # 3. Create park_factors table
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS park_factors (
                id SERIAL PRIMARY KEY,
                park_name VARCHAR(100) NOT NULL UNIQUE,
                hr_factor FLOAT NOT NULL DEFAULT 1.0,
                run_factor FLOAT NOT NULL DEFAULT 1.0,
                hits_factor FLOAT NOT NULL DEFAULT 1.0,
                era_factor FLOAT NOT NULL DEFAULT 1.0,
                whip_factor FLOAT NOT NULL DEFAULT 1.0,
                data_source VARCHAR(50),
                season INTEGER,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        db.commit()
        results["steps"].append({"name": "park_factors table", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "park_factors table", "status": f"error: {e}"})

    # 4. Seed default park factors
    try:
        result = db.execute(text("SELECT COUNT(*) FROM park_factors")).fetchone()
        if result[0] == 0:
            default_factors = [
                ('Yankee Stadium', 1.02, 1.01, 1.00, 0.99, 1.00),
                ('Dodger Stadium', 0.95, 0.97, 0.98, 1.01, 1.00),
                ('Coors Field', 1.25, 1.15, 1.10, 1.10, 1.05),
                ('Fenway Park', 1.08, 1.05, 1.03, 1.02, 1.01),
                ('Wrigley Field', 1.05, 1.04, 1.03, 1.01, 1.01),
                ('Oracle Park', 0.92, 0.95, 0.96, 0.98, 0.99),
                ('Truist Park', 0.99, 1.00, 0.99, 1.00, 1.00),
                ('Petco Park', 0.94, 0.96, 0.97, 1.00, 0.99),
                ('Citizens Bank Park', 1.09, 1.06, 1.04, 1.02, 1.01),
                ('Great American Ball Park', 1.15, 1.08, 1.05, 1.03, 1.02),
                ('American Family Field', 1.05, 1.03, 1.02, 0.99, 1.00),
                ('PNC Park', 0.97, 0.98, 0.98, 0.99, 1.00),
                ('LoanDepot Park', 0.96, 0.97, 0.97, 1.01, 1.00),
                ('Citi Field', 0.95, 0.97, 0.97, 1.00, 1.00),
                ('Nationals Park', 1.00, 1.00, 1.00, 1.00, 1.00),
                ('Tropicana Field', 0.94, 0.96, 0.97, 1.00, 0.99),
                ('Busch Stadium', 1.00, 1.00, 1.00, 1.00, 1.00),
                ('Comerica Park', 1.02, 1.02, 1.01, 1.00, 1.00),
                ('Kauffman Stadium', 1.00, 1.00, 1.00, 1.00, 1.00),
                ('Target Field', 0.98, 0.99, 0.99, 1.01, 1.00),
                ('Globe Life Field', 0.98, 0.99, 0.99, 1.00, 1.00),
                ('Angel Stadium', 0.98, 0.99, 0.99, 1.00, 1.00),
                ('Oakland Coliseum', 0.97, 0.98, 0.98, 1.01, 1.00),
                ('Rogers Centre', 1.03, 1.02, 1.01, 1.00, 1.00),
                ('T-Mobile Park', 0.94, 0.96, 0.97, 1.01, 1.00),
                ('Progressive Field', 1.02, 1.02, 1.01, 1.00, 1.00),
                ('Guaranteed Rate Field', 1.07, 1.05, 1.04, 1.01, 1.01),
            ]
            for park in default_factors:
                db.execute(text("""
                    INSERT INTO park_factors
                    (park_name, hr_factor, run_factor, hits_factor, era_factor, whip_factor, data_source, season)
                    VALUES (:park_name, :hr, :run, :hits, :era, :whip, 'fangraphs', 2025)
                """), {
                    'park_name': park[0], 'hr': park[1], 'run': park[2], 'hits': park[3],
                    'era': park[4], 'whip': park[5]
                })
            db.commit()
            results["steps"].append({"name": "park_factors seed", "status": f"inserted {len(default_factors)} parks"})
        else:
            results["steps"].append({"name": "park_factors seed", "status": "already seeded"})
    except Exception as e:
        results["steps"].append({"name": "park_factors seed", "status": f"error: {e}"})

    # 5. Create deployment_version table
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS deployment_version (
                id SERIAL PRIMARY KEY,
                git_commit_sha VARCHAR(100) NOT NULL UNIQUE,
                git_commit_date VARCHAR(50),
                build_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                app_version VARCHAR(50) DEFAULT 'dev',
                deployed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        db.commit()
        results["steps"].append({"name": "deployment_version table", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "deployment_version table", "status": f"error: {e}"})

    # 6. Populate deployment_version
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        commit_timestamp = subprocess.check_output(['git', 'show', '-s', '--format=%ci', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        db.execute(text("""
            INSERT INTO deployment_version (git_commit_sha, git_commit_date, app_version)
            VALUES (:sha, :commit_date, 'dev')
            ON CONFLICT (git_commit_sha)
            DO UPDATE SET build_timestamp = CURRENT_TIMESTAMP
        """), {'sha': sha, 'commit_date': commit_timestamp})
        db.commit()
        results["steps"].append({"name": "deployment_version seed", "status": f"SHA {sha[:12]}"})
    except Exception as e:
        results["steps"].append({"name": "deployment_version seed", "status": f"error: {e}"})

    # Verify migration
    verification = {}
    try:
        for table in ['weather_forecasts', 'park_factors', 'deployment_version']:
            result = db.execute(text("""
                SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :tbl)
            """), {'tbl': table}).fetchone()
            verification[table] = "EXISTS" if result[0] else "MISSING"

        result = db.execute(text("""
            SELECT constraint_name FROM information_schema.table_constraints
            WHERE table_name = 'probable_pitchers' AND constraint_name = '_pp_date_team_uc'
        """)).fetchone()
        verification["constraint _pp_date_team_uc"] = "EXISTS" if result else "MISSING"

        park_count = db.execute(text("SELECT COUNT(*) FROM park_factors")).fetchone()
        verification["park_factors count"] = park_count[0]

    except Exception as e:
        verification["error"] = str(e)

    results["verification"] = verification
    return results


@router.post("/admin/migrate/v31")
async def run_migration_v31(user: str = Depends(verify_admin_api_key), db: Session = Depends(get_db)):
    """
    Run V31 Rolling Stats Expansion migration.

    Adds to player_rolling_stats:
    - w_runs (decay-weighted runs scored)
    - w_tb (decay-weighted total bases)
    - w_qs (decay-weighted quality starts)
    """
    from sqlalchemy import text

    results = {"steps": []}

    # Columns to add to player_rolling_stats
    columns = [
        ("w_runs", "V31 Decay-weighted runs scored over the rolling window. Source: mlb_player_stats.runs (BDL). Drives z_runs for the R (Runs) batting category."),
        ("w_tb", "V31 Decay-weighted total bases over the rolling window. Computed as singles + 2*doubles + 3*triples + 4*home_runs per game. Drives z_tb for the TB (Total Bases) batting category."),
        ("w_qs", "V31 Decay-weighted quality starts over the rolling window. A quality start is IP >= 6.0 AND ER <= 3. Drives z_qs for the QS (Quality Starts) pitching category."),
    ]

    for col_name, comment in columns:
        try:
            # Check if column exists
            check_col = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'player_rolling_stats'
                AND column_name = :col_name
            """)
            result = db.execute(check_col, {"col_name": col_name}).fetchone()

            if result and result[0]:
                results["steps"].append({col_name: "already exists"})
            else:
                # Add column
                db.execute(text(f"""
                    ALTER TABLE player_rolling_stats
                    ADD COLUMN {col_name} DOUBLE PRECISION
                """))
                db.commit()
                results["steps"].append({col_name: "created"})
        except Exception as e:
            results["steps"].append({col_name: f"error: {e}"})

    # Verify migration
    verification = {}
    for col_name, _ in columns:
        result = db.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'player_rolling_stats' AND column_name = :col_name
        """), {"col_name": col_name}).fetchone()
        verification[col_name] = "EXISTS" if result else "MISSING"

    results["verification"] = verification
    return results


@router.post("/admin/migrate/v32")
async def run_migration_v32(user: str = Depends(verify_admin_api_key), db: Session = Depends(get_db)):
    """
    Run V32 Z-Score Expansion migration.

    Adds to player_scores:
    - z_r (league Z of w_runs)
    - z_h (league Z of w_hits)
    - z_tb (league Z of w_tb)
    - z_k_b (league Z of w_strikeouts_bat, lower-is-better)
    - z_ops (league Z of w_ops)
    - z_k_p (league Z of w_strikeouts_pit)
    - z_qs (league Z of w_qs)
    """
    from sqlalchemy import text

    results = {"steps": []}

    # Columns to add to player_scores
    columns = [
        ("z_r", "V31 League Z-score of w_runs (decay-weighted runs scored). For the R (Runs) batting category."),
        ("z_h", "V31 League Z-score of w_hits (decay-weighted hits). For the H (Hits) batting category."),
        ("z_tb", "V31 League Z-score of w_tb (decay-weighted total bases). For the TB (Total Bases) batting category."),
        ("z_k_b", "V31 League Z-score of w_strikeouts_bat (decay-weighted batter K). For the K_B (Batting Strikeouts) category. Lower-is-better: Z is negated."),
        ("z_ops", "V31 League Z-score of w_ops (decay-weighted OBP + SLG). For the OPS (On-Base Plus Slugging) batting category."),
        ("z_k_p", "V31 League Z-score of w_strikeouts_pit (decay-weighted pitcher K). For the K_P (Pitching Strikeouts) category."),
        ("z_qs", "V31 League Z-score of w_qs (decay-weighted quality starts). For the QS (Quality Starts) pitching category. QS = IP>=6 AND ER<=3."),
    ]

    for col_name, comment in columns:
        try:
            # Check if column exists
            check_col = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'player_scores'
                AND column_name = :col_name
            """)
            result = db.execute(check_col, {"col_name": col_name}).fetchone()

            if result and result[0]:
                results["steps"].append({col_name: "already exists"})
            else:
                # Add column
                db.execute(text(f"""
                    ALTER TABLE player_scores
                    ADD COLUMN {col_name} DOUBLE PRECISION
                """))
                db.commit()
                results["steps"].append({col_name: "created"})
        except Exception as e:
            results["steps"].append({col_name: f"error: {e}"})

    # Verify migration
    verification = {}
    for col_name, _ in columns:
        result = db.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'player_scores' AND column_name = :col_name
        """), {"col_name": col_name}).fetchone()
        verification[col_name] = "EXISTS" if result else "MISSING"

    results["verification"] = verification
    return results
