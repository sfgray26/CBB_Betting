"""
FastAPI application for CBB Edge Analyzer
Includes REST API, scheduled jobs, and monitoring
"""

from fastapi import FastAPI, Depends, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import text, func
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from dataclasses import asdict
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
    IngestedInjury,
    SessionLocal,
)
from backend.auth import verify_api_key, verify_admin_api_key
from backend.betting_model import CBBEdgeModel
from backend.services.analysis import run_nightly_analysis
from backend.services.clv import calculate_clv_full
from backend.services.bet_tracker import update_completed_games, capture_closing_lines
from backend.services.line_monitor import check_line_movements
from backend.services.performance import (
    calculate_summary_stats,
    calculate_clv_analysis,
    calculate_calibration,
    calculate_model_accuracy,
    calculate_timeline,
    generate_daily_snapshot,
    calculate_financial_metrics,
)
from backend.services.alerts import check_performance_alerts, run_alert_check

# TEST ENDPOINTS - REMOVE AFTER SYNC JOB TESTING
from backend.test_sync_jobs import router as _test_router
# END TEST ENDPOINTS

# ERA DIAGNOSTIC ENDPOINT - REMOVE AFTER TASK 10 COMPLETE
from backend.admin_endpoints_era import router as _era_diagnostic_router
# END ERA DIAGNOSTIC ENDPOINT

# VALIDATION AUDIT ENDPOINT - REMOVE AFTER TASK 11 COMPLETE
from backend.admin_endpoints_validation import router as _validation_audit_router
# END VALIDATION AUDIT ENDPOINT

# OPS/WHIP BACKFILL ENDPOINT - REMOVE AFTER TASK 26 COMPLETE
from backend.admin_backfill_ops_whip import router as _backfill_ops_whip_router
# END OPS/WHIP BACKFILL ENDPOINT

# STATCAST DIAGNOSTICS ENDPOINTS - REMOVE AFTER STATCAST VALIDATION COMPLETE
from backend.admin_statcast_diagnostics import router as _statcast_diag_router
# END STATCAST DIAGNOSTICS ENDPOINTS

# SCORING DIAGNOSTICS ENDPOINTS - REMOVE AFTER NSB ROLLOUT VALIDATED
from backend.admin_scoring_diagnostics import router as _scoring_diag_router
# END SCORING DIAGNOSTICS ENDPOINTS

# CONSTRAINT MIGRATION ENDPOINT - REMOVE AFTER YAHOO ID SYNC IS LIVE
from backend.admin_add_constraint import router as _constraint_migration_router
# END CONSTRAINT MIGRATION ENDPOINT

# DATA QUALITY MONITORING DASHBOARD
from backend.routers import data_quality
# END DATA QUALITY MONITORING DASHBOARD

from backend.services.recalibration import compute_dynamic_weights
from backend.services.discord_notifier import send_todays_bets
from backend.services.sentinel import run_nightly_health_check
from backend.services.health_monitor import check_pipeline_health, CRITICAL_CHAIN
from backend.services.dk_import import (
    parse_dk_csv, preview_import, apply_import,
    preview_direct_import, apply_direct_import,
)
from backend.services.odds_monitor import get_odds_monitor
from backend.services.portfolio import get_portfolio_manager
from backend.services.ratings import get_ratings_service
from backend.services.job_queue_service import submit_job as jq_submit, get_job_status as jq_status, process_pending_jobs as jq_process
from backend.utils.env_utils import get_float_env
from backend.stat_contract import YAHOO_ID_INDEX, SCORING_CATEGORY_CODES
from backend.utils.time_utils import today_et
from backend.schemas import (
    BetLogCreate,
    BetLogResponse,
    OutcomeUpdate,
    OutcomeResponse,
    AnalysisTriggerResponse,
    TodaysPredictionsResponse,
    DailyLineupResponse,
    WaiverWireResponse,
    LineupPlayerOut,
    StartingPitcherOut,
    RosterPlayerOut,
    RosterResponse,
    MatchupTeamOut,
    MatchupResponse,
    LineupApplyRequest,
    OracleFlaggedResponse,
    OraclePredictionDetail,
)
from backend.fantasy_baseball.yahoo_client_resilient import (
    YahooAuthError,
    YahooAPIError,
    ResilientYahooClient,
    get_yahoo_client,
    get_resilient_yahoo_client,
)
from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_optimizer
from backend.fantasy_baseball.ballpark_factors import load_park_factors
from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Scheduler instance - AsyncIOScheduler runs jobs inside FastAPI's event loop,
# allowing async job handlers (nightly_job, _opener_attack_job) to await coroutines.
scheduler = AsyncIOScheduler()

# DailyIngestionOrchestrator -- instantiated in lifespan() when
# ENABLE_INGESTION_ORCHESTRATOR=true. Declared at module level so the
# /admin/ingestion/status endpoint can reference it without an import cycle.
_ingestion_orchestrator = None

# MLBAnalysisService -- instantiated in lifespan() when
# ENABLE_MLB_ANALYSIS=true. Kept at module level for future status endpoints.
_mlb_analysis_service = None

# MLB probable-starts cache: {"data": {...}, "fetched_at": datetime}
_STARTS_CACHE: dict = {}


def _get_projection_freshness_report() -> dict:
    """Return the latest projection freshness report from the ingestion orchestrator."""
    if _ingestion_orchestrator is None:
        return {
            "checked_at": None,
            "violations": ["projection_freshness unavailable: ingestion orchestrator is disabled"],
            "violation_count": 1,
        }

    jobs = _ingestion_orchestrator.get_status() or {}
    report = jobs.get("projection_freshness") or {}
    violations = list(report.get("violations") or [])

    if report.get("checked_at") is None:
        violations.append("projection_freshness unavailable: no freshness report has been recorded yet")

    return {
        **report,
        "violations": violations,
        "violation_count": len(violations),
    }


def _enforce_projection_freshness(consumer: str, force_stale: bool = False) -> list[str]:
    """Block stale fantasy decisions unless the caller explicitly overrides the guard."""
    report = _get_projection_freshness_report()
    violations = list(report.get("violations") or [])
    if not violations:
        return []

    message = f"Projection freshness gate triggered for {consumer}"
    detail = {
        "error": "projection_freshness_violation",
        "message": message,
        "consumer": consumer,
        "force_stale_available": True,
        "freshness": report,
    }

    if not force_stale:
        raise HTTPException(status_code=503, detail=detail)

    logger.warning("%s -- proceeding due to force_stale override: %s", message, violations)
    return [f"Stale-data override active: {'; '.join(violations)}"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting CBB Edge Analyzer")

    # Ensure roster_acquisitions table exists (model was added after initial deploy).
    # create_all() is not wired to lifespan, so we check surgically to avoid
    # recreating unrelated tables.
    try:
        from sqlalchemy import inspect as _sa_inspect
        from backend.models import engine as _db_engine, RosterAcquisition, DailyAvailabilityOverride
        _inspector = _sa_inspect(_db_engine)
        if "roster_acquisitions" not in _inspector.get_table_names():
            RosterAcquisition.__table__.create(_db_engine)
            logger.info("Lifespan: created roster_acquisitions table")
        else:
            logger.debug("Lifespan: roster_acquisitions table already exists")
    except Exception as _tbl_exc:
        logger.warning("Lifespan: roster_acquisitions table check failed: %s", _tbl_exc)

    # Ensure daily_availability_overrides table exists (P0 availability guard).
    # Self-contained try block: does not rely on _inspector from the block above
    # (if that block threw before assigning _inspector, this block would NameError
    # and silently skip table creation).
    try:
        from sqlalchemy import inspect as _sa_inspect2
        from backend.models import engine as _db_engine2, DailyAvailabilityOverride as _DAO
        _dao_inspector = _sa_inspect2(_db_engine2)
        if "daily_availability_overrides" not in _dao_inspector.get_table_names():
            _DAO.__table__.create(_db_engine2)
            logger.info("Lifespan: created daily_availability_overrides table")
        else:
            logger.debug("Lifespan: daily_availability_overrides table already exists")
    except Exception as _dao_exc:
        logger.warning("Lifespan: daily_availability_overrides table check failed: %s", _dao_exc)

    # Start scheduler
    nightly_hour = int(os.getenv("NIGHTLY_CRON_HOUR", "3"))
    timezone = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")

    # Guard flags (INFRA-A, INFRA-B)
    cbb_active = os.getenv("CBB_SEASON_ACTIVE", "false").lower() == "true"
    fantasy_active = os.getenv("ENABLE_FANTASY_SCHEDULER", "false").lower() == "true"

    if cbb_active:
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

    if cbb_active:
        # O-10: Line movement monitor - runs every 30 minutes
        scheduler.add_job(
            _line_monitor_job,
            IntervalTrigger(minutes=30),
            id="line_monitor",
            name="Line Movement Monitor",
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

    if cbb_active:
        # Pre-warm ratings cache at 8 AM so nightly analysis uses fresh data
        ratings_prewarm_hour = int(os.getenv("RATINGS_PREWARM_HOUR", "8"))
        scheduler.add_job(
            _fetch_ratings_job,
            CronTrigger(hour=ratings_prewarm_hour, minute=0, timezone=timezone),
            id="fetch_ratings",
            name="Pre-warm Ratings Cache",
            replace_existing=True,
        )

    if fantasy_active:
        scheduler.add_job(
            check_expired_eta,
            CronTrigger(hour=3, minute=0, timezone="America/New_York"),
            id="check_expired_eta",
            name="check_expired_eta",
            replace_existing=True,
        )

        # Refresh pybaseball FanGraphs leaderboard caches at 7:30 AM daily
        scheduler.add_job(
            _pybaseball_fetch_job,
            CronTrigger(hour=7, minute=30, timezone=timezone),
            id="fetch_pybaseball",
            name="Refresh pybaseball Statcast Leaderboards",
            replace_existing=True,
        )

        # Daily Statcast ingestion + Bayesian projection updates at 6:00 AM ET
        # Runs after overnight games complete, before lineup decisions
        scheduler.add_job(
            _statcast_daily_ingestion_job,
            CronTrigger(hour=6, minute=0, timezone=timezone),
            id="statcast_daily_ingestion",
            name="Statcast Daily Ingestion + Bayesian Updates",
            replace_existing=True,
        )

        # OpenClaw autonomous waiver intelligence at 8:30 AM daily
        # PAUSED (2026-04-21): Disabled while baseball module is being implemented.
        # scheduler.add_job(
        #     _openclaw_morning_job,
        #     CronTrigger(hour=8, minute=30, timezone=timezone),
        #     id="openclaw_morning",
        #     name="OpenClaw Autonomous Morning Workflow",
        #     replace_existing=True,
        # )

    if cbb_active:
        # Odds monitor - poll every 5 minutes for line movements
        odds_monitor_interval = get_float_env("ODDS_MONITOR_INTERVAL_MIN", "5")
        scheduler.add_job(
            _odds_monitor_job,
            IntervalTrigger(minutes=odds_monitor_interval),
            id="odds_monitor",
            name="Odds Line Movement Monitor",
            replace_existing=True,
        )

        # Opening line attack - run when overnight lines are posted.
        # Books typically hang openers between 10 PM and midnight ET.
        # We run analysis at 10:30 PM and 12:30 AM to catch early value.
        # Enabled by default; set OPENER_ATTACK_ENABLED=false to disable.
        opener_enabled = os.getenv("OPENER_ATTACK_ENABLED", "true").lower() == "true"
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

        # Performance Sentinel - MAE, drawdown, pytest health check at 5:00 AM
        # (30 min after daily snapshot, ensuring fresh data is available)
        scheduler.add_job(
            _nightly_health_check_job,
            CronTrigger(hour=5, minute=0, timezone=timezone),
            id="nightly_health_check",
            name="Performance Sentinel Health Check",
            replace_existing=True,
        )

        # Morning Briefing - summarize today's slate at 7 AM ET (after ratings are fresh)
        scheduler.add_job(
            _morning_briefing_job,
            CronTrigger(hour=7, minute=0, timezone=timezone),
            id="morning_briefing",
            name="Morning Slate Briefing",
            replace_existing=True,
        )

        # End-of-day results - 11 PM ET
        scheduler.add_job(
            _end_of_day_results_job,
            CronTrigger(hour=23, minute=0, timezone=timezone),
            id="end_of_day_results",
            name="End-of-Day Results Summary",
            replace_existing=True,
        )

    if fantasy_active:
        # Fantasy Baseball: Nightly decision resolution - 11:59 PM ET
        # Resolves all pending lineup decisions with actual MLB stats
        scheduler.add_job(
            _nightly_decision_resolution_job,
            CronTrigger(hour=23, minute=59, timezone=timezone),
            id="nightly_decision_resolution",
            name="Nightly Fantasy Decision Resolution",
            replace_existing=True,
        )

    if cbb_active:
        # CBB tournament season is closed. Keep the legacy notifier function for
        # archival reference, but do not register tournament_bracket_notifier on
        # startup during the MLB fantasy platform pivot.
        # Weekly model parameter recalibration - Sunday 5 AM ET
        # Note: recalibration and sentinel both run at 5:00 AM; they are independent.
        scheduler.add_job(
            _weekly_recalibration_job,
            CronTrigger(day_of_week="sun", hour=5, minute=0, timezone=timezone),
            id="weekly_recalibration",
            name="Weekly Model Parameter Recalibration",
            replace_existing=True,
        )

    if fantasy_active:
        # Job queue processor — polls job_queue table every 5s for pending heavy ops
        scheduler.add_job(
            _process_job_queue_job,
            IntervalTrigger(seconds=5),
            id="job_queue_processor",
            name="Async Job Queue Processor",
            replace_existing=True,
        )

        # Yahoo ID sync — daily at 6 AM ET
        scheduler.add_job(
            _yahoo_id_sync_job_wrapper,
            CronTrigger(hour=6, minute=0, timezone=timezone),
            id="yahoo_id_sync",
            name="Yahoo Player ID Sync",
            replace_existing=True,
        )

    scheduler.start()
    logger.info(
        "Scheduler started: cbb_active=%s, fantasy_active=%s",
        cbb_active, fantasy_active
    )

    # Ingestion Orchestrator -- gated by env var (off by default, safe for Railway)
    global _ingestion_orchestrator
    if fantasy_active and os.getenv("ENABLE_INGESTION_ORCHESTRATOR", "false").lower() == "true":
        from backend.services.daily_ingestion import DailyIngestionOrchestrator
        _ingestion_orchestrator = DailyIngestionOrchestrator()
        _ingestion_orchestrator.start()
        logger.info("DailyIngestionOrchestrator started")
    else:
        logger.info("DailyIngestionOrchestrator disabled (ENABLE_INGESTION_ORCHESTRATOR not set or fantasy_active=False)")

    # MLB nightly analysis -- 9:00 AM ET daily
    # Only active when ENABLE_MLB_ANALYSIS=true (off by default during CBB overlap)
    global _mlb_analysis_service
    if os.getenv("ENABLE_MLB_ANALYSIS", "false").lower() == "true":
        from backend.services.mlb_analysis import MLBAnalysisService
        _mlb_analysis_service = MLBAnalysisService()
        scheduler.add_job(
            _run_mlb_analysis_job,
            CronTrigger(hour=9, minute=0, timezone=timezone),
            id="mlb_nightly_analysis",
            name="MLB Nightly Analysis",
            replace_existing=True,
        )
        logger.info("MLB nightly analysis enabled (09:00 %s)", timezone)

    if cbb_active:
        # Pre-warm reanalysis cache for OddsMonitor
        try:
            db = SessionLocal()
            try:
                from backend.models import Prediction
                from backend.services.odds_monitor import get_odds_monitor
                from backend.services.recalibration import load_current_params

                today_utc = datetime.now(ZoneInfo("America/New_York")).date()
                preds = db.query(Prediction).filter(Prediction.prediction_date == today_utc).all()

                if preds:
                    from backend.betting_model import ReanalysisEngine
                    params = load_current_params(db)
                    model = CBBEdgeModel(params)

                    import math as _math
                    _sd_mult = get_float_env("SD_MULTIPLIER", "0.85")
                    cache = {}
                    for p in preds:
                        if p.full_analysis:
                            fa = p.full_analysis
                            inputs = fa.get("inputs", {})
                            calcs = fa.get("calculations", {})

                            # full_analysis.inputs has no "game" key - reconstruct
                            # game_data directly from the SQLAlchemy Game relationship.
                            game_at = p.game.away_team or ""
                            game_ht = p.game.home_team or ""
                            _key = f"{game_at}@{game_ht}"

                            # Derive base_sd_override from odds total so the
                            # unchanged-spread invariant holds for pre-warmed engines.
                            _total = (inputs.get("odds", {}).get("total")
                                      or inputs.get("odds", {}).get("sharp_consensus_total"))
                            _base_sd = _math.sqrt(float(_total)) * _sd_mult if _total else None

                            try:
                                engine = ReanalysisEngine.from_analysis_pass(
                                    model=model,
                                    game_data={
                                        "home_team": game_ht,
                                        "away_team": game_at,
                                        "is_neutral": getattr(p.game, "is_neutral", False) or False,
                                    },
                                    odds=inputs.get("odds", {}),
                                    ratings=inputs.get("ratings", {}),
                                    injuries=inputs.get("injuries"),
                                    home_style=inputs.get("home_style"),
                                    away_style=inputs.get("away_style"),
                                    matchup_margin_adj=inputs.get("margin_components", {}).get("matchup_adj", 0.0),
                                    hours_to_tipoff=calcs.get("hours_to_tipoff"),
                                    concurrent_exposure=0.0,  # approximation for startup
                                    sharp_books_available=inputs.get("odds", {}).get("sharp_books_available", 0),
                                    integrity_verdict=p.integrity_verdict,
                                    base_sd_override=_base_sd,
                                    original_verdict=p.verdict,
                                )
                                cache[_key] = engine
                            except Exception:
                                continue

                    if cache:
                        get_odds_monitor().set_reanalysis_cache(cache)
                        logger.info("Lifespan: Pre-warmed reanalysis cache with %d engines", len(cache))
            finally:
                db.close()
        except Exception as startup_exc:
            logger.warning("Lifespan: Failed to pre-warm reanalysis cache: %s", startup_exc)

    if cbb_active:
        # EMAC-024: Register VERDICT_FLIP callback for real-time Discord alerts
        def _verdict_flip_handler(movement):
            if movement.event_type == "VERDICT_FLIP" and movement.fresh_analysis:
                from backend.services.discord_notifier import send_verdict_flip_alert
                try:
                    send_verdict_flip_alert(movement)
                except Exception as alert_exc:
                    logger.error("Failed to send verdict flip alert: %s", alert_exc)

        try:
            get_odds_monitor().on_significant_move(_verdict_flip_handler)
            logger.info("Lifespan: Registered VERDICT_FLIP Discord handler")
        except Exception as reg_exc:
            logger.warning("Lifespan: Failed to register movement handler: %s", reg_exc)

    # ── Startup catch-up: if nightly analysis was missed (service restarted after
    # 3 AM ET with no predictions for today), run it automatically as a background task.
    # APScheduler is in-memory - Railway deploys after 3 AM reset the next-run time to
    # tomorrow, so without this check today's games would never get analysed.
    async def _startup_catchup():
        if not cbb_active:
            return
        from pytz import timezone as _tz
        from datetime import datetime as _dt
        et = _tz("America/New_York")
        now_et = _dt.now(et)
        nightly_cutoff = int(os.getenv("NIGHTLY_CRON_HOUR", "3"))
        if now_et.hour < nightly_cutoff:
            # Before 3 AM ET - nightly job hasn't run yet today, nothing to catch up
            return
        # Check if today already has predictions.
        # Use ET date (not UTC) - the nightly job can run before midnight UTC
        # (e.g. 22:00 UTC = 6 PM ET), storing predictions with the ET date.
        # Querying UTC date after midnight UTC would miss those rows.
        today_et = now_et.date()
        db = SessionLocal()
        try:
            count = db.query(Prediction).filter(
                Prediction.prediction_date == today_et
            ).count()
        finally:
            db.close()
        if count > 0:
            logger.info("Lifespan: Today has %d predictions - no catch-up needed", count)
            return
        logger.warning(
            "Lifespan: No predictions found for %s (now_et=%s). "
            "Nightly job was likely missed due to a post-3AM deploy. Running catch-up analysis.",
            today_et, now_et.strftime("%H:%M ET"),
        )
        try:
            await nightly_job()
            logger.info("Lifespan: Catch-up analysis complete")
        except Exception as catchup_exc:
            logger.error("Lifespan: Catch-up analysis failed: %s", catchup_exc, exc_info=True)

    asyncio.create_task(_startup_catchup())

    # Load park factors into memory in the background (non-blocking).
    # Falls back to hardcoded PARK_FACTORS dict if this fails.
    async def _load_park_factors_bg():
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, load_park_factors)
        except Exception as _pf_exc:
            logger.warning("Could not load park factors on startup: %s", _pf_exc)

    asyncio.create_task(_load_park_factors_bg())

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

# END G-31 TEMPORARY VERIFICATION ENDPOINT

# --- Strangler-fig router mounts (Phase 5 - Complete) ---
# Fantasy routes are defined in backend/routers/fantasy.py
# Do NOT add inline routes here — they will shadow the router
from backend.routers.edge import router as _edge_router  # noqa: E402
from backend.routers.fantasy import router as _fantasy_router  # noqa: E402
from backend.routers.admin import router as _admin_router  # noqa: E402
from backend.routers.trade import router as _trade_router  # noqa: E402
app.include_router(_edge_router)
app.include_router(_fantasy_router)
app.include_router(_admin_router)
app.include_router(_trade_router)

# TEST ENDPOINTS - REMOVE AFTER SYNC JOB TESTING
app.include_router(_test_router, prefix="/test", tags=["test"])
# END TEST ENDPOINTS
# ERA DIAGNOSTIC ENDPOINT - REMOVE AFTER TASK 10 COMPLETE
app.include_router(_era_diagnostic_router, prefix="/admin", tags=["admin"])
# END ERA DIAGNOSTIC ENDPOINT
# VALIDATION AUDIT ENDPOINT - REMOVE AFTER TASK 11 COMPLETE
app.include_router(_validation_audit_router, prefix="/admin", tags=["admin"])
# END VALIDATION AUDIT ENDPOINT
# OPS/WHIP BACKFILL ENDPOINT - REMOVE AFTER TASK 26 COMPLETE
app.include_router(_backfill_ops_whip_router, tags=["admin"])
# END OPS/WHIP BACKFILL ENDPOINT
# STATCAST DIAGNOSTICS ENDPOINTS - REMOVE AFTER STATCAST VALIDATION COMPLETE
app.include_router(_statcast_diag_router, prefix="/admin", tags=["admin"])
# END STATCAST DIAGNOSTICS ENDPOINTS

# SCORING DIAGNOSTICS ENDPOINTS - REMOVE AFTER NSB ROLLOUT VALIDATED
app.include_router(_scoring_diag_router, prefix="/admin", tags=["admin"])
# END SCORING DIAGNOSTICS ENDPOINTS

# CONSTRAINT MIGRATION ENDPOINT - REMOVE AFTER YAHOO ID SYNC IS LIVE
app.include_router(_constraint_migration_router, prefix="/admin", tags=["admin"])
# END CONSTRAINT MIGRATION ENDPOINT

# DATA QUALITY MONITORING DASHBOARD (Phase 1: diagnostic instrumentation)
app.include_router(data_quality.router)

# --- end strangler-fig mounts ---

# MCP Server — exposes FastAPI endpoints as Model Context Protocol tools.
# Mounted at /mcp.  Test/admin routes are excluded from tool discovery.
# noinspection PyBroadException
try:
    from fastapi_mcp import FastApiMCP

    _mcp = FastApiMCP(
        app,
        name="CBB Edge API",
        description="Fantasy baseball + CBB betting API exposed as MCP tools",
        exclude_tags=[
            "test",
            "db-verify",
            "yahoo-debug",
            "yahoo-token",
            "yahoo-parsing-test",
            "yahoo-structure-dump",
            "admin",
        ],
    )
    _mcp.mount()
    logger.info("MCP server mounted at /mcp")
except Exception as _mcp_exc:
    logger.warning("MCP server setup failed (non-fatal): %s", _mcp_exc)

# CORS - reads ALLOWED_ORIGINS env var (comma-separated) or falls back to wildcard.
# API key auth means wildcard origins are safe; credentials are never cookie-based.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_origin_regex=None,
    allow_credentials=False,  # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/admin/audit-tables")
@app.get("/admin/audit/table-counts")
async def get_all_table_counts(user: str = Depends(verify_admin_api_key)):
    """
    Factual audit of all table row counts in the public schema.
    """
    from backend.models import SessionLocal
    from sqlalchemy import text
    db = SessionLocal()
    try:
        # Get all table names in public schema
        sql_tables = text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = [r[0] for r in db.execute(sql_tables).fetchall()]

        results = {}
        for table in tables:
            try:
                count_sql = text(f'SELECT count(*) FROM "{table}"')
                count = db.execute(count_sql).scalar()
                results[table] = count
            except Exception as e:
                results[table] = f"Error: {str(e)}"

        sorted_counts = dict(sorted(results.items()))
        return {
            "status": "success",
            "tables": list(sorted_counts.keys()),
            "table_counts": sorted_counts,
        }
    finally:
        db.close()


@app.get("/admin/diagnostics/field-coverage")
async def get_field_coverage(user: str = Depends(verify_admin_api_key)):
    """
    Diagnostic endpoint to verify K-33/K-34 field population.

    Returns non-null counts for scarcity_rank (position_eligibility),
    quality_score (probable_pitchers), w_runs/w_qs (player_rolling_stats),
    and z_r/z_k_p (player_scores). Use this to confirm backfill jobs ran
    successfully in production.
    """
    from backend.models import SessionLocal
    from sqlalchemy import text
    from zoneinfo import ZoneInfo
    db = SessionLocal()
    try:
        fields = {}

        # position_eligibility
        row = db.execute(text("""
            SELECT
              COUNT(*) AS total,
              COUNT(scarcity_rank) AS scarcity_rank_populated,
              COUNT(league_rostered_pct) AS league_rostered_pct_populated
            FROM position_eligibility
        """)).fetchone()
        fields["position_eligibility"] = {
            "total": row[0],
            "scarcity_rank_populated": row[1],
            "league_rostered_pct_populated": row[2],
        }

        # probable_pitchers
        row = db.execute(text("""
            SELECT
              COUNT(*) AS total,
              COUNT(quality_score) AS quality_score_populated
            FROM probable_pitchers
        """)).fetchone()
        fields["probable_pitchers"] = {
            "total": row[0],
            "quality_score_populated": row[1],
        }

        # player_rolling_stats
        row = db.execute(text("""
            SELECT
              COUNT(*) AS total,
              COUNT(w_runs) AS w_runs_populated,
              COUNT(w_qs) AS w_qs_populated
            FROM player_rolling_stats
        """)).fetchone()
        fields["player_rolling_stats"] = {
            "total": row[0],
            "w_runs_populated": row[1],
            "w_qs_populated": row[2],
        }

        # player_scores
        row = db.execute(text("""
            SELECT
              COUNT(*) AS total,
              COUNT(z_r) AS z_r_populated,
              COUNT(z_k_p) AS z_k_p_populated
            FROM player_scores
        """)).fetchone()
        fields["player_scores"] = {
            "total": row[0],
            "z_r_populated": row[1],
            "z_k_p_populated": row[2],
        }

        return {
            "status": "ok",
            "as_of": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "fields": fields,
        }
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Session O admin backfill endpoints
# ---------------------------------------------------------------------------

@app.post("/admin/actions/backfill-scarcity-rank")
async def backfill_scarcity_rank(user: str = Depends(verify_admin_api_key)):
    """
    O1 — Bulk-set scarcity_rank for all position_eligibility rows where it is NULL.

    Uses the same static POSITION_SCARCITY mapping as _sync_position_eligibility.
    Safe to run multiple times (WHERE scarcity_rank IS NULL is idempotent).
    """
    db = SessionLocal()
    try:
        result = db.execute(text("""
            UPDATE position_eligibility
            SET scarcity_rank = CASE primary_position
                WHEN 'C'  THEN 1
                WHEN 'SS' THEN 2
                WHEN '2B' THEN 3
                WHEN '3B' THEN 4
                WHEN 'CF' THEN 5
                WHEN 'SP' THEN 6
                WHEN 'RP' THEN 7
                WHEN 'LF' THEN 8
                WHEN 'RF' THEN 9
                WHEN '1B' THEN 10
                WHEN 'DH' THEN 11
                WHEN 'OF' THEN 12
                ELSE 99
            END
            WHERE scarcity_rank IS NULL
        """))
        db.commit()
        updated = result.rowcount

        # Coverage check
        coverage = db.execute(text("""
            SELECT primary_position,
                   COUNT(*) AS total,
                   COUNT(scarcity_rank) AS has_rank
            FROM position_eligibility
            GROUP BY primary_position
            ORDER BY MIN(scarcity_rank) NULLS LAST
        """)).fetchall()

        return {
            "status": "ok",
            "rows_updated": updated,
            "as_of": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "coverage": [
                {"position": r[0], "total": r[1], "has_rank": r[2],
                 "pct": round(100 * r[2] / r[1], 1) if r[1] else 0}
                for r in coverage
            ],
        }
    except Exception as exc:
        db.rollback()
        logger.error("backfill_scarcity_rank: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@app.post("/admin/actions/backfill-quality-scores")
async def backfill_quality_scores(user: str = Depends(verify_admin_api_key)):
    """
    O2 — Ensure the _pp_date_team_uc constraint exists, then set quality_score=0.0
    for all probable_pitchers rows where quality_score IS NULL.

    Root cause: the ON CONFLICT constraint used in _sync_probable_pitchers may not
    exist in the production DB if the Alembic migration was not run when the column
    was added. This causes all daily upserts to fail silently, leaving quality_score
    NULL. The 0.0 fill is a safe neutral value (same as the code-level default when
    no ERA data is available).

    After running this endpoint, trigger /admin/sync/probable-pitchers to backfill
    ERA-based quality scores for pitchers where MLBAM data is available.
    """
    db = SessionLocal()
    try:
        # Step 1: ensure the unique constraint exists so future upserts work
        constraint_created = False
        try:
            db.execute(text("""
                ALTER TABLE probable_pitchers
                ADD CONSTRAINT _pp_date_team_uc UNIQUE (game_date, team)
            """))
            db.commit()
            constraint_created = True
        except Exception:
            db.rollback()
            # Constraint already exists — expected on a healthy DB

        # Step 2: verify constraint presence
        constraint_exists = db.execute(text("""
            SELECT 1 FROM information_schema.table_constraints
            WHERE constraint_name = '_pp_date_team_uc'
              AND table_name = 'probable_pitchers'
        """)).fetchone() is not None

        # Step 3: set quality_score = 0.0 (neutral) where NULL
        result = db.execute(text("""
            UPDATE probable_pitchers
            SET quality_score = 0.0
            WHERE quality_score IS NULL
        """))
        db.commit()
        null_rows_patched = result.rowcount

        # Step 4: summary counts
        counts = db.execute(text("""
            SELECT
                COUNT(*) AS total,
                COUNT(quality_score) AS has_qs,
                AVG(quality_score) AS avg_qs
            FROM probable_pitchers
            WHERE game_date >= CURRENT_DATE
        """)).fetchone()

        return {
            "status": "ok",
            "constraint_existed": not constraint_created,
            "constraint_created_now": constraint_created,
            "constraint_present": constraint_exists,
            "null_rows_patched": null_rows_patched,
            "upcoming_pitchers": {
                "total": counts[0],
                "has_quality_score": counts[1],
                "avg_quality_score": round(float(counts[2]), 3) if counts[2] else None,
            },
            "as_of": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        }
    except Exception as exc:
        db.rollback()
        logger.error("backfill_quality_scores: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@app.post("/admin/actions/patch-null-teams")
async def patch_null_teams(user: str = Depends(verify_admin_api_key)):
    """
    O5 — Set team='Unknown' for player_projections rows where team IS NULL or empty.

    Motivation: Session J stores 'Unknown' for new MLBAM-lookup failures, but 311
    pre-existing rows from before Session J have NULL team. NULL causes the lineup
    optimizer to treat these players as un-slottable; 'Unknown' is the explicit
    sentinel value used by downstream logic.
    """
    db = SessionLocal()
    try:
        result = db.execute(text("""
            UPDATE player_projections
            SET team = 'Unknown'
            WHERE team IS NULL OR team = ''
        """))
        db.commit()
        patched = result.rowcount

        # Residual check
        remaining = db.execute(text("""
            SELECT COUNT(*) FROM player_projections
            WHERE team IS NULL OR team = ''
        """)).fetchone()[0]

        return {
            "status": "ok",
            "rows_patched": patched,
            "remaining_null_team": remaining,
            "as_of": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        }
    except Exception as exc:
        db.rollback()
        logger.error("patch_null_teams: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        db.close()


@app.post("/admin/run-migration-v27")
async def run_migration_v27(user: str = Depends(verify_admin_api_key)):
    """
    Run v27 migration (NSB Pipeline) in production.
    """
    import subprocess
    import sys
    try:
        # Run the script as a separate process using the app's python interpreter
        result = subprocess.run(
            [sys.executable, "scripts/migrate_v27_nsb.py"],
            capture_output=True,
            text=True,
            check=True
        )
        return {
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "status": "error",
            "stdout": e.stdout,
            "stderr": e.stderr,
            "error": str(e)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/admin/investigate/statcast-raw-columns")
async def investigate_statcast_raw_columns(target_date: str = "2026-04-12", user: str = Depends(verify_admin_api_key)):
    """
    Fetch raw Statcast data for a date and return column names + first row.
    """
    from pybaseball import statcast
    import pandas as pd
    try:
        df = await asyncio.to_thread(statcast, start_dt=target_date, end_dt=target_date)
        if df is None or df.empty:
            return {"status": "empty", "date": target_date}

        # Replace NaNs for JSON
        first_row = df.head(1).replace({pd.NA: None, float('nan'): None}).to_dict(orient="records")[0]

        return {
            "status": "success",
            "date": target_date,
            "columns": df.columns.tolist(),
            "first_row_sample": first_row
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/admin/investigate/statcast-quality")
async def investigate_statcast_quality(user: str = Depends(verify_admin_api_key)):
    """
    Investigate the rate of zero-quality metrics in statcast_performances.
    """
    from backend.models import SessionLocal, StatcastPerformance
    from sqlalchemy import func
    db = SessionLocal()
    try:
        total = db.query(StatcastPerformance).count()
        with_ev = db.query(StatcastPerformance).filter(StatcastPerformance.exit_velocity_avg > 0).count()
        with_xwoba = db.query(StatcastPerformance).filter(StatcastPerformance.xwoba > 0).count()
        with_cs = db.query(StatcastPerformance).filter(StatcastPerformance.cs > 0).count()
        with_pitches = db.query(StatcastPerformance).filter(StatcastPerformance.pitches > 0).count()

        # Sample of records with zero quality metrics
        sample_zeros = db.query(StatcastPerformance).filter(
            StatcastPerformance.exit_velocity_avg == 0,
            StatcastPerformance.xwoba == 0
        ).limit(10).all()

        return {
            "total_rows": total,
            "with_exit_velocity": with_ev,
            "with_xwoba": with_xwoba,
            "with_cs": with_cs,
            "with_pitches": with_pitches,
            "zero_metric_rate": round((total - with_ev) * 100 / total, 1) if total > 0 else 0,
            "sample_zeros": [
                {
                    "name": p.player_name,
                    "date": str(p.game_date),
                    "pa": p.pa,
                    "pitches": p.pitches,
                    "k_pit": p.k_pit
                } for p in sample_zeros
            ]
        }
    finally:
        db.close()


# ============================================================================
# SCHEDULED JOB
# ============================================================================

async def nightly_job():
    """Main nightly analysis job - runs at 3 AM ET by default."""
    logger.info("Starting nightly analysis job")
    try:
        results, cache = await run_nightly_analysis()
        logger.info("Nightly analysis complete: %s", results)

        # EMAC-021: Update OddsMonitor cache for real-time pulse
        try:
            get_odds_monitor().set_reanalysis_cache(cache)
        except Exception as cache_exc:
            logger.warning("Failed to update OddsMonitor cache: %s", cache_exc)

        try:
            send_todays_bets(results.get("bet_details"), results)
        except Exception as disc_exc:
            logger.warning("Discord notification failed: %s", disc_exc)
    except Exception as exc:
        logger.error("Nightly job failed: %s", exc, exc_info=True)


def _update_outcomes_job():
    """Settle completed game bets - runs every 2 hours."""
    try:
        results = update_completed_games()
        logger.info("Outcome update: %s", results)
    except Exception as exc:
        logger.error("Outcome update job failed: %s", exc, exc_info=True)


def _capture_lines_job():
    """Capture closing lines for imminent games - runs every 30 min."""
    try:
        results = capture_closing_lines()
        logger.info("Closing lines captured: %s", results)
    except Exception as exc:
        logger.error("Closing lines job failed: %s", exc, exc_info=True)


def _weekly_recalibration_job():
    """Auto-recalibrate model parameters weekly (Sunday 5 AM ET)."""
    try:
        from backend.services.recalibration import run_recalibration
        db = SessionLocal()
        try:
            result = run_recalibration(db, changed_by="scheduler", apply_changes=True)
            if result.get("skipped"):
                logger.info("Weekly recalibration skipped: %s", result.get("reason"))
            else:
                logger.info("Weekly recalibration complete: %s", result)
        finally:
            db.close()
    except Exception:
        logger.exception("Weekly recalibration job failed")


def _nightly_health_check_job():
    """Performance Sentinel - model accuracy, portfolio drawdown, pytest suite - runs at 5:00 AM."""
    try:
        result = run_nightly_health_check()
        logger.info("Sentinel health check complete: %s", result)
    except Exception:
        logger.exception("Sentinel health check job failed")


def check_expired_eta():
    """Mark injuries whose ETA has passed but whose status has not changed."""
    db = SessionLocal()
    try:
        today_et = datetime.now(ZoneInfo("America/New_York")).date()
        today_start_et = datetime(
            year=today_et.year,
            month=today_et.month,
            day=today_et.day,
            tzinfo=ZoneInfo("America/New_York"),
        )

        injuries = (
            db.query(IngestedInjury)
            .filter(
                IngestedInjury.return_date.isnot(None),
                IngestedInjury.return_date < today_start_et,
                IngestedInjury.expired_eta.is_(False),
            )
            .all()
        )

        marked_count = 0
        colt_emerson_found = False
        for injury in injuries:
            injury.expired_eta = True
            marked_count += 1
            if injury.player_name == "Colt Emerson":
                colt_emerson_found = True

        db.commit()
        logger.info("ETA watchdog: marked %d injuries as expired_eta=True", marked_count)
        if colt_emerson_found:
            logger.info("checked Colt Emerson")
    except Exception:
        db.rollback()
        logger.exception("ETA watchdog job failed")
    finally:
        db.close()


async def _process_job_queue_job():
    """Process pending async jobs every 5 seconds. Part of ARCH-001 Phase 1."""
    db = SessionLocal()
    try:
        await jq_process(db)
    except Exception:
        logger.exception("Job queue processor error")
    finally:
        db.close()


def _morning_briefing_job():
    """Morning slate briefing at 7 AM ET - sends Discord notification with today's bets."""
    import time as _time
    t0 = _time.monotonic()
    db = SessionLocal()
    try:
        from datetime import date as _date
        from backend.models import Prediction, Game
        from backend.services.scout import generate_morning_briefing_narrative
        from backend.services.discord_simple import send_morning_brief

        today = today_et()
        preds = (
            db.query(Prediction)
            .join(Game)
            .filter(func.date(Game.game_date) == today)
            .all()
        )
        n_bets = sum(1 for p in preds if p.verdict == "BET")
        n_considered = sum(1 for p in preds if p.verdict == "CONSIDER")

        top_bet = None
        bet_preds = [p for p in preds if p.verdict == "BET"]
        if bet_preds:
            top = max(bet_preds, key=lambda p: p.conservative_edge or 0.0)
            top_bet = "%s @ %s (%.1f%% edge)" % (
                top.game.away_team, top.game.home_team,
                (top.conservative_edge or 0.0) * 100,
            )

        narrative = generate_morning_briefing_narrative(n_bets, n_considered, top_bet)
        logger.info(
            "Morning Briefing: %d BET, %d CONSIDER. %s", n_bets, n_considered, narrative
        )

        duration = _time.monotonic() - t0  # noqa: F841

        # Convert to simplified format for Discord
        bet_details = [
            {
                "team": p.game.home_team if p.bet_side == "home" else p.game.away_team,
                "spread": p.spread if p.bet_side == "home" else -p.spread,
                "edge": p.conservative_edge or 0,
                "units": p.kelly_fraction or 0,
            }
            for p in bet_preds
        ]

        slate_summary = {
            "n_games": len(preds),
            "avg_clv": sum(p.clv_percent or 0 for p in bet_preds) / len(bet_preds) if bet_preds else 0,
        }

        try:
            send_morning_brief(bet_details, slate_summary)
        except Exception as discord_exc:
            logger.warning("Discord morning briefing send failed (non-fatal): %s", discord_exc)

    except Exception:
        logger.exception("Morning briefing job failed")
    finally:
        db.close()


def _end_of_day_results_job():
    """End-of-day results summary at 11 PM ET - settles today's bets and sends Discord recap."""
    from datetime import date
    from backend.services.discord_simple import send_eod_results

    try:
        db = SessionLocal()
        try:
            today = today_et()
            settled = (
                db.query(BetLog)
                .join(Game)
                .filter(
                    func.date(BetLog.timestamp) == today,
                    BetLog.outcome.isnot(None),
                    BetLog.outcome != -1,
                )
                .all()
            )

            if not settled:
                logger.info("End-of-day: No settled bets today")
                return

            results = []
            daily_pl = 0

            for b in settled:
                team = b.game.home_team if b.bet_side == "home" else b.game.away_team
                outcome_map = {1: "win", 0: "loss", 2: "push"}
                pl = (b.profit_loss_dollars or 0) / 100
                daily_pl += pl

                results.append({
                    "team": team,
                    "outcome": outcome_map.get(b.outcome, "unknown"),
                    "pl": pl,
                })

            send_eod_results(results, daily_pl)
            logger.info(
                "End-of-day results sent: %d bets, %.2f units P&L", len(results), daily_pl
            )
        finally:
            db.close()
    except Exception:
        logger.exception("End-of-day results job failed")


def _nightly_decision_resolution_job():
    """
    Nightly fantasy baseball decision resolution at 11:59 PM ET.

    Resolves all pending lineup decisions from the previous day with actual MLB stats.
    This enables accuracy tracking and trend analysis.
    """
    try:
        from backend.fantasy_baseball.nightly_resolution import resolve_yesterdays_decisions

        result = resolve_yesterdays_decisions()
        logger.info(
            "Nightly decision resolution complete: %d resolved, %d no game, %d failed",
            result.get("resolved", 0),
            result.get("no_game", 0),
            result.get("failed", 0)
        )
    except Exception:
        logger.exception("Nightly decision resolution job failed")


def _tournament_bracket_job():
    """
    Tournament bracket release notifier.

    Runs daily from March 15-17. On the day the bracket is released (Selection
    Sunday), The Odds API will start returning NCAAB tournament games. When we
    first detect ≥4 new NCAAB games with tips scheduled Mar 18-19 (First Four),
    we send a Discord notification and mark the bracket as notified so we only
    fire once.

    Uses a sentinel file (.bracket_notified_{year}) to prevent duplicate sends.
    """
    import requests as _requests
    from datetime import date, timezone as _timezone
    from backend.services.discord_notifier import _post, _bot_token

    try:
        today = today_et()
        year = today.year

        # Only run between March 14 and March 20 inclusive
        if not (today.month == 3 and 14 <= today.day <= 20):
            return

        sentinel = f".bracket_notified_{year}"
        if os.path.exists(sentinel):
            return

        api_key = os.getenv("THE_ODDS_API_KEY")
        if not api_key:
            return

        url = (
            f"https://api.the-odds-api.com/v4/sports/basketball_ncaab/events"
            f"?apiKey={api_key}&dateFormat=iso&regions=us"
        )
        resp = _requests.get(url, timeout=10)
        if resp.status_code != 200:
            logger.warning("Tournament bracket job: Odds API returned %d", resp.status_code)
            return

        events = resp.json()
        window_start = datetime(year, 3, 18, 0, 0, 0, tzinfo=_timezone.utc)
        window_end   = datetime(year, 3, 20, 23, 59, 59, tzinfo=_timezone.utc)

        first_four_games = [
            e for e in events
            if window_start
            <= datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
            <= window_end
        ]

        if len(first_four_games) < 4:
            return

        embed = {
            "title": "🏀 NCAA Tournament Bracket Released!",
            "description": (
                f"The {year} NCAA Tournament bracket is live. "
                f"{len(first_four_games)} First Four matchups detected."
            ),
            "color": 0x1E90FF,
            "fields": [
                {
                    "name": g["home_team"] + " vs " + g["away_team"],
                    "value": g["commence_time"][:10],
                    "inline": True,
                }
                for g in first_four_games[:8]
            ],
            "footer": {"text": "CBB Edge - Tournament Mode Active"},
            "timestamp": datetime.now(_timezone.utc).isoformat(),
        }

        if _post({"embeds": [embed]}):
            open(sentinel, "w").close()
            logger.info("Tournament bracket notification sent for %d", year)

        # --- Monte Carlo bracket simulation (non-fatal) ---
        try:
            from backend.services.bracket_simulator import BracketTeam, simulate_tournament
            from backend.services.tournament_data import fetch_tournament_bracket
            from backend.services.team_mapping import normalize_team_name

            bracket_seeds = fetch_tournament_bracket()
            kenpom_ratings = get_ratings_service().get_kenpom_ratings()

            if bracket_seeds and kenpom_ratings:
                teams = []
                kenpom_keys = list(kenpom_ratings.keys())
                for team_name, seed in bracket_seeds.items():
                    norm = normalize_team_name(team_name, kenpom_keys)
                    adj_em = kenpom_ratings.get(norm, 0.0) if norm else 0.0
                    teams.append(
                        BracketTeam(
                            name=team_name,
                            seed=seed,
                            region="Unknown",
                            adj_em=adj_em,
                        )
                    )

                if len(teams) >= 32:
                    result = simulate_tournament(teams, n_sims=5000)

                    # Build seed lookup so names are always shown as "#N Name"
                    seed_lookup = {t.name: t.seed for t in teams}

                    def _labeled(name: str) -> str:
                        s = seed_lookup.get(name)
                        return f"#{s} {name}" if s else name

                    f4_lines = "\n".join(
                        f"- {_labeled(t)} ({result.advancement_probs[t][4] * 100:.0f}% F4)"
                        for t in result.projected_final_four[:4]
                        if t in result.advancement_probs
                    )
                    champ_prob = result.advancement_probs.get(
                        result.projected_champion, [0.0] * 7
                    )[6]

                    bracket_embed = {
                        "title": "Bracket Projection - Monte Carlo",
                        "description": (
                            f"**Projected Champion:** {_labeled(result.projected_champion)}"
                            f" ({champ_prob * 100:.0f}%)\n\n"
                            f"**Final Four:**\n{f4_lines}"
                        ),
                        "color": 0x1E90FF,
                        "fields": [
                            {
                                "name": f"Upset Alert #{i + 1}",
                                "value": (
                                    f"#{a['dog_seed']} {a['underdog']} vs "
                                    f"#{a['fav_seed']} {a['favorite']}"
                                    f" - {a['upset_prob'] * 100:.0f}% upset chance"
                                ),
                                "inline": False,
                            }
                            for i, a in enumerate(result.upset_alerts[:3])
                        ],
                        "footer": {
                            "text": f"Based on {result.n_sims:,} simulated brackets"
                        },
                        "timestamp": datetime.now(_timezone.utc).isoformat(),
                    }
                    _post({"embeds": [bracket_embed]})
        except Exception as sim_exc:
            logger.warning("Bracket simulation failed (non-fatal): %s", sim_exc)

    except Exception:
        logger.exception("Tournament bracket job failed")


def _mlb_analysis_job():
    """
    MLB nightly analysis job (EMAC-080).

    Runs daily at 10:00 AM ET. Fetches today's MLB schedule, projects runs
    for each game, fetches market odds, calculates edge, and persists
    projections to the mlb_projections table.
    """
    import asyncio
    from datetime import date
    from backend.services.mlb_analysis import MLBAnalysisService

    try:
        service = MLBAnalysisService()
        projections = asyncio.get_event_loop().run_until_complete(
            service.run_analysis(target_date=date.today())
        )
        if projections:
            write_result = service.write_projections_to_db(projections)
            verify_result = service.verify_edge_calculation()
            logger.info(
                "MLB analysis job: %d projections, DB write=%s, edge verify=%s",
                len(projections),
                write_result.get("status"),
                verify_result.get("status"),
            )
        else:
            logger.info("MLB analysis job: no games to project today")
    except Exception:
        logger.warning("MLB analysis job failed", exc_info=True)


@app.get("/api/tournament/bracket-projection")
async def get_bracket_projection(
    n_sims: int = Query(default=10000, ge=1000, le=50000),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Retired NCAA Tournament bracket projection endpoint.

    CBB tournament simulation is closed for the season and must not import or
    execute bracket modules during the MLB fantasy platform pivot.
    """
    raise HTTPException(
        status_code=410,
        detail=(
            "NCAA Tournament bracket projection is retired for the closed CBB "
            "season. Use MLB fantasy and betting endpoints for active workflows."
        ),
    )


def _daily_snapshot_job():
    """Generate daily performance snapshot, adjust source weights, and run alert checks - runs at 4:30 AM."""
    db = SessionLocal()
    try:
        generate_daily_snapshot(db)
        # Dynamic ensemble weight adjustment - runs after snapshot so today's
        # MAE data is available in PerformanceSnapshot for the rolling window.
        try:
            weight_result = compute_dynamic_weights(db, changed_by="auto_daily")
            logger.info("Dynamic weight calibration: %s", weight_result.get("status"))
        except Exception as w_exc:
            logger.warning("Dynamic weight calibration failed (non-fatal): %s", w_exc)
        run_alert_check()
    except Exception as exc:
        logger.error("Daily snapshot job failed: %s", exc, exc_info=True)
    finally:
        db.close()


def _odds_monitor_job():
    """Poll odds API for line movements - runs every 5 min (configurable).

    Only active during the configured operating window (default 12-23 ET)
    to avoid burning API quota when no games are scheduled.
    """
    _tz_name = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")
    _start_h = int(os.getenv("ODDS_MONITOR_START_HOUR", "12"))
    _end_h   = int(os.getenv("ODDS_MONITOR_END_HOUR",   "23"))
    try:
        _local_hour = datetime.now(ZoneInfo(_tz_name)).hour
    except Exception:
        _local_hour = datetime.now(ZoneInfo("America/New_York")).hour  # fallback if tzdata missing

    if not (_start_h <= _local_hour < _end_h):
        logger.debug(
            "Odds monitor: outside window [%d, %d) %s (current=%d) - skipping",
            _start_h, _end_h, _tz_name, _local_hour,
        )
        return

    try:
        monitor = get_odds_monitor()
        # The monitor now uses the _reanalysis_cache populated by the latest analysis run
        result = monitor.poll()
        if result.get("significant_movements", 0) > 0:
            logger.info(
                "Odds monitor: %d significant movements detected",
                result["significant_movements"],
            )
    except Exception as exc:
        logger.error("Odds monitor job failed: %s", exc, exc_info=True)


def _line_monitor_job():
    """Check for significant line movements vs. active bets - runs every 30 min."""
    try:
        results = check_line_movements()
        logger.info("Line monitor check: %s", results)
    except Exception as exc:
        logger.error("Line monitor job failed: %s", exc, exc_info=True)


async def _fetch_ratings_job():
    """Pre-warm ratings cache at 8 AM - fetches all sources concurrently.

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


def _pybaseball_fetch_job():
    """Daily 7:30 AM refresh of pybaseball FanGraphs leaderboard caches."""
    try:
        from backend.fantasy_baseball.pybaseball_loader import fetch_all_statcast_leaderboards
        import backend.fantasy_baseball.statcast_loader as _sc
        fetch_all_statcast_leaderboards(year=2026)
        _sc._batter_cache.clear()
        _sc._pitcher_cache.clear()
        _sc._loaded_at = 0.0
        logger.info("pybaseball daily refresh complete")
    except Exception as e:
        logger.error("pybaseball fetch job failed: %s", e)


def _statcast_daily_ingestion_job():
    """
    Daily 6:00 AM Statcast ingestion + Bayesian projection updates.

    This is the critical data pipeline that:
    1. Pulls yesterday's Statcast data from Baseball Savant
    2. Validates data quality
    3. Runs Bayesian projection updates (prior + likelihood -> posterior)
    4. Stores updated projections for lineup/waiver decisions

    Runs before lineup decisions so we have fresh data.
    """
    try:
        from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion
        from datetime import date, timedelta

        # Run for yesterday (most recent completed day)
        target_date = today_et() - timedelta(days=1)

        logger.info("=" * 60)
        logger.info("Starting scheduled Statcast daily ingestion")
        logger.info(f"Target date: {target_date}")
        logger.info("=" * 60)

        result = run_daily_ingestion(target_date)

        if result.get('success'):
            logger.info("Statcast daily ingestion completed successfully")
            logger.info(f"  Records processed: {result.get('records_processed', 0)}")
            logger.info(f"  Projections updated: {result.get('projections_updated', 0)}")
            logger.info(f"  High confidence updates: {result.get('high_confidence_updates', 0)}")

            # Log big movers for monitoring
            big_movers = result.get('big_mover_details', [])
            if big_movers:
                logger.info("  Top movers:")
                for mover in big_movers[:5]:
                    delta = mover.get('delta', 0)
                    direction = "↑" if delta > 0 else "↓"
                    logger.info(f"    {mover.get('name')}: {mover.get('prior')} → {mover.get('posterior')} ({direction}{abs(delta):.3f})")
        else:
            logger.error(f"Statcast daily ingestion failed: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logger.exception(f"Statcast daily ingestion job failed: {e}")


def _yahoo_id_sync_job_wrapper():
    """
    Daily 6:00 AM Yahoo player ID sync.

    Fetches all players from Yahoo fantasy league and maps Yahoo IDs to BDL player IDs.
    Improves Yahoo roster matching accuracy for waiver recommendations and lineup optimization.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting scheduled Yahoo ID sync")
        logger.info("=" * 60)

        count = run_yahoo_id_sync_job()

        logger.info(f"Yahoo ID sync completed: {count} players synced")
    except Exception as e:
        logger.exception(f"Yahoo ID sync job failed: {e}")


async def _run_mlb_analysis_job():
    """Run MLB nightly analysis at 9:00 AM ET and log results."""
    try:
        from backend.services.mlb_analysis import MLBAnalysisService
        svc = MLBAnalysisService()
        projections = await svc.run_analysis()
        logger.info("MLB analysis complete: %d projections", len(projections))
    except Exception as exc:
        logger.error("MLB analysis job failed: %s", exc)


def _openclaw_morning_job():
    """Daily 8:30 AM OpenClaw autonomous waiver intelligence workflow.

    PAUSED (2026-04-21): OpenClaw is on hold until the baseball module is
    fully implemented. All Discord notifications and report generation are
    disabled to reduce noise and filesystem clutter.
    """
    logger.info("OpenClaw morning job skipped — paused until baseball module is complete")


async def _opener_attack_job():
    """
    Run analysis when overnight opening lines are posted.

    Bookmakers hang openers with lower limits because their models are
    vulnerable - they rely on sharp action to shape the line.  Running
    analysis immediately catches early value before the line moves.

    When BET verdicts are found, Discord alerts fire immediately so bets
    can be placed at the opening price rather than waiting for the 3 AM
    or 7 AM jobs (by which time sharp money may have moved the line).
    """
    logger.info("Opening line attack triggered - running analysis on fresh openers")
    try:
        results, cache = await run_nightly_analysis()

        # EMAC-021: Update OddsMonitor cache for real-time pulse
        try:
            get_odds_monitor().set_reanalysis_cache(cache)
        except Exception as cache_exc:
            logger.warning("Failed to update OddsMonitor cache: %s", cache_exc)

        bets = results.get("bets_recommended", 0)
        if bets > 0:
            logger.info(
                "Opener attack: %d bets found in %d games (%.1fs) - sending Discord alert",
                bets, results.get("games_analyzed", 0),
                results.get("duration_seconds", 0),
            )
            # Send Discord notification immediately so bets can be placed at
            # the opening price before sharp money moves the line.
            try:
                send_todays_bets(results.get("bet_details"), results)
            except Exception as disc_exc:
                logger.warning("Opener attack Discord notification failed: %s", disc_exc)
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
        "version": "9.0",
        "status": "operational",
        "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }


@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint with pipeline summary"""
    health = {"status": "healthy", "database": "connected", "scheduler": "running"}

    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error(f"Health check database error: {e}")
        health["status"] = "degraded"
        health["database"] = f"error: {str(e)}"

    if not scheduler.running:
        health["status"] = "degraded"
        health["scheduler"] = "stopped"

    # Add pipeline summary
    try:
        pipeline = check_pipeline_health(db)
        health["pipeline_summary"] = pipeline["summary"]
        # Overall status is degraded if any critical chain job is not healthy
        if pipeline["summary"]["stale"] > 0 or pipeline["summary"]["failed"] > 0:
            health["status"] = "degraded"
    except Exception as e:
        logger.error(f"Pipeline health check error: {e}")
        health["pipeline_summary"] = {"error": str(e)}

    return health


@app.get("/health/pipeline")
async def health_pipeline(db: Session = Depends(get_db)):
    """
    Detailed pipeline health endpoint.

    Returns per-job status with last run times and thresholds.
    Returns 503 if any critical chain job is stale or failed.
    No authentication required for uptime monitoring.
    """
    try:
        pipeline = check_pipeline_health(db)
        summary = pipeline["summary"]

        # Check if any critical chain job is stale or failed
        critical_unhealthy = False
        for job_name, job_info in pipeline["jobs"].items():
            if job_name in CRITICAL_CHAIN and job_info["status"] in ("stale", "failed"):
                critical_unhealthy = True
                break

        if critical_unhealthy:
            raise HTTPException(
                status_code=503,
                detail=f"Critical pipeline jobs unhealthy: {summary['stale']} stale, {summary['failed']} failed"
            )

        return pipeline
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pipeline health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health/db")
async def health_db(db: Session = Depends(get_db)):
    """
    Database health check with table row counts.

    Returns connection status and row counts for key tables.
    No authentication required for uptime monitoring.
    """
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database connection failed: {e}")

    # Get row counts for key tables
    tables = {
        "games": "games",
        "predictions": "predictions",
        "data_ingestion_logs": "data_ingestion_logs",
        "mlb_player_stats": "mlb_player_stats",
        "matchup_context": "matchup_context",
        "player_id_mapping": "player_id_mapping",
    }

    counts = {}
    for name, table in tables.items():
        try:
            result = db.execute(text(f"SELECT COUNT(*) FROM {table}"))
            counts[name] = result.scalar() or 0
        except Exception as e:
            counts[name] = f"error: {e}"

    return {
        "status": "connected",
        "checked_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        "table_counts": counts,
    }


# ============================================================================
# AUTHENTICATED ENDPOINTS - PREDICTIONS
# ============================================================================

@app.get("/api/predictions/today", response_model=TodaysPredictionsResponse)
async def get_todays_predictions(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all UPCOMING predictions from the latest analysis batch, deduplicated by game."""
    today_utc = datetime.now(ZoneInfo("America/New_York")).date()
    now_utc = datetime.now(ZoneInfo("America/New_York"))

    # run_tier priority: lower number = higher priority (nightly beats opener)
    _TIER_PRIORITY = {"nightly": 0, "opener": 1}

    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)
        .filter(Game.game_date > now_utc)
        .order_by(Game.game_date.asc())
        .options(joinedload(Prediction.game))
        .all()
    )

    # Deduplicate by game_id: prefer nightly > opener, then highest edge as tiebreaker.
    # This prevents duplicate rows when both opener_attack and nightly runs exist.
    seen: dict = {}
    for p in predictions:
        gid = p.game_id
        if gid not in seen:
            seen[gid] = p
        else:
            cur_pri = _TIER_PRIORITY.get(seen[gid].run_tier or "", 99)
            new_pri = _TIER_PRIORITY.get(p.run_tier or "", 99)
            if new_pri < cur_pri or (
                new_pri == cur_pri
                and (p.edge_conservative or 0) > (seen[gid].edge_conservative or 0)
            ):
                seen[gid] = p

    deduped = sorted(seen.values(), key=lambda p: p.game.game_date)

    return TodaysPredictionsResponse(
        date=today_utc,
        total_games=len(deduped),
        bets_recommended=len([p for p in deduped if p.verdict.startswith("Bet")]),
        predictions=deduped,
    )


@app.get("/api/predictions/today/all", response_model=TodaysPredictionsResponse)
async def get_todays_predictions_all(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """
    Get ALL predictions for today (including games that have started).
    Used to review bets after games have begun.
    """
    today_utc = datetime.now(ZoneInfo("America/New_York")).date()

    # run_tier priority: lower number = higher priority (nightly beats opener)
    _TIER_PRIORITY = {"nightly": 0, "opener": 1}

    predictions = (
        db.query(Prediction)
        .join(Game)
        .filter(Prediction.prediction_date == today_utc)
        .order_by(Game.game_date.asc())
        .options(joinedload(Prediction.game))
        .all()
    )

    # Deduplicate by game_id: prefer nightly > opener, then highest edge as tiebreaker.
    seen: dict = {}
    for p in predictions:
        gid = p.game_id
        if gid not in seen:
            seen[gid] = p
        else:
            cur_pri = _TIER_PRIORITY.get(seen[gid].run_tier or "", 99)
            new_pri = _TIER_PRIORITY.get(p.run_tier or "", 99)
            if new_pri < cur_pri or (
                new_pri == cur_pri
                and (p.edge_conservative or 0) > (seen[gid].edge_conservative or 0)
            ):
                seen[gid] = p

    deduped = sorted(seen.values(), key=lambda p: p.game.game_date)

    return TodaysPredictionsResponse(
        date=today_utc,
        total_games=len(deduped),
        bets_recommended=len([p for p in deduped if p.verdict.startswith("Bet")]),
        predictions=deduped,
    )


@app.get("/api/predictions/bets")
async def get_recommended_bets(
    days: int = Query(default=7, ge=1, le=30),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    """Get all recommended bets from the last N days"""
    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days)

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
    starting_bankroll = get_effective_bankroll(db)
    max_daily_pct     = get_float_env("MAX_DAILY_EXPOSURE_PCT", "20.0")
    max_daily_dollars = starting_bankroll * max_daily_pct / 100.0

    today_start = datetime.now(ZoneInfo("America/New_York")).replace(hour=0, minute=0, second=0, microsecond=0)
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
            "date": datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
            "message": "Portfolio capacity exhausted - no room for parlay sizing",
            "capital_allocated_dollars": round(already_allocated, 2),
            "max_daily_dollars": round(max_daily_dollars, 2),
            "parlays": [],
        }

    # ── Today's +EV straight bets ───────────────────────────────────────────
    today_utc = datetime.now(ZoneInfo("America/New_York")).date()
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
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Model calibration: predicted probability vs actual win rate + Brier score."""
    return calculate_calibration(db, days=days)


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


@app.get("/api/performance/financial-metrics")
async def get_financial_metrics(
    days: int = Query(default=90, ge=7, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Sharpe ratio, Sortino ratio, expected Kelly growth, max drawdown, Calmar."""
    return calculate_financial_metrics(db, days=days)


@app.get("/api/performance/by-team")
async def get_performance_by_team(
    days: int = Query(default=90, ge=7, le=365),
    min_bets: int = Query(default=2, ge=1, le=20, description="Minimum bets to include a team"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Per-team win/loss breakdown for settled bets.

    Helps identify teams with systematically outlier results that may indicate
    a team mapping error (e.g. a weak team's KenPom ratings being used for a
    strong team, or vice versa).

    Teams with win rates far above or below 50% and ≥ 3 bets are flagged as
    anomalies for manual review.
    """
    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days)
    bets = (
        db.query(BetLog)
        .join(Game)
        .options(joinedload(BetLog.game))
        .filter(
            BetLog.outcome.isnot(None),
            BetLog.outcome != -1,
            BetLog.timestamp >= cutoff,
        )
        .all()
    )

    if not bets:
        return {"teams": [], "total_bets": 0, "days": days}

    import re as _re

    def _extract_pick_team(pick: str) -> str:
        """
        Extract the team name from a pick string such as:
          "Northwestern -6.5"  → "Northwestern"
          "Kansas St. +3"      → "Kansas St."
          "Florida Int'l -1.5" → "Florida Int'l"
          "Kansas"             → "Kansas"   (moneyline)
        Strategy: strip a trailing spread/odds token that starts with + or -
        and is followed by digits.  Everything before that is the team name.
        """
        if not pick:
            return "Unknown"
        stripped = pick.strip()
        # Match a trailing numeric token like "-6.5", "+3", "-110", "+100"
        m = _re.match(r"^(.+?)\s+[+-]?\d+\.?\d*\s*$", stripped)
        if m:
            return m.group(1).strip()
        return stripped

    team_stats: dict = {}
    for b in bets:
        team_name = _extract_pick_team(b.pick or "")
        if not team_name:
            continue

        if team_name not in team_stats:
            team_stats[team_name] = {
                "team": team_name,
                "bets": 0,
                "wins": 0,
                "losses": 0,
                "total_pl_dollars": 0.0,
                "total_risked": 0.0,
                "edges": [],
            }
        s = team_stats[team_name]
        s["bets"] += 1
        s["total_pl_dollars"] += b.profit_loss_dollars or 0.0
        s["total_risked"] += b.bet_size_dollars or 0.0
        if b.outcome == 1:
            s["wins"] += 1
        else:
            s["losses"] += 1
        if b.conservative_edge is not None:
            s["edges"].append(b.conservative_edge)

    results = []
    for team_name, s in team_stats.items():
        if s["bets"] < min_bets:
            continue
        win_rate = s["wins"] / s["bets"] if s["bets"] > 0 else 0.0
        roi = s["total_pl_dollars"] / s["total_risked"] if s["total_risked"] > 0 else 0.0
        mean_edge = sum(s["edges"]) / len(s["edges"]) if s["edges"] else None
        # Flag as anomaly if win rate is suspiciously high (>80%) or low (<20%)
        # with at least 3 bets - possible mapping issue signal
        anomaly = s["bets"] >= 3 and (win_rate >= 0.80 or win_rate <= 0.20)
        results.append({
            "team": team_name,
            "bets": s["bets"],
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(win_rate, 4),
            "roi": round(roi, 4),
            "total_pl_dollars": round(s["total_pl_dollars"], 2),
            "mean_edge": round(mean_edge, 4) if mean_edge is not None else None,
            "anomaly_flag": anomaly,
        })

    results.sort(key=lambda x: x["win_rate"], reverse=True)
    anomalies = [r for r in results if r["anomaly_flag"]]

    return {
        "teams": results,
        "anomalies": anomalies,
        "total_bets": len(bets),
        "total_teams": len(results),
        "days": days,
    }


@app.get("/api/performance/source-weights")
async def get_source_weights(
    history_days: int = Query(default=30, ge=1, le=365),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Current dynamic source weights and 30-day change history."""
    from backend.services.recalibration import load_current_params
    current = load_current_params(db)
    weights = {
        "weight_kenpom":     current.get("weight_kenpom",     0.342),
        "weight_barttorvik": current.get("weight_barttorvik",  0.333),
        "weight_evanmiya":   current.get("weight_evanmiya",    0.325),
    }
    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=history_days)
    history = (
        db.query(ModelParameter)
        .filter(
            ModelParameter.parameter_name.in_(
                ["weight_kenpom", "weight_barttorvik", "weight_evanmiya"]
            ),
            ModelParameter.effective_date >= cutoff,
        )
        .order_by(ModelParameter.effective_date.desc())
        .limit(300)
        .all()
    )
    history_data = [
        {
            "date":      h.effective_date.isoformat() if h.effective_date else None,
            "parameter": h.parameter_name,
            "value":     h.parameter_value,
            "reason":    h.reason,
        }
        for h in history
    ]
    return {
        "current_weights": weights,
        "history":         history_data,
        "history_days":    history_days,
    }


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

            base_sd = get_float_env("BASE_SD", "11.0")

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


@app.post("/api/bets/{bet_id}/placed")
async def mark_bet_placed(
    bet_id: int,
    placed: bool = True,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Mark an existing BetLog as placed (executed=True) or unplaced (executed=False).

    Used by the Today's Bets UI checkbox to toggle whether a recommendation
    was actually placed at the sportsbook without creating a new BetLog entry.
    """
    bet = db.query(BetLog).filter(BetLog.id == bet_id).first()
    if not bet:
        raise HTTPException(status_code=404, detail="Bet not found")
    bet.executed = placed
    db.commit()
    logger.info("Bet %d marked as %s by %s", bet_id, "placed" if placed else "unplaced", user)
    return {"success": True, "bet_id": bet_id, "placed": placed}


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
    start = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days_back)
    end = datetime.now(ZoneInfo("America/New_York")) + timedelta(days=days_ahead)

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
    status: str = Query(default="all", description="all | pending | settled | cancelled | placed"),
    days: int = Query(default=60, ge=1, le=365),
    dedup: bool = Query(default=True, description="Keep only the first BetLog per game per day"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return bet logs with optional status filter and date window.

    By default deduplicates by (game_id, bet_date) so that multiple paper trade
    rows for the same game on the same day are collapsed to the first-created
    entry.  Pass dedup=false to see all raw rows.
    """
    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days)

    query = (
        db.query(BetLog)
        .join(Game)
        .filter(BetLog.timestamp >= cutoff)
        .options(joinedload(BetLog.game))
    )

    if status == "pending":
        query = query.filter(BetLog.outcome.is_(None))
    elif status == "settled":
        # Exclude cancelled/displaced bets (outcome=-1) from settled view
        query = query.filter(BetLog.outcome.isnot(None), BetLog.outcome != -1)
    elif status == "cancelled":
        query = query.filter(BetLog.outcome == -1)
    elif status == "placed":
        # "placed" - only bets that were actually executed (not paper trades)
        query = query.filter(BetLog.executed.is_(True), BetLog.outcome != -1)
    else:
        # "all" - exclude internal cancelled/displaced bookkeeping rows
        query = query.filter(BetLog.outcome != -1)

    bets = query.order_by(BetLog.id.asc()).all()

    # Deduplicate: keep first-created BetLog per (game_id, bet_date)
    if dedup:
        seen: set = set()
        deduped: list = []
        for b in bets:
            bet_date = b.timestamp.date() if b.timestamp else None
            key = (b.game_id, bet_date)
            if key not in seen:
                seen.add(key)
                deduped.append(b)
        bets = deduped

    # Sort by timestamp descending for display
    bets = sorted(bets, key=lambda b: b.timestamp or datetime.min, reverse=True)

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


@app.get("/api/closing-lines")
async def get_closing_lines_batch(
    game_ids: str = Query(description="Comma-separated game IDs"),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return closing lines for multiple games in one query. Returns {game_id: data_or_null}."""
    try:
        ids = [int(x.strip()) for x in game_ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="game_ids must be comma-separated integers")

    rows = (
        db.query(ClosingLine)
        .filter(ClosingLine.game_id.in_(ids))
        .order_by(ClosingLine.captured_at.desc())
        .all()
    )
    # Keep only the most recent capture per game
    seen: dict = {}
    for cl in rows:
        if cl.game_id not in seen:
            seen[cl.game_id] = {
                "game_id": cl.game_id,
                "captured_at": cl.captured_at.isoformat() if cl.captured_at else None,
                "spread": cl.spread,
                "spread_odds": cl.spread_odds,
                "total": cl.total,
                "total_odds": cl.total_odds,
                "moneyline_home": cl.moneyline_home,
                "moneyline_away": cl.moneyline_away,
            }
    # Fill nulls for requested IDs with no capture
    result = {gid: seen.get(gid) for gid in ids}
    return result


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
    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days)

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
        results, cache = await run_nightly_analysis()

        # EMAC-021: Update OddsMonitor cache for real-time pulse
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


@app.post("/admin/discord/test")
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
            "timestamp": datetime.now(ZoneInfo("America/New_York")).isoformat(),
        }]
    }
    ok = _post(payload)
    if ok:
        return {"status": "ok", "channel_id": _channel_id()}
    raise HTTPException(status_code=502, detail="Discord API call failed - check logs")


@app.post("/admin/discord/test-simple")
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

    today_utc = datetime.now(ZoneInfo("America/New_York")).date()
    now_utc = datetime.now(ZoneInfo("America/New_York"))

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

    # Deduplicate - keep the highest-edge prediction per game
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
        dry_run=true  - compute and return diagnostics without writing changes.
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


@app.get("/admin/recalibration/audit")
async def recalibration_audit(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Get recalibration audit data (admin only).

    Returns:
        - Settled bets count with prediction links
        - Current home_advantage and sd_multiplier values
        - Drift from baseline parameters
        - Recommendations for tournament prep
    """
    from sqlalchemy import func

    # Count settled bets with prediction links
    settled_with_pred = (
        db.query(BetLog)
        .filter(BetLog.outcome.isnot(None))
        .filter(BetLog.prediction_id.isnot(None))
        .count()
    )

    # Get current parameters
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

    # Calculate drift from baselines
    baseline_ha = 3.09
    baseline_sd = 0.85

    ha_drift = abs(ha_value - baseline_ha) / baseline_ha * 100
    sd_drift = abs(sd_value - baseline_sd) / baseline_sd * 100

    # Check last recalibration date
    last_recal = current_ha.effective_date if current_ha else None
    days_since = (datetime.now(ZoneInfo("America/New_York")) - last_recal).days if last_recal else None

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


@app.get("/admin/debug/duplicate-bets")
async def debug_duplicate_bets(
    days: int = Query(default=90, ge=1, le=365),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Find BetLog entries where multiple paper trades exist for the same game on
    the same calendar day.  These are duplicates that inflate bet counts and
    distort ROI / win-rate statistics.

    Returns each duplicate group with all matching bet IDs so they can be
    reviewed and the extras deleted via the admin panel or directly in the DB.
    """
    from sqlalchemy import func, cast, Date as SADate

    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days)

    # Fetch all paper trade bet logs in the window
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

    # Group by (game_id, calendar_date)
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


@app.get("/admin/debug/bets-last-24h")
async def debug_bets_last_24h(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Debug endpoint: Get all bets from last 24 hours.

    Returns simple list for debugging UI issues.
    """
    from datetime import timedelta

    since = datetime.now(ZoneInfo("America/New_York")) - timedelta(hours=24)

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


@app.post("/admin/cleanup/duplicate-bets")
async def cleanup_duplicate_bets(
    dry_run: bool = Query(default=True, description="If true, only report what would be deleted without deleting"),
    days: int = Query(default=365, ge=1, le=730, description="How far back to look for duplicates"),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Delete duplicate paper trade BetLog entries.

    A duplicate is any paper trade BetLog where the same game_id has more than
    one entry on the same calendar day.  The lowest id (first created) is kept;
    all others are deleted.

    By default dry_run=true - set dry_run=false to actually delete.
    """
    cutoff = datetime.now(ZoneInfo("America/New_York")) - timedelta(days=days)

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

    # Group by (game_id, calendar_date) - keep lowest id (first-created)
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


@app.post("/admin/force-update-outcomes")
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


@app.post("/admin/force-capture-lines")
async def force_capture_lines(user: str = Depends(verify_admin_api_key)):
    """Manually trigger the closing-line capture job (admin only)."""
    logger.info("Manual line capture triggered by %s", user)
    try:
        results = capture_closing_lines()
        return {"message": "Line capture complete", **results}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/board/refresh")
async def admin_board_refresh(user: str = Depends(verify_admin_api_key)):
    """Clear in-memory player board cache. Call after dropping new Steamer CSVs to data/projections/."""
    from backend.fantasy_baseball.player_board import reset_board_cache
    reset_board_cache()
    return {"status": "ok", "message": "Board cache cleared -- next request reloads from disk"}


@app.delete("/admin/bets/{bet_id}")
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


@app.delete("/admin/bets/orphaned/cleanup")
async def cleanup_orphaned_bets(
    dry_run: bool = Query(default=True),
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """Delete BetLog entries whose game_id no longer exists in the games table."""
    from sqlalchemy import text
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


@app.delete("/admin/games/{game_id}")
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
    alert.acknowledged_at = datetime.now(ZoneInfo("America/New_York"))
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


@app.get("/admin/pipeline-health")
async def admin_pipeline_health(
    db: Session = Depends(get_db),
    user: str = Depends(verify_admin_api_key),
):
    """Check freshness and row counts for all critical fantasy tables."""
    from backend.services.pipeline_validator import (
        check_table_health,
        pipeline_health_summary,
    )
    checks = check_table_health(db)
    return pipeline_health_summary(checks)


@app.get("/admin/version")
async def get_deployment_version(
    db: Session = Depends(get_db),
    user: str = Depends(verify_admin_api_key),
):
    """
    Return deployment fingerprint for production verification.

    Used by Layer 2 certification (Stage 1 and Stage 5) to confirm
    production is running the latest repo code.

    Returns:
    {
        "git_commit_sha": "abc123def456...",
        "git_commit_date": "2026-04-15T10:30:00Z",
        "build_timestamp": "2026-04-15T10:31:15Z",
        "app_version": "dev"
    }
    """
    from backend.models import DeploymentVersion
    import subprocess

    # Try to get from database first (authoritative)
    version = db.query(DeploymentVersion).order_by(DeploymentVersion.deployed_at.desc()).first()

    if version:
        return {
            "git_commit_sha": version.git_commit_sha,
            "git_commit_date": version.git_commit_date,
            "build_timestamp": version.build_timestamp.isoformat(),
            "app_version": version.app_version or "dev"
        }

    # Fallback: get from git directly
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode().strip()

        commit_date = subprocess.check_output(
            ['git', 'show', '-s', '--format=%ci', 'HEAD'],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode().strip()

        return {
            "git_commit_sha": sha,
            "git_commit_date": commit_date,
            "build_timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
            "app_version": "dev"
        }
    except Exception:
        # Ultimate fallback for environments without git
        return {
            "git_commit_sha": "unknown",
            "git_commit_date": None,
            "build_timestamp": datetime.now(ZoneInfo("UTC")).isoformat(),
            "app_version": "dev"
        }


@app.get("/admin/ingestion/status")
async def ingestion_status(user: str = Depends(verify_api_key)):
    """Return per-job status for the DailyIngestionOrchestrator, or disabled signal."""
    if _ingestion_orchestrator is None:
        return {"enabled": False, "jobs": {}}
    return {"enabled": True, "jobs": _ingestion_orchestrator.get_status()}


_PIPELINE_JOB_ORDER = [
    "mlb_game_log",
    "mlb_box_stats",
    "statcast",
    "rolling_windows",
    "player_scores",
    "player_momentum",
    "ros_simulation",
    "decision_optimization",
    "backtesting",
    "explainability",
    "snapshot",
]


@app.post("/admin/ingestion/run/{job_id}")
async def ingestion_run_job(
    job_id: str,
    user: str = Depends(verify_admin_api_key),
):
    """
    Manually trigger a single ingestion pipeline job.

    Valid job_id values (must be run in this order for data to flow):
      mlb_game_log -> mlb_box_stats -> rolling_windows ->
      player_scores -> player_momentum -> ros_simulation

    Returns the job result dict with status, records, elapsed_ms.
    Use /admin/ingestion/run-pipeline to run all six in sequence.
    """
    if _ingestion_orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="DailyIngestionOrchestrator is disabled (ENABLE_INGESTION_ORCHESTRATOR=false)",
        )
    try:
        result = await _ingestion_orchestrator.run_job(job_id)
        return {"job_id": job_id, "result": result}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Manual job trigger failed for %s: %s", job_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/ingestion/run-pipeline")
async def ingestion_run_pipeline(
    user: str = Depends(verify_admin_api_key),
):
    """
    Manually run the full MLB intelligence pipeline in order:
      mlb_game_log -> mlb_box_stats -> rolling_windows ->
      player_scores -> player_momentum -> ros_simulation

    Runs each job sequentially so downstream jobs see upstream data.
    Returns per-job results. Any job failure is recorded but does not abort
    subsequent jobs (pipeline degrades gracefully).
    """
    if _ingestion_orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="DailyIngestionOrchestrator is disabled (ENABLE_INGESTION_ORCHESTRATOR=false)",
        )
    results = {}
    for job_id in _PIPELINE_JOB_ORDER:
        try:
            results[job_id] = await _ingestion_orchestrator.run_job(job_id)
        except Exception as exc:
            logger.error("Pipeline run: job %s failed: %s", job_id, exc, exc_info=True)
            results[job_id] = {"status": "failed", "error": str(exc)}
    return {"pipeline": _PIPELINE_JOB_ORDER, "results": results}


@app.post("/admin/ingestion/steamer-csv")
async def ingest_steamer_csv_projections(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Ingest Steamer 2026 projections from CSV files into PlayerProjection table.

    Reads from data/projections/steamer_batting_2026.csv and
    data/projections/steamer_pitching_2026.csv, then writes all
    projections to the database with proper column mappings.

    Returns: dict with status, batters/pitchers counts, and rows written.
    """
    from backend.fantasy_baseball.csv_projection_ingestion import run_steamer_ingestion

    try:
        result = run_steamer_ingestion(db)
        return result
    except Exception as exc:
        logger.error("Steamer CSV ingestion failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/backfill/player-id-mapping")
async def backfill_player_id_mapping(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Manually trigger player ID mapping backfill from BDL API.

    Fetches all MLB players from BallDon'tLie and stores cross-reference
    mapping (BDL ID, MLBAM ID, Yahoo ID, full name, normalized name).

    Returns: dict with status, records_processed, elapsed_ms, table_count
    """
    from scripts.backfill_player_id_mapping import backfill_player_id_mapping

    try:
        result = backfill_player_id_mapping()
        return result
    except Exception as exc:
        logger.error("Player ID mapping backfill failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/backfill/positions")
async def backfill_positions(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Manually trigger position eligibility backfill from Yahoo Fantasy API.

    Fetches current position eligibility snapshot for all 30 MLB teams.

    Returns: dict with status, records_processed, teams_processed, elapsed_ms
    """
    from scripts.backfill_positions import backfill_position_eligibility

    try:
        result = backfill_position_eligibility()
        return result
    except Exception as exc:
        logger.error("Position eligibility backfill failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/backfill/probable-pitchers")
async def backfill_probable_pitchers(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Manually trigger probable pitchers backfill from BDL Games API.

    Fetches historical probable pitchers (March 20 - April 8, 2026).

    Returns: dict with status, records_processed, dates_processed, elapsed_ms
    """
    from scripts.backfill_probable_pitchers import backfill_probable_pitchers

    try:
        result = backfill_probable_pitchers()
        return result
    except Exception as exc:
        logger.error("Probable pitchers backfill failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/backfill/statcast")
async def backfill_statcast(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Manually trigger Statcast backfill from Baseball Savant CSV API.

    Fetches historical Statcast data (March 20 - April 8, 2026).
    WARNING: This can take 10-20 minutes due to API rate limits.

    Returns: dict with status, records_processed, dates_processed, elapsed_ms
    """
    from scripts.backfill_statcast import backfill_statcast

    try:
        result = backfill_statcast()
        return result
    except Exception as exc:
        logger.error("Statcast backfill failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/backfill/yahoo-keys")
async def backfill_yahoo_keys_endpoint(
    dry_run: bool = False,
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Manually trigger yahoo_key backfill from position_eligibility to player_id_mapping.

    Cross-references position_eligibility.yahoo_player_key with player_id_mapping
    by matching on normalized player names.

    This bridges the Yahoo Fantasy namespace to the BDL namespace.

    Query params:
        dry_run: If true, preview without writing to database

    Returns: dict with status, updated_count, skipped_count, errors, yahoo_key_count
    """
    from scripts.backfill_yahoo_keys import backfill_yahoo_keys

    try:
        result = backfill_yahoo_keys(db, dry_run=dry_run)
        return result
    except Exception as exc:
        logger.error("Yahoo keys backfill failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/link-orphans")
async def link_orphaned_eligibility(
    dry_run: bool = Query(default=True, description="If true, preview without writing to database"),
    verbose: bool = Query(default=False, description="Enable verbose logging"),
    user: str = Depends(verify_admin_api_key),
):
    """
    Link orphaned position_eligibility records via fuzzy name matching.

    Task 21: Links orphaned position_eligibility records to player_id_mapping
    using difflib.SequenceMatcher with 85% similarity threshold.

    Runs in a background thread to avoid blocking the FastAPI event loop.

    Target: Reduce orphans from ~477 to <50.

    Query params:
        dry_run: If true, preview without writing to database (default: True)
        verbose: Enable detailed logging of match attempts

    Returns: dict with status, linked_count, remaining_count, success_rate, elapsed_ms
    """
    import asyncio
    from backend.fantasy_baseball.orphan_linker import link_orphaned_records
    from backend.models import SessionLocal

    def _run_in_thread():
        db = SessionLocal()
        try:
            return link_orphaned_records(db, dry_run=dry_run, verbose=verbose)
        finally:
            db.close()

    try:
        result = await asyncio.to_thread(_run_in_thread)
        return result
    except Exception as exc:
        logger.error("Orphan linking failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/backfill/all")
async def backfill_all(
    user: str = Depends(verify_admin_api_key),
    db: Session = Depends(get_db),
):
    """
    Run all backfill scripts in dependency order via HTTP request.

    Executes: player_id_mapping -> positions -> probable_pitchers -> statcast
    Stops on critical failures. Provides progress summary.

    Returns: dict with overall success status and per-script results
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.backfill_all_data"],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 minutes
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout[-2000:],  # Last 2000 chars
            "stderr": result.stderr[-2000:] if result.stderr else None,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Backfill timeout (exceeded 30 minutes)")
    except Exception as exc:
        logger.error("Master backfill orchestration failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/export-projections", dependencies=[Depends(verify_admin_api_key)])
async def admin_export_projections():
    """
    Export FanGraphs RoS cache to Steamer CSV format.
    Bridges the FanGraphs daily fetch (3 AM) with load_full_board().
    Run this after fangraphs_ros job has completed at least once.
    """
    from backend.fantasy_baseball.projections_loader import export_ros_to_steamer_csvs
    from backend.services.daily_ingestion import _ROS_CACHE, _load_persisted_ros_cache

    # Try in-memory cache first, then persisted cache
    bat_raw = _ROS_CACHE.get("bat")
    pit_raw = _ROS_CACHE.get("pit")

    if not bat_raw and not pit_raw:
        bat_raw, pit_raw, cached_at = _load_persisted_ros_cache()
        if cached_at is None:
            return {
                "success": False,
                "error": "No FanGraphs RoS cache available. Run fangraphs_ros job first.",
            }

    result = export_ros_to_steamer_csvs(bat_raw or {}, pit_raw or {})

    # Clear the lru_cache so load_full_board() picks up new CSVs
    from backend.fantasy_baseball.projections_loader import load_full_board
    load_full_board.cache_clear()

    return {
        "success": True,
        "batting_rows": result["batting_rows"],
        "pitching_rows": result["pitching_rows"],
        "message": "Projections exported. load_full_board() cache cleared.",
    }


@app.get("/admin/explanations/{decision_id}")
async def get_explanation(
    decision_id: int,
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Return the stored explanation for a specific decision_results row.
    Returns 404 if no explanation exists for that decision_id.

    Auth: verify_api_key required.
    """
    from backend.models import DecisionExplanation as _DecisionExplanation
    row = db.query(_DecisionExplanation).filter(
        _DecisionExplanation.decision_id == decision_id
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="No explanation for decision_id={}".format(decision_id))
    return {
        "decision_id": row.decision_id,
        "bdl_player_id": row.bdl_player_id,
        "as_of_date": str(row.as_of_date),
        "decision_type": row.decision_type,
        "summary": row.summary,
        "factors": row.factors_json,
        "confidence_narrative": row.confidence_narrative,
        "risk_narrative": row.risk_narrative,
        "track_record_narrative": row.track_record_narrative,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
    }


def _snapshot_to_dict(row) -> dict:
    return {
        "as_of_date": str(row.as_of_date),
        "n_players_scored": row.n_players_scored,
        "n_momentum_records": row.n_momentum_records,
        "n_simulation_records": row.n_simulation_records,
        "n_decisions": row.n_decisions,
        "n_explanations": row.n_explanations,
        "n_backtest_records": row.n_backtest_records,
        "mean_composite_mae": row.mean_composite_mae,
        "regression_detected": row.regression_detected,
        "top_lineup_player_ids": row.top_lineup_player_ids,
        "top_waiver_player_ids": row.top_waiver_player_ids,
        "pipeline_jobs_run": row.pipeline_jobs_run,
        "pipeline_health": row.pipeline_health,
        "health_reasons": row.health_reasons,
        "summary": row.summary,
        "computed_at": row.computed_at.isoformat() if row.computed_at else None,
    }


@app.get("/admin/snapshot/latest")
async def get_latest_snapshot(db: Session = Depends(get_db)):
    """Return the most recent DailySnapshot row."""
    from backend.models import DailySnapshot as _DailySnapshot
    row = db.query(_DailySnapshot).order_by(_DailySnapshot.as_of_date.desc()).first()
    if row is None:
        raise HTTPException(status_code=404, detail="No snapshots available yet")
    return _snapshot_to_dict(row)


@app.get("/admin/snapshot/{snapshot_date}")
async def get_snapshot_by_date(snapshot_date: str, db: Session = Depends(get_db)):
    """
    Return the DailySnapshot for a specific date (YYYY-MM-DD).
    Returns 404 if no snapshot exists for that date.
    """
    from datetime import date as _date_type
    try:
        d = _date_type.fromisoformat(snapshot_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
    from backend.models import DailySnapshot as _DailySnapshot
    row = db.query(_DailySnapshot).filter(_DailySnapshot.as_of_date == d).first()
    if row is None:
        raise HTTPException(status_code=404, detail="No snapshot for {}".format(snapshot_date))
    return _snapshot_to_dict(row)


@app.get("/admin/portfolio/status")
async def get_portfolio_status(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return current portfolio state: exposure, drawdown, pending positions.

    Loads bankroll and pending positions from DB on each call so that
    drawdown_pct reflects real settled P&L rather than staying at 0%.
    """
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


@app.get("/admin/odds-monitor/status")
async def get_odds_monitor_status(user: str = Depends(verify_api_key)):
    """Return odds monitor status: tracked games, last poll time."""
    try:
        monitor = get_odds_monitor()
        status = monitor.get_status()
        status["status"] = "ok"
        return status
    except Exception as exc:
        logger.warning("Odds monitor status unavailable: %s", exc)
        return {
            "status": "degraded",
            "active": False,
            "games_tracked": 0,
            "last_poll": None,
            "quota_remaining": None,
            "quota_updated_at": None,
            "quota_is_low": False,
            "error": str(exc),
        }


@app.get("/admin/oracle/flagged", response_model=OracleFlaggedResponse)
async def get_oracle_flagged(
    days_back: int = Query(7, ge=1, le=90, description="Look-back window in days"),
    run_tier: Optional[str] = Query(None, description="Filter by run tier: opener|nightly|closing"),
    db: Session = Depends(get_db),
    user: str = Depends(verify_admin_api_key),
):
    """
    Return all predictions where the model diverged significantly from the
    KenPom + BartTorvik consensus (oracle_flag = TRUE).

    Useful for post-game review: were oracle-flagged predictions less accurate?
    """
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


@app.get("/admin/ratings/status")
async def get_ratings_status(user: str = Depends(verify_admin_api_key)):
    """
    Return live rating source coverage.

    Fetches (or returns cached) ratings from all three sources and reports
    how many teams each source is providing.  Use this to diagnose KenPom-only
    degraded mode before running nightly analysis.
    """
    from backend.services.ratings import get_ratings_service
    service = get_ratings_service()
    # Use cached data if < 6 hours old to avoid unnecessary scrape on status checks
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
                (__import__("datetime").datetime.now(ZoneInfo("America/New_York")) - service.cache_timestamp).total_seconds() / 3600
                if service.cache_timestamp else 0
            ),
            2,
        ),
    }


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
    return get_float_env("STARTING_BANKROLL", "1000")


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
        "env_starting_bankroll": get_float_env("STARTING_BANKROLL", "1000"),
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
# FEATURE FLAGS
# ============================================================================

_ALLOWED_FLAGS = {"draft_board_enabled"}


@app.get("/api/feature-flags")
async def get_feature_flags(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """Return feature flag values. Unset flags return their default."""
    defaults = {"draft_board_enabled": True}
    result = dict(defaults)
    for flag in _ALLOWED_FLAGS:
        row = _get_model_param(db, flag)
        if row is not None and row.parameter_value_json is not None:
            result[flag] = bool(row.parameter_value_json)
    return result


@app.post("/admin/feature-flags/{flag_name}")
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

    No paper trades required - creates brand-new BetLog rows for each wager,
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


# Kept inline: this route has no equivalent in backend/routers/fantasy.py and
# the frontend depends on it (frontend/lib/api.ts). All other fantasy/dashboard
# routes live in backend/routers/fantasy.py — do not add new inline routes.
@app.get("/api/fantasy/projections/canonical")
async def get_canonical_projections(
    limit: int = 100,
    player_type: Optional[str] = None,
    min_confidence: float = 0.0,
    db: Session = Depends(get_db),
):
    """Canonical projection surface for frontend consumption."""
    from backend.models import CanonicalProjection, PlayerIdentity
    q = db.query(CanonicalProjection)
    if player_type:
        q = q.filter(CanonicalProjection.player_type == player_type.upper())
    if min_confidence > 0:
        q = q.filter(CanonicalProjection.confidence_score >= min_confidence)
    rows = (
        q.order_by(CanonicalProjection.confidence_score.desc().nullslast())
         .limit(limit)
         .all()
    )
    mlbam_ids = [r.player_id for r in rows if r.player_id and r.player_id > 0]
    identity_map: dict = {}
    if mlbam_ids:
        identities = (
            db.query(PlayerIdentity)
            .filter(PlayerIdentity.mlbam_id.in_(mlbam_ids))
            .all()
        )
        identity_map = {i.mlbam_id: i.full_name for i in identities}
    return [
        {
            "player_id": r.player_id,
            "player_name": identity_map.get(r.player_id, f"player_{r.player_id}"),
            "player_type": r.player_type,
            "source_engine": r.source_engine,
            "projection_date": r.projection_date.isoformat() if r.projection_date else None,
            "confidence_score": r.confidence_score,
            "projected_pa": r.projected_pa,
            "projected_ip": r.projected_ip,
            "proj_hr": r.proj_hr,
            "proj_sb": r.proj_sb,
            "proj_r": r.proj_r,
            "proj_rbi": r.proj_rbi,
            "proj_avg": r.proj_avg,
            "proj_ops": r.proj_ops,
            "proj_era": r.proj_era,
            "proj_whip": r.proj_whip,
            "proj_w": r.proj_w,
            "proj_sv": r.proj_sv,
            "proj_k": r.proj_k,
            "xwoba": r.xwoba,
            "xera": r.xera,
            "savant_pitch_quality_score": r.savant_pitch_quality_score,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]




# ============================================================================
# YAHOO FANTASY BASEBALL - DEBUG ENDPOINTS
# ============================================================================

@app.get("/admin/yahoo/test")
async def yahoo_test(user: str = Depends(verify_admin_api_key)):
    """
    Test Yahoo API connectivity.
    Returns league name + authenticated team key.
    Requires YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REFRESH_TOKEN in env.
    """
    try:
        client = get_yahoo_client()
        league = client.get_league()
        team_key = client.get_my_team_key()
        return {
            "status": "ok",
            "connected": True,
            "league_name": league.get("name"),
            "league_key": client.league_key,
            "my_team_key": team_key,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/yahoo/roster-raw")
async def yahoo_roster_raw(user: str = Depends(verify_admin_api_key)):
    """
    Return the raw fantasy_content structure from Yahoo for your roster.
    Use this to inspect the exact shape Yahoo returns so parsing can be debugged.
    """
    try:
        client = get_yahoo_client()
        raw = client.get_roster_raw()
        return {"fantasy_content": raw}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/yahoo/roster")
async def yahoo_roster(user: str = Depends(verify_admin_api_key)):
    """
    Return your parsed Yahoo Fantasy roster.
    """
    try:
        client = get_yahoo_client()
        roster = client.get_roster()
        return {"count": len(roster), "players": roster}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/admin/investigate/ops-whip")
async def investigate_ops_whip_root_cause():
    """
    Temporary endpoint to investigate why ops/whip fields are NULL.
    Runs 6 diagnostic queries and returns comprehensive analysis.
    """
    try:
        db = SessionLocal()
        results = {}

        # Investigation 1: Check if obp/slg data exists
        inv1 = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(obp) as has_obp,
                COUNT(slg) as has_slg,
                COUNT(ops) as has_ops,
                COUNT(*) FILTER (WHERE obp IS NOT NULL AND slg IS NOT NULL) as has_both
            FROM mlb_player_stats
        """)).fetchone()
        results["investigation_1_source_data"] = {
            "total_rows": inv1.total_rows,
            "has_obp": inv1.has_obp,
            "has_slg": inv1.has_slg,
            "has_ops": inv1.has_ops,
            "has_both_obp_slg": inv1.has_both
        }

        # Investigation 2: Sample rows with obp/slg to see actual values
        inv2 = db.execute(text("""
            SELECT bdl_player_id, obp, slg, ops, game_date
            FROM mlb_player_stats
            WHERE obp IS NOT NULL OR slg IS NOT NULL
            ORDER BY game_date DESC
            LIMIT 5
        """)).fetchall()
        results["investigation_2_sample_rows"] = [
            {
                "bdl_player_id": row.bdl_player_id,
                "obp": float(row.obp) if row.obp else None,
                "slg": float(row.slg) if row.slg else None,
                "ops": float(row.ops) if row.ops else None,
                "game_date": row.game_date.isoformat() if row.game_date else None
            }
            for row in inv2
        ]

        # Investigation 3: Check if raw_payload has the data
        inv3 = db.execute(text("""
            SELECT
                COUNT(*) FILTER (WHERE raw_payload::text LIKE '%obp%') as has_obp_payload,
                COUNT(*) FILTER (WHERE raw_payload::text LIKE '%slg%') as has_slg_payload,
                COUNT(*) FILTER (WHERE raw_payload::text LIKE '%ops%') as has_ops_payload
            FROM mlb_player_stats
        """)).fetchone()
        results["investigation_3_raw_payload"] = {
            "has_obp_in_payload": inv3.has_obp_payload,
            "has_slg_in_payload": inv3.has_slg_payload,
            "has_ops_in_payload": inv3.has_ops_payload
        }

        # Investigation 4: Check whip components
        inv4 = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(walks_allowed) as has_bb,
                COUNT(hits_allowed) as has_h,
                COUNT(whip) as has_whip,
                COUNT(*) FILTER (WHERE walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL) as has_components
            FROM mlb_player_stats
        """)).fetchone()
        results["investigation_4_whip_components"] = {
            "total_rows": inv4.total_rows,
            "has_walks_allowed": inv4.has_bb,
            "has_hits_allowed": inv4.has_h,
            "has_whip": inv4.has_whip,
            "has_both_components": inv4.has_components
        }

        # Investigation 5: Check ERA anomaly
        inv5 = db.execute(text("""
            SELECT bdl_player_id, era, earned_runs, innings_pitched, game_date
            FROM mlb_player_stats
            WHERE era > 100
            ORDER BY era DESC
            LIMIT 1
        """)).fetchone()
        results["investigation_5_era_anomaly"] = {
            "found": inv5 is not None,
            "data": {
                "bdl_player_id": inv5.bdl_player_id,
                "era": float(inv5.era) if inv5 and inv5.era else None,
                "earned_runs": float(inv5.earned_runs) if inv5 and inv5.earned_runs else None,
                "innings_pitched": float(inv5.innings_pitched) if inv5 and inv5.innings_pitched else None,
                "game_date": inv5.game_date.isoformat() if inv5 and inv5.game_date else None
            } if inv5 else None
        }

        # Investigation 6: Orphaned position_eligibility
        inv6 = db.execute(text("""
            SELECT COUNT(*) as orphaned
            FROM position_eligibility pe
            LEFT JOIN player_id_mapping pim ON pe.yahoo_player_key = pim.yahoo_key
            WHERE pe.yahoo_player_key IS NOT NULL AND pim.yahoo_key IS NULL
        """)).fetchone()
        results["investigation_6_orphaned_positions"] = {
            "orphaned_records": inv6.orphaned
        }

        db.close()
        return results

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/admin/fantasy/reload-board", dependencies=[Depends(verify_admin_api_key)])
async def admin_reload_fantasy_board():
    """
    Force a fresh read of projection CSVs (data/projections/*.csv).
    Call this after dropping new Steamer/ZiPS exports into data/projections/.
    """
    from backend.fantasy_baseball.projections_loader import load_full_board
    from backend.fantasy_baseball import player_board

    load_full_board.cache_clear()
    player_board._BOARD = None  # Reset module-level sentinel

    board = player_board.get_board()
    return {"status": "ok", "players_loaded": len(board) if board else 0}


@app.post("/admin/pybaseball/refresh")
async def admin_refresh_pybaseball(year: int = 2025, user: str = Depends(verify_admin_api_key)):
    """Force-refresh pybaseball Statcast cache and invalidate in-memory statcast_loader cache."""
    from backend.fantasy_baseball.pybaseball_loader import fetch_all_statcast_leaderboards
    import backend.fantasy_baseball.statcast_loader as _sc
    fetch_all_statcast_leaderboards(year=year, force_refresh=True)
    _sc._batter_cache.clear()
    _sc._pitcher_cache.clear()
    _sc._loaded_at = 0.0
    return {"status": "ok", "year": year}


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    origin = request.headers.get("origin") or "*"
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
        headers={
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "false",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
