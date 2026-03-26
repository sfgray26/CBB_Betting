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
import asyncio
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
from backend.services.alerts import check_performance_alerts, persist_alerts, run_alert_check
from backend.services.recalibration import compute_dynamic_weights
from backend.services.discord_notifier import send_todays_bets
from backend.services.sentinel import run_nightly_health_check
from backend.services.dk_import import (
    parse_dk_csv, preview_import, apply_import,
    preview_direct_import, apply_direct_import,
)
from backend.services.odds_monitor import get_odds_monitor
from backend.services.portfolio import get_portfolio_manager
from backend.services.ratings import get_ratings_service
from backend.utils.env_utils import get_float_env
from backend.schemas import (
    BetLogCreate,
    BetLogResponse,
    OutcomeUpdate,
    OutcomeResponse,
    AnalysisTriggerResponse,
    TodaysPredictionsResponse,
    PredictionResponse,
    DailyLineupResponse,
    WaiverWireResponse,
    LineupPlayerOut,
    StartingPitcherOut,
    RosterPlayerOut,
    RosterResponse,
    MatchupTeamOut,
    MatchupResponse,
    LineupApplyPlayer,
    LineupApplyRequest,
    OracleFlaggedResponse,
    OraclePredictionDetail,
    MatchupSimulateRequest,
)
from backend.fantasy_baseball.yahoo_client import (
    YahooFantasyClient,
    YahooAuthError,
    YahooAPIError,
)
from backend.fantasy_baseball.yahoo_client_resilient import ResilientYahooClient
from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_optimizer

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Scheduler instance — AsyncIOScheduler runs jobs inside FastAPI's event loop,
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

    # O-10: Line movement monitor — runs every 30 minutes
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

    # Pre-warm ratings cache at 8 AM so nightly analysis uses fresh data
    ratings_prewarm_hour = int(os.getenv("RATINGS_PREWARM_HOUR", "8"))
    scheduler.add_job(
        _fetch_ratings_job,
        CronTrigger(hour=ratings_prewarm_hour, minute=0, timezone=timezone),
        id="fetch_ratings",
        name="Pre-warm Ratings Cache",
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

    # OpenClaw autonomous waiver intelligence at 8:30 AM daily
    scheduler.add_job(
        _openclaw_morning_job,
        CronTrigger(hour=8, minute=30, timezone=timezone),
        id="openclaw_morning",
        name="OpenClaw Autonomous Morning Workflow",
        replace_existing=True,
    )

    # Odds monitor — poll every 5 minutes for line movements
    odds_monitor_interval = get_float_env("ODDS_MONITOR_INTERVAL_MIN", "5")
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

    # Performance Sentinel — MAE, drawdown, pytest health check at 5:00 AM
    # (30 min after daily snapshot, ensuring fresh data is available)
    scheduler.add_job(
        _nightly_health_check_job,
        CronTrigger(hour=5, minute=0, timezone=timezone),
        id="nightly_health_check",
        name="Performance Sentinel Health Check",
        replace_existing=True,
    )

    # Morning Briefing — summarize today's slate at 7 AM ET (after ratings are fresh)
    scheduler.add_job(
        _morning_briefing_job,
        CronTrigger(hour=7, minute=0, timezone=timezone),
        id="morning_briefing",
        name="Morning Slate Briefing",
        replace_existing=True,
    )

    # End-of-day results — 11 PM ET
    scheduler.add_job(
        _end_of_day_results_job,
        CronTrigger(hour=23, minute=0, timezone=timezone),
        id="end_of_day_results",
        name="End-of-Day Results Summary",
        replace_existing=True,
    )

    # Tournament bracket release notifier — runs daily 6 PM ET, Mar 14–20
    scheduler.add_job(
        _tournament_bracket_job,
        CronTrigger(hour=18, minute=0, timezone=timezone),
        id="tournament_bracket_notifier",
        name="Tournament Bracket Release Notifier",
        replace_existing=True,
    )

    # Weekly model parameter recalibration — Sunday 5 AM ET
    # Note: recalibration and sentinel both run at 5:00 AM; they are independent.
    scheduler.add_job(
        _weekly_recalibration_job,
        CronTrigger(day_of_week="sun", hour=5, minute=0, timezone=timezone),
        id="weekly_recalibration",
        name="Weekly Model Parameter Recalibration",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "Scheduler started: nightly@%02d:00, outcomes every 2h + daily@04:00, "
        "lines every 30min, odds monitor every %dmin, snapshot@04:30, "
        "sentinel@05:00, briefing@07:00, ratings prewarm@%02d:00, recalibration@sun05:00, "
        "end_of_day@23:00, tournament_bracket@18:00 %s",
        nightly_hour, odds_monitor_interval, ratings_prewarm_hour, timezone,
    )

    # Ingestion Orchestrator -- gated by env var (off by default, safe for Railway)
    global _ingestion_orchestrator
    if os.getenv("ENABLE_INGESTION_ORCHESTRATOR", "false").lower() == "true":
        from backend.services.daily_ingestion import DailyIngestionOrchestrator
        _ingestion_orchestrator = DailyIngestionOrchestrator()
        _ingestion_orchestrator.start()
        logger.info("DailyIngestionOrchestrator started")
    else:
        logger.info("DailyIngestionOrchestrator disabled (ENABLE_INGESTION_ORCHESTRATOR not set)")

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

    # Pre-warm reanalysis cache for OddsMonitor
    try:
        db = SessionLocal()
        try:
            from backend.models import Prediction
            from backend.services.odds_monitor import get_odds_monitor
            from backend.services.recalibration import load_current_params
            
            today_utc = datetime.utcnow().date()
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

                        # full_analysis.inputs has no "game" key — reconstruct
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
    # APScheduler is in-memory — Railway deploys after 3 AM reset the next-run time to
    # tomorrow, so without this check today's games would never get analysed.
    async def _startup_catchup():
        from pytz import timezone as _tz
        from datetime import datetime as _dt
        et = _tz("America/New_York")
        now_et = _dt.now(et)
        nightly_cutoff = int(os.getenv("NIGHTLY_CRON_HOUR", "3"))
        if now_et.hour < nightly_cutoff:
            # Before 3 AM ET — nightly job hasn't run yet today, nothing to catch up
            return
        # Check if today already has predictions.
        # Use ET date (not UTC) — the nightly job can run before midnight UTC
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
            logger.info("Lifespan: Today has %d predictions — no catch-up needed", count)
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

# CORS — reads ALLOWED_ORIGINS env var (comma-separated) or falls back to wildcard.
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


# ============================================================================
# SCHEDULED JOB
# ============================================================================

async def nightly_job():
    """Main nightly analysis job — runs at 3 AM ET by default."""
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
    """Performance Sentinel — model accuracy, portfolio drawdown, pytest suite — runs at 5:00 AM."""
    try:
        result = run_nightly_health_check()
        logger.info("Sentinel health check complete: %s", result)
    except Exception:
        logger.exception("Sentinel health check job failed")


def _morning_briefing_job():
    """Morning slate briefing at 7 AM ET — sends Discord notification with today's bets."""
    import time as _time
    t0 = _time.monotonic()
    db = SessionLocal()
    try:
        from datetime import date as _date
        from backend.models import Prediction, Game
        from backend.services.scout import generate_morning_briefing_narrative
        from backend.services.discord_simple import send_morning_brief

        today = _date.today()
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
    """End-of-day results summary at 11 PM ET — settles today's bets and sends Discord recap."""
    from datetime import date
    from backend.services.discord_simple import send_eod_results

    try:
        db = SessionLocal()
        try:
            today = date.today()
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


def _tournament_bracket_job():
    """
    Tournament bracket release notifier.

    Runs daily from March 15–17. On the day the bracket is released (Selection
    Sunday), The Odds API will start returning NCAAB tournament games. When we
    first detect ≥4 new NCAAB games with tips scheduled Mar 18–19 (First Four),
    we send a Discord notification and mark the bracket as notified so we only
    fire once.

    Uses a sentinel file (.bracket_notified_{year}) to prevent duplicate sends.
    """
    import requests as _requests
    from datetime import date, timezone as _timezone
    from backend.services.discord_notifier import _post, _bot_token

    try:
        today = date.today()
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
            "footer": {"text": "CBB Edge — Tournament Mode Active"},
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
                        "title": "Bracket Projection — Monte Carlo",
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
                                    f" — {a['upset_prob'] * 100:.0f}% upset chance"
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


@app.get("/api/tournament/bracket-projection")
async def get_bracket_projection(
    n_sims: int = Query(default=10000, ge=1000, le=50000),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Monte Carlo NCAA Tournament bracket projection.

    Uses the V9.1 tournament module (composite KenPom+BartTorvik ratings,
    round-specific SD multipliers, historical seed upset blending).

    Falls back to data/bracket_2026.json when BALLDONTLIE_API_KEY is not set,
    so the endpoint always returns a result during tournament weeks.

    Query parameters:
      n_sims: Number of Monte Carlo simulations (1,000 – 50,000; default 10,000).
    """
    import json as _json
    from pathlib import Path as _Path
    from backend.tournament.matchup_predictor import TournamentTeam
    from backend.tournament.bracket_simulator import run_monte_carlo

    REGIONS = ["east", "south", "west", "midwest"]
    BRACKET_JSON = _Path(__file__).resolve().parent.parent / "data" / "bracket_2026.json"

    # --- 1. Build bracket from pre-built JSON (always available) ---
    if not BRACKET_JSON.exists():
        raise HTTPException(
            status_code=503,
            detail="bracket_2026.json not found. Re-deploy or run build_bracket_from_db.py.",
        )

    with open(BRACKET_JSON, encoding="utf-8") as _f:
        raw = _json.load(_f)

    bracket: dict = {}
    for region in REGIONS:
        if region not in raw:
            continue
        bracket[region] = [
            TournamentTeam(
                name=t["name"],
                seed=t["seed"],
                region=region,
                composite_rating=t.get("composite_rating", 0.0),
                kp_adj_em=t.get("kp_adj_em"),
                bt_adj_em=t.get("bt_adj_em"),
                pace=t.get("pace", 68.0),
                three_pt_rate=t.get("three_pt_rate", 0.35),
                def_efg_pct=t.get("def_efg_pct", 0.50),
                conference=t.get("conference", ""),
                tournament_exp=t.get("tournament_exp", 0.70),
            )
            for t in raw[region]
        ]

    if len(bracket) < 4:
        raise HTTPException(
            status_code=503,
            detail=f"bracket_2026.json is incomplete ({len(bracket)}/4 regions).",
        )

    # --- 2. Run Monte Carlo simulation ---
    try:
        results = run_monte_carlo(bracket, n_sims=n_sims, n_workers=2, base_seed=42)
    except Exception as exc:
        logger.error("Bracket Monte Carlo failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")

    # --- 3. Build response ---
    all_teams = [t for teams in bracket.values() for t in teams]
    seed_map = {t.name: t.seed for t in all_teams}
    region_map = {t.name: t.region for t in all_teams}

    sorted_champ = sorted(results.championship.items(), key=lambda x: -x[1])
    projected_champion = sorted_champ[0][0] if sorted_champ else None

    top_f4 = sorted(results.final_four.items(), key=lambda x: -x[1])[:4]
    projected_final_four = [t for t, _ in top_f4]

    upset_alerts = [
        {
            "team": t.name,
            "seed": t.seed,
            "region": t.region,
            "r64_win_prob": round(results.round_of_32.get(t.name, 0) * 100, 1),
        }
        for t in all_teams
        if t.seed >= 10 and results.round_of_32.get(t.name, 0) >= 0.35
    ]

    advancement_probs = {
        t: {
            "seed": seed_map.get(t, 0),
            "region": region_map.get(t, ""),
            "r32_pct": round(results.round_of_32.get(t, 0) * 100, 1),
            "s16_pct": round(results.sweet_sixteen.get(t, 0) * 100, 1),
            "e8_pct": round(results.elite_eight.get(t, 0) * 100, 1),
            "f4_pct": round(results.final_four.get(t, 0) * 100, 1),
            "runner_up_pct": round(results.runner_up.get(t, 0) * 100, 1),
            "champion_pct": round(results.championship.get(t, 0) * 100, 1),
        }
        for t in results.championship
    }

    return {
        "n_sims": results.n_sims,
        "data_source": "bracket_2026.json (V9.1 composite ratings)",
        "projected_champion": projected_champion,
        "projected_final_four": projected_final_four,
        "upset_alerts": upset_alerts,
        "advancement_probs": advancement_probs,
        "avg_upsets_per_tournament": round(results.avg_upsets_per_tournament, 1),
        "avg_championship_margin": round(results.avg_championship_margin, 1),
    }


def _daily_snapshot_job():
    """Generate daily performance snapshot, adjust source weights, and run alert checks — runs at 4:30 AM."""
    db = SessionLocal()
    try:
        generate_daily_snapshot(db)
        # Dynamic ensemble weight adjustment — runs after snapshot so today's
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
    """Check for significant line movements vs. active bets — runs every 30 min."""
    try:
        results = check_line_movements()
        logger.info("Line monitor check: %s", results)
    except Exception as exc:
        logger.error("Line monitor job failed: %s", exc, exc_info=True)


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


def _pybaseball_fetch_job():
    """Daily 7:30 AM refresh of pybaseball FanGraphs leaderboard caches."""
    try:
        from backend.fantasy_baseball.pybaseball_loader import fetch_all_statcast_leaderboards
        import backend.fantasy_baseball.statcast_loader as _sc
        fetch_all_statcast_leaderboards(year=2025)
        _sc._batter_cache.clear()
        _sc._pitcher_cache.clear()
        _sc._loaded_at = 0.0
        logger.info("pybaseball daily refresh complete")
    except Exception as e:
        logger.error("pybaseball fetch job failed: %s", e)


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
    """Daily 8:30 AM OpenClaw autonomous waiver intelligence workflow."""
    try:
        from backend.services.mcmc_simulator import MCMCWeeklySimulator
        from backend.services.waiver_edge_detector import WaiverEdgeDetector
        from backend.services.discord_router import DiscordRouter
        from backend.services.openclaw_autonomous import OpenClawAutonomousLoop
        sim = MCMCWeeklySimulator(n_sims=1000)
        loop = OpenClawAutonomousLoop(WaiverEdgeDetector(mcmc_simulator=sim), DiscordRouter())
        loop.run_morning_workflow()
    except Exception as e:
        logger.error("OpenClaw morning job failed: %s", e)


async def _opener_attack_job():
    """
    Run analysis when overnight opening lines are posted.

    Bookmakers hang openers with lower limits because their models are
    vulnerable — they rely on sharp action to shape the line.  Running
    analysis immediately catches early value before the line moves.

    When BET verdicts are found, Discord alerts fire immediately so bets
    can be placed at the opening price rather than waiting for the 3 AM
    or 7 AM jobs (by which time sharp money may have moved the line).
    """
    logger.info("Opening line attack triggered — running analysis on fresh openers")
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
                "Opener attack: %d bets found in %d games (%.1fs) — sending Discord alert",
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
    """Get all UPCOMING predictions from the latest analysis batch, deduplicated by game."""
    today_utc = datetime.utcnow().date()
    now_utc = datetime.utcnow()

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
    today_utc = datetime.utcnow().date()

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
    starting_bankroll = get_effective_bankroll(db)
    max_daily_pct     = get_float_env("MAX_DAILY_EXPOSURE_PCT", "20.0")
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
    cutoff = datetime.utcnow() - timedelta(days=days)
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
        # with at least 3 bets — possible mapping issue signal
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
    cutoff = datetime.utcnow() - timedelta(days=history_days)
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
        # Exclude cancelled/displaced bets (outcome=-1) from settled view
        query = query.filter(BetLog.outcome.isnot(None), BetLog.outcome != -1)
    elif status == "cancelled":
        query = query.filter(BetLog.outcome == -1)
    elif status == "placed":
        # "placed" — only bets that were actually executed (not paper trades)
        query = query.filter(BetLog.executed.is_(True), BetLog.outcome != -1)
    else:
        # "all" — exclude internal cancelled/displaced bookkeeping rows
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


@app.post("/admin/discord/test-simple")
async def discord_test_simple(user: str = Depends(verify_admin_api_key)):
    """Test the simplified Discord notification system (admin only)."""
    from backend.services.discord_simple import send_test_message
    
    success = send_test_message()
    if success:
        return {"status": "ok", "message": "Discord test messages sent to all configured channels"}
    raise HTTPException(
        status_code=502, 
        detail="Discord test failed — check DISCORD_BOT_TOKEN and channel IDs"
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

    cutoff = datetime.utcnow() - timedelta(days=days)

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

    By default dry_run=true — set dry_run=false to actually delete.
    """
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

    # Group by (game_id, calendar_date) — keep lowest id (first-created)
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
            detail=f"Game {game_id} has {len(bet_logs)} bet log(s) — use ?force=true to delete anyway"
        )

    bet_logs_deleted = 0
    if bet_logs and force:
        bet_logs_deleted = db.query(BetLog).filter(BetLog.game_id == game_id).delete()
        logger.warning("Admin %s force-deleting %d bet log(s) for game %d", user, bet_logs_deleted, game_id)

    predictions_deleted = db.query(Prediction).filter(Prediction.game_id == game_id).delete()
    closing_deleted = db.query(ClosingLine).filter(ClosingLine.game_id == game_id).delete()
    db.delete(game)
    db.commit()
    logger.info("Admin %s deleted game %d (%s @ %s) — %d predictions, %d closing lines, %d bet logs removed",
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


@app.get("/admin/ingestion/status")
async def ingestion_status(user: str = Depends(verify_api_key)):
    """Return per-job status for the DailyIngestionOrchestrator, or disabled signal."""
    if _ingestion_orchestrator is None:
        return {"enabled": False, "jobs": {}}
    return {"enabled": True, "jobs": _ingestion_orchestrator.get_status()}


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
    monitor = get_odds_monitor()
    return monitor.get_status()


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
                (__import__("datetime").datetime.utcnow() - service.cache_timestamp).total_seconds() / 3600
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
# FANTASY BASEBALL — PUBLIC API ENDPOINTS
# ============================================================================

@app.get("/api/fantasy/draft-board")
async def fantasy_draft_board(
    position: Optional[str] = Query(None, description="Filter by position (C, 1B, SP, RP, OF, ...)"),
    player_type: Optional[str] = Query(None, description="Filter by type: batter or pitcher"),
    tier_max: Optional[int] = Query(None, description="Only show players at or below this tier"),
    limit: int = Query(200, ge=1, le=500),
    user: str = Depends(verify_api_key),
):
    """
    Return the full ranked fantasy draft board (Steamer/ZiPS projections).
    Players are sorted by rank (ascending). Optionally filter by position/type/tier.
    """
    try:
        from backend.fantasy_baseball.player_board import get_board
        board = get_board()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Player board unavailable: {exc}")

    if position:
        pos_upper = position.upper()
        board = [p for p in board if pos_upper in p.get("positions", [])]
    if player_type:
        board = [p for p in board if p.get("type", "").lower() == player_type.lower()]
    if tier_max is not None:
        board = [p for p in board if p.get("tier", 99) <= tier_max]

    board = board[:limit]
    return {"count": len(board), "players": board}


@app.get("/api/fantasy/player/{player_id}")
async def fantasy_player_detail(
    player_id: str,
    user: str = Depends(verify_api_key),
):
    """Return a single player's full detail from the draft board."""
    try:
        from backend.fantasy_baseball.player_board import get_board
        board = get_board()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Player board unavailable: {exc}")

    player = next((p for p in board if p["id"] == player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")
    return player


@app.post("/api/fantasy/draft-session")
async def create_draft_session(
    my_draft_position: int = Query(..., ge=1, le=20),
    num_teams: int = Query(12, ge=4, le=20),
    num_rounds: int = Query(23, ge=10, le=30),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Create a new live-draft tracking session. Keepers are pre-inserted."""
    import secrets
    from backend.models import FantasyDraftSession, FantasyDraftPick
    from backend.fantasy_baseball.player_board import get_board, MY_KEEPERS

    session_key = secrets.token_hex(8)
    session = FantasyDraftSession(
        session_key=session_key,
        my_draft_position=my_draft_position,
        num_teams=num_teams,
        num_rounds=num_rounds,
        current_pick=1,
        is_active=True,
    )
    db.add(session)
    db.flush()  # get session.id

    # Pre-insert keeper picks (pick_number=0 sentinel = pre-draft keeper)
    board_by_id = {p["id"]: p for p in get_board()}
    for player_id, keeper_round in MY_KEEPERS.items():
        player = board_by_id.get(player_id)
        if not player:
            continue
        db.add(FantasyDraftPick(
            session_id=session.id,
            pick_number=0,
            round_number=keeper_round,
            drafter_position=my_draft_position,
            is_my_pick=True,
            player_id=player_id,
            player_name=player["name"],
            player_team=player.get("team"),
            player_positions=player.get("positions"),
            player_tier=player.get("tier"),
            player_adp=player.get("adp"),
            player_z_score=player.get("z_score"),
        ))

    db.commit()
    db.refresh(session)
    return {
        "session_key": session_key,
        "my_draft_position": my_draft_position,
        "num_teams": num_teams,
        "num_rounds": num_rounds,
        "keepers_preloaded": list(MY_KEEPERS.keys()),
        "message": "Draft session created with keepers pre-loaded.",
    }


@app.post("/api/fantasy/draft-session/{session_key}/pick")
async def record_draft_pick(
    session_key: str,
    player_id: str = Query(...),
    drafter_position: int = Query(..., ge=1, le=20),
    is_my_pick: bool = Query(False),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Record a pick in a live draft session and return recommendations for the next pick.
    """
    from backend.models import FantasyDraftSession, FantasyDraftPick
    from backend.fantasy_baseball.player_board import get_board
    from backend.fantasy_baseball.draft_engine import DraftState, DraftRecommender

    session = db.query(FantasyDraftSession).filter_by(
        session_key=session_key, is_active=True
    ).with_for_update().first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found or inactive")

    board = get_board()
    player = next((p for p in board if p["id"] == player_id), None)
    if player is None:
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found in board")

    pick_number = session.current_pick
    round_number = ((pick_number - 1) // session.num_teams) + 1

    pick = FantasyDraftPick(
        session_id=session.id,
        pick_number=pick_number,
        round_number=round_number,
        drafter_position=drafter_position,
        is_my_pick=is_my_pick,
        player_id=player["id"],
        player_name=player["name"],
        player_team=player.get("team"),
        player_positions=player.get("positions"),
        player_tier=player.get("tier"),
        player_adp=player.get("adp"),
        player_z_score=player.get("z_score"),
    )
    db.add(pick)
    session.current_pick = pick_number + 1
    db.commit()

    # Build recommendations for the next pick
    drafted_ids = {p.player_id for p in session.picks}
    try:
        state = DraftState(
            my_draft_position=session.my_draft_position,
            num_teams=session.num_teams,
            num_rounds=session.num_rounds,
        )
        state.pick_number = session.current_pick
        state.my_picks = [p.player_id for p in session.picks if p.is_my_pick]
        recs = DraftRecommender(state, board).recommend(top_n=5, drafted_ids=drafted_ids)
    except Exception:
        recs = [p for p in board if p["id"] not in drafted_ids][:5]

    return {
        "message": "Pick recorded",
        "pick_number": pick_number,
        "player_name": player["name"],
        "is_my_pick": is_my_pick,
        "next_recommendations": recs,
    }


@app.get("/api/fantasy/draft-session/{session_key}")
async def get_draft_session(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Return the current state and all picks for a draft session."""
    from backend.models import FantasyDraftSession

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    picks = [
        {
            "pick_number": p.pick_number,
            "round": p.round_number,
            "drafter_position": p.drafter_position,
            "is_my_pick": p.is_my_pick,
            "is_keeper": p.pick_number == 0,
            "player_id": p.player_id,
            "player_name": p.player_name,
            "player_team": p.player_team,
            "player_positions": p.player_positions,
            "player_tier": p.player_tier,
            "player_adp": p.player_adp,
        }
        for p in sorted(session.picks, key=lambda x: x.pick_number)
    ]
    my_picks = [p for p in picks if p["is_my_pick"]]

    return {
        "session_key": session_key,
        "my_draft_position": session.my_draft_position,
        "num_teams": session.num_teams,
        "num_rounds": session.num_rounds,
        "current_pick": session.current_pick,
        "total_picks": len(picks),
        "my_picks_count": len(my_picks),
        "is_active": session.is_active,
        "picks": picks,
        "my_picks": my_picks,
    }


@app.delete("/api/fantasy/draft-session/{session_key}")
async def delete_draft_session(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Delete a draft session and all its picks (for resetting a test session)."""
    from backend.models import FantasyDraftSession, FantasyDraftPick

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    db.query(FantasyDraftPick).filter_by(session_id=session.id).delete()
    db.delete(session)
    db.commit()
    return {"message": "Draft session deleted", "session_key": session_key}


@app.get("/api/fantasy/draft-sessions")
async def list_draft_sessions(
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """List all draft sessions (active and inactive) so you can find a session key to reset or delete."""
    from backend.models import FantasyDraftSession, FantasyDraftPick

    sessions = db.query(FantasyDraftSession).order_by(FantasyDraftSession.created_at.desc()).all()
    result = []
    for s in sessions:
        pick_count = db.query(FantasyDraftPick).filter_by(session_id=s.id).count()
        my_pick_count = db.query(FantasyDraftPick).filter_by(session_id=s.id, is_my_pick=True).count()
        result.append({
            "session_key": s.session_key,
            "my_draft_position": s.my_draft_position,
            "num_teams": s.num_teams,
            "num_rounds": s.num_rounds,
            "current_pick": s.current_pick,
            "total_picks_recorded": pick_count,
            "my_picks_recorded": my_pick_count,
            "is_active": s.is_active,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "updated_at": s.updated_at.isoformat() if s.updated_at else None,
        })
    return {"sessions": result, "count": len(result)}


@app.post("/api/fantasy/draft-session/{session_key}/reset")
async def reset_draft_session(
    session_key: str,
    my_draft_position: Optional[int] = Query(None, ge=1, le=20,
        description="Update draft position (e.g. 6). Omit to keep existing."),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Reset a draft session: clears all picks and resets current_pick to 1.
    Optionally updates my_draft_position (use this to correct a test session
    before the real draft begins).

    The session itself is preserved — use DELETE to fully remove it.
    """
    from backend.models import FantasyDraftSession, FantasyDraftPick

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    picks_deleted = db.query(FantasyDraftPick).filter_by(session_id=session.id).delete()
    session.current_pick = 1
    session.is_active = True
    if my_draft_position is not None:
        session.my_draft_position = my_draft_position
    db.commit()

    return {
        "message": "Draft session reset",
        "session_key": session_key,
        "picks_cleared": picks_deleted,
        "my_draft_position": session.my_draft_position,
        "ready_for_draft": True,
    }


@app.get("/api/fantasy/draft-session/value-board")
async def fantasy_value_board(
    drafted_ids: str = Query("", description="Comma-separated player IDs already drafted"),
    position: Optional[str] = Query(None, description="Filter by position (C, 1B, SP, RP, OF, ...)"),
    player_type: Optional[str] = Query(None, description="batter or pitcher"),
    tier_max: Optional[int] = Query(None, description="Exclude players below this tier"),
    limit: int = Query(100, ge=1, le=300),
    user: str = Depends(verify_api_key),
):
    """
    Advanced-metrics value board — ranks AVAILABLE players by a composite
    value_score that combines projection z-scores, ADP gaps, Statcast quality,
    and regression signals (BUY_LOW / BREAKOUT / AVOID).

    Pass drafted_ids as a comma-separated list of player IDs to filter out
    already-drafted players.  Falls back to standard z-score ranking if the
    Statcast overlay fails.
    """
    import copy
    from backend.fantasy_baseball.player_board import get_board
    from backend.fantasy_baseball.draft_analytics import (
        inject_advanced_analytics,
        compute_value_score,
    )

    try:
        raw_board = get_board()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Player board unavailable: {exc}")

    # Shallow-copy so we don't mutate the singleton cache
    board = [dict(p) for p in raw_board]

    # Attempt analytics overlay — silently degrades if CSVs missing
    inject_advanced_analytics(board)

    # Filter out drafted players + all league keepers (hardcoded fallback)
    from backend.fantasy_baseball.player_board import ALL_LEAGUE_KEEPERS
    exclude = ALL_LEAGUE_KEEPERS | {pid.strip() for pid in drafted_ids.split(",") if pid.strip()}
    board = [p for p in board if p["id"] not in exclude]

    # Optional filters
    if position:
        pos_upper = position.upper()
        board = [p for p in board if pos_upper in p.get("positions", [])]
    if player_type:
        board = [p for p in board if p.get("type", "").lower() == player_type.lower()]
    if tier_max is not None:
        board = [p for p in board if 0 < p.get("tier", 99) <= tier_max]

    # Sort by value_score descending
    board.sort(key=compute_value_score, reverse=True)

    # Attach computed value_score to each result for transparency
    for p in board:
        p["value_score"] = round(compute_value_score(p), 3)

    board = board[:limit]
    return {
        "count": len(board),
        "analytics_overlay": any(bool(p.get("statcast")) for p in board),
        "players": board,
    }


@app.post("/api/fantasy/draft-session/{session_key}/sync-yahoo")
async def sync_draft_from_yahoo(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Poll Yahoo's draftresults endpoint and sync any new picks into the session.

    Call this every 15-30 seconds during the live draft to keep drafted_ids
    current without manual entry.  Safe to call repeatedly — only new picks
    are recorded.

    Returns: picks_synced count, total picks in Yahoo, your current roster.
    """
    from backend.models import FantasyDraftSession, FantasyDraftPick
    from backend.fantasy_baseball.player_board import get_board
    from backend.fantasy_baseball.yahoo_client import YahooFantasyClient, YahooAuthError, YahooAPIError

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    try:
        client = YahooFantasyClient()
    except YahooAuthError as exc:
        raise HTTPException(status_code=401, detail=f"Yahoo auth not configured: {exc}")

    try:
        yahoo_picks = client.get_draft_results()
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo API error: {exc}")

    if not yahoo_picks:
        return {"picks_synced": 0, "total_yahoo_picks": 0, "message": "Draft not started or no picks yet"}

    # Build a lookup of player_key -> board player for name/position resolution
    board = get_board()
    # Yahoo player_key is like "mlb.p.12345" — we match against board by name
    # since we don't store Yahoo player keys in our board.
    # Build a name-normalised lookup from the board.
    def _norm(s: str) -> str:
        return s.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")

    board_by_id = {p["id"]: p for p in board}
    board_by_name = {_norm(p["name"]): p for p in board}

    # Already-recorded pick numbers in this session
    existing_pick_numbers = {
        pk.pick_number for pk in db.query(FantasyDraftPick).filter_by(session_id=session.id).all()
    }

    picks_synced = 0
    for raw in yahoo_picks:
        pick_num = int(raw.get("pick", 0))
        if pick_num == 0 or pick_num in existing_pick_numbers:
            continue

        round_num = int(raw.get("round", ((pick_num - 1) // session.num_teams) + 1))
        player_key = raw.get("player_key", "")

        # Determine if this is our pick based on pick number and draft position
        # Pick order: round 1 & 2 linear, round 3+ snake (see draft_engine.py)
        from backend.fantasy_baseball.draft_engine import build_full_pick_order
        full_order = build_full_pick_order(session.num_teams, session.num_rounds)
        pick_pos = 0
        if pick_num <= len(full_order):
            _, _, pick_pos = full_order[pick_num - 1]
        is_my_pick = (pick_pos == session.my_draft_position)

        # Resolve player name — try to fetch from Yahoo, fall back to player_key as ID
        player_name = player_key
        player_id = _norm(player_key)
        positions: list = []
        player_type = "batter"
        tier = 0
        adp = 999.0

        try:
            yahoo_player = client.get_player(player_key)
            player_name = yahoo_player.get("name") or player_key
            positions = yahoo_player.get("positions") or []
            player_id = _norm(player_name)
        except Exception:
            pass  # Use player_key as fallback name — pick still recorded

        # Try to enrich from our board
        board_match = board_by_id.get(player_id) or board_by_name.get(player_id)
        if board_match:
            player_type = board_match.get("type", "batter")
            tier = board_match.get("tier", 0)
            adp = board_match.get("adp", 999.0)
            if not positions:
                positions = board_match.get("positions", [])

        pick_record = FantasyDraftPick(
            session_id=session.id,
            pick_number=pick_num,
            round_number=round_num,
            player_id=player_id,
            player_name=player_name,
            player_team=yahoo_player.get("team", "") if "yahoo_player" in dir() else "",
            player_positions=",".join(positions),
            player_type=player_type,
            player_tier=tier,
            player_adp=adp,
            is_my_pick=is_my_pick,
        )
        db.add(pick_record)
        picks_synced += 1

    if picks_synced:
        # Advance session pick counter
        session.current_pick = max(session.current_pick, len(yahoo_picks) + 1)
        db.commit()

    my_picks = [
        pk.player_name for pk in
        db.query(FantasyDraftPick).filter_by(session_id=session.id, is_my_pick=True)
        .order_by(FantasyDraftPick.pick_number).all()
    ]

    return {
        "picks_synced": picks_synced,
        "total_yahoo_picks": len(yahoo_picks),
        "session_current_pick": session.current_pick,
        "my_roster_so_far": my_picks,
    }


@app.post("/api/fantasy/draft-session/{session_key}/sync-keepers")
async def sync_keepers_pre_draft(
    session_key: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Pre-draft keeper sweep. Call once when the Yahoo draft room opens (~30 min
    before picks start). Fetches all 12 teams' current rosters — any player
    already on a roster is a keeper — and inserts them as pick_number=0
    sentinel rows so the value-board and available-player pool are clean before
    the first pick.

    Safe to call repeatedly: existing keeper rows are not duplicated.

    Returns: keepers_found (total), keepers_inserted (new), keepers_skipped
             (already present), my_keepers (our own).
    """
    from backend.models import FantasyDraftSession, FantasyDraftPick
    from backend.fantasy_baseball.player_board import get_board
    from backend.fantasy_baseball.yahoo_client import YahooFantasyClient, YahooAuthError, YahooAPIError

    session = db.query(FantasyDraftSession).filter_by(session_key=session_key).first()
    if session is None:
        raise HTTPException(status_code=404, detail="Draft session not found")

    try:
        client = YahooFantasyClient()
    except YahooAuthError as exc:
        raise HTTPException(status_code=401, detail=f"Yahoo auth not configured: {exc}")

    try:
        my_team_key = client.get_my_team_key()
        all_teams = client.get_all_teams()
        all_rosters = client.get_all_rosters()
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Yahoo API error: {exc}")

    board = get_board()

    def _norm(s: str) -> str:
        return s.lower().replace(" ", "_").replace(".", "").replace("'", "").replace("-", "_")

    board_by_name = {_norm(p["name"]): p for p in board}

    # Players already in this session as keepers (pick_number=0)
    existing = {
        row.player_id
        for row in db.query(FantasyDraftPick.player_id)
        .filter_by(session_id=session.id, pick_number=0)
        .all()
    }

    # Map team_key -> draft position (1-indexed by league order)
    team_pos = {t["team_key"]: (i + 1) for i, t in enumerate(all_teams)}

    keepers_found = 0
    keepers_inserted = 0
    keepers_skipped = 0
    my_keepers: list[str] = []

    for team_key, roster in all_rosters.items():
        is_my_team = (team_key == my_team_key)
        drafter_pos = team_pos.get(team_key, 0)

        for player in roster:
            keepers_found += 1
            raw_name = player.get("name") or ""
            player_id = _norm(raw_name) if raw_name else player.get("player_key", "unknown")

            if player_id in existing:
                keepers_skipped += 1
                if is_my_team:
                    my_keepers.append(raw_name)
                continue

            board_match = board_by_name.get(player_id)
            positions = player.get("positions") or (board_match.get("positions") if board_match else []) or []

            db.add(FantasyDraftPick(
                session_id=session.id,
                pick_number=0,
                round_number=None,
                drafter_position=drafter_pos,
                is_my_pick=is_my_team,
                player_id=player_id,
                player_name=raw_name,
                player_team=player.get("team"),
                player_positions=",".join(positions) if isinstance(positions, list) else positions,
                player_tier=board_match.get("tier") if board_match else None,
                player_adp=board_match.get("adp") if board_match else None,
                player_z_score=board_match.get("z_score") if board_match else None,
            ))
            existing.add(player_id)
            keepers_inserted += 1
            if is_my_team:
                my_keepers.append(raw_name)

    db.commit()

    return {
        "status": "ok",
        "keepers_found": keepers_found,
        "keepers_inserted": keepers_inserted,
        "keepers_skipped": keepers_skipped,
        "my_keepers": my_keepers,
        "player_pool_ready": True,
        "message": (
            f"Keeper sweep complete. {keepers_inserted} new keepers loaded "
            f"across {len(all_rosters)} teams. Player pool clean for draft."
        ),
    }


@app.get("/api/fantasy/lineup/{lineup_date}", response_model=DailyLineupResponse)
async def get_fantasy_lineup_recommendations(
    lineup_date: str,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Return daily lineup recommendations for a given date.
    Wired to DailyLineupOptimizer for implied runs and park factors.
    """
    from datetime import date as date_type
    try:
        ld = date_type.fromisoformat(lineup_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")

    # Fetch Yahoo roster for player-specific rankings (best-effort)
    _lineup_roster: list = []
    try:
        _lineup_client = YahooFantasyClient()
        _lineup_roster = _lineup_client.get_roster()
    except Exception as _exc:
        logger.warning("Could not fetch Yahoo roster for lineup optimizer: %s", _exc)

    # Build projections from player board (best-effort)
    _lineup_projections: list = []
    if _lineup_roster:
        try:
            from backend.fantasy_baseball.player_board import get_or_create_projection as _get_lineup_proj
            _lineup_projections = [_get_lineup_proj(p) for p in _lineup_roster]
        except Exception as _exc:
            logger.warning("Could not load player board projections for lineup: %s", _exc)

    optimizer = get_lineup_optimizer()

    # --- Constraint-aware batter lineup ---
    lineup_warnings: list[str] = []
    batters: list[LineupPlayerOut] = []
    if _lineup_roster and _lineup_projections:
        try:
            solved_slots, lineup_warnings = optimizer.solve_lineup(
                roster=_lineup_roster,
                projections=_lineup_projections,
                game_date=lineup_date,
            )
            batters = [
                LineupPlayerOut(
                    player_id=s.player_name,
                    name=s.player_name,
                    team=s.player_team,
                    position=s.positions[0] if s.positions else "?",
                    implied_runs=round(s.implied_runs, 2),
                    park_factor=round(s.park_factor, 3),
                    lineup_score=round(s.lineup_score, 3),
                    status="START" if s.slot != "BN" else "BENCH",
                    assigned_slot=s.slot,
                    has_game=s.has_game,
                )
                for s in solved_slots
            ]
        except Exception as _exc:
            logger.warning("solve_lineup failed, falling back to score-rank: %s", _exc)
            lineup_warnings.append(f"Constraint solver unavailable: {_exc}")

    # Fallback: if solver produced nothing, use raw score ranking
    if not batters:
        report = optimizer.build_daily_report(
            game_date=lineup_date,
            roster=_lineup_roster or None,
            projections=_lineup_projections or None,
        )
        batters = [
            LineupPlayerOut(
                player_id=str(b.get("player_id", b.get("name", ""))),
                name=b.get("name", ""),
                team=b.get("team", ""),
                position=(b.get("positions") or ["OF"])[0],
                implied_runs=float(b.get("implied_runs", 0)),
                park_factor=float(b.get("park_factor", 1.0)),
                lineup_score=float(b.get("score", 0)),
                status="START" if i < 9 else "BENCH",
                assigned_slot=None,
            )
            for i, b in enumerate(report.get("batter_rankings", []))
        ]
    else:
        report = optimizer.build_daily_report(game_date=lineup_date)

    # --- Pitcher start detection ---
    pitchers: list[StartingPitcherOut] = []
    try:
        flagged_pitchers = optimizer.flag_pitcher_starts(
            roster=_lineup_roster,
            game_date=lineup_date,
        )
        # SP with a game today start; SP without sit; RP always listed
        sp_with_start = [p for p in flagged_pitchers if p["pitcher_slot"] == "SP" and p["has_start"]]
        sp_no_start = [p for p in flagged_pitchers if p["pitcher_slot"] == "SP" and not p["has_start"]]
        rps = [p for p in flagged_pitchers if p["pitcher_slot"] == "RP"]

        for p in sp_with_start + rps:
            pitchers.append(StartingPitcherOut(
                player_id=p.get("player_key") or p.get("name", ""),
                name=p.get("name", ""),
                team=p.get("team", ""),
                opponent_implied_runs=0.0,
                park_factor=1.0,
                sp_score=0.0,
                status="START",
            ))
        for p in sp_no_start:
            pitchers.append(StartingPitcherOut(
                player_id=p.get("player_key") or p.get("name", ""),
                name=p.get("name", ""),
                team=p.get("team", ""),
                opponent_implied_runs=0.0,
                park_factor=1.0,
                sp_score=0.0,
                status="NO_START",
            ))
        if sp_no_start:
            lineup_warnings.append(
                f"{len(sp_no_start)} SP(s) have no start today: "
                + ", ".join(p.get('name', '') for p in sp_no_start)
            )
    except Exception as _exc:
        logger.warning("flag_pitcher_starts failed: %s", _exc)

    games_list = report.get("games", [])
    if len(games_list) == 0:
        lineup_warnings.insert(0,
            "Odds API unavailable or no games today -- using projection-only scoring "
            "(all teams at league-average 4.5 runs). Lineup ranked by projected stats only."
        )

    # Detect active slot gaps — warn if suspiciously few active batters/pitchers
    _BENCH_SLOTS = {"BN", None}
    _batter_active = [b for b in batters if b.assigned_slot not in _BENCH_SLOTS]
    _pitcher_active = [p for p in pitchers if p.status == "START"]
    if len(_batter_active) < 6:
        lineup_warnings.append(
            f"Only {len(_batter_active)} active batter slots filled -- "
            "check bench/IL for promotable players."
        )
    if len(_pitcher_active) < 2:
        lineup_warnings.append(
            f"Only {len(_pitcher_active)} active pitcher slots filled -- "
            "consider streaming a SP."
        )

    return DailyLineupResponse(
        date=ld,
        batters=batters,
        pitchers=pitchers,
        games_count=len(games_list),
        no_games_today=len(games_list) == 0,
        lineup_warnings=lineup_warnings,
    )


@app.get("/api/fantasy/waiver", response_model=WaiverWireResponse)
async def get_fantasy_waiver_recommendations(
    position: Optional[str] = Query(None),
    sort: str = Query("need_score"),
    min_z_score: Optional[float] = Query(None),
    max_percent_owned: float = Query(90.0),
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=10, le=100),
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Return waiver wire recommendations.
    Pulls real free agents and waiver players from Yahoo API.
    """
    from datetime import date as date_type, timedelta
    from backend.schemas import WaiverPlayerOut

    today = date_type.today()
    week_end = today + timedelta(days=(6 - today.weekday()))

    matchup_opponent = "TBD"
    top_available: list = []
    two_start_pitchers: list = []
    category_deficits: list = []
    _closer_alert: Optional[str] = None
    _il_info: dict = {"used": 0, "total": 2, "available": 0}
    _faab_balance: Optional[float] = None

    # Maps Yahoo category display names to player board cat_scores keys.
    # cat_scores are already z-normalized with direction baked in (ERA/WHIP negative = good).
    _YAHOO_CAT_TO_BOARD = {
        "R": "r", "H": "h", "HR": "hr", "RBI": "rbi", "TB": "tb",
        "SB": "nsb", "AVG": "avg", "OPS": "ops",
        "W": "w", "L": "l", "K": "k_pit", "SO": "k_pit",
        "SV": "nsv", "ERA": "era", "WHIP": "whip",
        "QS": "qs", "K9": "k9", "K/9": "k9",
    }

    try:
        client = YahooFantasyClient()
        my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
        if not my_team_key:
            try:
                my_team_key = client.get_my_team_key()
            except Exception:
                my_team_key = ""

        # Fetch roster for IL capacity (best-effort; empty list if unavailable)
        my_roster: list[dict] = []
        if my_team_key:
            try:
                my_roster = client.get_roster()
            except Exception:
                pass

        # Fetch FAAB balance (best-effort; None if not a FAAB league or error)
        try:
            _faab_balance = client.get_faab_balance()
        except Exception:
            pass

        # Get free agents with pagination + position filter
        _fa_start = (page - 1) * per_page
        free_agents = client.get_free_agents(
            position=position or "", start=_fa_start, count=per_page
        )
        # Try to get matchup opponent
        try:
            matchups = client.get_scoreboard()
            for m in matchups:
                if isinstance(m, dict):
                    teams = m.get("teams", {})
                    team_keys_in_matchup = []
                    team_names = {}
                    # Handle both Yahoo response shapes
                    if isinstance(teams, list):
                        raw_entries = [item.get("team", []) for item in teams if isinstance(item, dict)]
                    elif isinstance(teams, dict):
                        count_t = int(teams.get("count", 0))
                        raw_entries = [teams.get(str(ti), {}).get("team", []) for ti in range(count_t)]
                    else:
                        raw_entries = []
                    for t_entry in raw_entries:
                        t_meta = {}
                        if isinstance(t_entry, list):
                            for sub in t_entry:
                                if isinstance(sub, list):
                                    for item in sub:
                                        if isinstance(item, dict):
                                            t_meta.update(item)
                                elif isinstance(sub, dict):
                                    t_meta.update(sub)
                        tk = t_meta.get("team_key", "")
                        tn = t_meta.get("name", "")
                        team_keys_in_matchup.append(tk)
                        team_names[tk] = tn
                    if my_team_key in team_keys_in_matchup:
                        for tk in team_keys_in_matchup:
                            if tk != my_team_key:
                                matchup_opponent = team_names.get(tk, "TBD")
                        break
        except Exception:
            pass

        # Build category deficits from scoreboard stats.
        # This must run before _to_waiver_player so need_score can be computed per player.
        # Failures here are swallowed — waiver list still returns, just without need scoring.
        try:
            from backend.schemas import CategoryDeficitOut
            if matchup_opponent != "TBD":
                # Re-fetch scoreboard to get per-category stats
                matchups2 = client.get_scoreboard()
                # Build stat_id → display name map for readable labels
                sid_map: dict[str, str] = {}
                try:
                    settings2 = client.get_league_settings()
                    stat_cats2 = (
                        settings2
                        .get("settings", [{}])[0]
                        .get("stat_categories", {})
                        .get("stats", [])
                    )
                    for entry in stat_cats2:
                        if isinstance(entry, dict):
                            s2 = entry.get("stat", {})
                            sid2 = str(s2.get("stat_id", ""))
                            abbr2 = s2.get("display_name") or s2.get("abbreviation") or s2.get("name") or sid2
                            if sid2:
                                sid_map[sid2] = abbr2
                except Exception:
                    pass
                lower_better = {"ERA", "WHIP"}
                for m2 in matchups2:
                    if not isinstance(m2, dict):
                        continue
                    teams2 = m2.get("teams", {})
                    team_stats_map: dict[str, dict] = {}
                    if isinstance(teams2, list):
                        team_entries2 = [item["team"] for item in teams2 if isinstance(item, dict) and "team" in item]
                    elif isinstance(teams2, dict):
                        count2 = int(teams2.get("count", 0))
                        team_entries2 = [teams2.get(str(ti2), {}).get("team", []) for ti2 in range(count2)]
                    else:
                        continue
                    for entry2 in team_entries2:
                        t_meta2: dict = {}
                        stats_raw2: list = []
                        items2 = entry2 if isinstance(entry2, list) else [entry2]
                        for sub2 in items2:
                            if isinstance(sub2, list):
                                for it2 in sub2:
                                    if isinstance(it2, dict):
                                        t_meta2.update(it2)
                            elif isinstance(sub2, dict):
                                if "team_stats" in sub2:
                                    inner2 = sub2["team_stats"].get("stats", [])
                                    if isinstance(inner2, list):
                                        stats_raw2 = inner2
                                else:
                                    t_meta2.update(sub2)
                        tk2 = t_meta2.get("team_key", "")
                        sd2: dict = {}
                        for st2 in stats_raw2:
                            if isinstance(st2, dict):
                                stobj = st2.get("stat", {})
                                if isinstance(stobj, dict):
                                    sid_k = str(stobj.get("stat_id", ""))
                                    key2 = sid_map.get(sid_k, sid_k)
                                    try:
                                        sd2[key2] = float(stobj.get("value", 0) or 0)
                                    except (TypeError, ValueError):
                                        sd2[key2] = 0.0
                        team_stats_map[tk2] = sd2
                    if my_team_key not in team_stats_map:
                        continue
                    my_stats = team_stats_map[my_team_key]
                    opp_key = next((k for k in team_stats_map if k != my_team_key), None)
                    if not opp_key:
                        continue
                    opp_stats = team_stats_map[opp_key]
                    for cat, my_val in my_stats.items():
                        opp_val = opp_stats.get(cat, 0.0)
                        if lower_better.issuperset({cat}):
                            deficit = my_val - opp_val  # positive = we are worse (higher ERA/WHIP)
                            winning = my_val < opp_val
                        else:
                            deficit = opp_val - my_val  # positive = we are behind
                            winning = my_val > opp_val
                        category_deficits.append(
                            CategoryDeficitOut(
                                category=cat,
                                my_total=my_val,
                                opponent_total=opp_val,
                                deficit=deficit,
                                winning=winning,
                            )
                        )
                    break  # found our matchup
        except Exception:
            category_deficits = []

        # Universal projection lookup — board players get rich projections,
        # call-ups / undrafted players get a conservative position-baseline proxy.
        from backend.fantasy_baseball.player_board import get_or_create_projection as _get_proj

        def _hot_cold_flag(cat_contributions: dict) -> Optional[str]:
            """Simple hot/cold based on category contribution z-scores."""
            scores = list(cat_contributions.values())
            if not scores:
                return None
            avg = sum(scores) / len(scores)
            if avg > 0.4:
                return "HOT"
            if avg < -0.3:
                return "COLD"
            return None

        def _to_waiver_player(p: dict) -> WaiverPlayerOut:
            positions = p.get("positions") or []
            name = (p.get("name") or "").strip()
            board_player = _get_proj(p)  # always returns something
            _raw_nsv = round(float((board_player.get("proj") or {}).get("nsv", 0.0)), 1)

            need_score = 0.0
            contributions: dict = {}

            if category_deficits:
                # In-season: weight player's per-category z-scores by our matchup deficits.
                cat_scores = board_player.get("cat_scores", {})
                for cd in category_deficits:
                    if cd.winning or cd.deficit <= 0:
                        continue
                    board_key = _YAHOO_CAT_TO_BOARD.get(cd.category)
                    if not board_key or board_key not in cat_scores:
                        continue
                    player_z = cat_scores[board_key]
                    if player_z <= 0:
                        continue
                    opp_total = abs(cd.opponent_total) or 1.0
                    deficit_weight = cd.deficit / opp_total
                    contribution = deficit_weight * player_z
                    contributions[cd.category] = round(contribution, 3)
                    need_score += contribution
            else:
                # Pre-season or no active matchup: use overall board z_score.
                need_score = board_player.get("z_score", 0.0)

            # Hot/cold flag derived from contribution scores (best-effort)
            _hc: Optional[str] = None
            try:
                _hc = _hot_cold_flag(contributions) if contributions else _hot_cold_flag(
                    {k: v for k, v in (board_player.get("cat_scores") or {}).items()}
                )
            except Exception:
                pass

            # Status / injury note pass-through from Yahoo metadata
            _status = p.get("status") or None
            _injury_note = p.get("injury_note") or None

            return WaiverPlayerOut(
                player_id=p.get("player_key") or "",
                name=name,
                team=p.get("team") or "",
                position=positions[0] if positions else "?",
                need_score=round(need_score, 3),
                category_contributions=contributions,
                owned_pct=p.get("percent_owned", 0.0),
                starts_this_week=p.get("starts_this_week", 0),
                projected_saves=_raw_nsv,
                hot_cold=_hc,
                status=_status,
                injury_note=_injury_note,
            )

        # Build + filter + sort top_available
        top_available = [_to_waiver_player(p) for p in free_agents]
        if min_z_score is not None:
            top_available = [p for p in top_available if p.need_score >= min_z_score]
        top_available = [p for p in top_available if p.owned_pct <= max_percent_owned]
        if sort == "percent_owned":
            top_available.sort(key=lambda x: x.owned_pct, reverse=True)
        else:
            top_available.sort(key=lambda x: x.need_score, reverse=True)

        # Two-start pitchers: SPs from free agents with 2+ probable starts this week
        import difflib as _difflib_starts
        from datetime import date as _dt, timedelta as _td
        _today = date_type.today()
        _week_end_ts = _today + _td(days=6)

        def _fetch_mlb_probable_starts(start_date: str, end_date: str) -> dict:
            """Return {pitcher_full_name_lower: starts_count} via public MLB Stats API (6h cached)."""
            import httpx as _httpx
            _now = datetime.utcnow()
            if _STARTS_CACHE.get("data") and _STARTS_CACHE.get("fetched_at"):
                age_h = (_now - _STARTS_CACHE["fetched_at"]).total_seconds() / 3600
                if age_h < 6:
                    return _STARTS_CACHE["data"]
            url = (
                "https://statsapi.mlb.com/api/v1/schedule"
                f"?sportId=1&startDate={start_date}&endDate={end_date}"
                "&gameType=R&hydrate=probablePitcher"
            )
            try:
                resp = _httpx.get(url, timeout=8.0)
                resp.raise_for_status()
            except Exception as _e:
                logger.warning("MLB Stats API schedule fetch failed: %s", _e)
                return _STARTS_CACHE.get("data") or {}
            starts: dict = {}
            for date_entry in resp.json().get("dates", []):
                for game in date_entry.get("games", []):
                    for side in ("home", "away"):
                        pitcher = (game.get("teams", {})
                                   .get(side, {})
                                   .get("probablePitcher", {}))
                        pname = (pitcher.get("fullName") or "").strip().lower()
                        if pname:
                            starts[pname] = starts.get(pname, 0) + 1
            _STARTS_CACHE["data"] = starts
            _STARTS_CACHE["fetched_at"] = _now
            return starts

        starts_map = _fetch_mlb_probable_starts(
            _today.strftime("%Y-%m-%d"), _week_end_ts.strftime("%Y-%m-%d")
        )

        sp_fas = [p for p in free_agents if "SP" in (p.get("positions") or [])]
        two_start_pitchers_raw = []
        for _sp in sp_fas[:50]:
            _sp_name = (_sp.get("name") or "").strip().lower()
            _starts = starts_map.get(_sp_name, 0)
            if _starts == 0 and starts_map:
                _best = max(
                    starts_map.keys(),
                    key=lambda k: _difflib_starts.SequenceMatcher(None, _sp_name, k).ratio(),
                    default=None,
                )
                if _best and _difflib_starts.SequenceMatcher(None, _sp_name, _best).ratio() >= 0.90:
                    _starts = starts_map[_best]
            if _starts >= 2:
                _sp["starts_this_week"] = _starts
                two_start_pitchers_raw.append(_to_waiver_player(_sp))
        two_start_pitchers = sorted(
            two_start_pitchers_raw, key=lambda x: x.need_score, reverse=True
        )[:5]

        # Closer alert — FAs with meaningful saves projection (nsv cat_score z > 0.5)
        _closer_fas = [f for f in top_available if f.category_contributions.get("nsv", 0) > 0.5]
        _closer_alert: Optional[str] = None
        if len(_closer_fas) == 0:
            _closer_alert = "NO_CLOSERS"
        elif len(_closer_fas) < 2:
            _closer_alert = "LOW_CLOSERS"

        # IL capacity
        from backend.services.waiver_edge_detector import il_capacity_info as _il_cap
        _il_info = _il_cap(my_roster) if my_roster else {"used": 0, "total": 2, "available": 0}

    except YahooAuthError as exc:
        logger.error("Waiver endpoint -- Yahoo auth error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo auth failed — refresh token may be expired. ({exc})",
        ) from exc
    except YahooAPIError as exc:
        logger.error("Waiver endpoint -- Yahoo API error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo API error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Waiver endpoint failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Unexpected error fetching waiver data: {exc}",
        ) from exc

    from backend.schemas import PaginationOut
    return WaiverWireResponse(
        week_end=week_end,
        matchup_opponent=matchup_opponent,
        category_deficits=category_deficits,
        top_available=top_available,
        two_start_pitchers=two_start_pitchers,
        pagination=PaginationOut(
            page=page,
            per_page=per_page,
            has_next=len(top_available) == per_page,
        ),
        closer_alert=_closer_alert,
        il_slots_used=_il_info["used"],
        il_slots_available=_il_info["available"],
        faab_balance=_faab_balance,
    )


@app.get("/api/fantasy/waiver/recommendations")
async def get_waiver_recommendations(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    """
    Actionable ADD/DROP/ADD_DROP recommendations.

    Algorithm:
    1. Fetch my roster + category deficits (matchup-aware if in-season)
    2. Fetch top free agents
    3. For each deficit category, find best available FA who helps
    4. Pair each ADD with the weakest same-position player on my roster
    5. Return ranked list of RosterMoveRecommendation

    Returns at most 5 recommendations ranked by need_score.
    """
    from datetime import date as date_type, timedelta
    from backend.schemas import (
        RosterMoveRecommendation, WaiverRecommendationsResponse,
        WaiverPlayerOut, CategoryDeficitOut,
    )
    from backend.fantasy_baseball.player_board import get_or_create_projection as _get_proj
    from backend.fantasy_baseball.statcast_loader import (
        build_statcast_signals, statcast_need_score_boost,
    )

    today = date_type.today()
    week_end = today + timedelta(days=(6 - today.weekday()))

    matchup_opponent = "TBD"
    category_deficits: list = []
    recommendations: list[RosterMoveRecommendation] = []

    try:
        client = YahooFantasyClient()
        my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
        if not my_team_key:
            try:
                my_team_key = client.get_my_team_key()
            except Exception:
                my_team_key = ""

        # Fetch my current roster
        my_roster: list[dict] = []
        if my_team_key:
            try:
                my_roster = client.get_roster()
            except Exception:
                pass

        # Fetch category deficits via scoreboard
        try:
            from backend.schemas import CategoryDeficitOut as _CDOut
            matchups = client.get_scoreboard()
            for m in matchups:
                if not isinstance(m, dict):
                    continue
                teams = m.get("teams", {})
                raw_entries = []
                if isinstance(teams, list):
                    raw_entries = [item.get("team", []) for item in teams if isinstance(item, dict)]
                elif isinstance(teams, dict):
                    count_t = int(teams.get("count", 0))
                    raw_entries = [teams.get(str(ti), {}).get("team", []) for ti in range(count_t)]
                team_keys_in_matchup = []
                team_stats: dict = {}
                team_names: dict = {}
                for t_entry in raw_entries:
                    t_meta: dict = {}
                    t_stat_cats: dict = {}
                    if isinstance(t_entry, list):
                        for sub in t_entry:
                            if isinstance(sub, list):
                                for item in sub:
                                    if isinstance(item, dict):
                                        t_meta.update(item)
                            elif isinstance(sub, dict):
                                t_meta.update(sub)
                                if "team_stats" in sub:
                                    stats_block = sub["team_stats"].get("stats", [])
                                    for s_entry in stats_block:
                                        if isinstance(s_entry, dict):
                                            s = s_entry.get("stat", {})
                                            t_stat_cats[s.get("stat_id")] = s.get("value")
                    tk = t_meta.get("team_key", "")
                    tn = t_meta.get("name", "")
                    team_keys_in_matchup.append(tk)
                    team_stats[tk] = t_stat_cats
                    team_names[tk] = tn
                if my_team_key in team_keys_in_matchup:
                    for tk in team_keys_in_matchup:
                        if tk != my_team_key:
                            matchup_opponent = team_names.get(tk, "TBD")
                    break
        except Exception:
            pass

        # Get free agents (larger pool for better recommendations)
        free_agents = client.get_free_agents(count=40)

        # Build scored FA list with universal projections
        def _score_fa(p: dict) -> WaiverPlayerOut:
            positions = p.get("positions") or []
            name = (p.get("name") or "").strip()
            bp = _get_proj(p)
            need_score = bp.get("z_score", 0.0)
            return WaiverPlayerOut(
                player_id=p.get("player_key") or "",
                name=name,
                team=p.get("team") or "",
                position=positions[0] if positions else "?",
                need_score=round(need_score, 3),
                category_contributions=bp.get("cat_scores", {}) if bp else {},
                owned_pct=p.get("percent_owned", 0.0),
                starts_this_week=0,
            )

        scored_fas = sorted(
            [_score_fa(p) for p in free_agents],
            key=lambda x: x.need_score,
            reverse=True,
        )

        # Non-blocking Statcast coverage log
        try:
            from backend.fantasy_baseball.pybaseball_loader import (
                log_statcast_coverage,
                load_pybaseball_batters,
                load_pybaseball_pitchers,
            )
            fa_names = [p.get("name", "") for p in free_agents]
            _sc = {**load_pybaseball_batters(2025), **load_pybaseball_pitchers(2025)}
            if _sc:
                log_statcast_coverage(fa_names, _sc, "waiver FAs")
        except Exception:
            pass

        # Build my roster with projections for drop candidate evaluation
        my_roster_scored: list[dict] = []
        for rp in my_roster:
            bp = _get_proj(rp)
            my_roster_scored.append({
                "name": (rp.get("name") or "").strip(),
                "player_key": rp.get("player_key") or "",
                "positions": rp.get("positions") or [],
                "z_score": bp.get("z_score", 0.0),
                "is_proxy": bp.get("is_proxy", False),
                "cat_scores": bp.get("cat_scores") or {},
                "starts_this_week": int(rp.get("starts_this_week", 1)),
                "status": rp.get("status"),
                "injury_note": rp.get("injury_note"),
                "is_undroppable": bool(rp.get("is_undroppable", 0)),
            })

        # Statuses that mean a player is occupying an IL/NA slot, not an active slot.
        # DTD players are borderline but still fill their active slot, so they remain droppable.
        _IL_STATUSES = {"IL", "IL10", "IL60", "NA", "OUT"}

        def _weakest_safe_to_drop(target_positions: list[str]) -> dict | None:
            """Return the weakest droppable player at the given positions, with 1-active-cover protection."""
            candidates = [
                rp for rp in my_roster_scored
                if not rp.get("is_undroppable", False)
                and any(pos in rp["positions"] for pos in target_positions)
            ]
            if not candidates:
                return None
            active = [p for p in candidates if p.get("status") not in _IL_STATUSES]
            if len(active) == 1:
                # Only one active at this position — protected, do not drop
                return None
            if len(active) == 0:
                # All injured — position already uncovered, anyone is droppable
                return min(candidates, key=lambda x: x.get("z_score") or 0.0)
            return min(active, key=lambda x: x.get("z_score") or 0.0)

        def _fmt_signals(signals: list[str], reg_delta: float, is_pitcher: bool) -> str:
            """Format FA Statcast signals as a rationale suffix."""
            parts = []
            if "BUY_LOW" in signals:
                metric = "xERA" if is_pitcher else "xwOBA"
                parts.append(f"BUY_LOW ({metric} delta={reg_delta:+.3f})")
            if "BREAKOUT" in signals:
                parts.append("BREAKOUT candidate")
            if not parts:
                return ""
            return " [" + "; ".join(parts) + "]"

        def _fmt_drop_signals(signals: list[str], reg_delta: float, is_pitcher: bool) -> str:
            """Format drop candidate Statcast signals as a rationale suffix."""
            parts = []
            if "SELL_HIGH" in signals:
                metric = "xERA" if is_pitcher else "xwOBA"
                parts.append(f"drop is SELL_HIGH ({metric} delta={reg_delta:+.3f})")
            if "HIGH_INJURY_RISK" in signals:
                parts.append("drop has HIGH_INJURY_RISK")
            if not parts:
                return ""
            return " [" + "; ".join(parts) + "]"

        # Generate recommendations for top FAs
        seen_drops: set[str] = set()
        for fa in scored_fas[:15]:
            if len(recommendations) >= 5:
                break

            fa_positions = [fa.position] if fa.position != "?" else []
            # What generic position group does this player fill?
            if fa.position in ("SP", "RP", "P"):
                pos_group = ["SP", "RP", "P"]
                pos_label = "pitching"
            elif fa.position in ("C",):
                pos_group = ["C"]
                pos_label = "catcher"
            elif fa.position in ("SS",):
                pos_group = ["SS"]
                pos_label = "shortstop"
            elif fa.position in ("2B",):
                pos_group = ["2B"]
                pos_label = "second base"
            elif fa.position in ("3B",):
                pos_group = ["3B"]
                pos_label = "third base"
            elif fa.position in ("1B",):
                pos_group = ["1B"]
                pos_label = "first base"
            elif fa.position in ("OF", "LF", "CF", "RF"):
                pos_group = ["OF", "LF", "CF", "RF"]
                pos_label = "outfield"
            else:
                pos_group = fa_positions
                pos_label = fa.position

            # Statcast enrichment for FA (non-blocking)
            fa_is_pitcher = fa.position in ("SP", "RP", "P")
            fa_signals, fa_reg_delta = build_statcast_signals(
                fa.name, fa_is_pitcher, fa.owned_pct
            )
            statcast_boost = statcast_need_score_boost(fa_signals)
            adjusted_need = fa.need_score + statcast_boost

            drop_candidate = _weakest_safe_to_drop(pos_group)
            if not drop_candidate:
                # No roster player at same position — still recommend add if FA is strong
                if adjusted_need >= 2.0:
                    signal_text = _fmt_signals(fa_signals, fa_reg_delta, fa_is_pitcher)
                    recommendations.append(RosterMoveRecommendation(
                        action="ADD",
                        add_player=fa,
                        drop_player_name=None,
                        drop_player_position=None,
                        rationale=(
                            f"Add {fa.name} ({fa.position}, {fa.team}) — "
                            f"projected z={fa.need_score:+.1f}{signal_text}. "
                            f"No {pos_label} to drop suggested; check bench."
                        ),
                        category_targets=[],
                        need_score=round(adjusted_need, 3),
                        confidence=0.5 if not _get_proj({"player_key": fa.player_id, "name": fa.name}).get("is_proxy") else 0.3,
                        statcast_signals=fa_signals,
                        regression_delta=fa_reg_delta,
                    ))
                continue

            # Statcast injury risk on drop candidate (pitcher only — higher risk = easier drop)
            drop_is_pitcher = drop_candidate["positions"][0] in ("SP", "RP", "P") if drop_candidate["positions"] else False
            drop_signals, drop_reg_delta = build_statcast_signals(
                drop_candidate["name"], drop_is_pitcher
            )
            drop_score_adj = drop_candidate["z_score"] + statcast_need_score_boost(drop_signals)

            # Skip if drop candidate is still better than FA after Statcast adjustment
            if drop_score_adj >= adjusted_need:
                continue

            # Skip if we already suggested dropping this player
            if drop_candidate["name"] in seen_drops:
                continue

            gain = adjusted_need - drop_candidate["z_score"]
            if gain < 0.5:
                continue  # Not worth the move

            seen_drops.add(drop_candidate["name"])
            fa_proj = _get_proj({"player_key": fa.player_id, "name": fa.name, "positions": [fa.position]})
            is_proxy = fa_proj.get("is_proxy", False)
            confidence = 0.75 if not is_proxy else 0.45

            signal_text = _fmt_signals(fa_signals, fa_reg_delta, fa_is_pitcher)
            drop_signal_text = _fmt_drop_signals(drop_signals, drop_reg_delta, drop_is_pitcher)

            rationale = (
                f"Add {fa.name} ({fa.position}, {fa.team}, {fa.owned_pct:.0f}% owned), "
                f"drop {drop_candidate['name']} ({drop_candidate['positions'][0] if drop_candidate['positions'] else '?'}). "
                f"Net gain: {gain:+.1f} ({drop_candidate['z_score']:+.1f} -> {adjusted_need:+.1f}){signal_text}{drop_signal_text}."
            )
            if is_proxy:
                rationale += " [Call-up — projections estimated.]"
            if drop_candidate.get("status") in _IL_STATUSES:
                from backend.services.waiver_edge_detector import il_capacity_info as _il_cap2
                if my_roster and _il_cap2(my_roster)["available"] > 0:
                    rationale = (
                        f"[IL slot free — move {drop_candidate['name']} to IL first] " + rationale
                    )
                else:
                    rationale += f" [Note: {drop_candidate['name']} is {drop_candidate['status']} — consider IL slot if available]"

            # MCMC win-probability simulation (non-blocking, graceful fallback)
            _mcmc = {}
            try:
                from backend.fantasy_baseball.mcmc_simulator import simulate_roster_move as _sim_move
                _add_for_mcmc = {
                    "name": fa.name,
                    "positions": [fa.position],
                    "cat_scores": dict(fa.category_contributions),
                    "starts_this_week": fa.starts_this_week,
                }
                _mcmc = _sim_move(
                    my_roster=my_roster_scored,
                    opponent_roster=[],  # league-average opponent
                    add_player=_add_for_mcmc,
                    drop_player_name=drop_candidate["name"],
                    n_sims=1000,
                )
                if _mcmc.get("mcmc_enabled") and abs(_mcmc["win_prob_gain"]) >= 0.005:
                    wp_before_pct = round(_mcmc["win_prob_before"] * 100)
                    wp_after_pct = round(_mcmc["win_prob_after"] * 100)
                    wp_gain_pct = round(_mcmc["win_prob_gain"] * 100)
                    rationale += (
                        f" Win prob: {wp_before_pct}% -> {wp_after_pct}%"
                        f" ({wp_gain_pct:+d}%)."
                    )
            except Exception:
                pass

            recommendations.append(RosterMoveRecommendation(
                action="ADD_DROP",
                add_player=fa,
                drop_player_name=drop_candidate["name"],
                drop_player_position=drop_candidate["positions"][0] if drop_candidate["positions"] else "?",
                rationale=rationale,
                category_targets=[],
                need_score=round(gain, 3),
                confidence=confidence,
                statcast_signals=fa_signals,
                regression_delta=fa_reg_delta,
                win_prob_before=_mcmc.get("win_prob_before", 0.0),
                win_prob_after=_mcmc.get("win_prob_after", 0.0),
                win_prob_gain=_mcmc.get("win_prob_gain", 0.0),
                category_win_probs=_mcmc.get("category_win_probs_after", {}),
                mcmc_enabled=_mcmc.get("mcmc_enabled", False),
            ))

    except YahooAuthError as exc:
        logger.error("Waiver recommendations endpoint -- Yahoo auth error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo auth failed — refresh token may be expired. ({exc})",
        ) from exc
    except YahooAPIError as exc:
        logger.error("Waiver recommendations endpoint -- Yahoo API error: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Yahoo API error: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Waiver recommendations endpoint failed unexpectedly: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Unexpected error: {exc}",
        ) from exc

    return WaiverRecommendationsResponse(
        week_end=week_end,
        matchup_opponent=matchup_opponent,
        recommendations=sorted(recommendations, key=lambda r: r.need_score, reverse=True),
        category_deficits=category_deficits,
    )


@app.post("/api/fantasy/lineup")
async def save_fantasy_lineup(
    payload: dict,
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """
    Save a daily lineup. Body: {lineup_date, platform, positions, projected_points, notes}
    """
    from backend.models import FantasyLineup
    from datetime import date as date_type

    lineup_date_raw = payload.get("lineup_date")
    if not lineup_date_raw:
        raise HTTPException(status_code=422, detail="lineup_date is required")
    try:
        lineup_date = date_type.fromisoformat(str(lineup_date_raw))
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")

    platform = payload.get("platform", "yahoo")
    positions = payload.get("positions", {})
    if not positions:
        raise HTTPException(status_code=422, detail="positions dict is required")

    existing = db.query(FantasyLineup).filter_by(
        lineup_date=lineup_date, platform=platform
    ).first()
    if existing:
        existing.positions = positions
        existing.projected_points = payload.get("projected_points")
        existing.notes = payload.get("notes")
        db.commit()
        return {"message": "Lineup updated", "id": existing.id}

    lineup = FantasyLineup(
        lineup_date=lineup_date,
        platform=platform,
        positions=positions,
        projected_points=payload.get("projected_points"),
        notes=payload.get("notes"),
    )
    db.add(lineup)
    db.commit()
    db.refresh(lineup)
    return {"message": "Lineup saved", "id": lineup.id}


@app.get("/api/fantasy/saved-lineup/{lineup_date}")
async def get_fantasy_lineup(
    lineup_date: str,
    platform: str = Query("yahoo"),
    db: Session = Depends(get_db),
    user: str = Depends(verify_api_key),
):
    """Retrieve a previously saved DK/Yahoo lineup for a given date."""
    from backend.models import FantasyLineup
    from datetime import date as date_type

    try:
        ld = date_type.fromisoformat(lineup_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")

    lineup = db.query(FantasyLineup).filter_by(lineup_date=ld, platform=platform).first()
    if lineup is None:
        raise HTTPException(status_code=404, detail="No lineup saved for this date")

    return {
        "lineup_date": lineup_date,
        "platform": lineup.platform,
        "positions": lineup.positions,
        "projected_points": lineup.projected_points,
        "actual_points": lineup.actual_points,
        "notes": lineup.notes,
    }


# ============================================================================
# YAHOO FANTASY BASEBALL — ROSTER / MATCHUP / LINEUP APPLY
# ============================================================================

@app.get("/api/fantasy/yahoo-diag")
async def yahoo_diag(user: str = Depends(verify_api_key)):
    """
    Diagnostic endpoint — returns Yahoo config status without making API calls.
    Safe to call: reveals which env vars are present (values redacted).
    Remove after debugging is complete.
    """
    client_id = os.getenv("YAHOO_CLIENT_ID", "")
    client_secret = os.getenv("YAHOO_CLIENT_SECRET", "")
    refresh_token = os.getenv("YAHOO_REFRESH_TOKEN", "")
    access_token = os.getenv("YAHOO_ACCESS_TOKEN", "")
    league_id = os.getenv("YAHOO_LEAGUE_ID", "72586")

    # Try constructor
    constructor_ok = False
    constructor_error = None
    try:
        _c = YahooFantasyClient()
        constructor_ok = True
    except YahooAuthError as e:
        constructor_error = str(e)
    except Exception as e:
        constructor_error = f"Unexpected: {e}"

    # Try token refresh (only if constructor passed)
    token_ok = False
    token_error = None
    if constructor_ok:
        try:
            _c._ensure_token()
            token_ok = True
        except YahooAuthError as e:
            token_error = str(e)
        except Exception as e:
            token_error = f"Unexpected: {e}"

    return {
        "env_vars_present": {
            "YAHOO_CLIENT_ID": bool(client_id),
            "YAHOO_CLIENT_SECRET": bool(client_secret),
            "YAHOO_REFRESH_TOKEN": bool(refresh_token),
            "YAHOO_ACCESS_TOKEN": bool(access_token),
            "YAHOO_LEAGUE_ID": league_id,
        },
        "client_id_length": len(client_id),
        "client_secret_length": len(client_secret),
        "refresh_token_length": len(refresh_token),
        "constructor_ok": constructor_ok,
        "constructor_error": constructor_error,
        "token_refresh_ok": token_ok,
        "token_refresh_error": token_error,
    }


@app.get("/api/fantasy/roster", response_model=RosterResponse)
async def get_fantasy_roster(user: str = Depends(verify_api_key)):
    """
    Return the authenticated user's current Yahoo roster enriched with z-scores.
    Returns 503 if Yahoo credentials are not configured.
    """
    from backend.fantasy_baseball.player_board import get_or_create_projection

    try:
        client = YahooFantasyClient()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    try:
        raw_players = client.get_roster(team_key=team_key)
    except YahooAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YahooAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    players_out: list[RosterPlayerOut] = []
    for p in raw_players:
        name = p.get("name") or ""
        proj = get_or_create_projection(p) if name else {}
        players_out.append(
            RosterPlayerOut(
                player_key=p.get("player_key") or "",
                name=name,
                team=p.get("team"),
                positions=p.get("positions") or [],
                status=p.get("status"),
                injury_note=p.get("injury_note"),
                z_score=proj.get("z_score"),
                is_undroppable=bool(p.get("is_undroppable", 0)),
                is_proxy=bool(proj.get("is_proxy", False)),
                cat_scores=proj.get("cat_scores") or {},
                selected_position=p.get("selected_position"),
            )
        )

    return RosterResponse(
        team_key=team_key,
        players=players_out,
        count=len(players_out),
    )


@app.get("/api/fantasy/matchup", response_model=MatchupResponse)
async def get_fantasy_matchup(user: str = Depends(verify_api_key)):
    """
    Return current week's matchup: opponent name + category-by-category breakdown.
    Returns stub with empty stats if scoreboard is unavailable (pre-season).
    """
    try:
        client = YahooFantasyClient()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    # Dynamically resolve team key so the game-year prefix is always current.
    # Fall back to env var if the API call fails.
    my_team_key = os.getenv("YAHOO_TEAM_KEY", "")
    if not my_team_key:
        try:
            my_team_key = client.get_my_team_key()
        except Exception:
            my_team_key = ""
    logger.info("Matchup: resolved my_team_key=%s", my_team_key)

    _stub_my = MatchupTeamOut(team_key=my_team_key, team_name="My Team", stats={})
    _stub_opp = MatchupTeamOut(team_key="", team_name="TBD", stats={})

    # Build stat_id → abbreviation map from league settings (best-effort)
    stat_id_map: dict[str, str] = {}
    try:
        settings = client.get_league_settings()
        stat_cats = (
            settings
            .get("settings", [{}])[0]
            .get("stat_categories", {})
            .get("stats", [])
        )
        for entry in stat_cats:
            if isinstance(entry, dict):
                s = entry.get("stat", {})
                sid = str(s.get("stat_id", ""))
                abbr = s.get("display_name") or s.get("abbreviation") or s.get("name") or sid
                if sid:
                    stat_id_map[sid] = abbr
    except Exception:
        pass  # fall back to raw stat_ids

    try:
        matchups = client.get_scoreboard()
        logger.info("Matchup scoreboard: %d matchups returned", len(matchups))
        if matchups:
            logger.info("Matchup[0] keys: %s", list(matchups[0].keys()) if isinstance(matchups[0], dict) else type(matchups[0]))
    except (YahooAuthError, YahooAPIError) as exc:
        logger.error("Matchup scoreboard fetch failed: %s", exc)
        return MatchupResponse(my_team=_stub_my, opponent=_stub_opp, message="Scoreboard unavailable -- Yahoo API error.")

    if not matchups:
        logger.warning("Matchup scoreboard returned empty list -- my_team_key=%s", my_team_key)
        return MatchupResponse(my_team=_stub_my, opponent=_stub_opp, message="No matchup data yet -- season may be starting.")

    # Determine current week from first matchup
    week: int | None = None
    is_playoffs = False

    def _extract_team_stats(team_entry) -> tuple[str, str, dict]:
        """Return (team_key, team_name, stats_dict) from Yahoo team entry.
        Stats are keyed by abbreviation (e.g. 'R', 'HR') when stat_id_map is populated,
        otherwise by raw stat_id string."""
        t_meta: dict = {}
        stats_raw: list = []
        entries = team_entry if isinstance(team_entry, list) else [team_entry]
        for sub in entries:
            if isinstance(sub, list):
                for item in sub:
                    if isinstance(item, dict):
                        t_meta.update(item)
            elif isinstance(sub, dict):
                if "team_stats" in sub:
                    inner = sub["team_stats"].get("stats", [])
                    if isinstance(inner, list):
                        stats_raw = inner
                else:
                    t_meta.update(sub)
        stats_dict: dict = {}
        for s in stats_raw:
            if isinstance(s, dict):
                stat = s.get("stat", {})
                if isinstance(stat, dict):
                    sid = str(stat.get("stat_id", ""))
                    key = stat_id_map.get(sid, sid)  # resolve to abbrev or fall back to id
                    val = stat.get("value", "")
                    if key:
                        stats_dict[key] = val
        return (
            t_meta.get("team_key", ""),
            t_meta.get("name", ""),
            stats_dict,
        )

    for m in matchups:
        if not isinstance(m, dict):
            continue
        w = m.get("week")
        if w:
            try:
                week = int(w)
            except (TypeError, ValueError):
                pass
        is_playoffs = bool(m.get("is_playoffs", 0))

        teams = m.get("teams", {})
        team_data: list[tuple[str, str, dict]] = []
        # Handle both Yahoo response shapes: indexed dict or list
        if isinstance(teams, list):
            for item in teams:
                if isinstance(item, dict) and "team" in item:
                    team_data.append(_extract_team_stats(item["team"]))
        elif isinstance(teams, dict):
            count_t = int(teams.get("count", 0))
            for ti in range(count_t):
                entry = teams.get(str(ti), {}).get("team", [])
                team_data.append(_extract_team_stats(entry))

        # Check if our team is in this matchup
        my_entry = next((t for t in team_data if t[0] == my_team_key), None)
        if my_entry is None:
            continue

        opp_entry = next((t for t in team_data if t[0] != my_team_key), None)
        if opp_entry is None:
            opp_entry = ("", "Unknown", {})

        return MatchupResponse(
            week=week,
            my_team=MatchupTeamOut(
                team_key=my_entry[0],
                team_name=my_entry[1],
                stats=my_entry[2],
            ),
            opponent=MatchupTeamOut(
                team_key=opp_entry[0],
                team_name=opp_entry[1],
                stats=opp_entry[2],
            ),
            is_playoffs=is_playoffs,
        )

    # Our team not found in any matchup (pre-season / bye week / key mismatch)
    all_keys = []
    for m in matchups:
        teams = m.get("teams", {}) if isinstance(m, dict) else {}
        if isinstance(teams, dict):
            for ti in range(int(teams.get("count", 0))):
                entry = teams.get(str(ti), {}).get("team", [])
                tk, _, _ = _extract_team_stats(entry)
                all_keys.append(tk)
        elif isinstance(teams, list):
            for item in teams:
                if isinstance(item, dict) and "team" in item:
                    tk, _, _ = _extract_team_stats(item["team"])
                    all_keys.append(tk)
    logger.warning("My team key %s not found in scoreboard teams: %s", my_team_key, all_keys)
    return MatchupResponse(week=week, my_team=_stub_my, opponent=_stub_opp, message="Your team was not found in the current week's matchup.")


@app.put("/api/fantasy/lineup/apply")
async def apply_fantasy_lineup(
    payload: LineupApplyRequest,
    user: str = Depends(verify_api_key),
    auto_correct: bool = True,  # Enable game-aware auto-correction by default
):
    """
    Push a lineup to Yahoo Fantasy with game-aware validation.
    
    Body: {date?: YYYY-MM-DD, players: [{player_key, position}]}
    
    The resilient client will:
    - Validate all players have games today
    - Auto-correct players with no games (if auto_correct=true)
    - Apply circuit breaker protection for API failures
    """
    from datetime import datetime as _dt

    try:
        # Use ResilientYahooClient for game-aware validation and auto-correction
        client = ResilientYahooClient()
    except YahooAuthError as exc:
        raise HTTPException(
            status_code=503,
            detail="Yahoo not configured -- set YAHOO_REFRESH_TOKEN",
        ) from exc

    apply_date = payload.date or _dt.utcnow().strftime("%Y-%m-%d")
    team_key = os.getenv("YAHOO_TEAM_KEY", "469.l.72586.t.7")

    # Pre-check: warn (but don't block) if no MLB games exist for this date
    apply_warnings: list[str] = []
    try:
        mlb_games = get_lineup_optimizer().fetch_mlb_odds(apply_date)
        if not mlb_games:
            apply_warnings.append(
                f"No MLB games scheduled for {apply_date} -- applying lineup in preseason mode."
            )
    except Exception as _exc:
        logger.warning("Could not pre-check MLB schedule for %s: %s", apply_date, _exc)

    # Build optimized_lineup dict expected by ResilientYahooClient
    # The resilient client's set_lineup_resilient expects a specific format
    optimized_lineup = {
        "starters": [
            {
                "id": p.player_key,
                "player_key": p.player_key,
                "position": p.position,
                "positions": [p.position],  # For validation
            }
            for p in payload.players
        ]
    }

    try:
        # Use the resilient method with game-aware validation
        result = await client.set_lineup_resilient(
            team_id=team_key,
            optimized_lineup=optimized_lineup,
            auto_correct=auto_correct,
        )
    except Exception as exc:
        logger.exception("Error applying lineup with ResilientYahooClient")
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not result.success:
        # Build error response with suggestions
        detail = {
            "success": False,
            "error": result.errors,
            "warnings": result.warnings,
            "suggested_action": result.suggested_action,
            "retry_possible": result.retry_possible,
        }
        raise HTTPException(status_code=422, detail=detail)

    # Build success response
    applied_count = len(result.changes) if result.changes else len(payload.players)
    
    return {
        "success": True,
        "applied": applied_count,
        "skipped": 0,  # With auto-correct, we handle skips internally
        "date": apply_date,
        "warnings": apply_warnings + result.warnings,
        "changes": result.changes,
        "auto_correct": auto_correct,
    }


# ============================================================================
# YAHOO FANTASY BASEBALL — DEBUG ENDPOINTS
# ============================================================================

@app.get("/admin/yahoo/test")
async def yahoo_test(user: str = Depends(verify_admin_api_key)):
    """
    Test Yahoo API connectivity.
    Returns league name + authenticated team key.
    Requires YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REFRESH_TOKEN in env.
    """
    try:
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
        client = YahooFantasyClient()
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


@app.get("/admin/yahoo/roster-raw")
async def yahoo_roster_raw(user: str = Depends(verify_admin_api_key)):
    """
    Return the raw fantasy_content structure from Yahoo for your roster.
    Use this to inspect the exact shape Yahoo returns so parsing can be debugged.
    """
    try:
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
        client = YahooFantasyClient()
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
        from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
        client = YahooFantasyClient()
        roster = client.get_roster()
        return {"count": len(roster), "players": roster}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/fantasy/matchup/simulate")
async def simulate_matchup(
    payload: MatchupSimulateRequest,
    user: str = Depends(verify_api_key),
):
    """
    Monte Carlo simulation of a weekly H2H matchup.

    Pass my_roster and opponent_roster as lists of player dicts with:
      cat_scores: {hr: float, r: float, ...}  -- z-scores per category
      positions: [str]                          -- position eligibility
      starts_this_week: int                     -- pitcher starts (default 1)
      name: str

    Returns win probability and per-category breakdown.
    n_sims is capped at 5000 for latency safety.
    """
    from backend.fantasy_baseball.mcmc_simulator import simulate_weekly_matchup
    n = min(max(100, payload.n_sims), 5000)
    result = simulate_weekly_matchup(
        my_roster=payload.my_roster,
        opponent_roster=payload.opponent_roster,
        n_sims=n,
    )
    return result


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
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
