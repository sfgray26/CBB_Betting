"""
backend/schedulers/edge_scheduler.py -- Edge service APScheduler setup.

Contains ONLY betting/analysis jobs. Fantasy jobs are in fantasy_scheduler.py.
Never import fantasy_baseball modules here.

Note: job functions are imported from backend.main during the strangler-fig
transition (Phase 1-5). In Phase 7 cleanup, they move here permanently.
"""
from __future__ import annotations

import logging
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler()


def start_edge_scheduler() -> None:
    """Register all edge service jobs and start the scheduler.

    Reads ENABLE_MAIN_SCHEDULER env var -- if false, logs and returns
    without starting (safe for UAT environments).
    """
    try:
        from backend.utils.deployment import main_scheduler_enabled
    except ImportError:
        def main_scheduler_enabled() -> bool:
            return os.getenv("ENABLE_MAIN_SCHEDULER", "true").lower() != "false"

    if not main_scheduler_enabled():
        logger.info("Edge scheduler disabled by ENABLE_MAIN_SCHEDULER=false")
        return

    # Lazy imports during transition -- job functions live in main.py until Phase 7.
    from backend.main import (  # noqa: PLC0415
        nightly_job,
        _update_outcomes_job,
        _capture_lines_job,
        _line_monitor_job,
        _daily_snapshot_job,
        _fetch_ratings_job,
        _odds_monitor_job,
        _opener_attack_job,
        _nightly_health_check_job,
        _morning_briefing_job,
        _end_of_day_results_job,
        _weekly_recalibration_job,
        _mlb_analysis_job,
    )
    from backend.utils.env_utils import get_float_env  # noqa: PLC0415

    nightly_hour = int(os.getenv("NIGHTLY_CRON_HOUR", "3"))
    tz = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")
    opener_enabled = os.getenv("OPENER_ATTACK_ENABLED", "true").lower() == "true"
    odds_monitor_interval = get_float_env("ODDS_MONITOR_INTERVAL_MIN", "5")
    ratings_prewarm_hour = int(os.getenv("RATINGS_PREWARM_HOUR", "8"))

    scheduler.add_job(nightly_job, CronTrigger(hour=nightly_hour, minute=0, timezone=tz),
                      id="nightly_analysis", name="Nightly Game Analysis", replace_existing=True)
    scheduler.add_job(_update_outcomes_job, IntervalTrigger(hours=2),
                      id="update_outcomes", name="Update Completed Game Outcomes", replace_existing=True)
    scheduler.add_job(_update_outcomes_job, CronTrigger(hour=4, minute=0, timezone=tz),
                      id="settle_games_daily", name="Daily Settle Completed Games", replace_existing=True)
    scheduler.add_job(_capture_lines_job, IntervalTrigger(minutes=30),
                      id="capture_closing_lines", name="Capture Closing Lines", replace_existing=True)
    scheduler.add_job(_line_monitor_job, IntervalTrigger(minutes=30),
                      id="line_monitor", name="Line Movement Monitor", replace_existing=True)
    scheduler.add_job(_daily_snapshot_job, CronTrigger(hour=4, minute=30, timezone=tz),
                      id="daily_snapshot", name="Daily Performance Snapshot", replace_existing=True)
    scheduler.add_job(_fetch_ratings_job, CronTrigger(hour=ratings_prewarm_hour, minute=0, timezone=tz),
                      id="fetch_ratings", name="Pre-warm Ratings Cache", replace_existing=True)
    scheduler.add_job(_odds_monitor_job, IntervalTrigger(minutes=odds_monitor_interval),
                      id="odds_monitor", name="Odds Line Movement Monitor", replace_existing=True)
    scheduler.add_job(_nightly_health_check_job, CronTrigger(hour=5, minute=0, timezone=tz),
                      id="nightly_health_check", name="Performance Sentinel Health Check", replace_existing=True)
    scheduler.add_job(_morning_briefing_job, CronTrigger(hour=7, minute=0, timezone=tz),
                      id="morning_briefing", name="Morning Slate Briefing", replace_existing=True)
    scheduler.add_job(_end_of_day_results_job, CronTrigger(hour=23, minute=0, timezone=tz),
                      id="end_of_day_results", name="End-of-Day Results Summary", replace_existing=True)
    scheduler.add_job(_weekly_recalibration_job, CronTrigger(day_of_week="sun", hour=5, minute=0, timezone=tz),
                      id="weekly_recalibration", name="Weekly Model Parameter Recalibration", replace_existing=True)
    # CBB tournament season is closed. Do not register the bracket notifier from
    # this standalone scheduler; MLB analysis remains active below.
    scheduler.add_job(_mlb_analysis_job, CronTrigger(hour=10, minute=0, timezone=tz),
                      id="mlb_analysis", name="MLB Nightly Analysis + Projection", replace_existing=True)

    if opener_enabled:
        opener_hour_1 = int(os.getenv("OPENER_ATTACK_HOUR_1", "22"))
        opener_min_1 = int(os.getenv("OPENER_ATTACK_MIN_1", "30"))
        opener_hour_2 = int(os.getenv("OPENER_ATTACK_HOUR_2", "0"))
        opener_min_2 = int(os.getenv("OPENER_ATTACK_MIN_2", "30"))
        scheduler.add_job(_opener_attack_job, CronTrigger(hour=opener_hour_1, minute=opener_min_1, timezone=tz),
                          id="opener_attack_2230", name="Opening Line Attack (10:30 PM)", replace_existing=True)
        scheduler.add_job(_opener_attack_job, CronTrigger(hour=opener_hour_2, minute=opener_min_2, timezone=tz),
                          id="opener_attack_0030", name="Opening Line Attack (12:30 AM)", replace_existing=True)
        logger.info("Opening line attack scheduler enabled (22:30, 00:30 %s)", tz)

    scheduler.start()
    logger.info("Edge scheduler started (%d jobs)", len(scheduler.get_jobs()))


def stop_edge_scheduler() -> None:
    """Stop the edge scheduler gracefully. Called from edge_app.py lifespan shutdown."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Edge scheduler stopped")
