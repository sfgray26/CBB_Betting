"""
backend/schedulers/fantasy_scheduler.py -- Fantasy service APScheduler setup.

Contains ONLY fantasy/MLB ingestion jobs.
Never imports from betting_model or analysis -- GUARDIAN FREEZE applies.

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


def start_fantasy_scheduler() -> None:
    """Register fantasy service jobs and optionally start the ingestion orchestrator.

    The job_queue_processor always starts. DailyIngestionOrchestrator is gated
    by ENABLE_INGESTION_ORCHESTRATOR env var.
    """
    # Lazy imports during transition -- job functions live in main.py until Phase 7.
    from backend.main import (  # noqa: PLC0415
        _process_job_queue_job,
        _pybaseball_fetch_job,
        _statcast_daily_ingestion_job,
        _nightly_decision_resolution_job,
    )

    tz = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")

    # Job queue processor -- always runs in fantasy service
    scheduler.add_job(_process_job_queue_job, IntervalTrigger(seconds=5),
                      id="job_queue_processor", name="Async Job Queue Processor", replace_existing=True)

    # Pybaseball / FanGraphs leaderboard refresh -- 7:30 AM daily
    scheduler.add_job(_pybaseball_fetch_job, CronTrigger(hour=7, minute=30, timezone=tz),
                      id="fetch_pybaseball", name="Refresh pybaseball Statcast Leaderboards",
                      replace_existing=True)

    # Statcast daily ingestion + Bayesian projection updates -- 6:00 AM ET
    scheduler.add_job(_statcast_daily_ingestion_job, CronTrigger(hour=6, minute=0, timezone=tz),
                      id="statcast_daily_ingestion", name="Statcast Daily Ingestion + Bayesian Updates",
                      replace_existing=True)

    # OpenClaw autonomous waiver intelligence -- 8:30 AM daily
    # PAUSED (2026-04-21): Disabled to reduce noise while baseball module is
    # being implemented. Re-enable when OpenClaw is needed again.
    # scheduler.add_job(_openclaw_morning_job, CronTrigger(hour=8, minute=30, timezone=tz),
    #                   id="openclaw_morning", name="OpenClaw Autonomous Morning Workflow",
    #                   replace_existing=True)

    # Nightly fantasy decision resolution -- 11:59 PM ET
    scheduler.add_job(_nightly_decision_resolution_job, CronTrigger(hour=23, minute=59, timezone=tz),
                      id="nightly_decision_resolution", name="Nightly Fantasy Decision Resolution",
                      replace_existing=True)

    # Ingestion Orchestrator (FanGraphs RoS, Yahoo ADP, ensemble blend, freshness gate)
    if os.getenv("ENABLE_INGESTION_ORCHESTRATOR", "false").lower() == "true":
        from backend.services.daily_ingestion import DailyIngestionOrchestrator  # noqa: PLC0415
        orchestrator = DailyIngestionOrchestrator()
        orchestrator.start()
        scheduler._fantasy_orchestrator = orchestrator  # type: ignore[attr-defined]
        logger.info("DailyIngestionOrchestrator started")
    else:
        logger.info("DailyIngestionOrchestrator disabled (ENABLE_INGESTION_ORCHESTRATOR not set)")

    # MLB analysis service (gated -- not enabled during CBB season)
    if os.getenv("ENABLE_MLB_ANALYSIS", "false").lower() == "true":
        from backend.main import _run_mlb_analysis_job  # noqa: PLC0415
        scheduler.add_job(_run_mlb_analysis_job, CronTrigger(hour=9, minute=0, timezone=tz),
                          id="mlb_nightly_analysis", name="MLB Nightly Analysis", replace_existing=True)
        logger.info("MLB nightly analysis scheduled for 09:00 %s", tz)

    scheduler.start()
    logger.info("Fantasy scheduler started (%d jobs)", len(scheduler.get_jobs()))


def stop_fantasy_scheduler() -> None:
    """Stop fantasy scheduler and orchestrator gracefully."""
    orchestrator = getattr(scheduler, "_fantasy_orchestrator", None)
    if orchestrator is not None:
        orchestrator.stop()
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Fantasy scheduler stopped")
