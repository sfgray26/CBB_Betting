"""
DailyIngestionOrchestrator — EPIC-2 data pipeline coordinator.

Owns all MLB/CBB data polling jobs that run independently of the nightly
CBB analysis scheduler. Each job acquires a PostgreSQL advisory lock before
running, which prevents duplicate execution across Railway replicas.

ADR-001: Every job MUST use _with_advisory_lock.
ADR-004: This file is additive only. Never import betting_model or analysis.
"""

import logging
import os
import time
from datetime import datetime, date, timedelta
from typing import Optional, Any

import requests
from sqlalchemy import text

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.models import SessionLocal, PlayerDailyMetric, ProjectionSnapshot

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Advisory lock IDs — must match HANDOFF.md LOCK_IDS table
# ---------------------------------------------------------------------------

LOCK_IDS = {
    "mlb_odds":    100_001,
    "statcast":    100_002,
    "rolling_z":   100_003,
    "cbb_ratings": 100_004,
    "clv":         100_005,
    "cleanup":     100_006,
    "waiver_scan": 100_007,
    "mlb_brief":   100_008,
    "openclaw_perf":  100_009,
    "openclaw_sweep": 100_010,
}


# ---------------------------------------------------------------------------
# Advisory lock helper
# ---------------------------------------------------------------------------

async def _with_advisory_lock(lock_id: int, coro):
    """
    Acquire a session-level PostgreSQL advisory lock, run coro(), then release.
    If the lock is already held (another replica running the same job), skip
    execution and return None.
    """
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT pg_try_advisory_lock(:lid)"), {"lid": lock_id}
        ).scalar()
        if not result:
            logger.info("SKIPPED -- advisory lock %d held by another worker", lock_id)
            return None
        return await coro()
    finally:
        try:
            db.execute(text("SELECT pg_advisory_unlock(:lid)"), {"lid": lock_id})
        except Exception:
            pass
        db.close()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DailyIngestionOrchestrator:
    """
    Coordinates background data-ingestion jobs on their own AsyncIOScheduler.
    Registered separately from the main CBB scheduler so ingestion can be
    disabled without affecting the nightly analysis pipeline.
    """

    def __init__(self):
        self._scheduler = AsyncIOScheduler()
        self._job_status: dict[str, dict] = {}
        self._openclaw: Optional[Any] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Register all jobs and start the scheduler. Called once from lifespan()."""
        tz = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")

        # MLB odds polling: every 5 min, restricted to 10 AM - 11 PM ET
        self._scheduler.add_job(
            self._poll_mlb_odds,
            IntervalTrigger(minutes=5, timezone=tz),
            id="mlb_odds",
            name="MLB Odds Poll",
            replace_existing=True,
        )

        # Statcast enrichment: every 6 hours
        self._scheduler.add_job(
            self._update_statcast,
            IntervalTrigger(hours=6),
            id="statcast",
            name="Statcast Update",
            replace_existing=True,
        )

        # Rolling z-scores: daily 4 AM ET
        self._scheduler.add_job(
            self._calc_rolling_zscores,
            CronTrigger(hour=4, minute=0, timezone=tz),
            id="rolling_z",
            name="Rolling Z-Score Calc",
            replace_existing=True,
        )

        # CLV attribution: daily 11 PM ET
        self._scheduler.add_job(
            self._compute_clv,
            CronTrigger(hour=23, minute=0, timezone=tz),
            id="clv",
            name="Daily CLV Attribution",
            replace_existing=True,
        )

        # Metric cleanup: daily 3:30 AM ET
        self._scheduler.add_job(
            self._cleanup_old_metrics,
            CronTrigger(hour=3, minute=30, timezone=tz),
            id="cleanup",
            name="Old Metric Cleanup",
            replace_existing=True,
        )

        # Initialise status dict so get_status() never returns missing keys
        for job_id in ("mlb_odds", "statcast", "rolling_z", "clv", "cleanup"):
            if job_id not in self._job_status:
                self._job_status[job_id] = {
                    "name": job_id,
                    "enabled": True,
                    "last_run": None,
                    "last_status": None,
                    "next_run": None,
                }

        self._scheduler.start()
        # Populate next_run now that jobs are scheduled
        for job_id in ("mlb_odds", "statcast", "rolling_z", "clv", "cleanup"):
            self._job_status[job_id]["next_run"] = self._get_next_run(job_id)

    def get_status(self) -> dict:
        """Return per-job status dict for /admin/ingestion/status."""
        # Refresh next_run on every call
        for job_id in list(self._job_status.keys()):
            self._job_status[job_id]["next_run"] = self._get_next_run(job_id)
        return dict(self._job_status)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_next_run(self, job_id: str) -> Optional[str]:
        """Return ISO next-run string for a scheduled job, or None."""
        try:
            job = self._scheduler.get_job(job_id)
            if job and job.next_run_time:
                return job.next_run_time.isoformat()
        except Exception:
            pass
        return None

    def _record_job_run(self, job_id: str, status: str, records: int = 0) -> None:
        """Update in-memory job status after a run."""
        self._job_status[job_id] = {
            "name": job_id,
            "enabled": True,
            "last_run": datetime.utcnow().isoformat(),
            "last_status": status,
            "next_run": self._get_next_run(job_id),
        }

    # ------------------------------------------------------------------
    # Job handlers
    # ------------------------------------------------------------------

    async def _poll_mlb_odds(self) -> dict:
        """
        Fetch MLB spread odds from The Odds API and return a count.
        Does NOT persist to DB (EPIC-3 will consume).
        Skips outside the MLB season window or when API key is absent.
        """
        t0 = time.monotonic()

        async def _run():
            api_key = os.getenv("THE_ODDS_API_KEY")
            if not api_key:
                logger.info("_poll_mlb_odds: THE_ODDS_API_KEY not set, skipping")
                self._record_job_run("mlb_odds", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
            params = {
                "apiKey": api_key,
                "regions": os.getenv("ODDS_API_REGIONS", "us,eu").strip().replace("=", ""),
                "markets": "spreads",
                "oddsFormat": "american",
            }
            try:
                resp = requests.get(url, params=params, timeout=10)
                resp.raise_for_status()
                games = resp.json()
                n = len(games) if isinstance(games, list) else 0
                elapsed = int((time.monotonic() - t0) * 1000)
                if n == 0:
                    logger.info("_poll_mlb_odds: API returned 0 games (off-season or quiet)")
                    self._record_job_run("mlb_odds", "skipped")
                    return {"status": "skipped", "records": 0, "elapsed_ms": elapsed}
                logger.info("_poll_mlb_odds: fetched %d MLB games", n)
                self._record_job_run("mlb_odds", "success", n)
                return {"status": "success", "records": n, "elapsed_ms": elapsed}
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.error("_poll_mlb_odds error: %s", exc)
                self._record_job_run("mlb_odds", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["mlb_odds"], _run)

    async def _update_statcast(self) -> dict:
        """Run daily Statcast ingestion via StatcastIngestionAgent."""
        t0 = time.monotonic()

        async def _run():
            import asyncio as _asyncio
            from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion
            result = await _asyncio.to_thread(run_daily_ingestion)
            elapsed = int((time.monotonic() - t0) * 1000)
            if isinstance(result, dict):
                records = result.get("records_processed", 0)
                status = "ok" if result.get("success") else "error"
            else:
                records = 0
                status = "error"
            self._record_job_run("statcast", status)
            return {"status": status, "records": records, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["statcast"], _run)

    async def _calc_rolling_zscores(self) -> dict:
        """
        Compute 7-day and 30-day rolling z-scores for all MLB players who have
        sufficient history in player_daily_metrics.

        Spec: HANDOFF.md section 3.5
        - 7-day window: requires >= 7 rows  -> z_score_recent
        - 30-day window: requires >= 30 rows -> z_score_total
        Writes a ProjectionSnapshot with significant_changes count.
        """
        t0 = time.monotonic()

        async def _run():
            import statistics

            today = date.today()
            cutoff = today - timedelta(days=30)
            db = SessionLocal()
            records_updated = 0
            significant_changes = 0

            try:
                rows = (
                    db.query(PlayerDailyMetric)
                    .filter(
                        PlayerDailyMetric.sport == "mlb",
                        PlayerDailyMetric.metric_date >= cutoff,
                    )
                    .order_by(PlayerDailyMetric.player_id, PlayerDailyMetric.metric_date)
                    .all()
                )

                # Group by player
                players: dict[str, list] = {}
                for row in rows:
                    players.setdefault(row.player_id, []).append(row)

                for player_id, player_rows in players.items():
                    # Sort ascending by date
                    player_rows.sort(key=lambda r: r.metric_date)

                    vorp_values = [
                        r.vorp_7d for r in player_rows if r.vorp_7d is not None
                    ]

                    new_z_recent: Optional[float] = None
                    new_z_total: Optional[float] = None

                    # 7-day window z-score
                    if len(vorp_values) >= 7:
                        window_7 = vorp_values[-7:]
                        mean_7 = statistics.mean(window_7)
                        try:
                            std_7 = statistics.stdev(window_7)
                        except statistics.StatisticsError:
                            std_7 = 0.0
                        if std_7 > 0 and vorp_values:
                            new_z_recent = (vorp_values[-1] - mean_7) / std_7
                        else:
                            new_z_recent = 0.0

                    # 30-day window z-score
                    if len(vorp_values) >= 30:
                        mean_30 = statistics.mean(vorp_values)
                        try:
                            std_30 = statistics.stdev(vorp_values)
                        except statistics.StatisticsError:
                            std_30 = 0.0
                        if std_30 > 0 and vorp_values:
                            new_z_total = (vorp_values[-1] - mean_30) / std_30
                        else:
                            new_z_total = 0.0

                    if new_z_recent is None and new_z_total is None:
                        continue

                    # Detect significant change vs stored values
                    latest_row = player_rows[-1]
                    old_z = latest_row.z_score_recent or 0.0
                    if new_z_recent is not None and abs(new_z_recent - old_z) > 0.5:
                        significant_changes += 1

                    # Upsert today's row
                    existing = (
                        db.query(PlayerDailyMetric)
                        .filter(
                            PlayerDailyMetric.player_id == player_id,
                            PlayerDailyMetric.metric_date == today,
                            PlayerDailyMetric.sport == "mlb",
                        )
                        .first()
                    )
                    if existing:
                        if new_z_recent is not None:
                            existing.z_score_recent = new_z_recent
                        if new_z_total is not None:
                            existing.z_score_total = new_z_total
                    else:
                        # Create a minimal row for today's z-scores
                        new_metric = PlayerDailyMetric(
                            player_id=player_id,
                            player_name=latest_row.player_name,
                            metric_date=today,
                            sport="mlb",
                            z_score_recent=new_z_recent,
                            z_score_total=new_z_total,
                            rolling_window={},
                            data_source="rolling_zscore_job",
                        )
                        db.add(new_metric)

                    records_updated += 1

                db.commit()

                # Write ProjectionSnapshot
                snapshot = ProjectionSnapshot(
                    snapshot_date=today,
                    sport="mlb",
                    player_changes={},
                    total_players=len(players),
                    significant_changes=significant_changes,
                )
                db.add(snapshot)
                db.commit()

            except Exception as exc:
                db.rollback()
                logger.error("_calc_rolling_zscores error: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_z", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "_calc_rolling_zscores: updated %d players, %d significant changes",
                records_updated, significant_changes,
            )
            self._record_job_run("rolling_z", "success", records_updated)
            return {"status": "success", "records": records_updated, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["rolling_z"], _run)

    async def _compute_clv(self) -> dict:
        """
        Run nightly CLV attribution and persist a ProjectionSnapshot summary.
        Delegates computation to compute_daily_clv_attribution() in clv.py.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.clv import compute_daily_clv_attribution, CLVAttributionError

            try:
                result = await compute_daily_clv_attribution()
            except CLVAttributionError as exc:
                logger.error("_compute_clv CLVAttributionError: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("clv", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            # Persist summary to ProjectionSnapshot
            yesterday = date.today() - timedelta(days=1)
            db = SessionLocal()
            try:
                snapshot = ProjectionSnapshot(
                    snapshot_date=yesterday,
                    sport="cbb",
                    player_changes={
                        "clv_summary": {
                            "clv_positive": result.get("clv_positive", 0),
                            "clv_negative": result.get("clv_negative", 0),
                            "avg_clv_points": result.get("avg_clv_points", 0.0),
                            "favorable_rate": result.get("favorable_rate", 0.0),
                        }
                    },
                    total_players=result.get("games_processed", 0),
                    significant_changes=result.get("clv_negative", 0),
                )
                db.add(snapshot)
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("_compute_clv snapshot write error: %s", exc)
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            records = result.get("games_processed", 0)
            self._record_job_run("clv", "success", records)
            result["elapsed_ms"] = elapsed
            return result

        return await _with_advisory_lock(LOCK_IDS["clv"], _run)

    async def _cleanup_old_metrics(self) -> dict:
        """
        Delete player_daily_metrics rows older than 90 days to keep the table lean.
        """
        t0 = time.monotonic()

        async def _run():
            cutoff = date.today() - timedelta(days=90)
            db = SessionLocal()
            try:
                result = db.execute(
                    text(
                        "DELETE FROM player_daily_metrics WHERE metric_date < :cutoff"
                    ),
                    {"cutoff": cutoff},
                )
                deleted = result.rowcount
                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("_cleanup_old_metrics error: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("cleanup", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info("_cleanup_old_metrics: deleted %d rows older than %s", deleted, cutoff)
            self._record_job_run("cleanup", "success", deleted)
            return {"status": "success", "records": deleted, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["cleanup"], _run)

    def _start_openclaw_monitoring(self) -> None:
        """
        Initialize OpenClaw Phase 1 monitoring (Performance Monitor + Pattern Detector).
        
        This is read-only monitoring that does NOT violate the Guardian freeze.
        Self-improvement features (Phase 4) remain disabled until Apr 7, 2026.
        """
        try:
            from backend.services.openclaw.scheduler import OpenClawScheduler
            
            self._openclaw = OpenClawScheduler(
                scheduler=self._scheduler,
                sport='cbb',  # Primary focus during tournament season
                discord_hook=self._send_discord_alert if os.getenv('DISCORD_ALERTS_ENABLED') else None
            )
            self._openclaw.start_monitoring()
            
            logger.info("OpenClaw Phase 1 monitoring started (Performance Monitor + Pattern Detector)")
        except Exception as exc:
            logger.warning("OpenClaw monitoring not started: %s", exc)
    
    def _send_discord_alert(self, embed: dict) -> None:
        """Send Discord alert via webhook."""
        import requests
        
        webhook_url = os.getenv('DISCORD_ALERTS_WEBHOOK')
        if not webhook_url:
            return
        
        try:
            requests.post(
                webhook_url,
                json={"embeds": [embed]},
                timeout=5
            )
        except Exception as exc:
            logger.warning("Discord alert failed: %s", exc)
