"""
DailyIngestionOrchestrator — EPIC-2 data pipeline coordinator.

Owns all MLB/CBB data polling jobs that run independently of the nightly
CBB analysis scheduler. Each job acquires a PostgreSQL advisory lock before
running, which prevents duplicate execution across Railway replicas.

ADR-001: Every job MUST use _with_advisory_lock.
ADR-004: This file is additive only. Never import betting_model or analysis.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, date, timedelta
from typing import Optional, Any
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.models import (
    SessionLocal,
    PlayerDailyMetric,
    ProjectionSnapshot,
    PlayerValuationCache,
    ProjectionCacheEntry,
    engine,
)
from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion
from backend.utils.time_utils import now_et, today_et

logger = logging.getLogger(__name__)


# Module-level mirror: RoS projections fetched by fangraphs_ros (100_012)
# and also persisted to projection_cache_entries for cross-process durability.
_ROS_CACHE: dict = {}
_ROS_CACHE_KEY = "fangraphs_ros"
_ROS_CACHE_TABLE_READY = False


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
    "valuation_cache": 100_011,
    "fangraphs_ros":   100_012,
    "yahoo_adp_injury": 100_013,
    "ensemble_update": 100_014,
    "projection_freshness": 100_015,
}


def _ensure_projection_cache_table() -> None:
    """Create the durable projection cache table if the migration has not run yet."""
    global _ROS_CACHE_TABLE_READY
    if _ROS_CACHE_TABLE_READY:
        return
    ProjectionCacheEntry.__table__.create(bind=engine, checkfirst=True)
    _ROS_CACHE_TABLE_READY = True


def _serialize_ros_frames(frames: Optional[dict]) -> dict[str, list[dict[str, Any]]]:
    """Convert a dict of pandas DataFrames into JSON-safe row lists."""
    if not frames:
        return {}

    serialized: dict[str, list[dict[str, Any]]] = {}
    for system_key, frame in frames.items():
        if frame is None:
            continue
        serialized[system_key] = json.loads(frame.to_json(orient="records", date_format="iso"))
    return serialized


def _deserialize_ros_frames(payload: Optional[dict]) -> dict[str, Any]:
    """Rebuild pandas DataFrames from persisted RoS row payloads."""
    if not payload:
        return {}

    import pandas as pd

    restored: dict[str, Any] = {}
    for system_key, rows in payload.items():
        restored[system_key] = pd.DataFrame(rows or [])
    return restored


def _store_persisted_ros_cache(
    bat_raw: Optional[dict],
    pit_raw: Optional[dict],
    fetched_at: datetime,
) -> None:
    """Persist raw Fangraphs payloads so downstream jobs survive process restarts."""
    _ensure_projection_cache_table()
    db = SessionLocal()
    try:
        entry = (
            db.query(ProjectionCacheEntry)
            .filter(ProjectionCacheEntry.cache_key == _ROS_CACHE_KEY)
            .first()
        )
        payload = {
            "bat": _serialize_ros_frames(bat_raw),
            "pit": _serialize_ros_frames(pit_raw),
        }
        if entry is None:
            entry = ProjectionCacheEntry(
                cache_key=_ROS_CACHE_KEY,
                payload=payload,
                fetched_at=fetched_at,
            )
            db.add(entry)
        else:
            entry.payload = payload
            entry.fetched_at = fetched_at
        db.commit()
    finally:
        db.close()


def _load_persisted_ros_cache(include_payload: bool = True) -> tuple[Optional[dict], Optional[dict], Optional[datetime]]:
    """Load the last persisted Fangraphs payload and fetched timestamp."""
    _ensure_projection_cache_table()
    db = SessionLocal()
    try:
        entry = (
            db.query(ProjectionCacheEntry)
            .filter(ProjectionCacheEntry.cache_key == _ROS_CACHE_KEY)
            .first()
        )
        if entry is None:
            return None, None, None
        if not include_payload:
            return None, None, entry.fetched_at
        payload = entry.payload or {}
        return (
            _deserialize_ros_frames(payload.get("bat")),
            _deserialize_ros_frames(payload.get("pit")),
            entry.fetched_at,
        )
    finally:
        db.close()


def _extract_blend_rows(blend_df: Any, metric_map: dict[str, str]) -> tuple[list[dict[str, Any]], int]:
    """Normalize a blend dataframe into upsert rows and count skipped entries."""
    if blend_df is None:
        return [], 0

    rows: list[dict[str, Any]] = []
    skipped = 0
    for _, row in blend_df.iterrows():
        player_id = row.get("player_id", "")
        if not player_id:
            skipped += 1
            continue

        metrics = {dest: row.get(src) for src, dest in metric_map.items()}
        if all(value is None for value in metrics.values()):
            skipped += 1
            continue

        rows.append(
            {
                "player_id": player_id,
                "player_name": row.get("name", player_id),
                **metrics,
            }
        )

    return rows, skipped


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

        # Player valuation cache: daily 6 AM ET
        # Only runs if FANTASY_LEAGUES env var is set (comma-separated league keys)
        _fantasy_leagues = os.getenv("FANTASY_LEAGUES", "")
        if _fantasy_leagues:
            self._scheduler.add_job(
                self._refresh_valuation_cache,
                CronTrigger(hour=6, minute=0, timezone=tz),
                id="valuation_cache",
                name="Player Valuation Cache Refresh",
                replace_existing=True,
            )

        # FanGraphs RoS projections: daily 3 AM ET (before ensemble at 5 AM)
        self._scheduler.add_job(
            self._fetch_fangraphs_ros,
            CronTrigger(hour=3, minute=0, timezone=tz),
            id="fangraphs_ros",
            name="FanGraphs RoS Fetch",
            replace_existing=True,
        )

        # Yahoo ADP + injury feed: every 4 hours
        self._scheduler.add_job(
            self._poll_yahoo_adp_injury,
            IntervalTrigger(hours=4),
            id="yahoo_adp_injury",
            name="Yahoo ADP & Injury Poll",
            replace_existing=True,
        )

        # Ensemble blend update: daily 5 AM ET (after RoS fetch at 3 AM)
        self._scheduler.add_job(
            self._update_ensemble_blend,
            CronTrigger(hour=5, minute=0, timezone=tz),
            id="ensemble_update",
            name="Ensemble Blend Update",
            replace_existing=True,
        )

        # Projection freshness SLA gate: hourly
        self._scheduler.add_job(
            self._check_projection_freshness,
            IntervalTrigger(hours=1),
            id="projection_freshness",
            name="Projection Freshness Check",
            replace_existing=True,
        )

        # Initialise status dict so get_status() never returns missing keys
        _all_job_ids = ["mlb_odds", "statcast", "rolling_z", "clv", "cleanup",
                        "fangraphs_ros", "yahoo_adp_injury", "ensemble_update",
                        "projection_freshness"]
        if _fantasy_leagues:
            _all_job_ids.append("valuation_cache")
        for job_id in _all_job_ids:
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
        for job_id in _all_job_ids:
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
            "last_run": now_et().isoformat(),
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
        """Daily Statcast enrichment — fetches yesterday's data and runs Bayesian projection updates."""      
        t0 = time.monotonic()
        async def _run():
            try:
                result = await asyncio.to_thread(run_daily_ingestion)
                elapsed = int((time.monotonic() - t0) * 1000)
                status = "success" if result.get("success") else "failed"

                if not result.get("success"):
                    logger.error(
                        "_update_statcast: ingestion reported failure -- %s",
                        result.get("error", "unknown error"),
                    )

                records = result.get("records_processed", 0) if isinstance(result, dict) else 0
                self._record_job_run("statcast", status)

                if isinstance(result, dict):
                    result["status"] = status
                    result["records"] = records
                    result["elapsed_ms"] = elapsed
                else:
                    result = {"status": status, "records": 0, "elapsed_ms": elapsed}
                return result
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.exception("_update_statcast: unhandled error -- %s", exc)
                self._record_job_run("statcast", "failed")
                return {"status": "failed", "records": 0, "error": str(exc), "elapsed_ms": elapsed}

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

            today = today_et()
            cutoff = today - timedelta(days=30)
            db = SessionLocal()
            records_updated = 0
            significant_changes = 0
            skipped_insufficient_data = 0
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
                        skipped_insufficient_data += 1
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
            # M-5: alert when most players lack sufficient history (first 7 days of season)
            total_seen = len(players)
            if total_seen > 0 and skipped_insufficient_data > 0:
                skip_pct = skipped_insufficient_data / total_seen * 100
                if skipped_insufficient_data == total_seen:
                    logger.warning(
                        "rolling_z: ALL %d players skipped — insufficient data "
                        "(need >= 7 days of vorp_7d). Season may be < 7 days old.",
                        total_seen,
                    )
                elif skip_pct > 50:
                    logger.warning(
                        "rolling_z: %d/%d players skipped (%.0f%%) — insufficient data. "
                        "z-scores will improve as season history accumulates.",
                        skipped_insufficient_data, total_seen, skip_pct,
                    )
                else:
                    logger.debug(
                        "rolling_z: %d players skipped for insufficient data (<7 rows)",
                        skipped_insufficient_data,
                    )
            self._record_job_run("rolling_z", "success", records_updated)
            return {"status": "success", "records": records_updated, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["rolling_z"], _run)

    async def _poll_yahoo_adp_injury(self) -> dict:
        """Fetch Yahoo ADP + injury status snapshot every 4 hours (lock 100_013).

        Pulls up to 100 players sorted by ADP (sort=DA) and caches their
        injury status in PlayerDailyMetric.  This feed is the sole source for
        detecting new injuries and ADP rank movements without polling each
        player individually.

        Data written: player status, injury_note, percent_owned updated on today's row.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.yahoo_client_resilient import (
                YahooFantasyClient, YahooAuthError, YahooAPIError,
            )
            try:
                client = YahooFantasyClient()
            except YahooAuthError as exc:
                logger.error("yahoo_adp_injury: auth error — %s", exc)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0}

            try:
                players = client.get_adp_and_injury_feed(pages=4, count_per_page=25)
            except (YahooAuthError, YahooAPIError) as exc:
                logger.error("yahoo_adp_injury: API error — %s", exc)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0}

            if not players:
                logger.warning("yahoo_adp_injury: no players returned")
                self._record_job_run("yahoo_adp_injury", "success", 0)
                return {"status": "success", "records": 0}

            today = today_et()
            db = SessionLocal()
            records_written = 0
            injury_flags = 0
            try:
                for p in players:
                    pid = p.get("player_key") or ""
                    if not pid:
                        continue
                    status_val = p.get("status") or None
                    injury_note = p.get("injury_note") or None
                    owned_pct = p.get("percent_owned", 0.0) or 0.0

                    if injury_note:
                        injury_flags += 1

                    existing = (
                        db.query(PlayerDailyMetric)
                        .filter(
                            PlayerDailyMetric.player_id == pid,
                            PlayerDailyMetric.metric_date == today,
                            PlayerDailyMetric.sport == "mlb",
                        )
                        .first()
                    )
                    if existing:
                        # Patch injury / ownership fields on existing row
                        existing.rolling_window = {
                            **(existing.rolling_window or {}),
                            "status": status_val,
                            "injury_note": injury_note,
                            "percent_owned": owned_pct,
                            "adp_updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                        }
                    else:
                        db.add(PlayerDailyMetric(
                            player_id=pid,
                            player_name=p.get("name", pid),
                            metric_date=today,
                            sport="mlb",
                            rolling_window={
                                "status": status_val,
                                "injury_note": injury_note,
                                "percent_owned": owned_pct,
                                "adp_updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                            },
                            data_source="yahoo_adp_injury",
                        ))
                    records_written += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("yahoo_adp_injury DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "yahoo_adp_injury: wrote %d rows (%d injury flags) in %dms",
                records_written, injury_flags, elapsed,
            )
            self._record_job_run("yahoo_adp_injury", "success", records_written)
            return {
                "status": "success",
                "records": records_written,
                "injury_flags": injury_flags,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["yahoo_adp_injury"], _run)

    async def _fetch_fangraphs_ros(self) -> dict:
        """Fetch daily Rest-of-Season projections from FanGraphs (lock 100_012).

        Runs at 3 AM ET, before the ensemble blend job at 5 AM.
        Stores raw blend results in a module-level cache so _update_ensemble_blend
        can use them without re-fetching.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.fangraphs_loader import (
                fetch_all_ros, compute_ensemble_blend,
            )

            bat_raw = fetch_all_ros("bat", delay_seconds=3.0)
            pit_raw = fetch_all_ros("pit", delay_seconds=3.0)
            fetched_at = now_et()

            bat_count = sum(len(df) for df in bat_raw.values()) if bat_raw else 0
            pit_count = sum(len(df) for df in pit_raw.values()) if pit_raw else 0

            # Mirror in memory for same-process handoff and persist for cross-process durability.
            _ROS_CACHE["bat"] = bat_raw
            _ROS_CACHE["pit"] = pit_raw
            _ROS_CACHE["fetched_at"] = fetched_at

            try:
                _store_persisted_ros_cache(bat_raw, pit_raw, fetched_at)
            except Exception as exc:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.error("fangraphs_ros: failed to persist durable cache: %s", exc)
                self._record_job_run("fangraphs_ros", "failed")
                return {
                    "status": "failed",
                    "bat_rows": bat_count,
                    "pit_rows": pit_count,
                    "elapsed_ms": elapsed,
                    "error": "durable cache persist failed",
                }

            elapsed = int((time.monotonic() - t0) * 1000)
            status = "ok" if (bat_raw or pit_raw) else "failed"
            logger.info(
                "fangraphs_ros: %d bat / %d pit rows from %d/%d systems; status=%s",
                bat_count, pit_count,
                len(bat_raw) + len(pit_raw), 8,  # 4 systems × 2 stat types
                status,
            )
            if status == "failed":
                logger.warning("fangraphs_ros: all FanGraphs fetches failed — cloudscraper or network issue")
            self._record_job_run("fangraphs_ros", status, bat_count + pit_count)
            return {"status": status, "bat_rows": bat_count, "pit_rows": pit_count, "elapsed_ms": elapsed}

        return await _with_advisory_lock(LOCK_IDS["fangraphs_ros"], _run)

    async def _update_ensemble_blend(self) -> dict:
        """Compute weighted ensemble blend and persist to PlayerDailyMetric (lock 100_014).

        Runs at 5 AM ET, 2 hours after _fetch_fangraphs_ros.
        Uses the durable Fangraphs cache from the earlier fetch when available.
        Blend columns written: blend_hr, blend_rbi, blend_avg, blend_era, blend_whip.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.fantasy_baseball.fangraphs_loader import (
                fetch_all_ros, compute_ensemble_blend,
            )

            # Use cached RoS data if fresh (< 4 hours); otherwise re-fetch
            bat_raw = None
            pit_raw = None
            cached_at = _ROS_CACHE.get("fetched_at")
            if cached_at:
                age_h = (now_et() - cached_at).total_seconds() / 3600
                if age_h < 4:
                    bat_raw = _ROS_CACHE.get("bat")
                    pit_raw = _ROS_CACHE.get("pit")

            if not bat_raw and not pit_raw:
                persisted_bat, persisted_pit, persisted_at = _load_persisted_ros_cache()
                if persisted_at is not None:
                    age_h = (now_et() - persisted_at).total_seconds() / 3600
                    if age_h < 4:
                        bat_raw = persisted_bat
                        pit_raw = persisted_pit
                        cached_at = persisted_at
                        _ROS_CACHE["bat"] = bat_raw
                        _ROS_CACHE["pit"] = pit_raw
                        _ROS_CACHE["fetched_at"] = persisted_at

            if not bat_raw and not pit_raw:
                logger.info("ensemble_update: cache miss — re-fetching FanGraphs RoS")
                bat_raw = fetch_all_ros("bat", delay_seconds=3.0)
                pit_raw = fetch_all_ros("pit", delay_seconds=3.0)
                if bat_raw or pit_raw:
                    fetched_at = now_et()
                    _ROS_CACHE["bat"] = bat_raw
                    _ROS_CACHE["pit"] = pit_raw
                    _ROS_CACHE["fetched_at"] = fetched_at
                    _store_persisted_ros_cache(bat_raw, pit_raw, fetched_at)

            if not bat_raw and not pit_raw:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.error("ensemble_update: no RoS data available — skipping blend")
                self._record_job_run("ensemble_update", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            bat_blend = compute_ensemble_blend(bat_raw or {}, stat_columns=["HR", "R", "RBI", "SB", "AVG"]) if bat_raw else None
            pit_blend = compute_ensemble_blend(pit_raw or {}, stat_columns=["ERA", "WHIP"]) if pit_raw else None

            today = today_et()
            db = SessionLocal()
            bat_rows, bat_skipped = _extract_blend_rows(
                bat_blend,
                {"HR": "blend_hr", "RBI": "blend_rbi", "AVG": "blend_avg"},
            )
            pit_rows, pit_skipped = _extract_blend_rows(
                pit_blend,
                {"ERA": "blend_era", "WHIP": "blend_whip"},
            )
            records_written = 0
            inserted = 0
            updated = 0
            skipped = bat_skipped + pit_skipped
            try:
                all_rows = bat_rows + pit_rows
                existing_ids: set[str] = set()
                if all_rows:
                    existing_ids = {
                        player_id
                        for (player_id,) in (
                            db.query(PlayerDailyMetric.player_id)
                            .filter(
                                PlayerDailyMetric.metric_date == today,
                                PlayerDailyMetric.sport == "mlb",
                                PlayerDailyMetric.player_id.in_([row["player_id"] for row in all_rows]),
                            )
                            .all()
                        )
                    }

                for row in all_rows:
                    stmt = pg_insert(PlayerDailyMetric.__table__).values(
                        player_id=row["player_id"],
                        player_name=row["player_name"],
                        metric_date=today,
                        sport="mlb",
                        rolling_window={},
                        data_source="ensemble_blend",
                        fetched_at=now_et(),
                        blend_hr=row.get("blend_hr"),
                        blend_rbi=row.get("blend_rbi"),
                        blend_avg=row.get("blend_avg"),
                        blend_era=row.get("blend_era"),
                        blend_whip=row.get("blend_whip"),
                    ).on_conflict_do_update(
                        index_elements=["player_id", "metric_date", "sport"],
                        set_={
                            "player_name": row["player_name"],
                            "data_source": "ensemble_blend",
                            "fetched_at": now_et(),
                            "blend_hr": row.get("blend_hr"),
                            "blend_rbi": row.get("blend_rbi"),
                            "blend_avg": row.get("blend_avg"),
                            "blend_era": row.get("blend_era"),
                            "blend_whip": row.get("blend_whip"),
                        },
                    )
                    db.execute(stmt)
                    if row["player_id"] in existing_ids:
                        updated += 1
                    else:
                        inserted += 1
                        existing_ids.add(row["player_id"])
                    records_written += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("ensemble_update DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("ensemble_update", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed, "inserted": inserted, "updated": updated, "skipped": skipped}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "ensemble_update: wrote %d player blend rows in %dms (inserted=%d updated=%d skipped=%d)",
                records_written,
                elapsed,
                inserted,
                updated,
                skipped,
            )
            self._record_job_run("ensemble_update", "success", records_written)
            return {
                "status": "success",
                "records": records_written,
                "elapsed_ms": elapsed,
                "inserted": inserted,
                "updated": updated,
                "skipped": skipped,
            }

        return await _with_advisory_lock(LOCK_IDS["ensemble_update"], _run)

    async def _check_projection_freshness(self) -> dict:
        """
        SLA gate: warn when projection data is stale but do NOT block anything.
        SLAs: ensemble_blend ≤ 12 h, statcast ≤ 6 h, Fangraphs RoS cache ≤ 12 h.
        Results stored in self._job_status["projection_freshness"] for /admin/ingestion/status.
        """
        t0 = time.monotonic()

        async def _run():
            from datetime import datetime
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo("America/New_York"))
            violations: list[str] = []
            report: dict = {"checked_at": now.isoformat(), "violations": violations}

            db = SessionLocal()
            try:
                # --- ensemble_blend SLA (12 hours) ---
                SLA_ENSEMBLE_H = 12
                result = db.execute(
                    text(
                        "SELECT MAX(date) FROM player_daily_metrics "
                        "WHERE data_source = 'ensemble_blend'"
                    )
                )
                latest_ensemble = result.scalar()
                if latest_ensemble is None:
                    msg = "ensemble_blend: no rows found — pipeline may not have run yet"
                    logger.warning("PROJECTION FRESHNESS: %s", msg)
                    violations.append(msg)
                else:
                    if hasattr(latest_ensemble, "tzinfo") and latest_ensemble.tzinfo is None:
                        latest_ensemble = latest_ensemble.replace(tzinfo=ZoneInfo("America/New_York"))
                    age_h = (now - latest_ensemble).total_seconds() / 3600
                    report["ensemble_blend_age_h"] = round(age_h, 1)
                    if age_h > SLA_ENSEMBLE_H:
                        msg = f"ensemble_blend stale: {age_h:.1f}h > SLA {SLA_ENSEMBLE_H}h"
                        logger.warning("PROJECTION FRESHNESS: %s", msg)
                        violations.append(msg)

                # --- statcast SLA (6 hours) ---
                SLA_STATCAST_H = 6
                result = db.execute(
                    text(
                        "SELECT MAX(date) FROM player_daily_metrics "
                        "WHERE data_source = 'statcast'"
                    )
                )
                latest_statcast = result.scalar()
                if latest_statcast is None:
                    msg = "statcast: no rows found — statcast ingestion may not have run yet"
                    logger.warning("PROJECTION FRESHNESS: %s", msg)
                    violations.append(msg)
                else:
                    if hasattr(latest_statcast, "tzinfo") and latest_statcast.tzinfo is None:
                        latest_statcast = latest_statcast.replace(tzinfo=ZoneInfo("America/New_York"))
                    age_h = (now - latest_statcast).total_seconds() / 3600
                    report["statcast_age_h"] = round(age_h, 1)
                    if age_h > SLA_STATCAST_H:
                        msg = f"statcast stale: {age_h:.1f}h > SLA {SLA_STATCAST_H}h"
                        logger.warning("PROJECTION FRESHNESS: %s", msg)
                        violations.append(msg)
            finally:
                db.close()

            # --- persisted Fangraphs RoS cache SLA (12 hours) ---
            SLA_ROS_H = 12
            _, _, ros_fetched_at = _load_persisted_ros_cache(include_payload=False)
            if ros_fetched_at is None:
                msg = "fangraphs_ros cache missing — durable RoS cache has not been persisted yet"
                logger.warning("PROJECTION FRESHNESS: %s", msg)
                violations.append(msg)
            else:
                if hasattr(ros_fetched_at, "tzinfo") and ros_fetched_at.tzinfo is None:
                    ros_fetched_at = ros_fetched_at.replace(tzinfo=ZoneInfo("America/New_York"))
                age_h = (now - ros_fetched_at).total_seconds() / 3600
                report["ros_cache_age_h"] = round(age_h, 1)
                if age_h > SLA_ROS_H:
                    msg = f"fangraphs_ros cache stale: {age_h:.1f}h > SLA {SLA_ROS_H}h"
                    logger.warning("PROJECTION FRESHNESS: %s", msg)
                    violations.append(msg)

            elapsed = round((time.monotonic() - t0) * 1000)
            report["elapsed_ms"] = elapsed
            report["violation_count"] = len(violations)

            if not violations:
                logger.debug("PROJECTION FRESHNESS: all SLAs met (checked in %d ms)", elapsed)

            self._job_status["projection_freshness"] = report
            return {"status": "success", **report}

        return await _with_advisory_lock(LOCK_IDS["projection_freshness"], _run)

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
            yesterday = today_et() - timedelta(days=1)
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
            cutoff = today_et() - timedelta(days=90)
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

    async def _refresh_valuation_cache(self) -> None:
        """
        Refresh player valuation cache for all configured leagues.
        Advisory lock: 100_011.
        """
        from backend.fantasy_baseball.valuation_worker import run_valuation_worker

        league_str = os.getenv("FANTASY_LEAGUES", "")
        if not league_str:
            logger.info("valuation_cache: FANTASY_LEAGUES not set -- skipping")
            return

        leagues = [lk.strip() for lk in league_str.split(",") if lk.strip()]

        async def _run():
            results = []
            for lk in leagues:
                try:
                    result = await run_valuation_worker(lk)
                    results.append(result)
                except Exception as exc:
                    logger.error("valuation_cache: failed for league=%s (%s)", lk, exc)
            return results

        self._job_status["valuation_cache"] = {
            "name": "valuation_cache",
            "enabled": True,
            "last_run": datetime.now(ZoneInfo("America/New_York")).isoformat(),
            "last_status": "running",
            "next_run": self._get_next_run("valuation_cache"),
        }

        try:
            results = await _with_advisory_lock(LOCK_IDS["valuation_cache"], _run)
            self._job_status["valuation_cache"]["last_status"] = "ok"
            logger.info("valuation_cache: complete -- %s", results)
        except Exception as exc:
            self._job_status["valuation_cache"]["last_status"] = f"error: {exc}"
            logger.error("valuation_cache: job failed (%s)", exc)

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
