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
import math
from datetime import datetime, date, timedelta
from typing import Optional, Any
from zoneinfo import ZoneInfo

import requests
from sqlalchemy import text, func
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
    MLBTeam,
    MLBGameLog,
    MLBOddsSnapshot,
    MLBPlayerStats,
    PlayerRollingStats,
    PlayerScore,
    PlayerMomentum,
    PlayerIDMapping,
    SimulationResult as SimulationResultORM,
    DecisionResult as DecisionResultORM,
    BacktestResult as BacktestResultORM,
    DecisionExplanation as DecisionExplanationORM,
    DailySnapshot as DailySnapshotORM,
    engine,
)
from backend.services.explainability_layer import ExplanationInput, explain_batch
from backend.services.snapshot_engine import SnapshotInput, build_snapshot
from backend.services.backtesting_harness import (
    BacktestInput,
    evaluate_cohort,
    summarize,
    load_golden_baseline,
    save_golden_baseline,
    BASELINE_PATH,
)
from backend.services.simulation_engine import simulate_all_players, REMAINING_GAMES_DEFAULT
from backend.services.decision_engine import (
    PlayerDecisionInput,
    optimize_lineup,
    optimize_waivers,
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
    "mlb_game_log":         100_016,
    "mlb_box_stats":        100_017,
    "rolling_windows":      100_018,
    "player_scores":        100_019,
    "player_momentum":      100_020,
    "ros_simulation":       100_021,
    "decision_optimization": 100_022,
    "backtesting":           100_023,
    "explainability":        100_024,
    "snapshot":              100_025,
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

        def _is_missing_metric(value: Any) -> bool:
            if value is None:
                return True
            if isinstance(value, float) and math.isnan(value):
                return True
            return False

        if all(_is_missing_metric(value) for value in metrics.values()):
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

        # MLB game log ingestion: daily 1 AM ET
        # Fetches yesterday (final scores) + today (schedule). Runs before rolling_z (3 AM).
        self._scheduler.add_job(
            self._ingest_mlb_game_log,
            CronTrigger(hour=1, minute=0, timezone=tz),
            id="mlb_game_log",
            name="MLB Game Log Ingestion",
            replace_existing=True,
        )

        # MLB player box stats ingestion: daily 2 AM ET (1 hour after game-log at 1 AM)
        # Fetches yesterday + today box stats. Runs before rolling_z (3 AM) and fangraphs_ros (3 AM).
        self._scheduler.add_job(
            self._ingest_mlb_box_stats,
            CronTrigger(hour=2, minute=0, timezone=tz),
            id="mlb_box_stats",
            name="MLB Box Stats Ingestion",
            replace_existing=True,
        )

        # Rolling window computation: daily 3 AM ET (after box stats at 2 AM)
        # Computes 7/14/30-day decay-weighted windows per player. Runs before Z-score calc (4 AM).
        self._scheduler.add_job(
            self._compute_rolling_windows,
            CronTrigger(hour=3, minute=0, timezone=tz),
            id="rolling_windows",
            name="Player Rolling Window Computation",
            replace_existing=True,
        )

        # Player Z-score scoring: daily 4 AM ET (after rolling windows at 3 AM)
        # Computes league Z-scores + 0-100 percentile ranks per player per window.
        self._scheduler.add_job(
            self._compute_player_scores,
            CronTrigger(hour=4, minute=0, timezone=tz),
            id="player_scores",
            name="Player Z-Score Scoring",
            replace_existing=True,
        )

        # Player momentum signals: daily 5 AM ET (after player_scores at 4 AM)
        # Computes delta-Z = Z_14d - Z_30d and assigns SURGING/HOT/STABLE/COLD/COLLAPSING.
        self._scheduler.add_job(
            self._compute_player_momentum,
            CronTrigger(hour=5, minute=0, timezone=tz),
            id="player_momentum",
            name="Player Momentum Signal Computation",
            replace_existing=True,
        )

        # RoS Monte Carlo simulation: daily 6 AM ET (after player_momentum at 5 AM)
        # Runs 1000-sim ROS projection per player from 14d rolling window.
        self._scheduler.add_job(
            self._run_ros_simulation,
            CronTrigger(hour=6, minute=0, timezone=tz),
            id="ros_simulation",
            name="RoS Monte Carlo Simulation",
            replace_existing=True,
        )

        # Decision optimization: daily 7 AM ET (after ros_simulation at 6 AM)
        # Runs greedy lineup + waiver analysis from player_scores + simulation_results.
        self._scheduler.add_job(
            self._run_decision_optimization,
            CronTrigger(hour=7, minute=0, timezone=tz),
            id="decision_optimization",
            name="Decision Engine Optimization",
            replace_existing=True,
        )

        # Backtesting harness: daily 8 AM ET (after decision_optimization at 7 AM)
        # Evaluates P16 simulation projections vs actual mlb_player_stats outcomes.
        self._scheduler.add_job(
            self._run_backtesting,
            CronTrigger(hour=8, minute=0, timezone=tz),
            id="backtesting",
            name="Backtesting Harness",
            replace_existing=True,
        )

        # Explainability: daily 9 AM ET (after backtesting at 8 AM)
        # Generates human-readable decision traces from all P14-P18 signals.
        self._scheduler.add_job(
            self._run_explainability,
            CronTrigger(hour=9, minute=0, timezone=tz),
            id="explainability",
            name="Explainability Engine",
            replace_existing=True,
        )

        # Snapshot: daily 10 AM ET (after explainability at 9 AM -- final pipeline stage)
        # Captures complete daily state: counts, health status, top players, regression flag.
        self._scheduler.add_job(
            self._run_snapshot,
            CronTrigger(hour=10, minute=0, timezone=tz),
            id="snapshot",
            name="Daily Snapshot",
            replace_existing=True,
        )

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
        _all_job_ids = ["mlb_game_log", "mlb_box_stats", "rolling_windows", "player_scores",
                        "player_momentum", "ros_simulation", "decision_optimization",
                        "backtesting", "explainability", "snapshot",
                        "mlb_odds", "statcast",
                        "rolling_z", "clv", "cleanup", "fangraphs_ros", "yahoo_adp_injury",
                        "ensemble_update", "projection_freshness"]
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

    async def run_job(self, job_id: str) -> dict:
        """
        Manually execute a single ingestion job by ID.

        Supported IDs (pipeline order):
          mlb_game_log -> mlb_box_stats -> rolling_windows -> player_scores
          -> player_momentum -> ros_simulation -> decision_optimization

        Returns the job's result dict.  Raises ValueError for unknown job_id.
        """
        _handlers = {
            "mlb_game_log":    self._ingest_mlb_game_log,
            "mlb_box_stats":   self._ingest_mlb_box_stats,
            "rolling_windows": self._compute_rolling_windows,
            "player_scores":   self._compute_player_scores,
            "player_momentum": self._compute_player_momentum,
            "ros_simulation":        self._run_ros_simulation,
            "decision_optimization": self._run_decision_optimization,
            "backtesting":           self._run_backtesting,
            "explainability":        self._run_explainability,
            "snapshot":              self._run_snapshot,
        }
        handler = _handlers.get(job_id)
        if handler is None:
            raise ValueError(f"Unknown job_id: {job_id!r}. Valid: {sorted(_handlers)}")
        return await handler()

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
        Fetch MLB spread odds via BallDontLie GOAT client (lock 100_001).

        Every poll (every 5 min):
          1. get_mlb_games(today) -> list[MLBGame]
          2. For each game: ensure mlb_team + mlb_game_log rows exist (idempotent upsert)
          3. For each game: get_mlb_odds(game_id) -> list[MLBBettingOdd]
          4. Upsert each odd into mlb_odds_snapshot on (game_id, vendor, snapshot_window)

        snapshot_window = now_et() rounded DOWN to the 30-min bucket.
        This makes each poll idempotent — re-running within the same 30-min window
        updates the row rather than inserting a duplicate.

        All data is Pydantic-validated. Typed attribute access only. No dict.get().
        raw_payload = odd.model_dump() on every upsert (dual-write).
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("_poll_mlb_odds: BDL init failed -- %s", exc)
                self._record_job_run("mlb_odds", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            date_str = today_et().isoformat()
            games = await asyncio.to_thread(bdl.get_mlb_games, date_str)

            if not games:
                elapsed = int((time.monotonic() - t0) * 1000)
                logger.info("_poll_mlb_odds: 0 games on %s (BDL)", date_str)
                self._record_job_run("mlb_odds", "success", 0)
                return {"status": "success", "records": 0, "elapsed_ms": elapsed}

            # snapshot_window: round current time DOWN to 30-min bucket (idempotent key)
            now = now_et()
            snapshot_window = now.replace(
                minute=(now.minute // 30) * 30, second=0, microsecond=0
            )

            total_odds = 0
            games_with_odds = 0
            db = SessionLocal()
            try:
                for game in games:
                    # Step 1: ensure dimension + fact rows exist before odds FK write
                    for team in (game.home_team, game.away_team):
                        stmt = pg_insert(MLBTeam.__table__).values(
                            team_id=team.id,
                            abbreviation=team.abbreviation,
                            name=team.name,
                            display_name=team.display_name,
                            short_name=team.short_display_name,
                            location=team.location,
                            slug=team.slug,
                            league=team.league,
                            division=team.division,
                            ingested_at=now,
                        ).on_conflict_do_update(
                            index_elements=["team_id"],
                            set_=dict(
                                abbreviation=team.abbreviation,
                                name=team.name,
                                display_name=team.display_name,
                                short_name=team.short_display_name,
                                location=team.location,
                                slug=team.slug,
                                league=team.league,
                                division=team.division,
                            ),
                        )
                        db.execute(stmt)

                    dt_utc = datetime.fromisoformat(game.date.replace("Z", "+00:00"))
                    game_date_et = dt_utc.astimezone(ZoneInfo("America/New_York")).date()
                    is_active = game.status in {"STATUS_FINAL", "STATUS_IN_PROGRESS"}

                    game_stmt = pg_insert(MLBGameLog.__table__).values(
                        game_id=game.id,
                        game_date=game_date_et,
                        season=game.season,
                        season_type=game.season_type,
                        status=game.status,
                        home_team_id=game.home_team.id,
                        away_team_id=game.away_team.id,
                        home_runs=game.home_team_data.runs if is_active else None,
                        away_runs=game.away_team_data.runs if is_active else None,
                        home_hits=game.home_team_data.hits if is_active else None,
                        away_hits=game.away_team_data.hits if is_active else None,
                        home_errors=game.home_team_data.errors if is_active else None,
                        away_errors=game.away_team_data.errors if is_active else None,
                        venue=game.venue,
                        attendance=(game.attendance or None) if is_active else None,
                        period=game.period,
                        raw_payload=game.model_dump(),
                        ingested_at=now,
                        updated_at=now,
                    ).on_conflict_do_update(
                        index_elements=["game_id"],
                        set_=dict(
                            status=game.status,
                            home_runs=game.home_team_data.runs if is_active else None,
                            away_runs=game.away_team_data.runs if is_active else None,
                            home_hits=game.home_team_data.hits if is_active else None,
                            away_hits=game.away_team_data.hits if is_active else None,
                            home_errors=game.home_team_data.errors if is_active else None,
                            away_errors=game.away_team_data.errors if is_active else None,
                            attendance=(game.attendance or None) if is_active else None,
                            period=game.period,
                            raw_payload=game.model_dump(),
                            updated_at=now,
                        ),
                    )
                    db.execute(game_stmt)

                    # Step 2: fetch and persist odds
                    odds = await asyncio.to_thread(bdl.get_mlb_odds, game.id)
                    if not odds:
                        continue

                    games_with_odds += 1
                    for odd in odds:
                        payload = odd.model_dump()
                        stmt = pg_insert(MLBOddsSnapshot.__table__).values(
                            odds_id=odd.id,
                            game_id=odd.game_id,
                            vendor=odd.vendor,
                            snapshot_window=snapshot_window,
                            spread_home=odd.spread_home_value,
                            spread_away=odd.spread_away_value,
                            spread_home_odds=odd.spread_home_odds,
                            spread_away_odds=odd.spread_away_odds,
                            ml_home_odds=odd.moneyline_home_odds,
                            ml_away_odds=odd.moneyline_away_odds,
                            total=odd.total_value,
                            total_over_odds=odd.total_over_odds,
                            total_under_odds=odd.total_under_odds,
                            raw_payload=payload,
                        ).on_conflict_do_update(
                            index_elements=["game_id", "vendor", "snapshot_window"],
                            set_=dict(
                                odds_id=odd.id,
                                spread_home=odd.spread_home_value,
                                spread_away=odd.spread_away_value,
                                spread_home_odds=odd.spread_home_odds,
                                spread_away_odds=odd.spread_away_odds,
                                ml_home_odds=odd.moneyline_home_odds,
                                ml_away_odds=odd.moneyline_away_odds,
                                total=odd.total_value,
                                total_over_odds=odd.total_over_odds,
                                total_under_odds=odd.total_under_odds,
                                raw_payload=payload,
                            ),
                        )
                        db.execute(stmt)
                        total_odds += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("_poll_mlb_odds DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("mlb_odds", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "_poll_mlb_odds: %d games, %d with odds, %d snapshots in %dms (window=%s)",
                len(games), games_with_odds, total_odds, elapsed,
                snapshot_window.strftime("%H:%M"),
            )
            self._record_job_run("mlb_odds", "success", total_odds)
            return {
                "status": "success",
                "records": total_odds,
                "games": len(games),
                "games_with_odds": games_with_odds,
                "snapshot_window": snapshot_window.isoformat(),
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["mlb_odds"], _run)

    async def _ingest_mlb_game_log(self) -> dict:
        """
        Daily MLB game-log ingestion (lock 100_016).

        Fetches two dates per run:
          - yesterday_et(): finalize scores for games that finished overnight
          - today_et():     seed today's scheduled games

        For each game:
          1. Upsert mlb_team (home + away) -- dimension before fact, FK dependency
          2. Upsert mlb_game_log on game_id:
             - Immutable: game_date, season, team_ids, venue, ingested_at
             - Updated:   status, scores, attendance, period, raw_payload, updated_at

        Scores (home_runs, away_runs, hits, errors) written only when game has started
        (STATUS_IN_PROGRESS or STATUS_FINAL). Pre-game rows store NULL for score fields.

        Anomaly check: logs WARNING if BDL returns 0 games for a date.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("mlb_game_log: BDL init failed -- %s", exc)
                self._record_job_run("mlb_game_log", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            today = today_et()
            yesterday = today - timedelta(days=1)
            dates = [yesterday.isoformat(), today.isoformat()]

            total_games = 0
            db = SessionLocal()
            try:
                for date_str in dates:
                    games = await asyncio.to_thread(bdl.get_mlb_games, date_str)

                    if not games:
                        logger.warning(
                            "mlb_game_log: 0 games returned for %s -- off-day or BDL error",
                            date_str,
                        )
                        continue

                    for game in games:
                        # Step 1: upsert both teams (dimension before fact -- FK dependency)
                        for team in (game.home_team, game.away_team):
                            stmt = pg_insert(MLBTeam.__table__).values(
                                team_id=team.id,
                                abbreviation=team.abbreviation,
                                name=team.name,
                                display_name=team.display_name,
                                short_name=team.short_display_name,
                                location=team.location,
                                slug=team.slug,
                                league=team.league,
                                division=team.division,
                                ingested_at=now_et(),
                            ).on_conflict_do_update(
                                index_elements=["team_id"],
                                set_=dict(
                                    abbreviation=team.abbreviation,
                                    name=team.name,
                                    display_name=team.display_name,
                                    short_name=team.short_display_name,
                                    location=team.location,
                                    slug=team.slug,
                                    league=team.league,
                                    division=team.division,
                                ),
                            )
                            db.execute(stmt)

                        # Step 2: convert UTC ISO 8601 game timestamp to ET date
                        dt_utc = datetime.fromisoformat(game.date.replace("Z", "+00:00"))
                        game_date_et = dt_utc.astimezone(ZoneInfo("America/New_York")).date()

                        # Scores are meaningful only when game has started
                        is_active = game.status in {"STATUS_FINAL", "STATUS_IN_PROGRESS"}
                        home_runs   = game.home_team_data.runs   if is_active else None
                        away_runs   = game.away_team_data.runs   if is_active else None
                        home_hits   = game.home_team_data.hits   if is_active else None
                        away_hits   = game.away_team_data.hits   if is_active else None
                        home_errors = game.home_team_data.errors if is_active else None
                        away_errors = game.away_team_data.errors if is_active else None
                        attendance  = (game.attendance or None)  if is_active else None

                        # Step 3: upsert game log -- idempotent on game_id
                        now = now_et()
                        payload = game.model_dump()
                        stmt = pg_insert(MLBGameLog.__table__).values(
                            game_id=game.id,
                            game_date=game_date_et,
                            season=game.season,
                            season_type=game.season_type,
                            status=game.status,
                            home_team_id=game.home_team.id,
                            away_team_id=game.away_team.id,
                            home_runs=home_runs,
                            away_runs=away_runs,
                            home_hits=home_hits,
                            away_hits=away_hits,
                            home_errors=home_errors,
                            away_errors=away_errors,
                            venue=game.venue,
                            attendance=attendance,
                            period=game.period,
                            raw_payload=payload,
                            ingested_at=now,
                            updated_at=now,
                        ).on_conflict_do_update(
                            index_elements=["game_id"],
                            set_=dict(
                                status=game.status,
                                home_runs=home_runs,
                                away_runs=away_runs,
                                home_hits=home_hits,
                                away_hits=away_hits,
                                home_errors=home_errors,
                                away_errors=away_errors,
                                attendance=attendance,
                                period=game.period,
                                raw_payload=payload,
                                updated_at=now,
                            ),
                        )
                        db.execute(stmt)
                        total_games += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("mlb_game_log DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("mlb_game_log", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "mlb_game_log: %d games upserted across %s in %dms",
                total_games, dates, elapsed,
            )
            self._record_job_run("mlb_game_log", "success", total_games)
            return {
                "status": "success",
                "records": total_games,
                "dates": dates,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["mlb_game_log"], _run)

    async def _ingest_mlb_box_stats(self) -> dict:
        """
        Daily MLB player box stats ingestion (lock 100_017).

        Runs at 2 AM ET -- 1 hour after game-log (1 AM) so mlb_game_log rows
        are present when we write the FK game_id into mlb_player_stats.

        For each stat row:
          - Validates via MLBPlayerStats Pydantic V2 contract (validation failures
            are logged at WARNING and skipped -- never kills the job)
          - Upserts into mlb_player_stats on (bdl_player_id, game_id)
          - Dual-write: raw_payload = stat.model_dump()

        Anomaly: if BDL returns 0 stat rows for dates that had scheduled games,
        logs WARNING (network blip / off-day -- does not raise).
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.balldontlie import BallDontLieClient
            try:
                bdl = BallDontLieClient()
            except ValueError as exc:
                logger.error("mlb_box_stats: BDL init failed -- %s", exc)
                self._record_job_run("mlb_box_stats", "skipped")
                return {"status": "skipped", "records": 0, "elapsed_ms": 0}

            today = today_et()
            yesterday = today - timedelta(days=1)
            date_strs = [yesterday.isoformat(), today.isoformat()]

            stats = await asyncio.to_thread(bdl.get_mlb_stats, date_strs)

            if not stats:
                logger.warning(
                    "mlb_box_stats: 0 stat rows returned for dates=%s -- off-day or BDL error",
                    date_strs,
                )
                self._record_job_run("mlb_box_stats", "success", 0)
                elapsed = int((time.monotonic() - t0) * 1000)
                return {"status": "success", "records": 0, "dates": date_strs, "elapsed_ms": elapsed}

            now = now_et()
            rows_upserted = 0
            db = SessionLocal()
            try:
                for stat in stats:
                    # game_date: prefer fetching from game_id FK's game_date; fall back
                    # to today if unknown (stats always belong to current polling window)
                    # We do a best-effort date: since we queried by date, use today as fallback.
                    # The migration note: game_date is NOT NULL -- we must supply a value.
                    # Use today as the safe fallback when stat has no date context.
                    game_date = today

                    payload = stat.model_dump()
                    stmt = pg_insert(MLBPlayerStats.__table__).values(
                        bdl_stat_id=stat.id,
                        bdl_player_id=stat.bdl_player_id,
                        game_id=stat.game_id,
                        game_date=game_date,
                        season=stat.season if stat.season is not None else 2026,
                        # Batting
                        ab=stat.ab,
                        runs=stat.r,
                        hits=stat.h,
                        doubles=stat.double,
                        triples=stat.triple,
                        home_runs=stat.hr,
                        rbi=stat.rbi,
                        walks=stat.bb,
                        strikeouts_bat=stat.so,
                        stolen_bases=stat.sb,
                        caught_stealing=stat.cs,
                        avg=stat.avg,
                        obp=stat.obp,
                        slg=stat.slg,
                        ops=stat.ops,
                        # Pitching
                        innings_pitched=stat.ip,
                        hits_allowed=stat.h_allowed,
                        runs_allowed=stat.r_allowed,
                        earned_runs=stat.er,
                        walks_allowed=stat.bb_allowed,
                        strikeouts_pit=stat.k,
                        whip=stat.whip,
                        era=stat.era,
                        # Audit
                        raw_payload=payload,
                        ingested_at=now,
                    ).on_conflict_do_update(
                        constraint="_mps_player_game_uc",
                        set_=dict(
                            bdl_stat_id=stat.id,
                            season=stat.season if stat.season is not None else 2026,
                            ab=stat.ab,
                            runs=stat.r,
                            hits=stat.h,
                            doubles=stat.double,
                            triples=stat.triple,
                            home_runs=stat.hr,
                            rbi=stat.rbi,
                            walks=stat.bb,
                            strikeouts_bat=stat.so,
                            stolen_bases=stat.sb,
                            caught_stealing=stat.cs,
                            avg=stat.avg,
                            obp=stat.obp,
                            slg=stat.slg,
                            ops=stat.ops,
                            innings_pitched=stat.ip,
                            hits_allowed=stat.h_allowed,
                            runs_allowed=stat.r_allowed,
                            earned_runs=stat.er,
                            walks_allowed=stat.bb_allowed,
                            strikeouts_pit=stat.k,
                            whip=stat.whip,
                            era=stat.era,
                            raw_payload=payload,
                        ),
                    )
                    db.execute(stmt)
                    rows_upserted += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("mlb_box_stats DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("mlb_box_stats", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "mlb_box_stats: %d rows upserted for dates=%s in %dms",
                rows_upserted, date_strs, elapsed,
            )
            self._record_job_run("mlb_box_stats", "success", rows_upserted)
            return {
                "status": "success",
                "records": rows_upserted,
                "dates": date_strs,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["mlb_box_stats"], _run)

    async def _compute_rolling_windows(self) -> dict:
        """
        Daily rolling window computation (lock 100_018, 3 AM ET).

        Runs after _ingest_mlb_box_stats (2 AM) so today's box stats are present.

        Algorithm:
          1. Query all mlb_player_stats rows for the past 30 days (max window)
          2. Compute 7/14/30-day decay-weighted windows for every player with data
          3. Upsert to player_rolling_stats on (bdl_player_id, as_of_date, window_days)

        Anomaly: logs WARNING if 0 players processed (likely off-day or box stats missing).
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.rolling_window_engine import compute_all_rolling_windows

            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            lookback_start = as_of_date - timedelta(days=30)

            db = SessionLocal()
            try:
                rows = (
                    db.query(MLBPlayerStats)
                    .filter(
                        MLBPlayerStats.game_date >= lookback_start,
                        MLBPlayerStats.game_date <= as_of_date,
                    )
                    .all()
                )
            except Exception as exc:
                db.close()
                logger.error("rolling_windows: DB query failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_windows", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            if not rows:
                logger.warning(
                    "rolling_windows: 0 stat rows found for window %s..%s -- off-day or box stats missing",
                    lookback_start, as_of_date,
                )
                db.close()
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_windows", "success", 0)
                return {
                    "status": "success",
                    "as_of_date": str(as_of_date),
                    "players_processed": 0,
                    "rows_upserted": 0,
                    "elapsed_ms": elapsed,
                }

            results = compute_all_rolling_windows(
                rows,
                as_of_date=as_of_date,
                window_sizes=[7, 14, 30],
            )

            players_processed = len({r.bdl_player_id for r in results})

            if players_processed == 0:
                logger.warning(
                    "rolling_windows: compute_all_rolling_windows returned 0 results for as_of_date=%s",
                    as_of_date,
                )

            now = datetime.now(ZoneInfo("America/New_York"))
            rows_upserted = 0
            try:
                for res in results:
                    stmt = pg_insert(PlayerRollingStats.__table__).values(
                        bdl_player_id=res.bdl_player_id,
                        as_of_date=res.as_of_date,
                        window_days=res.window_days,
                        games_in_window=res.games_in_window,
                        w_ab=res.w_ab,
                        w_hits=res.w_hits,
                        w_doubles=res.w_doubles,
                        w_triples=res.w_triples,
                        w_home_runs=res.w_home_runs,
                        w_rbi=res.w_rbi,
                        w_walks=res.w_walks,
                        w_strikeouts_bat=res.w_strikeouts_bat,
                        w_stolen_bases=res.w_stolen_bases,
                        w_avg=res.w_avg,
                        w_obp=res.w_obp,
                        w_slg=res.w_slg,
                        w_ops=res.w_ops,
                        w_ip=res.w_ip,
                        w_earned_runs=res.w_earned_runs,
                        w_hits_allowed=res.w_hits_allowed,
                        w_walks_allowed=res.w_walks_allowed,
                        w_strikeouts_pit=res.w_strikeouts_pit,
                        w_era=res.w_era,
                        w_whip=res.w_whip,
                        w_k_per_9=res.w_k_per_9,
                        computed_at=now,
                    ).on_conflict_do_update(
                        constraint="_prs_player_date_window_uc",
                        set_=dict(
                            games_in_window=res.games_in_window,
                            w_ab=res.w_ab,
                            w_hits=res.w_hits,
                            w_doubles=res.w_doubles,
                            w_triples=res.w_triples,
                            w_home_runs=res.w_home_runs,
                            w_rbi=res.w_rbi,
                            w_walks=res.w_walks,
                            w_strikeouts_bat=res.w_strikeouts_bat,
                            w_stolen_bases=res.w_stolen_bases,
                            w_avg=res.w_avg,
                            w_obp=res.w_obp,
                            w_slg=res.w_slg,
                            w_ops=res.w_ops,
                            w_ip=res.w_ip,
                            w_earned_runs=res.w_earned_runs,
                            w_hits_allowed=res.w_hits_allowed,
                            w_walks_allowed=res.w_walks_allowed,
                            w_strikeouts_pit=res.w_strikeouts_pit,
                            w_era=res.w_era,
                            w_whip=res.w_whip,
                            w_k_per_9=res.w_k_per_9,
                            computed_at=now,
                        ),
                    )
                    db.execute(stmt)
                    rows_upserted += 1

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("rolling_windows: DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("rolling_windows", "failed")
                return {"status": "failed", "records": 0, "elapsed_ms": elapsed}
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "rolling_windows: %d rows upserted for %d players, as_of_date=%s in %dms",
                rows_upserted, players_processed, as_of_date, elapsed,
            )
            self._record_job_run("rolling_windows", "success", rows_upserted)
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "players_processed": players_processed,
                "rows_upserted": rows_upserted,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["rolling_windows"], _run)

    async def _compute_player_scores(self) -> dict:
        """
        Daily Z-score scoring computation (lock 100_019, 4 AM ET).

        Runs after _compute_rolling_windows (3 AM) so player_rolling_stats is current.

        Algorithm:
          1. For each window_days in [7, 14, 30]:
             a. Query all player_rolling_stats WHERE as_of_date = yesterday AND window_days = N
             b. Call compute_league_zscores(rows, yesterday, N)
             c. Upsert each PlayerScoreResult to player_scores table
          2. Anomaly: WARN if 0 players scored for any window
          3. Return scored counts per window

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            from backend.services.scoring_engine import compute_league_zscores

            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

            scored_7d = 0
            scored_14d = 0
            scored_30d = 0

            db = SessionLocal()
            try:
                now = datetime.now(ZoneInfo("America/New_York"))

                for window_days in [7, 14, 30]:
                    try:
                        rows = (
                            db.query(PlayerRollingStats)
                            .filter(
                                PlayerRollingStats.as_of_date == as_of_date,
                                PlayerRollingStats.window_days == window_days,
                            )
                            .all()
                        )
                    except Exception as exc:
                        logger.error(
                            "player_scores: DB query failed for window_days=%d: %s",
                            window_days, exc,
                        )
                        elapsed = int((time.monotonic() - t0) * 1000)
                        self._record_job_run("player_scores", "failed")
                        return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

                    if not rows:
                        logger.warning(
                            "player_scores: 0 rolling_stats rows found for as_of_date=%s "
                            "window_days=%d -- off-day or rolling windows missing",
                            as_of_date, window_days,
                        )

                    score_results = compute_league_zscores(rows, as_of_date, window_days)

                    if not score_results:
                        logger.warning(
                            "player_scores: 0 players scored for as_of_date=%s window_days=%d",
                            as_of_date, window_days,
                        )

                    try:
                        for res in score_results:
                            stmt = pg_insert(PlayerScore.__table__).values(
                                bdl_player_id=res.bdl_player_id,
                                as_of_date=res.as_of_date,
                                window_days=res.window_days,
                                player_type=res.player_type,
                                games_in_window=res.games_in_window,
                                z_hr=res.z_hr,
                                z_rbi=res.z_rbi,
                                z_sb=res.z_sb,
                                z_avg=res.z_avg,
                                z_obp=res.z_obp,
                                z_era=res.z_era,
                                z_whip=res.z_whip,
                                z_k_per_9=res.z_k_per_9,
                                composite_z=res.composite_z,
                                score_0_100=res.score_0_100,
                                confidence=res.confidence,
                                computed_at=now,
                            ).on_conflict_do_update(
                                constraint="_ps_player_date_window_uc",
                                set_=dict(
                                    player_type=res.player_type,
                                    games_in_window=res.games_in_window,
                                    z_hr=res.z_hr,
                                    z_rbi=res.z_rbi,
                                    z_sb=res.z_sb,
                                    z_avg=res.z_avg,
                                    z_obp=res.z_obp,
                                    z_era=res.z_era,
                                    z_whip=res.z_whip,
                                    z_k_per_9=res.z_k_per_9,
                                    composite_z=res.composite_z,
                                    score_0_100=res.score_0_100,
                                    confidence=res.confidence,
                                    computed_at=now,
                                ),
                            )
                            db.execute(stmt)

                        db.commit()
                    except Exception as exc:
                        db.rollback()
                        logger.error(
                            "player_scores: DB write failed for window_days=%d: %s",
                            window_days, exc,
                        )
                        elapsed = int((time.monotonic() - t0) * 1000)
                        self._record_job_run("player_scores", "failed")
                        return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

                    count = len(score_results)
                    if window_days == 7:
                        scored_7d = count
                    elif window_days == 14:
                        scored_14d = count
                    else:
                        scored_30d = count

                    logger.info(
                        "player_scores: %d players scored for as_of_date=%s window_days=%d",
                        count, as_of_date, window_days,
                    )

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            total = scored_7d + scored_14d + scored_30d
            self._record_job_run("player_scores", "success", total)
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "scored_7d": scored_7d,
                "scored_14d": scored_14d,
                "scored_30d": scored_30d,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["player_scores"], _run)

    async def _compute_player_momentum(self) -> dict:
        """
        Daily momentum signal computation (lock 100_020, 5 AM ET).

        Runs after _compute_player_scores (4 AM) so player_scores is current.

        Algorithm:
          1. Query player_scores WHERE as_of_date = yesterday AND window_days = 14
          2. Query player_scores WHERE as_of_date = yesterday AND window_days = 30
          3. compute_all_momentum(scores_14d, scores_30d)
          4. Upsert each MomentumResult to player_momentum ON CONFLICT (_pm_player_date_uc)
          5. WARN if 0 results (off-day or scoring pipeline missing)
          6. WARN if any single signal exceeds 60% of total (indicates potential data issue)
          7. Return {"as_of_date": str(yesterday), "total": n, "signals": {signal: count}}

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            from collections import Counter
            from backend.services.momentum_engine import compute_all_momentum

            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)

            db = SessionLocal()
            results = []
            try:
                now = datetime.now(ZoneInfo("America/New_York"))

                try:
                    scores_14d = (
                        db.query(PlayerScore)
                        .filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 14,
                        )
                        .all()
                    )
                    scores_30d = (
                        db.query(PlayerScore)
                        .filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 30,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "player_momentum: DB query failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("player_momentum", "failed")
                    return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

                results = compute_all_momentum(scores_14d, scores_30d)

                if not results:
                    logger.warning(
                        "player_momentum: 0 players computed for as_of_date=%s -- "
                        "off-day or player_scores pipeline missing",
                        as_of_date,
                    )

                # Anomaly: warn if any single signal dominates (>60% of total)
                if results:
                    signal_counts: dict = Counter(r.signal for r in results)
                    total_check = len(results)
                    for sig, cnt in signal_counts.items():
                        if cnt / total_check > 0.60:
                            logger.warning(
                                "player_momentum: signal '%s' is %.1f%% of total (%d/%d) "
                                "for as_of_date=%s -- possible data issue",
                                sig, cnt / total_check * 100, cnt, total_check, as_of_date,
                            )

                try:
                    for res in results:
                        stmt = pg_insert(PlayerMomentum.__table__).values(
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            player_type=res.player_type,
                            delta_z=res.delta_z,
                            signal=res.signal,
                            composite_z_14d=res.composite_z_14d,
                            composite_z_30d=res.composite_z_30d,
                            score_14d=res.score_14d,
                            score_30d=res.score_30d,
                            confidence_14d=res.confidence_14d,
                            confidence_30d=res.confidence_30d,
                            confidence=res.confidence,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_pm_player_date_uc",
                            set_=dict(
                                player_type=res.player_type,
                                delta_z=res.delta_z,
                                signal=res.signal,
                                composite_z_14d=res.composite_z_14d,
                                composite_z_30d=res.composite_z_30d,
                                score_14d=res.score_14d,
                                score_30d=res.score_30d,
                                confidence_14d=res.confidence_14d,
                                confidence_30d=res.confidence_30d,
                                confidence=res.confidence,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "player_momentum: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("player_momentum", "failed")
                    return {"status": "failed", "records": 0, "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            total = len(results)
            signal_counts = Counter(r.signal for r in results)
            self._record_job_run("player_momentum", "success", total)
            logger.info(
                "player_momentum: %d players computed for as_of_date=%s signals=%s",
                total, as_of_date, dict(signal_counts),
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "total": total,
                "signals": dict(signal_counts),
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["player_momentum"], _run)

    async def _run_ros_simulation(self) -> dict:
        """
        Daily Rest-of-Season Monte Carlo simulation (lock 100_021, 6 AM ET).

        Runs after _compute_player_momentum (5 AM) so momentum layer is current.

        Algorithm:
          1. Query player_rolling_stats WHERE as_of_date = yesterday AND window_days = 14
          2. simulate_all_players(rows, remaining_games=REMAINING_GAMES_DEFAULT)
             -> list[SimulationResult dataclass]
          3. Upsert each result to simulation_results ON CONFLICT (_sr_player_date_uc)
          4. WARN if 0 players simulated (off-day or rolling_windows pipeline missing)
          5. Return {"as_of_date": str(yesterday), "players_simulated": n}

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 1: fetch 14d rolling window rows
                try:
                    rolling_rows = (
                        db.query(PlayerRollingStats)
                        .filter(
                            PlayerRollingStats.as_of_date == as_of_date,
                            PlayerRollingStats.window_days == 14,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "ros_simulation: DB query failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("ros_simulation", "failed")
                    return {"status": "failed", "players_simulated": 0, "elapsed_ms": elapsed}

                # Step 2: run simulations (CPU-bound -- offload to thread pool)
                sim_results = await asyncio.to_thread(
                    simulate_all_players,
                    rolling_rows,
                    REMAINING_GAMES_DEFAULT,
                    1000,
                )

                if not sim_results:
                    logger.warning(
                        "ros_simulation: 0 players simulated for as_of_date=%s -- "
                        "off-day or rolling_windows pipeline missing",
                        as_of_date,
                    )

                # Step 3: upsert results
                try:
                    for res in sim_results:
                        stmt = pg_insert(SimulationResultORM.__table__).values(
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            window_days=res.window_days,
                            remaining_games=res.remaining_games,
                            n_simulations=res.n_simulations,
                            player_type=res.player_type,
                            proj_hr_p10=res.proj_hr_p10,
                            proj_hr_p25=res.proj_hr_p25,
                            proj_hr_p50=res.proj_hr_p50,
                            proj_hr_p75=res.proj_hr_p75,
                            proj_hr_p90=res.proj_hr_p90,
                            proj_rbi_p10=res.proj_rbi_p10,
                            proj_rbi_p25=res.proj_rbi_p25,
                            proj_rbi_p50=res.proj_rbi_p50,
                            proj_rbi_p75=res.proj_rbi_p75,
                            proj_rbi_p90=res.proj_rbi_p90,
                            proj_sb_p10=res.proj_sb_p10,
                            proj_sb_p25=res.proj_sb_p25,
                            proj_sb_p50=res.proj_sb_p50,
                            proj_sb_p75=res.proj_sb_p75,
                            proj_sb_p90=res.proj_sb_p90,
                            proj_avg_p10=res.proj_avg_p10,
                            proj_avg_p25=res.proj_avg_p25,
                            proj_avg_p50=res.proj_avg_p50,
                            proj_avg_p75=res.proj_avg_p75,
                            proj_avg_p90=res.proj_avg_p90,
                            proj_k_p10=res.proj_k_p10,
                            proj_k_p25=res.proj_k_p25,
                            proj_k_p50=res.proj_k_p50,
                            proj_k_p75=res.proj_k_p75,
                            proj_k_p90=res.proj_k_p90,
                            proj_era_p10=res.proj_era_p10,
                            proj_era_p25=res.proj_era_p25,
                            proj_era_p50=res.proj_era_p50,
                            proj_era_p75=res.proj_era_p75,
                            proj_era_p90=res.proj_era_p90,
                            proj_whip_p10=res.proj_whip_p10,
                            proj_whip_p25=res.proj_whip_p25,
                            proj_whip_p50=res.proj_whip_p50,
                            proj_whip_p75=res.proj_whip_p75,
                            proj_whip_p90=res.proj_whip_p90,
                            composite_variance=res.composite_variance,
                            downside_p25=res.downside_p25,
                            upside_p75=res.upside_p75,
                            prob_above_median=res.prob_above_median,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_sr_player_date_uc",
                            set_=dict(
                                window_days=res.window_days,
                                remaining_games=res.remaining_games,
                                n_simulations=res.n_simulations,
                                player_type=res.player_type,
                                proj_hr_p10=res.proj_hr_p10,
                                proj_hr_p25=res.proj_hr_p25,
                                proj_hr_p50=res.proj_hr_p50,
                                proj_hr_p75=res.proj_hr_p75,
                                proj_hr_p90=res.proj_hr_p90,
                                proj_rbi_p10=res.proj_rbi_p10,
                                proj_rbi_p25=res.proj_rbi_p25,
                                proj_rbi_p50=res.proj_rbi_p50,
                                proj_rbi_p75=res.proj_rbi_p75,
                                proj_rbi_p90=res.proj_rbi_p90,
                                proj_sb_p10=res.proj_sb_p10,
                                proj_sb_p25=res.proj_sb_p25,
                                proj_sb_p50=res.proj_sb_p50,
                                proj_sb_p75=res.proj_sb_p75,
                                proj_sb_p90=res.proj_sb_p90,
                                proj_avg_p10=res.proj_avg_p10,
                                proj_avg_p25=res.proj_avg_p25,
                                proj_avg_p50=res.proj_avg_p50,
                                proj_avg_p75=res.proj_avg_p75,
                                proj_avg_p90=res.proj_avg_p90,
                                proj_k_p10=res.proj_k_p10,
                                proj_k_p25=res.proj_k_p25,
                                proj_k_p50=res.proj_k_p50,
                                proj_k_p75=res.proj_k_p75,
                                proj_k_p90=res.proj_k_p90,
                                proj_era_p10=res.proj_era_p10,
                                proj_era_p25=res.proj_era_p25,
                                proj_era_p50=res.proj_era_p50,
                                proj_era_p75=res.proj_era_p75,
                                proj_era_p90=res.proj_era_p90,
                                proj_whip_p10=res.proj_whip_p10,
                                proj_whip_p25=res.proj_whip_p25,
                                proj_whip_p50=res.proj_whip_p50,
                                proj_whip_p75=res.proj_whip_p75,
                                proj_whip_p90=res.proj_whip_p90,
                                composite_variance=res.composite_variance,
                                downside_p25=res.downside_p25,
                                upside_p75=res.upside_p75,
                                prob_above_median=res.prob_above_median,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "ros_simulation: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("ros_simulation", "failed")
                    return {"status": "failed", "players_simulated": 0, "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n = len(sim_results)
            self._record_job_run("ros_simulation", "success", n)
            logger.info(
                "ros_simulation: %d players simulated for as_of_date=%s "
                "remaining_games=%d elapsed_ms=%d",
                n, as_of_date, REMAINING_GAMES_DEFAULT, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "players_simulated": n,
                "remaining_games": REMAINING_GAMES_DEFAULT,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["ros_simulation"], _run)

    async def _run_decision_optimization(self) -> dict:
        """
        Daily Decision Engine optimization (lock 100_022, 7 AM ET).

        Runs after _run_ros_simulation (6 AM) so simulation_results is current.

        Algorithm:
          1. Query player_scores WHERE as_of_date = yesterday AND window_days = 14
          2. Query player_momentum WHERE as_of_date = yesterday (join on bdl_player_id)
          3. Query simulation_results WHERE as_of_date = yesterday (join on bdl_player_id)
          4. Build PlayerDecisionInput list from joined data
          5. Call optimize_lineup(players) and optimize_waivers(players, waiver_pool=[])
          6. Upsert DecisionResult ORM rows ON CONFLICT _dr_date_type_player_uc DO UPDATE
          7. Return summary dict

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 1: fetch player_scores (14d window)
                try:
                    score_rows = (
                        db.query(PlayerScore)
                        .filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 14,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "decision_optimization: player_scores query failed for %s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("decision_optimization", "failed")
                    return {"status": "failed", "lineup_decisions": 0, "waiver_decisions": 0, "elapsed_ms": elapsed}

                # Step 2: fetch player_momentum
                try:
                    momentum_rows = (
                        db.query(PlayerMomentum)
                        .filter(PlayerMomentum.as_of_date == as_of_date)
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "decision_optimization: player_momentum query failed for %s: %s",
                        as_of_date, exc,
                    )
                    momentum_rows = []

                # Step 3: fetch simulation_results
                try:
                    sim_rows = (
                        db.query(SimulationResultORM)
                        .filter(SimulationResultORM.as_of_date == as_of_date)
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "decision_optimization: simulation_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    sim_rows = []

                if not score_rows:
                    logger.warning(
                        "decision_optimization: 0 player_scores rows for %s -- "
                        "player_scores pipeline may not have run",
                        as_of_date,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("decision_optimization", "success", 0)
                    return {
                        "status": "success",
                        "as_of_date": str(as_of_date),
                        "lineup_decisions": 0,
                        "waiver_decisions": 0,
                        "elapsed_ms": elapsed,
                    }

                # Step 4: build lookup dicts for join
                momentum_by_id = {r.bdl_player_id: r for r in momentum_rows}
                sim_by_id      = {r.bdl_player_id: r for r in sim_rows}

                players = []
                for score in score_rows:
                    pid = score.bdl_player_id
                    mom  = momentum_by_id.get(pid)
                    sim  = sim_by_id.get(pid)

                    # Derive eligible_positions from player_type heuristic
                    # (full position eligibility requires Yahoo roster data -- stubbed here)
                    # Prefer simulation player_type; fall back to player_scores player_type
                    pt = (sim.player_type if sim else None) or score.player_type or "unknown"
                    if pt == "hitter":
                        eligible = ["Util"]
                    elif pt == "pitcher":
                        eligible = ["P"]
                    elif pt == "two_way":
                        eligible = ["Util", "P"]
                    else:
                        eligible = []

                    players.append(PlayerDecisionInput(
                        bdl_player_id=pid,
                        name=getattr(score, "player_name", str(pid)),
                        player_type=pt,
                        eligible_positions=eligible,
                        score_0_100=score.score_0_100 or 0.0,
                        composite_z=score.composite_z or 0.0,
                        momentum_signal=mom.signal if mom else "STABLE",
                        delta_z=mom.delta_z if mom else 0.0,
                        proj_hr_p50=sim.proj_hr_p50    if sim else None,
                        proj_rbi_p50=sim.proj_rbi_p50  if sim else None,
                        proj_sb_p50=sim.proj_sb_p50    if sim else None,
                        proj_avg_p50=sim.proj_avg_p50  if sim else None,
                        proj_k_p50=sim.proj_k_p50      if sim else None,
                        proj_era_p50=sim.proj_era_p50  if sim else None,
                        proj_whip_p50=sim.proj_whip_p50 if sim else None,
                        downside_p25=sim.downside_p25  if sim else None,
                        upside_p75=sim.upside_p75      if sim else None,
                    ))

                # Step 5: run decision engine (CPU-bound -- offload to thread pool)
                lineup_decision, lineup_results = await asyncio.to_thread(
                    optimize_lineup, players, as_of_date
                )
                # Waiver pool is empty for now -- no waiver pool query yet (stub)
                _waiver_decision, waiver_results = await asyncio.to_thread(
                    optimize_waivers, players, [], as_of_date
                )

                all_results = lineup_results + waiver_results

                # Step 6: upsert DecisionResult rows
                try:
                    for res in all_results:
                        stmt = pg_insert(DecisionResultORM.__table__).values(
                            as_of_date=res.as_of_date,
                            decision_type=res.decision_type,
                            bdl_player_id=res.bdl_player_id,
                            target_slot=res.target_slot,
                            drop_player_id=res.drop_player_id,
                            lineup_score=res.lineup_score,
                            value_gain=res.value_gain,
                            confidence=res.confidence,
                            reasoning=res.reasoning,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_dr_date_type_player_uc",
                            set_=dict(
                                target_slot=res.target_slot,
                                drop_player_id=res.drop_player_id,
                                lineup_score=res.lineup_score,
                                value_gain=res.value_gain,
                                confidence=res.confidence,
                                reasoning=res.reasoning,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "decision_optimization: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("decision_optimization", "failed")
                    return {
                        "status": "failed",
                        "lineup_decisions": 0,
                        "waiver_decisions": 0,
                        "elapsed_ms": elapsed,
                    }

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n_lineup = len(lineup_results)
            n_waiver = len(waiver_results)
            self._record_job_run("decision_optimization", "success", n_lineup + n_waiver)
            logger.info(
                "decision_optimization: %d lineup + %d waiver decisions for as_of_date=%s "
                "elapsed_ms=%d",
                n_lineup, n_waiver, as_of_date, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "lineup_decisions": n_lineup,
                "waiver_decisions": n_waiver,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["decision_optimization"], _run)

    async def _run_backtesting(self) -> dict:
        """
        Daily Backtesting Harness (lock 100_023, 8 AM ET).

        Runs after _run_decision_optimization (7 AM) so the full pipeline is current.

        Algorithm:
          1. Compute as_of_date = yesterday
          2. Query simulation_results WHERE as_of_date = yesterday AND window_days = 14
          3. For each sim_row, query mlb_player_stats for the 14-day actuals window
          4. Aggregate actuals: sum HR/RBI/SB/K, mean AVG, IP-weighted ERA/WHIP
          5. Build BacktestInput list and call evaluate_cohort via asyncio.to_thread
          6. Call summarize() with golden baseline loaded from BASELINE_PATH
          7. Save new golden baseline if no regression detected
          8. Upsert BacktestResultORM rows ON CONFLICT _br_player_date_uc DO UPDATE
          9. Return summary dict

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            window_start = as_of_date - timedelta(days=14)
            now = datetime.now(ZoneInfo("America/New_York"))

            results = []   # populated after evaluate_cohort; guard for finally path
            summary = None  # populated after summarize(); guard for finally path

            db = SessionLocal()
            try:
                # Step 1: fetch simulation_results for yesterday (14d window)
                try:
                    sim_rows = (
                        db.query(SimulationResultORM)
                        .filter(
                            SimulationResultORM.as_of_date == as_of_date,
                            SimulationResultORM.window_days == 14,
                        )
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "backtesting: simulation_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("backtesting", "failed")
                    return {"status": "failed", "n_players": 0, "elapsed_ms": elapsed}

                if not sim_rows:
                    logger.warning(
                        "backtesting: 0 simulation_results rows for as_of_date=%s -- "
                        "ros_simulation pipeline may not have run",
                        as_of_date,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("backtesting", "success", 0)
                    return {
                        "status": "success",
                        "as_of_date": str(as_of_date),
                        "n_players": 0,
                        "mean_composite_mae": None,
                        "regression_detected": False,
                        "elapsed_ms": elapsed,
                    }

                # Step 2: for each player, fetch actual stats from the 14-day window
                inputs = []
                for sim in sim_rows:
                    pid = sim.bdl_player_id
                    try:
                        stat_rows = (
                            db.query(MLBPlayerStats)
                            .filter(
                                MLBPlayerStats.bdl_player_id == pid,
                                MLBPlayerStats.game_date >= window_start,
                                MLBPlayerStats.game_date <= as_of_date,
                            )
                            .all()
                        )
                    except Exception as exc:
                        logger.warning(
                            "backtesting: stats query failed for player %d: %s",
                            pid, exc,
                        )
                        stat_rows = []

                    games_played = len(stat_rows)

                    # Aggregate batting totals
                    actual_hr  = None
                    actual_rbi = None
                    actual_sb  = None
                    actual_avg = None
                    actual_k   = None
                    actual_era = None
                    actual_whip = None

                    if stat_rows:
                        hr_vals  = [r.home_runs    for r in stat_rows if r.home_runs    is not None]
                        rbi_vals = [r.rbi           for r in stat_rows if r.rbi          is not None]
                        sb_vals  = [r.stolen_bases  for r in stat_rows if r.stolen_bases is not None]
                        avg_vals = [r.avg            for r in stat_rows if r.avg          is not None]
                        k_vals   = [r.strikeouts_pit for r in stat_rows if r.strikeouts_pit is not None]

                        actual_hr  = float(sum(hr_vals))  if hr_vals  else None
                        actual_rbi = float(sum(rbi_vals)) if rbi_vals else None
                        actual_sb  = float(sum(sb_vals))  if sb_vals  else None
                        actual_avg = sum(avg_vals) / len(avg_vals) if avg_vals else None
                        actual_k   = float(sum(k_vals))   if k_vals   else None

                        # IP-weighted ERA and WHIP aggregation
                        # innings_pitched stored as string e.g. "6.2" meaning 6 and 2/3
                        total_ip = 0.0
                        era_sum  = 0.0
                        whip_sum = 0.0
                        for r in stat_rows:
                            ip_str = r.innings_pitched
                            if ip_str is None:
                                continue
                            try:
                                parts = str(ip_str).split(".")
                                whole = int(parts[0])
                                frac  = int(parts[1]) if len(parts) > 1 else 0
                                ip_dec = whole + frac / 3.0
                            except (ValueError, IndexError):
                                ip_dec = 0.0
                            if ip_dec <= 0.0:
                                continue
                            total_ip += ip_dec
                            if r.era is not None:
                                era_sum += r.era * ip_dec
                            if r.whip is not None:
                                whip_sum += r.whip * ip_dec

                        if total_ip > 0.0:
                            actual_era  = era_sum  / total_ip
                            actual_whip = whip_sum / total_ip

                    inputs.append(BacktestInput(
                        bdl_player_id=pid,
                        as_of_date=as_of_date,
                        player_type=sim.player_type,
                        proj_hr_p50=sim.proj_hr_p50,
                        proj_rbi_p50=sim.proj_rbi_p50,
                        proj_sb_p50=sim.proj_sb_p50,
                        proj_avg_p50=sim.proj_avg_p50,
                        proj_k_p50=sim.proj_k_p50,
                        proj_era_p50=sim.proj_era_p50,
                        proj_whip_p50=sim.proj_whip_p50,
                        actual_hr=actual_hr,
                        actual_rbi=actual_rbi,
                        actual_sb=actual_sb,
                        actual_avg=actual_avg,
                        actual_k=actual_k,
                        actual_era=actual_era,
                        actual_whip=actual_whip,
                        games_played=games_played,
                    ))

                # Step 3: evaluate cohort (CPU-bound -- offload to thread pool)
                results = await asyncio.to_thread(evaluate_cohort, inputs)

                # Step 4: summarize with golden baseline
                baseline_data = load_golden_baseline(BASELINE_PATH)
                baseline_mae = baseline_data.get("mean_composite_mae")
                summary = summarize(results, window_start, as_of_date, baseline_mae)

                # Step 5: persist new baseline if no regression
                if not summary.regression_detected:
                    try:
                        save_golden_baseline(summary, BASELINE_PATH)
                    except Exception as exc:
                        logger.warning(
                            "backtesting: could not save golden baseline: %s", exc
                        )

                if summary.regression_detected:
                    logger.warning(
                        "backtesting: REGRESSION DETECTED as_of_date=%s "
                        "mean_composite_mae=%.4f baseline=%.4f delta=%.4f",
                        as_of_date,
                        summary.mean_composite_mae or 0.0,
                        baseline_mae or 0.0,
                        summary.regression_delta or 0.0,
                    )

                # Step 6: upsert BacktestResultORM rows
                try:
                    for res in results:
                        stmt = pg_insert(BacktestResultORM.__table__).values(
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            player_type=res.player_type,
                            games_played=res.games_played,
                            mae_hr=res.mae_hr,
                            rmse_hr=res.rmse_hr,
                            mae_rbi=res.mae_rbi,
                            rmse_rbi=res.rmse_rbi,
                            mae_sb=res.mae_sb,
                            rmse_sb=res.rmse_sb,
                            mae_avg=res.mae_avg,
                            rmse_avg=res.rmse_avg,
                            mae_k=res.mae_k,
                            rmse_k=res.rmse_k,
                            mae_era=res.mae_era,
                            rmse_era=res.rmse_era,
                            mae_whip=res.mae_whip,
                            rmse_whip=res.rmse_whip,
                            composite_mae=res.composite_mae,
                            direction_correct=res.direction_correct,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_br_player_date_uc",
                            set_=dict(
                                player_type=res.player_type,
                                games_played=res.games_played,
                                mae_hr=res.mae_hr,
                                rmse_hr=res.rmse_hr,
                                mae_rbi=res.mae_rbi,
                                rmse_rbi=res.rmse_rbi,
                                mae_sb=res.mae_sb,
                                rmse_sb=res.rmse_sb,
                                mae_avg=res.mae_avg,
                                rmse_avg=res.rmse_avg,
                                mae_k=res.mae_k,
                                rmse_k=res.rmse_k,
                                mae_era=res.mae_era,
                                rmse_era=res.rmse_era,
                                mae_whip=res.mae_whip,
                                rmse_whip=res.rmse_whip,
                                composite_mae=res.composite_mae,
                                direction_correct=res.direction_correct,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "backtesting: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("backtesting", "failed")
                    return {"status": "failed", "n_players": len(results), "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n = len(results)
            self._record_job_run("backtesting", "success", n)
            logger.info(
                "backtesting: %d players evaluated for as_of_date=%s "
                "mean_composite_mae=%s regression=%s elapsed_ms=%d",
                n, as_of_date, summary.mean_composite_mae,
                summary.regression_detected, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "n_players": n,
                "mean_composite_mae": summary.mean_composite_mae,
                "regression_detected": summary.regression_detected,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["backtesting"], _run)

    async def _run_explainability(self) -> dict:
        """
        Daily Explainability Engine (lock 100_024, 9 AM ET).

        Runs after _run_backtesting (8 AM) so all P14-P18 signals are current.

        Algorithm:
          1. Query decision_results WHERE as_of_date = yesterday
          2. For each decision, join player_scores (14d), player_momentum,
             simulation_results, backtest_results, and PlayerIDMapping for names
          3. Build ExplanationInput dataclasses; skip if player_scores row missing
          4. Call explain_batch(inputs) via asyncio.to_thread (CPU-bound)
          5. Upsert DecisionExplanationORM rows ON CONFLICT _de_decision_id_uc DO UPDATE
          6. Return summary dict with n_explained, n_skipped, elapsed_ms

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 1: fetch all decision_results for yesterday
                try:
                    decision_rows = (
                        db.query(DecisionResultORM)
                        .filter(DecisionResultORM.as_of_date == as_of_date)
                        .all()
                    )
                except Exception as exc:
                    logger.error(
                        "explainability: decision_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("explainability", "failed")
                    return {"status": "failed", "n_explained": 0, "n_skipped": 0, "elapsed_ms": elapsed}

                if not decision_rows:
                    logger.warning(
                        "explainability: 0 decision_results rows for as_of_date=%s -- "
                        "decision_optimization pipeline may not have run",
                        as_of_date,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("explainability", "success", 0)
                    return {
                        "status": "success",
                        "as_of_date": str(as_of_date),
                        "n_explained": 0,
                        "n_skipped": 0,
                        "elapsed_ms": elapsed,
                    }

                # Step 2: bulk-fetch supporting tables into dicts keyed by bdl_player_id
                try:
                    score_map = {
                        row.bdl_player_id: row
                        for row in db.query(PlayerScore).filter(
                            PlayerScore.as_of_date == as_of_date,
                            PlayerScore.window_days == 14,
                        ).all()
                    }
                except Exception as exc:
                    logger.error(
                        "explainability: player_scores query failed for %s: %s",
                        as_of_date, exc,
                    )
                    score_map = {}

                try:
                    momentum_map = {
                        row.bdl_player_id: row
                        for row in db.query(PlayerMomentum).filter(
                            PlayerMomentum.as_of_date == as_of_date,
                        ).all()
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: player_momentum query failed for %s: %s",
                        as_of_date, exc,
                    )
                    momentum_map = {}

                try:
                    sim_map = {
                        row.bdl_player_id: row
                        for row in db.query(SimulationResultORM).filter(
                            SimulationResultORM.as_of_date == as_of_date,
                        ).all()
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: simulation_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    sim_map = {}

                try:
                    backtest_map = {
                        row.bdl_player_id: row
                        for row in db.query(BacktestResultORM).filter(
                            BacktestResultORM.as_of_date == as_of_date,
                        ).all()
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: backtest_results query failed for %s: %s",
                        as_of_date, exc,
                    )
                    backtest_map = {}

                # Build a name-lookup dict from PlayerIDMapping (bdl_id -> full_name)
                try:
                    all_pids = set(d.bdl_player_id for d in decision_rows)
                    if decision_rows:
                        # also include drop_player_ids
                        for d in decision_rows:
                            if d.drop_player_id is not None:
                                all_pids.add(d.drop_player_id)
                    name_map = {
                        row.bdl_id: row.full_name
                        for row in db.query(PlayerIDMapping).filter(
                            PlayerIDMapping.bdl_id.in_(list(all_pids)),
                        ).all()
                        if row.bdl_id is not None
                    }
                except Exception as exc:
                    logger.warning(
                        "explainability: PlayerIDMapping query failed: %s", exc,
                    )
                    name_map = {}

                # Step 3: build ExplanationInput list
                inputs = []
                n_skipped = 0
                for dec in decision_rows:
                    pid = dec.bdl_player_id
                    score_row = score_map.get(pid)
                    if score_row is None:
                        # Cannot explain without Z-scores
                        n_skipped += 1
                        continue

                    momentum_row = momentum_map.get(pid)
                    sim_row = sim_map.get(pid)
                    bt_row = backtest_map.get(pid)

                    player_name = name_map.get(pid, "Player {}".format(pid))
                    drop_name = None
                    if dec.drop_player_id is not None:
                        drop_name = name_map.get(dec.drop_player_id, "Player {}".format(dec.drop_player_id))

                    inputs.append(ExplanationInput(
                        decision_id=dec.id,
                        as_of_date=as_of_date,
                        decision_type=dec.decision_type,
                        bdl_player_id=pid,
                        player_name=player_name,
                        target_slot=dec.target_slot,
                        drop_player_id=dec.drop_player_id,
                        drop_player_name=drop_name,
                        lineup_score=dec.lineup_score,
                        value_gain=dec.value_gain,
                        decision_confidence=dec.confidence if dec.confidence is not None else 0.0,
                        player_type=score_row.player_type,
                        score_0_100=score_row.score_0_100 if score_row.score_0_100 is not None else 0.0,
                        composite_z=score_row.composite_z if score_row.composite_z is not None else 0.0,
                        z_hr=score_row.z_hr,
                        z_rbi=score_row.z_rbi,
                        z_sb=score_row.z_sb,
                        z_avg=score_row.z_avg,
                        z_obp=score_row.z_obp,
                        z_era=score_row.z_era,
                        z_whip=score_row.z_whip,
                        z_k_per_9=score_row.z_k_per_9,
                        score_confidence=score_row.confidence if score_row.confidence is not None else 0.0,
                        games_in_window=score_row.games_in_window if score_row.games_in_window is not None else 0,
                        signal=momentum_row.signal if momentum_row else "STABLE",
                        delta_z=momentum_row.delta_z if momentum_row and momentum_row.delta_z is not None else 0.0,
                        proj_hr_p50=sim_row.proj_hr_p50 if sim_row else None,
                        proj_rbi_p50=sim_row.proj_rbi_p50 if sim_row else None,
                        proj_sb_p50=sim_row.proj_sb_p50 if sim_row else None,
                        proj_avg_p50=sim_row.proj_avg_p50 if sim_row else None,
                        proj_k_p50=sim_row.proj_k_p50 if sim_row else None,
                        proj_era_p50=sim_row.proj_era_p50 if sim_row else None,
                        proj_whip_p50=sim_row.proj_whip_p50 if sim_row else None,
                        prob_above_median=sim_row.prob_above_median if sim_row else None,
                        downside_p25=sim_row.downside_p25 if sim_row else None,
                        upside_p75=sim_row.upside_p75 if sim_row else None,
                        backtest_composite_mae=bt_row.composite_mae if bt_row else None,
                        backtest_games=bt_row.games_played if bt_row else None,
                    ))

                # Step 4: generate explanations (CPU-bound -- offload to thread pool)
                explanation_results = await asyncio.to_thread(explain_batch, inputs)

                if not explanation_results:
                    logger.warning(
                        "explainability: 0 explanations generated for as_of_date=%s "
                        "(inputs=%d, skipped=%d)",
                        as_of_date, len(inputs), n_skipped,
                    )

                # Step 5: upsert DecisionExplanationORM rows
                try:
                    for res in explanation_results:
                        factors_data = [
                            {
                                "name": f.name,
                                "value": f.value,
                                "label": f.label,
                                "weight": f.weight,
                                "narrative": f.narrative,
                            }
                            for f in res.factors
                        ]
                        stmt = pg_insert(DecisionExplanationORM.__table__).values(
                            decision_id=res.decision_id,
                            bdl_player_id=res.bdl_player_id,
                            as_of_date=res.as_of_date,
                            decision_type=res.decision_type,
                            summary=res.summary,
                            factors_json=factors_data,
                            confidence_narrative=res.confidence_narrative,
                            risk_narrative=res.risk_narrative,
                            track_record_narrative=res.track_record_narrative,
                            computed_at=now,
                        ).on_conflict_do_update(
                            constraint="_de_decision_id_uc",
                            set_=dict(
                                bdl_player_id=res.bdl_player_id,
                                as_of_date=res.as_of_date,
                                decision_type=res.decision_type,
                                summary=res.summary,
                                factors_json=factors_data,
                                confidence_narrative=res.confidence_narrative,
                                risk_narrative=res.risk_narrative,
                                track_record_narrative=res.track_record_narrative,
                                computed_at=now,
                            ),
                        )
                        db.execute(stmt)

                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "explainability: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("explainability", "failed")
                    return {"status": "failed", "n_explained": 0, "n_skipped": n_skipped, "elapsed_ms": elapsed}

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            n_explained = len(explanation_results)
            self._record_job_run("explainability", "success", n_explained)
            logger.info(
                "explainability: %d decisions explained for as_of_date=%s "
                "skipped=%d elapsed_ms=%d",
                n_explained, as_of_date, n_skipped, elapsed,
            )
            return {
                "status": "success",
                "as_of_date": str(as_of_date),
                "n_explained": n_explained,
                "n_skipped": n_skipped,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["explainability"], _run)

    async def _run_snapshot(self) -> dict:
        """
        Daily Snapshot Engine (lock 100_025, 10 AM ET).

        Runs after _run_explainability (9 AM) -- final stage of the daily pipeline.

        Algorithm:
          1. Compute as_of_date = yesterday
          2. Query count metrics from all 6 phase tables for that date
          3. Compute regression detection vs. historical average composite_mae
          4. Fetch top 5 lineup + top 3 waiver player IDs from decision_results
          5. Build SnapshotInput; call build_snapshot() via asyncio.to_thread
          6. Upsert DailySnapshotORM ON CONFLICT _ds_date_uc DO UPDATE all columns
          7. Return summary dict with health, counts, elapsed_ms

        ADR-004: Never import betting_model or analysis here.
        """
        t0 = time.monotonic()

        async def _run():
            as_of_date = datetime.now(ZoneInfo("America/New_York")).date() - timedelta(days=1)
            now = datetime.now(ZoneInfo("America/New_York"))

            db = SessionLocal()
            try:
                # Step 2: query count metrics from all phase tables
                n_players_scored = (
                    db.query(func.count(PlayerScore.id))
                    .filter(
                        PlayerScore.as_of_date == as_of_date,
                        PlayerScore.window_days == 14,
                    )
                    .scalar() or 0
                )

                n_momentum_records = (
                    db.query(func.count(PlayerMomentum.id))
                    .filter(PlayerMomentum.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_simulation_records = (
                    db.query(func.count(SimulationResultORM.id))
                    .filter(SimulationResultORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_decisions = (
                    db.query(func.count(DecisionResultORM.id))
                    .filter(DecisionResultORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_explanations = (
                    db.query(func.count(DecisionExplanationORM.id))
                    .filter(DecisionExplanationORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                n_backtest_records = (
                    db.query(func.count(BacktestResultORM.id))
                    .filter(BacktestResultORM.as_of_date == as_of_date)
                    .scalar() or 0
                )

                mean_mae = (
                    db.query(func.avg(BacktestResultORM.composite_mae))
                    .filter(BacktestResultORM.as_of_date == as_of_date)
                    .scalar()
                )  # may be None

                # Step 3: regression detection vs. historical baseline
                prev_avg = (
                    db.query(func.avg(BacktestResultORM.composite_mae))
                    .filter(
                        BacktestResultORM.as_of_date < as_of_date,
                        BacktestResultORM.composite_mae.isnot(None),
                    )
                    .scalar()
                )
                regression_detected = (
                    mean_mae is not None
                    and prev_avg is not None
                    and mean_mae > prev_avg * 1.20
                )

                # Step 4: top lineup and waiver player IDs
                top_lineup = [
                    r.bdl_player_id
                    for r in db.query(DecisionResultORM.bdl_player_id)
                    .filter(
                        DecisionResultORM.as_of_date == as_of_date,
                        DecisionResultORM.decision_type == "lineup",
                    )
                    .order_by(DecisionResultORM.lineup_score.desc())
                    .limit(5)
                    .all()
                ]

                top_waiver = [
                    r.bdl_player_id
                    for r in db.query(DecisionResultORM.bdl_player_id)
                    .filter(
                        DecisionResultORM.as_of_date == as_of_date,
                        DecisionResultORM.decision_type == "waiver",
                    )
                    .order_by(DecisionResultORM.value_gain.desc())
                    .limit(3)
                    .all()
                ]

                # Step 5: build SnapshotInput and compute result
                inp = SnapshotInput(
                    as_of_date=as_of_date,
                    n_players_scored=n_players_scored,
                    n_momentum_records=n_momentum_records,
                    n_simulation_records=n_simulation_records,
                    n_decisions=n_decisions,
                    n_explanations=n_explanations,
                    n_backtest_records=n_backtest_records,
                    mean_composite_mae=mean_mae,
                    regression_detected=regression_detected,
                    top_lineup_player_ids=top_lineup,
                    top_waiver_player_ids=top_waiver,
                    pipeline_jobs_run=[
                        "rolling_windows", "player_scores", "player_momentum",
                        "ros_simulation", "decision_optimization",
                        "backtesting", "explainability",
                    ],
                )

                result = await asyncio.to_thread(build_snapshot, inp)

                # Step 6: upsert DailySnapshotORM ON CONFLICT _ds_date_uc DO UPDATE
                try:
                    stmt = pg_insert(DailySnapshotORM.__table__).values(
                        as_of_date=result.as_of_date,
                        n_players_scored=result.n_players_scored,
                        n_momentum_records=result.n_momentum_records,
                        n_simulation_records=result.n_simulation_records,
                        n_decisions=result.n_decisions,
                        n_explanations=result.n_explanations,
                        n_backtest_records=result.n_backtest_records,
                        mean_composite_mae=result.mean_composite_mae,
                        regression_detected=result.regression_detected,
                        top_lineup_player_ids=result.top_lineup_player_ids,
                        top_waiver_player_ids=result.top_waiver_player_ids,
                        pipeline_jobs_run=result.pipeline_jobs_run,
                        pipeline_health=result.pipeline_health,
                        health_reasons=result.health_reasons,
                        summary=result.summary,
                        computed_at=now,
                    ).on_conflict_do_update(
                        constraint="_ds_date_uc",
                        set_=dict(
                            n_players_scored=result.n_players_scored,
                            n_momentum_records=result.n_momentum_records,
                            n_simulation_records=result.n_simulation_records,
                            n_decisions=result.n_decisions,
                            n_explanations=result.n_explanations,
                            n_backtest_records=result.n_backtest_records,
                            mean_composite_mae=result.mean_composite_mae,
                            regression_detected=result.regression_detected,
                            top_lineup_player_ids=result.top_lineup_player_ids,
                            top_waiver_player_ids=result.top_waiver_player_ids,
                            pipeline_jobs_run=result.pipeline_jobs_run,
                            pipeline_health=result.pipeline_health,
                            health_reasons=result.health_reasons,
                            summary=result.summary,
                            computed_at=now,
                        ),
                    )
                    db.execute(stmt)
                    db.commit()
                except Exception as exc:
                    db.rollback()
                    logger.error(
                        "snapshot: DB write failed for as_of_date=%s: %s",
                        as_of_date, exc,
                    )
                    elapsed = int((time.monotonic() - t0) * 1000)
                    self._record_job_run("snapshot", "failed")
                    return {
                        "status": "failed",
                        "as_of_date": str(as_of_date),
                        "pipeline_health": "FAILED",
                        "n_players_scored": n_players_scored,
                        "n_decisions": n_decisions,
                        "elapsed_ms": elapsed,
                    }

            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            self._record_job_run("snapshot", "success", n_players_scored)
            logger.info(
                "snapshot: pipeline_health=%s n_players_scored=%d n_decisions=%d "
                "as_of_date=%s elapsed_ms=%d",
                result.pipeline_health, n_players_scored, n_decisions, as_of_date, elapsed,
            )
            return {
                "as_of_date": str(as_of_date),
                "pipeline_health": result.pipeline_health,
                "n_players_scored": n_players_scored,
                "n_decisions": n_decisions,
                "elapsed_ms": elapsed,
            }

        return await _with_advisory_lock(LOCK_IDS["snapshot"], _run)

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
            from backend.services.yahoo_ingestion import get_validated_adp_feed
            try:
                client = YahooFantasyClient()
            except YahooAuthError as exc:
                logger.error("yahoo_adp_injury: auth error — %s", exc)
                self._record_job_run("yahoo_adp_injury", "failed")
                return {"status": "failed", "records": 0}

            try:
                players = get_validated_adp_feed(client)
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
                    if p.is_injured:
                        injury_flags += 1

                    existing = (
                        db.query(PlayerDailyMetric)
                        .filter(
                            PlayerDailyMetric.player_id == p.player_key,
                            PlayerDailyMetric.metric_date == today,
                            PlayerDailyMetric.sport == "mlb",
                        )
                        .first()
                    )
                    if existing:
                        # Patch injury / ownership fields on existing row
                        existing.rolling_window = {
                            **(existing.rolling_window or {}),
                            "status": p.status,
                            "injury_note": p.injury_note,
                            "percent_owned": p.percent_owned,
                            "adp_updated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                        }
                    else:
                        db.add(PlayerDailyMetric(
                            player_id=p.player_key,
                            player_name=p.name,
                            metric_date=today,
                            sport="mlb",
                            rolling_window={
                                "status": p.status,
                                "injury_note": p.injury_note,
                                "percent_owned": p.percent_owned,
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
            errors = 0
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
                    try:
                        with db.begin_nested():
                            db.execute(stmt)
                        if row["player_id"] in existing_ids:
                            updated += 1
                        else:
                            inserted += 1
                            existing_ids.add(row["player_id"])
                        records_written += 1
                    except Exception as row_exc:
                        errors += 1
                        logger.warning(
                            "ensemble_update: skip row %s -- %s",
                            row.get("player_id"),
                            row_exc,
                        )

                db.commit()
            except Exception as exc:
                db.rollback()
                logger.error("ensemble_update DB write failed: %s", exc)
                elapsed = int((time.monotonic() - t0) * 1000)
                self._record_job_run("ensemble_update", "failed")
                return {
                    "status": "failed",
                    "records": 0,
                    "elapsed_ms": elapsed,
                    "inserted": inserted,
                    "updated": updated,
                    "skipped": skipped,
                    "errors": errors,
                }
            finally:
                db.close()

            elapsed = int((time.monotonic() - t0) * 1000)
            logger.info(
                "ensemble_update: wrote %d player blend rows in %dms (inserted=%d updated=%d skipped=%d errors=%d)",
                records_written,
                elapsed,
                inserted,
                updated,
                skipped,
                errors,
            )
            self._record_job_run("ensemble_update", "success", records_written)
            return {
                "status": "success",
                "records": records_written,
                "elapsed_ms": elapsed,
                "inserted": inserted,
                "updated": updated,
                "skipped": skipped,
                "errors": errors,
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
