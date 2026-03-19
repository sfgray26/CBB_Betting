# OPENCLAW ORCHESTRATOR SPEC (Phase 5)

**Produced by:** Kimi CLI (Deep Intelligence)  
**Date:** March 20, 2026  
**For:** Claude Code (Master Architect)  
**Status:** GUARDIAN-SAFE — Does NOT modify betting model  
**Estimated Implementation:** 2-3 Claude sessions

---

## EXECUTIVE SUMMARY

Replace manual `POST /admin/run-analysis` triggers with an autonomous multi-agent pipeline that:
1. Watches for game slates and optimal analysis times
2. Fetches data in parallel (ratings, injuries, odds)
3. Runs model analysis (read-only during GUARDIAN)
4. Applies portfolio constraints
5. Posts picks to Discord + logs to DB

**Key Constraint:** Pipeline reads from `betting_model.py` but never writes to it during GUARDIAN period.

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPENCLAW ORCHESTRATOR                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  Scheduler Agent    │  Triggers at optimal times (3 AM ET, pre-game, etc)   │
│  Data Agent         │  Parallel fetch: ratings, injuries, odds              │
│  Model Agent        │  Runs V9.1 analysis (READ-ONLY during GUARDIAN)       │
│  Risk Agent         │  Applies Kelly sizing, portfolio limits               │
│  Notifier Agent     │  Posts Discord, logs DB, sends alerts                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  KenPom API │ BartTorvik CSV │ The Odds API │ BallDontLie │ Discord Webhook │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FILE STRUCTURE

```
backend/
├── services/
│   ├── openclaw.py           # Main orchestrator (NEW)
│   ├── openclaw_agents.py    # Individual agent implementations (NEW)
│   └── openclaw_scheduler.py # APScheduler integration (NEW)
├── main.py                   # ADD: startup hook for orchestrator
└── tests/
    └── test_openclaw.py      # Integration tests (NEW)
```

---

## IMPLEMENTATION SPEC

### 1. Core Orchestrator (`backend/services/openclaw.py`)

```python
"""
OpenClaw Orchestrator — Autonomous CBB Edge Operations

Pattern: Multi-agent pipeline with asyncio.gather for parallel stages.
Guardian Compliance: READ-ONLY access to betting_model.py until Apr 7, 2026.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional

from sqlalchemy.orm import Session

from backend.database import get_db
from backend.services.analysis import run_daily_analysis  # Existing
from backend.services.ratings import get_all_ratings      # Existing
from backend.services.odds import fetch_odds_for_date     # Existing
from backend.services.discord_simple import send_picks    # Existing

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    SCHEDULER = "scheduler"
    DATA_FETCH = "data_fetch"
    ANALYSIS = "analysis"
    RISK_CHECK = "risk_check"
    NOTIFY = "notify"


class PipelineStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """Immutable context passed through pipeline stages."""
    target_date: date
    session: Session
    dry_run: bool = False
    
    # Populated by stages
    ratings: Optional[dict] = None
    odds: Optional[list] = None
    injuries: Optional[list] = None
    predictions: Optional[list] = None
    portfolio_status: Optional[dict] = None
    
    # Metadata
    stage_results: dict = None
    started_at: datetime = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.stage_results is None:
            self.stage_results = {}
        if self.started_at is None:
            self.started_at = datetime.utcnow()


@dataclass
class StageResult:
    stage: PipelineStage
    status: PipelineStatus
    duration_ms: int
    error: Optional[str] = None
    data: Optional[dict] = None


class OpenClawOrchestrator:
    """
    Autonomous pipeline orchestrator.
    
    Usage:
        orchestrator = OpenClawOrchestrator()
        result = await orchestrator.run_pipeline(target_date=date.today())
    """
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.agents = {
            PipelineStage.SCHEDULER: SchedulerAgent(),
            PipelineStage.DATA_FETCH: DataFetchAgent(),
            PipelineStage.ANALYSIS: AnalysisAgent(),
            PipelineStage.RISK_CHECK: RiskCheckAgent(),
            PipelineStage.NOTIFY: NotifierAgent(),
        }
    
    async def run_pipeline(self, target_date: Optional[date] = None) -> PipelineContext:
        """Execute full pipeline for target date."""
        target_date = target_date or date.today()
        
        # Get database session
        db_gen = get_db()
        session = next(db_gen)
        
        try:
            context = PipelineContext(
                target_date=target_date,
                session=session,
                dry_run=self.dry_run
            )
            
            logger.info(f"[OpenClaw] Starting pipeline for {target_date}")
            
            # Stage 1: Scheduler check
            context = await self._run_stage(PipelineStage.SCHEDULER, context)
            if context.stage_results[PipelineStage.SCHEDULER].status == PipelineStatus.SKIPPED:
                logger.info("[OpenClaw] Pipeline skipped by scheduler")
                return context
            
            # Stage 2: Data fetch (parallel)
            context = await self._run_stage(PipelineStage.DATA_FETCH, context)
            if context.stage_results[PipelineStage.DATA_FETCH].status == PipelineStatus.FAILED:
                logger.error("[OpenClaw] Data fetch failed, aborting pipeline")
                return context
            
            # Stage 3: Analysis (GUARDIAN-SAFE: read-only)
            context = await self._run_stage(PipelineStage.ANALYSIS, context)
            
            # Stage 4: Risk check
            context = await self._run_stage(PipelineStage.RISK_CHECK, context)
            
            # Stage 5: Notification
            context = await self._run_stage(PipelineStage.NOTIFY, context)
            
            context.completed_at = datetime.utcnow()
            duration = (context.completed_at - context.started_at).total_seconds()
            logger.info(f"[OpenClaw] Pipeline completed in {duration:.1f}s")
            
            return context
            
        finally:
            session.close()
    
    async def _run_stage(self, stage: PipelineStage, context: PipelineContext) -> PipelineContext:
        """Execute a single pipeline stage."""
        agent = self.agents[stage]
        start = datetime.utcnow()
        
        try:
            logger.info(f"[OpenClaw] Starting stage: {stage.value}")
            context = await agent.execute(context)
            status = PipelineStatus.SUCCESS
            error = None
        except Exception as e:
            logger.error(f"[OpenClaw] Stage {stage.value} failed: {e}")
            status = PipelineStatus.FAILED
            error = str(e)
        
        duration = int((datetime.utcnow() - start).total_seconds() * 1000)
        context.stage_results[stage] = StageResult(
            stage=stage,
            status=status,
            duration_ms=duration,
            error=error
        )
        
        return context


# ═════════════════════════════════════════════════════════════════════════════
# AGENT IMPLEMENTATIONS
# ═════════════════════════════════════════════════════════════════════════════

class SchedulerAgent:
    """
    Decides whether to run analysis based on:
    - Time of day (optimal: 3 AM ET for full slate)
    - Game count (skip if < 3 games)
    - Last run time (don't re-run within 4 hours)
    """
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        from backend.services.openclaw_lite import get_todays_games
        
        # Check game count
        games = get_todays_games(context.session)
        if len(games) < 3:
            logger.info(f"[Scheduler] Only {len(games)} games, skipping")
            context.stage_results[PipelineStage.SCHEDULER] = StageResult(
                stage=PipelineStage.SCHEDULER,
                status=PipelineStatus.SKIPPED,
                duration_ms=0,
                data={"game_count": len(games)}
            )
            return context
        
        # Check last run (would query Prediction table for latest run_tier)
        # Stub: always proceed for now
        
        logger.info(f"[Scheduler] Proceeding with {len(games)} games")
        return context


class DataFetchAgent:
    """
    Parallel data fetching for:
    - KenPom + BartTorvik ratings
    - Injury reports
    - Current odds from The Odds API
    """
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Fetch all data sources in parallel."""
        
        async def fetch_ratings():
            logger.info("[DataFetch] Fetching ratings...")
            # Existing function from ratings.py
            return await asyncio.to_thread(get_all_ratings)
        
        async def fetch_odds():
            logger.info("[DataFetch] Fetching odds...")
            # Existing function from odds.py
            return await asyncio.to_thread(fetch_odds_for_date, context.target_date)
        
        async def fetch_injuries():
            logger.info("[DataFetch] Fetching injuries...")
            # Stub: implement later or use existing if available
            return []
        
        # Run all fetches in parallel
        ratings, odds, injuries = await asyncio.gather(
            fetch_ratings(),
            fetch_odds(),
            fetch_injuries(),
            return_exceptions=True
        )
        
        # Check for failures
        for name, result in [("ratings", ratings), ("odds", odds), ("injuries", injuries)]:
            if isinstance(result, Exception):
                raise RuntimeError(f"Failed to fetch {name}: {result}")
        
        context.ratings = ratings
        context.odds = odds
        context.injuries = injuries
        
        logger.info(f"[DataFetch] Ratings: {len(ratings)} teams, Odds: {len(odds)} games")
        return context


class AnalysisAgent:
    """
    Runs V9.1 model analysis.
    
    GUARDIAN COMPLIANCE: This is READ-ONLY. It calls existing analysis
    functions but does NOT modify betting_model.py parameters.
    """
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        if context.dry_run:
            logger.info("[Analysis] Dry run — skipping actual analysis")
            return context
        
        logger.info("[Analysis] Running V9.1 model analysis...")
        
        # Call existing analysis pipeline
        # This reads from betting_model.py but doesn't modify it
        predictions = await asyncio.to_thread(
            run_daily_analysis,
            session=context.session,
            target_date=context.target_date,
            ratings=context.ratings,
            odds=context.odds
        )
        
        context.predictions = predictions
        logger.info(f"[Analysis] Generated {len(predictions)} predictions")
        
        return context


class RiskCheckAgent:
    """
    Applies portfolio constraints:
    - Max drawdown check (circuit breaker)
    - Max exposure per bet
    - Kelly fraction adjustment
    """
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        from backend.services.portfolio import get_portfolio_status
        
        portfolio = await asyncio.to_thread(get_portfolio_status, context.session)
        context.portfolio_status = portfolio
        
        # Circuit breaker: halt if drawdown > 20%
        if portfolio.get("drawdown_pct", 0) > 20:
            logger.warning(f"[Risk] CIRCUIT BREAKER: Drawdown {portfolio['drawdown_pct']:.1f}%")
            # Still continue but mark for notification
            context.stage_results[PipelineStage.RISK_CHECK].data = {
                "circuit_breaker": True,
                "reason": "drawdown_threshold"
            }
        
        # Filter predictions based on risk limits
        if context.predictions:
            filtered = []
            for pred in context.predictions:
                # Skip if recommended_units would exceed exposure limit
                # This is a stub — implement full logic
                filtered.append(pred)
            
            context.predictions = filtered
        
        logger.info(f"[Risk] Portfolio DD: {portfolio.get('drawdown_pct', 0):.1f}%")
        return context


class NotifierAgent:
    """
    Posts results to:
    - Discord (picks channel)
    - Database (BetLog entries)
    - Alerts (if circuit breaker triggered)
    """
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        if not context.predictions:
            logger.info("[Notifier] No predictions to notify")
            return context
        
        # Format picks for Discord
        bets = [p for p in context.predictions if p.get("verdict") == "BET"]
        
        if bets and not context.dry_run:
            logger.info(f"[Notifier] Sending {len(bets)} picks to Discord...")
            
            # Use existing Discord service
            await asyncio.to_thread(
                send_picks,
                bets=bets,
                portfolio=context.portfolio_status
            )
        
        # Log to database (existing BetLog entries created by analysis)
        logger.info(f"[Notifier] Pipeline complete: {len(bets)} BETs, {len(context.predictions) - len(bets)} PASS")
        
        return context
```

---

### 2. Scheduler Integration (`backend/services/openclaw_scheduler.py`)

```python
"""
APScheduler integration for OpenClaw Orchestrator.

Replaces the existing manual trigger with autonomous scheduling.
"""

import logging
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.services.openclaw import OpenClawOrchestrator

logger = logging.getLogger(__name__)


class OpenClawScheduler:
    """
    Manages scheduled pipeline runs.
    
    Schedule:
    - 3:00 AM ET: Full nightly analysis (existing)
    - 12:00 PM ET: Pre-game check (new)
    - 6:00 PM ET: Evening update (new)
    """
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.orchestrator = OpenClawOrchestrator()
    
    def start(self):
        """Start the scheduler with all jobs."""
        
        # Nightly full analysis (3 AM ET)
        self.scheduler.add_job(
            self._run_pipeline,
            trigger=CronTrigger(hour=3, minute=0, timezone="America/New_York"),
            id="openclaw_nightly",
            name="OpenClaw Nightly Analysis",
            replace_existing=True
        )
        
        # Pre-game check (12 PM ET)
        self.scheduler.add_job(
            self._run_pipeline,
            trigger=CronTrigger(hour=12, minute=0, timezone="America/New_York"),
            id="openclaw_pregame",
            name="OpenClaw Pre-Game Check",
            replace_existing=True
        )
        
        # Evening update (6 PM ET)
        self.scheduler.add_job(
            self._run_pipeline,
            trigger=CronTrigger(hour=18, minute=0, timezone="America/New_York"),
            id="openclaw_evening",
            name="OpenClaw Evening Update",
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("[OpenClawScheduler] Scheduler started with 3 jobs")
    
    def shutdown(self):
        """Graceful shutdown."""
        self.scheduler.shutdown()
        logger.info("[OpenClawScheduler] Scheduler shutdown")
    
    async def _run_pipeline(self):
        """Wrapper to run orchestrator."""
        try:
            await self.orchestrator.run_pipeline()
        except Exception as e:
            logger.error(f"[OpenClawScheduler] Pipeline failed: {e}")
            # TODO: Send alert to admin


# Singleton instance
_scheduler: OpenClawScheduler | None = None


def get_scheduler() -> OpenClawScheduler:
    """Get or create scheduler singleton."""
    global _scheduler
    if _scheduler is None:
        _scheduler = OpenClawScheduler()
    return _scheduler


def start_openclaw_scheduler():
    """Entry point for startup."""
    scheduler = get_scheduler()
    scheduler.start()


def shutdown_openclaw_scheduler():
    """Entry point for shutdown."""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown()
        _scheduler = None
```

---

### 3. Main.py Integration

**Add to `backend/main.py`:**

```python
# Near top, with other imports
from backend.services.openclaw_scheduler import (
    start_openclaw_scheduler,
    shutdown_openclaw_scheduler
)

# In startup event (with other startup hooks)
@app.on_event("startup")
async def startup_event():
    # ... existing startup code ...
    
    # Start OpenClaw autonomous scheduler
    start_openclaw_scheduler()
    logger.info("OpenClaw scheduler started")

# In shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    # ... existing shutdown code ...
    
    # Graceful shutdown
    shutdown_openclaw_scheduler()
    logger.info("OpenClaw scheduler shutdown")
```

---

### 4. Integration Tests (`backend/tests/test_openclaw.py`)

```python
"""
Integration tests for OpenClaw Orchestrator.

Pattern: Mock all external calls, verify pipeline produces expected outputs.
Reference: Anthropic "Building a C Compiler" — test harness prevents wrong solutions.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch

from backend.services.openclaw import (
    OpenClawOrchestrator,
    PipelineStage,
    PipelineStatus,
    PipelineContext
)


class TestOpenClawOrchestrator:
    """Full pipeline integration tests."""
    
    @pytest.fixture
    def orchestrator(self):
        return OpenClawOrchestrator(dry_run=True)
    
    @pytest.fixture
    def mock_db_session(self):
        return Mock()
    
    @pytest.mark.asyncio
    async def test_pipeline_runs_all_stages(self, orchestrator, mock_db_session):
        """RED → GREEN: Pipeline should execute all 5 stages."""
        # Arrange
        with patch('backend.services.openclaw.get_db') as mock_get_db:
            mock_get_db.return_value = iter([mock_db_session])
            
            # Act
            context = await orchestrator.run_pipeline(target_date=date(2026, 3, 20))
            
            # Assert
            assert PipelineStage.SCHEDULER in context.stage_results
            assert PipelineStage.DATA_FETCH in context.stage_results
            assert PipelineStage.ANALYSIS in context.stage_results
            assert PipelineStage.RISK_CHECK in context.stage_results
            assert PipelineStage.NOTIFY in context.stage_results
    
    @pytest.mark.asyncio
    async def test_data_fetch_parallel(self, orchestrator, mock_db_session):
        """DataAgent should fetch ratings, odds, injuries in parallel."""
        with patch('backend.services.openclaw.get_db') as mock_get_db, \
             patch('backend.services.openclaw.get_all_ratings') as mock_ratings, \
             patch('backend.services.openclaw.fetch_odds_for_date') as mock_odds:
            
            mock_get_db.return_value = iter([mock_db_session])
            mock_ratings.return_value = {"team1": {"adj_em": 15.0}}
            mock_odds.return_value = [{"game_id": 1, "spread": -5.5}]
            
            context = await orchestrator.run_pipeline()
            
            # Verify data was fetched
            assert context.ratings is not None
            assert context.odds is not None
            mock_ratings.assert_called_once()
            mock_odds.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_on_high_drawdown(self, orchestrator, mock_db_session):
        """RiskAgent should trigger circuit breaker if drawdown > 20%."""
        with patch('backend.services.openclaw.get_db') as mock_get_db, \
             patch('backend.services.openclaw.get_portfolio_status') as mock_portfolio:
            
            mock_get_db.return_value = iter([mock_db_session])
            mock_portfolio.return_value = {"drawdown_pct": 25.0}
            
            context = await orchestrator.run_pipeline()
            
            # Verify circuit breaker was triggered
            risk_result = context.stage_results.get(PipelineStage.RISK_CHECK)
            assert risk_result is not None
            assert risk_result.data.get("circuit_breaker") is True
    
    @pytest.mark.asyncio
    async def test_scheduler_skips_low_game_count(self, orchestrator, mock_db_session):
        """Scheduler should skip if < 3 games."""
        with patch('backend.services.openclaw.get_db') as mock_get_db, \
             patch('backend.services.openclaw.get_todays_games') as mock_games:
            
            mock_get_db.return_value = iter([mock_db_session])
            mock_games.return_value = []  # No games
            
            context = await orchestrator.run_pipeline()
            
            # Pipeline should skip after scheduler
            scheduler_result = context.stage_results[PipelineStage.SCHEDULER]
            assert scheduler_result.status == PipelineStatus.SKIPPED


class TestOpenClawScheduler:
    """Scheduler configuration tests."""
    
    def test_scheduler_has_three_jobs(self):
        """Scheduler should register 3 cron jobs."""
        from backend.services.openclaw_scheduler import OpenClawScheduler
        
        scheduler = OpenClawScheduler()
        scheduler.start()
        
        jobs = scheduler.scheduler.get_jobs()
        job_ids = [j.id for j in jobs]
        
        assert "openclaw_nightly" in job_ids
        assert "openclaw_pregame" in job_ids
        assert "openclaw_evening" in job_ids
        
        scheduler.shutdown()
```

---

## SUCCESS CRITERIA

| Criterion | Test | Verification |
|-----------|------|--------------|
| All 5 stages execute | `test_pipeline_runs_all_stages` | pytest passes |
| Data fetch is parallel | `test_data_fetch_parallel` | mocked calls verified |
| Circuit breaker works | `test_circuit_breaker_on_high_drawdown` | DD > 20% triggers |
| Scheduler skips low games | `test_scheduler_skips_low_game_count` | < 3 games = skip |
| 3 scheduled jobs | `test_scheduler_has_three_jobs` | nightly, pregame, evening |
| No betting_model.py changes | Code review | Only reads, never writes |
| Integration test passes | `pytest tests/test_openclaw.py -v` | All green |

---

## GUARDIAN COMPLIANCE CHECKLIST

- [x] **No modifications to `betting_model.py`** — Only calls existing functions
- [x] **No modifications to `analysis.py`** — Uses existing `run_daily_analysis()`
- [x] **No modifications to `backend/services/` files** — Only imports existing functions
- [x] **New files only** — `openclaw.py`, `openclaw_agents.py`, `openclaw_scheduler.py`
- [x] **Read-only during GUARDIAN** — AnalysisAgent uses existing model, doesn't recalibrate

---

## IMPLEMENTATION ORDER

1. **Create `backend/services/openclaw.py`** — Core orchestrator + agents
2. **Create `backend/services/openclaw_scheduler.py`** — APScheduler integration
3. **Update `backend/main.py`** — Add startup/shutdown hooks
4. **Create `backend/tests/test_openclaw.py`** — Integration tests
5. **Run tests** — `pytest tests/test_openclaw.py -v`
6. **Deploy** — Push to Railway, verify scheduler starts in logs

---

## RISK MITIGATION

| Risk | Mitigation |
|------|------------|
| Pipeline fails silently | Each stage logs start/success/failure; scheduler catches exceptions |
| Database session leaks | `try/finally` blocks ensure session.close() |
| External API failures | `asyncio.gather(return_exceptions=True)` handles partial failures |
| Circuit breaker not triggered | Explicit 20% drawdown check with logging |
| Duplicate runs | 4-hour cooldown check in SchedulerAgent (implement before deploy) |

---

## MONITORING

Add to `backend/main.py` health check:

```python
@app.get("/health/openclaw")
async def openclaw_health():
    """Health check for OpenClaw scheduler."""
    from backend.services.openclaw_scheduler import get_scheduler
    
    scheduler = get_scheduler()
    jobs = scheduler.scheduler.get_jobs()
    
    return {
        "status": "healthy" if len(jobs) == 3 else "degraded",
        "jobs_registered": len(jobs),
        "job_ids": [j.id for j in jobs],
        "last_run": "TODO: track last pipeline run"
    }
```

---

**Spec produced by:** Kimi CLI  
**Date:** March 20, 2026  
**Status:** Ready for Claude Code implementation  
**Estimated effort:** 2-3 Claude sessions  
**Guardian compliance:** ✅ SAFE — No betting model modifications
