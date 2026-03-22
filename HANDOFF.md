# OPERATIONAL HANDOFF — EMAC-077 "DATA SUPERIORITY"

> **Ground truth as of March 23, 2026.** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior state: `EMAC-076` — Fantasy draft complete, Yahoo OAuth live, value-board endpoint deployed.
>
> **GUARDIAN FREEZE still active on CBB model files through April 7.**
> DO NOT touch `backend/betting_model.py`, `backend/services/analysis.py`, or any CBB model service.

---

## 0. CURRENT STATE — WHAT IS TRUE RIGHT NOW

| Subsystem | Status | Notes |
|-----------|--------|-------|
| V9.1 CBB Model | FROZEN until Apr 7 | Guardian active. See EMAC-076 §3 |
| Fantasy Draft | COMPLETE | Juan Soto kept. Draft session endpoints live. |
| Value-Board Endpoint | LIVE | `GET /api/fantasy/draft-session/value-board` w/ Statcast overlay |
| Yahoo OAuth Sync | LIVE | `POST /api/fantasy/draft-session/{key}/sync-yahoo` polls draftresults |
| Time-Series Schema | NOT EXISTS | Tables `player_daily_metrics`, `projection_snapshots` do not exist |
| Ingestion Orchestrator | NOT EXISTS | `backend/services/daily_ingestion.py` does not exist |
| OpenClaw Autonomous Loop | NOT EXISTS | `backend/services/openclaw_autonomous.py` does not exist |
| DiscordRouter | NOT EXISTS | `backend/services/discord_router.py` does not exist |
| WaiverEdgeDetector | NOT EXISTS | `backend/services/waiver_edge_detector.py` does not exist |
| EdgeGenerationEngine | NOT EXISTS | `backend/services/edge_engine.py` does not exist |
| Migration scripts dir | ABSENT | No `backend/migrations/` directory. Precedent: `scripts/migrate_v*.py` |
| Test suite | 647/650 pass | 3 pre-existing DB-auth failures — not our code |

**Existing scheduler (CRITICAL READ BEFORE EPIC-2):**
`main.py` line 96 instantiates `AsyncIOScheduler()` at module level and registers 14 jobs in
`lifespan()`. On Railway with multiple Uvicorn workers this scheduler fires in **every worker
process simultaneously**. The existing jobs are low-risk (read-only polls, idempotent writes) but
any new jobs in the Ingestion Orchestrator MUST be guarded by a PostgreSQL advisory lock.
See ADR-001 below — this is non-negotiable.

---

## 1. ARCHITECTURE DECISION RECORDS (ADRs)

These decisions are final. Agents must not re-open them. If you believe an ADR is wrong, write
a one-paragraph dissent in `reports/` and surface it during the next Architect review session.
Do not deviate from ADRs while implementing.

### ADR-001: Multi-Worker Scheduler Lock via PostgreSQL Advisory Locks

**Problem:** Railway deploys 2+ Uvicorn workers per dyno. `AsyncIOScheduler` starts in every
worker's event loop. Adding Statcast pulls, CLV attribution, and waiver scans to the existing
scheduler would fire each job N-workers times per trigger window.

**Decision:** Every new job registered in `DailyIngestionOrchestrator` MUST acquire a
PostgreSQL advisory lock before executing. Use `pg_try_advisory_lock(bigint)` (non-blocking).
If the lock is already held by another worker, the job logs `SKIPPED — lock held` and returns
immediately. Lock is released automatically when the DB session closes (transaction-level).

**Implementation contract:**
```python
# backend/services/daily_ingestion.py — required wrapper for all jobs
from sqlalchemy import text

LOCK_IDS = {
    'mlb_odds':      100_001,
    'statcast':      100_002,
    'rolling_z':     100_003,
    'cbb_ratings':   100_004,
    'clv':           100_005,
    'cleanup':       100_006,
    'waiver_scan':   100_007,
    'mlb_brief':     100_008,
}

async def _with_advisory_lock(lock_id: int, coro):
    """
    Acquire pg_try_advisory_lock(lock_id). If another worker holds it,
    skip silently. Always returns — never blocks.
    Caller awaits this wrapper, not the coro directly.
    """
    from backend.models import SessionLocal
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT pg_try_advisory_lock(:lid)"), {"lid": lock_id}
        ).scalar()
        if not result:
            logger.info("SKIPPED — advisory lock %d held by another worker", lock_id)
            return None
        return await coro()
    finally:
        db.execute(text("SELECT pg_advisory_unlock(:lid)"), {"lid": lock_id})
        db.close()
```

**Test requirement:** `tests/test_advisory_lock.py` — mock two concurrent calls to the same
job ID, assert only one executes the handler body.

### ADR-002: Migration Convention — `scripts/migrate_v8_post_draft.py`

**Problem:** No `alembic` or `backend/migrations/` directory exists. Prior migrations are in
`scripts/migrate_v*.py` (v3: actual_margin, v5: team_profiles, v6: D1 defaults).

**Decision:** Continue the existing convention. New migration file:
`scripts/migrate_v8_post_draft.py`

Every migration script MUST contain:
1. `def upgrade(db)` — DDL for the up-revision
2. `def downgrade(db)` — DDL for the down-revision (DROP TABLE / ALTER TABLE DROP COLUMN)
3. `if __name__ == "__main__":` block that runs upgrade with DB URL from env
4. A `--dry-run` flag that prints SQL without executing

**Rollback command:** `python scripts/migrate_v8_post_draft.py --downgrade` must be
safe to run at any point and restore the pre-EPIC-1 schema exactly.

### ADR-003: Epic Isolation — No Cross-Epic Work

Epics are strictly sequential. An agent must not begin EPIC-2 until EPIC-1's migration has
been verified on Railway. An agent must not begin EPIC-3 until EPIC-2's scheduler is running
and its `/admin/ingestion/status` endpoint returns healthy status for all jobs.

**Verification gates are defined in each Epic's "Exit Criteria" section below.**

### ADR-004: New Services Are Additive — No Modification to Guardian-Frozen Files

`DailyIngestionOrchestrator`, `EdgeGenerationEngine`, and `OpenClawAutonomousLoop` are NEW files.
They call existing services as imports. They do NOT modify:
- `backend/betting_model.py`
- `backend/services/analysis.py`
- `backend/services/clv.py` (extend only — add `compute_daily_clv_attribution()` to existing module)
- Any existing `scheduler.add_job()` call in `main.py`

Mount the new orchestrator as a **separate startup hook** via a conditional env var
`ENABLE_INGESTION_ORCHESTRATOR=true`. Default: false. This prevents accidental activation
on Railway before the feature is verified.

### ADR-005: No New Discord Channels Until Bot Has Verified Access

Before creating `DiscordRouter` routes to `#fantasy-waivers`, `#fantasy-lineups`, etc., confirm
the bot has access to those channels. The env var names already exist in `discord_notifier.py`
(`DISCORD_CHANNEL_FANTASY_WAIVERS`, `DISCORD_CHANNEL_FANTASY_LINEUPS`). Set them in Railway env
vars first. `DiscordRouter` reads the same env vars — it does NOT hardcode channel IDs.

---

## 2. EPIC-1: TIME-SERIES SCHEMA

**Owner:** Claude Code (Architect)
**Prerequisite:** None
**Status:** NOT STARTED
**Touches:** `scripts/migrate_v8_post_draft.py`, `backend/models.py`
**Does NOT touch:** Any service, any scheduler, any existing table

### 2.1 Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 1.1 | Write `upgrade()` for `player_daily_metrics` | `scripts/migrate_v8_post_draft.py` | [ ] |
| 1.2 | Write `downgrade()` for `player_daily_metrics` | same | [ ] |
| 1.3 | Write `upgrade()` for `projection_snapshots` | same | [ ] |
| 1.4 | Write `downgrade()` for `projection_snapshots` | same | [ ] |
| 1.5 | Add `pricing_engine` column to `predictions` (K-14 spec) | same | [ ] |
| 1.6 | Add SQLAlchemy ORM models for both new tables | `backend/models.py` | [ ] |
| 1.7 | Dry-run test locally | — | [ ] |
| 1.8 | Run migration on Railway | — | [ ] |
| 1.9 | Verify schema via `psql` or Railway DB console | — | [ ] |

### 2.2 Migration Script Specification

File: `scripts/migrate_v8_post_draft.py`

**upgrade() DDL — exact SQL to execute:**

```sql
-- Table 1: player_daily_metrics (sparse time-series)
CREATE TABLE IF NOT EXISTS player_daily_metrics (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    metric_date DATE NOT NULL,
    sport VARCHAR(10) NOT NULL CHECK (sport IN ('mlb', 'cbb')),

    -- Core value metrics
    vorp_7d FLOAT,
    vorp_30d FLOAT,
    z_score_total FLOAT,
    z_score_recent FLOAT,

    -- Statcast 2.0 (MLB only — NULL for CBB rows)
    blast_pct FLOAT,
    bat_speed FLOAT,
    squared_up_pct FLOAT,
    swing_length FLOAT,
    stuff_plus FLOAT,
    plv FLOAT,

    -- Flexible rolling windows (sparse JSONB)
    rolling_window JSONB DEFAULT '{}',

    -- Metadata
    data_source VARCHAR(50),
    fetched_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (player_id, metric_date, sport)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pdm_player_date
    ON player_daily_metrics (player_id, metric_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pdm_sport_date
    ON player_daily_metrics (sport, metric_date DESC)
    WHERE sport = 'mlb';

-- Table 2: projection_snapshots (delta audit trail)
CREATE TABLE IF NOT EXISTS projection_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    sport VARCHAR(10) NOT NULL CHECK (sport IN ('mlb', 'cbb')),
    player_changes JSONB NOT NULL DEFAULT '{}',
    total_players INTEGER,
    significant_changes INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ps_date_sport
    ON projection_snapshots (snapshot_date DESC, sport);

-- K-14: pricing_engine tracking on predictions
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS pricing_engine VARCHAR(20)
    CHECK (pricing_engine IN ('markov', 'gaussian', NULL));
```

**downgrade() DDL — exact SQL to restore prior state:**

```sql
ALTER TABLE predictions DROP COLUMN IF EXISTS pricing_engine;
DROP INDEX CONCURRENTLY IF EXISTS idx_ps_date_sport;
DROP TABLE IF EXISTS projection_snapshots;
DROP INDEX CONCURRENTLY IF EXISTS idx_pdm_sport_date;
DROP INDEX CONCURRENTLY IF EXISTS idx_pdm_player_date;
DROP TABLE IF EXISTS player_daily_metrics;
```

**Script skeleton:**
```python
#!/usr/bin/env python
"""
EMAC-077 EPIC-1 — Post-draft time-series schema migration.
Usage:
    python scripts/migrate_v8_post_draft.py              # run upgrade
    python scripts/migrate_v8_post_draft.py --downgrade  # run downgrade
    python scripts/migrate_v8_post_draft.py --dry-run    # print SQL, no execute
"""
import argparse
import os
import sys
from sqlalchemy import create_engine, text

UPGRADE_SQL = """...paste upgrade DDL here..."""
DOWNGRADE_SQL = """...paste downgrade DDL here..."""

def upgrade(engine, dry_run=False):
    ...

def downgrade(engine, dry_run=False):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downgrade", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    engine = create_engine(os.environ["DATABASE_URL"])
    if args.downgrade:
        downgrade(engine, dry_run=args.dry_run)
    else:
        upgrade(engine, dry_run=args.dry_run)
```

### 2.3 SQLAlchemy ORM Models

Add to `backend/models.py` after the existing model definitions:

```python
class PlayerDailyMetric(Base):
    __tablename__ = "player_daily_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    metric_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)

    vorp_7d = Column(Float)
    vorp_30d = Column(Float)
    z_score_total = Column(Float)
    z_score_recent = Column(Float)

    blast_pct = Column(Float)
    bat_speed = Column(Float)
    squared_up_pct = Column(Float)
    swing_length = Column(Float)
    stuff_plus = Column(Float)
    plv = Column(Float)

    rolling_window = Column(JSONB, default=dict)
    data_source = Column(String(50))
    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "metric_date", "sport"),
    )


class ProjectionSnapshot(Base):
    __tablename__ = "projection_snapshots"

    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)
    player_changes = Column(JSONB, nullable=False, default=dict)
    total_players = Column(Integer)
    significant_changes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 2.4 Required Tests — `tests/test_schema_v8.py`

**Minimum passing tests before EPIC-1 is complete:**

```
test_player_daily_metric_insert_and_unique_constraint
test_player_daily_metric_sport_check_constraint_rejects_invalid
test_projection_snapshot_insert_and_query
test_pricing_engine_column_exists_on_prediction
test_pricing_engine_rejects_invalid_values
test_downgrade_removes_all_new_tables
test_downgrade_removes_pricing_engine_column
```

Coverage target: 100% of new model code. Run with:
```bash
pytest tests/test_schema_v8.py -v
```

### 2.5 Exit Criteria for EPIC-1

All of the following must be TRUE before EPIC-2 starts:

- [ ] `pytest tests/test_schema_v8.py` — all 7 tests pass
- [ ] `python scripts/migrate_v8_post_draft.py --dry-run` — prints SQL, exits 0
- [ ] `python scripts/migrate_v8_post_draft.py` runs on Railway without error
- [ ] `\d player_daily_metrics` on Railway DB shows correct columns + UNIQUE constraint
- [ ] `\d projection_snapshots` on Railway DB shows correct columns
- [ ] `\d predictions` on Railway DB shows `pricing_engine` column
- [ ] `python scripts/migrate_v8_post_draft.py --downgrade` followed by `--upgrade` restores schema cleanly
- [ ] Full test suite still passes: `pytest tests/ -q` — no regressions

---

## 3. EPIC-2: INGESTION ORCHESTRATOR

**Owner:** Claude Code (Architect)
**Prerequisite:** EPIC-1 exit criteria satisfied
**Status:** NOT STARTED
**Touches:** `backend/services/daily_ingestion.py` (new), `backend/main.py` (mount hook only)
**Does NOT touch:** Any existing scheduler job. Any CBB model service.

### 3.1 Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 2.1 | Implement `DailyIngestionOrchestrator` skeleton | `backend/services/daily_ingestion.py` | [ ] |
| 2.2 | Implement `_with_advisory_lock()` wrapper (ADR-001) | same | [ ] |
| 2.3 | Implement `_poll_mlb_odds()` handler | same | [ ] |
| 2.4 | Implement `_update_statcast()` handler | same | [ ] |
| 2.5 | Implement `_calc_rolling_zscores()` — query `player_daily_metrics`, write results | same | [ ] |
| 2.6 | Implement `_compute_clv()` — delegate to `clv.compute_daily_clv_attribution()` | same | [ ] |
| 2.7 | Implement `_cleanup_old_metrics()` — purge rows older than 90 days | same | [ ] |
| 2.8 | Mount orchestrator in `main.py` lifespan under `ENABLE_INGESTION_ORCHESTRATOR=true` | `backend/main.py` | [ ] |
| 2.9 | Add `/admin/ingestion/status` endpoint | `backend/main.py` | [ ] |
| 2.10 | Implement CLV attribution addition to existing `clv.py` | `backend/services/clv.py` | [ ] |
| 2.11 | Write tests | `tests/test_ingestion_orchestrator.py` | [ ] |

### 3.2 API Contract: DailyIngestionOrchestrator

```python
# INPUT: configuration via env vars (no constructor params needed)
# ENABLE_INGESTION_ORCHESTRATOR=true to activate
# NIGHTLY_CRON_TIMEZONE=America/New_York (reuse existing var)

class DailyIngestionOrchestrator:

    def start(self) -> None:
        """
        Register all jobs and start APScheduler.
        Called once in lifespan() startup, ONLY when
        os.getenv('ENABLE_INGESTION_ORCHESTRATOR') == 'true'.
        """

    def get_status(self) -> dict:
        """
        OUTPUT shape (used by /admin/ingestion/status endpoint):
        {
            "job_id": {
                "name": str,
                "enabled": bool,
                "last_run": str | None,      # ISO datetime
                "last_status": str | None,   # "success" | "failed" | "skipped"
                "next_run": str | None       # ISO datetime from APScheduler
            }
        }
        """
```

**Job handler signature contract:**
```python
async def _handler_name(self) -> dict:
    """
    Returns a result dict with at minimum:
    {"status": "success" | "skipped", "records": int, "elapsed_ms": int}
    Raises on unrecoverable error (wrapper catches and alerts Discord).
    """
```

### 3.3 Mount in main.py — Exact Pattern

```python
# In lifespan() after existing scheduler.start() call:

if os.getenv("ENABLE_INGESTION_ORCHESTRATOR", "false").lower() == "true":
    from backend.services.daily_ingestion import DailyIngestionOrchestrator
    _ingestion_orchestrator = DailyIngestionOrchestrator()
    _ingestion_orchestrator.start()
    logger.info("DailyIngestionOrchestrator started")
else:
    _ingestion_orchestrator = None
    logger.info("DailyIngestionOrchestrator disabled (ENABLE_INGESTION_ORCHESTRATOR not set)")
```

```python
# New admin endpoint (add after existing /admin/scheduler/status):
@app.get("/admin/ingestion/status")
async def ingestion_status(user: str = Depends(verify_api_key)):
    if _ingestion_orchestrator is None:
        return {"enabled": False, "jobs": {}}
    return {"enabled": True, "jobs": _ingestion_orchestrator.get_status()}
```

### 3.4 CLV Attribution Extension

**File:** `backend/services/clv.py` — add this function (do NOT modify existing functions):

```python
async def compute_daily_clv_attribution() -> dict:
    """
    Automated CLV calculation comparing our projected spread vs closing line.
    Runs nightly at 11 PM ET via DailyIngestionOrchestrator.

    INPUT: None (reads from DB)
    OUTPUT: {
        "date": str,
        "games_processed": int,
        "clv_positive": int,       # games where we beat the closing line
        "clv_negative": int,
        "avg_clv_points": float,   # mean(|our_spread - closing_spread|)
        "favorable_rate": float,   # fraction of games where our side beat close
        "negative_streak_days": int | None,  # if streak detected
        "records": List[dict]      # per-game detail
    }
    Raises: CLVAttributionError on unrecoverable DB failure
    """
```

**Key implementation notes:**
- Query `Prediction` joined to `ClosingLine` via `game_id`
- A "favorable" CLV means our projected side moved in our favor from open to close
- Alert `discord_notifier.send_system_error()` if `negative_streak_days >= 7`
- Store per-game records in `ProjectionSnapshot` table (JSONB `player_changes` field)
  with `sport='cbb'` and `snapshot_date` = yesterday

### 3.5 Rolling Z-Score Calculation Specification

`_calc_rolling_zscores()` in `DailyIngestionOrchestrator`:

1. Query `player_daily_metrics` WHERE `sport='mlb'` AND `metric_date >= today - 30 days`
2. Group by `player_id`
3. For each player with >= 7 rows: compute `z_score_recent` from 7-day window
4. For each player with >= 30 rows: compute `z_score_total` from 30-day window
5. Upsert back to `player_daily_metrics` for today's date
6. Write summary to `ProjectionSnapshot` with `significant_changes` = count of players
   where `|new_z - old_z| > 0.5`

### 3.6 Required Tests — `tests/test_ingestion_orchestrator.py`

```
test_advisory_lock_prevents_double_execution
test_advisory_lock_releases_on_exception
test_orchestrator_get_status_returns_all_jobs
test_orchestrator_skips_disabled_jobs
test_rolling_zscore_calc_with_7_day_window
test_rolling_zscore_calc_skips_players_with_insufficient_data
test_cleanup_old_metrics_deletes_rows_before_90_days
test_cleanup_old_metrics_preserves_recent_rows
test_clv_attribution_returns_correct_shape
test_clv_attribution_detects_negative_streak
test_ingestion_status_endpoint_returns_enabled_false_when_not_started
```

Coverage target: >= 85% of `daily_ingestion.py` lines.

### 3.7 Exit Criteria for EPIC-2

- [ ] `pytest tests/test_ingestion_orchestrator.py` — all 11 tests pass
- [ ] Full test suite: `pytest tests/ -q` — no regressions vs 647/650 baseline
- [ ] `ENABLE_INGESTION_ORCHESTRATOR=false` (default): Railway logs show "orchestrator disabled" — existing behavior unaffected
- [ ] `ENABLE_INGESTION_ORCHESTRATOR=true` locally: scheduler starts, `/admin/ingestion/status` returns all 6 jobs with `last_status=null`
- [ ] Manual trigger test: call `_poll_mlb_odds()` directly, confirm it returns `{"status": "success", ...}` or graceful skip if MLB odds endpoint returns no data
- [ ] Advisory lock test: spin two asyncio tasks calling the same handler, confirm logs show exactly one "SKIPPED" entry
- [ ] Set `ENABLE_INGESTION_ORCHESTRATOR=true` in Railway, verify no duplicate job fires in `railway logs --follow` over a 10-minute window

---

## 4. EPIC-3: OPENCLAW AUTONOMOUS LOOP

**Owner:** Claude Code (Architect) + OpenClaw (execution target)
**Prerequisite:** EPIC-2 exit criteria satisfied
**Status:** NOT STARTED
**Touches:** (new files only, plus additive changes to 2 existing files)

### 4.1 Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 3.1 | Implement `WaiverEdgeDetector` | `backend/services/waiver_edge_detector.py` | [ ] |
| 3.2 | Implement `DiscordRouter` with rate limiting | `backend/services/discord_router.py` | [ ] |
| 3.3 | Add `send_batch_digest()` to existing discord_notifier | `backend/services/discord_notifier.py` | [ ] |
| 3.4 | Implement `OpenClawAutonomousLoop` | `backend/services/openclaw_autonomous.py` | [ ] |
| 3.5 | Wire MLB DFS section into `MorningBriefGenerator` | `backend/services/openclaw_briefs.py` | [ ] |
| 3.6 | Add `/api/fantasy/waiver` endpoint that returns `WaiverEdgeDetector` results | `backend/main.py` | [ ] |
| 3.7 | Write tests | `tests/test_openclaw_autonomous.py`, `tests/test_waiver_edge.py` | [ ] |
| 3.8 | Update HEARTBEAT.md with new autonomous loops | `HEARTBEAT.md` | [ ] |

### 4.2 API Contract: WaiverEdgeDetector

```python
# INPUT: None (reads from Yahoo API + player board)
# IMPORTANT: Yahoo API is rate-limited. Cap `get_free_agents()` calls at 1/30s.

class WaiverEdgeDetector:

    def detect_waiver_edges(self) -> list[dict]:
        """
        OUTPUT: list of waiver candidates, sorted descending by priority.
        Each dict shape:
        {
            "player": str,             # full name
            "player_id": str,          # normalized ID (matches player_board id key)
            "positions": list[str],
            "team": str,
            "z_score": float,
            "percent_rostered": float, # 0-100
            "tier": str,               # "must_add" | "strong_add" | "streamer"
            "priority": float,         # higher = more urgent
            "reason": str,
            "projected_war": float | None
        }
        Returns [] (empty list) — never raises — if Yahoo API is unavailable.
        Logs warning on Yahoo failure, does not propagate exception.
        """
```

**Graceful degradation requirement:** If `YahooFantasyClient()` raises `YahooAuthError`,
log the error and return `[]`. The autonomous loop must not crash because Yahoo is down.

### 4.3 API Contract: DiscordRouter

```python
# INPUT: IntelPackage dataclass (defined in openclaw_autonomous.py)
# OUTPUT: bool (True = delivered, False = rate-limited or failed)

@dataclass
class IntelPackage:
    channel: str        # key into CHANNELS dict (not a raw channel ID)
    embed: dict         # Discord embed payload (matches existing discord_notifier format)
    priority: int       # 1-5 (5 = critical, bypasses rate limit)
    timestamp: datetime
    mention_admin: bool = False

class DiscordRouter:
    RATE_LIMITS: dict[str, int]  # channel_key -> max_per_hour

    async def route(self, intel: IntelPackage) -> bool:
        """
        Attempt delivery with rate limit check.
        Priority >= 4: bypass rate limit, deliver immediately.
        Priority < 4 + rate limited: enqueue for batch digest.
        Returns True only if message was delivered in this call.
        Delegates actual HTTP call to discord_notifier.send_to_channel().
        """

    async def flush_batch(self, channel: str) -> bool:
        """
        Combine all queued messages for channel into a single embed and deliver.
        Called hourly by OpenClawAutonomousLoop for non-critical channels.
        """
```

**Critical constraint:** `DiscordRouter` calls `discord_notifier.send_to_channel()` — it does
NOT make its own HTTP calls to Discord. All actual Discord HTTP logic stays in
`discord_notifier.py`. `DiscordRouter` is a routing/rate-limiting layer only.

### 4.4 API Contract: OpenClawAutonomousLoop

```python
class OpenClawAutonomousLoop:
    """
    Registered as a single APScheduler job in DailyIngestionOrchestrator
    — NOT as a separate infinite loop. This prevents Railway process conflicts.
    """

    # Called by orchestrator at 7:00 AM ET
    async def run_morning_workflow(self) -> dict:
        """
        OUTPUT: {"brief_sent": bool, "waiver_sent": bool, "waiver_count": int}
        """

    # Called by orchestrator at 10:00 AM ET
    async def run_lineup_workflow(self) -> dict:
        """
        OUTPUT: {"lineup_sent": bool, "player_count": int}
        """

    # Called by orchestrator every 2h between 12 PM - 11 PM ET
    async def run_live_monitor(self) -> dict:
        """
        OUTPUT: {"escalations_sent": int, "games_checked": int}
        """

    # Called by orchestrator hourly
    async def run_telemetry_update(self) -> dict:
        """
        OUTPUT: {"telemetry_sent": bool, "token_budget_pct": float | None}
        """
```

**Scheduling in orchestrator** (add to `_register_default_jobs()` in EPIC-2):
```python
self.register_job(
    id='morning_workflow', name='Morning Brief + Waiver Scan',
    trigger=CronTrigger(hour=7, minute=0, timezone='America/New_York'),
    handler=lambda: self.openclaw_loop.run_morning_workflow()
)
self.register_job(
    id='lineup_workflow', name='Daily Lineup Optimization',
    trigger=CronTrigger(hour=10, minute=0, timezone='America/New_York'),
    handler=lambda: self.openclaw_loop.run_lineup_workflow()
)
self.register_job(
    id='live_monitor', name='Live Game Monitor',
    trigger=CronTrigger(hour='12-23', minute=0, timezone='America/New_York'),
    handler=lambda: self.openclaw_loop.run_live_monitor()
)
self.register_job(
    id='telemetry', name='OpenClaw Health Telemetry',
    trigger=CronTrigger(minute=0),
    handler=lambda: self.openclaw_loop.run_telemetry_update()
)
```

### 4.5 Morning Brief MLB Add-On Specification

**File:** `backend/services/openclaw_briefs.py`
**Function to add:** `collect_mlb_dfs_section(date_str: str) -> dict`

```python
# OUTPUT shape:
{
    "top_batters": [  # top 5 from DailyLineupOptimizer, sorted by score
        {"name": str, "team": str, "implied_runs": float, "park_factor": float, "score": float}
    ],
    "top_pitchers": [  # top 3 SPs with best park factor + low ERA
        {"name": str, "team": str, "opponent": str, "era": float, "park_factor": float}
    ],
    "slate_size": int,   # number of games today
    "best_park": str,    # park with highest run factor today
    "avoid_park": str    # park with lowest run factor today
}
# Returns {} on any exception — never raises
```

**Embed integration:** Add a "⚾ MLB DFS Outlook" section to the existing
`MorningBriefGenerator.generate_brief()` embed. Check `bool(mlb_addon)` before adding —
if empty dict, skip the section entirely.

### 4.6 Required Tests

**`tests/test_waiver_edge.py`:**
```
test_detect_waiver_edges_returns_list
test_detect_waiver_edges_returns_empty_on_yahoo_auth_error
test_calculate_pickup_edge_must_add_tier
test_calculate_pickup_edge_strong_add_tier
test_calculate_pickup_edge_streamer_tier
test_calculate_pickup_edge_returns_none_below_threshold
test_priority_sort_order_descending
```

**`tests/test_openclaw_autonomous.py`:**
```
test_morning_workflow_returns_correct_shape
test_morning_workflow_completes_if_waiver_detector_returns_empty
test_lineup_workflow_returns_correct_shape
test_live_monitor_returns_correct_shape
test_discord_router_rate_limit_blocks_low_priority
test_discord_router_critical_bypasses_rate_limit
test_discord_router_flush_batch_combines_queued
test_discord_router_delegates_to_discord_notifier_not_raw_http
```

Coverage target: >= 80% of `openclaw_autonomous.py` and `waiver_edge_detector.py`.

### 4.7 Exit Criteria for EPIC-3

- [ ] `pytest tests/test_waiver_edge.py tests/test_openclaw_autonomous.py` — all 15 tests pass
- [ ] Full test suite: `pytest tests/ -q` — total count >= 662 (647 + ~15 new)
- [ ] Set `ENABLE_INGESTION_ORCHESTRATOR=true` in Railway
- [ ] Set required Discord channel env vars in Railway (see §5.2)
- [ ] `railway logs --follow` at 7:00 AM ET — confirm single morning brief delivered to `#openclaw-briefs`
- [ ] `railway logs --follow` — confirm no duplicate job fires (advisory lock working)
- [ ] `/api/fantasy/waiver` endpoint returns non-empty list when real Yahoo data available

---

## 5. DEPLOYMENT CHECKLIST

### 5.1 Railway Environment Variables — Add Before EPIC-3

```bash
ENABLE_INGESTION_ORCHESTRATOR=true         # EPIC-2 activation
DISCORD_CHANNEL_FANTASY_WAIVERS=<id>       # from Discord server settings
DISCORD_CHANNEL_FANTASY_LINEUPS=<id>
DISCORD_CHANNEL_FANTASY_NEWS=<id>
DISCORD_CHANNEL_OPENCLAW_BRIEFS=<id>       # probably already set
DISCORD_CHANNEL_OPENCLAW_ESCALATIONS=<id>  # probably already set
DISCORD_CHANNEL_OPENCLAW_HEALTH=<id>       # probably already set
```

Verify all existing required vars are still set:
```bash
DATABASE_URL, THE_ODDS_API_KEY, KENPOM_API_KEY,
API_KEY_USER1, DISCORD_BOT_TOKEN,
YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REFRESH_TOKEN
```

### 5.2 Railway Deploy Sequence

```
Epic 1:  git push origin main
         railway run python scripts/migrate_v8_post_draft.py
         # verify schema via psql or Railway DB console

Epic 2:  git push origin main
         # set ENABLE_INGESTION_ORCHESTRATOR=true in Railway vars
         # watch railway logs --follow for "DailyIngestionOrchestrator started"
         # call GET /admin/ingestion/status and confirm all jobs listed

Epic 3:  git push origin main
         # set all DISCORD_CHANNEL_* vars
         # verify morning brief at 7 AM ET next day
```

### 5.3 Rollback Procedures

**Schema rollback (if Epic 1 breaks Railway startup):**
```bash
railway run python scripts/migrate_v8_post_draft.py --downgrade
# Confirm Railway restarts cleanly
```

**Epic 2 rollback (if orchestrator causes duplicate jobs):**
```bash
# In Railway env vars:
ENABLE_INGESTION_ORCHESTRATOR=false
# Railway auto-restarts with orchestrator disabled
# No code change or schema rollback required
```

**Epic 3 rollback (if Discord rate limits or loop errors):**
```bash
# In Railway env vars:
ENABLE_INGESTION_ORCHESTRATOR=false
# All Epic 3 code is inside the orchestrator — disabled instantly
```

---

## 6. PERFORMANCE & MONITORING CONTRACTS

### 6.1 Operational Metrics

| Metric | Target | Alert Threshold | Alert Destination |
|--------|--------|-----------------|-------------------|
| Daily ingestion success rate | >99% | <95% | `#openclaw-escalations` |
| Advisory lock skips per 24h | <5 | >20 | Log warning only |
| Morning brief latency | <30s | >60s | `#openclaw-health` |
| Waiver scan latency | <45s | >90s | `#openclaw-health` |
| Discord delivery success | >99% | <95% | `send_system_error()` |
| `player_daily_metrics` row count | <27,400/90d | >50,000 | Log warning (cleanup running?) |

### 6.2 Table Bloat Prevention

`_cleanup_old_metrics()` runs daily at 3:30 AM ET via `DailyIngestionOrchestrator`.
Retention: 90 days for `player_daily_metrics`.
Retention: indefinite for `projection_snapshots` (delta-compressed, small).

Expected steady-state size: ~300 players × 365 days = ~110,000 rows/year.
At ~500 bytes/row = ~55 MB/year. Acceptable without partitioning.

---

## 7. AGENT ROUTING — WHO DOES WHAT

| Task | Agent | Constraint |
|------|-------|-----------|
| All EPIC-1/2/3 implementation | Claude Code | No CBB model files |
| Railway `railway run python scripts/migrate_v8_post_draft.py` | Gemini (ops) | Run command only, no edits |
| Railway env var setup | Gemini (ops) | Verify then set |
| `railway logs --follow` monitoring | Gemini (ops) | Report back in HANDOFF.md |
| Post-implementation audit (whole corpus) | Kimi CLI | Read all new files + models.py, confirm no anti-patterns |
| Waiver report interpretation | OpenClaw | Reads output of `/api/fantasy/waiver` endpoint |
| V9.2 recalibration (Apr 7+) | Claude Code | After EPIC-3 complete. See HANDOFF.md §5.1 (prior version) |

---

## 8. UPDATED HEARTBEAT REGISTRY

Add these loops to `HEARTBEAT.md` after EPIC-2 and EPIC-3 are live:

### New Loop: MLB Odds Poll
- **Trigger:** Every 5 min, 10 AM–11 PM ET (EPIC-2 orchestrator)
- **Job ID:** `mlb_odds`
- **Owner:** DailyIngestionOrchestrator
- **Advisory lock ID:** 100_001

### New Loop: Statcast 2.0 Update
- **Trigger:** Every 6 hours (EPIC-2 orchestrator)
- **Job ID:** `statcast`
- **Advisory lock ID:** 100_002

### New Loop: Rolling Z-Scores
- **Trigger:** Daily 4 AM ET (EPIC-2 orchestrator)
- **Job ID:** `rolling_z`
- **Advisory lock ID:** 100_003

### New Loop: OpenClaw Morning Workflow (MLB + Waiver)
- **Trigger:** 7 AM ET daily (EPIC-3 via orchestrator)
- **Job ID:** `morning_workflow`
- **Output channels:** `#openclaw-briefs`, `#fantasy-waivers`
- **Advisory lock ID:** 100_008

### New Loop: CLV Attribution
- **Trigger:** 11 PM ET daily (EPIC-2 orchestrator)
- **Job ID:** `clv`
- **Advisory lock ID:** 100_005

---

## 9. HIVE WISDOM — LESSONS TO CARRY FORWARD

| Lesson | Source |
|--------|--------|
| `AsyncIOScheduler` fires in EVERY Uvicorn worker — use pg_try_advisory_lock for any new job | ADR-001 |
| `player_daily_metrics` UNIQUE constraint is `(player_id, metric_date, sport)` — always upsert, never insert-or-fail | Schema design |
| `discount_notifier.send_to_channel()` is the only function that should make raw Discord HTTP calls | ADR, discord_router contract |
| `WaiverEdgeDetector` must return `[]` not raise — autonomous loop must be failure-proof | ADR-004 |
| EPIC-2 orchestrator is gated by `ENABLE_INGESTION_ORCHESTRATOR=true` — off by default | ADR-004 |
| Set Discord channel env vars before running EPIC-3 or the router silently no-ops | discord_notifier behavior |
| `migrate_v8_post_draft.py --downgrade` is the atomic rollback — keep it working | ADR-002 |
| CBB model files still frozen until Apr 7 — V9.2 recalibration is a separate mission | GUARDIAN |

---

## 10. PRIOR ART — PRESERVE THESE

These are the items from EMAC-076 that must NOT be lost during EPIC implementation:

- `tasks/cbb_enhancement_plan.md` — V9.2 implementation roadmap
- `reports/K12_RECALIBRATION_SPEC_V92.md` — V9.2 params (apr 7)
- `reports/K13_POSSESSION_SIM_AUDIT.md` — K-14 pricing_engine spec
- `backend/services/haslametrics.py` — 3rd rating source, 12 tests, ready to wire post-Apr 7
- `backend/fantasy_baseball/draft_analytics.py` — value-board engine (written EMAC-077 pre-draft)
- CBB GUARDIAN: do not touch `betting_model.py`, `analysis.py` until Apr 7

---

## 11. IGNITION SWITCH

This is the single command to run on Monday morning to start EPIC-1. Run it from the project root after confirming `pytest tests/ -q` passes clean.

```bash
python scripts/migrate_v8_post_draft.py --dry-run && echo "DRY RUN OK — review SQL above, then run without --dry-run"
```

After reviewing the SQL output and confirming it matches the spec in §2.2, run:

```bash
python scripts/migrate_v8_post_draft.py
```

Then verify:

```bash
pytest tests/test_schema_v8.py -v && pytest tests/ -q
```

If both pass, EPIC-1 is complete. Proceed to EPIC-2.

---

**Document Version:** EMAC-077
**Last Updated:** March 23, 2026
**Status:** PLAN LOCKED — Awaiting EPIC-1 ignition Monday March 24
**Branch:** main
**Team:** Claude Code (Architect) · Kimi CLI (Audit) · OpenClaw (Execution Target) · Gemini (Ops/Railway only)
**Next operator:** Claude Code, Monday March 24 — run ignition command above, implement `scripts/migrate_v8_post_draft.py`
**Apr 7 mission:** V9.2 recalibration — see §10 and prior HANDOFF.md §6
