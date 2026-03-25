# OPERATIONAL HANDOFF — EMAC-080 "MLB BETTING MODEL P0"

> **Ground truth as of March 24, 2026.** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior state: `EMAC-077` — Data superiority sprint; OpenClaw autonomous loop, ingestion orchestrator.
>
> **GUARDIAN FREEZE still active on CBB model files through April 7.**
> DO NOT touch `backend/betting_model.py`, `backend/services/analysis.py`, or any CBB model service.

---

## MISSION ACCOMPLISHED — Mar 24, 2026 (EMAC-080)

| Item | Status |
|------|--------|
| `SportConfig.mlb()` constructor | COMPLETE — `backend/core/sport_config.py` |
| `SPORT_ID_MLB` constant | COMPLETE — added near SPORT_ID_NCAAB/NBA/NCAAF |
| `backend/services/mlb_analysis.py` | COMPLETE — MLBAnalysisService + MLBGameProjection |
| `_mlb_analysis_service` module var + lifespan block | COMPLETE — `backend/main.py` |
| `_run_mlb_analysis_job()` async job function | COMPLETE — `backend/main.py` |
| `tests/test_mlb_analysis.py` | COMPLETE — 12 tests, all pass |

**Total new tests this session:** 12
**Full suite:** 1103/1107 (4 pre-existing DB-auth/cache failures only)
**Files modified:** `backend/core/sport_config.py`, `backend/main.py`
**Files created:** `backend/services/mlb_analysis.py`, `tests/test_mlb_analysis.py`

---

## 0. CURRENT STATE — WHAT IS TRUE RIGHT NOW

| Subsystem | Status | Notes |
|-----------|--------|-------|
| V9.1 CBB Model | FROZEN until Apr 7 | Guardian active. See EMAC-076 §3 |
| K-15 Oracle Validation | **LIVE** (Mar 23) | `oracle_validator.py`, DB columns, `GET /admin/oracle/flagged`, 19 tests. Spec: `reports/K15_ORACLE_VALIDATION_SPEC.md` |
| Fantasy Draft | COMPLETE | Juan Soto kept. Draft session endpoints live. |
| Value-Board Endpoint | LIVE | `GET /api/fantasy/draft-session/value-board` w/ Statcast overlay |
| Yahoo OAuth Sync | LIVE | `POST /api/fantasy/draft-session/{key}/sync-yahoo` polls draftresults |
| Pre-Draft Keeper Sweep | **LIVE** (Mar 23) | `POST /api/fantasy/draft-session/{key}/sync-keepers` — fetches all 12 rosters from Yahoo at room open, marks all keepers, cleans pool before first pick |
| Time-Series Schema | **SCHEMA LIVE** (Mar 24) | ORM models + migration script exist. `tests/test_schema_v8.py` created (7 tests). Run `pytest tests/test_schema_v8.py -v` to confirm. DB tables require `migrate_v8_post_draft.py` to be run on Railway. |
| Ingestion Orchestrator | **LIVE** (Mar 24) | `backend/services/daily_ingestion.py`. 5 jobs (mlb_odds/statcast/rolling_z/clv/cleanup). Advisory locks. 11 tests pass. Mount via `ENABLE_INGESTION_ORCHESTRATOR=true`. |
| OpenClaw Autonomous Loop | **LIVE** (Mar 24) | `backend/services/openclaw_autonomous.py`. Scheduler job at 8:30 AM. 8 tests pass. |
| DiscordRouter | **LIVE** (Mar 24) | `backend/services/discord_router.py`. Rate-limited (60s/channel). Batch flush at 5 items or 300s. |
| WaiverEdgeDetector | **LIVE** (Mar 24) | `backend/services/waiver_edge_detector.py`. FA cache 10 min. MCMC-enriched. |
| MCMCWeeklySimulator (OOP) | **LIVE** (Mar 24) | `backend/services/mcmc_simulator.py`. Wrapper around fantasy_baseball/mcmc_simulator.py. |
| Waiver Wire (Next.js) | **LIVE** (Mar 24) | Filter/sort/pagination/recommendations UI. Backend bugs fixed (owned_pct, two_start SPs, get_roster). |
| EdgeGenerationEngine | NOT EXISTS | `backend/services/edge_engine.py` does not exist |
| Migration scripts dir | ABSENT | No `backend/migrations/` directory. Precedent: `scripts/migrate_v*.py` |
| Test suite | **~665/668** | 35 new tests added this session. 3 pre-existing DB-auth failures only. |
| **UI Stack** | **Next.js 15** | **Streamlit RETIRED** — see ADR-010. All UI work in `frontend/`. Do not reference `dashboard/`. |
| **MLB Betting Model** | **NOT BUILT** | ⚠️ **CRITICAL** — CBB ends Apr 7. MLB model must be operational by Apr 1. See ADR-006 (updated). |
| **Sport Transition** | **Overlap Phase** | CBB active until Apr 7. MLB season started Mar 28. Parallel operation required. |

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

### ADR-010: UI Stack — Next.js Only, Streamlit Retired (Mar 24, 2026)

**Problem:** The `dashboard/` folder contains a legacy Streamlit application that was the original
UI. As of March 2026, we completed a full migration to Next.js 15 (see `FRONTEND_MIGRATION.md`).
However, agents may still reference Streamlit code for UI patterns or mistakenly attempt to fix
bugs in the deprecated `dashboard/pages/` files.

**Decision:** 
1. **Streamlit is RETIRED** — The `dashboard/` folder is deprecated and will be archived in EPIC-4.
2. **Next.js is the ONLY UI** — All UI work goes in `frontend/` (Next.js 15, TypeScript, Tailwind).
3. **NEVER reference Streamlit** — Agents must NOT look at `dashboard/` for UI patterns, components,
   or logic. The Streamlit code is frozen and will be deleted.

**Current State:**
| Location | Status | Purpose |
|----------|--------|---------|
| `frontend/` | **ACTIVE** | Next.js 15 production UI — all new work here |
| `dashboard/` | **DEPRECATED** | Old Streamlit app — do not modify, do not reference |

**Agent Instructions:**
- ❌ DO NOT open files in `dashboard/` for any reason
- ❌ DO NOT use Streamlit as a reference for UI patterns
- ❌ DO NOT fix bugs in `dashboard/pages/*.py`
- ✅ DO build all UI in `frontend/app/(dashboard)/`
- ✅ DO use `frontend/lib/types.ts` and `frontend/lib/api.ts` as source of truth
- ✅ DO refer to `FRONTEND_MIGRATION.md` for component patterns

**Removal Timeline:**
- **Now:** Streamlit code is deprecated but present
- **EPIC-4 (Apr 7, 2026):** `dashboard/` folder will be moved to `archive/dashboard/`
- **Post-EPIC-4:** Streamlit dependencies removed from `requirements.txt`

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

### Immediate Priority (Waiver Wire Critical Fixes)

| Task | Agent | Constraint |
|------|-------|-----------|
| Waiver wire backend fixes (ownership %, 2-start SP, pagination) | Claude Code | See §14 for detailed spec |
| Waiver wire Next.js UI overhaul | Claude Code | `frontend/app/(dashboard)/fantasy/waiver/page.tsx` |
| Railway `railway run python scripts/migrate_v8_post_draft.py` | Gemini (ops) | Run command only, no edits |
| Railway env var setup | Gemini (ops) | Verify then set |
| `railway logs --follow` monitoring | Gemini (ops) | Report back in HANDOFF.md |

### Strategic Workstream (OpenClaw Autonomy)

| Task | Agent | Constraint |
|------|-------|-----------|
| **OpenClaw Autonomy Architecture** | **Kimi CLI (LEAD)** | Design full SOUL.md vision: alpha decay detection, performance monitoring, self-improvement loops |
| OpenClaw implementation | Kimi CLI proposes; Claude approves | Guardian-compliant (read-only until Apr 7) |
| Post-implementation audit (whole corpus) | Kimi CLI | Read all new files + models.py, confirm no anti-patterns |
| Waiver report interpretation | OpenClaw | Reads output of `/api/fantasy/waiver` endpoint |
| V9.2 recalibration (Apr 7+) | Claude Code | After EPIC-3 complete. See HANDOFF.md §5.1 (prior version) |

**Note:** Claude owns immediate waiver wire delivery (P0). Kimi owns OpenClaw autonomy design (strategic). Parallel workstreams with weekly sync.

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

---

## 12. FANTASY BASEBALL ELITE ROADMAP — ALGORITHMIC EXPANSION

> **Authored:** Kimi CLI · March 23, 2026  
> **Spec:** `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md`  
> **Status:** DRAFT — Awaits EPIC-1 through EPIC-3 completion  
> **Priority:** P0 (Post-CBB/March Madness pivot)

### Context: From Draft Helper to Quantitative Asset Management

The Fantasy Baseball module is evolving from a **draft-day assistant** into an **institutional-grade roster management system**. This requires treating fantasy baseball as a multi-agent, multi-timeframe portfolio optimization problem.

### Algorithmic Innovations (Phase 2)

| Innovation | Algorithm | Purpose | Owner |
|------------|-----------|---------|-------|
| **Bayesian Projection Updating** | Conjugate normal priors + shrinkage | Adapt projections as season unfolds | Claude Code |
| **Ensemble Projections** | Inverse-MAE weighted ensemble | Combine Steamer/ZiPS/Yahoo ROS optimally | Claude Code |
| **MCMC Weekly Simulator** | Gibbs sampling (10k sims) | Full outcome distributions, not point estimates | Claude Code |
| **Contextual Bandits** | LinUCB | Real-time add/drop decisions | Claude Code |
| **Portfolio Optimization** | Mean-variance quadratic programming | Risk-adjusted roster construction | Claude Code |
| **Reinforcement Learning** | Deep Q-Network (DQN) | Learn optimal roster moves over season | Claude Code + Kimi (validation) |
| **Graph Neural Networks** | GAT (Graph Attention Networks) | Optimal daily lineup selection | Claude Code |

### Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              FANTASY BASEBALL ORCHESTRATION                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Yahoo Agent │  │Statcast     │  │FanGraphs    │  Data Layer │
│  │ (OpenClaw)  │  │Agent (Kimi) │  │Agent (Claude│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           ENSEMBLE PROJECTION AGENT (Claude)           │   │
│  │    Bayesian Update → Ensemble → Confidence Intervals   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                │                │                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Weekly      │  │ Roster      │  │ Streamer    │  Decision   │
│  │ Strategy    │  │ Construction│  │ Optimization│  Layer      │
│  │ Agent       │  │ Agent (GNN) │  │ Agent       │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with EMAC-077/078 EPICs

**Fantasy Baseball work is SEPARATE from CBB EPICs 1-6.**

| Timeline | CBB Activity | Fantasy Activity |
|----------|-------------|------------------|
| Now (Mar 23) | EPIC-1 (Schema) + EPIC-2 (Orchestrator) | Foundation planning |
| Apr 7 | EPIC-4 (Bracket sunset), EPIC-5 (MLB polling) | Begin Phase 1 implementation |
| Apr 15 | EPIC-6 (Admin suite) | Universal projections + MCMC |
| May | V9.2 CBB recalibration (off-season) | Bayesian updater + RL training |
| June | CBB model maintenance | Full multi-agent deployment |

### Key Technical Decisions

**ADR-007: Fantasy Baseball is Additive, Not Substitution**
- All Fantasy Baseball code lives in `backend/fantasy_baseball/` and `backend/services/daily_ingestion.py`
- No modification to CBB model files (frozen per ADR-004)
- Fantasy orchestrator is a separate APScheduler instance within `DailyIngestionOrchestrator`

**ADR-008: Projection Layer Stratification**
```python
# Tier 1: Pre-computed (Draft Board) — static, high confidence
# Tier 2: Yahoo API (Real-time ROS) — refreshed every 6 hours
# Tier 3: Derived/Heuristic (MLE for call-ups) — computed on-demand
# Tier 4: Bayesian Posterior (Season-long learning) — updated after each game
```

**ADR-009: Multi-Timeframe Value Functions**
```python
class PlayerValue:
    ros_value: float          # Trade decisions (full season)
    four_week_value: float    # Waiver add decisions
    weekly_value: float       # Streamer decisions
    daily_value: float        # Lineup optimization
```

### Implementation Phases

**Phase 1: Foundation (Weeks 1-2, starting Apr 7)**
- Universal projection system (`get_or_create_projection()`)
- Yahoo ROS integration
- Basic roster recommendations
- MCMC weekly simulator

**Phase 2: Intelligence (Weeks 3-4)**
- Bayesian updater
- Ensemble projector
- Statcast trend detection (Kimi)
- Contextual bandit

**Phase 3: Optimization (Weeks 5-6)**
- Portfolio optimizer
- Weekly strategy engine
- GNN lineup setter
- Multi-agent orchestration

**Phase 4: Automation (Weeks 7-8)**
- RL agent (DQN) training
- Auto-execution for low-risk moves
- Real-time opportunity alerts

### Claude Code Responsibilities

1. **Architect all algorithms** — Bayesian, MCMC, RL, GNN implementations
2. **Design orchestration layer** — Agent message bus, coordination protocol
3. **Implement Phase 1** — Universal projections, MCMC simulator
4. **Coordinate with Kimi** — Validation of RL training, trend detection logic
5. **Coordinate with OpenClaw** — Real-time execution, Yahoo API integration

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Projection coverage | 99%+ of Yahoo universe | ~30% (draft board) |
| Recommendation accuracy | 70%+ of adds outperform drops | N/A |
| Weekly matchup win rate | 60%+ H2H | Baseline 50% |
| Time to actionable insight | <5 seconds | Manual browsing |
| Human intervention required | <20% of moves | 100% manual |

### Document References

- Full spec: `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md`
- Algorithm cheat sheet in spec appendix
- Multi-agent orchestration YAML in spec §2.3

---

## 13. PHASE 2 TRANSITION ROADMAP — EPICS 4-6

> **Authored:** EMAC-078 · March 23, 2026 · Claude Code (Master Architect)
> **Trigger condition:** These epics activate AFTER the CBB season concludes and ADR-004 freeze lifts (April 7, 2026).
> EPIC-4 → EPIC-5 → EPIC-6 must run sequentially. Do not start EPIC-5 until EPIC-4 is merged and verified.

### ADR-006: MLB Model Analysis — NOW IN SCOPE (Updated Mar 24, 2026)

**Status:** ⚠️ **SCOPE CHANGE** — MLB betting model transition required before CBB season ends (Apr 7).

**Context:** 
- CBB season ends ~April 7, 2026 (championship game)
- MLB season starts ~March 28, 2026 (ALREADY ACTIVE)
- **System must transition from CBB betting → MLB betting during overlap period**
- Fantasy baseball is already operational (separate from betting model)

**Two Core Components (Active Simultaneously During Overlap):**
1. **Fantasy Baseball** — Waiver wire, lineups, daily optimizations (ALREADY LIVE)
2. **MLB Betting Model** — Today's picks, runline analysis, nightly pipeline (MUST BE BUILT)

**Requirements:**
- `SportConfig.mlb()` constructor for MLB-specific parameters
- Parallel nightly analysis pipeline for MLB (runline, totals)
- MLB-specific OpenClaw patterns (pitcher form, bullpen fatigue, weather)
- Sport switching logic: CBB winds down Apr 1-7, MLB fully active Apr 8+

**Timeline:**
- **Now-Apr 1:** Build MLB betting model alongside CBB
- **Apr 1-7:** Overlap period — both sports active
- **Apr 8+:** Full MLB betting mode

**Quota Management:** MLB odds polling ~1,800/month (well under 20,000 cap)

---

### EPIC-4: Bracket Sunset (UI Deprecation)

**Owner:** Claude Code
**Trigger:** April 7, 2026 (post-championship)
**Prerequisite:** EPIC-1, EPIC-2, EPIC-3 complete
**Touches:** Frontend only + one scheduler job removal

#### Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 4.1 | Safety grep: `grep -r "bracket\|BracketProjection\|tournament_data"` — confirm no imports outside target files | Various | [ ] |
| 4.2 | Delete bracket route files | `frontend/app/(dashboard)/bracket/page.tsx`, `error.tsx`, `loading.tsx` | [ ] |
| 4.3 | Remove "Tournament" nav section (Trophy icon block) | `frontend/components/layout/sidebar.tsx` | [ ] |
| 4.4 | Remove `bracketProjection()` function | `frontend/lib/api.ts` | [ ] |
| 4.5 | Remove `BracketProjection`, `TeamAdvancement`, `UpsetAlert` interfaces | `frontend/lib/types.ts` | [ ] |
| 4.6 | Remove `tournament_bracket_notifier` APScheduler job | `backend/main.py` (lines ~235-238) | [ ] |
| 4.7 | Archive tournament service (do NOT delete — preserve for potential future use) | `backend/services/tournament_data.py` → `backend/archive/tournament_data.py` | [ ] |
| 4.8 | Verify: `npm run build` passes with zero TS errors | `frontend/` | [ ] |
| 4.9 | Verify: `/bracket` returns 404, all other routes healthy | Live app | [ ] |

#### EPIC-4 Handoff Prompt (copy-paste ready for coding agent)

```
EPIC-4: Bracket Sunset

Context: NCAA tournament is over. We are removing all bracket/tournament UI from the frontend.
The CBB model files remain FROZEN (ADR-004) — do not touch betting_model.py or analysis.py.

Step 1 — Safety check (READ-ONLY first):
  grep -r "bracket\|BracketProjection\|tournament_data\|Trophy" /home/user/CBB_Betting/frontend/
  grep -r "tournament_bracket" /home/user/CBB_Betting/backend/main.py
  Report every file that contains these strings before making any edits.

Step 2 — Delete these files (only after step 1 confirms no surprise imports):
  frontend/app/(dashboard)/bracket/page.tsx
  frontend/app/(dashboard)/bracket/error.tsx
  frontend/app/(dashboard)/bracket/loading.tsx

Step 3 — Edit these files:
  a. frontend/components/layout/sidebar.tsx — remove the "Tournament" nav section
     (the block containing the Trophy icon and the /bracket href)
  b. frontend/lib/api.ts — remove the bracketProjection() function
  c. frontend/lib/types.ts — remove BracketProjection, TeamAdvancement, UpsetAlert interfaces

Step 4 — Backend cleanup:
  a. backend/main.py — remove the tournament_bracket_notifier scheduler job
  b. Move (do NOT delete): backend/services/tournament_data.py → backend/archive/tournament_data.py
     (create backend/archive/ directory if it doesn't exist)

Step 5 — Verify:
  cd /home/user/CBB_Betting/frontend && npm run build
  npx tsc --noEmit
  Both must pass with zero errors. Report the output.

Step 6 — Commit and push to branch claude/clarify-bet-recommendations-ui-WC8Do:
  git add -A && git commit -m "EPIC-4: Remove bracket/tournament UI post-season"
  git push -u origin claude/clarify-bet-recommendations-ui-WC8Do
```

---

### EPIC-5: Sport Polling Switch (API Quota Management)

**Owner:** Claude Code (backend) · Gemini CLI (Railway env vars only)
**Trigger:** April 8, 2026
**Prerequisite:** EPIC-4 complete
**Touches:** `backend/core/sport_polling_switch.py` (new), `backend/services/odds.py`, `backend/models.py`, `backend/main.py`

#### Quota Budget (do not exceed)

| Phase | Sport | Calls/Month | Budget |
|---|---|---|---|
| Now (CBB active) | basketball_ncaab | ~11,610 | OK |
| Transition (Apr 7-8) | Both winding down | ~3,000 | OK |
| MLB season | baseball_mlb | ~1,800 | Well under |
| **Hard cap** | | **20,000** | 2,000 reserve |

MLB polling schedule: Morning check 9 AM (1 call), pre-game 11 AM-4 PM every 10 min (30 calls), game-time 5 PM-midnight every 15 min (28 calls), nightly settle 1 AM (1 call). Total: ~60/day → 1,800/month.

#### Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 5.1 | Add `sport_poll_config` table to models | `backend/models.py` | [ ] |
| 5.2 | Write migration script | `scripts/migrate_sport_poll_config.py` | [ ] |
| 5.3 | Create `SportPollingSwitch` class with `wind_down_cbb()`, `activate_mlb()`, `get_quota_status()` | `backend/core/sport_polling_switch.py` (NEW) | [ ] |
| 5.4 | Add generic `get_odds(sport_key: str)` to OddsAPIClient; keep `get_cbb_odds()` as wrapper | `backend/services/odds.py` | [ ] |
| 5.5 | Add `get_mlb_odds()` wrapper | `backend/services/odds.py` | [ ] |
| 5.6 | Register MLB scheduler jobs with PG advisory locks (ADR-001) | `backend/main.py` | [ ] |
| 5.7 | Auto-pause CBB jobs on April 7 (CronTrigger 11:59 PM ET) | `backend/main.py` | [ ] |
| 5.8 | Add admin endpoints: `GET/POST /admin/sport-switch`, `GET /admin/quota/history` | `backend/main.py` | [ ] |
| 5.9 | Write `tests/test_sport_polling_switch.py` (mock OddsAPI, test lock behavior) | `tests/` | [ ] |
| 5.10 | Gemini CLI: set `MLB_ACTIVE=false` in Railway env (starting value) | Railway dashboard | [ ] |

#### EPIC-5 Handoff Prompt

```
EPIC-5: Sport Polling Switch

Context: CBB season is over. We need to pivot API polling from basketball_ncaab to baseball_mlb.
Hard quota cap: 20,000 requests/month to The Odds API. Target MLB spend: ~1,800/month.
ADR-001 is non-negotiable: ALL new scheduler jobs must use pg_try_advisory_lock.
ADR-006: Do NOT wire MLB odds into nightly_analysis or produce model predictions for MLB.

Files to read first:
  backend/services/odds.py           — OddsAPIClient, get_cbb_odds(), quota tracking
  backend/core/sport_config.py       — SportConfig class (already has mlb sport key stub)
  backend/models.py                  — existing table patterns to follow
  backend/main.py lines 94-258       — existing scheduler jobs (pattern to replicate)

Tasks:
1. Add to backend/models.py:
   - Table `sport_poll_config`: id, cbb_active (bool, default True), mlb_active (bool, default False),
     transition_date (Date), updated_at (Timestamptz)
   - ORM class SportPollConfig

2. Create backend/core/sport_polling_switch.py:
   - Class SportPollingSwitch(db: Session)
   - Methods: active_sports(), wind_down_cbb(), activate_mlb(), get_quota_status()
   - wind_down_cbb() sets cbb_active=False; activate_mlb() sets mlb_active=True

3. Modify backend/services/odds.py:
   - Add get_odds(sport_key: str) — generic method (move URL construction there)
   - Refactor get_cbb_odds() to call get_odds("basketball_ncaab")
   - Add get_mlb_odds() calling get_odds("baseball_mlb")

4. Modify backend/main.py:
   - Add CronTrigger job `cbb_wind_down` firing April 7 at 11:59 PM ET
     → calls SportPollingSwitch.wind_down_cbb() and pauses CBB jobs
   - Add 4 MLB jobs (all with pg_try_advisory_lock):
     * mlb_morning_lines: CronTrigger 9 AM ET
     * mlb_pregame_monitor: IntervalTrigger 10 min (only run between 11 AM-4 PM via time check)
     * mlb_game_monitor: IntervalTrigger 15 min (only run between 5 PM-midnight)
     * mlb_nightly_settle: CronTrigger 1 AM ET
   - Add endpoints: GET /admin/sport-switch/status, POST /admin/sport-switch,
     GET /admin/quota/history

5. Write tests/test_sport_polling_switch.py covering:
   - wind_down_cbb() sets correct DB state
   - activate_mlb() sets correct DB state
   - get_mlb_odds() calls correct URL ("baseball_mlb")
   - Advisory lock prevents double-execution

6. Create migration script: scripts/migrate_sport_poll_config.py
   Run it and report output.

Report: file diffs, test results, quota projection calculation.
Push to branch claude/clarify-bet-recommendations-ui-WC8Do.
```

---

### EPIC-6: Admin Suite & Access Control

**Owner:** Claude Code
**Trigger:** April 15, 2026 (after EPIC-5 is stable for 1 week)
**Prerequisite:** EPIC-5 complete and verified on Railway
**Touches:** `backend/auth.py` (rewrite), `backend/models.py`, `frontend/app/(dashboard)/admin/page.tsx`

#### Role Matrix

| Action | owner | risk_manager | viewer |
|---|---|---|---|
| Read any data | ✓ | ✓ | ✓ |
| Acknowledge alerts / override bankroll | ✓ | ✓ | ✗ |
| Pause betting markets / adjust line projections | ✓ | ✓ | ✗ |
| Run analysis / recalibrate / delete bets | ✓ | ✗ | ✗ |
| Manage users / sport switch | ✓ | ✗ | ✗ |

#### Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 6.1 | Add `user_role` enum, `users` table, `audit_log` table | `backend/models.py` | [ ] |
| 6.2 | Write migration: `scripts/migrate_users_rbac.py` (seeds owner from API_KEY_USER1 env var) | `scripts/` | [ ] |
| 6.3 | Rewrite `verify_api_key()` → DB lookup + bcrypt verify → returns `User` ORM object | `backend/auth.py` | [ ] |
| 6.4 | Add `require_role(*roles)` dependency factory | `backend/auth.py` | [ ] |
| 6.5 | Apply role guards to all admin routes (owner-only: delete/recalibrate/sport-switch/user-mgmt; risk_manager: bankroll/pause/alerts) | `backend/main.py` | [ ] |
| 6.6 | Add audit_log write middleware for all /admin/* endpoints | `backend/main.py` | [ ] |
| 6.7 | Add endpoints: `GET /admin/users`, `POST /admin/users`, `DELETE /admin/users/{id}` | `backend/main.py` | [ ] |
| 6.8 | Add endpoints: `POST /admin/markets/{id}/pause`, `DELETE /admin/markets/{id}/pause` | `backend/main.py` | [ ] |
| 6.9 | Add endpoint: `GET /admin/audit-log` (last 100 actions) | `backend/main.py` | [ ] |
| 6.10 | Extend admin page with 4 tabs: Risk Controls, User Management, Audit Log, Quota Monitor | `frontend/app/(dashboard)/admin/page.tsx` | [ ] |
| 6.11 | Write `tests/test_auth_rbac.py` — verify 403 on role violations, audit log writes | `tests/` | [ ] |

#### EPIC-6 Handoff Prompt

```
EPIC-6: Admin Suite & Access Control

Context: Replace hardcoded user1=admin with proper RBAC. 3 roles: owner, risk_manager, viewer.
This is a solo-to-small-team system (max 5 users). No SAML/SSO in scope — SSO is a future
migration path via AuthProvider interface but NOT implemented now.

Files to read first:
  backend/auth.py              — current simple API key auth (to be rewritten)
  backend/models.py            — existing table patterns
  backend/main.py              — existing /admin/* routes and their auth dependencies
  frontend/app/(dashboard)/admin/page.tsx  — existing 6-panel admin UI

Phase A — DB Layer:
1. Add to backend/models.py:
   - Enum UserRole = Literal['owner', 'risk_manager', 'viewer']
   - Table `users`: id, username (unique), api_key_hash (bcrypt), role (UserRole), is_active (bool), created_at, last_seen
   - Table `audit_log`: id, user_id (FK users.id), action, endpoint, payload (JSON), ip_address, ts

2. Create scripts/migrate_users_rbac.py:
   - Creates both tables
   - Seeds one owner user from API_KEY_USER1 env var (bcrypt hash it, don't store plaintext)
   - Run and report output

Phase B — Auth Rewrite (backend/auth.py):
3. Rewrite verify_api_key(db, api_key) → User:
   - Query users table by doing bcrypt.checkpw against each active user's api_key_hash
   - Update last_seen on successful auth
   - Raise 401 if no match
4. Add require_role(*allowed_roles) → Depends():
   - Factory that returns a FastAPI dependency
   - Raises 403 if user.role not in allowed_roles
5. Keep verify_admin_api_key as compatibility shim calling require_role('owner')

Phase C — Route Guards (backend/main.py):
6. Apply require_role('owner') to: /admin/run-analysis, /admin/recalibrate, delete endpoints, /admin/sport-switch, user management endpoints
7. Apply require_role('owner', 'risk_manager') to: /admin/bankroll POST, /admin/alerts/*/acknowledge, /admin/markets/*/pause
8. Add audit_log write on every /admin/* endpoint (log action + user_id + endpoint + payload)

Phase D — New Endpoints:
9. GET /admin/users — list all users (owner only)
10. POST /admin/users — create user, generate API key, return key ONCE (owner only)
11. DELETE /admin/users/{id} — deactivate user (owner only)
12. POST /admin/markets/{id}/pause — pause a betting market (risk_manager+)
13. DELETE /admin/markets/{id}/pause — resume market (risk_manager+)
14. GET /admin/audit-log — last 100 entries (owner only)

Phase E — Frontend:
15. Extend frontend/app/(dashboard)/admin/page.tsx with a tabbed layout:
    - Tab 0: System Status (existing panels, unchanged)
    - Tab 1: Risk Controls (market pause toggles, line override inputs) — visible to risk_manager+
    - Tab 2: User Management (user list, add/revoke) — visible to owner only
    - Tab 3: Audit Log table (who/what/when) — visible to owner only
    - Tab 4: Quota Monitor (calls used, burn rate chart, 30-day trend) — visible to risk_manager+
    Role visibility: read current user's role from GET /api/me (new endpoint returning {username, role})

Phase F — Tests:
16. tests/test_auth_rbac.py:
    - risk_manager key → 403 on DELETE /admin/bets/1
    - risk_manager key → 200 on POST /admin/markets/1/pause
    - owner key → 200 on DELETE /admin/bets/1
    - Every admin action creates audit_log entry

Report: all test results, role matrix verification table.
Push to branch claude/clarify-bet-recommendations-ui-WC8Do.
```

---

## 14. OPENCLAW AUTONOMY WORKSTREAM — Strategic Initiative

> **Owner:** Kimi CLI (Deep Intelligence Unit) — **LEAD ARCHITECT**  
> **Status:** SPEC PHASE — Awaiting waiver wire completion  
> **Goal:** Transform OpenClaw from heuristic checker to autonomous "Soul" system per original vision  
> **Value:** Automated model monitoring, alpha decay detection, self-improvement  

### 14.1 Vision: From Tool to Autonomous Agent

Current OpenClaw (v3.0) is a **function** — called during analysis, returns verdict.

Full OpenClaw (v4.0+) is an **autonomous system** per SOUL.md:
- **Self-directed:** Monitors performance without human triggers
- **Self-aware:** Detects when model edge degrades (alpha decay)
- **Self-improving:** Proposes and implements code improvements
- **Always-on:** 24/7 monitoring, alerting, optimization

### 14.2 Core Mandates (from SOUL.md)

| Mandate | Current | Target | Owner |
|---------|---------|--------|-------|
| **Alpha Decay Detection** | ❌ None | ✅ Every 2 hours | Kimi CLI |
| **Structural Vulnerability** | ❌ Manual | ✅ Automated pattern analysis | Kimi CLI |
| **Roadmap Evolution** | ❌ Static | ✅ Living improvement proposals | Kimi CLI |
| **Autonomous Implementation** | ❌ Disabled | ⚠️ Post-Apr 7 (Guardian) | Kimi proposes; Claude implements |

### 14.3 Architecture Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPENCLAW AUTONOMOUS SYSTEM v4.0                   │
├─────────────────────────────────────────────────────────────────────┤
│  📊 Performance Monitor                                              │
│     ├── CLV decay tracker (every 2h)                                │
│     ├── Win rate trend analysis                                     │
│     ├── Conference/team pattern detection                           │
│     └── Alert when edge degrades > threshold                        │
│                                                                      │
│  🔍 Data Validator                                                   │
│     ├── Ratings source freshness checks                             │
│     ├── Odds line movement anomalies                                │
│     └── Injury data completeness                                    │
│                                                                      │
│  🧠 Learning Engine                                                  │
│     ├── Loss pattern analysis (conference, total, seed, HCA)        │
│     ├── Feature importance drift detection                          │
│     └── Automatic A/B test proposals                                │
│                                                                      │
│  📝 Roadmap Maintainer                                               │
│     ├── Auto-updates ROADMAP.md with findings                       │
│     ├── Ranks improvements by expected ROI                          │
│     └── Schedules implementation queue                              │
│                                                                      │
│  🔧 Self-Improvement (Post-Apr 7)                                    │
│     ├── Auto-recalibration triggers                                 │
│     ├── Weight adjustments based on backtests                       │
│     └── Safe code modification with rollback                        │
│                                                                      │
│  📢 Notifier                                                         │
│     ├── Discord alerts for drift, anomalies, improvements           │
│     ├── Morning brief with health summary                           │
│     └── Escalation queue for human review                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.4 Deliverables & Timeline (CORRECTED — Phase 1 Starts NOW)

**✅ Phase 0: Design (COMPLETE)**
- [x] Kimi CLI: Full architecture spec → `reports/OPENCLAW_AUTONOMY_SPEC_v4.md`
- [x] Kimi CLI: Database schema for time-series metrics
- [x] Kimi CLI: API contracts for all agents

**🚀 Phase 1: Foundation (START NOW — Mar 24)**
*Does NOT violate Guardian freeze (read-only monitoring)*
- [ ] Performance Monitor service (`backend/services/openclaw/performance_monitor.py`)
- [ ] CLV decay detection algorithm
- [ ] Pattern analysis engine
- [ ] Discord alerting integration
- [ ] Database migration (4 new tables)

**Phase 2: Intelligence (Mar 31-Apr 7)**
- [ ] Learning Engine with historical analysis
- [ ] Roadmap auto-maintenance
- [ ] A/B test framework design
- [ ] Weekly Monday 6 AM job

**Phase 3: Autonomy Setup (Apr 1-7)**
- [ ] Self-improvement framework (disabled mode)
- [ ] Rollback mechanism
- [ ] Safety constraint validation

**Phase 4: Activation (Apr 8+)**
- [ ] Enable self-improvement (auto-implementation)
- [ ] Full A/B test automation
- [ ] Continuous learning loop
- [ ] Weekly autonomy reports

### 14.5 Key Technical Decisions

**Database:** Use existing `player_daily_metrics` time-series schema (EPIC-1)

**Scheduling:** Integrate with `DailyIngestionOrchestrator` (EPIC-2)
- Performance checks every 2 hours
- Pattern analysis nightly
- Roadmap updates weekly

**Safety:** All self-improvement proposals require:
1. Kimi CLI review
2. Backtest validation
3. Claude Code approval (architect)
4. Rollback plan

**Guardian Compliance (Clarified):**

**ALLOWED NOW (Doesn't touch frozen CBB model files):**
- ✅ Read-only monitoring (querying bet_log, predictions tables)
- ✅ Pattern detection (analyzing outcomes)
- ✅ Proposal generation (writing to DB/markdown)
- ✅ Discord alerting (notifications)
- ✅ NEW infrastructure (monitoring tables, scheduler jobs)

**BLOCKED until Apr 7 (Would modify frozen files):**
- ❌ Auto-recalibration of betting_model.py parameters
- ❌ Auto-adjustment of WEIGHT_KENPOM etc.
- ❌ Self-modifying Python code

**Implementation Strategy:** Build everything NOW. Only the "auto-implement" switch stays off until Apr 8.

### 14.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Alpha decay detection latency | <4 hours | Time from edge degradation → alert |
| Pattern detection accuracy | >80% | Manual validation of flagged patterns |
| Improvement proposal quality | >70% accepted | Ratio of approved vs rejected proposals |
| Autonomous fix success rate | >95% | Rollbacks vs successful deployments |
| Human oversight reduction | 80% fewer manual reviews | Post-autonomy vs pre-autonomy |

### 14.7 Specification Delivered

**Status:** ✅ **DESIGN COMPLETE** — Full specification in `reports/OPENCLAW_AUTONOMY_SPEC_v4.md`

**What's Been Delivered:**
- Complete 6-agent architecture (Performance, Pattern, Learning, Roadmap, Self-Improvement, Notifier)
- Database schema for time-series metrics, vulnerabilities, proposals, A/B tests
- CLV decay detection algorithm with thresholds
- Pattern detection for conferences, totals, seeds, HCA
- Proposal generation and ranking logic
- A/B test framework design
- Safety constraints for autonomous implementation
- 4-phase implementation plan (Foundation → Intelligence → Autonomy → Activation)
- Discord integration specs
- Success metrics and KPIs

**Handoff Protocol:**

**When Claude completes waiver wire:**
1. Kimi CLI reviews Claude's waiver implementation
2. Kimi CLI provides audit feedback
3. Claude implements OpenClaw Phase 1 (Performance + Pattern agents) per spec
4. Kimi designs Phase 2 (Learning + Roadmap agents)
5. Weekly sync: Claude (implementation) + Kimi (architecture) + OpenClaw (execution)

---

**Document Version:** EMAC-082
**Last Updated:** March 25, 2026
**Status:** ACTIVE — EPIC-1/2/3 COMPLETE. MLB model LIVE + team wRC+ wired (14 tests pass). Integrity sweep hang FIXED. Next: EPIC-4 Bracket Sunset (Apr 7).
**Branch:** main
**Team:** Claude Code (Architect) · Kimi CLI (Audit) · OpenClaw (Execution Target) · Gemini (Ops/Railway only)
**Next operator (Claude Code):** EPIC-4 Bracket Sunset — run after CBB championship Apr 7. Prompt is in §13.
**Next operator (Gemini CLI):** (1) Set `ENABLE_MLB_ANALYSIS=true` in Railway env vars. (2) Set `INTEGRITY_SWEEP_ENABLED=false` in Railway env vars (CRITICAL — prevents container restart loop). (3) OpenClaw migration already applied (learning_journal, model_performance_metrics, roadmap_state, vulnerability_reports created).
**Next operator (Kimi CLI):** Audit MLB betting model + team wRC+ implementation. Key files: `backend/services/mlb_analysis.py`, `backend/fantasy_baseball/pybaseball_loader.py`, `backend/fantasy_baseball/advanced_metrics.py`. Also review OpenClaw CLI (`scripts/openclaw_cli.py`) for manual monitoring operations.
**CRITICAL REMINDER:** See ADR-010 — Next.js is the ONLY UI. Streamlit (`dashboard/`) is RETIRED. Never reference Streamlit code.
**Apr 7 mission:** V9.2 recalibration — see §10 and prior HANDOFF.md §6
**Workstream Split (PARALLEL EXECUTION):**
- **Claude (P0 — Done):** MLB betting model COMPLETE — `SportConfig.mlb()` + `mlb_analysis.py` + team wRC+ ingestion (14 tests pass)
- **Claude (P1 — Apr 7):** EPIC-4 Bracket Sunset — see §13 for copy-paste prompt
- **Kimi (P1 — IN PROGRESS):** OpenClaw Phase 1 COMPLETE — 24 tests pass. Pending Railway migration.
- **Gemini (Ops):** Set `INTEGRITY_SWEEP_ENABLED=false` + `ENABLE_MLB_ANALYSIS=true` in Railway
- **URGENT:** Set `INTEGRITY_SWEEP_ENABLED=false` NOW — app is in restart loop without it

**OpenClaw Phase 1 Status (COMPLETE - Pending Migration):**
- ✅ `backend/services/openclaw/` package created
- ✅ `performance_monitor.py` — CLV decay detection (15% CRITICAL, 8% WARNING), win rate tracking
- ✅ `pattern_detector.py` — CBB patterns (conference, seed, HCA, month, day-of-week), MLB patterns framework
- ✅ `database.py` — Guardian-gated DB layer (read-only until Apr 7)
- ✅ `scheduler.py` — APScheduler integration (every 2h performance check, daily 6 AM sweep)
- ✅ `v8_openclaw_monitoring.sql` — Migration with 4 tables + views
- ✅ `apply_openclaw_migration.py` — Migration script
- ✅ `daily_ingestion.py` updated to auto-start OpenClaw monitoring
- ✅ `tests/openclaw/` — 24 tests covering PerformanceMonitor and PatternDetector
- ⏳ PENDING: Run migration on Railway (`python scripts/migrations/apply_openclaw_migration.py`)
- ⏳ PENDING: Verify Discord alerting integration (requires webhook URL)

**OpenClaw Implementation Notes:**
- Read-only monitoring during Guardian freeze — write operations blocked until Apr 7
- Phase 1 delivers foundation: monitoring + detection without self-modification
- Phase 2-4 (Learning, Roadmap, Self-improvement) scheduled post-Apr 7 per spec
- CBB patterns: conference bias, seed mispricing, HCA errors, month/day drift
- MLB patterns: framework ready for pitch fatigue, platoon splits, Coors effect (requires MLB data layer)
