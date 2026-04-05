# Fantasy / Edge Decoupling Design
**Date:** 2026-04-04  
**Author:** Claude Code (Principal Architect)  
**Status:** Approved — ready for implementation planning  

---

## Context

The platform runs two operationally distinct products inside a single FastAPI monolith (`backend/main.py`, 239KB):

1. **Edge Service** — sports betting analysis engine. Currently CBB (V9.1). Expanding to MLB post-Apr 7 via BallDontLie GOAT API. Will eventually cover all major sports. Brand is sports-agnostic by design.
2. **Fantasy Service** — MLB fantasy baseball platform. Yahoo API, lineup optimisation, waiver analysis, ensemble projections. Completely standalone product.

The code-level isolation is already strong — zero illegal cross-imports exist, GUARDIAN FREEZE and ADR-004 are respected throughout. The coupling is **structural**: one app file, one scheduler, one models file, one Postgres instance. This structural coupling creates three operational problems:

- CBB freeze (EMAC-068) cannot isolate the Fantasy service — both live or die together on deploy.
- A Fantasy bug fix requires deploying alongside the frozen CBB model.
- The upcoming BDL migration and MLB expansion are cleaner if the Edge service owns its own data layer with no fantasy tables in scope.

The goal is a **Modular Monorepo with two independent Railway service entry points**, migrated via strangler fig so production is never broken.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Repo structure | Monorepo — same git repo | One `requirements.txt`, atomic cross-system commits, shared CI. Two repos is coordination overhead for this team size. |
| Service entry points | Two FastAPI apps: `edge_app.py` + `fantasy_app.py` | Independent Railway deploys, independent schedulers, independent scaling. |
| Databases | Separate Postgres per service | True data isolation. Edge owns betting tables; Fantasy owns all fantasy tables. No cross-DB queries ever. |
| Redis | Shared Railway instance, namespaced | One instance at this traffic scale. `edge:` and `fantasy:` prefixes enforced by `redis_client.py` helpers. |
| Migration strategy | Strangler fig | `main.py` stays live throughout. New entry points built alongside. Railway cut over once validated. `main.py` deleted last. |
| Naming | Sports-agnostic | No "cbb" in entry points, routers, or model files. Edge service is the betting/analysis engine regardless of sport. |
| `TeamProfile` | Edge DB only | Fantasy lineup optimisation is player-centric. Fantasy sources team context from Yahoo + Statcast directly. |
| `DBAlert`, `DataIngestionLog` | Duplicated into each DB | Lightweight. Each service owns its own alerts and ingestion logs. Cleaner than a shared table. |

---

## Target File Structure

### New files (created during migration)

```
backend/
├── edge_app.py                    ← Edge service entry point
├── fantasy_app.py                 ← Fantasy service entry point
├── db.py                          ← SQLAlchemy Base, engine factory, SessionLocal
├── redis_client.py                ← Shared Redis connection + namespace helpers
├── models_edge.py                 ← All edge/betting SQLAlchemy models
├── models_fantasy.py              ← All fantasy SQLAlchemy models
├── routers/
│   ├── __init__.py
│   ├── edge.py                    ← All 19 betting/analysis API routes
│   ├── fantasy.py                 ← All 20 fantasy API routes
│   └── admin.py                   ← Shared admin routes (health, ingestion status)
└── schedulers/
    ├── __init__.py
    ├── edge_scheduler.py          ← CBB jobs: nightly_analysis, outcomes, capture_lines, odds_monitor, clv, recalibration
    └── fantasy_scheduler.py       ← Fantasy jobs: fangraphs_ros, ensemble_update, yahoo_adp_injury, freshness, waiver_scan, statcast, rolling_z
```

### Files that change role

```
backend/main.py        → Strangler fig shim during migration; deleted in Phase 7
backend/models.py      → Re-export shim (from models_edge + models_fantasy) during transition; deleted in Phase 7
```

### Files that do not change

```
backend/betting_model.py           (GUARDIAN FREEZE — untouched)
backend/services/analysis.py       (GUARDIAN FREEZE — untouched)
backend/fantasy_baseball/          (all 35+ files — untouched)
backend/services/daily_ingestion.py
backend/services/job_queue_service.py
backend/core/
backend/utils/
```

---

## Service Topology

```
┌─────────────────────────────────────┐   ┌──────────────────────────────────────┐
│  EDGE SERVICE                       │   │  FANTASY SERVICE                     │
│  uvicorn backend.edge_app:app       │   │  uvicorn backend.fantasy_app:app     │
│                                     │   │                                      │
│  routers/edge.py                    │   │  routers/fantasy.py                  │
│  schedulers/edge_scheduler.py       │   │  schedulers/fantasy_scheduler.py     │
│  models_edge.py                     │   │  models_fantasy.py                   │
│                                     │   │                                      │
│  Data: KenPom, BartTorvik,          │   │  Data: Yahoo API, pybaseball,        │
│        OddsAPI (CBB until Apr 7),   │   │        FanGraphs RoS, OpenWeather,   │
│        BDL GOAT (MLB post Apr 7),   │   │        BDL injuries (TBD post Apr 7) │
│        DDGS/OpenClaw                │   │                                      │
│                                     │   │                                      │
│  DATABASE_URL → edge-postgres       │   │  FANTASY_DATABASE_URL → fantasy-pg   │
│  REDIS_URL → shared Redis (edge:)   │   │  REDIS_URL → shared Redis (fantasy:) │
└─────────────────────────────────────┘   └──────────────────────────────────────┘
           │                                           │
           └──────────────┬────────────────────────────┘
                          │
             ┌────────────▼─────────────┐
             │  Shared Railway Redis     │
             │  edge:*    → betting use  │
             │  fantasy:* → fantasy use  │
             └───────────────────────────┘
```

---

## Module Specifications

### `backend/db.py`

Extracted from `backend/models.py`. Contains only:
- `DATABASE_URL` env var read (with fallback)
- SQLAlchemy `engine` creation with pool config
- `SessionLocal` factory
- `Base` declarative base
- `get_db()` dependency (the retry wrapper currently in models.py)

Both `models_edge.py` and `models_fantasy.py` import `Base` and `SessionLocal` from here. Each service passes its own `DATABASE_URL` (Edge) or `FANTASY_DATABASE_URL` (Fantasy) at startup — the engine factory accepts the URL as a parameter so tests can inject a different DB.

```python
# backend/db.py — simplified interface
def make_engine(database_url: str) -> Engine: ...
def make_session_factory(engine: Engine) -> sessionmaker: ...
Base = declarative_base()
```

### `backend/redis_client.py`

```python
# backend/redis_client.py
import redis
import os

_client: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _client
    if _client is None:
        _client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
    return _client

class NamespacedCache:
    def __init__(self, prefix: str):
        self._prefix = prefix

    def key(self, k: str) -> str:
        return f"{self._prefix}:{k}"

    def get(self, k: str): return get_redis().get(self.key(k))
    def set(self, k: str, v, ex: int | None = None): get_redis().set(self.key(k), v, ex=ex)
    def delete(self, k: str): get_redis().delete(self.key(k))

edge_cache = NamespacedCache("edge")
fantasy_cache = NamespacedCache("fantasy")
```

### `backend/models_edge.py`

Contains: `Game`, `Prediction`, `BetLog`, `ClosingLine`, `PerformanceSnapshot`, `ModelParameter`, `DataFetch`, `TeamProfile`, `DBAlert`, `DataIngestionLog`

Imports `Base` from `backend/db.py`. No imports from `models_fantasy.py` — ever.

### `backend/models_fantasy.py`

Contains: `FantasyDraftSession`, `FantasyDraftPick`, `FantasyLineup`, `PlayerDailyMetric`, `StatcastPerformance`, `PlayerProjection`, `PatternDetectionAlert`, `UserPreferences`, `ProjectionCacheEntry`, `ProjectionSnapshot`, `PlayerValuationCache`, `JobQueue`, `DBAlert`, `DataIngestionLog`

`DBAlert` and `DataIngestionLog` are **redefined here** (not imported from models_edge). Each service owns its own copy in its own DB. No sync required.

Imports `Base` from `backend/db.py`. No imports from `models_edge.py` — ever. No imports from `betting_model.py` — ever (GUARDIAN FREEZE still applies).

### `backend/edge_app.py`

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.routers.edge import router as edge_router
from backend.routers.admin import router as admin_router
from backend.schedulers.edge_scheduler import start_edge_scheduler, stop_edge_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_edge_scheduler()
    yield
    stop_edge_scheduler()

app = FastAPI(title="Edge Analytics API", lifespan=lifespan)
app.include_router(edge_router, prefix="/api")
app.include_router(admin_router, prefix="/admin")
```

### `backend/fantasy_app.py`

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from backend.routers.fantasy import router as fantasy_router
from backend.routers.admin import router as admin_router
from backend.schedulers.fantasy_scheduler import start_fantasy_scheduler, stop_fantasy_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_fantasy_scheduler()
    yield
    stop_fantasy_scheduler()

app = FastAPI(title="Fantasy Baseball API", lifespan=lifespan)
app.include_router(fantasy_router, prefix="/api/fantasy")
app.include_router(admin_router, prefix="/admin")
```

---

## Redis Usage by Service

### Edge service (`edge:` namespace)

| Key pattern | TTL | Purpose |
|-------------|-----|---------|
| `edge:bdl:rate:{endpoint}` | 1s | BDL GOAT rate limiter (post Apr 7) |
| `edge:lines:{game_key}` | 300s | Line movement snapshot cache |
| `edge:bdl:response:{hash}` | 3600s | BDL API response dedup TTL |

### Fantasy service (`fantasy:` namespace)

| Key pattern | TTL | Purpose |
|-------------|-----|---------|
| `fantasy:ros:{type}:{date}` | 43200s (12h) | FanGraphs RoS projection cache — replaces `_ROS_CACHE` module dict |
| `fantasy:ensemble:{date}` | 43200s (12h) | Ensemble blend output cache — survives service restarts |
| `fantasy:yahoo:token` | 3500s | Yahoo OAuth access token — replaces env var round-trip |
| `fantasy:job:{job_id}` | 86400s | Async job results (replaces PostgreSQL job_queue for lightweight ops) |
| `fantasy:bdl:injuries:{date}` | 3600s | BDL injury feed cache (post Apr 7, if Fantasy uses BDL) |

---

## Database Split

### Edge Postgres (`DATABASE_URL`)

Existing Railway Postgres. No schema changes — all existing tables stay. Fantasy tables are eventually removed (Phase 6).

### Fantasy Postgres (`FANTASY_DATABASE_URL`)

New Railway Postgres service. Provisioned in Phase 4. Schema initialised by `python -m backend.models_fantasy` before first Fantasy service deploy.

**Migration sequence for Fantasy DB:**
1. Provision Railway Postgres → copy `DATABASE_URL` value as `FANTASY_DATABASE_URL`
2. Run `railway run --service fantasy python -m backend.models_fantasy` → creates all fantasy tables
3. Run `scripts/migrate_v9_live_data.py` and `scripts/migrate_v10_user_preferences.py` against Fantasy DB
4. Validate: `railway run --service fantasy python -c "from backend.models_fantasy import PlayerDailyMetric; print('ok')"`
5. Point Fantasy service at new DB in Railway env vars
6. Verify `blend_*` columns exist before enabling ingestion orchestrator

---

## Scheduler Split

### `backend/schedulers/edge_scheduler.py`

Jobs: `nightly_analysis`, `update_outcomes`, `settle_games_daily`, `capture_closing_lines`, `line_monitor`, `daily_snapshot`, `fetch_ratings`, `odds_monitor`, `opener_attack_*`, `nightly_health_check`, `weekly_recalibration`

Startup behaviour: reads `ENABLE_MAIN_SCHEDULER` (from `backend/utils/deployment.py` — already on infra branch). If false, logs "Edge scheduler disabled" and returns. This preserves the infra branch's env-flag work during the transition.

### `backend/schedulers/fantasy_scheduler.py`

Jobs: `statcast`, `rolling_z`, `waiver_scan`, `mlb_brief`, `fangraphs_ros`, `yahoo_adp_injury`, `ensemble_update`, `projection_freshness`, `job_queue_processor`, `valuation_cache`

Startup behaviour: reads `ENABLE_INGESTION_ORCHESTRATOR`. Wraps `DailyIngestionOrchestrator` if enabled.

---

## Migration Phases (Strangler Fig)

### Phase 1 — Infrastructure foundation (no behaviour change)
- Create `backend/db.py` — extract engine/SessionLocal/Base/get_db from `models.py`
- Create `backend/redis_client.py` — `NamespacedCache`, `edge_cache`, `fantasy_cache`
- Create `backend/routers/__init__.py`, `backend/schedulers/__init__.py`
- `models.py` becomes a re-export shim: `from backend.models_edge import *; from backend.models_fantasy import *`
- All existing imports continue to work — zero behaviour change
- Verify: `py_compile` + full pytest suite green

### Phase 2 — Model file split
- Create `backend/models_edge.py` with all edge model classes
- Create `backend/models_fantasy.py` with all fantasy model classes (redefine `DBAlert`, `DataIngestionLog`)
- Update `models.py` shim to import from both
- Verify: `py_compile` + full pytest suite green

### Phase 3 — Router extraction
- Create `backend/routers/edge.py` — extract all 19 betting routes from `main.py`
- Create `backend/routers/fantasy.py` — extract all 20 fantasy routes from `main.py`
- Create `backend/routers/admin.py` — extract all `/admin/*` routes
- `main.py` lifespan is simplified to: mount all three routers + start both schedulers
- Verify: `py_compile` + full pytest suite + smoke test each router prefix

### Phase 4 — Scheduler extraction
- Create `backend/schedulers/edge_scheduler.py` — extract CBB job registration from `main.py`
- Create `backend/schedulers/fantasy_scheduler.py` — extract fantasy job registration from `main.py`
- `main.py` lifespan now calls `start_edge_scheduler()` + `start_fantasy_scheduler()`
- Verify: scheduler jobs fire correctly after deploy

### Phase 5 — New entry points
- Create `backend/edge_app.py` — mounts edge router + edge scheduler only
- Create `backend/fantasy_app.py` — mounts fantasy router + fantasy scheduler only
- Test both locally: `uvicorn backend.edge_app:app` and `uvicorn backend.fantasy_app:app`
- Create second Railway service pointing at `main` branch with `uvicorn backend.fantasy_app:app`
- Validate Fantasy service live against existing DB before DB split
- Once validated: repoint Railway Edge service to `uvicorn backend.edge_app:app`
- `main.py` still exists — rollback is one env var change on Railway

### Phase 6 — DB and Redis integration
- Provision Fantasy Postgres on Railway
- Run Fantasy DB schema init against new instance
- Update Fantasy service `DATABASE_URL` → `FANTASY_DATABASE_URL` pointing at new Postgres
- **Data migration decision:** Most fantasy tables can start fresh (lineup history, draft sessions, statcast snapshots). `UserPreferences` is the only table with user-authored data worth preserving — write a one-shot migration script: `scripts/migrate_fantasy_db.py` that copies `user_preferences` rows from Edge Postgres to Fantasy Postgres. All other tables (PlayerDailyMetric, ProjectionCacheEntry, PlayerValuationCache, FantasyDraftSession, FantasyDraftPick, FantasyLineup, StatcastPerformance, PlayerProjection, PatternDetectionAlert, ProjectionSnapshot, JobQueue) are rebuilt from live API sources within 24h of the Fantasy service starting.
- Add `REDIS_URL` to both services on Railway
- Replace `_ROS_CACHE` dict with `fantasy_cache` Redis calls in `daily_ingestion.py`
- Replace Yahoo OAuth token env-var fetch with `fantasy_cache.get("fantasy:yahoo:token")`
- Verify ingestion orchestrator writes to Fantasy DB, edge scheduler writes to Edge DB

### Phase 7 — Cleanup
- Delete `backend/main.py`
- Delete `backend/models.py` re-export shim
- Remove `DEPLOYMENT_ROLE`, `ENABLE_MAIN_SCHEDULER`, `ENABLE_STARTUP_CATCHUP` env flags from infra branch (replaced by structural separation — Edge app simply doesn't import fantasy scheduler)
- Merge `infra/mlb-uat-isolation` branch — `league_contract.py`, `deployment.py`, Dockerfile guards all land
- Update Railway config — both services use the same repo and same `Dockerfile`. Railway differentiates them via **Start Command** override in each service's Settings tab: Edge service sets `uvicorn backend.edge_app:app --host 0.0.0.0 --port $PORT`, Fantasy service sets `uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT`. No separate Dockerfile needed.
- Update `CLAUDE.md`, `ORCHESTRATION.md`, `HANDOFF.md` to reflect new entry points
- Add `.superpowers/` to `.gitignore`

---

## Railway Configuration

### Edge service (`stable/cbb-prod` branch)

```
# railway.json (backend service)
CMD: uvicorn backend.edge_app:app --host 0.0.0.0 --port ${PORT:-8000}

Environment variables:
  DATABASE_URL          = <existing edge postgres>
  REDIS_URL             = <shared redis>
  DEPLOYMENT_ROLE       = edge-prod
  THE_ODDS_API_KEY      = <existing>
  NIGHTLY_CRON_HOUR     = 3
  ENABLE_MAIN_SCHEDULER = true
```

### Fantasy service (`main` branch)

```
# railway.json (fantasy service)
CMD: uvicorn backend.fantasy_app:app --host 0.0.0.0 --port ${PORT:-8000}

Environment variables:
  FANTASY_DATABASE_URL          = <new fantasy postgres>
  REDIS_URL                     = <shared redis>
  DEPLOYMENT_ROLE               = fantasy-prod
  YAHOO_CLIENT_ID               = <existing>
  YAHOO_CLIENT_SECRET           = <existing>
  YAHOO_REFRESH_TOKEN           = <existing>
  YAHOO_LEAGUE_ID               = 72586
  OPENWEATHER_API_KEY           = <existing>
  ENABLE_INGESTION_ORCHESTRATOR = true
```

---

## Testing Strategy

### During migration (Phases 1–5)
- Full pytest suite must stay green after every phase
- No new tests required — existing tests validate behaviour is preserved
- Each phase ends with `venv/Scripts/python -m py_compile` on all modified files

### Phase 5 validation (new entry points)
- Smoke test: `GET /health` returns 200 on both `edge_app` and `fantasy_app`
- Smoke test: `GET /api/predictions` works on edge service
- Smoke test: `GET /api/fantasy/lineup/{date}` works on fantasy service
- Verify schedulers: check `/admin/ingestion/status` on fantasy service shows all jobs initialised
- Verify edge scheduler: nightly_analysis registered in APScheduler job list

### Phase 6 validation (DB split)
- `railway run --service fantasy python -c "from backend.models_fantasy import PlayerDailyMetric; print(PlayerDailyMetric.__tablename__)"` → `player_daily_metrics`
- `railway run --service edge python -c "from backend.models_edge import Game; print(Game.__tablename__)"` → `games`
- Confirm no cross-DB queries exist: `grep -r "FANTASY_DATABASE_URL" backend/models_edge.py` → empty
- Run full pytest suite after DB split

### Post-cleanup (Phase 7)
- `grep -r "from backend.main" backend/` → empty (no imports of deleted file)
- `grep -r "from backend.models import" backend/` → only `models_edge.py` and `models_fantasy.py` (shim deleted)
- `grep -r "betting_model" backend/fantasy_baseball/` → empty (GUARDIAN FREEZE structurally enforced)

---

## Out of Scope

- **Frontend decoupling** — the Next.js frontend is a separate Railway service already. The infra branch deleted fantasy frontend pages as a SSOT reset; rebuild of fantasy UI is a separate spec.
- **BDL GOAT MLB migration** — the Edge service's BDL integration (post Apr 7) proceeds independently of this decoupling work. The new `edge_app.py` entry point makes that migration cleaner but does not depend on it.
- **CBB V9.2 recalibration (EMAC-068)** — unlocks Apr 7. Proceeds inside `betting_model.py` unchanged — this decoupling does not touch the model math.
- **Re-branding / repo rename** — sports-agnostic naming is baked into all new files (`edge_app.py`, `models_edge.py`, etc.). The repo rename (`cbb-edge` → new name) is a separate Git/Railway operation.

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| `models.py` re-export shim causes circular imports | Medium | Keep shim as a pure re-export with no logic; test at Phase 1 boundary |
| Live data migration from shared Postgres to Fantasy Postgres loses rows | High | Write idempotent migration script; validate row counts before and after; keep old tables in Edge DB for 2 weeks as fallback |
| Redis key collision between services | Low | `NamespacedCache` enforces prefix at write time; add assertion in `redis_client.py` that all keys start with known prefix |
| Scheduler double-firing during Phase 4 (both `main.py` and new schedulers active) | Medium | Phase 4 `main.py` lifespan calls `start_edge_scheduler()` + `start_fantasy_scheduler()` exclusively — removes inline job registration from `main.py` lifespan before the phase is complete |
| Fantasy DB missing `blend_*` columns after schema init | Medium | Phase 6 checklist includes explicit column verification before enabling orchestrator |
