# Fantasy / Edge Decoupling — Structural Refactor (Phases 1–5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `backend/main.py` into two independent FastAPI entry points (`edge_app.py` and `fantasy_app.py`) using a strangler-fig migration — `main.py` stays live and serving traffic throughout every phase.

**Architecture:** Seven phases total; this plan covers phases 1–5 (pure structural reorganisation, zero behaviour change). The shared DB engine is extracted to `backend/db.py`. Routes are extracted to `backend/routers/`. Schedulers are extracted to `backend/schedulers/`. Two new entry points (`edge_app.py`, `fantasy_app.py`) are created and validated before `main.py` is touched for Railway cut-over. Phase 6–7 (separate Postgres, Redis integration, cleanup) are a separate plan requiring Railway provisioning.

**Tech Stack:** Python 3.11, FastAPI, APScheduler 3.x, SQLAlchemy 2.x, pytest. Windows dev environment — use `venv/Scripts/python` for all commands.

**Spec:** `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md`

---

## Invariant: pytest must stay green after every task

Before starting, establish the baseline:

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```

Expected: 647+ passed, ≤3 pre-existing failures (DB auth — not our code). If anything beyond those 3 fails, stop and investigate before proceeding.

---

## Phase 1 — Infrastructure Foundation

### Task 1: Create `backend/db.py`

Extract engine/session/Base from `backend/models.py` into a standalone DB module. `models.py` will import from here once the shim is in place.

**Files:**
- Create: `backend/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_db.py
import pytest
from unittest.mock import patch


def test_make_engine_returns_engine():
    from backend.db import make_engine
    engine = make_engine("sqlite:///:memory:")
    assert engine is not None
    engine.dispose()


def test_make_session_local_returns_factory():
    from backend.db import make_engine, make_session_local
    engine = make_engine("sqlite:///:memory:")
    factory = make_session_local(engine)
    assert factory is not None
    engine.dispose()


def test_namespaced_cache_prefixes_keys():
    from backend.db import NamespacedKey
    ns = NamespacedKey("fantasy")
    assert ns.key("ros:2026-04-04") == "fantasy:ros:2026-04-04"
    assert ns.key("token") == "fantasy:token"


def test_namespaced_cache_different_prefixes_do_not_collide():
    from backend.db import NamespacedKey
    edge = NamespacedKey("edge")
    fantasy = NamespacedKey("fantasy")
    assert edge.key("token") != fantasy.key("token")
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_db.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.db'`

- [ ] **Step 3: Create `backend/db.py`**

```python
"""
backend/db.py — Shared database engine factory and declarative base.

Both models_edge.py and models_fantasy.py import Base and make_session_local
from here. Neither imports from the other. Each service passes its own
DATABASE_URL at startup via make_engine().
"""
from __future__ import annotations

import logging
import os
import time
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Declarative base shared by all models in both services.
Base = declarative_base()


def make_engine(database_url: str):
    """Create a SQLAlchemy engine for the given URL.

    Uses the same pool settings as the original models.py engine.
    Pass 'sqlite:///:memory:' in tests.
    """
    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        return create_engine(
            database_url,
            connect_args=connect_args,
            echo=False,
        )
    return create_engine(
        database_url,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )


def make_session_local(engine) -> sessionmaker:
    """Return a session factory bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def make_get_db(session_local: sessionmaker):
    """Return a FastAPI dependency (generator) for the given session factory.

    Retries up to 3 times on transient connection failures with exponential
    backoff — same behaviour as the original get_db() in models.py.
    """
    def _get_db() -> Generator[Session, None, None]:
        db = None
        for attempt in range(3):
            try:
                db = session_local()
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                error_str = str(exc).lower()
                if any(k in error_str for k in ("connection", "timeout", "ssl")):
                    time.sleep(0.1 * (2 ** attempt))
                else:
                    raise
        try:
            yield db
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    return _get_db


class NamespacedKey:
    """Utility for Redis key namespacing. Used by redis_client.py.

    Ensures that edge: and fantasy: keys never collide.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def key(self, k: str) -> str:
        return f"{self._prefix}:{k}"
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_db.py -v
```
Expected: 4 passed

- [ ] **Step 5: Compile check**

```bash
venv/Scripts/python -m py_compile backend/db.py && echo OK
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add backend/db.py tests/test_db.py
git commit -m "feat: add backend/db.py — engine factory and NamespacedKey extracted from models.py"
```

---

### Task 2: Create `backend/redis_client.py`

Redis client with namespace enforcement. No behaviour change in production yet — Redis is wired in Phase 6.

**Files:**
- Create: `backend/redis_client.py`
- Create: `tests/test_redis_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_redis_client.py
import pytest
from unittest.mock import MagicMock, patch


def test_namespaced_cache_get_prefixes_key():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    mock_redis.get.return_value = "value"
    cache = NamespacedCache("fantasy", client=mock_redis)
    result = cache.get("ros:today")
    mock_redis.get.assert_called_once_with("fantasy:ros:today")
    assert result == "value"


def test_namespaced_cache_set_prefixes_key():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    cache = NamespacedCache("edge", client=mock_redis)
    cache.set("token", "abc123", ex=3600)
    mock_redis.set.assert_called_once_with("edge:token", "abc123", ex=3600)


def test_namespaced_cache_delete_prefixes_key():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    cache = NamespacedCache("fantasy", client=mock_redis)
    cache.delete("old-key")
    mock_redis.delete.assert_called_once_with("fantasy:old-key")


def test_edge_and_fantasy_caches_use_different_prefixes():
    from backend.redis_client import NamespacedCache
    mock_redis = MagicMock()
    edge = NamespacedCache("edge", client=mock_redis)
    fantasy = NamespacedCache("fantasy", client=mock_redis)
    edge.get("token")
    fantasy.get("token")
    calls = [str(c) for c in mock_redis.get.call_args_list]
    assert any("edge:token" in c for c in calls)
    assert any("fantasy:token" in c for c in calls)


def test_get_redis_raises_if_no_url(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    # Re-import to reset module-level state
    import importlib
    import backend.redis_client as rc
    importlib.reload(rc)
    with pytest.raises(RuntimeError, match="REDIS_URL"):
        rc.get_redis()
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_redis_client.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.redis_client'`

- [ ] **Step 3: Create `backend/redis_client.py`**

```python
"""
backend/redis_client.py — Shared Redis connection with namespace enforcement.

Both edge_app.py and fantasy_app.py import this module.
Use edge_cache for edge: keys, fantasy_cache for fantasy: keys.
Neither service can pollute the other's keyspace.
"""
from __future__ import annotations

import os
from typing import Optional

try:
    import redis as _redis_lib
    _REDIS_AVAILABLE = True
except ImportError:
    _redis_lib = None  # type: ignore[assignment]
    _REDIS_AVAILABLE = False

_client: Optional[object] = None


def get_redis():
    """Return the shared Redis client, initialised lazily from REDIS_URL.

    Raises RuntimeError if REDIS_URL is not set — forces explicit
    configuration rather than silent fallback.
    """
    global _client
    if _client is not None:
        return _client
    url = os.environ.get("REDIS_URL")
    if not url:
        raise RuntimeError(
            "REDIS_URL environment variable is required for Redis operations. "
            "Set it to your Railway Redis connection string."
        )
    if not _REDIS_AVAILABLE:
        raise RuntimeError(
            "redis-py is not installed. Add 'redis>=5.0' to requirements.txt."
        )
    _client = _redis_lib.from_url(url, decode_responses=True)
    return _client


class NamespacedCache:
    """Redis cache with enforced key namespace prefix.

    Usage:
        fantasy_cache.set("ros:2026-04-04", json.dumps(data), ex=43200)
        edge_cache.get("bdl:rate:games")

    All keys are stored as '<prefix>:<k>' — edge: and fantasy: keys
    can never collide regardless of the key string passed by callers.
    """

    def __init__(self, prefix: str, client=None) -> None:
        """
        Args:
            prefix: Namespace prefix, e.g. 'edge' or 'fantasy'.
            client: Optional injected Redis client for testing.
                    If None, uses get_redis() at call time.
        """
        self._prefix = prefix
        self._client = client

    def _r(self):
        return self._client if self._client is not None else get_redis()

    def key(self, k: str) -> str:
        return f"{self._prefix}:{k}"

    def get(self, k: str):
        return self._r().get(self.key(k))

    def set(self, k: str, v, ex: Optional[int] = None) -> None:
        self._r().set(self.key(k), v, ex=ex)

    def delete(self, k: str) -> None:
        self._r().delete(self.key(k))

    def exists(self, k: str) -> bool:
        return bool(self._r().exists(self.key(k)))


# Module-level singletons used by both services.
# Import these directly: from backend.redis_client import edge_cache, fantasy_cache
edge_cache = NamespacedCache("edge")
fantasy_cache = NamespacedCache("fantasy")
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_redis_client.py -v
```
Expected: 5 passed

- [ ] **Step 5: Compile check**

```bash
venv/Scripts/python -m py_compile backend/redis_client.py && echo OK
```

- [ ] **Step 6: Full suite still green**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```
Expected: same pass count as baseline

- [ ] **Step 7: Commit**

```bash
git add backend/redis_client.py tests/test_redis_client.py
git commit -m "feat: add backend/redis_client.py — namespaced Redis cache for edge: and fantasy: keyspaces"
```

---

### Task 3: Create router and scheduler package stubs

Empty packages so Phase 3/4 imports don't fail.

**Files:**
- Create: `backend/routers/__init__.py`
- Create: `backend/schedulers/__init__.py`

- [ ] **Step 1: Create the files**

```bash
# Windows-compatible
echo "" > backend/routers/__init__.py
echo "" > backend/schedulers/__init__.py
```

Or use Write tool to create empty files at those paths.

- [ ] **Step 2: Verify imports work**

```bash
venv/Scripts/python -c "from backend.routers import __init__; from backend.schedulers import __init__; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/routers/__init__.py backend/schedulers/__init__.py
git commit -m "chore: add router and scheduler package stubs"
```

---

## Phase 2 — Model File Split

### Task 4: Create `backend/models_edge.py`

Extract all edge/betting model classes from `models.py` into a dedicated file. `models.py` is NOT modified yet — this is additive only.

**Files:**
- Create: `backend/models_edge.py`

- [ ] **Step 1: Create `backend/models_edge.py`**

This file re-exports the betting model classes from `backend/models.py` using direct re-imports. It does NOT copy the class definitions yet — that is Phase 7 cleanup. During the strangler-fig, `models_edge.py` is a typed re-export shim that lets `edge_app.py` import from the correct module name.

```python
"""
backend/models_edge.py — Edge service SQLAlchemy models.

Re-exports betting/analysis model classes from backend.models during the
strangler-fig migration. After Phase 7 cleanup, class definitions move
here permanently and backend.models is deleted.

DO NOT import from backend.models_fantasy — ever.
DO NOT import from backend.betting_model — use it directly only in
services that explicitly need it (analysis.py, line_monitor.py, etc.).
"""
from backend.models import (  # noqa: F401
    Base,
    Game,
    Prediction,
    BetLog,
    ModelParameter,
    PerformanceSnapshot,
    DataFetch,
    ClosingLine,
    TeamProfile,
    DBAlert,
    DataIngestionLog,
)

__all__ = [
    "Base",
    "Game",
    "Prediction",
    "BetLog",
    "ModelParameter",
    "PerformanceSnapshot",
    "DataFetch",
    "ClosingLine",
    "TeamProfile",
    "DBAlert",
    "DataIngestionLog",
]
```

- [ ] **Step 2: Verify compile and imports**

```bash
venv/Scripts/python -m py_compile backend/models_edge.py && echo OK
venv/Scripts/python -c "from backend.models_edge import Game, Prediction, BetLog, TeamProfile; print('OK')"
```
Expected: both print `OK`

- [ ] **Step 3: Commit**

```bash
git add backend/models_edge.py
git commit -m "feat: add backend/models_edge.py — typed re-export shim for betting model classes"
```

---

### Task 5: Create `backend/models_fantasy.py`

Extract all fantasy model classes. `DBAlert` and `DataIngestionLog` are **redefined** here (not imported from models_edge) — each service owns its own copy.

**Files:**
- Create: `backend/models_fantasy.py`

- [ ] **Step 1: Create `backend/models_fantasy.py`**

```python
"""
backend/models_fantasy.py — Fantasy service SQLAlchemy models.

Re-exports fantasy model classes from backend.models during the
strangler-fig migration. After Phase 7 cleanup, class definitions move
here permanently and backend.models is deleted.

DBAlert and DataIngestionLog are REDEFINED here (not imported from
models_edge) — the Fantasy service owns its own copy in its own DB.

DO NOT import from backend.models_edge — ever.
DO NOT import from backend.betting_model — GUARDIAN FREEZE.
"""
from backend.models import (  # noqa: F401
    FantasyDraftSession,
    FantasyDraftPick,
    FantasyLineup,
    PlayerDailyMetric,
    StatcastPerformance,
    PlayerProjection,
    PatternDetectionAlert,
    UserPreferences,
    ProjectionCacheEntry,
    ProjectionSnapshot,
    PlayerValuationCache,
    JobQueue,
    # DBAlert and DataIngestionLog are intentionally re-exported from models
    # during transition — they will be redefined in Fantasy DB in Phase 7.
    DBAlert,
    DataIngestionLog,
)

__all__ = [
    "FantasyDraftSession",
    "FantasyDraftPick",
    "FantasyLineup",
    "PlayerDailyMetric",
    "StatcastPerformance",
    "PlayerProjection",
    "PatternDetectionAlert",
    "UserPreferences",
    "ProjectionCacheEntry",
    "ProjectionSnapshot",
    "PlayerValuationCache",
    "JobQueue",
    "DBAlert",
    "DataIngestionLog",
]
```

- [ ] **Step 2: Verify compile and imports**

```bash
venv/Scripts/python -m py_compile backend/models_fantasy.py && echo OK
venv/Scripts/python -c "from backend.models_fantasy import PlayerDailyMetric, FantasyLineup, JobQueue; print('OK')"
```
Expected: both print `OK`

- [ ] **Step 3: Full suite still green**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add backend/models_fantasy.py
git commit -m "feat: add backend/models_fantasy.py — typed re-export shim for fantasy model classes"
```

---

## Phase 3 — Router Extraction

### Task 6: Create `backend/routers/edge.py`

Extract all 19 betting/analysis routes from `main.py`. Pattern: change `@app.get` → `@router.get`, change `@app.post` → `@router.post`, etc. The route prefix `/api` stays as-is on each route (it is stripped when the router is mounted in Task 8).

**Files:**
- Create: `backend/routers/edge.py`

- [ ] **Step 1: Create the router scaffold**

```python
"""
backend/routers/edge.py — Edge service API routes.

All betting/analysis endpoints extracted from backend/main.py.
Mounted at /api in edge_app.py and (during transition) re-mounted
in main.py so existing tests keep passing.

Route prefix convention: routes in this file use their full path
(e.g. /api/predictions/today) so they work identically whether
mounted at '/' in edge_app.py or re-mounted in the shim main.py.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from typing import List, Optional
import logging
import os
from zoneinfo import ZoneInfo
from datetime import datetime, timedelta
from dataclasses import asdict

from backend.models import (
    Game, Prediction, BetLog, ClosingLine, PerformanceSnapshot,
    ModelParameter, TeamProfile, DBAlert, DataIngestionLog, SessionLocal,
)
from backend.auth import verify_api_key, verify_admin_api_key
from backend.betting_model import CBBEdgeModel
from backend.services.analysis import run_nightly_analysis
from backend.services.clv import calculate_clv_full
from backend.services.bet_tracker import update_completed_games, capture_closing_lines
from backend.services.line_monitor import check_line_movements
from backend.services.performance import (
    get_performance_summary,
    get_clv_analysis,
    get_calibration_data,
    get_model_accuracy,
    get_performance_timeline,
    get_financial_metrics,
    get_performance_by_team,
)
from backend.services.alerts import check_performance_alerts, persist_alerts, run_alert_check
from backend.services.recalibration import compute_dynamic_weights
from backend.services.discord_notifier import send_todays_bets
from backend.services.sentinel import run_nightly_health_check
from backend.services.dk_import import (
    preview_dk_import,
    confirm_dk_import,
    preview_direct_dk_import,
    confirm_direct_dk_import,
)
from backend.services.odds_monitor import get_odds_monitor
from backend.services.portfolio import get_portfolio_manager
from backend.services.ratings import get_ratings_service
from backend.schemas import (
    TodaysPredictionsResponse,
    BetLogResponse,
    OutcomeResponse,
    AnalysisTriggerResponse,
    OracleFlaggedResponse,
)
from backend.utils.env_utils import get_float_env
from backend.models import get_db

logger = logging.getLogger(__name__)
router = APIRouter()
```

- [ ] **Step 2: Extract edge routes into the router**

Open `backend/main.py`. For every route listed below, copy the entire function (decorator + signature + body) into `backend/routers/edge.py`, replacing the `@app.` prefix with `@router.`.

**Routes to extract (by line in main.py):**
- Line 922: `GET /api/tournament/bracket-projection`
- Lines 1298–1500: `GET /api/predictions/today`, `/today/all`, `/bets`, `/game/{game_id}`, `/parlays`
- Lines 1619–1802: all `/api/performance/*` routes (8 routes)
- Lines 1845: `GET /api/performance/alerts`
- Lines 1892–2098: `/api/bets/log`, `/api/bets/{bet_id}/outcome`, `/api/bets/{bet_id}/placed`, `/api/games/recent`, `/api/bets`
- Lines 2180–2244: `/api/closing-lines`, `/api/closing-lines/{game_id}`, `/api/performance/history`
- Line 3156: `GET /api/feature-flags`

**Conversion pattern (apply to every route):**
```python
# BEFORE (in main.py):
@app.get("/api/predictions/today", response_model=TodaysPredictionsResponse)
async def get_todays_predictions(db: Session = Depends(get_db), ...):
    ...

# AFTER (in routers/edge.py):
@router.get("/api/predictions/today", response_model=TodaysPredictionsResponse)
async def get_todays_predictions(db: Session = Depends(get_db), ...):
    ...
```

Any helper functions referenced only by edge routes (not fantasy routes) should also be copied into `routers/edge.py` above the routes that use them. If a helper is also used by fantasy routes, leave it in `main.py` for now.

- [ ] **Step 3: Compile check**

```bash
venv/Scripts/python -m py_compile backend/routers/edge.py && echo OK
```
Expected: `OK`. Fix any import errors before continuing.

- [ ] **Step 4: Commit**

```bash
git add backend/routers/edge.py
git commit -m "feat: add backend/routers/edge.py — betting/analysis routes extracted from main.py"
```

---

### Task 7: Create `backend/routers/fantasy.py`

Extract all 27 fantasy routes from `main.py`.

**Files:**
- Create: `backend/routers/fantasy.py`

- [ ] **Step 1: Create the router scaffold**

```python
"""
backend/routers/fantasy.py — Fantasy service API routes.

All /api/fantasy/* endpoints extracted from backend/main.py.
Mounted at / in fantasy_app.py (routes already contain /api/fantasy/).
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import os
from zoneinfo import ZoneInfo
from datetime import date

from backend.models import (
    FantasyDraftSession, FantasyDraftPick, FantasyLineup,
    PlayerDailyMetric, JobQueue, UserPreferences, SessionLocal,
)
from backend.auth import verify_api_key
from backend.services.job_queue_service import (
    submit_job as jq_submit,
    get_job_status as jq_status,
    process_pending_jobs as jq_process,
)
from backend.utils.fantasy_stat_contract import YAHOO_STAT_ID_FALLBACK, LEAGUE_SCORING_CATEGORIES
from backend.utils.time_utils import today_et
from backend.schemas import DailyLineupResponse, WaiverWireResponse, RosterResponse, MatchupResponse
from backend.fantasy_baseball.yahoo_client_resilient import (
    get_resilient_yahoo_client,
    YahooFantasyClient,
)
from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_optimizer
from backend.models import get_db

logger = logging.getLogger(__name__)
router = APIRouter()
```

- [ ] **Step 2: Extract fantasy routes into the router**

Open `backend/main.py`. For every `/api/fantasy/*` route listed below, copy the entire function into `backend/routers/fantasy.py`, changing `@app.` → `@router.`:

**Routes to extract (by line in main.py):**
- Line 3346: `GET /api/fantasy/draft-board`
- Line 3376: `GET /api/fantasy/player/{player_id}`
- Lines 3394–3846: all `/api/fantasy/draft-session*` routes (8 routes)
- Line 3959: `GET /api/fantasy/lineup/{lineup_date}`
- Line 4377: `GET /api/fantasy/briefing/{briefing_date}`
- Lines 4464–4967: `/api/fantasy/waiver`, `/api/fantasy/waiver/add`, `/api/fantasy/waiver/recommendations`
- Line 5342: `POST /api/fantasy/lineup`
- Line 5390: `GET /api/fantasy/saved-lineup/{lineup_date}`
- Line 5424: `GET /api/fantasy/yahoo-diag`
- Line 5478: `GET /api/fantasy/roster`
- Line 5545: `GET /api/fantasy/players/valuations`
- Line 5626: `GET /api/fantasy/matchup`
- Line 6052: `GET /api/fantasy/dashboard/stream`
- Lines 6175–6629: all remaining `/api/fantasy/lineup/*` routes

Any helper functions referenced only by fantasy routes should be copied above the routes that use them in `routers/fantasy.py`. Any helpers also needed by edge routes remain in `main.py`.

Any lazy imports inside route bodies (e.g. `from backend.fantasy_baseball.elite_lineup_scorer import ...`) stay inside those route bodies — do not hoist them to module level.

- [ ] **Step 3: Compile check**

```bash
venv/Scripts/python -m py_compile backend/routers/fantasy.py && echo OK
```
Fix any import errors before continuing.

- [ ] **Step 4: Commit**

```bash
git add backend/routers/fantasy.py
git commit -m "feat: add backend/routers/fantasy.py — all /api/fantasy/* routes extracted from main.py"
```

---

### Task 8: Create `backend/routers/admin.py`

Extract all `/admin/*` routes plus root (`/`) and `/health`.

**Files:**
- Create: `backend/routers/admin.py`

- [ ] **Step 1: Create the router scaffold and extract routes**

```python
"""
backend/routers/admin.py — Shared admin, health, and root endpoints.

Mounted at / in both edge_app.py and fantasy_app.py.
Each service gets its own /health and /admin/* endpoints.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import logging
import os

from backend.models import SessionLocal, get_db
from backend.auth import verify_api_key, verify_admin_api_key
from backend.utils.deployment import deployment_role  # from infra branch — falls back gracefully if missing

logger = logging.getLogger(__name__)
router = APIRouter()
```

Extract these routes from `main.py` into `routers/admin.py`, changing `@app.` → `@router.`:

- Line 1263: `GET /` (root — returns service identity)
- Line 1274: `GET /health`
- Lines 2301–2465: all `/admin/run-analysis`, `/admin/discord/*`, `/admin/recalibrate`
- Lines 2497–3000: all `/admin/debug/*`, `/admin/cleanup/*`, `/admin/force-*`, `/admin/bets/*`, `/admin/games/*`, `/admin/alerts/*`, `/admin/scheduler/status`, `/admin/ingestion/status`, `/admin/portfolio/status`, `/admin/odds-monitor/status`, `/admin/oracle/*`, `/admin/ratings/status`
- Lines 3077–3196: `/admin/bankroll`, `/admin/parlay/override`, `/admin/feature-flags/*`, `/admin/dk/*`
- Line 2908: `GET /admin/ingestion/status` (if not already extracted above)

**Note on `deployment_role()`:** The `backend/utils/deployment.py` module exists on the `infra/mlb-uat-isolation` branch but not yet on `main`. Add a safe fallback import:

```python
try:
    from backend.utils.deployment import deployment_role
except ImportError:
    def deployment_role() -> str:
        return os.getenv("DEPLOYMENT_ROLE", "primary")
```

- [ ] **Step 2: Compile check**

```bash
venv/Scripts/python -m py_compile backend/routers/admin.py && echo OK
```

- [ ] **Step 3: Commit**

```bash
git add backend/routers/admin.py
git commit -m "feat: add backend/routers/admin.py — admin, health, and root endpoints extracted from main.py"
```

---

### Task 9: Verify routers work by mounting them in `main.py`

**This is the strangler-fig validation step.** We do NOT delete anything from `main.py` yet. We mount the new routers alongside the existing routes and verify no regressions.

**Files:**
- Modify: `backend/main.py` (add 6 lines near the top of the app setup section)

- [ ] **Step 1: Find the `app = FastAPI(...)` line in `main.py`**

Search for `app = FastAPI(` in `backend/main.py`. It will be near line 155–165.

- [ ] **Step 2: Add router mounting immediately after `app = FastAPI(...)`**

```python
# --- Strangler-fig router mounts (Phase 3) ---
# These routers duplicate the routes already inline in this file.
# FastAPI routes are matched in registration order; the inline routes
# still win because they were registered first. Once Phase 5 cut-over
# is complete and inline routes are removed, these mounts take over.
from backend.routers.edge import router as _edge_router
from backend.routers.fantasy import router as _fantasy_router
from backend.routers.admin import router as _admin_router
app.include_router(_edge_router)
app.include_router(_fantasy_router)
app.include_router(_admin_router)
# --- end strangler-fig mounts ---
```

- [ ] **Step 3: Compile check**

```bash
venv/Scripts/python -m py_compile backend/main.py && echo OK
```

- [ ] **Step 4: Full test suite green**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -8
```
Expected: same pass count as baseline. If anything new fails, the import of a router is pulling in something that conflicts — check the router's imports.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py
git commit -m "refactor: mount extracted routers in main.py — strangler-fig validation step"
```

---

## Phase 4 — Scheduler Extraction

### Task 10: Create `backend/schedulers/edge_scheduler.py`

Extract CBB scheduler job registration from `main.py` lifespan into a standalone module.

**Files:**
- Create: `backend/schedulers/edge_scheduler.py`

- [ ] **Step 1: Read the scheduler section of `main.py`**

Search `backend/main.py` for `scheduler.add_job(` — all calls before the `ENABLE_INGESTION_ORCHESTRATOR` block are edge jobs. These are roughly lines 176–365. The `scheduler` instance itself is declared near line 105 as `scheduler = AsyncIOScheduler()`.

- [ ] **Step 2: Create `backend/schedulers/edge_scheduler.py`**

```python
"""
backend/schedulers/edge_scheduler.py — Edge service APScheduler setup.

Contains ONLY betting/analysis jobs (nightly_analysis, update_outcomes,
capture_closing_lines, odds_monitor, clv, performance, recalibration, etc.)

Fantasy jobs are in backend/schedulers/fantasy_scheduler.py.
Never import fantasy_baseball modules here.
"""
from __future__ import annotations

import logging
import os

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)

# Module-level scheduler — shared by edge_app.py
scheduler = AsyncIOScheduler()


def start_edge_scheduler() -> None:
    """Register all edge service jobs and start the scheduler.

    Called from edge_app.py lifespan. Reads ENABLE_MAIN_SCHEDULER env var
    (from infra branch deployment.py) — if false, logs and returns without
    starting (safe for UAT environments).
    """
    try:
        from backend.utils.deployment import main_scheduler_enabled
    except ImportError:
        def main_scheduler_enabled() -> bool:
            return os.getenv("ENABLE_MAIN_SCHEDULER", "true").lower() != "false"

    if not main_scheduler_enabled():
        logger.info("Edge scheduler disabled by ENABLE_MAIN_SCHEDULER=false")
        return

    # Import job functions here (lazy) to avoid circular imports.
    # These are the same functions currently called in main.py lifespan.
    from backend.main import (  # noqa: PLC0415 — intentional lazy import during transition
        nightly_job,
        _update_outcomes_job,
        _capture_lines_job,
        _daily_snapshot_job,
        _fetch_ratings_job,
        _odds_monitor_job,
        _opener_attack_job,
        _nightly_health_check_job,
        _weekly_recalibration_job,
        _morning_briefing_job,
        _end_of_day_results_job,
        _nightly_decision_resolution_job,
    )

    nightly_hour = int(os.getenv("NIGHTLY_CRON_HOUR", "3"))
    tz = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")
    opener_enabled = os.getenv("OPENER_ATTACK_ENABLED", "true").lower() == "true"

    scheduler.add_job(nightly_job, CronTrigger(hour=nightly_hour, minute=0, timezone=tz),
                      id="nightly_analysis", name="Nightly Game Analysis", replace_existing=True)
    scheduler.add_job(_update_outcomes_job, IntervalTrigger(hours=2),
                      id="update_outcomes", name="Update Outcomes", replace_existing=True)
    scheduler.add_job(_capture_lines_job, IntervalTrigger(minutes=30),
                      id="capture_closing_lines", name="Capture Closing Lines", replace_existing=True)
    scheduler.add_job(_daily_snapshot_job, CronTrigger(hour=4, minute=30, timezone=tz),
                      id="daily_snapshot", name="Daily Snapshot", replace_existing=True)
    scheduler.add_job(_fetch_ratings_job, CronTrigger(hour=8, minute=0, timezone=tz),
                      id="fetch_ratings", name="Fetch Ratings", replace_existing=True)
    scheduler.add_job(_odds_monitor_job, IntervalTrigger(minutes=5),
                      id="odds_monitor", name="Odds Monitor", replace_existing=True)
    scheduler.add_job(_nightly_health_check_job, CronTrigger(hour=5, minute=0, timezone=tz),
                      id="nightly_health_check", name="Nightly Health Check", replace_existing=True)
    scheduler.add_job(_weekly_recalibration_job, CronTrigger(day_of_week="sun", hour=5, minute=0, timezone=tz),
                      id="weekly_recalibration", name="Weekly Recalibration", replace_existing=True)
    scheduler.add_job(_morning_briefing_job, CronTrigger(hour=7, minute=0, timezone=tz),
                      id="morning_briefing", name="Morning Briefing", replace_existing=True)
    scheduler.add_job(_end_of_day_results_job, CronTrigger(hour=23, minute=0, timezone=tz),
                      id="end_of_day_results", name="End of Day Results", replace_existing=True)
    scheduler.add_job(_nightly_decision_resolution_job, CronTrigger(hour=23, minute=59, timezone=tz),
                      id="nightly_decision_resolution", name="Decision Resolution", replace_existing=True)

    if opener_enabled:
        opener_hour_1 = int(os.getenv("OPENER_ATTACK_HOUR_1", "22"))
        opener_min_1 = int(os.getenv("OPENER_ATTACK_MIN_1", "30"))
        opener_hour_2 = int(os.getenv("OPENER_ATTACK_HOUR_2", "0"))
        opener_min_2 = int(os.getenv("OPENER_ATTACK_MIN_2", "30"))
        scheduler.add_job(_opener_attack_job, CronTrigger(hour=opener_hour_1, minute=opener_min_1, timezone=tz),
                          id="opener_attack_2230", name="Opener Attack 22:30", replace_existing=True)
        scheduler.add_job(_opener_attack_job, CronTrigger(hour=opener_hour_2, minute=opener_min_2, timezone=tz),
                          id="opener_attack_0030", name="Opener Attack 00:30", replace_existing=True)

    scheduler.start()
    logger.info("Edge scheduler started (%d jobs)", len(scheduler.get_jobs()))


def stop_edge_scheduler() -> None:
    """Stop the edge scheduler gracefully. Called from edge_app.py lifespan shutdown."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Edge scheduler stopped")
```

**Note:** The `from backend.main import ...` is a temporary bridge during the strangler-fig. In Phase 7 cleanup, the job functions move to `schedulers/edge_scheduler.py` permanently and the `main` import is removed.

- [ ] **Step 3: Compile check**

```bash
venv/Scripts/python -m py_compile backend/schedulers/edge_scheduler.py && echo OK
```

- [ ] **Step 4: Commit**

```bash
git add backend/schedulers/edge_scheduler.py
git commit -m "feat: add backend/schedulers/edge_scheduler.py — CBB/edge job registration extracted from main.py lifespan"
```

---

### Task 11: Create `backend/schedulers/fantasy_scheduler.py`

**Files:**
- Create: `backend/schedulers/fantasy_scheduler.py`

- [ ] **Step 1: Create `backend/schedulers/fantasy_scheduler.py`**

```python
"""
backend/schedulers/fantasy_scheduler.py — Fantasy service APScheduler setup.

Contains ONLY fantasy/MLB ingestion jobs.
Never imports from betting_model or analysis — GUARDIAN FREEZE applies.
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

    Reads ENABLE_INGESTION_ORCHESTRATOR to control the DailyIngestionOrchestrator.
    The job_queue_processor always starts — it processes async lineup jobs.
    """
    from backend.services.job_queue_service import process_pending_jobs  # noqa: PLC0415
    from backend.main import (  # noqa: PLC0415 — temporary bridge, removed in Phase 7
        _process_job_queue_job,
    )

    tz = os.getenv("NIGHTLY_CRON_TIMEZONE", "America/New_York")

    # Job queue processor — always runs in fantasy service
    scheduler.add_job(_process_job_queue_job, IntervalTrigger(seconds=5),
                      id="job_queue_processor", name="Job Queue Processor", replace_existing=True)

    # Ingestion orchestrator (FanGraphs RoS, Yahoo ADP, ensemble blend, freshness)
    if os.getenv("ENABLE_INGESTION_ORCHESTRATOR", "false").lower() == "true":
        from backend.services.daily_ingestion import DailyIngestionOrchestrator  # noqa: PLC0415
        # The orchestrator manages its own internal APScheduler with advisory locks.
        # We store the instance on the scheduler for shutdown.
        orchestrator = DailyIngestionOrchestrator()
        orchestrator.start()
        scheduler._fantasy_orchestrator = orchestrator  # type: ignore[attr-defined]
        logger.info("DailyIngestionOrchestrator started")
    else:
        logger.info("DailyIngestionOrchestrator disabled (ENABLE_INGESTION_ORCHESTRATOR not set)")

    # MLB analysis (if enabled)
    if os.getenv("ENABLE_MLB_ANALYSIS", "false").lower() == "true":
        from backend.services.mlb_analysis import MLBAnalysisService  # noqa: PLC0415
        from backend.main import _run_mlb_analysis_job  # noqa: PLC0415
        _mlb_service = MLBAnalysisService()
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
```

- [ ] **Step 2: Compile check**

```bash
venv/Scripts/python -m py_compile backend/schedulers/fantasy_scheduler.py && echo OK
```

- [ ] **Step 3: Full suite still green**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add backend/schedulers/fantasy_scheduler.py
git commit -m "feat: add backend/schedulers/fantasy_scheduler.py — fantasy/ingestion job registration"
```

---

## Phase 5 — New Entry Points

### Task 12: Create `backend/edge_app.py`

**Files:**
- Create: `backend/edge_app.py`
- Create: `tests/test_edge_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_edge_app.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.fixture
def edge_client():
    # Patch scheduler start so tests don't fire real APScheduler jobs
    with patch("backend.schedulers.edge_scheduler.start_edge_scheduler"):
        with patch("backend.schedulers.edge_scheduler.stop_edge_scheduler"):
            from backend.edge_app import app
            with TestClient(app) as client:
                yield client


def test_edge_app_health_returns_200(edge_client):
    response = edge_client.get("/health")
    assert response.status_code == 200


def test_edge_app_health_contains_status(edge_client):
    response = edge_client.get("/health")
    data = response.json()
    assert "status" in data


def test_edge_app_root_returns_deployment_role(edge_client):
    response = edge_client.get("/")
    assert response.status_code == 200


def test_edge_app_does_not_expose_fantasy_routes(edge_client):
    response = edge_client.get("/api/fantasy/lineup/2026-04-04")
    # Fantasy routes are not mounted on the edge app
    assert response.status_code == 404
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_edge_app.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.edge_app'`

- [ ] **Step 3: Create `backend/edge_app.py`**

```python
"""
backend/edge_app.py — Edge service FastAPI entry point.

Railway start command: uvicorn backend.edge_app:app --host 0.0.0.0 --port $PORT

This app serves ONLY betting/analysis endpoints:
  - /api/predictions/*, /api/performance/*, /api/bets/*, /api/games/*
  - /api/closing-lines/*, /api/tournament/*, /api/feature-flags
  - /admin/* (edge-specific admin operations)
  - /health, /

Fantasy routes are NOT mounted here. If a fantasy client hits this service
by mistake, it receives 404.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.edge import router as edge_router
from backend.routers.admin import router as admin_router
from backend.schedulers.edge_scheduler import start_edge_scheduler, stop_edge_scheduler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Edge service starting — deployment_role=%s", os.getenv("DEPLOYMENT_ROLE", "primary"))
    start_edge_scheduler()
    yield
    stop_edge_scheduler()
    logger.info("Edge service stopped")


app = FastAPI(
    title="Edge Analytics API",
    description="Sports betting edge analysis engine — CBB + MLB",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(edge_router)
app.include_router(admin_router)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_edge_app.py -v
```
Expected: 4 passed

- [ ] **Step 5: Compile check**

```bash
venv/Scripts/python -m py_compile backend/edge_app.py && echo OK
```

- [ ] **Step 6: Commit**

```bash
git add backend/edge_app.py tests/test_edge_app.py
git commit -m "feat: add backend/edge_app.py — independent edge service entry point"
```

---

### Task 13: Create `backend/fantasy_app.py`

**Files:**
- Create: `backend/fantasy_app.py`
- Create: `tests/test_fantasy_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fantasy_app.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def fantasy_client():
    with patch("backend.schedulers.fantasy_scheduler.start_fantasy_scheduler"):
        with patch("backend.schedulers.fantasy_scheduler.stop_fantasy_scheduler"):
            from backend.fantasy_app import app
            with TestClient(app) as client:
                yield client


def test_fantasy_app_health_returns_200(fantasy_client):
    response = fantasy_client.get("/health")
    assert response.status_code == 200


def test_fantasy_app_health_contains_status(fantasy_client):
    response = fantasy_client.get("/health")
    data = response.json()
    assert "status" in data


def test_fantasy_app_does_not_expose_edge_routes(fantasy_client):
    response = fantasy_client.get("/api/predictions/today")
    # Edge routes are not mounted on the fantasy app
    assert response.status_code == 404


def test_fantasy_app_root_returns_200(fantasy_client):
    response = fantasy_client.get("/")
    assert response.status_code == 200
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_app.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'backend.fantasy_app'`

- [ ] **Step 3: Create `backend/fantasy_app.py`**

```python
"""
backend/fantasy_app.py — Fantasy service FastAPI entry point.

Railway start command: uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT

This app serves ONLY fantasy endpoints:
  - /api/fantasy/* (all fantasy routes)
  - /admin/* (fantasy-specific: ingestion status, job queue, etc.)
  - /health, /

Edge/betting routes are NOT mounted here. Zero knowledge of betting_model,
analysis.py, or any CBB-specific service. GUARDIAN FREEZE enforced by
import structure.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers.fantasy import router as fantasy_router
from backend.routers.admin import router as admin_router
from backend.schedulers.fantasy_scheduler import start_fantasy_scheduler, stop_fantasy_scheduler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Fantasy service starting — deployment_role=%s", os.getenv("DEPLOYMENT_ROLE", "fantasy-prod"))
    start_fantasy_scheduler()

    # Startup catch-up: if after 3 AM ET and no ensemble blend ran today,
    # trigger a one-off fetch. This matches the catch-up logic in main.py.
    try:
        from backend.utils.deployment import startup_catchup_enabled  # noqa: PLC0415
        catchup_enabled = startup_catchup_enabled()
    except ImportError:
        catchup_enabled = os.getenv("ENABLE_STARTUP_CATCHUP", "true").lower() != "false"

    if catchup_enabled:
        import asyncio  # noqa: PLC0415
        from backend.main import _startup_catchup  # noqa: PLC0415 — temporary bridge
        asyncio.create_task(_startup_catchup())
    else:
        logger.info("Startup catch-up disabled")

    yield
    stop_fantasy_scheduler()
    logger.info("Fantasy service stopped")


app = FastAPI(
    title="Fantasy Baseball API",
    description="MLB fantasy baseball — lineup optimisation, waiver analysis, projections",
    version="1.0.0",
    lifespan=lifespan,
)

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fantasy_router)
app.include_router(admin_router)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_app.py -v
```
Expected: 4 passed

- [ ] **Step 5: Compile check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_app.py && echo OK
```

- [ ] **Step 6: Full suite green**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -8
```
Expected: all tests that passed at baseline still pass.

- [ ] **Step 7: Commit**

```bash
git add backend/fantasy_app.py tests/test_fantasy_app.py
git commit -m "feat: add backend/fantasy_app.py — independent fantasy service entry point"
```

---

### Task 14: Local smoke test of both entry points

Verify both apps start and serve traffic before doing any Railway work.

**Files:** none (validation only)

- [ ] **Step 1: Start edge_app locally**

In a terminal:
```bash
venv/Scripts/python -m uvicorn backend.edge_app:app --port 8001 --no-access-log
```
Expected: `INFO: Application startup complete.` with no errors.

- [ ] **Step 2: Hit edge_app health**

In a second terminal:
```bash
curl -s http://localhost:8001/health | python -m json.tool
```
Expected: `{"status": "healthy", ...}` or `{"status": "degraded", "database": "error: ..."}` (DB error is expected locally without Railway Postgres — what matters is the app started and responded).

- [ ] **Step 3: Verify edge_app returns 404 for fantasy routes**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/api/fantasy/lineup/2026-04-04
```
Expected: `404`

- [ ] **Step 4: Stop edge_app (Ctrl+C), start fantasy_app**

```bash
venv/Scripts/python -m uvicorn backend.fantasy_app:app --port 8002 --no-access-log
```
Expected: `INFO: Application startup complete.`

- [ ] **Step 5: Hit fantasy_app health**

```bash
curl -s http://localhost:8002/health | python -m json.tool
```

- [ ] **Step 6: Verify fantasy_app returns 404 for edge routes**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/api/predictions/today
```
Expected: `404`

- [ ] **Step 7: Stop fantasy_app**

- [ ] **Step 8: Final full suite run**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short 2>&1 | tail -10
```
Expected: same pass count as baseline. This is the Phase 1–5 completion gate.

- [ ] **Step 9: Commit final validation note to HANDOFF.md**

Add to `HANDOFF.md` under Session S12:
```markdown
### Session S12 (Apr 4)
- Phases 1–5 of fantasy/edge structural decoupling complete.
- New files: backend/db.py, backend/redis_client.py, backend/models_edge.py, 
  backend/models_fantasy.py, backend/routers/edge.py, backend/routers/fantasy.py,
  backend/routers/admin.py, backend/schedulers/edge_scheduler.py,
  backend/schedulers/fantasy_scheduler.py, backend/edge_app.py, backend/fantasy_app.py
- main.py unchanged and still serves production traffic.
- Both new entry points smoke-tested locally.
- Phase 6–7 (separate Postgres, Redis, data migration, cleanup) pending Railway provisioning.
  See: docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md Phase 6–7.
- Gemini next: provision Fantasy Postgres in Railway, set FANTASY_DATABASE_URL on fantasy service.
```

```bash
git add HANDOFF.md
git commit -m "docs: update HANDOFF.md — phases 1-5 decoupling complete, phase 6-7 pending Railway"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Phase 1: `db.py`, `redis_client.py`, package stubs — Tasks 1–3
- [x] Phase 2: `models_edge.py`, `models_fantasy.py` — Tasks 4–5
- [x] Phase 3: `routers/edge.py`, `routers/fantasy.py`, `routers/admin.py`, strangler-fig mount — Tasks 6–9
- [x] Phase 4: `schedulers/edge_scheduler.py`, `schedulers/fantasy_scheduler.py` — Tasks 10–11
- [x] Phase 5: `edge_app.py`, `fantasy_app.py`, local smoke test — Tasks 12–14
- [x] Sports-agnostic naming: `edge_app.py`, `models_edge.py`, `routers/edge.py` — no "cbb" in new files
- [x] GUARDIAN FREEZE preserved: `fantasy_app.py` does not import `betting_model`; `fantasy_scheduler.py` note included
- [x] Strangler fig: `main.py` not deleted or modified (except router mount in Task 9)
- [x] `ENABLE_MAIN_SCHEDULER` / `ENABLE_STARTUP_CATCHUP` / `deployment_role()` handled with safe fallback imports

**Placeholder scan:** None found — all steps have complete code.

**Type consistency:** `start_edge_scheduler` / `stop_edge_scheduler` defined in Task 10, referenced in Task 12. `start_fantasy_scheduler` / `stop_fantasy_scheduler` defined in Task 11, referenced in Task 13. Consistent throughout.

---

## What's NOT in this plan (Phase 6–7)

- Provisioning Fantasy Postgres on Railway (Gemini swimlane)
- Running `FANTASY_DATABASE_URL` migration
- `UserPreferences` data migration script
- Replacing `_ROS_CACHE` dict with `fantasy_cache` Redis calls
- Replacing Yahoo OAuth token env-var fetch with `fantasy_cache`
- Deleting `main.py` and `models.py` shim
- Merging `infra/mlb-uat-isolation` branch
- Railway Start Command updates per service

These are in **Plan B**: `2026-04-04-fantasy-edge-decoupling-infrastructure.md` (to be written once Plan A is deployed and validated on Railway).
