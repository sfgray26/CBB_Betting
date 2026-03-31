# Design: Fantasy Baseball Module Performance & Data Quality Fixes

**Date:** 2026-03-31
**Author:** Claude Code (Master Architect)
**Status:** Approved — pending implementation plan
**Source:** Forensic log audit (2026-03-31)

---

## Problem Statement

Four degraded conditions identified via production log analysis:

| # | Issue | Evidence | Impact |
|---|-------|----------|--------|
| 1 | Yahoo token refreshed on every request | `Yahoo tokens refreshed and persisted to .env` in every roster/matchup log | +500-1000ms latency per request |
| 2 | 6 CSV files re-parsed on every `/api/fantasy/roster` request | `Loaded 461 Steamer batters...` repeating in logs | Visible "very slow resolution" in UI |
| 3 | ADP matched 181/551 players (32%) | `ADP matched 181/551 players` in projections_loader logs | Players missing ADP ranking, incorrect tier assignment |
| 4 | Statcast ingestion job is a stub | `_update_statcast: statcast update not yet implemented` | Fantasy edge calculations using stale pre-season baselines |

---

## Issue 1: Yahoo Token Over-Refresh

### Root Cause

`YahooFantasyClient.__init__` sets `_token_expiry = 0.0` (instance variable). Every new instance unconditionally refreshes on first `_ensure_token()` call. The endpoint handlers create a new `YahooFantasyClient()` per request.

### Design: Module-level singleton via `get_yahoo_client()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`

- Add module-level `_client: Optional[YahooFantasyClient] = None` and `_client_lock = threading.Lock()`
- Add `get_yahoo_client() -> YahooFantasyClient` factory:
  - Fast path: return `_client` if already initialized
  - Slow path: acquire lock, double-check, construct and cache
- All callers in `backend/main.py` replace `YahooFantasyClient()` with `get_yahoo_client()`

**Thread safety:** `_client_lock` guards initialization only. Subsequent calls take the fast path with no lock. Token refresh uses the existing `_token_lock` inside the shared instance.

**What does NOT change:** `_ensure_token()`, `_refresh_access_token()`, `_token_lock`, `ResilientYahooClient` subclass, circuit breaker, stale cache manager.

**Testing:** New unit test asserts `get_yahoo_client() is get_yahoo_client()` (same object returned twice).

---

## Issue 2: ProjectionsLoader CSV Re-Read

### Root Cause

`load_full_board()` in `projections_loader.py` reads all 6 CSVs on every call. The `_BOARD` sentinel in `player_board.py` caches the result module-level, but `load_full_board()` itself has no internal cache — any direct caller bypasses the sentinel. Additionally, module re-import on Railway worker restarts resets `_BOARD = None`.

### Design: `@lru_cache(maxsize=1)` on `load_full_board()`

**File:** `backend/fantasy_baseball/projections_loader.py`

- Add `@functools.lru_cache(maxsize=1)` decorator to `load_full_board()`
- The default `None` argument path is the only production call path — this is what gets cached

**File:** `backend/main.py`

- Add `POST /admin/fantasy/reload-board` endpoint:
  - Calls `load_full_board.cache_clear()` to clear the lru_cache
  - Sets `player_board._BOARD = None` to reset the module-level sentinel
  - Calls `get_board()` to re-prime both caches in one shot
  - Returns `{"status": "ok", "players": len(board)}`
  - Uses same `@require_admin` dependency as other `/admin/` endpoints

**What does NOT change:** All CSV parsing logic, z-score computation, tier assignment, `player_board._BOARD` sentinel (still useful as a secondary guard).

---

## Issue 3: ADP Name Normalization

### Root Cause

`_make_player_id()` normalizes names to a lowercase underscore ID, but misses:
1. Name suffixes: `"Ronald Acuña Jr."` → `ronald_acuna_jr` vs ADP `"Ronald Acuna"` → `ronald_acuna`
2. Last-name-first format: some ADP sources export `"Ohtani, Shohei"` → `ohtani_shohei` vs `"shohei_ohtani"`
3. Missing accented char substitutions: `ê`, `ü`, `â` not handled

`_apply_adp()` has a partial fallback but it's fragile (`adp_id.replace("_", " ") in name_lower`).

### Design: Extend normalization + last-name fallback

**File:** `backend/fantasy_baseball/projections_loader.py`

**`_make_player_id()` changes:**
- Detect and flip last-name-first: if `","` in name, split on `","` and rejoin as `"First Last"`
- Strip suffixes before normalizing: remove ` jr`, ` sr`, ` ii`, ` iii`, ` iv` (case-insensitive, word-boundary aware)
- Add missing accented char substitutions: `ê→e`, `ü→u`, `â→a`, `ç→c`, `ï→i`

**`_apply_adp()` second-pass fallback:**
- When exact ID match fails, attempt match by last name + first initial
  - Extract from player ID: last token of underscore-split as last name, first char of first token as initial
  - e.g. `"shohei_ohtani"` → last=`"ohtani"`, initial=`"s"` → try `"s_ohtani"` key in a pre-built lookup
- Log match source (`exact` vs `initial_fallback`) at DEBUG level

**Expected outcome:** Match rate 32% → 80%+ based on common FanGraphs vs FantasyPros name format divergence.

**What does NOT change:** ADP CSV loading, player board structure, z-score/tier logic.

---

## Issue 4: Statcast Wire-Up

### Root Cause

`DailyIngestionOrchestrator._update_statcast()` in `daily_ingestion.py` (lines 238-245) is a 4-line stub that logs "not yet implemented" and returns `status: skipped`. The full implementation — `StatcastIngestionAgent`, `BayesianProjectionUpdater`, `run_daily_ingestion()` — already exists in `backend/fantasy_baseball/statcast_ingestion.py` and is untested in production.

### Design: Wire stub to existing implementation

**File:** `backend/services/daily_ingestion.py`

Replace the stub body with:
```python
from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion
result = await asyncio.to_thread(run_daily_ingestion)
self._record_job_run("statcast", result.get("status", "unknown"))
return result
```

**Advisory lock:** Already present in the stub (`_with_advisory_lock(LOCK_IDS["statcast"], _run)`) — preserved unchanged.

**Error handling:** `run_daily_ingestion()` returns a result dict with `success: bool`. If `success` is False, log the error and record `"failed"` via `_record_job_run`. No exception propagation — advisory lock must always release.

**Note on `datetime.utcnow()`:** `statcast_ingestion.py` uses `datetime.utcnow()` in two places (lines 447, 690, 706). Per project rules, these must be changed to `datetime.now(ZoneInfo("America/New_York"))` before wiring.

**What does NOT change:** `StatcastIngestionAgent`, `BayesianProjectionUpdater`, `run_daily_ingestion()` — no modifications to the ingestion logic itself. Advisory lock pattern, `_record_job_run`, job schedule (every 6h) — all preserved.

**Testing:** New test in `tests/` mocks `run_daily_ingestion` and asserts the stub calls it and records job status correctly.

---

## Implementation Order

Execute sequentially. Each fix is independent — no cross-dependencies.

1. Issue 1: Yahoo singleton (`yahoo_client_resilient.py` + `main.py` callers)
2. Issue 2: `lru_cache` on `load_full_board` + `/admin/fantasy/reload-board` endpoint
3. Issue 3: ADP name normalization (`projections_loader.py`)
4. Issue 4: Statcast wire-up (`daily_ingestion.py` + `datetime.utcnow` fixes in `statcast_ingestion.py`)

---

## Guardrails

- Do NOT modify Kelly math in `betting_model.py` (EMAC-068 frozen until Apr 7)
- Do NOT call BDL `/ncaab/v1/` endpoints
- Do NOT touch `dashboard/` (Streamlit retired)
- All new test files go in `tests/` only
- Use `datetime.now(ZoneInfo("America/New_York"))` — never `datetime.utcnow()`
- Verify with `venv/Scripts/python -m py_compile` before committing each file
