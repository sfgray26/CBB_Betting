# Fantasy Baseball Performance & Data Quality Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four production-degraded conditions in the fantasy baseball module: Yahoo token over-refresh, CSV re-read on every request, 32% ADP match rate, and Statcast ingestion stub.

**Architecture:** Four independent fixes applied sequentially. No cross-dependencies. Each fix is self-contained and can be tested and committed independently. Spec: `docs/superpowers/specs/2026-03-31-fantasy-baseball-perf-fixes-design.md`

**Tech Stack:** Python 3.11, FastAPI, `functools.lru_cache`, `threading.Lock`, `ZoneInfo`, pytest

---

## File Map

| File | Change |
|------|--------|
| `backend/fantasy_baseball/yahoo_client_resilient.py` | Add `get_yahoo_client()` + `get_resilient_yahoo_client()` singletons |
| `backend/main.py` | Replace all `YahooFantasyClient()` / `ResilientYahooClient()` constructor calls; add `/admin/fantasy/reload-board` endpoint |
| `backend/fantasy_baseball/projections_loader.py` | Add `@lru_cache(maxsize=1)` to `load_full_board()`; extend `_make_player_id()`; improve `_apply_adp()` fallback |
| `backend/services/daily_ingestion.py` | Wire `_update_statcast()` stub to `run_daily_ingestion()` |
| `backend/fantasy_baseball/statcast_ingestion.py` | Replace 3× `datetime.utcnow()` with `datetime.now(ZoneInfo("America/New_York"))` |
| `tests/test_fantasy_fixes.py` | New test file covering all four fixes |

---

## Task 1: Yahoo Client Singleton

**Files:**
- Modify: `backend/fantasy_baseball/yahoo_client_resilient.py` (end of file, after `ResilientYahooClient`)
- Modify: `backend/main.py` (13× `YahooFantasyClient()`, 1× `ResilientYahooClient()`)
- Test: `tests/test_fantasy_fixes.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_fantasy_fixes.py`:

```python
"""Tests for fantasy baseball performance and data quality fixes."""
import threading
import pytest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Task 1: Yahoo client singleton
# ---------------------------------------------------------------------------

def test_get_yahoo_client_returns_same_instance():
    """get_yahoo_client() must return the identical object on repeated calls."""
    # Reset module singleton so test is isolated
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._client = None

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "test_id",
        "YAHOO_CLIENT_SECRET": "test_secret",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_REFRESH_TOKEN": "test_refresh",
        "YAHOO_ACCESS_TOKEN": "test_access",
    }):
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client
        c1 = get_yahoo_client()
        c2 = get_yahoo_client()
        assert c1 is c2


def test_get_resilient_yahoo_client_returns_same_instance():
    """get_resilient_yahoo_client() must return the identical object on repeated calls."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._resilient_client = None

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "test_id",
        "YAHOO_CLIENT_SECRET": "test_secret",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_REFRESH_TOKEN": "test_refresh",
        "YAHOO_ACCESS_TOKEN": "test_access",
    }):
        from backend.fantasy_baseball.yahoo_client_resilient import get_resilient_yahoo_client
        c1 = get_resilient_yahoo_client()
        c2 = get_resilient_yahoo_client()
        assert c1 is c2


def test_singleton_thread_safety():
    """Concurrent calls must produce the same instance (no double-init)."""
    import backend.fantasy_baseball.yahoo_client_resilient as mod
    mod._client = None

    results = []

    with patch.dict("os.environ", {
        "YAHOO_CLIENT_ID": "test_id",
        "YAHOO_CLIENT_SECRET": "test_secret",
        "YAHOO_LEAGUE_ID": "12345",
        "YAHOO_REFRESH_TOKEN": "test_refresh",
        "YAHOO_ACCESS_TOKEN": "test_access",
    }):
        from backend.fantasy_baseball.yahoo_client_resilient import get_yahoo_client

        def grab():
            results.append(get_yahoo_client())

        threads = [threading.Thread(target=grab) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert len(set(id(r) for r in results)) == 1
```

- [ ] **Step 1.2: Run test — confirm FAIL**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py::test_get_yahoo_client_returns_same_instance -v
```

Expected: `FAILED` — `ImportError: cannot import name 'get_yahoo_client'`

- [ ] **Step 1.3: Add singleton factories to `yahoo_client_resilient.py`**

At the very end of `backend/fantasy_baseball/yahoo_client_resilient.py`, after the `ResilientYahooClient` class definition, add:

```python
# ---------------------------------------------------------------------------
# Module-level singletons — use these instead of constructing directly
# ---------------------------------------------------------------------------

_client: "Optional[YahooFantasyClient]" = None
_client_lock = threading.Lock()

_resilient_client: "Optional[ResilientYahooClient]" = None
_resilient_client_lock = threading.Lock()


def get_yahoo_client() -> "YahooFantasyClient":
    """
    Return the process-level YahooFantasyClient singleton.

    Thread-safe via double-checked locking. Token refresh is handled
    internally by _ensure_token() — callers never need to refresh manually.
    """
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = YahooFantasyClient()
    return _client


def get_resilient_yahoo_client() -> "ResilientYahooClient":
    """
    Return the process-level ResilientYahooClient singleton.

    Use this for endpoints that need circuit-breaker + stale-cache behaviour.
    """
    global _resilient_client
    if _resilient_client is not None:
        return _resilient_client
    with _resilient_client_lock:
        if _resilient_client is None:
            _resilient_client = ResilientYahooClient()
    return _resilient_client
```

- [ ] **Step 1.4: Run tests — confirm PASS**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py -v
```

Expected: 3 PASS

- [ ] **Step 1.5: Replace all constructor calls in `main.py`**

Find every `YahooFantasyClient()` instantiation in `backend/main.py`. There are 13. For each one, replace the pattern:

```python
# BEFORE (example pattern — repeated 13 times with minor variation):
try:
    client = YahooFantasyClient()
except YahooAuthError as exc:
    raise HTTPException(status_code=401, detail=f"Yahoo auth not configured: {exc}")
```

```python
# AFTER:
try:
    client = get_yahoo_client()
except YahooAuthError as exc:
    raise HTTPException(status_code=401, detail=f"Yahoo auth not configured: {exc}")
```

Also replace the one `ResilientYahooClient()` call (at the lineup-validate endpoint):

```python
# BEFORE:
client = ResilientYahooClient()
```

```python
# AFTER:
client = get_resilient_yahoo_client()
```

Confirm the imports at the top of main.py include `get_yahoo_client` and `get_resilient_yahoo_client`. Find the existing import line:

```python
from backend.fantasy_baseball.yahoo_client_resilient import (
    YahooFantasyClient,
    ...
)
```

Add `get_yahoo_client, get_resilient_yahoo_client` to that import block.

- [ ] **Step 1.6: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
venv/Scripts/python -m py_compile backend/main.py
```

Expected: no output (clean compile)

- [ ] **Step 1.7: Run full test suite**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py -q --tb=short
```

Expected: all PASS

- [ ] **Step 1.8: Commit**

```bash
git add backend/fantasy_baseball/yahoo_client_resilient.py backend/main.py tests/test_fantasy_fixes.py
git commit -m "perf: yahoo client singleton — eliminate per-request token refresh"
```

---

## Task 2: ProjectionsLoader `lru_cache`

**Files:**
- Modify: `backend/fantasy_baseball/projections_loader.py` (add decorator to `load_full_board`)
- Modify: `backend/main.py` (add `/admin/fantasy/reload-board` endpoint)
- Test: `tests/test_fantasy_fixes.py` (append)

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_fantasy_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 2: ProjectionsLoader lru_cache
# ---------------------------------------------------------------------------

def test_load_full_board_cached():
    """load_full_board() must return the same list object on repeated calls."""
    from backend.fantasy_baseball.projections_loader import load_full_board
    load_full_board.cache_clear()

    with patch("backend.fantasy_baseball.projections_loader.load_steamer_batting", return_value=[]):
        with patch("backend.fantasy_baseball.projections_loader.load_steamer_pitching", return_value=[]):
            with patch("pathlib.Path.exists", return_value=True):
                r1 = load_full_board()
                r2 = load_full_board()
                # Must be the exact same object (cached), not just equal
                assert r1 is r2


def test_load_full_board_cache_clear_triggers_reload():
    """cache_clear() must cause the next call to re-read CSVs."""
    from backend.fantasy_baseball.projections_loader import load_full_board
    load_full_board.cache_clear()

    call_count = {"n": 0}
    original_batting = __import__(
        "backend.fantasy_baseball.projections_loader",
        fromlist=["load_steamer_batting"]
    ).load_steamer_batting

    with patch("backend.fantasy_baseball.projections_loader.load_steamer_batting") as mock_bat:
        with patch("backend.fantasy_baseball.projections_loader.load_steamer_pitching", return_value=[]):
            with patch("pathlib.Path.exists", return_value=True):
                mock_bat.return_value = []
                load_full_board()
                load_full_board.cache_clear()
                load_full_board()
                assert mock_bat.call_count == 2
```

- [ ] **Step 2.2: Run tests — confirm FAIL**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py::test_load_full_board_cached -v
```

Expected: `FAILED` — `AttributeError: 'function' object has no attribute 'cache_clear'`

- [ ] **Step 2.3: Add `@lru_cache` to `load_full_board()`**

In `backend/fantasy_baseball/projections_loader.py`:

At the top, add to imports:
```python
from functools import lru_cache
```

Then decorate `load_full_board`:
```python
@lru_cache(maxsize=1)
def load_full_board(data_dir: Optional[Path] = None) -> Optional[list[dict]]:
```

No other changes to the function body.

- [ ] **Step 2.4: Run tests — confirm PASS**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py::test_load_full_board_cached tests/test_fantasy_fixes.py::test_load_full_board_cache_clear_triggers_reload -v
```

Expected: 2 PASS

- [ ] **Step 2.5: Add `/admin/fantasy/reload-board` endpoint to `main.py`**

Find the block of `/admin/fantasy/` endpoints in `backend/main.py` and add after the last one:

```python
@app.post("/admin/fantasy/reload-board", dependencies=[Depends(verify_admin_api_key)])
async def admin_reload_fantasy_board():
    """
    Force a fresh read of projection CSVs (data/projections/*.csv).
    Call this after dropping new Steamer/ZiPS exports into data/projections/.
    """
    from backend.fantasy_baseball.projections_loader import load_full_board
    from backend.fantasy_baseball import player_board

    load_full_board.cache_clear()
    player_board._BOARD = None  # Reset module-level sentinel

    board = player_board.get_board()
    return {"status": "ok", "players_loaded": len(board) if board else 0}
```

- [ ] **Step 2.6: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/projections_loader.py
venv/Scripts/python -m py_compile backend/main.py
```

Expected: clean

- [ ] **Step 2.7: Run tests**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py -q --tb=short
```

Expected: all PASS

- [ ] **Step 2.8: Commit**

```bash
git add backend/fantasy_baseball/projections_loader.py backend/main.py tests/test_fantasy_fixes.py
git commit -m "perf: lru_cache on load_full_board + admin reload endpoint"
```

---

## Task 3: ADP Name Normalization

**Files:**
- Modify: `backend/fantasy_baseball/projections_loader.py` (`_make_player_id`, `_apply_adp`)
- Test: `tests/test_fantasy_fixes.py` (append)

- [ ] **Step 3.1: Write the failing tests**

Append to `tests/test_fantasy_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 3: ADP name normalization
# ---------------------------------------------------------------------------

def test_make_player_id_strips_suffix():
    from backend.fantasy_baseball.projections_loader import _make_player_id
    assert _make_player_id("Ronald Acuña Jr.") == _make_player_id("Ronald Acuna")
    assert _make_player_id("Ken Griffey Jr.") == _make_player_id("Ken Griffey")
    assert _make_player_id("Cal Ripken Jr.") == _make_player_id("Cal Ripken")


def test_make_player_id_handles_last_name_first():
    from backend.fantasy_baseball.projections_loader import _make_player_id
    assert _make_player_id("Ohtani, Shohei") == _make_player_id("Shohei Ohtani")
    assert _make_player_id("Betts, Mookie") == _make_player_id("Mookie Betts")


def test_make_player_id_handles_extra_accents():
    from backend.fantasy_baseball.projections_loader import _make_player_id
    # ê, ü, â, ç, ï should all normalize to ASCII
    assert _make_player_id("José Ramîrez") == _make_player_id("Jose Ramirez")


def test_apply_adp_last_name_initial_fallback():
    """_apply_adp must match when exact ID fails but last+initial matches.

    Some ADP sources (e.g. FantasyPros) export abbreviated first names like
    "A. Judge" while the player board has "Aaron Judge". After normalization
    these produce different IDs ("a_judge" vs "aaron_judge"), but the initial
    fallback bridges them.
    """
    from backend.fantasy_baseball.projections_loader import _apply_adp, _make_player_id

    # ADP CSV has abbreviated first name — normalizes to "a_judge"
    adp_map = {"a_judge": 4.0}

    # Player board has full name — normalizes to "aaron_judge"
    players = [{
        "id": _make_player_id("Aaron Judge"),
        "name": "Aaron Judge",
        "adp": 999.0,
    }]

    _apply_adp(players, adp_map)
    assert players[0]["adp"] == 4.0
```

- [ ] **Step 3.2: Run tests — confirm FAIL**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py::test_make_player_id_strips_suffix tests/test_fantasy_fixes.py::test_make_player_id_handles_last_name_first -v
```

Expected: FAIL

- [ ] **Step 3.3: Rewrite `_make_player_id` in `projections_loader.py`**

Replace the existing `_make_player_id` function:

```python
import re as _re

# Suffixes to strip before normalizing (word-boundary, case-insensitive)
_SUFFIX_RE = _re.compile(r'\b(jr|sr|ii|iii|iv)\.?\s*$', _re.IGNORECASE)


def _make_player_id(name: str) -> str:
    """Normalize a player name to a stable ASCII identifier.

    Handles:
    - Accented characters (é, á, ó, ú, í, ñ, â, ê, ü, ç, ï, î)
    - Name suffixes (Jr., Sr., II, III, IV)
    - Last-name-first format ("Ohtani, Shohei" -> "shohei_ohtani")
    """
    if not name:
        return ""
    name = name.strip()

    # Flip last-name-first format
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        name = f"{parts[1]} {parts[0]}"

    # Strip generational suffixes
    name = _SUFFIX_RE.sub("", name).strip()

    # Normalize accented characters
    name = (name
            .replace("é", "e").replace("è", "e").replace("ê", "e")
            .replace("á", "a").replace("à", "a").replace("â", "a")
            .replace("ó", "o").replace("ò", "o").replace("ô", "o")
            .replace("ú", "u").replace("ù", "u").replace("û", "u").replace("ü", "u")
            .replace("í", "i").replace("ì", "i").replace("î", "i").replace("ï", "i")
            .replace("ñ", "n").replace("ç", "c"))

    return (name.lower()
            .replace(" ", "_").replace(".", "").replace("'", "")
            .replace(",", "").replace("-", "_"))
```

- [ ] **Step 3.4: Update `_apply_adp` to add last-name + initial fallback**

Replace the existing `_apply_adp` function:

```python
def _apply_adp(players: list[dict], adp_map: dict[str, float]) -> None:
    """Merge ADP data into player list in-place.

    Match strategy (in order):
    1. Exact normalized ID match
    2. Last name + first initial match (handles suffix/accent divergence)
    """
    # Pre-build last_initial -> adp_id map for fallback lookups
    # Key: "{first_initial}_{last_name}", Value: adp_id
    initial_map: dict[str, str] = {}
    for adp_id in adp_map:
        parts = adp_id.split("_")
        if len(parts) >= 2:
            first_initial = parts[0][0] if parts[0] else ""
            last_name = parts[-1]
            key = f"{first_initial}_{last_name}"
            # Don't overwrite — first entry wins (avoids false positives on common surnames)
            if key not in initial_map:
                initial_map[key] = adp_id

    matched_exact = 0
    matched_fallback = 0

    for p in players:
        pid = p["id"]

        # Pass 1: exact match
        if pid in adp_map:
            p["adp"] = adp_map[pid]
            matched_exact += 1
            continue

        # Pass 2: last name + first initial fallback
        parts = pid.split("_")
        if len(parts) >= 2:
            first_initial = parts[0][0] if parts[0] else ""
            last_name = parts[-1]
            key = f"{first_initial}_{last_name}"
            if key in initial_map:
                p["adp"] = adp_map[initial_map[key]]
                matched_fallback += 1

    total = matched_exact + matched_fallback
    logger.info(
        f"ADP matched {total}/{len(players)} players "
        f"(exact={matched_exact}, fallback={matched_fallback})"
    )
```

- [ ] **Step 3.5: Run tests — confirm PASS**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py -v
```

Expected: all PASS

- [ ] **Step 3.6: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/projections_loader.py
```

Expected: clean

- [ ] **Step 3.7: Commit**

```bash
git add backend/fantasy_baseball/projections_loader.py tests/test_fantasy_fixes.py
git commit -m "fix: ADP name normalization — suffixes, last-name-first, extended accents"
```

---

## Task 4: Statcast Wire-Up

**Files:**
- Modify: `backend/fantasy_baseball/statcast_ingestion.py` (fix 3× `datetime.utcnow()`)
- Modify: `backend/services/daily_ingestion.py` (replace stub)
- Test: `tests/test_fantasy_fixes.py` (append)

- [ ] **Step 4.1: Write the failing test**

Append to `tests/test_fantasy_fixes.py`:

```python
# ---------------------------------------------------------------------------
# Task 4: Statcast wire-up
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_update_statcast_calls_run_daily_ingestion():
    """_update_statcast must delegate to run_daily_ingestion, not return 'skipped'."""
    from backend.services.daily_ingestion import DailyIngestionOrchestrator

    mock_result = {
        "success": True,
        "date": "2026-03-30",
        "records_processed": 42,
        "projections_updated": 10,
    }

    with patch(
        "backend.services.daily_ingestion.run_daily_ingestion",
        return_value=mock_result,
    ) as mock_ingest:
        orchestrator = DailyIngestionOrchestrator.__new__(DailyIngestionOrchestrator)
        orchestrator._job_status = {}
        orchestrator._scheduler = MagicMock()

        result = await orchestrator._update_statcast()

        mock_ingest.assert_called_once()
        assert result["status"] != "skipped"
        assert result.get("records_processed") == 42


@pytest.mark.asyncio
async def test_update_statcast_records_failed_on_exception():
    """_update_statcast must record 'failed' if run_daily_ingestion raises."""
    from backend.services.daily_ingestion import DailyIngestionOrchestrator

    with patch(
        "backend.services.daily_ingestion.run_daily_ingestion",
        side_effect=RuntimeError("Baseball Savant timeout"),
    ):
        orchestrator = DailyIngestionOrchestrator.__new__(DailyIngestionOrchestrator)
        orchestrator._job_status = {}
        orchestrator._scheduler = MagicMock()

        result = await orchestrator._update_statcast()
        assert result["status"] == "failed"
```

- [ ] **Step 4.2: Run tests — confirm FAIL**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py::test_update_statcast_calls_run_daily_ingestion -v
```

Expected: `FAILED` — result status is `"skipped"` (stub still in place)

- [ ] **Step 4.3: Fix `datetime.utcnow()` in `statcast_ingestion.py`**

In `backend/fantasy_baseball/statcast_ingestion.py`, add to imports:

```python
from zoneinfo import ZoneInfo
```

Then replace all three occurrences of `datetime.utcnow()`:

Line ~447 (inside `store_performances`):
```python
# BEFORE:
created_at=datetime.utcnow()
# AFTER:
created_at=datetime.now(ZoneInfo("America/New_York"))
```

Line ~690 (inside `_store_updated_projection`, update branch):
```python
# BEFORE:
existing.updated_at = datetime.utcnow()
# AFTER:
existing.updated_at = datetime.now(ZoneInfo("America/New_York"))
```

Line ~706 (inside `_store_updated_projection`, create branch):
```python
# BEFORE:
updated_at=datetime.utcnow(),
# AFTER:
updated_at=datetime.now(ZoneInfo("America/New_York")),
```

- [ ] **Step 4.4: Replace the stub in `daily_ingestion.py`**

In `backend/services/daily_ingestion.py`, add to imports at the top of the file:

```python
from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion
```

Then replace the `_update_statcast` method (currently lines ~238-245) with:

```python
async def _update_statcast(self) -> dict:
    """Daily Statcast enrichment — fetches yesterday's data and runs Bayesian projection updates."""
    async def _run():
        try:
            result = await asyncio.to_thread(run_daily_ingestion)
            status = "ok" if result.get("success") else "failed"
            if not result.get("success"):
                logger.error(
                    "_update_statcast: ingestion reported failure — %s",
                    result.get("error", "unknown error"),
                )
            self._record_job_run("statcast", status)
            return result
        except Exception as exc:
            logger.exception("_update_statcast: unhandled error — %s", exc)
            self._record_job_run("statcast", "failed")
            return {"status": "failed", "records": 0, "error": str(exc)}

    return await _with_advisory_lock(LOCK_IDS["statcast"], _run)
```

Also verify `import asyncio` is present at the top of `daily_ingestion.py` (it should already be there).

- [ ] **Step 4.5: Run tests — confirm PASS**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py -q --tb=short
```

Expected: all PASS

- [ ] **Step 4.6: Syntax check both files**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/statcast_ingestion.py
venv/Scripts/python -m py_compile backend/services/daily_ingestion.py
```

Expected: clean

- [ ] **Step 4.7: Run targeted test suite**

```bash
venv/Scripts/python -m pytest tests/test_fantasy_fixes.py tests/test_mlb_analysis.py -q --tb=short
```

Expected: all PASS

- [ ] **Step 4.8: Commit**

```bash
git add backend/fantasy_baseball/statcast_ingestion.py backend/services/daily_ingestion.py tests/test_fantasy_fixes.py
git commit -m "feat: wire Statcast ingestion — replace daily_ingestion stub with StatcastIngestionAgent"
```

---

## Task 5: HANDOFF.md Update

- [ ] **Step 5.1: Update HANDOFF.md**

In `HANDOFF.md` §3 Technical State, update these rows:

```markdown
| **Yahoo token over-refresh** | ✅ FIXED (Mar 31) | Singleton via `get_yahoo_client()` — token refreshed once per process, not per request |
| **ProjectionsLoader CSV re-read** | ✅ FIXED (Mar 31) | `@lru_cache(maxsize=1)` on `load_full_board()`; reload via `POST /admin/fantasy/reload-board` |
| **ADP match rate 32%** | ✅ FIXED (Mar 31) | `_make_player_id` strips suffixes/flips last-name-first; `_apply_adp` adds initial fallback. Expect 80%+ match |
| **Statcast ingestion stub** | ✅ FIXED (Mar 31) | `_update_statcast` now calls `run_daily_ingestion()` from `statcast_ingestion.py`; Bayesian updates live |
```

Remove item #4 from §5 Next Session Roadmap (Statcast freshness — now complete).

- [ ] **Step 5.2: Commit HANDOFF**

```bash
git add HANDOFF.md
git commit -m "docs: update HANDOFF — mark four fantasy fixes complete"
```

---

## Final Verification

- [ ] **Run full test suite**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: all previously passing tests still pass + new tests in `test_fantasy_fixes.py` pass.
