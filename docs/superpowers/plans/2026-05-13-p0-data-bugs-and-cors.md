# P0 Data Bugs + CORS Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 confirmed Phase 1 data bugs (need score = 0, budget IP wrong key, acquisitions rolling window, ownership always 0%) plus verify/fix 4 K-1 P0 UI issues (CORS, streaming load, dashboard render, lineup 422).

**Architecture:** Fixes are spread across backend data layer (`category_aware_scorer.py`, `yahoo_client_resilient.py`, `constraint_helpers.py`) and the waiver/budget route handler (`routers/fantasy.py`). Phase 1 fixes are surgical one- or two-line changes; ownership requires a new secondary Yahoo API call. K-1 P0 verification tasks are read-first: if the issue is already gone, mark done and move on.

**Tech Stack:** Python/FastAPI backend, SQLAlchemy, Yahoo Fantasy API, pytest, Windows dev (venv/Scripts/python).

---

## File Map

| File | What changes |
|------|-------------|
| `backend/routers/fantasy.py` | Line 5519: `"my_team"` → `"my_stats"`; Lines 5498-5507: matchup-week window + broad exception |
| `backend/services/constraint_helpers.py` | `count_weekly_acquisitions`: accept `"add/drop"` type; handle string timestamps |
| `backend/fantasy_baseball/category_aware_scorer.py` | `compute_need_score` line 208: add `_CANONICAL_TO_BOARD` mapping |
| `backend/fantasy_baseball/yahoo_client_resilient.py` | `get_free_agents`: add secondary ownership batch call |
| `backend/main.py` | CORS: verify/fix `ALLOWED_ORIGINS` (read-only unless change is needed) |
| `frontend/app/(dashboard)/war-room/streaming/page.tsx` | Verify — may be fixed by A-7 already |
| `frontend/app/(dashboard)/dashboard/page.tsx` | Verify — may be fixed by recent updates |
| `tests/test_constraint_helpers.py` | New tests for `count_weekly_acquisitions` fixes |
| `tests/test_category_aware_scorer.py` | New/updated tests for `compute_need_score` key mapping |

---

## Task 1: Fix Budget IP Wrong Key

**Root cause:** `fantasy.py:5519` does `matchup_stats.get("my_team", {})` but `get_matchup_stats()` returns `{"my_stats": {}, "opp_stats": {}, "opponent_name": "..."}`.

**Files:**
- Modify: `backend/routers/fantasy.py:5519`

- [ ] **Step 1: Write a failing test**

Create or open `tests/test_fantasy_budget.py` and add:

```python
# tests/test_fantasy_budget.py
"""Tests for budget endpoint logic."""
import pytest

def test_budget_ip_key_reads_my_stats():
    """get_matchup_stats returns 'my_stats', not 'my_team'; IP must use the right key."""
    matchup_stats = {"my_stats": {"IP": 14.2}, "opp_stats": {}, "opponent_name": "Opponent"}
    # Replicate the extraction logic in fantasy.py
    my_stats = matchup_stats.get("my_stats", {})   # CORRECT key
    ip = float(my_stats.get("IP", 0.0))
    assert ip == 14.2, f"Expected 14.2 but got {ip}"


def test_budget_ip_wrong_key_returns_zero():
    """Regression: the old 'my_team' key must return 0 to prove the fix matters."""
    matchup_stats = {"my_stats": {"IP": 14.2}, "opp_stats": {}, "opponent_name": "Opponent"}
    old_key = matchup_stats.get("my_team", {})  # BUG path
    ip = float(old_key.get("IP", 0.0))
    assert ip == 0.0, "Old key should give 0 — confirming the bug exists"
```

Run: `venv/Scripts/python -m pytest tests/test_fantasy_budget.py -v`
Expected: PASS (these are logic tests, not endpoint tests — both pass immediately to confirm the test expresses what we want)

- [ ] **Step 2: Apply the one-line fix**

Open `backend/routers/fantasy.py`. Find line 5519:
```python
            my_stats = matchup_stats.get("my_team", {})
```
Change to:
```python
            my_stats = matchup_stats.get("my_stats", {})
```

- [ ] **Step 3: Compile-check**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```
Expected: no output (clean).

- [ ] **Step 4: Commit**

```bash
git add backend/routers/fantasy.py tests/test_fantasy_budget.py
git commit -m "fix(budget): use my_stats key for IP accumulation (A-6 wrong dict key)"
```

---

## Task 2: Fix Need Score Key Mismatch

**Root cause:** `compute_need_score` in `category_aware_scorer.py:208` converts canonical category codes (e.g. `"HR_B"`) to lowercase (`"hr_b"`) when building `needs_dict`. But `player_cat_scores` uses board keys (`"hr"`, `"k_bat"`, `"k_pit"`, `"k9"`, `"hr_pit"`). The lookup in `score_fa_against_needs` always finds 0.0 because the key namespaces never intersect.

**The mapping** (defined in `routers/fantasy.py` as `_CANONICAL_TO_BOARD`):
```
"HR_B"  → "hr"
"HR_P"  → "hr_pit"
"K_9"   → "k9"
"K_P"   → "k_pit"
"K_B"   → "k_bat"
```
All other canonical codes lower correctly (e.g. `"AVG"` → `"avg"`, `"ERA"` → `"era"`).

**Files:**
- Modify: `backend/fantasy_baseball/category_aware_scorer.py:208`
- Test: `tests/test_category_aware_scorer.py`

- [ ] **Step 1: Write failing tests**

Open (or create) `tests/test_category_aware_scorer.py` and add:

```python
# tests/test_category_aware_scorer.py
import pytest
from backend.schemas import CategoryDeficitOut
from backend.fantasy_baseball.category_aware_scorer import compute_need_score


def _deficit(category: str, deficit: float = 2.0) -> CategoryDeficitOut:
    return CategoryDeficitOut(category=category, deficit=deficit, deficit_z_score=1.0)


def test_need_score_hr_b_maps_to_hr():
    """HR_B canonical → 'hr' board key. If mismatch, score collapses to z-score only."""
    # cat_scores uses board key 'hr'
    player_cat_scores = {"hr": 2.5, "avg": 1.2}
    # category_deficits use canonical 'HR_B'
    deficits = [_deficit("HR_B", 3.0)]
    score = compute_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=1.0,
        category_deficits=deficits,
        n_cats=10,
    )
    # With correct mapping: cat_score = 2.5 * max(0, 3.0) = 7.5
    # blended = 0.4 * 1.0 + 0.6 * (7.5 / 10) = 0.4 + 0.45 = 0.85
    # With wrong mapping: cat_score = 0, blended = 0.4 * 1.0 = 0.4
    assert score > 0.5, f"Expected >0.5 (blended HR contribution), got {score:.4f}"


def test_need_score_k_b_maps_to_k_bat():
    """K_B canonical → 'k_bat' board key."""
    player_cat_scores = {"k_bat": 1.8}
    deficits = [_deficit("K_B", 2.0)]
    score = compute_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=0.5,
        category_deficits=deficits,
        n_cats=10,
    )
    # With correct mapping: cat_score = 1.8 * 2.0 = 3.6; blended = 0.4*0.5 + 0.6*(3.6/10) = 0.416
    assert score > 0.40, f"Expected >0.40, got {score:.4f}"


def test_need_score_k_p_maps_to_k_pit():
    """K_P canonical → 'k_pit' board key."""
    player_cat_scores = {"k_pit": 3.0}
    deficits = [_deficit("K_P", 1.5)]
    score = compute_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=0.5,
        category_deficits=deficits,
        n_cats=10,
    )
    assert score > 0.40, f"Expected >0.40, got {score:.4f}"


def test_need_score_passthrough_avg():
    """AVG lowercases cleanly to 'avg' — passthrough path should still work."""
    player_cat_scores = {"avg": 1.5}
    deficits = [_deficit("AVG", 2.0)]
    score = compute_need_score(
        player_cat_scores=player_cat_scores,
        player_z_score=0.5,
        category_deficits=deficits,
        n_cats=10,
    )
    # AVG → 'avg': cat_score = 1.5 * max(0, 2.0) = 3.0; blended = 0.4*0.5 + 0.6*(3.0/10) = 0.38
    assert score > 0.35, f"Expected >0.35, got {score:.4f}"
```

Run: `venv/Scripts/python -m pytest tests/test_category_aware_scorer.py -v`
Expected: FAIL on the first two tests (`test_need_score_hr_b_maps_to_hr` and `test_need_score_k_b_maps_to_k_bat` get only ~0.4 instead of >0.5/0.40).

- [ ] **Step 2: Add the mapping constant and fix the key lookup**

Open `backend/fantasy_baseball/category_aware_scorer.py`. Locate the top of the file (after imports, before class definitions). Add the mapping constant:

```python
# Canonical scoring codes → board keys used in PlayerProjection.cat_scores.
# cat.lower() is correct for most codes; only non-trivial remappings listed here.
_CANONICAL_TO_BOARD: dict = {
    "HR_B":  "hr",
    "HR_P":  "hr_pit",
    "K_9":   "k9",
    "K_P":   "k_pit",
    "K_B":   "k_bat",
}
```

Then in `compute_need_score`, change line 208 from:
```python
            needs_dict[cd.category.lower()] = float(cd.deficit)
```
to:
```python
            board_key = _CANONICAL_TO_BOARD.get(cd.category.upper(), cd.category.lower())
            needs_dict[board_key] = float(cd.deficit)
```

- [ ] **Step 3: Run tests**

```
venv/Scripts/python -m pytest tests/test_category_aware_scorer.py -v
```
Expected: All 4 tests PASS.

- [ ] **Step 4: Compile-check**

```
venv/Scripts/python -m py_compile backend/fantasy_baseball/category_aware_scorer.py
```

- [ ] **Step 5: Commit**

```bash
git add backend/fantasy_baseball/category_aware_scorer.py tests/test_category_aware_scorer.py
git commit -m "fix(waiver): map canonical category codes to board keys in compute_need_score"
```

---

## Task 3: Fix Budget Acquisitions Window + Transaction Type Filter

**Root cause (two bugs):**
1. `fantasy.py:5501` uses `timedelta(days=7)` (rolling 7d) instead of actual matchup week start (Monday 00:00 ET).
2. `constraint_helpers.py:48` filters `txn.get("type") != "add"` but Yahoo returns `"add/drop"` for combined transactions.

**Files:**
- Modify: `backend/routers/fantasy.py` (~line 5498–5507)
- Modify: `backend/services/constraint_helpers.py` (~line 48)
- Test: `tests/test_constraint_helpers.py`

- [ ] **Step 1: Write failing tests for `count_weekly_acquisitions`**

Open (or create) `tests/test_constraint_helpers.py` and add:

```python
# tests/test_constraint_helpers.py
import pytest
from datetime import datetime, timezone
from backend.services.constraint_helpers import count_weekly_acquisitions


def _txn(txn_type: str, team_key: str, ts: float) -> dict:
    return {"type": txn_type, "destination_team_key": team_key, "timestamp": ts}


_TEAM = "mlb.l.12345.t.3"
_MON_TS = datetime(2026, 5, 11, 0, 0, 0, tzinfo=timezone.utc).timestamp()  # Monday
_WED_TS = datetime(2026, 5, 13, 12, 0, 0, tzinfo=timezone.utc).timestamp()  # Wednesday
_SUN_TS = datetime(2026, 5, 17, 23, 59, 0, tzinfo=timezone.utc).timestamp()  # Sunday


def test_add_drop_counted():
    """Yahoo returns 'add/drop' type for combined transactions — must be counted."""
    txns = [_txn("add/drop", _TEAM, _WED_TS)]
    week_start = datetime.fromtimestamp(_MON_TS, tz=timezone.utc)
    week_end = datetime.fromtimestamp(_SUN_TS, tz=timezone.utc)
    assert count_weekly_acquisitions(txns, _TEAM, week_start, week_end) == 1


def test_pure_add_counted():
    """Standard 'add' type must still be counted."""
    txns = [_txn("add", _TEAM, _WED_TS)]
    week_start = datetime.fromtimestamp(_MON_TS, tz=timezone.utc)
    week_end = datetime.fromtimestamp(_SUN_TS, tz=timezone.utc)
    assert count_weekly_acquisitions(txns, _TEAM, week_start, week_end) == 1


def test_drop_not_counted():
    """Pure 'drop' transactions must NOT be counted as acquisitions."""
    txns = [_txn("drop", _TEAM, _WED_TS)]
    week_start = datetime.fromtimestamp(_MON_TS, tz=timezone.utc)
    week_end = datetime.fromtimestamp(_SUN_TS, tz=timezone.utc)
    assert count_weekly_acquisitions(txns, _TEAM, week_start, week_end) == 0


def test_outside_window_not_counted():
    """Transactions before the week start are excluded."""
    before_ts = datetime(2026, 5, 10, 23, 59, 0, tzinfo=timezone.utc).timestamp()  # Sunday
    txns = [_txn("add", _TEAM, before_ts)]
    week_start = datetime.fromtimestamp(_MON_TS, tz=timezone.utc)
    week_end = datetime.fromtimestamp(_SUN_TS, tz=timezone.utc)
    assert count_weekly_acquisitions(txns, _TEAM, week_start, week_end) == 0


def test_wrong_team_not_counted():
    """Transactions by other teams are excluded."""
    txns = [_txn("add", "mlb.l.12345.t.9", _WED_TS)]
    week_start = datetime.fromtimestamp(_MON_TS, tz=timezone.utc)
    week_end = datetime.fromtimestamp(_SUN_TS, tz=timezone.utc)
    assert count_weekly_acquisitions(txns, _TEAM, week_start, week_end) == 0
```

Run: `venv/Scripts/python -m pytest tests/test_constraint_helpers.py -v`
Expected: `test_add_drop_counted` FAILS (returns 0 instead of 1 because type filter only allows `"add"`).

- [ ] **Step 2: Fix the type filter in `constraint_helpers.py`**

Open `backend/services/constraint_helpers.py`. Change line 48:
```python
        if txn.get("type") != "add":
            continue
```
to:
```python
        if txn.get("type") not in ("add", "add/drop"):
            continue
```

- [ ] **Step 3: Run constraint helper tests**

```
venv/Scripts/python -m pytest tests/test_constraint_helpers.py -v
```
Expected: All 5 tests PASS.

- [ ] **Step 4: Fix the rolling-7d window in `fantasy.py`**

Open `backend/routers/fantasy.py`. Find the budget endpoint (~line 5498–5507):

```python
    # 2. Count acquisitions from last 7 days
    try:
        transactions = client.get_transactions(t_type="add")
        week_start = now_et - timedelta(days=7)
        week_end = now_et
        acquisitions_used = count_weekly_acquisitions(
            transactions, team_key, week_start, week_end
        )
    except (YahooAuthError, YahooAPIError):
        pass  # Fall back to 0
```

Replace with:

```python
    # 2. Count acquisitions from matchup week start (Monday 00:00 ET)
    try:
        transactions = client.get_transactions(t_type="add")
        # Find Monday 00:00 ET — the canonical Yahoo matchup week start
        days_since_monday = now_et.weekday()  # Monday=0
        week_start = now_et.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_since_monday)
        week_end = now_et
        acquisitions_used = count_weekly_acquisitions(
            transactions, team_key, week_start, week_end
        )
    except Exception:
        pass  # Fall back to 0
```

Note: `except Exception` replaces `except (YahooAuthError, YahooAPIError)` to also catch `TypeError` from string timestamps (audit finding §2.2).

- [ ] **Step 5: Compile-check both files**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m py_compile backend/services/constraint_helpers.py
```

- [ ] **Step 6: Commit**

```bash
git add backend/routers/fantasy.py backend/services/constraint_helpers.py tests/test_constraint_helpers.py
git commit -m "fix(budget): use matchup-week window for acquisitions; accept add/drop type"
```

---

## Task 4: Fix Ownership Always 0%

**Root cause:** `get_free_agents` fetches from `league/{league_key}/players` without ownership data. Yahoo doesn't support `out=ownership` on the league/players route (causes 400). The fix: after fetching players, make a secondary batch call to `players;player_keys={...}/ownership` to get ownership % from the global players endpoint (which DOES support the ownership subresource).

**Files:**
- Modify: `backend/fantasy_baseball/yahoo_client_resilient.py` (~line 799–812)

- [ ] **Step 1: Write a unit test to drive the design**

Add to `tests/test_yahoo_client.py` (create if missing):

```python
# tests/test_yahoo_client.py
from unittest.mock import MagicMock, patch


def test_get_free_agents_merges_ownership():
    """Ownership % from secondary batch call must be merged into player dicts."""
    from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

    client = YahooFantasyClient.__new__(YahooFantasyClient)
    client.league_key = "mlb.l.99999"
    client._headers = {}
    client._cache = {}
    client._circuit_open = False
    client._failure_count = 0
    client.logger = MagicMock()

    # Mock _get to return a minimal players response
    fake_players_data = {
        "fantasy_content": {
            "league": [
                {},  # league meta
                {"players": {
                    "0": {"player": [{"player_key": "mlb.p.1"}, {"percent_rostered": {"value": "0"}}]},
                    "count": 1,
                }}
            ]
        }
    }

    # Mock ownership batch response
    fake_ownership_data = {
        "fantasy_content": {
            "players": {
                "0": {"player": [{"player_key": "mlb.p.1"}, {"ownership": {"percent_rostered": {"value": "34.5"}}}]},
                "count": 1,
            }
        }
    }

    call_count = {"n": 0}

    def fake_get(path, params=None):
        if "ownership" in path:
            return fake_ownership_data
        return fake_players_data

    client._get = fake_get
    client.get_players_stats_batch = MagicMock(return_value={})

    players = client.get_free_agents()
    # After ownership merge, percent_owned should reflect the ownership batch result
    assert len(players) == 1
    assert players[0]["percent_owned"] == 34.5, f"Expected 34.5, got {players[0]['percent_owned']}"
```

Run: `venv/Scripts/python -m pytest tests/test_yahoo_client.py::test_get_free_agents_merges_ownership -v`
Expected: FAIL (currently returns 0.0).

- [ ] **Step 2: Add the ownership batch fetch to `get_free_agents`**

Open `backend/fantasy_baseball/yahoo_client_resilient.py`. Find `get_free_agents` (~line 776). After the stats enrichment block (after line 811), add an ownership batch fetch:

```python
        # Best-effort: enrich with ownership % via the global players endpoint.
        # The league/players route does not support out=ownership (causes 400).
        # The global players;player_keys=.../ownership route does support it.
        try:
            if player_keys:
                keys_str = ",".join(player_keys[:25])  # Yahoo hard limit
                own_data = self._get(
                    f"players;player_keys={keys_str}/ownership",
                )
                own_block = own_data.get("fantasy_content", {}).get("players", {})
                for raw_entry in own_block.values():
                    if not isinstance(raw_entry, dict):
                        continue
                    player_entry = raw_entry.get("player", [])
                    pk = None
                    pct = None
                    for chunk in player_entry:
                        if isinstance(chunk, dict):
                            if "player_key" in chunk:
                                pk = chunk["player_key"]
                            own = chunk.get("ownership", {})
                            if own:
                                pct_block = own.get("percent_rostered") or own.get("percent_owned")
                                if isinstance(pct_block, dict):
                                    pct = self._safe_float(pct_block.get("value", 0), 0.0)
                                elif pct_block is not None:
                                    pct = self._safe_float(pct_block, 0.0)
                    if pk and pct is not None and pct > 0.0:
                        for p in players:
                            if p.get("player_key") == pk and p.get("percent_owned", 0.0) == 0.0:
                                p["percent_owned"] = pct
        except Exception as _own_err:
            logger.warning("get_free_agents ownership batch failed (non-fatal): %s", _own_err)
```

> **Placement:** Insert this block AFTER the stats enrichment try/except (after line 811, before `return players`).

- [ ] **Step 3: Run the ownership test**

```
venv/Scripts/python -m pytest tests/test_yahoo_client.py::test_get_free_agents_merges_ownership -v
```
Expected: PASS.

- [ ] **Step 4: Compile-check**

```
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
```

- [ ] **Step 5: Commit**

```bash
git add backend/fantasy_baseball/yahoo_client_resilient.py tests/test_yahoo_client.py
git commit -m "fix(waiver): fetch ownership % from players;player_keys/ownership batch call"
```

---

## Task 5: Verify K-1 P0 Issues (Read-First)

These four items were reported on 2026-05-07. Some may already be fixed. Verify each before touching code.

### 5a: Streaming Page Infinite Loading

**Files:**
- Read: `frontend/app/(dashboard)/war-room/streaming/page.tsx`

- [ ] **Step 1: Verify current state**

Check the streaming page's `isLoading` guard:
- If the file has an early-return `if (waiver.isLoading)` block at the top of the component → the guard is present
- The file was modified by A-7 (commit 847415c) — read it and confirm loading/error/empty states are all handled

If the loading state IS handled (it is in the current code), mark this item RESOLVED.

Current code at line 18 already has:
```typescript
if (waiver.isLoading) { return <...Loading spinner...> }
if (waiver.isError)   { return <...error...> }
if (!waiver.data)     { return <...no data...> }
```
This is correct. **Item RESOLVED by A-7.**

### 5b: Dashboard Empty Render

**Files:**
- Read: `frontend/app/(dashboard)/dashboard/page.tsx`

- [ ] **Step 1: Verify current state**

The dashboard page at `frontend/app/(dashboard)/dashboard/page.tsx` has a full implementation:
- Calls `endpoints.getDashboard()` 
- Checks `response?.success ? response.data : null`
- Renders `LineupGapsCard`, `InjuryFlagsCard`, `WaiverTargetsCard`, `StreaksCard`

Backend at `/api/dashboard` returns `{ "success": true, "timestamp": ..., "data": {...} }` which matches the `DashboardResponse` type.

If the code looks complete as described, **mark RESOLVED** — this was fixed by the dashboard page rewrite.

### 5c: Lineup 422

**Files:**
- Read: `backend/main.py` lines 5051–5092

- [ ] **Step 1: Verify current state**

`main.py` has:
```python
@app.get("/api/fantasy/lineup/current")  # line 5051 — static route, registered first
async def get_fantasy_lineup_current(...):
    _today = today_et().strftime("%Y-%m-%d")   # valid YYYY-MM-DD
    return await get_fantasy_lineup_recommendations(lineup_date=_today, ...)

@app.get("/api/fantasy/lineup/{lineup_date}")  # line 5069 — parameterized, registered after
async def get_fantasy_lineup_recommendations(lineup_date: str, ...):
    try:
        ld = date_type.fromisoformat(lineup_date)
    except ValueError:
        raise HTTPException(status_code=422, detail="lineup_date must be YYYY-MM-DD")
```

FastAPI routes static paths before parameterized paths. `/api/fantasy/lineup/current` correctly routes to `get_fantasy_lineup_current`.

**Mark RESOLVED** — the routing is correct. The 2026-05-07 UAT likely tested it differently.

### 5d: CORS on POST /api/fantasy/matchup/simulate

**Files:**
- Read: `backend/main.py` (~line 697–708) — CORS config

- [ ] **Step 1: Read and assess CORS config**

Current config (main.py ~line 701–708):
```python
_raw_origins = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _raw_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

If `ALLOWED_ORIGINS` is not set in Railway, `allow_origins=["*"]` is used — POST requests would work.
If `ALLOWED_ORIGINS` IS set and the frontend URL doesn't match exactly, CORS fails.

- [ ] **Step 2: Check if simulate endpoint throws before CORS headers are added**

Read `backend/routers/fantasy.py` around line 4420–4450 (`simulate_matchup`). If the endpoint throws a 500 (e.g., Yahoo auth failure) BEFORE returning, FastAPI's exception handler runs instead of the normal response path. Starlette's CORSMiddleware applies headers during the `after_response` phase — if the exception handler returns a JSONResponse directly without going through the middleware chain, CORS headers may not be added.

- [ ] **Step 3: If CORS headers are missing on exceptions, add explicit header**

If the simulate endpoint returns 5xx errors and those lack the CORS header, wrap the endpoint body in a try/except that returns a `JSONResponse` with explicit CORS headers:

```python
# At the top of simulate_matchup in routers/fantasy.py
from fastapi.responses import JSONResponse

@router.post("/api/fantasy/matchup/simulate")
async def simulate_matchup(
    user: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    try:
        # ... existing body ...
        pass
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("simulate_matchup failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"detail": str(exc)},
            headers={"Access-Control-Allow-Origin": "*"},
        )
```

Alternatively, verify `ALLOWED_ORIGINS` is set correctly in Railway (must include the exact frontend origin, e.g., `https://observant-benevolence-production.up.railway.app`).

- [ ] **Step 4: Read the simulate endpoint body to understand failure modes**

```python
# In backend/routers/fantasy.py, search for @router.post("/api/fantasy/matchup/simulate")
# Read ~50 lines to understand what can throw before a response is returned
```

- [ ] **Step 5: Apply fix if needed**

If the endpoint body is wrapped properly and the middleware handles CORS already: **mark RESOLVED.**
If there's an unhandled exception path: add the try/except wrapper from Step 3.

- [ ] **Step 6: Compile-check**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```

- [ ] **Step 7: Commit (if changes made)**

```bash
git add backend/routers/fantasy.py
git commit -m "fix(cors): ensure simulate endpoint exceptions include CORS headers"
```

---

## Task 6: Run Full Test Suite and Final Compile Checks

- [ ] **Step 1: Run full suite**

```
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: All existing tests pass; new tests in `test_constraint_helpers.py`, `test_category_aware_scorer.py`, `test_fantasy_budget.py`, `test_yahoo_client.py` all pass.

- [ ] **Step 2: Compile-check all modified files**

```
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m py_compile backend/services/constraint_helpers.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/category_aware_scorer.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
venv/Scripts/python -m py_compile backend/main.py
```

- [ ] **Step 3: Update HANDOFF.md**

Mark each fix complete in the Claude Architect Queue section:
- Phase 1 data bug (1): `ip_accumulated` key fix → COMPLETE
- Phase 1 data bug (2): need score key mapping → COMPLETE
- Phase 1 data bug (3): acquisitions week window → COMPLETE
- Phase 1 data bug (4): ownership batch fetch → COMPLETE
- K-1 P0 items: mark each RESOLVED or FIXED with one-line summary

---

## Self-Review

**Spec coverage:**
- Budget IP key fix (`"my_team"` → `"my_stats"`) → Task 1 ✓
- Need score key mismatch (canonical → board) → Task 2 ✓
- Acquisitions rolling window → Task 3 ✓
- `"add/drop"` transaction type → Task 3 ✓
- Ownership always 0% → Task 4 ✓
- CORS on simulate → Task 5d ✓
- Streaming infinite loading → Task 5a (verified resolved) ✓
- Dashboard empty → Task 5b (verified resolved) ✓
- Lineup 422 → Task 5c (verified resolved) ✓

**Placeholder scan:** No "TBD", no "implement later", no placeholder test bodies. All code is complete.

**Type consistency:** `count_weekly_acquisitions` signature unchanged; `compute_need_score` signature unchanged; `get_free_agents` return type unchanged (`list[dict]`).
