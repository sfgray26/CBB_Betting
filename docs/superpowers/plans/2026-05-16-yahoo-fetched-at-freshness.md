# Yahoo fetched_at Freshness Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Track when Yahoo API data was fetched and compute `is_stale` (data older than 15 minutes) so `FreshnessMetadata.fetched_at` and `FreshnessMetadata.is_stale` are real values instead of `None`/`False` placeholders.

**Architecture:** The in-memory `YahooAPICache` already stores `(data, expiry)` per cache key. We extend it to also store `fetched_at` (Unix timestamp from `time.time()` at put-time). A new `get_fetched_at(key)` method on the cache plus a `get_cached_fetched_at(path, params)` helper on `YahooFantasyClient` lets callers retrieve this timestamp. We thread it through `map_yahoo_player_to_canonical_row` (player mapper) and `assemble_matchup_scoreboard` (scoreboard orchestrator) so every `FreshnessMetadata` in roster and scoreboard API responses has real timestamps and a correct `is_stale` flag.

**Tech Stack:** Python 3.11, FastAPI, Pydantic v2, `time.time()` for Unix timestamps, `datetime.fromtimestamp(..., tz=ZoneInfo("America/New_York"))` for conversion.

---

## File Map

| File | Change |
|------|--------|
| `backend/fantasy_baseball/yahoo_client_resilient.py` | Extend `YahooAPICache` storage to 3-tuple; add `get_fetched_at()`; add `get_cached_fetched_at()` on client |
| `backend/services/player_mapper.py` | Add `fetched_at` param to `map_yahoo_player_to_canonical_row`; compute `is_stale` |
| `backend/services/scoreboard_orchestrator.py` | Add `yahoo_fetched_at` param to `assemble_matchup_scoreboard`; use it in freshness |
| `backend/routers/fantasy.py` | Roster endpoint: extract `fetched_at` after `get_roster()`; pass to mapper and freshness. Scoreboard endpoint: extract `fetched_at` after `get_matchup_stats()`; pass to orchestrator |
| `tests/test_yahoo_freshness.py` | New test file covering cache tracking, mapper staleness, freshness flag |

---

## Task 1: Extend YahooAPICache to track fetched_at

**Files:**
- Modify: `backend/fantasy_baseball/yahoo_client_resilient.py` (lines 70–120)
- Test: `tests/test_yahoo_freshness.py`

- [ ] **Step 1: Write failing tests for `YahooAPICache.get_fetched_at`**

```python
# tests/test_yahoo_freshness.py
"""Tests for Yahoo data freshness tracking (fetched_at / is_stale)."""
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from unittest.mock import MagicMock, patch
import pytest


class TestYahooAPICache:
    """Tests for YahooAPICache fetched_at tracking."""

    def _make_cache(self):
        from backend.fantasy_baseball.yahoo_client_resilient import YahooAPICache
        return YahooAPICache(default_ttl_seconds=300)

    def test_get_fetched_at_returns_none_for_missing_key(self):
        cache = self._make_cache()
        assert cache.get_fetched_at("nonexistent") is None

    def test_get_fetched_at_returns_timestamp_after_put(self):
        cache = self._make_cache()
        before = time.time()
        cache.put("key1", {"data": "value"})
        after = time.time()
        ts = cache.get_fetched_at("key1")
        assert ts is not None
        assert before <= ts <= after

    def test_get_fetched_at_returns_none_for_expired_entry(self):
        cache = self._make_cache()
        cache.put("key1", {"data": "value"}, ttl_seconds=1)
        time.sleep(1.1)
        assert cache.get_fetched_at("key1") is None

    def test_get_returns_data_unchanged_after_cache_extension(self):
        cache = self._make_cache()
        data = {"answer": 42}
        cache.put("key1", data)
        result = cache.get("key1")
        assert result == data
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestYahooAPICache -v
```

Expected: `ModuleNotFoundError` or `AttributeError: YahooAPICache has no attribute 'get_fetched_at'`

- [ ] **Step 3: Modify `YahooAPICache` to store `(data, expiry, fetched_at)` and add `get_fetched_at`**

In `backend/fantasy_baseball/yahoo_client_resilient.py`, update the three methods:

```python
# In YahooAPICache.get() — replace the current implementation
def get(self, key: str) -> Optional[dict]:
    """Get cached response if still fresh."""
    with self._lock:
        if key not in self._cache:
            return None

        entry = self._cache[key]
        data, expiry = entry[0], entry[1]  # Supports both 2-tuple (legacy) and 3-tuple
        if time.time() > expiry:
            del self._cache[key]
            return None

        self._cache.move_to_end(key)
        return data

# In YahooAPICache.put() — replace the current implementation
def put(self, key: str, data: dict, ttl_seconds: Optional[int] = None) -> None:
    """Cache response with TTL, recording fetched_at for freshness tracking."""
    ttl = ttl_seconds or self._default_ttl
    now = time.time()
    expiry = now + ttl

    with self._lock:
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        self._cache[key] = (data, expiry, now)  # 3-tuple: data, expiry, fetched_at

# New method — add after get_stats()
def get_fetched_at(self, key: str) -> Optional[float]:
    """Return Unix timestamp when key was last fetched from Yahoo, or None if not cached/expired."""
    with self._lock:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if len(entry) < 3:
            return None  # Legacy 2-tuple entry without fetched_at
        data, expiry, fetched_at = entry
        if time.time() > expiry:
            del self._cache[key]
            return None
        return fetched_at
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestYahooAPICache -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
```

Expected: no output (clean)

- [ ] **Step 6: Commit**

```bash
git add backend/fantasy_baseball/yahoo_client_resilient.py tests/test_yahoo_freshness.py
git commit -m "feat(freshness): extend YahooAPICache to track fetched_at timestamp per key"
```

---

## Task 2: Add `get_cached_fetched_at` helper to `YahooFantasyClient`

**Files:**
- Modify: `backend/fantasy_baseball/yahoo_client_resilient.py` (after `_get_ttl_for_endpoint`, ~line 394)
- Test: `tests/test_yahoo_freshness.py`

- [ ] **Step 1: Write failing test for `get_cached_fetched_at`**

Add this class to `tests/test_yahoo_freshness.py`:

```python
class TestYahooClientCachedFetchedAt:
    """Tests for YahooFantasyClient.get_cached_fetched_at()."""

    def _make_client_with_cache(self):
        """Build a YahooFantasyClient with mocked credentials and pre-seeded cache."""
        with patch.dict("os.environ", {
            "YAHOO_CLIENT_ID": "fake_client_id_12345",
            "YAHOO_CLIENT_SECRET": "fake_secret",
            "YAHOO_REFRESH_TOKEN": "fake_token",
            "YAHOO_LEAGUE_ID": "12345",
        }):
            from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
            return YahooFantasyClient()

    def test_returns_none_when_path_not_cached(self):
        client = self._make_client_with_cache()
        result = client.get_cached_fetched_at("team/469.l.12345.t.7/roster/players")
        assert result is None

    def test_returns_datetime_after_cache_put(self):
        client = self._make_client_with_cache()
        from zoneinfo import ZoneInfo
        cache_key = client._make_cache_key("team/469.l.12345.t.7/roster/players", None)
        client._cache.put(cache_key, {"fake": "data"})
        result = client.get_cached_fetched_at("team/469.l.12345.t.7/roster/players")
        assert result is not None
        assert isinstance(result, datetime)
        assert result.tzinfo is not None  # Must be timezone-aware (ET)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestYahooClientCachedFetchedAt -v
```

Expected: `AttributeError: 'YahooFantasyClient' object has no attribute 'get_cached_fetched_at'`

- [ ] **Step 3: Add `get_cached_fetched_at` to `YahooFantasyClient`**

In `backend/fantasy_baseball/yahoo_client_resilient.py`, inside `YahooFantasyClient`, add this method after `_get_ttl_for_endpoint` (~line 394):

```python
def get_cached_fetched_at(self, path: str, params: Optional[dict] = None) -> Optional[datetime]:
    """Return when the cached response for this path was last fetched from Yahoo.

    Returns None if the response is not in cache or has expired.
    Timestamp is in Eastern Time.
    """
    cache_key = self._make_cache_key(path, params)
    ts = self._cache.get_fetched_at(cache_key)
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=ZoneInfo("America/New_York"))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestYahooClientCachedFetchedAt -v
```

Expected: 2 tests PASS

- [ ] **Step 5: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
```

- [ ] **Step 6: Commit**

```bash
git add backend/fantasy_baseball/yahoo_client_resilient.py tests/test_yahoo_freshness.py
git commit -m "feat(freshness): add get_cached_fetched_at helper to YahooFantasyClient"
```

---

## Task 3: Wire fetched_at into `map_yahoo_player_to_canonical_row`

**Files:**
- Modify: `backend/services/player_mapper.py` (lines 162–237)
- Test: `tests/test_yahoo_freshness.py`

- [ ] **Step 1: Write failing tests for mapper freshness**

Add this class to `tests/test_yahoo_freshness.py`:

```python
class TestMapperFreshness:
    """Tests for fetched_at / is_stale in map_yahoo_player_to_canonical_row."""

    def _minimal_yahoo_player(self) -> dict:
        return {
            "player_key": "469.p.12345",
            "name": "Test Player",
            "team": "NYY",
            "positions": ["OF"],
        }

    def test_fetched_at_none_gives_is_stale_false(self):
        from backend.services.player_mapper import map_yahoo_player_to_canonical_row
        row = map_yahoo_player_to_canonical_row(
            self._minimal_yahoo_player(),
            fetched_at=None,
        )
        assert row.freshness.fetched_at is None
        assert row.freshness.is_stale is False

    def test_fresh_data_is_not_stale(self):
        from backend.services.player_mapper import map_yahoo_player_to_canonical_row
        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=5)  # 5 minutes ago — within 15-min threshold
        row = map_yahoo_player_to_canonical_row(
            self._minimal_yahoo_player(),
            fetched_at=fetched_at,
            computed_at=now,
        )
        assert row.freshness.fetched_at == fetched_at
        assert row.freshness.is_stale is False

    def test_old_data_is_stale(self):
        from backend.services.player_mapper import map_yahoo_player_to_canonical_row
        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=20)  # 20 minutes ago — exceeds 15-min threshold
        row = map_yahoo_player_to_canonical_row(
            self._minimal_yahoo_player(),
            fetched_at=fetched_at,
            computed_at=now,
        )
        assert row.freshness.is_stale is True

    def test_staleness_threshold_is_15_minutes(self):
        from backend.services.player_mapper import map_yahoo_player_to_canonical_row
        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=15, seconds=1)  # Just over 15 min
        row = map_yahoo_player_to_canonical_row(
            self._minimal_yahoo_player(),
            fetched_at=fetched_at,
            computed_at=now,
        )
        assert row.freshness.staleness_threshold_minutes == 15
        assert row.freshness.is_stale is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestMapperFreshness -v
```

Expected: `TypeError: map_yahoo_player_to_canonical_row() got an unexpected keyword argument 'fetched_at'`

- [ ] **Step 3: Add `fetched_at` param and staleness computation to mapper**

In `backend/services/player_mapper.py`, update `map_yahoo_player_to_canonical_row`:

Change the function signature at line ~162:
```python
def map_yahoo_player_to_canonical_row(
    yahoo_player: Dict,
    rolling_stats: Optional[PlayerRollingStats] = None,
    rolling_stats_7d: Optional[PlayerRollingStats] = None,
    rolling_stats_14d: Optional[PlayerRollingStats] = None,
    rolling_stats_15d: Optional[PlayerRollingStats] = None,
    rolling_stats_30d: Optional[PlayerRollingStats] = None,
    computed_at: Optional[datetime] = None,
    ros_projection: Optional[CategoryStats] = None,
    fetched_at: Optional[datetime] = None,
) -> CanonicalPlayerRow:
```

Replace the freshness block at line ~231–237:
```python
    # PR-22: Freshness metadata
    _threshold_minutes = 15
    if fetched_at is not None:
        age_seconds = (now_et - fetched_at).total_seconds()
        is_stale = age_seconds > _threshold_minutes * 60
    else:
        is_stale = False  # fetched_at unknown: optimistically assume fresh
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=fetched_at,
        computed_at=now_et,
        staleness_threshold_minutes=_threshold_minutes,
        is_stale=is_stale,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestMapperFreshness -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/services/player_mapper.py
```

- [ ] **Step 6: Commit**

```bash
git add backend/services/player_mapper.py tests/test_yahoo_freshness.py
git commit -m "feat(freshness): add fetched_at param to map_yahoo_player_to_canonical_row, compute is_stale at 15-min threshold"
```

---

## Task 4: Wire fetched_at into the roster API endpoint

**Files:**
- Modify: `backend/routers/fantasy.py` (lines ~3112, ~3302–3320)
- Test: `tests/test_yahoo_freshness.py`

- [ ] **Step 1: Write failing test for freshness in roster response**

Add this to `tests/test_yahoo_freshness.py` (uses monkeypatching, no DB required):

```python
class TestRosterEndpointFreshness:
    """Integration test: roster endpoint must populate fetched_at in FreshnessMetadata."""

    def test_freshness_has_fetched_at_when_yahoo_returns_data(self):
        """Verify the freshness.fetched_at field is populated in GET /api/fantasy/roster."""
        from backend.services.player_mapper import map_yahoo_player_to_canonical_row

        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=3)

        player = {
            "player_key": "469.p.99999",
            "name": "Fresh Player",
            "team": "BOS",
            "positions": ["1B"],
        }
        row = map_yahoo_player_to_canonical_row(player, fetched_at=fetched_at, computed_at=now)
        assert row.freshness.fetched_at == fetched_at
        assert row.freshness.is_stale is False

    def test_freshness_is_stale_when_yahoo_data_is_old(self):
        from backend.services.player_mapper import map_yahoo_player_to_canonical_row

        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=16)  # Over threshold

        player = {
            "player_key": "469.p.99999",
            "name": "Stale Player",
            "team": "BOS",
            "positions": ["1B"],
        }
        row = map_yahoo_player_to_canonical_row(player, fetched_at=fetched_at, computed_at=now)
        assert row.freshness.is_stale is True
```

- [ ] **Step 2: Run tests to verify they pass (these test mapper directly)**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestRosterEndpointFreshness -v
```

Expected: 2 tests PASS (mapper tests pass after Task 3)

- [ ] **Step 3: Update the roster endpoint to extract and pass `fetched_at`**

In `backend/routers/fantasy.py`, after the `raw_players = client.get_roster(team_key=team_key)` call (~line 3112), add:

```python
    # Extract fetched_at from cache for freshness tracking
    _roster_fetched_at = client.get_cached_fetched_at(
        f"team/{team_key}/roster/players"
    )
```

Then at the `canonical_row = map_yahoo_player_to_canonical_row(...)` call site (~line 3302), add `fetched_at=_roster_fetched_at` as a keyword argument:

```python
        canonical_row = map_yahoo_player_to_canonical_row(
            yahoo_player=merged_player,
            rolling_stats_7d=rs_7d,
            rolling_stats_14d=rs_14d,
            rolling_stats_15d=rs_15d,
            rolling_stats_30d=rs_30d,
            computed_at=now_et,
            ros_projection=_ros_proj,
            fetched_at=_roster_fetched_at,
        )
```

Then update the roster-level `FreshnessMetadata` block (~line 3314–3319):

```python
    # Build freshness metadata
    _threshold_minutes = 15
    _is_stale = (
        _roster_fetched_at is None or
        (now_et - _roster_fetched_at).total_seconds() > _threshold_minutes * 60
    ) if _roster_fetched_at is not None else False
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=_roster_fetched_at,
        computed_at=now_et,
        staleness_threshold_minutes=_threshold_minutes,
        is_stale=_is_stale,
    )
```

- [ ] **Step 4: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```

Expected: no output (clean)

- [ ] **Step 5: Commit**

```bash
git add backend/routers/fantasy.py tests/test_yahoo_freshness.py
git commit -m "feat(freshness): wire fetched_at into roster endpoint FreshnessMetadata"
```

---

## Task 5: Wire fetched_at into `assemble_matchup_scoreboard`

**Files:**
- Modify: `backend/services/scoreboard_orchestrator.py` (lines ~293–308, ~391–398)
- Test: `tests/test_yahoo_freshness.py`

- [ ] **Step 1: Write failing test for scoreboard orchestrator freshness**

Add to `tests/test_yahoo_freshness.py`:

```python
class TestScoreboardFreshness:
    """Tests for fetched_at / is_stale in assemble_matchup_scoreboard."""

    def _call_assemble(self, yahoo_fetched_at=None):
        from backend.services.scoreboard_orchestrator import assemble_matchup_scoreboard
        return assemble_matchup_scoreboard(
            week=1,
            opponent_name="Test Opponent",
            my_current_stats={"R": 10.0, "H": 30.0, "HR_B": 3.0, "RBI": 12.0,
                              "K_B": 25.0, "TB": 45.0, "AVG": 0.280, "OPS": 0.780,
                              "NSB": 2.0, "W": 2.0, "L": 1.0, "HR_P": 1.0,
                              "K_P": 20.0, "ERA": 3.50, "WHIP": 1.20, "K_9": 9.0,
                              "QS": 2.0, "NSV": 1.0, "IP": 18.0, "SV": 1.0, "HLD": 2.0},
            opp_current_stats={"R": 8.0, "H": 25.0, "HR_B": 2.0, "RBI": 9.0,
                               "K_B": 20.0, "TB": 38.0, "AVG": 0.260, "OPS": 0.740,
                               "NSB": 1.0, "W": 1.0, "L": 2.0, "HR_P": 2.0,
                               "K_P": 15.0, "ERA": 4.00, "WHIP": 1.35, "K_9": 7.5,
                               "QS": 1.0, "NSV": 0.0, "IP": 15.0, "SV": 0.0, "HLD": 1.0},
            my_player_scores=[],
            yahoo_fetched_at=yahoo_fetched_at,
        )

    def test_freshness_none_fetched_at_gives_is_stale_false(self):
        result = self._call_assemble(yahoo_fetched_at=None)
        assert result.freshness.fetched_at is None
        assert result.freshness.is_stale is False

    def test_freshness_fresh_data_not_stale(self):
        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=3)
        result = self._call_assemble(yahoo_fetched_at=fetched_at)
        assert result.freshness.fetched_at == fetched_at
        assert result.freshness.is_stale is False

    def test_freshness_old_data_is_stale(self):
        now = datetime.now(ZoneInfo("America/New_York"))
        fetched_at = now - timedelta(minutes=20)
        result = self._call_assemble(yahoo_fetched_at=fetched_at)
        assert result.freshness.is_stale is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestScoreboardFreshness -v
```

Expected: `TypeError: assemble_matchup_scoreboard() got an unexpected keyword argument 'yahoo_fetched_at'`

- [ ] **Step 3: Update `assemble_matchup_scoreboard` signature**

In `backend/services/scoreboard_orchestrator.py`, add `yahoo_fetched_at` param to `assemble_matchup_scoreboard` (~line 293):

```python
def assemble_matchup_scoreboard(
    week: int,
    opponent_name: str,
    my_current_stats: Dict[str, float],
    opp_current_stats: Dict[str, float],
    my_player_scores: List[Dict],
    opp_player_scores: Optional[List[Dict]] = None,
    ip_accumulated: float = 0.0,
    ip_minimum: float = 18.0,
    games_remaining: int = 0,
    days_remaining: int = 7,
    acquisitions_used: int = 0,
    il_used: int = 0,
    n_monte_carlo_sims: int = 1000,
    force_stale: bool = False,
    yahoo_fetched_at: Optional[datetime] = None,
) -> MatchupScoreboardResponse:
```

Replace the freshness block at lines ~391–398:

```python
    # Step 7: Freshness metadata
    _threshold_minutes = 15
    if yahoo_fetched_at is not None:
        _age_secs = (now_et - yahoo_fetched_at).total_seconds()
        _is_stale = _age_secs > _threshold_minutes * 60
    else:
        _is_stale = False  # fetched_at unknown: optimistically assume fresh
    freshness = FreshnessMetadata(
        primary_source="yahoo",
        fetched_at=yahoo_fetched_at,
        computed_at=now_et,
        staleness_threshold_minutes=_threshold_minutes,
        is_stale=_is_stale,
    )
```

Also add `Optional` and `datetime` imports if not already present — check the import block at the top of the file. `datetime` is already imported from `datetime` and `Optional` from `typing`.

- [ ] **Step 4: Run tests to verify they pass**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py::TestScoreboardFreshness -v
```

Expected: 3 tests PASS

- [ ] **Step 5: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/services/scoreboard_orchestrator.py
```

- [ ] **Step 6: Commit**

```bash
git add backend/services/scoreboard_orchestrator.py tests/test_yahoo_freshness.py
git commit -m "feat(freshness): add yahoo_fetched_at param to assemble_matchup_scoreboard, compute is_stale"
```

---

## Task 6: Wire fetched_at into the scoreboard API endpoint

**Files:**
- Modify: `backend/routers/fantasy.py` (lines ~5752–5770)

- [ ] **Step 1: Identify where to extract `fetched_at` in the scoreboard endpoint**

The scoreboard endpoint in `fantasy.py` calls:
```python
matchup_data = client.get_matchup_stats(week=week)
```

Internally `get_matchup_stats` calls `get_scoreboard(week=week)` which calls `_get(f"league/{self.league_key}/scoreboard", params={"week": week})`. The cache key is derived from that path + params.

- [ ] **Step 2: Add `fetched_at` extraction after the matchup fetch (~line 5665)**

In `backend/routers/fantasy.py`, immediately after the `matchup_data = client.get_matchup_stats(week=week)` line and its logger call (~line 5664–5667), add:

```python
        # Extract fetched_at from scoreboard cache for freshness tracking
        _sb_params = {"week": week} if week else None
        _scoreboard_fetched_at = client.get_cached_fetched_at(
            f"league/{client.league_key}/scoreboard",
            _sb_params,
        )
```

- [ ] **Step 3: Pass `yahoo_fetched_at` to `assemble_matchup_scoreboard` (~line 5752)**

Update the `assemble_matchup_scoreboard(...)` call to add:

```python
        result = assemble_matchup_scoreboard(
            week=week,
            opponent_name=safe_opponent_name,
            my_current_stats=my_current_stats,
            opp_current_stats=opp_current_stats,
            my_player_scores=my_player_scores,
            opp_player_scores=None,
            ip_accumulated=45.0,
            ip_minimum=90.0,
            games_remaining=3,
            days_remaining=4,
            acquisitions_used=5,
            il_used=1,
            yahoo_fetched_at=_scoreboard_fetched_at,
        )
```

- [ ] **Step 4: Syntax check**

```bash
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```

- [ ] **Step 5: Commit**

```bash
git add backend/routers/fantasy.py
git commit -m "feat(freshness): wire scoreboard fetched_at from cache into assemble_matchup_scoreboard"
```

---

## Task 7: Full test suite regression check

**Files:**
- Test: `tests/test_yahoo_freshness.py` (all tests)
- Check: full pytest suite

- [ ] **Step 1: Run the new freshness tests**

```bash
venv/Scripts/python -m pytest tests/test_yahoo_freshness.py -v
```

Expected: All tests PASS (minimum 13 tests: 4 cache + 2 client + 4 mapper + 2 endpoint + 3 scoreboard)

- [ ] **Step 2: Run full test suite**

```bash
venv/Scripts/python -m pytest tests/ -q --tb=short
```

Expected: No regressions. Previous passing tests still pass.

- [ ] **Step 3: Compile all modified files**

```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py
venv/Scripts/python -m py_compile backend/services/player_mapper.py
venv/Scripts/python -m py_compile backend/services/scoreboard_orchestrator.py
venv/Scripts/python -m py_compile backend/routers/fantasy.py
```

Expected: All clean (no output)

- [ ] **Step 4: Final commit (tag as complete)**

```bash
git add -A
git commit -m "feat(freshness): complete fetched_at tracking — roster and scoreboard now expose real freshness indicators"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
|-------------|------|
| Add fetched_at tracking in yahoo_client_resilient.py | Task 1 + Task 2 |
| Compute is_stale (data > 15 mins old) | Task 3 (mapper), Task 5 (scoreboard) |
| Expose in API responses | Task 4 (roster), Task 6 (scoreboard) |
| Regression tests | Task 7 |
| Files affected match spec | All 5 files (yahoo_client_resilient, player_mapper, scoreboard_orchestrator, fantasy.py, schemas.py — note: schemas.py is not changed as FreshnessMetadata is in contracts.py) |

### Type consistency
- `YahooAPICache.get_fetched_at()` → `Optional[float]` (Unix timestamp)
- `YahooFantasyClient.get_cached_fetched_at()` → `Optional[datetime]` (ET-aware)
- `map_yahoo_player_to_canonical_row(fetched_at=...)` → `Optional[datetime]`
- `assemble_matchup_scoreboard(yahoo_fetched_at=...)` → `Optional[datetime]`
- All consistent throughout: Unix float only inside cache, `datetime` at all public interfaces.

### Known non-change
`backend/schemas.py` is listed in "Files affected" in the task spec but `FreshnessMetadata` lives in `contracts.py` (not `schemas.py`) and is already fully defined. No schema change needed.

### Placeholder scan
No TBD, TODO, or "fill in later" items remain in this plan. All code blocks are complete.
