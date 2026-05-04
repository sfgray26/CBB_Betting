# Ownership Wire-Up Spec — 2026-04-29

> **Agent:** Kimi CLI (research-only audit)  
> **Scope:** End-to-end audit of `league_rostered_pct` data path + `daily_lineup_optimizer.py` data-quality gaps  
> **Branch:** stable/cbb-prod, HEAD 8c7058c

---

## 1. Executive Summary

- **`league_rostered_pct` is hardcoded to `None`** in `_sync_position_eligibility` at `daily_ingestion.py:5502`. The ownership data is already available in the `get_league_rosters()` response (via `_parse_player` → `find_ownership` → `"percent_owned"` key), but the sync loop simply ignores it.
- **`get_adp_and_injury_feed()` is a richer, more reliable source** for ownership because it explicitly queries the `league/{league_key}/players` endpoint (which includes ownership blocks for all players), whereas `get_league_rosters()` fetches `league/{league_key}/teams/roster` where Yahoo may or may not embed ownership data.
- **Current `get_adp_and_injury_feed()` default covers only 100 players** (`pages=4 × count_per_page=25`). Yahoo leagues typically carry ~500–700 total players (rostered + free agents). To achieve full coverage, increase to at least `pages=20` (or `pages=10, count_per_page=50`).
- **`daily_lineup_optimizer.py` contains no dead-code references to missing columns.** It correctly reads `scarcity_rank` from `PositionEligibility` and `ProbablePitcherSnapshot.*` fields that all exist in the schema.
- **No N+1 API call pattern** in `get_league_rosters()`. It makes a single `teams/roster` call and parses all players locally.

---

## 2. TASK 1 — Ownership Data Path Audit

### 2.1 `yahoo_client_resilient.py` — `get_adp_and_injury_feed()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Line:** 756

```python
def get_adp_and_injury_feed(
    self,
    pages: int = 4,
    count_per_page: int = 25,
) -> list[dict]:
```

**Endpoint:** `GET league/{self.league_key}/players?start={start}&count={count}&sort=DA`  
**Requires `league_key`?** **YES** — uses `self.league_key` at line 776. Not global.  
**Sort order:** ADP ascending (`sort=DA`).  
**Returned player dict keys:** `player_key`, `name`, `team`, `positions`, `status`, `injury_note`, `is_undroppable`, **`percent_owned`**.

**Coverage analysis:**
- Default `pages=4 × count_per_page=25` = **100 players**.
- A typical Yahoo fantasy league has ~12 teams × 25 roster slots = **300 rostered** + **200–400 free agents** = **500–700 total**.
- **Recommendation:** Increase to `pages=20, count_per_page=25` (500 players) or `pages=25, count_per_page=25` (625 players) to ensure full coverage.
- **Risk:** Players sorted far down by ADP (deep free agents) will be missed with fewer pages. This is acceptable for ownership data because deep free agents are typically 0% owned, but for completeness the parameter should be increased.

### 2.2 `yahoo_client_resilient.py` — `_parse_player()` and `find_ownership()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Lines:** 1284–1402

The `find_ownership()` helper (lines 1319–1350) recursively searches the raw Yahoo player structure for ownership data. It checks these keys in order:

1. `ownership["percent_rostered"]` (Yahoo 2025+ format)
2. `ownership["percent_owned"]` (legacy format)
3. Flat `percent_rostered` at top level
4. Flat `percent_owned` dict at top level
5. `meta["percent_owned"]` fallback (pre-2025 format)

**Exact dict key returned in player dict:** `"percent_owned"` (line 1401).

```python
return {
    ...
    "percent_owned": owned_pct,
}
```

### 2.3 `yahoo_client_resilient.py` — `_parse_players_block()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Lines:** 1417–1436

Calls `_parse_player(p)` for each player, which preserves the `"percent_owned"` key. No ownership data is stripped.

### 2.4 `yahoo_client_resilient.py` — `get_league_rosters()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Lines:** 397–486

- **Single API call** to `league/{league_key}/teams/roster` (line 399).
- **No N+1 pattern** — iterates over all teams and all rostered players in memory.
- Calls `_parse_player()` for each player, so `"percent_owned"` **is** present in the returned dicts.
- **Caveat:** Yahoo's `teams/roster` endpoint may or may not include the `ownership` block in the response. If it doesn't, `percent_owned` will be `0.0` for all rostered players.

### 2.5 `daily_ingestion.py` — `_sync_position_eligibility()`

**File:** `backend/services/daily_ingestion.py`  
**Lines:** 5385–5546

**Current INSERT/UPDATE dict (line 5502):**
```python
stmt = pg_insert(PositionEligibility.__table__).values(
    yahoo_player_key=player_key,
    bdl_player_id=None,
    player_name=name,
    first_name="",
    last_name="",
    primary_position=primary,
    player_type=ptype,
    multi_eligibility_count=multi_count,
    scarcity_rank=scarcity_rank,
    league_rostered_pct=None,   # <-- HARDCODED NULL
    fetched_at=now,
    updated_at=now,
    **flags,
).on_conflict_do_update(
    ...
)
```

**The `get_league_rosters()` response already contains `percent_owned`**, but `_sync_position_eligibility` never reads it:
- `player_data.get("percent_owned", None)` is **never called**.
- `league_rostered_pct=None` is passed verbatim on every upsert.

**Recommended insertion point:**
- **File:** `backend/services/daily_ingestion.py`
- **Line:** ~5490 (just before the `stmt = pg_insert(...)` block)
- **Change:** Add `rostered_pct = player_data.get("percent_owned", None)` and pass it as `league_rostered_pct=rostered_pct`.

**Alternative richer source:** If `get_league_rosters()` does not reliably include ownership, also call `yahoo.get_adp_and_injury_feed(pages=20)` before the loop and build a lookup map:
```python
ownership_map = {
    p["player_key"]: p.get("percent_owned")
    for p in yahoo.get_adp_and_injury_feed(pages=20)
}
# Then inside loop:
rostered_pct = ownership_map.get(player_key, None)
```

---

## 3. TASK 2 — `daily_lineup_optimizer.py` Data Gap Audit

### 3.1 Columns Read from DB

| Table | Columns Read | Line(s) | Exists in Schema? | Production State |
|-------|-------------|---------|-------------------|------------------|
| `PositionEligibility` | `scarcity_rank` | 225, 228 | ✅ Yes | **9.8% populated** (235/2,389) |
| `PositionEligibility` | `primary_position` | 227 | ✅ Yes | 100% populated |
| `ProbablePitcherSnapshot` | `team` | 326 | ✅ Yes | Fresh (2026-04-29) |
| `ProbablePitcherSnapshot` | `opponent` | 327 | ✅ Yes | Fresh |
| `ProbablePitcherSnapshot` | `is_home` | 328 | ✅ Yes | Fresh |
| `ProbablePitcherSnapshot` | `park_factor` | 329 | ✅ Yes | Fresh |
| `MLBPlayerStats` (via `probable_pitcher_fallback.py`) | `game_date`, `innings_pitched`, `raw_payload`, `bdl_player_id` | 102–107 | ✅ Yes | Fresh |
| `PlayerIDMapping` (via `probable_pitcher_fallback.py`) | `bdl_id`, `mlbam_id`, `full_name` | 92–94 | ✅ Yes | Unknown coverage |

### 3.2 Data Quality Gaps

| Field | Impact on Optimizer | Current State | Risk Level |
|-------|---------------------|---------------|------------|
| `PositionEligibility.scarcity_rank` | Tiebreaker in `solve_lineup()` when two players have identical `lineup_score`. Falls back to static `_POSITION_SCARCITY` dict when DB returns NULL. | 9.8% populated | **LOW** — static fallback works but ignores live scarcity shifts |
| `ProbablePitcherSnapshot.*` | Used for schedule fallback when Odds API fails. | Fresh | **NONE** |
| `league_rostered_pct` | **NOT referenced** in `daily_lineup_optimizer.py` | 0% populated | **NONE** for optimizer (but HIGH for waiver edge) |

### 3.3 Dead Code Check

**Result: No dead code paths found.**

- `daily_lineup_optimizer.py` references `scarcity_rank` (lines 225, 228) → column exists ✅
- References `ProbablePitcherSnapshot.team/opponent/is_home/park_factor` (lines 326–329) → columns exist ✅
- References `MLBPlayerStats.game_date/innings_pitched/raw_payload/bdl_player_id` via `probable_pitcher_fallback.py` → columns exist ✅
- No references to `quality_score`, `league_rostered_pct`, `w_runs`, `z_r`, or other fields that might be missing.

---

## 4. TASK 3 — Batch Efficiency Audit

### 4.1 `get_league_rosters()` Efficiency

**Verdict: Already batched. No N+1 problem.**

- **One API call** to `league/{league_key}/teams/roster`.
- Parses all teams and all players in a single response locally.
- No additional network requests per team or per player.

### 4.2 `get_adp_and_injury_feed()` Coverage Gap

**Verdict: May miss deep free agents.**

- Sorts by ADP (`sort=DA`).
- Default `pages=4, count_per_page=25` = 100 players.
- A Yahoo league has ~500–700 total players. Deep free agents (low ownership, high ADP) are sorted to the bottom and may not appear in the first 100.
- **If `get_adp_and_injury_feed()` is used to populate `league_rostered_pct`**, the missing ~400 players will have NULL ownership. This is acceptable for free agents (they are 0% owned), but rostered players should all appear in the top 100.
- **If `get_league_rosters()` reliably includes ownership**, it is the preferred source because it covers 100% of rostered players with a single call.
- **Recommendation:** Test whether `get_league_rosters()` returns non-zero `percent_owned` values. If yes, use it (zero extra API calls). If no, fall back to `get_adp_and_injury_feed(pages=20)`.

---

## 5. Session O Implementation Spec

### 5.1 Fix `league_rostered_pct` in `_sync_position_eligibility`

**File:** `backend/services/daily_ingestion.py`  
**Line:** ~5490 (before the `stmt = pg_insert(...)` block)

**Current code (line 5502):**
```python
league_rostered_pct=None,
```

**Recommended change:**
```python
# Extract ownership from roster data (already parsed by _parse_player)
rostered_pct = player_data.get("percent_owned", None)
if rostered_pct == 0.0:
    rostered_pct = None  # Distinguish "0% owned" from "not fetched"
```

Then update lines 5502 and 5513:
```python
# INSERT values
league_rostered_pct=rostered_pct,

# ON CONFLICT DO UPDATE set_
"league_rostered_pct": rostered_pct,
```

**Estimated impact:** Restores ownership data for ~300 rostered players. Free agents will remain NULL unless `get_adp_and_injury_feed()` is also called.

### 5.2 Optional: Full Ownership Coverage via `get_adp_and_injury_feed()`

**File:** `backend/services/daily_ingestion.py`  
**Line:** ~5426 (after `all_players = await asyncio.to_thread(...)`)

**Insertion:**
```python
# Fetch ownership data for all players (rostered + free agents)
try:
    ownership_feed = await asyncio.to_thread(
        yahoo.get_adp_and_injury_feed,
        pages=20,  # ~500 players
        count_per_page=25,
    )
    ownership_map = {
        p["player_key"]: p.get("percent_owned")
        for p in ownership_feed
        if p.get("player_key")
    }
except Exception as exc:
    logger.warning("Failed to fetch ownership feed: %s", exc)
    ownership_map = {}
```

Then inside the player loop:
```python
rostered_pct = player_data.get("percent_owned", None)
if rostered_pct is None or rostered_pct == 0.0:
    rostered_pct = ownership_map.get(player_key, None)
```

**Estimated impact:** Covers all ~500–700 players with ownership data. Adds one API call per sync.

### 5.3 Increase `get_adp_and_injury_feed()` Default Pages

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`  
**Line:** 758

```python
def get_adp_and_injury_feed(
    self,
    pages: int = 20,   # was 4
    count_per_page: int = 25,
) -> list[dict]:
```

**Estimated impact:** Ensures full league coverage without caller-side parameter overrides.

---

## 6. Raw Evidence

### `daily_ingestion.py:5491–5518` (hardcoded `None`)
```python
stmt = pg_insert(PositionEligibility.__table__).values(
    yahoo_player_key=player_key,
    bdl_player_id=None,
    player_name=name,
    first_name="",
    last_name="",
    primary_position=primary,
    player_type=ptype,
    multi_eligibility_count=multi_count,
    scarcity_rank=scarcity_rank,
    league_rostered_pct=None,          # <-- LINE 5502
    fetched_at=now,
    updated_at=now,
    **flags,
).on_conflict_do_update(
    constraint="_pe_yahoo_uc",
    set_={
        "player_name": name,
        "primary_position": primary,
        "player_type": ptype,
        "multi_eligibility_count": multi_count,
        "scarcity_rank": scarcity_rank,
        "updated_at": now,
        **flags,
    },
)
```

### `yahoo_client_resilient.py:1392–1402` (return dict with `"percent_owned"`)
```python
return {
    "player_key": meta.get("player_key"),
    "player_id": meta.get("player_id"),
    "name": name,
    "team": meta.get("editorial_team_abbr"),
    "positions": [p for p in positions if p],
    "status": meta.get("status") or None,
    "injury_note": meta.get("injury_note") or None,
    "is_undroppable": meta.get("is_undroppable", 0) in (1, '1', True, 'true'),
    "percent_owned": owned_pct,       # <-- LINE 1401
}
```

### `yahoo_client_resilient.py:756–788` (`get_adp_and_injury_feed` signature)
```python
def get_adp_and_injury_feed(
    self,
    pages: int = 4,
    count_per_page: int = 25,
) -> list[dict]:
    ...
    for page in range(pages):
        start = page * count_per_page
        params = {"start": start, "count": count_per_page, "sort": "DA"}
        data = self._get(f"league/{self.league_key}/players", params=params)
```

---

*Report generated by Kimi CLI at 2026-04-29. Read-only audit — no files modified.*
