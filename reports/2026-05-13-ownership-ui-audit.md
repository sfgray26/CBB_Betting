# Ownership 0% Audit + UI Issues Analysis — 2026-05-13

> **Agent:** Kimi CLI (research audit)  
> **Scope:** End-to-end diagnosis of why all players show 0% ownership + current UI breakage  
> **Method:** Live UI inspection (Chrome DevTools MCP), production API response capture, code path tracing, PostgreSQL query

---

## 1. Executive Summary

**All players across the entire Fantasy Baseball UI show "0% owned."** This is not a display bug — the backend API is returning `ownership_pct: 0.0` for every player. Root cause: Yahoo Fantasy API does **not** include ownership data in the endpoints currently used for roster or waiver fetches. A working ownership endpoint exists but is only partially wired and capped at 25 players.

**Additionally, the Streaming Station page crashes** with a client-side `TypeError: Cannot read properties of undefined (reading 'toFixed')`.

---

## 2. Evidence from Live Production

### 2.1 Roster Page — Every Player Shows 0% Owned

Live DOM inspection of `/war-room/roster`:

| Player | Displayed Ownership | API `ownership_pct` |
|--------|--------------------|---------------------|
| Moisés Ballesteros | 0% owned | `0.0` |
| Pete Alonso | 0% owned | `0.0` |
| Juan Soto | 0% owned | `0.0` |
| Blake Snell | 0% owned | `0.0` |
| Gavin Williams | 0% owned | `0.0` |

Captured API response (`GET /api/fantasy/roster`) confirms every `CanonicalPlayerRow` has `"ownership_pct": 0.0`.

### 2.2 Dashboard — Waiver Targets Also 0% Owned

Live DOM inspection of `/dashboard`:

| Player | Displayed Ownership |
|--------|--------------------|
| Max Meyer | 0% owned |
| Landen Roupp | 0% owned |
| Colin Holderman | 0% owned |
| Erik Sabrowski | 0% owned |
| Grant Taylor | 0% owned |

### 2.3 Streaming Station — Client-Side Crash

Navigating to `/war-room/streaming` produces:

```
Application error: a client-side exception has occurred
Uncaught TypeError: Cannot read properties of undefined (reading 'toFixed')
```

Stack trace originates from `page-e53793a4175385f5.js:1:2495` (streaming page bundle).

---

## 3. Root Cause Analysis — Ownership 0%

### 3.1 Yahoo Endpoint Behavior

| Endpoint | Used By | Includes Ownership? |
|----------|---------|---------------------|
| `team/{team_key}/roster/players` | `get_roster()` | ❌ NO |
| `league/{league_key}/players` | `get_free_agents()`, `get_adp_and_injury_feed()` | ❌ NO (base response) |
| `players;player_keys={keys}/ownership` | Partially in `get_free_agents()` | ✅ YES |

**Key code comments from `yahoo_client_resilient.py`:**

```python
# FIX (April 21, 2026): out=ownership is NOT a valid subresource on
# league/.../players and causes a 400 error. Ownership data is fetched
# via get_players_stats_batch() in the best-effort enrichment step below.
```

This comment is misleading — `get_players_stats_batch()` fetches **stats**, not ownership.

### 3.2 Code Path: Roster Endpoint (`GET /api/fantasy/roster`)

**File:** `backend/routers/fantasy.py` (lines 2845–3059)

1. Calls `client.get_roster(team_key)` → Yahoo `team/{team_key}/roster/players`
2. `_parse_player()` recursively searches for `ownership` block — **not present in response**
3. `ownership_pct` defaults to `0.0`
4. `map_yahoo_player_to_canonical_row()` passes `0.0` through unchanged
5. API returns `ownership_pct: 0.0` for every player

**Missing step:** No ownership enrichment is performed for roster players.

### 3.3 Code Path: Waiver Endpoint (`GET /api/fantasy/waiver`)

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py` (lines 776–845)

`get_free_agents()` **does** attempt ownership enrichment:

```python
# Best-effort: enrich with ownership % via the global players endpoint.
# players;player_keys=.../ownership is valid on the global players collection.
try:
    if player_keys:
        keys_str = ",".join(player_keys[:25])  # ← BUG: only first 25 players!
        own_data = self._get(f"players;player_keys={keys_str}/ownership")
        ...
except Exception as _own_err:
    logger.warning("get_free_agents ownership batch failed (non-fatal): %s", _own_err)
```

**Two problems:**
1. **Only first 25 players get ownership data** — `player_keys[:25]` hard-caps the batch
2. **If the batch fails, it fails silently** — exception is caught and logged at WARNING level, players remain `0.0`

Historical Postman response (`20260422_210436/waiver.json`) confirms `"owned_pct": 0.0` was present even in April.

### 3.4 Code Path: PositionEligibility Sync Job

**File:** `backend/services/daily_ingestion.py` (lines 7107–7299)

The `_sync_position_eligibility` job tries to populate `league_rostered_pct`:

```python
ownership_by_key: dict[str, float] = {}
try:
    adp_players = await asyncio.to_thread(
        yahoo.get_adp_and_injury_feed,
        pages=10,           # 250 players
        count_per_page=25,
    )
    ownership_by_key = {
        p["player_key"]: p["percent_owned"]
        for p in adp_players
        if p.get("player_key") and isinstance(p.get("percent_owned"), (int, float))
    }
except Exception as exc:
    logger.warning("...Failed to fetch ownership feed...")
```

**Problem:** `get_adp_and_injury_feed()` uses `league/{league_key}/players` which **does not include ownership data**. Even if the call succeeds, `percent_owned` is `0.0` for all players, so `ownership_by_key` is either empty or filled with `0.0` values.

**Production database confirms:**

```sql
SELECT COUNT(*) as total_rows,
       COUNT(league_rostered_pct) as rows_with_ownership
FROM position_eligibility;
```

| total_rows | rows_with_ownership |
|-----------:|--------------------:|
| 2,389      | **0**               |

The `position_eligibility` job logs show `SUCCESS` with ~234 records processed, but `league_rostered_pct` remains NULL for all rows because the upstream data source returns no ownership.

---

## 4. Additional UI Issues Found

### 4.1 Streaming Station Crash (P0)

**Symptom:** White screen with "Application error: a client-side exception has occurred"
**Console:** `Uncaught TypeError: Cannot read properties of undefined (reading 'toFixed')`
**Likely cause:** `player.need_score.toFixed(1)` or `d.deficit_z_score.toFixed(2)` called on `undefined` value

**File:** `frontend/app/(dashboard)/war-room/streaming/page.tsx`

### 4.2 Waiver API Latency

Network tab shows `GET /api/fantasy/waiver?sort=need_score` taking >3s. No loading skeleton is visible during fetch.

### 4.3 Cross-Origin API Calls

Frontend on `observant-benevolence-production.up.railway.app` calls API on `fantasy-app-production-5079.up.railway.app`. CORS is configured (`access-control-allow-origin: *`), but this split-domain architecture can cause cache/session mismatches.

---

## 5. Recommended Fixes

### Fix 1: Add Ownership Enrichment to `get_roster()`

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`

After parsing roster players, batch-fetch ownership via the working global endpoint:

```python
def get_roster(self, team_key: Optional[str] = None) -> list[dict]:
    ...
    players = list(players_by_key.values())
    
    # Enrich with ownership % (same pattern as get_free_agents)
    self._enrich_ownership_batch(players)
    return players
```

Extract the existing ownership enrichment from `get_free_agents()` into a reusable `_enrich_ownership_batch(players)` helper that handles >25 players via chunking.

### Fix 2: Fix 25-Player Cap in Waiver Ownership Enrichment

**File:** `backend/fantasy_baseball/yahoo_client_resilient.py`

Change:
```python
keys_str = ",".join(player_keys[:25])  # ← BUG
```

To:
```python
for i in range(0, len(player_keys), 25):
    chunk_keys = player_keys[i:i + 25]
    keys_str = ",".join(chunk_keys)
    own_data = self._get(f"players;player_keys={keys_str}/ownership")
    ...
```

### Fix 3: Fix `_sync_position_eligibility` to Use Working Ownership Endpoint

**File:** `backend/services/daily_ingestion.py`

Replace the broken `get_adp_and_injury_feed()` ownership source with the global `players;player_keys=.../ownership` batch endpoint.

### Fix 4: Fix Streaming Station Crash

**File:** `frontend/app/(dashboard)/war-room/streaming/page.tsx`

Add null-guards before all `.toFixed()` calls:

```tsx
<span className="text-[#FFC000] text-xs font-mono">
  {player.need_score != null ? player.need_score.toFixed(1) : '-'}
</span>
```

And same for `d.deficit_z_score` in the category deficits map.

### Fix 5: Enrich Roster API from `PositionEligibility` Table

**File:** `backend/routers/fantasy.py`

As a fallback (and to reduce Yahoo API calls), query `PositionEligibility.league_rostered_pct` for all rostered players and merge it into the response before calling `map_yahoo_player_to_canonical_row()`.

---

## 6. Impact Assessment

| Impact | Severity | Notes |
|--------|----------|-------|
| Ownership data completely missing | **P1** | Affects roster, waiver, dashboard, streaming pages |
| Waiver targets show 0% + 0.00 need score | **P1** | Makes waiver recommendations unusable for decision-making |
| Streaming Station crashes | **P0** | Page is completely broken |
| PositionEligibility table has no ownership | **P2** | Blocks any downstream feature that depends on league_rostered_pct |

---

*Report generated by Kimi CLI at 2026-05-13. Live UI inspection + production API capture + PostgreSQL query confirmed all findings.*
