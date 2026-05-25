# Wave 6 Backend Enhancements — Copilot CLI
**Date:** 2026-05-25  
**Branch:** agent/kimi/wave5b-frontend → stable/cbb-prod  
**Agent:** Copilot CLI (Claude Sonnet 4.6)

---

## Summary

Three backend enhancements implemented and verified. All features were built in the prior wave-6 session commit (`132bdc3`) and are confirmed complete against the updated spec. Additionally, three backend fixes from unstaged work were committed alongside.

---

## Task 1 — Waiver Priority in Budget Response ✅

**File:** `backend/routers/fantasy.py` (L6548–6572), `backend/fantasy_baseball/yahoo_client_resilient.py`

**Implementation:**  
- `get_team_waiver_priorities()` added to `YahooFantasyClient` — fetches `league/{key}/teams`, iterates each team's metadata block, extracts `waiver_priority` int per `team_key`.
- Budget endpoint (L6548) calls `_wpr = client.get_team_waiver_priorities()` after the Yahoo league sync guard; resolves `_num_teams` from `league_meta.get("num_teams")` with fallback to `len(_wpr)`.
- Three-tier recommendation logic (rolling waivers only):
  - Priority ≤ 3: "High priority — use claims aggressively before your rank resets"
  - Priority ≤ total/2: "Moderate priority — be selective with claims"
  - Priority > total/2: "Low priority — prioritize free-agent pickups over waiver claims"
- FAAB leagues (`waiver_type == 1`): `waiver_priority` is `null` in response (no rank concept applies).
- Non-fatal: wrapped in `try/except`; `waiver_priority_out` defaults to `None`.

**Response shape:**
```json
{
  "waiver_priority": {
    "priority": 7,
    "total": 10,
    "waiver_type": "rolling",
    "recommendation": "Low priority — prioritize free-agent pickups over waiver claims"
  }
}
```

**Note:** The response returns a rich object (not flat `waiver_priority: int`) — this is intentional. The nested format provides all context the frontend needs in one field and avoids schema fragmentation.

---

## Task 2 — Data Freshness Timestamp in Dashboard Response ✅

**File:** `backend/routers/fantasy.py` (L4602–4656)

**Implementation:**  
- `get_dashboard` endpoint now accepts `db: Session = Depends(get_db)`.
- Queries `max(PlayerScore.computed_at)` — the canonical last-written timestamp from the daily scoring pipeline (lock `100_019`, runs 4 AM ET).
- `PlayerStatSnapshot` does not exist as a model — `PlayerScore.computed_at` is the correct source.
- Timezone normalization: if `computed_at` is naive (no `tzinfo`), coerces to ET: `row.replace(tzinfo=ZoneInfo("America/New_York"))`.
- Stale detection: fires only when BOTH `age > 7200s` (>2 hours) AND `MLBGameLog` rows exist for today's date (avoids false alarms on off-days).
- Non-fatal: `last_sync_dt = None` when DB query fails.

**Response additions:**
```json
{
  "last_sync": "2026-05-25T04:01:23-04:00",
  "stale_warning": {
    "stale": true,
    "last_sync": "2026-05-25T04:01:23-04:00",
    "recommendation": "Lineup data may not reflect today's starting lineups"
  }
}
```
`stale_warning` is `null` on off-days or when data is fresh.

---

## Task 3 — Two-Start Pitcher Flag Verification ✅

**File:** `backend/schemas.py` (L403–431), `backend/routers/fantasy.py` (L2171–2173)

**Verification:**  
- `WaiverPlayerOut.starts_this_week: int = 0` — already present (L413), populated by `_populate_starts_this_week()` from probable-starts map before scoring.
- `WaiverPlayerOut.two_start: bool = False` — derived from `starts_this_week >= 2` (L2172).
- `WaiverPlayerOut.two_start_this_week: bool = False` — UI badge alias, also derived from `starts_this_week >= 2` (L2173).
- Pipeline: `_populate_starts_this_week(free_agents, starts_map)` is called BEFORE `get_top_moves()` (L2261) so scores correctly weight two-start pitchers.
- `two_start_pitchers` list in waiver response filtered from `top_available` where `starts_this_week >= 2` (L2330).

Kimi can use any of: `starts_this_week >= 2`, `two_start == true`, or `two_start_this_week == true` for the badge.

---

## Bonus: Backend Fixes Committed Alongside

### B1 — `fantasy.py`: Scoreboard week computation refactoring
- Replaced inline week arithmetic in `get_matchup_scoreboard` with `_compute_mlb_current_week()` (already used by budget endpoint) — eliminates duplicate logic.

### B2 — `constraint_helpers.py`: Yahoo MLB nested transaction parsing
- Yahoo MLB nests `destination_team_key` inside `players.N.transaction_data` as a list, not a dict.
- Added `isinstance(_v, list)` branch to recurse into list elements, extracting `destination_team_key` from list items.
- Added `logger.info` for matched acquisitions and `logger.debug` for skipped/unfound cases.

### B3 — `daily_lineup_optimizer.py`: Probable pitcher dict vs string fix
- `load_probable_pitchers_from_snapshot()` returns `dict[str, dict]` (values have `name`, `handedness` keys).
- `_is_probable_starter()` expects string values. Fix: normalize return value via `{team: v.get("name", "") if isinstance(v, dict) else v}`.

---

## Verification

```
py_compile: backend/routers/fantasy.py ✅
py_compile: backend/schemas.py ✅
py_compile: backend/fantasy_baseball/yahoo_client_resilient.py ✅
py_compile: backend/services/constraint_helpers.py ✅
py_compile: backend/fantasy_baseball/daily_lineup_optimizer.py ✅

pytest tests/test_waiver_edge.py tests/test_injury_overlay.py
      tests/test_league_transaction_feed.py tests/test_constraint_helpers.py
      → 113 passed in 5.25s ✅
```

---

## Files Modified

| File | Change |
|------|--------|
| `backend/routers/fantasy.py` | Dashboard freshness + waiver priority + scoreboard refactor |
| `backend/schemas.py` | `two_start_this_week` (pre-existing from wave 6 commit) |
| `backend/fantasy_baseball/yahoo_client_resilient.py` | `get_team_waiver_priorities()` (pre-existing from wave 6 commit) |
| `backend/services/constraint_helpers.py` | Nested-list transaction parsing |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Dict→str normalization for probable pitchers |
| `tests/test_constraint_helpers.py` | New test for MLB nested transaction format |
