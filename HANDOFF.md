# OPERATIONAL HANDOFF — MARCH 30, 2026: UAT FIXES + DATA STRATEGY + POST-CBB ROADMAP

> **Ground truth as of March 30, 2026 (end of day).** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk posture · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior active crises: all resolved (see §8 archive).

---

## 1. Data Provider Strategy — LOCKED DECISIONS

### SUBSCRIPTIONS

| Provider | Status | Action |
|----------|--------|--------|
| **BallDontLie GOAT (NCAAB)** | ❌ CANCELLED | CBB season is over. Do NOT renew. |
| **OddsAPI Champion** | ⏳ CANCEL AFTER APR 7 | Still needed for CBB tournament. Cancel once bracket concludes. |
| **BallDontLie GOAT (MLB)** | ✅ SUBSCRIBE AFTER APR 7 | Replaces OddsAPI for all MLB use cases. $39.99/mo. |

**Net cost after transition: $39.99/mo (down from $49/mo OddsAPI Champion). Saves ~$108/yr.**

---

### WHY BALLDONTLIE FOR MLB (not OddsAPI)

1. **Unified stats + odds in one API** — eliminates the raw OddsAPI call in `mlb_analysis._fetch_mlb_odds()` and the raw call in `daily_ingestion._poll_mlb_odds()`. Both are currently unabstracted stubs with no circuit breaker.
2. **Webhooks** — 125+ MLB event types. Enables live game events for betting timing without polling.
3. **600 req/min rate limit** — vs OddsAPI's 90,000/month call budget (polling only).
4. **MCP server** — 250+ endpoints compatible with Claude Code agent workflows.
5. **Bookmaker coverage tradeoff** — BDL covers ~15-20 books vs OddsAPI's 40+. Pinnacle IS included (confirmed in `balldontlie.py` `PREFERRED_BOOKS`). Sharp consensus logic is preserved.

### WHY KEEP PYBASEBALL / STATCAST (do not replace with BDL)

BallDontLie does not expose Statcast-tier metrics: **xwOBA, barrel%, exit velocity, hard-hit%**. These are the core of the fantasy projection engine (`statcast_ingestion.py`, `pybaseball_loader.py`). Keep pybaseball for Statcast. Add BDL for:
- Real-time injury feed
- Box scores (live game data)
- Live event webhooks for betting triggers

---

## 2. Implementation Plan — Post-Apr 7

### Phase 1: Cancel OddsAPI, Subscribe BDL GOAT (MLB) — Day 1 after tournament

**Manual steps (human action required):**
1. Cancel OddsAPI Champion subscription.
2. Subscribe to BallDontLie GOAT ($39.99/mo).
3. Set `BALLDONTLIE_API_KEY` env var in Railway (already present — update to MLB-tier key if different).
4. Remove `THE_ODDS_API_KEY` from Railway env after confirming no CBB code paths call it.

### Phase 2: Expand `balldontlie.py` for MLB — Claude Code task

**File:** `backend/services/balldontlie.py`

Current state: NCAAB-only (`/ncaab/v1/` prefix, TOURNAMENT_SEASON = 2025).

Required additions:
- Add `MLB_PREFIX = "/mlb/v1"` constant alongside `NCAAB_PREFIX`
- Add `get_mlb_games(date)` — fetch today's schedule
- Add `get_mlb_odds(date)` — fetch moneyline/runline/totals per game
- Add `get_mlb_player_stats(season, player_ids)` — season batting/pitching stats
- Add `get_mlb_injuries()` — active IL list
- Add `get_mlb_box_score(game_id)` — live/final box score

The existing client structure (session headers, `_get()`, `_paginate()`, circuit breaker pattern) is solid — replicate it for MLB endpoints. Do NOT break the existing NCAAB methods.

### Phase 3: Migrate MLB Odds Callers — Claude Code task

**Two dirty raw-OddsAPI callers to fix:**

| File | Method | Current Problem | Fix |
|------|--------|-----------------|-----|
| `backend/services/mlb_analysis.py` | `_fetch_mlb_odds()` | Raw `requests.get` to OddsAPI, no circuit breaker, no sharp consensus logic | Replace with `get_bdl_client().get_mlb_odds(date)` |
| `backend/services/daily_ingestion.py` | `_poll_mlb_odds()` | Same — raw OddsAPI call, not using `OddsAPIClient` | Replace with BDL call; preserve advisory lock pattern |

Both callers already have graceful degradation (`return {}` / `return {"status": "skipped"}` on failure), so the swap is low-risk.

### Phase 4: Add BDL as Fantasy Enrichment Feed — Claude Code task

**Not a replacement — additive only.** Existing pybaseball/Statcast pipeline stays untouched.

Add to `daily_ingestion.py` scheduler:
- `_poll_mlb_injuries()` job — calls `get_mlb_injuries()`, writes to a new `PlayerInjuryStatus` table or updates `PlayerDailyMetric.injury_status`
- `_ingest_mlb_box_scores()` job — post-game box scores for rolling stat updates

This feeds the fantasy dashboard's injury display (currently sourced from Yahoo only, which lags by hours).

---

## 3. Current Technical State

| Component | Status | Notes |
|-----------|--------|-------|
| **DB Migrations v9/v10** | ✅ LIVE | Chained into Dockerfile CMD; `user_preferences` table confirmed present |
| **Yahoo client** | ✅ CONSOLIDATED | Single file: `yahoo_client_resilient.py`. Base class + resilient layer unified. |
| **Roster endpoint (`/api/fantasy/roster`)** | ✅ LIVE | 200 OK. |
| **Matchup endpoint (`/api/fantasy/matchup`)** | ✅ LIVE | Team mapping fixed. |
| **SSE stream (`/api/fantasy/dashboard/stream`)** | ✅ LIVE | `StreamingResponse`, `text/event-stream`, 60s interval. No `sse-starlette` dep. |
| **Matchup enrichment** | ✅ LIVE | `opponent_record` from standings. `*_projected_categories` from `PlayerDailyMetric`. |
| **CircuitBreaker** | ✅ FIXED | Catches `Exception` (not just `expected_exception`) so all error types trip the breaker. |
| **Weather integration** | ✅ LIVE | Provider: OpenWeatherMap (`OPENWEATHER_API_KEY` set). |
| **OR-Tools (Railway)** | ✅ LIVE | Installed via requirements.txt. |
| **Streamlit** | ✅ RETIRED | `dashboard/` untouched. Next.js is canonical UI. |
| **Test suite** | ✅ STABLE | 1199+ pass. |
| **RP-as-SP bug** | ✅ FIXED (Mar 29) | `pitcher_slot == "SP"` guard in `_get_probable_pitchers`. |
| **Yahoo stat category IDs** | ✅ FIXED (Mar 29) | `_YAHOO_STAT_FALLBACK` dict + all 22 frontend `STAT_LABELS`. |
| **UAT P0: No-game START** | ✅ FIXED (Mar 30) | Post-optimizer override loop demotes `START` → `BENCH` when `opponent` is empty. |
| **UAT P1: SP score 0.000** | ✅ FIXED (Mar 30) | `p.sp_score === 0 ? '—' : p.sp_score.toFixed(3)` in lineup page. |
| **UAT P1: UNKNOWN badge** | ✅ FIXED (Mar 30) | `FALLBACK_LABELS` map in `statusBadge()` — UNKNOWN/NO_START → "NO START", RP → "RELIEVER". |
| **UAT P1: Streamlit sidebar link** | ✅ FIXED (Mar 30) | Removed `localhost:8501` link from `sidebar.tsx` (Streamlit retired). |
| **UAT P1: Raw Pydantic errors** | ✅ FIXED (Mar 30) | Warning banner filters `validation error` / `Traceback` strings. |
| **UAT: Global "Dashboard" header** | ✅ FIXED (Mar 30) | Fantasy routes added to `PAGE_TITLES` in `header.tsx`. |
| **UAT: "Invalid Date ET"** | ✅ FIXED (Mar 30) | Null guard on `dashboard.timestamp` in dashboard page. |
| **Yahoo flatten_entry last-wins bug** | ✅ FIXED (Mar 30) | `if not stats_raw:` guard in `flatten_entry` — takes only first `team_stats` block. |
| **`_injury_lookup` bool crash** | ✅ FIXED (Mar 30) | `isinstance(p.get("status"), str)` guard — rejects Yahoo `status: False/True`. |
| **`fetch_mlb_odds` coverage logging** | ✅ FIXED (Mar 30) | Logs game list + warns on 0-game response for Railway diagnostics. |
| **MCMC Simulator** | ✅ LIVE (Mar 30) | Calibrated and wired into `_get_matchup_preview()`. New `mcmc_calibration.py` converts Yahoo rosters to MCMC format using player_board + PlayerDailyMetric z-scores. Returns win_probability + category advantages/disadvantages.
| **CBB V9.2 recalibration** | ⏸ BLOCKED | EMAC-068 — SNR/integrity scalar stacking correction. Do NOT touch Kelly math until Apr 7. |
| **`balldontlie.py`** | ⚠️ NCAAB-ONLY | Needs MLB endpoint expansion post-Apr 7 (see §2 Phase 2). |
| **`mlb_analysis._fetch_mlb_odds()`** | ⚠️ DIRTY | Raw OddsAPI call — no circuit breaker, no abstraction. Migrate to BDL post-Apr 7. |
| **`daily_ingestion._poll_mlb_odds()`** | ⚠️ DIRTY | Same — raw OddsAPI call. Migrate to BDL post-Apr 7. |
| **BDL NCAAB subscription** | ❌ CANCELLED | CBB season over. `balldontlie.py` NCAAB methods will 401 — do not call them. |

---

## 4. Advisory Lock IDs (do not reuse)

| Lock ID | Job |
|---------|-----|
| 100_001 | mlb_odds |
| 100_002 | statcast |
| 100_003 | rolling_z |
| 100_004 | cbb_ratings |
| 100_005 | clv |
| 100_006 | cleanup |
| 100_007 | waiver_scan |
| 100_008 | mlb_brief |
| 100_009 | openclaw_perf |
| 100_010 | openclaw_sweep |

Next available: **100_011** (mlb_injuries), **100_012** (mlb_box_scores)

---

## 5. Next Session Roadmap (Claude Code)

Priority order:

1. **CBB V9.2 recalibration** (EMAC-068) — Unblocks Apr 7. SNR/integrity scalar stacking correction. Do NOT touch Kelly math before then.

2. **Post-Apr 7: BDL MLB expansion** — Execute §2 Phases 1-4 in order. Confirm OddsAPI cancelled before writing any BDL MLB code to avoid calling a cancelled key.

3. **Statcast freshness** — `statcast_ingestion.py` exists but data is stale. The `_update_statcast()` job in `daily_ingestion.py` is a stub (`status: "skipped"`). Implement it to actually call `StatcastIngestionAgent` from `statcast_ingestion.py`.

4. **Historical MCMC validation** — Collect actual H2H matchup outcomes to validate win_probability calibration (backtesting). Current calibration uses proxy z-scores; empirical validation pending season data.

---

## 6. Architecture Decisions (Locked)

| Decision | Ruling | Reason |
|----------|--------|--------|
| Yahoo client split-brain | ELIMINATED | Single file: `yahoo_client_resilient.py` |
| Streamlit | RETIRED | Next.js only — never touch `dashboard/` |
| `openclaw_briefs.py` (old) | DELETED | `_improved` is canonical |
| Dashboard refresh strategy | SSE (IMPLEMENTED) | `StreamingResponse` text/event-stream. No sse-starlette dep. |
| Weather provider | OpenWeatherMap (LOCKED) | `OPENWEATHER_API_KEY` set |
| Test file location | `tests/` only | No test files in `backend/` subdirs |
| CBB recalibration | BLOCKED until Apr 7 | EMAC-068 — do not touch Kelly math before then |
| SSE keep-alive | `: keep-alive\n\n` comment line | Prevents Railway/nginx from closing idle SSE connections |
| Odds provider — CBB | OddsAPI → CANCEL post-tournament | Battle-tested for CBB; BDL NCAAB subscription already cancelled |
| Odds provider — MLB | BDL GOAT (post-Apr 7) | Unified stats+odds, webhooks, lower cost. Raw OddsAPI callers in mlb_analysis + daily_ingestion are stubs — low-risk swap |
| Stats provider — Statcast | pybaseball (LOCKED) | BDL does not expose xwOBA/barrel%/exit velocity. Do not replace. |
| Stats provider — injuries/box scores | BDL (additive, post-Apr 7) | Supplements Yahoo injury feed which lags by hours |

---

## 7. Delegation Bundles

### GEMINI CLI (Ops) — Standby

No new tasks since Mar 28 deploy was confirmed. When BDL MLB migration executes:

> 1. After Claude Code pushes BDL MLB changes, run py_compile on modified files:
>    ```
>    railway run python -m py_compile backend/services/balldontlie.py
>    railway run python -m py_compile backend/services/mlb_analysis.py
>    railway run python -m py_compile backend/services/daily_ingestion.py
>    ```
> 2. Confirm `BALLDONTLIE_API_KEY` is set in Railway env (MLB-tier key):
>    ```
>    railway variables | grep -i balldontlie
>    ```
> 3. Confirm `THE_ODDS_API_KEY` has been removed from Railway env after cancellation.
> 4. Trigger redeploy and smoke-test:
>    ```
>    railway run python -c "from backend.services.balldontlie import get_bdl_client; c = get_bdl_client(); print('BDL OK')"
>    ```
> Do NOT edit any `.py` or `.ts` files.

### KIMI CLI (Deep Intelligence Unit) — Standby

> No active coding tasks. Standing responsibilities:
> - If CBB recalibration (EMAC-068) status changes to unblocked, prepare the V9.2 spec memo.
> - Do NOT write to any production code files without an explicit Claude delegation bundle.
>
> Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge

---

## 8. Resolved Crises (Archive — Do Not Revisit)

| Crisis | Resolution | Date |
|--------|------------|------|
| `user_preferences` table missing | v9/v10 migrations chained into Dockerfile CMD | Mar 27 |
| Pydantic `status: False` → HTTP 500 | `_parse_player` + `RosterPlayerOut` guarded with `or None` | Mar 28 |
| Matchup "Team not found" | `m.get("teams") or m.get("0", {}).get("teams", {})` | Mar 28 |
| West Coast games showing no-game | `datetime.utcnow()` → `datetime.now(ZoneInfo("America/New_York"))` | Mar 28 |
| `injury_status` always None | `injury_status=p.get("status") or None` in `RosterPlayerOut` | Mar 28 |
| `_get_lineup_gaps` empty on `team_key=None` | `client.get_roster()` no-arg form | Mar 28 |
| UI routing cascade (Kimi hotfix) | Roster dedup, team key recursive parse, NaN float guard | Mar 28 |
| CircuitBreaker only counted `expected_exception` | `except Exception:` in `call()` and `call_async()` | Mar 28 |
| RP-as-SP in probable pitchers | `pitcher_slot == "SP"` guard in `_get_probable_pitchers` | Mar 29 |
| Yahoo stat category IDs returning raw numbers | `_YAHOO_STAT_FALLBACK` dict + 22-entry `STAT_LABELS` frontend map | Mar 29 |
| Yahoo `flatten_entry` last-wins bug (OBP>1, walks decimal, wins inflated) | `if not stats_raw:` guard in `flatten_entry` — takes only first `team_stats` block | Mar 30 |
| `_injury_lookup` Pydantic crash (`status: False` → bool) | `isinstance(p.get("status"), str)` guard in `_injury_lookup` dict comprehension | Mar 30 |
| No-game players (HOU/ARI/CLE/SF) receiving START | Post-optimizer override loop: `START` → `BENCH` when `opponent` is empty | Mar 30 |
| Global "Dashboard" title on all fantasy pages | Added `/fantasy/*` routes to `PAGE_TITLES` in `header.tsx` | Mar 30 |
| "Invalid Date ET" on dashboard timestamp | Null guard `dashboard.timestamp ? ... : 'N/A'` | Mar 30 |
| SP score shows 0.000 for no-start pitchers | `p.sp_score === 0 ? '—' : ...` in lineup page | Mar 30 |
| UNKNOWN status badge displayed verbatim | `FALLBACK_LABELS` map → "NO START"; RP → "RELIEVER" | Mar 30 |
| Streamlit localhost:8501 link in production sidebar | Removed entire block from `sidebar.tsx` | Mar 30 |
| Raw Pydantic validation errors in warning banner | Filter on "validation error" / "Traceback" strings | Mar 30 |
| MCMC Simulator calibration (B5) | `mcmc_calibration.py` created; wired into `_get_matchup_preview()`; returns win_probability + category advantages | Mar 30 |

---

## HANDOFF PROMPTS

### For Gemini CLI

```
You are Gemini CLI (Ops). Read this HANDOFF.md in full before acting.

Current status: standby. No active deploy tasks.

When Claude Code completes the BDL MLB migration (§2 Phases 2-3), you will be
asked to verify. At that point:

1. py_compile the three modified files (balldontlie.py, mlb_analysis.py, daily_ingestion.py)
2. Confirm BALLDONTLIE_API_KEY is present in Railway env (not the old NCAAB key)
3. Confirm THE_ODDS_API_KEY has been removed
4. Trigger redeploy and run the BDL smoke test shown in §7

Do NOT edit any .py or .ts files.
Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```

### For Kimi CLI (if initiating independently)

```
You are Kimi CLI (Deep Intelligence Unit). Read this HANDOFF.md in full.

No active coding tasks are assigned. Key context for this session:
- BallDontLie NCAAB subscription is CANCELLED. Do not call NCAAB endpoints.
- OddsAPI will be cancelled post-Apr 7. Do not build new features depending on it.
- BallDontLie GOAT (MLB) will be the new odds + enrichment provider post-Apr 7.
- EMAC-068 (CBB V9.2 recalibration) is still blocked until Apr 7. Do not touch Kelly math.
- MCMC calibration (B5) is COMPLETED as of Mar 30. Next: empirical validation against actual H2H outcomes.

Do NOT write to any production code files without an explicit Claude delegation bundle.
Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```
