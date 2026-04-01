# OPERATIONAL HANDOFF — MARCH 31, 2026: ARCH-001 CONTRACT LAYER + API-WORKER PATTERN

> **Ground truth as of March 31, 2026 (end of session 2).** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk posture · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Architecture reference: `reports/ARCHITECTURE_ANALYSIS_API_WORKER_PATTERN.md` (ARCH-001)
> Prior active crises: all resolved (see §9 archive).

---

## 0. Active Architecture Initiative — ARCH-003 (UI/UX REDESIGN - PENDING SIGN-OFF)

**Status:** Strategic Design Complete. **AWAITING CLAUDE CODE SIGN-OFF.**

Gemini CLI has produced a comprehensive refactor plan to transform the bloated frontend into a "Command Center."

### Core Directives (ARCH-003)
1. **Standardize UI**: Replace all 6 raw HTML tables with a unified `DataTable` (@/components/shared/data-table.tsx).
2. **Consolidate Monoliths**: Break down 800+ line page files into a `features/` directory structure.
3. **URL-as-State**: Use `nuqs` to move all filters, tabs, and date selections into search params.
4. **ET-Normalization**: Replace all local UTC date slicing with a shared `America/New_York` helper.

### Proposed Route Architecture
- `/dashboard` (The Pulse): Aggregated KPI health.
- `/fantasy/hq` (The Command Center): **[NEW]** Unified Lineup + Matchup view.
- `/fantasy/market` (The Explorer): Refactored high-density waiver browser.
- `/fantasy/roster` (The Lab): Owned player deep-dive.

### Refactor Hitlist (DELETE & REPLACE)
- **DELETE**: Duplicated `TableSkeleton` versions. **REPLACE**: Generic `DataTableSkeleton`.
- **DELETE**: Inline `statusBadge()` helpers. **REPLACE**: Centralized `@/components/shared/status-badge.tsx`.
- **DELETE**: Manual UTC date slicing in `api.ts`. **REPLACE**: ET-anchored helper.

---

## 0b. ARCH-001 & ARCH-002 (COMPLETE)

| Initiative | Status | Purpose |
|------------|--------|---------|
| **ARCH-001** | ✅ COMPLETE | Contract Layer + Async Job Queue + Phase 3 Integration. |
| **ARCH-002** | ✅ COMPLETE | Reliability Roadmap: Status normalization and BDL migration plan. |

---

## 1. REVIEW TASK: ARCH-003 Refactor Sign-Off — Claude Code Action Required

**Claude Code:** Review the refactor plan in §0 and the detailed report at `reports/GEMINI_UI_AUDIT_2026-04-01.md`. 

**Directive:** 
- If you approve the architectural direction of ARCH-003, please update §7 with a "GO" signal.
- Once sign-off is granted, Gemini CLI (Ops) will begin the component extraction and file deletion process.

---

## 2. Data Provider Strategy — LOCKED DECISIONS

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
| **Yahoo token over-refresh** | ✅ FIXED (Mar 31) | Singleton via `get_yahoo_client()` / `get_resilient_yahoo_client()` — token refreshed once per process, not per request |
| **ProjectionsLoader CSV re-read** | ✅ FIXED (Mar 31) | `@lru_cache(maxsize=1)` on `load_full_board()`; force reload via `POST /admin/fantasy/reload-board` |
| **ARCH-001 Phase 3: Frontend Integration** | ✅ VERIFIED | Lineup page reads valuations cache, async optimize polling, skeletons. |
| **asyncOptimizeLineup (api.ts)** | ✅ FIXED | Contract mismatch fixed (query params, not body). Smoke-tested on Railway. |
| **getPlayerValuations (api.ts)** | ✅ VERIFIED | Reading from cache (v12) with empty/stale degradation. |
| **`todayStr()` ET anchor** | ✅ FIXED (Apr 1) | `toLocaleDateString('en-CA', {timeZone: 'America/New_York'})` — West Coast users no longer see wrong date. |
| **matchup `loading.tsx` / `error.tsx`** | ✅ FIXED (Apr 1) | Created both — skeleton + error UI with retry button. |

---

## 4. Advisory Lock IDs (do not reuse)

| Lock ID | Job | Status |
|---------|-----|--------|
| 100_001 | mlb_odds | LIVE |
| 100_002 | statcast | LIVE |
| 100_003 | rolling_z | LIVE |
| 100_004 | cbb_ratings | LIVE |
| 100_005 | clv | LIVE |
| 100_006 | cleanup | LIVE |
| 100_007 | waiver_scan | LIVE |
| 100_008 | mlb_brief | LIVE |
| 100_009 | openclaw_perf | LIVE |
| 100_010 | openclaw_sweep | LIVE |
| 100_011 | valuation_cache | LIVE (Phase 2) |

Next available: **100_012** (mlb_injuries), **100_013** (mlb_box_scores)

---

## 5. Next Session Roadmap (Claude Code)

### Current State (Apr 1, 2026)
ARCH-001 is fully shipped: Phases 1 + 2 + 3 LIVE on Railway. All Gemini audit findings resolved (High + Medium severity). The system is **stable**. We are in a holding pattern until April 7 unlocks the next major workstream.

### Immediate (Before Apr 7) — COMPLETE
1. **Low-severity UI cleanup** ✅ — all 5 items done (Apr 1):
   - `fantasy/page.tsx`: hardcoded draft date removed; `DraftBoardTab` text loader replaced with table skeleton
   - `waiver/page.tsx`: disabled "Add" button → Yahoo deep link (`baseball.fantasysports.yahoo.com`); rec pulse divs → card-shaped skeletons
   - `frontend/lib/constants.ts`: created — `STAT_LABELS`, `RATIO_STATS`, `LOWER_IS_BETTER` exported
   - `matchup/page.tsx`: imports from `@/lib/constants` instead of local duplication
   - TypeScript passes after all changes

2. **Historical MCMC validation setup** — deferred post-Apr 7. Needs 4 weeks of season data first. Script will read H2H outcomes from Yahoo via `/api/fantasy/matchup` history and compute Brier score against `execution_decisions.win_probability`.

### April 7+ — EMAC-068 Unblocks
3. **CBB V9.2 recalibration** — Kimi has the K-12 spec memo ready. Fix SNR/integrity scalar stacking that makes the effective Kelly divisor ~3.4× instead of the calibrated 2.0×. This is the primary lever to improve the CBB win record. Do NOT touch any Kelly math before Apr 7.

4. **Cancel OddsAPI, subscribe BDL GOAT MLB** — Day 1 action after tournament concludes. Manual steps documented in §2 Phase 1. Claude Code then expands `balldontlie.py` with `/mlb/v1/` endpoints and migrates the two raw OddsAPI callers (§2 Phases 2-3). Advisory locks 100_012 + 100_013 reserved.

5. **MLB Betting Module** — After BDL is wired and tested. `mlb_analysis.py` is at stub level. Full implementation of edge calculation, Kelly sizing, and alert generation. Modelled on the CBB pipeline but using `mlb_analysis.py` as the entry point instead of `betting_model.py`.

6. **MCMC empirical calibration** — After 4 weeks of season data, run the Brier score script and adjust `win_probability` output scaling if needed.

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

### GEMINI CLI (Ops) — ACTIVE TASK

**Deploy ARCH-001 Phase 3 (frontend)**

```
You are Gemini CLI (Ops). ARCH-001 Phase 3 frontend has been built by Claude Code.
Your job is to deploy it to Railway.

Step 1: Trigger redeploy (Phase 3 is frontend-only — no migration needed)
  railway up

Step 2: Smoke-test the async optimize endpoint
  curl -s -X POST https://<app>.railway.app/api/fantasy/lineup/async-optimize \
    -H "X-API-Key: $API_KEY" \
    -d '{"target_date": "2026-04-01"}' | python -m json.tool
  # Should return: {"job_id": "...", "status": "queued", "poll_url": "..."}

Step 3: Smoke-test the valuations endpoint
  curl -s "https://<app>.railway.app/api/fantasy/players/valuations?date=2026-04-01" | python -m json.tool
  # Should return: {"cache_status": "fresh"|"stale"|"empty", ...}

Step 4: Verify the lineup page in the browser
  - Navigate to /fantasy/lineup
  - Confirm page loads without hanging (skeleton → data in <2s)
  - Confirm "Optimize Lineup" button shows progress bar (Queued → Processing → Complete)
  - Confirm "Proj" column appears in batters table (may show — if cache is empty)
  - Confirm cache banner appears if valuations are stale/empty

Step 5: Report results back via HANDOFF.md update

Do NOT edit any .py or .ts files.
```

**Forward-Looking: Gemini UI Audit** (✅ COMPLETE — Apr 1)

Gemini CLI has completed a read-only UI audit of the `frontend/app/(dashboard)/fantasy/` directory.

**Findings resolved by Claude Code (Apr 1):**

| Severity | Issue | Fix | Status |
|----------|-------|-----|--------|
| **HIGH** | `asyncOptimizeLineup` sends JSON body; backend expects query params → 422 | Changed to `?target_date=X&risk_tolerance=balanced` in `api.ts` | ✅ FIXED |
| **MEDIUM** | No `loading.tsx`/`error.tsx` for `/fantasy/matchup/` | Created both files with skeleton + error UI | ✅ FIXED |
| **MEDIUM** | `todayStr()` uses local browser time — West Coast users get wrong date after 9 PM PT | `toLocaleDateString('en-CA', { timeZone: 'America/New_York' })` | ✅ FIXED |
| **LOW** | Hardcoded "Draft: March 23 @ 7:30am" in `fantasy/page.tsx` | Deferred — next cleanup pass | ⏳ QUEUED |
| **LOW** | `STAT_LABELS` hardcoded in matchup page | Deferred — no user impact, centralize in next cleanup | ⏳ QUEUED |
| **LOW** | Waiver "Add" button permanently disabled | Deferred — convert to Yahoo deep link in next pass | ⏳ QUEUED |
| **LOW** | Inconsistent skeleton usage in `DraftBoardTab` and waiver recs | Deferred — cosmetic | ⏳ QUEUED |

TypeScript type-check passes after all fixes.


---

### KIMI CLI (Deep Intelligence Unit) — Standby

> No active coding tasks. Standing responsibilities:
> - If CBB recalibration (EMAC-068) status changes to unblocked, prepare the V9.2 spec memo.
> - ARCH-001 Phase 2 is complete. No further research needed on cache invalidation.
> - Do NOT write to any production code files without an explicit Claude delegation bundle.
>
> Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge

---

## 9. Elite Advancement Plan — ARCH-002

The "Elite Advancement Plan" identifies 5 high-priority architectural gaps to resolve for maximum reliability and scalability.

### Gap Status

| # | Gap | Status | Fix |
|---|-----|--------|-----|
| 1 | **Async Status Contract Drift** — backend used `running`/`done`, frontend expected `processing`/`completed` | ✅ FIXED Mar 31 | `job_queue_service.py` status strings updated |
| 2 | **Queue Error Masking** — `process_pending_jobs` marked jobs `done` even on logical errors | ✅ FIXED Mar 31 | `_run_lineup_optimization` now raises; caller handles retry/fail |
| 3 | **Fragmented MLB Provider** — raw OddsAPI calls in `mlb_analysis.py` + `daily_ingestion.py` | ⏳ Post-Apr 7 | Migrate to BDL GOAT MLB after subscription activates |
| 4 | **`main.py` Monolith** — 6,374-line file, no domain routers | ⏳ Deferred | Extract routers after MLB module stabilises |
| 5 | **Date UTC Drift** — `api.ts` used `toISOString().slice(0,10)` (UTC) instead of ET | ✅ FIXED Mar 31 | `etTodayStr()` in `constants.ts`; `dailyLineup` default updated |

### 90-Day Roadmap (ARCH-002)

*   **Phase 1: Reliability Contracts** ✅ COMPLETE (Mar 31) — Gaps 1, 2, 5 fixed. Contract tests pending (low priority).
*   **Phase 2: Data Source Unification** ⏳ GATED Apr 7 — Implement BDL adapter, migrate all MLB odds/injuries/schedule consumers.
*   **Phase 3: API Modularization** ⏳ DEFERRED — Decompose `main.py` into routers after MLB module is live.
*   **Phase 4: Performance & Calibration Ops** ⏳ DEFERRED — Empirical Brier score evaluation, recommendation ROI tracking, SLO dashboards (needs 4 weeks season data).

---

## 10. Resolved Crises (Archive — Do Not Revisit)

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

Current status: ARCH-001 Phase 3 fully deployed. Backend (CBB_Betting) and Frontend (observant-benevolence) confirmed LIVE.

Active tasks:
1. Gemini UI Audit COMPLETE (see reports/GEMINI_UI_AUDIT_2026-04-01.md).
2. Deployment timeout investigated: Error 4408 (Connection initialisation timeout) was a transient CLI/websocket issue during log streaming. Server-side deployment (ID: 60982b25) was SUCCESSFUL.
3. Smoke tests PASSED: async-optimize + valuations cache.

All pre-Apr-7 Claude Code tasks are now complete. System is stable.

Active task: deploy the latest changes (ARCH-002 Phase 1 + UI cleanup).
  railway up
  Verify:
    - Fantasy lineup page: async optimize button → job polls to completion (not timeout)
    - Lineup page default date: should match today's ET date (not UTC)
    - Fantasy page: no hardcoded draft date ("2026 season active" shown)
    - Waiver page: "Add" buttons link out to Yahoo Fantasy Baseball
    - Matchup page: loads category breakdown without errors

Backend changes deployed: backend/services/job_queue_service.py (status string fixes)
Frontend changes deployed: frontend/lib/constants.ts (etTodayStr), frontend/lib/api.ts (dailyLineup ET fix + asyncOptimize query params fix)

After deploy confirms, system is in full holding pattern until Apr 7.

Do NOT edit any .py or .ts files.
Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```

### For Kimi CLI (if initiating independently)

```
You are Kimi CLI (Deep Intelligence Unit). Read this HANDOFF.md in full.

Current session context:
- ARCH-001 Phases 1, 2, 3 are all COMPLETE, DEPLOYED, and VERIFIED on Railway.
- Gemini UI Audit is COMPLETE — all High + Medium findings fixed by Claude Code.
- BallDontLie NCAAB subscription is CANCELLED. Do not call NCAAB endpoints.
- OddsAPI will be cancelled post-Apr 7. Do not build new features depending on it.
- BallDontLie GOAT (MLB) will be the new odds + enrichment provider post-Apr 7.
- EMAC-068 (CBB V9.2 recalibration) is BLOCKED until Apr 7. Do not touch Kelly math.
- MCMC calibration (B5) is COMPLETED as of Mar 30. Next: empirical Brier score validation post-4 weeks.
- K-12 spec memo (V9.2 recalibration) should be ready to hand to Claude Code on Apr 7.

Do NOT write to any production code files without an explicit Claude delegation bundle.
Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```
