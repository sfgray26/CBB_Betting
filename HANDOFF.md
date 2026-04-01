# OPERATIONAL HANDOFF â€” APRIL 1, 2026: ARCH-003 FRONTEND UI/UX REFACTOR

> **Ground truth as of April 1, 2026.** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk posture Â· `AGENTS.md` for roles Â· `HEARTBEAT.md` for loops.
> Active initiative: ARCH-003 (frontend UI/UX â€” Gemini CLI executing). ARCH-001/002 complete.
> Prior active crises: all resolved (see Â§9 archive).

---

## 0. Active Architecture Initiative â€” ARCH-003 (UI/UX REDESIGN)

**Status:** âœ… initiative COMPLETE (Apr 1). All 7 frontend tasks shipped and verified.

### ARCH-003 Objective
Fix the worst fantasy UX pain points: fractured branding, inconsistent status display, poor scan-ability on the roster page, silent loading failures on waiver wire, and raw data leaking to UI. No backend changes. No route restructuring. No new data fetches unless strictly for UI state.

### HARD BOUNDARY â€” Gemini CLI Must Not Cross
- **NO** changes to any `.py` file
- **NO** changes to API endpoints, schemas, or database queries
- **NO** new `useQuery` hooks fetching from different endpoints
- **NO** changes to business logic (Kelly math, lineup optimization, projection scoring)
- **NO** changes to `api.ts` data-fetching functions
- TypeScript must continue to pass after every task

---

### Task F1 â€” Context-Aware Sidebar Branding
**File:** `frontend/components/layout/sidebar.tsx`

**Current issue:** "CBB EDGE" logo and Portfolio DD/Exp bottom panel show on `/fantasy/*` routes. Fantasy users see betting metrics they don't care about; the app feels like it's for someone else.

**Frontend-only change:**
- Use `usePathname()` (already imported) to detect `pathname.startsWith('/fantasy')`
- On fantasy routes: replace logo text "CBB EDGE" â†’ "FANTASY BASEBALL" (or "Fantasy HQ")
- On fantasy routes: hide the Portfolio DD/Exp bottom panel entirely (no alternative panel needed â€” just remove it from the DOM)
- On betting routes: behavior unchanged

**Acceptance criteria:**
- Navigating to `/fantasy/lineup` shows "FANTASY BASEBALL" in the sidebar logo area
- Portfolio DD/Exp chip is not visible on any `/fantasy/*` route
- Navigating to `/dashboard` or `/analytics` still shows "CBB EDGE" + Portfolio panel

---

### Task F2 â€” Centralized StatusBadge Component
**File:** `frontend/components/shared/status-badge.tsx` (new file)

**Current issue:** `statusBadge()` is duplicated inline in `lineup/page.tsx` and `roster/page.tsx` with inconsistent colors and missing status values. Some statuses fall through to raw strings.

**Frontend-only change:**
Create a shared `<StatusBadge status={string} />` React component with this exact color map:
- **Green** (`bg-emerald-500/15 text-emerald-400`): `ACTIVE`, `Active`, `active`, empty/null (treat as active)
- **Yellow** (`bg-amber-500/15 text-amber-400`): `DTD`, `QUESTIONABLE`, `PROBABLE`, `Probable`
- **Red** (`bg-rose-500/15 text-rose-400`): `IL`, `IL10`, `IL60`, `OUT`, `DL`, `IR`
- **Gray** (`bg-zinc-700 text-zinc-400`): `UNKNOWN`, `NO_START`, `NA`, `BENCH`, any unrecognized value

Display text rules:
- `IL10` â†’ "IL-10", `IL60` â†’ "IL-60"
- `NO_START` / `UNKNOWN` â†’ "NO START"
- `DTD` â†’ "DTD"
- Unrecognized â†’ show the raw value uppercased (not silently hide it)

**Then replace** the inline `statusBadge()` function in both `lineup/page.tsx` and `roster/page.tsx` with `<StatusBadge />`.

**Acceptance criteria:**
- Status badges have consistent colors on both lineup and roster pages
- No raw status strings visible to users (all values map to a readable label)
- No duplicate `statusBadge` function exists in the codebase after this task

---

### Task F3 â€” Roster Page: Status Column First + Icons
**File:** `frontend/app/(dashboard)/fantasy/roster/page.tsx`

**Current issue:** Status column is buried to the right, center-aligned, not visually distinct. Elite fantasy players scan for injuries first; current layout buries this.

**Frontend-only change:**
- Move the Status column to be the **second column** (immediately after player name)
- Add a Lucide icon inside the badge using `StatusBadge` from F2:
  - Active: `Activity` icon (green)
  - DTD/QUESTIONABLE: `AlertTriangle` icon (yellow)  
  - IL/OUT: `XCircle` icon (red)
  - Unknown/NA: no icon
- Keep all other columns in their current order

**Acceptance criteria:**
- Status is the second visible column on the roster page
- Icons appear inside status badges
- No layout breakage on mobile-width containers

---

### Task F4 â€” Status Tooltips on Lineup Page
**File:** `frontend/app/(dashboard)/fantasy/lineup/page.tsx`

**Current issue:** Question mark icons appear next to player names with no explanation. Status values are cryptic without context.

**Frontend-only change:**
- Wrap `<StatusBadge />` with `@radix-ui/react-tooltip` (already in `package.json`) â€” no new dependency
- Tooltip content for each status tier:
  - Active: "In the lineup / no injury designation"
  - DTD: "Day-to-Day â€” monitor before lock"
  - QUESTIONABLE/PROBABLE: "Listed as [status] â€” check beat reporters"
  - IL-10/IL-60: "On the [10/60]-day Injured List"
  - OUT: "Out â€” do not start"
  - NO START: "No game today"
  - Unknown: "Status unavailable from Yahoo"
- Remove any existing question mark / unknown icons that have no explanation

**Acceptance criteria:**
- Hovering any status badge shows a tooltip with a plain-English description
- No unexplained `?` or icon without hover text
- Tooltip uses `sideOffset={4}` for consistent positioning

---

### Task F5 â€” Waiver Wire: Loading Timeout + Empty State
**File:** `frontend/app/(dashboard)/fantasy/waiver/page.tsx`

**Current issue:** Infinite skeleton when Yahoo API is slow. No empty state. Users can't tell if data is loading vs. missing.

**Frontend-only change:**
- Add a `isStuck` state: set to `true` after 15 seconds if `isLoading` is still `true`
- When `isStuck`: replace skeleton with message: *"Taking longer than expected. Yahoo's API may be slow â€” try refreshing."* + a "Retry" button that calls `refetch()`
- When `!isLoading && players.length === 0`: show empty state: *"No waiver targets found. Projections update daily after 6 AM ET."*
- Use `useEffect` + `setTimeout` for the 15-second trigger; clear timeout on unmount and when `isLoading` becomes `false`

**Acceptance criteria:**
- After 15 seconds of loading, skeleton is replaced by the "Taking longer" message + Retry button
- When data returns empty array, empty state message appears (not infinite skeleton)
- Retry button successfully re-triggers the query

---

### Task F6 â€” Matchup Page: Suppress Impossible Stat Values
**File:** `frontend/app/(dashboard)/fantasy/matchup/page.tsx`

**Current issue:** Backend occasionally returns negative values (e.g., `-1 GS`). These display as raw negative numbers, which look like bugs to users.

**Frontend-only change:**
- In the `formatVal()` function, add a guard: if the value is a number and is negative, return `"â€”"` instead of the raw value
- Apply this guard to counting stats only (GS, W, SV, K, HR, RBI, R, SB). Ratio stats (ERA, WHIP) can legitimately be 0 but not negative â€” apply same rule.

**Acceptance criteria:**
- `-1 GS` displays as `â€”` in the matchup table
- No negative numbers visible anywhere in the matchup stat breakdown
- Valid zero values (e.g., `0 HR`) still display as `0`

---

### Task F7 â€” Category "57" Fallback Label
**File:** `frontend/app/(dashboard)/fantasy/matchup/page.tsx`

**Current issue:** When `STAT_LABELS["57"]` is undefined, the matchup table shows the raw `"57"` as the column header. This looks broken.

**Frontend-only change:**
- Update the display fallback in the category rendering from `` `${cat}` `` to `` `Cat. ${cat}` `` â€” so unmapped IDs show "Cat. 57" instead of "57"
- Do **not** add a mapping for "57" to STAT_LABELS â€” the correct label is unconfirmed and must be verified via Yahoo API first (backend task, not this session)

**Acceptance criteria:**
- Unmapped category IDs display as "Cat. 57" (or "Cat. 85") not the raw number
- The `STAT_LABELS` object is not modified

---

### ARCH-003 Verification Checklist
- [x] `/fantasy/lineup`: logo shows "FANTASY BASEBALL", Portfolio panel hidden
- [x] `/fantasy/lineup`: status badges are color-coded (green/yellow/red/gray)
- [x] `/fantasy/lineup`: hovering a status badge shows a tooltip
- [x] `/fantasy/lineup`: no unexplained `?` icons
- [x] `/fantasy/roster`: status column is second from left, with icons
- [x] `/fantasy/roster`: status badges use same colors as lineup page
- [x] `/fantasy/waiver`: after 15s loading â†’ "Taking longer" message + Retry button
- [x] `/fantasy/waiver`: empty data â†’ empty state message (not skeleton)
- [x] `/fantasy/matchup`: negative stat values display as `â€”`
- [x] `/fantasy/matchup`: unmapped categories show "Cat. N" not raw number
- [x] TypeScript type-check passes: `npx tsc --noEmit`
- [x] `/dashboard` and betting routes: sidebar unchanged (CBB EDGE + Portfolio panel visible)

---

## 0b. ARCH-001 & ARCH-002 (COMPLETE)

| Initiative | Status | Purpose |
|------------|--------|---------|
| **ARCH-001** | âœ… COMPLETE | Contract Layer + Async Job Queue + Phase 3 Integration. |
| **ARCH-002** | âœ… COMPLETE | Reliability Roadmap: Status normalization and BDL migration plan. |

---

## 0c. KIMI K-13: UI/UX UAT Analysis â€” COMPLETE (April 1, 2026)

**Status:** Analysis complete. Detailed report at `reports/KIMI_UAT_ANALYSIS_2026-04-01.md`

### Summary for Claude Code

Kimi CLI has completed a deep root-cause analysis of the Executive UI/UX UAT Report. This analysis is **independent of and complementary to** ARCH-003 â€” while ARCH-003 addresses structural frontend architecture, K-13 addresses specific data/logic/UX bugs identified in user acceptance testing.

### Critical Findings (P0/P1)

| Issue | Location | Root Cause | Fix Complexity |
|-------|----------|------------|----------------|
| **Category "57" leaking** | `constants.ts`, `main.py` | Deliberately excluded mapping, but ID displays raw | Low |
| **Injury names concatenated** | `yahoo_client_resilient.py` | Yahoo API parsing bug ("Jason Adam Quadriceps") | Low |
| **Platform identity crisis** | `sidebar.tsx`, `header.tsx` | Hardcoded "CBB EDGE" branding on fantasy routes | Medium |
| **Status badges inconsistent** | `lineup/page.tsx`, `roster/page.tsx` | No centralized status normalization | Medium |
| **Impossible stats (-1 GS)** | `main.py` matchup endpoint | Missing validation on Yahoo API data | Low |

### Key Technical Findings

1. **Category "57"**: Confirmed in `frontend/lib/constants.ts` line 31 and `backend/main.py` lines 5465-5466 as deliberately excluded. Requires Yahoo API verification to determine correct mapping (likely "BB" - Walks).

2. **Name Concatenation**: Yahoo's API occasionally returns injury info appended to player names. Requires regex sanitization in `_parse_player()`.

3. **Identity Crisis**: The sidebar (lines 118-121, 172-185) shows betting portfolio metrics (DD/Exp) even on fantasy routes. Needs route-aware conditional rendering.

4. **Status Warnings**: Yahoo returns boolean `status: False` which schemas now handle (schemas.py lines 495-501), but frontend badge mapping is inconsistent across pages.

### Implementation Guidance

- **Backend changes**: Low-risk, targeted fixes to data parsing and validation
- **Frontend changes**: Aligns with ARCH-003's goal of standardizing UI components (status badges, table skeletons)
- **Priority**: P0/P1 issues can be fixed independently of ARCH-003's larger refactor

### Files Requiring Modification

**Backend:**
- `backend/main.py` (stat_id mapping, stat validation)
- `backend/fantasy_baseball/yahoo_client_resilient.py` (name sanitization)
- `backend/services/dashboard_service.py` (timestamp timezone)

**Frontend:**
- `frontend/lib/constants.ts` (STAT_LABELS update)
- `frontend/components/layout/sidebar.tsx` (context-aware branding)
- `frontend/app/(dashboard)/fantasy/roster/page.tsx` (status column)
- `frontend/app/(dashboard)/fantasy/lineup/page.tsx` (status badges)
- `frontend/app/(dashboard)/fantasy/waiver/page.tsx` (loading states)

---

## 1. Sign-Off Log

| Initiative | Approved By | Date | Notes |
|------------|-------------|------|-------|
| ARCH-001 | Claude Code | Mar 31 | Contract layer + async job queue |
| ARCH-002 | Claude Code | Mar 31 | Reliability contracts Phase 1 |
| ARCH-003 | Claude Code | Apr 1 | Frontend-only UI/UX refactor (F1â€“F7); Gemini executing |
| K-14 to K-18 | Claude Code | Apr 1 | Kimi research tasks; no code changes authorized |

---

## 2. Data Provider Strategy â€” LOCKED DECISIONS

### SUBSCRIPTIONS

| Provider | Status | Action |
|----------|--------|--------|
| **BallDontLie GOAT (NCAAB)** | âŒ CANCELLED | CBB season is over. Do NOT renew. |
| **OddsAPI Champion** | â³ CANCEL AFTER APR 7 | Still needed for CBB tournament. Cancel once bracket concludes. |
| **BallDontLie GOAT (MLB)** | âœ… SUBSCRIBE AFTER APR 7 | Replaces OddsAPI for all MLB use cases. $39.99/mo. |

**Net cost after transition: $39.99/mo (down from $49/mo OddsAPI Champion). Saves ~$108/yr.**

---

### WHY BALLDONTLIE FOR MLB (not OddsAPI)

1. **Unified stats + odds in one API** â€” eliminates the raw OddsAPI call in `mlb_analysis._fetch_mlb_odds()` and the raw call in `daily_ingestion._poll_mlb_odds()`. Both are currently unabstracted stubs with no circuit breaker.
2. **Webhooks** â€” 125+ MLB event types. Enables live game events for betting timing without polling.
3. **600 req/min rate limit** â€” vs OddsAPI's 90,000/month call budget (polling only).
4. **MCP server** â€” 250+ endpoints compatible with Claude Code agent workflows.
5. **Bookmaker coverage tradeoff** â€” BDL covers ~15-20 books vs OddsAPI's 40+. Pinnacle IS included (confirmed in `balldontlie.py` `PREFERRED_BOOKS`). Sharp consensus logic is preserved.

### WHY KEEP PYBASEBALL / STATCAST (do not replace with BDL)

BallDontLie does not expose Statcast-tier metrics: **xwOBA, barrel%, exit velocity, hard-hit%**. These are the core of the fantasy projection engine (`statcast_ingestion.py`, `pybaseball_loader.py`). Keep pybaseball for Statcast. Add BDL for:
- Real-time injury feed
- Box scores (live game data)
- Live event webhooks for betting triggers

---

## 2. Implementation Plan â€” Post-Apr 7

### Phase 1: Cancel OddsAPI, Subscribe BDL GOAT (MLB) â€” Day 1 after tournament

**Manual steps (human action required):**
1. Cancel OddsAPI Champion subscription.
2. Subscribe to BallDontLie GOAT ($39.99/mo).
3. Set `BALLDONTLIE_API_KEY` env var in Railway (already present â€” update to MLB-tier key if different).
4. Remove `THE_ODDS_API_KEY` from Railway env after confirming no CBB code paths call it.

### Phase 2: Expand `balldontlie.py` for MLB â€” Claude Code task

**File:** `backend/services/balldontlie.py`

Current state: NCAAB-only (`/ncaab/v1/` prefix, TOURNAMENT_SEASON = 2025).

Required additions:
- Add `MLB_PREFIX = "/mlb/v1"` constant alongside `NCAAB_PREFIX`
- Add `get_mlb_games(date)` â€” fetch today's schedule
- Add `get_mlb_odds(date)` â€” fetch moneyline/runline/totals per game
- Add `get_mlb_player_stats(season, player_ids)` â€” season batting/pitching stats
- Add `get_mlb_injuries()` â€” active IL list
- Add `get_mlb_box_score(game_id)` â€” live/final box score

The existing client structure (session headers, `_get()`, `_paginate()`, circuit breaker pattern) is solid â€” replicate it for MLB endpoints. Do NOT break the existing NCAAB methods.

### Phase 3: Migrate MLB Odds Callers â€” Claude Code task

**Two dirty raw-OddsAPI callers to fix:**

| File | Method | Current Problem | Fix |
|------|--------|-----------------|-----|
| `backend/services/mlb_analysis.py` | `_fetch_mlb_odds()` | Raw `requests.get` to OddsAPI, no circuit breaker, no sharp consensus logic | Replace with `get_bdl_client().get_mlb_odds(date)` |
| `backend/services/daily_ingestion.py` | `_poll_mlb_odds()` | Same â€” raw OddsAPI call, not using `OddsAPIClient` | Replace with BDL call; preserve advisory lock pattern |

Both callers already have graceful degradation (`return {}` / `return {"status": "skipped"}` on failure), so the swap is low-risk.

### Phase 4: Add BDL as Fantasy Enrichment Feed â€” Claude Code task

**Not a replacement â€” additive only.** Existing pybaseball/Statcast pipeline stays untouched.

Add to `daily_ingestion.py` scheduler:
- `_poll_mlb_injuries()` job â€” calls `get_mlb_injuries()`, writes to a new `PlayerInjuryStatus` table or updates `PlayerDailyMetric.injury_status`
- `_ingest_mlb_box_scores()` job â€” post-game box scores for rolling stat updates

This feeds the fantasy dashboard's injury display (currently sourced from Yahoo only, which lags by hours).

---

## 3. Current Technical State

| Component | Status | Notes |
|-----------|--------|-------|
| **DB Migrations v9/v10** | âœ… LIVE | Chained into Dockerfile CMD; `user_preferences` table confirmed present |
| **Yahoo client** | âœ… CONSOLIDATED | Single file: `yahoo_client_resilient.py`. Base class + resilient layer unified. |
| **Roster endpoint (`/api/fantasy/roster`)** | âœ… LIVE | 200 OK. |
| **Matchup endpoint (`/api/fantasy/matchup`)** | âœ… LIVE | Team mapping fixed. |
| **SSE stream (`/api/fantasy/dashboard/stream`)** | âœ… LIVE | `StreamingResponse`, `text/event-stream`, 60s interval. No `sse-starlette` dep. |
| **Matchup enrichment** | âœ… LIVE | `opponent_record` from standings. `*_projected_categories` from `PlayerDailyMetric`. |
| **CircuitBreaker** | âœ… FIXED | Catches `Exception` (not just `expected_exception`) so all error types trip the breaker. |
| **Weather integration** | âœ… LIVE | Provider: OpenWeatherMap (`OPENWEATHER_API_KEY` set). |
| **OR-Tools (Railway)** | âœ… LIVE | Installed via requirements.txt. |
| **Streamlit** | âœ… RETIRED | `dashboard/` untouched. Next.js is canonical UI. |
| **Test suite** | âœ… STABLE | 1199+ pass. |
| **RP-as-SP bug** | âœ… FIXED (Mar 29) | `pitcher_slot == "SP"` guard in `_get_probable_pitchers`. |
| **Yahoo stat category IDs** | âœ… FIXED (Mar 29) | `_YAHOO_STAT_FALLBACK` dict + all 22 frontend `STAT_LABELS`. |
| **UAT P0: No-game START** | âœ… FIXED (Mar 30) | Post-optimizer override loop demotes `START` â†’ `BENCH` when `opponent` is empty. |
| **UAT P1: SP score 0.000** | âœ… FIXED (Mar 30) | `p.sp_score === 0 ? 'â€”' : p.sp_score.toFixed(3)` in lineup page. |
| **UAT P1: UNKNOWN badge** | âœ… FIXED (Mar 30) | `FALLBACK_LABELS` map in `statusBadge()` â€” UNKNOWN/NO_START â†’ "NO START", RP â†’ "RELIEVER". |
| **UAT P1: Streamlit sidebar link** | âœ… FIXED (Mar 30) | Removed `localhost:8501` link from `sidebar.tsx` (Streamlit retired). |
| **UAT P1: Raw Pydantic errors** | âœ… FIXED (Mar 30) | Warning banner filters `validation error` / `Traceback` strings. |
| **UAT: Global "Dashboard" header** | âœ… FIXED (Mar 30) | Fantasy routes added to `PAGE_TITLES` in `header.tsx`. |
| **UAT: "Invalid Date ET"** | âœ… FIXED (Mar 30) | Null guard on `dashboard.timestamp` in dashboard page. |
| **Yahoo flatten_entry last-wins bug** | âœ… FIXED (Mar 30) | `if not stats_raw:` guard in `flatten_entry` â€” takes only first `team_stats` block. |
| **`_injury_lookup` bool crash** | âœ… FIXED (Mar 30) | `isinstance(p.get("status"), str)` guard â€” rejects Yahoo `status: False/True`. |
| **`fetch_mlb_odds` coverage logging** | âœ… FIXED (Mar 30) | Logs game list + warns on 0-game response for Railway diagnostics. |
| **MCMC Simulator** | âœ… LIVE (Mar 30) | Calibrated and wired into `_get_matchup_preview()`. New `mcmc_calibration.py` converts Yahoo rosters to MCMC format using player_board + PlayerDailyMetric z-scores. Returns win_probability + category advantages/disadvantages.
| **CBB V9.2 recalibration** | â¸ BLOCKED | EMAC-068 â€” SNR/integrity scalar stacking correction. Do NOT touch Kelly math until Apr 7. |
| **`balldontlie.py`** | âš ï¸ NCAAB-ONLY | Needs MLB endpoint expansion post-Apr 7 (see Â§2 Phase 2). |
| **`mlb_analysis._fetch_mlb_odds()`** | âš ï¸ DIRTY | Raw OddsAPI call â€” no circuit breaker, no abstraction. Migrate to BDL post-Apr 7. |
| **`daily_ingestion._poll_mlb_odds()`** | âš ï¸ DIRTY | Same â€” raw OddsAPI call. Migrate to BDL post-Apr 7. |
| **BDL NCAAB subscription** | âŒ CANCELLED | CBB season over. `balldontlie.py` NCAAB methods will 401 â€” do not call them. |
| **Yahoo token over-refresh** | âœ… FIXED (Mar 31) | Singleton via `get_yahoo_client()` / `get_resilient_yahoo_client()` â€” token refreshed once per process, not per request |
| **ProjectionsLoader CSV re-read** | âœ… FIXED (Mar 31) | `@lru_cache(maxsize=1)` on `load_full_board()`; force reload via `POST /admin/fantasy/reload-board` |
| **ARCH-001 Phase 3: Frontend Integration** | âœ… VERIFIED | Lineup page reads valuations cache, async optimize polling, skeletons. |
| **ARCH-003 F1â€“F7 (UI Refactor)** | âœ… COMPLETE (Apr 1) | Shared StatusBadge, context-aware sidebar, waiver timeout, matchup fixes. |
| **Frontend Build & Lint Pass** | âœ… CLEAN | Fixed syntax errors in `lineup/page.tsx`, prop mismatch in `fantasy/page.tsx`, and UTF-8 encoding in `status-badge.tsx`. Verified with local `npm run build`. |
| **Railway Operations** | âœ… COMPLETE (Apr 1) | `INTEGRITY_SWEEP` disabled, `ENABLE_MLB_ANALYSIS` + `INGESTION` enabled. |
| **asyncOptimizeLineup (api.ts)** | âœ… FIXED | Contract mismatch fixed (query params, not body). Smoke-tested on Railway. |
| **getPlayerValuations (api.ts)** | âœ… VERIFIED | Reading from cache (v12) with empty/stale degradation. |
| **`todayStr()` ET anchor** | âœ… FIXED (Apr 1) | `toLocaleDateString('en-CA', {timeZone: 'America/New_York'})` â€” West Coast users no longer see wrong date. |
| **matchup `loading.tsx` / `error.tsx`** | ✅ FIXED (Apr 1) | Created both — skeleton + error UI with retry button. |
| **async-optimize Pydantic crash** | ✅ FIXED (Apr 1) | `_dispatch_job` now passes `target_date`+`risk_tolerance` directly; worker fetches Yahoo roster at execution time via `get_resilient_yahoo_client()` + `solve_smart_lineup()`. 3 bugs fixed. |
| **Name concat bug** | ✅ FIXED (Apr 1) | Regex in `_parse_player()` strips body-part keywords from player names. |
| **Stat ID 57/85 unmapped** | ✅ FIXED (Apr 1) | `_YAHOO_STAT_FALLBACK` now maps 57→BB, 85→OBP; `constants.ts` adds '57'→'Walks'. |
| **Negative stat values (-1 GS)** | ✅ FIXED (Apr 1) | `_extract_team_stats()` clamps `float(val) < 0` → “0”. |
| **Dashboard timestamp UTC** | ✅ FIXED (Apr 1) | `dashboard_service.py` lines 206 + 827 now use `datetime.now(ZoneInfo(“America/New_York”))`. |
| **In-season projection pipeline** | ⚠️ CRITICAL GAP | Pre-season CSVs at 100% weight while season is live. No FanGraphs RoS, no ensemble blender. K-19 spec in progress. |

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

### Immediate (Before Apr 7) â€” COMPLETE
1. **Low-severity UI cleanup** âœ… â€” all 5 items done (Apr 1):
   - `fantasy/page.tsx`: hardcoded draft date removed; `DraftBoardTab` text loader replaced with table skeleton
   - `waiver/page.tsx`: disabled "Add" button â†’ Yahoo deep link (`baseball.fantasysports.yahoo.com`); rec pulse divs â†’ card-shaped skeletons
   - `frontend/lib/constants.ts`: created â€” `STAT_LABELS`, `RATIO_STATS`, `LOWER_IS_BETTER` exported
   - `matchup/page.tsx`: imports from `@/lib/constants` instead of local duplication
   - TypeScript passes after all changes

2. **Historical MCMC validation setup** â€” deferred post-Apr 7. Needs 4 weeks of season data first. Script will read H2H outcomes from Yahoo via `/api/fantasy/matchup` history and compute Brier score against `execution_decisions.win_probability`.

### April 7+ â€” EMAC-068 Unblocks
3. **CBB V9.2 recalibration** â€” Kimi has the K-12 spec memo ready. Fix SNR/integrity scalar stacking that makes the effective Kelly divisor ~3.4Ã— instead of the calibrated 2.0Ã—. This is the primary lever to improve the CBB win record. Do NOT touch any Kelly math before Apr 7.

4. **Cancel OddsAPI, subscribe BDL GOAT MLB** â€” Day 1 action after tournament concludes. Manual steps documented in Â§2 Phase 1. Claude Code then expands `balldontlie.py` with `/mlb/v1/` endpoints and migrates the two raw OddsAPI callers (Â§2 Phases 2-3). Advisory locks 100_012 + 100_013 reserved.

5. **MLB Betting Module** â€” After BDL is wired and tested. `mlb_analysis.py` is at stub level. Full implementation of edge calculation, Kelly sizing, and alert generation. Modelled on the CBB pipeline but using `mlb_analysis.py` as the entry point instead of `betting_model.py`.

6. **MCMC empirical calibration** â€” After 4 weeks of season data, run the Brier score script and adjust `win_probability` output scaling if needed.

---

## 6. Architecture Decisions (Locked)

| Decision | Ruling | Reason |
|----------|--------|--------|
| Yahoo client split-brain | ELIMINATED | Single file: `yahoo_client_resilient.py` |
| Streamlit | RETIRED | Next.js only â€” never touch `dashboard/` |
| `openclaw_briefs.py` (old) | DELETED | `_improved` is canonical |
| Dashboard refresh strategy | SSE (IMPLEMENTED) | `StreamingResponse` text/event-stream. No sse-starlette dep. |
| Weather provider | OpenWeatherMap (LOCKED) | `OPENWEATHER_API_KEY` set |
| Test file location | `tests/` only | No test files in `backend/` subdirs |
| CBB recalibration | BLOCKED until Apr 7 | EMAC-068 â€” do not touch Kelly math before then |
| SSE keep-alive | `: keep-alive\n\n` comment line | Prevents Railway/nginx from closing idle SSE connections |
| Odds provider â€” CBB | OddsAPI â†’ CANCEL post-tournament | Battle-tested for CBB; BDL NCAAB subscription already cancelled |
| Odds provider â€” MLB | BDL GOAT (post-Apr 7) | Unified stats+odds, webhooks, lower cost. Raw OddsAPI callers in mlb_analysis + daily_ingestion are stubs â€” low-risk swap |
| Stats provider â€” Statcast | pybaseball (LOCKED) | BDL does not expose xwOBA/barrel%/exit velocity. Do not replace. |
| Stats provider â€” injuries/box scores | BDL (additive, post-Apr 7) | Supplements Yahoo injury feed which lags by hours |

---

## 7. Delegation Bundles

### GEMINI CLI (Ops) — ACTIVE: Deploy Session 3 Fixes + UI Re-Audit

**ARCH-003 F1–F7: COMPLETE.** New tasks for this session:

#### G-1: Deploy Session 3 Backend Fixes
```
railway up
```
Verify after deploy:
- `/api/fantasy/lineup/async-optimize` — submit a job, poll it to completion (no Pydantic crash)
- `/api/fantasy/roster` — player names have no injury suffixes (“Jason Adam” not “Jason Adam Quadriceps”)
- `/api/fantasy/matchup` — no negative stat values visible
- Dashboard timestamp shows ET time (e.g., “1:30 PM ET” not UTC offset)

#### G-2: Live Yahoo API Response Capture
**Purpose:** Claude needs exact JSON structures to fix remaining Yahoo parsing issues.

Run these Railway commands and save full output to `reports/GEMINI_YAHOO_RESPONSES_2026-04-01.md`:

```bash
# Roster raw response (first 3 players, full structure)
railway run python -c “
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import json
c = YahooFantasyClient()
raw = c.get_roster_raw()
print(json.dumps(raw, indent=2, default=str))
“ 2>&1 | head -200

# Free agents sample (5 players)
railway run python -c “
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import json
c = YahooFantasyClient()
fa = c.get_free_agents(count=5)
print(json.dumps(fa, indent=2, default=str))
“ 2>&1 | head -200

# League settings (stat category IDs)
railway run python -c “
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
import json
c = YahooFantasyClient()
s = c.get_league_settings()
print(json.dumps(s, indent=2, default=str))
“ 2>&1 | head -100
```

Save output to `reports/GEMINI_YAHOO_RESPONSES_2026-04-01.md`. This is the primary input for Claude’s next session.

#### G-3: UI Issues Re-Audit
**Context:** User says “countless UI issues remain” despite ARCH-003 completion. Do a fresh read-only audit of all fantasy pages and report what’s visually broken, confusing, or missing.

For each issue found:
- Page affected
- Exact description of what the user sees
- Severity (P0=blocks use, P1=confusing, P2=cosmetic)
- Whether it’s frontend-only or needs backend data fix

Save to `reports/GEMINI_UI_AUDIT2_2026-04-01.md`.

**Hard constraints:**
- Do NOT edit any `.py` file
- TypeScript must pass: `cd frontend && npx tsc --noEmit`


---

### KIMI CLI (Deep Intelligence Unit) — ACTIVE: K-19 In-Season Pipeline Spec

**K-14 through K-18: COMPLETE.** All backend fixes implemented by Claude Code (session 3). See §3 for status.

**New assignment:** K-19 — In-Season Projection Pipeline Architecture Spec

**K-14 through K-18: COMPLETE.** All backend fixes implemented by Claude Code (session 3):
- K-14: `_YAHOO_STAT_FALLBACK` now maps “57”→”BB”, “85”→”OBP”; `constants.ts` updated
- K-15: Name sanitization regex added to `_parse_player()` in `yahoo_client_resilient.py`
- K-17: `dashboard_service.py` now uses `datetime.now(ZoneInfo(“America/New_York”))` in 2 places
- K-18: Negative stat clamp added in `_extract_team_stats()` in `main.py`

---

#### K-19 — In-Season Projection Pipeline Architecture Spec
**Why Kimi:** K-17 overhaul report identified a CRITICAL gap: platform is running on pre-season CSVs while the MLB season is live. Architecture decisions required before Claude implements.

**Deliverable:** `reports/K19_INSEASON_PIPELINE_SPEC.md`

**Task:**
Read `reports/K17_INSEASON_PIPELINE_OVERHAUL.md` (your prior report) and the following files:
- `backend/fantasy_baseball/projections_loader.py` — understand current CSV load logic
- `backend/services/daily_ingestion.py` — understand scheduler and job patterns
- `backend/fantasy_baseball/statcast_ingestion.py` — understand Bayesian update path

Then produce a spec that answers these exact architectural questions for Claude:

1. **FanGraphs RoS Downloads** — which FanGraphs export URLs are publicly accessible without auth? What are the exact CSV column names for Steamer RoS batting and pitching? Is rate-limiting a concern?

2. **Ensemble Blender Design** — the prior report proposes ATC 30% / Steamer 20% / ZiPS 20% / THE BAT 15% / Statcast 15%. How should this weight decay as the season progresses? What’s the simplest schema change that stores blended projections without breaking the existing `PlayerDailyMetric` table?

3. **Yahoo ADP/Injury Polling** — what Yahoo Fantasy API endpoints return waiver wire ADP and injury designations? What’s the pagination pattern? Should this use the existing `yahoo_client_resilient.py` base class or a new lightweight async client?

4. **Statcast UTC Bug Fix Scope** — K-17 identified `statcast_ingestion.py` uses UTC for “yesterday” date. Confirm: is the fix simply replacing `datetime.utcnow().date() - timedelta(days=1)` with `(datetime.now(ZoneInfo(“America/New_York”)) - timedelta(days=1)).date()`? Are there other date anchors in that file?

5. **Lock IDs** — confirm the 4 new lock IDs from K-17 overhaul (100_012 through 100_015) don’t conflict with the advisory lock table in HANDOFF.md §4. Current next available is 100_012.

**Output format:** Numbered answers to the 5 questions above. For each: recommendation + confidence level + any blocker that requires human action (e.g., API key, FanGraphs account). Keep total under 800 words — Claude needs decisions, not prose.

**Hard constraints:** Read-only. Do NOT modify any file.

---

## 8. Yahoo Ingestion Pipeline — Current State

### Session 3 Fixes (COMPLETE — April 1, 2026)
All K-14 through K-18 backend fixes shipped. Test suite: 1091 pass, 1 pre-existing failure.

### What is working (confirmed live)
| Component | Status | Notes |
|-----------|--------|-------|
| `yahoo_client_resilient.py` | âœ… LIVE | Token refresh singleton, all roster/lineup/matchup methods |
| `statcast_ingestion.py` | âœ… LIVE | Bayesian projection update wired into daily scheduler |
| `projections_loader.py` | âœ… LIVE | Steamer CSV â†’ player_board; `@lru_cache`; reload via `/admin/fantasy/reload-board` |
| `daily_ingestion.py` waiver_scan job | âœ… LIVE | Lock 100_007 |
| `daily_ingestion.py` statcast job | âœ… LIVE | Lock 100_002 |

### Known Issues Requiring Backend Fixes (Claude Code, post-Apr 7 or earlier)
| Issue | File | Complexity | Blocked on |
|-------|------|-----------|------------|
| Name concatenation ("Jason Adam Quadriceps") | `yahoo_client_resilient.py` `_parse_player()` | Low | K-15 spec |
| Impossible stat values (-1 GS) | `main.py` matchup endpoint | Low | K-18 spec |
| Stat ID "57"/"85" unmapped | `main.py` `_YAHOO_STAT_FALLBACK` | Low | K-14 confirmation |
| `datetime.utcnow()` in dashboard_service | `dashboard_service.py` line ~206 | Low | K-17 audit |

### Kimi Research Findings Summary (All Reports Complete)

| Task | Report | Key Finding | Fix Location | Priority |
|------|--------|-------------|--------------|----------|
| **K-14** | `reports/K14_STAT57_CONFIRMATION.md` | Stat 57 = "BB" (Walks); Stat 85 = "OBP". Both deliberately excluded from `_YAHOO_STAT_FALLBACK` due to mapping uncertainty. | `main.py` line ~5505 | Add "57": "BB" after API verification |
| **K-15** | `reports/K15_NAME_CONCAT_SPEC.md` | Injury text ("Quadriceps") concatenated to names from Yahoo API position field. | `yahoo_client_resilient.py` lines 914-918 in `_parse_player()` | Add regex to strip body part keywords |
| **K-16** | `reports/K16_INGESTION_AUDIT.md` | 6 jobs analyzed. Statcast uses UTC for "yesterday" (wrong during EDT). CSV columns reject entire file on mismatch. | `statcast_ingestion.py` line ~160; `projections_loader.py` | ET date anchor; partial CSV loading |
| **K-17 (Legacy)** | `reports/K17_DATETIME_AUDIT.md` | 117 `datetime.utcnow()` usages. 2 MUST FIX: dashboard timestamp (user-visible), probable pitchers date. | `dashboard_service.py` lines 206, 827 | `datetime.now(ZoneInfo("America/New_York"))` |
| **K-17 (In-Season)** | `reports/K17_INSEASON_PIPELINE_OVERHAUL.md` | **CRITICAL:** Pre-season CSV reliance is HIGH RISK. Platform needs Fangraphs RoS downloads, ensemble blender, and Yahoo ADP polling. Statcast UTC bug documented with exact fix. | Multiple files — see report Section 7 | **CRITICAL — Season is live** |
| **K-18** | `reports/K18_STAT_VALIDATION_SPEC.md` | -1 GS comes from Yahoo raw data without validation. | `main.py` line 5577 in `_extract_team_stats()` | Clamp negative values to 0 or "0" |

### In-Season Pipeline Crisis — NEW FINDING (K-17 Overhaul Report)

**Status:** The platform is operating on **pre-season projection CSVs from March 2026** while the MLB season is live. This is a **HIGH RISK** configuration.

**Critical Issues Identified:**
1. **No Fangraphs RoS Downloads** — No automated daily fetch of Steamer/ZiPS/ATC/THE BAT RoS projections
2. **No Ensemble Blender** — Projections are 100% pre-season; no Statcast Bayesian integration
3. **Statcast UTC/ET Bug** — "Yesterday" calculation uses local server time, not ET (duplicate/missed games)
4. **No Yahoo ADP/Injury Polling** — Waiver decisions use stale pre-season ADP
5. **Pre-Season CSVs at 100% Weight** — Should be decaying to 5-10% after Week 2

**New Lock IDs Reserved:**
- 100_012: `fangraphs_ros` (Daily 4 AM ET)
- 100_013: `yahoo_adp_injury` (Every 4 hours)
- 100_014: `ensemble_update` (Daily 5 AM ET)
- 100_015: `projection_freshness_check` (Every 1 hour)

**Required Data Freshness SLA:**
| Data Type | Max Age | Current Status |
|-----------|---------|----------------|
| Rest-of-Season Projections | < 12 hours | ❌ NOT IMPLEMENTED |
| Statcast Metrics | < 6 hours | ⚠️ UTC bug present |
| Yahoo ADP/Injuries | < 4 hours | ❌ Not automated |
| Ensemble Projections | < 12 hours | ❌ NOT IMPLEMENTED |

**See `reports/K17_INSEASON_PIPELINE_OVERHAUL.md` for complete:**
- Fangraphs RoS download URLs and automation script
- Ensemble blender architecture with weights (ATC 30%, Steamer 20%, etc.)
- Statcast UTC fix with exact code snippet
- Production-grade monitoring with `job_execution_log` table schema
- Mermaid architecture diagram for In-Season Pipeline v2.0
- Prioritized action items (CRITICAL/HIGH/MEDIUM/LOW)

### Backend Fixes â€” UNBLOCKED
Claude Code can now implement all fixes in one session (~1 hour):
1. **K-17 Fix:** ET timezone in `dashboard_service.py` (2 locations)
2. **K-15 Fix:** Name sanitization regex in `yahoo_client_resilient.py`
3. **K-18 Fix:** Negative stat validation guard in `main.py`
4. **K-14 Fix:** Stat ID 57/85 mapping (after live verification)

**No schema changes. All are data sanitization guards. Test suite unaffected.**

### Architecture Intent
The Yahoo ingestion pipeline feeds three surfaces:
1. **Roster page** â€” `_parse_player()` data via `/api/fantasy/roster`
2. **Matchup page** â€” `_extract_team_stats()` data via `/api/fantasy/matchup`
3. **Daily lineup optimizer** â€” `_injury_lookup` dict fed into start/bench decisions

All three are currently serving data. The bugs above cause display issues only â€” no data loss, no incorrect lineup decisions (the optimizer already guards against bad status with the `isinstance(str)` fix). Fix priority is UX, not correctness.

---

## 8b. Delegation Matrix â€” Full View

| Task | Owner | Why | Deliverable | Risk | Dependency |
|------|-------|-----|-------------|------|-----------|
| ARCH-003 F1â€“F7 (UI) | Gemini CLI | Frontend-only, no py files | 7 .tsx changes + tsc pass | Low | None |
| K-14 Stat ID "57" confirm | Kimi CLI | Long-context doc research | `K14_STAT57_CONFIRMATION.md` | None | None |
| K-15 Name concat spec | Kimi CLI | Source trace, no code change | `K15_NAME_CONCAT_SPEC.md` | None | None |
| K-16 Ingestion failure audit | Kimi CLI | Read-only pipeline analysis | `K16_INGESTION_AUDIT.md` | None | None |
| K-17 UTC datetime audit | Kimi CLI | Grep + classify across files | `K17_DATETIME_AUDIT.md` | None | None |
| K-18 Impossible stats spec | Kimi CLI | Matchup endpoint tracing | `K18_STAT_VALIDATION_SPEC.md` | None | None |
| Backend fixes (K-14/15/17/18) | Claude Code | Architectural judgment needed | Code changes in py files | Low | K-14/15/17/18 specs |
| BDL MLB expansion | Claude Code | Contract design | `balldontlie.py` MLB methods | Medium | Apr 7 + BDL subscription |
| CBB V9.2 recalibration | Claude Code | Kelly math â€” no delegation | `betting_model.py` update | High | Apr 7 (EMAC-068) |
| MCMC Brier score validation | Claude Code | Needs 4 weeks season data | Calibration script | Low | Apr 28+ |

### Hard Boundary â€” Kimi Must Not Cross
- **No edits to any `.py` file** â€” Kimi is analysis-only
- **No schema or API contract changes** â€” these require Claude sign-off
- **No speculative implementation** â€” if unsure, document the uncertainty and escalate
- **No Kelly math** â€” EMAC-068 block applies until Apr 7, full stop
- **Escalate to Claude** if any finding is ambiguous, contradictory, or requires a non-obvious tradeoff

---

## 9. Elite Advancement Plan â€” ARCH-002

The "Elite Advancement Plan" identifies 5 high-priority architectural gaps to resolve for maximum reliability and scalability.

### Gap Status

| # | Gap | Status | Fix |
|---|-----|--------|-----|
| 1 | **Async Status Contract Drift** â€” backend used `running`/`done`, frontend expected `processing`/`completed` | âœ… FIXED Mar 31 | `job_queue_service.py` status strings updated |
| 2 | **Queue Error Masking** â€” `process_pending_jobs` marked jobs `done` even on logical errors | âœ… FIXED Mar 31 | `_run_lineup_optimization` now raises; caller handles retry/fail |
| 3 | **Fragmented MLB Provider** â€” raw OddsAPI calls in `mlb_analysis.py` + `daily_ingestion.py` | â³ Post-Apr 7 | Migrate to BDL GOAT MLB after subscription activates |
| 4 | **`main.py` Monolith** â€” 6,374-line file, no domain routers | â³ Deferred | Extract routers after MLB module stabilises |
| 5 | **Date UTC Drift** â€” `api.ts` used `toISOString().slice(0,10)` (UTC) instead of ET | âœ… FIXED Mar 31 | `etTodayStr()` in `constants.ts`; `dailyLineup` default updated |

### 90-Day Roadmap (ARCH-002)

*   **Phase 1: Reliability Contracts** âœ… COMPLETE (Mar 31) â€” Gaps 1, 2, 5 fixed. Contract tests pending (low priority).
*   **Phase 2: Data Source Unification** â³ GATED Apr 7 â€” Implement BDL adapter, migrate all MLB odds/injuries/schedule consumers.
*   **Phase 3: API Modularization** â³ DEFERRED â€” Decompose `main.py` into routers after MLB module is live.
*   **Phase 4: Performance & Calibration Ops** â³ DEFERRED â€” Empirical Brier score evaluation, recommendation ROI tracking, SLO dashboards (needs 4 weeks season data).

---

## 10. Resolved Crises (Archive â€” Do Not Revisit)

| Crisis | Resolution | Date |
|--------|------------|------|
| `user_preferences` table missing | v9/v10 migrations chained into Dockerfile CMD | Mar 27 |
| Pydantic `status: False` â†’ HTTP 500 | `_parse_player` + `RosterPlayerOut` guarded with `or None` | Mar 28 |
| Matchup "Team not found" | `m.get("teams") or m.get("0", {}).get("teams", {})` | Mar 28 |
| West Coast games showing no-game | `datetime.utcnow()` â†’ `datetime.now(ZoneInfo("America/New_York"))` | Mar 28 |
| `injury_status` always None | `injury_status=p.get("status") or None` in `RosterPlayerOut` | Mar 28 |
| `_get_lineup_gaps` empty on `team_key=None` | `client.get_roster()` no-arg form | Mar 28 |
| UI routing cascade (Kimi hotfix) | Roster dedup, team key recursive parse, NaN float guard | Mar 28 |
| CircuitBreaker only counted `expected_exception` | `except Exception:` in `call()` and `call_async()` | Mar 28 |
| RP-as-SP in probable pitchers | `pitcher_slot == "SP"` guard in `_get_probable_pitchers` | Mar 29 |
| Yahoo stat category IDs returning raw numbers | `_YAHOO_STAT_FALLBACK` dict + 22-entry `STAT_LABELS` frontend map | Mar 29 |
| Yahoo `flatten_entry` last-wins bug (OBP>1, walks decimal, wins inflated) | `if not stats_raw:` guard in `flatten_entry` â€” takes only first `team_stats` block | Mar 30 |
| `_injury_lookup` Pydantic crash (`status: False` â†’ bool) | `isinstance(p.get("status"), str)` guard in `_injury_lookup` dict comprehension | Mar 30 |
| No-game players (HOU/ARI/CLE/SF) receiving START | Post-optimizer override loop: `START` â†’ `BENCH` when `opponent` is empty | Mar 30 |
| Global "Dashboard" title on all fantasy pages | Added `/fantasy/*` routes to `PAGE_TITLES` in `header.tsx` | Mar 30 |
| "Invalid Date ET" on dashboard timestamp | Null guard `dashboard.timestamp ? ... : 'N/A'` | Mar 30 |
| SP score shows 0.000 for no-start pitchers | `p.sp_score === 0 ? 'â€”' : ...` in lineup page | Mar 30 |
| UNKNOWN status badge displayed verbatim | `FALLBACK_LABELS` map â†’ "NO START"; RP â†’ "RELIEVER" | Mar 30 |
| Streamlit localhost:8501 link in production sidebar | Removed entire block from `sidebar.tsx` | Mar 30 |
| Raw Pydantic validation errors in warning banner | Filter on "validation error" / "Traceback" strings | Mar 30 |
| MCMC Simulator calibration (B5) | `mcmc_calibration.py` created; wired into `_get_matchup_preview()`; returns win_probability + category advantages | Mar 30 |

---

## HANDOFF PROMPTS

### For Gemini CLI

```
You are Gemini CLI (Ops). Read HANDOFF.md §7 (Gemini G-1/G-2/G-3) before acting.

Current status: ARCH-003 COMPLETE. Session 3 backend fixes deployed (Claude Code).
System is stable on Railway.

Active tasks (in order):
1. G-1: Deploy session 3 backend changes: railway up
   Verify: async-optimize job completes without Pydantic crash, player names clean,
           no negative stats in matchup, dashboard timestamp shows ET

2. G-2: Capture live Yahoo API JSON responses (roster raw, free agents x5, league settings)
   Save full output to: reports/GEMINI_YAHOO_RESPONSES_2026-04-01.md
   Commands are in HANDOFF.md §7 G-2 section.

3. G-3: Fresh UI re-audit — user reports “countless issues remain.”
   Read-only pass across all fantasy pages. Document what’s broken/confusing.
   Save to: reports/GEMINI_UI_AUDIT2_2026-04-01.md

HARD RULES:
- Do NOT edit any .py file
- TypeScript must pass: cd frontend && npx tsc --noEmit before any commit
- Do NOT add new API endpoints or useQuery hooks

Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```

### For Kimi CLI

```
You are Kimi CLI (Deep Intelligence Unit).
Read HANDOFF.md §7 (Kimi K-19) before acting.

Current status:
- K-14 through K-18 COMPLETE. All 4 backend fixes implemented by Claude Code (session 3).
- ARCH-003 UI refactor COMPLETE (Gemini).
- Async-optimize Pydantic crash FIXED (Claude Code, session 3).
- System stable. Season is live.

Your active task: K-19 — In-Season Projection Pipeline Architecture Spec
Details in HANDOFF.md §7. Output to: reports/K19_INSEASON_PIPELINE_SPEC.md

CRITICAL CONTEXT:
- The platform is running on pre-season projection CSVs. MLB season is live and in Week 1.
- No FanGraphs RoS downloads, no ensemble blender, no Yahoo ADP/injury polling automation.
- This is now the highest-priority unblocked research item.
- Your K-17 overhaul report identified this; K-19 is the architecture spec to fix it.

KEY CONSTRAINTS:
- EMAC-068 still BLOCKED until Apr 7: Do NOT touch Kelly math, betting_model.py
- Do NOT write to any production code file
- K-12 (V9.2 recalibration spec) should still be ready to hand to Claude on Apr 7

Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```
@'

---

## 16.4. Railway Operations Log (DevOps Lead)

| Operation | Command | Status | Date |
|-----------|---------|--------|------|
| Disable Integrity Sweep | railway variables set INTEGRITY_SWEEP_ENABLED=false | âœ… COMPLETE | Apr 1, 2026 |
| Enable MLB Analysis | railway variables set ENABLE_MLB_ANALYSIS=true | âœ… COMPLETE | Apr 1, 2026 |
| Enable Ingestion Orchestrator | railway variables set ENABLE_INGESTION_ORCHESTRATOR=true | âœ… COMPLETE | Apr 1, 2026 |
| ARCH-003 UI Refactor | Execute F1-F7 frontend changes | âœ… COMPLETE | Apr 1, 2026 |
| Frontend Build/Lint | cd frontend; npx next lint; npx tsc --noEmit | âœ… CLEAN | Apr 1, 2026 |

'@

