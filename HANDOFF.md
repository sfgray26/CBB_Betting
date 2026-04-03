# HANDOFF.md — Fantasy Baseball Platform Master Plan (In-Season 2026 Edition)

> **Date:** April 2, 2026 (updated session S8) | **Author:** Claude Code (Master Architect)
> **Risk Level:** ELEVATED — in-season with pre-season CSV fallback still active

---

## Executive Summary

The platform is live for the 2026 MLB fantasy season. The Yahoo API pipeline is functional and data is flowing to all four surfaces (roster, matchup, waiver, lineup optimizer). However, the system is operating with a structural debt: pre-season Steamer CSVs are still the primary projection source while the season is active. Kimi audits K-17 through K-23 collectively expose twelve critical bugs (data corruption, silent failures, broken UI contracts) and one existential architectural gap (stale projections at 100% weight in-season). The immediate mandate is twofold: (1) close all critical bugs — twelve are identified, eight are already fixed — and (2) implement the in-season projection pipeline before pre-season CSVs become materially wrong, which is now. Every remaining open item has a clear owner and a concrete next deliverable. No vague work items exist in this document.

### Session S9 Hotfix (Apr 3)

- Fixed backend cold-start crash in Railway: `FileNotFoundError` for `/app/frontend/lib/fantasy-stat-contract.json`.
- Root cause: `.dockerignore` excludes `frontend/`, but backend contract loader imported that file at module import time.
- Remediation: `backend/utils/fantasy_stat_contract.py` now checks both frontend and backend-local contract paths.
- Added backend runtime copy at `backend/utils/fantasy_stat_contract.json` so backend service boots even when frontend is absent from image.

### Session S10 Hotfix (Apr 3)

- Hardened lineup actuation pipeline at `PUT /api/fantasy/lineup/apply`:
  - Enforced ET date default (`America/New_York`) for apply date.
  - Added strict incoming player identifier sanitization (must resolve to `mlb.p.XXXXX`).
  - Added roster-backed position enrichment so OF payloads can resolve to LF/CF/RF eligibility when needed.
- Weather fetch compatibility fix in `backend/fantasy_baseball/weather_fetcher.py`:
  - Switched OneCall request to `data/2.5/onecall` (free-tier compatible path).
  - Added fallback to `data/2.5/weather` before degrading to estimated weather.
- Matchup payload alignment fix in `GET /api/fantasy/matchup`:
  - Team stat extraction now filters to active scoring categories from league settings to prevent OBP/K-BB column drift.
- Waiver wire reliability improvements in `GET /api/fantasy/waiver`:
  - Normalized outgoing strikeout key for frontend (`K(P)` → `K`).
  - Prevented NSV projection fallback for non-RP players.
- Added direct waiver actuation endpoint:
  - `POST /api/fantasy/waiver/add?add_player_key=...&drop_player_key=...`
  - Frontend wired to call endpoint from waiver tables (replaces Yahoo deep-link-only flow).
- Frontend lineup UX cleanup:
  - Added Debug Mode toggle to hide/show Implied Runs and Park Factor columns.
  - Smart Score now rendered on normalized 0-100 scale for readability.
  - Apply payload only sends canonical Yahoo player keys.

---

## Session S8 Checkpoint (Apr 2)

### Mission Accomplished

- Added ET-safe shared time helpers in `backend/utils/time_utils.py` and applied them to ingestion job status plus the verified fantasy lineup/statcast job hot paths.
- Activated the projection freshness guard on `GET /api/fantasy/lineup/{lineup_date}` with `force_stale=true` override support and explicit 503 error payloads.
- Reworked fallback weather estimation so hitter scoring and HR factor respond to estimated temperature rather than a fixed neutral value, and marked fallback responses with `fallback_mode=True`.
- Replaced the duplicated Yahoo/fantasy stat metadata with one shared contract in `frontend/lib/fantasy-stat-contract.json`, consumed by both frontend constants and backend services.
- Replaced process-local Fangraphs RoS dependency with a durable `projection_cache_entries` handoff path used by `fangraphs_ros`, `ensemble_update`, and the projection freshness check.
- Added regression coverage for the new freshness gate and fallback weather behavior.

### Technical State

| Area | State | Evidence |
|------|-------|----------|
| ET date anchoring | PARTIALLY HARDENED | `daily_ingestion.py` and fantasy lineup/statcast scheduled helpers now use ET helper functions |
| Lineup freshness gate | ACTIVE | `backend/main.py` blocks stale lineup requests with 503 unless `force_stale=true` |
| Weather fallback realism | ACTIVE | `weather_fetcher.py` fallback score now uses estimated temperature and exposes `fallback_mode` |
| Stat contract | CANONICALIZED | `frontend/lib/fantasy-stat-contract.json` now feeds `frontend/lib/constants.ts`, `backend/main.py`, and `backend/fantasy_baseball/category_tracker.py` |
| RoS handoff durability | ACTIVE | `daily_ingestion.py` persists Fangraphs payloads in `projection_cache_entries` via `ProjectionCacheEntry` |
| Verification | PASS | `py_compile` passed on touched files; `pytest tests/test_ingestion_orchestrator.py tests/test_fantasy_stat_contract.py tests/test_waiver_integration.py tests/test_weather_fetcher.py -q --tb=short` → 43 passed; `cd frontend && npx tsc --noEmit` passed |

### Delegation Bundles

- Claude next: A5 atomic ensemble upsert and A6 retry taxonomy.
- Gemini next: no new frontend work until Claude finishes the shared stat contract export required for stable fantasy UI semantics.
- Kimi next: no new research required for this checkpoint.

### HANDOFF PROMPTS

#### Claude Code

Implement Phase A continuation in `c:\Users\sfgra\repos\Fixed\cbb-edge`. Read `HANDOFF.md`, `ORCHESTRATION.md`, `IDENTITY.md`, and `HEARTBEAT.md` first. Continue fantasy stabilization with these tasks only: (1) convert the ensemble write path in `backend/services/daily_ingestion.py` to atomic upsert semantics with inserted/updated/skipped counters using the new durable Fangraphs cache path, and (2) split retryable vs fatal exceptions in `backend/services/job_queue_service.py`. Validate with `venv\Scripts\python -m py_compile backend\services\daily_ingestion.py backend\services\job_queue_service.py backend\main.py backend\models.py` and the relevant pytest subset. Report exact file changes, verification output summary, and any migration requirement.

#### Gemini CLI

Do not edit Python. Stand by for frontend implementation after Claude lands the shared stat contract export. When unblocked, target only fantasy frontend files and verify with `cd frontend && npx tsc --noEmit` before any deploy.

### Architect Review Queue

- Decide whether the freshness gate should remain hard 503 for every lineup request or be relaxed for historical dates only.
- Decide whether `fallback_mode` should surface directly in frontend weather chips once Gemini resumes UI work.

---

## Core Principles

- **Contracts first.** Every layer must publish an explicit schema. No implicit data shapes, no "should be fine" assumptions. If a Yahoo API key name changes, it fails loudly, not silently.
- **Determinism before probability.** Fix the data pipeline before tuning the model. A correct batting average beats a sophisticated model fed corrupt stats.
- **Decisions before automation.** Architect the ensemble blender on paper before writing a scheduler job. Premature automation of a bad design just runs the bad design faster.
- **Intelligence before integration.** The MCMC win probability is only as good as its inputs. Do not add a fourth projection system until the three active sources are clean and fresh.
- **Every layer exists only to make the next layer simpler — never smarter.** Ingestion produces clean rows. Scoring produces a ranked list. The UI displays it. No layer compensates for broken upstream.

---

## Major Issues Acknowledged & Mitigation Plan

| Issue | Source | Severity | Current Mitigation | Owner | Status |
|-------|--------|----------|--------------------|-------|--------|
| Name concat with injury text ("Jason Adam Quadriceps") | K-15 | HIGH | Regex strip in `_parse_player()` | Claude | FIXED S3 |
| `date.today()` UTC bias in Statcast ingestion | K-16 / K-19 | HIGH | ET anchor via `ZoneInfo` in `run_daily_ingestion()` | Claude | FIXED S4 |
| Dashboard timestamp shown in UTC to users | K-17 | HIGH | `datetime.now(ZoneInfo("America/New_York"))` at `dashboard_service.py` lines 206, 827 | Claude | FIXED S3 |
| Stat IDs 57/83/85 unmapped (BB/NSV/OBP blank) | K-14 | HIGH | `_YAHOO_STAT_FALLBACK` dict + `constants.ts` STAT_LABELS | Claude | FIXED S3 |
| Impossible negative stat values (-1 GS) | K-18 | MEDIUM | `_NON_NEGATIVE_STATS` frozenset clamp; NSB explicitly excluded | Claude | FIXED S4 |
| Waiver owned% = 0% for all players | K-20 | CRITICAL | Removed `"out": "metadata"`; lookup changed to `percent_rostered` | Claude | FIXED S4 |
| Playoffs hallucination (Week 2 shows "PLAYOFFS") | K-22 | CRITICAL | `is_playoffs = raw_playoffs and (week >= 20)` guard | Claude | FIXED S4 |
| Async lineup optimizer Pydantic crash (5 validation errors) | K-21 | CRITICAL | Worker fetches Yahoo data at execution time; correct `solve_smart_lineup()` call | Claude | FIXED S3 |
| SQL Syntax Error in Job Queue (`:result::jsonb`) | G-1 | CRITICAL | Changed to `CAST(:result AS jsonb)` in `job_queue_service.py` | Claude | FIXED S6 |
| `apiFetch` converts nested error to `[object Object]` | K-21 | CRITICAL | Parse nested detail; extract readable string | Gemini | PENDING G-5 |
| "Score" column is `smart_score`; mislabelled; can be negative | K-21 | HIGH | Rename to "Smart Score" + tooltip; PROJ falls back to `implied_runs` | Gemini | PENDING G-6/G-7 |
| K/9 value (16.20) appearing in Walks column | K-22 | HIGH | Decimal guard in `_extract_team_stats()`: float BB != int -> reject + warn | Claude | FIXED S5 |
| Missing NSV on matchup (1 save shown as 0) | K-22 | HIGH | Stat ID 83 extraction + fallback mapping audit | Claude + Gemini | PENDING G-8 |
| Stat labels missing: K/BB, GS, NSV in `constants.ts` | K-22 | MEDIUM | Add string + numeric ID mappings | Gemini | PENDING G-8 |
| Matchup score calculation wrong | K-26 | CRITICAL | `MATCHUP_DISPLAY_ONLY` set excludes IP/H_AB/GS; `LOWER_IS_BETTER` expanded with L/26/27/29; `MATCHUP_STAT_ORDER` enforces display order; ScoreBanner filters display-only before score calc | Claude | **FIXED S7** |
| PROBABLE badge shown yellow/injury (should be green/SP) | K-19 domain | MEDIUM | Map PROBABLE to emerald/STARTING | Gemini | PENDING G-10 |
| Waiver "Key Stats" blank (shows deficit analysis, not stats) | K-20 | HIGH | Added `stats: dict` to `WaiverPlayerOut`; `get_free_agents()` now passes `out=stats,percent_owned`; `_to_waiver_player()` extracts stats | Claude | FIXED S6 |
| Settings page is a read-only JSON dump | K-23 | CRITICAL | Rebuild with Switch/Select/Slider; hide z-score behind presets | Gemini | PENDING G-9 |
| Pre-season CSV at 100% weight while season is live | K-17 | CRITICAL | `fangraphs_loader.py` built with ensemble blend; needs daily_ingestion wiring (100_012 + 100_014) | Claude | IN PROGRESS S6 |
| No FanGraphs RoS download | K-17 / K-19 | CRITICAL | `fangraphs_loader.py` built + wired: `_fetch_fangraphs_ros()` job (100_012, daily 3 AM ET) populates `_ROS_CACHE`; `_update_ensemble_blend()` job (100_014, daily 5 AM ET) computes + upserts blend columns | Claude | **FIXED S7** |
| No ensemble blender (ATC/Steamer/ZiPS/THE BAT) | K-17 / K-19 | CRITICAL | `_update_ensemble_blend()` job wired (100_014, daily 5 AM ET); reads `_ROS_CACHE` or re-fetches, calls `compute_ensemble_blend()`, upserts `blend_hr/rbi/avg/era/whip` | Claude | **FIXED S7** |
| No Yahoo ADP/injury polling | K-17 / K-19 | HIGH | `get_adp_and_injury_feed()` added to `yahoo_client_resilient.py`; `_poll_yahoo_adp_injury()` job wired in `daily_ingestion.py` (100_013, every 4h) | Claude | **FIXED S7** |
| No projection freshness alerting | K-16 / K-17 | HIGH | `_check_projection_freshness()` implemented and `GET /api/fantasy/lineup/{lineup_date}` now blocks on violations unless `force_stale=true`; report stored in `_job_status` | Claude | **FIXED S8** |
| CSV loader rejects full file on column mismatch | K-16 | MEDIUM | Header validation warning added to both loaders; per-row errors upgraded to WARNING | Claude | FIXED S5 |
| CBB V9.2 recalibration (SNR/integrity scalar stacking) | K-12 | BLOCKED | EMAC-068 — do not touch Kelly math before Apr 7 | Claude | BLOCKED APR 7 |
| OddsAPI to BDL GOAT MLB migration | CLAUDE.md | BLOCKED | Cancel post-tournament; expand `balldontlie.py` | Claude | BLOCKED APR 7 |

---

## In-Season Data Strategy & Architecture

### Single Source of Truth: Dynamic RoS + Statcast Bayesian Ensemble

```
Layer 1: Rest-of-Season Projections (counting stats only)
  Sources: ATC 30% / THE BAT 30% / Steamer 20% / ZiPS 20%
  Fetch:   Daily 3 AM ET via cloudscraper (FanGraphs public, no auth required)
  Lock:    100_012

Layer 2: Statcast Underlying Metrics (trend modifiers ONLY — NOT counting stats)
  Sources: pybaseball (Baseball Savant) — xwOBA, Barrel%, Exit Velocity
  Use:     Boost/penalty multiplier on RoS blend — never blended with HR/RBI directly
  Lock:    100_002 (live; UTC bug fixed S4)

Layer 3: Ensemble Blender
  Schema:  5 new columns on PlayerDailyMetric: blend_hr, blend_rbi, blend_avg, blend_era, blend_whip
  Runs:    Daily 5 AM ET after RoS download completes
  Lock:    100_014

Layer 4: Yahoo ADP + Injury Feed
  Source:  YahooFantasyClient (existing base class, new pagination method)
  Cadence: Every 4 hours
  Lock:    100_013

Layer 5: Projection Freshness Guard
  Alert if: RoS > 12h, Statcast > 6h, Yahoo injuries > 4h
  Gate:    Block lineup optimization if any SLA violated
  Lock:    100_015
```

### Pre-Season CSV: Fallback Only

`projections_loader.py` Steamer CSV is retained as fallback **for players not yet in the ensemble** (call-ups, trades). It must never be blended against RoS projections. The `@lru_cache` on `load_full_board()` is acceptable for this fallback use case only.

### Data Freshness SLAs

| Data Type | Max Age | Current Status |
|-----------|---------|----------------|
| Rest-of-Season Projections | < 12 hours | NOT IMPLEMENTED |
| Statcast Metrics | < 6 hours | UTC bug fixed — otherwise live |
| Yahoo ADP / Injuries | < 4 hours | Not automated |
| Ensemble Projections | < 12 hours | NOT IMPLEMENTED |
| Player Roster / Status | < 1 hour | Live via Yahoo client |

---

## Current Pipeline State

| Job | Lock ID | Cadence | Status | Notes |
|-----|---------|---------|--------|-------|
| `statcast` | 100_002 | Daily 2 AM ET | LIVE | UTC bug fixed (S4) |
| `rolling_z` | 100_003 | Daily 3 AM ET | LIVE | M-5 fixed: WARNs when all or majority of players skipped |
| `waiver_scan` | 100_007 | Daily 6 AM ET | LIVE | Owned% fix deployed |
| `mlb_brief` | 100_008 | Daily 7 AM ET | LIVE | |
| `valuation_cache` | 100_011 | On demand + scheduled | LIVE | ARCH-001 Phase 2 |
| `mlb_odds` | 100_001 | Every 30 min | DIRTY | Raw OddsAPI — migrate to BDL Apr 7 |
| `fangraphs_ros` | 100_012 | Daily 3 AM ET | **LIVE** | `_fetch_fangraphs_ros()` — populates `_ROS_CACHE` |
| `yahoo_adp_injury` | 100_013 | Every 4 hours | **LIVE** | `_poll_yahoo_adp_injury()` — upserts status/injury to `rolling_window` |
| `ensemble_update` | 100_014 | Daily 5 AM ET | **LIVE** | `_update_ensemble_blend()` — upserts blend_hr/rbi/avg/era/whip |
| `projection_freshness_check` | 100_015 | Every 1 hour | **LIVE** | `_check_projection_freshness()` feeds active lineup SLA gate |

---

## Future Roadmap

### Phase 1 — Close All Critical Bugs (Target: Apr 3)

| # | Item | Owner | Deliverable |
|---|------|-------|-------------|
| 1.1 | Fix `apiFetch` error parsing | Gemini G-5 | `api.ts` lines 66-75 — readable error messages |
| 1.2 | Smart Score rename + tooltip + PROJ fallback | Gemini G-6/G-7 | `lineup/page.tsx` — column rename, `implied_runs` fallback |
| 1.3 | Fix stat label mappings (K/BB, GS, NSV, HLD) | Gemini G-8 | `constants.ts` — complete stat label map |
| 1.4 | Settings page rebuild | Gemini G-9 | `settings/page.tsx` — Switch/Select/Slider |
| 1.5 | PROBABLE badge green/STARTING | Gemini G-10 | `status-badge.tsx` — correct color map |
| 1.6 | WaiverPlayerOut add `stats` field | Claude | `schemas.py` + `main.py` `_to_waiver_player()` + `yahoo_client_resilient.py` `out=stats` | **DONE S6** |
| 1.7 | K/9 bleed into Walks column | Claude | `main.py` `_extract_team_stats()` — decimal type guard |
| 1.8 | CSV partial loader | Claude | `projections_loader.py` — column intersection; never reject full file |

### Phase 2 — In-Season Pipeline (Target: Apr 7)

| # | Item | Owner | Deliverable |
|---|------|-------|-------------|
| 2.1 | FanGraphs RoS downloader | Claude | `fangraphs_loader.py` — 4 systems, cloudscraper, ensemble blend | **DONE S6** |
| 2.2 | `PlayerDailyMetric` schema extension | Claude | SQL migration `add_blend_columns.sql` + model columns | **DONE S6** |
| 2.3 | Ensemble blender job (100_014) | Claude | `daily_ingestion.py` — weighted blend logic | **DONE S7** |
| 2.4 | Yahoo ADP/injury poller (100_013) | Claude | `yahoo_client_resilient.py` — pagination + DB write | **DONE S7** |
| 2.5 | Projection freshness alerting (100_015) | Claude | `daily_ingestion.py` — SLA gate, alert-only | **DONE S7** |
| 2.6 | OddsAPI to BDL GOAT MLB migration | Claude | `balldontlie.py` — `/mlb/v1/` endpoints | BLOCKED APR 7 |
| 2.7 | CBB V9.2 recalibration (EMAC-068 unlocks Apr 7) | Claude | `betting_model.py` — scalar stacking fix per K-12 | BLOCKED APR 7 |

### Phase 3 — Intelligence Upgrades (Target: Apr 21, after 3 weeks live data)

| # | Item | Owner | Deliverable |
|---|------|-------|-------------|
| 3.1 | MCMC empirical calibration (Brier score) | Claude | Script: H2H outcomes vs `win_probability` |
| 3.2 | Matchup pace/projections + remaining games | Claude + Gemini | `/api/fantasy/matchup/pace` + frontend display |
| 3.3 | Discord notification pipeline | Gemini | OAuth + per-type toggles in settings |
| 3.4 | MLB betting module (full build) | Claude | Edge calc + Kelly sizing via BDL |

---

## Delegation Matrix

| Area | Owner | Responsibility | Next Deliverable |
|------|-------|----------------|-----------------|
| Backend Python | Claude | All `.py` files, DB migrations, scheduler jobs | H-1 WaiverPlayerOut stats; Phase 2 pipeline |
| Frontend TypeScript | Gemini | All Next.js files; zero `.py` changes | G-5 through G-10 + deploy |
| Deep research / audit | Kimi | Read-only; output to `reports/K*.md` | K-24 Yahoo player_stats spec; K-25 FanGraphs column map |
| Railway DevOps | Gemini | `railway up`, smoke tests | Deploy after G-5 through G-10 complete |
| CBB recalibration | Claude | K-12 spec to V9.2 | After EMAC-068 unblocks Apr 7 |
| MLB odds migration | Claude | BDL GOAT integration | After subscription swap Apr 7 |

---

## Active Delegation Bundles

### GEMINI -- Frontend Bug Closures (G-5 through G-10)

**Guardrails:** Zero `.py` changes. TypeScript must pass (`cd frontend && npx tsc --noEmit`) after every task.
**Deploy:** After all G-tasks pass TypeScript: `railway up`. Then verify each acceptance criterion live.

| Task | File | Change | Status |
|------|------|--------|--------|
| G-5 | `frontend/lib/api.ts` lines 66-75 | Handle nested error objects in `body.detail` | ✅ FIXED |
| G-6 | `lineup/page.tsx` line 252 | Header "Score" to "Smart Score" + Radix Tooltip | ✅ FIXED |
| G-7 | `lineup/page.tsx` lines 283-293 | PROJ fallback to `implied_runs` | ✅ FIXED |
| G-8 | `frontend/lib/constants.ts` | Fix stat label mappings (Holds, K/BB, GS, NSB) | ✅ FIXED |
| G-9 | `settings/page.tsx` | Rebuild with Switch/Select/Slider + Z-Score presets | ✅ COMPLETE |
| G-10 | `status-badge.tsx` | Map PROBABLE to green "STARTING" badge | ✅ FIXED |
| G-L2 | `waiver/page.tsx` line 349 | Change maxOwned default from 90 to 50 | ✅ FIXED |

---

### KIMI -- Research to Unblock Phase 2 Backend

**Guardrails:** Read-only. No code changes. Output to `reports/K24_*.md` and `reports/K25_*.md`.

### KIMI -- Research COMPLETE

**K-24 and K-25 research delivered.** All findings documented in:
- `reports/K24_YAHOO_PLAYER_STATS_SPEC.md` — Yahoo API contract for player stats
- `reports/K25_FANGRAPHS_COLUMN_MAP.md` — FanGraphs RoS column headers

#### K-24 Findings Summary

**Recommendation for `_to_waiver_player()`:** Reuse existing `get_free_agents()` call by adding `out=stats,metadata,percent_owned` parameter. This returns stats in the same API call (no batching needed for 25 players/page).

| Stat | ID | Path |
|------|-----|------|
| HR | 7 | `player[1].player_stats.stats[n].stat.value` |
| RBI | 8 | Same structure |
| R | 6 | Same structure |
| SB | 5 | Same structure |
| AVG | 3 | Same structure |
| ERA | 26 | Same structure |
| WHIP | 27 | Same structure |

**Batch limit:** 25 player keys per call (Yahoo hard limit).

#### K-25 Findings Summary

**Column consistency:** ALL systems (ATC, THE BAT, ZiPS DC, Steamer) use identical column headers.

| Stat | Column | Notes |
|------|--------|-------|
| HR | `HR` | Consistent all systems |
| RBI | `RBI` | Consistent all systems |
| R | `R` | Consistent all systems |
| AVG | `AVG` | Consistent all systems |
| SB | `SB` | Consistent all systems |
| ERA | `ERA` | Consistent all systems |
| WHIP | `WHIP` | Consistent all systems |
| Strikeouts | `SO` | **Not "K" — all systems use "SO"** |
| Innings | `IP` | Format: "182.1" |
| Name | `Name` | Format: "Last, First" |

**Scraping:** All systems require `cloudscraper` (Cloudflare protected).

**URLs:**
- ATC: `projections.aspx?...type=atc`
- THE BAT: `projections.aspx?...type=thebat`
- ZiPS DC: `projections.aspx?...type=zipsdc`
- Steamer RoS: `projections.aspx?...type=steamerr`

---

### K-26: Matchup Category Alignment & Logic Issues — NEW CRITICAL

**Research:** `reports/K26_MATCHUP_CATEGORY_ALIGNMENT_SPEC.md`

**Problem:** The Matchup page displays categories in random order and miscalculates the overall score. The 9-6 score (or similar) is **wrong** and doesn't match Yahoo's official standings.

**Root Causes:**
1. **Non-scoring stats counted**: H/AB and IP are display-only reference stats but are being counted toward win/loss totals
2. **Missing inverse logic**: Batter K (strikeouts) should award "Win" to the *lower* value (fewer strikeouts is better for batters), but currently higher wins
3. **Random ordering**: Stats display in Yahoo API order instead of logical Batters→Pitchers sections

**Required Category Order (Batters):**
| # | Stat | Scoring? | Win Condition |
|---|------|----------|---------------|
| 1 | H/AB | ❌ Display Only | N/A |
| 2 | R | ✅ Yes | Higher |
| 3 | H | ✅ Yes | Higher |
| 4 | HR | ✅ Yes | Higher |
| 5 | RBI | ✅ Yes | Higher |
| 6 | **K** | ✅ Yes | **LOWER** ⚠️ |
| 7 | TB | ✅ Yes | Higher |
| 8 | AVG | ✅ Yes | Higher |
| 9 | OPS | ✅ Yes | Higher |
| 10 | NSB | ✅ Yes | Higher |

**Required Category Order (Pitchers):**
| # | Stat | Scoring? | Win Condition |
|---|------|----------|---------------|
| 1 | IP | ❌ Display Only | N/A |
| 2 | W | ✅ Yes | Higher |
| 3 | L | ✅ Yes | Lower |
| 4 | HR | ✅ Yes | Lower |
| 5 | K | ✅ Yes | Higher |
| 6 | ERA | ✅ Yes | Lower |
| 7 | WHIP | ✅ Yes | Lower |
| 8 | K/9 | ✅ Yes | Higher |
| 9 | QS | ✅ Yes | Higher |
| 10 | NSV | ✅ Yes | Higher |

**Files to Modify:**
- `frontend/lib/constants.ts` — Add `MATCHUP_CATEGORIES` config with ordering, section, and `lower_is_better` flags
- `frontend/app/(dashboard)/fantasy/matchup/page.tsx` — Update `MatchupTable` and `ScoreBanner` to use ordered categories and exclude display-only stats from scoring

**Key Logic Fix:**
```typescript
// In constants.ts
export const MATCHUP_CATEGORIES = {
  H_AB: { label: 'H/AB', section: 'batting', scoring: false },
  R: { label: 'Runs', section: 'batting', scoring: true, lowerIsBetter: false },
  // ... etc
  K: { label: 'Strikeouts', section: 'batting', scoring: true, lowerIsBetter: true }, // ⚠️ KEY FIX
  IP: { label: 'Innings Pitched', section: 'pitching', scoring: false },
  // ... etc
}

// In matchup/page.tsx ScoreBanner
const scoringCats = allCats.filter(cat => MATCHUP_CATEGORIES[cat]?.scoring !== false)
```

---

## Immediate Action Items

### Complete This Session (S7 — Apr 1, 2026)

| # | Item | Status |
|---|------|--------|
| H-2 | Matchup Category Alignment Fix (constants.ts + matchup/page.tsx) | ✅ DONE |
| M-5 | rolling_z silent skip alert | ✅ DONE |
| 2.3 | Ensemble blender (100_014) wired to daily_ingestion | ✅ DONE |
| 2.4 | Yahoo ADP/injury poller (100_013) | ✅ DONE |
| 2.5 | Projection freshness SLA gate (100_015) | ✅ DONE |
| 2.1 | fangraphs_ros (100_012) wired to daily_ingestion | ✅ DONE |

### Pending (Claude)

| # | Item | Dependency | File | Effort |
|---|------|-----------|------|--------|
| Phase 2.6 | OddsAPI → BDL GOAT MLB migration | Apr 7 subscription swap | `balldontlie.py` — `/mlb/v1/` endpoints | Large |
| Phase 2.7 | CBB V9.2 recalibration (EMAC-068 unlocks Apr 7) | Apr 7 | `betting_model.py` | Large |
| Phase 3.1 | Expose `projection_freshness` report via `/admin/ingestion/status` | None | `main.py` admin route | Small |
| Phase 3.2 | Upgrade freshness gate to BLOCK lineup opt when SLA violated | After 3.1 stable | `daily_ingestion.py` + `main.py` lineup route | Medium |

### S7 Completed (All closed this session)

| # | Item | File | Status |
|---|------|------|--------|
| H-2 | Matchup Category Alignment Fix | `frontend/lib/constants.ts` + `matchup/page.tsx` | ✅ DONE |
| M-5 | `rolling_z` silent skip alert | `backend/services/daily_ingestion.py` | ✅ DONE |
| 2.1 | FanGraphs RoS wired to scheduler (100_012) | `daily_ingestion.py` | ✅ DONE |
| 2.3 | Ensemble blender wired to scheduler (100_014) | `daily_ingestion.py` | ✅ DONE |
| 2.4 | Yahoo ADP/injury poller (100_013) | `yahoo_client_resilient.py` + `daily_ingestion.py` | ✅ DONE |
| 2.5 | Projection freshness SLA gate (100_015) | `daily_ingestion.py` | ✅ DONE |


## Advisory Lock Registry

| Lock ID | Job | Status |
|---------|-----|--------|
| 100_001 | mlb_odds (OddsAPI — migrate Apr 7) | LIVE / DIRTY |
| 100_002 | statcast | LIVE |
| 100_003 | rolling_z | LIVE (fragile) |
| 100_004 | cbb_ratings | LIVE |
| 100_005 | clv | LIVE |
| 100_006 | cleanup | LIVE |
| 100_007 | waiver_scan | LIVE |
| 100_008 | mlb_brief | LIVE |
| 100_009 | openclaw_perf | LIVE |
| 100_010 | openclaw_sweep | LIVE |
| 100_011 | valuation_cache | LIVE |
| 100_012 | fangraphs_ros | LIVE |
| 100_013 | yahoo_adp_injury | LIVE |
| 100_014 | ensemble_update | LIVE |
| 100_015 | projection_freshness_check | LIVE |

**Next available:** 100_016

---

## Hard Stops

| Rule | Reason |
|------|--------|
| Do NOT modify Kelly math in `betting_model.py` | EMAC-068 — blocked until Apr 7 |
| Do NOT call BDL `/ncaab/v1/` endpoints | Subscription cancelled — will 401 |
| Do NOT add OddsAPI dependencies | Phase out; BDL MLB replaces post-Apr 7 |
| Do NOT touch `dashboard/` (Streamlit) | Retired — Next.js is canonical |
| Do NOT use `datetime.utcnow()` for game times | Use `datetime.now(ZoneInfo("America/New_York"))` |
| Do NOT write test files outside `tests/` | Architecture locked |
| Do NOT blend Statcast xwOBA/Barrel% with counting stats | K-19 domain: Statcast = trend modifier only |
| Do NOT weight pre-season CSVs against RoS projections | K-17: pre-season CSV = fallback for missing players only |

---

*Handoff complete. New HANDOFF.md is clean, contract-driven, and ready for execution. All K-15-K-23 issues acknowledged and roadmap simplified.*

---

## Session 4 Summary — Gemini CLI (Ops) — Apr 1, 2026

### G-1: Deployment & Verification
- **Build Fixed:** Resolved UTF-8 encoding issue in `status-badge.tsx` that was blocking production builds.
- **Deploy:** `railway up` successful.
- **Verification:**
    - Player Names: CLEAN (no injury suffixes).
    - Matchup Stats: NO NEGATIVES (GS clamped to 0).
    - Dashboard Timestamp: ET ANCHORED (EDT/-04:00).
    - **Async-Optimize Job:** CRITICAL FAILURE. Job hangs in `processing` due to SQL Syntax Error in `job_queue_service.py` (line 124: `:result::jsonb`). Worker transaction aborts. Verified via live polling and code inspection.

### G-2: Yahoo API Capture
- Captured full roster, free agents sample, and league settings JSON.
- Saved to `reports/GEMINI_YAHOO_RESPONSES_2026-04-01.md`.
- High-fidelity data confirmed for Claude's parsing logic fixes.

### G-3: UI Re-Audit
- Fresh read-only pass across fantasy pages.
- Documented 10+ UX/UI issues in `reports/GEMINI_UI_AUDIT2_2026-04-01.md`.
- Key findings: dense tables, confusing column naming (Score vs Proj), and missing interactivity in dashboard flags.

---

## Session 7 Summary — Claude Code (Architect) — Apr 1, 2026

**All 6 items in Claude's S7 queue completed and pushed.**

### S7-1: H-2 — Matchup Category Alignment Fix

**Problem:** Matchup score was miscalculated because IP and H/AB (display-only reference stats) were counted as scoring categories, Losses (L) was not in `LOWER_IS_BETTER`, and stats displayed in random Yahoo API order.

**Files changed:**
- `frontend/lib/constants.ts` — Added `MATCHUP_DISPLAY_ONLY` set (`IP`, `H/AB`, `GS`, `21`, `50`, `62`); expanded `LOWER_IS_BETTER` to include `L`, `26`, `27`, `29`; added `MATCHUP_STAT_ORDER` array enforcing Batters→Pitchers section order
- `frontend/app/(dashboard)/fantasy/matchup/page.tsx` — `MatchupTable` now uses `orderedCats` (MATCHUP_STAT_ORDER first, unknowns appended); display-only stats show "(ref)" label and "—" in Edge column; `ScoreBanner` filters `MATCHUP_DISPLAY_ONLY` before computing score

### S7-2: M-5 — rolling_z Silent Skip Alert

**Problem:** `_calc_rolling_zscores` returned `records=0` with no log output when the season was < 7 days old and all players lacked sufficient history.

**File changed:** `backend/services/daily_ingestion.py`
- Added `skipped_insufficient_data` counter
- `logger.warning` when all or majority (>50%) of players are skipped
- `logger.debug` for minor skips (< 50%)
- **Note:** There was also an indentation bug introduced here (missing `try:` line before the rows query) — this was discovered and fixed via py_compile during S7

### S7-3: 100_012 — fangraphs_ros Wired to Scheduler

**File changed:** `backend/services/daily_ingestion.py`
- Added `fangraphs_ros: 100_012` to `LOCK_IDS`
- Added module-level `_ROS_CACHE: dict = {}` for inter-job data sharing
- Registered `_fetch_fangraphs_ros` job — daily 3 AM ET
- Implemented `_fetch_fangraphs_ros()`: calls `fetch_all_ros("bat")` + `fetch_all_ros("pit")` from `fangraphs_loader.py`, stores results in `_ROS_CACHE` with `fetched_at` timestamp

### S7-4: 100_014 — ensemble_update Wired to Scheduler

**File changed:** `backend/services/daily_ingestion.py`
- Added `ensemble_update: 100_014` to `LOCK_IDS`
- Registered `_update_ensemble_blend` job — daily 5 AM ET (after fangraphs_ros at 3 AM)
- Implemented `_update_ensemble_blend()`: reads `_ROS_CACHE` (re-fetches if stale), calls `compute_ensemble_blend()`, upserts `blend_hr/blend_rbi/blend_avg/blend_era/blend_whip` to `PlayerDailyMetric`

### S7-5: 100_013 — Yahoo ADP/Injury Poller

**Files changed:**
- `backend/fantasy_baseball/yahoo_client_resilient.py` — Added `get_adp_and_injury_feed(pages=4, count_per_page=25)`: fetches players sorted by `sort=DA` (ADP order), 4 pages × 25 players, per-page `YahooAPIError` catch for partial tolerance
- `backend/services/daily_ingestion.py` — Added `yahoo_adp_injury: 100_013`; registered `_poll_yahoo_adp_injury` job (every 4 hours); implemented method that calls client, upserts `status`/`injury_note`/`percent_owned` into `PlayerDailyMetric.rolling_window`

### S7-6: 100_015 — Projection Freshness SLA Gate

**File changed:** `backend/services/daily_ingestion.py`
- Added `projection_freshness: 100_015` to `LOCK_IDS`
- Registered `_check_projection_freshness` job (every 1 hour)
- Implemented `_check_projection_freshness()`:
  - Queries DB: latest `ensemble_blend` row (SLA: 12h) and latest `statcast` row (SLA: 6h)
  - Checks `_ROS_CACHE["fetched_at"]` in-memory (SLA: 12h)
  - `logger.warning` per violation with staleness hours
  - Stores full report in `self._job_status["projection_freshness"]` for `/admin/ingestion/status`
  - Alert-only — does NOT block anything yet (Phase 3.2 will add the hard gate)

### S7 Test Results

```
1222 passed, 4 pre-existing failures, 3 warnings
Pre-existing failures:
  - test_betting_model.py::TestExposureAccounting × 3  (no local Postgres — DB auth fails)
  - test_tournament_data.py::TestTournamentDataClient::test_cache_expired  (stale bracket cache)
```

`tests/test_ingestion_orchestrator.py::test_orchestrator_get_status_returns_all_jobs` was updated to include the 4 new job IDs.

### Next Up for Claude (post-S7)

| # | Item | When | File |
|---|------|------|------|
| Phase 3.1 | Expose `projection_freshness` report via `/admin/ingestion/status` | Any time | `main.py` admin route |
| Phase 3.2 | Upgrade freshness gate to BLOCK lineup opt when SLA violated | After 3.1 stable | `daily_ingestion.py` + lineup route |
| Phase 2.6 | OddsAPI → BDL GOAT MLB migration | Apr 7 | `balldontlie.py` |
| Phase 2.7 | CBB V9.2 recalibration (EMAC-068) | Apr 7 | `betting_model.py` |
