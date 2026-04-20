# HANDOFF.md — MLB Platform Operating Brief

> Date: April 20, 2026 | Author: Claude Code (Master Architect)
> Status: **UAT remediation session complete (Apr 20). 6 bugs fixed, 2245/2245 tests passing. Deploy needed — see Gemini bundle below.**

---

## SESSION DELTA — April 20, 2026 (UAT Remediation)

### What Was Done

UAT was run against Railway production (`uat_findings_fresh.md`): 53 PASS / 15 FAIL. Root causes diagnosed and fixed. **Critical discovery: `ConfigDict` missing from schemas.py caused the ENTIRE fantasy router to fail on import, making scoreboard/budget/optimize return 404 and explaining many observed failures.**

### Fixes Applied (Local — Need Deployment)

| File | Fix |
|------|-----|
| `backend/schemas.py` | Add `ConfigDict` import (was missing — crashed router on import). Add `FreshnessMetadata` import from contracts (resolves forward-ref null in roster response). |
| `backend/routers/fantasy.py` | Fix `decisions/status` AttributeError: `today_et()` returns `date`, not `datetime` — `now_et.date()` call removed. |
| `backend/routers/fantasy.py` | Fix NSB "X/Y" non-numeric in matchup: parse fraction to numerator. |
| `backend/routers/fantasy.py` | Fix briefing `overall_confidence` from int % to float 0-1 (divide by 100). |
| `backend/routers/fantasy.py` | Fix IL player drop guard: derive effective status from `selected_position` slot when `status` is None; add z>=6.0 superstar protection to prevent dropping elite players. |
| `backend/main.py` | Roster endpoint: add `get_players_stats_batch()` call to populate `season_stats` for all roster players. Status default changed from "playing" → "Active". |
| `scripts/uat_validator.py` | Fix field name mismatch: check `name`/`player_key` (Python field names) not `player_name`/`yahoo_player_key` (aliases). |
| `tests/test_budget_api.py` | Update test paths from `/budget` → `/api/fantasy/budget` to match renamed route. |

### Post-Fix Test Results

```
2245 passed, 3 skipped, 0 failed
```

### Remaining After Deployment

After Gemini deploys, expected UAT improvements:
- `GET /api/fantasy/scoreboard` → was 404, will be 200 (router now loads)
- `GET /api/fantasy/budget` → was 404, will be 200
- `POST /api/fantasy/roster/optimize` → was 404, will be 200
- `GET /api/fantasy/decisions/status` → was 500, will be 200
- `GET /api/fantasy/matchup` NSB → was "0/0", will be "0" (numeric)
- `GET /api/fantasy/roster` freshness → was null, will be dict
- `GET /api/fantasy/roster` season_stats → was 0%, will be >0% (Yahoo stats batch)

**Known gaps still open after deployment:**
- Waiver `matchup_opponent='TBD'` — Yahoo scoreboard parsing fragile (not fixed this session)
- Briefing `categories/starters` — depends on `category_tracker.get_category_needs()` returning data (empty when scoreboard parsing fails)
- Proxy players (Ballesteros, Antonacci, Murakami) — genuinely not in Steamer/ZiPS; stats will come from Yahoo season stats batch
- No Statcast/advanced stats on roster endpoint (future work)

---

## GEMINI DELEGATION BUNDLE — Deploy April 20 Fixes

**HANDOFF PROMPT FOR GEMINI CLI:**

```
Deploy the April 20 UAT remediation fixes to Railway.

Files changed locally (all tests passing at 2245/0 fail):
- backend/schemas.py — ConfigDict import added (critical: was crashing router)
- backend/routers/fantasy.py — decisions/status fix, NSB fix, briefing fix, IL guard
- backend/main.py — roster season_stats batch fetch added
- scripts/uat_validator.py — field name corrections
- tests/test_budget_api.py — route path updated

Steps:
1. Run: venv/Scripts/python -m py_compile backend/schemas.py backend/routers/fantasy.py backend/main.py
2. Verify all pass with "OK"
3. Run: venv/Scripts/python -m pytest tests/ -q --tb=short
4. Verify 2245 passed, 0 failed
5. git add backend/schemas.py backend/routers/fantasy.py backend/main.py backend/fantasy_baseball/daily_briefing.py scripts/uat_validator.py tests/test_budget_api.py tasks/uat_findings_fresh.md HANDOFF.md
6. git commit -m "fix(uat): fix router crash, decisions/status 500, NSB parse, IL guard, roster stats"
7. git push origin stable/cbb-prod
8. Confirm Railway auto-deploys (check Railway dashboard)
9. After deploy, run: venv/Scripts/python scripts/uat_validator.py --base-url "https://fantasy-app-production-5079.up.railway.app" --api-key "j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg" --output tasks/uat_findings_post_deploy.md
10. Report: count of PASS/FAIL/WARN. Expected: scoreboard/budget/optimize now 200, decisions/status now 200, NSB numeric.
```

---

> Previous session date: April 18, 2026 | Author: Claude Code (Master Architect)
> Status: Phase 0–4 COMPLETE. **Phase 4.5a Quality Remediation P1-P3-P5 COMPLETE.** P4 (statcast integration) remains optional. Kimi framework audit (verified) exposed 3 critical foundation gaps that must be fixed before UAT is meaningful. Phase 4.5b UAT ready to proceed. L3E deferred. Do not reopen Layer 2 except for regressions.

UI Specification Audit: reports/2026-04-17-ui-specification-contract-audit.md
Comprehensive application audit: reports/2026-04-15-comprehensive-application-audit.md
Framework quality audit: reports/2026-04-18-framework-audit.md
Raw-ingestion contract audit: reports/2026-04-05-raw-ingestion-audit.md
Historical context: HANDOFF_ARCHIVE.md

---

## Mission Accomplished

- Layer 2 certification is complete in production.
- Deployment truth is live and versioned.
- Raw ingestion audit logging is healthy.
- `probable_pitchers` is populating successfully in production.
- `park_factors` is seeded and available for DB-backed reads with fallback.
- `weather_forecasts` exists in schema but remains deferred; request-time weather is the live path.
- The Layer 2 hard gate is lifted.
- **UI Specification Contract Audit completed — authoritative field-level mapping produced, 9-phase gated implementation plan adopted.**

---

## UI Contract Authority

The document `reports/2026-04-17-ui-specification-contract-audit.md` is the authoritative mapping between the locked UI specification and backend readiness. The UI spec defines the contract; the backend serves it; not the other way around.

### Field Readiness Summary

The UI specification requires **110 fields** across 6 canonical pages plus global header and cross-cutting requirements.

| Status | Count | Percentage |
|--------|-------|------------|
| READY | 19 | 17% |
| PARTIAL | 27 | 25% |
| MISSING | 64 | 58% |

### Top 5 Blockers (Ranked by Fields Blocked)

1. **ROW projection pipeline does not exist** — Blocks 18 fields across Matchup Scoreboard, My Roster, Waiver Wire, and Streaming pages. This is the single highest-priority gap.

2. **Rolling stats cover only 9 of 18 categories** — Missing R, TB from batting; W, L, HRA, K(pitching), QS, NSV from pitching. Creates 27+ cell gaps across all player rows.

3. **Projections cover only 8 of 18 categories** — Missing H, K(B), TB, NSB, W, L, HRA, SV/NSV, HLD, QS. Affects all projection displays.

4. **Per-player games-remaining-this-week missing** — Required for ROW pipeline computation; blocks scoreboard games remaining and waiver filters.

5. **Acquisition count not tracked** — Yahoo transactions fetched but not counted or week-filtered; blocks global header acquisition display and waiver budget.

### Canonical Pages (Priority Order)

| Priority | Page | Fields Ready | Status |
|----------|------|-------------|--------|
| P1 | Matchup Scoreboard | 3 of 16 (19%) | Load-bearing page; defining features all MISSING |
| P1 | My Roster | 2 of 10 (20%) | Core roster management |
| P2 | Waiver Wire | 1 of 8 (13%) | Marginal value computation incomplete |
| P2 | Probable Pitchers / Streaming | 4 of 12 (33%) | Ratio risk and recommendations MISSING |
| P3 | Trade Analyzer | 0 of 9 (0%) | Entire page blocked |
| P3 | Season Dashboard | 0 of 6 (0%) | Projections and diagnosis incomplete |

### Operating Principle

The UI specification is now the requirements document. All downstream backend work must reference the field-level mapping in the audit report. No frontend implementation should proceed without backend readiness confirmed against this audit.

---

## Core Doctrine

The MLB data platform is now validated at Layer 2. Work proceeds through a gated 9-phase plan to achieve UI contract readiness.

The architecture remains layered:

| Layer | Name | Purpose | Status |
|------|------|---------|--------|
| 0 | Immutable Decision Contracts | Canonical contracts, schemas, IDs, and validation boundaries | **COMPLETE — Phase 0 delivered stat_contract package + 6 UI contracts** |
| 1 | Pure Stateless Intelligence | Deterministic pure functions over validated inputs | **PARTIAL — delta-to-flip and margin delivered (Phase 2 category_math.py). IP pace and acquisition count delivered (Phase 1 constraint_helpers.py). Remaining: ratio risk quantifier, category-count delta extractor** |
| 2 | Data and Adaptation | Ingestion, validation, persistence, observability, freshness, provenance | Certified Complete — 7 data gap tasks identified for Phase 1 |
| 3 | Derived Stats and Scoring | Rolling stats, player scores, context-enriched features | **COMPLETE — Phase 2 delivered: 15/18 Z-scores, ROW projector, category math. 4 greenfield categories (W, L, HR_P, NSV) deferred to Phase 2b.** L3A/L3B/L3D/L3F remain complete. |
| 4 | Decision Engines and Simulation | Lineup logic, waiver logic, matchup engines, Monte Carlo | **COMPLETE — Phase 3 delivered: H2H Monte Carlo + MCMC aligned to v2 18-cat codes, ROW→Simulation bridge, ratio risk quantifier, category-count delta.** |
| 5 | APIs and Service Presentation | FastAPI contracts, dashboards, admin views | **COMPLETE — Phase 4 delivered: 5 P1 endpoints (scoreboard, budget, roster, move, optimize) + scoreboard orchestrator + player mapper. 50 new tests.** |
| 6 | Frontend and UX | Next.js pages, interactions, polish | **GATED — Phase 5: build P1 pages after Phase 4.5 UAT passes. 15 components salvageable, 9 CBB pages to archive.** |

### Operating Rule

- **Phase 0 is COMPLETE.** stat_contract package loaded, 6 UI contracts validated, 30/30 tests passing.
- **Phase 1 is COMPLETE.** V1→V2 consumer migration + 7 data gap closures delivered, 2029 tests passing.
- **Phase 2 is COMPLETE.** 15/18 Z-scores computed, ROW projector + category math delivered, 96 new tests passing. 4 greenfield categories (W, L, HR_P, NSV) awaiting upstream data.
- **Phase 3 is COMPLETE.** H2H Monte Carlo + MCMC aligned to 18-cat v2 codes, ROW→Simulation bridge, ratio risk quantifier, category-count delta extractor.
- **Phase 4 is COMPLETE.** 5 P1 API endpoints delivered (scoreboard, budget, roster, move, optimize), scoreboard orchestrator, player mapper. 50 new tests, 2233 total passing.
- **Phase 4.5a Quality Remediation is NEXT.** Foundation gaps exposed by Kimi audit must be fixed before UAT. See "Active Workstream" for the full gap analysis and remediation plan.
- **Phase 4.5b UAT follows 4.5a.** Manual sanity-check of live API data — meaningful only after quality gaps are closed.
- Do not reopen Layer 2 as an active workstream unless a production regression is observed.
- Do not start frontend (Phase 5) until Phase 4.5b UAT passes.
- The 9-phase plan defines the sequenced path. Do not skip phases.

---

## Current Production Truth

Verified production state as of April 17, 2026 (18:00 UTC):

| Area | Current Truth | Status |
|------|---------------|--------|
| Deployment state | Fresh (Build: 2026-04-16T02:00:54) | Healthy |
| `data_ingestion_logs` | 66 rows | Healthy |
| `probable_pitchers` | 94 rows | Healthy |
| `/admin/pipeline-health` | `overall_healthy: true` | Healthy |
| `mlb_player_stats` | 7249 rows | Healthy |
| `statcast_performances` | 7408 rows | Healthy |
| `park_factors` | 27 parks seeded, DB-backed reads active | Healthy |
| `weather_forecasts` | Table exists, EMPTY (request-time weather used instead) | Deferred |
| `/admin/diagnose-scoring/layer3-freshness` | Endpoint live, 13 tests passing | Healthy |
| `/admin/diagnose-decision/pipeline-freshness` | Endpoint live, 8 tests passing | Healthy |

### Operational Interpretation

- Layer 2 is certified complete.
- Park factor authority is DB-backed with fallback (ballpark_factors.py → ParkFactor table → PARK_FACTORS constant → neutral 1.0).
- Weather context exists in schema but is NOT populated; request-time weather (weather_fetcher.py) remains the live path for consumers like smart_lineup_selector.py.
- Layer 3 scoring (player_scores) does NOT consume weather - pure rolling-window Z-score computation remains appropriate for multi-day windows.
- The production data spine is no longer the blocker.
- The next bottleneck is completing the data contracts and pipelines required by the UI specification.

---

## Layer 2 Certification Record

Final production verification:

- `probable_pitchers` row count: 94
- Latest `probable_pitchers` job result: SUCCESS (Job ID 65)
- `data_ingestion_logs` contains recent durable rows
- `/admin/pipeline-health` returns `overall_healthy: true`
- `/admin/version` exists and deployment versioning is live

Layer 2 acceptance criteria status:

1. Production is running latest repo code: PASS
2. `data_ingestion_logs` has recent durable rows: PASS
3. Health endpoints report correctly: PASS
4. `probable_pitchers` contains usable rows: PASS
5. Raw MLB source tables are fresh and internally consistent: PASS
6. `park_factors` is persisted canonically; `weather_forecasts` remains deferred by design: PASS
7. Scoring code remains pure and does not depend on persisted weather context: PASS

Layer 2 verdict: PASS

---

## Layer Status

### Layer 0 — Immutable Decision Contracts

Status: **COMPLETE — Phase 0 delivered stat_contract package + 6 UI contracts**

**Deliverables:**
- `backend/stat_contract/` package (5 modules): schema, registry, builder, loader, __init__
- `fantasy_stat_contract.json` validated and loaded at import time
- 6 UI contracts in `backend/contracts.py`:
  - `CategoryStatusTag` enum (LOCKED_WIN, LOCKED_LOSS, BUBBLE, LEANING_WIN, LEANING_LOSS)
  - `IPPaceFlag` enum (BEHIND, ON_TRACK, AHEAD)
  - `ConstraintBudget` Pydantic model
  - `FreshnessMetadata` Pydantic model
  - `CategoryStats` Pydantic model (validates 18 canonical categories)
  - `MatchupScoreboardRow` and `MatchupScoreboardResponse` Pydantic models
  - `PlayerGameContext` and `CanonicalPlayerRow` Pydantic models

**Gate 0 verification:** 30/30 tests passing. All py_compile checks passing. CategoryStats validator derives from loaded contract (no hardcoded frozensets).

### Layer 1 — Pure Stateless Intelligence

Status: **COMPLETE — All pure functions delivered across Phases 1–3**

Phase 1 delivered `classify_ip_pace()` + 6 constraint helpers. Phase 2 delivered delta-to-flip calculator + category math. Phase 3 delivered ratio risk quantifier + category-count delta extractor.

### Layer 2 — Data and Adaptation

Status: **Certified Complete — Phase 1 data gap closures delivered**

Regressions only. Do not run a new Layer 2 roadmap unless production evidence degrades.

**Phase 1 deliverables (COMPLETE):**
- V1→V2 consumer migration: `main.py`, `routers/fantasy.py`, `category_tracker.py`, `smart_lineup_selector.py` all use `backend.stat_contract` v2 codes
- Old v1 artifacts deleted: `backend/utils/fantasy_stat_contract.py`, `backend/utils/fantasy_stat_contract.json`, `tests/test_fantasy_stat_contract.py`
- 7 pure functions in `backend/services/constraint_helpers.py`: acquisition counter, IP extractor, IP pace classifier, games-remaining, standings parser, opposing-SP lookup, playing-today status
- 23 tests in `tests/test_constraint_helpers.py` + 3 migration verification tests
- Full suite: 2029 passed, 3 skipped

### Layer 3 — Derived Stats and Scoring

Status: **COMPLETE — Phase 2 expanded to 18 categories + ROW pipeline**

Historical accomplishments (preserved):
- **L3A (scoring spine)** — Complete. `GET /api/fantasy/players/{bdl_player_id}/scores` endpoint live with 13 tests.
- **L3B (context authority)** — Complete. Scoped park factor consolidation with DB-backed reads.
- **L3D (observability)** — Complete. `/admin/diagnose-scoring/layer3-freshness` endpoint live with 13 tests.
- **L3F (decision read surface)** — Complete. `GET /api/fantasy/decisions` endpoint live with 13 tests.
- **Decision pipeline observability** — Complete. `/admin/diagnose-decision/pipeline-freshness` endpoint live with 8 tests.

**Phase 2 deliverables (COMPLETE):**
- 15/18 Z-scores (scoring_engine.py)
- ROW projector (`backend/services/row_projector.py`) — 31 tests
- Category math (`backend/services/category_math.py`) — 39 tests
- 4 greenfield categories (W, L, HR_P, NSV) deferred — awaiting upstream Yahoo data

### L3E. Market-Implied Probability Integration

**Status: DEFERRED — future enhancement backlog (not active)**

This work remains preserved as a complete specification but requires an explicit policy gate before becoming active. The proposed use of The Odds API for MLB player props currently conflicts with CLAUDE.md hard-stop rules. Do not conflate L3E with Phase 0-2 work.

### Layer 4 — Decision Engines and Simulation

Status: **COMPLETE — Phase 3 delivered v2 alignment + bridge + pure functions**

Phase 3 deliverables:
- H2H Monte Carlo v2 alignment — 18 categories, canonical codes, dynamic win threshold
- MCMC simulator v2 alignment — canonical category keys
- ROW→Simulation bridge adapter (`backend/services/row_simulation_bridge.py`)
- Ratio risk quantifier (pure function in `category_math.py`)
- Category-count delta extractor (pure function in `category_math.py`)
- Lineup optimization (lineup_constraint_solver.py) — batting-only, full-roster extension deferred

### Layer 5 — APIs and Service Presentation

Status: **COMPLETE — Phase 4 delivered 5 P1 API endpoints (50 new tests)**

**Phase 4 deliverables:**
- `GET /api/fantasy/scoreboard` — MatchupScoreboardResponse with 18 category rows, Monte Carlo, budget (26 tests)
- `GET /api/fantasy/budget` — ConstraintBudget with IP pace, acquisition warning (6 tests)
- `GET /api/fantasy/roster` — Extended to return CanonicalPlayerRow with rolling_14d (4 tests)
- `POST /api/fantasy/roster/move` — Slot swaps and IL moves via Yahoo set_lineup (7 tests)
- `POST /api/fantasy/roster/optimize` — Greedy roster optimization from rolling stats (7 tests)
- `backend/services/scoreboard_orchestrator.py` — Orchestrates L1-L4 for scoreboard assembly
- `backend/services/player_mapper.py` — Yahoo + rolling stats → CanonicalPlayerRow (17 tests)

**New contracts:** CanonicalRosterResponse, RosterMoveRequest/Response, RosterOptimizeRequest/Response, PlayerSlotAssignment

**Live data status:**
- `/api/fantasy/roster` → LIVE (Yahoo + rolling stats)
- `/api/fantasy/roster/move` → LIVE (Yahoo set_lineup)
- `/api/fantasy/roster/optimize` → LIVE (Yahoo + rolling stats)
- `/api/fantasy/scoreboard` → MOCK DATA (contract-valid, not wired to Yahoo matchup data yet)
- `/api/fantasy/budget` → MOCK DATA (contract-valid, not wired to Yahoo transactions yet)

**Pre-existing endpoints (unchanged):**
- `GET /api/fantasy/decisions` — live with 13 tests
- `GET /api/fantasy/players/{id}/scores` — live with 13 tests
- `GET /api/fantasy/lineup/{date}` — live
- `GET /api/fantasy/waiver` — live
- `GET /api/fantasy/waiver/recommendations` — live
- `GET /api/fantasy/matchup` — live (raw data, not scoreboard shape)
- `POST /api/fantasy/matchup/simulate` — live
- `GET /api/fantasy/briefing/{date}` — live
- `GET /api/dashboard` — live

### Layer 6 — Frontend and UX

Status: **GATED — Phase 5 after Phase 4.5 UAT passes**

Phase 5 will build P1 pages (Matchup Scoreboard + My Roster) after UAT confirms API data quality. Frontend readiness assessment in the UI audit found: 15 components salvageable, 9 CBB pages to archive, 6 canonical pages to build.

---

## Active Workstream: Phase 4.5a — Quality Remediation (Foundation Gaps)

**Status: NEXT**

The Kimi framework audit (`reports/2026-04-18-framework-audit.md`) applied elite fantasy baseball principles to the platform and exposed foundational quality gaps. Claude verified each claim against the codebase. 3 confirmed critical gaps make UAT meaningless without remediation. The highest-ROI fix is not adding new math — it's wiring existing math together.

### Kimi Audit Findings — Verification Results

| # | Claim | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | `smart_lineup_selector.py` orphaned | **REFUTED** | Wired into main.py API routes via `get_smart_selector()`. NOT in daily pipeline — architectural gap, not orphaning. |
| 2 | `decision_engine.py` primitive scoring | **CONFIRMED** | `_lineup_score()` uses `0.6×score + 0.3×momentum + 0.1×proj`. No category awareness, no matchup context, no LOWER_IS_BETTER. |
| 3 | `simulation_engine.py` only 7 stats | **CONFIRMED** | Covers HR, RBI, SB, AVG (batting) + K, ERA, WHIP (pitching). Missing 11 of 18 v2 categories. Separate from `h2h_monte_carlo.py` (which has 18). |
| 4 | `statcast_performances` never consumed | **PARTIALLY CONFIRMED** | 7K+ rows written by ingestion. Read ONLY by `data_reliability_engine.py` for validation. NOT consumed by scoring, decisions, or simulation. |
| 5 | `h2h_monte_carlo.py` not in production | **REFUTED** | Wired into `scoreboard_orchestrator.py` (Phase 4). Called by `GET /api/fantasy/scoreboard`. |
| 6 | `category_tracker.py` orphaned | **REFUTED** | Called in main.py for matchup category needs via `get_category_tracker()`. |
| 7 | `elite_lineup_scorer.py` orphaned | **REFUTED** | Used in async optimization routes in main.py and routers/fantasy.py. |
| 8 | Waiver pool empty | **REFUTED** | `_run_decision_optimization()` fetches 25 free agents from Yahoo, resolves via name fallback. |

### Decision Pipeline Gap Analysis

Four distinct decision paths exist. The daily pipeline is the most sophisticated; the API optimize endpoint is the most primitive. Neither uses the full 18-category system.

| Component | Daily Lineup | Daily Waivers | API Optimize | API Scoreboard |
|-----------|--------------|---------------|--------------|----------------|
| Data Source | player_scores + momentum + sim_results | Same | rolling_stats only | player_scores + ROW projections |
| Scoring | `_lineup_score()` (60/30/10 weighted) | `_composite_value()` (HR+RBI+SB) | Simple rolling avg | ROW projections per category |
| Monte Carlo | Inputs via downside_p25/upside_p75 | Same | None | H2HOneWinSimulator (18-cat) |
| Category Awareness | None | None | None | Full 18-cat (L1 category math) |
| LOWER_IS_BETTER | Not applied | Not applied | Not applied | Implicit in L1 math |
| Matchup Context | None | None | None | Full (my vs opp per category) |

**Root cause:** Sophisticated modules exist (smart_lineup_selector, elite_lineup_scorer, h2h_monte_carlo, category_tracker) but each lives in a separate API route. The daily automated pipeline and the API optimize endpoint bypass all of them.

### Phase 4.5a Remediation Plan

**Priority 1: Wire mock endpoints to live data**
- [x] Wire `GET /api/fantasy/scoreboard` to real Yahoo matchup stats (replace hardcoded `my_current_stats`/`opp_current_stats`)
- [x] Wire `GET /api/fantasy/budget` to real Yahoo transaction counts (replace hardcoded `acquisitions_used`/`il_used`)

**Priority 2: Fix API optimize endpoint quality**
- [x] Fix scoring to respect `LOWER_IS_BETTER` from stat_contract (ERA, WHIP, K_B, L, HR_P)
- [x] Add `frozen=True` to `RosterMoveRequest` and `RosterOptimizeRequest` contracts
- [x] Fix `target_date` vs `as_of_date` mismatch in optimize endpoint
- [x] Fix `assert len(...) >= 0` no-op assertion in test_roster_optimize_api.py

**Priority 3: Expand simulation_engine.py coverage**
- [x] Expand from 7→15 categories (added R, H, TB, NSB, K_B, OPS to batting; QS, K_9 to pitching). Note: W, L, HR_P, NSV remain deferred as not available in PlayerRollingStats.

**Priority 4: Integrate statcast x-stats into scoring (OPTIONAL — data quality review required first)**
- [ ] Wire `statcast_performances` data (xwOBA, barrel%, exit_velocity) into player scoring for luck-adjusted projections
- [ ] Scope: scoring_engine or decision_engine consumption — TBD based on data freshness review

**Priority 5: Fix pre-existing test failures**
- [x] `test_nsb_pipeline.py` (6 failures) — SimpleNamespace fixtures missing `runs` attribute
- [x] `test_h2h_monte_carlo.py` (1 failure) — Win threshold expectation mismatch (13-cat vs 18-cat)
- [x] `test_mcmc_simulator.py` (1 failure) — Empty opponent roster causes 0.5 fallback

### Phase 4.5a Gate Criteria

All Priority 1-3-5 items are COMPLETE. Priority 4 (statcast integration) remains optional.

- Scoreboard returns real Yahoo matchup data (not mock) — **COMPLETE**
- Budget returns real Yahoo transaction counts (not mock) — **COMPLETE**
- Optimize endpoint correctly handles LOWER_IS_BETTER categories — **COMPLETE**
- simulation_engine.py expanded to 15 categories — **COMPLETE**
- All existing tests still pass (2233+ + 8 fixed) — **COMPLETE**

---

## Phase 4.5b — UAT (Manual API Sanity Check)

**Status: BLOCKED by Phase 4.5a**

Once quality remediation is complete, manually test all P1 API endpoints against real Yahoo data to confirm data integrity, sensible values, and correct contract shapes.

### UAT Readiness Assessment

| Endpoint | Data Source | UAT Ready? | Notes |
|----------|-----------|------------|-------|
| `GET /api/fantasy/roster` | Yahoo + rolling stats | **YES** | Live data, CanonicalPlayerRow with rolling_14d |
| `POST /api/fantasy/roster/move` | Yahoo set_lineup | **YES** | Mutates real Yahoo lineup |
| `POST /api/fantasy/roster/optimize` | Yahoo + rolling stats | **After 4.5a P2** | LOWER_IS_BETTER fix required first |
| `GET /api/fantasy/scoreboard` | **MOCK → needs wiring** | **After 4.5a P1** | Currently returns hardcoded stats |
| `GET /api/fantasy/budget` | **MOCK → needs wiring** | **After 4.5a P1** | Currently returns hardcoded counts |

### UAT Checklist (Manual)

Once Phase 4.5a is complete:

- [ ] `GET /api/fantasy/roster` — Verify player names, positions, teams match Yahoo. Confirm rolling_14d stats are non-null for active players. Check status mapping (IL, DTD, etc.)
- [ ] `GET /api/fantasy/scoreboard` — Verify 18 category rows present. Confirm my_current/opp_current match Yahoo matchup page. Check projected_margin sign convention (positive = winning). Verify ERA/WHIP lower-is-better logic.
- [ ] `GET /api/fantasy/budget` — Verify acquisition count matches Yahoo. Confirm IP pace flag makes sense. Check acquisition_warning triggers at 6+.
- [ ] `POST /api/fantasy/roster/optimize` — Verify recommended starters make sense (best players starting, not benchwarming). Check position eligibility is respected. Confirm LOWER_IS_BETTER categories are scored correctly.
- [ ] `POST /api/fantasy/roster/move` — Test one safe move (BN → Util swap). Verify Yahoo reflects the change. Test rollback.

---

## Immediate Priority Queue: 9-Phase Gated Implementation Plan

| Phase | Layer Focus | Key Deliverable | Gate Criteria | Status |
|-------|------------|----------------|---------------|--------|
| 0 | L0 | 6 Pydantic contracts + stat_contract package | All compile, all reference 18 categories, no optional fields for required data | **COMPLETE** |
| 1 | L2 | V1→V2 migration + 7 data gap closures | All consumers on v2, 7 helpers tested, 2029 tests passing | **COMPLETE** |
| 2 | L3 | 18-category rolling stats + projections + ROW pipeline | ROW projections stable for full matchup week | **COMPLETE** |
| 3 | L1 + L4 | Pure functions + engine wiring | H2H Monte Carlo with projected finals produces non-degenerate results | **COMPLETE** |
| 4 | L5 | P1 page APIs (scoreboard, budget, roster, optimize) | All endpoints return complete data per contract | **COMPLETE** |
| 4.5a | L4-L5 | Quality Remediation — wire live data, fix scoring, expand coverage | Scoreboard/budget return real Yahoo data, LOWER_IS_BETTER fixed, tests green | **NEXT** |
| 4.5b | L5 | UAT — manual sanity-check of live API data | User confirms data accuracy before UI work | Blocked by 4.5a |
| 5 | L6 | Matchup Scoreboard + My Roster pages | Pages render with live data, mobile-optimized | Blocked by 4.5b |
| 6 | L3-L5 | P2 page backends (waiver v2, streaming) | Endpoints return complete data | Blocked by Phase 5 |
| 7 | L6 | Waiver Wire + Streaming pages | Pages render with live data | Blocked by Phase 6 |
| 8-9 | L3-L6 | P3 pages (Trade + Season Dashboard) | Complete | Blocked by Phase 7 |

---

## Frontend Readiness Brief

> NOTE: Superseded by the 9-phase gated plan. See "Active Workstream" and "Immediate Priority Queue" for the current plan.

Frontend is NOT the active workstream. When frontend execution resumes, use the documents below as the canonical briefing set and preserve backend-first sequencing.

### Frontend source-of-truth docs

1. `DESIGN.md` — Primary visual authority for the current design direction
2. `reports/2026-04-10-revolut-design-implementation-plan.md` — Token and component implementation plan
3. `docs/superpowers/plans/2026-04-12-next-steps-assessment.md` — Fantasy-first frontend roadmap
4. `FRONTEND_MIGRATION.md` — Historical frontend implementation record and guardrails
5. `reports/2026-03-12-api-ground-truth.md` — Contract authority for frontend TypeScript shapes
6. `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md` — Architectural guardrail

### UI Salvage Assessment (from UI Contract Audit)

**Keep:** 15 UI components, API client, auth flow, query client, layout mechanism, Tailwind config, utils, login page, alerts page, admin page

**Refactor:** Badge (new CategoryStatusTag variants), sidebar (canonical page hierarchy), layout header (add constraint display)

**Archive (CBB):** 9 pages, CBB type definitions

**Rebuild from scratch:** Matchup Scoreboard, My Roster, Waiver Wire, Streaming, Trade Analyzer, Season Dashboard, global header, CanonicalPlayerRow component, CategoryStatusTag component, fantasy TypeScript types

### Frontend activation gates

- Do not start frontend implementation while backend decision trust is still under validation.
- Frontend may consume validated backend outputs; it must not invent or pressure backend contracts prematurely.
- The first frontend initiative, when opened, should be fantasy-first and use existing `/api/fantasy/*` endpoints rather than redesigning archived CBB-first views.
- Any frontend type work must be grounded in backend route/schema truth or the API ground-truth report, not inferred from runtime UI errors.

---

## Architect Review Queue

### Open Questions from UI Contract Audit (7 remaining — Q10, Q11 resolved by Phase 0 contract)

**Q1:** Yahoo API rate limits for scoreboard/transactions/roster calls. Determines caching strategy.

**Q2-Q3:** Are W, L, SV, HLD, QS available in Yahoo player season stats? Affects rolling stats source.

**Q4:** Does the league use FAAB or priority-based waivers? Affects ConstraintBudget contract.

**Q5:** For opponent ROW projections: per-player or pace-based? Affects P2-5 scope.

**Q7:** What defines the matchup week boundary? Affects acquisition counting and games-remaining windows.

**Q8:** Acceptable scoreboard response time? Determines on-demand vs pre-compute strategy.

**Q9:** How should canonical player row handle trade context (same player as sending/receiving)?

**Resolved:** Q6 (America/New_York confirmed), Q10 (HLD is supporting stat, not scoring — see stat_contract JSON), Q11 (K_B is lower-is-better — confirmed in LOWER_IS_BETTER frozenset).

### Existing Contracts Note

The existing `contracts.py` has `UncertaintyRange`, `LineupOptimizationRequest`, `PlayerValuationReport`, and `ExecutionDecision`. Phase 0 new contracts should live alongside these, not replace them.

### L3E Deferral Confirmation

L3E (Market-Implied Probabilities) remains deferred and unchanged. Do not conflate it with Phase 0-2 work.

### Passive Monitoring

- Keep `/admin/version`, ingestion logs, and `probable_pitchers` on passive regression watch.
- Treat canonical environment snapshots beyond current weather and park persistence as backlog, not active recovery work.
- If production health regresses, reopen Layer 2 explicitly rather than mixing regression response into Phase 0-2 work.

---

## Delegation Bundles

### Gemini CLI

No active delegation.

Use Gemini only if a production regression check, Railway deploy, log tail, or read-only production validation is required.

### Kimi CLI

Phase 2 research delegation available. See HANDOFF PROMPTS below for K1 (rolling stats audit) and K2 (ROW projection spec).

---

## HANDOFF PROMPTS

Phase 2 research prompts ready for Kimi CLI delegation (K1 and K2). Implementation prompt at `CLAUDE_PHASE1_IMPLEMENTATION_PROMPT.md` is historical (Phase 1 complete). Phase 2 implementation prompt to be created after K1/K2 research memos are delivered.

---

Last Updated: April 18, 2026 (Phase 4.5a P1-P3-P5 COMPLETE + test fixtures fixed. Test suite: 2239 passing. 6 failures are DB connection only (environment issue). Phase 4.5b UAT READY.)
