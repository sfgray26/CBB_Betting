# HANDOFF.md — MLB Platform Operating Brief

> Date: April 18, 2026 | Author: Claude Code (Master Architect)
> Status: Phase 0 COMPLETE. Phase 1 COMPLETE. Phase 2 COMPLETE (15 Z-scores, ROW projector, category math — 96 new tests). Phase 3 NEXT (pure functions + engine wiring). L3E deferred. Do not reopen Layer 2 except for regressions.

UI Specification Audit: reports/2026-04-17-ui-specification-contract-audit.md
Comprehensive application audit: reports/2026-04-15-comprehensive-application-audit.md
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
| 4 | Decision Engines and Simulation | Lineup logic, waiver logic, matchup engines, Monte Carlo | **ACTIVE — Phase 3: wire H2H Monte Carlo + MCMC to v2 codes and projected data. Align simulation categories with 18-category contract.** |
| 5 | APIs and Service Presentation | FastAPI contracts, dashboards, admin views | **GATED — Phase 4: build scoreboard, budget, and roster endpoints after Phase 3 gate passes** |
| 6 | Frontend and UX | Next.js pages, interactions, polish | **GATED — Phase 5: build P1 pages after Phase 4 gate passes. 15 components salvageable, 9 CBB pages to archive.** |

### Operating Rule

- **Phase 0 is COMPLETE.** stat_contract package loaded, 6 UI contracts validated, 30/30 tests passing.
- **Phase 1 is COMPLETE.** V1→V2 consumer migration + 7 data gap closures delivered, 2029 tests passing.
- **Phase 2 is COMPLETE.** 15/18 Z-scores computed, ROW projector + category math delivered, 96 new tests passing. 4 greenfield categories (W, L, HR_P, NSV) awaiting upstream data.
- Phase 3 (pure functions + engine wiring) is NEXT.
- Do not reopen Layer 2 as an active workstream unless a production regression is observed.
- Layers 4, 5, and 6 are gated until their prerequisite phases pass gate criteria.
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

Status: **Partially delivered — IP pace classifier in Phase 1, 4 pure functions remain for Phase 3**

Phase 1 delivered `classify_ip_pace()` in `backend/services/constraint_helpers.py`. Remaining pure functions (delta-to-flip calculator, ratio risk quantifier, category-count delta extractor, TB calculator) will be implemented in Phase 3 after Layer 3 projected data is available.

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

Status: **ACTIVE — Phase 2 will expand coverage and build ROW pipeline**

Historical accomplishments (preserved):
- **L3A (scoring spine)** — Complete. `GET /api/fantasy/players/{bdl_player_id}/scores` endpoint live with 13 tests.
- **L3B (context authority)** — Complete. Scoped park factor consolidation with DB-backed reads.
- **L3D (observability)** — Complete. `/admin/diagnose-scoring/layer3-freshness` endpoint live with 13 tests.
- **L3F (decision read surface)** — Complete. `GET /api/fantasy/decisions` endpoint live with 13 tests.
- **Decision pipeline observability** — Complete. `/admin/diagnose-decision/pipeline-freshness` endpoint live with 8 tests.

**Phase 2 focus:**
- Expand rolling stats from 9 to 18 categories (add R, TB, W, L, HRA, K, QS, NSV)
- Expand projections from 8 to 18 categories
- Build ROW (Rest-of-Week) projection pipeline
- Team-level ROW aggregation
- Opponent ROW projection
- Category classification with directionality
- Delta-to-flip calculator
- Freshness timestamp propagation

### L3E. Market-Implied Probability Integration

**Status: DEFERRED — future enhancement backlog (not active)**

This work remains preserved as a complete specification but requires an explicit policy gate before becoming active. The proposed use of The Odds API for MLB player props currently conflicts with CLAUDE.md hard-stop rules. Do not conflate L3E with Phase 0-2 work.

### Layer 4 — Decision Engines and Simulation

Status: **ACTIVE — Phase 3**

Phase 3 wires existing simulation engines to v2 canonical codes and the ROW projection pipeline. The gate lifts when H2H Monte Carlo produces non-degenerate results with projected data.

Existing engines (ready, awaiting v2 alignment):
- H2H Monte Carlo simulation (h2h_monte_carlo.py) — uses v1 codes: K, K/9, SB. Missing: H, K_B, TB, L, HR_P, NSV
- MCMC roster-move simulation (mcmc_simulator.py) — uses z-score convention with non-canonical keys (k_pit, k9)
- Lineup optimization (lineup_constraint_solver.py) — batting-only, needs full-roster extension (Phase 3+)
- Drop candidate selection (_weakest_safe_to_drop)

New deliverables (Phase 3):
- H2H Monte Carlo v2 alignment (18 categories, canonical codes)
- MCMC simulator v2 alignment (canonical category keys)
- ROW→Simulation bridge adapter
- Ratio risk quantifier (pure function)
- Category-count delta extractor (pure function)

### Layer 5 — APIs and Service Presentation

Status: **GATED — Phase 4 after Phase 3 gate passes**

Phase 4 will build P1 page APIs (scoreboard, budget, roster, optimize) after Phase 3 gate passes. All endpoints will return complete data per Layer 0 contracts and include freshness metadata.

**Existing endpoints:**
- `GET /api/fantasy/decisions` — live with 13 tests
- `GET /api/fantasy/players/{id}/scores` — live with 13 tests
- `GET /api/fantasy/lineup/{date}` — live
- `GET /api/fantasy/waiver` — live
- `GET /api/fantasy/waiver/recommendations` — live
- `GET /api/fantasy/roster` — live
- `GET /api/fantasy/matchup` — live (raw data, not scoreboard shape)
- `POST /api/fantasy/matchup/simulate` — live
- `GET /api/fantasy/briefing/{date}` — live
- `GET /api/dashboard` — live

**Phase 4 endpoints to build:**
- `GET /api/fantasy/scoreboard` — MatchupScoreboardResponse contract
- `GET /api/fantasy/budget` — ConstraintBudget contract
- Extend `GET /api/fantasy/roster` to return CanonicalPlayerRow
- `POST /api/fantasy/roster/move` — slot swaps and IL moves
- `POST /api/fantasy/roster/optimize` — full-roster optimization

### Layer 6 — Frontend and UX

Status: **GATED — Phase 5 after Phase 4 gate passes**

Phase 5 will build P1 pages (Matchup Scoreboard + My Roster) after Phase 4 gate passes. Frontend readiness assessment in the UI audit found: 15 components salvageable, 9 CBB pages to archive, 6 canonical pages to build.

---

## Active Workstream: Phase 3 — Pure Functions + Engine Wiring

**Status: NEXT**

Phase 3 wires the simulation layer (H2H Monte Carlo, MCMC) to v2 canonical codes and the ROW projection pipeline from Phase 2. It also builds remaining L1 pure functions.

### Prerequisites (verified)
- Phase 0: stat_contract package loaded, 18 canonical codes, UI contracts ✓
- Phase 1: v2 consumer migration complete, 7 constraint helpers available ✓
- Phase 2: 15/18 Z-scores, ROW projector (`backend/services/row_projector.py`), category math (`backend/services/category_math.py`), 96 new tests ✓

### Key Deliverables
1. **H2H Monte Carlo v2 alignment** — Update `h2h_monte_carlo.py` categories from v1 codes (K, K/9, SB) to v2 (K_P, K_9, NSB, + add H, K_B, TB, L, HR_P, NSV)
2. **MCMC simulator v2 alignment** — Update `mcmc_simulator.py` category keys to match canonical codes
3. **ROW→Simulation bridge** — Adapter that converts `ROWProjectionResult` into the player dict format expected by simulation engines
4. **Remaining L1 pure functions** — Ratio risk quantifier, category-count delta extractor
5. **Tests** — Simulation produces non-degenerate results with projected data

---

## Immediate Priority Queue: 9-Phase Gated Implementation Plan

| Phase | Layer Focus | Key Deliverable | Gate Criteria | Status |
|-------|------------|----------------|---------------|--------|
| 0 | L0 | 6 Pydantic contracts + stat_contract package | All compile, all reference 18 categories, no optional fields for required data | **COMPLETE** |
| 1 | L2 | V1→V2 migration + 7 data gap closures | All consumers on v2, 7 helpers tested, 2029 tests passing | **COMPLETE** |
| 2 | L3 | 18-category rolling stats + projections + ROW pipeline | ROW projections stable for full matchup week | **COMPLETE** |
| 3 | L1 + L4 | Pure functions + engine wiring | H2H Monte Carlo with projected finals produces non-degenerate results | **NEXT** |
| 4 | L5 | P1 page APIs (scoreboard, budget, roster, optimize) | All endpoints return complete data per contract | Blocked by Phase 3 |
| 5 | L6 | Matchup Scoreboard + My Roster pages | Pages render with live data, mobile-optimized | Blocked by Phase 4 |
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

Last Updated: April 18, 2026 (Phase 1 COMPLETE: v1→v2 migration + 7 data gap closures, 2029 tests passing. Phase 2 is NEXT: 18-category rolling stats + ROW projection pipeline.)
