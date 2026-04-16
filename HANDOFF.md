# HANDOFF.md — MLB Platform Operating Brief

> Date: April 16, 2026 | Author: Claude Code (Master Architect)
> Status: Layer 2 certified complete. Layer 3B consolidation complete. Layer 3D observability complete. API endpoint live with auth. Do not reopen Layer 2 except for regressions.

Full audit: reports/2026-04-15-comprehensive-application-audit.md
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

---

## Core Doctrine

The MLB data platform is now validated at Layer 2. Work may resume above Layer 2, but execution should remain disciplined and sequenced.

The architecture remains layered:

| Layer | Name | Purpose | Status |
|------|------|---------|--------|
| 0 | Immutable Decision Contracts | Canonical contracts, schemas, IDs, and validation boundaries | Stable |
| 1 | Pure Stateless Intelligence | Deterministic pure functions over validated inputs | Available |
| 2 | Data and Adaptation | Ingestion, validation, persistence, observability, freshness, provenance | Certified Complete |
| 3 | Derived Stats and Scoring | Rolling stats, player scores, context-enriched features | Active |
| 4 | Decision Engines and Simulation | Lineup logic, waiver logic, matchup engines, Monte Carlo | Hold until the first Layer 3 objective is stable |
| 5 | APIs and Service Presentation | FastAPI contracts, dashboards, admin views | Maintenance |
| 6 | Frontend and UX | Next.js pages, interactions, polish | Maintenance |

### Operating Rule

- Do not reopen Layer 2 as an active workstream unless a production regression is observed.
- Do not activate broad Layer 4 work yet.
- Use Layer 3 as the single active engineering lane.

---

## Current Production Truth

Verified production state as of April 16, 2026 (18:00 UTC):

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

### Operational Interpretation

- Layer 2 is certified complete.
- Park factor authority is DB-backed with fallback (ballpark_factors.py → ParkFactor table → PARK_FACTORS constant → neutral 1.0).
- Weather context exists in schema but is NOT populated; request-time weather (weather_fetcher.py) remains the live path for consumers like smart_lineup_selector.py.
- Layer 3 scoring (player_scores) does NOT consume weather - pure rolling-window Z-score computation remains appropriate for multi-day windows.
- The production data spine is no longer the blocker.
- The next bottleneck is downstream scoring construction, not ingestion stability.

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
Status: STABLE

- No contract expansion is currently required.

### Layer 1 — Pure Stateless Intelligence
Status: AVAILABLE

- Pure logic may be extended as required by Layer 3 scoring work.

### Layer 2 — Data and Adaptation
Status: CERTIFIED COMPLETE

- Regressions only.
- Do not run a new Layer 2 roadmap unless production evidence degrades.

### Layer 3 — Derived Stats and Scoring
Status: ACTIVE

- This is the only active engineering workstream now.

**Layer 3B Context Authority Audit (2026-04-16):**

Authoritative scoring path: `player_rolling_stats → compute_league_zscores() → player_scores`

Key findings:
- Scoring is PURE Z-score computation over rolling stats - NO park factors, NO weather
- Park factor fragmentation: 5+ hardcoded copies across codebase (ballpark_factors.py, mlb_analysis.py, daily_lineup_optimizer.py, two_start_detector.py, weather_ingestion.py)
- scoring_engine.py has DB-backed get_park_factor() helper but it's UNUSED by main scoring
- Weather infrastructure exists (weather_ingestion.py) but weather_forecasts table is EMPTY; request-time weather (weather_fetcher.py) is the live path

Risk severity: HIGH (fragmentation) > MEDIUM (unused helper confusion) > LOW (weather table empty but request-time path exists)

**Scoped consolidation COMPLETE (2026-04-16):**
- `ballpark_factors.py:get_park_factor()` now reads from ParkFactor table first, with fallback to PARK_FACTORS constant → neutral 1.0
- 9 focused tests added (test_ballpark_factors.py)
- Weather remains deferred for Layer 3 scoring (appropriate for rolling windows; request-time weather via weather_fetcher.py serves immediate-decision needs)

### Layer 4 — Decision Engines and Simulation
Status: HOLD

- Do not resume broader decision-engine work until the first Layer 3 scoring objective is complete and stable.

### Layer 5 — APIs and Service Presentation
Status: MAINTENANCE

- Only changes needed to expose validated Layer 3 outputs should be made.

### Layer 6 — Frontend and UX
Status: MAINTENANCE

- No new UI initiative should begin until Layer 3 outputs are defined.

---

## Frontend Readiness Brief

Frontend is NOT the active workstream. When frontend execution resumes, use the documents below as the canonical briefing set and preserve backend-first sequencing.

### Frontend source-of-truth docs

1. `DESIGN.md`
	- Primary visual authority for the current design direction.
	- Use the Revolut-inspired system: Aeonik Pro display typography, Inter body, near-black/white binary, pill buttons, zero shadows.

2. `reports/2026-04-10-revolut-design-implementation-plan.md`
	- Token and component implementation plan for Tailwind/CSS.
	- Use for concrete frontend build execution once UI work is officially active.

3. `docs/superpowers/plans/2026-04-12-next-steps-assessment.md`
	- Fantasy-first frontend roadmap.
	- Important product guidance: do NOT spend cycles redesigning dead CBB surfaces before the fantasy product has a usable interface.

4. `FRONTEND_MIGRATION.md`
	- Historical frontend implementation record and guardrails.
	- Useful for patterns, auth, client-fetching rules, and type-discipline; NOT the source of current product priority.

5. `reports/2026-03-12-api-ground-truth.md`
	- Contract authority for frontend TypeScript shapes.
	- Frontend types should be derived from backend truth, never guessed from browser errors.

6. `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md`
	- Architectural guardrail for product separation.
	- Frontend work should reinforce Fantasy as the active product and avoid deepening coupling to frozen CBB concerns.

### Frontend activation gates

- Do not start frontend implementation while backend decision trust is still under validation.
- Frontend may consume validated backend outputs; it must not invent or pressure backend contracts prematurely.
- The first frontend initiative, when opened, should be fantasy-first and use existing `/api/fantasy/*` endpoints rather than redesigning archived CBB-first views.
- Any frontend type work must be grounded in backend route/schema truth or the API ground-truth report, not inferred from runtime UI errors.
- Treat `DESIGN.md` as the style guide and `reports/2026-04-10-revolut-design-implementation-plan.md` as the implementation recipe.
- If frontend work starts, scope it as a bounded execution lane with its own prompt and do not mix it into backend stabilization tasks.

---

## Active Workstream

### L3A. Derived Stats And Scoring Spine

**Status: Complete (2026-04-16)**

Implemented `GET /api/fantasy/players/{bdl_player_id}/scores` - the first authoritative Layer 3 scoring exposure.

**Completed:**
- Endpoint implementation in `backend/routers/fantasy.py` with verify_api_key auth
- Pydantic schemas (`PlayerScoresResponse`, `PlayerScoreOut`, `PlayerScoreCategoryBreakdown`)
- 13 comprehensive test cases covering validation, response contract, and auth requirements
- Supports window_days=7/14/30 (defaults to 14)
- Supports optional as_of_date query parameter (defaults to latest available)
- Returns 400 for invalid window_days, 404 for missing scores, 401 for missing auth
- Exposes hitter categories (z_hr, z_rbi, z_nsb, z_avg, z_obp) and pitcher categories (z_era, z_whip, z_k_per_9)

### L3B. Context Authority Consolidation

**Status: Complete (2026-04-16)**

Scoped park factor consolidation completed:
- `ballpark_factors.py:get_park_factor()` now resolves: DB → PARK_FACTORS constant → 1.0 neutral
- 9 focused tests added (test_ballpark_factors.py)
- Weather remains explicitly deferred for Layer 3 scoring (request-time weather via weather_fetcher.py serves immediate-decision needs)

### L3D. Layer 3 Observability

**Status: Complete (2026-04-16)**

Layer 3 freshness endpoint `/admin/diagnose-scoring/layer3-freshness` is live and fully tested:
- Returns freshness verdict (healthy/stale/partial/missing)
- Provides row counts by window (7/14/30)
- Shows latest audit log entries for rolling_windows and player_scores jobs
- 13 comprehensive tests in test_admin_scoring_diagnostics.py

**Out of scope for this phase:**

- simulation expansion
- waiver-system breadth work
- optimizer redesign
- frontend feature expansion
- broad Layer 4 activation
- weather_forecasts table population (request-time weather remains sufficient)

---

## Immediate Priority Queue

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| P0 | Define the first Layer 3 scoring objective and success criteria | Claude | Complete |
| P0 | Audit current derived-stats and scoring code path for gaps | Claude | Complete |
| P1 | Identify one authoritative scoring output for downstream consumers | Claude | Complete |
| P1 | Audit context authority in scoring path (3B) | Claude | Complete |
| P2 | Consolidate ballpark_factors.py to DB-backed read | Claude | Complete |
| P2 | Add Layer 3 freshness observability endpoint | Claude | Complete |
| P3 | Decide whether any Layer 5 response shape changes are needed after scoring output stabilizes | Claude | Pending |

---

## Architect Review Queue

- Keep `/admin/version`, ingestion logs, and `probable_pitchers` on passive regression watch.
- Treat canonical environment snapshots beyond current weather and park persistence as backlog, not active recovery work.
- Do not reopen simulation or decision-layer expansion until Layer 3 outputs are demonstrably trustworthy.
- Do not start frontend implementation until the backend decision pipeline is trusted and the frontend lane is opened explicitly.
- When frontend work opens, use `DESIGN.md` plus the April 10 and April 12 planning docs as the briefing pack; treat `FRONTEND_MIGRATION.md` as historical implementation context only.
- If production health regresses, reopen Layer 2 explicitly rather than mixing regression response into Layer 3 work.

---

## Delegation Bundles

### Gemini CLI

No active delegation.

Use Gemini only if a production regression check, Railway deploy, log tail, or read-only production validation is required.

### Kimi CLI

No active delegation.

Use Kimi for a bounded Layer 3 analysis memo only after Claude defines the exact scoring objective.

---

## HANDOFF PROMPTS

No active handoff prompt is currently open. Create a new prompt only after the first Layer 3 objective is explicitly defined.

---

Last Updated: April 16, 2026 (18:00 UTC - frontend readiness docs added; backend-first sequencing preserved)