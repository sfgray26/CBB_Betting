# CBB Edge — Fantasy Baseball Platform: Production-Grade Assessment

**Date:** 2026-07-10  
**Scope:** Backend (FastAPI/SQLAlchemy/PostgreSQL), frontend (Next.js 15/React 19), data pipelines, tests, and operational tooling for the Fantasy Baseball module. CBB betting code was reviewed for cross-cutting issues only.  
**Methodology:** Read-only audit. Parallel deep-dives by subdomain were synthesized into this report. No code was modified.  
**Artifact location:** `reports/2026-07-10-production-grade-assessment.md`

---

## 1. Executive Summary

### Overall platform maturity score: **4 / 10**

The platform is **functionally rich** and has recently shipped working fixes for projection coverage and IL exclusion, but it is **not production-grade** today. The most serious problems are a **critical authentication regression on roster-mutation endpoints**, **massive route duplication between `main.py` and the extracted `fantasy.py` router**, and **data-pipeline workarounds that mask root-cause data-quality failures**. Combined with low true test coverage, a frontend build failure, and missing global error handling, the system cannot be declared production-ready without remediation.

### Top 3 risks that could kill the product

1. **Unauthenticated roster mutation** — `POST /api/fantasy/roster/move` and `POST /api/fantasy/roster/bulk-apply` have no API-key guard. Anyone on the internet can change the user's Yahoo lineup.  
2. **Route shadowing makes the deployed API diverge from the source code** — `backend/main.py` still registers inline fantasy routes before mounting `backend/routers/fantasy.py`, so edits to the router may have no production effect.  
3. **Silent data-pipeline degradation** — the `player_id_mapping` "corruption workaround" is still in the hot path, external API clients lack retries, and broad `except Exception` handlers return "healthy" defaults, allowing bad data to reach users undetected.

### Top 3 quick wins (high impact, low effort)

1. **Add `Depends(verify_api_key)` to the two roster-mutation endpoints** (~5 min each).  
2. **Fix the frontend build** by removing the `any` type in `components/__tests__/matchup-strip.test.tsx:44` and remove the `http://localhost:8000` fallback in `frontend/lib/api.ts:69`.  
3. **Instrument the projection-coverage workaround** — emit a warning/metric whenever `find_alternative_player_score` is triggered, and treat `covered_workaround` as yellow, not green.

---

## 2. Critical Issues (P0 — Must Fix Before Production)

| # | Issue | Location | Evidence | User Impact | Recommended Fix | Effort |
|---|-------|----------|----------|-------------|-----------------|--------|
| P0-1 | **Roster-mutation endpoints are unauthenticated** | `backend/routers/fantasy.py:4158` and `:4550` | `async def move_roster_player(req: Request, request: RosterMoveRequest):` and `async def bulk_apply_roster_moves(request: BulkRosterMoveRequest):` — neither has `user: str = Depends(verify_api_key)`. | Any HTTP client can move players or apply a full lineup. The most destructive security finding in the audit. | Add `user: str = Depends(verify_api_key)` to both signatures. Audit the 15 total unauthenticated routes in `fantasy.py` and either guard or explicitly mark them public. | XS |
| P0-2 | **Inline routes in `main.py` shadow the `fantasy.py` router** | `backend/main.py:642-654` mounts routers *after* defining inline routes; `backend/main.py:4493`, `:4523`, `:4541`, `:5106`, `:5124`, `:5566`, `:5881`, `:7182` still define `/api/fantasy/*` and `/api/dashboard/*` inline. | ```python
# backend/main.py:642-654
# ... FastAPI routes are matched in registration order; the inline routes
# still win because they were registered first. Once Phase 5 cut-over
# is complete and inline routes are removed, these mounts take over.
``` | Changes made to `backend/routers/fantasy.py` may not execute in production. The deployed code can silently diverge from the repository. | Complete the "Phase 5 cut-over": delete all inline fantasy/dashboard routes from `main.py`; keep `main.py` as app factory + lifespan + scheduler + router mounts only. Run a smoke test for every `/api/fantasy/*` path before and after. | L |
| P0-3 | **Frontend production build is failing** | `frontend/components/__tests__/matchup-strip.test.tsx:44` | ESLint `no-explicit-any` error blocks `npm run build`. | Cannot deploy the frontend until fixed. | Fix the `any` type or exclude test files from the production lint pass. | XS |
| P0-4 | **API client falls back to `localhost:8000` in production** | `frontend/lib/api.ts:69` and `frontend/app/login/page.tsx:7` | `const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'` | If `NEXT_PUBLIC_API_URL` is missing, every deployed user calls their own machine and sees broken functionality. | Remove the fallback; fail the build if the env var is missing, or default to an invalid sentinel that throws at runtime. | XS |
| P0-5 | **Hardcoded Yahoo token refresh endpoint returns plaintext token** | `backend/admin_yahoo_token_refresh.py:14-49` | ```python
return {
    "access_token": access_token,
    "railway_update_command": f'railway variables set YAHOO_ACCESS_TOKEN="{access_token}"',
}
``` | If this file is ever mounted (it is not currently), any caller gets the live Yahoo access token and a ready-to-run command. | Delete the file. If a refresh helper is needed, make it admin-authenticated and do not return the full token. | XS |
| P0-6 | **SQL injection risk in `cat_scores_builder.py`** | `backend/services/cat_scores_builder.py:293-317` | ```python
id_list = ", ".join(f"'{pid}'" for pid in null_ids)
team_query = text(f"SELECT DISTINCT ON (player_id) player_id, team FROM statcast_performances WHERE player_id IN ({id_list}) ...")
``` | A malicious or malformed `player_id` containing a single quote can break the query or inject SQL. | Use SQLAlchemy `bindparam(..., expanding=True)` or `column.in_(...)` instead of string interpolation. | S |

---

## 3. High Priority (P1 — Fix Before Beta Launch)

| # | Issue | Location | Evidence | User Impact | Recommended Fix | Effort |
|---|-------|----------|----------|-------------|-----------------|--------|
| P1-1 | **Projection-coverage "corruption workaround" masks future data-quality issues** | `backend/services/player_id_resolver.py:286-374`, `backend/routers/fantasy.py:5274-5285`, `backend/services/projection_coverage.py:137-140` | `find_alternative_player_score()` is still called in the optimizer hot path. `projection_coverage.py` treats `covered_workaround` as healthy coverage. | Users see confident lineup scores even when the identity-mapping layer is silently patching missing data. Root causes stay hidden. | Remove workaround from optimizer hot path; surface `covered_workaround` as a warning; alert if any roster player triggers it for 7+ consecutive days. | M |
| P1-2 | **CORS defaults to wildcard origins** | `backend/main.py:710-722` | `allow_origins=_allowed_origins or ["*"]` with `allow_methods=["*"]`, `allow_headers=["*"]`. | Combined with unauth endpoints, arbitrary origins can invoke read/write endpoints from a victim's browser. | Require `ALLOWED_ORIGINS` in production; fail startup if empty. Default to `[]`, not `["*"]`. | XS |
| P1-3 | **Bulk roster apply is not atomic** | `backend/routers/fantasy.py:4550-4875`, `backend/fantasy_baseball/yahoo_client_resilient.py:1399-1426` | Endpoint calls `client.set_lineup(team_key, lineup)` once. Yahoo client falls back to per-player PUT on `game_id` mismatch. No rollback exists. | Partial lineup changes can occur with no way to undo them. | Document as "best-effort atomic"; validate locally first; return explicit `applied`/`skipped`/`warnings` and surface skipped players in the UI. | S |
| P1-4 | **Global exception handler converts `ValidationError` to opaque 500** | `backend/main.py:8011-8023` | `@app.exception_handler(Exception)` returns `{"detail": "Internal server error", "type": type(exc).__name__}` for all unhandled exceptions. | Callers cannot distinguish bad input (422) from server failure (500). Field-level validation errors are lost. | Add explicit handlers for `RequestValidationError` and `ValidationError` returning 422 with field details. | S |
| P1-5 | **External API clients lack retry logic** | `backend/services/balldontlie.py:79-83`, `:306-336`; `backend/ingestion/fangraphs_scraper.py:33-102`; `backend/ingestion/savant_scraper.py:55-64` | Single-attempt `session.get` with `break` on exception. | A transient 5xx/429/timeout at 02:00 can leave data stale until the next day. | Add uniform retry/backoff (e.g., `tenacity`) to all external GETs: 3 attempts, exponential backoff, fail only after retries. | M |
| P1-6 | **Broad exception handlers return "healthy" defaults** | `backend/services/daily_ingestion.py:381-396`, `:699-739`, `:742-813`, `:816-924`, `:927-972`; `backend/fantasy_baseball/yahoo_client_resilient.py:570-598`, `:736-756`, `:1020-1124` | Many helpers catch `Exception` and return `{}`, `0`, `None`, or `True`. | Failures in Statcast enrichment, ownership refresh, or opportunity lookup are invisible to users; dashboards show stale/missing data as if healthy. | Return structured results (`success`, `rows_affected`, `error_count`) and emit metrics/alerts on non-zero error counts. | M |
| P1-7 | **Unbounded DB queries / N+1 patterns** | `backend/routers/data_quality.py:273-297`; `backend/routers/fantasy.py:8886`; `backend/services/health_monitor.py:53-69` | Loop issues one query per player; `PlayerIdentity.all()` loads entire table; few `joinedload` usages. | Latency and memory growth as player pool grows; can cause 503s on dashboard route. | Replace loops with `IN`/`EXISTS` queries; add pagination; eager-load relationships. | M |
| P1-8 | **Duplicate scoring/category logic produces inconsistent values** | `backend/routers/fantasy.py:234-240`, `:7833-7838`; `backend/fantasy_baseball/category_aware_scorer.py:34-40`; `backend/services/cat_scores_builder.py:49-60`; `backend/fantasy_baseball/player_board.py:689-700`; `backend/services/need_score.py:15-16`; `backend/main.py:6178-6200` | `_CANONICAL_TO_BOARD` is copy-pasted; z-score is implemented in multiple files; need-score has a service wrapper, a direct scorer call, and an ad-hoc loop in `main.py`. | Same player can have different scores/verdicts across waiver, dashboard, optimizer, and decisions. | Centralize canonical-category maps and z-score/need-score math in `backend/core/` or `backend/services/`; delete inline redefinitions. | M |
| P1-9 | **No global error boundary and fragmented cache keys** | Missing `frontend/app/error.tsx` and `frontend/app/global-error.tsx`; `frontend/lib/query-client.ts:31-76`; `frontend/components/layout/header.tsx:54` | Two duplicate `ErrorBoundary` components exist; header refresh invalidates *all* queries; many War Room keys are not in `FANTASY_QUERY_KEYS`. | Unhandled component errors crash whole pages; global refresh misses stale data; over-invalidation wastes backend resources. | Add root `error.tsx`/`global-error.tsx`; consolidate boundaries; centralize all fantasy query keys and use prefix/predicate invalidation. | M |
| P1-10 | **Reported test coverage is misleading** | `.coveragerc:3-8` omits `backend/main.py`, `backend/schemas.py`, `backend/fantasy_baseball/*`, `scripts/*`, migrations. | Measured coverage is 48 %, but the entry point, schema layer, Yahoo client, and optimizer are excluded. | Regressions in auth, schemas, and lineup optimization can ship without failing coverage gates. | Run a second CI coverage job with no omissions; set a full-backend coverage gate. | S |
| P1-11 | **Admin authorization is hardcoded to `user1`** | `backend/auth.py:85` | `if user != "user1": raise HTTPException(status_code=403, ...)` | No support for multiple admins or role-based access. Acceptable for 1–2 users but must be documented as intentional. | Document; consider `ADMIN_API_KEYS` env var if team grows. | XS |
| P1-12 | **API key stored in localStorage + non-HttpOnly cookie** | `frontend/lib/auth.ts:9-46` | Key lives in `localStorage` and a `SameSite=Strict` cookie without `Secure`/`HttpOnly`. | XSS can steal the API key. | Remove the cookie and read the key from `localStorage` on client requests only; if cookie is required for SSR, make it `HttpOnly; Secure`. | S |


---

## 4. Medium Priority (P2 — Fix During Beta)

| # | Issue | Location | Evidence | User Impact | Recommended Fix | Effort |
|---|-------|----------|----------|-------------|-----------------|--------|
| P2-1 | **`backend/routers/fantasy.py` is a 8,977-line god router** | `backend/routers/fantasy.py` | ~62 endpoints spanning draft, lineup, waiver, roster, matchup, dashboard, decisions, streaming, scoring. | Impossible to review/test/deploy subdomains independently; high merge-conflict risk. | Split along subdomain boundaries: `routers/draft.py`, `routers/lineup.py`, `routers/waiver.py`, `routers/roster.py`, `routers/matchup.py`, `routers/decisions.py`, `routers/dashboard.py`; keep `fantasy.py` as a thin aggregator. | L |
| P2-2 | **`backend/main.py` is 8,028 lines and mixes app factory with legacy routes/scheduler/business logic** | `backend/main.py` | ~147 inline `@app.*` routes plus scheduler wiring and helper functions. | Same shadowing/coupling problem as the fantasy router; difficult to reason about. | Remove legacy routes; move business logic to services; keep only lifespan, scheduler, middleware, and router mounts. | L |
| P2-3 | **Import cycles force lazy-import workarounds** | `backend/services/projection_coverage.py:47-52`; `backend/routers/fantasy.py:124-133`, `:247-253`; `backend/main.py` ↔ `backend/routers/admin.py` | Services import helpers from `routers/fantasy`; routers lazy-import orchestrator from `main`. | Violates layering; slows startup; complicates type checking; can fail at runtime. | Move shared roster/identity helpers to a neutral `backend/services/roster_identity_service.py`; move shared scheduler/health utilities to `backend/core/`. | M |
| P2-4 | **Business logic and direct DB writes leak into routers** | `backend/routers/fantasy.py` | 85 direct `db.query/add/commit/delete/execute` calls and 11 raw `text()` SQL statements. | Routers own persistence decisions; unit testing business rules requires FastAPI test client. | Introduce subdomain services (`DraftService`, `LineupService`, `WaiverService`, `RosterService`) and move transactions there. | L |
| P2-5 | **Frontend "god" page components** | `frontend/app/(dashboard)/war-room/roster/page.tsx` (1,346 lines); `frontend/app/(dashboard)/war-room/waiver/page.tsx` (805 lines); `frontend/app/(dashboard)/dashboard/_components/dashboard-client.tsx` (749 lines) | Mix data fetching, mutations, optimistic updates, stats formatting, and UI. | Hard to test/review; large blast radius. | Extract data hooks into `frontend/hooks/`; pure helpers into `*.utils.ts`; sub-components into `/_components/`. | M |
| P2-6 | **Missing loading skeletons on high-traffic routes** | Only 3 `loading.tsx` files exist: `today`, `bracket`, `admin`. | War Room roster/waiver/streaming rely on inline spinners. | Layout shift and poor perceived performance. | Add `loading.tsx` to `/war-room/roster`, `/war-room/waiver`, `/war-room/streaming`, `/decisions`. | S |
| P2-7 | **Mobile table overflow and fixed-width elements** | `frontend/components/ui/data-table.tsx:58-60`; `frontend/components/streaming/streaming-recommendations.tsx:430`; `frontend/components/yahoo-roster-view.tsx:203-210` | Tables use horizontal scroll; some fixed widths (`w-16`, `min-w-[2.5rem]`). | Dense tables are hard to use on small screens. | Audit table consumers; consider card-based mobile layouts; remove unnecessary fixed widths. | S |
| P2-8 | **Plain-text logs with no correlation IDs** | `backend/main.py:131-135` | `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')`; no request-id middleware. | Hard to query logs; cannot trace a request across services/DB/external calls. | Switch to JSON logging (`structlog` or JSON formatter); add ASGI middleware for `X-Request-ID`. | S-M |
| P2-9 | **Global exception handler does not alert operators** | `backend/main.py:8011-8023` | Logs unhandled exceptions but does not call alerting code. | 500s may go unnoticed until a user reports them. | Wire global handler to dispatch a `CRITICAL` Discord/email/SMS alert for unhandled exceptions. | S |
| P2-10 | **Data-quality dashboard lacks accuracy metrics** | `backend/routers/data_quality.py:25-367` | Tracks freshness, coverage, null fields, failure rate; no projection-accuracy tracking. | Cannot detect that projections are drifting or systematically wrong. | Add accuracy metrics: compare projections to actual box scores (MAE/RMSE by source) and surface on dashboard. | M |
| P2-11 | **No frontend unit-test script** | `frontend/package.json` | Scripts: `dev`, `build`, `start`, `lint`, `test:e2e`; no `test`. | Existing component tests (`streaming-recommendations.test.tsx`, `matchup-strip.test.tsx`) are not runnable via npm. | Add `vitest` (or `jest`) and a `test` script. | S |
| P2-12 | **Duplicate/overlapping scheduler jobs** | `backend/main.py` lifespan + `backend/services/daily_ingestion.py` | Two schedulers run; `yahoo_id_sync` runs at 04:30 (orchestrator) and 06:00 (legacy). | Race conditions, duplicate writes to `player_id_mapping`, wasted resources. | Consolidate on `DailyIngestionOrchestrator`; remove duplicate legacy fantasy jobs from `main.py`. | M |
| P2-13 | **Identity normalization has divergent implementations** | `backend/routers/fantasy.py:352-367`; `backend/services/player_id_resolver.py:28-43`; `backend/fantasy_baseball/id_resolution_service.py:35-43` | Router strips suffixes (`jr.`, `sr.`, `ii`, etc.) and periods; resolvers do not. | Jr./Sr./accented-name players can mismatch between sync and optimizer fallback paths. | Centralize normalization in one utility; add unit tests for accents and suffixes. | S-M |
| P2-14 | **Diagnostic/test routers are still mounted in production** | `backend/main.py:53-79` | Routers marked "REMOVE AFTER ..." are mounted: `_test_router`, `_era_diagnostic_router`, `_validation_audit_router`, `_backfill_ops_whip_router`, `_statcast_diag_router`, `_scoring_diag_router`, `_constraint_migration_router`. | Increased attack surface; potential for destructive backfills if keys leak. | Gate on `ENABLE_DIAGNOSTIC_ENDPOINTS=true`; remove after linked tasks close. | S |

---

## 5. Low Priority (P3 — Polish)

| # | Issue | Location | Evidence | User Impact | Recommended Fix | Effort |
|---|-------|----------|----------|-------------|-----------------|--------|
| P3-1 | **Legacy `datetime.utcnow()` usage throughout CBB modules** | `backend/services/analysis.py`, `performance.py`, `ratings.py`, `sharp_money.py`, `recalibration.py`, etc. (see architecture audit for full list). | Violates `AGENTS.md` quality gate. | Off-by-a-day bugs for ET fantasy deadlines; inconsistent log timestamps. | Add a lint rule banning `datetime.utcnow()`; migrate incrementally, starting with fantasy-facing modules. | M |
| P3-2 | **Unused `js-cookie` dependency** | `frontend/package.json` | Listed but no source import. | Slightly larger bundle; dependency confusion. | Remove or use consistently. | XS |
| P3-3 | **Test suite contains real sleeps and brittle assertions** | `tests/test_matchup_api.py:90` (`time.sleep(6)`); `tests/test_waiver_recovery.py:127` (`time.sleep(1.1)`); `tests/test_yahoo_actions.py` (assertions on internal transaction ID format). | Slow, flaky, or brittle tests. | Replace sleeps with deterministic stubs/clocks; replace internal-format assertions with property-based matchers. | S |
| P3-4 | **Frontend React Query presets are not applied consistently** | `frontend/lib/query-client.ts:12-25` | Default `staleTime`/`gcTime` are reasonable, but pages redeclare their own values inconsistently. | Suboptimal cache behavior. | Define query presets (`live`, `frequent`, `reference`) and apply consistently. | S |
| P3-5 | **`backend/admin_yahoo_token_refresh.py` style: bare `except Exception`** | `backend/admin_yahoo_token_refresh.py:57-59` | Catches all exceptions and returns error string. | Information leakage; poor error handling. | Remove file or catch specific exceptions. | XS |
| P3-6 | **Duplicate `ErrorBoundary` components** | `frontend/components/error-boundary.tsx` and `frontend/components/ui/error-boundary.tsx` | Two nearly identical class components. | Maintenance confusion. | Consolidate into one boundary. | S |

---

## 6. Architectural Recommendations

### Should the monolithic `fantasy.py` be split?

**Yes.** At 8,977 lines and ~62 endpoints, it is a clear god router. The current file comment even describes itself as a "strangler-fig extraction from `backend/main.py`." Split it along subdomain boundaries:

- `routers/draft.py` — draft board, sessions, picks, sync
- `routers/lineup.py` — saved lineups, elite optimization, scarcity analysis
- `routers/waiver.py` — waiver wire, recommendations, add/drop actions
- `routers/roster.py` — roster view, move, bulk-apply, optimize
- `routers/matchup.py` — matchup preview, simulation, scoreboard
- `routers/decisions.py` — decision log, status, accuracy
- `routers/dashboard.py` — dashboard, streaks, waiver targets, stream

Keep `backend/routers/fantasy.py` as a thin aggregator that imports and mounts the subdomain routers. This is a **large** change (2–3 days) but unlocks independent testing, review, and deployment of each subdomain.

### Should the frontend adopt a state management library (Zustand, Redux)?

**Not yet.** The frontend already uses TanStack Query v5 as its primary server-state cache, which is the right choice for this architecture. The problem is not missing global state but **poor query-key governance** and **no global error boundary**. Before adding Zustand/Redux:

1. Centralize every War Room query key in `FANTASY_QUERY_KEYS`.
2. Adopt a shared prefix or `predicate`-based invalidation strategy.
3. Add root `error.tsx`/`global-error.tsx`.

If a true client-only global UI state emerges (e.g., toast notifications, pending-action queue), a small Zustand store would be appropriate, but Redux is overkill.

### Should the data pipeline use a message queue or event-driven architecture?

**Not as a first step.** The current `DailyIngestionOrchestrator` + APScheduler + PostgreSQL advisory-lock pattern is adequate for a single-tenant fantasy league. The more urgent improvements are:

1. **Idempotency and retry** for every external fetch.
2. **Structured per-job results** (`success`, `rows_affected`, `error_count`) instead of silent swallowing.
3. **Alerting** on stale/failed jobs.

If the platform scales to many leagues or real-time streaming workloads, then migrate to a job queue (e.g., Celery + Redis/RabbitMQ or Railway's native queue) with event-driven triggers. Today, a well-instrumented scheduler is cheaper and sufficient.

### Should there be a dedicated data quality service?

**Yes.** Data quality is currently split between `backend/routers/data_quality.py`, `backend/services/health_monitor.py`, inline checks in `daily_ingestion.py`, and the projection-coverage endpoint. Consolidate into a `backend/services/data_quality_service.py` that:

- Owns freshness, coverage, null-field, failure-rate, and **accuracy** metrics.
- Publishes a single data-quality snapshot used by the API and the scheduler health monitor.
- Emits alerts when thresholds are breached.

This removes duplication and gives operators one place to reason about data health.

---

## 7. Testing Strategy

### What tests are missing?

1. **Contract/API tests for the top fantasy endpoints** — `TestClient` tests for `/api/fantasy/waiver`, `/api/fantasy/roster`, `/api/fantasy/lineup`, `/api/fantasy/matchup`, `/api/fantasy/scoreboard` with a seeded test DB.
2. **Integration tests that exercise Yahoo → DB → API** — mock the Yahoo client boundary, run ingestion, call the API, assert response shape.
3. **End-to-end tests against a real backend** — at least one Playwright smoke test that hits a staging/local backend for the roster-move path (currently all E2E specs mock every endpoint).
4. **Coverage for `backend/main.py`, `backend/schemas.py`, and `backend/fantasy_baseball/*`** — remove these omissions from CI coverage.
5. **Data-pipeline tests** — run `DailyIngestionOrchestrator` stages with mocked external sources against a throwaway DB.
6. **Security tests** — unauthenticated mutation endpoints should return 401; CORS should reject unknown origins.

### What testing patterns should be adopted?

- **Contract testing** — use FastAPI `TestClient` + Pydantic response models to ensure API contracts do not drift.
- **Property-based tests** for scoring math (`need_score`, z-score, category directions) to catch subtle formula regressions.
- **Component tests** with `vitest`/`React Testing Library` for critical UI components (`streaming-recommendations`, `action-modal`, `yahoo-roster-view`).
- **E2E with Playwright** against a seeded local or staging environment for the critical user journeys: login → view roster → optimize → apply lineup → waiver action.
- **Mutation testing** (e.g., `mutmut`) on scoring modules after coverage improves.

### How to prevent regressions?

1. **Fail CI on flake8 F-errors and on any `datetime.utcnow()` in new code.**
2. **Require full-backend coverage to not decrease** in PRs.
3. **Add a smoke-test job** that starts the app and hits `/health`, `/api/fantasy/projection-status`, and `/api/fantasy/roster` with a test API key.
4. **Add a pre-deploy checklist** that verifies route registration order after any `main.py` or router change.
5. **Tag tests by tier** (`unit`, `integration`, `e2e`) so CI can run fast unit tests on PRs and full suite nightly.

---

## 8. Production Readiness Checklist

Before declaring the platform "production-grade," the following must be completed:

### Security
- [ ] `POST /api/fantasy/roster/move` and `POST /api/fantasy/roster/bulk-apply` require `verify_api_key`.
- [ ] All 15 unauthenticated routes in `backend/routers/fantasy.py` are either guarded or explicitly public.
- [ ] `backend/admin_yahoo_token_refresh.py` is deleted.
- [ ] `ALLOWED_ORIGINS` is required in production and CORS no longer defaults to `["*"]`.
- [ ] `backend/services/cat_scores_builder.py` uses parameterized `IN` clauses.
- [ ] Diagnostic/test routers are gated by `ENABLE_DIAGNOSTIC_ENDPOINTS=true` or removed.
- [ ] API-key cookie is `HttpOnly; Secure` or removed in favor of `localStorage` for client-only usage.

### Architecture / Code Health
- [ ] Inline fantasy/dashboard routes are removed from `backend/main.py`.
- [ ] `backend/routers/fantasy.py` is split into subdomain routers.
- [ ] Import cycles between routers, services, and `main.py` are broken.
- [ ] Canonical scoring maps, z-score, and need-score logic live in single-source modules.
- [ ] Routers delegate persistence to subdomain services.

### Data Pipeline Integrity
- [ ] `find_alternative_player_score` is removed from the optimizer hot path or emits an alert on every trigger.
- [ ] External API clients have uniform retry/backoff.
- [ ] Silent `except Exception` handlers return structured results and emit metrics.
- [ ] Manual `source='manual'` rows in `player_id_mapping` are protected from overwrite.
- [ ] A source-of-truth registry documents precedence for projections, injury status, ownership %, and stat semantics.

### Frontend Reliability
- [ ] `npm run build` passes in CI.
- [ ] `NEXT_PUBLIC_API_URL` fallback to `localhost:8000` is removed.
- [ ] Root `error.tsx` and `global-error.tsx` exist.
- [ ] All War Room query keys are centralized and invalidation is predictable.
- [ ] Loading skeletons exist for `/war-room/roster`, `/war-room/waiver`, `/war-room/streaming`, `/decisions`.

### Testing
- [ ] Full-backend coverage gate is in place (no omissions for `main.py`, `schemas.py`, `fantasy_baseball/*`).
- [ ] `TestClient` contract tests exist for the top 5 fantasy endpoints.
- [ ] At least one E2E Playwright test runs against a seeded backend.
- [ ] Full test suite runs in CI in under 10 minutes (currently times out at 5 minutes).

### Observability & Operations
- [ ] Logs are JSON-formatted in production.
- [ ] `X-Request-ID` middleware propagates correlation IDs.
- [ ] Global exception handler dispatches `CRITICAL` alerts.
- [ ] Data-quality dashboard includes accuracy metrics.
- [ ] Railway 503 root cause on dashboard route is mitigated (N+1 fixes + pagination + query optimization).
- [ ] Runbook exists for Yahoo token refresh, scheduler recovery, and pipeline starvation.

### Performance & Load
- [ ] Load test the waiver and roster endpoints with realistic player counts.
- [ ] All list endpoints support pagination or streaming.
- [ ] DB hot paths are reviewed for N+1 queries and missing indexes.

### User Acceptance
- [ ] End-to-end UAT of login → roster → optimize → apply lineup → waiver add/drop → matchup simulation.
- [ ] Verify mobile responsiveness on iOS Safari and Android Chrome.
- [ ] Confirm error messages are user-friendly (no raw Yahoo XML).

---

## Appendix: Key Evidence Summary

- `backend/routers/fantasy.py` — 8,977 lines, ~62 endpoints, 85 direct DB calls, 11 raw SQL blocks.
- `backend/main.py` — 8,028 lines, ~147 inline routes, registers fantasy routes before mounting `routers/fantasy.py`.
- Static import-cycle detection found **6 cycles**, including `daily_ingestion → projection_coverage → routers/fantasy → daily_ingestion` and `main → test_sync_jobs → daily_ingestion → projection_coverage → routers/fantasy → main`.
- Test inventory: 198 Python test files, ~3,175 test functions, **3,276 tests** collected, but full suite timed out at 300 s.
- Measured coverage: **48 %** overall; true coverage is lower because `.coveragerc` omits `main.py`, `schemas.py`, `fantasy_baseball/*`, `scripts/*`, and migrations.
- `backend/routers/fantasy.py` itself is only **31 %** covered; `backend/services/daily_ingestion.py` is **15 %** covered.
- Frontend: 2 component tests, 2 Playwright E2E specs, **no `test` script** in `package.json`, and a current build failure.

*End of assessment.*
