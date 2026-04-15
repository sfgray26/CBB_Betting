# CBB Edge — Task Tracker
*Updated: 2026-04-15 (handoff recovery checkpoint) | Architect: Claude Sonnet 4.6 | Mission: Fantasy Baseball stabilization Phase A*

> **Canonical source:** `HANDOFF.md` — full specs, ADRs, exit criteria for each EPIC.
> This file is the status board. HANDOFF.md has the implementation detail.

---

## System Status

| Subsystem | Status | Notes |
|-----------|--------|-------|
| V9.1 CBB Model | FROZEN until Apr 7 | Guardian active |
| Frontend (Next.js) Phases 0-5 | DONE | All pages live on main |
| Fantasy Draft + Yahoo OAuth | DONE (Mar 20) | Live draft, value-board, sync endpoint |
| Fantasy Lineup / Waiver endpoints | DONE (Mar 20) | `/fantasy/lineup`, `/fantasy/waiver` live |
| Admin Risk Dashboard | DONE (Mar 20) | `/admin` — 4-panel ops view |
| MLB+PGA Expansion Blueprint | REVIEWED (Mar 23) | Critical bugs identified — see `reports/EXPANSION_ARCHITECTURE_MLB_PGA_BLUEPRINT.md`. No implementation tickets open yet. |
| EPIC-1: Time-Series Schema | **NOT STARTED** | First priority |
| EPIC-2: Ingestion Orchestrator | NOT STARTED | Blocked on EPIC-1 exit criteria |
| EPIC-3: Edge Generation Engine | NOT STARTED | Blocked on EPIC-2 |
| EPIC-4: Waiver Edge Detector | NOT STARTED | Blocked on EPIC-3 |
| EPIC-5: Sport Polling Switch | NOT STARTED | CBB wind-down, MLB ramp-up |
| EPIC-6: Discord Router | NOT STARTED | Blocked on EPIC-4+5 |

---

## GUARDIAN FREEZE (until Apr 7)

**DO NOT TOUCH:**
- `backend/betting_model.py`
- `backend/services/analysis.py`
- Any CBB model services

---

## Active Priority Queue

### 0A. Fantasy Production Audit Hotfix — Apr 3 (COMPLETED)

| Task | File | Done? |
|------|------|-------|
| Harden lineup apply payload: strict Yahoo key sanitization + OF fallback + ET date | `backend/main.py`, `frontend/app/(dashboard)/fantasy/lineup/page.tsx` | [x] |
| Fix weather API compatibility: OneCall free-tier fallback chain | `backend/fantasy_baseball/weather_fetcher.py` | [x] |
| Filter matchup payload to active scoring categories only | `backend/main.py` | [x] |
| Stabilize waiver stat rendering and NSV reliever-only fallback | `backend/main.py` | [x] |
| Add direct waiver add/drop API actuation + frontend button wiring | `backend/main.py`, `frontend/lib/api.ts`, `frontend/app/(dashboard)/fantasy/waiver/page.tsx` | [x] |
| Validation: py_compile + frontend typecheck | backend + frontend | [x] |

**Review:** Critical / High fantasy production regressions addressed in code with compile and type checks passing. Remaining medium UX redesign items (dashboard widget model and roster trend data source) need separate scoped implementation.

### 0. Fantasy Stabilization — Phase A (ACTIVE)
**Spec:** `HANDOFF.md` fantasy in-season pipeline sections | **Priority:** highest until stale-data risk is removed

| Task | File | Done? |
|------|------|-------|
| Unify ET date anchoring for ingestion + fantasy lineup hot paths | `backend/services/daily_ingestion.py`, `backend/main.py`, `backend/utils/time_utils.py` | [x] |
| Enforce projection freshness gate on lineup endpoint with `force_stale` override | `backend/main.py` | [x] |
| Make fallback weather scoring temperature-aware and expose fallback flag | `backend/fantasy_baseball/weather_fetcher.py` | [x] |
| Add regression coverage for freshness gate and weather fallback | `tests/test_waiver_integration.py`, `tests/test_weather_fetcher.py` | [x] |
| Canonicalize backend/frontend stat ID contract | `frontend/lib/fantasy-stat-contract.json`, `backend/utils/fantasy_stat_contract.py`, `backend/main.py`, `backend/fantasy_baseball/category_tracker.py`, `frontend/lib/constants.ts` | [x] |
| Replace `_ROS_CACHE` with durable DB-backed handoff | `backend/services/daily_ingestion.py`, `backend/models.py` | [x] |
| Convert ensemble write path to atomic upsert counters | `backend/services/daily_ingestion.py` | [ ] |
| Split retryable vs fatal job queue failures | `backend/services/job_queue_service.py` | [ ] |

**Checkpoint verification completed:** `py_compile` on touched backend files plus targeted pytest subset (`test_ingestion_orchestrator.py`, `test_fantasy_stat_contract.py`, `test_waiver_integration.py`, `test_weather_fetcher.py`) are green, and `frontend` passes `npx tsc --noEmit`.

### 0C. P0 Observability + Health Semantics — Apr 15 (ACTIVE)
**Spec:** `HANDOFF.md` P0 queue | **Priority:** restore durable pipeline auditing and remove false-green health signals

| Task | File | Done? |
|------|------|-------|
| Persist durable job audit rows at the advisory-lock boundary | `backend/services/daily_ingestion.py` | [x] |
| Reclassify empty `probable_pitchers` and `data_ingestion_logs` as degraded in health validation | `backend/services/pipeline_validator.py`, `backend/admin_endpoints_validation.py` | [x] |
| Validate compile + targeted orchestrator / validation tests | backend + tests | [x] |
| Verify live row creation and degraded endpoint output in production | Railway / admin endpoints | [ ] |

**Review:** Code implementation is complete and targeted validation passed locally (`py_compile`, `tests/test_ingestion_orchestrator.py`, `tests/test_admin_validation_audit.py`). Remaining work is production verification after the next scheduled or manual ingestion run.

### 0D. Probable Pitcher Resilience — Apr 15 (ACTIVE)
**Spec:** `HANDOFF.md` probable-pitcher resilience section | **Priority:** restore usable starter inputs for two-start and matchup-quality features

| Task | File | Done? |
|------|------|-------|
| Add shared conservative fallback helper for probable-pitcher inference | `backend/services/probable_pitcher_fallback.py` | [x] |
| Use fallback inference in scheduled probable-pitcher sync when official probables are absent | `backend/services/daily_ingestion.py` | [x] |
| Prefer persisted probable-pitcher snapshots before live API fetches in lineup reads | `backend/fantasy_baseball/daily_lineup_optimizer.py` | [x] |
| Add regression coverage for snapshot preference and conservative 5-day-cycle inference | `tests/test_ingestion_orchestrator.py`, `tests/test_probable_pitcher_fallback.py` | [x] |
| Validate compile + targeted probable-pitcher tests | backend + tests | [x] |
| Verify live `probable_pitchers` population in production | Railway / admin endpoints | [ ] |

**Review:** Local implementation is complete and validated (`py_compile`, `tests/test_ingestion_orchestrator.py`, `tests/test_probable_pitcher_fallback.py`). Remaining work is production verification that scheduled runs now populate usable `probable_pitchers` rows.

### 0B. HANDOFF Recovery — Apr 15 (ACTIVE)
**Spec:** `HANDOFF.md` operator recovery rewrite | **Priority:** immediate documentation cleanup so pipeline work resumes from a reliable source of truth

| Task | File | Done? |
|------|------|-------|
| Remove contradictory pipeline status across resolved vs pending items | `HANDOFF.md` | [x] |
| Reframe executive state into resolved, live risks, upstream blockers, and next actions | `HANDOFF.md` | [x] |
| Clarify MLB provider policy: BDL primary, Odds API allowed where additive or resilient | `HANDOFF.md`, `CLAUDE.md` | [x] |
| Tighten fantasy pipeline priority queue around observability, probable pitchers, environment factors, and decision breadth | `HANDOFF.md` | [x] |
| Add review notes and confirm cold-start readability | `HANDOFF.md` | [x] |

**Review:** HANDOFF now distinguishes infrastructure health from fantasy decision-quality health, removes contradictory migration/commit state, and documents a deliberate dual-source MLB data policy with BDL preferred and Odds API allowed where it adds signal or resilience.

---

### 1. EPIC-1 — Time-Series Schema (DO FIRST)
**Spec:** `HANDOFF.md §2` | **No prerequisites**

| Task | File | Done? |
|------|------|-------|
| Write `upgrade()` + `downgrade()` for `player_daily_metrics` | `scripts/migrate_v8_post_draft.py` | [ ] |
| Write `upgrade()` + `downgrade()` for `projection_snapshots` | same | [ ] |
| Add `pricing_engine` column to `predictions` (K-14) | same | [ ] |
| Add SQLAlchemy ORM models (`PlayerDailyMetric`, `ProjectionSnapshot`) | `backend/models.py` | [ ] |
| Write 7 required tests | `tests/test_schema_v8.py` | [ ] |
| Dry-run test locally | — | [ ] |
| Run migration on Railway | — | [ ] |
| Verify schema on Railway DB | — | [ ] |

**Exit gate:** All 8 criteria in `HANDOFF.md §2.5` must be TRUE before EPIC-2 starts.

---

### 2. EPIC-2 — Ingestion Orchestrator (after EPIC-1)
**Spec:** `HANDOFF.md §3` | **Prerequisite: EPIC-1 exit criteria**

Key deliverables:
- `backend/services/daily_ingestion.py` — `DailyIngestionOrchestrator` with advisory lock wrapper (ADR-001)
- `backend/services/clv.py` — add `compute_daily_clv_attribution()` (additive only)
- `/admin/ingestion/status` endpoint
- `tests/test_ingestion_orchestrator.py`
- Mount under `ENABLE_INGESTION_ORCHESTRATOR=true` env var

---

### 3. Task 3: Verify Cross-System Joins (BLOCKED)
**Spec:** Execute two verification queries to confirm position_eligibility, player_id_mapping, and mlb_player_stats joins work

**BLOCKER**: Environment access constraints prevent query execution:
- Local PostgreSQL: No local DB instance
- `railway run`: Cannot resolve Railway internal DNS (postgres-ygnv.railway.internal)
- `railway shell`: Interactive only, not scriptable

**REQUIRED**: Temporary verification endpoint `/admin/debug-verify-joins` in backend/main.py → execute via curl → capture results → remove endpoint

**DELEGATION BUNDLE**: Gemini CLI to create endpoint, execute queries, report results

---

## Post-Apr 7 Backlog (GUARDIAN-LOCKED)

| Item | Spec | Priority |
|------|------|----------|
| V9.2 params (sd_mult 1.0→0.80, ha 2.419→2.85, SNR_KELLY_FLOOR 0.50→0.75) | `reports/K12_RECALIBRATION_SPEC_V92.md` | HIGH |
| Wire Haslametrics as 3rd rating source | `docs/THIRD_RATING_SOURCE.md` | HIGH |
| `pricing_engine` field on Prediction (K-14) — done in EPIC-1 | — | COMPLETE |
| Oracle validation (K-15) | `reports/K15_ORACLE_VALIDATION_SPEC.md` | **COMPLETE** (Mar 23) |
| Fantasy Baseball Yahoo OAuth wiring | `docs/MLB_FANTASY_ROADMAP.md` | LOW |
| Pre-draft keeper sweep endpoint | `backend/main.py` `POST /sync-keepers` | **COMPLETE** (Mar 23) |

---

## MLB+PGA Expansion (Future — Not Yet Scheduled)

Blueprint reviewed Mar 23. **6 critical bugs identified** — must be fixed before any implementation:
1. `sport_type` enum references — replace with VARCHAR migrations
2. `calculate_stuff_plus` variable shadowing (`break_z`)
3. `mlb_player_stats` UNION ALL — mismatched columns
4. `pga_shots` RANGE partition on VARCHAR — use LIST or drop partition
5. `can_request()` priority bypass — clv_critical skips all budget guards
6. CBB budget (800/mo) incompatible with live system (~11,610/mo) without EPIC-5 first

No implementation tickets open until EMAC-077 EPIC-1 is complete and tournament is past Apr 7.

---

## Pending Manual Actions

| Item | Owner | Action |
|------|-------|--------|
| Push `v0.8.0-cbb-stable` tag | User | `git push origin v0.8.0-cbb-stable` |
| Confirm `NEXT_PUBLIC_API_URL` in Railway frontend | User | Railway dashboard |
| Set `ENABLE_INGESTION_ORCHESTRATOR=false` in Railway | User | Before EPIC-2 is deployed |

---

## Done Archive

| Description | Date |
|-------------|------|
| MLB+PGA Expansion Blueprint reviewed, critical bugs documented | Mar 23 |
| K-15 Oracle Validation System (oracle_validator.py, DB columns, admin endpoint, 19 tests) | Mar 23 |
| Pre-draft keeper sweep: POST /api/fantasy/draft-session/{key}/sync-keepers | Mar 23 |
| Fantasy Season Ops — full stack (/fantasy/lineup, /fantasy/waiver + backend endpoints) | Mar 20 |
| Admin Risk Dashboard (/admin — 4-panel) | Mar 20 |
| Fantasy Draft Assistant (Live Draft tab, snake order, roster panel) | Mar 20 |
| Fantasy DB Migration v7 on Railway | Mar 20 |
| Frontend Phase 5 (error.tsx + loading.tsx) | Mar 20 |
| Frontend Phase 4 Mobile and PWA (viewport, manifest, drawer, responsive grids) | Mar 20 |
| Frontend Phase 3 Tournament (/bracket 10k MC sims) | Mar 19 |
| Frontend Phase 2 Trading (/today, /live-slate, /odds-monitor) | Mar 19 |
| Frontend Phase 1 Core Analytics (5 pages, OpenClaw validated) | Mar 18 |
| Frontend Phase 0 scaffold + auth + layout | Mar 18 |
| Railway CORS fix | Mar 18 |
| Monte Carlo bracket simulator | Mar 16 |
| Discord morning brief + EOD results jobs fixed | Mar 16 |
| Team mapping hardening (29 St variants, 78 tests) | Mar 16 |
| V9.1 confidence engine (SNR + integrity scalars) | Mar 12 |
