# HANDOFF.md — Fantasy Baseball Platform Master Plan (In-Season 2026)

> **Date:** April 5, 2026 (updated Session S13) | **Author:** Claude Code (Master Architect)
> **Risk Level:** ELEVATED — Fantasy data layer under embargo; raw ingestion unvalidated

---

## ACTIVE DIRECTIVES (read before every session)

### DIRECTIVE 1 — Fantasy Data-First Mandate (STRIP-BACK)

**The assumption that the fantasy pipeline is "LIVE" and reliable is false.**

The projection pipeline was built layer-by-layer but the raw ingestion foundations have not been independently validated. We are executing a deliberate strip-back: validate the data floor before any math runs on top of it.

**HARD EMBARGO — do not lift without explicit human instruction:**
- Lineup optimization
- Projection blending
- Ensemble update (job 100_014)
- FanGraphs RoS ingestion (job 100_012)
- Yahoo ADP/injury polling (job 100_013)
- Any derived stats (wOBA, FIP, barrel%, etc.)

**The only active priority:** prove that raw Yahoo roster state and BallDontLie MLB data can be ingested daily with zero silent failures, validated end-to-end through strict Pydantic V2 models, before any row is written to the relational database or surface reaches the UI.

**Nothing proceeds to the DB or UI until:** incoming payloads pass strict Pydantic V2 validation models. Every field, every type, every nullable must be explicitly declared and verified against live API responses.

### DIRECTIVE 2 — Phase 6-7 Deployment Sequence (Boot/Migration Deadlock Fix)

The previous handoff had Gemini deploying the standalone fantasy service before the database schema existed. That will crash on boot. The correct sequence is strictly ordered:

```
Step 1 (Gemini):  Provision Fantasy Postgres. Set FANTASY_DATABASE_URL on the fantasy service.
                  DO NOT start the service yet.

Step 2 (Claude):  Run database migrations against FANTASY_DATABASE_URL.
                  Verify schema with: SELECT table_name FROM information_schema.tables
                  WHERE table_schema = 'public';

Step 3 (Gemini):  Deploy fantasy_app.py standalone service ONLY after Claude confirms Step 2.
                  Start Command: uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT
```

**Gemini must not deploy until receiving explicit confirmation from Claude that migrations ran clean.**

### DIRECTIVE 3 — Strangler-Fig Scheduler Duplication (Race Condition Fix)

If the new `fantasy_app.py` boots with `ENABLE_INGESTION_ORCHESTRATOR=true` while the legacy `main.py` service is still running, both containers will execute the same ingestion jobs simultaneously against the same database. This will cause duplicate writes, advisory lock timeouts, and data corruption.

**Required sequence before booting the new fantasy service:**

1. Set `ENABLE_FANTASY_SCHEDULER=false` on the **legacy CBB `main.py` service** in Railway. This disables its fantasy scheduler start path without affecting CBB betting jobs.
2. Verify the legacy service has redeployed and its fantasy scheduler is not running (check `/admin/scheduler/status`).
3. Only then start the fantasy_app.py service.

This variable must be respected in `backend/main.py`'s lifespan. **Claude to implement** `ENABLE_FANTASY_SCHEDULER` guard in the lifespan before Phase 6-7 cut-over.

### DIRECTIVE 4 — Redis / Advisory Lock Contention

The advisory lock registry (100_001–100_015) uses **PostgreSQL `pg_try_advisory_lock`**, not Redis. Once the fantasy service has its own Postgres (Phase 6-7), the lock namespaces are physically separate — no collision possible.

**However, during the transition period** (both services pointing at the same Postgres), the edge and fantasy services share the same advisory lock integer namespace. Lock IDs must not overlap. Current assignment is safe: 100_001–100_010 are edge/CBB jobs; 100_011–100_015 are fantasy jobs. This must not change.

**Redis shared instance:** Both services share one Railway Redis. The `NamespacedCache` in `backend/redis_client.py` enforces `edge:` and `fantasy:` key prefixes at the application layer. This is sufficient for cache keys. It does **not** protect advisory locks (those are Postgres). No separate Redis instance is required provided the key prefix discipline is maintained in all new code.

**Requirement for any new Redis key written by either service:** it must use `edge_cache.set(...)` or `fantasy_cache.set(...)` — never raw `redis.set(...)`.

### DIRECTIVE 5 — No LLM Time-Gates

Do not write "execute on April 7" or any date-based trigger in agent instructions. LLMs cannot be trusted to parse the current system date reliably.

**CBB V9.2 recalibration (EMAC-068):** Embargoed until the human developer explicitly inputs the trigger phrase:

> **"The EMAC-068 block is lifted. Execute Task A."**

Until that phrase is received, treat Kelly math as completely frozen. Do not read the spec. Do not plan the implementation. Do not estimate timing.

**OddsAPI → BDL migration:** Embargoed until the human developer explicitly inputs:

> **"OddsAPI is cancelled. Execute the BDL migration."**

---

## Platform State — April 4, 2026

| System | State | Notes |
|--------|-------|-------|
| Fantasy projection pipeline | **EMBARGOED** | Jobs 100_012, 100_013, 100_014 disabled pending raw data validation |
| Fantasy raw ingestion (Yahoo + BDL) | **NEEDS VALIDATION** | Pydantic V2 contract verification not yet done |
| Fantasy lineup optimizer | **EMBARGOED** | Blocked until ingestion layer certified |
| CBB Betting Model | **FROZEN** | Kelly math blocked — lift only on human trigger phrase |
| OddsAPI Champion | **EXPIRING** | Migrate only on human trigger phrase |
| BDL NCAAB | **DEAD** | Subscription cancelled — never call `/ncaab/v1/` |
| Fantasy/Edge structural split | **PHASES 1-5 DONE** | New entry points exist; Phase 6-7 needs correct deploy sequence (Directive 2) |

---

## What's Actually Pending (Ordered by Priority)

### Priority 1 — Validate Raw Ingestion Layer (Fantasy, Claude)

**Do this before anything else.**

The task: write strict Pydantic V2 models for Yahoo and BallDontLie API responses, run live fetches, and prove the data is clean before allowing any downstream job to consume it.

**Deliverables:**

1. `backend/fantasy_baseball/validators/yahoo_roster_validator.py` — Pydantic V2 model for Yahoo roster API response. Every field typed. Nullable fields explicit. Validate a live `get_team_roster()` call and log any validation errors.

2. `backend/fantasy_baseball/validators/yahoo_waiver_validator.py` — Pydantic V2 model for Yahoo free agent API response. Validate a live `get_free_agents()` call.

3. `backend/fantasy_baseball/validators/bdl_game_validator.py` — Pydantic V2 model for BallDontLie `/mlb/v1/games` response. Validate a live fetch for today's schedule.

4. Each validator must: parse the live payload, log every field that is missing or type-mismatched, return a `ValidationReport` with pass/fail counts.

5. No data is written to the database during this phase. Validation only.

6. Report: per-endpoint pass rate, any schema mismatches against what the code currently assumes, and explicit confirmation that the raw data is safe to consume.

**Only after this is confirmed clean:** re-enable 100_013 (Yahoo ADP/injury). Jobs 100_012 and 100_014 remain embargoed until the blending logic is re-validated separately.

### Priority 2 — ENABLE_FANTASY_SCHEDULER Guard (Claude, before Phase 6-7)

Implement the `ENABLE_FANTASY_SCHEDULER` environment variable guard in `backend/main.py` lifespan. When `ENABLE_FANTASY_SCHEDULER=false`, the legacy service must not start the fantasy ingestion orchestrator, the job_queue_processor for fantasy jobs, or any of jobs 100_012–100_015.

This is required before Phase 6-7 cut-over (Directive 3).

### Priority 3 — Phase 6-7 Infrastructure (Gemini → Claude → Gemini)

Execute strictly per Directive 2 sequence. Do not skip steps.

### Priority 4 — CBB V9.2 Recalibration

**Embargoed.** Execute only on human trigger: `"The EMAC-068 block is lifted. Execute Task A."`

### Priority 5 — OddsAPI → BDL Migration

**Embargoed.** Execute only on human trigger: `"OddsAPI is cancelled. Execute the BDL migration."`

---

## Job Registry — Current Status

| Job | Lock | Cadence | Status |
|-----|------|---------|--------|
| `statcast` | 100_002 | Daily 2 AM ET | LIVE |
| `rolling_z` | 100_003 | Daily 3 AM ET | LIVE |
| `waiver_scan` | 100_007 | Daily 6 AM ET | LIVE |
| `mlb_brief` | 100_008 | Daily 7 AM ET | LIVE |
| `valuation_cache` | 100_011 | On demand | LIVE |
| `mlb_odds` | 100_001 | Every 30 min | DIRTY — migrate on trigger |
| `fangraphs_ros` | 100_012 | Daily 3 AM ET | **EMBARGOED** |
| `yahoo_adp_injury` | 100_013 | Every 4h | **EMBARGOED** |
| `ensemble_update` | 100_014 | Daily 5 AM ET | **EMBARGOED** |
| `projection_freshness_check` | 100_015 | Every 1h | LIVE (gate active; SLA will flag embargoed jobs as missing — expected) |

**Next available lock ID:** 100_016

**Advisory lock namespace rule:** 100_001–100_010 are edge/CBB. 100_011–100_015 are fantasy. Do not cross-assign.

---

## Hard Stops

| Rule | Reason |
|------|--------|
| Do NOT run ensemble blender (100_014) | Directive 1 — raw data not validated |
| Do NOT run FanGraphs RoS fetch (100_012) | Directive 1 — embargoed |
| Do NOT run Yahoo ADP/injury poll (100_013) | Directive 1 — embargoed |
| Do NOT run lineup optimization | Directive 1 — embargoed |
| Do NOT modify Kelly math in `betting_model.py` | Frozen — lift only on human trigger phrase |
| Do NOT call BDL `/ncaab/v1/` endpoints | Subscription cancelled — will 401 |
| Do NOT add `THE_ODDS_API_KEY` dependencies | Phasing out — migrate only on human trigger |
| Do NOT touch `dashboard/` (Streamlit) | Retired — Next.js is canonical |
| Do NOT use `datetime.utcnow()` for game times | Use `datetime.now(ZoneInfo("America/New_York"))` |
| Do NOT write test files outside `tests/` | Architecture locked |
| Do NOT import `betting_model` from fantasy modules | GUARDIAN FREEZE / ADR-004 |
| Do NOT import `backend.models_edge` from `backend.models_fantasy` | Hard architectural boundary |
| Do NOT deploy fantasy_app.py before DB migrations run | Directive 2 — boot/migration deadlock |
| Do NOT boot new fantasy service before disabling legacy fantasy scheduler | Directive 3 — race condition |
| Do NOT write raw Redis keys without namespace prefix | Directive 4 — use edge_cache or fantasy_cache |

---

## HANDOFF PROMPTS

### Claude Code — Priority 1: Raw Ingestion Validation

Execute in `C:\Users\sfgra\repos\Fixed\cbb-edge`. Read `HANDOFF.md` and `CLAUDE.md` first.

**Context (read carefully):** The fantasy projection pipeline is under a data-first embargo. All ensemble blending, lineup optimization, and derived stats are disabled until the raw ingestion layer is independently validated through strict Pydantic V2 models. Your job is to build those validators and run them against live APIs. No database writes. No math. Validation only.

**Task 1 — Yahoo Roster Validator:**

Create `backend/fantasy_baseball/validators/__init__.py` (empty) and `backend/fantasy_baseball/validators/yahoo_roster_validator.py`.

The validator must:
- Define a Pydantic V2 model hierarchy for the Yahoo roster API response structure. Start with a real live call: `from backend.fantasy_baseball.yahoo_client_resilient import get_resilient_yahoo_client; client = get_resilient_yahoo_client(); roster = client.get_team_roster()` — inspect the raw response and model every field.
- Define a `ValidationReport` dataclass: `{endpoint: str, total_records: int, passed: int, failed: int, field_errors: list[str]}`.
- The `validate_roster(raw_response)` function must return a `ValidationReport`. Log every field that is null when it shouldn't be, wrong type, or absent.
- Run it: `railway run python -c "from backend.fantasy_baseball.validators.yahoo_roster_validator import validate_roster; ..."` and report the result.

**Task 2 — Yahoo Waiver Validator:**

Same pattern. Create `backend/fantasy_baseball/validators/yahoo_waiver_validator.py`. Validate a live `get_free_agents()` response. The key fields to validate: `player_key`, `name`, `eligible_positions`, `percent_rostered` (must be float 0-100, not string, not None). Log any stat ID that appears in the response but is not in `YAHOO_STAT_ID_FALLBACK`.

**Task 3 — BallDontLie Game Validator:**

Create `backend/fantasy_baseball/validators/bdl_game_validator.py`. Hit the BDL MLB games endpoint for today's date. Validate: `game_id` (int), `home_team` (object with `id`, `full_name`), `visitor_team`, `date` (ISO date string), `status` (one of `Final`, `In Progress`, `Scheduled`). Report any games where `status` is an unexpected value.

**Task 4 — Test coverage:**

Write `tests/test_validators.py` with unit tests for each validator using mocked payloads that intentionally include bad data (wrong types, missing fields). Tests must demonstrate that the validator catches the bad data and reports it rather than silently passing.

Run: `venv/Scripts/python -m pytest tests/test_validators.py -q --tb=short`

**Task 5 — Compile check all new files:**
```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/validators/yahoo_roster_validator.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/validators/yahoo_waiver_validator.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/validators/bdl_game_validator.py
```

**Task 6 — Report back with:**
- Pass/fail counts from live validation runs
- Any field mismatches found (schema vs actual API response)
- Explicit recommendation: is job 100_013 (Yahoo ADP/injury poll) safe to re-enable? State the evidence.
- Any schema assumptions in existing code that are wrong

Commit: `git commit -m "feat: Pydantic V2 validators for Yahoo and BDL raw ingestion -- data-first mandate"`

---

### Claude Code — Priority 2: ENABLE_FANTASY_SCHEDULER Guard

Execute only after Priority 1 validation is complete and reported.

**Context:** Before the new `fantasy_app.py` service can be deployed standalone, the legacy `main.py` service must stop running fantasy scheduler jobs. Otherwise both containers execute the same ingestion jobs simultaneously against the same database (race condition, duplicate writes, advisory lock timeouts).

**Task:** Add an `ENABLE_FANTASY_SCHEDULER` environment variable guard to `backend/main.py` lifespan.

When `ENABLE_FANTASY_SCHEDULER=false`:
- Do NOT start the `DailyIngestionOrchestrator`
- Do NOT register the `job_queue_processor` job (100_016+ fantasy jobs)
- Do NOT register jobs 100_012, 100_013, 100_014, 100_015
- DO continue running all CBB/edge jobs (nightly_analysis, update_outcomes, etc.)
- Log clearly: `"Fantasy scheduler disabled by ENABLE_FANTASY_SCHEDULER=false"`

Default value if env var is absent: `true` (backwards-compatible — legacy service keeps running as-is until Gemini explicitly sets it to false during Phase 6-7).

Write a test in `tests/test_main_scheduler_guard.py` that mocks `os.getenv("ENABLE_FANTASY_SCHEDULER", "true")` returning `"false"` and asserts that `DailyIngestionOrchestrator` is never instantiated during lifespan startup.

Compile check: `venv/Scripts/python -m py_compile backend/main.py && echo OK`
Test: `venv/Scripts/python -m pytest tests/test_main_scheduler_guard.py -q`

Commit: `git commit -m "feat: ENABLE_FANTASY_SCHEDULER guard in main.py lifespan -- prevents scheduler race on cut-over"`

---

### Gemini CLI — Phase 6-7 Infrastructure (Execute ONLY after Claude confirms Priority 2 is done)

**You are Step 1 and Step 3. Claude is Step 2. Do not do Step 3 before Claude does Step 2.**

Read this entire prompt before taking any action.

**Step 1 — Provision Fantasy Postgres and set env vars:**

```bash
railway status
railway service list
```

In the Railway dashboard, add a new PostgreSQL database service. Name it `fantasy-postgres`. Copy the `DATABASE_URL` it generates.

On the fantasy service (the service that will run `fantasy_app.py`), set these environment variables:
```
FANTASY_DATABASE_URL   = <connection string from fantasy-postgres>
DEPLOYMENT_ROLE        = fantasy-prod
ENABLE_MAIN_SCHEDULER  = false
ENABLE_FANTASY_SCHEDULER = true
ENABLE_INGESTION_ORCHESTRATOR = false    ← keep disabled until raw data validated
```

On the **legacy CBB main.py service**, set:
```
ENABLE_FANTASY_SCHEDULER = false
```

Then trigger a redeploy of the legacy CBB service and confirm it comes up healthy:
```bash
curl -s https://<cbb-service-url>/health | python -m json.tool
```
Expected: `{"status": "healthy", ...}` and logs should show `"Fantasy scheduler disabled by ENABLE_FANTASY_SCHEDULER=false"`.

**Report to Claude:** "Step 1 complete. FANTASY_DATABASE_URL is set. Legacy service redeployed with ENABLE_FANTASY_SCHEDULER=false. Health check passed."

**Step 2 — WAIT.** Do not proceed until Claude explicitly confirms: "Migrations complete. Fantasy DB schema verified. Ready for Step 3."

**Step 3 — Deploy fantasy_app.py standalone:**

Only after receiving Claude's confirmation above:

In Railway, set the Start Command on the fantasy service to:
```
uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT
```

Deploy. Smoke test:
```bash
curl -s https://<fantasy-service-url>/health | python -m json.tool
curl -s -o /dev/null -w "%{http_code}" https://<fantasy-service-url>/api/predictions/today
# Expected: 404 (edge route not mounted on fantasy service)
curl -s -o /dev/null -w "%{http_code}" https://<fantasy-service-url>/api/fantasy/roster
# Expected: 200 (fantasy route is mounted)
```

Report: fantasy service URL, health check output, smoke test HTTP codes.

**Zero `.py` file changes throughout. TypeScript check if any frontend files touched: `cd frontend && npx tsc --noEmit`.**

---

### Kimi CLI — Raw Ingestion Spec Audit (Unblock Priority 1)

Read-only. No code changes. Output to `reports/K27_RAW_INGESTION_AUDIT.md`.

**Context:** The fantasy pipeline is under a data-first embargo. Before Claude writes Pydantic V2 validators, we need a clear audit of what the Yahoo and BallDontLie APIs actually return vs. what the codebase currently assumes they return.

**Research Task 1 — Yahoo API Response Audit:**

Read `backend/fantasy_baseball/yahoo_client_resilient.py` in full. For each method that makes a live API call (get_team_roster, get_free_agents, get_matchup_stats, get_league_settings, get_adp_and_injury_feed), document:
- What the code assumes the response shape is
- What fields it accesses (e.g., `player[1]['player_stats']`)
- Which fields are accessed without a null check
- Which stat IDs are hardcoded vs. derived from the response

**Research Task 2 — BallDontLie MLB Schema:**

Read `backend/services/balldontlie.py`. Document the current API call structure for MLB endpoints. If there are no MLB endpoints yet (only NBA/NCAAB), state that explicitly — it means the BDL MLB integration does not exist yet and must be built from scratch.

**Research Task 3 — Silent Failure Points:**

Read `backend/services/daily_ingestion.py` focusing on the `_fetch_fangraphs_ros`, `_poll_yahoo_adp_injury`, and `_update_ensemble_blend` functions. Identify every place where an API response field is accessed without type validation — these are silent failure points where bad data corrupts the DB silently.

**Report in `reports/K27_RAW_INGESTION_AUDIT.md`:**
- Per-method table: endpoint, fields accessed, null-safety (yes/no), type-checked (yes/no)
- List of the top-5 highest-risk silent failure points
- Whether BDL MLB endpoints exist in the codebase or must be built
- Recommended field list for Claude's Pydantic V2 models

---

## Session History (Recent)

### S13 — Frontend Cleanup (Apr 5)

Removed all remaining fantasy-specific frontend surface. Deleted `frontend/app/(dashboard)/settings/page.tsx` (432 lines — UserPreferences/waiver sliders, 100% fantasy). Removed Settings nav section from sidebar. Removed `UserPreferences`, `UserPreferencesResponse`, `DashboardPanel` TypeScript types and `getUserPreferences`/`updateUserPreferences` API methods. Bracket Simulator hidden via `SHOW_BRACKET = false` flag in `sidebar.tsx` (page on disk, not linked).

Running UI now: betting analytics only (Dashboard, Performance, CLV, Bet History, Calibration, Alerts, Today's Bets, Live Slate, Odds Monitor, Admin). No fantasy UI. No settings. No bracket.

Commit: `cc1e7ce`

### S12 — Structural Decoupling + Fantasy UI Removal (Apr 4)

Phases 1-5 of the fantasy/edge modular monorepo split complete. `main.py` unchanged — strangler-fig, production unaffected.

New files: `backend/db.py`, `backend/redis_client.py`, `backend/models_edge.py`, `backend/models_fantasy.py`, `backend/routers/{edge,fantasy,admin}.py`, `backend/schedulers/{edge,fantasy}_scheduler.py`, `backend/edge_app.py`, `backend/fantasy_app.py`.

Fantasy UI fully removed: `frontend/app/(dashboard)/fantasy/` (17 files), `frontend/components/shared/status-badge.tsx`, `frontend/lib/fantasy-stat-contract.json`, `frontend/tests/e2e/fantasy_baseball.spec.ts`, sidebar fantasy nav section, fantasy API methods, fantasy TypeScript types.

Test suite: 1256 passed, 4 pre-existing failures unchanged.

### S11 — Ensemble Hardening (Apr 3)

Per-row `db.begin_nested()` savepoints in `_update_ensemble_blend`. Fatal bare-except in `process_pending_jobs`.

### S9/S10 — Fantasy Hotfixes (Apr 3)

Cold-start fix, lineup apply hardening, direct waiver actuation, matchup scoring fix, weather fallback.

### S7/S8 — Projection Pipeline (Apr 1-2)

Jobs 100_012–100_015 built and wired. Now embargoed pending raw data validation.

---

## Architecture Reference

### Entry Points

| Entry Point | Command | Status |
|-------------|---------|--------|
| `backend/main.py` | `uvicorn backend.main:app` | LIVE in Railway (CBB production) |
| `backend/edge_app.py` | `uvicorn backend.edge_app:app` | Built — not yet deployed standalone |
| `backend/fantasy_app.py` | `uvicorn backend.fantasy_app:app` | Built — awaiting Phase 6-7 deploy sequence |

### Key File Map

| What | Where |
|------|-------|
| Fantasy routes | `backend/routers/fantasy.py` |
| Edge/betting routes | `backend/routers/edge.py` |
| Admin/health routes | `backend/routers/admin.py` |
| Shared DB engine factory | `backend/db.py` |
| Redis namespace helpers | `backend/redis_client.py` (use `edge_cache` / `fantasy_cache` only) |
| Decoupling spec (Phases 6-7) | `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md` |
| Kelly math (FROZEN) | `backend/core/kelly.py`, `backend/betting_model.py` |
| Yahoo OAuth client | `backend/fantasy_baseball/yahoo_client_resilient.py` |
| Ingestion scheduler | `backend/services/daily_ingestion.py` |
