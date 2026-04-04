# HANDOFF.md — Fantasy Baseball Platform Master Plan (In-Season 2026)

> **Date:** April 4, 2026 (updated Session S12) | **Author:** Claude Code (Master Architect)
> **Risk Level:** STABLE — fantasy pipeline live, CBB frozen pending Apr 7 unlock

---

## Platform State — April 4, 2026

| System | State | Notes |
|--------|-------|-------|
| Fantasy Baseball | **LIVE** | All 12 critical bugs closed. Projection pipeline active. |
| CBB Betting Model | **FROZEN** | EMAC-068 — Kelly math blocked until Apr 7 |
| OddsAPI Champion | **EXPIRING** | Cancel post-Apr 7; BDL GOAT MLB replaces it |
| BDL NCAAB | **DEAD** | Subscription cancelled — never call `/ncaab/v1/` |
| Fantasy/Edge decoupling | **PHASES 1-5 DONE** | New entry points built; Phase 6-7 pending Railway provisioning |
| CBB V9.2 recalibration | **UNLOCKS APR 7** | K-12 spec ready; implement within 48h of unlock |

---

## Session History (Recent)

### S12 — Structural Decoupling (Apr 4)

Phases 1-5 of the fantasy/edge modular monorepo split complete. `main.py` unchanged — strangler-fig, production unaffected.

**New files added:**

| File | Purpose |
|------|---------|
| `backend/db.py` | Engine factory, `NamespacedKey` — 4 tests |
| `backend/redis_client.py` | `NamespacedCache`, `edge_cache`/`fantasy_cache` — 5 tests |
| `backend/models_edge.py` | Re-export shim — betting model classes |
| `backend/models_fantasy.py` | Re-export shim — fantasy model classes |
| `backend/routers/edge.py` | 24 betting/analysis routes |
| `backend/routers/fantasy.py` | 35 fantasy routes |
| `backend/routers/admin.py` | 37 admin/health/root routes |
| `backend/schedulers/edge_scheduler.py` | Edge APScheduler (job fns bridged from main.py) |
| `backend/schedulers/fantasy_scheduler.py` | Fantasy APScheduler (job fns bridged from main.py) |
| `backend/edge_app.py` | Edge service entry point — 4 isolation tests |
| `backend/fantasy_app.py` | Fantasy service entry point — 4 isolation tests |

**Test suite:** 1256 passed, 4 pre-existing failures (unchanged).
**Route isolation verified:** `edge_app` returns 404 for `/api/fantasy/*`; `fantasy_app` returns 404 for `/api/predictions/*`.
**Spec:** `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md`

### S11 — Ensemble Hardening (Apr 3)

- A5: `_update_ensemble_blend` — per-row `db.begin_nested()` savepoints. Row failures skip without aborting batch.
- A6: `process_pending_jobs` — bare `except Exception` made fatal (prefixed `[unexpected:TypeName]`). No retry slot consumed for programming errors.

### S9/S10 — Fantasy Hotfixes (Apr 3)

- Cold-start fix: `fantasy_stat_contract.py` dual-path loader + backend-local JSON copy.
- Lineup apply: ET date enforcement, player key sanitization, OF enrichment.
- Waiver: `POST /api/fantasy/waiver/add` endpoint for direct Yahoo actuation.
- Matchup: active scoring category filter prevents OBP/K-BB column drift.
- Weather: free-tier OneCall path + `data/2.5/weather` fallback.

### S7/S8 — Projection Pipeline (Apr 1-2)

- Jobs 100_012 (FanGraphs RoS @ 3 AM ET), 100_013 (Yahoo ADP/injury every 4h), 100_014 (ensemble blend @ 5 AM ET), 100_015 (freshness gate every 1h) — all LIVE.
- `GET /api/fantasy/lineup/{lineup_date}` blocks with 503 when SLA violated (force_stale=true override available).

---

## What's Pending

### Apr 7 Unlocks (Claude)

| Item | File(s) | Notes |
|------|---------|-------|
| CBB V9.2 recalibration | `backend/betting_model.py`, `backend/services/analysis.py` | K-12 spec in `reports/`. Fix SNR/integrity scalar stacking. Effective divisor ~3.4x → ~1.0x target. |
| OddsAPI → BDL GOAT MLB | `backend/services/balldontlie.py`, `backend/services/odds.py`, `backend/services/daily_ingestion.py` | Subscribe to BDL GOAT. Add `/mlb/v1/` endpoints. Migrate `_poll_mlb_odds()` and `_fetch_mlb_odds()` off raw OddsAPI. |

### Architecture Phase 6-7 (Gemini → then Claude)

Gemini provisions the Railway services. Claude implements Plan B once `FANTASY_DATABASE_URL` is confirmed.

| Step | Owner | Action |
|------|-------|--------|
| Provision Fantasy Postgres | Gemini | Add new PostgreSQL service in Railway under the fantasy service |
| Set env vars | Gemini | `FANTASY_DATABASE_URL` on fantasy service; verify `DATABASE_URL` on edge service |
| Deploy fantasy standalone | Gemini | Override Start Command: `uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT` |
| Deploy edge standalone | Gemini | Override Start Command: `uvicorn backend.edge_app:app --host 0.0.0.0 --port $PORT` |
| Plan B implementation | Claude | Per `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md` Phase 6-7: `UserPreferences` migration, Redis wiring, `main.py` cleanup |

---

## Current Pipeline State

| Job | Lock | Cadence | Status |
|-----|------|---------|--------|
| `statcast` | 100_002 | Daily 2 AM ET | LIVE |
| `rolling_z` | 100_003 | Daily 3 AM ET | LIVE |
| `waiver_scan` | 100_007 | Daily 6 AM ET | LIVE |
| `mlb_brief` | 100_008 | Daily 7 AM ET | LIVE |
| `valuation_cache` | 100_011 | On demand + scheduled | LIVE |
| `fangraphs_ros` | 100_012 | Daily 3 AM ET | LIVE |
| `yahoo_adp_injury` | 100_013 | Every 4h | LIVE |
| `ensemble_update` | 100_014 | Daily 5 AM ET | LIVE |
| `projection_freshness_check` | 100_015 | Every 1h | LIVE |
| `mlb_odds` | 100_001 | Every 30 min | DIRTY — migrate Apr 7 |

**Next available lock ID:** 100_016

---

## Data Freshness SLAs

| Data Type | Max Age | Status |
|-----------|---------|--------|
| Rest-of-Season Projections | < 12h | LIVE (100_012 @ 3 AM ET) |
| Ensemble Blend | < 12h | LIVE (100_014 @ 5 AM ET) |
| Yahoo ADP / Injuries | < 4h | LIVE (100_013 every 4h) |
| Statcast Metrics | < 6h | LIVE (100_002 @ 2 AM ET) |
| Projection Freshness Gate | < 1h check | LIVE (100_015 every 1h) — 503 on SLA violation |
| Player Roster / Status | < 1h | LIVE via Yahoo client |

---

## Hard Stops

| Rule | Reason |
|------|--------|
| Do NOT modify Kelly math in `betting_model.py` | EMAC-068 — blocked until Apr 7 |
| Do NOT call BDL `/ncaab/v1/` endpoints | Subscription cancelled — will 401 |
| Do NOT add `THE_ODDS_API_KEY` dependencies | Phasing out post-Apr 7 |
| Do NOT touch `dashboard/` (Streamlit) | Retired — Next.js is canonical |
| Do NOT use `datetime.utcnow()` for game times | Use `datetime.now(ZoneInfo("America/New_York"))` |
| Do NOT write test files outside `tests/` | Architecture locked |
| Do NOT import `betting_model` from fantasy modules | GUARDIAN FREEZE / ADR-004 |
| Do NOT import `backend.models_edge` from `backend.models_fantasy` | Hard architectural boundary |
| Do NOT blend Statcast xwOBA/Barrel% with counting stats | Statcast = trend modifier only (K-19) |
| Do NOT weight pre-season CSVs against RoS projections | Pre-season CSV = fallback for missing players only |

---

## Advisory Lock Registry

| Lock | Job | Status |
|------|-----|--------|
| 100_001 | mlb_odds (OddsAPI) | DIRTY — migrate Apr 7 |
| 100_002 | statcast | LIVE |
| 100_003 | rolling_z | LIVE |
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
| 100_016 | *(next available)* | — |

---

## Delegation Matrix

| Area | Owner | Constraint |
|------|-------|------------|
| Backend Python — core logic, schema, algorithms | Claude | Read HANDOFF.md before every session |
| Railway DevOps — provisioning, deploy, smoke tests | Gemini | Zero `.py` changes |
| Frontend TypeScript | Gemini | `cd frontend && npx tsc --noEmit` after every change |
| Deep research, spec memos | Kimi | Output to `reports/K*.md` — no prod code |

---

## HANDOFF PROMPTS

### Claude Code — Apr 7 Unlock (CBB V9.2 + BDL Migration)

Execute on or immediately after April 7, 2026 in `C:\Users\sfgra\repos\Fixed\cbb-edge` on branch `stable/cbb-prod`.

**Read first:** `HANDOFF.md`, `CLAUDE.md`, `reports/K12_CBB_V9_2_RECALIBRATION_SPEC.md`

**Task A — CBB V9.2 Recalibration (EMAC-068 unlocks Apr 7):**

The CBB model has a calibration mismatch introduced in V9.1: the SNR kelly scalar (~0.70) and integrity scalar (~0.85) are stacked on top of half-Kelly (÷2.0), producing an effective divisor of ~3.4× instead of the calibrated ~1.0×. The 663-bet calibration dataset did not include these scalars, so `sd_mult=1.0` and `ha=2.419` are wrong for V9.1. The model requires ~6-8% raw edge to emit a 2.5% conservative edge — far too conservative.

Fix per the K-12 spec:
1. In `backend/betting_model.py`: recalibrate `sd_mult` and `ha` to account for the scalar stack.
2. In `backend/core/kelly.py`: verify the effective Kelly divisor after SNR + integrity scalars are applied.
3. In `backend/services/analysis.py`: bump `model_version` to `v9.2`.
4. Validate with `venv/Scripts/python -m pytest tests/test_betting_model.py tests/test_kelly.py -q --tb=short`.
5. Run `venv/Scripts/python -m py_compile backend/betting_model.py backend/core/kelly.py backend/services/analysis.py && echo OK`.
6. Commit: `git commit -m "feat: CBB V9.2 recalibration -- correct scalar stacking per EMAC-068 / K-12 spec"`

**Task B — OddsAPI → BDL GOAT MLB Migration:**

OddsAPI Champion subscription ends Apr 7. BDL GOAT MLB replaces it for all MLB odds work.

1. Subscribe to BDL GOAT plan at balldontlie.io and retrieve the new API key.
2. In `backend/services/balldontlie.py`: add `/mlb/v1/` endpoints (games, scores, odds). Keep `/nba/` endpoints intact. Never add `/ncaab/` (subscription cancelled).
3. In `backend/services/daily_ingestion.py`: migrate `_poll_mlb_odds()` (lock 100_001) off raw OddsAPI to the new BDL client.
4. In `backend/services/mlb_analysis.py` (GUARDIAN FREEZE still applies — do NOT import `betting_model`): update `_fetch_mlb_odds()` to use BDL client.
5. Update Railway env vars: add `BALLDONTLIE_API_KEY`, remove `THE_ODDS_API_KEY`.
6. Validate: `venv/Scripts/python -m pytest tests/test_mlb_analysis.py -q --tb=short`.
7. Commit: `git commit -m "feat: migrate mlb_odds from OddsAPI to BDL GOAT -- lock 100_001"`

Report: which lines changed, py_compile output for all modified files, test result summary.

---

### Gemini CLI — Railway Service Provisioning (Phase 6-7 Prep)

Execute in `C:\Users\sfgra\repos\Fixed\cbb-edge`. **Zero Python file changes.** All work is Railway CLI and env var configuration.

**Context:** The codebase now has two independent FastAPI entry points:
- `backend/edge_app.py` → edge betting/analysis service
- `backend/fantasy_app.py` → fantasy baseball service

Both are on branch `stable/cbb-prod`. `backend/main.py` still runs in production on the existing Railway service (unchanged — strangler-fig). The goal is to provision a second Railway service for fantasy and point it at the new entry point.

**Task 1 — Verify current Railway state:**
```bash
railway status
railway service list
railway variables --service <current-backend-service-name>
```
Report: current service names, which service serves the backend, current env vars (redact secrets).

**Task 2 — Provision Fantasy Postgres:**
In the Railway dashboard (or CLI), add a new PostgreSQL plugin/service under the fantasy service project. Name it `fantasy-postgres`. Note the `DATABASE_URL` it generates — this becomes `FANTASY_DATABASE_URL`.

**Task 3 — Create fantasy Railway service:**
In Railway, create a new service in the same project. Set:
- Root directory: same repo (`stable/cbb-prod` branch)
- Start Command: `uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT`
- Env vars:
  - `FANTASY_DATABASE_URL` = connection string from Task 2
  - `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_REFRESH_TOKEN` = copy from existing service
  - `OPENWEATHER_API_KEY` = copy from existing service
  - `ENABLE_INGESTION_ORCHESTRATOR` = `true`
  - `DEPLOYMENT_ROLE` = `fantasy-prod`
  - `ENABLE_MAIN_SCHEDULER` = `false`

**Task 4 — Smoke test fantasy service:**
```bash
curl -s https://<fantasy-service-url>/health | python -m json.tool
curl -s -o /dev/null -w "%{http_code}" https://<fantasy-service-url>/api/predictions/today
```
Expected: `/health` returns `{"status": ...}`. `/api/predictions/today` returns 404 (edge route not mounted).

**Task 5 — Report back to Claude:**
Provide the fantasy service URL, the `FANTASY_DATABASE_URL` connection string (redact password), and confirmation that Tasks 1-4 completed. Claude will then implement Plan B (database migration, `UserPreferences` copy, Redis wiring, `main.py` cleanup).

Do NOT run database migrations — that is Claude's job in Plan B. Do NOT modify any `.py` files.

---

### Kimi CLI — Pre-V9.2 Recalibration Analysis (Apr 7 Prep)

Read-only research task. Output to `reports/K27_V9_2_CALIBRATION_AUDIT.md`. No code changes.

**Context:** The CBB V9.1 model has a known calibration mismatch (EMAC-068). The effective Kelly divisor is ~3.4× due to stacking of the SNR scalar (~0.70) and integrity scalar (~0.85) on top of half-Kelly (÷2.0). The 663-bet calibration dataset did not include these scalars.

**Research questions:**

1. Read `backend/core/kelly.py`. Trace the Kelly fraction computation from raw edge to final unit size. Calculate what the effective divisor is when `snr_kelly_scalar=0.70`, `integrity_kelly_scalar=0.85`, and the base Kelly half-fraction is applied. Show the math.

2. Read `backend/betting_model.py`. Find `sd_mult` and `ha` (home advantage). What values are they currently set to? What would they need to be for the model to produce unit sizes consistent with the original calibration dataset (i.e., effective divisor ~1.0×)?

3. Read `reports/K12_CBB_V9_2_RECALIBRATION_SPEC.md`. Summarize the recommended parameter changes. Identify any gaps or risks in the K-12 spec that Claude should know before implementing.

4. Read `backend/services/analysis.py`. Identify which lines reference `snr_kelly_scalar` and `integrity_kelly_scalar`. Does the multiplication order match the K-12 spec's correction?

Report in `reports/K27_V9_2_CALIBRATION_AUDIT.md` with: the math trace, current parameter values, recommended values, K-12 spec gaps, and a confidence rating on whether the K-12 fix is sufficient.

---

## Architecture Reference

### Entry Points (after S12)

| Entry Point | Command | Purpose | Status |
|-------------|---------|---------|--------|
| `backend/main.py` | `uvicorn backend.main:app` | Current prod (Railway) | LIVE — strangler-fig |
| `backend/edge_app.py` | `uvicorn backend.edge_app:app` | Edge service (betting) | BUILT — not yet deployed standalone |
| `backend/fantasy_app.py` | `uvicorn backend.fantasy_app:app` | Fantasy service | BUILT — not yet deployed standalone |

### Key File Map

| What | Where |
|------|-------|
| Fantasy routes | `backend/routers/fantasy.py` |
| Edge/betting routes | `backend/routers/edge.py` |
| Admin/health routes | `backend/routers/admin.py` |
| Edge scheduler jobs | `backend/schedulers/edge_scheduler.py` |
| Fantasy scheduler jobs | `backend/schedulers/fantasy_scheduler.py` |
| Shared DB engine factory | `backend/db.py` |
| Redis namespace helpers | `backend/redis_client.py` |
| Decoupling spec (Phases 6-7) | `docs/superpowers/specs/2026-04-04-fantasy-edge-decoupling-design.md` |
| Kelly math (FROZEN) | `backend/core/kelly.py`, `backend/betting_model.py` |
| Yahoo OAuth client | `backend/fantasy_baseball/yahoo_client_resilient.py` |
| Ingestion scheduler | `backend/services/daily_ingestion.py` |
| CBB V9.2 spec | `reports/K12_CBB_V9_2_RECALIBRATION_SPEC.md` |
