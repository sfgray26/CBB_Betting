# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-04-23 14:30 UTC | **Architect:** Claude Code (Master Architect)
> **Status:** Phase 6 (Data Payload & API Root Cause) — cat_scores extraction completed, integration tests passing, ready for production backfill execution.

> **Previous Status:** ⚠️ **CRITICAL PRODUCTION REGRESSION** — Waiver endpoints regressed to 503 as of 21:09 UTC (2026-04-21). A deployment between 19:01 and 21:09 reintroduced the Yahoo API `out=ownership` bug (K-20). Waiver functionality **completely unavailable**.

---

## 1. Mission Accomplished — Latest Session (2026-04-23)

### Phase 6: Data Payload Recovery — Test Infrastructure Complete

**Objective:** Extract cat_scores backfill logic into testable module and validate with SQLite integration tests (avoiding ORM mocking anti-pattern).

**Work Completed:**

1. **Created `backend/services/cat_scores_builder.py`** (310 lines)
   - Extracted from `backend/routers/data_quality.py` endpoint (was ~200 lines of inline closures)
   - Pure functions: `classify_player()`, `compute_cat_scores()`, `_zscore()`
   - Full pipeline: `run_backfill(db)` — load → classify → score → team-lookup → write → verify
   - Dialect-aware SQL: PostgreSQL uses `DISTINCT ON`, `::text`, `::jsonb`; SQLite uses `GROUP BY`, native JSON

2. **Refactored `backend/routers/data_quality.py`**
   - `/backfill-cat-scores` endpoint reduced from ~200 lines to ~10 lines
   - Now delegates to `backend.services.cat_scores_builder.run_backfill(db)`
   - Maintains production behavior (pitcher defaults for ambiguous rows)

3. **Created `tests/test_cat_scores_backfill.py`** (307 lines)
   - 12 integration tests using SQLite in-memory database
   - Fixture patches all JSONB columns to JSON for SQLite compatibility
   - Covers: classify batter/pitcher/ambiguous, z-score computation, full pipeline, team lookup, edge cases
   - **All 12 tests passing**

4. **Test Suite Results**
   - New integration tests: **12/12 PASS**
   - Full suite: **2364 PASS / 8 FAIL / 3 SKIP**
   - 8 failures are pre-existing (test_openclaw_autonomous.py, test_openclaw_lite.py, test_nsb_pipeline.py)
   - **Zero regressions** from this change

**Files Modified:**
- `backend/services/cat_scores_builder.py` (new)
- `backend/routers/data_quality.py` (refactored endpoint)
- `tests/test_cat_scores_backfill.py` (new)

**Next Step (Production Execution):**
Execute `POST /api/admin/data-quality/backfill-cat-scores` against Railway production to populate 345 empty cat_scores rows. See Phase 6 plan for verification steps.

---

## 1. Mission Accomplished — Previous Session (2026-04-21)

### ⚠️ CRITICAL: Production Regression Detected (21:09 UTC)

**Fresh audit** (`reports/2026-04-21-production-data-quality-audit-v3.md`) reveals:
- **Waiver endpoints regressed from 200 → 503** between 19:01 and 21:09 UTC
- **Root cause:** Yahoo API `out=ownership` parameter error reintroduced (K-20 fix reverted)
- **Impact:** Waiver wire **completely unavailable** — no free agents, no recommendations, no ownership data
- **Positive changes:** BDL IDs improved (0/23 → 19/23), injury data improved (0/23 → 3/23)
- **Persistent issues:** Rolling windows 100% null, MLBAM IDs 100% null, universal-drop bug active

### Session Work (Pre-Regression)

Post-deploy UAT v5 (`tasks/uat_findings_post_deploy_v5.md`) revealed three live API failures. All root-caused with focused regression tests:

**Roster enrichment null (P0):** Canonical handler never called `get_players_stats_batch()`, so season_stats remained null for all 23 players. **Fixed:** Added batch hydration call in roster route.

**Waiver `matchup_opponent = "TBD"` (P0):** Inline scoreboard parser did 2-level descent; Yahoo nests `team_key` one level deeper. **Fixed:** Extracted recursive walker from matchup endpoint into shared helper.

**Waiver `category_deficits = []` (P0):** Cascaded from matchup issue  deficit block gated on `matchup_opponent != "TBD"`. **Fixed:** Shared helper now feeds both opponent and deficits in single call.

Also bundled: Apr 21 Postman P0/P1 fixes (MCMC negative-gain gate, numeric stat_id filter, briefing MONITOR routing, roster ImportError hoist). Targeted fantasy suite: **72 passed**. Full suite: **309 passed / 0 regressions**. All changes uncommitted and ready for Wave 2 deploy. See `memory/2026-04-21.md` for file-level detail.

---

## 2. Current State

### 2.1 Deploy State

| Slice | Status | Commit(s) | Production Impact |
|-------|--------|-----------|-------------------|
| Apr 20 UAT Remediation | **Committed** | `a2e2e56`, `791f6fa`, `3347937` | Live in prod |
| Apr 21 Lineup/Admin Repair | **Committed** | `2749276`, `9147f83`, `80889dc`, `8ca2ebe` | Live in prod |
| **Unknown Deploy (19:01-21:09 UTC)** | **⚠️ REVERTED K-20 FIX** | Unknown | **Waiver 200→503 regression** |
| Apr 21 Postman P0/P1 + UAT v5 Fixes | **Local/uncommitted** | — | Needs validation against regressed prod |

### 2.2 Phase Plan Progress

| Phase | Focus | Status |
|-------|-------|--------|
| 0-1 | Contracts + V1V2 migration | **COMPLETE** |
| 2 | 18-category rolling stats + ROW projector | **INCOMPLETE** (ROW pipeline missing, 9/18 categories) |
| 3 | Pure functions + H2H Monte Carlo | **COMPLETE** |
| 4 | P1 API endpoints | **ROUTES OK, DATA DEGRADED** |
| 4.5a | Quality remediation | **COMPLETE** (P4 Statcast deferred) |
| 4.5b | UAT | **HTTP 95/100 PASS, DATA FAILING** |
| 5 | Frontend | **BLOCKED** (B1: ROW pipeline, B2: rolling stats 9/18, B3: ROS projections) |

### 2.3 Open Defects (Prioritized)

| # | Severity | Defect | Evidence | Next Action |
|---|----------|--------|----------|-------------|
| **0** | **P0** | **⚠️ Waiver endpoints 503 — Yahoo API `out=ownership` regression** | v3 audit 21:09 UTC | **IMMEDIATE:** Identify unknown deploy between 19:01-21:09, verify K-20 fix presence, redeploy correct version |
| 1 | P1 | Roster rolling windows 100% null (7d/14d/15d/30d, ROS, ROW, game_context) | v3 audit: 0/23 populated | Wave 3: Diagnose rolling_stats ingestion + PlayerIDMapping |
| 2 | P1 | MLBAM IDs 100% null (blocks Statcast joins) | v3 audit: 0/23 populated | Wave 3: PlayerIDMapping ingestion diagnosis |
| 3 | P1 | Universal-drop bug — all 24 recs drop Seiya Suzuki | v3 audit: 24/24 same drop | Architect review + code fix |
| 4 | P1 | Lineup pitcher warning noise — 7 SP no start, 0 active slots | Legacy issue | Wave 3: Pitcher-start detection logic |
| 5 | P2 | Briefing uses legacy v1 category names (HR, SB, K, SV) | v3 audit: 11 categories | Wave 3: Migrate to v2 canonicals |
| 6 | P2 | Schema pollution — K_P mislabeled, batters have pitcher stats | Legacy issue | Wave 3: Stat schema cleanup |
| 7 | P3 | Impossible ROS projections (0.00 ERA, 0.00 WHIP) | v3 audit: 5 instances | See `tasks/architect_review.md` Decision #5 |
| 8 | P3 | Draft board age=0 for 92.5% (185/200 players) | v3 audit | Low priority |
| 9 | P3 | NSB composite test failure | Pre-existing | See `tasks/architect_review.md` Decision #1 |

**Positive changes (v3 audit):** BDL IDs 0/23 → 19/23, injury data 0/23 → 3/23.

**Previously resolved (now uncertain due to regression):** Roster season_stats null, waiver matchup="TBD", waiver category_deficits=[] — need revalidation against regressed prod state.

---

## 3. Delegation Bundles

### 3.1 For Gemini CLI — Emergency Waiver Recovery + Wave 2 Deploy

**PRIORITY 0 (BEFORE Wave 2):** Investigate and fix waiver 503 regression.

**Steps:**
1. Check Railway deployment history between 19:01-21:09 UTC (2026-04-21)
2. Identify which commit is currently deployed (`GET /admin/version`)
3. Verify presence/absence of K-20 fix (Yahoo client should NOT include `out=ownership` parameter)
4. If K-20 fix missing: redeploy commit `8ca2ebe` or later (contains K-20)
5. Validate: `GET /api/fantasy/waiver` should return 200, not 503
6. Report back: deployment timeline, current commit SHA, waiver endpoint status

**PRIORITY 1 (After waiver recovery):** Deploy Apr 21 uncommitted fixes to Railway. See `docs/plan/fantasy-recovery-2026-04/plan.yaml` task `wave2-deploy-fixes`.

**Uncommitted files:**
```
backend/routers/fantasy.py
backend/fantasy_baseball/daily_briefing.py
backend/fantasy_baseball/smart_lineup_selector.py
tests/test_waiver_recommendations_gates.py  (new)
tests/test_daily_briefing_no_game_contract.py  (new)
tests/test_roster_waiver_enrichment_contract.py  (new)
```

**Pre-deploy validation:**
```bash
# 1. Syntax check
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/daily_briefing.py
venv/Scripts/python -m py_compile backend/fantasy_baseball/smart_lineup_selector.py

# 2. Targeted regression sweep (expect 72 passed)
venv/Scripts/python -m pytest \
  tests/test_waiver_recommendations_gates.py \
  tests/test_daily_briefing_no_game_contract.py \
  tests/test_roster_waiver_enrichment_contract.py \
  tests/test_waiver_edge.py \
  tests/test_waiver_integration.py \
  tests/test_dashboard_service_waiver_targets.py \
  -q --tb=short
```

**Deploy:**
```bash
git add backend/routers/fantasy.py \
        backend/fantasy_baseball/daily_briefing.py \
        backend/fantasy_baseball/smart_lineup_selector.py \
        tests/test_waiver_recommendations_gates.py \
        tests/test_daily_briefing_no_game_contract.py \
        tests/test_roster_waiver_enrichment_contract.py

git commit -m "fix(fantasy): Wave 2  roster enrichment, waiver matchup/deficits, MCMC gate, briefing routing"

git push origin stable/cbb-prod
```

Wait for Railway auto-deploy. Confirm `/admin/version` reflects new commit SHA.

**Post-deploy validation:**

Capture responses to `postman_collections/responses/2026-04-22/`.

1. **CRITICAL:** `GET /api/fantasy/waiver` → expect **200**, not 503 (verify `out=ownership` regression fixed)
2. **CRITICAL:** `GET /api/fantasy/waiver/recommendations` → expect **200**, not 503
3. `GET /api/fantasy/roster` → expect 200, `players_with_stats > 0%`, BDL IDs >80% populated
4. `GET /api/fantasy/briefing/2026-04-22` → expect opponents != "TBD" where probable pitcher known
5. Waiver stats → expect NO "38" key in stats dict, `matchup_opponent` != "TBD", `category_deficits` populated
6. Waiver recs → expect no recommendation with `mcmc.win_prob_gain < 0` when `mcmc_enabled=true`

**Smoke tests (should not regress):**
- `GET /api/fantasy/lineup/2026-04-22`
- `GET /admin/audit-tables`
- `GET /admin/odds-monitor/status`
- `GET /admin/yahoo/test`

**Report back:** HTTP status + 1-line body summary for endpoints 1-4. Explicit yes/no on each expected behavior. Any surprise regressions in smoke tests. If any endpoint fails, provide exact status code and JSON body. **Do not patch code**  escalate to Claude Code.

---

## 4. References

**Operational Documents:**
- **Plan:** `docs/plan/fantasy-recovery-2026-04/plan.yaml` (10 waves, 60-90 days to UI enablement)
- **Executive Roadmap:** `docs/EXECUTIVE_ROADMAP.md` (honest assessment, realistic timeline)
- **Session Logs:** `memory/2026-04-20.md`, `memory/2026-04-21.md`
- **Architect Review:** `tasks/architect_review.md` (7 code decisions, 7 UI contract questions)

**Research & Audits:**
- **⚠️ Latest Production Audit (21:09 UTC):** `reports/2026-04-21-production-data-quality-audit-v3.md` (waiver 503 regression)
- **Production Data Audit (reviewed):** `reports/2026-04-21-production-data-quality-audit-reviewed.md`
- **UI Contract Audit:** `reports/2026-04-17-ui-specification-contract-audit.md` (110 fields: 17% READY, 25% PARTIAL, 58% MISSING)
- **Framework Audit:** `reports/2026-04-18-framework-audit.md` (Kimi analysis + Claude verdicts)

**Historical Context:**
- Pre-April 17 history: `HANDOFF_ARCHIVE.md`
- Git commit trail: `git log -p` on listed commit SHAs

---

*Last updated: 2026-04-21 21:30 UTC — **CRITICAL**: Waiver 503 regression detected (v3 audit). Emergency recovery required before Wave 2 deploy. Session logs in `memory/`, architectural decisions in `tasks/architect_review.md`.*

---

## 16.4 DEVOPS OPERATIONS LOG (Apr 23, 2026)

| Date | Operation | Status | Notes |
|------|-----------|--------|-------|
| 2026-04-23 | Disable Integrity Sweep | **COMPLETE** | INTEGRITY_SWEEP_ENABLED=false |
| 2026-04-23 | Enable MLB Analysis | **COMPLETE** | ENABLE_MLB_ANALYSIS=true |
| 2026-04-23 | Enable Ingestion Orchestrator | **COMPLETE** | ENABLE_INGESTION_ORCHESTRATOR=true |
| 2026-04-23 | Production Deployment | **COMPLETE** | `railway up` pushed latest changes (including SQL cast fix) |
| 2026-04-23 | Cat Scores Backfill | **COMPLETE** | 350 rows populated. Target met (0 remaining). |
| 2026-04-23 | CSV Projection Ingest | **COMPLETE** | 25 player projections backfilled from CSV. |
| 2026-04-23 | Yahoo Error Diagnosis | **COMPLETE** | Waiver endpoint 200 OK; `pybaseball` 403 (FanGraphs) detected in logs. |

| 2026-04-22 | MLBAM ID Backfill | **COMPLETE** | 6,567/10,000 players populated with MLBAM IDs. |
| 2026-04-22 | Cat Scores Backfill | **COMPLETE** | 344/345 rows populated with z-scores. 0 rows remain empty. |

