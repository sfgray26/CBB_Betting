## SESSION F — Bug Fixes (2026-04-28, archived)

> All DevOps items from this session are complete. Constraint dropped, names backfilled, FANTASY_LEAGUES set. See Section 1 for status.

**Test suite:** 2433 pass / 3 skip / 7 xfail / 0 fail (up from 2364 pass / 8 fail)

#### Fixes Applied

1. **data_quality.py — timezone mismatch (P2)**
   - `PlayerProjection.updated_at` and `DataIngestionLog.started_at` are naive `DateTime` columns.
   - Was comparing against tz-aware `now - timedelta(days=7)` → PostgreSQL raises `can't subtract offset-naive and offset-aware datetimes`
   - Fix: `cutoff_naive = (now - timedelta(days=7)).replace(tzinfo=None)` used for all 3 filter expressions
   - File: `backend/routers/data_quality.py`

2. **Seiya Suzuki universal-drop bug (P1) — two-part fix**
   - Root cause A: `drop_candidate_value` used `hash(name)` as tiebreaker — consistent within a Python process, always picked the same player (Seiya Suzuki) when all other fields were equal
   - Root cause B: `_weakest_safe_to_drop` had no guard against "all candidates have empty scoring data" state
   - Fix A (`waiver_edge_detector.py`): replaced `hash(name)` → `player_id` (Yahoo key, unique per player)
   - Fix B (`fantasy.py`): added `all_data_missing` guard in `_weakest_safe_to_drop` — returns None when all candidates have `cat_scores={}`, `z_score=0.0`, `adp≥9000`, `tier≥999`

3. **scoring_engine.py — weighted SUM vs MEAN regression (P1)**
   - Commit `6b54c3f` changed weighted SUM (P1-4/P1-5 design) to weighted MEAN, breaking `test_nsb_pipeline.py`
   - Fix: reverted to weighted SUM per documented P1-4/P1-5 intent (two-way players valued higher for contributing to more categories)
   - File: `backend/services/scoring_engine.py`
   - Updated `test_scoring_engine.py::test_composite_z_is_weighted_mean` → `test_composite_z_is_weighted_sum` to match

4. **OpenClaw pre-existing failures (7 tests)**
   - OpenClaw deliberately paused 2026-04-21 (stubs returning immediately)
   - Tests were testing active behavior that no longer exists
   - Fix: marked 7 tests as `xfail(strict=True)` with reason documenting the pause
   - Files: `tests/test_openclaw_autonomous.py`, `tests/test_openclaw_lite.py`

5. **Box stats pipeline diagnostic endpoint (NEW)**
   - Added `GET /admin/pipeline/box-stats-health` to diagnose rolling windows 100% null
   - Reports: mlb_game_log row count/dates, mlb_player_stats null rates (ab, ip, both), player_rolling_stats counts by window + w_ab/w_ip null rates, player_scores latest date
   - Verdict field auto-detects: empty table vs 100% null mapping vs healthy
   - File: `backend/routers/admin.py`

#### Pending Investigation (needs Gemini to hit prod endpoint)
- **Rolling windows 100% null**: root cause still unknown. Run `GET /admin/pipeline/box-stats-health` to determine if mlb_player_stats is empty (no box stats ever ingested) vs null field mapping issue
- Most likely: mlb_game_log empty → game_ids=[] → box stats job skips → rolling windows finds 0 rows

#### DevOps Status (all complete as of Session G)
- ✅ Duplicate constraint dropped by Gemini
- ✅ Numeric player names backfilled (6 players — see Section 1)
- ✅ `FANTASY_LEAGUES` env var set in Railway
- ✅ Box stats health confirmed HEALTHY (12,297 rows, 3.5% both-null)

---

## SESSION F PART 2 — MCP & Skills Infrastructure (2026-04-28, archived)

> **Owner:** Kimi CLI (Deep Intelligence Unit)  
> **Scope:** Agent tooling improvements for Gemini CLI (DevOps) and Kimi CLI (Research)

### 2.1 Gemini CLI Improvements

**Updated `.gemini/settings.json`** with MCP servers:
- `@railway` — Deployment, logs, service management (trust=false)
- `@postgres-readonly` — Read-only database queries via `crystaldba/postgres-mcp` (trust=false, `--access-mode=restricted`)

**Created `GEMINI.md`** — Project context file auto-loaded by Gemini CLI on startup. Defines role, permitted ops, escalation path, and MCP tool reference.

**Created 2 new skills:**
| Skill | File | Purpose |
|-------|------|---------|
| `pre-deploy` | `.gemini/skills/pre-deploy/SKILL.md` | py_compile → tests → env-check → health-check → `railway up` (gated) |
| `post-deploy` | `.gemini/skills/post-deploy/SKILL.md` | Verify Railway status, logs, health endpoints after deploy |

**Gemini skills inventory (now 6):** `db-migrate`, `env-check`, `health-check`, `railway-logs`, `pre-deploy`, `post-deploy`

### 2.2 Kimi CLI Improvements

**Registered 5 MCP servers** in `~/.kimi/mcp.json`:
| Server | Transport | Status |
|--------|-----------|--------|
| `railway` | stdio | ✅ Connected |
| `postgres` | stdio (npx, no Docker) | ✅ Connected |
| `sequential-thinking` | stdio | ✅ Connected |
| `context7` | HTTP | ✅ Connected (free tier) |

> Note: `balldontlie` MCP removed — package does not exist on npm. BDL data is already in Railway DB; query via `@postgres` instead.

**Created `.kimi/skills/` directory with 3 research skills:**
| Skill | File | Purpose |
|-------|------|---------|
| `deep-db-audit` | `.kimi/skills/deep-db-audit/SKILL.md` | Comprehensive DB integrity audit using `@postgres-audit` MCP |
| `codebase-analysis` | `.kimi/skills/codebase-analysis/SKILL.md` | Multi-file architecture mapping and anti-pattern detection |
| `research-memo` | `.kimi/skills/research-memo/SKILL.md` | Enforces structured `reports/YYYY-MM-DD-<topic>.md` format |

**Created `.kimi/project_context.md`** — Kimi CLI project context with AGENTS.md constraints, session startup rules, MCP tool reference, and output rules.

### 2.3 Shared Infrastructure

**Created `mcp-shared-config.json`** — Reference MCP config template for all agents (gitignored).
**Created `scripts/setup_mcp_agents.ps1`** — One-command setup script for recreating MCP + skills config on new machines.
**Updated `.gitignore`** — Added `.kimi/`, `kimi-mcp-config.json`, `mcp-shared-config.json` to prevent secret leakage.

### 2.4 Remaining Manual Steps

| Step | Owner | Blocker |
|------|-------|---------|
| Set `RAILWAY_API_TOKEN` in `.env` | User | Need token from Railway dashboard → Account Settings → Tokens |
| Set `GITHUB_PERSONAL_ACCESS_TOKEN` in `.env` (optional) | User | Create at github.com/settings/tokens (scopes: repo, read:org) |
| Set `CONTEXT7_API_KEY` in `.env` (optional) | User | For higher rate limits at context7.com/dashboard |
| Register GitHub MCP in Kimi | User | After token obtained: `kimi mcp add --transport stdio github -- npx -y @modelcontextprotocol/server-github` |
| Register GitHub MCP in Claude Code | Claude | After token obtained |

**Auto-load `.env` setup:**
- PowerShell profile updated at `$PROFILE`
- `.env` loads automatically when PowerShell starts in `cbb-edge/`
- Manual load: `. .\scripts\load-env.ps1`

**Verification:**
```powershell
.\scripts\test_mcp.ps1
```

**Interactive test commands:**
```bash
# Gemini
gemini
@railway List services
@postgres-readonly List tables

# Kimi
kimi
@postgres-audit List tables
@balldontlie Get today's MLB games
@sequential-thinking Think through a complex problem
```

---

## 1. Mission Accomplished — Previous Session (2026-04-27)

### K-32 P0 Remediation — Clarified & Corrected

**Key Finding:** K-32 audit contained incorrect assumptions about BDL `/lineups` endpoint and fantasy_lineups INSERT logic.

**Actually Completed:**
1. ✅ Admin endpoint verified — `data_quality.py` syntax OK, uses correct `DataIngestionLog.started_at`
2. ✅ Migration verified — `create_ingested_injuries.sql` exists and ran successfully (170 rows in prod)
3. ✅ BDL injuries wired — Job scheduled at lock 100_033, actively ingesting
4. ✅ Projection freshness fixed — `datetime.date` → `datetime.datetime` conversion working (71% success)
5. ✅ Fantasy lineups INSERT — Logic EXISTS (lines 1378-1397), fixed banned `datetime.utcnow()` call
6. ✅ Probable pitchers — Uses MLB Stats API (lock 100_028), NOT BDL. K-37 confirmed BDL lacks this data.

**Audit Corrections:**
- **Issue #7 "fantasy_lineups empty"**: INSERT logic EXISTS at fantasy.py:1378-1397. Table empty because endpoint hasn't been called in production (or silently failing). Fixed `datetime.utcnow()` violation.
- **Issue #8 "BDL lineups endpoint"**: INCORRECT. System uses MLB Stats API for probable pitchers (`_sync_probable_pitchers`, lock 100_028). BDL does NOT expose probable pitcher data (K-37 confirmed).

**Ready for Execution (DevOps):**
7. 🔄 Numeric name backfill — Script ready, needs execution: `railway run python scripts/backfill_numeric_player_names.py`
8. 🔄 Valuation cache — Worker exists, needs `FANTASY_LEAGUES` env var set in Railway

**Files Modified:**
- `backend/routers/fantasy.py:1385` — Fixed `datetime.utcnow()` → `datetime.now(ZoneInfo("America/New_York"))`

**Next Steps (Gemini):**
- Run backfill script: `railway run python scripts/backfill_numeric_player_names.py`
- Verify/set `FANTASY_LEAGUES` env var in Railway dashboard
- Call lineup endpoint to verify persistence: `GET /api/fantasy/lineup/2026-04-27`

---

## 1. Mission Accomplished — Previous Session (2026-04-23)

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
| 1 | P1 | Roster rolling windows 100% null | v3 audit + diagnostic endpoint added | **Gemini: `GET /admin/pipeline/box-stats-health` → report verdict** |
| 2 | P1 | MLBAM IDs 100% null (blocks Statcast joins) | v3 audit: 0/23 populated | Wave 3: PlayerIDMapping ingestion diagnosis |
| ~~3~~ | ~~P1~~ | ~~Universal-drop bug — all 24 recs drop Seiya Suzuki~~ | **FIXED Session F** | Two-part fix: `hash(name)` → `player_id`, data-guard in `_weakest_safe_to_drop` |
| 4 | P1 | Lineup pitcher warning noise — 7 SP no start, 0 active slots | Legacy issue | Wave 3: Pitcher-start detection logic |
| 5 | P2 | Briefing uses legacy v1 category names (HR, SB, K, SV) | v3 audit: 11 categories | Wave 3: Migrate to v2 canonicals |
| 6 | P2 | Schema pollution — K_P mislabeled, batters have pitcher stats | Legacy issue | Wave 3: Stat schema cleanup |
| 7 | P3 | Impossible ROS projections (0.00 ERA, 0.00 WHIP) | v3 audit: 5 instances | See `tasks/architect_review.md` Decision #5 |
| 8 | P3 | Draft board age=0 for 92.5% (185/200 players) | v3 audit | Low priority |
| ~~9~~ | ~~P3~~ | ~~NSB composite test failure~~ | **FIXED Session F** | scoring_engine.py reverted to weighted SUM, test updated |

**Positive changes (v3 audit):** BDL IDs 0/23 → 19/23, injury data 0/23 → 3/23.

**Previously resolved (now uncertain due to regression):** Roster season_stats null, waiver matchup="TBD", waiver category_deficits=[] — need revalidation against regressed prod state.

---

## 3. Delegation Bundles
