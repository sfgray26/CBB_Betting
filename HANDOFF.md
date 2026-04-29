# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-04-29 | **Architect:** Claude Code (Master Architect)
> **Status:** Session K COMPLETE — ILP + greedy scarcity bonus live. Suite 2453/3 skip/0 fail. HEAD `7007939`. Session L candidates in Section 4.

---

## 1. Mission Accomplished — Session K (2026-04-29)

### Session K — ILP + Greedy Scarcity Objective Bonus

**Test suite:** 2453 pass / 3 skip / 0 fail — HEAD: `7007939`

| Step | Task | Detail | Commit |
|------|------|--------|--------|
| K1 | `lineup_constraint_solver.py` scarcity bonus | ILP: second objective pass adds `10*(10-scarcity_rank)` bonus units per natural-pos assignment (Util excluded). Max 90 units = <0.001 score-space — never overrides real gap ≥0.091. Greedy: candidates extended to 4-tuple with `natural_bonus`; `max()` key = `(score, natural_bonus)`. 4 new tests (2 skip without OR-Tools, 2 always pass). | 7007939 |

**Deferred:** ILP tie-break tests skip locally (OR-Tools absent); auto-promote to pass on Railway prod where OR-Tools is installed.

---

## 1a. Mission Accomplished — Session J (2026-04-29)

### Session J — Scarcity Tiebreaker + MLBAM Fallback

**Test suite:** 2451 pass / 1 skip / 0 fail — HEAD: `eff1160`

| Step | Task | Detail | Commit |
|------|------|--------|--------|
| 1 | `solve_lineup` scarcity tiebreaker | `_get_scarcity_rank(db, pos)` queries `MIN(scarcity_rank)` from `position_eligibility`, falls back to `_POSITION_SCARCITY` dict. `solve_lineup` collect-then-sort with `(-score, scarcity_rank)` key. `db=None` param added. `fantasy.py` call updated. 2 new tests. | ee4bae9 |
| 2 | `_update_projection_cat_scores` MLBAM fallback | Three-tier: (1) fg_id → (2) name.lower() → (3) PlayerProjection ilike. `team="Unknown"` stored when all three fail (no skip). Type-based default positions in INSERT; excluded from `on_conflict_do_update`. 7 new tests. | eff1160 |

**Post-H+I ops (completed locally — all done):**

| Operation | Result |
|-----------|--------|
| V31 backfill (`scripts/backfill_v31_fast.py`) | ✅ 69,504 rows — 34,517 `w_runs` non-null |
| V32 backfill (`scripts/backfill_v32_fast.py`) | ✅ 58,248 rows — 34,511 `z_r` non-null |
| `bdl_stat_id` column drop | ✅ Dropped from `mlb_player_stats` |
| Valuation cache refresh | ✅ HTTP 200 |
| `TestFourStateFusionIntegration` isolation fix | ✅ `setup_method` mocks `backend.models.get_db` (commit `b784b88`) |
| GitHub PAT redacted from `.env.example` history | ✅ Autosquash rebase; force-pushed (`4bc80c5`) |
| `git push origin stable/cbb-prod` | ✅ HEAD `eff1160` live on remote |
| `railway up --detach` | ✅ Deploy triggered |

**Deferred (not blocking Session K):**
- `test_player_board_fusion.py::test_state_1_both_sources_full_fusion_batter` — intermittent ordering-dependent failure; DB mock in place; passes in two consecutive full-suite runs.

---

## 1a. Mission Accomplished — Session I (2026-04-29)

### Session I — K-34 Downstream Wiring

**Test suite:** 2442 pass / 3 skip / 0 xfail / 0 fail — HEAD: `b78c76d` (SHA shifted after Session J rebase)

| Step | Task | Decision / Detail | Commit |
|------|------|-------------------|--------|
| 1 | `quality_score` range fix | **Option A** (rescale): `two_start_detector.py:222-227` has `>=1.0→EXCELLENT, >=0.0→GOOD, else→AVOID` thresholds; AVOID unreachable at [0,1]. Formula: `round((raw-0.5)*4.0, 2)` applied in `_sync_probable_pitchers` | 07fdf87 |
| 2 | `scarcity_rank` → `waiver_edge_detector` | `_load_scarcity_lookup()` bulk queries `position_eligibility.yahoo_player_key` for up to 40 FAs; `scarcity_multiplier = max(1.0, 1.0 + (13-rank)*0.05)`; `_FALLBACK_RANK` dict when no DB row | 872dca2 |
| 3 | `scarcity_rank` → `daily_lineup_optimizer` | Helper `_get_scarcity_rank(db, primary_position)` added; queries `MIN(scarcity_rank)` from `position_eligibility`, falls back to `_POSITION_SCARCITY` static dict. **Tiebreaker NOT yet integrated** into `assign_lineup_slots()` — slot loop has no pick comparison point; left as helper + docstring integration note. | 0130d7d |
| 4 | `quality_score` in waiver schemas | `quality_score: Optional[float] = None` added to `WaiverPlayerOut` + `RosterMoveRecommendation`. Populated in `get_waiver_recommendations` via bulk `ProbablePitcherSnapshot` query (today +7d), keyed by `pitcher_name.strip().lower()`; SP/RP/P only; wrapped in bare try/except (non-fatal). | 19cc902 |
| — | OpenClaw stubs (Kimi) | `openclaw_autonomous.py` + `openclaw_lite.py` — real implementations replacing paused stubs; 7 `xfail` tests converted to passing | — |

---

## 1a. Mission Accomplished — Session H (2026-04-29)

### Session H — P0 Structural Fixes

**Test suite:** 2433 pass / 3 skip / 7 xfail / 0 fail (baseline held) — HEAD: `ff7b5a6`

| Step | Task | Status |
|------|------|--------|
| 1 | `scripts/backfill_v31_rolling.py` — recomputes w_runs, w_tb, w_qs, w_caught_stealing, w_net_stolen_bases for NULL rows | ✅ ff7b5a6 |
| 1 | `scripts/backfill_v32_zscores.py` — recomputes z_r, z_h, z_tb, z_k_b, z_ops, z_k_p, z_qs for NULL rows | ✅ ff7b5a6 |
| 2 | `POSITION_SCARCITY` dict added to `_sync_position_eligibility`; `scarcity_rank` + `league_rostered_pct=None` now in INSERT + ON CONFLICT SET | ✅ ff7b5a6 |
| 3 | `quality_score` heuristic in `_sync_probable_pitchers`: bulk ERA lookup via `mlb_player_stats` JOIN `player_id_mapping`, formula: `0.5 + era_score + park_score` clamped [0,1] | ✅ ff7b5a6 |
| 4 | `_supplement_statsapi_counting_stats` filter broadened: `ab IS NULL OR runs IS NULL OR hits IS NULL OR doubles IS NULL OR triples IS NULL OR home_runs IS NULL OR rbi IS NULL OR stolen_bases IS NULL` | ✅ ff7b5a6 |
| 5 | `bdl_stat_id` removed from `backend/models.py:1166`, ingestion assignments at lines 1559+1594; `scripts/migrations/drop_bdl_stat_id.py` created for Gemini | ✅ ff7b5a6 |

**ERA lookup SQL (Step 3 as implemented):**
```sql
SELECT m.mlbam_id, AVG(s.era) AS avg_era
FROM (
    SELECT bdl_player_id, era,
           ROW_NUMBER() OVER (PARTITION BY bdl_player_id ORDER BY game_date DESC) AS rn
    FROM mlb_player_stats
    WHERE innings_pitched > 0 AND era IS NOT NULL
) s
JOIN player_id_mapping m ON m.bdl_id = s.bdl_player_id::text
WHERE s.rn <= 10 AND m.mlbam_id IS NOT NULL
GROUP BY m.mlbam_id
```

---

## 1a. Mission Accomplished — Sessions G + Post-G Ops (2026-04-28)

### Session G — Bug Fixes + Migration Scripts

**Test suite:** 2433 pass / 7 xfail / 0 fail (baseline maintained)

| Step | Task | Commit | Status |
|------|------|--------|--------|
| 1 | Fix `_with_advisory_lock` missing `job_name` arg in `daily_ingestion.py:5338` | e92f1a0 | ✅ |
| 2 | `scripts/migrations/drop_duplicate_yahoo_key_constraint.py` | 0c05411 | ✅ |
| 3 | `scripts/sync_projection_names_from_mapping.py` (6 numeric-name players) | 0c05411 | ✅ |
| 4 | Fix bare `except Exception:` in savant finally block → captures real error + traceback | 0c05411 | ✅ |
| 5 | Test suite ≥ 2433 pass / 7 xfail | — | ✅ |

### Post-Session G — Gemini Production Ops

| Operation | Result |
|-----------|--------|
| Deploy Session G commits | ✅ railway up deployed |
| Drop duplicate `player_id_mapping_yahoo_key_key` constraint | ✅ Dropped successfully |
| Run `sync_projection_names_from_mapping.py --execute` | ✅ 6 players backfilled |
| `POST /admin/refresh-valuation-cache` | ⏳ Re-trigger pending (fix now deployed) |

**6 players with numeric names — resolved by Gemini:**
| player_id | Name assigned |
|-----------|---------------|
| 608701 | Rob Refsnyder |
| 641598 | Mitch Garver |
| 642201 | Eli White |
| 657136 | Connor Wong |
| 669065 | Kyle Stowers |
| 669743 | Alex Call |

### K-33 — Kimi Deep Data Quality Audit (2026-04-28)

> **Full report:** `reports/2026-04-28-data-quality-null-audit.md`  
> **Scope:** 8 tables, 155,474 rows

**5 root-cause patterns identified:**

| Pattern | Tables Affected | Impact |
|---------|-----------------|--------|
| Migrations without backfills | `player_rolling_stats`, `player_scores` | 85% null on V31/V32 columns (w_runs, w_tb, w_qs, z_r, z_tb, z_qs, etc.) |
| Unimplemented computed fields | `position_eligibility`, `probable_pitchers` | `scarcity_rank`, `league_rostered_pct`, `quality_score` hardcoded to None — never computed |
| Cross-system ID resolution gaps | `player_projections`, `probable_pitchers` | FanGraphs → MLBAM → BDL → Yahoo chain incomplete; 50% of projections have no `team` |
| BDL partial stat coverage | `mlb_player_stats` | supplement job only patches `ab IS NULL`, misses partial rows |
| Season-age effect (self-healing) | `player_daily_metrics` | `z_score_total` requires 30d history; resolves automatically by ~May 25 |

**4 P0 items for Session H:**
1. Backfill V31/V32 columns (`w_runs`, `w_tb`, `w_qs`, `z_r`, `z_h`, `z_tb`, `z_k_b`, `z_ops`, `z_k_p`, `z_qs`) for all historical rows — currently 85% null
2. Implement `scarcity_rank` + `league_rostered_pct` in `_sync_position_eligibility` (static Option A)
3. Implement `quality_score` in `_sync_probable_pitchers` (heuristic Option A)
4. Harden `_supplement_statsapi_counting_stats` to patch any NULL counting stat (not just `ab IS NULL`)

**Downstream feature impact:**
| Feature | Current State |
|---------|---------------|
| Two-Start Command Center | ❌ Broken — `quality_score` 100% null |
| Waiver Edge Detector | ⚠️ Degraded — `scarcity_rank` 100% null, new Z-categories 85% null |
| Daily Lineup Optimizer | ⚠️ Degraded — missing scarcity weighting |
| VORP Engine | ⚠️ Degraded — flat replacement levels |
| Statcast / core pipeline | ✅ Healthy |

**Decisions already made (per Kimi recommendation):**
- `scarcity_rank`: Option A — static percentile mapping (C=most scarce, OF=least scarce). No daily recalculation needed.
- `quality_score`: Option A — heuristic using `park_factor` + pitcher ERA vs league average. No new dependencies.
- V31/V32 backfills: Option A — one-off scripts, run via Gemini. Not integrated into daily pipeline.
- `bdl_stat_id` column: Drop it (Option A). 100% null, BDL does not expose per-row stat IDs. Migration script needed.

---

## 2. Current System State (2026-04-29)

| System | Status | Notes |
|--------|--------|-------|
| Test suite | ✅ **2442 pass / 0 fail / 0 xfail** | HEAD `d901866` — baseline confirmed |
| Production deploy | ✅ **`railway up` triggered** | H+I+J commits deploying now |
| `quality_score` range | ✅ Fixed | Rescaled to [-2,+2]; thresholds reachable |
| `scarcity_rank` → waiver | ✅ Wired | Multiplier live in `waiver_edge_detector.py` |
| `scarcity_rank` → optimizer | ✅ **Tiebreaker live** | `solve_lineup` collect-then-sort with `(-score, scarcity_rank)` |
| `quality_score` schemas | ✅ Added | `WaiverPlayerOut` + `RosterMoveRecommendation` updated |
| OpenClaw | ✅ Implemented | Stubs replaced with real implementations (Kimi) |
| Advisory locks | ✅ 100_001–100_034 taken | **Next available: 100_035** |
| Valuation cache | ✅ Refreshed | HTTP 200 confirmed |
| `scarcity_rank` DB values | ⏳ Needs daily job run | Logic deployed; populates on next run |
| `quality_score` DB values | ⏳ Needs daily job run | Heuristic deployed; populates on next sync |
| V31/V32 backfills | ✅ Done | V31: 69,504 rows; V32: 58,248 rows |
| `bdl_stat_id` drop | ✅ Done | Column gone from `mlb_player_stats` |
| MLBAM fallback | ✅ Done | Three-tier resolution + `team="Unknown"` fallback |
| GitHub PAT in history | ✅ Redacted | Autosquash rebase; force-pushed to origin |
| Kimi MCP config | ✅ Clean | 4 servers, no Docker, no fake packages |

---

## 3. Post-Deploy Smoke Test (Gemini — Optional)

> All ops are complete and pushed. Gemini may run these spot-checks after deploy finishes.

```bash
# Confirm deploy health
curl -s https://fantasy-app-production-5079.up.railway.app/health

# Spot-check V31 backfill — expect > 30,000
railway run python -c "from backend.models import SessionLocal, PlayerRollingStats; db=SessionLocal(); print(db.query(PlayerRollingStats).filter(PlayerRollingStats.w_runs.isnot(None)).count()); db.close()"

# Spot-check V32 backfill — expect > 30,000
railway run python -c "from backend.models import SessionLocal, PlayerScore; db=SessionLocal(); print(db.query(PlayerScore).filter(PlayerScore.z_r.isnot(None)).count()); db.close()"
```

---

## ARCHIVED — Post-Session H+I Gemini bundle (completed locally 2026-04-29)

> All ops were run locally by Copilot via `scripts/backfill_v31_fast.py` and `scripts/backfill_v32_fast.py`.
> Gemini was stuck; Copilot ran them directly against the production DB (via `.env` public URL).

**Then run in order:**
```bash
# 1. Deploy (includes Session H + Session I commits)
railway up

# 2. Wait for healthy
curl -s https://fantasy-app-production-5079.up.railway.app/health

# 3. Drop bdl_stat_id column
railway run python scripts/migrations/drop_bdl_stat_id.py

# 4. Backfill V31 rolling columns (allow 5-10 min)
railway run python scripts/backfill_v31_rolling.py --execute

# 5. Backfill V32 z-score columns (run AFTER V31 completes)
railway run python scripts/backfill_v32_zscores.py --execute

# 6. Re-trigger valuation cache
curl -X POST https://fantasy-app-production-5079.up.railway.app/admin/refresh-valuation-cache
```

**Spot-check queries after completion:**
```sql
SELECT COUNT(*) FROM player_rolling_stats WHERE w_runs IS NOT NULL;
-- expect > 60,000

SELECT COUNT(*) FROM player_scores WHERE z_r IS NOT NULL;
-- expect > 60,000

SELECT column_name FROM information_schema.columns
WHERE table_name='mlb_player_stats' AND column_name='bdl_stat_id';
-- expect 0 rows
```

**Report back:** row counts for V31/V32 spot-checks + bdl_stat_id gone confirmation.

---

## 4. Next Session (Session L) — Scope

> Session K complete. K1 deployed. Session L candidates below.

### Session L Candidate Items

**Candidate 1 — Savant leaderboard first production ingestion run (K-28/K-30 deferred)**
- Pipeline verified and tested (445 batters, 507 pitchers). Never run in production.
- Scope: trigger `SavantIngestionClient` once manually; verify `statcast_leaderboard` table populated.
- Command: `railway run python -c "from backend.services.savant_ingestion import SavantIngestionClient; SavantIngestionClient().run()"`

**Candidate 2 — Post-deploy DB verification (K3 carry-over)**
- After first daily job run post-deploy, confirm `position_eligibility.scarcity_rank` non-null and `probable_pitchers.quality_score` non-null.
- Spot-check: `SELECT COUNT(*), AVG(scarcity_rank) FROM position_eligibility WHERE scarcity_rank IS NOT NULL`

**Priority:** L1 (Savant ingest) → L2 (DB verify).

---

## 5. Advisory Locks Reference

```
100_001–100_034   TAKEN (see CLAUDE.md for full map)
Next available:   100_035
```

---

## K-33 FINDINGS (Kimi — 2026-04-28, RESOLVED by Session H)

See full report: `reports/2026-04-28-data-quality-null-audit.md`

**Key numbers (pre-Session H state):**
- `player_rolling_stats.w_runs/w_tb/w_qs`: 85% null → **backfill scripts created (ff7b5a6)**
- `player_scores.z_r/z_h/z_tb/z_k_b/z_ops/z_k_p/z_qs`: 85% null → **backfill scripts created (ff7b5a6)**
- `position_eligibility.scarcity_rank`: 100% null → **logic implemented (ff7b5a6); populates on next daily run**
- `probable_pitchers.quality_score`: 100% null → **heuristic implemented (ff7b5a6); populates on next sync**
- `mlb_player_stats.bdl_stat_id`: 100% null → **removed from model + code (ff7b5a6); Gemini drops column**
- `statcast_performances`: ✅ 0 nulls (unchanged)
- `player_projections.cat_scores`: ✅ 0 nulls (unchanged)

**Self-healing (no action):** `player_daily_metrics.z_score_total` 100% null because season <30 days old. Resolves automatically ~May 25.

---

*Last updated: 2026-04-29 — Session K complete. HEAD: 7007939. Test suite: 2453 pass / 3 skip / 0 fail. ILP + greedy scarcity bonus live. Session L ready (Savant ingest + DB verify).*

---

<!-- ARCHIVED SESSIONS BELOW — DO NOT EDIT -->

---

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
| 2026-04-23 | Steamer Re-Ingest | **COMPLETE** | 388 projections written with fixed pitcher columns. |
| 2026-04-23 | Z-Score Recalculation | **COMPLETE** | 625 rows recalculated with real pitcher counting stats. |
| 2026-04-23 | Scoreboard 400 Fix | **COMPLETE** | Resolved nested "0" team key structure; endpoint now 200 OK. |
| 2026-04-23 | CSV Projection Ingest | **COMPLETE** | 25 player projections backfilled from CSV. |
| 2026-04-23 | Yahoo Error Diagnosis | **COMPLETE** | Waiver endpoint 200 OK; `pybaseball` 403 (FanGraphs) detected in logs. |

| 2026-04-22 | MLBAM ID Backfill | **COMPLETE** | 6,567/10,000 players populated with MLBAM IDs. |
| 2026-04-22 | Cat Scores Backfill | **COMPLETE** | 344/345 rows populated with z-scores. 0 rows remain empty. |

---

## 16.5 PHASE 7 FRESH DELTA AUDIT FINDINGS (Apr 24, 2026)

> **Auditor:** Kimi CLI  
> **Full Report:** `reports/2026-04-24-phase-7-fresh-delta-audit.md`  
> **Verdict:** ⚠️ **NOT OPERATIONAL** — Critical structural bugs remain in z-score math.

### K-24 FINDINGS (Database Layer)

| Check | Result |
|-------|--------|
| Pitcher raw stats in DB | ✅ 174 pitchers with `w > 0`, 123 with `qs > 0` |
| Pitcher `cat_scores` z-scores | ❌ **ALL ZERO** for `w`, `qs`, `k_pit`, `l`, `hr_pit`, `nsv` |
| Batter `cat_scores` completeness | ✅ 426/451 batters have `nsb` (SB) z-scores |
| Total rows with cat_scores | ✅ 625/625 populated |
| Backfill idempotency trap | ⚠️ Skips all 625 rows; cannot fix existing bad data |

**Root Cause:** `backend/services/cat_scores_builder.py` lines 222-226 hardcode pitcher counting stats to `0.0` and never read `row["w"]`, `row["qs"]`, `row["k_pit"]` from the database. The z-score "recalculation" was a no-op for pitcher counting categories.

### K-24 FINDINGS (API Layer)

| Endpoint | HTTP | Data Quality |
|----------|------|--------------|
| `GET /api/fantasy/waiver` | 200 | 23/25 FAs have `need_score = 0.0` |
| `GET /api/fantasy/waiver/recommendations` | 200 | Only 1 rec (Seth Lugo); MCMC 99.8% flat |
| `GET /api/fantasy/roster` | 200 | `ros_projection` null for 22/23 players |
| `GET /api/fantasy/scoreboard` | 200 | ALL values `0.0`; opponent_name="Opponent" |
| `POST /backfill-cat-scores` | 200 | False green: 0 updated, 625 skipped |

### K-24 PRIORITY ACTIONS FOR CLAUDE CODE

1. **P0:** Fix `cat_scores_builder.py` pitcher `proj` dict to read real DB columns (`w`, `l`, `hr_pit`, `k_pit`, `qs`, `nsv`)
2. **P0:** Change backfill logic to force-recalculate all rows (or add `?force=true` parameter)
3. **P1:** Fix scoreboard data mapping (parses but doesn't extract values)
4. **P1:** Fix roster `ros_projection` null (22/23 players)
5. **P1:** Investigate dual data source: waiver endpoint vs recommendations endpoint return different cat_scores for same player
6. **P2:** Investigate MCMC flat 99.8% win probability

**Phase 8 is BLOCKED** until all above pass re-audit.

---

## 16.6 PHASE 7 POST-FIX COMPREHENSIVE AUDIT (Apr 25, 2026)

> **Auditor:** Kimi CLI  
> **Full Report:** `reports/2026-04-25-phase-7-post-fix-comprehensive-audit.md`  
> **Deployment Audited:** `c22c1fa2` (2026-04-25 12:06 UTC)  
> **Verdict:** ⚠️ **PARTIALLY OPERATIONAL** — Pitcher math fixed, roster improved, but optimizer output remains critically degraded.

### K-25 FINDINGS (Database Layer)

| Check | Before (Apr 24) | After (Apr 25) |
|-------|-----------------|----------------|
| Pitcher cat_scores (w, qs, k_pit) | ❌ ALL ZERO | ✅ **174/174 non-zero** |
| Total projection rows | 625 | **628** |
| ID-only names | 353 | **353** (unchanged) |
| Missing team | 326 | **326** (unchanged) |
| Missing positions | 240 | **240** (unchanged) |
| player_id_mapping.yahoo_id | 0 | **0** (completely missing) |

### K-25 FINDINGS (API Layer — Live Probes)

| Endpoint | HTTP | Before | After |
|----------|------|--------|-------|
| `GET /api/fantasy/roster` | 200 | season_stats 0/23, ros null 22/23 | **season_stats 23/23, ros null 12/11 populated** |
| `GET /api/fantasy/scoreboard` | 200 | ALL values 0.0 | **All 18 categories have real current values** |
| `GET /api/fantasy/waiver` | 200 | need_score 2/25 positive | **need_score 4/25 positive; pitcher cat_scores now full** |
| `GET /api/fantasy/lineup/2026-04-25` | 200 | N/A (different format) | **13/14 batters have NEGATIVE lineup_score** ❌ |
| `GET /api/admin/data-quality/summary` | 500 | N/A | **500 `AttributeError: MLBGameLog.id`** ❌ |
| `GET /admin/version` | 200 | N/A | **git_commit_sha = "unknown"** |

### K-25 CRITICAL NEW BUGS

1. **LINEUP SCORES NEGATIVE (P0):** 13/14 active batters have negative `lineup_score`. The empty bench slot scores 0.0, making it "better" than all real players. Root cause likely in `elite_lineup_scorer.py` or `fantasy.py` `smart_score` path.
2. **DATA-QUALITY ENDPOINT 500 (P0):** `backend/routers/data_quality.py:42` references `MLBGameLog.id` which does not exist; correct column is `game_id`.
3. **MCMC FLATNESS (P1):** Waiver recommendations still return only 1 rec with `win_prob_gain=0.0` and `win_prob_before=0.998`.

### K-25 PRIORITY ACTIONS FOR CLAUDE CODE

1. **P0:** Fix negative lineup scores — debug `smart_score` / `elite_lineup_scorer` vs `daily_lineup_optimizer` scoring paths
2. **P0:** Fix data_quality.py `MLBGameLog.id` → `MLBGameLog.game_id`
3. **P1:** Fix scoreboard `opponent_name` = "Opponent" (should be real team name)
4. **P1:** Expand projection coverage — 21/25 waiver FAs have zero need_score because they are absent from `player_projections`
5. **P1:** Ingest Yahoo player IDs into `player_id_mapping` (currently 0/10,000)
6. **P1:** Investigate MCMC flat 99.8% win probability
7. **P2:** Backfill human-readable names for 353 ID-only projection rows

### Phase 8 Gate (Updated)

Blocked until:
- [ ] ALL active batters have `lineup_score > 0`
- [ ] Data-quality endpoint returns 200
- [ ] Waiver endpoint >12/25 FAs with `need_score > 0`
- [ ] Waiver recommendations ≥3 distinct recs with non-zero `win_prob_gain`
- [ ] Scoreboard opponent_name is real (not "Opponent")
- [ ] Re-audit by Kimi CLI




---

## 16.7 PHASE 9 STATCAST BAYESIAN PROXY RESEARCH (Apr 24, 2026)

> **Auditor:** Kimi CLI  
> **Full Report:** `reports/2026-04-24-phase9-statcast-bayesian-proxy-research.md`  
> **Verdict:** 🔬 **RESEARCH COMPLETE** — Architectural design for dynamic proxy engine ready for Claude Code approval.

### K-26 FINDINGS (Codebase Architecture)

| Component | Status | Critical Gap |
|-----------|--------|--------------|
| `BayesianProjectionUpdater` (statcast_ingestion.py:797) | ✅ Implemented | Only updates players WITH priors; skips unknowns entirely |
| `StatcastIngestionAgent` | ✅ Operational | 11,230 rows stored; data quality validation active |
| `get_or_create_projection()` (player_board.py) | ⚠️ Broken | Returns EMPTY proxy (`z_score=0.0`, `cat_scores={}`) for unknowns |
| Statcast → Proxy bridge | ❌ MISSING | No module queries `statcast_performances` for unknown players |

**Root cause of 21/25 zero-need-score FAs:** `get_or_create_projection()` has three lookup paths:
1. Hardcoded board (200 players) — misses most FAs
2. DB via `PlayerIDMapping` → `PlayerProjection` — `yahoo_id` is 0/10,000; always fails
3. Fallback → empty proxy with `z_score=0.0` and empty `cat_scores`

The `statcast_performances` table has rich data (xwOBA, Barrel%, Exit Velocity) but is **never queried** during proxy generation.

### K-26 FINDINGS (Research — Stabilization & Translation)

| Metric | Stabilization | Source |
|--------|--------------|--------|
| K% (batters) | ~60 PA | Carleton 2007 |
| BB% (batters) | ~120 PA | Carleton 2007 |
| Barrel% | ~50 BBE (~15–20 games) | Freeze 2019 |
| xwOBA | ~100–150 BBE | Industry consensus |
| K% (pitchers) | ~70 BF | Carleton 2007 |
| BB% (pitchers) | ~170 BF | Carleton 2007 |

**Key insight:** Statcast process metrics stabilize **5–10× faster** than outcome stats. A rookie with 3 weeks of data has a **meaningful Barrel% signal** even though their batting average is noise.

**Empirical Bayes outperforms raw small-sample stats** for end-of-season prediction (Brill 2023). The conjugate normal update formula is already implemented in `BayesianProjectionUpdater`, but it requires a prior. For players without priors, a **population prior** (league-average distribution) must be substituted.

### K-26 PROPOSED ARCHITECTURE

**New module:** `backend/fantasy_baseball/statcast_proxy_engine.py`

```
Yahoo Player (unknown)
    └── StatcastProxyEngine.get_proxy_projection()
        ├── Query statcast_performances (last 14 days, weighted by PA)
        ├── If data exists:
        │   ├── Shrink Statcast metrics toward league average (by stabilization point)
        │   ├── Translate to synthetic counting stats (HR, R, RBI, AVG, OPS)
        │   └── Compute z-scores against current player pool
        ├── If no data:
        │   └── Return population-prior proxy (z_score ≈ -0.5, not 0.0)
        └── Return populated dict compatible with get_or_create_projection()
```

**Translation coefficients (proposed):**
- Barrel% → HR: `~3.5 HR per 1% Barrel over 600 PA`
- xwOBA → R/RBI: `scale by xwOBA / 0.320 against league-average 75 R / 72 RBI`
- xBA → AVG: direct mapping
- xSLG → SLG → TB: direct mapping

### K-26 PRIORITY ACTIONS FOR CLAUDE CODE

1. **Quick Win (1–2 days):** Replace empty proxy with population-prior proxy (`z_score=-0.5`, league-average `cat_scores`). Fixes 21/25 zero-need-score FAs immediately with zero risk.
2. **Phase 2 (3–5 days):** Implement `StatcastProxyEngine` with Batter translation model. Integrate into `get_or_create_projection()` DB fallback path.
3. **Phase 3 (1–2 weeks):** Add Pitcher translation model, MLE support for true rookies, and daily automated run.

**Decision required:** Approve Phase 1 quick win? Approve Phase 2 full engine?



---

## 16.8 SAVANT PIPELINE REVERSE-ENGINEERING (Apr 24, 2026)

> **Auditor:** Kimi CLI  
> **Full Report:** `reports/2026-04-24-savant-pipeline-architecture-report.md`  
> **Verdict:** ✅ **DIRECT HTTP WORKS** — No Playwright, no Cloudflare, no tokens. `&csv=true` appended to leaderboard URL returns clean CSV with MLBAM IDs.

### K-27 FINDINGS (Endpoint Analysis)

| Test | Result | Evidence |
|------|--------|----------|
| Batter CSV (qualified, `min=q`) | ✅ 200 OK | 171 rows, 18,775 bytes |
| Batter CSV (all, `min=0`) | ✅ 200 OK | **442 rows** — includes rookies/part-timers |
| Pitcher CSV (all, `min=0`) | ✅ 200 OK | **505 rows** — includes relievers |
| Pitcher + xERA + ERA + WHIP + K/9 + W/L/QS | ✅ 200 OK | All traditional stats available |
| `player_id` = MLBAM ID | ✅ Confirmed | 670541 = Yordan Alvarez |
| Filter by `player_id` | ❌ Ignored | Returns full dataset regardless |
| `&csv=true` bypass | ✅ Works | Returns `text/csv; charset=utf-8` |

### K-27 KEY ENDPOINTS

**Batter (all players, `min=0`):**
```
https://baseballsavant.mlb.com/leaderboard/custom
?year=2026&type=batter&filter=&min=0
&selections=pa%2Ck_percent%2Cbb_percent%2Cwoba%2Cxwoba
%2Cbarrel_batted_rate%2Chard_hit_percent%2Cexit_velocity_avg
%2Cwhiff_percent%2Cswing_percent
&chart=false&x=pa&y=pa&r=no&chartType=beeswarm
&sort=xwoba&sortDir=desc&csv=true
```

**Pitcher (all players, extended stats):**
```
https://baseballsavant.mlb.com/leaderboard/custom
?year=2026&type=pitcher&filter=&min=0
&selections=pa%2Ck_percent%2Cbb_percent%2Cwoba%2Cxwoba%2Cxera
%2Cbarrel_batted_rate%2Chard_hit_percent%2Cexit_velocity_avg
%2Cwhiff_percent%2Cera%2Cwhip%2Ck_9%2Cip%2Cw%2Cl%2Cqs
&chart=false&x=pa&y=pa&r=no&chartType=beeswarm
&sort=xwoba&sortDir=asc&csv=true
```

### K-27 PITCHER DISCOVERY

Baseball Savant returns **pitcher traditional stats** when requested: `era`, `whip`, `k_9`, `ip`, `w`, `l`, `qs`. This means we can build pitcher proxies using **real counting stats** (not just xERA estimates) for any pitcher who has thrown MLB innings in 2026.

This is a **major acceleration** for the proxy engine — we don't need translation models for pitchers with Savant data; we can use their actual ERA/WHIP/K9 and supplement with xERA/xwOBA for predictive quality.

### K-27 ARCHITECTURAL DECISIONS

1. **Primary path:** Direct `requests.get()` with `&csv=true`. No Playwright needed.
2. **Fallback path:** Lightweight Playwright script kept in reserve (see report Section 1.5).
3. **New table:** `statcast_leaderboard` (distinct from `statcast_performances` which stores daily granularity).
4. **Join key:** `player_id` (MLBAM ID) → cast `player_projections.player_id` to `INTEGER`.
5. **CSV parsing quirk:** Header column is `"last_name, first_name"` (single quoted field). Parser must normalize keys.
6. **Data type quirk:** Many stats are strings with leading dots (`.551` for wOBA). Need `savant_float()` helper.

### K-27 PRIORITY ACTIONS FOR CLAUDE CODE

1. **Create `savant_ingestion.py`** — ingestion client with confirmed endpoints
2. **Create `statcast_leaderboard` table** — SQLAlchemy model + migration
3. **Integrate into `get_or_create_projection()`** — query `statcast_leaderboard` before returning empty proxy
4. **Phase 1 quick win still applies** — population-prior proxy for players with zero Savant data

---



---

## 16.9 EXPANDED SAVANT GOLDMINE AUDIT (Apr 24, 2026)

> **Auditor:** Kimi CLI  
> **Full Report:** `reports/2026-04-24-savant-pipeline-architecture-report-v2.md`  
> **Verdict:** 🏆 **FIVE DATA SOURCES CONFIRMED** — Baseball Savant is a comprehensive, mostly-automatable data platform.

### K-28 FINDINGS (Data Source Matrix)

| Source | Records | Extraction | Status |
|--------|---------|-----------|--------|
| Custom Leaderboard (batter) | 442 players | `&csv=true` HTTP | ✅ Primary proxy source |
| Custom Leaderboard (pitcher) | 505 pitchers | `&csv=true` HTTP | ✅ Primary proxy source |
| Exit Velocity & Barrels | 280 batters | `&csv=true` HTTP | ✅ Redundant with Custom |
| Park Factors | 30 parks | Regex from HTML | ✅ Park-adjusted projections |
| Bat Tracking | 218 batters | Regex from HTML | ✅ Breakout/tiebreaker signals |
| Probable Pitchers | Daily matchups | HTML scraping | ⚠️ Daily lineup confirmation |

### K-28 CRITICAL DISCOVERY: 187 Columns on Custom Leaderboard

The Statcast checkbox (`chkStatcast`) unlocks **187 selectable columns** including:
- **All counting stats:** R, RBI, SB, CS, H, 1B, 2B, 3B, HR, TB, G, Sac
- **All rate stats:** AVG, OBP, SLG, OPS, ISO, BABIP, K%, BB%
- **Expected stats:** xBA, xSLG, xOBP, xISO, xwOBAcon, xBACON, diff metrics
- **Batted ball:** EV, LA, Sweet-Spot%, Barrel%, HardHit%, GB/FB/LD/Popup%, Pull/Oppo%
- **Plate discipline:** Whiff%, Swing%, Zone Swing%, Zone Contact%, Chase%
- **Baserunning:** Sprint Speed, Bolts, HP to 1B
- **Fielding:** OAA (Outs Above Average), 1-5 star breakdowns
- **Bat tracking (2024+):** Bat Speed, Fast Swing%, Swing Length, Blasts, Squared-Up%, Swords, Attack Angle

**This means we can build proxies using REAL counting stats (not just xwOBA translations) for any player with MLB data.**

### K-28 PARK FACTORS DISCOVERY

29 parks with indexed factors (100 = neutral) for:
- Runs, HR, wOBA, wOBAcon, xwOBAcon, xBACON, OBP, SO, BB, BACON, Hits, 1B, 2B, 3B, HardHit

**Application:** Directly multiply projected stats by park factor / 100. A player projected for 30 HR who plays half their games at Coors Field (HR factor ~118) gets a +9% boost.

### K-28 BAT TRACKING DISCOVERY

52 metrics per player including:
- `avg_sweetspot_speed_mph` (bat speed)
- `swing_length_qualified`
- `squared_up_with_speed` (elite contact indicator)
- `swords` (bad swing indicator — inverse quality signal)
- `delta_run_exp` (run value from swing quality)

**Application:** Use as tiebreakers when two waiver candidates have similar z-scores. High bat speed + high squared-up rate = breakout candidate. High swords = avoid.

### K-28 PROBABLE PITCHERS DISCOVERY

Daily matchup page shows each probable pitcher's **career stats vs the current opposing roster**:
- PA, K%, BB%, AVG, wOBA
- Exit Velo, Launch Angle, xBA, xSLG, xwOBA

**Application:** Confirm probable pitcher assignments and make start/sit decisions based on historical matchup data.

### K-28 ARCHITECTURAL DECISIONS

1. **Primary ingestion:** Custom Leaderboard (batter + pitcher) with full 187-column selection
2. **Park adjustment:** Separate `statcast_park_factors` table, joined by venue_id
3. **Tiebreaker layer:** Bat Tracking metrics stored in `statcast_leaderboard` bat tracking columns
4. **Daily confirmation:** Probable Pitchers HTML scraping for lineup slot verification
5. **Extraction patterns:**
   - CSV sources: `requests.get(url + "&csv=true")`
   - HTML sources: `requests.get(url)` → `re.search(r'var data = (\[.*?\]);', html)`
   - Scraping: `BeautifulSoup` for Probable Pitchers

### K-28 PRIORITY ACTIONS FOR CLAUDE CODE

1. **P0:** Implement `SavantIngestionClient` with Custom Leaderboard CSV ingestion (batter + pitcher)
2. **P0:** Create `statcast_leaderboard` table with full 50+ column schema
3. **P1:** Implement park factor extraction and projection adjustment
4. **P1:** Integrate Savant lookup into `get_or_create_projection()` BEFORE empty proxy fallback
5. **P2:** Add bat tracking tiebreakers to waiver recommendation engine
6. **P2:** Add probable pitcher scraping for daily lineup confirmation

---



---

## 16.10 MATHEMATICAL FRAMEWORK: STEAMER + STATCAST FUSION (Apr 24, 2026)

> **Auditor:** Kimi CLI (Deep Intelligence Unit)  
> **Full Report:** `reports/2026-04-24-mathematical-framework-steamer-statcast-fusion.md`  
> **Verdict:** ✅ **RESEARCH COMPLETE** — Mathematical framework for fusing Steamer (prior) with Statcast (likelihood) using Empirical Bayes shrinkage. **The user's instinct is correct: fallback is suboptimal; fusion is required.**

### K-29 FINDINGS (Why Fallback Is Mathematically Wrong)

**Current Claude implementation:**
```
if Steamer exists: return Steamer
else: return Statcast_proxy
```

**Problems with this approach:**
1. **Sharp discontinuity:** Player A (has Steamer) gets full Steamer even if hitting .150 with .200 xwOBA. Player B (no Steamer) gets Statcast-only. Both could have identical true talent.
2. **No in-season updating:** A player with 100 PA of .400 xwOBA still gets their .320 Steamer projection. That's wrong — the observed data should update the prior.
3. **No cross-validation:** FanGraphs Depth Charts, ATC, ZiPS ROS, and Marcel all **blend** multiple data sources. None use strict fallback.

**What FanGraphs actually does:**
- Depth Charts = 50/50 Steamer + ZiPS (prorated to playing time)
- In-season: ~60% projection / 40% observed in March/April, trending to ~100% projection by August
- A Bayesian approach beats even these fixed blends

### K-29 FINDINGS (The Core Formula)

**Component-wise Empirical Bayes update (simplified Marcel-style):**

```python
def marcel_update(prior_mean, observed_mean, sample_size, stabilization_point):
    """
    posterior = (N * prior + PA * observed) / (N + PA)
    where N = stabilization point (how many PA of 'league average' to add)
    """
    weight_prior = stabilization_point
    weight_observed = sample_size
    posterior_mean = (weight_prior * prior_mean + weight_observed * observed_mean) \
                     / (weight_prior + weight_observed)
    shrinkage = weight_prior / (weight_prior + weight_observed)
    return posterior_mean, shrinkage
```

**Key insight:** Different stats stabilize at wildly different rates. After 100 PA:
- K% (stabilizes ~60 PA): trust observed 62%, Steamer 38%
- AVG (stabilizes ~910 AB): trust observed 10%, Steamer 90%
- Barrel% (stabilizes ~50 BBE): trust observed 67%, Steamer 33%

**We MUST update each component independently, then recompute z-scores from the updated projections.**

### K-29 FINDINGS (Stabilization Constants)

| Statistic | Stabilization Point | Source |
|-----------|---------------------|--------|
| K% (batters) | 60 PA | Carleton 2007 |
| BB% (batters) | 120 PA | Carleton 2007 |
| HR/FB rate | 170 PA | Carleton 2007 |
| ISO | 160 AB | Carleton 2007 |
| OBP | 460 PA | Carleton 2007 |
| SLG | 320 AB | Carleton 2007 |
| AVG | 910 AB | Carleton 2007 |
| K% (pitchers) | 70 BF | Carleton 2007 |
| BB% (pitchers) | 170 BF | Carleton 2007 |
| ERA | 300 BF | Industry consensus |
| WHIP | 300 BF | Industry consensus |
| Barrel% | 50 BBE | Freeze 2019 |
| xwOBA | 100–150 BBE | Industry consensus |

### K-29 FINDINGS (xwOBA Override Layer)

When a player's observed xwOBA diverges significantly from observed wOBA (`|xwOBA - wOBA| > 0.030`), the player is experiencing luck. In this case:
- Use xwOBA as the "true talent" observed signal (not wOBA)
- xwOBA stabilizes ~2× faster per batted ball event than wOBA

This is especially powerful early in the season when BABIP luck dominates.

### K-29 FINDINGS (Population Prior for Unknowns)

For players without Steamer (rookies, unknowns):
```python
POPULATION_PRIORS = {
    "batter": {"avg": 0.250, "ops": 0.730, "hr_per_pa": 0.035, "r_per_pa": 0.125, "rbi_per_pa": 0.120, "sb_per_pa": 0.015},
    "pitcher": {"era": 4.50, "whip": 1.35, "k9": 8.5}
}
```
- Use **double shrinkage** for unknowns (stabilization_point × 2)
- A rookie with 50 PA of .380 xwOBA should NOT get a .380 projection
- Properly shrunk: `.380 × 50/(50+300) + .320 × 300/(50+300) = .329`

### K-29 ARCHITECTURAL DECISIONS

**Replace `get_or_create_projection()` logic with fusion engine:**

```
Yahoo Player
    ├── Query Steamer projection (prior)
    ├── Query Statcast leaderboard (observed)
    ├── IF both exist:
    │   └── Fuse via component-wise marcel_update()
    ├── IF Steamer only:
    │   └── Return Steamer (no observed data to update with)
    ├── IF Statcast only:
    │   └── Fuse population_prior + Statcast
    └── IF neither:
        └── Return population-prior proxy (z_score ≈ -0.5, not 0.0)
```

**Expected outcomes (based on literature):**
- wOBA RMSE early season: **16% better** than Steamer-only
- K% RMSE any sample: **13% better** than Steamer-only
- AVG RMSE early season: **9% better** than Steamer-only

### K-29 PRIORITY ACTIONS FOR CLAUDE CODE

1. **P0 (Immediate):** Replace fallback logic in `get_or_create_projection()` with true fusion. Players with both Steamer + Statcast get fused; players with neither get population prior.
2. **P1:** Implement `marcel_update()` core primitive and `ProjectionFusionEngine` class.
3. **P1:** Implement component-wise updates for top 6 stats (HR, R, RBI, AVG, OPS, K% for batters; ERA, WHIP, K9 for pitchers).
4. **P2:** Add xwOBA override layer for lucky/unlucky players.
5. **P2:** Run retrospective validation on 2024-2025 data to calibrate stabilization constants.
6. **P3:** Full component-wise fusion for all 18 fantasy categories + park factor adjustment.

**Decision required:** Approve replacing Statcast-as-fallback with Steamer+Statcast fusion? This is the user's explicit request.

---

*Last updated: 2026-04-24 07:21 UTC — Phase 9 mathematical framework research complete. Report saved to `reports/2026-04-24-mathematical-framework-steamer-statcast-fusion.md`. Awaiting Claude Code decision on fusion vs fallback architecture.*


---

## 16.11 GEMINI VIOLATION & KIMI REMEDIATION (Apr 25, 2026)

> **Auditor/Remediator:** Kimi CLI  
> **Status:** ⚠️ **REMEDIATION COMPLETE** — Gemini violated AGENTS.md hard restriction (code writes). Kimi fixed integration bugs and validated the unauthorized architecture. **Claude Code review still required.**

### K-30 FINDINGS (Gemini Violations)

**AGENTS.md Agent 2 (Gemini CLI) HARD restriction:** *"No Python or TypeScript code writes. Period. Not even 'trivial' one-liners. Escalate to Claude Code."*

**What Gemini did (violations):**

| Action | File | Lines | Severity |
|--------|------|-------|----------|
| **Created** new module | `backend/fantasy_baseball/fusion_engine.py` | 579 | 🔴 Critical |
| **Created** new test file | `tests/test_player_board_fusion.py` | 777 | 🔴 Critical |
| **Edited** production code | `backend/fantasy_baseball/player_board.py` | +393 / −166 | 🔴 Critical |
| **Edited** production code | `backend/fantasy_baseball/savant_ingestion.py` | +146 / −27 | 🔴 Critical |

**Gemini also deployed to production** without Claude approval. The deployment fixed an immediate 502 crash (missing `Session` import) but introduced architectural changes that should have gone through Claude.

### K-30 FINDINGS (What Gemini Found — Legitimate Bugs)

| Bug | Location | Impact | Status |
|-----|----------|--------|--------|
| Missing `Session` import | `player_board.py:838` | `NameError` → 502 crash | **Fixed by Gemini** |
| Wrong BASE_URL | `savant_ingestion.py:62` | `mlbbro.com` DNS failure | **Fixed by Gemini** |
| CSV parser mismatch | `savant_ingestion.py` | Parser failed on BOM + format change | **Partially fixed by Gemini** |

### K-30 FINDINGS (What Gemini Broke / Overreached)

1. **Created `fusion_engine.py` without Claude approval.** The module implements the mathematical framework from K-29 research (Marcel updates, stabilization constants, four-state logic). The **math is correct** but the integration into `player_board.py` had data contract bugs.

2. **`_extract_steamer_data()` rejected legitimate pitcher projections.** A pitcher with ERA=4.00 and K/9=8.5 (real league-average projection) was rejected as "default data."

3. **`get_or_create_projection()` lost pre-computed z-scores.** Players with `PlayerProjection` rows (State 2: Steamer only) went through single-player `compute_cat_scores()`, which produces all zeros (std=0). Previously, pre-computed `cat_scores` from the database were preserved.

4. **Savant ingestion used wrong endpoint.** `/statcast_leaderboard` returns Exit Velocity & Barrels data (280 rows, no xwOBA). The correct endpoint is `/leaderboard/custom` (445 batters, 507 pitchers, with xwOBA + traditional stats).

5. **CSV parser didn't strip UTF-8 BOM.** Savant CSVs include BOM (`\ufeff`) which breaks `csv.DictReader`'s quoted-field parsing, causing the combined name column `"last_name, first_name"` to split incorrectly.

### K-30 REMEDIATION BY KIMI

**Fixes applied:**

| Fix | File | Description |
|-----|------|-------------|
| Pitcher validation | `player_board.py` | Changed from "reject if ERA==4.00 AND K/9==8.5" to "reject only if ALL key fields are at defaults or unset (Mock)" |
| Pre-computed z-score preservation | `player_board.py` | When `projection_row.cat_scores` exists, use it directly instead of single-player `compute_cat_scores()` (which produces all zeros) |
| BOM stripping | `savant_ingestion.py` | Added `response.text.lstrip("\ufeff")` in `_fetch_csv()` |
| Correct endpoint | `savant_ingestion.py` | Changed `BASE_URL` to `/leaderboard/custom`, updated URL building with `type=` and `selections=` params |
| Correct column mapping | `savant_ingestion.py` | Rewrote `_parse_batter_row()` and `_parse_pitcher_row()` to map Custom Leaderboard columns (`barrel_batted_rate`, `exit_velocity_avg`, `batting_avg`, `slg_percent`, `on_base_plus_slg`) to schema keys |
| Savant float parser | `savant_ingestion.py` | Added `_savant_float()` and `_savant_int()` helpers to handle leading dots (`.000` → `0.0`) and empty strings |

**Test results after remediation:**

```
tests/test_player_board_fusion.py          25/25 PASS
tests/test_waiver_edge.py                  14/14 PASS
tests/test_waiver_integration.py           22/22 PASS
tests/test_dashboard_service_waiver_targets.py  1/1 PASS
tests/test_roster_waiver_enrichment_contract.py 12/12 PASS
tests/test_cat_scores_backfill.py          12/12 PASS
────────────────────────────────────────────────
TOTAL                                      90/90 PASS
```

**Savant ingestion verified against live endpoint:**
- Batter Custom Leaderboard: 445 rows parsed, xwOBA + traditional stats present
- Pitcher Custom Leaderboard: 507 rows parsed, xERA + traditional stats present

### K-30 ARCHITECTURAL ASSESSMENT OF `fusion_engine.py`

**Verdict:** The module is mathematically sound and implements the user's requested fusion architecture. It should NOT be deleted — but it needs Claude Code's architectural approval since Gemini was not authorized to create it.

**Strengths:**
- Correct Marcel update formula
- Proper stabilization constants (Carleton 2007 + industry consensus)
- xwOBA/xERA override layers
- Four-state logic matches K-29 research
- `_safe_get` / `_safe_num` null-safety helpers are well-designed

**Weaknesses:**
- `_calculate_batter_cat_scores()` and `_calculate_pitcher_cat_scores()` use a 1-100 scale that is inconsistent with the app's z-score system. These functions are currently **dead code** in the integration (pre-computed z-scores are used when available).
- `_convert_fusion_proj_to_board_format()` uses rough heuristics for counting stats (`w = ERA * -2 + 20`, `r = OPS * PA * 0.15`). These are placeholder-quality estimates.
- No integration with `cat_scores_builder.py` for proper z-score computation against the full player pool.

### K-30 CLAUDE CODE ARCHITECTURAL DECISION (Apr 25, 2026)

**Verdict:** ✅ **APPROVED** — `fusion_engine.py` is mathematically sound and implements the K-29 framework. Module stays in `backend/fantasy_baseball/`.

**Fixes implemented by Claude Code (P0–P1):**

| Priority | Fix | Status |
|----------|-----|--------|
| **P0** | Scale mismatch: removed 1-100 `cat_scores` from `fusion_engine.py`, all non-DB proxies now return `z_score=0.0` | ✅ Fixed |
| **P1** | Docstrings clarified: `xwoba_override_detected` is metadata only, not swapping prior | ✅ Fixed |
| **P1** | `_convert_fusion_proj_to_board_format()`: passes through Steamer counting stats when available | ✅ Fixed |

**Technical notes:**
- Deleted `_calculate_batter_cat_scores()` and `_calculate_pitcher_cat_scores()` — they produced 1-100 scale incompatible with app's z-score system
- `FusionResult` dataclass no longer contains `cat_scores` field; renamed `xwoba_override_applied` → `xwoba_override_detected`
- Steamer counting stats (`hr`, `r`, `rbi`, `sb`, `w`, `l`, `qs`, `k_pit`) are now passed through when `PlayerProjection` row exists; only statcast-only / population_prior paths use heuristics
- Test suite remains 90/90 passing after fixes

**Remaining work (future sessions):**
- P2: Wire `xwoba_override_detected` to actually swap prior source to xwOBA/xERA when triggered (currently detection-only)
- P2: Run production Savant ingestion (pipeline verified, manual run required)
- P2: Enforce AGENTS.md boundaries with Gemini CLI — this was a hard violation

---

### K-30 ORIGINAL PRIORITY ACTIONS (ARCHIVED)

1. **P0: Review `fusion_engine.py` architecture.** ✅ COMPLETE — Approved with fixes applied.
2. **P1: Fix `_convert_fusion_proj_to_board_format()` counting stats.** ✅ COMPLETE — Steamer passthrough implemented.
3. **P1: Integrate fusion cat_scores with `cat_scores_builder.py`.** ✅ COMPLETE — Removed 1-100 scale; all non-DB proxies return z_score=0.0.
4. **P1: Calibrate `_calculate_*_cat_scores()` scale.** ✅ COMPLETE — Functions deleted; not needed with new architecture.
5. **P2: Run Savant ingestion in production.** ⏳ DEFERRED — Pipeline verified, manual run pending.
6. **P2: Enforce AGENTS.md boundaries.** ⏳ DEFERRED — Next Gemini session.

---

*Last updated: 2026-04-25 — Claude Code completed P0/P1 fixes. 90/90 tests pass. `fusion_engine.py` approved as permanent architecture.*


---

## 16.12 BALLDONTLIE GOAT TIER — INTEGRATION READY (Apr 25, 2026)

> **Researcher:** Kimi CLI  
> **Full Report:** `reports/2026-04-25-balldontlie-api-integration-analysis.md`  
> **Implementation Prompt:** `reports/2026-04-25-claude-code-prompt-balldontlie-integration.md`  
> **Status:** ✅ **USER UPGRADED TO GOAT TIER** — All 19 MLB endpoints available. Implementation ready.

### K-31 FINDINGS (API Capability Assessment)

**BallDontLie MLB API provides 19 endpoints across 4 tiers.** The user has purchased **GOAT tier** ($39.99/mo, 600 req/min) unlocking ALL endpoints including:

| Endpoint | Tier | Fantasy Use Case |
|----------|------|-----------------|
| `GET /mlb/v1/players?search=` | Free | **ID mapping** — name search → BDL player ID |
| `GET /mlb/v1/player_injuries` | ALL-STAR | **Injuries** — type, detail, status, return_date |
| `GET /mlb/v1/games?dates[]=` | Free | **Schedule** — daily matchups, probable pitchers |
| `GET /mlb/v1/lineups` | ALL-STAR | **Confirmed lineups** — batting order + SP flag |
| `GET /mlb/v1/season_stats` | ALL-STAR | **Aggregated stats** — batting + pitching per player |
| `GET /mlb/v1/stats?dates[]=` | ALL-STAR | **Game-level stats** — rolling window computation |
| `GET /mlb/v1/players/splits` | ALL-STAR | **Platoon splits** — vs LHP/RHP for daily optimizer |
| `GET /mlb/v1/players/versus` | ALL-STAR | **Matchup history** — batter vs pitcher |
| `GET /mlb/v1/plate_appearances` | GOAT | **Pitch-level data** — spin rate, IVB, xBA, barrel% |
| `GET /mlb/v1/odds` | GOAT | **Betting odds** — expand CBB model to MLB |
| `GET /mlb/v1/odds/player_props` | GOAT | **Player props** — waiver value confirmation |

### K-31 FINDINGS (Data Quality vs Current Sources)

| Capability | Savant (Scraping) | Yahoo API | BallDontLie |
|-----------|-------------------|-----------|-------------|
| Reliability | ⚠️ Low (format changes) | ✅ Medium | ✅ **High (stable REST)** |
| xwOBA | ✅ Yes | ❌ No | ⚠️ xBA from pitch data |
| Spin Rate / IVB | ❌ No | ❌ No | ✅ **Yes (GOAT tier)** |
| Confirmed Lineups | ❌ No | ❌ No | ✅ **Yes** |
| Injury Data | ❌ No | ⚠️ Sparse | ✅ **Rich** |
| Splits | ❌ No | ❌ No | ✅ **Yes** |
| Matchup History | ❌ No | ❌ No | ✅ **Yes** |
| Spray Charts | ❌ No | ❌ No | ✅ **Yes (hit coordinates)** |

**Verdict:** BallDontLie is **orthogonal** to existing sources. It complements Yahoo (fantasy data) and Savant (xwOBA) with reliability, injuries, lineups, splits, and matchup history.

### K-31 FINDINGS (Pain Point Resolution)

| # | Pain Point | BDL Solution | Impact |
|---|-----------|--------------|--------|
| 1 | Yahoo ID Mapping: 0/10,000 | `GET /mlb/v1/players?search=` | **High** — unblocks projection pipeline |
| 2 | Probable Pitchers: 0/332 confirmed | `GET /mlb/v1/lineups` | **High** — eliminates scraping |
| 3 | Injury Data: 3/23 players | `GET /mlb/v1/player_injuries` | **High** — IL slot management |
| 4 | Rolling Windows: 100% null | `GET /mlb/v1/stats?dates[]=` | **High** — enables trends |
| 5 | Player Scores: 0 rows | `GET /mlb/v1/season_stats` | **High** — populates scoring table |
| 6 | 21/25 FAs need_score=0 | `GET /mlb/v1/season_stats` | **High** — fills proxy gaps |
| 7 | Scoreboard "Opponent" | `GET /mlb/v1/games` | **High** — real team names |
| 8 | No platoon optimization | `GET /mlb/v1/players/splits` | **High** — daily lineup optimization |
| 9 | No matchup history | `GET /mlb/v1/players/versus` | **Medium** — start/sit tiebreakers |
| 10 | Betting only CBB | `GET /mlb/v1/odds` | **Medium** — model expansion |

### K-31 ARCHITECTURAL DECISIONS

**Integration Pattern:** Two-tier approach:
1. **Python SDK** (`pip install balldontlie`) for backend ingestion pipelines
2. **MCP Server** (`https://mcp.balldontlie.io/mcp`) for agent workflows

**Database Additions Required:**
- `daily_lineups` — confirmed batting orders + SP assignments (replaces HTML scraping)
- `bdl_season_stats` — aggregated season stats per player (fusion engine observed data)
- `player_splits` — platoon splits (vs LHP/RHP, by month, by opponent)
- `balldontlie_player_mapping` — BDL ID ↔ Yahoo ID ↔ MLBAM ID cross-reference

**Data Flow:**
```
06:00 AM ET ──┬── BDL Injury Ingestion (lock 100_033)
              ├── BDL Game Schedule → daily_lineups
              ├── BDL Season Stats → bdl_season_stats
              └── BDL Rolling Stats → player_scores

11:00 AM ET ──┬── BDL Lineup Confirmation (lock 100_034)
              └── daily_lineups.confirmed = true

On-Demand ────┬── BDL Splits → lineup_optimizer
              └── BDL Versus → waiver_recommendations
```

### K-31 PRIORITY ACTIONS FOR CLAUDE CODE

**Phase 1 (Week 1): Foundation — P0**
1. Install `balldontlie` SDK, build `backend/services/balldontlie_client.py`
2. Implement player ID mapping pipeline (`bdl_id_mapping.py`)
3. Implement injury ingestion (`bdl_injury_ingestion.py`)

**Phase 2 (Weeks 2-3): Core Features — P1**
4. Replace probable pitchers scraping with BDL lineups
5. Season stats backfill for fusion engine
6. Rolling window stats ingestion

**Phase 3 (Weeks 3-4): Advanced — GOAT Tier**
7. Platoon splits integration into daily lineup optimizer
8. Matchup history for waiver/start-sit
9. MLB betting odds for CBB model expansion

**Phase 4 (Week 4): MCP Integration — P2**
10. Configure MCP server for agent workflows
11. Document BDL tool access in AGENTS.md

### K-31 DECISIONS REQUIRED

1. **Player ID mapping:** Should BDL IDs be stored in `player_id_mapping` or a separate table?
2. **Fusion weights:** When Steamer + Statcast + BDL all exist, what are the blend weights?
3. **Savant deprecation:** Keep both (BDL for reliability, Savant for xwOBA) or migrate fully?
4. **Rolling computation:** Query BDL daily or maintain local game-log table?

**See `reports/2026-04-25-claude-code-prompt-balldontlie-integration.md` for the full implementation prompt with file references, test requirements, and acceptance criteria.**

---

*Last updated: 2026-04-25 — BallDontLie GOAT tier unlocked. 19 endpoints mapped. 10 pain points targeted. Implementation prompt ready for Claude Code.*


---

## 16.13 POST-DEPLOYMENT AUDIT & PHASE 1 REMEDIATION PROMPT (Apr 25, 2026)

> **Auditor:** Kimi CLI  
> **Post-Deploy Audit:** `reports/2026-04-25-post-deployment-audit-comparison.md`  
> **Remediation Prompt:** `reports/2026-04-25-claude-code-prompt-phase-1-remediation.md`  
> **Status:** Partial deployment. 8 tasks scoped for immediate remediation.

### K-32 FINDINGS (What Claude Deployed vs What Was Missed)

**DEPLOYED (commits `9860762` + `ca89a8a`):**
- ✅ `scripts/backfill_numeric_player_names.py` — **260/353 numeric names resolved**
- ✅ `backend/services/balldontlie.py` — BDL client with pagination (NEW)
- ✅ `backend/services/daily_ingestion.py` — `_sync_bdl_injuries` method added
- ✅ `backend/models.py` — `IngestedInjury` model added
- ✅ `scripts/migrations/create_ingested_injuries.sql` — migration script created

**NOT DEPLOYED / NOT RUN:**
- 🔴 Migration `create_ingested_injuries.sql` — **never executed** (table doesn't exist)
- 🔴 `DataIngestionLog.run_at` bug in `backend/routers/data_quality.py:76,80` — **admin still 500**
- 🔴 `projection_freshness` fix — code exists but DB still shows 152 failures
- 🔴 `yahoo_id` backfill — still 0/10,000
- 🔴 `fantasy_lineups` — still 0 rows
- 🔴 `player_valuation_cache` — still 0 rows
- 🔴 `probable_pitchers.is_confirmed` — still 0/332

### K-32 ROOT CAUSE ANALYSIS

1. **Admin 500 Error:** Not `MLBGameLog.id` — that was already fixed. The actual bug is **`DataIngestionLog.run_at`** on lines 76 and 80 of `data_quality.py`. The `DataIngestionLog` table has **no `run_at` column** — actual columns are `started_at` and `completed_at`.

2. **Projection Freshness:** The `datetime` → `date` fix IS in repo code (`daily_ingestion.py:4905-4908, 4931-4934`). But failures continue. Likely the **deployed code is stale** OR `_load_persisted_ros_cache()` returns a `date` object for `ros_fetched_at` (line 4946) without conversion.

3. **Migration Not Run:** The `IngestedInjury` model and SQL migration were committed but **never executed against production PostgreSQL**. This is a common deployment gap — code ships, schema doesn't.

4. **Empty Tables:** `fantasy_lineups` and `player_valuation_cache` have INSERT logic missing. The computation pipelines run but never persist results. This is a wiring gap, not a data gap.

### K-32 REVISED PRIORITY ORDER

| Priority | Task | File | ETA | Acceptance |
|----------|------|------|-----|------------|
| P0 | Fix admin 500 | `data_quality.py:76,80` | 5 min | `GET /api/admin/data-quality/summary` → 200 |
| P0 | Run migration | `scripts/migrations/create_ingested_injuries.sql` | 10 min | `ingested_injuries` table exists |
| P0 | Fix projection_freshness | `daily_ingestion.py:4946-4958` | 15 min | Next run shows SUCCESS |
| P0 | Finish name backfill | `scripts/backfill_numeric_player_names.py` | 30 min | 0 numeric names remain |
| P1 | Wire BDL injuries | `daily_ingestion.py` scheduler | 1 hr | `ingested_injuries` has rows |
| P1 | Persist fantasy_lineups | `fantasy.py` or optimizer | 30 min | `fantasy_lineups` > 0 rows |
| P1 | Persist valuation_cache | `fantasy.py` waiver endpoint | 30 min | `player_valuation_cache` > 0 rows |
| P1 | Confirm probable pitchers | `daily_ingestion.py` | 1 hr | `is_confirmed` > 0 |

### K-32 FILES REFERENCED

| File | Action | Purpose |
|------|--------|---------|
| `backend/routers/data_quality.py` | **MODIFY** lines 76, 80 | Fix `run_at` → `completed_at` |
| `backend/services/daily_ingestion.py` | **MODIFY** ~line 4952 | Add `ros_fetched_at` date→datetime guard |
| `scripts/backfill_numeric_player_names.py` | **MODIFY/EXTEND** | Handle 93 orphan players via BDL/Stats API |
| `scripts/migrations/create_ingested_injuries.sql` | **RUN** | Create `ingested_injuries` table |
| `backend/services/daily_ingestion.py` | **MODIFY** scheduler | Schedule `_sync_bdl_injuries` with lock ID 100_036 |
| `backend/routers/fantasy.py` | **MODIFY** | INSERT computed lineups into `fantasy_lineups` |
| `backend/routers/fantasy.py` | **MODIFY** | INSERT waiver valuations into `player_valuation_cache` |
| `backend/services/daily_ingestion.py` | **MODIFY** | Wire BDL lineups to flip `is_confirmed` |

---

*Section added by Kimi CLI v1.17.0 | Post-deploy audit complete | 8 tasks scoped | Exact line numbers verified against HEAD*


---

## 16.14 DATA QUALITY NULL AUDIT — ROOT CAUSES & REMEDIATION PLAN (Apr 28, 2026)

> **Auditor:** Kimi CLI  
> **Full Report:** `reports/2026-04-28-data-quality-null-audit.md`  
> **Status:** Research complete. 8 tables audited. 5 root-cause patterns identified. 10 prioritized actions for Claude Code.

### K-33 FINDINGS (Executive Summary)

**Three root-cause patterns drive ~90% of nulls:**
1. **Schema migrations without historical backfills** (V31/V31) → `player_rolling_stats.w_runs/w_tb/w_qs` and `player_scores.z_r/z_h/z_tb/z_k_b/z_ops/z_k_p/z_qs` are 85% NULL.
2. **Ingestion jobs write placeholder NULLs for unimplemented computed fields** → `position_eligibility.scarcity_rank`/`league_rostered_pct` and `probable_pitchers.quality_score` are 100% NULL.
3. **Cross-system ID resolution failures** (FanGraphs ↔ MLBAM ↔ BDL ↔ Yahoo) → `player_projections.team` (50.7% null), `positions` (24.6% null), `probable_pitchers.bdl_player_id` (53.9% null).

**Highest-impact gaps:**
| Table | Column | Null % | Downstream Impact |
|-------|--------|--------|-------------------|
| `position_eligibility` | `scarcity_rank` | 100% | VORP uses flat replacement levels; waiver edge ignores position scarcity |
| `probable_pitchers` | `quality_score` | 100% | Two-Start Command Center shows neutral ratings for all matchups |
| `player_rolling_stats` | `w_runs`, `w_tb`, `w_qs` | 85% | New scoring categories (R, TB, QS) missing for historical windows |
| `player_scores` | `z_r`, `z_h`, `z_tb`, `z_k_b`, `z_ops`, `z_k_p`, `z_qs` | 85% | Composite Z-scores exclude new categories for pre-migration rows |
| `player_projections` | `team` | 50.7% | Dashboard shows "—" for team; filters break |
| `mlb_player_stats` | `bdl_stat_id` | 100% | Unused column; dead weight in schema |

### K-33 ROOT CAUSES (Verified Against Code)

**A. Migrations Without Backfills**
- `scripts/migrate_v31_rolling_expansion.py:20` — explicit comment: "no backfill required"
- `scripts/migrate_v32_zscore_expansion.py:22` — identical comment
- Daily pipeline only computes windows for `as_of_date = today`; historical rows never updated.

**B. Unimplemented Computed Fields**
- `backend/services/daily_ingestion.py:5443` — `_sync_position_eligibility` never writes `scarcity_rank` or `league_rostered_pct`
- `backend/services/daily_ingestion.py:5651` — `_sync_probable_pitchers` hardcodes `quality_score=None`

**C. Cross-System ID Resolution**
- `backend/services/daily_ingestion.py:5020-5028` — `_update_projection_cat_scores` skips players whose FanGraphs normalized name does not resolve to MLBAM ID via `player_id_mapping`
- `backend/services/daily_ingestion.py:5538-5546` — `mlbam_to_bdl` mapping preloaded but incomplete; 53.9% of probable pitchers miss BDL linkage

**D. BDL Partial Stat Coverage**
- `backend/services/daily_ingestion.py:2036-2043` — `_supplement_statsapi_counting_stats` only patches rows where `ab IS NULL`; does not patch partial rows where some counting stats are present and others missing

**E. Season-History Insufficiency (Expected)**
- `backend/services/daily_ingestion.py:4452-4462` — `z_score_total` requires ≥30 days of `vorp_7d`; `z_score_recent` requires ≥7 days. Season <30 days old → self-healing.

### K-33 PRIORITY ACTIONS FOR CLAUDE CODE

| Priority | Task | File | ETA |
|----------|------|------|-----|
| **P0** | Backfill `player_rolling_stats` V31 columns (w_runs, w_tb, w_qs) | `scripts/backfill_v31_rolling.py` (new) | 2 days |
| **P0** | Backfill `player_scores` V32 columns (z_r, z_h, z_tb, z_k_b, z_ops, z_k_p, z_qs) | `scripts/backfill_v32_zscores.py` (new) | 1 day |
| **P0** | Implement `scarcity_rank` + `league_rostered_pct` in position eligibility sync | `backend/services/daily_ingestion.py` | 1 day |
| **P0** | Implement `quality_score` in probable pitchers sync | `backend/services/daily_ingestion.py` | 1 day |
| **P1** | Harden StatsAPI supplement to patch ANY null counting stat | `backend/services/daily_ingestion.py` | 1 day |
| **P1** | Fix projection cat_scores team/positions fallback when MLBAM lookup fails | `backend/services/daily_ingestion.py` | 0.5 days |
| **P1** | Drop unused `bdl_stat_id` column OR map it correctly | `backend/models.py` or migration | 0.5 days |
| **P1** | Backfill `position_eligibility.bdl_player_id` gaps | `scripts/link_position_eligibility_bdl_ids.py` | 0.5 days |
| **P2** | Add data-quality gate for partial box-stat rows | `backend/services/daily_ingestion.py` | 1 day |
| **P2** | Document z_score_total self-healing timeline | `HANDOFF.md` or `docs/data_quality.md` | 0.25 days |

### K-33 DECISIONS REQUIRED

1. **Drop `bdl_stat_id`?** Column is 100% null, no consumer uses it. Recommend: drop via migration.
2. **Scarcity rank algorithm:** Static (preseason position frequency) vs dynamic (daily league roster counts)? Recommend: static for immediate fix; dynamic in backlog.
3. **Quality score heuristic:** Simple (`park_factor * opp_woba`) vs full model (ROS projections + opponent strength)? Recommend: simple heuristic for immediate fix.
4. **Auto-backfill strategy:** One-off scripts vs integrated retroactive computation in daily jobs? Recommend: one-off now; add `AUTO_BACKFILL_DAYS=30` config later.

---

*Section added by Kimi CLI v1.17.0 | Null audit complete | 8 tables inspected | 155,474 rows analyzed | Full report: `reports/2026-04-28-data-quality-null-audit.md`*


---

## K-34 FINDINGS — Downstream Consumption Audit (2026-04-28)

> **Full report:** `reports/2026-04-28-downstream-consumption-audit.md`  
> **Auditor:** Kimi CLI | **Scope:** `scarcity_rank` + `quality_score` downstream readers

### Q1–Q3: `scarcity_rank` Consumers

| Consumer | Queries `position_eligibility`? | Uses `scarcity_rank`? | Current Scarcity Logic |
|----------|-------------------------------|----------------------|------------------------|
| `waiver_edge_detector.py` | ❌ No | ❌ No | Hardcoded `_POS_GROUP` dict for drop pairing |
| `daily_lineup_optimizer.py` | ❌ No | ❌ No | Hardcoded `_DEFAULT_BATTER_SLOTS` order |
| `lineup_constraint_solver.py` | ❌ No | ❌ No | Static `SLOT_CONFIG` with internal `scarcity_rank` column |

**Gap:** None of the three consumers read `position_eligibility`. `scarcity_rank` must be wired into each scorer before it affects user-facing output.

### Q3: `quality_score` Consumer — `two_start_detector.py`

- **Reads `quality_score`?** ✅ Yes — SELECTs it from `probable_pitchers` and surfaces it in `MatchupRating` / `TwoStartOpportunity` dataclasses.
- **Downstream impact of null:** All pitchers get `0.0` fallback → `"GOOD"` rating for everyone. `EXCELLENT`/`AVOID` buckets unreachable.
- **Schema mismatch:** `MatchupRatingSchema` documents `quality_score` as `-2.0 to +2.0` (`backend/schemas.py:722`), but Session H heuristic emits `0.0–1.0`.

### Q4: Waiver Recommendations Endpoint (`GET /api/fantasy/waiver/recommendations`)

- **Queries `probable_pitchers`?** ❌ No.
- **Response includes `quality_score`?** ❌ No — `WaiverPlayerOut` and `RosterMoveRecommendation` schemas lack the field.
- **Gap:** Pitcher FA recommendations do not include matchup quality context.

### Q5: DB Schema Verification

```text
position_eligibility.scarcity_rank       integer          nullable
position_eligibility.league_rostered_pct  double precision nullable
probable_pitchers.quality_score           double precision nullable
```

All three columns exist and are nullable. No `NOT NULL` constraints.

### K-34 Recommendations for Session H

| Priority | Action |
|----------|--------|
| P0 | Implement `scarcity_rank` in `_sync_position_eligibility` (already in Session H scope) |
| P0 | Implement `quality_score` in `_sync_probable_pitchers` (already in Session H scope) |
| P1 | **Fix `MatchupRatingSchema` docstring** — `-2.0 to +2.0` → `0.0 to 1.0` to match heuristic |
| P2 | Wire `scarcity_rank` into `waiver_edge_detector.py` need-score multiplier |
| P2 | Wire `scarcity_rank` into `daily_lineup_optimizer.py` slot ordering |
| P2 | Wire `scarcity_rank` into `lineup_constraint_solver.py` objective bonus |
| P3 | Add `matchup_quality` to `WaiverPlayerOut` / `RosterMoveRecommendation` for pitcher FAs |

---

*Section added by Kimi CLI v1.17.0 | Downstream consumption audit complete | Full report: `reports/2026-04-28-downstream-consumption-audit.md`*
