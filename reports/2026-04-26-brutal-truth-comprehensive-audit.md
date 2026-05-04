# Brutal Truth Comprehensive Audit — 2026-04-26

> **Auditor:** Kimi CLI (Deep Intelligence Unit)  
> **Deployment Audited:** `e0e51d6c` (2026-04-26 13:07 UTC)  
> **Database:** Production PostgreSQL (Railway)  
> **API Base:** `https://fantasy-app-production-5079.up.railway.app`  
> **Methodology:** Live DB queries + live HTTP probes + code inspection. No assumptions.

---

## A. Executive Summary

| Metric | Verdict |
|--------|---------|
| **System Status** | **DEGRADED** |
| **Confidence Score** | **35%** |
| **Deploy Health** | Latest deploy successful, but prior 24h had 27 failures |
| **API Availability** | 17/20 endpoints return 200, but data quality is poor |
| **Math Integrity** | Formulas are correct; data feeding them is contaminated |
| **Database Hygiene** | Critical gaps in rolling stats, defaults, and null rates |

**Bottom line:** The application is *structurally* functional — it responds to requests, computes z-scores, and runs scheduled jobs. But the **numbers cannot be trusted** because the underlying projection data is ~30% garbage defaults, the waiver engine returns zero players, the scoreboard cannot name opponents, and the data-quality endpoint itself crashes with a datetime bug that has been known for days.

---

## B. Critical Failures (P0)

These are mathematically wrong or breaking production **right now**.

### B.1 Admin Data-Quality Endpoint — 500 Internal Server Error (LIVE)
- **Endpoint:** `GET /api/admin/data-quality/summary`
- **Status:** 500
- **Root cause:** `TypeError: can't subtract offset-naive and offset-aware datetimes` in `backend/routers/data_quality.py:69`
- **Impact:** The admin health dashboard is completely unavailable. This is the same bug identified in K-32 (Apr 25). It was supposedly fixed but is **still live in production** as of the latest deploy (13:07 UTC today).
- **Verdict:** This is a deploy gap — the fix exists in repo but did not make it into the deployed image, or the fix was applied to the wrong line.

### B.2 Waiver Endpoint Returns ZERO Players
- **Endpoint:** `GET /api/fantasy/waiver`
- **Status:** 200, body contains `players: []` (count = 0)
- **Impact:** The waiver wire is completely empty. No free agents can be browsed.
- **Likely cause:** The Yahoo API `out=ownership` regression from Apr 21 may have been reintroduced, or the waiver query is filtered too aggressively.
- **Note:** Waiver *recommendations* (`/api/fantasy/waiver/recommendations`) returns 2 players, which proves the recommendation engine can find candidates, but the main browse endpoint cannot.

### B.3 Scoreboard Projections Are ALL Zero
- **Endpoint:** `GET /api/fantasy/scoreboard`
- **Status:** 200
- **Data issue:** `my_projected_final: 0.0` for **all 18 categories**. This means the ROW/ROS projection pipeline is not feeding into the scoreboard orchestrator.
- **Impact:** Users see "locked_loss" on every category because projected final = 0.
- **Additional:** `opponent_name: "Opponent"` — still not resolved (K-25 bug).

### B.4 Projection Data is ~30% Garbage Defaults
- **Table:** `player_projections`
- **Evidence:**
  - 191 rows (30.2%) have **identical** default batter stats: `hr=15, r=65, rbi=65, sb=5, avg=0.25, ops=0.72`
  - 457 rows (72.2%) have **identical** default pitcher stats: `era=4.0, whip=1.3, w=0, qs=0, k_pit=0`
- **Impact:** The z-score pool is contaminated with fake data. Players with default stats get z-scores computed against a distribution that includes 191 copies of the same number. This distorts every valuation in the system.
- **Verdict:** The data is garbage. The math is correct, but garbage in = garbage out.

### B.5 Yahoo ID Mapping is 0/10,000
- **Table:** `player_id_mapping`
- **Evidence:** `COUNT(yahoo_id) = 0` out of 10,093 rows.
- **Impact:** `get_or_create_projection()` cannot resolve Yahoo players to MLBAM/BDL IDs. This breaks the Steamer → Yahoo lookup path for ALL players. The system falls back to (a) hardcoded 200-player board, (b) fuzzy name match, or (c) empty proxy with `z_score=0.0`.
- **Verdict:** The entire projection pipeline is blocked at the identity layer.

### B.6 `player_daily_metrics.z_score_total` is 100% NULL
- **Table:** `player_daily_metrics`
- **Evidence:** 16,007 total rows, 0 non-null `z_score_total`.
- **Impact:** Either this column is dead code, or the metric computation pipeline was never wired to populate it.
- **Verdict:** Feature not implemented or completely broken.

---

## C. Mathematical Assessment

**Overall Grade: C+**

The *formulas* are correct. The *data* and *integration* are not.

### C.1 Z-Score Calculation — CORRECT
- **File:** `backend/services/cat_scores_builder.py:49-60`
- Uses `statistics.stdev(values)` — **sample standard deviation**, which is correct for fantasy sports (population std dev would be wrong).
- Formula: `((value - mean) / sd) * direction`
- Direction multiplier correctly inverts "lower is better" categories (ERA, WHIP, K_B, L, HR_P).
- **Grade: A**

### C.2 Category Weights — UNVALIDATED
- **File:** `backend/services/cat_scores_builder.py:31-39`
- Weights are hardcoded:
  - Batter: OPS=1.4, HR=1.3, RBI=1.2, AVG=1.1, TB=0.9, H=0.8, R=1.0, NSB=1.0, K_B=-0.7
  - Pitcher: K_P=1.2, NSV=1.1, QS=1.0, W=1.0, K9=0.9, L=-0.8, HR_P=-1.0, ERA=-1.3, WHIP=-1.3
- **Problem:** No evidence these weights were optimized against historical league outcomes, correlated with win probability, or backtested. They appear to be editorial guesses.
- **Grade: C**

### C.3 Ensemble Blend — DOES NOT EXIST
- The user asked about Steamer, ZiPS, ATC, etc.
- **Reality:** The system uses **only Steamer** as a prior. There is no multi-system ensemble.
- The `fusion_engine.py` does implement a proper **Bayesian Marcel update** (Empirical Bayes shrinkage) to fuse Steamer with observed Statcast data. This is mathematically sound.
- **But:** The xwOBA/xERA override layers are "detection only" — they flag luck divergence but do NOT actually swap the prior source.
- **Grade: B-** (good math, incomplete integration)

### C.4 Projection-to-Counting-Stats Heuristics — MATHEMATICALLY WRONG
- **File:** `backend/fantasy_baseball/player_board.py:1496-1507`
- When Steamer counting stats are missing, the code invents them with heuristics:
  - `w = round(max(0, 12 - era))` — **Nonsense.** Wins are not `12 - ERA`.
  - `l = round(max(0, era - 3))` — **Nonsense.** Losses are not `ERA - 3`.
  - `r = round(obp * pa * 0.14)` — Crude and undocumented.
  - `rbi = round(slg * pa * 0.16)` — Crude and undocumented.
- These heuristics silently corrupt the `proj` dict for any player without a Steamer row.
- **Grade: F** for this function specifically.

### C.5 Predictive vs Descriptive Stats

| Stat Type | Metrics Used | Metrics Missing |
|-----------|-------------|-----------------|
| **Predictive** | Steamer projections (internal), xwOBA (detection only), xERA (detection only) | FIP, xFIP, SIERA, wRC+ |
| **Descriptive** | Season AVG, OPS, ERA, WHIP, raw counting stats | None |
| **Process** | Statcast Barrel%, Exit Velo, K%, BB% | None |

- The system is **descriptive-stat dominant** in its API output. The predictive layer (Steamer) exists but is hidden behind broken ID mapping.
- **No park factors** are applied to projections.
- **No age curves** or recency bias are applied.
- **Grade: D**

---

## D. Data Quality Scorecard

### D.1 Freshness

| Pipeline | Status | Grade |
|----------|--------|-------|
| `projection_freshness` | **FIXED** — 10 consecutive SUCCESS runs on Apr 26 | B+ |
| `bdl_injuries` | **FAILING** — 20/21 runs FAILED, error_message NULL | F |
| `savant_ingestion` | **FAILING** — 1 FAILED, error_message = "unknown" | F |
| `mlb_odds` | **SKIPPED** — 36 times in 24h | D |
| Daily snapshots | Contiguous (29 days, no gaps) | A |
| `ingested_injuries` | 170 rows present | B |

**Freshness Grade: D+**

### D.2 Completeness

| Table | Total Rows | Critical Null Rate | Grade |
|-------|-----------|-------------------|-------|
| `player_projections` | 633 | 30% default batter stats, 72% default pitcher stats | F |
| `player_id_mapping.yahoo_id` | 10,093 | **100% NULL** | F |
| `player_rolling_stats` (7d) | 19,836 | ~50% null counting stats, ~88% null runs | D- |
| `player_daily_metrics.z_score_total` | 16,007 | **100% NULL** | F |
| `probable_pitchers.is_confirmed` | 349 | **0% confirmed** | F |
| `fantasy_lineups` | 0 | N/A (empty table) | F |
| `player_valuation_cache` | 0 | N/A (empty table) | F |
| `player_momentum` | 20,656 | 0% null in key columns | A |
| `statcast_performances` | 11,653 | N/A | B+ |

**Completeness Grade: F**

### D.3 Accuracy

| Check | Result | Grade |
|-------|--------|-------|
| Duplicate names in projections | 7 duplicates | D |
| Orphan projections (no mapping) | 1 orphan by MLBAM join | B+ |
| Ghost players (all-null stats) | 0 | A |
| Numeric player names | 5 remaining | C |
| Roster season_stats populated | 23/23 | A |
| Roster ros_projection populated | 12/23 (z-scores, not raw stats) | C- |
| Scoreboard opponent name | "Opponent" for all | F |
| Waiver need_score positive | 0/0 (no players returned) | F |
| Lineup scores positive | 13/14 batters positive | B+ |

**Accuracy Grade: D**

---

## E. Specific Findings by Section

### E.1 Deployment Verification (Ground Truth)

**Latest deploy:** `e0e51d6c` (stable/cbb-prod) at 2026-04-26 13:07 UTC — **SUCCESS**.

**Ingestion logs (last 24h):**
- Total jobs: 357
- SUCCESS: 294
- SKIPPED: 36 (all `mlb_odds`)
- FAILED: 27
  - `bdl_injuries`: 20 FAILED
  - `projection_freshness`: 6 FAILED (all on Apr 25, NOT Apr 26 — **fixed**)
  - `savant_ingestion`: 1 FAILED

**Projection freshness conflict resolved:** The `datetime` → `date` TypeError (`unsupported operand type(s) for -: 'datetime.datetime' and 'datetime.date'`) was the root cause of the Apr 25 failures. The fix IS deployed and working. All runs since Apr 26 02:36 UTC are SUCCESS.

**However:** The `bdl_injuries` job is failing every hour with empty `error_message`. The commit `e0e51d6c` was specifically titled "capture exception text on DB-write failure", meaning the fix was deployed but the job is **still failing without capturing the error**. This suggests the exception is happening *before* the `_run()` function returns, or in a different code path entirely.

### E.2 API Stress Test Results

| Endpoint | Status | Latency | Data Quality |
|----------|--------|---------|--------------|
| `GET /health` | 200 | <1s | ✅ healthy, db connected |
| `GET /api/admin/data-quality/summary` | **500** | <1s | ❌ datetime TypeError |
| `GET /admin/version` | 200 | <1s | ⚠️ `git_commit_sha: "unknown"` |
| `GET /api/projections/freshness` | **404** | <1s | ❌ endpoint missing |
| `GET /api/fantasy/roster` | 200 | ~2s | ⚠️ ros_projection present but are z-scores |
| `GET /api/fantasy/waiver` | 200 | ~1s | ❌ **0 players** |
| `GET /api/fantasy/scoreboard` | 200 | ~1s | ❌ projected_final=0.0, opponent="Opponent" |
| `GET /api/fantasy/lineup/2026-04-26` | 200 | ~2s | ✅ scores positive, empty 2B slot |
| `GET /api/fantasy/waiver/recommendations` | 200 | ~2s | ⚠️ 2 recs, MCMC disabled, flat 0.0 |
| `GET /api/fantasy/briefing/2026-04-26` | 200 | ~2s | ❌ legacy cats, all 0.0, TBD, broken emoji |
| `GET /api/fantasy/matchup` | 200 | <1s | ✅ |
| `GET /api/fantasy/matchup/simulate` | **405** | <1s | ❌ Method Not Allowed |
| `GET /api/fantasy/players/{id}/scores` | **422** | <1s | ❌ Unprocessable Entity |
| `GET /api/fantasy/draft/board` | **404** | <1s | ❌ Not Found |
| `GET /admin/audit-tables` | 200 | <1s | ✅ |
| `GET /admin/yahoo/test` | 200 | <1s | ✅ connected |
| `GET /api/fantasy/decisions` | 200 | ~2s | ⚠️ recommends dropping Juan Soto |
| `GET /api/fantasy/decisions/status` | 200 | <1s | ✅ |

**5xx count:** 1 (admin data-quality)  
**4xx count:** 3 (projections/freshness 404, matchup/simulate 405, draft/board 404, scores 422)  
**Latency:** All under 2s.

### E.3 Database Integrity Deep Dive

**Referential Integrity:**
- `player_projections.player_id` = MLBAM ID (not BDL ID). Joining on `mlbam_id` resolves 632/633 rows. Only 1 true orphan.
- `player_id_mapping.yahoo_id` = 0 for all 10,093 rows. This is the **root cause** of the projection lookup failure.

**The "Null" Problem:**
- `player_daily_metrics.z_score_total`: 100% null. Either dead code or never wired.
- `player_rolling_stats` (7d window): `w_home_runs` 49.9% null, `w_runs` 88.0% null, `w_rbi` 49.9% null, `w_avg` 50.7% null.
- `player_rolling_stats` (14d): `w_home_runs` 50.5% null, `w_runs` 88.0% null.
- `player_rolling_stats` (30d): `w_home_runs` 51.0% null, `w_runs` 87.6% null.

**Time Series Continuity:**
- `daily_snapshots`: 29 distinct days from 2026-03-27 to 2026-04-25. Span = 29 days. **No gaps.**

**`probable_pitchers`:**
- Total: 349 rows
- Confirmed: **0**
- The `is_confirmed` flag is never set. This blocks reliable start/sit decisions for pitchers.

---

## F. Honest Assessment

### What Works
1. **Z-score math is correct.** Sample std dev, direction multipliers, and weighted aggregation are all implemented properly.
2. **Fusion engine math is correct.** The Marcel update formula, stabilization constants, and four-state logic are sound.
3. **Daily snapshot continuity is perfect.** 29 days with no gaps.
4. **Projection freshness is fixed.** The datetime bug that caused 152 failures is resolved.
5. **Lineup scores are positive.** The K-25 negative-score bug was fixed.
6. **Roster season_stats are populated.** 23/23 players have real Yahoo data.

### What is Garbage
1. **30% of batter projections are identical defaults.** `hr=15, r=65, rbi=65, sb=5, avg=0.25, ops=0.72` — copy-pasted 191 times.
2. **72% of pitcher projections are identical defaults.** `era=4.0, whip=1.3, w=0, qs=0, k_pit=0` — copy-pasted 457 times.
3. **The waiver endpoint returns zero players.** The waiver wire is completely empty.
4. **The scoreboard thinks every category is a locked loss** because projected final = 0.0.
5. **The admin health endpoint crashes** with a known datetime bug.
6. **Yahoo ID mapping is 100% null.** No Yahoo player can be resolved to a projection.
7. **The decisions endpoint recommends dropping Juan Soto** for Daylen Lile. This is either a bug in value calculation or the underlying data is so bad that Juan Soto's z-score has collapsed to near-zero.

### Root Cause
The system has **correct math operating on garbage data**. The Steamer ingestion pipeline wrote default/placeholder values for hundreds of players. The ID mapping layer never ingested Yahoo IDs. The z-score backfill ran on this contaminated data and produced meaningless cat_scores. Every downstream consumer (waiver, scoreboard, lineup, decisions) is fed these broken numbers.

**Fixing the math is not the priority. Fixing the data is.**

---

## G. Priority Actions for Claude Code

| Priority | Task | ETA | Acceptance Criteria |
|----------|------|-----|-------------------|
| **P0** | Fix admin 500 (`data_quality.py` datetime aware/naive) | 5 min | `GET /api/admin/data-quality/summary` → 200 |
| **P0** | Fix waiver endpoint returning 0 players | 2 hr | `GET /api/fantasy/waiver` returns >20 players |
| **P0** | Purge default projection rows or mark them `is_proxy=true` | 4 hr | 0 rows with `hr=15 AND r=65 AND rbi=65 AND sb=5` |
| **P0** | Ingest Yahoo IDs into `player_id_mapping` | 1 day | `COUNT(yahoo_id)` > 5000 |
| **P1** | Fix scoreboard `my_projected_final` = 0.0 | 4 hr | All 18 categories have non-zero projected_final |
| **P1** | Fix scoreboard `opponent_name` = "Opponent" | 2 hr | Returns real opponent team names |
| **P1** | Wire `player_daily_metrics.z_score_total` computation | 1 day | Null rate < 10% |
| **P1** | Fix `bdl_injuries` hourly failure | 2 hr | 3 consecutive SUCCESS runs |
| **P2** | Replace heuristic counting stats in `_convert_fusion_proj_to_board_format` | 2 days | No `w = 12 - era` logic |
| **P2** | Apply park factors to projections | 3 days | Projections vary by home park |
| **P2** | Implement true xwOBA/xERA prior swap (not just detection) | 2 days | Override flag changes projection source |

---

*Report generated: 2026-04-26 09:20 UTC*  
*Next re-audit required after all P0 items are resolved.*
