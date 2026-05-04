# Brutal Truth Full-System Audit (K-34)

**Auditor:** Kimi CLI (Deep Context Specialist)  
**Timestamp:** 2026-04-27 (derived from production DB state)  
**Deployment:** Production Railway (`fantasy-app-production-5079.up.railway.app`)  
**Previous Audit:** K-33 (2026-04-26)  
**Integrity Scalar Framework:** `cbb-identity` (CONFIRMED/CAUTION/VOLATILE/ABORT)

---

## Executive Summary

**Confidence Score: 38%** (up from 35% in K-33, but for the wrong reasons)

**Brutal Truth Verdict:** The system is **partially operational** with **severe identity layer degradation**. While backend pipelines (Statcast, BDL injuries, odds) are now stable and several metrics have improved (MLBAM IDs 66%, probable pitchers 28% confirmed, VORP populating 1,689 records), the **Yahoo ID sync is actively failing** with database constraint violations. Without Yahoo IDs, the waiver engine, lineup optimizer, and decision pipeline cannot resolve roster players to projection rows — rendering the fantasy intelligence layer **functionally blind**.

The most dangerous finding: **651 out of 636 projection rows contain default/placeholder values** (191 batter defaults + 460 pitcher defaults, with ~15 rows fully defaulted in both dimensions). The z-score pool is **more contaminated than previously estimated** because overlapping default patterns were missed in prior audits.

---

## Audit Findings

### 1. Identity Layer (Integrity Scalar: ABORT)

| ID Type | Coverage | Status | Change from K-33 |
|---------|----------|--------|------------------|
| **Yahoo ID** | **0 / 10,096 (0%)** | 🔴 **ABORT** | Unchanged — sync job now failing |
| **MLBAM ID** | **6,663 / 10,096 (66%)** | 🟡 CAUTION | ✅ **+66 pp** massive improvement |
| **BDL ID** | **10,000 / 10,096 (99%)** | ✅ CONFIRMED | ✅ **+99 pp** massive improvement |

**Yahoo ID Sync — ACTIVE FAILURE (P0):**
```
Job: yahoo_id_sync
Status: FAILED (4 times in last 24h)
Error 1: UniqueViolation: duplicate key value violates unique constraint "_pim_bdl_id_uc"
       Key (bdl_id)=(1607) already exists.
Error 2: UniqueViolation: duplicate key value violates unique constraint "player_id_mapping_yahoo_key_key"
       Key (yahoo_key)=(469.p.60776) already exists.
Error 3: UniqueViolation: duplicate key value violates unique constraint "_pim_bdl_id_uc"
       Key (bdl_id)=(241414) already exists.
Error 4: UndefinedObject: constraint "_pim_yahoo_key_uc" for table "player_id_mapping" does not exist
```

**Root cause analysis:** The sync job uses `ON CONFLICT ON CONSTRAINT _pim_yahoo_key_uc DO UPDATE`, but that constraint **does not exist** in production. The table has `_pim_bdl_id_uc` and `player_id_mapping_yahoo_key_key` instead. The upsert logic is referencing the wrong constraint name, causing inserts to fail when duplicates are encountered rather than updating existing rows.

**Impact:**
- 0% Yahoo ID coverage means `get_or_create_projection()` cannot resolve ANY Yahoo roster player to a `PlayerProjection` row
- All Yahoo players receive population-prior fallback (`z_score=0.0`)
- Waiver recommendations use need_score with degraded inputs
- Lineup optimizer scores all players at baseline

---

### 2. Data Quality (Integrity Scalar: ABORT)

#### 2.1 Stub Projection Contamination

```sql
-- Batter defaults: 191 rows
WHERE hr = 15 AND r = 65 AND rbi = 65 AND sb = 5 AND avg = 0.25 AND ops = 0.72

-- Pitcher defaults: 460 rows
WHERE era = 4.0 AND whip = 1.3 AND w = 0 AND qs = 0 AND k_pit = 0

-- Total projection rows: 636
```

**Critical discovery:** 191 + 460 = 651, but total rows = 636. **~15 rows match BOTH patterns**, meaning they have default values for BOTH batter and pitcher stats simultaneously. These are "zombie rows" — fully defaulted across all columns.

**Contamination rate:** 651 / 636 = **102.4%** of rows have at least one default pattern (with overlap). In reality:
- 191 rows are batter-only defaults
- 460 rows are pitcher-only defaults
- ~15 rows are fully defaulted (both)

**All 636 rows have `cat_scores` populated.** The backfill scored the contaminated data, polluting the z-score distribution.

#### 2.2 `player_daily_metrics` — Empty for Today

```sql
SELECT COUNT(*) FROM player_daily_metrics WHERE metric_date = CURRENT_DATE;
-- Result: 0
```

The table has **0 rows for the current date**. This explains why:
- `z_score_total` is null for "today"
- `vorp_7d` appears null for "today"

However, the VORP job reports SUCCESS with `records_processed = 1,689`, suggesting it writes to `metric_date = CURRENT_DATE - 1` (intentional per prior Claude note). The rolling_z job processed 772 records. The player_scores job processed 2,628 records.

**This is not necessarily a bug** — but it means real-time daily metrics are not available for same-day analysis.

#### 2.3 `probable_pitchers` — Partially Improved

| Metric | K-33 Value | Current Value | Change |
|--------|-----------|---------------|--------|
| Total rows | 349 | 411 | +62 |
| `is_confirmed = true` | 0 (0%) | 115 (28%) | ✅ **+28 pp** |

**28% confirmation is operational progress** but still means 72% of probable pitchers are unconfirmed. The `b6a882a` fix is partially working — some pitchers now flip to confirmed, but the majority do not.

---

### 3. Pipeline Freshness

| Job | Last 24h Status | Records | Assessment |
|-----|-----------------|---------|------------|
| `projection_freshness` | SUCCESS (every hour) | 0 | ✅ Healthy |
| `bdl_injuries` | SUCCESS (every hour) | 163-168 | ✅ Healthy |
| `statcast` | SUCCESS | 814-866 | ✅ Healthy |
| `mlb_odds` | SUCCESS (~80%) / SKIPPED (~20%) | 29-96 | ⚠️ Advisory lock contention |
| `yahoo_id_sync` | **FAILED × 4** | 0-367 | 🔴 **Broken** |
| `vorp` | SUCCESS | 1,689 | ✅ **Fixed** |
| `rolling_z` | SUCCESS | 772 | ✅ Healthy |
| `player_scores` | SUCCESS | 2,628 | ✅ Healthy |
| `savant_ingestion` | FAILED × 1 | 0 | ⚠️ "unknown" error |
| `probable_pitchers` | SUCCESS | 82-84 | ✅ Healthy |

**Key improvements since K-33:**
- VORP: From 0% to 1,689 records processed ✅
- BDL injuries: Consistent SUCCESS with error capture working ✅
- Statcast: Regular SUCCESS (814-866 rows) ✅
- Probable pitchers: 0% → 28% confirmed ✅

**New degradations since K-33:**
- Yahoo ID sync: Now actively failing (was just 0% before; now failing with DB errors) 🔴
- Savant ingestion: Failed with "unknown" error ⚠️

---

### 4. Mathematical Soundness Assessment

| Component | Grade | Notes |
|-----------|-------|-------|
| Bayesian fusion engine (`fusion_engine.py`) | **A-** | Marcel formula correct; `PitcherCountingStatFormulas` added in `kimi/fix-pitcher-heuristics` branch replaces placeholder W/L/QS heuristics with research-backed math (Bill James Pythagorean, Mastersball methodology). **Not yet merged to production.** |
| Z-score computation (`cat_scores_builder.py`) | **A** | Sample stdev (N-1), correct direction multipliers. Operating on contaminated data pool. |
| Projection heuristics (production) | **F** | `w = 12 - era`, `l = era - 3` still live in production. Fix committed to feature branch only. |
| Yahoo ID → Projection resolution | **F** | 0% resolution rate. All players get population prior. |
| VORP computation | **B** | Now populating 1,689 records. Lagged by 1 day (intentional design). |
| Overall math grade | **C** | Correct formulas operating on garbage data + unresolved identity layer. |

---

## Critical Fix List

### P0 — System-Blocking

| # | Issue | Evidence | Fix |
|---|-------|----------|-----|
| 1 | **Yahoo ID sync DB constraint mismatch** | `UndefinedObject: constraint "_pim_yahoo_key_uc" does not exist`; `_pim_bdl_id_uc` violations | Rename constraint reference in sync job to match actual production constraint, or drop/recreate constraints to match code expectations |
| 2 | **0% Yahoo ID coverage** | 0/10,096 rows populated | Fix sync job + backfill all Yahoo IDs from roster/API |
| 3 | **651 contaminated projection rows** | 191 batter + 460 pitcher defaults | Flag/remove zombie rows; re-run cat_scores backfill after cleaning |

### P1 — Degraded Experience

| # | Issue | Evidence | Fix |
|---|-------|----------|-----|
| 4 | **Probable pitchers 72% unconfirmed** | 296/411 `is_confirmed = false` | Investigate why `bool(pitcher_data)` is falsy for majority; inspect upstream payload |
| 5 | **Savant ingestion failure** | `error_message = "unknown"` | Add proper error capture to savant ingestion path |
| 6 | **Placeholder W/L/QS in production** | `w = 12 - era` still in `player_board.py:1498` | Merge `kimi/fix-pitcher-heuristics` branch after Claude review |
| 7 | **player_daily_metrics empty for today** | 0 rows for `CURRENT_DATE` | Verify if intentional (yesterday-write pattern) or pipeline gap |

### P2 — Operational Debt

| # | Issue | Evidence | Fix |
|---|-------|----------|-----|
| 8 | `mlb_odds` 20% skipped rate | SKIPPED entries in ingestion logs | Advisory lock contention — may need lock ID redistribution |
| 9 | Admin 500 (timezone mismatch) | `TypeError: can't subtract offset-naive and offset-aware datetimes` | One-line fix in `data_quality.py` (still unmerged from K-33) |

---

## Remediation Proposals

### Delegation Bundle 1: Fix Yahoo ID Sync (P0)

```json
{
  "task_id": "20260427-fix-yahoo-id-sync",
  "assignee": "Claude Code",
  "branch": "claude/fix-yahoo-id-sync-constraints",
  "scope": [
    "backend/services/daily_ingestion.py",
    "backend/models.py"
  ],
  "goal": "Fix yahoo_id_sync unique constraint violations. The job references constraint '_pim_yahoo_key_uc' which does not exist. Determine correct constraint name and fix upsert logic.",
  "criteria": "1. yahoo_id_sync runs SUCCESS for 3 consecutive hours. 2. player_id_mapping.yahoo_id populated > 5000 rows. 3. No UniqueViolation errors in logs."
}
```

### Delegation Bundle 2: Purge Contaminated Projections (P0)

```json
{
  "task_id": "20260427-purge-stub-projections",
  "assignee": "Claude Code",
  "branch": "claude/purge-stub-projections",
  "scope": [
    "backend/services/cat_scores_builder.py",
    "scripts/migrations/"
  ],
  "goal": "Remove or flag 191 batter-default and 460 pitcher-default stub rows from player_projections. Recompute cat_scores for clean pool.",
  "criteria": "1. 0 rows match default batter pattern. 2. 0 rows match default pitcher pattern. 3. cat_scores backfill produces non-zero z-scores for remaining rows."
}
```

### Delegation Bundle 3: Review & Merge Pitcher Heuristics Fix (P1)

```json
{
  "task_id": "20260427-merge-pitcher-heuristics",
  "assignee": "Claude Code",
  "branch": "kimi/fix-pitcher-heuristics",
  "scope": [
    "backend/fantasy_baseball/player_board.py",
    "backend/fantasy_baseball/fusion_engine.py",
    "tests/test_player_board_fusion.py"
  ],
  "goal": "Differential review of kimi/fix-pitcher-heuristics branch. Merge to stable/cbb-prod if approved.",
  "criteria": "1. Code review complete. 2. All tests pass. 3. No regressions in waiver/lineup endpoints."
}
```

---

## Raw Evidence Log

### Database Queries Executed

```sql
-- Identity coverage
SELECT 'yahoo_id', COUNT(*), COUNT(yahoo_id), ROUND(COUNT(yahoo_id)*100.0/COUNT(*),1) FROM player_id_mapping;
-- Result: 10096 total, 0 populated, 0.0%

SELECT 'mlbam_id', COUNT(*), COUNT(mlbam_id), ROUND(COUNT(mlbam_id)*100.0/COUNT(*),1) FROM player_id_mapping;
-- Result: 10096 total, 6663 populated, 66.0%

SELECT 'bdl_id', COUNT(*), COUNT(bdl_id), ROUND(COUNT(bdl_id)*100.0/COUNT(*),1) FROM player_id_mapping;
-- Result: 10096 total, 10000 populated, 99.0%

-- Contamination
SELECT COUNT(*) FROM player_projections WHERE hr = 15 AND r = 65 AND rbi = 65 AND sb = 5 AND avg = 0.25 AND ops = 0.72;
-- Result: 191

SELECT COUNT(*) FROM player_projections WHERE era = 4.0 AND whip = 1.3 AND w = 0 AND qs = 0 AND k_pit = 0;
-- Result: 460

SELECT COUNT(*) FROM player_projections;
-- Result: 636

-- Probable pitchers
SELECT is_confirmed, COUNT(*) FROM probable_pitchers GROUP BY is_confirmed;
-- Result: false=296, true=115

-- Pipeline (last 24h failures only)
SELECT job_type, error_message FROM data_ingestion_logs
WHERE started_at >= NOW() - INTERVAL '24 hours' AND status = 'FAILED';
-- yahoo_id_sync: UniqueViolation (_pim_bdl_id_uc, player_id_mapping_yahoo_key_key, _pim_yahoo_key_uc missing)
-- savant_ingestion: "unknown"
```

---

*Audit complete. No sugarcoating applied.*
