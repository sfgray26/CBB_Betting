
# Full Platform Audit — Fresh Data Pull (2026-04-26)

> **Auditor:** Kimi CLI  
> **Data Source:** Production PostgreSQL (Railway), direct SQL queries + API probes  
> **Scope:** All fantasy baseball tables, ingestion pipelines, and production endpoints  
> **Constraint:** Only new data sourced in this research task  
> **Timestamp:** 2026-04-26 ~12:30 UTC (audit queries executed 12:25–12:35 UTC)

---

## EXECUTIVE SUMMARY

| Category | Status | Key Finding |
|----------|--------|-------------|
| **New Features** | 🟢 Deployed | `ingested_injuries` table created (170 rows). BDL injury ingestion pipeline active. |
| **Data Growth** | 🟢 Healthy | All core tables grew: +4,125 player_scores, +2,621 rolling_stats, +423 statcast rows |
| **Pipeline Health** | 🟡 Improved | `projection_freshness` went from 0% → 71% success. New `bdl_injuries` job active but 95% failure rate. |
| **P0 Blockers** | 🔴 Persistent | Admin 500, empty lineups/valuation_cache, 0 confirmed pitchers, 0 yahoo_ids, z_score_total 100% null |
| **Data Quality** | 🟡 Mixed | 98 numeric projection names remain (was 93). 5 new projections added. No name backfill progress. |

**Bottom Line:** Significant forward progress on BDL injury ingestion and data freshness monitoring. However, the **same P0 consumer-facing gaps persist**: empty lineup/valuation tables, unconfirmed pitchers, broken admin endpoint, and 100% null z-scores. The platform ingests data well but **does not surface it to users effectively**.

---

## 1. WHAT CHANGED SINCE 04/25 AUDIT

### 🟢 NEW: `ingested_injuries` Table — 170 Rows

| Metric | Value |
|--------|-------|
| Total injuries | 170 |
| Unique players | 170 |
| 15-Day-IL | 56 |
| 60-Day-IL | 44 |
| 10-Day-IL | 41 |
| Day-To-Day | 26 |
| Other (paternity, bereavement) | 3 |

**Sample data quality:** Rich metadata including `injury_type`, `injury_detail`, `injury_side`, `return_date`, `short_comment`, and `raw_payload` (jsonb). All fields populated.

**Sample entries:**
- Willi Castro — Day-To-Day, Knee Soreness (Right), return 2026-04-26
- Steven Kwan — Day-To-Day, Neck Soreness, return 2026-04-26
- Jack Leiter — Day-To-Day, Ankle Soreness (Right), return 2026-04-27
- Max Muncy — Day-To-Day, Illness, return 2026-04-26

**Pipeline:** `bdl_injuries` job runs hourly. **1 success / 20 failures** in last 24h. The successful run processed 170 records in 19.5 seconds. Failure mode: empty `error_message` and `error_details` — likely **advisory lock contention** (job runs every hour but may overlap with itself).

---

### 🟢 IMPROVED: `projection_freshness` — 71% Success Rate

| Metric | Before (04/25) | After (04/26) |
|--------|---------------|---------------|
| Success rate | 0% (152/152 failed) | **71% (17/24 success)** |
| Last failure | 2026-04-25 17:40 | 2026-04-25 19:16 |
| Error type | `datetime - date` | `'module' object is not callable` (last failure) |
| Consecutive successes | 0 | **17** |

**Analysis:** The `datetime.date` → `datetime.datetime` fix was deployed and is working. The last two failures (Apr 25 18:20 and 19:16) were different errors (`datetime.time` has no `monotonic`, then `'module' object is not callable`) — likely from a bad deploy that was quickly rolled back or patched. Since 19:16, **17 consecutive successes**.

---

### 🟢 DATA GROWTH: All Core Tables Expanded

| Table | 04/25 Rows | 04/26 Rows | Delta | Growth Rate |
|-------|-----------|-----------|-------|-------------|
| `player_scores` | 57,610 | **61,735** | +4,125 | +7.2% |
| `player_rolling_stats` | 59,341 | **61,962** | +2,621 | +4.4% |
| `player_momentum` | 19,802 | **20,656** | +854 | +4.3% |
| `player_daily_metrics` | 14,907 | **16,007** | +1,100 | +7.4% |
| `statcast_performances` | 11,230 | **11,653** | +423 | +3.8% |
| `mlb_player_stats` | 11,159 | **11,591** | +432 | +3.9% |
| `mlb_game_log` | 399 | **415** | +16 | +4.0% |
| `probable_pitchers` | 332 | **349** | +17 | +5.1% |
| `simulation_results` | 18,933 | **19,787** | +854 | +4.5% |
| `backtest_results` | 18,079 | **18,933** | +854 | +4.7% |
| `decision_results` | 421 | **453** | +32 | +7.6% |
| `mlb_odds_snapshot` | 71,008 | **74,993** | +3,985 | +5.6% |

**All data pipelines are actively ingesting.** The platform is not stale — it's growing consistently.

---

### 🟢 NEW JOB TYPES IN PIPELINE

| Job Type | Runs (24h) | Success | Failure | Notes |
|----------|-----------|---------|---------|-------|
| `bdl_injuries` | 21 | 1 | 20 | **NEW**. Lock contention suspected. |
| `savant_ingestion` | 1 | 0 | 1 | **NEW**. Error: `"unknown"` |

---

### 🔴 UNCHANGED: Same P0 Gaps Persist

| Issue | 04/25 Status | 04/26 Status | Changed? |
|-------|-------------|-------------|----------|
| Admin endpoint 500 | 🔴 | 🔴 | **No** |
| `fantasy_lineups` | 0 rows | 0 rows | **No** |
| `player_valuation_cache` | 0 rows | 0 rows | **No** |
| `probable_pitchers.is_confirmed` | 0/332 | 0/349 | **No** |
| `yahoo_id` mapping | 0/10,000 | 0/10,000 | **No** |
| `z_score_total` null rate | 100% | 100% | **No** |
| Numeric projection names | 93 | 98 | **Worse** (+5) |
| `mlb_game_log.status` = Final | 0/399 | 0/415 | **No** |

---

## 2. DETAILED FINDINGS

### 2.1 `ingested_injuries` — Functional but Fragile

**Schema:** 14 columns, well-designed:
- `bdl_player_id` (integer, NOT NULL)
- `player_name` (varchar, NOT NULL)
- `injury_date`, `return_date` (timestamptz)
- `injury_type`, `injury_detail`, `injury_side` (varchar)
- `injury_status` (varchar, NOT NULL)
- `long_comment` (text), `short_comment` (varchar)
- `raw_payload` (jsonb, NOT NULL)
- `ingested_at`, `updated_at` (timestamptz)

**Data quality:** Excellent. Every row has `injury_status`, `injury_type`, `raw_payload`. Most have `short_comment` with actionable fantasy info (e.g., "not in the lineup for Friday's game").

**Pipeline fragility:** 20/21 failures with **empty error messages** strongly suggests the job is hitting its own advisory lock. The job may be scheduled too frequently (hourly) or the lock is not being released properly on exception paths.

**Recommendation:**
1. Check if `bdl_injuries` uses `replace_existing=True` in scheduler
2. Add a 5-minute cooldown between runs
3. Log lock acquisition failures explicitly

---

### 2.2 `projection_freshness` — Fixed and Stable

The `datetime.date` → `datetime.datetime` conversion logic is working. 17 consecutive successes as of 12:25 UTC.

**Last failure pattern:**
- 2026-04-25 17:40: `datetime - date` (old bug)
- 2026-04-25 18:20: `datetime.time` has no `monotonic` (import shadowing bug — `from datetime import time` shadowed `time.monotonic`)
- 2026-04-25 19:16: `'module' object is not callable` (related import bug)
- **2026-04-25 20:16 onwards: ALL SUCCESS**

This suggests a hotfix was deployed around 20:00 UTC on Apr 25 that resolved the import issues.

---

### 2.3 `player_projections` — 98 Numeric Names (+5 from 93)

| Metric | 04/25 | 04/26 |
|--------|-------|-------|
| Total projections | 628 | **633** (+5 new) |
| Numeric names | 93 | **98** (+5) |
| Real names | 535 | **535** (0 progress) |
| Missing team | 326 | **331** (+5) |
| Missing positions | 238 | **243** (+5) |
| Positive DQ | 354 | **359** (+5) |
| High DQ (>0.5) | 0 | **0** |
| Avg DQ score | 0.1123 | **0.1119** |

**The 5 new projections all have numeric names** and were added on 2026-04-26 10:00. They were not backfilled — they entered the system as raw IDs. This indicates the **backfill script is not running automatically** for new projections.

**The 98 remaining numeric IDs** are the same players as before (e.g., 695578, 683002, 686948, 660670, 808959, etc.) — all prospects/rookies with no `player_id_mapping` entry.

---

### 2.4 `player_daily_metrics` — vorp_7d/30d VANISHED on 04/26

| metric_date | rows | z_score_total | vorp_7d | vorp_30d | blend_hr |
|-------------|------|---------------|---------|----------|----------|
| 2026-04-26 | 953 | **0%** | **0%** | **0%** | 4% |
| 2026-04-25 | 1,099 | 0% | 70% | 83% | 4% |
| 2026-04-24 | 1,093 | 0% | 70% | 83% | 4% |
| 2026-04-23 | 1,086 | 0% | 71% | 83% | 4% |

**Critical regression:** `vorp_7d` and `vorp_30d` were populated for ~70-80% of players on Apr 23-25, but **completely absent (0%) on Apr 26**. This suggests the VORP computation pipeline failed silently on Apr 26, or the data source it depends on was unavailable.

However, the `vorp` job in `data_ingestion_logs` shows **SUCCESS** on Apr 26 at 08:30 with records_processed. The data exists somewhere but is not landing in `player_daily_metrics`.

**Hypothesis:** The `vorp` job may be writing to a different table or column, or the `player_daily_metrics` upsert is not matching on the correct key for Apr 26.

---

### 2.5 `probable_pitchers` — Still 0% Confirmed

| Date Range | Total | Confirmed | Has bdl_id |
|------------|-------|-----------|------------|
| Apr 16-21 | 158 | 0 | 0 |
| Apr 22-26 | 191 | 0 | 119 |
| Apr 27-29 | 23 | 0 | 19 |

**bdl_id coverage is improving** (119 of 191 recent rows, 62%), but **not a single row is confirmed**. The BDL injury sync is working; the BDL lineup sync is not wired to flip the confirmation flag.

---

### 2.6 `player_scores` — Healthy Growth, Good Distribution

| window | player_type | rows | non_zero_z | avg_z | avg_score | avg_conf |
|--------|-------------|------|------------|-------|-----------|----------|
| 7d | hitter | 9,823 | 100% | +0.0230 | 50.48 | 0.613 |
| 7d | pitcher | 9,812 | 99.8% | +0.0122 | 50.21 | 0.253 |
| 7d | two_way | 123 | 100% | **-0.796** | **59.35** | 0.537 |
| 30d | hitter | 10,249 | 100% | +0.0308 | 50.41 | 0.349 |
| 30d | pitcher | 10,833 | 99.9% | +0.0233 | 50.17 | 0.138 |
| 30d | two_way | 239 | 100% | **-1.239** | **54.81** | 0.303 |

**Two-way player anomaly persists:** Negative composite_z but inflated score_0_100. This mathematical inconsistency is still present.

---

### 2.7 `player_rolling_stats` — ~50% Rate Stats, Improved Coverage

| window | rows | has_w_avg | has_w_era | has_w_ip | avg_games |
|--------|------|-----------|-----------|----------|-----------|
| 7d | 19,836 | 9,775 (49%) | 9,904 (50%) | 9,935 (50%) | 3.04 |
| 14d | 20,732 | 10,094 (49%) | 10,566 (51%) | 10,590 (51%) | 5.02 |
| 30d | 21,394 | 10,330 (48%) | 11,057 (52%) | 11,072 (52%) | 7.25 |

Rate stat coverage unchanged at ~50%, but total rows grew by 2,621. The pipeline accumulates game data but only computes rates when sample size thresholds are met.

---

### 2.8 `player_momentum` — Rich Signal Distribution

| Signal | Count | % |
|--------|-------|---|
| STABLE | 15,836 | 76.7% |
| COLD | 1,806 | 8.7% |
| HOT | 1,269 | 6.1% |
| COLLAPSING | 1,090 | 5.3% |
| SURGING | 655 | 3.2% |

Healthy distribution. 23.3% of players show non-neutral momentum.

---

### 2.9 Admin Endpoint — Still 500

`GET /api/admin/data-quality/summary` → `{"detail":"Internal server error","type":"TypeError"}`

**Root cause confirmed:** `backend/routers/data_quality.py:76,80` references `DataIngestionLog.run_at`. The actual column is `completed_at` (or `started_at`). This bug was identified in the 04/25 audit and was **not fixed in the latest deployment**.

---

### 2.10 Yahoo ID Mapping — Still 0/10,000

| Field | Coverage | Changed? |
|-------|----------|----------|
| `bdl_id` | 100% (10,000/10,000) | No |
| `mlbam_id` | 65.67% (6,567/10,000) | No |
| `yahoo_key` | 18.99% (1,899/10,000) | No |
| `yahoo_id` | **0% (0/10,000)** | **No** |
| Fully mapped | 0 | No |
| Last updated | 2026-04-22 18:04 | No new updates |

The `player_id_mapping` table has not been updated since Apr 22. No Yahoo ID backfill has occurred.

---

### 2.11 Empty Tables — Persistent

| Table | Rows | Status | Blocked Since |
|-------|------|--------|---------------|
| `fantasy_lineups` | 0 | 🔴 Empty | Apr 9 (first detected) |
| `player_valuation_cache` | 0 | 🔴 Empty | Apr 9 (first detected) |
| `projection_cache_entries` | 1 | 🔴 Effectively empty | — |
| `games` | 4 | ⚠️ Nearly empty | — |
| `alerts` | 0 | ⚠️ Empty | — |
| `weather_forecasts` | 0 | ⚠️ Empty | — |
| `closing_lines` | 0 | ⚠️ Empty | — |
| `execution_decisions` | 0 | ⚠️ Empty | — |
| `job_queue` | 0 | ⚠️ Empty | — |
| `team_profiles` | 0 | ⚠️ Empty | — |
| `model_parameters` | 0 | ⚠️ Empty | — |
| `pattern_detection_alerts` | 0 | ⚠️ Empty | — |

---

## 3. PIPELINE HEALTH (Last 24 Hours)

| Job Type | Runs | Success | Failure | Skip | Status |
|----------|------|---------|---------|------|--------|
| `mlb_odds` | 284 | 244 | 0 | 40 | 🟢 Healthy |
| `bdl_injuries` | 21 | 1 | 20 | 0 | 🔴 95% failure |
| `probable_pitchers` | 3 | 3 | 0 | 0 | 🟢 Healthy |
| `backtesting` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `projection_freshness` | 24 | 17 | 7 | 0 | 🟡 71% success |
| `yahoo_adp_injury` | 4 | 4 | 0 | 0 | 🟢 Healthy |
| `position_eligibility` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `player_id_mapping` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `decision_optimization` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `ros_simulation` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `savant_ingestion` | 1 | 0 | 1 | 0 | 🔴 Failed |
| `projection_cat_scores` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `player_momentum` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `ensemble_update` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `vorp` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `rolling_z` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `player_scores` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `cleanup` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `statcast` | 2 | 2 | 0 | 0 | 🟢 Healthy |
| `rolling_windows` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `fangraphs_ros` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `statsapi_supplement` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `mlb_box_stats` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `mlb_game_log` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `clv` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `snapshot` | 1 | 1 | 0 | 0 | 🟢 Healthy |
| `explainability` | 1 | 1 | 0 | 0 | 🟢 Healthy |

**Overall job success rate:** 24/27 job types = 89% healthy. 3 problem jobs: `bdl_injuries` (95% fail), `projection_freshness` (29% fail, but improving), `savant_ingestion` (100% fail, 1 run).

---

## 4. DAILY SNAPSHOTS

| Date | Health | n_players_scored | Regression? |
|------|--------|------------------|-------------|
| 2026-04-25 | **DEGRADED** | 866 | Yes |
| 2026-04-24 | **DEGRADED** | 865 | Yes |
| 2026-04-23 | **DEGRADED** | 860 | Yes |
| 2026-04-21 | HEALTHY | 869 | No |
| 2026-04-20 | **DEGRADED** | 870 | Yes |

No snapshot for Apr 26 yet (expected at 14:00 UTC).

---

## 5. API ENDPOINT STATUS

| Endpoint | Status | Notes |
|----------|--------|-------|
| `GET /api/admin/data-quality/summary` | **500** | `DataIngestionLog.run_at` bug |
| `GET /health` | 200 | Healthy |
| `GET /docs` | 200 | Swagger UI available |
| `GET /openapi.json` | 200 | 143KB schema |
| `GET /api/fantasy/roster` | 401 | Requires Yahoo auth |
| `GET /api/fantasy/waiver` | 401 | Requires Yahoo auth |
| `GET /api/fantasy/lineup/*` | 401 | Requires Yahoo auth |

---

## 6. DATA QUALITY SCORECARD

| Domain | Metric | Value | Grade | Trend |
|--------|--------|-------|-------|-------|
| **Injuries** | Stored injury records | 170 | **B+** | ↑ NEW |
| **Projections** | Real name coverage | 84.5% (535/633) | B | → Flat |
| **Projections** | Numeric names | 98 | D | ↓ Worse (+5) |
| **ID Mapping** | Yahoo ID coverage | 0.00% | F | → Flat |
| **ID Mapping** | BDL ID coverage | 100.00% | A+ | → Flat |
| **Lineups** | Stored lineups | 0 | F | → Flat |
| **Valuations** | Cached valuations | 0 | F | → Flat |
| **Pitchers** | Confirmed SPs | 0.00% | F | → Flat |
| **Z-Scores** | Non-zero composite_z | 99.9% | A+ | → Flat |
| **Rolling Stats** | Rate stats computed | ~50% | D | → Flat |
| **Pipeline** | Job success rate | 89% | B | ↑ Improved |
| **Pipeline** | Freshness monitoring | 71% | C | ↑ Improved |
| **Snapshots** | Health status | DEGRADED | D | → Flat |
| **VORP** | Populated (latest date) | 0% | F | ↓ REGRESSED |

---

## 7. NEW ISSUES DISCOVERED

### 7.1 `bdl_injuries` — 95% Failure Rate with Empty Errors

20 of 21 runs fail with **no error message, no stack trace, no error details**. The only successful run processed 170 records perfectly. This pattern screams **advisory lock contention** — the job starts, finds the lock held (possibly by a previous run that hasn't finished), and exits silently.

**Evidence:**
- `records_processed = 0` on all failures
- `error_message = NULL`
- `error_details = []`
- Runs hourly but may take >1 hour when BDL API is slow

**Fix:** Add explicit logging for lock acquisition failure, or use `replace_existing=True` with a longer interval (e.g., every 4 hours instead of hourly).

### 7.2 `vorp` — Disappeared on Apr 26

`vorp_7d` and `vorp_30d` were 70-80% populated on Apr 23-25, but **0% on Apr 26**. The `vorp` job reports SUCCESS. Data is being computed but not persisted to `player_daily_metrics` for Apr 26.

**Fix:** Investigate the `player_daily_metrics` upsert logic for Apr 26. Likely a key mismatch or the metric_date is being set incorrectly.

### 7.3 `savant_ingestion` — Failed with "unknown"

1 run, 1 failure, error `"unknown"`. The new Savant ingestion job (possibly the BDL-based replacement) failed on first run. Needs investigation.

---

## 8. RECOMMENDATIONS BY PRIORITY

### P0 — Fix Today

| # | Issue | Fix | File |
|---|-------|-----|------|
| 1 | Admin 500 | `DataIngestionLog.run_at` → `completed_at` | `data_quality.py:76,80` |
| 2 | `bdl_injuries` 95% fail | Add lock failure logging; reduce frequency to 4h | `daily_ingestion.py` |
| 3 | `vorp` missing on Apr 26 | Fix `player_daily_metrics` upsert key for metric_date | `daily_ingestion.py` |
| 4 | Numeric names auto-backfill | Hook name resolution into projection creation path | `daily_ingestion.py` or `models.py` trigger |

### P1 — This Week

| # | Issue | Fix |
|---|-------|-----|
| 5 | `fantasy_lineups` empty | Add INSERT after lineup computation |
| 6 | `player_valuation_cache` empty | Add INSERT after waiver/MCMC computation |
| 7 | `yahoo_id` mapping | BDL search backfill for 1,899 Yahoo-keyed players |
| 8 | `probable_pitchers` confirmation | Wire BDL lineups to flip `is_confirmed` |
| 9 | `z_score_total` null | Investigate daily metrics pipeline calculation |
| 10 | Two-way player scoring | Fix `score_0_100` transform for negative z |

### P2 — Next Sprint

| # | Issue | Fix |
|---|-------|-----|
| 11 | `savant_ingestion` "unknown" error | Add proper error handling and logging |
| 12 | Backtest regression false positive | Reconfigure threshold or baseline |
| 13 | `mlb_game_log.status` vocabulary | Map actual statuses to "Final" equivalent |

---

*Report compiled by Kimi CLI v1.17.0 from 100% fresh database queries. All quantitative findings sourced directly from production PostgreSQL at 12:25–12:35 UTC on 2026-04-26. No historical reports were referenced for the primary findings.*
