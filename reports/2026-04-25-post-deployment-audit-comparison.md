# Post-Deployment Audit Comparison (2026-04-25)

> **Auditor:** Kimi CLI  
> **Baseline Audit:** `reports/2026-04-25-full-platform-audit-fresh-data.md` (07:25–08:00 UTC)  
> **Post-Deployment Audit:** 18:06 UTC same day  
> **Deployment Window:** Between baseline and post-deployment scans  
> **Scope:** Identify exactly what changed and what remained broken after Claude's deployment

---

## SUMMARY OF CHANGES

| Metric | Baseline | Post-Deploy | Delta | Verdict |
|--------|----------|-------------|-------|---------|
| **Projection numeric names** | 353/628 | **93/628** | **−260 fixed** | ✅ **FIXED** |
| **Projection real names** | 275/628 | **535/628** | **+260 resolved** | ✅ **FIXED** |
| **Yahoo ID mapping** | 0/10,000 | 0/10,000 | 0 | 🔴 **UNCHANGED** |
| **Fantasy lineups table** | 0 rows | 0 rows | 0 | 🔴 **UNCHANGED** |
| **Valuation cache** | 0 rows | 0 rows | 0 | 🔴 **UNCHANGED** |
| **Probable pitchers confirmed** | 0/332 | 0/332 | 0 | 🔴 **UNCHANGED** |
| **z_score_total null rate** | 100% | 100% | 0 | 🔴 **UNCHANGED** |
| **projection_freshness failures** | 152/152 | 152/152 | 0 | 🔴 **UNCHANGED** |
| **Admin endpoint** | 500 error | 500 error | 0 | 🔴 **UNCHANGED** |
| **Player scores** | 57,610 rows | 57,610 rows | 0 | ➖ **UNCHANGED** |
| **Rolling stats** | 59,341 rows | 59,341 rows | 0 | ➖ **UNCHANGED** |
| **Pipeline success rate** | 93.5% | 93.5% | 0 | ➖ **UNCHANGED** |

**Bottom Line:** Claude deployed **one fix** — the projection name backfill — and it was **successful for 260 of 353 numeric names (73.7% hit rate)**. The remaining 93 numeric names are players with no `player_id_mapping` entry. All other P0/P1 issues remain unaddressed.

---

## DETAILED FINDINGS

### ✅ WHAT WAS FIXED: Projection Name Backfill

**Before:** 353/628 projections had numeric-only names (MLBAM IDs stored as strings).

**After:** 93/628 projections have numeric-only names. **260 names successfully resolved.**

**Sample of fixed names:**

| player_id | Before | After | Team | Positions |
|-----------|--------|-------|------|-----------|
| 608324 | `608324` | **Alex Bregman** | CHC | 3B |
| 663538 | `663538` | **Nico Hoerner** | CHC | 2B |
| 665833 | `665833` | **Oneil Cruz** | PIT | OF |
| 621493 | `621493` | **Taylor Ward** | BAL | OF |
| 664023 | `664023` | **Ian Happ** | CHC | OF |
| 607043 | `607043` | **Brandon Nimmo** | TEX | OF |
| 687263 | `687263` | **Zach Neto** | LAA | SS |
| 621566 | `621566` | **Matt Olson** | ATL | 1B |
| 682928 | `682928` | **CJ Abrams** | WSH | SS |
| 545361 | `545361` | **Mike Trout** | LAA | OF |

**How the fix worked:** Cross-referenced `player_id_mapping` via `bdl_id` → `full_name` and updated `player_projections.player_name`.

---

### 🔴 WHAT REMAINS BROKEN: P0 Issues

#### 1. `player_id_mapping.yahoo_id`: Still 0/10,000 (0%)

| Field | Value | Changed? |
|-------|-------|----------|
| Total rows | 10,000 | No |
| yahoo_id filled | **0** | No |
| yahoo_key filled | 1,899 | No |
| mlbam_id filled | 6,567 | No |
| bdl_id filled | 10,000 | No |
| fully mapped (all three) | 0 | No |
| Last updated | 2026-04-22 18:04 | No new updates |

**Impact:** Zero progress on the most critical mapping gap. The 1,899 Yahoo roster players still have no `yahoo_id`, breaking Yahoo API → projection lookup chains.

**Why it wasn't fixed:** This requires live BDL API calls (`GET /mlb/v1/players?search=`) to resolve Yahoo player names → BDL IDs. The fix may have been skipped or deprioritized.

---

#### 2. `fantasy_lineups`: Still 0 Rows

No INSERT logic was added to the lineup optimizer. Table remains completely empty.

---

#### 3. `player_valuation_cache`: Still 0 Rows

No INSERT logic was added to the waiver/MCMC pipeline. Table remains completely empty.

---

#### 4. `probable_pitchers.is_confirmed`: Still 0/332 (0%)

| Date Range | Total | Confirmed | Has bdl_id |
|------------|-------|-----------|------------|
| All dates | 332 | **0** | 119 |
| Last update | — | — | 2026-04-25 12:30 |

No BDL lineup confirmation logic was wired in.

---

#### 5. `player_daily_metrics.z_score_total`: Still 100% Null

| metric_date | rows | has_z_total | Changed? |
|-------------|------|-------------|----------|
| 2026-04-25 | 952 | **0 (0%)** | No |
| 2026-04-24 | 1,093 | **0 (0%)** | No |
| 2026-04-23 | 1,086 | **0 (0%)** | No |

The calculation pipeline that produces `z_score_total` was not fixed.

---

#### 6. `projection_freshness`: Still 152 Consecutive Failures (100%)

| Metric | Value | Changed? |
|--------|-------|----------|
| Total runs (7d) | 152 | +0 (no new runs since baseline) |
| Successes | 0 | No |
| Failures | 152 | No |
| Last failure | 2026-04-25 17:40 | Most recent scheduled run still fails |
| Error | `datetime - date` TypeError | No |

The 15-minute datetime type fix was **not deployed**.

---

#### 7. Admin Endpoint: Still 500

`GET /api/admin/data-quality/summary` → `{"detail":"Internal server error","type":"TypeError"}`

The `MLBGameLog.id` → `MLBGameLog.game_id` fix was **not deployed**.

---

### 🔴 REMAINING NUMERIC NAMES: The 93 Holdouts

All 93 remaining numeric-name projections share these traits:

- `team = NULL`
- `positions = NULL`
- `data_quality_score` between 0.10 and 0.34
- `player_name` = raw MLBAM ID string

**Sample of remaining unfixed IDs:**

| player_id | player_name | team | positions | DQ Score |
|-----------|-------------|------|-----------|----------|
| 695578 | `695578` | NULL | NULL | 0.34 |
| 683002 | `683002` | NULL | NULL | 0.315 |
| 686948 | `686948` | NULL | NULL | 0.305 |
| 660670 | `660670` | NULL | NULL | 0.30 |
| 670541 | `670541` | NULL | NULL | 0.295 |
| 808959 | `808959` | NULL | NULL | 0.285 |
| 802415 | `802415` | NULL | NULL | 0.27 |
| 805779 | `805779` | NULL | NULL | 0.27 |

**Root cause:** These players have **no entry in `player_id_mapping`** or their `bdl_id` doesn't resolve to a name in the mapping table. Many IDs in the 800xxx range are likely **prospects or 2026 rookies** not yet in the BDL mapping database.

**Recommendation:** For these 93, use BDL `GET /mlb/v1/players?search=` API calls to resolve names by MLBAM ID, or fall back to MLB Stats API name lookup.

---

## PIPELINE HEALTH: NO CHANGE

All ingestion jobs continue on their existing schedules. No new job types were added. No new tables were created.

| Job Type | Status | Records (24h) | Notes |
|----------|--------|---------------|-------|
| `mlb_odds` | ✅ SUCCESS/SKIPPED | 171 / 114 | CBB odds ingestion |
| `projection_freshness` | 🔴 FAILED (23x) | 0 | Same datetime bug |
| `snapshot` | ✅ SUCCESS | 1 | Daily snapshot |
| `explainability` | ✅ SUCCESS | 1 | Decision explanations |
| `probable_pitchers` | ✅ SUCCESS | 3 | Still not confirming |
| `backtesting` | ✅ SUCCESS | 1 | Regression detected |
| `yahoo_adp_injury` | ✅ SUCCESS | 5 | 27 injury flags/run |
| `position_eligibility` | ✅ SUCCESS | 1 | 2,389 rows |
| `player_id_mapping` | ✅ SUCCESS | 1 | No new yahoo_ids |
| `decision_optimization` | ✅ SUCCESS | 1 | Waiver decisions |
| `ros_simulation` | ✅ SUCCESS | 1 | ROS sims |
| `projection_cat_scores` | ✅ SUCCESS | 1 | Cat scores |
| `player_momentum` | ✅ SUCCESS | 1 | Momentum signals |
| `ensemble_update` | ✅ SUCCESS | 1 | Ensemble model |
| `vorp` | ✅ SUCCESS | 1 | VORP calc |
| `rolling_z` | ✅ SUCCESS | 1 | Rolling z-scores |
| `player_scores` | ✅ SUCCESS | 1 | Score computation |
| `cleanup` | ✅ SUCCESS | 1 | Cleanup job |
| `statcast` | ✅ SUCCESS | 3 | Savant ingestion |
| `rolling_windows` | ✅ SUCCESS | 1 | Rolling windows |
| `fangraphs_ros` | ✅ SUCCESS | 1 | FanGraphs ROS |
| `statsapi_supplement` | ✅ SUCCESS | 1 | MLB Stats API |
| `mlb_box_stats` | ✅ SUCCESS | 1 | Box scores |
| `mlb_game_log` | ✅ SUCCESS | 1 | Game logs |
| `clv` | ✅ SUCCESS | 1 | Closing line value |

---

## DAILY SNAPSHOTS: STILL DEGRADED

| Date | Health | n_players_scored | Regression? |
|------|--------|------------------|-------------|
| 2026-04-25 | **DEGRADED** | 866 | Yes |
| 2026-04-24 | **DEGRADED** | 865 | Yes |
| 2026-04-23 | **DEGRADED** | 860 | Yes |
| 2026-04-21 | HEALTHY | 869 | No |
| 2026-04-20 | **DEGRADED** | 870 | Yes |
| 2026-04-19 | **DEGRADED** | 873 | Yes |
| 2026-04-18 | **DEGRADED** | 867 | Yes |

The backtesting regression flag was not addressed.

---

## SCORECARD: BEFORE vs AFTER

| Domain | Metric | Before | After | Grade Change |
|--------|--------|--------|-------|--------------|
| **Projections** | Real name coverage | 43.8% | **85.2%** | F → **B** |
| **Projections** | Numeric name count | 353 | **93** | — |
| **ID Mapping** | Yahoo ID coverage | 0.0% | 0.0% | F → F |
| **ID Mapping** | BDL ID coverage | 100.0% | 100.0% | A+ → A+ |
| **Lineups** | Stored lineups | 0 | 0 | F → F |
| **Valuations** | Cached valuations | 0 | 0 | F → F |
| **Pitchers** | Confirmed SPs | 0.0% | 0.0% | F → F |
| **Injuries** | Stored records | 0 | 0 | F → F |
| **Pipeline** | Freshness monitoring | 0.0% | 0.0% | F → F |
| **Z-Scores** | Non-zero composite_z | 99.9% | 99.9% | A+ → A+ |
| **Rolling Stats** | Rate stats computed | ~50% | ~50% | D → D |
| **Pipeline** | Job success rate | 93.5% | 93.5% | B → B |

---

## RECOMMENDATIONS

### Immediate (Next Deployment)

1. **Finish the name backfill** — Resolve the remaining 93 numeric names via BDL API search or MLB Stats API lookup
2. **Fix `projection_freshness`** — Cast `datetime` to `date` before subtraction (literally a 1-line fix)
3. **Fix admin 500** — Change `MLBGameLog.id` to `MLBGameLog.game_id` (1-line fix)

### This Week

4. **Populate `fantasy_lineups`** — Add INSERT after lineup optimization computes results
5. **Populate `player_valuation_cache`** — Add INSERT after MCMC/waiver computation
6. **Fix `z_score_total`** — Investigate why daily metrics pipeline never computes this field
7. **Backfill `yahoo_id`** — Use BDL search to map the 1,899 Yahoo-keyed players

### Next Sprint

8. **Confirm probable pitchers** — Wire BDL lineups to flip `is_confirmed`
9. **Create `player_injuries` table** — Persist Yahoo + BDL injury data
10. **Address backtest regression** — Determine if DEGRADED flag is false positive

---

*Comparison compiled by Kimi CLI v1.17.0. All data sourced fresh from production PostgreSQL at 18:06 UTC. No historical reports were referenced for the quantitative comparisons — only the prior audit's raw numbers were used as the baseline.*
