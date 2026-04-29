# Data Quality Null Audit — Player Stats & Projections

**Date:** 2026-04-28  
**Auditor:** Kimi CLI  
**Full Report:** `reports/2026-04-28-data-quality-null-audit.md`  
**Scope:** `player_projections`, `mlb_player_stats`, `player_daily_metrics`, `player_rolling_stats`, `player_scores`, `position_eligibility`, `probable_pitchers`, `statcast_performances`

---

## Executive Summary

- **8 tables audited** covering 155,474 total rows. **6 tables have critical null gaps** that degrade or disable downstream fantasy features.
- **Three root-cause patterns** drive ~90% of the nulls: (1) schema migrations without historical backfills, (2) ingestion jobs that write placeholder NULLs for unimplemented computed fields, and (3) cross-system ID resolution failures (Yahoo ↔ BDL ↔ MLBAM) that block population of `team`, `positions`, and `bdl_player_id`.
- **Highest impact:** `position_eligibility.scarcity_rank` and `league_rostered_pct` are 100% null, disabling scarcity-based VORP and waiver recommendations. `probable_pitchers.quality_score` is 100% null, breaking matchup quality ratings in the Two-Start Command Center.
- **Most fixable:** `player_projections.team` (311 nulls) and `positions` (151 nulls) can be patched with a single backfill from FanGraphs RoS cache. `mlb_player_stats` counting-stat gaps can be closed by hardening the MLB Stats API supplement job.
- **Season-age effect:** `player_daily_metrics.z_score_total` is 100% null because the season is <30 days old and the rolling-Z job requires 30 days of VORP history. This is expected and will self-heal.

---

## Findings

### 1. `player_projections` — 613 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `team` | 311 | 50.7% | ⚠️ P1 | `_update_projection_cat_scores` only upserts players with resolved `mlbam_id`; unmatched players keep NULL team |
| `positions` | 151 | 24.6% | ⚠️ P1 | Same root cause: FanGraphs → MLBAM ID lookup fails for ~25% of blend rows |
| `cat_scores` | 0 | 0% | ✅ | Fully populated by ensemble blend job (line 5039, `daily_ingestion.py`) |
| All stat columns (hr, r, rbi, sb, era, whip, etc.) | 0 | 0% | ✅ | Default values enforced by SQLAlchemy model (`default=15`, `default=4.00`, etc.) |

**Code reference:** `backend/services/daily_ingestion.py:5018-5081` — `_update_projection_cat_scores` builds `name_to_mlbam` from `PlayerIDMapping` and skips any player whose normalized name does not resolve to an MLBAM ID. No fallback to raw FanGraphs player ID.

---

### 2. `mlb_player_stats` — 12,297 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `bdl_stat_id` | **12,297** | **100%** | ⚠️ P2 | BDL API `/stats` endpoint does not expose a stable per-row `id` field; `stat.id` in the Pydantic contract is likely None or unmapped |
| `ab`, `runs`, `hits`, `doubles`, `triples`, `strikeouts_bat`, `stolen_bases` | 3,973 | 32.3% | ⚠️ P1 | BDL returns NULL for counting stats on many pitcher rows and partial batting rows |
| `home_runs`, `rbi`, `walks` | 3,548 | 28.9% | ⚠️ P1 | Same BDL partial-population issue |
| `avg`, `obp`, `slg`, `ops` | 3,549 | 28.9% | ⚠️ P1 | `ops` is computed in ingestion; `avg`/`obp`/`slg` come from BDL and are sometimes missing |
| `innings_pitched`, `earned_runs` | 8,722 | 70.9% | ⚠️ P2 | Expected: pure hitters do not pitch |
| `hits_allowed`, `runs_allowed`, `walks_allowed` | 8,920 | 72.5% | ⚠️ P2 | Expected: pure hitters do not pitch |
| `whip` | 10,432 | 84.8% | ⚠️ P2 | Computed only when `bb_allowed`, `h_allowed`, and `ip` are all present; many rows fail this guard |
| `era` | 8,726 | 70.9% | ⚠️ P2 | Computed only when `er` and `ip` present; also missing for hitters |
| `strikeouts_pit` | 185 | 1.5% | ✅ | BDL reliably provides this even for hitters (sometimes 0) |
| `game_id` | 0 | 0% | ✅ | FK populated from `mlb_game_log` |
| `caught_stealing` | 0 | 0% | ✅ | Backfilled to `0` when BDL omits it (line 1546, `daily_ingestion.py`) |

**Code reference:** `backend/services/daily_ingestion.py:1999-2158` — `_supplement_statsapi_counting_stats` patches NULL counting stats from MLB Stats API, but only for rows where `ab IS NULL`. It does not patch rows where `ab` is present but `runs` or `doubles` are missing, leaving partial stat rows in the table.

---

### 3. `player_daily_metrics` — 18,229 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `z_score_total` | **18,229** | **100%** | ⚠️ P2 | `_calc_rolling_zscores` requires ≥30 days of `vorp_7d` history; season started <30 days ago |
| `blast_pct`, `bat_speed`, `squared_up_pct`, `swing_length`, `stuff_plus`, `plv` | **18,229** | **100%** | ⚠️ P3 | Model docstring: "Statcast 2.0 (MLB only — always NULL for CBB rows)"; no ingestion job writes these |
| `blend_hr`, `blend_rbi`, `blend_avg` | 17,299 | 94.9% | ❌ P0 | `_update_ensemble_blend` only writes blend data for players whose `player_id` appears in FanGraphs RoS DataFrames; most Yahoo/BDL players do not match |
| `blend_era`, `blend_whip` | 17,255 | 94.7% | ❌ P0 | Same as above |
| `z_score_recent` | 12,153 | 66.7% | ⚠️ P2 | Requires ≥7 days of `vorp_7d`; many players have <7 days of history |
| `vorp_7d` | 7,256 | 39.8% | ⚠️ P2 | Computed only for players with position data and 7d player_scores; gaps when Yahoo sync fails |
| `vorp_30d` | 5,728 | 31.4% | ⚠️ P2 | Same as above but for 30d window |
| `data_source` | 7,947 | 43.6% | ⚠️ P2 | Rows inserted without explicit `data_source` (e.g., rolling z-score inserts set it, but VORP upserts do not always) |

**Code reference:** `backend/services/daily_ingestion.py:4717-4888` — `_update_ensemble_blend` extracts blend rows from FanGraphs DataFrames keyed by `player_id` (a normalized name string). If the DataFrame uses a different ID scheme than the database rows, the blend values never land.

---

### 4. `player_rolling_stats` — 67,232 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `w_runs` | 57,169 | 85.0% | ❌ P0 | Added in V31 migration with **no backfill**; only new rows computed after migration have it |
| `w_tb` | 57,169 | 85.0% | ❌ P0 | Same V31 migration issue |
| `w_qs` | 56,164 | 83.5% | ❌ P0 | Same V31 migration issue |
| `w_caught_stealing` | 49,692 | 73.9% | ⚠️ P1 | Added in P27 migration; partial backfill existed but historical rows remain NULL |
| `w_net_stolen_bases` | 49,692 | 73.9% | ⚠️ P1 | Same P27 issue |
| `w_avg`, `w_slg`, `w_ops` | ~34,500 | ~51% | ⚠️ P2 | Expected: pitchers have no batting data; but also NULL when `w_ab == 0` |
| `w_obp` | 34,424 | 51.2% | ⚠️ P2 | Same as above |
| `w_ab` through `w_stolen_bases` (batting sums) | ~34,000 | ~50.6% | ⚠️ P2 | Expected pitcher/hitter split |
| `w_ip` through `w_strikeouts_pit` (pitching sums) | ~32,800 | ~48.9% | ⚠️ P2 | Expected pitcher/hitter split |
| `w_era`, `w_whip`, `w_k_per_9` | ~32,900 | ~49% | ⚠️ P2 | Expected pitcher/hitter split |
| `w_games` | 0 | 0% | ✅ | Always populated (M3 fix) |

**Code reference:** `scripts/migrate_v31_rolling_expansion.py:20` — explicit design note: "Nullable additive columns only -- zero downtime, no backfill required." This decision leaves 85% of rows permanently NULL for the new columns.

---

### 5. `player_scores` — 66,981 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `z_r` | 56,918 | 85.0% | ❌ P0 | Added in V32 migration with **no backfill** |
| `z_h` | 56,918 | 85.0% | ❌ P0 | Same V32 migration issue |
| `z_tb` | 56,918 | 85.0% | ❌ P0 | Same V32 migration issue |
| `z_k_b` | 56,918 | 85.0% | ❌ P0 | Same V32 migration issue |
| `z_ops` | 57,065 | 85.2% | ❌ P0 | Same V32 migration issue |
| `z_k_p` | 55,913 | 83.5% | ❌ P0 | Same V32 migration issue |
| `z_qs` | 55,913 | 83.5% | ❌ P0 | Same V32 migration issue |
| `z_nsb` | 49,431 | 73.8% | ⚠️ P1 | Added in P27; partial backfill |
| `z_avg`, `z_hr`, `z_rbi`, `z_sb` | ~33,700 | ~50.4% | ⚠️ P2 | Expected: pitcher rows have no batting Z-scores |
| `z_obp` | 34,163 | 51.0% | ⚠️ P2 | Same as above |
| `z_era`, `z_whip`, `z_k_per_9` | ~32,675 | ~48.8% | ⚠️ P2 | Expected: hitter rows have no pitching Z-scores |
| `composite_z`, `score_0_100`, `confidence` | 0 | 0% | ✅ | Always computed and written |

**Code reference:** `scripts/migrate_v32_zscore_expansion.py:22` — same "no backfill required" design note as V31.

---

### 6. `position_eligibility` — 2,389 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `scarcity_rank` | **2,389** | **100%** | ❌ P0 | `_sync_position_eligibility` never writes this field |
| `league_rostered_pct` | **2,389** | **100%** | ❌ P0 | `_sync_position_eligibility` never writes this field |
| `bdl_player_id` | 375 | 15.7% | ⚠️ P1 | Yahoo sync upserts `bdl_player_id=None`; backfilled later by `scripts/link_position_eligibility_bdl_ids.py` but gaps remain |
| `primary_position` | 0 | 0% | ✅ | Always populated from Yahoo positions list |
| `player_name` | 0 | 0% | ✅ | Always populated |

**Code reference:** `backend/services/daily_ingestion.py:5343-5494` — `_sync_position_eligibility` builds boolean flags, primary position, and player type, but `scarcity_rank` and `league_rostered_pct` are omitted from both the INSERT and the ON CONFLICT UPDATE set.

---

### 7. `probable_pitchers` — 430 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| `quality_score` | **430** | **100%** | ❌ P0 | Explicitly hardcoded to `None` in `_sync_probable_pitchers` (line 5651) |
| `bdl_player_id` | 232 | 53.9% | ⚠️ P1 | MLBAM → BDL ID mapping misses when `player_id_mapping` lacks the MLBAM ID |
| `opponent`, `is_home`, `pitcher_name`, `mlbam_id`, `game_time_et`, `park_factor` | 0 | 0% | ✅ | Populated from MLB Stats API |

**Code reference:** `backend/services/daily_ingestion.py:5640-5653` — `quality_score=None` is passed directly into the upsert. No computation logic exists in the job.

---

### 8. `statcast_performances` — 12,323 rows

| Column | Nulls | % Null | Severity | Evidence |
|--------|-------|--------|----------|----------|
| **All columns** | **0** | **0%** | ✅ | Statcast ingestion pipeline (`statcast_ingestion.py`) populates every column with either real data or `0` defaults |

---

## Root Cause Analysis

### Root Cause A: Migrations Without Backfills (P0)

**Tables affected:** `player_rolling_stats`, `player_scores`

**Why it happens:** The V31 and V32 migrations added columns to support new scoring categories (Runs, Total Bases, Quality Starts, etc.). The migration scripts explicitly state "no backfill required" for zero-downtime deployment. However, the daily pipeline only computes windows for the current date; it does not retroactively recompute historical rolling stats. Consequently, all rows created before the migration date remain permanently NULL for the new columns.

**Impact:** The dashboard, waiver edge detector, and lineup optimizer that query `player_scores` for the latest date see NULL for the new category Z-scores. If any consumer tries to use `z_r`, `z_tb`, or `z_qs` in a composite calculation, it silently drops those players from ranking or produces incorrect composites.

**Evidence:**
- `scripts/migrate_v31_rolling_expansion.py:20` — "nullable additive columns only -- zero downtime, no backfill required"
- `scripts/migrate_v32_zscore_expansion.py:22` — identical comment

---

### Root Cause B: Unimplemented Computed Fields (P0)

**Tables affected:** `position_eligibility`, `probable_pitchers`

**Why it happens:** The schema was designed with placeholder columns for future features (`scarcity_rank`, `league_rostered_pct`, `quality_score`). The ingestion jobs that write to these tables were never updated to compute and populate these values. In `probable_pitchers`, the developer explicitly passed `None` rather than implementing a matchup quality algorithm.

**Impact:**
- `position_eligibility`: VORP engine (`_compute_vorp`) loads position data but cannot compute replacement-level Z-scores by position because `scarcity_rank` is missing. Waiver recommendations that should prioritize scarce positions (C, SS, CF) are instead treated as flat.
- `probable_pitchers`: Two-Start Detector and Smart Lineup Selector expect `quality_score` in their schemas. When the frontend reads NULL, matchup previews show blank ratings or default to neutral (0.0), removing a key edge-detection signal.

**Evidence:**
- `backend/services/daily_ingestion.py:5443` — `scarcity_rank` is absent from INSERT/UPDATE
- `backend/services/daily_ingestion.py:5651` — `quality_score=None` hardcoded

---

### Root Cause C: Cross-System ID Resolution Gaps (P1)

**Tables affected:** `player_projections`, `probable_pitchers`, `position_eligibility`

**Why it happens:** The fantasy baseball pipeline uses three separate player ID namespaces:
1. **FanGraphs** → provides projections keyed by a normalized name slug
2. **BDL (BallDontLie)** → provides box stats keyed by `bdl_player_id`
3. **MLBAM** → provides Statcast and Yahoo integration keyed by `mlbam_id`
4. **Yahoo** → provides roster/eligibility keyed by `yahoo_player_key`

The `player_id_mapping` table is the bridge, but it is not fully populated. The `_update_projection_cat_scores` job resolves FanGraphs names → MLBAM IDs, and when the lookup fails, the player is skipped entirely. Similarly, `probable_pitchers` resolves MLBAM IDs → BDL IDs, and when the mapping is missing, `bdl_player_id` remains NULL.

**Impact:** ~50% of `player_projections` rows have no `team` because the MLBAM lookup failed. ~54% of `probable_pitchers` have no `bdl_player_id`, which breaks joins to `player_scores` and `player_momentum` for matchup analysis.

**Evidence:**
- `backend/services/daily_ingestion.py:5020-5028` — `mlbam_id = name_to_mlbam.get(fg_id)`; if None, row is skipped
- `backend/services/daily_ingestion.py:5538-5546` — `mlbam_to_bdl` preloaded but missing many mappings

---

### Root Cause D: BDL API Partial Stat Coverage (P1)

**Tables affected:** `mlb_player_stats`

**Why it happens:** BDL's `/mlb/v1/stats` endpoint does not guarantee complete box stats for every player in every game. Pitchers often have NULL batting counting stats, and some games return rate stats (avg, obp, slg) without the underlying counting stats (ab, h, r, etc.). The existing `_supplement_statsapi_counting_stats` job (2:30 AM) patches gaps where `ab IS NULL`, but it does not patch rows where `ab` is present yet `runs` or `doubles` are missing.

**Impact:** Rolling window accuracy is degraded for affected players because the weighted sums undercount their contributions. The `_validate_mlb_stats` guard also rejects impossible ERA values, which is correct, but does not flag partial-stat rows.

**Evidence:**
- `backend/services/daily_ingestion.py:2036-2043` — supplement job filters on `ab IS NULL` only
- `backend/services/daily_ingestion.py:2113` — `_patch_counting_stats_batter` only patches individual NULL fields, but the job only runs for rows already flagged by the `ab IS NULL` filter

---

### Root Cause E: Season-History Insufficiency (P2)

**Tables affected:** `player_daily_metrics`

**Why it happens:** `z_score_total` requires 30 consecutive days of `vorp_7d` data; `z_score_recent` requires 7 days. The MLB season is currently <30 days old, so no player has enough history. This is a time-bound issue, not a code defect.

**Impact:** The rolling-Z dashboard panel shows empty sparklines for the first month of the season. This is expected behavior but should be communicated to users.

**Evidence:**
- `backend/services/daily_ingestion.py:4452-4462` — explicit length checks: `if len(vorp_values) >= 30` and `>= 7`

---

## Downstream Impact Assessment

| Feature | Dependent Table(s) | Null Sensitivity | Current Risk |
|---------|-------------------|------------------|--------------|
| **Daily Lineup Optimizer** | `player_scores`, `position_eligibility`, `probable_pitchers` | High | ⚠️ Degraded — missing `scarcity_rank` removes position-scarcity weighting; missing `quality_score` removes matchup penalty |
| **Waiver Edge Detector** | `player_scores`, `player_momentum`, `position_eligibility` | High | ⚠️ Degraded — same scarcity gap; new Z-score categories (z_r, z_tb, z_qs) are NULL for historical rows, biasing composite_z downward for players with old data |
| **Two-Start Command Center** | `probable_pitchers` | High | ❌ Broken — `quality_score` is 100% null; `two_start_detector.py` expects a float and will coerce to `0.0`, making every matchup look neutral |
| **VORP Computation** | `player_scores`, `position_eligibility` | Medium | ⚠️ Degraded — VORP runs but uses flat replacement levels because scarcity ranks are missing |
| **Backtesting Harness** | `mlb_player_stats`, `simulation_results` | Medium | ⚠️ Degraded — partial box stats cause inaccurate actuals aggregation; ERA/WHIP aggregation skips games with missing IP |
| **Explainability Layer** | `player_scores`, `player_momentum` | Low | ✅ Functional — momentum and core Z-scores are populated; missing new categories are simply omitted from narratives |
| **Statcast Proxy Projections** | `statcast_performances` | Low | ✅ Healthy — 0 nulls across all columns |

---

## Priority Actions for Claude Code

| Priority | Task | File | ETA | Acceptance Criteria |
|----------|------|------|-----|---------------------|
| **P0** | Backfill `player_rolling_stats.w_runs`, `w_tb`, `w_qs` for all historical rows | `scripts/backfill_v31_rolling.py` (new) | 2 days | Run retroactive `compute_rolling_window` for each (player, date, window) where the column is NULL and source stats exist |
| **P0** | Backfill `player_scores.z_r`, `z_h`, `z_tb`, `z_k_b`, `z_ops`, `z_k_p`, `z_qs` | `scripts/backfill_v32_zscores.py` (new) | 1 day (after P0-1) | Re-run `compute_league_zscores` for each historical `as_of_date` and `window_days` where new columns are NULL |
| **P0** | Implement `scarcity_rank` + `league_rostered_pct` in `_sync_position_eligibility` | `backend/services/daily_ingestion.py` | 1 day | Scarcity rank = percentile of position frequency (C=1, OF=12); rostered_pct = Yahoo `%owned` from ADP feed |
| **P0** | Implement `quality_score` in `_sync_probable_pitchers` | `backend/services/daily_ingestion.py` + `backend/fantasy_baseball/two_start_detector.py` | 1 day | Algorithm: `park_factor` adjusted by opposing team OPS vs pitcher handedness (use `statcast_batter_metrics` proxy) |
| **P1** | Harden `_supplement_statsapi_counting_stats` to patch *any* NULL counting stat, not just `ab IS NULL` | `backend/services/daily_ingestion.py` | 1 day | Change filter from `ab IS NULL` to `ab IS NULL OR runs IS NULL OR hits IS NULL OR ...`; patch all missing fields per row |
| **P1** | Fix `_update_projection_cat_scores` team/positions fallback | `backend/services/daily_ingestion.py` | 0.5 days | When `mlbam_id` lookup fails, fallback to raw FanGraphs `player_id` and write `team`/`positions` from the CSV row directly |
| **P1** | Add `bdl_stat_id` removal or proper mapping | `backend/models.py` or `backend/services/daily_ingestion.py` | 0.5 days | Either remove the unused column (clean schema) or map it correctly if BDL provides a game-level stat ID in a nested object |
| **P1** | Backfill `position_eligibility.bdl_player_id` gaps | `scripts/link_position_eligibility_bdl_ids.py` | 0.5 days | Run the existing script and verify <5% nulls remain |
| **P2** | Add data-quality gate to reject partial box-stat rows | `backend/services/daily_ingestion.py` | 1 day | If a row has `ab > 0` but `hits IS NULL`, log warning and skip or flag for supplement job |
| **P2** | Document `z_score_total` self-healing timeline | `docs/data_quality.md` or `HANDOFF.md` | 0.25 days | Explain to users that 30-day Z-scores require season history and will appear automatically after day 30 |
| **P3** | Deprecate or implement Statcast 2.0 columns | `backend/models.py` + ingestion job | Future sprint | Either remove `blast_pct`, `bat_speed`, etc. from schema if unused, or wire Savant leaderboard ingestion to populate them |

---

## Decisions Required

1. **Decision:** Should we delete the `bdl_stat_id` column entirely?
   - **Option A:** Drop column — cleanest, since no consumer uses it and BDL does not provide the data. Requires a migration script.
   - **Option B:** Keep column and populate from `raw_payload.id` if BDL nests a stat ID inside the player object — low value, extra complexity.
   - **Recommendation:** Option A (drop). The column has been 100% null for the table's entire lifetime.

2. **Decision:** How should `scarcity_rank` be computed?
   - **Option A:** Static scarcity based on 2026 preseason position frequency (C=most scarce, OF=least scarce) — simple, deterministic.
   - **Option B:** Dynamic scarcity recomputed daily from `position_eligibility` counts across the Yahoo league — more accurate but requires league roster data.
   - **Recommendation:** Option A for immediate fix; Option B as enhancement tracked in backlog.

3. **Decision:** Should probable-pitcher `quality_score` use a simple heuristic or a full model?
   - **Option A:** Heuristic: `park_factor * (1 + opp_woba_above_league_avg) * (1 - pitcher_era_below_league_avg)` — fast, no new dependencies.
   - **Option B:** Full model using `simulation_results` ROS projections + `team_profiles` opponent strength — most accurate but depends on ROS pipeline health.
   - **Recommendation:** Option A for immediate fix; Option B as v2 enhancement.

4. **Decision:** Should V31/V32 backfills be run as one-off scripts or integrated into the daily pipeline?
   - **Option A:** One-off scripts executed manually via Railway CLI — faster to ship, but historical gaps reappear if migrations run again.
   - **Option B:** Integrate retroactive computation into `_compute_rolling_windows` and `_compute_player_scores` so any future migration columns are automatically backfilled N days — more robust, slightly slower daily job.
   - **Recommendation:** Option A for immediate relief; add an `AUTO_BACKFILL_DAYS=30` config to the daily jobs as follow-up.

---

## Appendix

### A. Verification Queries

```sql
-- Count empty cat_scores (used by /api/admin/data-quality/summary)
SELECT COUNT(*) FROM player_projections
WHERE cat_scores IS NULL OR CAST(cat_scores AS TEXT) = '{}';
-- Result: 0 ✅

-- Count blend population in player_daily_metrics
SELECT COUNT(*) FROM player_daily_metrics WHERE blend_hr IS NOT NULL;
-- Result: 930 (5.1% of 18,229)

-- Count V31 column population
SELECT COUNT(*) FROM player_rolling_stats WHERE w_runs IS NOT NULL;
-- Result: 10,063 (15.0% of 67,232)

-- Count V32 column population
SELECT COUNT(*) FROM player_scores WHERE z_r IS NOT NULL;
-- Result: 10,063 (15.0% of 66,981)
```

### B. Pipeline Job Order & Dependencies

```
01:00  mlb_game_log
02:00  mlb_box_stats
02:30  statsapi_supplement   <-- patches NULL counting stats
03:00  fangraphs_ros
03:00  rolling_windows       <-- consumes mlb_player_stats
04:00  player_scores         <-- consumes player_rolling_stats
04:30  yahoo_id_sync
04:30  vorp                  <-- consumes player_scores + position_eligibility
05:00  player_momentum       <-- consumes player_scores
05:00  ensemble_update       <-- writes blend_* to player_daily_metrics
05:30  projection_cat_scores <-- writes team/cat_scores to player_projections
06:00  ros_simulation        <-- consumes player_rolling_stats (14d)
07:00  decision_optimization <-- consumes player_scores + momentum + simulation
07:15  position_eligibility  <-- writes Yahoo positions
08:30  probable_pitchers     <-- writes MLB Stats API schedule
```

### C. Relevant Code Paths

| Concern | File | Line Range |
|---------|------|------------|
| MLB box stats ingestion | `backend/services/daily_ingestion.py` | 1439-1663 |
| StatsAPI supplement | `backend/services/daily_ingestion.py` | 1999-2158 |
| Rolling window computation | `backend/services/daily_ingestion.py` | 2220-2386 |
| Z-score computation | `backend/services/daily_ingestion.py` | 2388-2554 |
| Ensemble blend | `backend/services/daily_ingestion.py` | 4717-4888 |
| Projection cat_scores | `backend/services/daily_ingestion.py` | 4890-5111 |
| Position eligibility sync | `backend/services/daily_ingestion.py` | 5343-5494 |
| Probable pitchers sync | `backend/services/daily_ingestion.py` | 5496-5707 |
| Rolling engine (pure) | `backend/services/rolling_window_engine.py` | 1-393 |
| Scoring engine (pure) | `backend/services/scoring_engine.py` | 1-495 |
| V31 migration | `scripts/migrate_v31_rolling_expansion.py` | 1-130 |
| V32 migration | `scripts/migrate_v32_zscore_expansion.py` | 1-161 |
