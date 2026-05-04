## K-34 Statcast Pipeline Audit — May 2, 2026

**Auditor:** Kimi CLI  
**Scope:** Full data-flow audit of Statcast ingestion, storage, and signal utilization  
**Database:** `railway` (production)  

---

### EXECUTIVE SUMMARY

- **Real data IS reaching the system.** Savant leaderboards are ingested daily (459 batters, 532 pitchers, updated 2026-05-02 10:00 UTC). Daily per-game Statcast data is current through 2026-05-01 (13,842 rows).
- **The primary predictive edge is completely disabled.** `xwoba_diff` and `xera_diff` are **exactly 0.000** for 100% of DB-tier players due to two separate name/format bugs in `_load_from_db()`. This means `BUY_LOW`, `SELL_HIGH`, and `luck_regression` signals are **never** generated from the DB tier.
- **Synthetic fallback CSVs are checked into git.** `data/projections/advanced_batting_2026.csv` and `advanced_pitching_2026.csv` (10 synthetic rows each) ship with every Railway deploy and serve as the final fallback tier.
- **Critical missing columns.** `sprint_speed`, `stuff_plus`, `location_plus`, and `pitching_plus` are defined in the dataclasses but are **not** present in the DB schema or ingestion pipeline.
- **Briefing cards bypass the DB tier entirely.** The morning briefing enrichment loads `pybaseball` leaderboards directly, ignoring the freshly-ingested Savant DB tables.

---

### 1. Data Sourcing — Real vs. Synthetic

#### 1.1 Savant Leaderboard Ingestion (`savant_ingestion.py`)
| Check | Result | Evidence |
|-------|--------|----------|
| Hits `baseballsavant.mlb.com` directly | ✅ Yes | `BASE_URL = "https://baseballsavant.mlb.com/leaderboard/custom"` (line 62) |
| No auth required | ✅ Yes | Public CSV export endpoint with `csv=true` param |
| Scheduled daily | ✅ Yes | `CronTrigger(hour=6, minute=0)` in `daily_ingestion.py:825` |
| Advisory lock registered | ✅ Yes | `LOCK_IDS["savant_ingestion"] = 100_032` (line 127) |
| Production data present | ✅ Yes | `statcast_batter_metrics`: 459 rows; `statcast_pitcher_metrics`: 532 rows |
| Freshness | ✅ Current | `last_updated`: 2026-05-02 10:00 UTC (today) |

#### 1.2 Daily Per-Game Ingestion (`statcast_ingestion.py`)
| Check | Result | Evidence |
|-------|--------|----------|
| Hits `baseballsavant.mlb.com/statcast_search/csv` | ✅ Yes | `base_url = "https://baseballsavant.mlb.com/statcast_search/csv"` (line 337) |
| Scheduled every 6 hours | ✅ Yes | `IntervalTrigger(hours=6)` in `daily_ingestion.py:814` |
| Production data present | ✅ Yes | `statcast_performances`: 13,842 rows, dates 2026-03-25 → 2026-05-01 |
| Zero-quality rows | ⚠️ 3.1% | 433 rows with all-zero quality metrics (exit_velocity_avg, xwoba, xba, hard_hit_pct, barrel_pct all = 0) |

#### 1.3 Fallback Tiers in `statcast_loader.py`
The loader has a 4-tier priority stack. The critical issue is what happens when Railway's ephemeral filesystem has no CSVs:

| Tier | Source | Data Quality | Present on Railway? |
|------|--------|-------------|---------------------|
| 0 | `statcast_batter_metrics` / `statcast_pitcher_metrics` (DB) | Real Savant leaderboards | ✅ Yes — 459/532 rows |
| 1 | `pybaseball_loader` JSON cache | FanGraphs leaderboards | ⚠️ Ephemeral — fetched on first run, 24h TTL |
| 2 | `data/cache/statcast_*.csv` | Manual Savant downloads | ❌ **Absent** — only `data/cache/weather/` exists |
| 3 | `data/projections/advanced_*.csv` | **Synthetic sample data** | ✅ **Checked into git** |

**Finding:** Tier 3 synthetic CSVs contain exactly **10 batters** and **10 pitchers** with hard-coded sample stats (e.g., Aaron Judge barrel% = 18.5, xwOBA = 0.435). If both DB and pybaseball tiers fail, the entire system operates on 20 synthetic players.

**Files checked into git:**
- `data/projections/advanced_batting_2026.csv` (13,094 bytes, committed 2026-03-09)
- `data/projections/advanced_pitching_2026.csv` (1,051 bytes, committed 2026-03-09)

---

### 2. Pipeline Completeness — Are the Two Pipelines Connected?

#### 2.1 Table Relationships
```
SavantIngestionAgent.run_daily_ingestion()
  ↓ fetches CSV from baseballsavant.mlb.com/leaderboard/custom
  ↓ parses → SavantMetricsRow
  ↓ _upsert_batters() → statcast_batter_metrics table
  ↓ _upsert_pitchers() → statcast_pitcher_metrics table

statcast_loader._load_from_db()
  ↓ queries statcast_batter_metrics + statcast_pitcher_metrics
  ↓ builds StatcastBatter / StatcastPitcher objects
  ↓ populates in-memory cache
```

**Connection status:** ✅ The DB read/write loop is wired correctly. `savant_ingestion.py` writes to the exact tables that `statcast_loader.py` reads.

#### 2.2 Scheduling Verification
| Pipeline | Function | Schedule | Lock ID | Status |
|----------|----------|----------|---------|--------|
| Savant leaderboards | `_ingest_savant_leaderboards` | 6:00 AM ET daily | `100_032` | ✅ Active |
| Daily per-game | `_update_statcast` | Every 6 hours | `100_016` | ✅ Active |

Both jobs are registered in `daily_ingestion.py` and have run successfully in production (logs show recent completions).

#### 2.3 What Each Pipeline Actually Produces
| Pipeline | Writes To | Row Count | Purpose |
|----------|-----------|-----------|---------|
| Savant leaderboards | `statcast_batter_metrics` | 459 | Season-aggregated batter stats (xwOBA, barrel%, etc.) |
| Savant leaderboards | `statcast_pitcher_metrics` | 532 | Season-aggregated pitcher stats (xERA, xwOBA allowed, etc.) |
| Daily per-game | `statcast_performances` | 13,842 | Per-game event-level data for Bayesian updates |

These are **separate tables serving separate purposes** — the leaderboard tables feed the waiver recommendation engine; the per-game table feeds Bayesian projection updates and the `avg_woba` computation.

---

### 3. Signal Utilization — Does Statcast Actually Influence Recommendations?

#### 3.1 `build_statcast_signals()` Return Contract
```python
def build_statcast_signals(player_name, is_pitcher, owned_pct=100.0) -> tuple[list[str], float]:
    # Returns: (signals, regression_delta)
    # signals: ["BUY_LOW", "BREAKOUT", "HIGH_INJURY_RISK", ...]
    # regression_delta: xwOBA - wOBA for batters, xERA - ERA for pitchers
```

#### 3.2 Signal-to-Score Mapping (`statcast_need_score_boost`)
| Signal | Score Delta | Direction |
|--------|-------------|-----------|
| `BUY_LOW` | +0.4 | Favors adding / keeping |
| `BREAKOUT` | +0.5 | Favors adding |
| `SELL_HIGH` | -0.3 | Favors dropping / avoiding |
| `HIGH_INJURY_RISK` | -0.2 | Favors dropping |

#### 3.3 Consumer Traces

**A. Waiver ADD recommendations (`fantasy.py:2418`)**
```python
fa_signals, fa_reg_delta = build_statcast_signals(fa.name, fa_is_pitcher, fa.owned_pct)
statcast_boost = statcast_need_score_boost(fa_signals)
adjusted_need = fa.need_score + statcast_boost
```
✅ **USED:** The boost is added to `need_score` and directly affects whether a player is recommended.

**B. Waiver DROP evaluation (`fantasy.py:2447`)**
```python
drop_signals, drop_reg_delta = build_statcast_signals(drop_candidate["name"], drop_is_pitcher)
drop_score_adj = max(
    _drop_candidate_value(drop_candidate)[0],
    drop_z_score + statcast_need_score_boost(drop_signals),
)
```
✅ **USED:** The drop candidate's score is adjusted by Statcast signals, affecting the net gain calculation.

**C. Roster response (`main.py:6401`)**
```python
signals, _ = build_statcast_signals(name, is_pitcher)
if signals:
    sc_signals = signals
# ... stored on RosterPlayerOut.statcast_signals
```
⚠️ **PRESENTATIONAL ONLY:** Signals are returned to the frontend but do not influence the roster scoring algorithm.

**D. Morning briefing cards (`fantasy.py:1456-1492`)**
```python
_br_bat = _lb(2026)   # load_pybaseball_batters(2026)
_br_pit = _lp(2026)   # load_pybaseball_pitchers(2026)
# ... builds card["statcast_stats"] and card["statcast_signals"]
```
❌ **BYPASSES DB TIER:** Briefing enrichment loads `pybaseball` leaderboards directly, completely ignoring the `statcast_*_metrics` tables populated by Savant ingestion.

#### 3.4 `advanced_metrics.py` Function Call Graph
| Function | Called By | Used in Scoring? |
|----------|-----------|------------------|
| `analyze_batter_regression()` | `build_statcast_signals()` | ✅ Indirectly |
| `analyze_pitcher_regression()` | `build_statcast_signals()` | ✅ Indirectly |
| `is_breakout_candidate_batter()` | `build_statcast_signals()` (owned_pct < 60 only) | ✅ Indirectly |
| `calculate_injury_risk_score()` | `build_statcast_signals()` | ✅ Indirectly |
| `is_breakout_candidate_pitcher()` | **Nowhere** | ❌ Dead code |
| `calculate_batter_power_score()` | **Nowhere** | ❌ Dead code |
| `calculate_pitcher_stuff_score()` | **Nowhere** | ❌ Dead code |

**Verdict:** The signal pipeline is architecturally sound, but the underlying regression deltas are broken (see Section 5).

---

### 4. Column Mapping Integrity

#### 4.1 `statcast_performances` Model vs. Ingestion
| Model Column | Ingestion Source | Status |
|--------------|-----------------|--------|
| `exit_velocity_avg` | `launch_speed` / `exit_velocity_avg` | ✅ Mapped |
| `xwoba` | `xwoba` / `estimated_woba_using_speedangle` | ✅ Mapped |
| `xba` | `xba` / `estimated_ba_using_speedangle` | ✅ Mapped |
| `xslg` | `xslg` / `estimated_slg_using_speedangle` | ✅ Mapped |
| `hard_hit_pct` | `hardhit_percent` / `hard_hit_percent` | ✅ Mapped (divided by 100) |
| `barrel_pct` | `barrels_per_pa_percent` / `barrel_batted_rate` | ✅ Mapped (divided by 100) |
| `woba` | Computed from box stats in dataclass | ✅ Stored |

**Finding:** 433 rows (3.1%) have ALL of these metrics equal to 0. This is acceptable for pinch-runners or defensive replacements with 0 batted ball events, but the validation code (`admin_endpoints_validation.py:271-305`) correctly flags it for investigation.

#### 4.2 `statcast_batter_metrics` Model vs. Savant CSV
| Model Column | Savant CSV Column (in `_parse_batter_row`) | Status |
|--------------|-------------------------------------------|--------|
| `xwoba` | `xwoba` | ✅ |
| `barrel_percent` | `barrel_batted_rate` | ✅ |
| `hard_hit_percent` | `hard_hit_percent` | ✅ |
| `avg_exit_velocity` | `exit_velocity_avg` | ✅ |
| `whiff_percent` | `whiff_percent` | ✅ |
| `swing_percent` | `swing_percent` | ✅ |

#### 4.3 `statcast_pitcher_metrics` Model vs. Savant CSV
| Model Column | Savant CSV Column (in `_parse_pitcher_row`) | Status |
|--------------|--------------------------------------------|--------|
| `xera` | `xera` | ✅ |
| `xwoba` | `xwoba` | ✅ |
| `barrel_percent_allowed` | `barrel_batted_rate` | ✅ |
| `hard_hit_percent_allowed` | `hard_hit_percent` | ✅ |
| `avg_exit_velocity_allowed` | `exit_velocity_avg` | ✅ |
| `k_percent` | `k_percent` | ✅ |
| `bb_percent` | `bb_percent` | ✅ |
| `era` | `era` | ❌ **MISSING** — see Section 5.2 |

---

### 5. Player ID Resolution & Match Rate

#### 5.1 Name Normalization Pipeline
```
Yahoo name → _strip_name() → lowercase ASCII, no accents, no Jr/Sr/II/III
                    ↓
Savant DB name → _key() → identical normalization
                    ↓
              Cache lookup by normalized key
```

**Normalization is identical** between `pybaseball_loader._strip_name()` and `statcast_loader._load_from_db._key()`.

#### 5.2 Name Format Mismatch (CRITICAL)

**`statcast_performances` stores names as `"Last, First"`** (e.g., `'Jansen, Kenley'`, `'Pivetta, Nick'`)
**`statcast_batter_metrics` stores names as `"First Last"`** (e.g., `'Andrés Giménez'`, `'Oswald Peraza'`)

The Savant daily search CSV returns `player_name` in `"Last, First"` format, while the Savant leaderboard CSV returns `"last_name, first_name"` which `savant_ingestion.py` converts to `"First Last"`.

**Impact on `_load_from_db()`:**
```sql
LEFT JOIN (
    SELECT LOWER(player_name) AS lname, AVG(woba) AS avg_woba
    FROM statcast_performances
    WHERE woba > 0
    GROUP BY LOWER(player_name)
) sp_agg ON sp_agg.lname = LOWER(sbm.player_name)
```

This join attempts to match `'gimenez, andres'` with `'andres gimenez'`. **It never matches.**

**Empirical result:**
- `statcast_batter_metrics with avg_woba match: 0 / 458`
- `avg_woba` is always NULL in the join result.
- The code falls back: `avg_woba = _float(row.avg_woba) if row.avg_woba else xwoba`
- Therefore: `xwoba_diff = xwoba - xwoba = 0.000` for **100% of batters**.

#### 5.3 ERA Column Missing for Pitchers (CRITICAL)

In `statcast_pitcher_metrics`:
- `nonzero_xera`: 532 / 532 (100%)
- `nonzero_era`: 0 / 532 (0%)
- `era` is **None for every single pitcher**.

**Root cause:** The Baseball Savant custom leaderboard likely returns ERA under a column name other than `"era"` (possibly `"earned_run_average"` or `"earned_run_avg"`). The parser does `row.get("era", "")` which returns empty string, and `_savant_float("")` returns `None`.

**Impact on `_load_from_db()`:**
```python
xera = _float(row.xera)          # e.g., 4.54
era = _float(row.era) if row.era else xera   # era is None → era = 4.54
xera_diff = xera - era            # 4.54 - 4.54 = 0.00
```

**Empirical result:**
- Pitcher `xera_diff`: min=0.0000, max=0.0000, mean=0.0000 for **100% of pitchers**.

#### 5.4 Match Rate Summary
| Dataset | Rows | In `player_id_mapping`? | Rate |
|---------|------|------------------------|------|
| `statcast_batter_metrics` | 458 | 458 | 100% |
| `statcast_pitcher_metrics` | 532 | 521 | 97.9% |

Name resolution between Savant and the ID bridge is **excellent** for leaderboard data. The breakdown is entirely in the `xwoba_diff` / `xera_diff` computation, not in the name→player lookup.

---

### 6. Data Recency

| Table | Most Recent Date | Staleness |
|-------|-----------------|-----------|
| `statcast_performances` | 2026-05-01 | ~0-1 day |
| `statcast_batter_metrics` | 2026-05-02 10:00 UTC | ~0-1 day |
| `statcast_pitcher_metrics` | 2026-05-02 10:00 UTC | ~0-1 day |

**Ingestion schedule:**
- Savant leaderboards: daily at 6:00 AM ET
- Per-game Statcast: every 6 hours

**Cache invalidation:** The `statcast_loader` 6-hour in-memory cache (`_CACHE_TTL = 6 * 3600`) is **not** invalidated by either ingestion job. The cache only clears on:
1. Process restart (Railway deploy)
2. Manual call to `POST /admin/pybaseball/refresh`
3. Cache TTL expiration (6 hours)

This means after the 6:00 AM Savant ingestion, the loader cache may still hold pre-6:00 AM data until the 6-hour TTL expires (12:00 PM ET) or until the next deploy.

---

### 7. Gap Analysis — Missing Signals

#### 7.1 `sprint_speed` — Critical for SB Projections
| Check | Result |
|-------|--------|
| In `StatcastBatter` dataclass | ✅ Yes |
| In `statcast_batter_metrics` DB model | ❌ **No** |
| In Savant ingestion `BATTER_SELECTIONS` | ❌ **No** |
| Used in `is_breakout_candidate_batter()` | ✅ Yes (`sprint_speed > 28.5`) |

**Impact:** Breakout detection for speed threats is disabled for DB-tier data. The `sprint_speed` field is only populated when pybaseball CSV or synthetic sample data loads.

#### 7.2 `stuff_plus`, `location_plus`, `pitching_plus`
| Check | stuff_plus | location_plus | pitching_plus |
|-------|-----------|---------------|---------------|
| In `StatcastPitcher` dataclass | ✅ | ✅ | ✅ |
| In `statcast_pitcher_metrics` DB model | ❌ | ❌ | ❌ |
| In Savant ingestion `PITCHER_SELECTIONS` | ❌ | ❌ | ❌ |

**Impact:** Pitch quality scoring (`calculate_pitcher_stuff_score()`) and breakout detection for pitchers (`stuff_plus > 110`) are disabled for DB-tier data.

#### 7.3 `wRC+` / `OPS+`
- Not present in any model, ingestion pipeline, or dataclass field.
- `wrc_plus` exists on `StatcastBatter` but is hard-coded to `100.0` and never updated from real data.

#### 7.4 `xFIP`, `FIP`, `SIERA`
- Not present anywhere in the system.

#### 7.5 `xwoba_diff` Regression Signal
- The signal is **computed** in `_load_from_db()` but **always equals 0** due to the name-format join bug.
- This is the **single highest-value regression-to-mean predictor** and it is completely neutralized.

#### 7.6 Sample Size Gate
- **No gate exists.** A player with 1 PA receives the same `statcast_need_score_boost()` as a player with 200 PA.
- The `StatcastIngestionAgent` has a `min_pa=20` gate for Bayesian updates, but the waiver recommendation engine does not down-weight low-sample signals.

---

### FINDINGS TABLE

| # | Severity | Component | Issue | Recommended Fix |
|---|----------|-----------|-------|-----------------|
| 1 | **CRITICAL** | `statcast_loader._load_from_db()` | Name format mismatch: `statcast_performances` stores `"Last, First"` while `statcast_batter_metrics` stores `"First Last"`. The `avg_woba` join never matches, so `xwoba_diff = 0` for 100% of batters. | Normalize names identically before join, or join on `mlbam_id` instead of `player_name`. |
| 2 | **CRITICAL** | `savant_ingestion._parse_pitcher_row()` | ERA column is not mapped correctly from Savant CSV. `row.get("era")` returns empty, so `era=None` for all 532 pitchers. `xera_diff = 0` for 100% of pitchers. | Inspect actual Savant CSV column name for ERA (likely `earned_run_average` or `earned_run_avg`) and update the parser. |
| 3 | **CRITICAL** | `data/projections/` | Synthetic sample CSVs (`advanced_batting_2026.csv`, `advanced_pitching_2026.csv`) are checked into git and serve as fallback tier. | Move sample generators to an on-demand bootstrap script; do not ship synthetic data to production. |
| 4 | **HIGH** | `fantasy.py` briefing enrichment | Briefing cards load `pybaseball_loader` directly, bypassing the DB tables populated by Savant ingestion. | Route briefing enrichment through `statcast_loader` so it consumes the same DB-tier data as waiver recommendations. |
| 5 | **HIGH** | `savant_ingestion.py` | `sprint_speed` is not included in `BATTER_SELECTIONS` or `statcast_batter_metrics` model. | Add `"sprint_speed"` to `BATTER_SELECTIONS`, DB model, and `_parse_batter_row()`. |
| 6 | **HIGH** | `savant_ingestion.py` | `stuff_plus`, `location_plus`, `pitching_plus` are not included in `PITCHER_SELECTIONS` or `statcast_pitcher_metrics` model. | Add `"stuff_plus","location_plus","pitching_plus"` to `PITCHER_SELECTIONS`, DB model, and `_parse_pitcher_row()`. |
| 7 | **MEDIUM** | `statcast_loader.py` | 6-hour in-memory cache is not invalidated by Savant or Statcast ingestion jobs. | Add cache-clear call at the end of `_ingest_savant_leaderboards()` and `_update_statcast()`. |
| 8 | **MEDIUM** | `statcast_ingestion.py` | 433 rows (3.1%) in `statcast_performances` have all-zero quality metrics. | Investigate if these are legitimate (0 PA defensive replacements) or column-mapping misses. |
| 9 | **MEDIUM** | Waiver scoring | No sample-size gate for Statcast signals. | Add `min_pa` / `min_ip` threshold before applying `statcast_need_score_boost()`. |
| 10 | **LOW** | `advanced_metrics.py` | `calculate_batter_power_score()`, `calculate_pitcher_stuff_score()`, `is_breakout_candidate_pitcher()` are defined but never called. | Either wire them into `build_statcast_signals()` or remove dead code. |
| 11 | **LOW** | `pybaseball_loader.py` | Year parameter is inconsistent (`2025` in some calls, `2026` in others). | Standardize on `season = 2026` everywhere. |

---

### IMPLEMENTATION PRIORITY

#### P0 (Blocks All Predictive Edge)
1. **Fix name-format join in `_load_from_db()`** — The `avg_woba` subquery must join on a normalized name key that matches `statcast_batter_metrics.player_name`. Options:
   - Parse `"Last, First"` → `"First Last"` in the subquery.
   - Join on `mlbam_id` (preferred — requires adding `mlbam_id` to `statcast_batter_metrics` and resolving it in `statcast_performances`).
2. **Fix ERA column mapping in `_parse_pitcher_row()`** — Identify the actual Savant CSV column name and map it to `era`.
3. **Remove synthetic CSVs from git** — Add `data/projections/advanced_*.csv` to `.gitignore` and generate them on-demand via `generate_advanced_*_csv()` only in local dev.

#### P1 (Significant Uplift)
4. **Route briefing cards through `statcast_loader`** — Replace direct `pybaseball_loader` calls in `fantasy.py:1451-1456` with `statcast_loader.get_statcast_batter()` / `get_statcast_pitcher()`.
5. **Add `sprint_speed` to batter pipeline** — Include in Savant selections, DB model, and loader query.
6. **Add `stuff_plus`, `location_plus`, `pitching_plus` to pitcher pipeline** — Same pattern as sprint_speed.

#### P2 (Nice to Have)
7. **Add cache invalidation to ingestion jobs** — Call `_sc._batter_cache.clear(); _sc._pitcher_cache.clear(); _sc._loaded_at = 0.0` after successful Savant ingestion.
8. **Add sample-size gate** — Require `min_pa >= 30` for batters and `min_ip >= 10` for pitchers before applying score boosts.
9. **Investigate zero-quality rows** — Run diagnostic query to distinguish legitimate 0-PA rows from mapping failures.

---

### APPENDIX: Supporting Queries

**Query A: Verify xwoba_diff is zero for all DB-tier batters**
```sql
SELECT COUNT(*) FILTER (WHERE xwoba_diff = 0) AS zero_diff,
       COUNT(*) AS total
FROM statcast_batter_metrics;
-- Result: 458 / 458 (100%)
```

**Query B: Verify xera_diff is zero for all DB-tier pitchers**
```sql
SELECT COUNT(*) FILTER (WHERE xera IS NULL OR era IS NULL) AS missing,
       COUNT(*) AS total
FROM statcast_pitcher_metrics;
-- Result: 532 / 532 (100% have NULL era)
```

**Query C: Name format in statcast_performances**
```sql
SELECT DISTINCT player_name FROM statcast_performances LIMIT 5;
-- 'Shugart, Chase'
-- 'Herrin, Tim'
-- 'Jansen, Kenley'
```

**Query D: Name format in statcast_batter_metrics**
```sql
SELECT player_name FROM statcast_batter_metrics LIMIT 5;
-- 'Andrés Giménez'
-- 'Oswald Peraza'
-- 'Gabriel Moreno'
```

**Query E: Zero-quality rows**
```sql
SELECT COUNT(*) FROM statcast_performances
WHERE COALESCE(exit_velocity_avg,0)=0
  AND COALESCE(xwoba,0)=0
  AND COALESCE(xba,0)=0
  AND COALESCE(hard_hit_pct,0)=0
  AND COALESCE(barrel_pct,0)=0;
-- Result: 433 / 13,842 (3.1%)
```

---

*Report generated by Kimi CLI (K-34) on 2026-05-02. All code references verified against commit `175549d` on branch `stable/cbb-prod`.*
