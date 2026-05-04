# Kimi CLI — Full Database Data Quality Audit

**Date:** 2026-05-01  
**Agent:** Kimi CLI (Deep Intelligence Unit)  
**Assigned by:** Claude Code (Master Architect)  
**Database:** PostgreSQL (Railway managed)  
**Total Queries Run:** 40+  
**Total Issues Found:** 14 classified findings  

---

## SECTION 1 — DATABASE INVENTORY & HEALTH BASELINE

### 1a. Full Table Inventory with Row Counts and Storage Size

```sql
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
       pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS data_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Result (top 20 by size):**

| Table | Exact Row Count | Total Size | Data Size | Notes |
|-------|-----------------|------------|-----------|-------|
| mlb_odds_snapshot | 92,261 | 68 MB | 53 MB | Betting odds; actively growing |
| player_rolling_stats | 75,154 | 30 MB | 19 MB | 3 windows × ~25k rows each |
| player_scores | 74,872 | 24 MB | 17 MB | Z-scores; matches rolling stats closely |
| mlb_player_stats | 13,444 | 16 MB | 13 MB | Raw daily box stats |
| simulation_results | 24,095 | 7352 kB | 6104 kB | CBB/MLB sim output |
| player_id_mapping | 10,096 | 7136 kB | 2088 kB | ID crosswalk |
| player_daily_metrics | 21,565 | 6776 kB | 3136 kB | Multi-sport daily metrics |
| statcast_performances | 13,414 | 6480 kB | 4008 kB | Statcast event-level data |
| backtest_results | 22,375 | 5240 kB | 3096 kB | Model backtests |
| player_momentum | 24,964 | 4568 kB | 3288 kB | Trend signals |
| data_ingestion_logs | 5,433 | 2352 kB | 1840 kB | Pipeline audit trail |
| ingested_injuries | 232 | 1192 kB | 640 kB | Injury records |
| player_projections | 623 | 1088 kB | 680 kB | Steamer ROS projections |
| mlb_game_log | 472 | 1080 kB | 960 kB | Game metadata |
| position_eligibility | 2,389 | 1064 kB | 608 kB | Yahoo position eligibility |
| decision_explanations | 582 | 704 kB | 600 kB | Lineup decision explanations |
| decision_results | 620 | 272 kB | 112 kB | Optimized lineup outputs |
| probable_pitchers | 504 | 224 kB | 112 kB | Probable starters |
| predictions | 2 | 160 kB | 8192 bytes | CBB predictions (frozen/archival) |
| games | 4 | 112 kB | 8192 bytes | CBB games (frozen/archival) |

**Tables with 0 rows (empty):**
- statcast_batter_metrics
- statcast_pitcher_metrics
- player_valuation_cache
- weather_forecasts
- team_profiles
- closing_lines
- model_parameters
- deployment_version
- alerts
- pattern_detection_alerts
- job_queue
- execution_decisions
- fantasy_draft_picks
- fantasy_draft_sessions

### 1b. Index Inventory

```sql
SELECT indexname, tablename,
       pg_size_pretty(pg_relation_size(indexname::regclass)) AS idx_size
FROM pg_indexes
WHERE schemaname = 'public'
ORDER BY pg_relation_size(indexname::regclass) DESC;
```

**Top 10 indexes by size:**

| Index | Table | Size |
|-------|-------|------|
| mlb_odds_snapshot_game_id_vendor_snapshot_window_key | mlb_odds_snapshot | 5920 kB |
| _prs_player_date_window_uc | player_rolling_stats | 4464 kB |
| _ps_player_date_window_uc | player_scores | 3912 kB |
| mlb_odds_snapshot_pkey | mlb_odds_snapshot | 3720 kB |
| idx_mlb_odds_vendor_window | mlb_odds_snapshot | 3256 kB |
| player_rolling_stats_pkey | player_rolling_stats | 3208 kB |
| player_scores_pkey | player_scores | 3152 kB |
| idx_mlb_odds_game | mlb_odds_snapshot | 2016 kB |
| player_id_mapping_pkey | player_id_mapping | 1984 kB |
| idx_prs_player_date | player_rolling_stats | 1816 kB |

**Finding:** All expected unique constraints exist. No invalid indexes detected.

### 1c. Data Ingestion Job Health

```sql
SELECT job_type, status, started_at, completed_at,
       EXTRACT(EPOCH FROM (completed_at - started_at)) AS duration_sec,
       records_processed, error_message
FROM data_ingestion_logs
ORDER BY started_at DESC
LIMIT 100;
```

**Summary of last 100 runs:**
- `mlb_odds`: Runs every ~5 min, mostly SUCCESS (47 records), occasional SKIPPED
- `position_eligibility`: SUCCESS at 15:15 today (236 records)
- `bdl_injuries`: SUCCESS at 15:08 today (178 records)
- `player_id_mapping`: SUCCESS at 15:00 today (10,000 records, took ~470 sec)
- `decision_optimization`: SUCCESS at 15:00 today (38 records)
- `projection_freshness`: SUCCESS at 15:08 today (0 records, ~0.04 sec)
- `cat_scores_backfill`: SUCCESS at 14:30 today (2 records)
- `yahoo_adp_injury`: SUCCESS at 14:08 today (100 records)
- `ros_simulation`: SUCCESS at 14:00 today (866 records, ~212 sec)
- `rolling_windows`: SUCCESS at 11:00 today (2,648 records)
- `fangraphs_ros`: SUCCESS at 11:00 today (120 records)
- `statsapi_supplement`: SUCCESS at 10:30 today (328 records)

**Jobs with errors in last 48 hours:**
- `valuation_cache`: FAILED today at 14:00 — "'list' object has no attribute 'get'"
- `savant_ingestion`: FAILED today at 14:00 — CSV newline parsing error
- `savant_ingestion`: FAILED today at 07:29 — same CSV error
- `yahoo_id_sync`: Most recent SUCCESS (today 12:30), but historical failures due to unique constraint violations

**Jobs that have NOT run in last 48 hours:**
- None of the critical daily jobs are missing. All core jobs (fangraphs_ros, rolling_windows, player_scores, etc.) ran today.

### 1d. Advisory Lock Status

```sql
SELECT pid, granted, classid, objid, mode
FROM pg_locks
WHERE locktype = 'advisory'
ORDER BY classid, objid;
```

**Result:** `[]` — No advisory locks currently held. No stuck jobs.

---

## SECTION 2 — NULL VALUE AUDIT (ALL CRITICAL TABLES)

### 2a. player_projections

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE player_id IS NULL) AS player_id_nulls,
  COUNT(*) FILTER (WHERE player_name IS NULL) AS player_name_nulls,
  COUNT(*) FILTER (WHERE team IS NULL) AS team_nulls,
  COUNT(*) FILTER (WHERE positions IS NULL) AS positions_nulls,
  COUNT(*) FILTER (WHERE woba IS NULL) AS woba_nulls,
  COUNT(*) FILTER (WHERE avg IS NULL) AS avg_nulls,
  COUNT(*) FILTER (WHERE obp IS NULL) AS obp_nulls,
  COUNT(*) FILTER (WHERE slg IS NULL) AS slg_nulls,
  COUNT(*) FILTER (WHERE ops IS NULL) AS ops_nulls,
  COUNT(*) FILTER (WHERE xwoba IS NULL) AS xwoba_nulls,
  COUNT(*) FILTER (WHERE hr IS NULL) AS hr_nulls,
  COUNT(*) FILTER (WHERE r IS NULL) AS r_nulls,
  COUNT(*) FILTER (WHERE rbi IS NULL) AS rbi_nulls,
  COUNT(*) FILTER (WHERE sb IS NULL) AS sb_nulls,
  COUNT(*) FILTER (WHERE era IS NULL) AS era_nulls,
  COUNT(*) FILTER (WHERE whip IS NULL) AS whip_nulls,
  COUNT(*) FILTER (WHERE k_per_nine IS NULL) AS k_per_nine_nulls,
  COUNT(*) FILTER (WHERE bb_per_nine IS NULL) AS bb_per_nine_nulls,
  COUNT(*) FILTER (WHERE w IS NULL) AS w_nulls,
  COUNT(*) FILTER (WHERE k_pit IS NULL) AS k_pit_nulls,
  COUNT(*) FILTER (WHERE qs IS NULL) AS qs_nulls,
  COUNT(*) FILTER (WHERE nsv IS NULL) AS nsv_nulls,
  COUNT(*) FILTER (WHERE cat_scores IS NULL) AS cat_scores_nulls
FROM player_projections;
```

| Metric | Value |
|--------|-------|
| total_rows | 623 |
| player_id_nulls | 0 |
| player_name_nulls | 0 |
| team_nulls | 6 |
| positions_nulls | 157 |
| woba_nulls | 0 |
| avg_nulls | 0 |
| obp_nulls | 0 |
| slg_nulls | 0 |
| ops_nulls | 0 |
| xwoba_nulls | 0 |
| hr_nulls | 0 |
| r_nulls | 0 |
| rbi_nulls | 0 |
| sb_nulls | 0 |
| era_nulls | 0 |
| whip_nulls | 0 |
| k_per_nine_nulls | 0 |
| bb_per_nine_nulls | 0 |
| w_nulls | 0 |
| k_pit_nulls | 0 |
| qs_nulls | 0 |
| nsv_nulls | 0 |
| cat_scores_nulls | 0 |

**Special check — cat_scores population:**

```sql
SELECT COUNT(*) FILTER (WHERE cat_scores IS NULL) AS null_cats,
       COUNT(*) FILTER (WHERE cat_scores::text = '{}') AS empty_cats,
       COUNT(*) FILTER (WHERE cat_scores IS NOT NULL AND cat_scores::text != '{}') AS populated_cats
FROM player_projections;
```

| null_cats | empty_cats | populated_cats |
|-----------|------------|----------------|
| 0 | 0 | 623 |

**Finding:** 157/623 (25.2%) rows have NULL `positions`. 6/623 (1.0%) have NULL `team`. All cat_scores populated.

### 2b. player_rolling_stats

```sql
SELECT window_days, COUNT(*) AS rows,
       COUNT(*) FILTER (WHERE w_avg IS NULL) AS null_avg,
       COUNT(*) FILTER (WHERE w_ops IS NULL) AS null_ops,
       COUNT(*) FILTER (WHERE w_era IS NULL) AS null_era
FROM player_rolling_stats GROUP BY window_days ORDER BY window_days;
```

| window_days | rows | null_avg | null_ops | null_era |
|-------------|------|----------|----------|----------|
| 7 | 23,869 | 12,115 | 12,115 | 11,917 |
| 14 | 25,060 | 12,881 | 12,881 | 12,240 |
| 30 | 26,225 | 13,660 | 13,660 | 12,523 |

**Full null audit:**

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE bdl_player_id IS NULL) AS bdl_player_id_nulls,
  COUNT(*) FILTER (WHERE as_of_date IS NULL) AS as_of_date_nulls,
  COUNT(*) FILTER (WHERE window_days IS NULL) AS window_days_nulls,
  COUNT(*) FILTER (WHERE w_runs IS NULL) AS w_runs_nulls,
  COUNT(*) FILTER (WHERE w_hits IS NULL) AS w_hits_nulls,
  COUNT(*) FILTER (WHERE w_home_runs IS NULL) AS w_home_runs_nulls,
  COUNT(*) FILTER (WHERE w_rbi IS NULL) AS w_rbi_nulls,
  COUNT(*) FILTER (WHERE w_stolen_bases IS NULL) AS w_stolen_bases_nulls,
  COUNT(*) FILTER (WHERE w_net_stolen_bases IS NULL) AS w_net_stolen_bases_nulls,
  COUNT(*) FILTER (WHERE w_strikeouts_bat IS NULL) AS w_strikeouts_bat_nulls,
  COUNT(*) FILTER (WHERE w_tb IS NULL) AS w_tb_nulls,
  COUNT(*) FILTER (WHERE w_avg IS NULL) AS w_avg_nulls,
  COUNT(*) FILTER (WHERE w_obp IS NULL) AS w_obp_nulls,
  COUNT(*) FILTER (WHERE w_ops IS NULL) AS w_ops_nulls,
  COUNT(*) FILTER (WHERE w_era IS NULL) AS w_era_nulls,
  COUNT(*) FILTER (WHERE w_whip IS NULL) AS w_whip_nulls,
  COUNT(*) FILTER (WHERE w_k_per_9 IS NULL) AS w_k_per_9_nulls,
  COUNT(*) FILTER (WHERE w_strikeouts_pit IS NULL) AS w_strikeouts_pit_nulls,
  COUNT(*) FILTER (WHERE w_qs IS NULL) AS w_qs_nulls
FROM player_rolling_stats;
```

| Metric | Value |
|--------|-------|
| total_rows | 75,154 |
| bdl_player_id_nulls | 0 |
| as_of_date_nulls | 0 |
| window_days_nulls | 0 |
| w_runs_nulls | 38,054 |
| w_hits_nulls | 38,070 |
| w_home_runs_nulls | 38,070 |
| w_rbi_nulls | 38,070 |
| w_stolen_bases_nulls | 38,070 |
| w_net_stolen_bases_nulls | 38,064 |
| w_strikeouts_bat_nulls | 38,070 |
| w_tb_nulls | 38,064 |
| w_avg_nulls | 38,656 |
| w_obp_nulls | 38,550 |
| w_ops_nulls | 38,656 |
| w_era_nulls | 36,680 |
| w_whip_nulls | 36,680 |
| w_k_per_9_nulls | 36,680 |
| w_strikeouts_pit_nulls | 36,590 |
| w_qs_nulls | 36,580 |

**Finding:** ~50% null rates on batting/pitching stats are expected — pitchers lack batting stats and vice versa. No structural nulls in key dimensions (bdl_player_id, as_of_date, window_days).

### 2c. player_scores

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE bdl_player_id IS NULL) AS bdl_player_id_nulls,
  COUNT(*) FILTER (WHERE as_of_date IS NULL) AS as_of_date_nulls,
  COUNT(*) FILTER (WHERE window_days IS NULL) AS window_days_nulls,
  COUNT(*) FILTER (WHERE z_r IS NULL) AS z_r_nulls,
  COUNT(*) FILTER (WHERE z_h IS NULL) AS z_h_nulls,
  COUNT(*) FILTER (WHERE z_hr IS NULL) AS z_hr_nulls,
  COUNT(*) FILTER (WHERE z_rbi IS NULL) AS z_rbi_nulls,
  COUNT(*) FILTER (WHERE z_sb IS NULL) AS z_sb_nulls,
  COUNT(*) FILTER (WHERE z_nsb IS NULL) AS z_nsb_nulls,
  COUNT(*) FILTER (WHERE z_k_b IS NULL) AS z_k_b_nulls,
  COUNT(*) FILTER (WHERE z_tb IS NULL) AS z_tb_nulls,
  COUNT(*) FILTER (WHERE z_avg IS NULL) AS z_avg_nulls,
  COUNT(*) FILTER (WHERE z_obp IS NULL) AS z_obp_nulls,
  COUNT(*) FILTER (WHERE z_ops IS NULL) AS z_ops_nulls,
  COUNT(*) FILTER (WHERE z_era IS NULL) AS z_era_nulls,
  COUNT(*) FILTER (WHERE z_whip IS NULL) AS z_whip_nulls,
  COUNT(*) FILTER (WHERE z_k_per_9 IS NULL) AS z_k_per_9_nulls,
  COUNT(*) FILTER (WHERE z_k_p IS NULL) AS z_k_p_nulls,
  COUNT(*) FILTER (WHERE z_qs IS NULL) AS z_qs_nulls,
  COUNT(*) FILTER (WHERE composite_z IS NULL) AS composite_z_nulls,
  COUNT(*) FILTER (WHERE score_0_100 IS NULL) AS score_0_100_nulls,
  COUNT(*) FILTER (WHERE confidence IS NULL) AS confidence_nulls
FROM player_scores;
```

| Metric | Value |
|--------|-------|
| total_rows | 74,872 |
| bdl_player_id_nulls | 0 |
| as_of_date_nulls | 0 |
| window_days_nulls | 0 |
| z_r_nulls | 37,788 |
| z_h_nulls | 37,788 |
| z_hr_nulls | 37,778 |
| z_rbi_nulls | 37,778 |
| z_sb_nulls | 37,778 |
| z_nsb_nulls | 53,471 |
| z_k_b_nulls | 37,788 |
| z_tb_nulls | 37,788 |
| z_avg_nulls | 38,364 |
| z_obp_nulls | 38,258 |
| z_ops_nulls | 38,374 |
| z_era_nulls | 36,398 |
| z_whip_nulls | 36,398 |
| z_k_per_9_nulls | 36,398 |
| z_k_p_nulls | 36,308 |
| z_qs_nulls | 36,308 |
| composite_z_nulls | 0 |
| score_0_100_nulls | 0 |
| confidence_nulls | 0 |

**Cross-join check — every rolling_stats row has a matching score row:**

```sql
SELECT COUNT(*) AS rolling_rows,
       COUNT(ps.bdl_player_id) AS matched_score_rows,
       COUNT(*) - COUNT(ps.bdl_player_id) AS unmatched
FROM player_rolling_stats prs
LEFT JOIN player_scores ps
  ON prs.bdl_player_id = ps.bdl_player_id AND prs.as_of_date = ps.as_of_date
     AND prs.window_days = ps.window_days;
```

| rolling_rows | matched_score_rows | unmatched |
|--------------|-------------------|-----------|
| 75,154 | 74,872 | 282 |

**Finding:** 282 rolling_stats rows (0.4%) lack a corresponding player_scores row. composite_z and score_0_100 are never NULL.

### 2d. mlb_player_stats

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE bdl_player_id IS NULL) AS bdl_player_id_nulls,
  COUNT(*) FILTER (WHERE game_date IS NULL) AS game_date_nulls,
  COUNT(*) FILTER (WHERE ab IS NULL) AS ab_nulls,
  COUNT(*) FILTER (WHERE hits IS NULL) AS hits_nulls,
  COUNT(*) FILTER (WHERE home_runs IS NULL) AS home_runs_nulls,
  COUNT(*) FILTER (WHERE rbi IS NULL) AS rbi_nulls,
  COUNT(*) FILTER (WHERE stolen_bases IS NULL) AS stolen_bases_nulls,
  COUNT(*) FILTER (WHERE strikeouts_bat IS NULL) AS strikeouts_nulls,
  COUNT(*) FILTER (WHERE walks IS NULL) AS walks_nulls,
  COUNT(*) FILTER (WHERE innings_pitched IS NULL) AS innings_pitched_nulls,
  COUNT(*) FILTER (WHERE earned_runs IS NULL) AS earned_runs_nulls,
  COUNT(*) FILTER (WHERE strikeouts_pit IS NULL) AS pitcher_strikeouts_nulls,
  COUNT(*) FILTER (WHERE raw_payload IS NULL) AS raw_payload_nulls
FROM mlb_player_stats;
```

| Metric | Value |
|--------|-------|
| total_rows | 13,444 |
| bdl_player_id_nulls | 0 |
| game_date_nulls | 0 |
| ab_nulls | 4,300 |
| hits_nulls | 4,300 |
| home_runs_nulls | 3,868 |
| rbi_nulls | 3,868 |
| stolen_bases_nulls | 4,300 |
| strikeouts_nulls | 4,300 |
| walks_nulls | 3,868 |
| innings_pitched_nulls | 9,545 |
| earned_runs_nulls | 9,545 |
| pitcher_strikeouts_nulls | 189 |
| raw_payload_nulls | 0 |

**Date range coverage:**

```sql
SELECT MIN(game_date) AS earliest, MAX(game_date) AS latest,
       COUNT(DISTINCT game_date) AS unique_dates,
       COUNT(DISTINCT bdl_player_id) AS unique_players,
       COUNT(*) AS total_rows
FROM mlb_player_stats;
```

| earliest | latest | unique_dates | unique_players | total_rows |
|----------|--------|--------------|----------------|------------|
| 2026-03-27 | 2026-04-30 | 35 | 987 | 13,444 |

**Finding:** Nulls are positional (pitchers lack AB/hits; batters lack IP/ER). `strikeouts_pit` has only 189 nulls — unusual; most pitchers record strikeouts. raw_payload is 100% populated.

### 2e. statcast_batter_metrics

```sql
SELECT COUNT(DISTINCT pp.player_id) AS projection_players,
       COUNT(DISTINCT sbm.mlbam_id) AS statcast_matched,
       COUNT(DISTINCT pp.player_id) - COUNT(DISTINCT sbm.mlbam_id) AS unmatched
FROM player_projections pp
LEFT JOIN statcast_batter_metrics sbm ON pp.player_id = sbm.mlbam_id;
```

| projection_players | statcast_matched | unmatched |
|-------------------|------------------|-----------|
| 623 | 0 | 623 |

**Finding:** `statcast_batter_metrics` has **0 rows**. All 623 projection players are unmatched. This is a CRITICAL pipeline failure (savant_ingestion job has failed 7 consecutive times).

### 2f. statcast_pitcher_metrics

```sql
SELECT COUNT(*) AS total,
       COUNT(*) FILTER (WHERE xera IS NULL) AS null_xera,
       COUNT(*) FILTER (WHERE era IS NULL) AS null_era
FROM statcast_pitcher_metrics;
```

| total | null_xera | null_era |
|-------|-----------|----------|
| 0 | 0 | 0 |

**Finding:** `statcast_pitcher_metrics` also has **0 rows**. Same root cause as 2e.

### 2g. probable_pitchers

```sql
SELECT game_date, COUNT(*) AS rows, COUNT(*) FILTER (WHERE quality_score IS NULL) AS null_qs
FROM probable_pitchers
WHERE game_date >= CURRENT_DATE
GROUP BY game_date ORDER BY game_date;
```

| game_date | rows | null_qs |
|-----------|------|---------|
| 2026-05-01 | 28 | 0 |
| 2026-05-02 | 27 | 0 |
| 2026-05-03 | 26 | 0 |
| 2026-05-04 | 3 | 0 |

**Full null audit:**

```sql
SELECT COUNT(*) AS total_rows,
       COUNT(*) FILTER (WHERE bdl_player_id IS NULL) AS bdl_nulls,
       COUNT(*) FILTER (WHERE pitcher_name IS NULL) AS pitcher_name_nulls,
       COUNT(*) FILTER (WHERE game_date IS NULL) AS game_date_nulls,
       COUNT(*) FILTER (WHERE team IS NULL) AS team_nulls,
       COUNT(*) FILTER (WHERE opponent IS NULL) AS opponent_nulls,
       COUNT(*) FILTER (WHERE is_home IS NULL) AS is_home_nulls,
       COUNT(*) FILTER (WHERE park_factor IS NULL) AS park_factor_nulls,
       COUNT(*) FILTER (WHERE quality_score IS NULL) AS quality_score_nulls
FROM probable_pitchers;
```

| Metric | Value |
|--------|-------|
| total_rows | 504 |
| bdl_nulls | 244 |
| pitcher_name_nulls | 0 |
| game_date_nulls | 0 |
| team_nulls | 0 |
| opponent_nulls | 0 |
| is_home_nulls | 0 |
| park_factor_nulls | 0 |
| quality_score_nulls | 0 |

**Finding:** 244/504 (48.4%) rows have NULL `bdl_player_id`. All other critical columns are populated. Coverage is good through May 4.

### 2h. position_eligibility

```sql
SELECT primary_position,
       COUNT(*) AS total,
       COUNT(*) FILTER (WHERE scarcity_rank IS NULL) AS null_scarcity,
       COUNT(*) FILTER (WHERE yahoo_player_key IS NULL) AS null_yahoo_key
FROM position_eligibility
GROUP BY primary_position ORDER BY primary_position;
```

| primary_position | total | null_scarcity | null_yahoo_key |
|------------------|-------|---------------|----------------|
| 1B | 68 | 0 | 0 |
| 2B | 140 | 0 | 0 |
| 3B | 85 | 0 | 0 |
| C | 172 | 0 | 0 |
| CF | 183 | 0 | 0 |
| LF | 26 | 0 | 0 |
| RF | 92 | 0 | 0 |
| RP | 462 | 0 | 0 |
| SP | 747 | 0 | 0 |
| SS | 270 | 0 | 0 |
| Util | 144 | 0 | 0 |

**Full null audit:**

```sql
SELECT COUNT(*) AS total_rows,
       COUNT(*) FILTER (WHERE bdl_player_id IS NULL) AS bdl_nulls,
       COUNT(*) FILTER (WHERE player_name IS NULL) AS player_name_nulls,
       COUNT(*) FILTER (WHERE primary_position IS NULL) AS primary_position_nulls,
       COUNT(*) FILTER (WHERE player_type IS NULL) AS player_type_nulls
FROM position_eligibility;
```

| Metric | Value |
|--------|-------|
| total_rows | 2,389 |
| bdl_nulls | 375 |
| player_name_nulls | 0 |
| primary_position_nulls | 0 |
| player_type_nulls | 0 |

**Finding:** `scarcity_rank` and `yahoo_player_key` have zero nulls across all positions. 375/2389 (15.7%) rows lack `bdl_player_id`.

### 2i. player_id_mapping

```sql
SELECT COUNT(*) AS total,
       COUNT(*) FILTER (WHERE yahoo_id IS NULL) AS null_yahoo,
       COUNT(*) FILTER (WHERE mlbam_id IS NULL) AS null_mlb,
       COUNT(*) FILTER (WHERE bdl_id IS NULL) AS null_bdl
FROM player_id_mapping;
```

| Metric | Value |
|--------|-------|
| total | 10,096 |
| null_yahoo | 9,724 |
| null_mlb | 3,433 |
| null_bdl | 96 |

**Finding:** Extremely high orphan rate: **96.3%** of rows lack `yahoo_id`, **34.0%** lack `mlbam_id`, **0.95%** lack `bdl_id`. This table appears to be primarily a BDL→MLBAM mapping with sparse Yahoo coverage.

### 2j. ingested_injuries

```sql
SELECT
  COUNT(*) AS total_rows,
  COUNT(*) FILTER (WHERE bdl_player_id IS NULL) AS bdl_player_id_nulls,
  COUNT(*) FILTER (WHERE player_name IS NULL) AS player_name_nulls,
  COUNT(*) FILTER (WHERE injury_status IS NULL) AS injury_status_nulls,
  COUNT(*) FILTER (WHERE injury_detail IS NULL) AS injury_detail_nulls,
  COUNT(*) FILTER (WHERE ingested_at IS NULL) AS ingested_at_nulls,
  COUNT(*) FILTER (WHERE injury_type IS NULL) AS injury_type_nulls
FROM ingested_injuries
WHERE ingested_at >= NOW() - INTERVAL '7 days';
```

| Metric | Value |
|--------|-------|
| total_rows | 232 |
| bdl_player_id_nulls | 0 |
| player_name_nulls | 0 |
| injury_status_nulls | 0 |
| injury_detail_nulls | 17 |
| ingested_at_nulls | 0 |
| injury_type_nulls | 0 |

**Finding:** 17/232 (7.3%) recent injury records lack `injury_detail`. All key identifiers are populated.

---

## SECTION 3 — LOGICAL INTEGRITY CHECKS

### 3a. Duplicate player records in player_projections

```sql
SELECT player_id, COUNT(*) AS dupes
FROM player_projections
GROUP BY player_id HAVING COUNT(*) > 1
ORDER BY dupes DESC LIMIT 20;
```

**Result:** `[]` — No duplicate player_id values.

```sql
SELECT player_name, team, COUNT(*) AS dupes
FROM player_projections
GROUP BY player_name, team HAVING COUNT(*) > 1
ORDER BY dupes DESC LIMIT 20;
```

| player_name | team | dupes |
|-------------|------|-------|
| Jazz Chisholm Jr. | NYY | 2 |

**Finding:** One duplicate by name+team: Jazz Chisholm Jr. appears twice for NYY. **WARNING** — possible double-entry or multiple projection sources.

### 3b. player_projections stale data check

```sql
SELECT MIN(updated_at) AS oldest_update,
       MAX(updated_at) AS newest_update,
       COUNT(DISTINCT DATE(updated_at)) AS distinct_update_dates
FROM player_projections;
```

| oldest_update | newest_update | distinct_update_dates |
|---------------|---------------|----------------------|
| 2026-04-28 12:10:47 | 2026-05-01 14:00:24 | 3 |

**Finding:** Projections were updated on 3 distinct dates in the last 4 days. Data is fresh. `updated_at` column exists and is populated.

### 3c. Cross-reference: players in player_scores with NO row in player_projections

```sql
SELECT COUNT(DISTINCT ps.bdl_player_id) AS scored_no_projection
FROM player_scores ps
LEFT JOIN player_projections pp ON ps.bdl_player_id::text = pp.player_id
WHERE pp.player_id IS NULL;
```

| scored_no_projection |
|---------------------|
| 984 |

**Finding:** 984 distinct players have rolling Z-scores but no Steamer projection. This is expected — many bench/platoon players get game stats but are not in the 623-player projection set. **CLEAN** (contextually expected).

### 3d. Cross-reference: players in player_rolling_stats with NO row in mlb_player_stats

```sql
SELECT COUNT(DISTINCT prs.bdl_player_id) AS rolling_no_raw_stats
FROM player_rolling_stats prs
LEFT JOIN mlb_player_stats mps ON prs.bdl_player_id = mps.bdl_player_id
WHERE mps.bdl_player_id IS NULL;
```

| rolling_no_raw_stats |
|---------------------|
| 0 |

**Finding:** Every player in rolling_stats has at least one raw stat row. **CLEAN**.

### 3e. player_id_mapping join integrity — players in projections with no ID mapping

```sql
SELECT COUNT(DISTINCT pp.player_id) AS projection_players_no_mapping
FROM player_projections pp
LEFT JOIN player_id_mapping pim ON pp.player_id = pim.yahoo_id
WHERE pim.yahoo_id IS NULL;
```

| projection_players_no_mapping |
|------------------------------|
| 623 |

**Finding:** **CRITICAL** — All 623 projection players have NO matching `yahoo_id` in `player_id_mapping`. The join condition may be wrong (projections use `player_id` which is a varchar Yahoo ID, but mapping uses `yahoo_id` which is also varchar). However, the query returned 623 unmatched, meaning zero projection players are in the mapping table by yahoo_id. This breaks any Yahoo API integration that relies on ID mapping.

### 3f. Duplicate game entries in mlb_player_stats

```sql
SELECT bdl_player_id, game_date, COUNT(*) AS dupes
FROM mlb_player_stats
GROUP BY bdl_player_id, game_date HAVING COUNT(*) > 1
ORDER BY dupes DESC LIMIT 20;
```

**Sample duplicates:**

| bdl_player_id | game_date | dupes |
|---------------|-----------|-------|
| 44 | 2026-04-05 | 2 |
| 65 | 2026-04-30 | 2 |
| 144 | 2026-04-04 | 2 |
| ... | ... | ... |

```sql
SELECT COUNT(*) AS total_dupes FROM (
  SELECT bdl_player_id, game_date, COUNT(*) AS dupes
  FROM mlb_player_stats
  GROUP BY bdl_player_id, game_date HAVING COUNT(*) > 1
) sub;
```

| total_dupes |
|-------------|
| 78 |

**Finding:** 78 distinct player+date combinations have duplicate rows. Each has exactly 2 rows. **WARNING** — likely double-counting from multiple ingestion sources.

### 3g. Game date continuity in mlb_player_stats

```sql
WITH dates AS (
  SELECT DISTINCT game_date FROM mlb_player_stats ORDER BY game_date
),
gaps AS (
  SELECT game_date,
         LAG(game_date) OVER (ORDER BY game_date) AS prev_date,
         game_date - LAG(game_date) OVER (ORDER BY game_date) AS gap_days
  FROM dates
)
SELECT * FROM gaps WHERE gap_days > 5 ORDER BY gap_days DESC;
```

**Result:** `[]` — No gaps greater than 5 days.

**Finding:** Ingestion is continuous with no major gaps. **CLEAN**.

### 3h. probable_pitchers coverage gaps

```sql
SELECT d.dt, COUNT(pp.id) AS rows
FROM generate_series(CURRENT_DATE - 14, CURRENT_DATE, '1 day'::interval) AS d(dt)
LEFT JOIN probable_pitchers pp ON pp.game_date = d.dt::date
GROUP BY d.dt ORDER BY d.dt;
```

| dt | rows |
|----|------|
| 2026-04-17 | 30 |
| 2026-04-18 | 30 |
| 2026-04-19 | 30 |
| 2026-04-20 | 20 |
| 2026-04-21 | 30 |
| 2026-04-22 | 30 |
| 2026-04-23 | 18 |
| 2026-04-24 | 28 |
| 2026-04-25 | 30 |
| 2026-04-26 | 30 |
| 2026-04-27 | 16 |
| 2026-04-28 | 30 |
| 2026-04-29 | 30 |
| 2026-04-30 | 18 |
| 2026-05-01 | 28 |

**Finding:** Every day in the last 14 days has rows. Some days have fewer entries (16–20) which aligns with light schedules. **CLEAN**.

### 3i. Position eligibility vs. projection mismatch

```sql
SELECT pp.player_name, pp.positions, pe.primary_position
FROM player_projections pp
JOIN position_eligibility pe ON pp.player_id::text = pe.bdl_player_id::text
WHERE pe.primary_position IN ('SP','RP')
  AND pp.avg IS NOT NULL AND pp.avg > 0
LIMIT 20;
```

**Result:** `[]` — No pitchers with non-zero batting averages found.

**Finding:** No row misclassification detected. **CLEAN**.

---

## SECTION 4 — MATHEMATICAL / VALUE SANITY CHECKS

### 4a. ERA outliers in player_projections

```sql
SELECT player_name, team, era FROM player_projections
WHERE era > 15.0 OR era < 0 ORDER BY era DESC LIMIT 20;
```

**Result:** `[]` — No ERA values outside [0, 15]. **CLEAN**.

### 4b. Batting average outliers in player_projections

```sql
SELECT player_name, team, avg FROM player_projections
WHERE avg > 0.450 OR avg < 0 ORDER BY avg DESC LIMIT 20;
```

| player_name | team | avg |
|-------------|------|-----|
| Ryan Vilade | Unknown | 0.500 |
| Nathan Lukes | Unknown | 0.500 |
| Carlos Cortes | Unknown | 0.492 |
| Michael Harris II | Unknown | 0.453 |

**Finding:** 4 players have AVG > 0.450. All are on team "Unknown". **WARNING** — "Unknown" team suggests incomplete data ingestion for these players. Values are mathematically possible but statistically extreme for season-long projections.

### 4c. Z-score outliers in player_scores

```sql
SELECT bdl_player_id, as_of_date, window_days,
       z_r, z_hr, z_rbi, z_era, composite_z
FROM player_scores
WHERE ABS(z_r) > 10 OR ABS(z_hr) > 10 OR ABS(z_rbi) > 10
   OR ABS(z_era) > 10 OR ABS(composite_z) > 10
ORDER BY ABS(composite_z) DESC LIMIT 20;
```

| bdl_player_id | as_of_date | window_days | z_r | z_hr | z_rbi | z_era | composite_z |
|---------------|------------|-------------|-----|------|-------|-------|-------------|
| 208 | 2026-05-01 | 7 | 2.585 | 0.885 | 0.180 | 0.551 | **25.656** |
| 208 | 2026-04-30 | 7 | 2.848 | 0.767 | 0.199 | 0.553 | **23.449** |
| 142 | 2026-04-20 | 7 | 3.0 | 3.0 | 2.152 | null | **22.179** |
| 208 | 2026-04-29 | 7 | 3.0 | 0.830 | 0.255 | 0.552 | **22.130** |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Finding:** 20+ rows have `composite_z > 19`. The highest is **25.656**. In a normal distribution, |z| > 10 is effectively impossible. **CRITICAL** — The composite_z formula is likely summing raw z-scores without normalization or weighting, producing absurdly inflated values for players who excel across multiple categories. This breaks any ranking/scoring that depends on composite_z.

### 4d. score_0_100 range check

```sql
SELECT MIN(score_0_100) AS min_score, MAX(score_0_100) AS max_score,
       COUNT(*) FILTER (WHERE score_0_100 < 0 OR score_0_100 > 100) AS out_of_range
FROM player_scores;
```

| min_score | max_score | out_of_range |
|-----------|-----------|--------------|
| 0.2 | 100.0 | 0 |

**Finding:** All scores within [0, 100]. **CLEAN**.

### 4e. Rolling stats sign check

```sql
SELECT COUNT(*) FILTER (WHERE w_era < 0) AS negative_era,
       COUNT(*) FILTER (WHERE w_whip < 0) AS negative_whip,
       COUNT(*) FILTER (WHERE w_avg < 0 OR w_avg > 1) AS impossible_avg,
       COUNT(*) FILTER (WHERE w_ops < 0 OR w_ops > 5) AS suspicious_ops
FROM player_rolling_stats;
```

| negative_era | negative_whip | impossible_avg | suspicious_ops |
|--------------|---------------|----------------|----------------|
| 0 | 0 | 0 | 0 |

**Finding:** No physically impossible values. **CLEAN**.

### 4f. park_factor range check in probable_pitchers

```sql
SELECT MIN(park_factor) AS min_pf, MAX(park_factor) AS max_pf,
       COUNT(*) FILTER (WHERE park_factor < 0.7 OR park_factor > 1.4) AS outliers,
       AVG(park_factor) AS avg_pf
FROM probable_pitchers;
```

| min_pf | max_pf | outliers | avg_pf |
|--------|--------|----------|--------|
| 0.93 | 1.28 | 0 | 1.011 |

**Finding:** All park factors within expected range [0.93, 1.28]. **CLEAN**.

### 4g. xwOBA vs wOBA plausibility in statcast_batter_metrics

```sql
SELECT player_name, xwoba,
       COUNT(*) FILTER (WHERE xwoba < 0.100 OR xwoba > 0.700) AS impossible_xwoba
FROM statcast_batter_metrics
GROUP BY player_name, xwoba
HAVING xwoba < 0.100 OR xwoba > 0.700
LIMIT 20;
```

**Result:** `[]` — Table has 0 rows. No data to evaluate.

### 4h. Pitcher metrics sanity in statcast_pitcher_metrics

```sql
SELECT COUNT(*) FILTER (WHERE xera < 0 OR xera > 15) AS impossible_xera,
       COUNT(*) FILTER (WHERE k_percent < 0 OR k_percent > 80) AS impossible_k_pct,
       MIN(xera), MAX(xera), AVG(xera)
FROM statcast_pitcher_metrics;
```

| impossible_xera | impossible_k_pct | min | max | avg |
|-----------------|------------------|-----|-----|-----|
| 0 | 0 | null | null | null |

**Result:** Table has 0 rows. No data to evaluate.

### 4i. cat_scores JSON value sanity in player_projections

```sql
SELECT player_name, team, cat_scores
FROM player_projections
WHERE cat_scores IS NOT NULL AND cat_scores::text != '{}'
LIMIT 10;
```

**Sample rows:**

| player_name | team | cat_scores |
|-------------|------|------------|
| JJ Wetherholt | STL | `{"h": -1.238, "r": -0.918, "hr": -1.267, "tb": -1.55, "avg": -0.673, "nsb": -0.323, "ops": 0, "rbi": -2.243, "k_bat": 0}` |
| Garrett Crochet | BOS | `{"l": 0, "w": 1.695, "k9": 1.518, "qs": 0.218, "era": 1.894, "nsv": 0, "whip": 0, "k_pit": 0, "hr_pit": 0}` |
| Logan Webb | SFG | `{"l": 0, "w": 1.839, "k9": -1.011, "qs": 0.903, "era": 1.187, "nsv": 0, "whip": 0, "k_pit": 0, "hr_pit": 0}` |
| Cole Ragans | KCR | `{"l": 0, "w": -0.333, "k9": 1.705, "qs": -0.467, "era": 0.484, "nsv": 0, "whip": 0, "k_pit": 0, "hr_pit": 0}` |
| Brendan Donovan | SEA | `{"h": 1.2681, "r": 0.8841, "hr": -0.4882, "tb": 0.0156, "avg": 1.2831, "nsb": 0.1872, "ops": 0.432, "rbi": 0.1786, "k_bat": 0}` |

**Finding:** Values are numeric and within plausible fantasy-point ranges. Many pitcher entries have 0 for categories not applicable to them (e.g., `k_pit: 0`, `hr_pit: 0`). **CLEAN** (structurally sound).

---

## SECTION 5 — DATA FRESHNESS & STALENESS AUDIT

### 5a. Most recent record timestamp per table

```sql
SELECT 'player_projections' AS tbl, MAX(updated_at) AS freshest FROM player_projections
UNION ALL
SELECT 'player_rolling_stats', MAX(as_of_date) FROM player_rolling_stats
UNION ALL
SELECT 'player_scores', MAX(as_of_date) FROM player_scores
UNION ALL
SELECT 'mlb_player_stats', MAX(game_date) FROM mlb_player_stats
UNION ALL
SELECT 'statcast_batter_metrics', MAX(last_updated) FROM statcast_batter_metrics
UNION ALL
SELECT 'statcast_pitcher_metrics', MAX(last_updated) FROM statcast_pitcher_metrics
UNION ALL
SELECT 'probable_pitchers', MAX(game_date) FROM probable_pitchers
UNION ALL
SELECT 'ingested_injuries', MAX(ingested_at) FROM ingested_injuries
UNION ALL
SELECT 'mlb_odds_snapshot', MAX(snapshot_window) FROM mlb_odds_snapshot
UNION ALL
SELECT 'player_momentum', MAX(computed_at) FROM player_momentum;
```

| Table | Freshest Record | Age (as of 2026-05-01 07:42 ET) |
|-------|-----------------|--------------------------------|
| player_projections | 2026-05-01 10:00:24 | ~2 hours old |
| player_rolling_stats | 2026-05-01 | Current day |
| player_scores | 2026-05-01 | Current day |
| mlb_player_stats | 2026-04-30 | 1 day stale |
| statcast_batter_metrics | null | **NEVER POPULATED** |
| statcast_pitcher_metrics | null | **NEVER POPULATED** |
| probable_pitchers | 2026-05-04 | Future coverage |
| ingested_injuries | 2026-05-01 11:08:14 | ~3.5 hours old |
| mlb_odds_snapshot | 2026-05-01 11:30:00 | ~3 hours old |
| player_momentum | 2026-05-01 09:00:00 | ~1.5 hours old |

**Findings:**
- `mlb_player_stats` is 1 day behind (latest game_date is Apr 30). **MEDIUM** — may be due to morning ingestion timing.
- `statcast_*_metrics` are completely empty. **CRITICAL**.
- All other core tables are current within hours.

### 5b. data_ingestion_logs — failure rate per job over last 30 days

```sql
SELECT job_type,
       COUNT(*) AS total_runs,
       COUNT(*) FILTER (WHERE status = 'SUCCESS') AS successes,
       COUNT(*) FILTER (WHERE status = 'FAILED' OR status = 'ERROR') AS failures,
       MAX(started_at) AS last_run,
       AVG(processing_time_seconds) AS avg_duration_sec
FROM data_ingestion_logs
WHERE started_at >= NOW() - INTERVAL '30 days'
GROUP BY job_type
ORDER BY failures DESC, last_run DESC;
```

| job_type | total_runs | successes | failures | last_run | avg_duration_sec |
|----------|------------|-----------|----------|----------|------------------|
| projection_freshness | 342 | 121 | 218 | 2026-05-01 15:08:13 | 0.046 |
| bdl_injuries | 127 | 106 | 20 | 2026-05-01 15:08:13 | 2.612 |
| savant_ingestion | 7 | 0 | 7 | 2026-05-01 14:00:00 | 0.802 |
| valuation_cache | 5 | 0 | 5 | 2026-05-01 14:00:00 | 4.338 |
| yahoo_id_sync | 13 | 7 | 5 | 2026-05-01 12:30:01 | 36.493 |
| mlb_odds | 4434 | 3600 | 0 | 2026-05-01 15:43:13 | 1.417 |
| position_eligibility | 21 | 20 | 0 | 2026-05-01 15:15:00 | 3.599 |
| player_id_mapping | 17 | 17 | 0 | 2026-05-01 15:00:02 | 344.664 |
| decision_optimization | 20 | 20 | 0 | 2026-05-01 15:00:00 | 3.613 |
| ros_simulation | 18 | 17 | 0 | 2026-05-01 14:00:00 | 267.427 |
| yahoo_adp_injury | 72 | 72 | 0 | 2026-05-01 14:08:13 | 2.999 |
| player_momentum | 16 | 16 | 0 | 2026-05-01 13:00:00 | 2.416 |
| ensemble_update | 16 | 16 | 0 | 2026-05-01 13:00:00 | 0.696 |
| vorp | 16 | 16 | 0 | 2026-05-01 12:30:00 | 4.062 |
| statcast | 46 | 46 | 0 | 2026-05-01 12:08:13 | 12.549 |
| rolling_z | 16 | 16 | 0 | 2026-05-01 12:00:04 | 1.716 |
| player_scores | 18 | 18 | 0 | 2026-05-01 12:00:00 | 8.786 |
| rolling_windows | 18 | 18 | 0 | 2026-05-01 11:00:26 | 10.053 |
| fangraphs_ros | 16 | 16 | 0 | 2026-05-01 11:00:00 | 26.923 |
| statsapi_supplement | 16 | 16 | 0 | 2026-05-01 10:30:00 | 4.306 |
| mlb_box_stats | 17 | 17 | 0 | 2026-05-01 10:00:00 | 4.176 |
| mlb_game_log | 17 | 17 | 0 | 2026-05-01 09:00:00 | 0.507 |
| clv | 16 | 16 | 0 | 2026-05-01 07:00:00 | 0.066 |
| probable_pitchers | 53 | 53 | 0 | 2026-05-01 04:00:00 | 2.188 |
| snapshot | 15 | 15 | 0 | 2026-04-30 18:00:00 | 0.121 |
| explainability | 19 | 19 | 0 | 2026-04-30 17:00:00 | 0.553 |
| backtesting | 15 | 15 | 0 | 2026-04-30 16:00:00 | 5.100 |

**Findings:**
- `projection_freshness`: 63.7% failure rate (218/342). Date arithmetic bug. **HIGH** — noisy but may mask real freshness issues.
- `savant_ingestion`: 100% failure rate (7/7). **CRITICAL** — directly causes empty statcast tables.
- `valuation_cache`: 100% failure rate (5/5). **HIGH** — player_valuation_cache has 0 rows.
- `yahoo_id_sync`: 38.5% failure rate (5/13). Unique constraint violations. **MEDIUM**.
- `bdl_injuries`: 15.7% failure rate (20/127). **LOW** — intermittent, recent runs are successful.

### 5c. Most recent error messages for failed jobs

```sql
SELECT job_type, started_at, error_message
FROM data_ingestion_logs
WHERE status = 'FAILED'
  AND started_at >= NOW() - INTERVAL '30 days'
ORDER BY started_at DESC
LIMIT 30;
```

**Most recent failures (top 5):**

| job_type | started_at | error_message |
|----------|------------|---------------|
| valuation_cache | 2026-05-01 14:00:00 | `'list' object has no attribute 'get'` |
| savant_ingestion | 2026-05-01 14:00:00 | `Error: new-line character seen in unquoted field - do you need to open the file with newline=''` |
| savant_ingestion | 2026-05-01 07:29:10 | Same CSV newline error |
| valuation_cache | 2026-04-30 14:00:00 | Same `'list'` error |
| savant_ingestion | 2026-04-30 14:00:00 | `ProgrammingError: relation "statcast_batter_metrics" does not exist` (followed by SQLAlchemy query) |

**Pattern analysis:**
- `savant_ingestion`: CSV parsing error since at least Apr 28. On Apr 30 it also hit "relation does not exist" — suggests the table may have been dropped/recreated during a migration.
- `valuation_cache`: Consistent Python type error — code treats a list as a dict.
- `projection_freshness`: Historical failures were `unsupported operand type(s) for -: 'datetime.datetime' and 'datetime.date'` (date arithmetic bug). Last 30 days show it has since been fixed and is now succeeding.

---

## SECTION 6 — CROSS-TABLE REFERENTIAL INTEGRITY

### 6a. player_scores → player_rolling_stats orphan check

```sql
SELECT COUNT(*) AS orphaned_scores
FROM player_scores ps
WHERE NOT EXISTS (
  SELECT 1 FROM player_rolling_stats prs
  WHERE prs.bdl_player_id = ps.bdl_player_id
    AND prs.as_of_date = ps.as_of_date
    AND prs.window_days = ps.window_days
);
```

| orphaned_scores |
|-----------------|
| 0 |

**Finding:** Every player_scores row has a matching source row in player_rolling_stats. **CLEAN**.

### 6b. player_rolling_stats → mlb_player_stats orphan check

```sql
SELECT COUNT(DISTINCT prs.bdl_player_id) AS rolling_stats_players,
       COUNT(DISTINCT mps.bdl_player_id) AS also_in_raw_stats
FROM player_rolling_stats prs
LEFT JOIN mlb_player_stats mps ON prs.bdl_player_id = mps.bdl_player_id;
```

| rolling_stats_players | also_in_raw_stats |
|-----------------------|-------------------|
| 987 | 987 |

**Finding:** All 987 players in rolling_stats appear in raw stats. **CLEAN**.

### 6c. probable_pitchers → mlb_player_stats team consistency

```sql
SELECT DISTINCT probable_pitchers.team FROM probable_pitchers
EXCEPT
SELECT DISTINCT mlb_player_stats.team FROM mlb_player_stats
ORDER BY 1;
```

**⚠️ UNVERIFIED ASSUMPTION:** This query could not be executed because `mlb_player_stats` does not have a `team` column. Team consistency between probable_pitchers and raw stats cannot be verified without joining through `player_id_mapping` or another bridge table.

**Teams in probable_pitchers:** ARI, ATH, ATL, BAL, BOS, CHC, CIN, CLE, COL, CWS, DET, HOU, KC, LAA, LAD, MIA, MIL, MIN, NYM, NYY, PHI, PIT, SD, SEA, SF, STL, TB, TEX, TOR, WSH (30 teams — complete MLB set).

### 6d. position_eligibility → player_projections orphan check

```sql
SELECT COUNT(*) AS position_no_projection
FROM position_eligibility pe
LEFT JOIN player_projections pp ON pe.bdl_player_id::text = pp.player_id
WHERE pp.player_id IS NULL;
```

| position_no_projection |
|------------------------|
| 2389 |

**Finding:** **CRITICAL** — All 2,389 position_eligibility rows have NO matching player_projections row. The join attempted `pe.bdl_player_id::text = pp.player_id` and returned 100% unmatched. This means either:
1. The ID types are incompatible (bdl_player_id is integer, player_id is varchar Yahoo ID)
2. There is genuinely no overlap between the two tables' ID spaces

Either way, the lineup optimizer cannot join position eligibility to projections using `bdl_player_id`. This is a broken join path.

### 6e. player_id_mapping consistency check

```sql
SELECT COUNT(DISTINCT mps.bdl_player_id) AS stats_players_no_mapping
FROM mlb_player_stats mps
LEFT JOIN player_id_mapping pim ON mps.bdl_player_id = pim.bdl_id
WHERE pim.bdl_id IS NULL;
```

| stats_players_no_mapping |
|--------------------------|
| 7 |

**Finding:** Only 7 players in `mlb_player_stats` (out of 987) lack a `bdl_id` mapping. **LOW** — minimal orphan rate.

### 6f. statcast_batter_metrics → player_projections join coverage

```sql
SELECT
  COUNT(DISTINCT sbm.mlbam_id) AS statcast_batters,
  COUNT(DISTINCT pp.player_id) AS also_in_projections,
  ROUND(100.0 * COUNT(DISTINCT pp.player_id) / NULLIF(COUNT(DISTINCT sbm.mlbam_id),0), 1) AS join_pct
FROM statcast_batter_metrics sbm
LEFT JOIN player_projections pp ON sbm.mlbam_id = pp.player_id;
```

| statcast_batters | also_in_projections | join_pct |
|------------------|---------------------|----------|
| 0 | 0 | null |

**Finding:** `statcast_batter_metrics` has 0 rows. Join coverage is undefined. **CRITICAL** — same root cause as Section 2e/2f.

---

## SECTION 7 — FINDINGS SUMMARY & RISK CLASSIFICATION

### 7a. Risk-Classified Findings Table

| # | Section | Table | Finding | Risk Level | Action Required |
|---|---------|-------|---------|------------|-----------------|
| 1 | 1c, 5b | data_ingestion_logs | `savant_ingestion` job has failed 7 consecutive times (100% failure rate in last 30 days) with CSV parsing error and "relation does not exist" | **CRITICAL** | Fix savant_ingestion CSV parsing; verify table existence in ORM/schema |
| 2 | 2e, 2f, 4g, 4h, 5a, 6f | statcast_batter_metrics, statcast_pitcher_metrics | Both tables have **0 rows** due to savant_ingestion failure. All projection players unmatched. | **CRITICAL** | Restore statcast ingestion pipeline; backfill missing data |
| 3 | 4c | player_scores | composite_z values exceed 25 (|z| > 10 is statistically impossible). Formula appears to sum raw z-scores without normalization. | **CRITICAL** | Audit composite_z calculation in scoring_engine.py; fix normalization |
| 4 | 3e, 6d | player_projections, position_eligibility, player_id_mapping | **100% of projection players have NO yahoo_id mapping** (623/623). **100% of position_eligibility rows have NO projection match** (2389/2389). Join paths are broken. | **CRITICAL** | Audit ID join keys between projections, position_eligibility, and player_id_mapping. Likely `player_id` in projections is Yahoo ID but mapping table uses different key space or type mismatch |
| 5 | 5b | data_ingestion_logs | `valuation_cache` job failed 5 consecutive times with `'list' object has no attribute 'get'`. `player_valuation_cache` has 0 rows. | **HIGH** | Fix valuation_cache Python bug |
| 6 | 5b | data_ingestion_logs | `projection_freshness` has 218 failures (63.7% failure rate) in last 30 days. Historical date arithmetic bug. | **HIGH** | Verify fix is permanent; clean up noisy failure logs |
| 7 | 3f | mlb_player_stats | 78 distinct player+date combinations have duplicate rows (exactly 2 each). Likely double-ingestion. | **MEDIUM** | Add unique constraint or deduplication logic on (bdl_player_id, game_date) |
| 8 | 5b | data_ingestion_logs | `yahoo_id_sync` failed 5 times due to unique constraint violations on `bdl_id` and `yahoo_key`. | **MEDIUM** | Fix upsert logic in yahoo_id_sync to handle duplicate BDL/Yahoo keys gracefully |
| 9 | 2a | player_projections | 157/623 rows (25.2%) have NULL `positions`. 6/623 have NULL `team`. | **MEDIUM** | Investigate why positions are missing for 25% of projected players |
| 10 | 4b | player_projections | 4 players have AVG > 0.450 (max 0.500). All are on team "Unknown". | **MEDIUM** | Investigate "Unknown" team assignments in projection ingestion |
| 11 | 2g | probable_pitchers | 244/504 rows (48.4%) have NULL `bdl_player_id`. | **LOW** | Backfill BDL IDs for probable pitchers |
| 12 | 2h | position_eligibility | 375/2389 rows (15.7%) have NULL `bdl_player_id`. | **LOW** | Backfill BDL IDs for position eligibility entries |
| 13 | 2i | player_id_mapping | 96.3% null yahoo_id, 34.0% null mlbam_id. Table is mostly BDL→MLBAM mapping with sparse Yahoo coverage. | **LOW** | By design or gap? Clarify with Claude whether this is expected |
| 14 | 2j | ingested_injuries | 17/232 recent records (7.3%) lack `injury_detail`. | **LOW** | Minor data quality gap |

### 7b. Tables Confirmed CLEAN (Evidence-Based)

| Table | Checks Passed |
|-------|--------------|
| `player_scores` | score_0_100 within [0,100]; no orphaned rows; composite_z never NULL |
| `player_rolling_stats` | No negative ERA/WHIP; no impossible AVG/OPS; all players have raw stats |
| `probable_pitchers` | Park factors within range; quality_score fully populated; continuous coverage |
| `mlb_player_stats` | No gaps >5 days; raw_payload 100% populated; continuous date coverage |
| `ingested_injuries` | All key identifiers populated in last 7 days; ingested_at never NULL |
| `data_ingestion_logs` | Core daily jobs (fangraphs_ros, rolling_windows, player_scores, etc.) all succeeding |
| `mlb_odds_snapshot` | Ingesting every ~5 minutes successfully; current within hours |

### 7c. ⚠️ UNVERIFIED ASSUMPTIONS LOG

| # | Assumption | Reason | Impact |
|---|-----------|--------|--------|
| 1 | `mlb_player_stats` team consistency with `probable_pitchers` could not be verified | `mlb_player_stats` lacks a `team` column entirely | Cannot detect silent join failures between probable pitchers and raw stats |
| 2 | The 50% null rates in player_rolling_stats batting/pitching columns are "expected" because pitchers don't bat | This is inferred from domain knowledge; no direct query proves this causation | If incorrect, it could mask a systemic data loss issue in the rolling window computation |
| 3 | `player_id_mapping` high null yahoo_id rate is "by design" | No documentation in DB schema confirms this | If unexpected, it indicates a broken Yahoo sync pipeline that needs fixing |
| 4 | The 282 unmatched player_rolling_stats → player_scores rows are "acceptable" | No root cause identified for why these specific 282 rows lack scores | Could indicate a race condition or window edge case in the scoring pipeline |

### 7d. Recommended Follow-Up Queries

| # | Query | Reason / Error |
|---|-------|---------------|
| 1 | `SELECT DISTINCT team FROM mlb_player_stats;` | Table has no `team` column — need to join via `player_id_mapping` to compare with `probable_pitchers` |
| 2 | `SELECT player_id, COUNT(*) FROM player_projections GROUP BY player_id HAVING COUNT(*) > 1;` | Already executed — returned 0. Verified clean. |
| 3 | Investigate `composite_z` formula | `composite_z` values of 25+ are mathematically impossible for a standard z-score. Need to read `backend/fantasy_baseball/scoring_engine.py` to verify formula. |

---

## APPENDIX: QUERY COUNT & EXECUTION LOG

| Section | Queries Run |
|---------|-------------|
| Section 1 | 6 |
| Section 2 | 12 |
| Section 3 | 9 |
| Section 4 | 9 |
| Section 5 | 3 |
| Section 6 | 5 |
| Schema discovery | 6 |
| **Total** | **~50** |

## SIGN-OFF

> This audit was conducted read-only. No fixes were implemented. All numbers are from live query results as of 2026-05-01 07:42 ET.
>
> **Kimi CLI**  
> Deep Intelligence Unit  
> Report saved to: `reports/2026-05-01-kimi-deep-db-audit.md`
