# Pipeline Quality Audit — 2026-04-29

> **Agent:** Kimi CLI (read-only audit)  
> **Scope:** Sessions H–M structural backfills (V31 rolling stats, V32 z-scores, scarcity_rank, quality_score, MLBAM fallback)  
> **Season Day:** 35 (started 2026-03-25)

---

## 1. Executive Summary

- **V31 rolling-stats backfill is STUCK.** Only 48.6% of `player_rolling_stats` rows on the latest date have `w_runs` / `w_tb` filled. The backfill script repeatedly timed out and left uncommitted idle-in-transaction locks that blocked all subsequent UPDATEs. Claude must finish this backfill before any downstream V32/z-score logic can be trusted.
- **Z-score coverage is ~50%.** `player_scores` has complete batter z-scores for 395/797 rows (49.6%) on the 7-day window and pitcher z-scores for 405/797 (50.8%). The V32 backfill appears partially complete but needs verification after V31 is fixed.
- **scarcity_rank and quality_score are effectively unpopulated.** Only ~8–17% of position-eligibility records have `scarcity_rank`, and ZERO probable pitchers have `quality_score`. Both features are wired in code but the daily-sync path that should populate them is not running or is silently failing.
- **Cross-table join is broken.** `player_projections.player_id` (varchar, appears to be MLBAM/FG IDs) does not cast to `position_eligibility.bdl_player_id` (integer). The end-to-end pipeline currently relies on a fragile name-based fallback that covers only ~22% of eligible players.
- **Data freshness is healthy.** All core tables except `statcast_performances` (2 days stale) have been updated today.

---

## 2. Q1–Q8 Results

### Q1 — Roster Coverage (`player_projections`, last 7 days)

| Metric | Count | Rate |
|--------|-------|------|
| total_projections | 613 | — |
| has_cat_scores | 613 | 100% ✅ |
| has_team (any) | 302 | 49.3% |
| has_real_team | 302 | 49.3% |
| has_positions | 462 | 75.4% |
| **fully_complete** | **302** | **49.3%** |

**Interpretation:** Every projection has category scores, but half lack a team and a quarter lack positions. Only 302 of 613 are fully usable for lineup optimization.

---

### Q2 — Z-Score Coverage (`player_scores`, most recent window)

Latest `as_of_date` = 2026-04-29.

| window_days | total | z_r_ok | z_h_ok | z_ops_ok | z_k_p_ok | z_k_b_ok | batter_complete | pitcher_complete |
|-------------|-------|--------|--------|----------|----------|----------|-----------------|------------------|
| 7 | 797 | 399 (50.1%) | 399 (50.1%) | 395 (49.6%) | 405 (50.8%) | 399 (50.1%) | 395 (49.6%) | 405 (50.8%) |
| 14 | 861 | 424 (49.2%) | 424 (49.2%) | 418 (48.5%) | 450 (52.3%) | 424 (49.2%) | 418 (48.5%) | 450 (52.3%) |
| 30 | 960 | 455 (47.4%) | 455 (47.4%) | 447 (46.6%) | 528 (55.0%) | 455 (47.4%) | 447 (46.6%) | 528 (55.0%) |

**Interpretation:** Roughly half of all player_score rows have complete z-scores. Pitcher coverage is slightly better than batter coverage. This is consistent with a partial V32 backfill.

---

### Q3 — Rolling Stats Coverage (`player_rolling_stats`, latest date)

Latest `as_of_date` = 2026-04-29.

| Metric | Count | Rate |
|--------|-------|------|
| total | 2,628 | — |
| w_runs_ok | 1,278 | 48.6% |
| w_tb_ok | 1,278 | 48.6% |
| w_qs_ok | 1,383 | 52.6% |
| batter_complete | 1,278 | 48.6% |
| pitcher_complete | 1,383 | 52.6% |

**Interpretation:** V31 columns are populated for only ~48–53% of rows. The backfill repeatedly failed due to lock contention and script timeouts. **This blocks all downstream V32 z-score accuracy** because z-scores are derived from rolling windows.

---

### Q4 — Position Eligibility (`scarcity_rank`)

| primary_position | players | has_rank | % | avg_rank | has_rostered_pct |
|------------------|---------|----------|---|----------|------------------|
| SP | 747 | 66 | 8.8% | 6.0 | 0 |
| RP | 462 | 37 | 8.0% | 7.0 | 0 |
| CF | 183 | 31 | 16.9% | 5.0 | 0 |
| SS | 270 | 27 | 10.0% | 2.0 | 0 |
| RF | 92 | 15 | 16.3% | 9.0 | 0 |
| 1B | 68 | 15 | 22.1% | 10.0 | 0 |
| 2B | 140 | 13 | 9.3% | 3.0 | 0 |
| C | 172 | 12 | 7.0% | 1.0 | 0 |
| 3B | 85 | 12 | 14.1% | 4.0 | 0 |
| LF | 26 | 6 | 23.1% | 8.0 | 0 |
| Util | 144 | 1 | 0.7% | 99.0 | 0 |

**Interpretation:** `scarcity_rank` is populated for only ~8–23% of players per position. `league_rostered_pct` is completely empty (0% across all positions). The scarcity tiebreaker in `solve_lineup` will rarely fire.

---

### Q5 — Probable Pitchers (`quality_score`)

| game_date | pitchers | has_qs | avg_qs | min_qs | max_qs |
|-----------|----------|--------|--------|--------|--------|
| 2026-04-29 | 30 | 0 | — | — | — |
| 2026-04-30 | 16 | 0 | — | — | — |
| 2026-05-01 | 8 | 0 | — | — | — |
| 2026-05-02 | 7 | 0 | — | — | — |

**Interpretation:** `quality_score` is 0% populated across all upcoming game dates. The heuristic is wired but the daily sync that should write it is not executing.

---

### Q6 — MLBAM Fallback Residue (`team="Unknown"`)

| team | count |
|------|-------|
| null | 311 |

| unknown_team_with_cat_scores |
|------------------------------|
| 0 |

**Interpretation:** 311 projections have `NULL` team (not `"Unknown"`). The Session J fallback that should store `"Unknown"` is not running or is silently failing. No "Unknown"-team projections carry cat_scores.

---

### Q7 — End-to-End Readiness Check (name-based join)

> **Note:** The original query used `pp.player_id::integer = pe.bdl_player_id`, which returns **0 matches** because `player_projections.player_id` stores MLBAM/FG-style IDs (varchar) while `position_eligibility.bdl_player_id` stores BallDontLie integer IDs. The table below uses a `LOWER(name)` fallback join.

| primary_position | eligible_players | with_cat_scores | % | with_z_scores | % | with_scarcity_rank | % |
|------------------|------------------|-----------------|---|---------------|---|--------------------|---|
| SP | 747 | 97 | 13.0% | 202 | 27.0% | 66 | 8.8% |
| RP | 462 | 38 | 8.2% | 193 | 41.8% | 37 | 8.0% |
| SS | 270 | 64 | 23.7% | 62 | 23.0% | 27 | 10.0% |
| CF | 183 | 86 | 47.0% | 89 | 48.6% | 31 | 16.9% |
| C | 172 | 55 | 32.0% | 63 | 36.6% | 12 | 7.0% |
| Util | 144 | 3 | 2.1% | 5 | 3.5% | 1 | 0.7% |
| 2B | 140 | 56 | 40.0% | 62 | 44.3% | 13 | 9.3% |
| RF | 92 | 41 | 44.6% | 42 | 45.7% | 15 | 16.3% |
| 3B | 85 | 29 | 34.1% | 27 | 31.8% | 12 | 14.1% |
| 1B | 68 | 27 | 39.7% | 27 | 39.7% | 15 | 22.1% |
| LF | 26 | 13 | 50.0% | 13 | 50.0% | 6 | 23.1% |

**Interpretation:** Even with name-based joining, cat_score coverage is abysmal for pitchers (8–13%) and mediocre for batters (23–50%). The primary breakage is the ID-schema mismatch between projections and eligibility/scores tables.

---

### Q8 — Data Freshness

| Table | Latest Timestamp | Staleness |
|-------|------------------|-----------|
| player_projections | 2026-04-29 10:00:19 | ✅ Current |
| player_scores | 2026-04-29 00:00:00 | ✅ Current |
| player_rolling_stats | 2026-04-29 00:00:00 | ✅ Current |
| position_eligibility | 2026-04-29 20:46:43 | ✅ Current |
| probable_pitchers | 2026-04-29 12:30:01 | ✅ Current |
| statcast_performances | 2026-04-27 00:00:00 | ⚠️ 2 days stale |

---

## 3. Prioritized Gap List

| Priority | Gap | Impact | Evidence |
|----------|-----|--------|----------|
| **HIGH** | V31 rolling-stats backfill incomplete | Z-scores, waiver edge, and lineup valuation all depend on rolling windows. 51% of rows are NULL. | Q3: 1,278 / 2,628 rows missing w_runs |
| **HIGH** | Cross-table ID mismatch | `player_projections` cannot join to `position_eligibility` or `player_scores` via player ID. Lineup optimizer, waiver detector, and dashboard all join these tables. | Q7: 0 matches via `player_id::integer = bdl_player_id`; name fallback covers only 22% |
| **HIGH** | `quality_score` 0% populated | Probable-pitcher streaming recommendations rely on this field. | Q5: 0 / 61 pitchers have quality_score |
| **MED** | `scarcity_rank` < 15% coverage | Scarcity tiebreaker in `solve_lineup` and `lineup_constraint_solver` rarely fires. | Q4: 8–23% per position |
| **MED** | `league_rostered_pct` 0% | Waiver-edge detector uses this for rostered-vs-available comparison. | Q4: 0% across all positions |
| **MED** | 311 projections with NULL team | Lineup optimizer excludes these from positional slotting; waiver detector can't compute category edge. | Q6: 311 NULL teams |
| **LOW** | `statcast_performances` 2 days stale | Used for xwOBA and Statcast-derived projections. Slightly degrades projection quality. | Q8: latest = 2026-04-27 |

---

## 4. Session O Recommendations

### 4.1 Fix V31 Backfill (CRITICAL)
- **File:** `scripts/backfill_v31_rolling.py`
- **Change:** The current ORM-per-row approach leaves idle-in-transaction locks when the script times out. Rewrite to:
  1. Use raw `psycopg2` with `statement_timeout=60000`.
  2. Compute rolling windows locally, then bulk-update each date with a single `UPDATE ... SET col = CASE id ... END` per column.
  3. Commit after **each date** so a timeout only loses one date's work, not the entire run.
  4. Add `pg_terminate_backend` cleanup step in a pre-flight check to kill any lingering idle-in-transaction sessions from previous failed runs.
- **Estimated impact:** Unblocks all downstream z-score and valuation logic.

### 4.2 Fix Cross-Table Player ID Bridge (CRITICAL)
- **File:** `backend/models.py` or a new mapping table
- **Change:** `player_projections.player_id` (MLBAM/FG varchar) does not match `position_eligibility.bdl_player_id` (integer). Options:
  1. **Preferred:** Add a `mapping` table or JSON column that stores `(yahoo_player_key, bdl_player_id, mlbam_id, fg_id)` and populate it during daily sync.
  2. **Fallback:** Update the ingestion pipeline that writes `player_projections` to also store `bdl_player_id`.
  3. **Immediate workaround:** Change `daily_lineup_optimizer.py` and `waiver_edge_detector.py` joins from ID-based to name-based + team fuzzy match, with a warning log when name collisions occur.
- **Estimated impact:** Restores end-to-end pipeline connectivity. Without this, the lineup optimizer is optimizing against partial data.

### 4.3 Wire `quality_score` Daily Sync
- **File:** `backend/services/probable_pitcher_sync.py` (or wherever daily probable-pitcher ingestion lives)
- **Change:** After inserting probable pitchers, run the `quality_score` heuristic and update the `quality_score` column in the same transaction. Currently the heuristic function exists but is never called in the sync path.
- **Estimated impact:** Enables streaming-pitcher recommendations.

### 4.4 Wire `scarcity_rank` and `league_rostered_pct` Daily Sync
- **File:** `backend/services/position_eligibility_sync.py` (or equivalent)
- **Change:** Ensure the daily sync that populates `position_eligibility` also computes and writes `scarcity_rank` and `league_rostered_pct`. Currently only 8–17% of rows have these fields. Verify the sync logic isn't skipping them due to a missing `commit()` or a conditional branch.
- **Estimated impact:** Enables scarcity tiebreaker in lineup optimization.

### 4.5 Fix MLBAM Fallback to Store `"Unknown"`
- **File:** `backend/services/mlbam_team_lookup.py` (or equivalent)
- **Change:** When the MLBAM lookup fails, explicitly set `team = 'Unknown'` instead of leaving `NULL`. Also update any existing NULL teams to `'Unknown'` so the `NOT IN ('', 'Unknown')` check in Q1 correctly identifies them.
- **Estimated impact:** Cleaner data semantics; downstream queries can distinguish "not yet looked up" from "lookup failed."

### 4.6 Investigate Statcast Staleness
- **File:** `backend/services/statcast_ingest.py` (or equivalent)
- **Change:** `statcast_performances` is 2 days behind. Check if the ingestion job is failing silently or if the data source (Baseball Savant) has a lag. Add an alert if lag exceeds 24 hours.
- **Estimated impact:** Low; projections degrade slowly with 2-day-old Statcast data.

---

## 5. Raw SQL Results

### Q1
```
 total_projections | has_cat_scores | has_team | has_real_team | has_positions | fully_complete
-------------------+----------------+----------+---------------+---------------+----------------
               613 |            613 |      302 |           302 |           462 |            302
```

### Q2
```
 window_days | total | z_r_ok | z_h_ok | z_ops_ok | z_k_p_ok | z_k_b_ok | batter_complete | pitcher_complete
-------------+-------+--------+--------+----------+----------+----------+-----------------+------------------
           7 |   797 |    399 |    399 |      395 |      405 |      399 |             395 |              405
          14 |   861 |    424 |    424 |      418 |      450 |      424 |             418 |              450
          30 |   960 |    455 |    455 |      447 |      528 |      455 |             447 |              528
```

### Q3
```
 total | w_runs_ok | w_tb_ok | w_qs_ok | batter_complete | pitcher_complete
-------+-----------+---------+---------+-----------------+------------------
  2628 |      1278 |    1278 |    1383 |            1278 |             1383
```

### Q4
```
 primary_position | players | has_rank |       avg_rank       | has_rostered_pct
------------------+---------+----------+----------------------+------------------
 SP               |     747 |       66 | 6.0000000000000000   |                0
 RP               |     462 |       37 | 7.0000000000000000   |                0
 CF               |     183 |       31 | 5.0000000000000000   |                0
 SS               |     270 |       27 | 2.0000000000000000   |                0
 RF               |      92 |       15 | 9.0000000000000000   |                0
 1B               |      68 |       15 | 10.0000000000000000  |                0
 2B               |     140 |       13 | 3.0000000000000000   |                0
 C                |     172 |       12 | 1.00000000000000000000|               0
 3B               |      85 |       12 | 4.0000000000000000   |                0
 LF               |      26 |        6 | 8.0000000000000000   |                0
 Util             |     144 |        1 | 99.0000000000000000  |                0
```

### Q5
```
  game_date   | pitchers | has_qs | avg_qs | min_qs | max_qs
--------------+----------+--------+--------+--------+--------
 2026-04-29  |       30 |      0 |        |        |
 2026-04-30  |       16 |      0 |        |        |
 2026-05-01  |        8 |      0 |        |        |
 2026-05-02  |        7 |      0 |        |        |
```

### Q6
```
 team | count
------+-------
      |   311

 unknown_team_with_cat_scores
------------------------------
                            0
```

### Q7 (name-based join)
```
 primary_position | eligible_players | with_cat_scores | with_z_scores | with_scarcity_rank
------------------+------------------+-----------------+---------------+--------------------
 SP               |              747 |              97 |           202 |                 66
 RP               |              462 |              38 |           193 |                 37
 SS               |              270 |              64 |            62 |                 27
 CF               |              183 |              86 |            89 |                 31
 C                |              172 |              55 |            63 |                 12
 Util             |              144 |               3 |             5 |                  1
 2B               |              140 |              56 |            62 |                 13
 RF               |               92 |              41 |            42 |                 15
 3B               |               85 |              29 |            27 |                 12
 1B               |               68 |              27 |            27 |                 15
 LF               |               26 |              13 |            13 |                  6
```

### Q8
```
        tbl           |         latest
----------------------+-------------------------
 player_projections   | 2026-04-29 10:00:19.921
 player_scores        | 2026-04-29 00:00:00
 player_rolling_stats | 2026-04-29 00:00:00
 position_eligibility | 2026-04-29 20:46:43.259
 probable_pitchers    | 2026-04-29 12:30:01.219
 statcast_performances| 2026-04-27 00:00:00
```

---

*Report generated by Kimi CLI at 2026-04-29. Read-only audit — no data modified.*
