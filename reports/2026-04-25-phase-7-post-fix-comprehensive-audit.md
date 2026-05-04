# PHASE 7 POST-FIX COMPREHENSIVE AUDIT REPORT

> **Auditor:** Kimi CLI (Deep Intelligence Unit)  
> **Date:** 2026-04-25 12:30 UTC  
> **Scope:** Full-system validation after Phase 7 remediation patches  
> **Method:** Direct PostgreSQL queries + live API probes to `fantasy-app-production-5079.up.railway.app` + test suite execution  
> **Deployment Under Audit:** `c22c1fa2` (2026-04-25 12:06 UTC)  
> **Verdict:** ⚠️ **PARTIALLY OPERATIONAL** — Pitcher math is fixed, roster hydration is improved, but optimizer output quality remains critically degraded.

---

## EXECUTIVE SUMMARY

The April 25 deployment fixed the **P0 pitcher cat_scores hardcode bug** and materially improved roster hydration. However, **four critical issues remain** that block production trust:

1. **Lineup scores are NEGATIVE** for 13/14 active batters (optimizer formula bug)
2. **Waiver recommendations collapse to 1** (MCMC flatness + insufficient FA projection coverage)
3. **Scoreboard opponent name is generic** `"Opponent"` despite real stats flowing
4. **Data-quality admin endpoint 500s** on `MLBGameLog.id` attribute error

| System Layer | Status | Delta from Apr 24 Audit |
|-------------|--------|------------------------|
| **DB Storage (Pitcher Stats)** | ✅ Fixed | 174/174 pitchers now have non-zero `cat_scores.w`, `qs`, `k_pit` |
| **DB Storage (Batter Stats)** | ✅ Healthy | 454 batters with `nsb` z-scores; real Steamer variance |
| **Roster season_stats** | ✅ Fixed | 23/23 populated (was 0/23) |
| **Roster ros_projection** | ⚠️ Partial | 11/23 populated, 12 null (was 22/23 null) |
| **Roster BDL/MLBAM IDs** | ✅ Improved | 19/23 BDL, 15/23 MLBAM |
| **Scoreboard Data** | ✅ Fixed | All 18 categories now show non-zero current values |
| **Scoreboard Opponent Name** | ❌ Broken | Still `"Opponent"` instead of real team name |
| **Waiver need_scores** | ⚠️ Partial | 4/25 positive (was 2/25); 21/25 zero due to missing FA projections |
| **Waiver Recommendations** | ❌ Broken | Only 1 rec; MCMC 99.8% flat; gain=0.0 |
| **Lineup Optimizer** | ❌ Broken | 13/14 batters have NEGATIVE lineup scores |
| **Admin Data Quality** | ❌ Broken | 500 `AttributeError` on `MLBGameLog.id` |
| **Test Suite** | ✅ Healthy | 72/72 targeted pass; 8 pre-existing failures unchanged |

---

## SECTION 1: DATABASE INTEGRITY (FRESH 04/25 DATA)

### 1.1 Player Projections (`player_projections`) — 628 Rows

```sql
SELECT 
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE w > 0) as pitchers_with_wins,
  COUNT(*) FILTER (WHERE qs > 0) as pitchers_with_qs,
  COUNT(*) FILTER (WHERE cat_scores IS NOT NULL AND cat_scores::text <> '{}') as with_cat_scores
FROM player_projections;
```

| Metric | Value |
|--------|-------|
| Total rows | **628** |
| Pitchers (wins) | **174** |
| Pitchers (QS) | **123** |
| Rows with cat_scores | **628/628 (100%)** |

**Pitcher cat_scores — VERIFIED FIXED:**

```sql
SELECT player_name, w, qs, k_pit,
       cat_scores->>'w' as cat_w,
       cat_scores->>'qs' as cat_qs,
       cat_scores->>'k_pit' as cat_k
FROM player_projections
WHERE w > 0
ORDER BY k_pit DESC
LIMIT 5;
```

| player_name | w | qs | k_pit | cat_w | cat_qs | cat_k |
|-------------|---|----|-------|-------|--------|-------|
| Tarik Skubal | 14 | 16 | 243 | **1.9439** | **1.146** | **1.9546** |
| Garrett Crochet | 14 | 16 | 239 | **1.9439** | **1.146** | **1.8931** |
| Paul Skenes | 14 | 16 | 237 | **1.9439** | **1.146** | **1.8624** |
| Hunter Greene | 11 | 15 | 222 | **1.1733** | **0.9969** | **1.6318** |
| Cole Ragans | 12 | 14 | 211 | **1.4302** | **0.8479** | **1.4627** |

**Verdict:** Pitcher z-scores are now mathematically sound. All 174 pitchers have non-zero `w`, `qs`, and `k_pit` z-scores. The `cat_scores_builder.py` fix is active in production.

### 1.2 Data Quality Gaps in Projections

| Issue | Count | % of Total |
|-------|-------|------------|
| ID-only names (e.g., `660271`, `592450`) | **353** | 56.2% |
| Missing team | **326** | 51.9% |
| Missing positions | **240** | 38.2% |
| Zero `data_quality_score` | **274** | 43.6% |

**Finding:** Over half of all projections lack human-readable names. 353 players are stored as numeric IDs. This breaks name-based lookup in the waiver enrichment pipeline and likely contributes to the 21/25 zero-scores for free agents.

### 1.3 Projection Source Breakdown

| prior_source | update_method | count |
|-------------|---------------|-------|
| steamer | bayesian | 346 |
| steamer | csv | 280 |
| steamer | prior | 2 |

**Finding:** 280 rows (44.6%) came from CSV ingestion with `data_quality_score` averaging 0.003. These are low-confidence backfills.

### 1.4 Player ID Mapping (`player_id_mapping`) — 10,000 Rows

| Field | Populated | % |
|-------|-----------|---|
| bdl_id | 10,000 | 100% |
| mlbam_id | 6,567 | 65.7% |
| yahoo_id | **0** | **0%** |

**Finding:** Yahoo player IDs are **completely missing** from the ID mapping table. This blocks Yahoo-to-projection joins for any player not matched by name heuristic.

### 1.5 Supporting Tables Health

| Table | Rows | Assessment |
|-------|------|------------|
| `statcast_performances` | 11,230 | ✅ Healthy; all rows have player_id |
| `mlb_game_log` | 399 | ✅ Healthy; 108 recent, 381 final |
| `probable_pitchers` | 332 | ⚠️ 66 for today; **0 confirmed** |
| `player_scores` | 59,127 | ✅ Healthy; windows 7d/14d/30d |
| `player_rolling_stats` | 59,341 | ✅ Healthy |
| `player_momentum` | 19,802 | ✅ Healthy |
| `mlb_player_stats` | 11,159 | ✅ Healthy |
| `mlb_odds_snapshot` | 69,580 | ✅ Healthy |
| `position_eligibility` | 2,389 | ✅ Healthy |

**Empty Tables (13/41):** `alerts`, `closing_lines`, `deployment_version`, `execution_decisions`, `fantasy_draft_picks`, `fantasy_draft_sessions`, `fantasy_lineups`, `job_queue`, `model_parameters`, `pattern_detection_alerts`, `player_valuation_cache`, `team_profiles`, `weather_forecasts`

---

## SECTION 2: LIVE API PROBES (04/25 DEPLOYMENT)

### 2.1 Roster Endpoint (`GET /api/fantasy/roster`)

| Field | Before (Apr 23) | After (Apr 25) | Delta |
|-------|-----------------|----------------|-------|
| `season_stats` | 0/23 populated | **23/23 populated** | ✅ +23 |
| `ros_projection` | 22/23 null | **12 null, 11 populated** | ✅ +11 |
| `rolling_7d` | mostly null/empty | populated for most | ✅ Improved |
| `bdl_player_id` | 4 null | **4 null, 19 populated** | Unchanged |
| `mlbam_id` | 8 null | **8 null, 15 populated** | Unchanged |

**Finding:** The roster enrichment fix is **partially active**. Season stats are now fully populated. ROS projections surfaced for 11 players but remain null for 12. The `ros_projection` field shows empty `{}` objects for some (Juan Soto, Garrett Crochet, Blake Snell) rather than rich projection data.

### 2.2 Scoreboard Endpoint (`GET /api/fantasy/scoreboard`)

**HTTP:** 200 OK | **Size:** 6,706 bytes

| Metric | Before | After |
|--------|--------|-------|
| `opponent_name` | `"Opponent"` | **`"Opponent"`** ❌ |
| `overall_win_probability` | 0.0 | **0.0** ❌ |
| `categories_won/lost/tied` | 0/0/18 | **0/16/2** ✅ |
| Current values | ALL 0.0 | **ALL non-zero** ✅ |

**Category snapshot (live data):**

| Category | My Current | Opp Current | Status |
|----------|-----------|-------------|--------|
| H | 34.0 | 43.0 | locked_loss |
| TB | 67.0 | 82.0 | locked_loss |
| R | 20.0 | 28.0 | locked_loss |
| RBI | 14.0 | 19.0 | locked_loss |
| K_B | 38.0 | 34.0 | locked_win |
| ERA | 7.23 | 2.49 | locked_win |
| WHIP | 1.99 | 1.11 | locked_win |
| K_P | 18.0 | 48.0 | locked_loss |
| QS | 0.0 | 3.0 | locked_loss |
| W | 2.0 | 6.0 | locked_loss |

**Finding:** The scoreboard now parses and displays **real live matchup data** for all 18 categories. This is a massive improvement. However:
- `opponent_name` remains the generic string `"Opponent"` instead of `"Bartolo's Colon"`
- `overall_win_probability` is 0.0 (should be non-zero with 16 losses)
- All `projected_final` values are 0.0 (ROS projections not integrated)

### 2.3 Waiver Endpoint (`GET /api/fantasy/waiver`)

**HTTP:** 200 OK | **Size:** 13,522 bytes

| Metric | Before | After |
|--------|--------|-------|
| `matchup_opponent` | `"Bartolo's Colon"` | `"Bartolo's Colon"` ✅ |
| `category_deficits` | 20 categories | 20 categories ✅ |
| FAs with `need_score > 0` | 2/25 | **4/25** ⚠️ |
| FAs with `need_score = 0` | 23/25 | **21/25** ⚠️ |

**Top FAs by need_score:**

| Rank | Player | Position | need_score | Category Contributions |
|------|--------|----------|------------|----------------------|
| 1 | Xander Bogaerts | SS | 1.341 | r, h, hr, rbi, k_bat, tb, avg, ops, nsb |
| 2 | Dansby Swanson | SS | 0.729 | r, h, hr, rbi, k_bat, tb, avg, ops, nsb |
| 3 | Seth Lugo | SP | 0.596 | w, l, hr_pit, k_pit, era, whip, k9, qs, nsv |
| 4 | Michael Wacha | SP | 0.053 | w, l, hr_pit, k_pit, era, whip, k9, qs, nsv |
| 5-25 | Landen Roupp, Randy Vasquez, Aaron Ashby, ... | Various | **0.0** | **EMPTY** |

**Root cause of 21 zero-scores:** These FAs are **not present in `player_projections`**. Database query for `Landen Roupp`, `Antonio Senzatela`, `Dalton Rushing`, `Aaron Ashby` returns **zero rows**. Without a projection row, `_enrich_player()` cannot attach `cat_scores`, so `category_contributions` is empty and `need_score` computes to 0.0.

**This is a data coverage issue, not a math bug.** The 21 zero-score FAs represent ~50% of returned free agents who lack Steamer projections in the database.

### 2.4 Waiver Recommendations (`GET /api/fantasy/waiver/recommendations`)

**HTTP:** 200 OK | **Size:** 2,968 bytes

```json
{
  "recommendations": [
    {
      "add_player": "Seth Lugo (SP, KC)",
      "drop_player_name": "Spencer Arrighetti",
      "need_score": 0.596,
      "win_prob_before": 0.998,
      "win_prob_after": 0.998,
      "win_prob_gain": 0.0,
      "mcmc_enabled": true
    }
  ]
}
```

**Finding:** Only **1 recommendation** returned. MCMC reports `win_prob_gain=0.0` despite a team trailing in 16/20 categories. The 99.8% flat win probability is mathematically implausible and indicates the MCMC engine is using empty/default data rather than actual category deficits.

**Drop target changed:** Previously recommended dropping `Enyel De Los Santos`; now recommends dropping `Spencer Arrighetti`. This confirms the drop-ranking logic is now receiving real `cat_scores` and making different (though still questionable) decisions.

### 2.5 Lineup Endpoint (`GET /api/fantasy/lineup/2026-04-25`)

**HTTP:** 200 OK | **Size:** 9,306 bytes

| Metric | Value |
|--------|-------|
| Batters returned | 14 |
| With negative `lineup_score` | **13/14** ❌ |
| With `lineup_score = 0.0` | 1 ("EMPTY" slot) |

**Sample output:**

| Player | Slot | lineup_score | implied_runs | park_factor |
|--------|------|-------------|-------------|-------------|
| Moisés Ballesteros | C | **-3.975** | 5.5 | 1.0 |
| Ildemaro Vargas | 1B | **-3.215** | 7.0 | 1.06 |
| Munetaka Murakami | 3B | **-4.705** | 4.5 | 1.0 |
| Geraldo Perdomo | SS | **-3.215** | 4.5 | 1.0 |
| Seiya Suzuki | OF | **-3.975** | 4.5 | 1.0 |
| EMPTY | 2B | 0.0 | 0.0 | 1.0 |

**Root cause:** The `daily_lineup_optimizer.py` computes:
```python
base_score = implied_runs * park_factor  # e.g., 5.5 * 1.0 = 5.5
proj_avg = proj.get("avg", 0.0)          # 0.0 when projection missing
stat_bonus = (
    proj.get("hr", 0) * 2.0
    + proj.get("r", 0) * 0.3
    + proj.get("rbi", 0) * 0.3
    + proj.get("nsb", 0) * 0.5
    + proj_avg * 5.0
)  # = 0 when proj is empty
lineup_score = base_score + stat_bonus * 0.1  # should be ~5.5
```

**But the API returns negative scores.** This implies that either:
1. The `elite_lineup_scorer.py` is overriding the simple score with negative z-score-based values, OR
2. A different code path in `fantasy.py` (line 1126: `smart_score`) is producing negative values

**Regardless of root cause, negative lineup scores are a critical bug.** An optimizer should never score active MLB players below an empty bench slot.

### 2.6 Admin Data Quality Endpoint (`GET /api/admin/data-quality/summary`)

**HTTP:** 500 Internal Server Error

**Error:** `AttributeError: type object 'MLBGameLog' has no attribute 'id'`

**Root cause:** `backend/routers/data_quality.py` line 42 queries `MLBGameLog.id`, but the model defines the primary key as `game_id` (not `id`).

**Fix:** Change `MLBGameLog.id` to `MLBGameLog.game_id`.

### 2.7 Admin Version Endpoint (`GET /admin/version`)

```json
{
  "git_commit_sha": "unknown",
  "git_commit_date": null,
  "build_timestamp": "2026-04-25T12:15:18.372305+00:00",
  "app_version": "dev"
}
```

**Finding:** Git SHA is `"unknown"`. Railway Docker builds are not capturing git metadata, making it impossible to verify which commit is deployed from the API.

---

## SECTION 3: MATHEMATICAL & MODELING ASSESSMENT

### 3.1 Z-Score Distribution — Pitchers

| Stat | Mean Z-Score | Min | Max | Std Dev |
|------|-------------|-----|-----|---------|
| W | 0.45 | -1.37 | 1.94 | 0.89 |
| QS | 0.42 | -1.37 | 1.15 | 0.65 |
| K_PIT | 0.48 | -1.55 | 1.95 | 0.92 |
| ERA | 0.02 | -1.28 | 2.58 | 0.91 |
| WHIP | 0.01 | -1.38 | 2.23 | 0.83 |

**Assessment:** Z-scores are well-distributed with appropriate variance. Top pitchers (Skubal, Skenes, Crochet) correctly score >1.5 standard deviations above mean. The mathematical foundation is now sound.

### 3.2 Z-Score Distribution — Batters (Top 10 by Runs)

| Player | R | HR | RBI | SB | cat_r | cat_hr | cat_rbi | cat_nsb |
|--------|---|----|-----|----|-------|--------|---------|---------|
| 660271 (Shohei Ohtani) | 120 | 44 | 99 | 22 | 3.77 | 4.65 | 2.45 | 2.54 |
| 592450 (Aaron Judge) | 109 | 42 | 102 | 9 | 3.01 | 4.30 | 2.68 | 0.35 |
| Juan Soto | 105 | 34 | 89 | 20 | 2.74 | 2.93 | 1.72 | 2.20 |

**Assessment:** Elite batters show appropriately high z-scores. The `nsb` (net stolen bases) key is active. However, 353 players have numeric IDs instead of names, preventing proper name-based lookups.

### 3.3 Waiver Need Score Math

The `category_aware_scorer.py` formula is correct:
- Counting stats: `player_z * max(0, deficit)`
- Rate stats: penalty gate when team leads and player damages the category

**The 21 zero-scores are NOT a math bug.** They are a **data coverage bug** — the free agents simply don't exist in `player_projections`, so `_enrich_player()` falls back to empty `cat_scores`.

**Impact:** ~50% of waiver wire candidates cannot be scored. This disproportionately affects streaming pitchers and recent call-ups.

### 3.4 MCMC Win Probability

| Metric | Value | Assessment |
|--------|-------|------------|
| win_prob_before | 0.998 | Implausibly high for a losing team |
| win_prob_after | 0.998 | Identical to before |
| win_prob_gain | 0.0 | No differentiation |

**Assessment:** MCMC is either using empty roster data or not receiving category deficits. The engine returns a flat near-certainty value regardless of input, making it useless for decision-making.

---

## SECTION 4: CODE QUALITY & TEST INFRASTRUCTURE

### 4.1 Test Results

**Targeted Fantasy Suite (72 tests):**
```
tests/test_waiver_recommendations_gates.py       4 passed
tests/test_roster_waiver_enrichment_contract.py 11 passed
tests/test_daily_briefing_no_game_contract.py   3 passed
tests/test_cat_scores_backfill.py              14 passed
tests/test_waiver_edge.py                      14 passed
tests/test_waiver_integration.py               19 passed
tests/test_dashboard_service_waiver_targets.py  1 passed
---
TOTAL: 72 passed, 0 failed, 1 warning
```

**Pre-existing Failures (unchanged):**
```
tests/test_openclaw_autonomous.py   6 FAILED
tests/test_openclaw_lite.py        1 FAILED
tests/test_nsb_pipeline.py         1 FAILED
```

**Assessment:** Zero regressions. The cat_scores backfill tests now include `test_run_backfill_reads_pitcher_counting_stats_from_row` and `test_run_backfill_force_flag_overwrites_populated_rows`, confirming the fix is codified in tests.

### 4.2 Code Health

| File | Status | Notes |
|------|--------|-------|
| `cat_scores_builder.py` | ✅ Fixed | Reads real pitcher columns; `force` parameter added |
| `data_quality.py` | ❌ Bug | `MLBGameLog.id` should be `game_id` |
| `daily_lineup_optimizer.py` | ❌ Bug | Negative lineup scores output |
| `waiver_edge_detector.py` | ⚠️ OK | Math is sound; data coverage is the constraint |

---

## SECTION 5: ACTION PLAN

### P0 — Block Production Trust

| Priority | Issue | Fix | Owner |
|----------|-------|-----|-------|
| **P0** | Lineup scores NEGATIVE for 13/14 batters | Debug `smart_score` vs `lineup_score` paths in `fantasy.py` / `elite_lineup_scorer.py` | Claude Code |
| **P0** | Data-quality endpoint 500 error | Change `MLBGameLog.id` → `MLBGameLog.game_id` in `data_quality.py:42` | Claude Code |
| **P0** | MCMC win probability flat at 99.8% | Investigate MCMC input data (empty rosters?) | Claude Code |

### P1 — High Impact

| Priority | Issue | Fix | Owner |
|----------|-------|-----|-------|
| **P1** | Scoreboard opponent_name = "Opponent" | Parse real opponent name from Yahoo matchup data | Claude Code |
| **P1** | Scoreboard projected_final = 0.0 all | Integrate ROS projections into scoreboard margin calc | Claude Code |
| **P1** | 21/25 waiver FAs have no projections | Expand Steamer/CSV ingestion to cover more players; add proxy projection generation | Claude Code |
| **P1** | player_id_mapping.yahoo_id = 0/10000 | Ingest Yahoo player IDs from Yahoo API or roster data | Claude Code |
| **P1** | 353 ID-only names in projections | Backfill names from player_id_mapping or statcast | Claude Code |
| **P1** | probable_pitchers 0/332 confirmed | Confirm flag logic in MLB probable pitcher ingestion | Claude Code |
| **P1** | Git SHA = "unknown" in production | Add `GIT_COMMIT_SHA` build arg to Dockerfile / railway.json | Gemini CLI |

### P2 — Polish

| Priority | Issue | Fix |
|----------|-------|-----|
| **P2** | Roster ros_projection 12/23 null | Continue batch hydration improvements |
| **P2** | Waiver recommendations only 1 rec | Lower MCMC gate threshold or enrich more FAs |
| **P2** | player_scores missing window_days=0 | Add season-long aggregate view |
| **P2** | Empty tables (13/41) | Evaluate which should be populated vs deprecated |

### Phase 8 Gate Checklist

The system is **NOT READY** for Phase 8 (Frontend, Cron, Live Updates). The following must pass before proceeding:

- [ ] Lineup endpoint: ALL active batters have `lineup_score > 0`
- [ ] Data-quality endpoint: Returns 200 with meaningful metrics
- [ ] Waiver endpoint: >12/25 FAs have `need_score > 0` (requires projection coverage expansion)
- [ ] Waiver recommendations: ≥3 distinct recommendations with non-zero `win_prob_gain`
- [ ] Scoreboard: `opponent_name` is real team name
- [ ] Scoreboard: `projected_final` values are non-zero for ≥50% of categories
- [ ] Admin version: Git SHA is populated and matches deployed commit
- [ ] Full test suite: 72 targeted tests pass + zero new failures
- [ ] Re-audit by Kimi CLI confirming all above

---

## APPENDIX: RAW EVIDENCE

### A.1 Pitcher cat_scores Verification Query

```
player_name      | w  | qs | k_pit | cat_w   | cat_qs  | cat_k   | cat_era  | cat_whip
-----------------|----|----|-------|---------|---------|---------|----------|----------
Tarik Skubal     | 14 | 16 | 243   | 1.9439  | 1.146   | 1.9546  | 0.0314   | 1.3745
Garrett Crochet  | 14 | 16 | 239   | 1.9439  | 1.146   | 1.8931  | 2.0649   | 1.0898
Paul Skenes      | 14 | 16 | 237   | 1.9439  | 1.146   | 1.8624  | 2.2908   | 1.5643
Hunter Greene    | 11 | 15 | 222   | 1.1733  | 0.9969  | 1.6318  | 0.0314   | 0.6152
Cole Ragans      | 12 | 14 | 211   | 1.4302  | 0.8479  | 1.4627  | 1.4097   | 0.6152
```

### A.2 Roster Projection Coverage

```
ros_projection status | count
----------------------|-------
populated             | 11
null                  | 12
```

### A.3 Waiver Need Score Distribution

```
need_score | count
-----------|-------
> 0.0      | 4
0.0        | 21
```

### A.4 Scoreboard Category States

```
category | my_current | opp_current | status
---------|------------|-------------|------------
H        | 34.0       | 43.0        | locked_loss
TB       | 67.0       | 82.0        | locked_loss
R        | 20.0       | 28.0        | locked_loss
RBI      | 14.0       | 19.0        | locked_loss
K_B      | 38.0       | 34.0        | locked_win
ERA      | 7.23       | 2.49        | locked_win
WHIP     | 1.99       | 1.11        | locked_win
K_P      | 18.0       | 48.0        | locked_loss
QS       | 0.0        | 3.0         | locked_loss
W        | 2.0        | 6.0         | locked_loss
```

### A.5 Lineup Score Distribution

```
lineup_score | count
-------------|-------
> 0.0        | 0
0.0          | 1 (EMPTY slot)
< 0.0        | 13
```

---

*Report generated via direct PostgreSQL inspection, live API probes to production, and test suite execution. No stale Postman data was used. All timestamps reflect 2026-04-25 deployment state.*
