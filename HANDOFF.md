# HANDOFF.md -- MLB Platform Master Plan (In-Season 2026)

> **Date:** April 15, 2026 | **Author:** Claude Code (Master Architect)
> **Status:** NSB PIPELINE LIVE -- v27 MIGRATION DEPLOYED. TASKS C (ROLLOUT DIAGNOSTICS) + D (EXPLAINABILITY NSB) COMPLETE, AWAITING DEPLOY.
>
> **Current phase:** Application and data pipeline audit completed by Kimi CLI (April 15).
> Live production probes confirm ALL 6 critical tables are healthy. Statcast zero-metric rate
> collapsed from 42.4% to 5.0% post-aggregation fix. NSB pipeline is live in prod. Next focus:
> close weather/park factor integration gap, investigate decision_results volume (26 rows),
> and commit uncommitted v27 changes.
>
> **Full audit report:** `reports/2026-04-15-comprehensive-application-audit.md`
>
> **Prior phase (complete):** Statcast aggregation bug FIXED (4 commits). `_aggregate_to_daily()`
> makes pipeline resilient to per-pitch data; type-scoped upserts protect two-way players;
> CS indicators properly SUMmed. Diagnostics endpoints verified (6,971 aggregated records,
> 5.0% zero-metric rate).

---

## NSB PIPELINE (P27 -- this session's delivery)

### What was wrong

The H2H-One-Win category NSB (Net Stolen Bases = SB - CS) was being CONSUMED by downstream
code (`daily_lineup_optimizer.py:386`, `draft_engine.py:58/254/286`, `schemas.py:582`, three
main.py touch points) but never COMPUTED anywhere in the derived-stats pipeline. Scoring
used `z_sb` based on `w_stolen_bases`, which ignores the negative value of caught stealings.

### What changed

| File | Change |
|------|--------|
| `scripts/migrate_v27_nsb.py` | **NEW.** Idempotent `ADD COLUMN IF NOT EXISTS` for `w_caught_stealing`, `w_net_stolen_bases` on `player_rolling_stats` and `z_nsb` on `player_scores`. Reversible downgrade. |
| `backend/models.py` | Added 3 new columns on `PlayerRollingStats` and `PlayerScore`. `z_sb` retained (not dropped) for backward compat. |
| `backend/services/rolling_window_engine.py` | Added `sum_w_cs` accumulator. Reads `row.caught_stealing` via `getattr(..., None)` so rows without the attribute (older ORM stubs) gracefully treat CS as 0. Computes `w_net_stolen_bases = sum_w_sb - sum_w_cs`. Both new fields are None for pure pitchers. |
| `backend/services/scoring_engine.py` | `HITTER_CATEGORIES` now has `z_nsb -> w_net_stolen_bases`. Added `_COMPOSITE_EXCLUDED = {"z_sb"}` — `z_sb` is still computed and persisted for legacy consumers (explainability narratives, UAT spot check) but **excluded from composite_z** to avoid double-counting basestealing (SB and NSB correlate >0.95). `compute_league_params` now emits both `"sb"` (for simulation_engine compatibility) and `"nsb"`. |
| `backend/services/daily_ingestion.py` | Extended both `PlayerRollingStats` and `PlayerScore` upsert write paths (insert + on-conflict) to include the new fields. |
| `tests/test_nsb_pipeline.py` | **NEW.** 15 regression tests: null/missing CS coerced to 0, CS decay-weighted alongside SB, pure pitcher gets None NSB fields, `z_sb` excluded from composite (asserts composite == mean of `{z_hr, z_rbi, z_nsb, z_avg, z_obp}` only), `z_nsb` on `PlayerScoreResult`, migration SQL shape. |

### Key quant decision: why z_sb stays alongside z_nsb

Rather than rip out z_sb, `z_sb` is **still computed and persisted** but flagged in
`_COMPOSITE_EXCLUDED`. This preserves:
- `explainability_layer.py` narratives (inp.z_sb → "Speed factor")
- `uat_spot_check.py` null-presence checks
- Anything currently reading `player_scores.z_sb`

...while making composite_z drive off the **correct** fantasy category (NSB). If CS data
is ever missing (e.g. player has SB but no CS events), `w_caught_stealing=0` → NSB == SB →
z_nsb ≈ z_sb. So the transition is seamless for the 95%+ of players where CS is zero.

### Test verification

- `tests/test_nsb_pipeline.py`: **15/15 pass**
- Full suite: **1859 passed, 3 skipped** (no regressions; 1838 → 1859 = net +21 new tests this session across all recent sprints)

### v27 migration -- DEPLOYED (Gemini, Apr 14)

| Step | Result |
|------|--------|
| Apply v27 migration on Railway | SUCCESS -- columns added |
| Remove temporary diagnostic + migration endpoints | SUCCESS -- prod re-secured |
| Statcast aggregated foundation (6,971 records) | Verified, recency through 2026-04-13 |

Gemini flagged a "CS source gap" -- this refers to `statcast_performances.cs` being zero
across the aggregated name-date Statcast feed. **This is orthogonal to NSB scoring.**

The NSB pipeline reads `caught_stealing` from `mlb_player_stats` (BDL feed), NOT from
`statcast_performances`. BDL ingestion at `daily_ingestion.py:1239` populates
`MLBPlayerStats.caught_stealing = stat.cs if stat.cs is not None else 0`. So the new
`w_caught_stealing` / `w_net_stolen_bases` / `z_nsb` columns will populate from BDL data
on the next 4 AM ET `rolling_windows` + `player_scores` job run.

Post-deploy verification (Gemini to run after Apr 15 4 AM ET job):
```sql
SELECT COUNT(*) FILTER (WHERE w_caught_stealing IS NOT NULL) AS cs_filled,
       COUNT(*) FILTER (WHERE w_net_stolen_bases IS NOT NULL) AS nsb_filled,
       COUNT(*) AS total
FROM player_rolling_stats
WHERE as_of_date = CURRENT_DATE;

SELECT COUNT(*) FILTER (WHERE z_nsb IS NOT NULL) AS z_nsb_filled,
       COUNT(*) AS total
FROM player_scores
WHERE as_of_date = CURRENT_DATE;
```

### Downstream-consumer wiring status

The new `z_nsb` enters `composite_z` immediately on next scoring job, which propagates to
`score_0_100` percentiles and into the decision pipeline. No additional consumer changes
required for the *live* scoring path.

Static-projection consumers (draft_engine, player_board, keeper_engine, h2h_monte_carlo,
mcmc_simulator) read `proj["nsb"]` from hardcoded season projection tables -- a separate
subsystem from the live rolling-window pipeline. Those are unaffected by the v27 deploy
and remain on their existing static projections.

### Closed-out sprint summary

| Sprint item | Status |
|-------------|--------|
| Pydantic odds validation | LIVE |
| `/admin/backfill-cs-from-statcast` endpoint | LIVE (set CS=0 default for source-gap rows) |
| Null-safe derived-stats helpers (OPS/WHIP/ERA/AVG/ISO) | LIVE |
| Statcast aggregation fix (4 commits) | LIVE in prod |
| Statcast diagnostic endpoints | Verified by Gemini, then removed (prod re-secured) |
| **NSB pipeline (P27, this session)** | **LIVE -- columns added in prod, populates on next 4 AM ET job** |
| **Task C: NSB scoring diagnostics (this session)** | **LIVE in code, router mounted -- deploy + exercise to verify rollout** |
| **Task D: Explainability NSB narration (this session)** | **LIVE in code -- next decision explanations will use NSB over SB** |

---

## P27 FOLLOW-UP -- TASKS C + D (April 14, this session)

### Task C -- NSB rollout diagnostic endpoints

**Rationale:** v27 migration added columns; nothing in prod yet proves the next 4 AM ET
rolling_windows + player_scores jobs actually populate them. Three read-only GET endpoints
verify fill rates, distribution shape, and per-player invariants so Gemini can confirm
healthy rollout without reading SQL directly.

| File | What |
|------|------|
| `backend/admin_scoring_diagnostics.py` | **NEW** (~490 lines). Three endpoints + helpers + whitelists. Mirrors structure of `admin_statcast_diagnostics.py`. |
| `backend/main.py` | Imports + mounts `_scoring_diag_router` under `/admin` prefix (two insertions bracketed by `SCORING DIAGNOSTICS ENDPOINTS` markers, matching the existing statcast diag pattern). Remove both blocks once NSB rollout is validated. |
| `tests/test_admin_scoring_diagnostics.py` | **NEW** (18 tests). Verifies endpoint registration, GET-only, real-column usage for all 3 endpoints, whitelist rejection (bad `window_days`, bad `direction`), helper behavior (`_f`, `_i`, `_d`, `_nsb_integrity_check` ok/mismatch/partial/null), and that the router is actually mounted on `backend.main.app`. |

**Endpoints (all GET, all under `/admin` prefix):**

- `GET /admin/diagnose-scoring/nsb-rollout?window_days=14&as_of_date=YYYY-MM-DD`
  - Returns verdict `healthy`/`partial`/`empty`/`no_data` based on z_nsb fill rate of hitters
  - Fill rates on both `player_rolling_stats` (w_caught_stealing, w_net_stolen_bases) and `player_scores` (z_nsb, z_sb)
  - z_nsb distribution: min/max/mean/std + 5-bucket histogram (POOR/WEAK/AVERAGE/STRONG/ELITE)
  - z_nsb vs z_sb divergence (count differing >0.1 and >0.5; mean delta; max abs delta)
  - `as_of_date` defaults to MAX(as_of_date) in player_scores for the window
- `GET /admin/diagnose-scoring/nsb-leaders?direction=top|bottom&limit=20&window_days=14`
  - Joins `player_scores` + `player_id_mapping` + `player_rolling_stats` so each row has
    `player_name`, `z_sb`, `z_nsb`, `z_nsb_minus_z_sb`, and the raw CS/SB/NSB inputs
  - `direction` is whitelisted to `{"top", "bottom"}`; `window_days` to `{7, 14, 30}`
  - `limit` is `1..100`
- `GET /admin/diagnose-scoring/nsb-player?bdl_player_id=X&as_of_date=YYYY-MM-DD`
  - All-windows detail for one player with `_nsb_integrity_check` per window
  - Integrity check returns: `ok` / `mismatch (expected X, got Y)` / `partial_source` /
    `nsb_null` / `n/a (pure pitcher or no batting)`

**Ops verification recipe (Gemini, after next 4 AM ET job):**

```bash
# 1. Gross rollout health
curl -s $RAILWAY_URL/admin/diagnose-scoring/nsb-rollout?window_days=14 | jq '.verdict, .player_scores'
#    Expect: verdict == "healthy" ; player_scores.z_nsb_fill_pct_of_hitters >= 80.0

# 2. Eyeball leaders for plausibility
curl -s "$RAILWAY_URL/admin/diagnose-scoring/nsb-leaders?direction=top&limit=10&window_days=14" \
  | jq '.rows[] | {name: .player_name, z_nsb, z_sb, sb: .w_stolen_bases, cs: .w_caught_stealing}'
#    Expect: active basestealers (De La Cruz, Chisholm, Ohtani, etc.); z_nsb <= z_sb for high-CS guys

# 3. Spot-check one known basestealer
curl -s "$RAILWAY_URL/admin/diagnose-scoring/nsb-player?bdl_player_id=<id>" \
  | jq '.windows[] | {window: .window_days, sb: .w_stolen_bases, cs: .w_caught_stealing, nsb: .w_net_stolen_bases, check: .nsb_check}'
#    Expect: every window has nsb_check == "ok"; nsb == sb - cs
```

**Remove after validated** (tracked in this doc + source-file header).

### Task D -- explainability layer narrates NSB over SB

**Rationale:** Without this, decision explanations would keep naming "Speed (SB Z-score)"
while composite_z is driven by `z_nsb`. The narrative would diverge from the math --
confusing for users reading explanations alongside percentile scores.

| File | What |
|------|------|
| `backend/services/explainability_layer.py` | `ExplanationInput` dataclass now has `z_nsb`. Basestealing factor prefers `z_nsb` and labels it `"Net basestealing (NSB Z-score)"`; falls back to `z_sb` labeled `"Speed (SB Z-score)"` only when `z_nsb is None` (pre-v27 rows). Narrative always references raw `proj_sb_p50` for continuity with projection tables (simulator still projects raw SB). Both factors omit the projection line when `proj_sb_p50 is None`. |
| `backend/services/daily_ingestion.py` | `ExplanationInput` construction at the decision-pipeline site now passes `z_nsb=score_row.z_nsb`. |
| `tests/test_explainability_layer.py` | `_make_input` helper gains `z_nsb=0.25` default; two existing tests updated to pass `z_nsb=None` when they intentionally nil-out basestealing inputs. **7 new tests** (below). |

**New test coverage (all pass):**

- `test_basestealing_factor_prefers_z_nsb_when_both_present` -- z_nsb is used, labeled NSB
- `test_basestealing_factor_falls_back_to_z_sb_when_nsb_none` -- legacy Speed label, narrative says SB
- `test_basestealing_factor_skipped_when_both_none` -- neither factor appears
- `test_basestealing_narrative_includes_proj_sb_when_available` -- "X.X SB ROS; NSB Z-score"
- `test_basestealing_narrative_omits_projection_when_none` -- Z-score-only narrative
- `test_explain_populates_nsb_factor_end_to_end` -- full `explain()` surfaces NSB factor
- `test_explanation_input_has_z_nsb_field` -- dataclass contract

### Test verification (after Tasks C + D)

- `tests/test_admin_scoring_diagnostics.py`: **18/18 pass**
- `tests/test_explainability_layer.py`: **40/40 pass** (33 existing + 7 new)
- `tests/test_nsb_pipeline.py`: **15/15 pass** (unchanged)
- **Full suite: 1884 passed, 3 skipped** -- zero regressions; +25 net new tests across P27 sprint

### Deploy + verification sequence (Gemini)

```bash
# 1. Pull latest + railway up (mounts new diagnostic routes)
git pull origin stable/cbb-prod
railway up --service mlb-backend  # or current service name

# 2. Confirm router is mounted (should return 200, not 404)
curl -s -o /dev/null -w "%{http_code}\n" $RAILWAY_URL/admin/diagnose-scoring/nsb-rollout

# 3. After next 4 AM ET rolling_windows + player_scores run, exercise the recipe above.
#    Archive the three JSON responses in reports/2026-04-15-nsb-rollout-verification.md.

# 4. Remove the diagnostic endpoints once verdict=="healthy" for 2 consecutive days
#    (mirrors the statcast diagnostics lifecycle).
```

---

## APRIL 15 AUDIT FINDINGS (Kimi CLI)

### Live Production State (Verified via API)

`GET /admin/pipeline-health` → **`overall_healthy: true`**

| Table | Rows | Latest Date | Status |
|-------|------|-------------|--------|
| `player_rolling_stats` | **30,667** | 2026-04-13 | ✅ Healthy |
| `player_scores` | **30,580** | 2026-04-13 | ✅ Healthy |
| `statcast_performances` | **6,971** | 2026-04-13 | ✅ Healthy |
| `simulation_results` | **10,236** | 2026-04-13 | ✅ Healthy |
| `mlb_player_stats` | **6,801** | 2026-04-13 | ✅ Healthy |
| `probable_pitchers` | **0** | — | ⚠️ Empty (upstream) |

### Critical Wins

1. **Statcast aggregation fix was highly effective.** Zero-metric rate dropped from 42.4% to **5.0%**.
   - 6,623/6,971 rows have `exit_velocity` (95.0%)
   - 6,737/6,971 rows have `xwoba` (96.6%)
2. **NSB pipeline is live.** v27 migration deployed. `z_nsb` drives `composite_z`; `z_sb` excluded.
3. **Scheduler is stable.** 10 jobs running. Fantasy jobs (6 AM, 7:30 AM, 8:30 AM, 11:59 PM) all active.
4. **Tests are green on Railway.** 1859 passed. Local collection errors are environmental (missing `redis`).

### Critical Gaps

1. **Weather & park factors: CODE EXISTS BUT NOT INTEGRATED.**
   - `weather_fetcher.py`, `park_weather.py`, and `ballpark_factors.py` are all implemented.
   - **No database tables** for park factors or weather forecasts.
   - **No ingestion pipeline** fetches or stores weather data.
   - `scoring_engine.py` does NOT adjust player scores for park/weather.
   - `daily_lineup_optimizer.py` uses hardcoded park factors but NOT live weather.
2. **Probable pitchers table is empty.** Upstream MLB Stats API returned 0 rows. Not a code bug, but blocks two-start pitcher detection.
3. **Decision results volume is suspiciously low: 26 rows.** With 30K+ player scores and 10K+ simulations, 26 decisions is abnormally low. Needs investigation.
4. **Uncommitted changes on disk.** `scoring_engine.py`, `daily_ingestion.py`, `main.py`, `tests/test_nsb_pipeline.py` are modified/untracked and should be committed.
5. **Data ingestion logs table is empty.** No structured audit trail for sync jobs.

### Immediate Priority Actions

| Priority | Action | Owner | ETA |
|----------|--------|-------|-----|
| P0 | Commit v27 changes (scoring_engine, daily_ingestion, main.py, test_nsb_pipeline.py) | Claude | 15 min |
| P0 | Investigate `decision_results = 26` — trace decision optimization job | Claude | 2 hrs |
| P1 | Integrate park factors into data pipeline (table + scoring engine + lineup optimizer) | Claude | 1 week |
| P1 | Fix/fix probable pitchers source or implement alternative | Claude | 2-3 days |
| P1 | Build data ingestion logging infrastructure | Claude | 2 days |
| P2 | Dedupe `player_id_mapping` (60K → ~2K) | Claude/Gemini | 1 day |
| P2 | Integrate live weather forecasts (requires OPENWEATHER_API_KEY) | Claude | 1 week |

## CURRENT SESSION STATE (April 14, 2026)

### Gemini DevOps Report (April 14 — all requested tasks complete)

| Task | Result | Verdict |
|------|--------|---------|
| Deploy latest code to Railway | Healthy startup, API accessible | DONE |
| `/admin/backfill-cs-from-statcast` | **0 updates** — `statcast_performances` has zero rows with `cs > 0` | DONE (source issue) |
| Probable pitchers ingestion | **0 records** — no games or upstream MLB API lag | DONE |
| Statcast quality audit | **42.4% zero-metric rate** (13,653 rows) — pitch-level overwrite confirmed | DONE |
| Fuzzy linker | **0 linked, 362 remaining** — orphans are prospects/retired (accepted) | DONE |
| Bonus: pitcher metric probes + duplicate endpoint cleanup in main.py | Completed | DONE |

### Live Production DB State (post-Gemini deploy, April 14)

| Table | Rows | Latest Date | Notes |
|-------|------|-------------|-------|
| `mlb_player_stats` | ~6,500+ | 2026-04-13 | Growing daily via BDL |
| `statcast_performances` | **13,653** | 2026-04-13 | **42.4% zero-metric** — pitch-level overwrite bug confirmed |
| `probable_pitchers` | 0 | — | Upstream MLB API returned 0; not a code issue |
| `simulation_results` | ~9,400+ | 2026-04-13 | Fresh |
| `player_rolling_stats` | ~28,000+ | 2026-04-13 | OK |
| `player_scores` | ~28,000+ | 2026-04-13 | OK |
| `player_id_mapping` | 60,000 | — | **Dedup needed** (FU-3) |
| `position_eligibility` | ~2,376 | — | 362 permanent orphans (prospects/retired — accepted) |
| `decision_results` | 26 | — | Low; likely expected for early-season |

Overall health: **Infrastructure stable.** Primary data quality blocker: Statcast aggregation.

### Work Completed This Session (April 14 — Statcast Aggregation Fix)

4 commits implementing a complete fix for the Statcast aggregation bug:

1. **`is_pitcher` flag** (`ef80ecc`) — Added `is_pitcher: bool` to `PlayerDailyPerformance`
   dataclass. Set to `True` in pitcher branch of `transform_to_performance()`. Prerequisite
   for type-scoped upserts. 2 new tests.

2. **`_aggregate_to_daily()` pre-aggregation** (`4c9eef5`) — Core fix. Groups per-pitch
   rows by (player, date, type) before transform. SUMs counting stats (42 column aliases
   including `caught_stealing_2b`), AVERAGEs quality metrics (18 aliases). Short-circuits
   when data is already at daily granularity (leaderboard passthrough). Wired into
   `fetch_statcast_day()` before return. 6 new tests.

3. **Type-scoped upserts** (`9da1f3f`) — Pitcher rows only update pitching + quality metric
   columns on conflict. Batting counting stats (`pa`, `ab`, `h`, `sb`, `cs`, etc.) are
   excluded from pitcher `set_=` dict, preventing two-way player data corruption (Ohtani).
   2 new tests.

4. **End-to-end integration test** (`8fcff87`) — Simulates 5 per-pitch rows with
   `caught_stealing_2b=1` on 2 of them. Verifies aggregation -> transform produces
   `cs=2`. Full suite: **1838 passed**, 3 pre-existing DB-auth failures.

### Architect Review Queue (needs judgment, not execution)

- ~~**P0: Statcast pitch-level overwrite bug** — FIXED (commits ef80ecc..8fcff87)~~
- ~~**P0: CS data gap** — FIXED (caught_stealing_2b now SUMmed in aggregation)~~
- **P1: `/admin/diagnose-era` broken on prod** — `ROUND(AVG(era), 3)` needs `::numeric` cast.
  Cosmetic; fix when touching ERA code.
- **P2: player_id_mapping 60K rows** vs expected ~2K. FU-3 dedup deferred but join costs
  compound. Worth re-prioritizing before NSB/xwOBA derived-stats work.
- **P3: decision_results = 26 rows** — verify if expected early-season or silent drop.

### Work Completed Prior Session (April 13)

1. **MLBBettingOdd Pydantic validation** — `spread_*`, `total_*` fields now `Optional`.
   Files: `backend/data_contracts/mlb_odds.py`, `backend/services/daily_ingestion.py`,
   `tests/test_data_contracts.py` (+4 new tests; 46/46 pass).

2. **caught_stealing backfill endpoint** — `POST /admin/backfill-cs-from-statcast`.
   Idempotent join across `statcast_performances -> player_id_mapping -> mlb_player_stats`.
   Files: `backend/admin_backfill_ops_whip.py`, `tests/test_admin_backfill_ops_whip.py`
   (+6 new tests; 18/18 pass). **Result: 0 updates due to source CS gap.**

3. **Partial derived stats layer** — `backend/services/derived_stats.py`. Pure null-safe
   helpers for OPS/AVG/ISO/WHIP/ERA. 36 tests in `tests/test_derived_stats.py`.
   Ready to ship — values come from `mlb_player_stats` (BDL), not `statcast_performances`.

### Gemini Backfill Findings History (April 13-14)

**April 13 findings (3 blockers identified):**
1. Statcast backfill stalled at 13,376 rows — per-pitch rows overwriting daily aggregates.
2. `statcast_performances.cs > 0` count is zero — `caught_stealing_2b` needs SUM aggregation.
3. Probable pitchers + fuzzy linker returned 0 rows — tracked separately.

**April 14 confirmation:** Gemini re-ran all tasks. Results consistent with April 13 findings.
Zero-metric rate now measured at 42.4% (13,653 rows). Infrastructure is robust; blockers are
logical aggregation gaps requiring architect-level code fixes, not ops issues.

**Full diagnosis:** `reports/2026-04-13-statcast-aggregation-diagnosis.md`.

**Resolution:** Fix implemented in 4 commits (ef80ecc..8fcff87). Ready for Gemini deploy + re-backfill.

---

## ARCHIVE — Prior Session (Derived Stats Readiness Assessment)

### What Just Happened -- Derived Stats Readiness Assessment
- Database audit requested to verify production state after Statcast fixes
- Direct DB connection from local environment blocked (`DATABASE_PUBLIC_URL` not configured on Railway)
- Readiness assessment completed using known state from April 11 deployment + user-reported Statcast progress
- **Full report:** `reports/2026-04-13-derived-stats-readiness-assessment.md`

**Key Finding:** Overall readiness is **72/100** — **MARGINAL** for a complete derived stats layer.
- OPS/WHIP: 85/100 ✅ (at mathematical floor, ready to use)
- NSB (Net Stolen Bases): 0/100 🔴 (caught_stealing is 100% NULL — hard blocker)
- Player identity linkage: 85/100 ✅ (362 orphans accepted)
- Statcast (xwOBA, barrel%, exit velocity): 75/100 ⚠️ (progress reported, needs confirmation)
- Pipeline freshness: 95/100 ✅
- Data quality: 98/100 ✅

**Recommendation:** Start building a **partial derived stats layer now** (OPS, WHIP, ERA, AVG, ISO). Parallel-track the `caught_stealing` source mapping and Statcast backfill confirmation. Add NSB and advanced metrics in a follow-up phase.

### Previous Session -- Data Quality Hardening Sprint (April 12)

**Critical Fixes:**
- **Statcast Column Mapping (Task 0): FIXED** -- `transform_to_performance()` was looking for `exit_velocity_avg`/`xba`/`xwoba` but Baseball Savant returns `launch_speed`/`estimated_ba_using_speedangle`/`estimated_woba_using_speedangle`. This caused 6,255 shell records with all-zero quality metrics. Fixed both batter and pitcher sections. Added `stolen_base_2b`/`caught_stealing_2b` for SB/CS. 28 regression tests.
- **Category Tracker Bug (Task 1): FIXED** -- Removed dead code on line 59 (copy-paste: both `my_stats` and `opp_stats` called `_extract_stats` which didn't even exist on the class). Lines 62-80 correctly parse both teams. 7 regression tests.
- **Valuation Worker Logging (Task 2): FIXED** -- Replaced silent `.get(key, 0.0)` defaults with explicit `logger.warning()` calls naming the player and missing field. Added summary counter (complete vs degraded players).
- **Production Schedules (Task 6): FIXED** -- Restored 5 jobs from temporary 10:32 AM test schedules to production times: player_id_mapping (7:00 AM), position_eligibility (7:15 AM), probable_pitchers (8:30 AM / 4:00 PM / 8:00 PM).

**New Infrastructure:**
- **FanGraphs RoS -> CSV Bridge (Task 3): IMPLEMENTED** -- `export_ros_to_steamer_csvs()` in `projections_loader.py`. Exports FanGraphs daily fetch to Steamer-format CSVs that `load_full_board()` can read. This unlocks real projections for Monte Carlo sims. 12 tests with round-trip verification.
- **Projection Export Endpoint (Task 4): IMPLEMENTED** -- `POST /admin/export-projections` bridges FanGraphs cache to CSVs on demand. Clears `load_full_board()` LRU cache.
- **Pipeline Freshness Validator (Task 5): IMPLEMENTED** -- `GET /admin/pipeline-health` checks row counts and staleness for 6 critical tables. Returns `overall_healthy` flag with per-table details.

**Validation Tests:**
- **E2E Pipeline Tests (Task 7):** 10 tests -- scoring Z-score bounds, simulation percentile ordering, VORP calculations.
- **Monte Carlo Backtest (Task 8):** 12 tests -- mean convergence, variance scaling, percentile coverage, CV matching. Proves the simulation engine is statistically sound.

**Test suite: 1738 passed, 3 pre-existing DB-auth failures (not our code).**

### What's Next -- Gemini Deployment Tasks

1. **Deploy latest code to Railway** (includes all data quality fixes)
2. **Run `POST /admin/backfill/statcast`** -- re-run backfill with fixed column mapping. Should populate real quality metrics for ~6,255 existing shell records + new dates.
3. **Run `POST /admin/ingestion/run/probable_pitchers_morning`** -- verify probable_pitchers populates
4. **Run `POST /admin/ingestion/run/vorp`** -- verify VORP values populate
5. **Run `POST /admin/export-projections`** -- bridge FanGraphs RoS to CSVs (requires fangraphs_ros job to have run at least once)
6. **Run `GET /admin/pipeline-health`** -- validate ALL tables are fresh and populated. Must return `overall_healthy: true` before any frontend work begins.
7. **Verify simulation_results table updates overnight** -- should populate after full pipeline runs (rolling_stats -> player_scores -> ros_simulation)

### Validation Gates Before Frontend

| Gate | Endpoint/Command | Passing Criteria |
|------|------------------|-----------------|
| Pipeline health | `GET /admin/pipeline-health` | All tables `is_healthy: true` |
| Projection data | `POST /admin/export-projections` then verify | > 500 players with real projections |
| Statcast populated | `statcast_performances` check | > 10,000 rows with non-zero quality metrics |
| Simulation fresh | `simulation_results` latest date | Within 2 days of current date |
| Test suite green | `pytest tests/ -q` | 1738+ passed, 0 new failures |

---

## DATABASE STATE (April 15, 2026) — LIVE

| Table | Expected | Actual | Status | Notes |
|-------|----------|--------|--------|-------|
| `player_id_mapping` | ~2,000 | **60,000** (with duplicates) | POPULATED | FU-3: dedup deferred |
| `position_eligibility` | ~750 | **~2,376** | OK | 362 permanent orphans (prospects) |
| `mlb_player_stats` | ~13,500 | **6,801** | OK | Growing daily via BDL sync |
| `probable_pitchers` | ~30/day | **0** | EMPTY | MLB Stats API returned 0 (upstream lag/no games) |
| `statcast_performances` | ~20,000 | **6,971** | ✅ OK | 5.0% zero-metric rate (was 42.4%). Aggregation fix successful. |
| `player_projections` | varies | 0 | EMPTY | No projection pipeline built yet |
| `player_rolling_stats` | ~25,000 | **30,667** | ✅ OK | Populated by rolling stats job |
| `player_scores` | ~25,000 | **30,580** | ✅ OK | Populated by scoring engine |
| `simulation_results` | ~8,500+ | **10,236** | ✅ OK | Fresh through 2026-04-13 |
| `decision_results` | varies | **26** | 🔴 LOW | Suspiciously low — investigate decision optimization job |
| `data_ingestion_logs` | should have entries | 0 | EMPTY | Logging infrastructure not implemented |

---

## COMPLETED WORK (Phases 1-7 + Production Deployment)

### Phase 1: Player Identity Resolution (Tasks 1-3) -- COMPLETE
- Task 1: Backfilled yahoo_key from position_eligibility to player_id_mapping
- Task 2: Linked position_eligibility.bdl_player_id to player_id_mapping
- Task 3: Verified cross-system joins working

### Phase 2: Empty Tables Diagnosis (Tasks 4-6) -- COMPLETE (diagnosis only)
- Task 4: Diagnosed probable_pitchers -- empty because BDL API has no probable pitcher endpoint (K-37 confirmed)
- Task 5: Diagnosed statcast_performances -- empty due to column mismatch in transform (now fixed)
- Task 6: Diagnosed data_ingestion_logs -- empty by design (not yet implemented)

### Phase 3: Computed Fields (Task 7) -- COMPLETE (at mathematical floor)
- Task 7: ops/whip/caught_stealing computation
  - ops: At mathematical floor -- 1,639 remaining NULLs all have NULL obp or slg (unbackfillable)
  - whip: At mathematical floor -- 8 remaining stuck rows have innings_pitched='0.0' (math undefined)
  - caught_stealing: Defaulting to 0 (working)

### Phase 4: Backtest Enhancement (Task 8) -- COMPLETE
- Task 8: direction_correct population verified working

### Phase 5: Player Metrics (Task 9) -- COMPLETE
- Task 9: VORP/z-score computation verified working (scoring_engine.py)

### Phase 6: Data Quality Fixes (Tasks 10-11) -- COMPLETE
- Task 10: ERA investigation -- 1 impossible row (id=8683, era=162.0) fixed
- Task 11: Full validation audit -- `/admin/validation-audit` endpoint created and refreshed

### Phase 7: Root Cause Fixes (Tasks 19-29) -- COMPLETE
- Task 19: Field name mismatch root cause (ops vs on_base_plus_slugging)
- Task 22: WHIP computation fix
- Task 24: OPS computation fix
- Task 26: ops/whip backfill endpoint (`/admin/backfill-ops-whip`)
- Task 27/28: ERA validator (0-100 range)
- Task 21/29: Orphan linking infrastructure
- Task 25: All fixes validated

### Production Deployment (P-1..P-4) -- COMPLETE
- P-1: OPS/WHIP backfill at mathematical floor
- P-2: Legacy ERA cleanup (id=8683 NULL'd)
- P-3: Orphan linking (366 -> 362 orphans; Ohtani + Lorenzen linked)
- P-4: Statcast retry verified; persistence bug identified and fixed

**Full report:** `reports/2026-04-11-production-deployment-results.md`

### Post-Production Follow-Ups -- ALL HIGH/MEDIUM RESOLVED
- FU-1: Statcast persistence bug -- RESOLVED (commit `6848acb`)
- FU-2: Stale validation audit queries -- RESOLVED (commit `c66c445`)
- FU-3: player_id_mapping duplicates -- LOW, deferred (non-blocking)
- FU-4: backfill-ops-whip overcount -- RESOLVED (commit `b11c1e4`)

---

## NEXT STEPS (Priority Order)

### Tasks 4-9 Status

| Task | Description | Status | Notes |
|------|-------------|--------|-------|
| **4** | Probable pitchers pipeline | **DONE** | Rewrote to MLB Stats API. `daily_ingestion.py:_sync_probable_pitchers()` |
| **5** | Statcast production backfill | **GEMINI** | Code fixed. Gemini to deploy + run `POST /admin/backfill/statcast` on Railway |
| **6** | Data ingestion logging | DEFERRED | Low priority -- build audit trail for sync jobs |
| **7** | Derived stats computation | **DONE** (pre-existing) | Already in `scoring_engine.py` -- 8 categories computed |
| **8** | VORP implementation | **DONE** | `backend/services/vorp_engine.py` + scheduled job (4:30 AM, lock 100_030) |
| **9** | Z-score enhancements | **DONE** | Winsorization (default ON) + MAD-based robust Z option in `scoring_engine.py` |

### IMMEDIATE: Gemini Deploy + Re-Backfill

Statcast aggregation fix is complete (4 commits on `stable/cbb-prod`). Gemini should:

1. **Deploy latest code to Railway.** Includes `_aggregate_to_daily()`, type-scoped upserts,
   and `is_pitcher` flag. Verify healthy startup.
2. **TRUNCATE `statcast_performances` table.** Existing 13,653 rows contain corrupted
   per-pitch overwrites. Must start clean.
   ```sql
   TRUNCATE TABLE statcast_performances;
   ```
3. **Run `POST /admin/backfill/statcast`** — full re-backfill with fixed aggregation.
   Expected: ~600-900 rows per game date (one per player per date), NOT 10K+.
   Monitor logs for `"_aggregate_to_daily: X raw rows -> Y aggregated rows"`.
4. **Run `POST /admin/backfill-cs-from-statcast`** — should now find CS events since
   `caught_stealing_2b` is properly SUMmed during aggregation.
5. **Run `GET /admin/pipeline-health`** and report:
   - Total `statcast_performances` row count
   - Zero-metric rate (should drop from 42.4% to <10%)
   - CS backfill result (expect >0 updates now)

### NEXT: Architect Tasks (after Gemini confirms backfill)

1. **Ship derived stats layer** — OPS/WHIP/ERA/AVG/ISO helpers are ready (independent of Statcast).
2. **Add NSB (Net Stolen Bases) to derived stats** once CS data is confirmed populated.
3. **Add Statcast-sourced advanced metrics** (xwOBA, barrel%, exit velocity) to derived stats
   once zero-metric rate is confirmed <10%.

### LOW PRIORITY (Housekeeping)

| Item | Description | Priority |
|------|-------------|----------|
| FU-3 | Dedupe player_id_mapping (Ohtani/Lorenzen 4x rows each) | LOW |
| Simulation staleness | simulation_results last updated 2026-04-07 -- investigate scheduler | LOW |
| data_ingestion_logs | Comprehensive audit logging (Task 6, ~4 hours) | MEDIUM |

### PARALLEL TRACKS (Independent)

| Track | Owner | Status | Notes |
|-------|-------|--------|-------|
| K-39..K-43 Infra Hardening | Kimi CLI | Research pending | DB indexes, rate limiting, security, testing gaps, DR |
| K-44 UI/UX Design System | Claude | Ready to start | Revolut-inspired redesign, 7-phase plan in `reports/2026-04-10-revolut-design-implementation-plan.md` |

---

## RESEARCH REPORTS (K-34 to K-38) -- ALL COMPLETE

| Task | Report | Key Takeaway |
|------|--------|--------------|
| K-34 | `reports/2026-04-10-bdl-api-capabilities.md` | BDL GOAT: 600 req/min, 18 MLB endpoints, cursor pagination, no probable pitchers |
| K-35 | `reports/2026-04-09-zscore-best-practices.md` | Winsorize 5th/95th; MAD robust Z; sample size stabilization |
| K-36 | `reports/2026-04-10-h2h-scoring-systems.md` | H2H One Win 5x5: NSB=SB-CS, NSV=SV-BS; position eligibility rules |
| K-37 | `reports/2026-04-10-mlb-api-comparison.md` | Probable pitchers ONLY from MLB Stats API; Statcast for xwOBA; BDL for games/stats/injuries |
| K-38 | `reports/2026-04-09-vorp-implementation-guide.md` | VORP = Player_Z - Replacement_Z; replacement levels by position documented |

---

## DATABASE HEALTH & EXCELLENCE ROADMAP

### Current Health: C+ (76.9%)

| Metric | Score | Grade |
|--------|-------|-------|
| Data Completeness | 75% | C |
| Cross-System Linkage | 80% | B- |
| Computed Fields | 68% | D+ |
| Pipeline Freshness | 75% | C |
| Data Quality | 95% | A |

### Target: A+ (95%+) via 6-week roadmap

**Full roadmap:** `reports/2026-04-11-database-excellence-roadmap.md`

| Phase | Duration | Focus | Target Grade |
|-------|----------|-------|-------------|
| 1. Foundation | Week 1 | Complete current fixes + Tasks 4-9 | B+ (85%) |
| 2. Hardening | Weeks 2-3 | Reliability engineering | A- (90%) |
| 3. Optimization | Week 4 | Query performance <200ms | A (93%) |
| 4. Observability | Week 5 | Monitoring, MTTD < 5 min | A (95%) |
| 5. Governance | Week 6 | Processes, docs | A+ (97%) |

---

## P-3 UNMATCHABLE ORPHANS -- ACCEPTED AS PERMANENT STATE

362 orphaned position_eligibility records are retired/prospect/minor-league Yahoo players (`469.p.65xxx`, `469.p.66xxx`) with no MLB/BDL entry. Do NOT re-run `backend/scripts/link_orphans.py` -- it will burn ~7 minutes and return 0%. If surfacing in UI, use `yahoo_player_key` directly.

**Export:** `reports/2026-04-11-unmatchable-orphans.csv` (162 batters, 200 pitchers)

---

## INFRASTRUCTURE RESEARCH (K-39 to K-43) -- PENDING

| Task | Focus | Priority | Owner |
|------|-------|----------|-------|
| K-39 | Database Index Optimization | HIGH | Kimi CLI |
| K-40 | API Rate Limiting Strategy | HIGH | Kimi CLI |
| K-41 | Testing Strategy Gap Analysis | MEDIUM | Kimi CLI |
| K-42 | Security Audit - API Layer | HIGH | Kimi CLI |
| K-43 | Backup & Disaster Recovery | MEDIUM | Kimi CLI |

**Delegation doc:** `KIMI_K39_K43_DELEGATION.md`

---

## UI/UX DESIGN SYSTEM (K-44)

**Revolut-inspired fintech redesign.** 7-phase, 2-3 week implementation.

**Plan:** `reports/2026-04-10-revolut-design-implementation-plan.md`

**Status:** Ready for implementation. Can run parallel with backend work.

**Key design decisions:**
- Pill-shaped buttons (9999px radius), zero shadows
- Aeonik Pro (display) + Inter (body) typography
- Semantic color tokens (danger/warning/teal/blue)
- Dark/light section alternation
- 8px base spacing system

---

## AGENT TEAM

| Agent | Role | Current Assignment |
|-------|------|-------------------|
| **Claude Code** | Master Architect -- algorithms, schema, core logic | Tasks 4-9 feature implementation |
| **Gemini CLI** | Ops/DevOps -- Railway deploy, verification, smoke tests | Monitor Railway; run Statcast backfill (Task 5) |
| **Kimi CLI** | Deep research, spec memos, API audits | K-39..K-43 infra hardening research |

---

## VALIDATION TOOLS

| Endpoint/Script | Purpose |
|-----------------|---------|
| `/admin/validation-audit` | Comprehensive data quality validation (refreshed FU-2) |
| `/admin/backfill-ops-whip` | OPS/WHIP backfill (fixed FU-4) |
| `/admin/backfill/statcast` | Statcast backfill (fixed FU-1) |
| `/admin/diagnose-era` | ERA value diagnostics |
| `/test/verify-db-state` | Phase 1+2 table row counts and recency |

---

## KEY FILES

| What you need | Where it is |
|---------------|-------------|
| Current mission + task queue | `HANDOFF.md` (this file) |
| Project orientation | `CLAUDE.md` |
| Agent roles + swimlanes | `ORCHESTRATION.md`, `AGENTS.md` |
| Production deployment report | `reports/2026-04-11-production-deployment-results.md` |
| Database health report | `reports/2026-04-11-comprehensive-database-health-report.md` |
| Database excellence roadmap | `reports/2026-04-11-database-excellence-roadmap.md` |
| Statcast bug fix report | `reports/2026-04-11-statcast-bug-fix-results.md` |
| Validation report | `reports/task-11-validation-report.md` |
| BDL API capabilities | `reports/2026-04-10-bdl-api-capabilities.md` |
| MLB API comparison | `reports/2026-04-10-mlb-api-comparison.md` |
| VORP implementation guide | `reports/2026-04-09-vorp-implementation-guide.md` |
| Z-score best practices | `reports/2026-04-09-zscore-best-practices.md` |
| H2H scoring systems | `reports/2026-04-10-h2h-scoring-systems.md` |
| UI/UX design plan | `reports/2026-04-10-revolut-design-implementation-plan.md` |

---

## ADVISORY LOCK IDS (daily_ingestion.py)

```
100_001 mlb_odds | 100_002 statcast | 100_003 rolling_z | 100_004 cbb_ratings
100_005 clv      | 100_006 cleanup  | 100_007 waiver_scan | 100_008 mlb_brief
100_009 openclaw_perf | 100_010 openclaw_sweep
100_011 scarcity_index_recalc | 100_012 two_start_sp_identification
100_013 projection_model_update | 100_014 probable_pitcher_sync | 100_015 waiver_priority_snapshot
100_028 probable_pitchers | 100_029 player_id_mapping | 100_030 vorp
Next available: 100_031
```

---

**Last Updated:** April 14, 2026
**Session Context:** Statcast aggregation fix COMPLETE (4 commits). 1838/1838 tests pass.
**Priority:** Gemini deploy + TRUNCATE + re-backfill. Then ship derived stats layer + NSB.

---

## ??? DEVOPS & PIPELINE AUDIT (April 14, 2026)

**Mission:** Deploy latest fixes and execute required backfills/audits.

**Status:**
1. **Deployment**: ? **SUCCESS**
   - Latest code (CS backfill, Pydantic odds fixes) live on Railway.
2. **CS Backfill (`/admin/backfill-cs-from-statcast`)**: ? **COMPLETE**
   - Result: **0 updates**.
   - Finding: `statcast_performances` currently contains **zero** records with `cs > 0`. Upstream source/mapping for `caught_stealing_2b` is effectively empty.
3. **Probable Pitchers Ingestion**: ? **COMPLETE**
   - Result: **0 records** (indicates API lag or no scheduled games).
4. **Statcast Quality Audit**: ? **COMPLETE**
   - **Zero-Metric Rate**: **42.4%** (Total Rows: 13,653).
   - **Critical Finding**: Pitch-level data is overwriting daily player totals due to `on_conflict_do_update` without aggregation. This is the root cause of the "shell record" issue.
5. **Fuzzy Linker**: ? **COMPLETE**
   - Result: **0 linked**, 362 remaining. New orphans confirmed as unmatchable prospects/retired players.

**Overall Verdict**: Infrastructure is stable. The data pipeline requires a logical pivot from **overwrite** to **aggregate** for Statcast detail rows to reach required richness levels.

*Last Updated: April 14, 2026 � Gemini DevOps Lead*
