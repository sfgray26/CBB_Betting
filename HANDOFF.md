# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-05-13 | **Architect:** Claude Code (Master Architect)
> **Branch:** `stable/cbb-prod` | **HEAD:** 7d18395 (Phase 1 P0 data bugs fixed)
> **Deploy:** `/health` = `{"status":"healthy","database":"connected","scheduler":"running"}` (d319beb live on Railway)

---

## System Status

| Component | Status | Notes |
|---|---|---|
| Railway deployment | LIVE | FastAPI + uvicorn, us-west1 |
| `canonical_projections` | LIVE (822 rows) | SAVANT_ADJUSTED=469, STATIC_BOARD=353 — 0 errors |
| `category_impacts` | LIVE (5,350 rows) | Per-category z-scores + marginal numerator/denominator columns |
| `player_projections` (FanGraphs RoS) | LIVE (9,686 rows) | prior_source=fangraphs_ros, updated 2026-05-06 15:15 ET |
| `player_projections.player_type` | fully populated | C-A1 confirmed: 0 NULL rows |
| `player_identities` | 454/454 resolved | 12-row backfill complete 2026-05-06 |
| `CANONICAL_PROJECTION_V1` flag | **true** | Nightly job runs at 11 PM ET |
| `market_signals_enabled` | **true** | avg_market_score=51.53 after d319beb — CALIBRATED |
| `feature_matchup_enabled` | **true** | C-4 complete |
| `opportunity_enabled` | **true** | D-2 complete. Fix deployed in current HEAD — upserted=0 bug patched |
| `/api/fantasy/decisions` route | **LIVE** | Returns ~800 daily decisions — A-4 resolved (pipeline working) |
| Stuff+/Location+ (FanGraphs) | BLOCKED | Cloudflare blocks Railway IP — P2, do not attempt |
| Savant Pitch Quality | **LIVE** | 554 scores seeded; avg_confidence=0.191 (below 0.3 threshold) — flags remain false |
| Savant Park Factors | **LIVE** | 28 venues / 56 rows seeded |
| `player_market_signals` | LIVE (10,466 rows) | avg_market_score=51.53, BUY_LOW=135 / FAIR=618 / SLEEPER=131 — healthy |
| `player_opportunity` | **LIVE + SIGNAL** | 2738 rows, avg_z=0.0 (centered), min=-0.598, max=4.082, avg_conf=0.120 |
| `mlb_player_stats` | LIVE (15,407 rows) | Confirmed populating — now feeds opportunity engine |

---

## Code Fixes This Session (2026-05-13)

| Fix | Bug | Patch |
|-----|-----|-------|
| Budget IP key `"my_team"` → `"my_stats"` (A-6) | `get_matchup_stats()` returns `"my_stats"` not `"my_team"` — IP always read as 0.0 | `routers/fantasy.py:5519`. Commit d0976b4 |
| Need score key mismatch | `compute_need_score` lowercased canonical codes (`"hr_b"`) but `cat_scores` uses board keys (`"hr"`) — all need scores = 0 | Added `_CANONICAL_TO_BOARD` mapping in `category_aware_scorer.py`. Commit bd180a4 |
| Budget acquisitions window | Rolling 7-day window instead of matchup week (Mon 00:00 ET); `"add/drop"` type excluded from count | `routers/fantasy.py:5501`, `constraint_helpers.py:48`. Commit d2fbc43 |
| Ownership always 0% | `get_free_agents` fetched from `league/{lk}/players` without `out=ownership` (Yahoo 400). Secondary batch call to `players;player_keys/ownership` needed | `yahoo_client_resilient.py`. Commit 425f9d6 |
| Simulate roster fetch unhandled exception | `_fetch_rosters_for_simulate` called outside try/except — Yahoo auth error would bypass CORS middleware | Added try/except around roster fetch in `simulate_matchup`. Commit ff2c8b9 |
| K-1 P0 streaming/dashboard/lineup 422 | All three verified RESOLVED by prior session's code (A-7 streaming fix, dashboard page rewrite, FastAPI static-route priority) | No code change needed |

---

## Code Fixes This Session (2026-05-12)

| Fix | Bug | Patch |
|-----|-----|-------|
| `opportunity_update` upserted=0 | First FK violation (`bdl_player_id` not in `player_id_mapping`) poisoned SQLAlchemy session; all 2738 subsequent `db.execute()` raised `PendingRollbackError` silently caught | SAVEPOINT/RELEASE/ROLLBACK wrapper per row in `_compute_opportunity` loop (~line 3399). Also removed dead `pg_insert(text(...).columns)` |
| `yahoo_id_sync` UniqueViolation (`_pim_bdl_id_uc`) kills entire transaction | INSERT path uses only `_pim_yahoo_key_uc` as ON CONFLICT target; if two players resolve to same `bdl_id`, `_pim_bdl_id_uc` fires at `db.commit()` → full rollback → 0 updates returned | SAVEPOINT/RELEASE/ROLLBACK around `db.execute(stmt)` in the "new row" INSERT path (~line 2459); failed rows logged as `insert_conflict` and skipped |
| ✅ Draft board data leak in waiver targets | `get_or_create_projection()` checked draft board (hardcoded player rankings) BEFORE database, returning "Gavin Williams" draft data instead of real Steamer/Statcast projections from DB | Commented out draft board fallback in `player_board.py` lines 1030-1063; function now queries DB first. Commit 2f6f1f7 pushed. |

## Codex Feature Branch Notes (2026-05-07)

Branch: `codex-fantasy-predictive-quality-gates`

Purpose: implement pre-merge predictive-quality hardening for Claude audit before stable/prod merge.

Changes:
- Add denominator-aware rate gates in `backend/services/scoring_engine.py`: AVG/OBP/OPS require `w_ab >= 20`; ERA/WHIP/K9 require `w_ip >= 8`. Counting stats still score in small samples.
- Filter invalid/null market rows in `backend/services/waiver_edge_detector.py` before using market score as a waiver tiebreaker.
- Normalize `daily_snapshots.pipeline_jobs_run` in `backend/services/snapshot_engine.py` to fantasy pipeline job names only.
- Give canonical RoS/Steamer and component-based fusion rows bounded non-zero confidence in `backend/fantasy_baseball/projection_assembly_service.py` instead of zeroing them solely because sample size is missing.

**Note:** Kimi's K-NEXT-2 report incorrectly identified a duplicate `_sync_yahoo_id_mapping` at line 7682. Verified: only one definition exists (line 2204). The real bug was the missing `_pim_bdl_id_uc` conflict guard on the INSERT path. Both bugs now fixed.

---

## Previously Deployed Fixes (d319beb — 2026-05-06)

| Commit | Bug | Fix | Status |
|--------|-----|-----|--------|
| `21f96df` | `ros_projection_refresh` UniqueViolation for two-way players | `bat_processed_ids` set + `db.flush()` + pitcher merge preserves batting stats | ✅ |
| `8388062` | `canonical_projections.player_id` INT4 overflow for large Yahoo IDs | `Column(Integer)` → `Column(BigInteger)` | ✅ |
| `12b7f5a` | `migration_dedupe_player_id_mapping` UniqueViolation on `_pim_yahoo_key_uc` | Step 3a-pre: NULL loser `yahoo_key` values before merge | ✅ |
| `b799aec` | Frontend `/decisions` page 404 | Added `GET /api/fantasy/decisions` + `/decisions/status` | ✅ |
| `612d351` | `backfill_player_type.py` hardcoded credential | Full rewrite — removed credential, `--dry-run`, two-way logic fixed | ✅ |
| `6f44ebb` | `market_signals_update` column not found | `score_0_100` + `window_days=14` + `MAX(as_of_date)` fallback | ✅ |
| `ecfa5ba` | `market_signals_update` upserted=0 | 8 SQL INSERT param names fixed | ✅ |
| `63c936b` | `avg_market_score=99.65` / 881 BUY_LOW | `skill_gap_pct / 100.0` normalization fix | ✅ |
| `d319beb` | Savant pitch quality: all scores 100.0 (ip=NULL) | Use `pa` as IP proxy for `sample_confidence` | ✅ |

---

## DevOps Queue

### D-3/D-4/D-5: COMPLETE ✅
- opportunity_update: 2738/2738 upserted, avg_z=0.0 (centered), min=-0.598, max=4.082, avg_conf=0.120

### D-6: Validate yahoo_id_sync (b82bc14 already live — no deploy needed)

The previous 502 was an HTTP gateway timeout — job ran in background. The `_pim_bdl_id_uc` SAVEPOINT fix is deployed. Trigger and wait 3 min before checking:

```bash
curl.exe -X POST https://<railway-url>/admin/ingestion/run/yahoo_id_sync \
     -H "X-API-Key: $ADMIN_API_KEY"
# 502 is expected (job takes >60s) — do NOT treat as failure
# Wait 3 minutes, then check job_runs:
railway run python -c "
from backend.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
rows = db.execute(text('''
    SELECT status, records_processed, error_message, started_at
    FROM job_runs WHERE job_name = 'yahoo_id_sync'
    ORDER BY started_at DESC LIMIT 3
''')).fetchall()
for r in rows: print(r)
db.close()
"
# Expected: status=success (or completed), records_processed > 0
# insert_conflict warnings in Railway logs are EXPECTED and fine
# STOP and escalate to Claude only if status=failed with a NEW error type
```

---

## Claude Architect Queue

### ✅ A-1: Fix `player_type` NULL — COMPLETE
### ✅ A-2: Yahoo ID sync coverage — COMPLETE
### ✅ A-3: Frontend `/decisions` endpoint — COMPLETE

### ✅ A-4: Decision pipeline starved — RESOLVED (2026-05-11)

**Verdict:** Phantom problem — pipeline is working correctly.

**Diagnostic Results (2026-05-11):**
- `mlb_game_log`: 609 rows ✅
- `mlb_player_stats`: 17,459 rows ✅
- `player_rolling_stats`: 102,040 rows ✅
- `player_scores`: 99,037 rows ✅
- `decision_results`: 832 rows ✅

**Root Cause:** HANDOFF.md claim that "decision_results table is empty" was outdated. The pipeline has been functional since at least 2026-05-06. `/api/fantasy/decisions` endpoint returns live recommendations (~800 daily decision results).

### A-5: Dead `_refresh_ros_projections` v1 at line ~5952 (low priority)
Python last-definition-wins: v2 at line ~6822 is active. Remove v1. Do when deploying for something else.

### ✅ A-6 (P1): Wire IP tracking in `/api/fantasy/budget` — COMPLETE (2026-05-13)
**Root cause:** `matchup_stats.get("my_team", {})` → key was wrong; correct key is `"my_stats"`.
**Fix applied:** `backend/routers/fantasy.py:5519` — single-line change. Commit d0976b4.

### A-7 (P2): UI clarity issues — ✅ COMPLETE (2026-05-12)
**Report:** User screenshots showing dashboard confusion

**Issues identified:**
1. **Running counts lack context** — Just shows numbers without category names or whether higher/lower is better
2. **Bubble ratings unclear** — Visual bubbles don't communicate what they represent (z-scores? win probabilities?)
3. **✅ Waiver targets not showing diverse players** — FIXED (2026-05-12): Two-phase fix applied:
   - Commit 2f6f1f7: Disabled draft board fallback that was overriding DB data
   - Commit f6e5a6f: Re-enabled draft board fallback ONLY for players not in database
   
   **Result:**
   - Players in database (Gavin Williams) → use real Steamer/Statcast projections ✅
   - Players NOT in database (Christopher Sanchez) → use draft board fallback ✅
   - Tested: Christopher Sanchez (Cy Young contender) now appears with tier 2, ADP 28 ✅

**UI Fixes Applied (2026-05-12):**

**1. Running Counts (Category Deficits) — FIXED**
   - Added category labels using CATEGORY_LABEL mapping (HR_B → HR, K_B → K, etc.)
   - Added color-coded arrows: green (TrendingUp) for ahead, red (TrendingDown) for behind
   - Correct direction logic for LOWER_IS_BETTER categories (K_B, L, HR_P, ERA, WHIP):
     - Negative z-score = good (ahead) for lower-is-better
     - Positive z-score = good (ahead) for higher-is-better
   - Added explanatory text: "(Negative = behind league average)"

**2. Bubble Ratings — FIXED**
   - Added tooltips to status tags showing exact win probability:
     - SAFE: "85% win prob - Safe lead"
     - LEAD: "70% win prob - Leaning ahead"
     - BUBBLE: "50% win prob - Could go either way"
     - BEHIND: "30% win prob - Leaning behind"
     - LOST: "10% win prob - Unlikely to win"
   - Added visual legend above category battlefield showing ranges:
     - SAFE >85% | LEAD 65-85% | BUBBLE 35-65% | BEHIND 15-35% | LOST <15%

**Files modified:**
- `frontend/app/(dashboard)/war-room/streaming/page.tsx` — Category deficits display
- `frontend/components/war-room/category-battlefield.tsx` — Status tooltips + legend

**Result:** Running counts now show direction (good/bad) with visual indicators; bubble ratings are self-explanatory with tooltips and legend.

---

## Kimi Research Queue

### K-NEXT-1: Savant pitch quality ✅ COMPLETE — flags remain FALSE

Scores now differentiated (88.8–112.6) after `pa` proxy fix (d319beb). But `avg_sample_confidence=0.191` is below the >0.3 threshold. This is a season-length issue — PA accumulates over time. Re-check late May when avg PA > 120. No code change needed.

### K-NEXT-2: Yahoo ID sync — ✅ RESOLVED (architect fix applied)

Kimi's report incorrectly identified a duplicate function. Actual bug: INSERT at line ~2459 only handled `_pim_yahoo_key_uc` on conflict, allowing `_pim_bdl_id_uc` to kill the entire transaction on any duplicate bdl_id. Fixed with SAVEPOINT pattern.

---

## K-1 UI UAT FINDINGS (2026-05-07)

Production UI audit completed. Full report: `reports/2026-05-07-ui-uat-audit.md`

### P0 (Blocking) — ✅ ALL RESOLVED (2026-05-13)
1. **✅ CORS on `POST /api/fantasy/matchup/simulate`** — Simulate roster fetch wrapped in try/except; Yahoo auth errors now return clean HTTPException (CORS middleware applies headers to all HTTPException responses). Commit ff2c8b9.
2. **✅ Streaming page infinite loading** — Verified RESOLVED by A-7 (commit 847415c): streaming page has proper isLoading/isError/!data early-return guards.
3. **✅ Dashboard empty despite API data** — Verified RESOLVED: dashboard/page.tsx has full implementation calling getDashboard with correct response.success check.
4. **✅ `/api/fantasy/lineup/current` always 422** — Verified RESOLVED: main.py registers static `/current` route before parameterized `/{lineup_date}`; FastAPI routes static first.

### P1 (Degraded)
5. **Budget API not integrated into UI** — `/api/fantasy/budget` is healthy (576ms) but no frontend page calls it; no budget panel exists.
6. **Matchup API latency ~3,182ms** — Exceeds 2s threshold. No loading skeleton visible during fetch.
7. **My Roster & Waiver Wire are placeholders** — "COMING NEXT" pages only.
8. **Favicon 404** on every page load.

### P2 (Polish)
9. Login form API key input lacks `id`/`name` attribute (accessibility warning).
10. `ip_accumulated` mocked at `0.0` — no UI impact since budget is not displayed.

---

## Feature Flag State

| Flag | Value | Gate Condition |
|---|---|---|
| `CANONICAL_PROJECTION_V1` | **true** | Nightly `canonical_projection_refresh` job active |
| `market_signals_enabled` | **true** | CALIBRATED — avg 51.53 after d319beb |
| `feature_matchup_enabled` | **true** | Active |
| `opportunity_enabled` | **true** | Fix deployed in HEAD — validate after D-3 |
| `statcast_stuff_plus_enabled` | false | Blocked by Cloudflare — do not enable |
| `statcast_location_plus_enabled` | false | Blocked by Cloudflare — do not enable |
| `savant_pitch_quality_enabled` | false | avg_confidence=0.191 < 0.3 — re-check late May |
| `savant_pitch_quality_waiver_signals_enabled` | false | Enable after main flag |
| `savant_pitch_quality_projection_adjustments_enabled` | false | Enable last |

---

## Known Infrastructure Blockers

### Stuff+/Location+ (FanGraphs/Cloudflare — P2)
FanGraphs routes through Cloudflare with IP-reputation blocking. Railway IP range blocked. Resolution:
1. Manual CSV snapshot from browser (monthly)
2. FanGraphs API subscription (~$80/year)
3. Savant Pitch Quality (in-house) as proxy — activate when confidence threshold met

### Savant Pitch Quality (inactive — confidence too low)
All 554 scores seeded. avg_confidence=0.191 (pa proxy gives signal but early-season PA counts are low). Re-validate late May. Feature flags remain false.

---

## Architecture Decisions (Locked)

| Decision | Rule |
|---|---|
| Counting-stat pipeline | Hybrid provenance: HR/SB from Bayesian rates × PA; R/RBI/SV static Steamer; W formula; K = K/9 × IP/9 |
| Advanced metrics storage | No new table. `statcast_batter_metrics` + `statcast_pitcher_metrics` → denormalized into `canonical_projections` |
| TeamContext | Ephemeral runtime dataclass. Quarantined players (PENDING_REVIEW) excluded from PA/IP denominators |
| CBB betting model | FROZEN — season closed. Do not modify `betting_model.py` |
| OddsAPI | 20k/month — CBB archival closing lines only. All MLB odds via BDL |

---

## Advisory Lock Registry

```
100_001 mlb_odds        | 100_002 statcast          | 100_003 rolling_z         | 100_004 cbb_ratings
100_005 clv             | 100_006 cleanup           | 100_007 waiver_scan        | 100_008 mlb_brief
100_009 openclaw_perf   | 100_010 openclaw_sweep    | 100_011 valuation_cache    | 100_012 fangraphs_ros
100_013 yahoo_adp_injury| 100_014 ensemble_update   | 100_015 projection_freshness| 100_016 mlb_game_log
100_017 mlb_box_stats   | 100_018 rolling_windows   | 100_019 player_scores      | 100_020 player_momentum
100_021 ros_simulation  | 100_022 decision_optimization | 100_023 backtesting    | 100_024 explainability
100_025 snapshot        | 100_026 statsapi_supplement | 100_027 position_eligibility | 100_028 probable_pitchers
100_029 player_id_mapping | 100_030 vorp            | 100_031 projection_cat_scores | 100_032 savant_ingestion
100_033 bdl_injuries    | 100_034 yahoo_id_sync     | 100_035 cat_scores_backfill | 100_036 ros_projection_refresh
100_037 opportunity_update | 100_038 market_signals_update | 100_039 matchup_context_update | 100_040 canonical_projection_refresh
Next available: 100_041
```
