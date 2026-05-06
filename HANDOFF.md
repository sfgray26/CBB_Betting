# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-05-06 | **Architect:** Claude Code (Master Architect)
> **Branch:** `stable/cbb-prod` | **HEAD:** `63c936b` (not yet deployed)
> **Deploy:** `/health` = `{"status":"healthy","database":"connected","scheduler":"running"}` (HEAD on Railway: `ecfa5ba`)

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
| `market_signals_enabled` | **true** | C-2 complete — re-run needed after `63c936b` to get correct scores |
| `feature_matchup_enabled` | **true** | C-4 complete |
| `opportunity_enabled` | **false** | Ready to enable — C-A1 confirmed 0 NULL player_types |
| `/api/fantasy/decisions` route | **LIVE** | Returns empty until P17 runs |
| Stuff+/Location+ (FanGraphs) | BLOCKED | Cloudflare blocks Railway IP — P2, do not attempt |
| Savant Pitch Quality | **LIVE** | 550 scores seeded; all three flags remain false (pending K-NEXT-1 review) |
| Savant Park Factors | **LIVE** | 28 venues / 56 rows seeded |
| `player_market_signals` table | partially broken | Two bugs fixed in commits below — needs re-run to validate |

---

## Code Fixes Deployed This Session (2026-05-06)

| Commit | Bug | Fix | Deployed? |
|--------|-----|-----|-----------|
| `21f96df` | `ros_projection_refresh` UniqueViolation for two-way players | `bat_processed_ids` set + `db.flush()` + pitcher merge preserves batting stats | ✅ |
| `8388062` | `canonical_projections.player_id` INT4 overflow for large Yahoo IDs | `Column(Integer)` → `Column(BigInteger)` in ORM + migration | ✅ |
| `12b7f5a` | `migration_dedupe_player_id_mapping` UniqueViolation on `_pim_yahoo_key_uc` | Added Step 3a-pre: NULL loser `yahoo_key` values before merge runs | ✅ |
| `b799aec` | Frontend `/decisions` page 404 | Added `GET /api/fantasy/decisions` + `/decisions/status` to `main.py` | ✅ |
| `612d351` | `backfill_player_type.py` hardcoded credential, missing `'both'` case | Full rewrite — removed credential, `--dry-run`, fixed `::jsonb` and two-way logic | ✅ |
| `6f44ebb` | `market_signals_update` column `skill_gap_percentile` not found | `score_0_100` + `window_days=14` + `MAX(as_of_date)` fallback | ✅ |
| `ecfa5ba` | `market_signals_update` — `upserted=0` despite "success" | 8 SQL INSERT param names fixed to match `res` dict keys | ✅ |
| `63c936b` | `market_signals_update` — `avg_market_score=99.65`, 881/884 tagged BUY_LOW | `skill_gap_percentile=skill_gap_pct / 100.0` — `score_0_100` is [0-100], function expects [0-1] | **❌ PENDING DEPLOY** |

---

## DevOps Queue (ordered) — HEAD `63c936b`

All C-1 → C-6 are **COMPLETE**. One new item:

### D-1: Deploy `63c936b` + validate market signals calibration
```bash
# 1. Deploy
railway up --detach
# 2. Wait for health
curl.exe https://<railway-url>/health
# 3. Re-run market signals
curl.exe -X POST https://<railway-url>/admin/ingestion/run/market_signals_update \
     -H "X-API-Key: $ADMIN_API_KEY"
# Expected AFTER fix:
#   upserted > 5000
#   avg_market_score in range [40, 75]  (was 99.65 — that was the bug)
#   tag_distribution shows mix of BUY_LOW / FAIR / HOT_PICKUP / SLEEPER (was 881 BUY_LOW / 3 FAIR)
# STOP and escalate to Claude if upserted=0 or avg_market_score still > 95
```

### D-2: Enable `opportunity_enabled` flag (no deploy needed)
C-A1 confirmed 0 NULL player_types. Enable the flag directly in DB:
```sql
UPDATE feature_flags SET enabled=true, updated_at=NOW()
WHERE flag_name='opportunity_enabled';
-- verify:
SELECT flag_name, enabled FROM feature_flags WHERE flag_name='opportunity_enabled';
```
Then trigger one run to validate:
```bash
curl.exe -X POST https://<railway-url>/admin/ingestion/run/opportunity_update \
     -H "X-API-Key: $ADMIN_API_KEY"
# Expected: status=success, total > 0
```

**Reporting protocol:** After each task, report: exit code + last 5 stdout lines + verify query result. STOP on first failure.

---

## Claude Architect Queue

### ✅ A-1: Fix `player_type` NULL in `player_projections` — COMPLETE
C-A1 ran, 0 nulls found — already fully populated.

### ✅ A-2: Yahoo ID sync coverage — COMPLETE

### ✅ A-3: Frontend `/decisions` endpoint missing — COMPLETE

### A-4 (next): Investigate P17 decision_optimization pipeline
`decision_results` is empty — the planning/decision phase has never produced output.
Likely cause: prerequisite pipeline phases (P13–P16) may have empty tables or errors.
Check job_runs for the last 7 days for rolling_windows (100_018), player_scores (100_019), decision_optimization (100_022).

### A-5: Dead `_refresh_ros_projections` v1 at line ~5952 in daily_ingestion.py
Python last-definition-wins: v2 at line ~6822 is active, v1 is dead code. Remove v1 (low priority).

---

## Kimi Research Queue

### K-NEXT-1: Savant pitch quality distribution validation
Run after Codex completes C-5. Pull `SELECT player_name, savant_pitch_quality, sample_confidence FROM savant_pitch_quality_scores WHERE season=2026 ORDER BY savant_pitch_quality DESC LIMIT 20` and the bottom 20. Cross-check vs known 2026 pitcher quality (Skenes, Wheeler, Flaherty at top; roster fillers at bottom). Report distribution shape, range, anomalies.
Output: `reports/2026-05-XX-savant-pitch-quality-validation.md`

### K-NEXT-2: Yahoo ID sync gap analysis ✅ COMPLETE
**Report:** `reports/2026-05-06-yahoo-id-sync-gap-analysis.md`

**Key Findings:**
1. **Job is BROKEN since May 4** — Every run fails with `UniqueViolation: _pim_bdl_id_uc`. Zero enrichments for 48h.
2. **Root cause:** Duplicate rows in `player_id_mapping` (1,513 dupes across 707 names) + the "simplified" `_sync_yahoo_id_mapping` (line 7682, which overwrites the guarded version at line 2204) removed the `bdl_id` conflict check.
3. **3.7% coverage = shallow enumeration, not poor matching.** The sync only sees ~394 Yahoo players/day (rosters + top-25 FAs per position) vs ~10,000 BDL players. Match rate for seen players is ~94%.
4. **451 Yahoo rows (18.5%) still have NULL `bdl_id`** — the sync never backfills existing Yahoo rows.
5. **No fuzzy matching implemented** despite docstring claim. Exact-name-only lookup.
6. **Proposed fixes:** (P0) Restore conflict guard + dedupe table; (P1) Paginate FAs to 200/position + backfill missing bdl_ids; (P1) Use `mlbam_id` bridge from FanGraphs API to eliminate name collisions.

**Immediate action for Claude:** Remove duplicate method definition at line 7682, restore bdl_id conflict guard, run table dedupe.

---

## Feature Flag State

| Flag | Value | Gate Condition |
|---|---|---|
| `CANONICAL_PROJECTION_V1` | **true** | Nightly `canonical_projection_refresh` job active |
| `market_signals_enabled` | **true** | Active — re-run after D-1 deploy to get calibrated scores |
| `feature_matchup_enabled` | **true** | Active |
| `opportunity_enabled` | **false** | Ready — enable via D-2 (C-A1 complete) |
| `statcast_stuff_plus_enabled` | false | Blocked by Cloudflare — do not enable |
| `statcast_location_plus_enabled` | false | Blocked by Cloudflare — do not enable |
| `savant_pitch_quality_enabled` | false | Enable after K-NEXT-1 validation passes |
| `savant_pitch_quality_waiver_signals_enabled` | false | Enable after main flag + waiver wiring review |
| `savant_pitch_quality_projection_adjustments_enabled` | false | Enable last — projection impact requires full validation |

---

## Known Infrastructure Blockers

### Stuff+/Location+ (FanGraphs/Cloudflare — P2)
FanGraphs routes `leaders-legacy.aspx` through Cloudflare with IP-reputation blocking.
Railway's IP range is blocked. Code is correct; columns exist in `statcast_pitcher_metrics`.
**Do not attempt Cloudflare bypass.** Resolution options when signal is needed:
1. Manual CSV snapshot from browser (monthly) → `data/fangraphs_pitcher_quality_YYYY.csv` → backfill
2. FanGraphs API subscription (~$80/year) bypasses Cloudflare
3. Savant Pitch Quality (in-house, already implemented) as proxy

### Savant Pitch Quality (inactive — pending Railway rollout)
Code path: `backend/fantasy_baseball/savant_pitch_quality.py` → `savant_pitch_quality_scores` table.
All three feature flags disabled. Codex runs C-5 to activate DB. Kimi runs K-NEXT-1 to validate.
Claude reviews K-NEXT-1 output before enabling any flags.

### Savant Park Factors (inactive — pending Railway rollout)
Snapshot: `data/park_factors/savant_park_factors_2025_3yr.json` (28 venues).
Codex runs C-6. TB/OAK fall back to legacy constants until Savant has stable data for their parks.

---

## Architecture Decisions (Locked — see git log for full rationale)

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
