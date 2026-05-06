# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-05-06 | **Architect:** Claude Code (Master Architect)
> **Branch:** `stable/cbb-prod` | **HEAD:** pending commit (opportunity + yahoo_id_sync fixes)
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
| `/api/fantasy/decisions` route | **LIVE** | Returns empty — A-4 blocking (decision pipeline starved) |
| Stuff+/Location+ (FanGraphs) | BLOCKED | Cloudflare blocks Railway IP — P2, do not attempt |
| Savant Pitch Quality | **LIVE** | 554 scores seeded; avg_confidence=0.191 (below 0.3 threshold) — flags remain false |
| Savant Park Factors | **LIVE** | 28 venues / 56 rows seeded |
| `player_market_signals` | LIVE (10,466 rows) | avg_market_score=51.53, BUY_LOW=135 / FAIR=618 / SLEEPER=131 — healthy |
| `player_opportunity` | LIVE (flag enabled) | upserted=0 bug fixed in HEAD — needs deploy + re-run |
| `mlb_player_stats` | UNKNOWN | Root cause of empty decision pipeline — Gemini to diagnose (see D-3) |

---

## Code Fixes This Session (2026-05-06 — pending commit)

| Fix | Bug | Patch |
|-----|-----|-------|
| `opportunity_update` upserted=0 | First FK violation (`bdl_player_id` not in `player_id_mapping`) poisoned SQLAlchemy session; all 2738 subsequent `db.execute()` raised `PendingRollbackError` silently caught | SAVEPOINT/RELEASE/ROLLBACK wrapper per row in `_compute_opportunity` loop (~line 3399). Also removed dead `pg_insert(text(...).columns)` |
| `yahoo_id_sync` UniqueViolation (`_pim_bdl_id_uc`) kills entire transaction | INSERT path uses only `_pim_yahoo_key_uc` as ON CONFLICT target; if two players resolve to same `bdl_id`, `_pim_bdl_id_uc` fires at `db.commit()` → full rollback → 0 updates returned | SAVEPOINT/RELEASE/ROLLBACK around `db.execute(stmt)` in the "new row" INSERT path (~line 2459); failed rows logged as `insert_conflict` and skipped |

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

## DevOps Queue — HEAD needs deploy

### D-3: Deploy HEAD + validate both fixes

```bash
# 1. Deploy
railway up --detach

# 2. Health check
curl.exe https://<railway-url>/health

# 3. Trigger opportunity_update and verify upserted > 0
curl.exe -X POST https://<railway-url>/admin/ingestion/run/opportunity_update \
     -H "X-API-Key: $ADMIN_API_KEY"
# Expected: upserted > 0 (was 0 due to session-poison bug now fixed)
# If still 0: report the warning logs — bdl_id FK mismatches will now be logged per row

# 4. Trigger yahoo_id_sync and verify no UniqueViolation
curl.exe -X POST https://<railway-url>/admin/ingestion/run/yahoo_id_sync \
     -H "X-API-Key: $ADMIN_API_KEY"
# Expected: status=success, records > 0 (was failing with UniqueViolation / 0 updates)
# If insert_conflict warnings appear in logs, that is EXPECTED — they now skip gracefully

# 5. Run A-4 diagnostic queries against prod DB (use Railway DB proxy):
railway run python -c "
from backend.database import SessionLocal
from sqlalchemy import text
db = SessionLocal()
print('mlb_player_stats:', db.execute(text('SELECT COUNT(*), MAX(game_date) FROM mlb_player_stats')).fetchone())
print('mlb_game_log:', db.execute(text('SELECT COUNT(*), MAX(game_date) FROM mlb_game_log')).fetchone())
print('rolling_stats today:', db.execute(text('SELECT COUNT(*) FROM player_rolling_stats WHERE as_of_date = CURRENT_DATE')).fetchone())
print('player_scores today:', db.execute(text('SELECT COUNT(*) FROM player_scores WHERE as_of_date = CURRENT_DATE AND window_days = 14')).fetchone())
print('decision_results today:', db.execute(text('SELECT COUNT(*) FROM decision_results WHERE as_of_date = CURRENT_DATE')).fetchone())
db.close()
"
# Report all 5 counts + MAX(game_date) values. STOP and escalate to Claude if any count is 0.
```

**Reporting protocol:** After each step, report exit code + key output lines. STOP on first failure.

---

## Claude Architect Queue

### ✅ A-1: Fix `player_type` NULL — COMPLETE
### ✅ A-2: Yahoo ID sync coverage — COMPLETE
### ✅ A-3: Frontend `/decisions` endpoint — COMPLETE

### A-4 (ACTIVE): Decision pipeline starved — `decision_results` empty

The pipeline is: `mlb_box_stats` (2 AM) → `rolling_windows` (3 AM) → `player_scores` (4 AM) → `decision_optimization` (7 AM).

Each phase returns `status: success` with 0 records when upstream is empty — making the scheduler appear green while nothing flows through. Root cause suspected: `mlb_game_log` or `mlb_player_stats` is empty or stale.

**Waiting for D-3 step 5 diagnostic results from Gemini before prescribing fix.**

Once Gemini reports the 5 counts:
- If `mlb_game_log` count = 0: `_ingest_mlb_game_log` job is broken — investigate BDL game log call
- If `mlb_player_stats` count = 0 but `mlb_game_log` has rows: `_ingest_mlb_box_stats` is failing silently — investigate BDL `get_mlb_stats` call
- If `mlb_player_stats` has rows but `player_rolling_stats` is empty: `_compute_rolling_windows` has a bug
- If `player_rolling_stats` has rows but `player_scores` is empty: `_compute_player_scores` has a bug

### A-5: Dead `_refresh_ros_projections` v1 at line ~5952 (low priority)
Python last-definition-wins: v2 at line ~6822 is active. Remove v1. Do when deploying for something else.

---

## Kimi Research Queue

### K-NEXT-1: Savant pitch quality ✅ COMPLETE — flags remain FALSE

Scores now differentiated (88.8–112.6) after `pa` proxy fix (d319beb). But `avg_sample_confidence=0.191` is below the >0.3 threshold. This is a season-length issue — PA accumulates over time. Re-check late May when avg PA > 120. No code change needed.

### K-NEXT-2: Yahoo ID sync — ✅ RESOLVED (architect fix applied)

Kimi's report incorrectly identified a duplicate function. Actual bug: INSERT at line ~2459 only handled `_pim_yahoo_key_uc` on conflict, allowing `_pim_bdl_id_uc` to kill the entire transaction on any duplicate bdl_id. Fixed with SAVEPOINT pattern.

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
