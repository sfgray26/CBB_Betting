# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-05-06 | **Architect:** Claude Code (Master Architect)
> **Branch:** `stable/cbb-prod` | **HEAD:** `612d351`
> **Deploy:** `/health` = `{"status":"healthy","database":"connected","scheduler":"running"}`

---

## System Status

| Component | Status | Notes |
|---|---|---|
| Railway deployment | LIVE | FastAPI + uvicorn, us-west1 |
| `canonical_projections` | LIVE (445 rows) | SAVANT_ADJUSTED=234, STATIC_BOARD=211 — 0 errors after BIGINT fix |
| `category_impacts` | LIVE (5,350 rows) | Per-category z-scores + marginal numerator/denominator columns |
| `player_projections` (FanGraphs RoS) | LIVE (9,686 rows) | prior_source=fangraphs_ros, updated 2026-05-06 15:15 ET |
| `player_projections.player_type` | ~71% NULL | Backfill script ready (A-1) — DevOps to run |
| `player_identities` | 454/454 resolved | 12-row backfill complete 2026-05-06 |
| `CANONICAL_PROJECTION_V1` flag | **true** | Nightly job runs at 11 PM ET |
| `market_signals_enabled` | false | Pending: C-7 → C-2 |
| `feature_matchup_enabled` | false | Pending: C-3 → C-4 |
| `opportunity_enabled` | false | Unblock after A-1 (player_type backfill) runs |
| `/api/fantasy/decisions` route | **LIVE** | Commit `b799aec` — was 404, now serves stored decisions |
| Stuff+/Location+ (FanGraphs) | BLOCKED | Cloudflare blocks Railway IP — P2, do not attempt |
| Savant Pitch Quality | Code done, DB pending | Pending Codex migration (Task C-5) |
| Savant Park Factors | Code done, DB pending | Pending Codex migration (Task C-6) |

---

## Code Fixes Deployed This Session (2026-05-06)

All bugs listed here are fixed and deployed. **No code action needed — just run the rollout steps below.**

| Commit | Bug | Fix |
|--------|-----|-----|
| `21f96df` | `ros_projection_refresh` UniqueViolation for two-way players (Ohtani, etc.) | `bat_processed_ids` set + `db.flush()` + pitcher merge preserves batting stats |
| `8388062` | `canonical_projections.player_id` INT4 overflow for large Yahoo IDs | `Column(Integer)` → `Column(BigInteger)` in ORM + migration script |
| `12b7f5a` | `migration_dedupe_player_id_mapping` UniqueViolation on `_pim_yahoo_key_uc` | Added Step 3a-pre: NULL loser `yahoo_key` values before merge runs |
| `b799aec` | Frontend `/decisions` page calling routes that returned 404 | Added `GET /api/fantasy/decisions` and `GET /api/fantasy/decisions/status` to `main.py` |
| `612d351` | `backfill_player_type.py` had hardcoded DB credential, missing `'both'` case, bad `::jsonb` cast | Full rewrite — removed credential, added `--dry-run`, fixed `::jsonb` and two-way logic |
| earlier | `yahoo_id_sync` UniqueViolation every run (48h outage) | Removed duplicate `_sync_yahoo_id_mapping` v2 (no conflict guard); v1 (guarded) is canonical |
| earlier | Yahoo sync covered only ~3.7% of BDL universe | FA enumeration paginated to 200/position (was 25) |
| earlier | `market_signals_update`: `column "name" does not exist` | `SELECT name` → `SELECT full_name` in `_compute_market_signals` |
| earlier | `opportunity_update`: `column "espn_id" does not exist` | Fixed JOIN to use `pim.mlbam_id::text = pp.player_id` |

---

## Codex DevOps Queue (ordered)

### ✅ C-1: BIGINT migration — COMPLETE
`canonical_projections.player_id` is now BIGINT in production. 445 upserted, 0 errors verified.

### C-7: Dedupe player_id_mapping — RETRY NEEDED
Fix committed (`12b7f5a`). Previous run failed on `_pim_yahoo_key_uc`; root cause fixed.
```bash
railway run python scripts/migration_dedupe_player_id_mapping.py --dry-run
# Confirm dry-run shows "Step 3a-pre" in output (the new NULL-losers step)
railway run python scripts/migration_dedupe_player_id_mapping.py
# Expected: SUCCESS, rows_before=11036, merged_count≈109, no UniqueViolation
# Sanity:
#   SELECT COUNT(*), COUNT(DISTINCT normalized_name) FROM player_id_mapping;
#   SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL AND bdl_id IS NULL;
#   -- target: orphans < 50 (was 451)
```
**STOP if UniqueViolation recurs on a different key — escalate to Claude with the key name.**

### C-A1: Backfill player_type (after C-7)
```bash
railway run python scripts/backfill_player_type.py --dry-run
# Confirm: shows ~9,000+ rows with sensible hitter/pitcher/both split
railway run python scripts/backfill_player_type.py
# Expected: "DONE — updated N rows" where N > 8000
# Then enable opportunity flag:
# UPDATE feature_flags SET enabled=true WHERE flag_name='opportunity_enabled';
```

### C-7b: Trigger yahoo_id_sync (after C-7)
```bash
curl -X POST https://<railway-url>/admin/ingestion/run/yahoo_id_sync \
     -H "X-API-Key: $ADMIN_API_KEY"
# Expected: {"status":"success","matched":>=600,"unmatched":<200}
# First successful run validates the 48h outage fix
```

### C-2: Enable market signals (after C-7)
```bash
railway connect postgres
# in psql:
UPDATE feature_flags SET enabled=true, updated_at=NOW() WHERE flag_name='market_signals_enabled';
# wait 90s for config cache TTL, then:
curl -X POST https://<railway-url>/admin/ingestion/run/market_signals_update \
     -H "X-API-Key: $ADMIN_API_KEY"
# Expected: {"status": "success", "updated": <N>}
```

### C-3: Seed matchup context flag
```bash
railway run python scripts/seed_matchup_context_flag.py
# Expected: "Seeded feature_matchup_enabled=false" (or "already exists, skipped")
```

### C-4: Enable matchup context (after C-3 + one successful cron run)
```bash
railway connect postgres
# in psql:
UPDATE feature_flags SET enabled=true, updated_at=NOW() WHERE flag_name='feature_matchup_enabled';
```

### C-5: Savant pitch quality rollout
```bash
railway run python scripts/migration_savant_pitch_quality.py
railway run python scripts/seed_savant_pitch_quality_flags.py
railway run python scripts/backfill_savant_pitch_quality.py
# Verify: SELECT COUNT(*) FROM savant_pitch_quality_scores WHERE season = 2026; -- > 200
```

### C-6: Savant park factors rollout
```bash
railway run python scripts/migration_savant_park_factors.py
railway run python scripts/seed_savant_park_factors.py
# Verify: SELECT COUNT(*) FROM park_factors; -- >= 28
```

**Reporting protocol:** After each task, report: exit code + last 5 stdout lines + verify query result. STOP on first failure.

---

## Claude Architect Queue

### ✅ A-1: Fix `player_type` NULL in `player_projections` — SCRIPT READY
`scripts/backfill_player_type.py` rewritten and committed (`612d351`). Handed to DevOps as C-A1.

### ✅ A-2: Yahoo ID sync coverage — COMPLETE
Root cause found (K-NEXT-2), duplicate method removed, pagination fix deployed. Validate via C-7b.

### ✅ A-3: Frontend `/decisions` endpoint missing — COMPLETE
`GET /api/fantasy/decisions` and `GET /api/fantasy/decisions/status` added to `main.py` (`b799aec`).
Joins `decision_results` + `decision_explanations` + `player_id_mapping` name lookup.
Returns `{decisions: [{decision, explanation}], count, as_of_date, decision_type}`.
Note: `decision_results` is populated by the P17 decision optimization job — rows will be empty
until that pipeline phase has run at least once.

### A-4 (P1): Sprint 3 — Lineup UI data binding
Wire `game_time`, `player_id`, `SP` scores into lineup optimizer API response.
Prerequisite: C-A1 (player_type backfill) confirmed in production.

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
| `market_signals_enabled` | false | Enable after C-2 confirmed success |
| `feature_matchup_enabled` | false | Enable after C-3/C-4 and one clean cron run |
| `statcast_stuff_plus_enabled` | false | Blocked by Cloudflare — do not enable |
| `statcast_location_plus_enabled` | false | Blocked by Cloudflare — do not enable |
| `savant_pitch_quality_enabled` | false | Enable after K-NEXT-1 validation passes |
| `savant_pitch_quality_waiver_signals_enabled` | false | Enable after main flag + waiver wiring review |
| `savant_pitch_quality_projection_adjustments_enabled` | false | Enable last — projection impact requires full validation |
| `opportunity_enabled` | false | Enable when A-1 (player_type backfill) is complete |

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
