# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-05-06 | **Architect:** Claude Code (Master Architect)
> **Branch:** `stable/cbb-prod` | **HEAD:** `ae1f823`
> **Deploy:** `/health` = `{"status":"healthy","database":"connected","scheduler":"running"}`

---

## System Status

| Component | Status | Notes |
|---|---|---|
| Railway deployment | LIVE | FastAPI + uvicorn, us-west1 |
| `canonical_projections` | LIVE (445 rows) | SAVANT_ADJUSTED=234, STATIC_BOARD=211 |
| `category_impacts` | LIVE (5,350 rows) | Per-category z-scores |
| `player_projections` (FanGraphs RoS) | LIVE (9,686 rows) | Updated 2026-05-06 15:15:53 |
| `player_identities` | 454/454 resolved | 12-row backfill complete 2026-05-06 |
| `CANONICAL_PROJECTION_V1` flag | **true** | Nightly job runs at 11 PM ET |
| `market_signals_enabled` | false | Pending Codex enable (Task C-2) |
| `feature_matchup_enabled` | false | Pending Codex seed + enable (Task C-3/C-4) |
| Stuff+/Location+ (FanGraphs) | BLOCKED | Cloudflare blocks Railway IP — P2 |
| Savant Pitch Quality | Code done, DB pending | Pending Codex migration (Task C-5) |
| Savant Park Factors | Code done, DB pending | Pending Codex migration (Task C-6) |

---

## P0 BLOCKING — Must Run Before Next Nightly Job

`canonical_projections.player_id` is `INTEGER` in the production DB. Commit `8388062` fixed the ORM to `BigInteger`. The migration script is written but not yet run. Until it runs, 9 Yahoo-ID-only players (Johnny Brito, JP France, Kenny Hernandez, Brent Honeywell Jr., etc.) fail assembly with `NumericValueOutOfRange`.

**Codex: run this first, before anything else:**
```
railway run python scripts/migration_canonical_player_id_bigint.py
```
Expected output: `"player_id column altered to BIGINT -- done"`
Verify: `SELECT pg_typeof(player_id) FROM canonical_projections LIMIT 1;` → `bigint`

---

## Codex DevOps Queue (ordered — complete P0 before proceeding)

### C-1 (P0): BIGINT migration
```bash
railway run python scripts/migration_canonical_player_id_bigint.py
# verify: SELECT pg_typeof(player_id) FROM canonical_projections LIMIT 1; -- bigint
```

### C-2: Enable market signals (after C-1)
```bash
railway connect postgres
# in psql:
UPDATE feature_flags SET flag_value='true', updated_at=NOW() WHERE flag_name='market_signals_enabled';
# wait 90s for config cache TTL, then:
curl -X POST https://<railway-url>/admin/ingestion/run/market_signals_update \
     -H "X-Admin-API-Key: $ADMIN_API_KEY"
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
UPDATE feature_flags SET flag_value='true', updated_at=NOW() WHERE flag_name='feature_matchup_enabled';
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

### A-1 (P0): Fix `player_type` NULL in `player_projections`
71% of rows have `NULL player_type`. The `ProjectionAssemblyService` cannot classify
these players correctly. Write `scripts/backfill_player_type.py`:
- Infer from `positions` array: any of `('SP','RP','P')` → `'pitcher'`; else → `'batter'`; both → `'both'`
- Run locally to verify, then hand to Codex for Railway execution.

### A-2 (P0): Investigate Yahoo ID sync coverage
`_sync_yahoo_id_mapping()` achieves ~3.7% coverage. Most waiver FAs never get BDL ID
enrichment, so market signals and matchup context are silently skipped for them.
Assign K-NEXT-2 to Kimi first (root cause analysis), then implement fix.

### A-3 (P1): Resolve frontend waiver endpoint mismatch
Frontend calls: `GET /api/fantasy/decisions?decision_type=waiver`
Backend serves: `GET /api/fantasy/waiver/recommendations`
Confirm with user: add alias route OR fix Next.js call. Do not change both sides.

### A-4 (P1): Sprint 3 — Lineup UI data binding (Milestone 10)
Wire `game_time`, `player_id`, `SP` scores into lineup optimizer API response.
Prerequisite: A-1 and A-2 resolved first.

---

## Kimi Research Queue

### K-NEXT-1: Savant pitch quality distribution validation
Run after Codex completes C-5. Pull `SELECT player_name, savant_pitch_quality, sample_confidence FROM savant_pitch_quality_scores WHERE season=2026 ORDER BY savant_pitch_quality DESC LIMIT 20` and the bottom 20. Cross-check vs known 2026 pitcher quality (Skenes, Wheeler, Flaherty at top; roster fillers at bottom). Report distribution shape, range, anomalies.
Output: `reports/2026-05-XX-savant-pitch-quality-validation.md`

### K-NEXT-2: Yahoo ID sync gap analysis
Inspect `_sync_yahoo_id_mapping()` in `backend/services/daily_ingestion.py`. Identify why
coverage is 3.7% — name normalization failures? `player_id_mapping` gaps? Missing BDL entries?
Propose fix: fuzzy threshold, BDL name normalization, or manual map patch.
Output: `reports/2026-05-XX-yahoo-id-sync-gap-analysis.md`

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
