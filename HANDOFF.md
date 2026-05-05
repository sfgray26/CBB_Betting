# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-05-05 | **Architect:** Claude Code (Master Architect)
> **Status:** EPIC 2 sprint_speed ✅ LIVE. Stuff+/Location+ deferred P2. Savant Pitch Quality implemented behind disabled flags. Savant park factors implemented as canonical snapshot source.
> **HEAD:** Run `git log --oneline -1` for latest branch commit; Savant Pitch Quality implemented in current branch.
> **Deploy status:** ✅ LIVE — `/health` = `{"status":"healthy","database":"connected","scheduler":"running"}`.

---

## Sprint 1b Architecture Decisions (2026-05-05) — WRITTEN AND LOCKED

These three decisions unblock Sprint 2. They were deferred from Sprint 1b and are
now resolved based on direct code inspection of the production pipeline.

---

### Decision 1: Counting-Stat Pipeline

**Decision: Hybrid with explicit provenance per field. Bayesian stays rate-stat-only.
Counting stats are sourced from the FanGraphs RoS ensemble blend (Steamer/ATC/ZIPS),
written to `player_projections` by `_update_ensemble_blend`. No Bayesian extension planned.**

Rationale:
- `BayesianProjectionUpdater` (statcast_ingestion.py:797) conjugate-updates wOBA only.
  Extending it to HR/SB/R/RBI would require a non-conjugate likelihood or Marcel-style
  regression model. That is Sprint 3+ scope.
- `player_board.py` already documents: "Fusion engine does NOT compute cat_scores".
  The ensemble blend IS the counting-stat prior — it is not a gap, it is the design.
- `CanonicalProjection` projection fields populated by source:

| Field | Source table | Populated by |
|---|---|---|
| `proj_avg`, `proj_ops`, `proj_era`, `proj_whip` | `player_projections.avg/.ops/.era/.whip` | Bayesian posterior (wOBA-anchored) + ensemble passthrough |
| `proj_hr`, `proj_r`, `proj_rbi`, `proj_sb` | `player_projections.hr/.r/.rbi/.sb` | FanGraphs RoS ensemble blend (`_update_ensemble_blend`) |
| `proj_w`, `proj_sv`, `proj_k` | `player_projections.w/.sv/.k_pit` | FanGraphs RoS ensemble blend |
| `proj_xwoba`, `proj_xera` | `statcast_batter_metrics.xwoba`, `statcast_pitcher_metrics.xera` | Savant leaderboard ingestion (`_ingest_savant_leaderboards`) |

Sprint 2 `ProjectionService.build_canonical()` reads from `player_projections` for
counting stats and `statcast_batter/pitcher_metrics` for Savant rates. Both tables are
live and current.

---

### Decision 2: Advanced Metrics Storage — No New Table

**Decision: `player_advanced_metrics` will NOT be created as a new table.
`statcast_batter_metrics` and `statcast_pitcher_metrics` are the canonical advanced-metrics
store. `CanonicalProjection` denormalizes the relevant fields inline at build time.**

Rationale:
- `StatcastBatterMetrics` already holds: `xwoba`, `barrel_percent`, `hard_hit_percent`,
  `avg_exit_velocity`, `whiff_percent`, `sprint_speed`.
- `StatcastPitcherMetrics` already holds: `xera`, `xwoba`, `barrel_percent_allowed`,
  `hard_hit_percent_allowed`, `k_percent`, `bb_percent`, `k_9`, `whiff_percent`,
  `stuff_plus`, `location_plus`.
- `CanonicalProjection` already has these as direct columns
  (`proj_xwoba`, `proj_xera`, `barrel_pct`, `k_pct`, `bb_pct`, etc.).
- A new merged table would be a third copy of data that already has two live copies.
  The only legitimate reason to add it would be time-series history (one row per
  player per date). That requirement does not exist in the current product.
- The SLA monitoring bug (querying `player_daily_metrics WHERE data_source='statcast'`
  instead of `statcast_batter_metrics.last_updated`) has been fixed in
  `_check_projection_freshness` as of this session.

If a future requirement for time-series advanced metrics emerges, extend
`StatcastBatterMetrics` / `StatcastPitcherMetrics` with a `metric_date` compound key
rather than creating a fourth storage layer.

---

### Decision 3: TeamContext Contract

**Decision: Phase B `WaiverValuationService` TeamContext uses `StatcastPerformance`
actuals (season-to-date) as the primary accumulator source. For players without
MLBAM resolution, Yahoo `player_projections` priors are the fallback.
Quarantined players are excluded from roster-specific delta; they do not block the call.**

Full contract:

**Source of truth:** `StatcastPerformance` (table: `statcast_performances`)
- Fields used: `pa`, `hr`, `r`, `rbi`, `sb`, `h`, `ab` (season accumulators)
- Join key: `mlbam_id`
- Populated by: `_update_statcast` → `run_daily_ingestion` (every 6 hours)

**Refresh cadence:** Tied to `_update_statcast` interval (6 hours).
TeamContext is recomputed on each `WaiverValuationService` call; no separate cache layer.

**Fallback for unresolved players (no MLBAM ID):**
Use `PlayerProjection.hr/.r/.rbi/.sb` as season-accumulator estimates.
These are RoS full-season numbers, not actuals — apply a prorated discount:
`prorate(proj_stat, games_elapsed / 162)` using `get_remaining_games(today)` from
`simulation_engine.py`.

**Quarantine player handling:**
- Players in `identity_quarantine` are excluded from the roster-specific delta
  (they contribute 0 to `current_team_hr`, etc.).
- They are NOT excluded from the league-baseline calculation — league baseline comes
  from the full active-player pool in `cat_scores_builder.py`.
- They ARE excluded from waiver recommendations (existing behavior, unchanged).
- Log a warning when a quarantine-excluded player is on the active roster.

**Rate-category denominator safety:**
- If active roster has fewer than 5 MLBAM-resolved players (identity coverage < ~30%),
  do not compute the Phase B roster-specific delta. Fall back to Phase A (league-median)
  marginal impact only. Log at WARNING level.

---

### Bug Fixes Applied This Session

| Bug | Fix | File |
|---|---|---|
| SLA monitor queries wrong table for statcast | Changed `_check_projection_freshness` to query `GREATEST(MAX(last_updated) FROM statcast_batter_metrics, statcast_pitcher_metrics)` | `backend/services/daily_ingestion.py` |
| `savant_ingestion` job registered twice (6 AM + 2 AM) | Removed dead 6 AM registration; kept 2 AM P0-B FIX as sole entry | `backend/services/daily_ingestion.py` |
| `ros_projection_refresh` registered twice (3:35 AM + 3:30 AM) | Removed dead 3:35 AM registration; kept 3:30 AM P0-A FIX as sole entry | `backend/services/daily_ingestion.py` |

---

## Executive Summary

**Completed this session (May 5, 2026):**
| Item | Result |
|------|--------|
| **Sprint 2: Canonical Projection System** | ✅ COMPLETE — Steps 1-6 of 7 implemented (step 7 = flag enable pending prod validation) |
| Sprint 2 Step 1: `fusion_engine.py` `to_season_counts()` | ✅ 10 tests — hybrid counting-stat provenance (HR/SB from rates, R/RBI/SV static, W formula, K formula) |
| Sprint 2 Step 2: `id_resolution_service.py` `get_quarantined_identity_ids()` | ✅ 5 tests — returns PENDING_REVIEW proposed_player_ids as set |
| Sprint 2 Step 3: `projection_assembly_service.py` (new) | ✅ 34 tests — assembles CanonicalProjection rows + CategoryImpact z-scores; source_engine SAVANT_ADJUSTED vs STATIC_BOARD |
| Sprint 2 Step 4: `daily_ingestion.py` canonical_projection_refresh job | ✅ lock 100_040, 11 PM ET cron, checks CANONICAL_PROJECTION_V1 flag |
| Sprint 2 Step 5: `team_context.py` + `waiver_valuation_service.py` (new) | ✅ 17 tests — TeamContext ephemeral dataclass, add/drop surplus for all 10 fantasy cats |
| Sprint 2 Step 6: `category_aware_scorer.py` TeamContext depth_factor | ✅ 7 new tests (29 total) — backwards-compatible, depth_factor clamps [0.8, 1.2] |
| Sprint 2 Step 7: Enable CANONICAL_PROJECTION_V1 | ⏳ PENDING — awaits production validation (first nightly run + identity_miss rate check) |
| Architecture decisions written + locked | ✅ Sprint 1b decisions in HANDOFF.md — Decision 1 (hybrid counting), Decision 2 (no player_advanced_metrics), Decision 3 (TeamContext contract) |
| Full test suite | ✅ **2684 passed / 4 skipped / 0 failed** (confirmed clean run) |
| Phase 4: Market Signals Engine | ✅ COMPLETE — All 4 PRs implemented |
| PR 4.3/4.4: Market engine | ✅ 29 tests passing (pure computation module) |
| PR 4.2: Market signals job | ✅ Scheduled 8:30 AM ET (lock 100_038) |
| PR 4.5: Market score tiebreaker | ✅ Integrated into waiver_edge_detector |
| EPIC 2: sprint_speed ingestion | ✅ LIVE (Savant — no Cloudflare) |
| EPIC 2: Stuff+/Location+ pipeline | ⚠️ P2 DEFERRED — FanGraphs/Cloudflare blocks Railway IP |
| EPIC 2: fangraphs_scraper.py | ✅ Code correct — 21 tests passing |
| EPIC 2: DB columns added | ✅ stuff_plus / location_plus in statcast_pitcher_metrics |
| EPIC 2: Feature flags | ✅ Disabled (statcast_stuff_plus_enabled=false) |
| Savant Pitch Quality | ✅ IMPLEMENTED — in-house Savant pitcher breakout score, inactive until DB migration/backfill/validation |
| Savant Park Factors | ✅ IMPLEMENTED — Baseball Savant Statcast park factor snapshot replaces legacy ESPN/BR constants as canonical DB source |
| Phase 4 handoff SQL corrected | ✅ feature_flags not threshold_config |
| Team role change: Codex replaces Gemini for blocking DevOps | ✅ Approved |

**Phase 4 Deliverables:**
- Pure computation module (`backend/services/market_engine.py`) — ownership_velocity, ownership_deltas, add_drop_ratio, market_score (0-100)
- Daily job (8:30 AM ET) — computes Yahoo ownership-based market intelligence
- Tiebreaker integration — market_score as tertiary sort key in waiver recommendations
- Confidence gating — low sample sizes dampen contrarian signals
- Feature flags — `market_signals_enabled` = false, `opportunity_enabled` = false

**Previous session (May 2-3, 2026):**
| Item | Result |
|------|--------|
| M34 migration (junction DB) | ✅ Gemini ran successfully — 441 hitters + 176 pitchers classified |
| cat_scores backfill | ✅ 617 players now have valid cat_scores |
| MCMC win_prob=0.763 fix | ✅ Deployed (`2a736cc`) — `_build_proxy_cat_scores()` returns `{}` when `total_z=0.0` |

**K-34 P0 Statcast bugs (also complete):**
| Bug | Fix | Commit | Status |
|-----|-----|--------|--------|
| Bug 1: `team` in required_cols blocks all ingestion | Removed `team` from validator | `ba033a5` | ✅ Deployed |
| Bug 2: xwoba_diff=0 — name JOIN mismatch | JOIN on MLBAM ID instead of name | `2c2dd1f` | ✅ Deployed |
| Bug 3: xera_diff=0 — ERA null for all pitchers | Computed ERA subquery from statcast_performances | `0426b7e` | ✅ Deployed |

**Statcast ingestion confirmed**: 723 records/day, `is_valid: true, error_count: 0`

**New Savant Pitch Quality path (May 5):**
- Design: `docs/superpowers/specs/2026-05-05-savant-pitch-quality-design.md`
- Plan: `docs/superpowers/plans/2026-05-05-savant-pitch-quality.md`
- Calculator: `backend/fantasy_baseball/savant_pitch_quality.py`
- ORM table: `SavantPitchQualityScore` → `savant_pitch_quality_scores`
- Migration: `scripts/migration_savant_pitch_quality.py`
- Backfill: `scripts/backfill_savant_pitch_quality.py`
- Flags: `scripts/seed_savant_pitch_quality_flags.py`

Feature flags default disabled:
- `savant_pitch_quality_enabled=false`
- `savant_pitch_quality_waiver_signals_enabled=false`
- `savant_pitch_quality_projection_adjustments_enabled=false`

Purpose: replace the blocked automated FanGraphs Stuff+/Location+ path for waiver/breakout detection with a transparent Savant-native 100-centered score. This does **not** activate production waiver or projection behavior until migration, backfill, distribution validation, and explicit flag enablement.

### Savant Park Factors (canonical projection context)

**Status:** Code implemented, DB rollout pending.

Baseball Savant Statcast park factors are now the canonical park-factor source:
- Snapshot: `data/park_factors/savant_park_factors_2025_3yr.json`
- Loader/converter: `backend/fantasy_baseball/savant_park_factors.py`
- Runtime lookup: `backend/fantasy_baseball/ballpark_factors.py`
- ORM: `ParkFactor` expanded with Savant columns (`woba_factor`, `xwobacon_factor`, `so_factor`, `bb_factor`, 1B/2B/3B factors, etc.)
- Migration: `scripts/migration_savant_park_factors.py`
- Seeder: `scripts/seed_savant_park_factors.py`

Important logic change:
- Savant indexes are 100-centered; loader converts to 1.00-centered factors.
- `park_adjusted_era()` now multiplies by the environment factor. A hitter-friendly park (`era_factor > 1`) should raise projected ERA; the old division direction understated pitcher risk.
- Savant 2025 three-year table currently returns 28 MLB venues. TB/OAK remain covered by the legacy constant fallback until Savant has stable rolling-year data for their current homes.

Railway rollout commands when approved:
```powershell
railway run python scripts/migration_savant_park_factors.py
railway run python scripts/seed_savant_park_factors.py
```

---

## Current Sprint: Lineup UI Data Binding (May 3-7, 2026)

**Milestones**:
- [x] **Milestone 1** - M34 player_type discriminator code | DONE
- [x] **Milestone 2** - Yahoo API caching | DONE
- [x] **Milestone 3** - K-34 Statcast audit (Kimi) | DONE
- [x] **Milestone 4** - K-34 P0 bug fixes (3/3) | DONE (May 2)
- [x] **Milestone 5** - M34 migration on correct DB → cat_scores live | DONE (May 2, Gemini)
- [x] **Milestone 6** - MCMC baseline win_prob=0.763 fix | DONE (May 2, `2a736cc`)
- [x] **Milestone 7** - 48h signal validation | ✅ PASSED (May 3) — cat_scores 100% non-zero, xwOBA diffs vary
- [x] **Milestone 8** - ADP collision fix (Yainer Diaz) | ✅ DONE (May 3) — first_3_char disambiguation
- [ ] **Milestone 9** - Full test suite green gate | ⏳ IN PROGRESS (running)
- [ ] **Milestone 10** - Lineup UI data binding (player_id, game_time, SP scores) | **NEXT**

**Completed This Session (May 3):**
| Item | Result |
|------|--------|
| Signal validation (Task 1) | ✅ PASSED — cat_scores 100% non-zero (20/20 hitters), xwOBA signals present |
| game_time spot-check (Task 2) | ✅ VERIFIED — commit 9426f69 deployed (player_id, game_time UTC, SP detection) |
| ADP collision fix (Task 3) | ✅ DONE (a9cc2ce) — first_3_char disambiguation for y_diaz collisions |
| Full test suite (Task 4) | ✅ PASSED — all tests green (exit code 0) |
| HANDOFF.md update (Task 5) | ✅ DONE — Milestones 7-10 complete |

**Session Summary:**
All signal validation gates passed. Production confirmed non-zero cat_scores and 
xwOBA signals. ADP collision bug fixed. Full test suite green. Critical analysis 
completed: identified gap between "institutional-grade" vision (SYSTEM_ARCHITECTURE_ANALYSIS.md) 
and actual implementation (basic fantasy platform with static projections).

**Next Session:** Fix P0 data quality bugs (player_type NULL, Yahoo ID sync), then decide on performance strategy.

---

## Week 1 Complete: Agent Reports Summary

**All verification tasks complete (May 3, 2026):**
| Agent | Task | Status | Report |
|-------|------|--------|--------|
| Kimi | Data Quality Audit | ✅ Complete | `reports/2026-05-03-data-quality-audit.md` |
| Gemini | Endpoint Verification | ✅ Complete | `reports/2026-05-03-endpoint-verification.md` |
| Gemini | Performance Investigation | ✅ Complete | `reports/2026-05-03-optimizer-performance.md` |
| Claude | Week 1 Synthesis | ✅ Complete | `reports/2026-05-03-week1-synthesis.md` |

**Critical Findings:**
- **2 P0 data bugs found:** player_type NULL (71%), Yahoo ID coverage 3.7%
- **N+1 confirmed:** ballpark_factors.py queries DB in loop → 27s waiver delay
- **Performance baseline:** Optimizer 0.28s ✅, Dashboard 10s ⚠️, Waiver 27s ❌
- **All endpoints work:** No 500 errors, all return 200

**Next Actions (See Week 1 Synthesis):**
1. Fix player_type NULL (backfill from positions) → 45 min
2. Fix Yahoo ID sync (create scheduler job) → 1h 15min
3. Fix N+1 (bulk-load park_factors) → 1 hour
4. Debug matchup_preview null → 30 min

**Total estimated time:** 3 hours (all reversible, low risk)

**Recommendation:** Fix all P0 bugs now for clean foundation (see `reports/2026-05-03-week1-synthesis.md`)

---

## Critical Analysis Summary (May 3, 2026)

**Vision vs. Reality Gap:**
- **Promised:** Institutional-grade quantitative asset management with Bayesian updating, MCMC simulation, GNNs, contextual bandits
- **Actual:** Functional fantasy platform with **static** Steamer projections from March 9, basic Statcast ingestion, 38 endpoints (quality unknown)

**Tier 1 Gaps (Foundational — Must Verify):**
1. Deployed features not tested end-to-end (38 endpoints, unknown which work)
2. Data quality unknown (NULL checks, rolling window accuracy, Statcast freshness)
3. Performance issues (lineup optimizer timing out at 30s)

**Tier 2 Gaps (High Impact — Missing):**
4. Bayesian projection updating (projections don't learn from 2026 season)
5. Matchup quality engine (no pitcher xERA, no platoon splits)
6. MCMC UI integration (simulation works but users can't see "70% chance to win HR")

**Minimum Viable "Institutional-Grade" Feature:**
**Bayesian Projection Updating** — Posterior = Prior × Likelihood. Projections update daily with new Statcast data. Success: Rookie called up May 1 → projection updated by May 2.

**Recommendation:** Complete Tasks 1-3 (verification), then build Bayesian updater + live data pipeline before UI work.

---

## Open Investigations

### 48h Signal Validation (P1 — gate before UI work)
All structural fixes are deployed. Now need 48h of real-world data to confirm signals are real:
- `xwoba_diff` and `xera_diff` should show non-zero distribution across players (not all 0.0)
- `win_prob` in MCMC matchup endpoint should vary 0.3–0.8 across different matchups (not locked at 0.763)
- Sample check: `SELECT player_name, xwoba_diff, xera_diff FROM player_scores WHERE updated_at > NOW() - INTERVAL '48 hours' LIMIT 20`

**Gate**: Do NOT proceed to frontend dashboard signal display until both conditions confirmed.


---

## Known Blockers — Infrastructure

### Stuff+ / Location+ (FanGraphs/Cloudflare — P2)

**Status:** Code complete, data unavailable from Railway.

FanGraphs routes `leaders-legacy.aspx` through Cloudflare with IP-reputation blocking. Railway's IP range is blocked regardless of User-Agent headers. The UA patch (`_patch_pybaseball_user_agent`) applies and executes, but Cloudflare returns 403 before the response reaches the application layer.

**Confirmed:** `pybaseball_loader.py` also fails to fetch live FanGraphs data from Railway — it works via 24-hour file cache seeded from developer machines.

**Impact:** `PLUS_STUFF` signal never fires. This was already the case before EPIC 2 (columns existed in ORM, never populated). Not a regression.

**Feature flags:** `statcast_stuff_plus_enabled=false`, `statcast_location_plus_enabled=false` — safe to leave disabled.

**P2 resolution options (pick one when signal becomes needed):**
1. **Manual CSV snapshot** — User downloads FanGraphs pitching leaderboard CSV from browser monthly → drops at `data/fangraphs_pitcher_quality_YYYY.csv` → backfill script reads it. Requires ~15 lines added to `fangraphs_scraper.py`.
2. **Savant proxy** — Replace with Savant-native metrics (xERA, barrel%, whiff%) which don't require FanGraphs. Good proxy but not Stuff+/Location+ specifically.
3. **FanGraphs API subscription** — Paid API access bypasses Cloudflare. ~$80/year.

**Do not:** spend more time on Cloudflare bypass (UA tricks, rotating proxies, cloudscraper) — these fail at the IP-reputation layer regardless.

### Savant Pitch Quality (FanGraphs replacement path — inactive)

**Status:** Code implemented, DB rollout pending.

The platform now has a Savant-native pitcher quality score for waiver and breakout detection:

```text
savant_pitch_quality =
  35% arsenal_quality
  30% bat_missing_skill
  20% contact_suppression
  15% command_stability
  + trend_adjustment
  x sample_confidence
```

Signals produced by the calculator:
- `BREAKOUT_ARM`
- `SKILL_CHANGE`
- `STREAMER_UPSIDE`
- `WATCHLIST`
- `RATIO_RISK`

**Agent routing:**
- Claude: owns architecture review, final activation decision, and any waiver/projection integration.
- Codex: may run migration/backfill/verification and implement bounded calculator/script fixes.
- Kimi: may use MLB MCP/Savant research to audit endpoint fields and score validity.
- Gemini: may run only pre-approved Railway scripts; no code edits.

**Railway rollout commands when approved:**
```powershell
railway run python scripts/migration_savant_pitch_quality.py
railway run python scripts/seed_savant_pitch_quality_flags.py
railway run python scripts/backfill_savant_pitch_quality.py
```

**Validation queries after backfill:**
```sql
SELECT COUNT(*) FROM savant_pitch_quality_scores WHERE season = 2026;
SELECT player_name, savant_pitch_quality, sample_confidence, signals
FROM savant_pitch_quality_scores
WHERE season = 2026
ORDER BY as_of_date DESC, savant_pitch_quality DESC
LIMIT 20;
```

**Gate:** Keep all three `savant_pitch_quality_*` feature flags disabled until distribution and known-player sanity checks pass.

---

## Canonical Projection Architecture Decisions (May 5, 2026)

> **Status:** DECIDED — unblocks Sprint 2 implementation.
> Applies to: `fusion_engine.py`, `statcast_ingestion.py`, `player_board.py`, future `projection_assembly_service.py`, `waiver_valuation_service.py`, `category_impact_builder.py`.

### Sprint 1b → Sprint 2 State Boundary

**Sprint 1b (COMPLETE):**
- V35 schema deployed: `canonical_projections`, `category_impacts`, `divergence_flags`, `player_identities`, `identity_quarantine`
- `IdentityResolutionService` implemented (`id_resolution_service.py`)
- `player_board.py` ilike fallback removed (exact-match resolution only)
- `backfill_player_identities.py` run on Railway: 6,997 rows in `player_identities`
- `CANONICAL_PROJECTION_V1` feature flag seeded, disabled
- Test suite green (2604 pass)

**Sprint 2 scope:** ProjectionAssemblyService + CategoryImpactBuilder + WaiverValuationService (TeamContext). These three decisions define the contracts Sprint 2 must implement.

---

### Decision 1 — Counting-Stat Pipeline: Hybrid Provenance by Field

**Decision:** Option C — hybrid. Each category has an explicit provenance rule.

| Stat | Source | Method |
|------|--------|--------|
| `proj_hr` | Fusion (Bayesian) | `hr_per_pa × projected_pa` |
| `proj_sb` | Fusion (Bayesian) | `sb_per_pa × projected_pa` |
| `proj_r` | Static Steamer board | Direct from `player_board.py` tuple (no formula) |
| `proj_rbi` | Static Steamer board | Direct from `player_board.py` tuple (no formula) |
| `proj_w` | Formula | `PitcherCountingStatFormulas.project_wins(proj_era, projected_ip)` |
| `proj_sv` | Static Steamer board | Direct from `player_board.py` tuple (closer role required) |
| `proj_k` | Formula | `proj_k9 × projected_ip / 9` |

**Rationale:**
- HR and SB are individual-skill outputs (power, speed) derivable from per-PA rates. HR/PA stabilizes at 170 PA — enough signal by May for season regulars.
- R and RBI depend on lineup position and teammate OBP — no individual-level Statcast formula is reliable. Static Steamer season totals are the best prior we have and they update slowly enough (projected through season end) that mid-season drift is tolerable.
- W depends on team run support — ERA and IP are the best individual proxies. `PitcherCountingStatFormulas` (already in `fusion_engine.py`) handles this.
- SV requires role designation (closer vs. non-closer) that cannot be inferred from Statcast. Static Steamer encodes this via projected SV totals.
- K (strikeouts pitched) is a direct rate stat. K/9 × IP / 9 is exact.

**`projected_pa` / `projected_ip` source in Sprint 2:** Read from `player_board.py` hardcoded tuples (batter `pa` column, pitcher `ip` column). These are Steamer full-season projections. The `ProjectionAssemblyService` will store them in `canonical_projections.projected_pa` / `projected_ip` for all downstream derivations.

**Do NOT extend `BayesianProjectionUpdater` to compute R/RBI/SV.** These stats cannot be reliably updated per-player from Statcast alone. The Bayesian update layer (rate stats only) remains correct as designed.

**Affected files — Sprint 2:**
- `backend/fantasy_baseball/fusion_engine.py`: Add `to_season_counts(projected_pa: float, projected_ip: float) -> dict` on `FusionResult`. Maps `hr_per_pa × pa → proj_hr`, `sb_per_pa × pa → proj_sb`, `k_per_nine × ip / 9 → proj_k`, calls `PitcherCountingStatFormulas` for `proj_w`.
- `backend/fantasy_baseball/projection_assembly_service.py` (NEW): Orchestrates all source reads, applies hybrid provenance, writes `CanonicalProjection` rows with `source_engine=BAYESIAN` (fusion-updated) or `STATIC_BOARD` (board-only).
- `backend/fantasy_baseball/statcast_ingestion.py` (`BayesianProjectionUpdater`): Sprint 2 migration — write output to `canonical_projections` instead of (or in addition to) `player_projections`. No counting stat extension.
- `backend/services/daily_ingestion.py`: Job `_refresh_canonical_projections()` wired (post-Statcast, ~11 PM ET, lock **100_040** — 100_039 taken by matchup_context_update).

---

### Decision 2 — Advanced-Metrics Storage: No Third Table

**Decision:** `player_advanced_metrics` table does NOT exist and should NOT be created.

**Canonical arrangement:**

| Table | Role | Writer | Reader |
|-------|------|--------|--------|
| `statcast_batter_metrics` | Savant batter leaderboard data | Savant ingestion pipeline | `ProjectionAssemblyService` only |
| `statcast_pitcher_metrics` | Savant pitcher leaderboard data | Savant ingestion pipeline (+ FanGraphs for Stuff+/Location+ when unblocked) | `ProjectionAssemblyService` only |
| `canonical_projections` | Denormalized read surface for all scoring consumers | `ProjectionAssemblyService` | `WaiverValuationService`, `CategoryAwareScorer`, dashboard, all future services |
| `savant_pitch_quality_scores` | In-house pitcher quality score (time-series, purpose-specific) | `SavantPitchQualityCalculator` | Waiver signals (when enabled) |

**Read-write contract:**
1. Ingestion jobs write to `statcast_batter_metrics` / `statcast_pitcher_metrics` (upsert on `mlbam_id`, keyed per season).
2. `ProjectionAssemblyService` joins both ingestion tables into `canonical_projections`, populating `woba`, `xwoba`, `barrel_pct`, `hardhit_pct`, `xslg`, `xba` (batters) and `era`, `whip`, `k9`, `fip`, `xera`, `csw_pct`, `swstr_pct`, `savant_pitch_quality_score` (pitchers).
3. All downstream reads go to `canonical_projections` only — never direct joins to ingestion tables from scoring or valuation logic.

**Why not a merged Savant+FanGraphs table:** Both ingestion tables already store the columns. A merge table adds a third write target with no schema benefit. Denormalization lives in `canonical_projections` where it belongs.

**Why not a time-series layer now:** Season-to-date snapshots are sufficient for 2026. Daily delta tracking is a Sprint 3+ concern when we need trend signals for waiver alerts. The `savant_pitch_quality_scores` table with `as_of_date` is the pattern to follow when that time comes.

**Affected files — Sprint 2:**
- No schema changes. All tables exist.
- `backend/fantasy_baseball/projection_assembly_service.py` (NEW): Contains the JOIN logic reading from `statcast_batter_metrics` and `statcast_pitcher_metrics`.
- `category_aware_scorer.py`: Unchanged in Sprint 2. Continues reading `cat_scores` from board layer.

---

### Decision 3 — TeamContext Contract: Runtime Struct, Quarantine Players Excluded

**Decision:** `TeamContext` is a **runtime-assembled ephemeral dataclass**, not a DB table. Assembled fresh per waiver/lineup API call.

**Contract:**
```python
@dataclass
class TeamContext:
    roster_player_ids: list[int]          # canonical player_identities.id (resolved only)
    projected_pa_by_player: dict[int, float]   # batter PA remaining this season
    projected_ip_by_player: dict[int, float]   # pitcher IP remaining this season
    rate_pa_denominator: float            # sum(projected_pa_by_player.values())
    rate_ip_denominator: float            # sum(projected_ip_by_player.values())
    quarantined_player_ids: set[int]      # players in identity_quarantine PENDING_REVIEW
```

**Assembly sources:**
1. Yahoo roster API (live) → `roster_player_ids` after identity resolution
2. `CanonicalProjection` → `projected_pa` / `projected_ip` per player
3. `IdentityResolutionService.get_quarantined_ids(session)` (new method) → exclusion set

**Quarantine behavior:**
- Players in `identity_quarantine` with `status=PENDING_REVIEW` contribute 0 PA and 0 IP.
- They are excluded from `roster_player_ids` entirely.
- Rate category denominators (AVG, ERA, WHIP, K9) use only resolved players' projected PA/IP.
- Effect: slightly conservative rate denominators (team accrues fewer expected AB/IP than a fully-resolved roster). This is intentional — unresolved identity = treat as unknown = do not pollute rate math.

**Refresh cadence:** No persistence. Built on-demand. Yahoo roster state (IL moves, daily pickups) invalidates any cached version within hours. The `CanonicalProjection` data refreshes nightly — no intra-day staleness concern there.

**Affected files — Sprint 2:**
- `backend/fantasy_baseball/waiver_valuation_service.py` (NEW): Assembles `TeamContext`, computes add/drop surplus using `CanonicalProjection` as data source.
- `backend/fantasy_baseball/id_resolution_service.py`: Add `get_quarantined_ids(session: Session) -> set[int]` method.
- `backend/fantasy_baseball/category_aware_scorer.py`: Extend `compute_need_score()` to accept optional `TeamContext` for rate-denominator weighting (backwards-compatible: defaults to current behavior when `None`).

---

### Sprint 2 Implementation Sequence

Sequence is ordered by dependency. Each item is an independent commit.

1. ✅ **`fusion_engine.py`** — `to_season_counts()` added. 10 tests in `test_fusion_engine.py`.
2. ✅ **`id_resolution_service.py`** — `get_quarantined_identity_ids()` added. 5 tests in `test_id_resolution_service.py` (new file).
3. ✅ **`projection_assembly_service.py`** (new) — Full assembly pipeline. 34 tests in `test_projection_assembly_service.py` (new file).
4. ✅ **`daily_ingestion.py`** — Job `_refresh_canonical_projections()` wired. Lock **100_040** (100_039 was already taken by `matchup_context_update`). ~11 PM ET cron. In `_all_job_ids` and dispatch dict.
5. ✅ **`team_context.py`** + **`waiver_valuation_service.py`** (new) — TeamContext dataclass + WaiverValuationService. 17 tests.
6. ✅ **`category_aware_scorer.py`** — `compute_need_score()` extended with optional `TeamContext` depth_factor. 7 new tests (29 total).
7. ⏳ **Feature flag** — Enable `CANONICAL_PROJECTION_V1` after first nightly run validates identity_miss rate and CanonicalProjection row counts.

**Sprint 2 status:** Steps 1-6 DONE. Step 7 requires production run first.

### Blockers Removed

- Ambiguity about `proj_hr`/`proj_sb`/`proj_r` source → **resolved**: hybrid provenance table above
- Ambiguity about whether to create `player_advanced_metrics` → **resolved**: no; `canonical_projections` is the read surface
- Ambiguity about `TeamContext` persistence → **resolved**: runtime struct, never persisted

### Blockers Remaining

- `projected_pa`/`projected_ip` values for players not on `player_board.py` (e.g. promoted rookies mid-season). Sprint 2 workaround: use population-prior PA (450 for hitters, 130 for SPs, 60 for RPs) when player is absent from board. Permanent fix is a roster-aware playing-time model (Sprint 3).
- FanGraphs Stuff+/Location+ blocked by Cloudflare at Railway IP layer (Decision 2 is unaffected — columns exist in `statcast_pitcher_metrics`, remain NULL until P2 resolution).
- `WaiverValuationService` depends on `CanonicalProjection` being populated for a reasonable fraction of the roster. First-run bootstrap required (run `ProjectionAssemblyService` manually or await first nightly job).

---

## Delegation Queue (P0 — Week 1 Priority)

### 🔧 Task 0: Phase 4 + EPIC 2 Deployment (NEW — May 5, 2026)
**Assigned:** Codex (Implementation Lead + Deployment Verification)
**Status:** ⏳ PENDING — Gemini removed from blocking DevOps ownership
**Estimated time:** 30 minutes

**Context:**
Phase 4 (Market Signals Engine) is complete but requires DevOps verification before production activation.

**Tasks:**
1. **Run migration on Railway:**
   ```bash
   railway ssh
   python scripts/migration_player_market_signals.py
   ```
   Expected output: "PR 4.1 migration ready." Verify table created:
   ```sql
   SELECT table_name FROM information_schema.tables
   WHERE table_name = 'player_market_signals';
   ```

2. **Seed feature flags in threshold_config:**
   ```sql
   INSERT INTO threshold_config (key, value, description)
   VALUES
     ('market_signals_enabled', 'false', 'PR 4: Market score tiebreaker in waiver recommendations'),
     ('opportunity_enabled', 'false', 'PR 3: Opportunity adjustment in player_scores')
   ON CONFLICT (key) DO NOTHING;
   ```
   Verify:
   ```sql
   SELECT key, value FROM threshold_config WHERE key LIKE '%_enabled';
   ```

3. **Verify daily job scheduler:**
   - Check that lock ID 100_038 is registered: `grep "market_signals_update" backend/services/daily_ingestion.py`
   - Verify job scheduled for 8:30 AM ET: `grep -A5 "market_signals_update" backend/services/daily_ingestion.py | grep "8:30"`
   - Check logs for job execution: `railway logs --filter "market_signals_update"`

4. **Run tests on Railway:**
   ```bash
   railway ssh
   python -m pytest tests/test_market_engine.py tests/test_waiver_edge.py -v --tb=short
   ```
   Expected: All 43 tests passing (29 market_engine + 14 waiver_edge)

5. **Verify advisory lock:**
   ```sql
   SELECT pg_try_advisory_lock(100_038);
   -- Should return 't' (true)
   SELECT pg_advisory_unlock(100_038);
   ```

**Success criteria:**
- [ ] Migration runs without errors
- [ ] player_market_signals table exists
- [ ] Feature flags seeded correctly
- [ ] All 43 tests pass on Railway
- [ ] Lock ID 100_038 available
- [ ] Daily job scheduled (will run at next 8:30 AM ET)

**Report template:**
```markdown
## Phase 4 Deployment Verification Report

**Date:** 2026-05-05
**Engineer:** Gemini CLI

### Migration Status
- player_market_signals table: [CREATED/FAILED]
- Indexes created: [YES/NO]

### Feature Flags
- market_signals_enabled: [SEEDED/MISSING]
- opportunity_enabled: [SEEDED/MISSING]

### Test Results
- test_market_engine.py: [X/29] passing
- test_waiver_edge.py: [X/14] passing
- Total: [X/43] passing

### Scheduler Verification
- Lock ID 100_038: [AVAILABLE/TAKEN]
- Job scheduled: [YES/NO]
- Next run: [8:30 AM ET tomorrow]

### Issues Found
[None or list of issues]

### Recommendation
[PROCEED TO PRODUCTION / NEEDS FIXES]
```

**Escalation:** If migration fails or tests don't pass on Railway, report specific error messages for Architect review.

---

### 🔧 Task 0b: EPIC 2 Railway DB Migration + Backfill (NEW — May 5, 2026)
**Assigned:** Codex (Implementation Lead)
**Status:** ⏳ PENDING — code committed `c213ba2`, needs Railway execution

**Context:**
EPIC 2 (Stuff+/Location+) scraper and daily job are complete and committed. The DB columns
don't exist yet in Railway — the migration must run before any data lands.

**Step 1: Apply column migration**
```bash
railway run python -c "
import psycopg2, os
conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('ALTER TABLE statcast_pitcher_metrics ADD COLUMN IF NOT EXISTS stuff_plus FLOAT')
cur.execute('ALTER TABLE statcast_pitcher_metrics ADD COLUMN IF NOT EXISTS location_plus FLOAT')
conn.commit()
cur.close(); conn.close()
print('Migration complete.')
"
```
Or run: `railway run python -c "$(cat scripts/migration_add_statcast_advanced.sql)"` (pure SQL variant).

**Step 2: Run backfill**
```bash
railway run python scripts/backfill_statcast_pitcher_advanced.py
```
Expected: logged coverage ≥ 70% for both stuff_plus and location_plus.

**Step 3: Seed feature flags**
```bash
railway run python scripts/seed_statcast_advanced_flags.py
```
Expected: "Successfully seeded 4 feature flags"

**Step 4: Verify**
```sql
SELECT COUNT(*) FILTER (WHERE stuff_plus IS NOT NULL), COUNT(*)
FROM statcast_pitcher_metrics WHERE season = 2026;
```
Expected: non-zero / total ≥ 70%.

**Success criteria:**
- [ ] `statcast_pitcher_metrics.stuff_plus` column exists
- [ ] `statcast_pitcher_metrics.location_plus` column exists
- [ ] Backfill logged ≥ 70% coverage
- [ ] Feature flags seeded (statcast_stuff_plus_enabled=false, statcast_location_plus_enabled=false)

**Escalate to Architect if:** migration errors, coverage < 30%, or Savant returns empty CSV.

---

### ✅ Task 1: Fantasy Feature Verification
**Assigned:** Gemini CLI (DevOps)
**Status:** ✅ COMPLETE
**Report:** `reports/2026-05-03-endpoint-verification.md`

**Findings:**
- All endpoints return 200 (no broken endpoints)
- Optimizer fast: 0.28s (uses pre-calculated DB scores)
- Dashboard slow: 19.34s → 9.95s after caching (50% improvement)
- Waiver very slow: 23.78s → 26.82s (N+1 bottleneck)
- N+1 confirmed in ballpark_factors.py

**Escalation:** None needed, performance optimization complete

---

### ✅ Task 2: Data Quality Audit
**Assigned:** Kimi CLI (Deep Intelligence)
**Status:** ✅ COMPLETE
**Report:** `reports/2026-05-03-data-quality-audit.md`

**Findings:**
- ✅ cat_scores coverage: 100% (621/621)
- ✅ Statcast fresh: 18.4 hours
- ✅ ESPN overlap: ~15/20 (75%)
- ❌ **P0 BUG #1:** player_type NULL for 71% (441/621)
- ❌ **P0 BUG #2:** Yahoo ID coverage 3.7% (372/10,096)
- ⚠️ Pitcher ERA correlation weak: r=0.1569
- ⚠️ player_scores 20.4 hours stale

**Escalation:** Fix P0 bugs immediately (see Week 1 Synthesis report)

---

### ✅ Task 3: Optimizer Performance Investigation
**Assigned:** Gemini CLI (DevOps)
**Status:** ✅ COMPLETE
**Report:** `reports/2026-05-03-optimizer-performance.md`

**Findings:**
- Caching implemented: lru_cache on _get_player_board (5min TTL)
- Dashboard: 19.34s → 9.95s (50% improvement)
- Waiver: 26.82s (no improvement - different bottleneck)
- N+1 root cause: ballpark_factors.py queries DB for every player
- Missing indexes: None found (all present)

**Escalation:** Bulk-load park_factors to fix remaining N+1

---

## P0 Data Quality Fixes (Immediate - 2 hours)

---

## P0 Data Quality Fixes — STATUS: PARTIAL COMPLETE

**Deployment:** Commit `e64c0c4` (May 3, 2026) | **Railway:** ✅ LIVE

### Fix #1: player_type NULL Backfill ✅ SUCCESS
- **Before:** 441 NULL rows (71%)
- **After:** 0 NULLs remaining
- **Impact:** Batter routing now works correctly

### Fix #2: Yahoo ID Sync ⚠️ BUG FIXED — Pending execution
- **Before:** 3.7% coverage (372/10,096)
- **After:** Bug fixed (wrong method name), job scheduled for 6 AM ET daily
- **Issue:** AttributeError `get_league_players` → Fixed to `get_league_rosters`
- **Status:** Job will run overnight at 6 AM ET, verify coverage >50% tomorrow

### Fix #3: Park Factors Bulk-Load ✅ SUCCESS
- **Before:** Waiver endpoint 27s (N+1 queries)
- **After:** Waiver endpoint 0.3s (90x faster!)
- **Proof:** 81 park factors loaded on startup, cached in memory

**Next:** Verify Yahoo ID coverage after 6 AM ET job runs, then proceed to Milestone 10

---

## Deployment Queue

Nothing pending. `2a736cc` deployed via `railway up --detach`.

**Next deploy trigger**: After synthetic CSV gitignore (P2) or signal validation reveals additional fixes.

---

## Historical Archives

---

## Historical Archives

Previous sessions are archived in `docs/handoff_archives/`:
- April 21-27: Sessions F-H → [session-f-through-h.md](docs/handoff_archives/2026-04-21-to-27-session-f-through-h.md)
- April 28-29: Sessions I-M → [session-i-through-m.md](docs/handoff_archives/2026-04-28-through-29-sessions-h-through-m.md)
- April 29-30: Sessions N-U → [session-n-through-u.md](docs/handoff_archives/2026-04-29-through-30-sessions-n-through-u.md)
- April 30: Gemini deployment W+X+Y → [gemini-deployment-wxy.md](docs/handoff_archives/2026-04-30-gemini-deployment-wxy.md)
- May 1: Session Z (statcast fixes) → [session-z-statcast-fixes.md](docs/handoff_archives/2026-05-01-session-z-statcast-fixes.md)
- May 1: Sessions Y+AA → [sessions-y-and-aa.md](docs/handoff_archives/2026-05-01-sessions-y-and-aa.md)
- May 1: Session AC (odds migration) → [session-ac-odds-migration.md](docs/handoff_archives/2026-05-01-session-ac-odds-migration.md)

**Kimi Research Findings**:
- Data quality audit (K-33) → [reports/kimi-k33-data-quality-crisis.md](reports/kimi-k33-data-quality-crisis.md)
- All findings → [reports/](reports/)

---

## Database Connection Reference

> ⚠️ TWO Postgres services exist in this Railway project. Use the correct one.

| Service | Proxy URL | Purpose |
|---------|-----------|---------|
| **Postgres-ygnV** ✅ CORRECT | `junction.proxy.rlwy.net:45402` | Fantasy app + MLB data (production) |
| Postgres ❌ WRONG | `shinkansen.proxy.rlwy.net:17252` | Old CBB betting DB — archived |

**Correct public URL for local migrations/audits:**
```
postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway
```

PowerShell usage:
```powershell
$env:DATABASE_URL = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
.\venv\Scripts\python scripts/your_script.py
```

**Verified row counts (May 2, 2026) — correct DB:**
- `statcast_performances`: 13,842 ✅
- `player_projections`: 617 ✅
- `mlb_game_log`: 490 ✅
- `mlb_player_stats`: 13,809 ✅
- `player_scores`: 77,517 ✅

**Session AI finding "all tables empty" was WRONG** — it queried the CBB Postgres, not Postgres-ygnV.

---

## Technical Reference

**Advisory Locks** (next available: 100_039 — no new locks this session):
- 100_001 mlb_odds | 100_002 statcast | 100_003 rolling_z | 100_004 cbb_ratings
- 100_005 clv | 100_006 cleanup | 100_007 waiver_scan | 100_008 mlb_brief
- 100_009 openclaw_perf | 100_010 openclaw_sweep
- 100_011 scarcity_index_recalc | 100_012 two_start_sp_identification
- 100_013 projection_model_update | 100_014 probable_pitcher_sync | 100_015 waiver_priority_snapshot
- 100_016 ros_projection_refresh | 100_034 yahoo_id_sync | 100_036 ros_projection_refresh
- 100_037 opportunity_update | 100_038 market_signals_update

**Key Files**:
- `backend/main.py` - FastAPI app, scheduler jobs
- `backend/fantasy_baseball/` - Fantasy platform modules
- `backend/services/mlb_analysis.py` - MLB betting analysis
- `backend/services/daily_lineup_optimizer.py` - Daily lineup optimization
- `backend/services/balldontlie.py` - BDL client (GOAT MLB tier)
- `backend/services/market_engine.py` - NEW: Market signals pure computation (PR 4.3/4.4)
- `backend/services/opportunity_engine.py` - NEW: Playing-time opportunity scoring (PR 3.2/3.3)
- `backend/services/waiver_edge_detector.py` - UPDATED: Market score tiebreaker (PR 4.5)

**Documentation Links**:
- [MLB Fantasy Roadmap](docs/MLB_FANTASY_ROADMAP.md)
- [Implementation Plans](docs/superpowers/plans/)
- [CLAUDE.md](CLAUDE.md) - Project orientation
- [ORCHESTRATION.md](ORCHESTRATION.md) - Agent team swimlanes
- [AGENTS.md](AGENTS.md) - Agent roles and responsibilities
