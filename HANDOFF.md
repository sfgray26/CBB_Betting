# HANDOFF.md -- MLB Platform Master Plan (In-Season 2026)

> **Date:** April 12, 2026 | **Author:** Claude Code (Master Architect)
> **Status:** DATA QUALITY PHASE COMPLETE -- ALL FOLLOW-UPS RESOLVED -- READY FOR FEATURE WORK
>
> **Current phase:** Tasks 4-9 feature implementation using K-34..K-38 research.
> All 4 production follow-ups from P-1..P-4 are resolved. Database at C+ (76.9%), target B+ via quick wins.

---

## CURRENT SESSION STATE (April 12, 2026)

### What Just Happened
- P-1..P-4 production deployment fixes: ALL COMPLETE
- FOLLOW-UP 1 (Statcast persistence bug): RESOLVED -- commit `6848acb`
- FOLLOW-UP 2 (Stale validation audit): RESOLVED -- commit `c66c445`
- FOLLOW-UP 3 (player_id_mapping dedupes): LOW priority, deferred
- FOLLOW-UP 4 (backfill-ops-whip overcount): RESOLVED -- commit `b11c1e4`

### What's Next
**Tasks 4-9: Feature implementation** -- all research (K-34..K-38) is complete and unblocked.

---

## DATABASE STATE (April 12, 2026)

| Table | Expected | Actual | Status | Notes |
|-------|----------|--------|--------|-------|
| `player_id_mapping` | ~2,000 | ~20,000 (with duplicates) | POPULATED | FU-3: dedup deferred |
| `position_eligibility` | ~750 | ~750 | OK | 362 permanent orphans (prospects) |
| `mlb_player_stats` | ~13,500 | ~5,632 | PARTIAL | Growing daily via BDL sync |
| `probable_pitchers` | ~30/day | 0 | EMPTY | BDL API lacks probable pitcher data (K-37 confirmed). Needs MLB Stats API source. |
| `statcast_performances` | ~20,000 | 0 (code fixed, backfill not yet run) | FIX DEPLOYED | Run `POST /admin/backfill/statcast` on Railway to populate |
| `player_projections` | varies | 0 | EMPTY | No projection pipeline built yet |
| `player_rolling_stats` | ~25,000 | ~25,581 | OK | Populated by rolling stats job |
| `player_scores` | ~25,000 | ~25,506 | OK | Populated by scoring engine |
| `simulation_results` | ~8,500+ | ~8,523 | STALE | Last updated 2026-04-07 -- needs investigation |
| `decision_results` | varies | varies | OK | Fed by simulation engine |
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

### IMMEDIATE: Tasks 4-9 Feature Implementation

All K-34..K-38 research is complete. These tasks build the production data pipeline:

| Task | Description | Research | Key Finding | Priority |
|------|-------------|----------|-------------|----------|
| **4** | Probable pitchers pipeline | K-37 | BDL lacks this data; must use MLB Stats API `hydrate=probablePitcher` | HIGH |
| **5** | Statcast production backfill | (FU-1 fix) | Code fixed; run backfill on Railway, verify row counts | HIGH |
| **6** | Data ingestion logging | -- | Build audit trail for all sync jobs | MEDIUM |
| **7** | Derived stats computation | K-34, K-36 | BDL `/stats` + `/season_stats` for daily box scores; H2H scoring formulas | HIGH |
| **8** | VORP implementation | K-38 | Formula: Player_Z - Replacement_Z by position; replacement levels documented | MEDIUM |
| **9** | Z-score enhancements | K-35 | Winsorization at 5th/95th; MAD-based robust Z-scores; sample size stabilization | MEDIUM |

**Specific implementation notes:**

**Task 4 (Probable Pitchers):** The current `_sync_probable_pitchers()` in daily_ingestion.py fetches games from BDL and tries `getattr(game, 'home_probable', None)` -- but `MLBGame` has no such field. BDL does not provide probable pitcher data (K-37 confirmed). Must switch to MLB Stats API: `https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher`. The `mlb_analysis.py` already uses statsapi for schedule -- extend it for probable pitchers.

**Task 5 (Statcast Backfill):** Code is fixed (commit `6848acb`). Need to deploy to Railway and run `POST /admin/backfill/statcast`. Expected: ~15K rows for March 20 - April 11.

**Task 7 (Derived Stats):** BDL `/season_stats` provides pre-aggregated season data. Need to compute: NSB=SB-CS, NSV=SV-BS for H2H scoring (K-36). Position eligibility rules: 5 GS/10 GP batters, 3 starts SP, 5 apps RP.

**Task 8 (VORP):** Replacement levels by position: C=-5.5, 1B=-3.0, 2B/SS=-4.0, 3B=-3.5, OF=-2.5. Multi-eligible players use scarcest position. Formula: Player Total Z-Score - Replacement Level Z-Score.

**Task 9 (Z-Score Enhancements):** scoring_engine.py already computes Z-scores with Z_CAP=3.0. Enhancements: add Winsorization at 5th/95th percentiles, MAD-based robust option, sample size minimum thresholds.

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
100_028 probable_pitchers
Next available: 100_029
```

---

**Last Updated:** April 12, 2026
**Session Context:** All follow-ups resolved. Beginning Tasks 4-9 feature implementation.
**Priority:** HIGH -- probable pitchers pipeline (Task 4) + Statcast backfill (Task 5) first, then derived stats (Task 7).
