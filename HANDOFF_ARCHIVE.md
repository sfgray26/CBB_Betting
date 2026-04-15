# HANDOFF.md Archive — Historical Context & Reference

> **Purpose:** This file contains historical sections removed from `HANDOFF.md` to reduce
> per-session token load. Read only when you need background on completed work.
> **Last archived:** April 15, 2026

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

### Previous Session -- Data Quality Hardening Sprint (April 12)

**Critical Fixes:**
- **Statcast Column Mapping (Task 0): FIXED** -- `transform_to_performance()` column mismatch fixed.
- **Category Tracker Bug (Task 1): FIXED** -- Removed dead code on line 59.
- **Valuation Worker Logging (Task 2): FIXED** -- Replaced silent `.get(key, 0.0)` defaults with explicit warnings.
- **Production Schedules (Task 6): FIXED** -- Restored 5 jobs from test schedules to production times.

**New Infrastructure:**
- **FanGraphs RoS -> CSV Bridge (Task 3): IMPLEMENTED**
- **Projection Export Endpoint (Task 4): IMPLEMENTED**
- **Pipeline Freshness Validator (Task 5): IMPLEMENTED**

**Test suite:** 1738 passed, 3 pre-existing DB-auth failures.

---

## COMPLETED WORK (Phases 1-7 + Production Deployment)

### Phase 1: Player Identity Resolution (Tasks 1-3) -- COMPLETE
### Phase 2: Empty Tables Diagnosis (Tasks 4-6) -- COMPLETE (diagnosis only)
### Phase 3: Computed Fields (Task 7) -- COMPLETE (at mathematical floor)
### Phase 4: Backtest Enhancement (Task 8) -- COMPLETE
### Phase 5: Player Metrics (Task 9) -- COMPLETE
### Phase 6: Data Quality Fixes (Tasks 10-11) -- COMPLETE
### Phase 7: Root Cause Fixes (Tasks 19-29) -- COMPLETE
### Production Deployment (P-1..P-4) -- COMPLETE

**Full report:** `reports/2026-04-11-production-deployment-results.md`

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

**Current Health: C+ (76.9%)** — superseded by April 15 audit (B+).

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

362 orphaned position_eligibility records are retired/prospect/minor-league Yahoo players with no MLB/BDL entry. Do NOT re-run `backend/scripts/link_orphans.py`.

**Export:** `reports/2026-04-11-unmatchable-orphans.csv`

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

---

## AGENT TEAM

| Agent | Role | Current Assignment |
|-------|------|-------------------|
| **Claude Code** | Master Architect -- algorithms, schema, core logic | Tasks 4-9 feature implementation |
| **Gemini CLI** | Ops/DevOps -- Railway deploy, verification, smoke tests | Monitor Railway; run backfills |
| **Kimi CLI** | Deep research, spec memos, API audits | K-39..K-43 infra hardening research |

---

## VALIDATION TOOLS

| Endpoint/Script | Purpose |
|-----------------|---------|
| `/admin/validation-audit` | Comprehensive data quality validation |
| `/admin/backfill-ops-whip` | OPS/WHIP backfill |
| `/admin/backfill/statcast` | Statcast backfill |
| `/admin/diagnose-era` | ERA value diagnostics |

---

## KEY FILES

| What you need | Where it is |
|---------------|-------------|
| Current mission + task queue | `HANDOFF.md` |
| Project orientation | `CLAUDE.md` |
| Agent roles + swimlanes | `ORCHESTRATION.md`, `AGENTS.md` |
| Production deployment report | `reports/2026-04-11-production-deployment-results.md` |
| Database health report | `reports/2026-04-11-comprehensive-database-health-report.md` |
| VORP implementation guide | `reports/2026-04-09-vorp-implementation-guide.md` |
| Z-score best practices | `reports/2026-04-09-zscore-best-practices.md` |
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

## OUTDATED: DEVOPS & PIPELINE AUDIT (April 14, 2026)

*Superseded by April 15 comprehensive audit. Kept here for historical reference only.*

- Statcast quality audit (pre-fix): **42.4% zero-metric rate** (13,653 rows). Root cause: pitch-level overwrite.
- Fuzzy linker: 0 linked, 362 remaining orphans (prospects/retired).
- CS backfill: 0 updates due to empty `cs > 0` source data.
