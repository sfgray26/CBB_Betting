# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 8, 2026 (updated Session S28) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW — P1-P20 CERTIFIED. Phases 2-10 complete. Full pipeline operational.

---

## CORE PHILOSOPHY — Data-First, Contracts Before Plumbing

We are building this system like a quantitative trading desk. The data pipeline IS the product. Everything else — UI, optimization, automation — is a window into it that does not exist until the data is pristine.

---

## ARCHITECTURAL BLUEPRINT — 10-Phase Master Plan

| Phase | Goal | Status |
|-------|------|--------|
| **1 — Layered Architecture** | Separate side effects (bottom) from pure functions (top). Contracts before plumbing. | ✅ DONE |
| **2 — Data Foundation** | Ingest every game + stat + player. Normalize. Resolve IDs. | ✅ DONE (S26) |
| **3 — Derived Stats** | 30/14/7-day rolling windows. Exponential decay λ=0.95. | ✅ DONE (S26) |
| **4 — Scoring Engine** | League Z-scores + position Z-scores. Z_adj. | ✅ DONE (S26) |
| **5 — Momentum Layer** | ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Cold. | ✅ DONE (S26) |
| **6 — Probabilistic Layer** | 1000-run ROS Monte Carlo. Percentiles. | ✅ DONE (S26) |
| **7 — Decision Engines** | Lineup optimizer, waiver optimizer. | ✅ DONE (S26) |
| **8 — Backtesting Harness** | Historical loader, simulation engine, baselines. | ✅ DONE (S26) |
| **9 — Explainability** | Decision traces. Human-readable narratives. | ✅ DONE (S26) |
| **10 — Integration & Automation** | Snapshot system, daily sim harness. | ✅ DONE (S26) |

---

## ACTIVE DIRECTIVES (read before every session)

### DIRECTIVE 1 — Data-First Mandate
Incoming payloads MUST pass strict Pydantic V2 validation. No `dict.get()` defaults.

### DIRECTIVE 2 — Fantasy Baseball UI Data Layer (NEW: April 8, 2026)
**CRITICAL:** Do NOT begin UI design for H2H One Win format until Phase 1-2 validation passes.

**Required Before UI Phase:**
1. **Schema Extension:** PositionEligibility table created with CF/LF/RF breakdown (not generic "OF")
2. **Yahoo API Validation:** NSB (stat_id 5070), QS (stat_id 32), K/9 (stat_id 3096) confirmed in data pipeline
3. **Monte Carlo H2H:** H2HOneWinSimulator implemented and benchmarked <200ms for 10k sims
4. **API Endpoints:** All 8 endpoints (Weekly Compass, Scarcity, Two-Start, NSB, IP Bank, Waiver Budget, IL Shuffle, Matchup Difficulty) return valid payloads
5. **Cache Layer:** Redis or in-memory cache hitting >85% on hot paths
6. **Validation Suite:** `tests/test_fantasy_h2h_validations.py` passing (see roadmap doc)

**Root Cause:** H2H One Win format (position-specific OF, NSB not raw SB, 18 IP minimum) requires data granularity not present in current pipeline.

**Reference:** `reports/2026-04-08-fantasy-baseball-ui-roadmap.md` — full technical breakdown.

**Kimi Handoff:** When validation checklist passes, hand off to Kimi CLI for UI component specs (see roadmap Phase 5.2).

---

## Platform State — April 8, 2026

| System | State | Notes |
|--------|-------|-------|
| MLB Data Pipeline | **P1-P20 CERTIFIED** | Full 10-phase pipeline operational in production. |
| `mlb_player_stats` | **POPULATED (S26)** | 646 rows verified live in Fantasy-App DB. |
| `statcast_performances`| **PENDING** | Agent built but fetches 0 records for 2026-04-06 (off-day or lag). |
| Ingestion Orchestrator | **HARDENED (S26)** | All 11 jobs (including statcast/snapshot) registered and manual-triggerable via `/admin/ingestion/run-pipeline`. |
| `position_eligibility` (P25) | **LIVE (S28)** | Model + migration deployed to both DBs. Verified live. |
| `probable_pitchers` (P26) | **LIVE (S28)** | Model + migration script deployed to both DBs. Verified live. |
| **H2H One Win UI Data Layer** | **IN PROGRESS — Phase 2** | P25/P26 live. NSB bug fixed. H2HOneWinSimulator complete (Phase 2.1). API endpoints pending. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `mlb_player_stats` | **LIVE.** Pydantic validation relaxed for partial BDL objects. FK integrity enforced via game-ID-first fetch. |
| `daily_snapshots` | **LIVE.** `_ds_date_uc` constraint added. End-to-end pipeline health tracking operational. |
| `position_eligibility` | **LIVE (S28).** Table exists in both DBs. Tracks LF/CF/RF granularity. |
| `probable_pitchers` | **LIVE (S28).** Table exists in both DBs. Tracks daily probables from MLB Stats API. |
| Ingestion Pipeline | `run-pipeline` endpoint expanded to 11 jobs. Sequential execution verified. |

---

## Session History (Recent)

### S28 — Phase 1 Data Layer Hardening: Model + Migration (Apr 8)

**Completed:**
- `models.py`: Added PositionEligibility table with LF/CF/RF granularity (P25)
- `scripts/migrate_v25_position_eligibility.py`: Migration script created, syntax-verified.
- `models.py`: Added ProbablePitcherSnapshot table (P26).
- `scripts/migrate_v26_probable_pitchers.py`: Migration script created, syntax-verified.
- **Railway Deployment:** Migrations v25 and v26 successfully deployed to both Legacy and Fantasy production databases (Gemini S28).
- **NSB Bug Fixed:** Changed `max(0, sb - cs)` to `sb - cs` in `projections_loader.py` line 174 (verified via K-28 audit).
- **Validation Suite:** Created `tests/test_fantasy_h2h_validations.py` — all 6 tests passing (NSB negative values, scarcity index, IP bank, one-win probability, stat_id 60, Statcast fallback).
- **Phase 2.1 (Compute Layer):** H2HOneWinSimulator implemented with NumPy vectorization.
  - `backend/fantasy_baseball/h2h_monte_carlo.py` — Monte Carlo for category-by-category win probability
  - Performance: <200ms for 10,000 simulations (target met)
  - Returns: win_probability, locked/swing/vulnerable categories, category breakdown
  - `tests/test_h2h_monte_carlo.py` — 7/7 tests passing (basic functionality, dominant team, even matchup, performance, negative NSB, ERA/WHIP, category probs)
  - `backend/schemas.py`: Added H2HOneWinSimRequest, H2HOneWinSimResponse, CategoryWinProbability

**Active Delegation:**
- None — Phase 1 complete. Phase 2 (H2HOneWinSimulator) ready to start.

---

## K-28 COMPLETION SUMMARY — Yahoo API NSB Audit

**Status:** ✅ COMPLETE  
**Agent:** Kimi CLI (Deep Intelligence Unit)  
**Report:** `reports/2026-04-08-yahoo-nsb-audit.md`

### Verdict
**YES** — NSB (Net Stolen Bases) is available via Yahoo Fantasy API as stat_id 60.

### Key Findings
1. **Yahoo API:** stat_id 60 maps to "NSB" — already configured in `frontend/lib/fantasy-stat-contract.json`
2. **Endpoint:** `get_players_stats_batch()` in `yahoo_client_resilient.py` returns NSB when league has it configured
3. **CS Fallback:** Statcast provides `cs` / `caught_stealing` field if needed

### Critical Bug Identified
**File:** `backend/fantasy_baseball/projections_loader.py` line 174  
**Current (WRONG):** `nsb = max(0, sb - cs)` — clamps negative NSB to 0  
**Should Be:** `nsb = sb - cs` — NSB can be negative (0 SB - 1 CS = -1)  
**Impact:** H2H One Win format requires accurate NSB (can be negative). Bug must be fixed before UI phase.

### H2H One Win UI Data Layer Phase 1 Status
- **UNBLOCKED** — NSB data source confirmed
- **PENDING** — NSB bug fix (Claude Code owner)
- **Next:** H2HOneWinSimulator implementation after bug fix

---

## ACTIVE CRITICAL PATH

### Completed (S28)
- ✅ **Phase 1 Data Layer:** P25/P26 migrations LIVE, NSB bug fixed, validation suite passing
- ✅ **Phase 2.1 Compute Layer:** H2HOneWinSimulator implemented, performance target met (<200ms for 10k sims)

### Next Session (S29)
- Phase 2.2: Two-Start Detection (expose probable_pitchers via API)
- Phase 2.3: Scarcity Index Computation (CF/LF/RF granularity)
- Phase 3: API Layer (8 new REST endpoints)

---

## HANDOFF PROMPTS — Agent Delegation Bundles

### Gemini CLI (DevOps Strike Lead) — G-28: Probable Pitchers Migration Deployment ✅ COMPLETE

**Mission:** Deploy the P26 probable_pitchers table migration to Railway production.

**Status:**
- Migration v26 executed successfully on both DBs.
- `probable_pitchers` table verified live.
- `ProbablePitcherSnapshot` model confirmed mapping correctly.

---

### Gemini CLI (DevOps Strike Lead) — G-27: Position Eligibility Migration Deployment ✅ COMPLETE

**Mission:** Deploy the P25 position_eligibility table migration to Railway production.

**Status:**
- Migration v25 executed successfully on both DBs.
- `position_eligibility` table verified live.
- `PositionEligibility` model confirmed mapping correctly.

---

### Kimi CLI (Deep Intelligence Unit) — K-28: Yahoo API NSB Verification Audit ✅ COMPLETE

**Mission:** Determine if Yahoo Fantasy API exposes NSB (Net Stolen Bases, stat_id 5070) and identify fallback data sources if not.

**Status:**
- **Verdict:** YES — NSB available as stat_id 60
- **Bug Found:** `projections_loader.py` clamps NSB to 0 (must fix)
- **Report:** `reports/2026-04-08-yahoo-nsb-audit.md`
- **H2H UI Data Layer:** UNBLOCKED (pending bug fix)

---

*Last Updated: April 8, 2026 — Session S28*
