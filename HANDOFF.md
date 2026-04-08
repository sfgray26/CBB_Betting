# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 7, 2026 (updated Session S26) | **Author:** Claude Code (Master Architect)
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
| **H2H One Win UI Data Layer** | **BLOCKED — Phase 1 Required** | Kimi research complete (2026-04-08); technical roadmap issued. CF/LF/RF granularity, NSB/QS/K/9 data gaps identified. See Directive 2. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `mlb_player_stats` | **LIVE.** Pydantic validation relaxed for partial BDL objects. FK integrity enforced via game-ID-first fetch. |
| `daily_snapshots` | **LIVE.** `_ds_date_uc` constraint added. End-to-end pipeline health tracking operational. |
| Ingestion Pipeline | `run-pipeline` endpoint expanded to 11 jobs. Sequential execution verified. |

---

## Session History (Recent)

### S26 — Pipeline Hardening & Phase 10 Complete (Apr 7)

**Hardening:**
- `mlb_player_stats`: Relaxed validation for `jersey`, `birth_place`, `height`, etc. to handle incomplete BDL responses.
- `mlb_box_stats`: Refactored to fetch `game_ids` from DB first, ensuring FK integrity. Added `begin_nested` savepoints for row-level robustness.
- `statcast`: Fixed date range (Baseball Savant uses strict inequalities) and parameter encoding (removed manual `%7C` which was double-encoded).
- `daily_snapshots`: Added missing `_ds_date_uc` unique constraint to both DBs.
- `run-pipeline`: Expanded to all 11 jobs. Fixed `statcast` job registration.

**Verification:**
- `mlb_player_stats`: **646 rows** successfully upserted and verified in production.
- Pipeline: End-to-end manual trigger success (returning results for all 11 jobs).

---

### Gemini CLI — Ingestion Audit & Fix (S26) ✅ COMPLETE

**Status:**
- Root cause for 0-record ingestion identified (Strict validation + FK violations).
- Models patched and deployed.
- DB Constraints manually added to Legacy/Fantasy DBs.
- Pipeline manual trigger expanded and verified.
- `mlb_player_stats` count: 646.

---

### S27 — Fantasy Baseball UI Roadmap Analysis (Apr 8) ✅ COMPLETE

**Kimi Research Review:**
- Reviewed `reports/2026-04-08-fantasy-baseball-ui-ux-research.md` (Kimi's H2H One Win UI/UX research)
- 8 UI features identified: Weekly Compass, Position Scarcity, Two-Start Command Center, NSB Efficiency, IP Bank, Waiver Priority, IL Shuffle, Matchup Difficulty
- 7 UI-optimized view models specified (Section 2.2 of research doc)
- Validation checklist defined (Section 4.5)

**Current State Audit:**
- `models.py`: MLB Phase 2 tables live (P11-P20), but NO PositionEligibility table for CF/LF/RF granularity
- `yahoo_client_resilient.py`: OAuth client robust; stat batch endpoint returns {stat_id: value} dicts; NSB (5070), QS (32), K/9 (3096) NOT confirmed in data pipeline
- `schemas.py`: Current response models (DailyLineupResponse, WaiverWireResponse, RosterMoveRecommendation) partial; missing 7 UI-optimized view models from research
- `dashboard_service.py`: Dashboard aggregation exists (DashboardService class), but no WeeklyCompass or IPBank models
- `waiver_edge_detector.py`: Position group mapping exists but CF/LF/RF NOT distinguished (_POS_GROUP maps OF to ["OF","LF","CF","RF"])
- `daily_lineup_optimizer.py`: Two-start detection exists (_fetch_probable_pitchers_for_date), ProbablePitcherInfo model in schemas but not exposed via API

**Critical Gaps Identified:**
1. **Yahoo API NSB:** stat_id 5070 may not be exposed via Fantasy API; fallback to Statcast CS required
2. **Position Granularity:** Current system treats OF as monolithic; research requires CF/LF/RF for scarcity calculations
3. **Monte Carlo:** P16 ROS simulation exists but optimizes for points; H2H One Win requires category-by-category win probability
4. **WebSocket Layer:** Missing entirely for real-time updates (research specifies <400ms latency target)
5. **Cache Layer:** No Redis; all queries hit DB directly

**Deliverable Created:**
- `reports/2026-04-08-fantasy-baseball-ui-roadmap.md` — 8-week technical implementation roadmap
  - Phase 1: Data Layer Hardening (Weeks 1-2) — Schema extensions, models, validation suite
  - Phase 2: Compute Layer (Weeks 3-4) — Monte Carlo H2H, two-start detection, scarcity index
  - Phase 3: API Layer (Weeks 5-6) — 8 new REST endpoints, error contracts, caching
  - Phase 4: Data Pipeline (Week 7) — 5 new daily jobs, Yahoo sync validation
  - Phase 5: UI Component Specs (Week 8+) — Delegated to Kimi CLI after validation

**Decision:** PROCEED to Phase 1 (Data Layer Hardening). Do NOT begin UI design until validation checklist passes.

**HANDOFF.md Updates:**
- Added Directive 2: Fantasy Baseball UI Data Layer requirements
- Updated Platform State to track UI readiness
- Roadmap document saved for Kimi handoff reference

**Next Steps (Immediate):**
1. Create PositionEligibility model + migration script
2. Verify Yahoo API returns NSB (stat_id 5070) or identify fallback
3. Build H2HOneWinSimulator prototype
4. Implement validation suite (`tests/test_fantasy_h2h_validations.py`)
