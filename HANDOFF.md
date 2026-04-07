# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 6, 2026 (updated Session S23) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW-MODERATE — P1-P16 certified. Phases 2-6 complete. Next: P17 (Decision Engines — Phase 7).

---

## CORE PHILOSOPHY — Data-First, Contracts Before Plumbing

We are building this system like a quantitative trading desk. The data pipeline IS the product. Everything else — UI, optimization, automation — is a window into it that does not exist until the data is pristine.

**Five non-negotiable principles:**

1. **Data First:** The data pipeline is the entire product right now.
2. **Contracts Before Plumbing:** Define the shape of reality (Pydantic V2 models) before writing the API clients that fetch it.
3. **One Feed at a Time:** Do not move to odds or injuries until the core game schedule is pristine.
4. **No Silent Failures:** `dict.get()` with defaults is suppression, not validation. Every input passes strict schema validation.
5. **Strict Embargo:** All downstream logic (optimization, matchups, waivers, ensemble blending) remains cut off until the data floor is certified.

**Layer model:**

```
Layer 0: Decision Contracts (Pydantic V2 models — immutable truth of what valid data looks like)
Layer 1: Pure Intelligence Functions (stateless transforms: raw -> validated)
Layer 2: Data Adapters (API clients, ingestion — swappable plumbing)
Layer 3: Orchestration (schedulers, job queues — when things run)
Layer 4: Presentation (API endpoints, UI — the face)
```

Build bottom-up. Never build Layer 2 without Layer 0 contracts. Never build Layer 4 without Layer 1 proven.

---

## ARCHITECTURAL BLUEPRINT — 10-Phase Master Plan

This is the north star. Every session's work maps to one of these phases. Never skip a phase. Never build a higher phase without the lower phase proven.

| Phase | Goal | Status |
|-------|------|--------|
| **1 — Layered Architecture** | Separate side effects (bottom) from pure functions (top). Contracts before plumbing. | ✅ DONE — layer model established, Pydantic contracts in place |
| **2 — Data Foundation** | Ingest every game + stat + player. Normalize. Resolve IDs. Never compute from raw API. Build as standalone microservice: idempotent, raw+normalized dual-write, schema drift detection, anomaly detection. | ✅ DONE — Phase 2 complete (S20). All tables live and verified in production. |
| **3 — Derived Stats** | 30/14/7-day rolling windows. Exponential decay λ=0.95. Per-game aggregation. Hitter + pitcher parity. | ✅ DONE — Phase 3 complete (S21). `player_rolling_stats` verified live in production. |
| **4 — Scoring Engine** | League Z-scores + position Z-scores. Z_adj = 0.7·Z_league + 0.3·Z_position. Confidence regression. 0–100 output. | ✅ DONE — Phase 4 complete (S22). `player_scores` verified live in production. |
| **5 — Momentum Layer** | ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Cold / Collapsing / Breakout / Collapse. | ✅ DONE — Phase 5 complete (S22). `player_momentum` verified live in production. |
| **6 — Probabilistic Layer** | 1000-run ROS Monte Carlo. Percentiles (P10/25/50/75/90). Risk metrics. P(top-10/25/50). | ✅ DONE — Phase 6 complete (S23). `simulation_results` verified live in production. |
| **7 — Decision Engines** | Lineup optimizer, waiver optimizer, trade evaluator. World-with vs world-without sim. | 🔄 IN PROGRESS — unblocked by Phase 6 (S23) |
| **8 — Backtesting Harness** | Historical loader, simulation engine, baselines, golden regression detector. | **EMBARGO** — after Phase 7 |
| **9 — Explainability** | Decision traces. "Why this player over that one?" Human-readable explanations for every action. | **EMBARGO** — after Phase 8 |
| **10 — Integration & Automation** | Snapshot system, daily sim harness, configurable weights, risk modes, UI/API. | **EMBARGO** — last |

### Core Tenets (non-negotiable)
1. **Pure functions at top, side effects at bottom.** Deterministic, testable, stable.
2. **Context-aware scoring, not raw stats.** Time decay + Z-scores + position adjustment + confidence regression.
3. **Probabilistic thinking, not point estimates.** Monte Carlo → distributions → risk-aware decisions.
4. **Decision engines, not rankings.** Lineups, waivers, trades — all optimized, not sorted.
5. **Closed-loop validation.** Backtesting harness → metrics → golden baseline → regression detection.
6. **Explainability everywhere.** Every decision has a reason.

---

## ACTIVE DIRECTIVES (read before every session)

### DIRECTIVE 1 — Data-First Mandate (STRICT EMBARGO)

**HARD EMBARGO — do not lift without explicit human instruction:**
- Lineup optimization (Phase 7)
- Projection blending / ensemble update (job 100_014)
- FanGraphs RoS ingestion (job 100_012)
- Any new UI surface

**Nothing proceeds to the DB or UI until:** incoming payloads pass strict Pydantic V2 validation models. Every field, every type, every nullable must be explicitly declared and verified against live API responses.

---

## Platform State — April 6, 2026

| System | State | Notes |
|--------|-------|-------|
| CBB Season | **CLOSED** | Permanently archived. |
| MLB Data Pipeline | **P1-P16 CERTIFIED** | All contracts + BDL/Yahoo clients + jobs 100_001/100_013/100_016-100_021 wired + full schema live. Phase 6 Monte Carlo simulation verified in production (S23). P17 Decision Engines next. |
| Fantasy/Edge structural split | **PHASES 1-7 DONE** | Fantasy-App live — isolated DB, isolated scheduler. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | NCAAB + MLB. All verified clients returning validated objects. |
| `player_id_resolver.py` | **CLEAN (S20)** — Cache-first lookup → pybaseball fallback → persist. |
| `rolling_window_engine.py`| **CLEAN (S21)** — Exponential decay computations λ=0.95. |
| `scoring_engine.py` | **CLEAN (S22)** — League Z-scores, percentile scoring, confidence. |
| `momentum_engine.py` | **CLEAN (S22)** — Delta Z derivation (14d vs 30d), momentum signals. |
| `simulation_engine.py` | **CLEAN (S23)** — 1000-run ROS Monte Carlo, risk metrics. |
| `daily_ingestion._run_ros_simulation()` | **BUILT (S23)** — lock 100_021, daily 6 AM ET. Upserts to `simulation_results`. |

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

### P17 — Decision Engines [Phase 7]

**Claude Task:** Build `backend/services/decision_engine.py`:
- Lineup optimizer: greedy search vs ILP for daily roster slots.
- Waiver intelligence: world-with vs world-without simulation.
- Build `DecisionResult` ORM + migration `migrate_v20_decision_results.py`.
- Implement `_run_decision_optimization()` job (Lock 100_022, 7 AM ET).

### P18 — Backtesting Harness [Phase 8]
Historical loader, simulation engine, baselines.

---

## Session History (Recent)

### S23 — P16 Complete: Monte Carlo Probabilistic Layer (Apr 6)

**P16:** `simulation_engine.py` built. `simulation_results` schema deployed (`migrate_v19`) and verified in production. `_run_ros_simulation()` job registered (Lock 100_021). 1000-run Monte Carlo, percentile projections, risk metrics.

### S22 — P15 Complete: Momentum Signals (Apr 6)

**P15:** `momentum_engine.py` built. `player_momentum` schema deployed (`migrate_v18`) and verified in production. `_compute_player_momentum()` job registered (Lock 100_020).

---

### Gemini CLI — P16 Deploy: migrate_v19 (S23) ✅ COMPLETE

**Status:**
- `migrate_v19` deployed to Legacy and Fantasy DBs.
- `simulation_results` table verified live.
- `ros_simulation` job registered (Lock 100_021).
- Full suite verified: 1443 pass (only expected failures remaining).

### Gemini CLI — P15 Deploy: migrate_v18 (S22) ✅ COMPLETE

**Status:**
- `migrate_v18` deployed to Legacy and Fantasy DBs.
- `player_momentum` table verified live.
- `player_momentum` job registered (Lock 100_020).
