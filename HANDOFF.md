# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 7, 2026 (updated Session S25) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW-MODERATE — P1-P19 certified. Phases 2-9 complete. Next: P20 (Integration & Automation).

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
| **7 — Decision Engines** | Lineup optimizer, waiver optimizer, trade evaluator. World-with vs world-without sim. | ✅ DONE — Phase 7 complete (S23). `decision_results` verified live in production. |
| **8 — Backtesting Harness** | Historical loader, simulation engine, baselines, golden regression detector. | ✅ DONE — Phase 8 complete (S24). `backtest_results` verified live in production. |
| **9 — Explainability** | Decision traces. "Why this player over that one?" Human-readable explanations for every action. | ✅ DONE — Phase 9 complete (S25). `decision_explanations` verified live in production. |
| **10 — Integration & Automation** | Snapshot system, daily sim harness, configurable weights, risk modes, UI/API. | 🔄 IN PROGRESS — unblocked by Phase 9 (S25) |

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
- Projection blending / ensemble update (job 100_014)
- FanGraphs RoS ingestion (job 100_012)
- Any new UI surface

**Nothing proceeds to the DB or UI until:** incoming payloads pass strict Pydantic V2 validation models. Every field, every type, every nullable must be explicitly declared and verified against live API responses.

---

## Platform State — April 6, 2026

| System | State | Notes |
|--------|-------|-------|
| CBB Season | **CLOSED** | Permanently archived. |
| MLB Data Pipeline | **P1-P19 CERTIFIED** | All contracts + BDL/Yahoo clients + jobs 100_001/100_013/100_016-100_024 wired + full schema live. Phase 9 Explainability Layer verified in production (S25). P20 Integration next. |
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
| `decision_engine.py` | **CLEAN (S23)** — Greedy lineup optimization, waiver world-with/without simulation. |
| `backtesting_harness.py` | **CLEAN (S24)** — Historical accuracy metrics, regression detection. |
| `daily_ingestion._run_backtesting()` | **BUILT (S24)** — lock 100_023, daily 8 AM ET. Upserts to `backtest_results`. |
| `explainability_layer.py` | **CLEAN (S25)** — Z-score factor ranking, NL generation, lineup/waiver summaries, risk/track-record narratives. |
| `daily_ingestion._run_explainability()` | **BUILT (S25)** — lock 100_024, daily 9 AM ET. Upserts to `decision_explanations`. |
| `GET /admin/explanations/{decision_id}` | **LIVE (S25)** — Returns full explanation JSON for any decision; 404 if not yet computed. |

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

### P20 — Integration & Automation [Phase 10]

**Claude Task:** Wire the full 9-phase pipeline into a single daily snapshot system:
- Build `backend/services/snapshot_engine.py`: captures a complete daily state (all 8 pipeline outputs) into a single `DailySnapshot` record.
- Build `DailySnapshot` ORM + migration `migrate_v23_daily_snapshots.py`.
- Implement `GET /admin/snapshot/latest` and `GET /admin/snapshot/{date}` endpoints.
- Wire `_run_snapshot()` job (Lock 100_025, 10 AM ET) — runs after explainability (9 AM).
- Snapshot includes: n_players_scored, n_decisions, n_explanations, n_backtest_records, regression_detected flag, top_lineup_players (list of top-5 by lineup_score), top_waiver_adds (list of top-3 by value_gain).

### P21 — Next Steps (Post-P20)
UI integration (Next.js pages), configurable weights, risk mode controls.

---

## Session History (Recent)

### S25 — P19 Complete: Explainability Layer (Apr 7)

**P19:** `explainability_layer.py` built. `decision_explanations` schema deployed (`migrate_v22`) and verified in production. `_run_explainability()` job registered (Lock 100_024, 9 AM ET). Z-score factor ranking with ELITE/STRONG/AVERAGE/WEAK/POOR labels, NL summaries for lineup/waiver decisions, confidence/risk/track-record narratives. `GET /admin/explanations/{decision_id}` endpoint live. 33/33 tests pass.

### S24 — P18 Complete: Backtesting Harness (Apr 6)

**P18:** `backtesting_harness.py` built. `backtest_results` schema deployed (`migrate_v21`) and verified in production. `_run_backtesting()` job registered (Lock 100_023). Historical accuracy metrics (RMSE, MAE), regression detection against baseline.

### S23 — P17 Complete: Decision Engines (Apr 6)

**P17:** `decision_engine.py` built. `decision_results` schema deployed (`migrate_v20`) and verified in production. `_run_decision_optimization()` job registered (Lock 100_022). Greedy lineup optimization + waiver simulation.

---

### Gemini CLI — P19 Deploy: migrate_v22 (S25)

**Tasks:**
1. `railway run python scripts/migrate_v22_explanations.py --dry-run`
2. `railway run python scripts/migrate_v22_explanations.py`
3. Verify: `railway run python -c "from backend.models import DecisionExplanation; print('OK')"`
4. Smoke test: `POST /admin/ingestion/run/explainability`
5. Verify endpoint: `GET /admin/explanations/1` (expect 404 until first daily run completes)

### Gemini CLI — P18 Deploy: migrate_v21 (S24) ✅ COMPLETE

**Status:**
- `migrate_v21` deployed to Legacy and Fantasy DBs.
- `backtest_results` table verified live.
- `backtesting` job registered (Lock 100_023) and smoke tested via API.

### Gemini CLI — P17 Deploy: migrate_v20 (S23) ✅ COMPLETE

**Status:**
- `migrate_v20` deployed to Legacy and Fantasy DBs.
- `decision_results` table verified live.
- `decision_optimization` job registered (Lock 100_022).
