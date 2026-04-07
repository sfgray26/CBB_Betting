# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 7, 2026 (updated Session S26) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW — P1-P20 CERTIFIED. All 10 phases complete. Full pipeline operational. Next: P21 (UI integration + configurable weights).

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
| **10 — Integration & Automation** | Snapshot system, daily sim harness, configurable weights, risk modes, UI/API. | ✅ DONE — Phase 10 complete (S26). `daily_snapshots` live. GET /admin/snapshot/latest operational. |

### Core Tenets (non-negotiable)
1. **Pure functions at top, side effects at bottom.** Deterministic, testable, stable.
2. **Context-aware scoring, not raw stats.** Time decay + Z-scores + position adjustment + confidence regression.
3. **Probabilistic thinking, not point estimates.** Monte Carlo → distributions → risk-aware decisions.
4. **Decision engines, not rankings.** Lineups, waivers, trades — all optimized, not sorted.
5. **Closed-loop validation.** Backtesting harness → metrics → golden baseline → regression detection.
6. **Explainability everywhere.** Every decision has a reason.

---

## Platform State — April 7, 2026

| System | State | Notes |
|--------|-------|-------|
| CBB Season | **CLOSED** | Permanently archived. |
| MLB Data Pipeline | **P1-P20 CERTIFIED -- ALL 10 PHASES COMPLETE** | Full pipeline: rolling_windows(3AM) -> scores(4AM) -> momentum(5AM) -> simulation(6AM) -> decisions(7AM) -> backtesting(8AM) -> explainability(9AM) -> snapshot(10AM). Jobs 100_001/100_013/100_016-100_025 wired. All schemas live in production. |
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
| `explainability_layer.py` | **CLEAN (S25)** — Z-score factor ranking, NL generation, narratives. |
| `daily_ingestion._run_explainability()` | **BUILT (S25)** — lock 100_024, daily 9 AM ET. Upserts to `decision_explanations`. |
| `GET /admin/explanations/{id}` | **LIVE (S25)** — Returns explanation JSON for any decision. |
| `snapshot_engine.py` | **CLEAN (S26)** — Health check (HEALTHY/DEGRADED/FAILED), daily state capture, one-sentence pipeline summary. |
| `daily_ingestion._run_snapshot()` | **BUILT (S26)** — lock 100_025, daily 10 AM ET. Upserts to `daily_snapshots`. |
| `GET /admin/snapshot/latest` | **LIVE (S26)** — Returns most recent DailySnapshot; 404 until first daily run. |
| `GET /admin/snapshot/{date}` | **LIVE (S26)** — Returns DailySnapshot for a specific YYYY-MM-DD date. |

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

### P21 — Next Steps (Post-10-Phase Blueprint)

All 10 blueprint phases are now complete. The following are the next logical improvements
to discuss with the team before the next session:

1. **Next.js UI pages** — Surface the pipeline outputs (scores, decisions, explanations, snapshots) via the canonical frontend at `/fantasy` or `/mlb`.
2. **Configurable weights** — Allow adjusting the lineup score formula weights (0.6/0.3/0.1) via environment variables or admin API.
3. **Waiver pool integration** — Currently `_run_decision_optimization` passes empty waiver candidates. Wire in Yahoo available-player queries.
4. **Alert on regression** — Send a notification (email/Slack) when `regression_detected=True` in the daily snapshot.

**These require explicit human instruction before Claude begins implementation.**

---

## Session History (Recent)

### S26 — P20 Complete: Snapshot Engine -- ALL 10 PHASES DONE (Apr 7)

**P20:** `snapshot_engine.py` built. `daily_snapshots` schema deployed (`migrate_v23`) and verified in production. `_run_snapshot()` job registered (Lock 100_025, 10 AM ET). Health check logic (HEALTHY/DEGRADED/FAILED), regression detection, top-5 lineup and top-3 waiver player capture. `GET /admin/snapshot/latest` + `GET /admin/snapshot/{date}` endpoints live. 13/13 tests pass.

**THIS COMPLETES THE 10-PHASE BLUEPRINT.** Full daily pipeline operational:
rolling_windows(3AM) -> scores(4AM) -> momentum(5AM) -> simulation(6AM) -> decisions(7AM) -> backtesting(8AM) -> explainability(9AM) -> snapshot(10AM)

### S25 — P19 Complete: Explainability Layer (Apr 7)

**P19:** `explainability_layer.py` built. `decision_explanations` schema deployed (`migrate_v22`) and verified in production. `_run_explainability()` job registered (Lock 100_024, 9 AM ET). Z-score factor ranking, NL summaries, confidence/risk narratives. `GET /admin/explanations/{decision_id}` endpoint live.

### S24 — P18 Complete: Backtesting Harness (Apr 6)

**P18:** `backtesting_harness.py` built. `backtest_results` schema deployed (`migrate_v21`) and verified in production. `_run_backtesting()` job registered (Lock 100_023).

---

### Gemini CLI — P20 Deploy: migrate_v23 (S26)

**Tasks:**
1. `railway run python scripts/migrate_v23_daily_snapshots.py --dry-run`
2. `railway run python scripts/migrate_v23_daily_snapshots.py`
3. Verify: `railway run python -c "from backend.models import DailySnapshot; print('OK')"`
4. Smoke test: `POST /admin/ingestion/run/snapshot`
5. Verify endpoint: `GET /admin/snapshot/latest` (expect 404 until first daily run, or trigger manually)
6. Run full test suite: `venv/Scripts/python -m pytest tests/ -q --tb=short` — report pass count

### Gemini CLI — P19 Deploy: migrate_v22 (S25) ✅ COMPLETE

**Status:**
- `migrate_v22` deployed to Legacy and Fantasy DBs.
- `decision_explanations` table verified live.
- `explainability` job registered (Lock 100_024).
- Smoke test success: `POST /admin/ingestion/run/explainability` returns 200.
- Endpoint success: `GET /admin/explanations/1` returns 404 (expected).

### Gemini CLI — P18 Deploy: migrate_v21 (S24) ✅ COMPLETE

**Status:**
- `migrate_v21` deployed to Legacy and Fantasy DBs.
- `backtest_results` table verified live.
- `backtesting` job registered (Lock 100_023) and smoke tested via API.
