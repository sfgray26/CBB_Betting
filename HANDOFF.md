# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 6, 2026 (updated Session S23) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW-MODERATE — P1-P15 certified. Phases 2-5 complete. Next: Gemini deploy migrate_v18, then P16 (Monte Carlo probabilistic layer).

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
| **5 — Momentum Layer** | ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Cold / Collapsing / Breakout / Collapse. | ✅ DONE — `momentum_engine.py` built, `player_momentum` schema ready (S23). Pending: Gemini deploy migrate_v18. |
| **6 — Probabilistic Layer** | 1000-run ROS Monte Carlo. Percentiles (P10/25/50/75/90). Risk metrics. P(top-10/25/50). | ⏳ BLOCKED on migrate_v18 deploy |
| **7 — Decision Engines** | Lineup optimizer, waiver optimizer, trade evaluator. World-with vs world-without sim. | **HARD EMBARGO** — do not touch |
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
- Any derived stats / rolling windows (Phase 3 — complete)
- Any scoring / scores DB persistence (Phase 4 — complete)
- Any new UI surface
- Monte Carlo / probabilistic layer (Phase 6)

**Nothing proceeds to the DB or UI until:** incoming payloads pass strict Pydantic V2 validation models. Every field, every type, every nullable must be explicitly declared and verified against live API responses.

### DIRECTIVE 2 — Phase 6-7 Deployment Sequence (Infrastructure Track) ✅ COMPLETE

```
Step 1 (Gemini):  Provision Fantasy Postgres. Set FANTASY_DATABASE_URL.
                  ✅ COMPLETE (Postgres-ygnV provisioned; URL set).

Step 2 (Gemini):  Run migrations v8-v13 against FANTASY_DATABASE_URL.
                  ✅ COMPLETE — 26 tables verified live.

Step 3 (Gemini):  Deploy fantasy_app.py.
                  ✅ COMPLETE — Fantasy-App live at https://fantasy-app-production-5079.up.railway.app
                  Health: {"status":"healthy","database":"connected","scheduler":"running"}
```

### DIRECTIVE 3 — Strangler-Fig Scheduler Duplication (Race Condition Fix) ✅ COMPLETE

1. `ENABLE_FANTASY_SCHEDULER=false` on legacy `main.py` service. ✅ VERIFIED
2. Legacy service scheduler confirmed not running. ✅ VERIFIED
3. `fantasy_app.py` deployed — runs its own scheduler (pybaseball, statcast, openclaw, job_queue_processor). ✅ VERIFIED

---

## Platform State — April 6, 2026

| System | State | Notes |
|--------|-------|-------|
| CBB Season | **CLOSED** | Permanently archived. |
| MLB Data Pipeline | **P1-P15 CERTIFIED (pending deploy)** | All contracts + BDL/Yahoo clients + jobs 100_001/100_013/100_016-100_020 wired + full schema live. Phase 5 momentum engine built (S23). Gemini to run migrate_v18 in production. |
| Fantasy/Edge structural split | **PHASES 1-7 DONE** | Fantasy-App live — isolated DB, isolated scheduler. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | NCAAB + MLB. All verified clients returning validated objects. |
| `player_id_resolver.py` | **CLEAN (S20)** — Cache-first lookup → pybaseball fallback → persist. |
| `rolling_window_engine.py`| **CLEAN (S21)** — Exponential decay computations λ=0.95. |
| `scoring_engine.py` | **CLEAN (S22)** — League Z-scores, percentile scoring, confidence. |
| `daily_ingestion._compute_player_scores()` | **BUILT (S22)** — lock 100_019, daily 4 AM ET. Upserts to `player_scores`. |
| `momentum_engine.py` | **BUILT (S23)** — pure computation, zero I/O. classify_signal, compute_player_momentum, compute_all_momentum. |
| `daily_ingestion._compute_player_momentum()` | **BUILT (S23)** — lock 100_020, daily 5 AM ET. Upserts to `player_momentum`. |

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

### P15 — Momentum signals [Phase 5] -- COMPLETE (S23)

All deliverables built and tested. Awaiting Gemini to run migrate_v18 in production.

### P16 — Probabilistic Layer [Phase 6]
1000-run ROS Monte Carlo. Percentiles. Risk metrics.

---

## Session History (Recent)

### S23 — P15 Complete: Momentum Layer (Apr 6)

**P15:** `momentum_engine.py` built (pure, zero I/O). `PlayerMomentum` ORM added to `models.py`. `migrate_v18_player_momentum.py` written. `_compute_player_momentum()` job registered (Lock 100_020, 5 AM ET). 23 tests in `test_momentum_engine.py` covering all boundary semantics. `test_ingestion_orchestrator.py` updated with `player_momentum` job. Gemini to deploy migrate_v18 to both Railway DBs.

### S22 — P14 Complete: Z-score Scoring Engine (Apr 6)

**P14:** `scoring_engine.py` built. `player_scores` schema deployed (`migrate_v17`) and verified in production. `_compute_player_scores()` job registered (Lock 100_019). League Z-scores, Z-cap ±3.0, percentile scores.

### S21 — P13 Complete: Rolling Windows (Apr 6)

**P13:** `rolling_window_engine.py` built. `player_rolling_stats` schema deployed (`migrate_v16`) and verified in production. `_compute_rolling_windows()` job registered (Lock 100_018).

---

### Gemini CLI — P15 Deploy: migrate_v18 (S23) -- READY FOR DEPLOY

**Task:** Deploy `player_momentum` table to both Railway databases.

**Commands (run exactly):**
```bash
# Legacy DB
railway run --service cbb-edge python scripts/migrate_v18_player_momentum.py

# Fantasy DB
railway run --service fantasy-app python scripts/migrate_v18_player_momentum.py

# Verify table exists on Legacy DB
railway run --service cbb-edge python -c "
from backend.models import engine
from sqlalchemy import text
with engine.connect() as c:
    r = c.execute(text(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='player_momentum'\"))
    print('player_momentum exists:', r.scalar() == 1)
"
```

**Compile-check before deploy:**
```bash
venv/Scripts/python -m py_compile scripts/migrate_v18_player_momentum.py
venv/Scripts/python -m py_compile backend/models.py
venv/Scripts/python -m py_compile backend/services/daily_ingestion.py
venv/Scripts/python -m py_compile backend/services/momentum_engine.py
```

**Report back:** Confirm table created + row count (expect 0 rows initially).

---

### Gemini CLI — P14 Deploy: migrate_v17 (S22) ✅ COMPLETE

**Status:**
- `migrate_v17` deployed to Legacy and Fantasy DBs.
- `player_scores` table verified live.
- `player_scores` job registered (Lock 100_019).

### Gemini CLI — P13 Deploy: migrate_v16 (S21) ✅ COMPLETE

**Status:**
- `migrate_v16` deployed to Legacy and Fantasy DBs.
- `player_rolling_stats` table verified live.
- `rolling_windows` job registered (Lock 100_018).
