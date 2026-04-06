# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 6, 2026 (updated Session S22) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW-MODERATE — P1-P14 certified. Phases 2-4 complete. Next: Gemini deploy migrate_v17, then P15 (momentum signals ΔZ).

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
| **4 — Scoring Engine** | League Z-scores + position Z-scores. Z_adj = 0.7·Z_league + 0.3·Z_position. Confidence regression. 0–100 output. | ✅ DONE — `scoring_engine.py` built, `player_scores` schema ready (S22). Pending: Gemini deploy migrate_v17. Position Z deferred (no position data yet). |
| **5 — Momentum Layer** | ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Cold / Collapsing / Breakout / Collapse. | ⏳ BLOCKED on migrate_v17 deploy |
| **6 — Probabilistic Layer** | 1000-run ROS Monte Carlo. Percentiles (P10/25/50/75/90). Risk metrics. P(top-10/25/50). | ⏳ BLOCKED on Phase 5 |
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
| MLB Data Pipeline | **P1-P14 CERTIFIED** | All contracts + BDL/Yahoo clients + jobs 100_001/100_013/100_016-100_019 wired + full schema live. Phase 4 scoring engine built (S22). Pending: Gemini deploy migrate_v17. |
| Fantasy/Edge structural split | **PHASES 1-7 DONE** | Fantasy-App live — isolated DB, isolated scheduler. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | NCAAB + MLB. All verified clients returning validated objects. |
| `player_id_resolver.py` | **CLEAN (S20)** — Cache-first lookup → pybaseball fallback → persist. |
| `rolling_window_engine.py`| **CLEAN (S21)** — Exponential decay computations λ=0.95. |
| `daily_ingestion._compute_rolling_windows()` | **BUILT (S21)** — lock 100_018, daily 3 AM ET. Upserts to `player_rolling_stats`. |

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

### P14 — Scoring engine: Z-scores + confidence ✅ COMPLETE (S22)

`backend/services/scoring_engine.py` — pure computation: `HITTER_CATEGORIES`, `PITCHER_CATEGORIES`, `PlayerScoreResult`, `compute_league_zscores()`.
- Hitter categories: HR, RBI, SB, AVG, OBP (5 categories)
- Pitcher categories: ERA (inverted), WHIP (inverted), K/9 (3 categories)
- League Z-scores only (position Z deferred — no position data in schema yet)
- Z capped ±3.0. MIN_SAMPLE=5. Confidence = games/window_days capped at 1.0
- score_0_100 = percentile rank within player_type
- stdlib math only (no numpy/scipy)
`backend/models.py` — `PlayerScore` ORM. Natural key `(bdl_player_id, as_of_date, window_days)`.
`scripts/migrate_v17_player_scores.py` — dry-run verified.
`backend/services/daily_ingestion.py` — `_compute_player_scores()` lock 100_019, 4 AM ET.
Tests: `tests/test_scoring_engine.py` — 21/21 pass (incl. Z-cap fixture fix: need n≥12 for Z>3.0).
Full suite: 1399/1403 pass.

**Pending: Gemini deploy migrate_v17 to both services.**

### P15 — Momentum signals [Phase 5]
Compare 14d vs 30d Z-scores. ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Stable / Cold / Collapsing. Blueprint: Phase 5. **Do not start until migrate_v17 is deployed.**

---

## Session History (Recent)

### S22 — P14 Complete: Z-score Scoring Engine (Apr 6)

**P14:** `scoring_engine.py` — `compute_league_zscores()`. 5 hitter + 3 pitcher categories. League Z-scores, Z-cap ±3.0, MIN_SAMPLE=5, percentile score_0_100, confidence. stdlib only.
**Schema:** `player_scores` ORM + `migrate_v17`. Natural key `(bdl_player_id, as_of_date, window_days)`.
**Job:** `_compute_player_scores()` lock 100_019, 4 AM ET.
**Test fix:** Z-cap tests require n≥12 players (population std: max Z = √(n-1), need n≥11 for Z>3.0).
**Suite:** 1399/1403 pass (22 new tests, 4 pre-existing failures unchanged).
**Next:** Gemini deploy migrate_v17 → P15 momentum signals.

---

### S21 — P13 Complete: Rolling Windows (Apr 6)

**P13:** `rolling_window_engine.py` built. `player_rolling_stats` schema deployed (`migrate_v16`) and verified in production. `_compute_rolling_windows()` job registered (Lock 100_018). Correctness enforced for `parse_ip("6.2")=6.667`.

### S20 — P11 + P12 Complete (Apr 6)

**P11:** `MLBPlayerStats` Pydantic contract + `get_mlb_stats()` BDL client + `mlb_player_stats` ORM + `migrate_v15` + `_ingest_mlb_box_stats()` job (lock 100_017, 2 AM ET).
**P12:** `PlayerIDResolver` service — cache + pybaseball fallback + persist.

---

### Gemini CLI — P14 Deploy: migrate_v17 (S22) ← NEXT GEMINI TASK

**Context:** Session S22 built `player_scores` schema (P14). Deploy to both services.

```bash
railway run python scripts/migrate_v17_player_scores.py
railway run --service fantasy-app python scripts/migrate_v17_player_scores.py
```

Verify: `railway run python -c "from backend.models import SessionLocal; from sqlalchemy import text; db=SessionLocal(); print(db.execute(text(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='player_scores'\")).scalar()); db.close()"`

Expected: `1`. Report back.

---

### Gemini CLI — P13 Deploy: migrate_v16 (S21) ✅ COMPLETE

**Status:**
- `migrate_v16` deployed to Legacy and Fantasy DBs.
- `player_rolling_stats` table verified live.
- `rolling_windows` job registered (Lock 100_018).

### Gemini CLI — P11 Deploy: migrate_v15 (S20) ✅ COMPLETE

**Status:**
- `migrate_v15` deployed to Legacy and Fantasy DBs.
- `mlb_player_stats` table verified live.
- `mlb_box_stats` job registered (Lock 100_017).
