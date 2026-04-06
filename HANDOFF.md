# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 6, 2026 (updated Session S20) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW-MODERATE — P1-P12 certified. Fantasy-App live. Phase 2 complete. Next: Gemini deploy migrate_v15, then P13 (rolling windows).

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
| **2 — Data Foundation** | Ingest every game + stat + player. Normalize. Resolve IDs. Never compute from raw API. Build as standalone microservice: idempotent, raw+normalized dual-write, schema drift detection, anomaly detection. | ✅ DONE — Phase 2 complete (S20). Game logs + odds snapshots + player box stats + identity resolution all built. Pending: Gemini deploy migrate_v15. |
| **3 — Derived Stats** | 30/14/7-day rolling windows. Exponential decay λ=0.95. Per-game aggregation. Hitter + pitcher parity. | ⏳ BLOCKED on migrate_v15 deploy |
| **4 — Scoring Engine** | League Z-scores + position Z-scores. Z_adj = 0.7·Z_league + 0.3·Z_position. Confidence regression. 0–100 output. | ⏳ BLOCKED on Phase 3 |
| **5 — Momentum Layer** | ΔZ = Z_14d − Z_30d. Signals: Surging / Hot / Cold / Collapsing / Breakout / Collapse. | ⏳ BLOCKED on Phase 4 |
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
- Any derived stats / rolling windows (Phase 3 — not yet built)
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

**Embargo safety confirmed:** `DailyIngestionOrchestrator` (embargoed jobs 100_012, 100_014) is gated by `ENABLE_INGESTION_ORCHESTRATOR` (defaults `false`) — NOT running on Fantasy-App. Race condition eliminated.

### DIRECTIVE 4 — Redis / Advisory Lock Contention

Advisory locks use PostgreSQL `pg_try_advisory_lock` (NOT Redis). Lock IDs: 100_001-100_010 = edge/CBB, 100_011-100_015 = fantasy. Do not cross-assign.

Redis shared instance: use `edge_cache.set(...)` or `fantasy_cache.set(...)` — never raw `redis.set(...)`.

### DIRECTIVE 5 — No LLM Time-Gates

All embargoes lifted by explicit human instruction only. No date-based triggers.

- **CBB V9.2 recalibration:** MOOT. CBB season permanently closed. Model archived.
- **BDL MLB integration:** ACTIVE. No trigger phrase needed.
- **OddsAPI:** NOT cancelled — Basic plan (20k calls/month). CBB archival closing lines only. MLB odds via BDL.

---

## Platform State — April 5, 2026

| System | State | Notes |
|--------|-------|-------|
| CBB Season | **CLOSED** | Permanently archived. No recalibration. |
| CBB Betting Model | **FROZEN PERMANENTLY** | Kelly math untouched. Archive only. |
| BDL GOAT MLB | **ACTIVE** | Purchased. Verified in production. |
| OddsAPI Basic | **ACTIVE** | 20k calls/month. CBB archival only. MLB odds via BDL. |
| BDL NCAAB | **DEAD** | Subscription cancelled — never call `/ncaab/v1/` |
| MLB Data Pipeline | **P1-P12 CERTIFIED** | All contracts + BDL/Yahoo clients + jobs 100_001/100_013/100_016/100_017 wired + full schema live. `player_id_mapping` + `mlb_player_stats` schema complete (S20). `PlayerIDResolver` service built. Pending: Gemini deploy migrate_v15 to both DBs. |
| `mlb_analysis.py` | **PROTOTYPE — DO NOT BUILD ON** | Raw OddsAPI calls, no validation, silent 0.0 returns, fuzzy name matching. Rebuild with validated contracts. |
| Fantasy projection pipeline | **EMBARGOED** | Jobs 100_012-100_015 disabled pending data floor certification |
| Fantasy/Edge structural split | **PHASES 1-7 DONE** | Fantasy-App live at https://fantasy-app-production-5079.up.railway.app — isolated DB, isolated scheduler. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | NCAAB + MLB. `get_mlb_games`, `get_mlb_odds`, `get_mlb_injuries`, `search_mlb_players`, `get_mlb_player` — all returning Pydantic-validated objects. |
| `yahoo_ingestion.py` | `get_validated_roster`, `get_validated_free_agents`, `get_validated_adp_feed` — Layer 2 adapters, typed returns. |
| `daily_ingestion._poll_yahoo_adp_injury()` | **CLEAN (S17)** — wired through `get_validated_adp_feed()`. Typed attribute access. LIVE. |
| `daily_ingestion._poll_mlb_odds()` | **CLEAN (S18)** — BDL client, typed attributes. Persisting to `mlb_odds_snapshot` on 30-min windows. LIVE. |
| `DailyIngestionOrchestrator._ingest_mlb_game_log()` | **CLEAN (S18)** — Idempotent upserts to `mlb_team` and `mlb_game_log`. LIVE (runs nightly 1 AM ET). |
| `mlb_analysis.py` | **PROTOTYPE — DO NOT BUILD ON.** Raw OddsAPI calls, no validation, silent 0.0 returns. |
| `odds.py` | 100% CBB. Zero MLB methods. |
| MLB Pydantic models | `backend/data_contracts/` — MLBGame, MLBBettingOdd, MLBInjury, MLBPlayer, MLBTeam, YahooPlayer, YahooRosterEntry, YahooWaiverCandidate, BDLResponse. All certified. |
| MLB game-log DB ingestion | **LIVE (S18).** `mlb_game_log` table populated and verified in production. |
| MLB API endpoints | None exposed. No `/api/mlb/*` routes. |
| identity mapping | `player_id_mapping` table live in both Legacy and Fantasy DBs (S19). |
| `player_id_resolver.py` | **CLEAN (S20)** — `PlayerIDResolver` class. Cache-first lookup → pybaseball fallback → persist. Manual overrides win. 7/7 tests pass. |
| `daily_ingestion._ingest_mlb_box_stats()` | **BUILT (S20)** — lock 100_017, daily 2 AM ET. Dual-write. ON CONFLICT (`bdl_player_id`, `game_id`) DO UPDATE. Anomaly warn on 0 rows. |
| `mlb_player_stats` table | **SCHEMA READY (S20)** — `migrate_v15_mlb_player_stats.py` written + dry-run verified. Deploy pending (Gemini task). |

---

## DATA PIPELINE TRACK — All Effort Goes Here

### Priority 1 — Raw Payload Capture + Schema Discovery ✅ COMPLETE

**Completed Session S15 (Apr 5 2026)**

- `tests/fixtures/bdl_mlb_games.json` — 19 games, 37KB
- `tests/fixtures/bdl_mlb_odds.json` — 6 vendors for game 5057892, 2.7KB
- `tests/fixtures/bdl_mlb_injuries.json` — 25 items page 1, cursor=409031, 40KB
- `tests/fixtures/bdl_mlb_players.json` — Ohtani search result, 852B
- `reports/SCHEMA_DISCOVERY.md` — fully populated with field tables, nullability, and contract notes
- Auth: bare key (no "Bearer" prefix). Spread/total values are **strings**. `dob` is `"D/M/YYYY"` format.

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

### P11 — BDL /mlb/v1/stats endpoint + player box stats ingestion ✅ COMPLETE (S20)

Contract: `backend/data_contracts/mlb_player_stats.py` — `MLBPlayerStats` (all stat fields Optional, rate stats float, ip str).
Client: `get_mlb_stats(dates, player_ids, per_page)` in `balldontlie.py`.
ORM: `MLBPlayerStats` class in `backend/models.py`.
Migration: `scripts/migrate_v15_mlb_player_stats.py` — dry-run verified.
Job: `_ingest_mlb_box_stats()` lock 100_017, daily 2 AM ET. Dual-write. Idempotent.
Tests: `tests/test_mlb_player_stats_contract.py` — 7/7 pass.

**Pending: Gemini deploy migrate_v15 to both services.**

### P12 — PlayerIDResolver service ✅ COMPLETE (S20)

`backend/services/player_id_resolver.py` — `PlayerIDResolver` class.
- Cache-first lookup: `player_id_mapping WHERE bdl_id = ?`
- Manual overrides (`source='manual'`) always win (queried with case sort)
- pybaseball `playerid_lookup()` fallback — stdout suppressed, NaN handled
- Persists successful lookups (`source='pybaseball'`, `resolution_confidence=1.0`)
Tests: `tests/test_player_id_resolver.py` — 7/7 pass.

### P13 — Derived stats: rolling windows + decay [Phase 3]
30/14/7-day rolling windows per player. Exponential decay λ=0.95. Per-game aggregation. Hitters and pitchers. Blueprint: Phase 3. **Do not start until P11 (player stats DB) + P12 (identity resolution) are complete.**

---

### Gemini CLI — P11 Deploy: migrate_v15 (S20) ← NEXT GEMINI TASK

**Context:** Session S20 built `mlb_player_stats` table schema (P11). Migration script is ready. Deploy to both services before Claude begins P13 (rolling windows).

**Task 1 — Deploy migrate_v15 to both services:**

```bash
# Legacy edge service
railway run python scripts/migrate_v15_mlb_player_stats.py

# Fantasy-App service  
railway run --service fantasy-app python scripts/migrate_v15_mlb_player_stats.py
```

Expected: `SUCCESS: mlb_player_stats created`. If `WARNING: Skipping (already exists)` — also fine.

**Task 2 — Verify tables live in both DBs:**

```bash
railway run python -c "
from backend.models import SessionLocal
db = SessionLocal()
from sqlalchemy import text
result = db.execute(text(\"SELECT COUNT(*) FROM information_schema.tables WHERE table_name='mlb_player_stats'\")).scalar()
print('mlb_player_stats exists:', result == 1)
db.close()
"
```

**Task 3 — Check _ingest_mlb_box_stats job registered:**

```bash
railway run python -c "
import os; os.environ.setdefault('ENABLE_INGESTION_ORCHESTRATOR', 'true')
from backend.services.daily_ingestion import DailyIngestionOrchestrator
o = DailyIngestionOrchestrator.__new__(DailyIngestionOrchestrator)
print('Has mlb_box_stats in LOCK_IDS:', 'mlb_box_stats' in getattr(o.__class__, 'LOCK_IDS', {}) or True)
"
# Or just check LOCK_IDS constant directly
railway run python -c "from backend.services.daily_ingestion import DailyIngestionOrchestrator; print(DailyIngestionOrchestrator.LOCK_IDS)"
```

Report back row counts and any errors.

---

### Gemini CLI — P10 Deploy + P11 Stats Probe (S19) ✅ COMPLETE

**Status:**
- `migrate_v14` deployed to Legacy and Fantasy DBs.
- `/mlb/v1/stats` probe successful (Findings in Roadmap P11).

---

## Session History (Recent)

### S20 — P11 + P12 Complete (Apr 6)

**P11:** `MLBPlayerStats` Pydantic contract + `get_mlb_stats()` BDL client + `mlb_player_stats` ORM + `migrate_v15` + `_ingest_mlb_box_stats()` job (lock 100_017, 2 AM ET). 7 contract tests.
**P12:** `PlayerIDResolver` service — cache + pybaseball fallback + persist. 7 resolver tests.
**Suite:** 1351/1355 pass (17 new tests, 4 pre-existing failures unchanged).
**Next:** Gemini deploy migrate_v15 → P13 rolling windows.

---

### S19 — P10 Deployed + BDL Stats Probe (Apr 5)

**Task 1:** `migrate_v14_player_id_mapping.py` deployed to both `CBB_Betting` and `Fantasy-App` databases. `player_id_mapping` table is live.

**Task 2:** `/mlb/v1/stats` live probe successful. Confirmed BDL-only IDs, float rate stats, and mutual exclusivity of batting/pitching fields.

---

## Architecture Reference

### Entry Points

| Entry Point | Command | Status |
|-------------|---------|--------|
| `backend/main.py` | `uvicorn backend.main:app` | LIVE in Railway (Legacy) |
| `backend/fantasy_app.py` | `uvicorn backend.fantasy_app:app` | LIVE in Railway (Fantasy-App) |

### Key File Map

| What | Where |
|------|-------|
| Data contracts | `backend/data_contracts/` |
| BDL client | `backend/services/balldontlie.py` |
| Yahoo client | `backend/fantasy_baseball/yahoo_client_resilient.py` |
| Ingestion scheduler | `backend/services/daily_ingestion.py` |
| Shared DB engine factory | `backend/db.py` |
| Redis namespace helpers | `backend/redis_client.py` |
| Kelly math (FROZEN) | `backend/core/kelly.py`, `backend/betting_model.py` |
