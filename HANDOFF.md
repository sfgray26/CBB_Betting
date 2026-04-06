# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 6, 2026 (updated Session S18) | **Author:** Claude Code (Master Architect)
> **Risk Level:** LOW-MODERATE — P1-P7 data floor certified. Fantasy-App live. Phase 2 structurally complete. Next: P8 (odds persistence), then Phase 3 (rolling windows).

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
| **2 — Data Foundation** | Ingest every game + stat + player. Normalize. Resolve IDs. Never compute from raw API. Build as standalone microservice: idempotent, raw+normalized dual-write, schema drift detection, anomaly detection. | 🔄 IN PROGRESS — Game logs + odds snapshots live. Missing: player box stats (/mlb/v1/stats) + identity resolution (Kimi K-A, K-B) |
| **3 — Derived Stats** | 30/14/7-day rolling windows. Exponential decay λ=0.95. Per-game aggregation. Hitter + pitcher parity. | ⏳ BLOCKED on Phase 2 DB ingestion |
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
                  ✅ COMPLETE — 26 tables confirmed (mlb_team, mlb_game_log, player_daily_metrics, etc.)

Step 3 (Gemini):  Deploy fantasy_app.py.
                  ✅ COMPLETE — Fantasy-App live at https://fantasy-app-production-5079.up.railway.app
                  Health: {"status":"healthy","database":"connected","scheduler":"running"}
```

### DIRECTIVE 3 — Strangler-Fig Scheduler Duplication (Race Condition Fix) ✅ COMPLETE

1. `ENABLE_FANTASY_SCHEDULER=false` on legacy `main.py` service. ✅ VERIFIED
2. Legacy service scheduler confirmed not running. ✅ VERIFIED
3. `fantasy_app.py` deployed — runs its own scheduler (pybaseball, statcast, openclaw, job_queue_processor).

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
| BDL GOAT MLB | **ACTIVE** | Purchased. Zero `/mlb/v1/` code exists yet — build from scratch. |
| OddsAPI Basic | **ACTIVE** | 20k calls/month. CBB archival only. MLB odds via BDL. |
| BDL NCAAB | **DEAD** | Subscription cancelled — never call `/ncaab/v1/` |
| MLB Data Pipeline | **P1-P8 CERTIFIED** | All contracts + BDL/Yahoo clients + jobs 100_001/100_013/100_016 wired + full schema live + odds snapshots persisting on 30-min windows. Phase 2 structurally complete. P9 next: BDL /mlb/v1/stats + player box stats ingestion (Phase 3 bridge). |
| `mlb_analysis.py` | **PROTOTYPE — DO NOT BUILD ON** | Raw OddsAPI calls, no validation, silent 0.0 returns, fuzzy name matching. Rebuild with validated contracts. |
| Fantasy projection pipeline | **EMBARGOED** | Jobs 100_012-100_015 disabled pending data floor certification |
| Fantasy/Edge structural split | **PHASES 1-7 DONE** | Fantasy-App live at https://fantasy-app-production-5079.up.railway.app — isolated DB, isolated scheduler. |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | NCAAB + MLB. `get_mlb_games`, `get_mlb_odds`, `get_mlb_injuries`, `search_mlb_players`, `get_mlb_player` — all returning Pydantic-validated objects. |
| `yahoo_ingestion.py` | `get_validated_roster`, `get_validated_free_agents`, `get_validated_adp_feed` — Layer 2 adapters, typed returns. |
| `daily_ingestion._poll_yahoo_adp_injury()` | **CLEAN (S17)** — wired through `get_validated_adp_feed()`. Typed attribute access. `is_injured` checks all 3 signals. LIVE. |
| `daily_ingestion._poll_mlb_odds()` | **CLEAN (S17)** — BDL client, typed `MLBGame`/`MLBBettingOdd` attributes, `asyncio.to_thread` for sync client. No DB persistence yet (P7 adds game table). |
| `mlb_analysis.py` | **PROTOTYPE — DO NOT BUILD ON.** Raw OddsAPI calls, no validation, silent 0.0 returns. |
| `odds.py` | 100% CBB. Zero MLB methods. |
| MLB Pydantic models | `backend/data_contracts/` — MLBGame, MLBBettingOdd, MLBInjury, MLBPlayer, MLBTeam, YahooPlayer, YahooRosterEntry, YahooWaiverCandidate, BDLResponse. All certified. |
| MLB game-log DB ingestion | **NOT YET BUILT.** BDL client returns validated `MLBGame` objects but nothing writes them to DB daily. Phase 2 blocker. |
| MLB API endpoints | None exposed. No `/api/mlb/*` routes. |

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

**Yahoo capture:** PENDING — requires live OAuth session. Separate Railway task.

### Priority 2 — Layer 0: Pydantic V2 Decision Contracts ✅ COMPLETE

**Completed Session S15 (Apr 5 2026)**

```
backend/data_contracts/__init__.py        -- re-exports all contracts
backend/data_contracts/mlb_team.py        -- MLBTeam (shared sub-model)
backend/data_contracts/mlb_player.py      -- MLBPlayer (shared sub-model, dob validator)
backend/data_contracts/mlb_game.py        -- MLBGame, MLBTeamGameData, MLBScoringPlay
backend/data_contracts/mlb_odds.py        -- MLBBettingOdd (spread/total as str + float properties)
backend/data_contracts/mlb_injury.py      -- MLBInjury (nullable detail/side)
backend/data_contracts/pagination.py      -- BDLMeta, BDLResponse[T] generic
```

**Tests: `tests/test_data_contracts.py` — 18/18 pass (100% parse rate confirmed)**

Key decisions locked in contracts:
- `spread_home_value`, `spread_away_value`, `total_value` typed as `str` (API sends strings)
- `attendance` as `Optional[int]` (0 for pre-game, None guard for edge cases)
- `dob` stored as `str` — `"D/M/YYYY"` format, not ISO 8601 — custom validator rejects non-strings
- `college`, `draft` nullable (>60% null in live sample)

**Yahoo contracts (yahoo_roster.py, yahoo_waiver.py):** PENDING Yahoo live capture.

### Priority 3 — Layer 2: Validated BDL MLB Client ✅ COMPLETE

**Completed Session S15 (Apr 5 2026)**

All four methods added to `BallDontLieClient` in `backend/services/balldontlie.py`:

| Method | Commit | Tests |
|--------|--------|-------|
| `get_mlb_games(date)` → `list[MLBGame]` | `ebaabd9` | 6/6 |
| `get_mlb_odds(game_id)` → `list[MLBBettingOdd]` | `539644e` | 6/6 |
| `get_mlb_injuries()` → `list[MLBInjury]` | `c1ccc34` | 6/6 |
| `search_mlb_players(query)` → `list[MLBPlayer]` | `ebf133f` | 3/3 |
| `get_mlb_player(player_id)` → `MLBPlayer \| None` | `ebf133f` | 3/3 |

`tests/test_balldontlie_mlb.py` — 24/24 pass.
All MLB data layer tests combined: 42/42.

**One open item:** `/players/{id}` response envelope unverified — both direct-object and `{"data": {...}}` shapes handled defensively. Confirm actual shape in next live API session.

**Future BDL MLB endpoints (post-embargo, ordered by value):**
1. `/mlb/v1/lineups` — starting lineup cards. Critical for fantasy DFS roster lock decisions.
2. `/mlb/v1/stats` — per-game box stats per player (hits, HR, ERA, IP).
3. `/mlb/v1/season_stats` — season aggregates + WAR.
4. `/mlb/v1/plate_appearances` — full Statcast-level per-PA data (may overlap pybaseball).

### Priority 4 — Layer 2: Validated Yahoo Ingestion ✅ COMPLETE

**Completed Session S16 (Apr 5 2026)**

Files created:
- `backend/data_contracts/yahoo_player.py` — YahooPlayer base (strict=True, is_injured property)
- `backend/data_contracts/yahoo_roster.py` — YahooRosterEntry (adds selected_position)
- `backend/data_contracts/yahoo_waiver.py` -- YahooWaiverCandidate (adds stats dict)
- `backend/data_contracts/__init__.py` -- updated to export all three Yahoo models
- `backend/services/yahoo_ingestion.py` -- Layer 2 adapter wrapping client methods
- `tests/test_yahoo_contracts.py` -- 36 tests, all fixture parse rates verified

Key decisions locked:
- `status: Optional[bool]` -- strict=True means string "IL" is rejected at validation boundary
- `is_injured` checks all three independent signals: status, injury_note, "IL" in positions
- `percent_owned: float` -- always present, null rejected by contract
- `selected_position` on YahooRosterEntry only -- not on base or waiver model
- `stats: Optional[dict[str,str]]` on YahooWaiverCandidate -- stat 60 stays as "H/AB" string
- "NA" in positions is accepted (Yahoo "Not Active" token) -- does NOT trigger is_injured

**Yahoo live capture COMPLETE (S15, Apr 5)** -- fixtures in `tests/fixtures/yahoo_*.json`.

#### Critical findings from live capture (override K27 assumptions):

| Finding | Impact on contracts |
|---------|---------------------|
| `status` is `Optional[bool]` (`True`=injured, `None`=healthy) — NOT a string | Any code doing `status == "IL"` is silently broken |
| `injury_note` is independent of `status` — Verlander has `status=None` + `injury_note="Hip"` | Must check BOTH fields for injury detection |
| `positions` includes `"IL"` for IL players — reliable injury signal | Third injury signal to check |
| `percent_owned: float` always present, always normalized | K27 `percent_rostered` concern is moot |
| `selected_position` only in roster response, not FA or ADP | Separate models needed for each endpoint |
| Stat ID 60 returns `"H/AB"` combined format e.g. `"8/20"` — NOT same as ID 8 (raw H) | Not a duplicate — different semantics |
| Stat IDs 28 vs 42: likely pitching K vs batting K | Build separate fields, not same mapping |

### Priority 5 — Wire job 100_013 through validated layer ✅ COMPLETE

**Completed Session S17 (Apr 5 2026)**

`daily_ingestion._poll_yahoo_adp_injury()` now routes through `get_validated_adp_feed(client)`. All `dict.get()` removed. `p.is_injured` replaces `if injury_note:` — catches Verlander-style cases (status=None + injury_note set + "IL" in positions). 1334/1338 full suite (4 pre-existing).

---

## INFRASTRUCTURE TRACK (parallel, non-blocking, never delays data pipeline)

### INFRA-A — Disable CBB Scheduler Jobs

Gate behind `CBB_SEASON_ACTIVE` env var (default `false`):
- ✅ COMPLETE: `CBB_SEASON_ACTIVE` set to `false` in Railway.
- Disable: `nightly_analysis`, `fetch_ratings`, `opener_attack`, `odds_monitor`
- Keep: `update_outcomes`, `capture_closing_lines`, `daily_snapshot` (archival)

### INFRA-B — ENABLE_FANTASY_SCHEDULER Guard

Add guard to `backend/main.py` lifespan. When `false`, disable all fantasy scheduler jobs (100_012-100_015, DailyIngestionOrchestrator, job_queue_processor).
- ✅ COMPLETE: `ENABLE_FANTASY_SCHEDULER` set to `false` in Railway.

### INFRA-C — Phase 6-7 Railway Deployment ✅ COMPLETE

All three steps complete. Fantasy-App live and isolated. 26-table schema confirmed on Postgres-ygnV.

### INFRA-D — Disable Projection Freshness Gate (100_015)

Job 100_015 is listed LIVE but gates on embargoed jobs — permanently tripped, producing false violations. Disable alongside embargoed jobs until data floor is certified.
- ✅ COMPLETE: `ENABLE_PROJECTION_FRESHNESS` set to `false` in Railway.

### INFRA-E — CBB Archive (housekeeping, low urgency)

Rename stale references (cbb-architect -> mlb-architect, etc.). Documentation only.

---

## Job Registry

| Job | Lock | Cadence | Status |
|-----|------|---------|--------|
| `statcast` | 100_002 | Daily 2 AM ET | LIVE |
| `rolling_z` | 100_003 | Daily 3 AM ET | LIVE |
| `waiver_scan` | 100_007 | Daily 6 AM ET | LIVE |
| `mlb_brief` | 100_008 | Daily 7 AM ET | LIVE |
| `valuation_cache` | 100_011 | On demand | LIVE |
| `mlb_odds` | 100_001 | Every 30 min | DIRTY — rebuild with BDL validated client |
| `fangraphs_ros` | 100_012 | Daily 3 AM ET | **EMBARGOED** |
| `yahoo_adp_injury` | 100_013 | Every 4h | **LIVE** -- wired through `get_validated_adp_feed()` (P5 complete, S17) |
| `ensemble_update` | 100_014 | Daily 5 AM ET | **EMBARGOED** |
| `projection_freshness_check` | 100_015 | Every 1h | **DISABLE** (gates on embargoed jobs = false violations) |

**Next available lock ID:** 100_017

**Advisory lock namespace:** 100_001-100_010 = edge/CBB. 100_011-100_015 = fantasy. Do not cross-assign.

---

## Hard Stops

| Rule | Reason |
|------|--------|
| Do NOT build Layer 2 (API clients) without Layer 0 contracts | Core philosophy — contracts before plumbing |
| Do NOT build more than one BDL endpoint at a time | Core philosophy — one feed at a time |
| Do NOT use `dict.get()` as validation in new data code | Core philosophy — no silent failures |
| Do NOT run embargoed jobs (100_012, 100_014) | Directive 1 — data floor not certified (100_013 is now LIVE) |
| Do NOT run lineup optimization or ensemble blending | Directive 1 — downstream embargo |
| Do NOT modify Kelly math in `betting_model.py` | CBB season closed — model archived permanently |
| Do NOT call BDL `/ncaab/v1/` endpoints | Subscription cancelled — will 401 |
| Do NOT use OddsAPI for MLB features | 20k/month budget — MLB odds via BDL only |
| Do NOT build on `mlb_analysis.py` as-is | Prototype with silent failures — rebuild with contracts |
| Do NOT touch `dashboard/` (Streamlit) | Retired — Next.js is canonical |
| Do NOT use `datetime.utcnow()` | Use `datetime.now(ZoneInfo("America/New_York"))` |
| Do NOT write test files outside `tests/` | Architecture locked |
| Do NOT import `betting_model` from fantasy modules | GUARDIAN FREEZE / ADR-004 |
| Do NOT deploy fantasy_app.py before DB migrations | Directive 2 |
| Do NOT boot new fantasy service before disabling legacy scheduler | Directive 3 |
| Do NOT write raw Redis keys without namespace prefix | Directive 4 |

---

## FORWARD ROADMAP — Ordered by Blueprint Phase

All work below is Phase 2 (Data Foundation) unless labelled otherwise. Do NOT begin a later item until the prior one is complete and tested.

### P6 — Fix `_poll_mlb_odds()`: OddsAPI → BDL validated client ✅ COMPLETE (S17)
`daily_ingestion._poll_mlb_odds()` now uses `BallDontLieClient.get_mlb_games(date)` + `.get_mlb_odds(game_id)`. Typed `MLBGame`/`MLBBettingOdd` attribute access. Sync client wrapped in `asyncio.to_thread`. No OddsAPI calls for MLB anywhere. Blueprint: Phase 2.

### P7a — Schema: mlb_team, mlb_game_log, mlb_odds_snapshot ✅ COMPLETE (S17)

ORM models in `backend/models.py` (MLBTeam, MLBGameLog, MLBOddsSnapshot). Migration script: `scripts/migrate_v13_mlb_game_log.py`. Dry-run verified. 1334/1338 full suite (no regressions).

**To deploy:** `railway run python scripts/migrate_v13_mlb_game_log.py` (Gemini task)

### P7b — Daily MLB game-log ingestion job ✅ COMPLETE (S17)

`DailyIngestionOrchestrator._ingest_mlb_game_log()` — lock 100_016, daily 1 AM ET.
- Fetches yesterday + today via `BallDontLieClient.get_mlb_games()`
- Upserts `mlb_team` (dim) before `mlb_game_log` (fact) — FK dependency enforced
- Scores written NULL for STATUS_SCHEDULED; actual values for IN_PROGRESS/FINAL
- UTC ISO 8601 → ET date conversion via `ZoneInfo("America/New_York")`
- Idempotent: `ON CONFLICT (game_id) DO UPDATE` — safe to replay
- Dual-write: `raw_payload=game.model_dump()` on every upsert
- Anomaly check: WARNING if 0 games returned for any date
- 14/14 orchestrator tests pass. 1334/1338 full suite (4 pre-existing).

### P8 — Wire _poll_mlb_odds persistence to mlb_odds_snapshot ✅ COMPLETE (S18)

`_poll_mlb_odds` now persists to DB on every 5-min poll. Each poll:
1. Upserts `mlb_team` + `mlb_game_log` (makes the job self-sufficient — no FK violation risk)
2. Fetches odds per game via `get_mlb_odds(game.id)`
3. Upserts each `MLBBettingOdd` into `mlb_odds_snapshot` on `(game_id, vendor, snapshot_window)`
4. `snapshot_window` = now_et() rounded to 30-min bucket — idempotent per window

`odd.model_dump()` → `raw_payload` on every row (dual-write). 92/92 data+orchestrator tests pass.

### P9 — BDL /mlb/v1/stats endpoint + player box stats ingestion [Phase 2 → Phase 3 bridge]
`get_mlb_games(date)` already returns validated `list[MLBGame]`. Nothing persists them to DB. This is the **keystone of the entire intelligence stack** — every phase above it (rolling windows, Z-scores, momentum, Monte Carlo, decisions) is noise without it.

**Build it as a production microservice:**
- **Dual-write:** store raw BDL payload (JSON blob) + normalized row — never lose the original
- **Idempotent:** upsert on `(game_id, date)` — safe to re-run on crash or replay
- **Validate before persist:** every `MLBGame` passes contract before any DB write
- **Schema drift detection:** log at ERROR if unexpected fields appear or required fields go null
- **Anomaly detection:** alert if 0 games returned for a date with scheduled games; alert if <50% of expected box stats present
- **Lock ID:** 100_016 (next available)

This job is the source of truth for rolling windows, time decay, Z-scores, momentum, Monte Carlo, lineup optimization, trade evaluation, and backtesting. If this is flaky, everything above it is noise.

### P8 — Player identity resolution [Phase 2]
Map `MLBPlayer.id` (BDL internal ID) → `mlbamId` (MLB Stats API standard) → internal `player_id`. Without this, stats from different sources (BDL, pybaseball/Statcast, Yahoo) cannot be joined. Build a resolution table and a lookup function. Blueprint: Phase 2 (Identity Resolution layer).

### P9 — Derived stats: rolling windows + decay [Phase 3]
30/14/7-day rolling windows per player. Exponential decay λ=0.95. Per-game aggregation. Hitters and pitchers. Blueprint: Phase 3. **Do not start until P7 (game-log DB) is complete.**

### P10 — Scoring engine: Z-scores + confidence [Phase 4]
League Z-scores + position Z-scores. Z_adj = 0.7·Z_league + 0.3·Z_position. Confidence regression. 0–100 output. Blueprint: Phase 4. **Do not start until P9 is complete.**

---

## HANDOFF PROMPTS

### Claude Code — Priority 6: Rebuild `_poll_mlb_odds()` Through BDL Validated Client

Execute in `C:\Users\sfgra\repos\Fixed\cbb-edge`. Read `HANDOFF.md` and `CLAUDE.md` first.

**Context:** P1-P5 certified. Job 100_013 (`yahoo_adp_injury`) is now wired through `get_validated_adp_feed()`. P6 targets the other dirty job: `_poll_mlb_odds()` in `daily_ingestion.py`, which currently calls OddsAPI directly via raw `requests.get()` — violating the MLB-odds-via-BDL rule and the no-silent-failures rule.

**What exists:**
- `backend/services/balldontlie.py` — `get_mlb_odds(game_id)` returns `list[MLBBettingOdd]` (Pydantic-validated). `get_mlb_games(date)` returns `list[MLBGame]`.
- `backend/data_contracts/mlb_odds.py` — `MLBBettingOdd` with `spread_home_value`/`spread_away_value`/`total_value` as `str` + float properties.
- `daily_ingestion._poll_mlb_odds()` — currently calls OddsAPI. Needs to be replaced.

**The change:** Wire `_poll_mlb_odds()` through the BDL client. Pattern mirrors what P5 did for Yahoo:
1. Call `bdl_client.get_mlb_games(date)` to get today's game IDs
2. For each game, call `bdl_client.get_mlb_odds(game_id)` → `list[MLBBettingOdd]`
3. Access fields via typed attributes (`odd.spread_home_value`, `odd.spread_away_float`, etc.) — no `dict.get()`
4. Write to DB

**Hard stops:**
- Do NOT call OddsAPI (`THE_ODDS_API_KEY`) for any MLB data
- Do NOT use `dict.get()` on API responses
- Lock ID for `mlb_odds` is `100_001` — already in `LOCK_IDS`

**Compile + test:**
```bash
venv/Scripts/python -m py_compile backend/services/daily_ingestion.py
venv/Scripts/python -m pytest tests/test_balldontlie_mlb.py tests/test_data_contracts.py -q --tb=short
venv/Scripts/python -m pytest tests/ -q --tb=short
```

All tests pass + no new failures → commit → update HANDOFF.md platform state to P6 CERTIFIED.

---

### Kimi CLI — Raw Ingestion Spec Audit (COMPLETE)

**Status:** `reports/K27_RAW_INGESTION_AUDIT.md` delivered.

**Summary:**
- **Yahoo:** 21 stat IDs hardcoded, many with duplicate mappings (28/42 both = K). `_parse_player()` does recursive 5-level search for ownership data — actual path unknown.
- **BDL:** ZERO MLB code exists. NCAAB patterns (auth, pagination, base URL) are reusable but `/mlb/v1/` endpoints never called.
- **OddsAPI:** `mlb_analysis.py` and `daily_ingestion.py` both call The Odds API directly via `requests.get()`. No validation — returns `{}` or `[]` on any failure.
- **Top 5 Silent Failures:**
  1. `get_players_stats_batch()` enriches with empty `stats={}` on any API error
  2. `_fetch_mlb_odds()` returns `{}` if API key missing — no error raised
  3. `statsapi.schedule()` throws → caught → returns `[]` (graceful but invisible)
  4. `_parse_player()` recursive flattening hides Yahoo's actual response structure
  5. `_poll_mlb_odds()` treats 0 games as "expected" — could be API key expired

**Recommended Pydantic Contracts (Layer 0):**
```python
class YahooPlayer(BaseModel):
    player_key: str
    player_id: str
    name: str
    team: str
    positions: list[str]
    status: Optional[str] = None
    injury_note: Optional[str] = None
    percent_owned: float = 0.0

class BDLMLBGame(BaseModel):  # VERIFY WITH CLAUDE'S CAPTURE
    id: int
    date: str
    home_team: str
    away_team: str
    status: str
    home_score: Optional[int] = None
    away_score: Optional[int] = None
```

**Delta to Resolve:**
| Assumption | Reality Check |
|------------|---------------|
| Stat ID 28/42 both = K (pitcher) | Need to distinguish batting K vs pitching K |
| Ownership at 4+ possible paths | Claude's capture will show actual path depth |
| `/mlb/v1/` mirrors `/ncaab/v1/` | ENDPOINTS MAY NOT EXIST — verify first |
| OddsAPI returns `away_team`/`home_team` strings | May be ID-based in MLB endpoint |

### Gemini CLI — P8 Verification + System Health Check (S18)

**Context:** P8 just wired `_poll_mlb_odds` to write to `mlb_odds_snapshot`. The `mlb_game_log` ingestion job runs nightly at 1 AM ET. Verify both are working in production.

**Task 1 — Verify mlb_game_log populated after overnight run:**

```bash
railway run python -c "
from backend.models import SessionLocal, MLBGameLog, MLBTeam
db = SessionLocal()
game_count = db.query(MLBGameLog).count()
team_count = db.query(MLBTeam).count()
from sqlalchemy import func
latest = db.query(func.max(MLBGameLog.updated_at)).scalar()
print(f'mlb_game_log rows: {game_count}')
print(f'mlb_team rows: {team_count}')
print(f'latest updated_at: {latest}')
db.close()
"
```

Expected: game_count > 0 (MLB season is active). If 0, check Railway logs for the 1 AM ET job run.

**Task 2 — Verify mlb_odds_snapshot is being populated (run after next 5-min poll):**

```bash
railway run python -c "
from backend.models import SessionLocal, MLBOddsSnapshot
from sqlalchemy import func
db = SessionLocal()
count = db.query(MLBOddsSnapshot).count()
latest = db.query(func.max(MLBOddsSnapshot.snapshot_window)).scalar()
vendors = db.query(MLBOddsSnapshot.vendor).distinct().all()
print(f'mlb_odds_snapshot rows: {count}')
print(f'latest snapshot_window: {latest}')
print(f'vendors seen: {[v[0] for v in vendors]}')
db.close()
"
```

Expected: rows > 0 with vendors like draftkings, fanduel, etc.

**Task 3 — Both service health checks:**

```bash
# Legacy edge service
curl -f https://<legacy-backend-url>/health

# Fantasy-App
curl -f https://fantasy-app-production-5079.up.railway.app/health
```

**Report back:** For each task, report actual row counts and any errors. If mlb_game_log is empty, paste the Railway log lines from the 1 AM ET job run.

---

### Kimi CLI — Phase 3 Research: BDL Stats Endpoint + Player Identity Resolution (S18)

**Context:** Phase 2 data foundation is complete (mlb_team, mlb_game_log, mlb_odds_snapshot all live and populating). Phase 3 requires rolling windows per player per game. Two research tasks are needed before Claude can build Phase 3.

**Working directory:** `C:\Users\sfgra\repos\Fixed\cbb-edge`

---

**Research Task K-A: BDL /mlb/v1/stats Endpoint Specification**

BDL GOAT MLB subscription is active (API key: BALLDONTLIE_API_KEY env var). We need to build a validated ingestion layer for per-player per-game box stats.

Read `backend/services/balldontlie.py` to understand the existing client patterns (auth, pagination, `_mlb_get` method, Pydantic contract pattern). Read `backend/data_contracts/mlb_game.py` to understand the existing MLBGame contract.

Then query the BDL `/mlb/v1/stats` endpoint (if you have API access) OR research what fields it exposes. We need:

1. **Full field list** — what stats are returned per record? (hits, HR, RBI, K, IP, ERA, etc.)
2. **Record granularity** — is it per-player per-game, or season aggregates?
3. **Response shape** — same `{"data": [...], "meta": {"next_cursor": ...}}` pagination as other BDL endpoints?
4. **Nullable fields** — which fields are null for pitchers vs hitters?
5. **Player identifier** — does it return BDL `player.id` or mlbam ID?
6. **Rate limits** — how many requests per minute/hour?

Produce a spec memo at `reports/K_A_BDL_STATS_SPEC.md` with:
- Field table (name, type, nullable, notes)
- Recommended Pydantic contract skeleton
- Any gotchas (string values, nested objects, etc.)

---

**Research Task K-B: Player Identity Resolution Design**

The system uses three player ID systems that must be joined for Phase 3:
- **BDL player ID** (`MLBPlayer.id`) — integer, BDL internal
- **mlbam ID** — integer, MLB Stats API standard (used by pybaseball/Statcast)
- **Yahoo player key** (`YahooPlayer.player_key`) — string like "mlb.p.8967"

Read:
- `backend/data_contracts/mlb_player.py` — BDL MLBPlayer contract
- `backend/data_contracts/yahoo_player.py` — Yahoo YahooPlayer contract
- `backend/fantasy_baseball/yahoo_client_resilient.py` — Yahoo client (grep for `player_id`, `mlbam`)

Produce a spec memo at `reports/K_B_IDENTITY_RESOLUTION_SPEC.md` with:
1. **Current state** — what IDs are available from each source right now
2. **Join strategy** — best approach to map BDL ID → mlbam ID → Yahoo key
3. **Data source recommendation** — can pybaseball's `playerid_lookup()` provide the mapping? What about the Yahoo `player_key` format — does it embed mlbam ID?
4. **Proposed schema** — what a `player_identity` mapping table should look like
5. **Bootstrap approach** — how to seed the mapping table on first run with minimal API calls

Do NOT write any production code. Research and spec only. Output goes to `reports/`.

---

### Gemini CLI — Directive 2 Step 2: Fantasy DB Migrations + Step 3 Deploy (S17)

**Context:** `Postgres-ygnV` is provisioned and `FANTASY_DATABASE_URL` is set on the legacy service. Claude has confirmed the schema is ready. Execute Steps 2 and 3 of Directive 2 in strict order.

**Step 2 — Run migrations against FANTASY_DATABASE_URL:**

The fantasy service needs the full schema. Run each migration script in order, substituting `FANTASY_DATABASE_URL` for `DATABASE_URL`:

```bash
# Dry-run first to verify SQL:
railway run --service <fantasy-service> \
  DATABASE_URL=$FANTASY_DATABASE_URL \
  python scripts/migrate_v13_mlb_game_log.py --dry-run

# Run in order (skip any that error with "already exists"):
railway run DATABASE_URL=$FANTASY_DATABASE_URL python scripts/migrate_v9_live_data.py
railway run DATABASE_URL=$FANTASY_DATABASE_URL python scripts/migrate_v10_user_preferences.py
railway run DATABASE_URL=$FANTASY_DATABASE_URL python scripts/migrate_v11_blend_columns.py
railway run DATABASE_URL=$FANTASY_DATABASE_URL python scripts/migrate_v11_job_queue.py
railway run DATABASE_URL=$FANTASY_DATABASE_URL python scripts/migrate_v12_valuation_cache.py
railway run DATABASE_URL=$FANTASY_DATABASE_URL python scripts/migrate_v13_mlb_game_log.py
```

Verify schema after each:
```bash
railway run DATABASE_URL=$FANTASY_DATABASE_URL python -c "
from sqlalchemy import create_engine, text, inspect
import os
engine = create_engine(os.environ['DATABASE_URL'])
tables = inspect(engine).get_table_names()
print('Tables:', sorted(tables))
"
```

Expected tables include: `player_daily_metrics`, `player_valuation_cache`, `projection_cache_entries`, `projection_snapshots`, `mlb_team`, `mlb_game_log`, `mlb_odds_snapshot`.

**Step 3 — Deploy fantasy_app.py (ONLY after Step 2 confirmed):**

Start command: `uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT`

Verify health after deploy: `railway run curl https://<fantasy-service-url>/health`

**Report back:** For each migration script, report [SUCCESS] or [ERROR: message]. For the health check, report the HTTP status. Do NOT start Step 3 until all Step 2 migrations confirm success.

---

### Gemini CLI — Railway Branch Verification (S16)

**Context:** `stable/cbb-prod` was merged into `main` on April 5, 2026 (commit `1ed8c8d`). This pushed 8 commits of S15-S16 data layer work (Pydantic contracts, validated BDL client, Yahoo ingestion). Railway should auto-deploy from `main`. This prompt verifies Railway is watching the right branch and that env vars survived.

**Working directory:** `C:\Users\sfgra\repos\Fixed\cbb-edge`

**What to verify (in Railway dashboard or via `railway` CLI):**

1. **Backend service branch:** Confirm the `CBB_Betting` (or equivalent backend) Railway service is set to deploy from `main`, not a pinned commit or old branch.

2. **Frontend service branch:** Confirm the Next.js frontend Railway service is also watching `main`.

3. **Env var survival:** After the redeploy triggered by the push, verify these env vars are still set on the backend service:
   - `ENABLE_FANTASY_SCHEDULER=false`
   - `CBB_SEASON_ACTIVE=false`
   - `ENABLE_PROJECTION_FRESHNESS=false`
   - `FANTASY_DATABASE_URL` (should point to `Postgres-ygnV`)

4. **Dockerfile CMD (read-only check):** The current Dockerfile CMD runs:
   ```
   python scripts/migrate_v9_live_data.py && python scripts/migrate_v10_user_preferences.py && python -m backend.models && uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```
   This is CORRECT for now. Do NOT change it — Phase 6-7 service split is a separate task.

5. **Smoke test:** After confirming the redeploy completed, hit the health endpoint:
   ```
   railway run curl https://<backend-url>/health
   ```
   Confirm it returns 200. Do NOT run any migration scripts or scheduler commands.

**Report back:** For each item above, report [CONFIRMED] or [ACTION NEEDED] with exact Railway dashboard values observed. Do not make changes beyond what is listed here — if anything needs changing, flag it and wait for Claude to review.

---

## Session History (Recent)

### S17 — P5+P6 Certified: Yahoo + MLB odds both wired through validated contracts (Apr 5)

**P5:** `_poll_yahoo_adp_injury` → `get_validated_adp_feed()`. All `dict.get()` gone. `p.is_injured` checks 3 signals. Job 100_013 LIVE.

**P6:** `_poll_mlb_odds` rebuilt. OddsAPI eliminated. Now: `BallDontLieClient.get_mlb_games(today)` → per-game `get_mlb_odds(game.id)`. Typed `MLBGame.away_team.abbreviation`, `MLBBettingOdd.spread_home_value`, `.vendor` etc. Sync client wrapped in `asyncio.to_thread`. No DB persistence yet (P7 adds game table).

**Blueprint added to HANDOFF:** 10-phase north star, Phase 2 production-grade spec (dual-write, idempotent, schema drift + anomaly detection).

1334/1338 full suite (4 pre-existing). 78/78 data layer tests.

### S15 — P1+P2 Certified: Live Capture + Layer 0 Contracts (Apr 5)

`daily_ingestion._poll_yahoo_adp_injury()` now calls `get_validated_adp_feed(client)` from `yahoo_ingestion.py` instead of raw `client.get_adp_and_injury_feed()`. All `dict.get()` calls replaced with typed `YahooPlayer` attribute access. Injury detection upgraded from `if injury_note:` to `p.is_injured` (catches all 3 signals: status, injury_note, "IL" in positions — catches Verlander-style cases). 1334/1338 full suite (4 pre-existing failures, none new). Job 100_013 is LIVE on its 4-hour cadence.

### S15 — P1+P2 Certified: Live Capture + Layer 0 Contracts (Apr 5)

`railway run python scripts/capture_api_payloads.py` succeeded: 19 games, 6 odds vendors, 25 injuries, 1 player. Auth confirmed: bare key (no Bearer). All four BDL endpoints are accessible.

`backend/data_contracts/` created with 6 model files + `__init__.py`:
- `MLBTeam`, `MLBPlayer`, `MLBGame`, `MLBTeamGameData`, `MLBScoringPlay`, `MLBBettingOdd`, `MLBInjury`
- `BDLMeta`, `BDLResponse[T]` generic pagination wrapper

Key contract decisions: spread/total values typed as `str` (API sends strings, not floats); `dob` stored as `str` (`"D/M/YYYY"` format); 5 nullable player fields verified from live sample.

`tests/test_data_contracts.py` — 18/18 pass. 100% parse rate on all captured fixtures.

`reports/SCHEMA_DISCOVERY.md` fully populated with field tables, nullability, and non-obvious contract notes.

**P3 ready:** `get_mlb_games()` is the next commit.

### S14 — Philosophy Alignment + HANDOFF Rewrite (Apr 5)

Three-persona audit (Quant Dev, Architect, Elite Player) of HANDOFF.md. Found priorities were backwards: BDL endpoint construction listed above data validation. Corrected to contracts-before-plumbing philosophy. Separated infrastructure track from data pipeline track. Stripped all downstream references (optimization, matchups, waivers) from visible roadmap.

### S13 — Frontend Cleanup (Apr 5)

Removed settings page (432 lines, 100% fantasy). Hidden bracket via `SHOW_BRACKET = false`. Running UI: betting analytics only. Commit: `cc1e7ce`

### S12 — Structural Decoupling + Fantasy UI Removal (Apr 4)

Phases 1-5 of fantasy/edge split. New entry points (`edge_app.py`, `fantasy_app.py`). Fantasy UI fully removed (17 files). Test suite: 1256 passed, 4 pre-existing failures.

### S11 — Ensemble Hardening (Apr 3)

Per-row savepoints in `_update_ensemble_blend`. Fatal bare-except in `process_pending_jobs`.

---

## Architecture Reference

### Entry Points

| Entry Point | Command | Status |
|-------------|---------|--------|
| `backend/main.py` | `uvicorn backend.main:app` | LIVE in Railway |
| `backend/edge_app.py` | `uvicorn backend.edge_app:app` | Built — not yet deployed |
| `backend/fantasy_app.py` | `uvicorn backend.fantasy_app:app` | Built — awaiting Phase 6-7 |

### Key File Map

| What | Where |
|------|-------|
| Data contracts (TO BUILD) | `backend/data_contracts/` |
| BDL client (NCAAB only, MLB TBD) | `backend/services/balldontlie.py` |
| MLB analysis (PROTOTYPE - rebuild) | `backend/services/mlb_analysis.py` |
| Fantasy routes | `backend/routers/fantasy.py` |
| Edge/betting routes | `backend/routers/edge.py` |
| Admin/health routes | `backend/routers/admin.py` |
| Shared DB engine factory | `backend/db.py` |
| Redis namespace helpers | `backend/redis_client.py` |
| Kelly math (FROZEN) | `backend/core/kelly.py`, `backend/betting_model.py` |
| Yahoo OAuth client | `backend/fantasy_baseball/yahoo_client_resilient.py` |
| Ingestion scheduler | `backend/services/daily_ingestion.py` |
