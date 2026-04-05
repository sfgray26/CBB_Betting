# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 5, 2026 (updated Session S14) | **Author:** Claude Code (Master Architect)
> **Risk Level:** ELEVATED — Data pipeline under construction; raw ingestion unvalidated

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

## ACTIVE DIRECTIVES (read before every session)

### DIRECTIVE 1 — Data-First Mandate (STRICT EMBARGO)

**HARD EMBARGO — do not lift without explicit human instruction:**
- Lineup optimization
- Projection blending
- Ensemble update (job 100_014)
- FanGraphs RoS ingestion (job 100_012)
- Yahoo ADP/injury polling (job 100_013)
- Any derived stats (wOBA, FIP, barrel%, etc.)
- Any new UI surface

**Nothing proceeds to the DB or UI until:** incoming payloads pass strict Pydantic V2 validation models. Every field, every type, every nullable must be explicitly declared and verified against live API responses.

### DIRECTIVE 2 — Phase 6-7 Deployment Sequence (Infrastructure Track)

```
Step 1 (Gemini):  Provision Fantasy Postgres. Set FANTASY_DATABASE_URL.
                  DO NOT start the service yet.

Step 2 (Claude):  Run database migrations against FANTASY_DATABASE_URL.
                  Verify schema with: SELECT table_name FROM information_schema.tables
                  WHERE table_schema = 'public';

Step 3 (Gemini):  Deploy fantasy_app.py ONLY after Claude confirms Step 2.
                  Start Command: uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT
```

### DIRECTIVE 3 — Strangler-Fig Scheduler Duplication (Race Condition Fix)

Before booting new fantasy service:
1. Set `ENABLE_FANTASY_SCHEDULER=false` on legacy `main.py` service in Railway.
2. Verify legacy service redeployed and fantasy scheduler is not running.
3. Only then start fantasy_app.py.

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
| MLB Data Pipeline | **UNDER CONSTRUCTION** | No Pydantic contracts. No validated clients. Ground-up build. |
| `mlb_analysis.py` | **PROTOTYPE — DO NOT BUILD ON** | Raw OddsAPI calls, no validation, silent 0.0 returns, fuzzy name matching. Rebuild with validated contracts. |
| Fantasy projection pipeline | **EMBARGOED** | Jobs 100_012-100_015 disabled pending data floor certification |
| Fantasy/Edge structural split | **PHASES 1-5 DONE** | Phase 6-7 in infrastructure track |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | 100% NCAAB. Zero MLB code. Auth + pagination patterns reusable. |
| `mlb_analysis.py` | Calls OddsAPI via raw `requests.get()`. No retry. No validation. Returns `0.0` silently on errors. |
| `daily_ingestion._poll_mlb_odds()` | Raw OddsAPI. Validates only `isinstance(games, list)`. No DB persistence. |
| `daily_ingestion._poll_yahoo_adp_injury()` | Checks `if not pid` only. No schema validation. Writes directly to DB. |
| `odds.py` | 100% CBB. Zero MLB methods. |
| MLB Pydantic models | None exist. |
| MLB API endpoints | None exposed. No `/api/mlb/*` routes. |

---

## DATA PIPELINE TRACK — All Effort Goes Here

### Priority 1 — Raw Payload Capture + Schema Discovery

**READ-ONLY reconnaissance. No code changes. No DB writes.**

Before writing any wrapper code or validation model, capture what the APIs actually return. The delta between what the code ASSUMES and what the API ACTUALLY sends is where every bug lives.

**BDL MLB API (`/mlb/v1/`):**

BDL GOAT MLB is purchased. `balldontlie.py` has the auth pattern (Bearer token via `BALLDONTLIE_API_KEY`, base URL `https://api.balldontlie.io`, cursor pagination). But zero `/mlb/v1/` calls have ever been made.

Tasks:
1. Hit `GET /mlb/v1/games?dates[]={today}` with existing BDL API key. Capture full raw JSON. Document every field, type, nullable.
2. Hit `GET /mlb/v1/odds?game_id={id}` for a real game. Capture. Document market types, sportsbook names, line format.
3. Hit `GET /mlb/v1/injuries` if endpoint exists. Capture. Document.
4. Hit `GET /mlb/v1/players?search={name}`. Capture. Document.
5. Save captured payloads to `tests/fixtures/bdl_mlb_*.json` (test fixtures, not production code).

**Yahoo Fantasy API:**

Yahoo client exists and works. But field-level response shapes have never been documented.

Tasks:
1. Call `get_team_roster()` via Railway. Capture raw response. Document every field path.
2. Call `get_free_agents()` via Railway. Capture. Document.
3. Save to `tests/fixtures/yahoo_*.json`.

**Kimi K27 audit runs in parallel** — reads the codebase to map what code ASSUMES. Claude captures what APIs ACTUALLY return. The delta is the bug map.

**Acceptance gate:** Every data source has a captured real payload and a documented field map before moving to Priority 2.

### Priority 2 — Layer 0: Pydantic V2 Decision Contracts

Define canonical domain models. These are the IMMUTABLE CONTRACTS. No raw dicts cross layer boundaries.

**Location: `backend/data_contracts/`**

```
backend/data_contracts/__init__.py
backend/data_contracts/mlb_game.py        -- MLBGame, MLBTeam
backend/data_contracts/mlb_odds.py        -- MLBOddsLine, MLBMarket, MLBSportsbook
backend/data_contracts/mlb_injury.py      -- MLBInjuryReport, MLBInjuredPlayer
backend/data_contracts/yahoo_roster.py    -- YahooRosterEntry, YahooPlayerStats
backend/data_contracts/yahoo_waiver.py    -- YahooWaiverCandidate
backend/data_contracts/validation.py      -- ValidationReport (shared)
```

**Requirements:**
- Pydantic V2 `BaseModel` with `model_config = ConfigDict(strict=True)`
- Every field typed. `Optional[T]` with explicit `None` default for nullables.
- `model_validator` / `field_validator` where business rules apply (e.g., `percent_rostered` must be 0-100).
- Tests: feed captured payloads from Priority 1 through each model. **100% parse rate required.**

**Acceptance gate:** Every captured payload parses with zero `ValidationError` before moving to Priority 3.

### Priority 3 — Layer 2: Validated BDL MLB Client (one endpoint at a time)

Build API client methods in `backend/services/balldontlie.py`. Each method returns Pydantic-validated domain objects. NEVER raw dicts.

**Strict sequence — do not skip ahead:**

**3a.** `get_mlb_games(date: str) -> list[MLBGame]`
- Calls `/mlb/v1/games?dates[]={date}`
- Parses through `MLBGame` contract
- Handles pagination (reuse existing NCAAB cursor pattern)
- Empty list on API error (logged, never silent)
- Test with captured fixture. Prove against live API. **Commit. Only then proceed.**

**3b.** `get_mlb_odds(game_id: int) -> list[MLBOddsLine]`
- Test. Prove. **Commit. Proceed.**

**3c.** `get_mlb_injuries() -> list[MLBInjuredPlayer]`
- Test. Prove. **Commit. Proceed.**

**3d.** `get_mlb_box_score(game_id: int) -> MLBBoxScore` (if BDL exposes)
- Test. Prove. **Commit.**

**Each step is its own commit.** If 3b reveals the BDL odds schema differs from our contract, we fix the contract (Priority 2) FIRST — not after.

### Priority 4 — Layer 2: Validated Yahoo Ingestion

Existing Yahoo client methods STAY (hardened across 11 sessions). Add a validation LAYER on top.

1. Wrap `get_team_roster()` return in `YahooRosterEntry` contract parsing.
2. Wrap `get_free_agents()` return in `YahooWaiverCandidate` contract parsing.
3. Any field that fails validation: logged with exact field name, expected type, actual value. Never suppressed.
4. Test against captured Yahoo fixtures.

**Acceptance criteria for lifting job 100_013 embargo:**
- Live `get_adp_and_injury_feed()` parses 100% through contracts
- Zero `ValidationError` on any player record
- Any `null` in `percent_rostered` explicitly handled (logged + default to 0.0, documented in contract)
- Written confirmation with pass/fail counts
- Human approval before re-enabling

---

## INFRASTRUCTURE TRACK (parallel, non-blocking, never delays data pipeline)

These are operationally important but architecturally irrelevant to data quality. Execute at any time without blocking the data pipeline.

### INFRA-A — Disable CBB Scheduler Jobs

Gate behind `CBB_SEASON_ACTIVE` env var (default `false`):
- Disable: `nightly_analysis`, `fetch_ratings`, `opener_attack`, `odds_monitor`
- Keep: `update_outcomes`, `capture_closing_lines`, `daily_snapshot` (archival)

### INFRA-B — ENABLE_FANTASY_SCHEDULER Guard

Add guard to `backend/main.py` lifespan. When `false`, disable all fantasy scheduler jobs (100_012-100_015, DailyIngestionOrchestrator, job_queue_processor).

### INFRA-C — Phase 6-7 Railway Deployment

Execute per Directive 2 sequence. Strict ordering.

### INFRA-D — Disable Projection Freshness Gate (100_015)

Job 100_015 is listed LIVE but gates on embargoed jobs — permanently tripped, producing false violations. Disable alongside embargoed jobs until data floor is certified.

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
| `yahoo_adp_injury` | 100_013 | Every 4h | **EMBARGOED** |
| `ensemble_update` | 100_014 | Daily 5 AM ET | **EMBARGOED** |
| `projection_freshness_check` | 100_015 | Every 1h | **DISABLE** (gates on embargoed jobs = false violations) |

**Next available lock ID:** 100_016

**Advisory lock namespace:** 100_001-100_010 = edge/CBB. 100_011-100_015 = fantasy. Do not cross-assign.

---

## Hard Stops

| Rule | Reason |
|------|--------|
| Do NOT build Layer 2 (API clients) without Layer 0 contracts | Core philosophy — contracts before plumbing |
| Do NOT build more than one BDL endpoint at a time | Core philosophy — one feed at a time |
| Do NOT use `dict.get()` as validation in new data code | Core philosophy — no silent failures |
| Do NOT run embargoed jobs (100_012, 100_013, 100_014) | Directive 1 — data floor not certified |
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

## HANDOFF PROMPTS

### Claude Code — Priority 1: Raw Payload Capture

Execute in `C:\Users\sfgra\repos\Fixed\cbb-edge`. Read `HANDOFF.md` and `CLAUDE.md` first.

**Context:** We are building validated data contracts before API client plumbing. Step 1 is capturing real API responses to document exactly what each source returns. This is READ-ONLY reconnaissance. No wrapper code. No DB writes.

**Task 1 — BDL MLB Payload Capture:**

`balldontlie.py` has the auth pattern but zero MLB code. Use the existing `BALLDONTLIE_API_KEY` and base URL `https://api.balldontlie.io`.

```python
# Run locally or via railway run:
import requests, json, os

API_KEY = os.environ["BALLDONTLIE_API_KEY"]
headers = {"Authorization": f"Bearer {API_KEY}"}
base = "https://api.balldontlie.io"

# 1. Games for today
r = requests.get(f"{base}/mlb/v1/games", headers=headers, params={"dates[]": "2026-04-05"})
with open("tests/fixtures/bdl_mlb_games.json", "w") as f:
    json.dump(r.json(), f, indent=2)

# 2. Odds for a specific game (use a game_id from step 1)
game_id = <first_game_id_from_above>
r = requests.get(f"{base}/mlb/v1/odds", headers=headers, params={"game_id": game_id})
with open("tests/fixtures/bdl_mlb_odds.json", "w") as f:
    json.dump(r.json(), f, indent=2)

# 3. Injuries (try — may not exist)
r = requests.get(f"{base}/mlb/v1/injuries", headers=headers)
with open("tests/fixtures/bdl_mlb_injuries.json", "w") as f:
    json.dump(r.json(), f, indent=2)

# 4. Player search
r = requests.get(f"{base}/mlb/v1/players", headers=headers, params={"search": "Ohtani"})
with open("tests/fixtures/bdl_mlb_players.json", "w") as f:
    json.dump(r.json(), f, indent=2)
```

For EACH response, document in `reports/SCHEMA_DISCOVERY.md`:
- Every field name and its type
- Which fields are nullable
- Pagination structure (`meta`, `next_cursor`, etc.)
- Any unexpected fields or values

**Task 2 — Yahoo Payload Capture:**

```python
# Via railway run (requires Yahoo OAuth):
from backend.fantasy_baseball.yahoo_client_resilient import get_resilient_yahoo_client
import json

client = get_resilient_yahoo_client()

roster = client.get_team_roster()
with open("tests/fixtures/yahoo_roster.json", "w") as f:
    json.dump(roster, f, indent=2, default=str)

free_agents = client.get_free_agents()
with open("tests/fixtures/yahoo_free_agents.json", "w") as f:
    json.dump(free_agents, f, indent=2, default=str)
```

Document in `reports/SCHEMA_DISCOVERY.md`:
- Every field path accessed by existing code vs. what actually comes back
- Fields accessed without null checks (the delta = bugs waiting to happen)

**Task 3 — Output:**

Commit fixture files: `git add tests/fixtures/bdl_mlb_*.json tests/fixtures/yahoo_*.json reports/SCHEMA_DISCOVERY.md`

Report: field maps for each API. Do NOT write any wrapper code or Pydantic models yet. This is recon only.

---

### Kimi CLI — Raw Ingestion Spec Audit (parallel with Priority 1)

Read-only. No code changes. Output to `reports/K27_RAW_INGESTION_AUDIT.md`.

**Context:** Claude is capturing live API payloads. Your job is the other half: audit what the CODEBASE currently assumes these APIs return. The delta between Claude's capture and your audit is the bug map.

**Research Task 1 — Yahoo API Assumptions:**

Read `backend/fantasy_baseball/yahoo_client_resilient.py` in full. For each method (get_team_roster, get_free_agents, get_matchup_stats, get_league_settings, get_adp_and_injury_feed):
- What fields does the code access?
- Which accesses have no null check?
- Which stat IDs are hardcoded?

**Research Task 2 — BDL Client Patterns:**

Read `backend/services/balldontlie.py`. Document:
- Auth pattern, base URL, pagination structure
- Which NCAAB patterns are reusable for MLB
- Current MLB support: explicitly state "ZERO" if none

**Research Task 3 — `mlb_analysis.py` OddsAPI Calls:**

Read `backend/services/mlb_analysis.py`. Document:
- Every direct OddsAPI call (these will be rebuilt, not migrated)
- What fields `_fetch_mlb_odds()` expects
- All silent failure paths (returns 0.0, empty dict, etc.)

**Research Task 4 — Silent Failure Audit:**

Read `backend/services/daily_ingestion.py`. Focus on `_poll_mlb_odds`, `_poll_yahoo_adp_injury`, `_fetch_fangraphs_ros`, `_update_ensemble_blend`. Every field access without type validation = silent failure point.

**Report:**
- Per-method table: endpoint, fields accessed, null-safe (y/n), type-checked (y/n)
- Top-5 silent failure risks
- Recommended field list for Pydantic V2 contracts

---

## Session History (Recent)

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
