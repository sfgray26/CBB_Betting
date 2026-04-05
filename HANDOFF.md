# HANDOFF.md — MLB Platform Master Plan (In-Season 2026)

> **Date:** April 5, 2026 (updated Session S15) | **Author:** Claude Code (Master Architect)
> **Risk Level:** MODERATE — Layer 0 contracts certified; Priority 3 (validated BDL client) is next

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
                  ✅ COMPLETE (Postgres-ygnV provisioned; URL set).

Step 2 (Claude):  Run database migrations against FANTASY_DATABASE_URL.
                  Verify schema with: SELECT table_name FROM information_schema.tables
                  WHERE table_schema = 'public';

Step 3 (Gemini):  Deploy fantasy_app.py ONLY after Claude confirms Step 2.
                  Start Command: uvicorn backend.fantasy_app:app --host 0.0.0.0 --port $PORT
```

### DIRECTIVE 3 — Strangler-Fig Scheduler Duplication (Race Condition Fix)

Before booting new fantasy service:
1. Set `ENABLE_FANTASY_SCHEDULER=false` on legacy `main.py` service in Railway. ✅ COMPLETE
2. Verify legacy service redeployed and fantasy scheduler is not running. ✅ VERIFIED
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
| MLB Data Pipeline | **P1+P2+P3+P4 CERTIFIED** | Layer 0 contracts (MLB + Yahoo) + validated BDL client + Yahoo ingestion layer complete. 78/78 total tests pass (42 BDL + 36 Yahoo). P5 (job 100_013 re-enable) is next pending human approval. |
| `mlb_analysis.py` | **PROTOTYPE — DO NOT BUILD ON** | Raw OddsAPI calls, no validation, silent 0.0 returns, fuzzy name matching. Rebuild with validated contracts. |
| Fantasy projection pipeline | **EMBARGOED** | Jobs 100_012-100_015 disabled pending data floor certification |
| Fantasy/Edge structural split | **PHASES 1-5 DONE** | Phase 6-7 in infrastructure track |

### Ground Truth: What Actually Exists

| Component | Reality |
|-----------|---------|
| `balldontlie.py` | NCAAB + MLB. `get_mlb_games`, `get_mlb_odds`, `get_mlb_injuries`, `search_mlb_players`, `get_mlb_player` — all returning Pydantic-validated objects. |
| `mlb_analysis.py` | Calls OddsAPI via raw `requests.get()`. No retry. No validation. Returns `0.0` silently on errors. |
| `daily_ingestion._poll_mlb_odds()` | Raw OddsAPI. Validates only `isinstance(games, list)`. No DB persistence. |
| `daily_ingestion._poll_yahoo_adp_injury()` | Checks `if not pid` only. No schema validation. Writes directly to DB. |
| `odds.py` | 100% CBB. Zero MLB methods. |
| MLB Pydantic models | `backend/data_contracts/` — MLBGame, MLBBettingOdd, MLBInjury, MLBPlayer, MLBTeam, BDLResponse. 18/18 tests pass. |
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

#### Next action: Build Yahoo contracts, then wrap client methods

**Files to create:**
- `backend/data_contracts/yahoo_player.py` — `YahooPlayer` base (shared fields)
- `backend/data_contracts/yahoo_roster.py` — `YahooRosterEntry` (adds `selected_position`)
- `backend/data_contracts/yahoo_waiver.py` — `YahooWaiverCandidate` (adds `stats: dict`)

**Existing client methods STAY.** Add validation at their call sites or in a thin wrapper layer.

**Acceptance criteria for lifting job 100_013 embargo:**
- `get_adp_and_injury_feed()` parses 100% through `YahooPlayer` contract
- Zero `ValidationError` on any of the 100 ADP feed players
- `status: Optional[bool]` correct (not Optional[str])
- `injury_note` checked independently from `status`
- Written confirmation with pass/fail counts
- Human approval before re-enabling

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

### INFRA-C — Phase 6-7 Railway Deployment

Execute per Directive 2 sequence. Strict ordering.
- ✅ Step 1 COMPLETE: Provisioned `Postgres-ygnV`. Set `FANTASY_DATABASE_URL` on legacy service.

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
| `yahoo_adp_injury` | 100_013 | Every 4h | **EMBARGOED** -- P4 contracts certified; awaiting human approval to re-enable |
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

### Claude Code — Priority 4: Yahoo Pydantic Contracts + Validation Layer

Execute in `C:\Users\sfgra\repos\Fixed\cbb-edge`. Read `HANDOFF.md` and `CLAUDE.md` first.

**Context:** BDL MLB data layer is certified (42/42 tests). Yahoo live fixtures are captured in `tests/fixtures/yahoo_*.json`. Priority 4 builds Pydantic V2 contracts for the Yahoo response shapes and adds a validation layer on top of the existing client methods. The Yahoo client STAYS — do not rewrite it.

**Ground truth is in `reports/SCHEMA_DISCOVERY.md` Yahoo section.** Read that before writing any model.

**Critical facts from live capture (override K27 assumptions):**
- `status: Optional[bool]` — `True` = injured, `None` = healthy. NOT a string.
- `injury_note` is independent of `status` — must check BOTH for injury detection
- `positions` includes `"IL"` for IL players — third injury signal
- `percent_owned: float` — always present, never null
- `selected_position: str` — roster only, NOT in FA or ADP responses
- `stats: dict[str, str]` — FA only; stat 60 is `"H/AB"` format, NOT same as stat 8

**Step 1 — Build contracts in `backend/data_contracts/`:**

Create `backend/data_contracts/yahoo_player.py`:
```python
class YahooPlayer(BaseModel):
    model_config = ConfigDict(strict=True)
    player_key: str
    player_id: str
    name: str
    team: str
    positions: list[str]
    status: Optional[bool] = None      # True=injured, None=healthy. NOT a string.
    injury_note: Optional[str] = None  # Body part only. Independent of status.
    is_undroppable: bool
    percent_owned: float               # Always present, 0.0-100.0
```

Create `backend/data_contracts/yahoo_roster.py`:
```python
class YahooRosterEntry(YahooPlayer):
    selected_position: str             # Where slotted: "C", "SP", "BN", etc.
```

Create `backend/data_contracts/yahoo_waiver.py`:
```python
class YahooWaiverCandidate(YahooPlayer):
    stats: Optional[dict[str, str]] = None  # Stat ID -> value string. May be absent.
```

Update `backend/data_contracts/__init__.py` to export all three.

**Step 2 — Tests in `tests/test_yahoo_contracts.py`:**
- Parse all 24 roster items through `YahooRosterEntry` — 100% required
- Parse all 25 FA items through `YahooWaiverCandidate` — 100% required
- Parse all 100 ADP items through `YahooPlayer` — 100% required
- Test `status=True` parsed correctly as bool (not coerced from string)
- Test player with `status=None` + `injury_note="Hip"` parsed correctly

**Step 3 — Add validation at call sites:**
Wrap the three key client methods in `backend/services/yahoo_ingestion.py` (new file):
```python
def get_validated_roster(client: YahooFantasyClient) -> list[YahooRosterEntry]:
    raw = client.get_roster()
    return [YahooRosterEntry.model_validate(p) for p in raw]

def get_validated_free_agents(client: YahooFantasyClient) -> list[YahooWaiverCandidate]:
    raw = client.get_free_agents()
    return [YahooWaiverCandidate.model_validate(p) for p in raw]

def get_validated_adp_feed(client: YahooFantasyClient) -> list[YahooPlayer]:
    raw = client.get_adp_and_injury_feed()
    return [YahooPlayer.model_validate(p) for p in raw]
```

Log any `ValidationError` at ERROR level with field + value. Never suppress.

**Compile checks:**
```bash
venv/Scripts/python -m py_compile backend/data_contracts/yahoo_player.py
venv/Scripts/python -m py_compile backend/data_contracts/yahoo_roster.py
venv/Scripts/python -m py_compile backend/data_contracts/yahoo_waiver.py
venv/Scripts/python -m py_compile backend/services/yahoo_ingestion.py
venv/Scripts/python -m pytest tests/test_yahoo_contracts.py -v --tb=short
```

All tests pass → commit → report back with pass/fail counts.

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

---

## Session History (Recent)

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
