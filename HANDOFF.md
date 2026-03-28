# OPERATIONAL HANDOFF — MARCH 28, 2026: STRUCTURAL REFACTOR + PHASE B1/B2 COMPLETE

> **Ground truth as of March 28, 2026.** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior state: `EMAC-086` + Kimi build-fix pass (March 27).
>
> **CRITICAL CONTEXT:** Structural refactor complete. Yahoo client consolidated.
> Dead code deleted. Streamlit retired. Dashboard B1 stubs wired. Weather integration confirmed complete.
> Test suite: 1198 pass / 5 fail (all pre-existing).

---

## 1. Mission Accomplished (This Session)

| Task | Status | Notes |
|------|--------|-------|
| **Yahoo client consolidation** | ✅ DONE | `yahoo_client.py` deleted — inlined into `yahoo_client_resilient.py` |
| **openclaw_briefs.py deleted** | ✅ DONE | All callers re-pointed to `openclaw_briefs_improved.py` |
| **Tests moved** | ✅ DONE | `test_lineup_validator.py` + `test_waiver_recovery.py` → `tests/` |
| **Streamlit retired** | ✅ DONE | Never touch `dashboard/` again. Next.js is the only UI. |
| **State updates ingested** | ✅ DONE | v9/v10 migrations live, coroutine patched, Kimi on UI/API mapping |
| **Dashboard B1 stubs wired** | ✅ DONE | `_get_waiver_targets`, `_get_matchup_preview`, `_get_probable_pitchers` wired to live services |
| **Weather provider decision** | ✅ LOCKED | OpenWeatherMap — already fully implemented in `weather_fetcher.py` + `smart_lineup_selector.py` |

---

## 2. Technical State Table

| Component | Status | Notes |
|-----------|--------|-------|
| **DB Migrations v9/v10** | ✅ LIVE | Chained into Dockerfile CMD by prior session |
| **Coroutine leak** (`openclaw_autonomous.py:33`) | ✅ FIXED | `asyncio.run()` applied |
| **yahoo_client.py** | ✅ DELETED | `YahooFantasyClient` now lives in `yahoo_client_resilient.py` |
| **openclaw_briefs.py** | ✅ DELETED | Superseded by `openclaw_briefs_improved.py` |
| **Streamlit dashboard** | ✅ RETIRED | `dashboard/` is dead code — do not touch |
| **Build pipeline** | ✅ GREEN | Python syntax clean, TypeScript builds |
| **Test suite** | ✅ 1198/1203 | 3 pre-existing DB-auth failures + 2 pre-existing logic failures |
| **Dashboard B1 stubs** | ✅ WIRED | `dashboard_service.py` — waiver/matchup/pitcher panels now live |
| **Weather integration (B2)** | ✅ COMPLETE | `smart_lineup_selector.py` + `weather_fetcher.py` + `park_weather.py` — fully wired |
| **UI/API Mapping (Kimi)** | ✅ DONE | Roster dedup, matchup parsing, NaN fixes in `yahoo_client_resilient.py` — see `reports/yahoo-client-hotfix-march28.md` |
| **Weather API key** | ✅ KEY SET | `OPENWEATHER_API_KEY` is present in Railway (March 28) |
| **MCMC Simulator** | ❌ SCAFFOLDED | `mcmc_simulator.py` exists but not calibrated |
| **OR-Tools (Railway)** | 🔄 PENDING DEPLOY | Added to requirements.txt — next Railway deploy will install. Greedy fallback active in interim. |

---

## 3. Pre-existing Test Failures (Do Not Fix Without Analysis)

| Test | Failure | Root Cause |
|------|---------|------------|
| `test_betting_model.py::TestExposureAccounting` (3x) | `psycopg2.OperationalError` | Local DB not running — Railway-only |
| `test_tournament_data.py::test_cache_expired` | Cache returns data when test expects `{}` | TTL logic mismatch |
| `test_waiver_recovery.py::TestCircuitBreaker::test_opens_after_threshold` | `ValueError` propagates instead of being caught | CircuitBreaker swallows wrong exception type — newly visible after moving file to `tests/` |

---

## 4. Delegation Bundles

### CLAUDE CODE (Master Architect) — Next Session

> **Priority 1: OR-Tools — add to requirements.txt (low urgency)**
>
> Gemini confirmed OR-Tools NOT installed in Railway (March 28).
> `lineup_constraint_solver.py` already has a greedy fallback (`ORTOOLS_AVAILABLE = False` path) —
> so the optimizer still works, just non-optimally.
> Action: add `ortools` to `requirements.txt` to re-enable true ILP optimization.
> Risk: LOW. Greedy fallback is correct, just slightly suboptimal for multi-position players.
>
> **Priority 2: Dashboard SSE refresh**
>
> Decision locked: SSE via FastAPI `EventSourceResponse` (over WebSocket — simpler, Railway-compatible).
> Wire: `/api/fantasy/dashboard/stream` endpoint → push panel updates every 60s.
> Panels to stream: waiver_targets, matchup_preview, probable_pitchers, streaks.
>
> **Priority 3: Matchup preview enrichment**
>
> `_get_matchup_preview` returns `win_prob=0.5`, `opponent_record=""`, `my_projected_categories={}`.
> Add opponent record from `client.get_standings()`.
> Add category projections from `PlayerDailyMetric.z_score_recent` + `rolling_window` JSONB.
> Do NOT wire MCMC yet — that is B5.
>
> **Priority 4: CircuitBreaker test fix (or accept-as-is)**
>
> `test_waiver_recovery.py::test_opens_after_threshold` — see Architect Review Queue item 1.
> Either fix the CircuitBreaker to raise `CircuitOpenError` after threshold,
> or update the test to `pytest.raises(ValueError)`.

### KIMI CLI (Deep Intelligence Unit) — Currently Executing

> Complete `yahoo_client_resilient.py` nested dict parsing hotfix.
> Fix roster deduplication bug (Roster page showing duplicates).
> Fix Matchup page "Team not found" error — likely a `get_my_team_key()` parse failure.
> Fix NaN projections — likely a `_parse_player` float cast failure.
>
> When complete: update HANDOFF.md with findings. Do NOT write to production code
> outside `yahoo_client_resilient.py` without Claude review.

### GEMINI CLI (Ops) — COMPLETE (March 28)

> Both ops checks done. Findings recorded in Technical State Table.
> Next task: await OR-Tools decision from Claude before any Railway changes.

---

## 5. Architecture Decisions (Locked)

| Decision | Ruling | Reason |
|----------|--------|--------|
| Yahoo client split-brain | ELIMINATED | Single file: `yahoo_client_resilient.py` |
| Streamlit | RETIRED | Next.js only — never touch `dashboard/` |
| openclaw_briefs (old) | DELETED | `_improved` is canonical |
| Dashboard refresh | SSE (pending impl) | Over WebSocket — simpler, Railway-compatible |
| Weather provider | OpenWeatherMap (LOCKED) | Already fully implemented — env var: `OPENWEATHER_API_KEY` |
| Test location | `tests/` only | No test files in `backend/` subdirs |

---

## 6. Architect Review Queue

1. **`test_waiver_recovery.py::test_opens_after_threshold`** — CircuitBreaker in `backend/fantasy_baseball/circuit_breaker.py` propagates the original exception instead of raising `CircuitOpenError` after threshold. Decide: fix the CircuitBreaker, fix the test, or accept as-is.
2. **MCMC Simulator calibration** — `backend/fantasy_baseball/mcmc_simulator.py` exists but is unvalidated. Before wiring to lineup optimizer, needs a calibration pass against historical matchup outcomes. Schedule for mid-term roadmap.
3. **V9.1 CBB recalibration (EMAC-068)** — Still blocked until post-Apr 7 per prior session. Do not touch Kelly math until then.

---

## HANDOFF PROMPTS

### For Kimi CLI

```
You are Kimi CLI (Deep Intelligence Unit). Read this HANDOFF.md in full.

Your current task:
1. Complete the nested dict parsing hotfix in `backend/fantasy_baseball/yahoo_client_resilient.py`
   - Fix: Roster page showing duplicate players
   - Fix: Matchup page "Team not found" error
   - Fix: NaN projections (likely in `_parse_player` float cast)

2. After fixing, run: `python -m py_compile backend/fantasy_baseball/yahoo_client_resilient.py`
   and confirm: PASS

3. Save a structured fix summary to `reports/yahoo-client-hotfix-march28.md` with:
   - Each bug, root cause, fix applied, line numbers
   - Any NEW bugs discovered

4. Update HANDOFF.md section 2 (Technical State Table) to mark UI/API Mapping as DONE.

IMPORTANT: Do NOT touch any other files. Claude will review your changes before merging.
Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```

### For Gemini CLI

```
You are Gemini CLI (Ops). Read HANDOFF.md. Your task is ops-only, no code.

1. Run: railway run pip show ortools
   Report: installed version or "NOT INSTALLED"

2. Run: railway variables | grep -i openweather
   Report: whether OPENWEATHER_API_KEY is set (do not print the value).
   Note: the correct key name is OPENWEATHER_API_KEY (not OPENWEATHERMAP_API_KEY or WEATHERAPI_KEY).

3. Update HANDOFF.md section 2 row "OR-Tools (Railway)" with your finding.

Do NOT edit any .py or .ts files.
```
