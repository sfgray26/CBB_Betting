# OPERATIONAL HANDOFF — MARCH 28, 2026: SSE + MATCHUP ENRICHMENT

> **Ground truth as of March 28, 2026 (second session).** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk posture · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior active crises: all resolved (see archive below).

---

## 1. Mission Accomplished (This Session)

| Task | Status | Notes |
|------|--------|-------|
| **Priority A: CircuitBreaker fix** | ✅ DONE | `except self.expected_exception` → `except Exception`; all failures now count toward threshold. `test_opens_after_threshold`, `test_rejects_calls_when_open`, `test_failures_increment_count` all pass. |
| **Priority B: SSE endpoint** | ✅ DONE | `GET /api/fantasy/dashboard/stream` wired via `StreamingResponse` (no external deps). Streams 4 panels every 60s. `?panels=` filter supported. Yahoo auth errors yield `event: error` events. |
| **Priority C: Matchup enrichment** | ✅ DONE | `_get_matchup_preview` now populates `opponent_record` (W-L-T from Yahoo standings) and `my_projected_categories`/`opponent_projected_categories` (7-day rolling avg from `PlayerDailyMetric`). Win prob stays 0.5 (MCMC is B5). |

---

## 2. Technical State Table

| Component | Status | Notes |
|-----------|--------|-------|
| **DB Migrations v9/v10** | ✅ LIVE | Chained into Dockerfile CMD; `user_preferences` table confirmed present |
| **Yahoo client** | ✅ CONSOLIDATED | Single file: `yahoo_client_resilient.py`. Base class + resilient layer unified. |
| **Roster endpoint (`/api/fantasy/roster`)** | ✅ LIVE | 200 OK. |
| **Matchup endpoint (`/api/fantasy/matchup`)** | ✅ LIVE | Team mapping fixed. |
| **SSE stream (`/api/fantasy/dashboard/stream`)** | ✅ NEW | `StreamingResponse`, `text/event-stream`, 60s interval. No `sse-starlette` dep needed. |
| **Matchup enrichment** | ✅ NEW | `opponent_record` from standings. `*_projected_categories` from `PlayerDailyMetric`. |
| **CircuitBreaker** | ✅ FIXED | Catches `Exception` (not just `expected_exception`) so all error types trip the breaker. |
| **Weather integration** | ✅ LIVE | Provider: OpenWeatherMap. |
| **OR-Tools (Railway)** | ✅ LIVE (March 28, 2026) | Installed via requirements.txt. |
| **Kimi UI/API hotfix** | ✅ MERGED | Roster dedup, `get_my_team_key()` recursive parse, `_safe_float()` NaN guard. |
| **Streamlit** | ✅ RETIRED | `dashboard/` untouched. Next.js is canonical UI. |
| **Test suite** | ✅ STABLE | CircuitBreaker test fixed. Overall: 1199+ pass. |
| **MCMC Simulator** | ❌ SCAFFOLDED | `mcmc_simulator.py` exists, not calibrated. B5 roadmap item. |
| **CBB V9.1 recalibration** | ⏸ BLOCKED | EMAC-068 — blocked until post-Apr 7. Do not touch Kelly math. |

---

## 3. Files Modified This Session

| File | Change |
|------|--------|
| `backend/fantasy_baseball/circuit_breaker.py` | `except self.expected_exception:` → `except Exception:` in both `call()` and `call_async()`. Removed unused `expected_exception` branch. |
| `backend/main.py` | Added `GET /api/fantasy/dashboard/stream` SSE endpoint (~115 lines). Added `import json as _json` and `_ALL_PANELS` module-level constant. |
| `backend/services/dashboard_service.py` | `_get_matchup_preview` enriched with `opponent_record` + projected categories. Added static methods `_extract_team_standings`, `_fetch_team_record`, `_project_categories_from_db`. |

---

## 4. Resolved Crises (Archive — Do Not Revisit)

| Crisis | Resolution | Date |
|--------|------------|------|
| `user_preferences` table missing | v9/v10 migrations chained into Dockerfile CMD | Mar 27, 2026 |
| Pydantic `status: False` → HTTP 500 | `_parse_player` + `RosterPlayerOut` constructor guarded with `or None` | Mar 28, 2026 |
| Matchup "Team not found" | `m.get("teams") or m.get("0", {}).get("teams", {})` | Mar 28, 2026 |
| West Coast games showing no-game | `datetime.utcnow()` → `datetime.now(ZoneInfo("America/New_York"))` | Mar 28, 2026 |
| `injury_status` always None | Added `injury_status=p.get("status") or None` to `RosterPlayerOut` constructor | Mar 28, 2026 |
| `_get_lineup_gaps` empty on `team_key=None` | `client.get_roster()` (no-arg form) replaces silent `[]` | Mar 28, 2026 |
| UI routing cascade (Kimi hotfix) | Roster dedup, team key recursive parse, NaN float guard | Mar 28, 2026 |
| CircuitBreaker only counted `expected_exception` errors | `except Exception:` in `call()` and `call_async()` | Mar 28, 2026 |

---

## 5. Delegation Bundles

### GEMINI CLI (Ops) — Active

> 1. Run `py_compile` verification for the two modified files:
>    ```
>    railway run python -m py_compile backend/fantasy_baseball/circuit_breaker.py && echo "OK circuit_breaker"
>    railway run python -m py_compile backend/services/dashboard_service.py && echo "OK dashboard_service"
>    railway run python -m py_compile backend/main.py && echo "OK main"
>    ```
> 2. Trigger Railway redeploy (OR-Tools was already added in prior session; confirm it's still live).
> 3. Smoke-test the SSE endpoint after deploy:
>    ```
>    curl -N -H "X-API-Key: $API_KEY" "https://<railway-url>/api/fantasy/dashboard/stream?panels=streaks" | head -20
>    ```
>    Expected: streaming SSE text with `event: streaks` or `event: error`.
> 4. Report back: compile status, deploy status, SSE smoke test result.
>
> Do NOT edit any `.py` or `.ts` files.

### KIMI CLI (Deep Intelligence Unit) — Standby

> No active coding tasks. Next delegation will come when CBB recalibration (EMAC-068) is
> unblocked post-Apr 7.
>
> If initiating a new session unprompted: read this HANDOFF.md + `HEARTBEAT.md`.
> Do not write to any production code files without a Claude delegation bundle.

### CLAUDE CODE — Next Session Roadmap

> 1. ~~**Matchup RP-as-SP bug**~~ ✅ DONE (Mar 29): Added `p.get("pitcher_slot") != "SP"` guard to `_get_probable_pitchers` in `dashboard_service.py`. Edwin Diaz / Jason Adam no longer appear as starters.
> 2. **Yahoo stat category IDs** ✅ DONE (Mar 29): `_YAHOO_STAT_FALLBACK` dict pre-seeds `stat_id_map` in matchup + waiver endpoints. Frontend `STAT_LABELS` completed with all 22 numeric IDs.
> 3. **MCMC Simulator calibration** (B5) — `backend/fantasy_baseball/mcmc_simulator.py` needs
>    validation against historical matchup data before wiring into `win_probability`.
> 4. **CBB V9.2 recalibration** (EMAC-068) — Unblocks Apr 7. SNR/integrity scalar stacking
>    correction. Do not touch Kelly math until then.

---

## 6. Architecture Decisions (Locked)

| Decision | Ruling | Reason |
|----------|--------|--------|
| Yahoo client split-brain | ELIMINATED | Single file: `yahoo_client_resilient.py` |
| Streamlit | RETIRED | Next.js only — never touch `dashboard/` |
| `openclaw_briefs.py` (old) | DELETED | `_improved` is canonical |
| Dashboard refresh strategy | SSE (IMPLEMENTED) | `StreamingResponse` text/event-stream. No sse-starlette dep. |
| Weather provider | OpenWeatherMap (LOCKED) | Fully implemented; `OPENWEATHER_API_KEY` set |
| Test file location | `tests/` only | No test files in `backend/` subdirs |
| CBB recalibration | BLOCKED until Apr 7 | EMAC-068 — do not touch Kelly math before then |
| SSE keep-alive | `: keep-alive\n\n` comment line | Prevents Railway/nginx from closing idle SSE connections |

---

## 7. Architect Review Queue

1. ~~**Matchup page RP-as-SP bug**~~ ✅ RESOLVED Mar 29 — `pitcher_slot == "SP"` guard added.
2. **MCMC Simulator calibration** — `backend/fantasy_baseball/mcmc_simulator.py` unvalidated. B5.
3. **CBB V9.1 recalibration (EMAC-068)** — Blocked until post-Apr 7.
4. **`_project_categories_from_db` day-0 scenario** — On Opening Day (no metrics in DB yet),
   returns `{}`. This is correct behavior. Will populate as Statcast data flows in.

---

## HANDOFF PROMPTS

### For Gemini CLI

```
You are Gemini CLI (Ops). Read this HANDOFF.md in full.

Your tasks this session:

1. Verify Python syntax is clean on the three modified files:
   cd /path/to/cbb-edge
   python -m py_compile backend/fantasy_baseball/circuit_breaker.py && echo "OK circuit_breaker"
   python -m py_compile backend/services/dashboard_service.py && echo "OK dashboard_service"
   python -m py_compile backend/main.py && echo "OK main"

2. Trigger Railway redeploy if any changes are pending (git status will show).

3. After deploy, smoke-test the SSE endpoint:
   curl -N -H "X-API-Key: YOUR_API_KEY" \
     "https://YOUR_RAILWAY_URL/api/fantasy/dashboard/stream?panels=streaks" \
     --max-time 10
   Expected output: lines starting with "event: streaks" or "event: error".

4. Confirm OR-Tools is still installed:
   railway run python -c "from ortools.sat.python import cp_model; print('OR-Tools: OK')"

5. Report all results. If py_compile fails on any file, paste the exact error.

Do NOT edit any .py or .ts files.
Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```

### For Kimi CLI (if initiating independently)

```
You are Kimi CLI (Deep Intelligence Unit). Read this HANDOFF.md in full.

No active coding tasks are assigned. Your standing responsibilities:
- If any new `reports/` files have been requested in HANDOFF Delegation Bundles, execute them.
- If CBB recalibration (EMAC-068) status changes to unblocked, prepare the V9.2 spec memo.
- Do NOT write to any production code files without an explicit Claude delegation bundle.

Working directory: C:/Users/sfgra/repos/Fixed/cbb-edge
```
