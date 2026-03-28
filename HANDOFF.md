# OPERATIONAL HANDOFF — MARCH 28, 2026: STATE CONSOLIDATION

> **Ground truth as of March 28, 2026 EOD.** Author: Claude Code (Master Architect).
> See `IDENTITY.md` for risk posture · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior active crises: DB migration failure, UI routing cascade, production outage — all resolved.

---

## 1. Mission Accomplished (This Session)

| Task | Status | Notes |
|------|--------|-------|
| **Yahoo client consolidation** | ✅ DONE | `yahoo_client.py` deleted — inlined into `yahoo_client_resilient.py` |
| **openclaw_briefs.py deleted** | ✅ DONE | All callers re-pointed to `openclaw_briefs_improved.py` |
| **Tests moved to `tests/`** | ✅ DONE | `test_lineup_validator.py` + `test_waiver_recovery.py` |
| **Streamlit retired** | ✅ DONE | `dashboard/` is dead code. Next.js is the only UI. Never touch again. |
| **Dashboard B1 stubs wired** | ✅ DONE | `_get_waiver_targets`, `_get_matchup_preview`, `_get_probable_pitchers` → live services |
| **Dashboard DB fallback** | ✅ DONE | `_get_lineup_gaps` roster bug fixed; `_get_streaks` falls back to DB when Yahoo down |
| **Weather integration (B2)** | ✅ DONE | `smart_lineup_selector.py` → `weather_fetcher.py` → `park_weather.py` — fully wired |
| **Production outage resolved** | ✅ DONE | Roster 500 fixed (Pydantic bool→str); injury_status wired; matchup team mapping; UTC→ET |
| **OR-Tools added to deps** | ✅ DONE | `ortools>=9.8` in `requirements.txt`; greedy fallback active until Railway redeploys |
| **EMAC doc consolidation** | ✅ DONE | HANDOFF, HEARTBEAT, AGENTS, ORCHESTRATION, IDENTITY all rewritten to current reality |

---

## 2. Technical State Table

| Component | Status | Notes |
|-----------|--------|-------|
| **DB Migrations v9/v10** | ✅ LIVE | Chained into Dockerfile CMD; `user_preferences` table confirmed present |
| **Yahoo client** | ✅ CONSOLIDATED | Single file: `yahoo_client_resilient.py`. Base class + resilient layer unified. |
| **Roster endpoint (`/api/fantasy/roster`)** | ✅ LIVE | 200 OK. Pydantic `status` bool→None fixed. `injury_status` now populated. |
| **Matchup endpoint (`/api/fantasy/matchup`)** | ✅ LIVE | Team mapping fixed — handles Yahoo indexed format (`m["0"]["teams"]`). Diagnostic log added. |
| **UTC timezone bug** | ✅ FIXED | `daily_lineup_optimizer.py` — all 3 `datetime.utcnow()` → `datetime.now(ZoneInfo("America/New_York"))` |
| **Weather integration** | ✅ LIVE | Provider: OpenWeatherMap. Env var `OPENWEATHER_API_KEY` confirmed set in Railway. |
| **OR-Tools** | 🔄 PENDING DEPLOY | In `requirements.txt`. Greedy fallback active in interim. |
| **Kimi UI/API hotfix** | ✅ MERGED | Roster dedup, `get_my_team_key()` recursive parse, `_safe_float()` NaN guard. |
| **Streamlit** | ✅ RETIRED | `dashboard/` untouched. Next.js is canonical UI. |
| **Test suite** | ✅ STABLE | 1198 pass / 5 pre-existing failures (3 DB-auth, 1 TTL logic, 1 CircuitBreaker) |
| **Build pipeline** | ✅ GREEN | Python syntax clean across all modified files; TypeScript builds |
| **MCMC Simulator** | ❌ SCAFFOLDED | `mcmc_simulator.py` exists, not calibrated. B5 roadmap item. |
| **CBB V9.1 recalibration** | ⏸ BLOCKED | EMAC-068 — blocked until post-Apr 7. Do not touch Kelly math. |

---

## 3. Resolved Crises (Archive — Do Not Revisit)

| Crisis | Resolution | Date |
|--------|------------|------|
| `user_preferences` table missing | v9/v10 migrations chained into Dockerfile CMD | Mar 27, 2026 |
| Pydantic `status: False` → HTTP 500 | `_parse_player` + `RosterPlayerOut` constructor guarded with `or None` | Mar 28, 2026 |
| Matchup "Team not found" | `m.get("teams") or m.get("0", {}).get("teams", {})` | Mar 28, 2026 |
| West Coast games showing no-game | `datetime.utcnow()` → `datetime.now(ZoneInfo("America/New_York"))` | Mar 28, 2026 |
| `injury_status` always None | Added `injury_status=p.get("status") or None` to `RosterPlayerOut` constructor | Mar 28, 2026 |
| `_get_lineup_gaps` empty on `team_key=None` | `client.get_roster()` (no-arg form) replaces silent `[]` | Mar 28, 2026 |
| UI routing cascade (Kimi hotfix) | Roster dedup, team key recursive parse, NaN float guard | Mar 28, 2026 |

---

## 4. Delegation Bundles

### CLAUDE CODE (Master Architect) — Next Session

> **Priority 1: Railway redeploy**
>
> `ortools>=9.8` is now in `requirements.txt`. Next Railway redeploy will install it and restore
> ILP optimization in `lineup_constraint_solver.py` (currently running greedy fallback).
> No code changes needed — just trigger deploy via Gemini.

> **Priority 2: Dashboard SSE endpoint**
>
> Decision locked: SSE via FastAPI `EventSourceResponse` (simpler than WebSocket, Railway-compatible).
> Wire: `GET /api/fantasy/dashboard/stream` → yield panel updates every 60s.
> Panels to stream: `waiver_targets`, `matchup_preview`, `probable_pitchers`, `streaks`.
> Target file: `backend/main.py` (new route) + `backend/services/dashboard_service.py` (yield helper).

> **Priority 3: Matchup preview enrichment**
>
> `_get_matchup_preview` is live but returns `win_prob=0.5`, `opponent_record=""`, `my_projected_categories={}`.
> Next pass: populate `opponent_record` from `client.get_standings()`.
> Populate `my_projected_categories` / `opponent_projected_categories` from
> `PlayerDailyMetric.z_score_recent` + `rolling_window["7d"]["avg"]` for rostered players.
> Do NOT wire MCMC win probability yet — that is B5, post-calibration.

> **Priority 4: CircuitBreaker test decision**
>
> `test_waiver_recovery.py::TestCircuitBreaker::test_opens_after_threshold` — pre-existing failure.
> `backend/fantasy_baseball/circuit_breaker.py` propagates `ValueError` instead of `CircuitOpenError`.
> Options: (a) fix CircuitBreaker to catch and re-raise as `CircuitOpenError`, or
> (b) update test to `pytest.raises(ValueError)`. Option (b) is faster; option (a) is correct.

### KIMI CLI (Deep Intelligence Unit) — Standby

> Kimi's doc refactor (MASTER_DOCUMENT_INDEX.md + deprecation headers) is complete and accepted.
> No active coding tasks. Next delegation will come when Matchup enrichment is scoped or
> when the CBB recalibration (EMAC-068) is unblocked post-Apr 7.
>
> If initiating a new session unprompted: read this HANDOFF.md + `HEARTBEAT.md`.
> Do not write to any production code files without a Claude delegation bundle.

### GEMINI CLI (Ops) — Active

> 1. Trigger Railway redeploy to install `ortools>=9.8` from updated `requirements.txt`.
>    Confirm: `railway run python -c "from ortools.sat.python import cp_model; print('ok')"`
> 2. Monitor logs for 30 min post-deploy — confirm no new errors.
> 3. Report back: `OR-Tools: INSTALLED` or any error seen in deploy logs.
>
> Do NOT edit any `.py` or `.ts` files.

---

## 5. Architecture Decisions (Locked)

| Decision | Ruling | Reason |
|----------|--------|--------|
| Yahoo client split-brain | ELIMINATED | Single file: `yahoo_client_resilient.py` |
| Streamlit | RETIRED | Next.js only — never touch `dashboard/` |
| `openclaw_briefs.py` (old) | DELETED | `_improved` is canonical |
| Dashboard refresh strategy | SSE (pending impl) | Over WebSocket — simpler, Railway-compatible |
| Weather provider | OpenWeatherMap (LOCKED) | Fully implemented in `weather_fetcher.py`; `OPENWEATHER_API_KEY` set |
| Test file location | `tests/` only | No test files in `backend/` subdirs |
| CBB recalibration | BLOCKED until Apr 7 | EMAC-068 — do not touch Kelly math before then |

---

## 6. Architect Review Queue

1. **`test_waiver_recovery.py::test_opens_after_threshold`** — CircuitBreaker in `backend/fantasy_baseball/circuit_breaker.py` propagates `ValueError` instead of `CircuitOpenError` after threshold. Decide: fix the CircuitBreaker or update the test.
2. **MCMC Simulator calibration** — `backend/fantasy_baseball/mcmc_simulator.py` exists but unvalidated. Schedule calibration pass for B5 roadmap (post-matchup enrichment).
3. **CBB V9.1 recalibration (EMAC-068)** — Blocked until post-Apr 7. SNR/integrity scalar stacking issue confirmed. Do not touch Kelly math until then.
4. **Matchup page RP-as-SP bug** — Edwin Díaz, Jason Adam (RPs) appearing in Starting Pitchers table. Root cause: SP filter not excluding by `pitcher_slot == "SP"`. Defer until after SSE work.

---

## HANDOFF PROMPTS

### For Gemini CLI

```
You are Gemini CLI (Ops). Read this HANDOFF.md in full.

Your sole task this session:
1. Trigger a Railway redeploy (push or manual trigger) so that requirements.txt changes take effect.
   The key change: `ortools>=9.8` was added to requirements.txt.

2. After deploy completes, run:
   railway run python -c "from ortools.sat.python import cp_model; print('OR-Tools: INSTALLED')"
   Report the output.

3. Monitor Railway logs for 5 minutes post-deploy.
   Report: any new errors seen, or "No new errors."

4. Update HANDOFF.md Section 2 row "OR-Tools" to ✅ LIVE with today's date.

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
