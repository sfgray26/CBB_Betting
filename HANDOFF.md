# OPERATIONAL HANDOFF (EMAC-027)

> Ground truth as of EMAC-027. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-027 — OpenClaw O-5 Root Cause Fix + Integrity Sweep Hardening

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-11: API Deduplication | OK | `/api/predictions/today` deduplicates by `game_id`. nightly > opener priority. (EMAC-020) |
| A-12: UI Dedup Guard | OK | `1_Todays_Bets.py` last-resort client-side dedup guard. (EMAC-020) |
| G-5: ReanalysisEngine Wiring | OK | `analysis.py` returns `(summary, cache)` tuple. `main.py` passes cache to `OddsMonitor`. (EMAC-021) |
| A-15: base_sd_override Invariant | OK | `CachedGameContext` stores dynamic SD. Unchanged-spread invariant holds. (EMAC-021) |
| A-16: ReanalysisEngine Tests | OK | 4 unit tests in `TestReanalysisEngine`. (EMAC-021) |
| G-6: VERDICT_FLIP Callback | OK | `OddsMonitor.poll()` detects true PASS->BET flips on significant spread moves. (EMAC-022/023) |
| A-17: VERDICT_FLIP Audit Fixes | OK | 4 bugs fixed. `_verdict_flip_fired` one-fire set. `original_verdict` in context. Pre-warm correct. (EMAC-023) |
| G-7: VERDICT_FLIP Discord Alert | OK | `_verdict_flip_handler` in lifespan. `send_verdict_flip_alert()` hardened. (EMAC-025) |
| A-18: Discord Alert Hardening | OK | 4 crash paths fixed in `send_verdict_flip_alert()`. (EMAC-025) |
| O-5: Integrity Sweep Async | OK | **CLOSED.** sync-in-async bug fixed. `asyncio.to_thread()` wrapper. 14 tests added. (EMAC-027) |
| Full test suite | OK | **427/427 passing.** |
| G-3: Railway Env Vars | PENDING | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. EMAC-027 FINDINGS — O-5 CLOSED

**O-5 was outstanding since EMAC-017.** OpenClaw repeatedly failed to deliver log evidence because the task was premised on a code bug: the integrity sweep was never truly async.

**Root Cause — Sync-in-Async (`_ddgs_and_check`):**
`_ddgs_and_check()` was declared `async def` but all I/O was synchronous blocking:
- `DDGS().text()` — synchronous HTTP to DuckDuckGo
- `perform_sanity_check()` → `requests.post(OLLAMA_URL)` — synchronous HTTP to Ollama

`asyncio.gather() + Semaphore(8)` provided zero concurrency. The log line fired, but all 8 "concurrent" checks ran serially on the event loop main thread.

**Fix (`backend/services/analysis.py`):**
- Extracted blocking logic into `_ddgs_and_check_sync(game)` — plain sync function
- `_ddgs_and_check(game)` now wraps it: `return await asyncio.to_thread(_ddgs_and_check_sync, game)`
- Thread pool concurrency verified by timing test: 5 x 50ms tasks complete in <0.20s

**Tests Added (`tests/test_integrity_sweep.py`, 14 tests):**
- `TestDdgsAndCheckSync` (5 tests): happy path, DDGS failure, scout failure, empty results, edge in verdict
- `TestDdgsAndCheckAsync` (3 tests): is coroutine, result passthrough, concurrency timing proof
- `TestIntegritySweep` (6 tests): empty input, keyed by game_key, fault isolation, VOLATILE passthrough, >8 games, missing edge field

---

### 3. EMAC-025 FINDINGS — VERDICT_FLIP DISCORD HARDENING

**Reviewed:** `discord_notifier.py::send_verdict_flip_alert()`, `main.py` lifespan callback

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| B-5 | `f"{edge:.1%}"` with `edge=None` -> TypeError | Critical | `edge = analysis.edge_conservative or 0.0` |
| B-6 | `f"{units:.2f}u"` with `units=None` -> TypeError | Critical | `units = analysis.recommended_units or 0.0` |
| B-7 | `f"{movement.old_value:+.1f}"` with `old_value=None` -> TypeError | Critical | Conditional string: `"N/A"` when None |
| B-8 | Fragile `.split('u ')` pick string parsing | High | Replaced with `calcs.get("bet_side", "home")` from `full_analysis["calculations"]` |

Callback ordering: OK (handler registered after `set_reanalysis_cache()`).
Callback accumulation: OK (registered in `lifespan()`, not `nightly_job()`).

---

### 4. V9 SYSTEM STATUS — FULLY OPERATIONAL

```
+----------------------------------------------------------+
|  V9 OPERATIONAL STACK                                    |
+------------------+---------------------------------------+
| Layer            | Status                                |
+------------------+---------------------------------------+
| Model (v9.0)     | OK  SNR + Integrity Kelly scalars     |
| Analysis         | OK  Returns (summary, cache) tuple    |
| Integrity Sweep  | OK  asyncio.to_thread, truly async    |
| ReanalysisEngine | OK  Unchanged-spread invariant holds  |
| OddsMonitor      | OK  VERDICT_FLIP with one-fire dedup  |
| Startup Pre-Warm | OK  Correct game_data + base_sd       |
| API Dedup        | OK  nightly > opener priority         |
| UI Dedup         | OK  Last-resort guard in Todays_Bets  |
| Discord Alerts   | OK  send_verdict_flip_alert hardened  |
| Callback Wiring  | OK  lifespan, no accumulation risk    |
| Test suite       | OK  427/427 passing                   |
+------------------+---------------------------------------+
```

**Outstanding:** G-3 Railway env vars — user action only, no code change.

---

### 5. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (USER ACTION — not a code change)**
- Go to Railway Dashboard for the CBB Edge project
- Add: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200
- Without these, Railway runs on Python env defaults — model will function but without tuned scalars

No other Gemini tasks. G-7 is closed. V9 stack is fully operational.

---

### 6. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**O-5 is CLOSED.** The sync-in-async bug was fixed at the code level (EMAC-027). No re-verification needed.

**Next Task: O-6 — Prod Integrity Verdict Spot-Check**
After G-3 Railway env vars are set and a live nightly analysis runs:
1. Query the DB for the most recent Prediction where `verdict` starts with "Bet"
2. Check `full_analysis["calculations"]["integrity_verdict"]`
3. If it is NOT `"Sanity check unavailable"` — report `O-6 CONFIRMED: [verdict string]`
4. If all verdicts are unavailable — Ollama is not configured in Railway; escalate as G-task for Gemini

---

### 7. ARCHITECT REVIEW QUEUE (Next EMAC)

- **SNR re-audit**: Run when n >= 20 alpha bets in DB: `python scripts/audit_confidence.py --days 90 --min-bets 20`
- **Season-end calibration**: Re-tune `sd_multiplier` and `home_advantage` with actual vs projected margins. Script: `scripts/generate_real_insights.py`
- **Scout hardening**: `backend/services/scout.py` uses free functions (not singleton). Consider singleton + connection pooling if Ollama proves unreliable in prod
- **V9 live verification**: After G-3, trigger manual analysis and confirm `snr_kelly_scalar` + `integrity_kelly_scalar` appear in a real Prediction's `full_analysis` JSON

---

### 8. HIVE WISDOM (Operational Lessons)

| Lesson | Source |
|--------|--------|
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id`. | EMAC-019 |
| Inflated bet count was run_tier dedup failure, not model logic. Fix at API + UI layers. | EMAC-020 |
| Always store `base_sd_override` in context. `None` != "same as original" — it means "use model default". | EMAC-021 |
| `full_analysis.inputs` has no "game" or "game_data" key. Reconstruct `game_data` from `p.game` DB relationship. | EMAC-023 |
| `MagicMock()._ctx.field` is truthy — always explicitly set mock context fields that drive conditional logic. | EMAC-023 |
| Scout `integrity_verdict` != model `verdict`. Never use one as a proxy for the other. | EMAC-023 |
| One-fire sets must be cleared on cache refresh — otherwise a new nightly run cannot trigger new flips. | EMAC-023 |
| Discord embed fields using `:.1%` or `:.2f` on None crash silently in prod. Guard with `or 0.0`. | EMAC-025 |
| Never parse verdict strings to extract pick info — use `full_analysis["calculations"]["bet_side"]` directly. | EMAC-025 |
| Register Discord/alert handlers in lifespan, not in nightly_job — prevents callback accumulation. | EMAC-025 |
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. `asyncio.gather` only yields at real `await` points. Use `asyncio.to_thread()` for all blocking network calls inside async functions. | EMAC-027 |
| When a recurring task is blocked for multiple sessions, suspect a code bug first — audit before re-delegating. | EMAC-027 |

---

### 9. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-028 — Railway Env Vars (G-3 Final)
You are the DevOps Strike Lead (Gemini CLI).

G-7 is closed. V9 Discord alerting and integrity sweep are fully operational (427/427 tests passing).
Your only outstanding task is G-3 (user action in Railway Dashboard — NOT a code change).

Action required:
1. Go to Railway Dashboard for the CBB Edge project.
2. Add environment variables: SNR_KELLY_FLOOR=0.5, INTEGRITY_CAUTION_SCALAR=0.75, INTEGRITY_VOLATILE_SCALAR=0.5
3. Redeploy the service.
4. Verify GET /health returns 200.
5. Report back with the health check response body.
6. Update HEARTBEAT.md with the verification date.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-028 — O-6 Prod Integrity Verdict Spot-Check
You are the Integrity Execution Unit (OpenClaw).
O-5 is CLOSED. The sync-in-async bug was fixed in code (EMAC-027). Read HANDOFF.md Section 6 for O-6.

Prerequisites: G-3 Railway env vars must be set and a live nightly analysis must have run.

Execute:
1. Query the DB for the most recent Prediction where verdict starts with "Bet".
2. Read full_analysis["calculations"]["integrity_verdict"] from that row.
3. If NOT "Sanity check unavailable": report "O-6 CONFIRMED: [verdict string]"
4. If all are "Sanity check unavailable": Ollama not in Railway — escalate to Gemini as new G-task.
5. Update HEARTBEAT.md with today's date and result.
```

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-028 — SNR Re-Audit + Season Calibration Prep
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 7 for your review queue.

Tasks:
1. SNR re-audit: Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.
   If n < 20 alpha bets, document current count and defer.
   If n >= 20, analyze output and flag any SNR floor bypass patterns.
2. Season calibration prep: Review `scripts/generate_real_insights.py` and document
   what data is needed from the DB for end-of-season recalibration.
   Identify which tables/fields feed sd_multiplier and home_advantage re-tuning.
3. V9 live verification: If G-3 is done, query DB for most recent prediction and
   confirm full_analysis["calculations"] contains snr, snr_kelly_scalar,
   integrity_verdict, integrity_kelly_scalar fields.
4. Run pytest tests/ -q --tb=short — must pass (427+).
5. Update HANDOFF.md to EMAC-029.
```
