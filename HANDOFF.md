# OPERATIONAL HANDOFF (EMAC-029)

> Ground truth as of EMAC-029. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-028 — SNR Re-Audit + Season Calibration Script Audit

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
| O-5: Integrity Sweep Async | OK | CLOSED. sync-in-async bug fixed. `asyncio.to_thread()` wrapper. 14 tests added. (EMAC-027) |
| A-19: SNR Re-Audit | DEFERRED | V9 data too new. Run after recalibration reports `status: ok` for first time. (EMAC-028) |
| A-20: Calibration Script Audit | OK | `generate_real_insights.py` is a narrative demo, NOT a calibration script. 3 bugs documented. True calibration: `backend/services/recalibration.py`. (EMAC-028) |
| Full test suite | OK | **427/427 passing.** |
| G-3: Railway Env Vars | PENDING | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. EMAC-028 FINDINGS

#### A. SNR Re-Audit — DEFERRED

Script (`scripts/audit_confidence.py`) logic is correct and sound. Cannot run: no DB connection available in dev, and V9 data is too new to hit the 20-alpha-bet minimum.

**What the script does (confirmed):**
- Queries Predictions with `snr IS NOT NULL` + `actual_margin IS NOT NULL` + `verdict LIKE "Bet%"` + within 90-day window
- Buckets into 4 SNR tiers (Alpha 0.90+, Strong 0.70-0.90, Moderate 0.50-0.70, Weak 0-0.50)
- Computes cover rate per tier, integrity group breakdown (CONFIRMED/CAUTION/VOLATILE), combined Kelly scalar
- Recommends raising/lowering `SNR_KELLY_FLOOR` based on gap between Alpha and Weak tier cover rates:
  - Gap < 3pp: raise floor to 0.65 (SNR not predictive)
  - Gap 3-8pp: maintain at 0.50 (adding value, within noise)
  - Gap > 8pp: lower floor to 0.35 (SNR strongly predictive)

**Deferral trigger:** Run AFTER `run_recalibration()` first returns `status: ok` (signals >= 30 settled BetLog rows, sufficient for SNR tier analysis).

#### B. Calibration Script Audit — FINDINGS

**HANDOFF.md Section 7 was wrong.** `scripts/generate_real_insights.py` is a **narrative demo tool** — it reads `tmp_today_data.json` and calls the Ollama LLM to produce scouting text. It does NOT touch `sd_multiplier` or `home_advantage`.

**True calibration script:** `backend/services/recalibration.py::run_recalibration()` — reads settled BetLogs, computes actual vs projected margins, re-tunes model parameters.

**Bugs confirmed in `generate_real_insights.py` (low priority — dev demo only):**
- Line 10: `sys.path.append(os.getcwd())` at module level — violates convention (must be inside `__main__` guard)
- Line 18: `open("tmp_today_data.json")` — relative path; fails outside project root
- Line 21: `print(f"GENERATING INSIGHTS...")` with emoji — CP-1252 crash on Windows terminal
- Line 41: `n_considered=5` hardcoded — not derived from actual CONSIDER count

---

### 3. V9 SYSTEM STATUS — FULLY OPERATIONAL

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

### 4. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (USER ACTION — not a code change)**
- Go to Railway Dashboard for the CBB Edge project
- Add: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200
- Without these, Railway runs on Python env defaults

No other Gemini tasks. V9 stack is fully operational.

---

### 5. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**O-5 is CLOSED.** The sync-in-async bug was fixed at code level (EMAC-027).

**Task O-6 — Prod Integrity Verdict Spot-Check**
Prerequisites: G-3 must be set + a live nightly analysis must have run.
1. Query DB for most recent Prediction where `verdict` starts with "Bet"
2. Check `full_analysis["calculations"]["integrity_verdict"]`
3. If NOT "Sanity check unavailable": report `O-6 CONFIRMED: [verdict string]`
4. If all unavailable: Ollama not in Railway — escalate as new G-task for Gemini
5. Update HEARTBEAT.md with today's date and result

---

### 6. ARCHITECT REVIEW QUEUE (Next EMAC)

- **SNR re-audit (A-19)**: Deferred. Run `python scripts/audit_confidence.py --days 90 --min-bets 20` AFTER `run_recalibration()` returns `status: ok` for the first time (>= 30 settled bets).
- **V9 live verification**: After G-3, trigger manual analysis and confirm `snr_kelly_scalar` + `integrity_kelly_scalar` appear in a real Prediction's `full_analysis` JSON.
- **`generate_real_insights.py` cleanup**: Fix 3 bugs (sys.path, relative path, emoji). Low priority — dev demo only.
- **Recalibration readiness**: Once >= 30 settled BetLogs exist, run `backend/services/recalibration.py` to produce first real `sd_multiplier`/`home_advantage` update. Verify it writes to the `model_parameters` DB table.

---

### 7. HIVE WISDOM (Operational Lessons)

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
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. `asyncio.gather` only yields at real `await` points. | EMAC-027 |
| When a recurring task is blocked for multiple sessions, suspect a code bug first — audit before re-delegating. | EMAC-027 |
| Script names in HANDOFF may not match their actual function. Read the code to verify before referencing. | EMAC-028 |
| The true calibration entry point is `backend/services/recalibration.py::run_recalibration()` — not any script in `/scripts/`. | EMAC-028 |

---

### 8. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-029 — Railway Env Vars (G-3 Final)
You are the DevOps Strike Lead (Gemini CLI).

V9 stack is fully operational (427/427 tests passing). Your only task is G-3.

Action required (NOT a code change):
1. Go to Railway Dashboard for the CBB Edge project.
2. Add environment variables: SNR_KELLY_FLOOR=0.5, INTEGRITY_CAUTION_SCALAR=0.75, INTEGRITY_VOLATILE_SCALAR=0.5
3. Redeploy the service.
4. Verify GET /health returns 200.
5. Report back with the health check response body.
6. Update HEARTBEAT.md with the verification date.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-029 — O-6 Prod Integrity Verdict Spot-Check
You are the Integrity Execution Unit (OpenClaw).
O-5 is CLOSED. Read HANDOFF.md Section 5 for O-6.

Prerequisites: G-3 Railway env vars must be set and a live nightly analysis must have run.

Execute:
1. Query DB for the most recent Prediction where verdict starts with "Bet".
2. Read full_analysis["calculations"]["integrity_verdict"] from that row.
3. If NOT "Sanity check unavailable": report "O-6 CONFIRMED: [verdict string]"
4. If all are "Sanity check unavailable": Ollama not in Railway — escalate to Gemini as new G-task.
5. Update HEARTBEAT.md with today's date and result.
```

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-029 — V9 Live Verification + First Recalibration Readiness Check
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 6 for your review queue.

Tasks:
1. V9 live verification: If G-3 is done, trigger POST /admin/run-analysis and query the DB
   for the most recent Prediction. Confirm full_analysis["calculations"] contains:
   snr, snr_kelly_scalar, integrity_verdict, integrity_kelly_scalar.
   If G-3 is not done, skip and note it.

2. Recalibration readiness: Query the DB for count of BetLog rows where outcome IS NOT NULL.
   If count >= 30: Run backend/services/recalibration.py::run_recalibration() directly or
   via POST /admin/recalibrate. Confirm it writes updated parameters to the model_parameters table.
   If count < 30: Document the current count and defer.

3. Fix generate_real_insights.py bugs (low priority):
   - Line 10: Move sys.path.append inside __main__ guard
   - Line 18: Use os.path.join(os.path.dirname(__file__), "..", "tmp_today_data.json")
   - Line 21: Remove emoji from print statement (CP-1252 Windows terminal issue)

4. Run pytest tests/ -q --tb=short — must pass (427+).
5. Update HANDOFF.md to EMAC-030.
```
