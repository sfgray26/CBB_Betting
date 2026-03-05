# OPERATIONAL HANDOFF (EMAC-030)

> Ground truth as of EMAC-030. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-029 — Recalibration Run + generate_real_insights.py Fixes

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-11: API Deduplication | OK | `/api/predictions/today` deduplicates by `game_id`. nightly > opener priority. (EMAC-020) |
| A-12: UI Dedup Guard | OK | `1_Todays_Bets.py` last-resort client-side dedup guard. (EMAC-020) |
| G-5: ReanalysisEngine Wiring | OK | `analysis.py` returns `(summary, cache)` tuple. `main.py` passes cache to `OddsMonitor`. (EMAC-021) |
| A-15: base_sd_override Invariant | OK | `CachedGameContext` stores dynamic SD. Unchanged-spread invariant holds. (EMAC-021) |
| A-16: ReanalysisEngine Tests | OK | 4 unit tests in `TestReanalysisEngine`. (EMAC-021) |
| G-6: VERDICT_FLIP Callback | OK | `OddsMonitor.poll()` detects true PASS->BET flips. (EMAC-022/023) |
| A-17: VERDICT_FLIP Audit Fixes | OK | 4 bugs fixed. One-fire set. `original_verdict` in context. Pre-warm correct. (EMAC-023) |
| G-7: VERDICT_FLIP Discord Alert | OK | `_verdict_flip_handler` in lifespan. `send_verdict_flip_alert()` hardened. (EMAC-025) |
| A-18: Discord Alert Hardening | OK | 4 crash paths fixed in `send_verdict_flip_alert()`. (EMAC-025) |
| O-5: Integrity Sweep Async | OK | CLOSED. `asyncio.to_thread()` wrapper. 14 tests. (EMAC-027) |
| A-19: SNR Re-Audit | DEFERRED | V9 DB data too new. Trigger: run after first successful `run_recalibration()` with V9 rows. (EMAC-028) |
| A-20: Calibration Script Audit | OK | `generate_real_insights.py` corrected — it is a narrative demo. True calibration: `recalibration.py`. (EMAC-028) |
| A-21: First Recalibration | OK | **COMPLETE.** 663 settled BetLogs. 2 parameters updated. Written to `model_parameters`. (EMAC-029) |
| A-22: generate_real_insights.py | OK | 3 bugs fixed: sys.path, relative path, emoji. (EMAC-029) |
| G-3: Railway Env Vars | PENDING | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |
| V9 Live Verification | PENDING | G-3 not set. Most recent DB prediction is v8.0. Verify after G-3 + fresh nightly run. |
| Full test suite | OK | **427/427 passing.** |

---

### 2. EMAC-029 FINDINGS

#### A. Recalibration — COMPLETE (663 settled bets)

First recalibration with substantial data. `run_recalibration()` analyzed 200 most-recent settled bets and updated two parameters:

| Parameter | Old | New | Signal | Interpretation |
|-----------|-----|-----|--------|----------------|
| `home_advantage` | 2.000 | **2.419** | ha_bias = -1.675 | Model was under-predicting home team margins by 1.7 pts on average. Home court is worth more than we were giving it. |
| `sd_multiplier` | 0.970 | **1.000** | overconfidence = +0.064, brier = 0.331 | Model was 6.4% overconfident — probabilities too extreme. Widening distribution by returning to 1.0. |

Both changes written to `model_parameters` DB table (IDs 17, 18, effective 2026-03-05).

**sd_multiplier oscillation note:** This is the third time sd_multiplier has flipped between 0.97 and 1.0 (Feb 23, Feb 25, Feb 27, Mar 5). The overconfidence signal is small and noisy at the edge. Consider adding a minimum change threshold (e.g., only update if |delta| > 0.05) to prevent flip-flopping once data volume grows. Not urgent.

**Prior recalibration history (model_parameters table):**
- 2026-02-23: ha 1.59→1.5, sd 0.94→0.97
- 2026-02-25: sd 0.97→1.0
- 2026-02-27: sd 1.0→0.97, ha 1.5→2.0
- 2026-03-04: weight_kenpom 0.45→0.444, weight_barttorvik 0.40→0.409 (auto_daily)
- 2026-03-05: ha 2.0→2.419, sd 0.97→1.0 (EMAC-029)

#### B. V9 Live Verification — BLOCKED

G-3 env vars not set. Most recent Prediction in DB: `model_version=v8.0`, `snr=None`, `integrity_verdict=None`. This is expected — no V9 nightly analysis has run against prod yet. V9 fields will appear after G-3 is set in Railway and the next nightly job fires.

#### C. generate_real_insights.py — 3 bugs fixed

- `sys.path.append(os.getcwd())` at module level → moved inside `__main__` guard as `sys.path.insert(0, ...)`
- `from backend.services.scout import ...` at module level → moved inside `run_real_test()` (lazy import, decouples from path setup)
- Hard-coded `open("tmp_today_data.json")` → `os.path.join(os.path.dirname(__file__), "..", "tmp_today_data.json")`
- Emoji removed from print statement (CP-1252 Windows terminal)

---

### 3. V9 SYSTEM STATUS — OPERATIONAL (pending prod verification)

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
| Recalibration    | OK  ha=2.419, sd=1.0 (200 bets)       |
| V9 DB Verify     | PENDING  Needs G-3 + fresh nightly    |
| Test suite       | OK  427/427 passing                   |
+------------------+---------------------------------------+
```

---

### 4. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (USER ACTION — not a code change)**
- Go to Railway Dashboard for the CBB Edge project
- Add: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200
- This unblocks V9 live verification (next nightly run will write v9.0 predictions with SNR fields)

---

### 5. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**O-5 is CLOSED.**

**Task O-6 — Prod Integrity Verdict Spot-Check**
Prerequisites: G-3 must be set + live nightly analysis must have run + prediction `model_version = 'v9.0'` in DB.
1. Query DB: `SELECT integrity_verdict, full_analysis FROM predictions WHERE model_version='v9.0' ORDER BY created_at DESC LIMIT 1`
2. Check `full_analysis["calculations"]["integrity_verdict"]`
3. If NOT "Sanity check unavailable": report `O-6 CONFIRMED: [verdict]`
4. If all unavailable: Ollama not in Railway — escalate to Gemini as G-8

---

### 6. ARCHITECT REVIEW QUEUE (Next EMAC)

- **V9 live verification (A-23)**: After G-3 is set and nightly runs, query DB for a v9.0 Prediction. Confirm `snr`, `snr_kelly_scalar`, `integrity_verdict`, `integrity_kelly_scalar` in `full_analysis["calculations"]`.
- **SNR re-audit (A-19)**: Deferred until V9 data accumulates. Run `python scripts/audit_confidence.py --days 90 --min-bets 20` once V9 predictions start being settled.
- **sd_multiplier oscillation**: Consider adding `MIN_RECAL_DELTA=0.05` guard to `recalibration.py` to prevent flip-flopping between 0.97 and 1.0 on noisy small signals.
- **Season-end**: At season end, run full recalibration on the complete dataset to re-tune all parameters with N > 500 V9-era bets.

---

### 7. HIVE WISDOM (Operational Lessons)

| Lesson | Source |
|--------|--------|
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id`. | EMAC-019 |
| Inflated bet count was run_tier dedup failure, not model logic. Fix at API + UI layers. | EMAC-020 |
| Always store `base_sd_override` in context. `None` != "same as original" — means "use model default". | EMAC-021 |
| `full_analysis.inputs` has no "game" or "game_data" key. Reconstruct from `p.game` DB relationship. | EMAC-023 |
| `MagicMock()._ctx.field` is truthy — always explicitly set mock context fields that drive conditional logic. | EMAC-023 |
| Scout `integrity_verdict` != model `verdict`. Never use one as proxy for the other. | EMAC-023 |
| One-fire sets must be cleared on cache refresh — otherwise new nightly runs cannot trigger new flips. | EMAC-023 |
| Discord embed fields using `:.1%` or `:.2f` on None crash silently. Guard with `or 0.0`. | EMAC-025 |
| Never parse verdict strings — use `full_analysis["calculations"]["bet_side"]` directly. | EMAC-025 |
| Register Discord/alert handlers in lifespan, not nightly_job — prevents callback accumulation. | EMAC-025 |
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. | EMAC-027 |
| When a recurring task is blocked for many sessions, suspect a code bug — audit before re-delegating. | EMAC-027 |
| Script names in HANDOFF may not match their actual function. Always read the code to verify. | EMAC-028 |
| True calibration entry point is `backend/services/recalibration.py::run_recalibration()`. | EMAC-028 |
| `sys.path` manipulation belongs inside `if __name__ == "__main__":` guards, never at module level. | EMAC-029 |
| `ModelParameter` columns: `id, effective_date, parameter_name, parameter_value, parameter_value_json, reason, changed_by, created_at`. No `updated_at` or `version`. | EMAC-029 |
| sd_multiplier oscillates at the noise boundary (0.97 <-> 1.0). Add a min-delta guard to recalibration before season end. | EMAC-029 |

---

### 8. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-030 — Railway Env Vars (G-3 Final)
You are the DevOps Strike Lead (Gemini CLI).

V9 code is fully operational and tested (427/427). Recalibration ran successfully with 663 settled bets.
The only blocker to V9 live verification is G-3 (Railway Dashboard — NOT a code change).

Action required:
1. Go to Railway Dashboard for the CBB Edge project.
2. Add: SNR_KELLY_FLOOR=0.5, INTEGRITY_CAUTION_SCALAR=0.75, INTEGRITY_VOLATILE_SCALAR=0.5
3. Redeploy the service.
4. Verify GET /health returns 200.
5. Report back with the health check response body.
6. Update HEARTBEAT.md with the verification date.

Once G-3 is done, the next nightly analysis will write v9.0 predictions with full SNR and
integrity fields, enabling O-6 verification by OpenClaw.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-030 — O-6 Prod Integrity Verdict Spot-Check
You are the Integrity Execution Unit (OpenClaw). O-5 is CLOSED.
Read HANDOFF.md Section 5 for O-6 prerequisites.

Prerequisites (both must be true before proceeding):
- G-3 Railway env vars are set (confirm with GET /health response)
- A nightly analysis has run and produced predictions with model_version='v9.0'

Execute:
1. Query: SELECT id, verdict, integrity_verdict, full_analysis->>'calculations'
   FROM predictions WHERE model_version='v9.0' ORDER BY created_at DESC LIMIT 1
2. Check if full_analysis["calculations"]["integrity_verdict"] is set and not "Sanity check unavailable".
3. Report "O-6 CONFIRMED: [verdict]" or escalate to Gemini as G-8 if Ollama is not in Railway.
4. Update HEARTBEAT.md with today's date and result.
```

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-030 — V9 Live Verification + sd_multiplier Oscillation Fix
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 6 for your review queue.

Tasks:
1. V9 live verification: After G-3 is confirmed set in Railway, trigger POST /admin/run-analysis.
   Query DB for most recent Prediction where model_version='v9.0'.
   Confirm full_analysis["calculations"] contains: snr, snr_kelly_scalar,
   integrity_verdict, integrity_kelly_scalar.
   If G-3 not done yet: skip and note it.

2. sd_multiplier oscillation fix: Add a MIN_RECAL_DELTA guard to recalibration.py.
   If |old_value - new_value| < threshold (suggest 0.03), skip the update and log
   "Recalibration delta too small to apply ({delta:.4f} < {threshold})".
   This prevents flip-flopping between 0.97 and 1.0 on noisy signals.
   Add 2 unit tests: one where delta is below threshold (no update), one above (update applies).

3. Run pytest tests/ -q --tb=short — must pass (427+).
4. Update HANDOFF.md to EMAC-031.
```
