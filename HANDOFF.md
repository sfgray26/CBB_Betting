# OPERATIONAL HANDOFF (EMAC-026)

> Ground truth as of EMAC-026. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-025 — VERDICT_FLIP Discord Peer Review + G-7 Close-Out

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-11: API Deduplication | ✅ | `/api/predictions/today` deduplicates by `game_id`. nightly > opener priority. (EMAC-020) |
| A-12: UI Dedup Guard | ✅ | `1_Todays_Bets.py` last-resort client-side dedup guard. (EMAC-020) |
| G-5: ReanalysisEngine Wiring | ✅ | `analysis.py` returns `(summary, cache)` tuple. `main.py` passes cache to `OddsMonitor`. (EMAC-021) |
| A-15: base_sd_override Invariant | ✅ | `CachedGameContext` stores dynamic SD. Unchanged-spread invariant holds. (EMAC-021) |
| A-16: ReanalysisEngine Tests | ✅ | 4 unit tests in `TestReanalysisEngine`. (EMAC-021) |
| G-6: VERDICT_FLIP Callback | ✅ | `OddsMonitor.poll()` detects true PASS->BET flips on significant spread moves. (EMAC-022/023) |
| A-17: VERDICT_FLIP Audit Fixes | ✅ | 4 bugs fixed (see below). `_verdict_flip_fired` one-fire set. `original_verdict` in context. Pre-warm reconstructed correctly. |
| G-7: VERDICT_FLIP Discord Alert | ✅ | `_verdict_flip_handler` in lifespan. `send_verdict_flip_alert()` hardened. (EMAC-025) |
| A-18: Discord Alert Hardening | ✅ | 4 crash paths fixed in `send_verdict_flip_alert()`. (EMAC-025) |
| Full test suite | ✅ | **413/413 passing.** |
| G-3 Railway Env Vars | ⚠️ | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. EMAC-025 PEER REVIEW FINDINGS — 4 CRASH PATHS FIXED

**Reviewed:** `discord_notifier.py::send_verdict_flip_alert()`, `main.py` lifespan callback registration

**Callback Ordering:** PASS — handler registered at line 283, `set_reanalysis_cache()` at line 266. Handler goes in after cache is set. No ordering bug.

**Callback Accumulation:** PASS — `_verdict_flip_handler` is defined and registered in `lifespan()`, not inside `nightly_job()`. Runs once at startup, not once per nightly run. No duplicate accumulation.

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| B-5 | `f"{edge:.1%}"` with `edge=None` → TypeError | **Critical** | Added `or 0.0` guard: `edge = analysis.edge_conservative or 0.0` |
| B-6 | `f"{units:.2f}u"` with `units=None` → TypeError | **Critical** | Added `or 0.0` guard: `units = analysis.recommended_units or 0.0` |
| B-7 | `f"{movement.old_value:+.1f}"` with `old_value=None` → TypeError | **Critical** | Conditional: `old_str = f"{movement.old_value:+.1f}" if movement.old_value is not None else "N/A"` |
| B-8 | Fragile `.split('u ')` pick string parsing — breaks on any non-standard verdict | **High** | Replaced with calcs-based resolution: `bet_side = calcs.get("bet_side", "home")` from `analysis.full_analysis["calculations"]` |

---

### 3. V9 SYSTEM STATUS — FULLY OPERATIONAL

All V9 components correctly wired and tested:

```
+----------------------------------------------------------+
|  V9 OPERATIONAL STACK                                    |
+------------------+---------------------------------------+
| Layer            | Status                                |
+------------------+---------------------------------------+
| Model (v9.0)     | OK  SNR + Integrity Kelly scalars     |
| Analysis         | OK  Returns (summary, cache) tuple    |
| ReanalysisEngine | OK  Unchanged-spread invariant holds  |
| OddsMonitor      | OK  VERDICT_FLIP with one-fire dedup  |
| Startup Pre-Warm | OK  Correct game_data + base_sd       |
| API Dedup        | OK  nightly > opener priority         |
| UI Dedup         | OK  Last-resort guard in Todays_Bets  |
| Discord Alerts   | OK  send_verdict_flip_alert hardened  |
| Callback Wiring  | OK  lifespan, no accumulation risk    |
| Test suite       | OK  413/413 passing                   |
+------------------+---------------------------------------+
```

**Outstanding:** Railway env vars (G-3) — requires user action, not code.

---

### 4. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (FINAL REMINDER — USER ACTION)**
- Add to Railway Dashboard: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200.
- Without these, Railway runs with Python defaults. Not a code change — dashboard action only.

**No new Gemini tasks.** G-7 is closed. V9 stack is fully operational.

---

### 5. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**Task O-5 — Async Verification (outstanding since EMAC-017)**
- Call `POST /admin/run-analysis` with `X-API-Key` header.
- Paste the exact log lines showing `"Triggering concurrent integrity sweep for N candidates..."` with timestamps.
- Verbal claim without log evidence will be rejected.

---

### 6. ARCHITECT REVIEW QUEUE (Next EMAC)

- **SNR re-audit**: Trigger when n >= 20 alpha bets in DB. Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.
- **Season-end calibration**: At end of season, run full recalibration with actual vs projected margins to re-tune sd_multiplier and home_advantage. Script: `scripts/generate_real_insights.py`.
- **Scout hardening**: `backend/services/scout.py` uses free functions (not singleton). Gemini can harden if Ollama proves unreliable in prod.
- **V9 live verification**: After Railway env vars are set, trigger a manual analysis and confirm `snr_kelly_scalar` and `integrity_kelly_scalar` appear in DB `full_analysis` JSON for at least one prediction.

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
| One-fire sets must be cleared on cache refresh — otherwise a new nightly run can't trigger new flips. | EMAC-023 |
| Discord embed fields using `:.1%` or `:.2f` on None fields crash silently in prod. Always guard with `or 0.0`. | EMAC-025 |
| Never parse verdict strings to extract pick info — use `full_analysis["calculations"]["bet_side"]` directly. | EMAC-025 |
| Callback accumulation risk: register Discord/alert handlers in lifespan, not in nightly_job. | EMAC-025 |

---

### 8. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-026 — Railway Env Vars (G-3 Final)
You are the DevOps Strike Lead (Gemini CLI).

G-7 is closed. V9 Discord alerting is fully operational (413/413 tests passing).
Your only outstanding task is G-3 (user action in Railway Dashboard).

Action required (NOT a code change):
1. Go to Railway Dashboard for the CBB Edge project.
2. Add environment variables: SNR_KELLY_FLOOR=0.5, INTEGRITY_CAUTION_SCALAR=0.75, INTEGRITY_VOLATILE_SCALAR=0.5
3. Redeploy the service.
4. Verify GET /health returns 200.
5. Report back with health check response body.

If already done, verify by checking GET /health response and confirm the vars appear in Railway's variable list.
Update HEARTBEAT.md with the verification date.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-026 — O-5 Async Sweep Live Log Evidence (final)
You are the Integrity Execution Unit (OpenClaw).
Read HANDOFF.md Section 5 for your directive.

Execute:
1. Call POST /admin/run-analysis with X-API-Key header.
2. Capture the ACTUAL log output. Paste the exact line containing "integrity sweep" with timestamp.
3. Report "ASYNC CONFIRMED: [exact log line]" or "SYNC -- needs fix: [reason]".
4. Update HEARTBEAT.md integrity_sweep.last_verified with today's date.
```

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-027 — SNR Re-Audit + Season Calibration Prep
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 6 for your review queue.

Tasks:
1. SNR re-audit: Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.
   If n < 20 alpha bets, document current count and defer.
   If n >= 20, analyze output and flag any SNR floor bypass patterns.
2. Season calibration prep: Review `scripts/generate_real_insights.py` and document
   what data is needed from the DB to run end-of-season recalibration.
   Identify which tables/fields feed sd_multiplier and home_advantage re-tuning.
3. V9 live verification: If Railway env vars (G-3) have been set, query the DB for
   the most recent prediction and verify `full_analysis["calculations"]` contains
   `snr`, `snr_kelly_scalar`, `integrity_verdict`, `integrity_kelly_scalar` fields.
4. Run pytest tests/ -q --tb=short — must pass.
5. Update HANDOFF.md to EMAC-028.
```
