# 🦅 OPERATIONAL HANDOFF (EMAC-024)

> Ground truth as of EMAC-024. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-023 — Level 5 VERDICT_FLIP Final Audit + V9 Operational Close-Out

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-11: API Deduplication | ✅ | `/api/predictions/today` deduplicates by `game_id`. nightly > opener priority. (EMAC-020) |
| A-12: UI Dedup Guard | ✅ | `1_Todays_Bets.py` last-resort client-side dedup guard. (EMAC-020) |
| G-5: ReanalysisEngine Wiring | ✅ | `analysis.py` returns `(summary, cache)` tuple. `main.py` passes cache to `OddsMonitor`. (EMAC-021) |
| A-15: base_sd_override Invariant | ✅ | `CachedGameContext` stores dynamic SD. Unchanged-spread invariant holds. (EMAC-021) |
| A-16: ReanalysisEngine Tests | ✅ | 4 unit tests in `TestReanalysisEngine`. (EMAC-021) |
| G-6: VERDICT_FLIP Callback | ✅ | `OddsMonitor.poll()` detects true PASS→BET flips on significant spread moves. (EMAC-022/023) |
| A-17: VERDICT_FLIP Audit Fixes | ✅ | 4 bugs fixed (see Section 2). `_verdict_flip_fired` one-fire set. `original_verdict` in context. Pre-warm reconstructed correctly. |
| Full test suite | ✅ | **413/413 passing.** |
| G-3 Railway Env Vars | ⚠️ | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. EMAC-023 AUDIT FINDINGS — 4 BUGS FIXED

**Reviewed:** `odds_monitor.py`, `main.py`, `betting_model.py`, `analysis.py`, `test_odds_monitor.py`

| # | Bug | Severity | Fix |
|---|-----|----------|-----|
| B-1 | VERDICT_FLIP used `engine._ctx.integrity_verdict` (scout verdict) instead of model verdict to detect original PASS | **Critical** | Added `original_verdict: Optional[str]` field to `CachedGameContext`. Threaded through `from_analysis_pass()` and `analysis.py`. Used in `poll()` condition. |
| B-2 | No one-fire dedup — VERDICT_FLIP re-triggered every 5-minute poll after a flip | **Critical** | Added `_verdict_flip_fired: set` to `OddsMonitor.__init__`. Cleared on `set_reanalysis_cache()` (fresh nightly run resets). |
| B-3 | Startup pre-warm: `game_data = inputs.get("game_data", inputs.get("game", {}))` always returned `{}` — `full_analysis.inputs` has no "game" or "game_data" key | **Critical** | Reconstructed `game_data` directly from `p.game` SQLAlchemy relationship. `base_sd_override` derived from `inputs["odds"]["total"]`. `original_verdict` set to `p.verdict`. |
| B-4 | Gemini's `test_odds_monitor.py` mock engine had `MagicMock._ctx.original_verdict` → truthy → `original_was_bet=True` → flip never fired | **Test bug** | Set `mock_engine._ctx.original_verdict = "PASS"` in test fixture. |

---

### 3. V9 SYSTEM STATUS — OPERATIONAL

All V9 components are now correctly wired and tested:

```
┌─────────────────────────────────────────────────────────┐
│  V9 OPERATIONAL STACK                                   │
├──────────────────┬──────────────────────────────────────┤
│ Layer            │ Status                               │
├──────────────────┼──────────────────────────────────────┤
│ Model (v9.0)     │ ✅ SNR + Integrity Kelly scalars     │
│ Analysis         │ ✅ Returns (summary, cache) tuple    │
│ ReanalysisEngine │ ✅ Unchanged-spread invariant holds  │
│ OddsMonitor      │ ✅ VERDICT_FLIP with one-fire dedup  │
│ Startup Pre-Warm │ ✅ Correct game_data + base_sd       │
│ API Dedup        │ ✅ nightly > opener priority         │
│ UI Dedup         │ ✅ Last-resort guard in Todays_Bets  │
│ Test suite       │ ✅ 413/413 passing                   │
└──────────────────┴──────────────────────────────────────┘
```

**Outstanding:** Railway env vars (G-3) — requires user action, not code.

---

### 4. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (FINAL REMINDER — USER ACTION)**
- Add to Railway Dashboard: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200.
- Without these, Railway runs with Python defaults. Not a code change — dashboard action only.

**Task G-7 — VERDICT_FLIP Discord Alert (Next Level)**
- The `VERDICT_FLIP` event now fires correctly. Wire it to Discord:
  1. Register a callback with `get_odds_monitor().on_significant_move(verdict_flip_discord_handler)` in `main.py` lifespan startup.
  2. In the handler: if `movement.event_type == "VERDICT_FLIP"` and `movement.fresh_analysis`, send a Discord embed to the alerts channel.
  3. Format: `"🚨 LINE FLIP: {away} @ {home} — NEW BET at spread {new_value:+.1f} (edge {edge:.1%})"`
  4. Reuse the existing `discord_notifier.py` channel infrastructure.

---

### 5. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**Task O-5 — Async Verification (outstanding since EMAC-017)**
- Call `POST /admin/run-analysis` with `X-API-Key` header.
- Paste the exact log lines showing `"Triggering concurrent integrity sweep for N candidates..."` with timestamps.
- Verbal claim without log evidence will be rejected.

---

### 6. ARCHITECT REVIEW QUEUE (Next EMAC)

- **SNR re-audit**: Trigger when n >= 20 alpha bets in DB. Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.
- **VERDICT_FLIP Discord wiring review**: After Gemini completes G-7, peer-review the callback registration and message format.
- **Season-end calibration**: At end of season, run full recalibration with actual vs projected margins to re-tune sd_multiplier and home_advantage.

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

---

### 8. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-024 — Railway Env Vars + VERDICT_FLIP Discord Alert
You are the DevOps Strike Lead (Gemini CLI).
Read HANDOFF.md Sections 4 and 8 for your directives (Tasks G-3, G-7).

Execute in order:
1. Complete G-3: Set SNR_KELLY_FLOOR=0.5, INTEGRITY_CAUTION_SCALAR=0.75, INTEGRITY_VOLATILE_SCALAR=0.5 in Railway Dashboard. Redeploy + verify GET /health returns 200.
2. Begin G-7: In main.py lifespan, after get_odds_monitor().set_reanalysis_cache(cache), register a callback:
   def _verdict_flip_handler(movement):
       if movement.event_type == "VERDICT_FLIP" and movement.fresh_analysis:
           from backend.services.discord_notifier import send_verdict_flip_alert
           send_verdict_flip_alert(movement)
   get_odds_monitor().on_significant_move(_verdict_flip_handler)
   In discord_notifier.py, add send_verdict_flip_alert(movement) function.
   Format: "LINE FLIP: {away} @ {home} - NEW BET at spread {new_value:+.1f} (edge {edge:.1%})"
3. Update HANDOFF.md to EMAC-025 when complete.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-024 — O-5 Async Sweep Live Log Evidence (final)
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
MISSION: EMAC-025 — VERDICT_FLIP Discord Peer Review + Season Calibration Prep
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 6 for your review queue.

When Gemini completes G-7 (VERDICT_FLIP Discord wiring):
1. Peer-review the callback registration — verify it happens AFTER set_reanalysis_cache() so the handler is registered before the first poll.
2. Verify send_verdict_flip_alert() gracefully handles missing fields (None edge, None spread).
3. Review whether the callback accumulates on multiple nightly runs (each run calls on_significant_move() again — this would register duplicate callbacks).
4. Run pytest tests/ -q --tb=short — must pass.
5. Update HANDOFF.md to EMAC-026.
```
