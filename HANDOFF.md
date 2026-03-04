# 🦅 OPERATIONAL HANDOFF (EMAC-022)

> Ground truth as of EMAC-022. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-021 — Level 5 Wiring Peer Review + ReanalysisEngine Test Coverage

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-11: API Deduplication | ✅ | `/api/predictions/today` deduplicates by `game_id`. nightly > opener priority. (EMAC-020) |
| A-12: UI Dedup Guard | ✅ | `1_Todays_Bets.py` last-resort client-side dedup guard. (EMAC-020) |
| G-5: ReanalysisEngine Wiring | ✅ | `analysis.py` returns `(summary, cache)` tuple. `main.py` passes cache to `OddsMonitor` via `set_reanalysis_cache()`. Startup pre-warm reconstructs engines from today's DB predictions. |
| A-15: base_sd_override Bug Fix | ✅ | **Critical fix.** `CachedGameContext` now stores `base_sd_override`. `reanalyze()` uses cached SD when `new_total=None`. `from_analysis_pass()` accepts and stores `base_sd_override`. `analysis.py` passes `dynamic_base_sd`. Unchanged-spread invariant now holds. |
| A-16: ReanalysisEngine Tests | ✅ | 4 unit tests in `TestReanalysisEngine`: unchanged-spread invariant, bet_side flip on 22pt shift, new_total updates SD, factory snapshots all context fields. |
| Full test suite | ✅ | **412/412 passing** (up from 408). |
| G-3 Railway Env Vars | ⚠️ | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. LEVEL 5 WIRING — PEER REVIEW FINDINGS

**Reviewed:** `analysis.py`, `betting_model.py`, `odds_monitor.py`, `main.py`

**APPROVED with one critical fix (A-15):**

| Finding | Severity | Resolution |
|---------|----------|-----------|
| `_game_key` format consistent: `f"{away}@{home}"` in both analysis.py and main.py startup pre-warm | ✅ Clean | No action needed |
| `game_input` is the correct variable passed to both `analyze_game()` and `from_analysis_pass()` | ✅ Clean | No action needed |
| `@dataclass` decorator confirmed on `CachedGameContext` | ✅ Clean | No action needed |
| `base_sd_override=None` in `reanalyze()` when `new_total=None` — model fell back to `self.base_sd=11.0` instead of the dynamic SD from the original analysis | **CRITICAL BUG** | Fixed: store `base_sd_override` in `CachedGameContext`; thread through `from_analysis_pass()` and `analysis.py` |
| `set_reanalysis_cache()` in `odds_monitor.py` correctly replaces the instance cache dict | ✅ Clean | No action needed |

**The base_sd_override bug** would have caused `reanalyze(same_spread)` to return a DIFFERENT verdict than `analyze_game()` in any game where a game total was available (all real games). Fixed before tests were written.

---

### 3. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Task G-3 — Railway Env Vars (STILL PENDING USER ACTION)**
- Add to Railway Dashboard: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200.

**Task G-6 — OddsMonitor Real-Time Callback (Level 5 Completion)**
- The `_reanalysis_cache` is now wired into the OddsMonitor. Complete the loop:
  1. In `odds_monitor.py`, when a significant line movement is detected in `poll()`, look up the engine: `engine = self._reanalysis_cache.get(game_key)`.
  2. If found, call `updated = engine.reanalyze(new_spread=new_line.spread)`.
  3. If `updated.verdict.startswith("Bet")` and the original prediction was `"PASS"`, fire the movement callback with a `"VERDICT_FLIP"` event type.
  4. Log: `"Real-Time Pulse: %s flipped PASS→BET at new spread %.1f (edge=%.3f)", game_key, new_spread, updated.edge_conservative`
  5. Test in `tests/test_odds_monitor.py`.
- This is EMAC-023 scope.

---

### 4. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**Task O-5 — Async Verification (outstanding since EMAC-017)**
- Call `POST /admin/run-analysis` with `X-API-Key` header.
- Paste the exact log lines showing `"Triggering concurrent integrity sweep for N candidates..."` with timestamps.
- Verbal claim without log evidence will be rejected.

---

### 5. ARCHITECT REVIEW QUEUE (Next EMAC)

- **Level 5 completion**: After Gemini completes G-6 (OddsMonitor VERDICT_FLIP callback), peer-review the callback logic and add test coverage.
- **SNR re-audit**: Trigger when n >= 20 alpha bets in DB. Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.
- **Startup pre-warm edge case**: The lifespan pre-warm in `main.py` uses `inputs.get("game_data", inputs.get("game", {}))` to reconstruct `game_data`. Verify this correctly extracts `home_team`/`away_team` for predictions stored by the new code path (which uses `game_input` not `game_data` as the key).

---

### 6. HIVE WISDOM (Operational Lessons)

| Lesson | Source |
|--------|--------|
| Verified `/admin/scheduler/status` tracks the new Performance Sentinel job. | EMAC-017 |
| CLI env var setting is brittle in restricted shells; Railway Dashboard UI is source of truth for secrets. | EMAC-017 |
| `GameAnalysis.notes` populated for all game runs — full LLM pipeline confirmed. | EMAC-018 |
| OpenClaw verbal claims without log evidence are unacceptable for O-5 class tasks. | EMAC-017/018 |
| n=50 in directive was aspirational. DB had only 3 resolved BETs. Always verify DB count via audit script before acting on sample-size claims. | EMAC-019 |
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id` which is not unique per-render. | EMAC-019 |
| Inflated bet count (28 on 77-game slate) was run_tier deduplication failure, not model logic. Fix at API + UI layers. | EMAC-020 |
| When writing `reanalyze()`, always store the original `base_sd_override` in context. `None` does not mean "same as original" — it means "use model default". Fail to store it = broken unchanged-spread invariant. | EMAC-021 |

---

### 7. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-022 — Railway Env Vars + OddsMonitor VERDICT_FLIP Callback
You are the DevOps Strike Lead (Gemini CLI).
Read HANDOFF.md Sections 3 and 7 for your directives (Tasks G-3, G-6).

Execute in order:
1. Complete G-3: Set SNR_KELLY_FLOOR=0.5, INTEGRITY_CAUTION_SCALAR=0.75, INTEGRITY_VOLATILE_SCALAR=0.5 in Railway Dashboard. Redeploy + verify GET /health returns 200.
2. Begin G-6: In backend/services/odds_monitor.py, inside poll() where significant movements are logged, add:
   engine = self._reanalysis_cache.get(game_key)
   if engine:
       updated = engine.reanalyze(new_spread=new_line_spread)
       if updated.verdict.startswith("Bet") and original_verdict == "PASS":
           fire callback with event_type="VERDICT_FLIP", game_key=game_key, new_analysis=updated
   Log: logger.info("Real-Time Pulse: %s flipped PASS->BET at spread %.1f (edge=%.3f)", ...)
3. Add test in tests/test_odds_monitor.py covering the VERDICT_FLIP path.
4. Update HANDOFF.md to EMAC-023 when complete.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-022 — O-5 Async Sweep Live Log Evidence (final)
You are the Integrity Execution Unit (OpenClaw).
Read HANDOFF.md Section 4 for your directive.

Execute:
1. Call POST /admin/run-analysis with X-API-Key header.
2. Capture the ACTUAL log output. Paste the exact line containing "integrity sweep" with timestamp.
3. Report "ASYNC CONFIRMED: [exact log line]" or "SYNC -- needs fix: [reason]".
4. Update HEARTBEAT.md integrity_sweep.last_verified with today's date.
```

#### PROMPT FOR CLAUDE CODE
```
MISSION: EMAC-023 — Level 5 VERDICT_FLIP Peer Review + Startup Pre-Warm Audit
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 5 for your review queue.

When Gemini completes G-6 (VERDICT_FLIP callback in odds_monitor.py):
1. Peer-review the callback logic — verify game_key lookup is consistent with _reanalysis_cache keys.
2. Verify the VERDICT_FLIP only fires once per game per session (no repeated alerts on same line).
3. Audit the lifespan pre-warm in main.py: confirm inputs.get("game_data", inputs.get("game", {})) correctly extracts home_team/away_team for predictions stored by current analysis.py (which uses game_input dict).
4. Run pytest tests/ -q --tb=short — must pass.
5. Update HANDOFF.md to EMAC-024.
```
