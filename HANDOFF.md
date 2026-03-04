# 🦅 OPERATIONAL HANDOFF (EMAC-020)

> Ground truth as of EMAC-020. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

### 1. MISSION INTEL (Ground Truth)

**Operator:** Claude Code (Master Architect)
**Mission accomplished:** EMAC-019 — SNR Calibration Audit, ReanalysisEngine Design, Circuit Breaker Audit, Peer Review.

**Technical State (cumulative):**

| Component | Status | Detail |
|-----------|--------|--------|
| A-8: SNR Calibration Audit | ✅ | n=3 resolved BET predictions (not 50 as anticipated). 0/3 cover rate — statistically meaningless. `SNR_KELLY_FLOOR=0.5` maintained. Re-audit when n >= 20 per tier. |
| A-9: ReanalysisEngine | ✅ | `CachedGameContext` + `ReanalysisEngine` added to `betting_model.py`. Exposes `reanalyze(new_spread, new_total)` + factory `from_analysis_pass()`. Ready for Level 5 wiring. |
| A-10: Circuit Breaker Audit | ✅ | Single integrity abort path at `betting_model.py:1608`, already prefixed `INTEGRITY_ABORT:`. All other reject paths use distinct prefixes. No changes needed. |
| Peer Review: pred_id fix | ✅ | **APPROVED.** `pred_id` (Prediction PK) is the correct unique key. Full assessment in Section 2. |
| Full test suite | ✅ | **408/408 passing** after ReanalysisEngine addition. |
| G-3 Railway Env Vars | ⚠️ | **USER ACTION REQUIRED.** Set `SNR_KELLY_FLOOR`, `INTEGRITY_CAUTION_SCALAR`, `INTEGRITY_VOLATILE_SCALAR` in Railway Dashboard. |

---

### 2. ARCHITECT PEER REVIEW — Gemini's pred_id Fix

**File reviewed:** `dashboard/pages/1_Todays_Bets.py`

**Change:** Widget keys derived from `pred_id` (Prediction PK) instead of `game_id`.

**Assessment: APPROVED — correct fix, correct data model alignment.**

Reasoning:
- `game_id` is NOT unique per page render on busy slates — multiple predictions can exist for the same game (re-runs, test data). `pred_id` is the Prediction PK and is guaranteed unique.
- Streamlit requires unique widget keys within a page. Using `pred_id` eliminates the `DuplicateWidgetID` exception on multi-bet slates.
- The bet log `notes` field correctly preserves both IDs for traceability: `f"Logged from Today's Bets | pred_id={pred_id}"`.
- Session state keys (`log_key`, `logged_{pred_id}`, form + input keys) are all consistently `pred_id`-scoped.

**No follow-up action required.**

---

### 3. DELEGATION BUNDLE: Gemini CLI (DevOps Strike Lead)

**Objective:** Railway env vars + Level 5 Real-Time Pulse wiring.

**Task G-3 — Railway Env Vars (STILL PENDING USER ACTION)**
- Add to Railway Dashboard: `SNR_KELLY_FLOOR=0.5`, `INTEGRITY_CAUTION_SCALAR=0.75`, `INTEGRITY_VOLATILE_SCALAR=0.5`
- Redeploy + verify `GET /health` returns 200.

**Task G-5 — ReanalysisEngine Wiring (Level 5 Prep)**
- `ReanalysisEngine` + `CachedGameContext` are live in `betting_model.py`. Wire them in `analysis.py`:
  1. After the BET/CONSIDER model.analyze_game() call, create an engine and store it:
     ```python
     engine = ReanalysisEngine.from_analysis_pass(model, game_data, odds_input, ratings, ...)
     _reanalysis_cache[_game_key] = engine
     ```
  2. Return `_reanalysis_cache` from `run_nightly_analysis()` alongside the summary dict.
  3. Pass the cache to the `OddsMonitor` so its callback can call `engine.reanalyze(new_spread)`.
- This is EMAC-021 scope.

---

### 4. DELEGATION BUNDLE: OpenClaw (Integrity Execution Unit)

**Task O-5 — Async Verification (outstanding since EMAC-017)**
- Call `POST /admin/run-analysis` with `X-API-Key` header.
- Paste the exact log lines showing `"Triggering concurrent integrity sweep for N candidates..."` with timestamps.
- Verbal claim without log evidence will be rejected.

---

### 5. ARCHITECT REVIEW QUEUE (Next EMAC)

- **Level 5 wiring**: After Gemini completes G-5, peer-review `analysis.py` changes + add ReanalysisEngine tests.
- **SNR re-audit**: Trigger when n >= 20 alpha bets in DB. Run `python scripts/audit_confidence.py --days 90 --min-bets 20`.

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

---

### 7. HANDOFF PROMPTS — COPY AND PASTE THESE

#### PROMPT FOR GEMINI CLI
```
MISSION: EMAC-020 — Railway Env Vars + Level 5 ReanalysisEngine Wiring
You are the DevOps Strike Lead (Gemini CLI).
Read HANDOFF.md Sections 3 and 7 for your directives (Tasks G-3, G-5).

Execute in order:
1. Complete G-3: Set SNR_KELLY_FLOOR, INTEGRITY_CAUTION_SCALAR, INTEGRITY_VOLATILE_SCALAR in Railway Dashboard. Redeploy + verify GET /health returns 200.
2. Begin G-5: Wire ReanalysisEngine.from_analysis_pass() into the BET/CONSIDER loop in analysis.py. Store engines in _reanalysis_cache dict keyed by game_key. Return cache from run_nightly_analysis(). Pass to OddsMonitor callback.
3. Update HANDOFF.md to EMAC-021 when complete.
```

#### PROMPT FOR OPENCLAW
```
MISSION: EMAC-020 — O-5 Async Sweep Live Log Evidence (final)
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
MISSION: EMAC-021 — ReanalysisEngine Test Coverage + Level 5 Peer Review
You are Claude Code, Master Architect for CBB Edge Analyzer.
Read HANDOFF.md Section 5 for your review queue.

When Gemini completes G-5 (ReanalysisEngine wiring in analysis.py):
1. Peer-review the analysis.py changes — verify cache is keyed correctly and passed to OddsMonitor.
2. Add 4 tests to tests/test_betting_model.py:
   - reanalyze() with unchanged spread returns same verdict as analyze_game()
   - reanalyze() with spread moved past threshold flips PASS -> BET verdict
   - reanalyze() with new_total updates base_sd_override correctly
   - from_analysis_pass() factory correctly snapshots all context fields
3. Run pytest tests/ -q --tb=short — must pass.
4. Update HANDOFF.md to EMAC-022.
```
