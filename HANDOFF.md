# OPERATIONAL HANDOFF (EMAC-045)

> Ground truth as of EMAC-045. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-045 — BallDontLie API contract corrected (endpoint, field, season offset). Pre-tournament assessment written. 464/464 tests passing.

| Component | Status | Detail |
|-----------|--------|--------|
| Railway API | OK | Live, healthy. All syntax errors resolved. |
| V9 Model | OK | SNR + Integrity + Seed-Spread Kelly scalars. `model_version='v9.0'` |
| Env Var Parsing | OK | `get_float_env` on all float reads. Zero plain `float(os.getenv)`. |
| CI Syntax Guard | OK | `py_compile` step in `deploy.yml` before pytest. |
| Analysis Pipeline | OK | Nightly running. 0 bets = correct conservatism (K-3). V9 recal at 50 bets. |
| Seed-Spread Scalars (A-26 T2) | LIVE — INACTIVE | Code correct. No-op until `BALLDONTLIE_API_KEY` set in Railway (G-13). |
| Tournament SD Bump (A-26 T1) | READY | 1.15x when `is_neutral=True`. Active for neutral-site games. |
| Integrity Sweep | LIVE | Async, 8-worker concurrent. 0 BET games since V9 launch — correct. |
| O-6 Integrity Spot-Check | OPEN | OpenClaw to verify `integrity_verdict` in prod predictions. |
| O-9 Tiered Escalation | UNWIRED | Claude to implement before March 18. |
| O-8 Pre-Tournament Baseline | PENDING | Kimi designs, OpenClaw executes March 16 ~9 PM ET. |
| Calibration | OK | ha=2.419, sd_mult=1.0 (V8-era). V9 recal after 50 settled V9 bets. |
| Gemini Trust Level | RESTORING | G-11 clean. Single-file tasks only until 2 more clean executions. |

---

## 2. TEAM ROLES

| Agent | Role | Trust | Current Focus |
|-------|------|-------|---------------|
| Claude Code | Master Architect | FULL | O-9 tiered escalation (before March 18) |
| Gemini CLI | DevOps Strike Lead | RESTORING | G-13: BALLDONTLIE_API_KEY in Railway |
| Kimi CLI | Deep Intelligence + **OpenClaw Config Owner** | FULL | K-6: O-8 baseline script design |
| OpenClaw | Integrity Execution | FULL | O-6 spot-check |

**Gemini rule:** No multi-file Python refactors. Single file or non-Python only. `py_compile` + 464 tests before every push.

**OpenClaw ownership:** Kimi owns OpenClaw configuration, prompt tuning, and optimization. OpenClaw executes what Kimi designs.

---

## 3. COMPLETED WORK

| Mission | Who | What |
|---------|-----|------|
| A-26 T2 | Claude | Seed-spread Kelly scalars. 26 new tests. API contract fixed (endpoint, field, season offset). |
| A-26 T1 | Claude | Tournament SD bump 1.15x when is_neutral=True. |
| A-25 | Claude | Neutral-site fix in parse_odds_for_game. |
| K-4 | Kimi | A-26 T2 architecture spec. |
| K-3 | Kimi | Model quality audit: 0 bets = correct conservatism. |
| K-2 | Kimi | Seed data research: BallDontLie API. |
| K-1 | Kimi | Tournament intelligence: SD bump 1.15x. |
| G-12 | Claude | py_compile step in deploy.yml. |
| G-11 | Gemini | Railway env var audit: cleaned =VALUE vars. |
| EMAC-037–045 | Claude | get_float_env on all float reads; all Railway syntax errors fixed. |

---

## 4. ACTIVE MISSIONS

---

### CLAUDE CODE — O-9: Wire Tiered Escalation [CRITICAL — before March 18]

Create `backend/services/coordinator.py` with an `escalate_if_needed(game, verdict)` function. Wire into `analysis.py` after `_integrity_sweep()`. Logging-only for now (no Kimi API call required until post-March 18).

**Escalation triggers:** `units >= 1.5` · `tournament_round >= 4` · `"VOLATILE" in verdict`

Add tests to `tests/test_coordinator.py`. 464 must still pass. Advance HANDOFF to EMAC-046.

---

### GEMINI CLI — G-13: Add BALLDONTLIE_API_KEY to Railway [HIGH — before March 16]

Railway Variables tab → add `BALLDONTLIE_API_KEY` with the GOAT-tier key. No Python changes.

Verify: trigger manual analysis, check logs for `"BallDontLie bracket request: season=2025"`. Update HANDOFF.md G-13 to COMPLETE. Advance title to EMAC-046.

---

### KIMI CLI — K-6: Design O-8 Pre-Tournament Baseline [HIGH — before March 16]

Design `scripts/openclaw_baseline.py` for OpenClaw execution March 16 ~9 PM ET. 68 teams, DDGS + qwen2.5:3b, JSON output (`team → {seed, region, risk_level, summary}`).

Read `backend/services/scout.py` and `reports/openclaw-capabilities-assessment.md` first. Save spec to `reports/k6-o8-baseline-spec.md`. Update HANDOFF.md K-6 to COMPLETE. Advance title to EMAC-046.

---

### OPENCLAW — O-6: Integrity Spot-Check [MEDIUM — run now]

`GET /api/predictions/today`. Check if `integrity_verdict` is populated. Expected: all null.

Report: `O-6: Not triggered — correct` or `O-6: BROKEN — escalate to Kimi`. Update HEARTBEAT.md status tracker. Update HANDOFF.md O-6 to COMPLETE. Advance title to EMAC-046.

---

## 5. DEPENDENCY CHAIN

```
G-13 (Gemini) --> BALLDONTLIE_API_KEY set --> seed scalars activate March 16
K-6 (Kimi)    --> O-8 baseline spec ready --> OpenClaw executes March 16 ~9 PM ET
O-6 (OpenClaw)--> integrity spot-check --> HEARTBEAT updated
O-9 (Claude)  --> tiered escalation wired --> must be live before March 18

Bracket March 16 6 PM ET
  --> BallDontLie data ~8 PM ET
  --> Seed scalars active in analysis
  --> O-8 baseline batch (OpenClaw)
  --> March 18: First Four begins
```

---

## 6. PRE-TOURNAMENT CHECKLIST

| # | Item | Status | Owner |
|---|------|--------|-------|
| 1 | System Assessment | ✅ | Claude |
| 2 | Team Readiness Confirmed | ✅ | Claude |
| 3 | A-26 T2 Implemented + API Corrected | ✅ | Claude |
| 4 | BALLDONTLIE_API_KEY in Railway | ⬜ | Gemini (G-13) |
| 5 | Seed-Spread Scalar Defaults Verified | ✅ | Claude |
| 6 | O-9 Tiered Escalation Wired | ⬜ | Claude |
| 7 | O-8 Baseline Script Ready | ⬜ | Kimi design / OpenClaw exec |
| 8 | O-10 Line Movement Monitor | ⬜ | Gemini + OpenClaw |
| 9 | Railway Deploy Pipeline Tested | ⬜ | Gemini |
| 10 | Rollback Plan Documented | ✅ | Claude |

---

## 7. ARCHITECT REVIEW QUEUE

**Pre-March 18:**
- O-9 implementation (coordinator.py + analysis.py wiring)
- Review K-6 spec before OpenClaw executes O-8

**March 16–18:**
- Verify seed scalars fire in logs after bracket loads
- Review O-8 baseline output, flag HIGH-risk teams
- Deploy O-10 line movement monitor (Gemini leads)

**Ongoing:**
- A-27: Weekly calibration review — MAE drift, document in memory/
- A-28: MIN_BET_EDGE 2.0% experiment via Railway env var
- V9 recalibration at 50 settled V9 bets

**Post-tournament:**
- SNR re-audit A-19, season-end recalibration, OpenClaw classifier (Kimi leads)

---

## 8. HIVE WISDOM

| Lesson | Source |
|--------|--------|
| `pred_id` is the correct Streamlit widget key — never `game_id`. | EMAC-019 |
| `full_analysis.inputs` has no "game" key. Reconstruct from `p.game` relationship. | EMAC-023 |
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. | EMAC-027 |
| `sd_multiplier` oscillates at noise boundary. Min-delta guard (0.03) prevents flip-flopping. | EMAC-031 |
| Use Kimi for tasks requiring >50K tokens simultaneously. | EMAC-034 |
| Tournament SD bump applies AFTER all other SD penalties. Order matters. | EMAC-036 |
| Gemini large-scale refactors drop closing parens. Always `py_compile` every .py before commit. | EMAC-038 |
| `get_float_env` must be used for ALL float env reads. Any new float env var MUST use it. | EMAC-042 |
| `main.py` syntax errors bypass `pytest` (no DB). Always `py_compile` after changes. | EMAC-039 |
| Uncommitted local changes are invisible to Railway. Verify push before blaming the fix. | EMAC-042 |
| BallDontLie: endpoint `/bracket`, field `name` (not `full_name`), `season=year-1`. | EMAC-045 |
| `sd_mult=1.0` widens distribution, compresses edges. V9 recal after 50+ V9 bets settle. | K-3 |

---

## 9. HANDOFF PROMPTS

### CLAUDE CODE
```
MISSION: EMAC-045 — O-9 Tiered Escalation (critical before March 18)
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge

STATE: 464/464 tests. Railway live. A-26 T2 deployed (inactive until BALLDONTLIE_API_KEY set).
TASK: Create backend/services/coordinator.py. Implement escalate_if_needed(game, verdict).
Wire into analysis.py after _integrity_sweep(). Logging-only — no Kimi API call required yet.
Escalation triggers: units>=1.5, tournament_round>=4, "VOLATILE" in verdict.
Tests: add tests/test_coordinator.py. Run pytest tests/ -q (464 must pass).
py_compile all touched files. Commit. Update HANDOFF.md O-9 to COMPLETE. Advance to EMAC-046.

GUARDIAN: py_compile + 464 tests before approving any Gemini commit.
```

### GEMINI CLI
```
MISSION: G-13 — Add BALLDONTLIE_API_KEY to Railway Variables
Read HANDOFF.md Section 4 G-13 for exact steps. Railway UI only — no Python changes.
After setting: trigger manual analysis, verify logs show "BallDontLie bracket request: season=2025".
Update HANDOFF.md G-13 row to COMPLETE. Advance title to EMAC-046.
SCOPE RULE: one Railway env var add. Do not expand scope.
```

### KIMI CLI
```
MISSION: K-6 — Design O-8 Pre-Tournament Baseline Script
You are Deep Intelligence Unit AND OpenClaw Config Owner for CBB Edge Analyzer.
Read: backend/services/scout.py, reports/openclaw-capabilities-assessment.md, HEARTBEAT.md.
Design scripts/openclaw_baseline.py for OpenClaw to execute March 16 ~9 PM ET.
Output: 68-team JSON map (team -> seed, region, risk_level, summary). Use DDGS + qwen2.5:3b.
Save spec to reports/k6-o8-baseline-spec.md.
Update HANDOFF.md K-6 to COMPLETE. Advance title to EMAC-046.
```

### OPENCLAW
```
MISSION: O-6 — V9 Integrity Spot-Check
GET https://cbbbetting-production.up.railway.app/api/predictions/today
Header: X-API-Key: <your key>
Check: is integrity_verdict populated in any prediction?
Expected: all null (0 BET-tier games = sweep not triggered = correct).
Report: "O-6: Not triggered — correct" or "O-6: BROKEN — escalate to Kimi".
Update HEARTBEAT.md status tracker row for O-6.
Update HANDOFF.md O-6 row to COMPLETE. Advance title to EMAC-046.
```
