# OPERATIONAL HANDOFF (EMAC-050)

> Ground truth as of EMAC-049. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-049 — BALLDONTLIE_API_KEY verified in Railway (G-13). A-27 calibration review complete. Parameters frozen.  
**OpenClaw Fix:** HANDOFF.md auto-update now working via `.openclaw/handoff_writer.py` — integrity sweep results and baseline completion will auto-update this file.

| Component | Status | Detail |
|-----------|--------|--------|
| Railway API | OK | Live, healthy. All syntax errors resolved. |
| V9 Model | OK | SNR + Integrity + Seed-Spread Kelly scalars. `model_version='v9.0'` |
| Env Var Parsing | OK | `get_float_env` on all float reads. Zero plain `float(os.getenv)`. |
| CI Syntax Guard | OK | `py_compile` step in `deploy.yml` before pytest. |
| Analysis Pipeline | OK | Nightly running. 0 bets = correct conservatism (K-3). V9 recal at 50 bets. |
| Seed-Spread Scalars (A-26 T2) | LIVE | Active. BALLDONTLIE_API_KEY verified in Railway. |
| Tournament SD Bump (A-26 T1) | READY | 1.15x when `is_neutral=True`. Active for neutral-site games. |
| Integrity Sweep | LIVE | Async, 8-worker concurrent. 0 BET games since V9 launch — correct. |
| O-6 Integrity Spot-Check | OPEN | OpenClaw to verify `integrity_verdict` in prod predictions. Note: OpenClaw HANDOFF.md integration now fixed — see `.openclaw/handoff_writer.py`. |
| O-9 Tiered Escalation | LIVE (logging) | coordinator.py created. Logs ESCALATION_FLAGGED on units>=1.5, neutral_site, VOLATILE. Kimi API routing post-March 18. |
| O-8 Pre-Tournament Baseline | READY | Script created. OpenClaw executes March 16 ~9 PM ET. Discord errors fixed in v2.1 — see TROUBLESHOOTING.md. |
| Calibration | OK | ha=2.419, sd_mult=1.0 (V8-era). V9 recal after 50 settled V9 bets. |
| Gemini Trust Level | RESTORING | G-13 clean. Single-file tasks only until 1 more clean execution. |

---

## 1.5. OPENCLAW HANDOFF.md INTEGRATION (NEW)

**Status:** ✅ FIXED — OpenClaw can now update HANDOFF.md autonomously

**How it works:**
- `.openclaw/handoff_writer.py` provides `update_openclaw_status()` function
- Called by `analysis.py` after each integrity sweep (line ~953)
- Called by `scripts/openclaw_baseline.py` after O-8 baseline completion
- Updates Section 1.5 (this section) with current status

**What gets updated:**
- Last integrity sweep timestamp and game count
- Circuit breaker state
- O-8 baseline completion status (when run)
- Active alerts (VOLATILE >20%, ABORT/RED FLAG, etc.)

**Next Task Note:** OpenClaw will auto-update this section. Do not manually edit Section 1.5 — let OpenClaw manage it.

---

## 2. TEAM ROLES

| Agent | Role | Trust | Current Focus |
|-------|------|-------|---------------|
| Claude Code | Master Architect | FULL | Monitoring — no active code missions. Unblock: O-10 architecture when Gemini ready. |
| Gemini CLI | DevOps Strike Lead | RESTORING | G-14: Railway Deploy Pipeline Tested |
| Kimi CLI | Deep Intelligence + **OpenClaw Config Owner** | FULL | K-6: O-8 baseline script design |
| OpenClaw | Integrity Execution | FULL | O-6 spot-check |

**Gemini rule:** No multi-file Python refactors. Single file or non-Python only. `py_compile` + 474 tests before every push.

**OpenClaw ownership:** Kimi owns OpenClaw configuration, prompt tuning, and optimization. OpenClaw executes what Kimi designs.

---

## 3. COMPLETED WORK

| Mission | Who | What |
|---------|-----|------|
| A-27 | Claude | Weekly calibration review. Params frozen. See memory/calibration.md. |
| G-13 | Gemini | BALLDONTLIE_API_KEY set in Railway. Verified logs "BallDontLie bracket request: season=2025". |
| O-9 | Claude | Tiered escalation coordinator. Logs ESCALATION_FLAGGED on units>=1.5, neutral_site, VOLATILE. 10 tests. |
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

### CLAUDE CODE — A-28: MIN_BET_EDGE 2.0% Experiment [MEDIUM]

Verify `MIN_BET_EDGE` env var is wired in `betting_model.py` and controls the 2% threshold.
If not wired, add it using `get_float_env("MIN_BET_EDGE", "2.0")`.
Document expected impact: raising from 2% → 2.5% would further reduce BET-tier volume (~15-20% reduction estimated).
No Railway changes needed — document how operator sets it.
Update `.env.example` if missing. Add 1-2 tests if the wiring is new.
No commit required unless code change needed — report findings only.

**STATUS (EMAC-047 findings):** `MIN_BET_EDGE` is ALREADY FULLY WIRED.
- Location: `backend/betting_model.py` lines 1774-1775 (D4 block).
- Current default: **2.5%** (not 2.0% — already tuned more conservatively than spec assumed).
- Uses `get_float_env("MIN_BET_EDGE", "2.5")` — fully env-overridable.
- Pass reason string: `f"Edge {edge:.1%} below MIN_BET_EDGE {floor:.1%} — signal too marginal to size"`.
- Missing from `.env.example` — added in EMAC-047.
- **No code change required.** Operator raises threshold via Railway env var `MIN_BET_EDGE=3.0` (etc.).

---

### GEMINI CLI — G-14: Railway Deploy Pipeline Tested [LOW]

Trigger a push to `main` and verify `.github/workflows/deploy.yml` successfully:
1. Passes `py_compile` on all modified files.
2. Passes 474/474 tests.
3. Successfully deploys to Railway without 502/503 errors.

Update HANDOFF.md G-14 to COMPLETE. Advance title to EMAC-051.

---

### KIMI CLI — K-6: Design O-8 Pre-Tournament Baseline [COMPLETE]

**Status:** ✅ SPEC COMPLETE — Script ready for execution

**Deliverables:**
- Spec: `reports/k6-o8-baseline-spec.md` — Full design document with architecture, schema, and integration plan
- Script: `scripts/openclaw_baseline.py` — Production-ready Python implementation

**Execution Plan:**
- **When:** March 16, 2026 ~9:00 PM ET (after bracket reveal at 6 PM, after A-26 T2 implementation by Claude)
- **Who:** OpenClaw (autonomous execution)
- **Prerequisites:**
  1. ✅ Script created and tested
  2. ✅ `BALLDONTLIE_API_KEY` in Railway (G-13)
  3. ⏳ Ollama running with qwen2.5:3b on execution host

**Output:**
- `data/pre_tournament_baseline_2026.json` — 68-team risk map with seed, region, risk_level, summary
- `reports/o8_baseline_summary_2026.md` — Human-readable summary
- HANDOFF.md auto-updated with baseline results

**Command:**
```bash
python scripts/openclaw_baseline.py --year 2026
```

**OpenClaw v2.1 + Discord Fix (2026-03-07):**
| Issue | Status | Details |
|-------|--------|---------|
| Discord "Unknown target" errors | ✅ FIXED | **Root cause identified: Claude CLI Discord integration (NOT OpenClaw)**. Disabled broken integration in `.env`. See `.claude/DISCORD_ERRORS_EXPLAINED.md` |
| OpenClaw notifications | ✅ ACTIVE | File logging to `.openclaw/notifications/*.log` — working correctly |
| WebSocket 1005/1006 disconnects | ✅ DOCUMENTED | Normal auto-reconnect behavior — see TROUBLESHOOTING.md |
| Notification wiring | ✅ ADDED | `high_stakes_escalation` and `integrity_volatile` triggers implemented |
| Notification logs | ✅ ACTIVE | `.openclaw/notifications/YYYY-MM-DD.log` captures all alerts |

**Key Distinction:** The persistent "Unknown target" and WebSocket errors were from **Claude CLI's built-in Discord client** (using `DISCORD_BOT_TOKEN`), NOT from OpenClaw. OpenClaw has its own notification system that logs to file by default. I've commented out the broken Claude Discord tokens in `.env` to stop the errors.

**References:** `.claude/DISCORD_ERRORS_EXPLAINED.md` | `.openclaw/TROUBLESHOOTING.md` | `.openclaw/README.md`

---

## 5. DEPENDENCY CHAIN

```
G-13 (Gemini) --> BALLDONTLIE_API_KEY set --> seed scalars activate March 16 [COMPLETE]
K-6 (Kimi)    --> O-8 baseline spec ready --> OpenClaw executes March 16 ~9 PM ET
O-6 (OpenClaw)--> integrity spot-check --> HEARTBEAT updated
O-9 (Claude)  --> tiered escalation wired --> LIVE (coordinator.py + logging active)

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
| 4 | BALLDONTLIE_API_KEY in Railway | ✅ | Gemini (G-13) |
| 5 | Seed-Spread Scalar Defaults Verified | ✅ | Claude |
| 6 | O-9 Tiered Escalation Wired | ✅ | Claude |
| 7 | O-8 Baseline Script Ready | ✅ | Kimi design / OpenClaw exec |
| 8 | O-10 Line Movement Monitor | ⬜ | Gemini + OpenClaw |
| 9 | Railway Deploy Pipeline Tested | ⬜ | Gemini |
| 10 | Rollback Plan Documented | ✅ | Claude |

---

## 7. ARCHITECT REVIEW QUEUE

**Pre-March 18:**
- Review K-6 spec before OpenClaw executes O-8

**March 16–18:**
- Verify seed scalars fire in logs after bracket loads
- Review O-8 baseline output, flag HIGH-risk teams
- Deploy O-10 line movement monitor (Gemini leads)

**Ongoing:**
- A-27: REVIEWED 2026-03-07. Parameters frozen at ha=2.419, sd_mult=1.0. V9 recal pending (need 50 settled V9 bets). See memory/calibration.md.
- A-28: MIN_BET_EDGE experiment — COMPLETE (already wired at 2.5% default). Operator raises via Railway env var.
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
MISSION: EMAC-050 — Monitoring mode. No active code missions.
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge

STATE: 474/474 tests. Railway live. V9 fully deployed.
A-27 calibration review COMPLETE — see memory/calibration.md.
Parameters frozen: ha=2.419, sd_mult=1.0. V9 recal pending (need 50 settled V9 bets).

NEXT UNBLOCK: O-10 line movement monitor architecture — wait for Gemini readiness signal.

GUARDIAN: py_compile + 474 tests before approving any Gemini commit.
```

### GEMINI CLI
```
MISSION: G-14 — Railway Deploy Pipeline Tested
Trigger a push to main and verify .github/workflows/deploy.yml successfully:
1. Passes py_compile on all modified files.
2. Passes 474/474 tests.
3. Successfully deploys to Railway without 502/503 errors.

Update HANDOFF.md G-14 to COMPLETE. Advance title to EMAC-051.
```

### KIMI CLI
```
MISSION: K-6 — Design O-8 Pre-Tournament Baseline Script
You are Deep Intelligence Unit AND OpenClaw Config Owner for CBB Edge Analyzer.
Read: backend/services/scout.py, reports/openclaw-capabilities-assessment.md, HEARTBEAT.md.
Design scripts/openclaw_baseline.py for OpenClaw to execute March 16 ~9 PM ET.
Output: 68-team JSON map (team -> seed, region, risk_level, summary). Use DDGS + qwen2.5:3b.
Save spec to reports/k6-o8-baseline-spec.md.
Update HANDOFF.md K-6 to COMPLETE. Advance title to EMAC-051.
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
Update HANDOFF.md O-6 row to COMPLETE. Advance title to EMAC-051.
```
