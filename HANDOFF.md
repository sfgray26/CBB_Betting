# OPERATIONAL HANDOFF (EMAC-053)

> Ground truth as of EMAC-052. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-052 — A-29 dead .openclaw import removed from analysis.py. G-14/G-15 validated. 481/481 tests passing.

| Component | Status | Detail |
|-----------|--------|--------|
| Railway API | OK | Live, healthy. All syntax errors resolved. |
| V9 Model | OK | SNR + Integrity + Seed-Spread Kelly scalars. `model_version='v9.0'` |
| Env Var Parsing | OK | `get_float_env` on all float reads. Zero plain `float(os.getenv)`. |
| CI Syntax Guard | OK | `py_compile` step in `deploy.yml` before pytest. |
| Analysis Pipeline | OK | Nightly running. 0 bets = correct conservatism (K-3). V9 recal at 50 bets. |
| Line Movement Monitor (O-10) | LIVE | Active. Runs every 30m. Checks consensus vs. bet log spread. |
| Seed-Spread Scalars (A-26 T2) | LIVE | Active. BALLDONTLIE_API_KEY verified in Railway. |
| Tournament SD Bump (A-26 T1) | READY | 1.15x when `is_neutral=True`. Active for neutral-site games. |
| Integrity Sweep | LIVE | Async, 8-worker concurrent. 0 BET games since V9 launch — correct. |
| O-6 Integrity Spot-Check | ✅ COMPLETE | 2026-03-07. Verified 133 predictions — all null verdicts (correct, no BET-tier games). |
| O-9 Tiered Escalation | LIVE (logging) | coordinator.py created. Logs ESCALATION_FLAGGED on units>=1.5, neutral_site, VOLATILE. Kimi API routing post-March 18. |
| O-8 Pre-Tournament Baseline | READY | Script created. OpenClaw executes March 16 ~9 PM ET. Discord errors fixed in v2.1 — see TROUBLESHOOTING.md. |
| Calibration | OK | ha=2.419, sd_mult=1.0 (V8-era). V9 recal after 50 settled V9 bets. |
| Gemini Trust Level | FULL | Restoration complete. Standard senior engineer workflow. |

---

## 1.5. OPENCLAW Status (Auto-Updated 2026-03-07 16:35)

| Component | Status | Detail |
|-----------|--------|--------|
| ✅ Circuit Breaker | CLOSED | Auto-reset after 60s if OPEN |
| 🔄 Last Integrity Sweep | No recent sweep | — |
| 📊 O-8 Baseline | ⏳ PENDING | Execute March 16 ~9 PM ET |

**Active Alerts:**
- None

---

## 2. TEAM ROLES

| Agent | Role | Trust | Current Focus |
|-------|------|-------|---------------|
| Claude Code | Master Architect | FULL | A-30: Wire Nightly Health Check APScheduler + morning briefing audit |
| Gemini CLI | DevOps Strike Lead | FULL | G-16: Post-Deploy Verification of O-10 |
| Kimi CLI | Deep Intelligence + **OpenClaw Config Owner** | FULL | K-7: Design A-30 Nightly Health Check thresholds |
| OpenClaw | Integrity Execution | FULL | O-8 Baseline execution March 16 ~9 PM ET |

**Gemini rule:** RESTORATION COMPLETE. standard senior engineer workflow resumed. `py_compile` + 474+ tests before every push.

**OpenClaw ownership:** Kimi owns OpenClaw configuration, prompt tuning, and optimization. OpenClaw executes what Kimi designs.

---

## 3. COMPLETED WORK

| Mission | Who | What |
|---------|-----|------|
| A-29 | Claude | Remove dead .openclaw relative import from analysis.py. Non-breaking cleanup. 481/481 tests. |
| O-10 | Claude | BET_ADVERSE_MOVE detection in odds_monitor.py: event-driven, T-2h golden window, >2pt moves, 4 tests. |
| G-15 | Gemini | O-10 Line Movement Monitor implemented. Scheduled every 30m. Discord alerts wired. |
| O-6 | OpenClaw | V9 Integrity Spot-Check complete. Verified all 133 predictions have null integrity_verdict (0 BET-tier games = sweep not triggered = correct). |
| G-14 | Gemini | Railway Deploy Pipeline Tested. Verified 474 tests + py_compile + Railway startup. |
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

### CLAUDE CODE — A-30: Wire Nightly Health Check + Morning Briefing Audit [MEDIUM]

The HEARTBEAT defines a Nightly Health Check (4:30 AM ET daily) but it is not wired as an APScheduler job.
Also: scout.py has `generate_morning_briefing()` — verify if this is called anywhere in main.py.

Steps:
1. Read `backend/main.py` scheduler jobs section to see what is currently scheduled.
2. Read `backend/services/scout.py` to find `generate_morning_briefing()` signature.
3. If morning briefing is NOT scheduled: add `_morning_briefing_job` as APScheduler cron at 7 AM ET daily.
4. Write `_nightly_health_check_job()` in main.py: logs MAE, predictions count, bets, drawdown. Warns if MAE > 3 pts.
5. Add health check job to APScheduler at 4:30 AM ET (after daily_snapshot at 4 AM).
6. Update HEARTBEAT.md: Nightly Health Check -> LIVE.
7. py_compile + 481 tests. Commit.
8. Update HANDOFF.md A-30 to COMPLETE. Advance to EMAC-054.

Constraints: Single file (main.py). No DB schema changes.

---

### GEMINI CLI — G-16: Post-Deploy Verification of O-10 [LOW]

After push to main and deployment:
1. Verify `line_monitor` job is scheduled in `/admin/scheduler/status`.
2. Check Railway logs for `Starting check_line_movements job`.
3. Verify Discord bot receives line monitor alerts if movement occurs.

Update HANDOFF.md G-16 to COMPLETE. Advance title to EMAC-053.

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

### OPENCLAW — O-6: Integrity Spot-Check [✅ COMPLETE]

**Result:** `O-6: Not triggered — correct`

All 133 predictions verified — all have `null` integrity_verdict. This is expected because 0 BET-tier games means the Integrity Sweep was not triggered. HEARTBEAT.md updated.

---

## 5. DEPENDENCY CHAIN

```
G-13 (Gemini) --> BALLDONTLIE_API_KEY set --> seed scalars activate March 16 [COMPLETE]
K-6 (Kimi)    --> O-8 baseline spec ready --> OpenClaw executes March 16 ~9 PM ET
O-6 (OpenClaw)--> integrity spot-check --> HEARTBEAT updated
O-9 (Claude)  --> tiered escalation wired --> LIVE (coordinator.py + logging active)
G-15 (Gemini) --> O-10 line monitor wired --> LIVE (runs every 30m) [COMPLETE]

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
| 8 | O-10 Line Movement Monitor | ✅ | Gemini (G-15) |
| 9 | Railway Deploy Pipeline Tested | ✅ | Gemini (G-14) |
| 10 | Rollback Plan Documented | ✅ | Claude |

---

## 7. ARCHITECT REVIEW QUEUE

**Pre-March 18:**
- Review K-6 spec before OpenClaw executes O-8

**March 16–18:**
- Verify seed scalars fire in logs after bracket loads
- Review O-8 baseline output, flag HIGH-risk teams
- Monitor O-10 line movement monitor (Gemini leads)

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
| Gemini O-10 (G-15) and Claude O-10 are COMPLEMENTARY: G-15=DB-driven position monitor (30m), Claude=event-driven in-memory golden-window check. Both needed. | EMAC-052 |
| Dead imports in try/except blocks are invisible failures. Remove them -- do not paper over with broader except. | EMAC-052 |

---

## 9. HANDOFF PROMPTS

### CLAUDE CODE
```
MISSION: EMAC-053 — A-30: Wire Nightly Health Check APScheduler + morning briefing audit.
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge

STATE: 481/481 tests. Railway live. V9 fully deployed.
A-29 COMPLETE (dead .openclaw import removed from analysis.py).
O-10 LIVE (G-15 DB-driven + Claude event-driven golden-window).
Parameters frozen: ha=2.419, sd_mult=1.0. V9 recal pending (need 50 settled V9 bets).

NEXT: Implement A-30 (see Section 4). py_compile + 481 tests before commit.
GUARDIAN: py_compile + 481 tests before approving any Gemini commit.
```

### GEMINI CLI
```
MISSION: G-16 — Post-Deploy Verification of O-10 (still open)
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge
1. Verify line_monitor job is scheduled in /admin/scheduler/status.
2. Check Railway logs for "Starting check_line_movements job".
3. Verify Discord bot receives line monitor alerts if movement occurs.

Update HANDOFF.md G-16 to COMPLETE. Advance title to EMAC-054.
```

### KIMI CLI
```
MISSION: K-7 — Design A-30 Nightly Health Check thresholds
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge
K-6 COMPLETE. scripts/openclaw_baseline.py ready for March 16.

Review HEARTBEAT.md Nightly Health Check spec.
Read backend/services/performance.py for available metrics (MAE, ROI, etc.).
Recommend thresholds for _nightly_health_check_job:
  - MAE warning threshold (currently proposed 3.0 pts)
  - Drawdown warning vs halt levels
  - Min predictions per night for a meaningful check

Output: reports/k7-health-check-thresholds.md
Update HANDOFF.md K-7 to COMPLETE. Advance to EMAC-054.
```

### OPENCLAW
```
MISSION: O-6 ✅ COMPLETE — O-8 PENDING EXECUTION March 16 ~9 PM ET

O-6 RESULT: O-6: Not triggered — correct (all 133 predictions have null integrity_verdict).

NEXT: O-8 Pre-Tournament Baseline — EXECUTE March 16, 2026 ~9:00 PM ET
Command: python scripts/openclaw_baseline.py --year 2026
Prerequisites: Ollama running with qwen2.5:3b, BALLDONTLIE_API_KEY set in Railway
Output: data/pre_tournament_baseline_2026.json + reports/o8_baseline_summary_2026.md
```
