# OPERATIONAL HANDOFF (EMAC-045)

> Ground truth as of EMAC-044. Operator: Claude Code (Master Architect).
> Read `IDENTITY.md` for risk policy. Read `AGENTS.md` for roles. Read `HEARTBEAT.md` for loops.

---

## 1. SYSTEM STATUS

**Last completed:** EMAC-045 — BallDontLie API bug fixed (endpoint `/bracket`, field `name`, `season=year-1`). Pre-tournament assessment report created. Rollback plan documented. HANDOFF.md updated. 464/464 tests passing.

| Component | Status | Detail |
|-----------|--------|--------|
| V9 Model | OK | SNR + Integrity Kelly scalars. `model_version='v9.0'` |
| Railway API | OK | Live. All syntax errors resolved (EMAC-039/040/042). |
| Env Var Parsing | OK | `get_float_env` applied to ALL float reads. Zero plain `float(os.getenv)` remaining. |
| CI Syntax Guard (G-12) | COMPLETE | `py_compile` step added to `deploy.yml`. |
| Railway Env Var Audit (G-11) | COMPLETE | Root-cause fix: variables cleaned in Railway UI. Verified healthy. |
| Dashboard API | OK | `API_URL` = Railway production. Data Cleanup endpoint added (`DELETE /admin/games/{id}`). |
| Analysis Pipeline | OK | Daily runs continue through March 15. `bets_recommended=0` is correct conservatism per K-3. V9 recal at 50 bets. |
| Tournament SD Bump (A-26 T1) | OK | `TOURNAMENT_MODE_SD_BUMP` (1.15x) ready for tournament. Not active in regular season. |
| Neutral-Site Fix (A-25) | OK | `parse_odds_for_game` extracts `neutral_site` -> `is_neutral`. |
| Recalibration | OK | ha=2.419, sd_mult=1.0. V8-calibrated; V9-specific recalibration after 50 settled V9 bets. |
| Railway DB | OK | PostgreSQL connected. 9 tables initialized. Nightly running. |
| OpenClaw v2.0 | OK | O-7 coordinator validation: 4/4 tests. Integrity sweep wired in analysis pipeline. |
| Integrity Spot-Check (O-6) | OPEN | Verify integrity_verdict fields in production predictions. Assigned to OpenClaw. |
| Seed-Spread Scalars (A-26 T2) | COMPLETE | Implemented ahead of schedule. 464 tests passing. No-op until bracket loaded (BALLDONTLIE_API_KEY needed in Railway). |
| A-26 T2 Architecture Spec (K-4) | COMPLETE | Spec at `reports/2026-03-16-a26t2-implementation-spec.md`. Implemented EMAC-044. |
| BallDontLie API Bug (7.5) | FIXED | Endpoint `/bracket`, field `name`, season-1 offset. Fixed EMAC-045. |
| SNR Re-Audit (A-19) | DEFERRED | Needs 20+ settled V9-era bets. |
| Gemini Trust Level | RESTORING | G-11 (env audit) complete. G-12 in CI. Trust restoring after clean execution. |

---

## 1.5. OPENCLAW STATUS (Auto-Updated)

**Last Updated:** 2026-03-06 by Kimi CLI (diagnostic audit)

| Component | Status | Detail |
|-----------|--------|--------|
| Coordinator | ✅ HEALTHY | v2.0, circuit breaker CLOSED, O-7 4/4 tests passed |
| Integrity Sweep | ✅ ACTIVE | Wired in `analysis.py`, runs on BET-tier games |
| Memory System | ✅ CREATED | `memory/` dir + `.openclaw/operational-state.json` initialized |
| Telemetry | ⚠️ PARTIAL | Sweep runs but no structured logging yet (see Issue Analysis) |
| Scheduler | ❌ NOT ACTIVE | Health checks defined in HEARTBEAT.md but not triggered |
| Tiered Escalation (O-9) | ❌ NOT WIRED | Coordinator exists but `analysis.py` bypasses it |
| HANDOFF Updates | ❌ MANUAL | No auto-update mechanism from OpenClaw |

**Active Alerts:**
- OpenClaw telemetry gap — sweep results not persisted (Issue #4 in analysis)
- Tiered escalation to Kimi not yet wired (O-9 incomplete)
- Nightly health check never runs (no scheduler)

**Reference:** Full issue analysis at `reports/openclaw-issue-analysis.md`

---

## 2. SYSTEM CONTEXT

### Model Quality — K-3 Verdict (FINAL)

**0 bets on 12 games = correct conservatism. Not a bug.**

Root causes:
1. `sd_mult=1.0` (vs default 0.85) = +17.6% wider SD → ~2.8pp edge compression
2. `ha=2.419` (vs default 3.09) = -21.7% less HCA → compressed home-team margins
3. V9 structural mismatch: 663 V8-era calibration bets had no SNR/integrity scalars. V9 applies 0.25–1.0x combined Kelly scalar — structural under-sizing relative to calibrated thresholds.

**No code changes needed now.** V9-specific recalibration is the long-term fix, triggered after 50 settled V9-era bets.

### Team State

| Agent | Role | Trust | Notes |
|-------|------|-------|-------|
| Claude Code | Master Architect | FULL | Standby until March 16. Guardian of all commits. |
| Gemini CLI | DevOps Strike Lead | RESTORING | G-11 complete. Single-file tasks only. |
| Kimi CLI | Deep Intelligence | FULL | K-4 complete. |
| OpenClaw | Integrity Execution | FULL | O-6 queued. Integrity sweep live in pipeline. |

**Gemini scope rule:** No multi-file Python refactors. Each task must be a single file or a non-Python file. Before committing any `.py` file: `python -m py_compile <file>.py`. Before any push: `pytest tests/ -q --tb=no` (438 must pass).

---

## 3. COMPLETED WORK (ARCHIVE)

| Mission | Who | What |
|---------|-----|------|
| A-26 T2 | Claude | Seed-spread Kelly scalars: tournament_data.py + analysis.py enrichment + betting_model.py scalars. 464 tests. |
| G-11 | Gemini | Railway env var audit: Cleaned ` =VALUE` vars in UI. Verified healthy. |
| K-4 | Kimi | A-26 T2 Architecture Spec: `reports/2026-03-16-a26t2-implementation-spec.md` |
| K-1 | Kimi | Tournament intelligence: SD bump 1.15x validated |
| K-2 | Kimi | Seed data research: BallDontLie API recommended, ESPN free fallback |
| K-3 | Kimi | Model quality audit: 0 bets = correct conservatism confirmed |
| G-12 | Claude (took over from degraded Gemini) | `py_compile` step in `deploy.yml` |
| A-25 | Claude | Neutral-site fix: `parse_odds_for_game` → `is_neutral` |
| A-26 T1 | Claude | Tournament SD bump (1.15x) when `is_neutral=True` |
| EMAC-037–042 | Claude | `get_float_env` applied to all 37 float env reads; Railway SyntaxErrors fixed |

---

## 4. REGULAR SEASON OPERATIONS (Through March 15)

**Status:** Daily analysis continues. Tournament prep is PARALLEL, not a replacement.

### Daily Operations Checklist
- [ ] Nightly analysis runs successfully (12 AM ET)
- [ ] Odds API fetching without errors
- [ ] Ratings data fresh (KenPom, BartTorvik, EvanMiya)
- [ ] Integrity sweep completing for BET-tier games
- [ ] Predictions persisted to Railway DB
- [ ] Paper trades created for BET verdicts
- [ ] Performance tracking accumulating

### Regular Season Model Optimization (Parallel Track)

**A-27: V9 Calibration Monitoring [NEW — ONGOING]**
**Owner:** Claude (weekly review)

**Context:** 
- K-3 identified V9 structural mismatch with V8 calibration
- Need 50 settled V9-era bets for recalibration
- Current: ~12 games analyzed, 0 bets (edge compression working as designed)

**Weekly Tasks:**
1. Review `GET /api/performance/model-accuracy` — track MAE vs V8 baseline
2. Check edge distribution: what % of games have `edge_conservative > 0`? > 2.5%?
3. Document any systematic biases (home/away, favorites/dogs)
4. If MAE drifts > 3 pts from 30-day baseline → escalate

**Output:** Weekly calibration brief added to `memory/` (e.g., `memory/2026-03-10-calibration.md`)

---

**A-28: Edge Detection Optimization [NEW — ONGOING]**
**Owner:** Claude (as time permits)

**Hypothesis:** Current `MIN_BET_EDGE=2.5%` may be too conservative for regular season,
given `sd_mult=1.0` widening and V9 scalars compressing effective Kelly.

**Experiment:** 
- Reduce `MIN_BET_EDGE` to 2.0% for 1 week (March 10-17)
- Monitor: bet rate, win rate, CLV (closing line value)
- Compare to prior week

**Decision Rule:**
- If win rate >= 52% and positive CLV → keep 2.0% for tournament
- If win rate < 50% → revert to 2.5%

**Implementation:** Via Railway env var `MIN_BET_EDGE=2.0` (no code change)

---

**A-29: OpenClaw Integration Hardening [NEW — ONGOING]**
**Owner:** OpenClaw + Gemini

**Current:** Integrity sweep runs on BET-tier games only.
**Enhancement:** Run on CONSIDER-tier games too (lower stakes, more samples).

**Rationale:**
- More data for OpenClaw calibration
- Practice for tournament load
- Catch false negatives (good edges marked CONSIDER due to integrity flags)

**Implementation:**
1. Modify `analysis.py` line ~919: include CONSIDER games in sweep inputs
2. Log CONSIDER-tier verdicts separately (don't block, just observe)
3. After 50 games, compare: CONSIDER tier outcomes vs verdicts

---

## 5. ACTIVE MISSIONS

---

### OPENCLAW — O-6: V9 Integrity Spot-Check [OPEN — MEDIUM PRIORITY]

**What:** Verify that V9 integrity verdicts (`VOLATILE`/`CAUTION`/`CONFIRMED`) are being populated in production predictions.

**Context:**
- Analysis pipeline: `_integrity_sweep()` runs ONLY on BET-tier games (not PASS/CONSIDER).
- Last slate had 0 BET-tier games. So integrity_verdict is expected to be null for all predictions — this is NOT a bug.
- The spot-check goal is to verify the field exists and the pipeline didn't silently error.

**Your tasks:**

1. Call `GET https://cbbbetting-production.up.railway.app/api/predictions/today` with header `X-API-Key: <your key>`.
2. For each prediction in the response:
   - Check if `integrity_verdict` field exists anywhere in the JSON (top-level or nested under `full_analysis.calculations`).
   - Check if any verdict is `"Sanity check unavailable"` — this would indicate a silent Ollama failure.
3. Expected result: `integrity_verdict` is null or absent for all predictions (correct — 0 BET-tier games means sweep was not triggered).
4. Report one of:
   - `O-6 STATUS: Not triggered — correct (0 BET-tier games in slate, integrity_verdict=null for all)`
   - `O-6 STATUS: Active — N predictions have verdicts: [list them]`
   - `O-6 STATUS: BROKEN — "Sanity check unavailable" on prediction [id]`

**Output:**
- Update `HEARTBEAT.md` with O-6 status and timestamp.
- Update HANDOFF.md Section 1 `| Integrity Spot-Check (O-6) |` row to COMPLETE with verdict.
- Update title to EMAC-044.

---

### CLAUDE CODE — A-26 Task 2: Seed-Spread Kelly Scalars [DEFERRED — March 16]

**Earliest start:** 8 PM ET March 16 (BallDontLie has bracket data ~2h after 6 PM reveal).

**Input:** K-4 spec from Kimi (`reports/2026-03-16-a26t2-implementation-spec.md`).

**Implementation steps:**
1. Create `backend/services/tournament_data.py` — BallDontLie seed lookup with TTL cache + ESPN fallback.
2. Enrich `game_data` in `analysis.py::_analyze_games_pass2()` with `seed_home`, `seed_away`.
3. Add `_seed_spread_kelly_scalar()` to `betting_model.py` after integrity scalar in `analyze_game()`.
4. Add env vars to `.env.example`: `SEED_DATA_SOURCE=balldontlie`, `BALLDONTLIE_API_KEY=`.
5. Add `TestSeedSpreadScalars` to `tests/test_betting_model.py` (5+ tests per K-4 spec).
6. Run `pytest tests/ -q` — all 438+ must pass.
7. `python -m py_compile` on every touched `.py` file before committing.

**Guardian rule:** Do NOT start before K-4 spec is complete. If K-4 spec is missing, Kimi first.

---

## 6. OPENCLAW STRATEGIC LEVERAGE - NEW MISSIONS

Based on `reports/openclaw-capabilities-assessment.md`, OpenClaw is significantly 
underutilized. The following missions leverage its unique real-time intelligence
capabilities:

### OPENCLAW — O-8: Pre-Tournament Intelligence Baseline [NEW — HIGH PRIORITY]

**What:** Batch process all 68 tournament teams March 16-17 to establish injury/availability baseline.

**Why:** Enter tournament with known risk factors for every team, not just BET-tier games.

**Implementation:**
```bash
# Run March 16 ~9 PM ET (after BallDontLie bracket available)
python scripts/openclaw_baseline.py --output data/pre_tournament_intel.json
```

**Tasks:**
1. Create `scripts/openclaw_baseline.py`:
   - Fetch tournament bracket from BallDontLie
   - For each of 68 teams: DDGS search `{team} injury news March 2026`
   - qwen2.5:3b summarizes health status per team
   - Output: JSON with team → risk_level (low/medium/high) + summary

2. Output format:
```json
{
  "Duke Blue Devils": {
    "seed": 1,
    "region": "South",
    "risk_level": "low",
    "summary": "No significant injuries reported",
    "last_updated": "2026-03-16T21:00:00Z"
  },
  ...
}
```

3. Generate heatmap report (markdown) for operator review.

**Timeline:** Execute March 16 9 PM ET — 2 hours before First Four.
**Cost:** ~$0.50 (68 teams × 2 searches × $0.001)
**Owner:** OpenClaw (with Claude review of script)

---

### OPENCLAW — O-9: Wire Tiered Escalation to Kimi [NEW — CRITICAL]

**What:** Connect existing `coordinator.py` escalation rules to live analysis pipeline.

**Current State:** Rules exist but are NOT wired to `analysis.py`.

**Escalation Triggers (from coordinator.py):**
- `recommended_units >= 1.5`
- `tournament_round >= 4` (Sweet 16+)
- `integrity_verdict contains "VOLATILE"`

**Implementation in `analysis.py`:**
```python
# After _integrity_sweep() completes:
for game in bet_tier_games:
    context = TaskContext(
        recommended_units=game['edge'],
        tournament_round=game.get('round', 1),
        integrity_verdict=_integrity_results.get(game_key)
    )
    
    if (context.recommended_units >= 1.5 or 
        context.tournament_round >= 4 or 
        "VOLATILE" in context.integrity_verdict):
        # Escalate to Kimi
        verdict = await coordinator.route_to_kimi(
            task_type=TaskType.INTEGRITY_CHECK,
            context=context,
            prompt=build_kimi_prompt(game)
        )
    else:
        # Use local OpenClaw result
        verdict = _integrity_results[game_key]
```

**Timeline:** Must be complete before March 18 (First Four).
**Owner:** Claude (with OpenClaw testing)
**Complexity:** Medium — requires coordinator integration testing.

---

### OPENCLAW — O-10: Line Movement Monitoring [NEW — TOURNAMENT PHASE]

**What:** Real-time monitoring for sharp line moves within 2 hours of tipoff.

**Why:** Sharp money often knows something the model doesn't. Exit if line moves against us.

**Implementation:**
1. New APScheduler job: `scripts/line_movement_monitor.py`
2. Poll Odds API every 15 minutes for games with `hours_to_tipoff < 2`
3. If spread moves >2 points against our recommended side:
   - Trigger OpenClaw re-check
   - If new red flags → auto-downgrade or ABORT
   - Alert operator

**Timeline:** Deploy March 18, run continuously through tournament.
**Owner:** Gemini (infrastructure) + OpenClaw (execution)
**Complexity:** Medium — requires new endpoint + alerting.

---

### OPENCLAW — O-11: Cross-Game Correlation Detection [NEW — TOURNAMENT PHASE]

**What:** Detect slate-wide risks (e.g., 3+ games in same conference all flagged VOLATILE).

**Why:** Portfolio-level risk management. If systemic risk detected, reduce overall exposure.

**Implementation:**
```python
# After integrity sweep:
volatile_by_conference = defaultdict(int)
for game in slate:
    if "VOLATILE" in game['verdict']:
        volatile_by_conference[game['conference']] += 1

for conf, count in volatile_by_conference.items():
    if count >= 3:
        logger.warning(f"SYSTEMIC_RISK: {conf} has {count} VOLATILE games")
        # Trigger exposure reduction
```

**Timeline:** Tournament phase.
**Owner:** OpenClaw
**Complexity:** Low — 20-line addition to existing sweep.

---

## 7. DEPENDENCY CHAIN

```
G-11 (Gemini — Railway env var cleanup)
  --> COMPLETE

O-6 (OpenClaw — integrity spot-check)
  --> UNBLOCKED — run now
  --> Output feeds HEARTBEAT.md

K-4 (Kimi — A-26 T2 spec)
  --> COMPLETE

Bracket reveals March 16 @ 6 PM ET
  + K-4 spec complete
  --> A-26 T2 implementation (Claude, March 16-18)
  --> Tournament starts March 18
```

---

## 7.5. BALLDONTLIE API RESEARCH (Kimi CLI)

**Status:** COMPLETE — Research documented at `reports/balldontlie-api-research.md`

### Key Findings

| Item | Detail |
|------|--------|
| **Endpoint** | `GET https://api.balldontlie.io/ncaab/v1/bracket` |
| **Auth** | Header: `Authorization: YOUR_API_KEY` |
| **Tier Required** | GOAT (highest tier) |
| **Seed Location** | `response.data[].home_team.seed` / `away_team.seed` |
| **Season Param** | Offset by -1 (use `season=2025` for 2026 tournament) |

### Round ID Mapping
```
1 = First Four / Play-in
2 = Round of 64
3 = Round of 32
4 = Sweet 16
5 = Elite 8
6 = Final Four
7 = Championship
```

### Sample Response (seed extraction)
```json
{
  "home_team": {
    "name": "Duke",
    "seed": "1",
    ...
  },
  "away_team": {
    "name": "Mount St. Mary's",
    "seed": "16",
    ...
  }
}
```

### Implementation Ready
- ✅ API contract confirmed
- ✅ Authentication mechanism documented
- ✅ Seed field location verified
- ⚠️ Requires GOAT tier API key (not yet in Railway)

**Action Required:** Add `BALLDONTLIE_API_KEY` to Railway env vars before March 16.

---

## 8. ARCHITECT REVIEW QUEUE

### Regular Season (Now — March 15)
- **A-27 V9 Calibration Monitoring:** Weekly review of model accuracy. Trigger recalibration if MAE drifts > 3 pts or at 50 settled V9 bets (whichever first).
- **A-28 Edge Detection Optimization:** Test `MIN_BET_EDGE=2.0%` vs 2.5% for 1 week. Measure bet rate, win rate, CLV. Decision by March 17.
- **A-29 OpenClaw Hardening:** Include CONSIDER-tier games in integrity sweep for more training data. Evaluate false negative rate after 50 games.
- **Daily Ops Guardian:** Monitor nightly runs for errors. Any failure → immediate triage.

### Tournament Phase (March 16 — April 7)
- **March 16 window:** A-26 T2 must be implemented and deployed in < 48h after bracket. Prep Railway deploy pipeline.
- **March 18 readiness:** O-8 (pre-tournament baseline) complete, O-9 (tiered escalation) wired, O-10 (line monitor) deployed.
- **Sweet 16+ (March 22+):** Activate tiered integrity escalation (Elite 8+ → Kimi automatic).
- **V9 recalibration trigger:** At 50 settled V9-era bets, run recalibration. May hit during tournament — monitor daily.

### Post-Season (April+)
- **SNR re-audit (A-19):** After 20+ settled V9-era bets.
- **Season-end recalibration:** Full V9-era dataset (target N > 500). Off-season task.
- **OpenClaw model training:** Custom classifier for integrity checks (summer project).

---

## 9. HIVE WISDOM

| Lesson | Source |
|--------|--------|
| `pred_id` (Prediction PK) is the correct Streamlit widget key — never `game_id`. | EMAC-019 |
| Always store `base_sd_override` in context. `None` != "same as original". | EMAC-021 |
| `full_analysis.inputs` has no "game" key. Reconstruct from `p.game` DB relationship. | EMAC-023 |
| One-fire sets must be cleared on cache refresh. | EMAC-023 |
| `async def` without `asyncio.to_thread` wrapping sync I/O = ZERO concurrency. | EMAC-027 |
| True calibration entry point is `backend/services/recalibration.py::run_recalibration()`. | EMAC-028 |
| `sys.path` manipulation belongs inside `if __name__ == "__main__":` guards only. | EMAC-029 |
| sd_multiplier oscillates at noise boundary. Min-delta guard (0.03) prevents flip-flopping. | EMAC-031 |
| `parse_odds_for_game` result dict is the single source of truth for game metadata. | EMAC-034 |
| Use Kimi for tasks requiring >50K tokens simultaneously (performance audits, codebase-wide). | EMAC-034 |
| Single-elimination tournament variance is 15-25% higher. Apply SD bump (1.15x) when is_neutral=True. | K-1 |
| Tournament SD bump applies AFTER all other SD penalties. Order matters — bump last. | EMAC-036 |
| Gemini's large-scale refactors drop closing parens. NEVER approve a multi-file Gemini refactor without running `py_compile` on every modified .py file. | EMAC-038 |
| `get_float_env` (backend/utils/env_utils.py) must be used for ALL env var float reads. Any new float env var MUST use it. | EMAC-037/038/042 |
| `main.py` syntax errors bypass `pytest` (no DB). Always run `python -m py_compile backend/main.py` after any main.py change. | EMAC-039 |
| When Gemini breaks a file: restore from pre-Gemini commit (`git checkout HASH -- file.py`), apply legitimate changes via Python regex. Fastest path. | EMAC-039 |
| `sd_mult=1.0` widens distribution, compresses edges. V9-specific recalibration after 50+ V9 bets settle. | EMAC-040/K-3 |
| Uncommitted local changes are invisible to Railway. Always verify changes are pushed before attributing errors to the code fix. | EMAC-042 |
| BallDontLie uses `season=year-1` offset (2025 for 2026 tournament). Field is `name` not `full_name`. Endpoint is `/bracket` not `/march_madness_bracket`. | EMAC-045 / Kimi K-5 |

---

## 10. PRE-TOURNAMENT CHECKLIST (T-2 Days)

**Status:** IN PROGRESS — 4/10 items green. Assessment complete (EMAC-045).

This checklist must be 100% green before March 18 (First Four). Assessment at
`reports/2026-03-16-project-state-assessment.md`.

| # | Item | Status | Owner | Verification |
|---|------|--------|-------|--------------|
| 1 | System Status Review Complete | ✅ | Claude | Assessment doc at reports/2026-03-16-project-state-assessment.md |
| 2 | Team Readiness Confirmed | ⬜ | Claude | All agents 8+/10 readiness |
| 3 | A-26 T2 Spec Reviewed + Implemented | ✅ | Claude | 464 tests passing, commit 7ee0207 |
| 4 | BallDontLie API Key Set | ⬜ | Gemini | Railway env var `BALLDONTLIE_API_KEY` needed |
| 5 | Seed-Spread Scalar Defaults Verified | ✅ | Claude | 0.75/0.75/0.80 confirmed + env var overridable |
| 6 | OpenClaw Tiered Escalation Wired | ⬜ | Claude | O-9 complete |
| 7 | Pre-Tournament Baseline Script Ready | ⬜ | OpenClaw | O-8 script tested |
| 8 | Line Movement Monitor Deployed | ⬜ | Gemini | O-10 in production |
| 9 | Railway Deploy Pipeline Tested | ⬜ | Gemini | Deploy <10 min verified |
| 10 | Rollback Plan Documented | ✅ | Claude | Rollback via Railway env vars + git revert documented in assessment |

**Legend:**
- ⬜ = Not started
- 🟡 = In progress
- ✅ = Complete
- 🔴 = Blocked/Risk

---

## 11. HANDOFF PROMPTS

### PROMPT FOR CLAUDE CODE — COMPREHENSIVE ARCHITECT ASSESSMENT & PLANNING
```
MISSION: EMAC-043 — Master Architect Strategic Review & Dual-Track Operations
You are Claude Code, Master Architect for CBB Edge Analyzer.
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge

CONTEXT:
We are operating on TWO PARALLEL TRACKS:

TRACK 1: REGULAR SEASON (Business as Usual)
- Daily analysis runs continue through March 15 (Selection Sunday)
- Model optimization continues (V9 calibration, edge detection improvements)
- Regular betting operations and performance tracking
- 438 tests must continue passing

TRACK 2: TOURNAMENT PREPARATION (March 16-18 readiness)
- A-26 T2 implementation (seed-spread scalars)
- OpenClaw enhancements (tiered escalation, baseline monitoring)
- Infrastructure hardening for high-stakes games

DO NOT pause Track 1 for Track 2. Both proceed in parallel.

YOUR TASKS — READ AND EXECUTE IN ORDER:

TASK 1: PROJECT STATE ASSESSMENT (30 min)
────────────────────────────────────────────
Read and assess the following:
1. AGENTS.md — Verify all agent roles are current and accurate
2. IDENTITY.md — Confirm risk posture is appropriate for tournament variance
3. HEARTBEAT.md — Check if monitoring loops are comprehensive
4. All files in reports/ — Understand open questions and technical debt

OUTPUT: Create reports/2026-03-16-project-state-assessment.md with:
- GREEN (healthy) / YELLOW (monitor) / RED (fix before tournament) status for each component
- List of 3 highest-risk items that could fail during tournament
- Recommended mitigations for each risk

TASK 2: TEAM ROLES & CAPABILITIES REVIEW (20 min)
────────────────────────────────────────────────────
Assess team readiness for tournament operations:

| Agent | Current Role | Tournament Readiness | Gaps |
|-------|-------------|---------------------|------|
| Claude (you) | Architect, A-26 T2 implementer | ? | ? |
| Gemini | DevOps, env management | ? | ? |
| Kimi | Deep Intelligence, escalation | ? | ? |
| OpenClaw | Integrity, real-time checks | ? | ? |

OUTPUT: Add "Team Readiness Matrix" section to your assessment doc with:
- Readiness score 1-10 for each agent
- Specific gaps that must be closed before March 18
- Resource reallocation recommendations

TASK 3: IMPROVEMENT OPPORTUNITIES IDENTIFICATION (20 min)
──────────────────────────────────────────────────────────
Based on your assessment and the OpenClaw capabilities assessment 
(reports/openclaw-capabilities-assessment.md), identify:

1. QUICK WINS (can implement before March 18):
   - Items requiring <4h effort with measurable impact

2. TOURNAMENT-CRITICAL (must have for March 18-April 7):
   - Items that directly impact betting decisions or risk management

3. POST-TOURNAMENT (nice to have, April+):
   - Technical debt and capability improvements

OUTPUT: Add "Improvement Roadmap" section with prioritized list.

TASK 4: HANDOFF.md UPDATE (15 min)
──────────────────────────────────
Update HANDOFF.md based on your assessment:
- Update Section 1 (System Status) with any new risks identified
- Add Section 9: "PRE-TOURNAMENT CHECKLIST" with 10 items that must be green before March 18
- Update your prompt in Section 8 with any new standing instructions

TASK 5: CRITICAL PATH IDENTIFICATION (15 min)
─────────────────────────────────────────────
Define the critical path from NOW → March 18 tournament start:

NOW → [Task A] → [Task B] → ... → Tournament Ready

Identify:
- Single point of failure tasks (if X fails, we're not ready)
- Parallelizable work (what can agents do simultaneously)
- Contingency plans (if something fails at T-24h)

OUTPUT: Add "Critical Path" diagram to assessment doc.

DELIVERABLES:
1. reports/2026-03-16-project-state-assessment.md (comprehensive)
2. Updated HANDOFF.md with new Section 9
3. Brief summary of top 3 risks and mitigations (add to HANDOFF.md Section 1)

TIMELINE: Complete within 2 hours. This is blocking tournament preparation planning.
```

---

### PROMPT FOR CLAUDE CODE — DUAL-TRACK OPERATIONS MODE
```
MISSION: EMAC-043 — Master Architect: Regular Season + Tournament Prep
You are Claude Code, Master Architect for CBB Edge Analyzer.
Working directory: C:\Users\sfgra\repos\Fixed\cbb-edge

SYSTEM STATE (all confirmed):
- 464/464 tests passing. Railway live and healthy.
- A-26 T2 COMPLETE. Seed-spread Kelly scalars live (no-op until BALLDONTLIE_API_KEY set in Railway).
- G-11 COMPLETE. Railway env vars cleaned at source.
- G-12 COMPLETE. CI syntax guard (py_compile) installed.
- K-4 COMPLETE. Spec implemented ahead of schedule.

DUAL-TRACK OPERATIONS:

TRACK 1 — REGULAR SEASON (Daily through March 15):
- [ ] Monitor nightly analysis runs for errors
- [ ] Weekly A-27 calibration review (Fridays)
- [ ] A-28 edge detection experiment (MIN_BET_EDGE=2.0% trial)
- [ ] Continue model optimization as opportunities arise

TRACK 2 — TOURNAMENT PREPARATION (Parallel):
- [ ] Review A-26 T2 spec (reports/2026-03-16-a26t2-implementation-spec.md)
- [ ] Prep implementation branch for March 16
- [ ] O-9: Wire tiered escalation (coordinate with OpenClaw)
- [ ] O-8: Review pre-tournament baseline script

GUARDIAN DUTIES (Always Active):
- Review all commits before merge (run `py_compile` on .py files)
- If Gemini breaks a file: restore from git, apply surgical fixes
- Verify 438 tests pass before any push
- Monitor Railway deploys for health

EMERGENCY PROTOCOL:
If nightly analysis fails → Triage within 1 hour
  1. Check Railway logs for error location
  2. If env var issue → assign Gemini
  3. If model logic issue → fix or revert
  4. If data source down → monitor, don't panic (use cached data)
```

---

### PROMPT FOR OPENCLAW — DUAL-TRACK OPERATIONS
```
MISSION: EMAC-043 — Integrity Execution Unit: Daily Ops + Tournament Prep
You are the Integrity Execution Unit (OpenClaw) for CBB Edge Analyzer.

DUAL-TRACK RESPONSIBILITIES:

TRACK 1 — REGULAR SEASON (Through March 15):
1. Daily integrity sweeps for all BET-tier games
2. O-6 spot-check: Verify integrity_verdict fields are populating correctly
3. A-29 enhancement: Include CONSIDER-tier games in sweep (observation mode)
4. Performance monitoring: Track latency, success rate, verdict distribution

TRACK 2 — TOURNAMENT PREPARATION:
1. O-8: Pre-tournament baseline script (test with 2025 data if available)
2. O-9: Support Claude with tiered escalation wiring (testing)
3. Hardware check: Verify Ollama/qwen2.5:3b can handle tournament load

DAILY HEARTBEAT CHECKS:
- [ ] `ollama ps` shows qwen2.5:3b loaded
- [ ] Last integrity sweep completed without errors
- [ ] DDGS requests succeeding (not rate-limited)
- [ ] Average latency < 30s per 8-game batch

IMMEDIATE TASK (O-6):
Verify V9 integrity verdicts in production:
GET https://cbbbetting-production.up.railway.app/api/predictions/today
Header: X-API-Key: j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg

Check for:
- integrity_verdict field exists in predictions
- No "Sanity check unavailable" errors
- VOLATILE rate reasonable (< 30% of slate)

OUTPUT:
- Update HEARTBEAT.md with daily status
- Report any anomalies immediately
```
