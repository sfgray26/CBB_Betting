# Pre-Tournament Project State Assessment
**Date:** 2026-03-07 (T-11 days to First Four)
**Author:** Claude Code (Master Architect)
**Session:** EMAC-045
**Scope:** Full system review for tournament readiness. March 18 (First Four) is the hard deadline.

---

## Section 1: System Health Dashboard

| Component | Status | Verdict | Notes |
|-----------|--------|---------|-------|
| Railway API and DB | GREEN | Healthy | All syntax errors resolved (EMAC-039/040/042). PostgreSQL connected. 9 tables. |
| V9 Model (SNR + Integrity Kelly) | GREEN | Healthy | SNR scalar, integrity scalar, seed-spread scalar all live. model_version='v9.0'. |
| Env Var Parsing (get_float_env) | GREEN | Healthy | Applied to all 37 float env reads. G-11 + G-12 resolved root cause. Zero plain float(os.getenv) remaining. |
| CI Pipeline (py_compile guard) | GREEN | Healthy | G-12 complete. py_compile step in deploy.yml catches syntax errors before deploy. |
| Analysis Pipeline (nightly runs) | YELLOW | Monitor | Daily runs continue. 0 bets on 12 games is correct conservatism (K-3 confirmed). Watch for pipeline errors. |
| A-26 T2 Seed-Spread Scalars | YELLOW | Needs Key | Code is live and correct (464 tests). No-op until BALLDONTLIE_API_KEY set in Railway. API bug fixed this session (EMAC-045). |
| OpenClaw Integrity Sweep | YELLOW | Monitor | Sweep wired. O-6 spot-check assigned but not yet executed. O-9 tiered escalation NOT wired to analysis.py yet. |
| Calibration State | YELLOW | Monitor | ha=2.419, sd_mult=1.0 (V8-era params). V9-specific recalibration deferred until 50 settled V9 bets. Structural mismatch known and accepted. |

---

## Section 2: Top 3 Pre-Tournament Risks

### Risk 1: BALLDONTLIE_API_KEY not set in Railway by March 16

**Description:** The seed-spread Kelly scalar system (A-26 T2) is fully implemented and tested. However, it is a strict no-op when `BALLDONTLIE_API_KEY` is absent from Railway environment. If Gemini does not add the key before 8 PM ET on March 16 (approximately 2 hours after bracket reveal), the system runs the entire tournament without seed differentiation. Upsets like a 12-seed beating a 5-seed would not trigger the scalar adjustments designed for exactly that scenario.

**Likelihood:** Medium. Gemini has the task. Railway env var changes are low complexity. Risk is scheduling, not technical.

**Impact:** Medium. System degrades gracefully to 1.0x scalar (no crash, no wrong sizing). However, the entire engineering investment in A-26 T2 produces zero tournament-phase value. The 15/16 seed upset-proofing and the 1-seed cover boost both go dark.

**Mitigation:**
- Assign Gemini this task immediately with a hard deadline of March 14 (before bracket to allow verification time).
- Verify via: `GET /api/predictions/today` on March 16 and confirm `seed_home` / `seed_away` fields are non-null in at least one prediction after bracket reveal.
- Fallback: if key is not set by March 18, the system operates safely without seed scalars. No bet will be incorrectly sized upward.

---

### Risk 2: O-9 Tiered Escalation not wired before March 18

**Description:** The coordinator.py escalation rules (route to Kimi for Elite 8+, >= 1.5u sizing, VOLATILE verdicts) exist in code but are not connected to analysis.py. The current pipeline runs OpenClaw integrity checks but the routing layer that would trigger Kimi for high-stakes games is absent. From First Four onward, every game could theoretically benefit from this escalation, but it will not fire.

**Likelihood:** High (it is confirmed unwired as of EMAC-044).

**Impact:** Medium for First Four and Round of 64 (most games are standard BET-tier). High from Sweet 16 onward where game stakes increase and model edge is harder to establish. A VOLATILE verdict on a 1.5u Elite 8 bet currently just applies the 0.5x scalar and bets anyway; Kimi second opinion is never triggered.

**Mitigation:**
- Claude must implement O-9 before March 18. Estimated 2-3 hours of engineering.
- Implementation path is already documented in HANDOFF.md Section 6 with pseudocode.
- Minimum viable version: add the TaskContext routing check in analysis.py Pass 2. Can defer full coordinator.py integration to post-tournament.
- If O-9 misses the deadline: manually review any game flagged VOLATILE before sizing up.

---

### Risk 3: V9 Calibration Mismatch with V8-Era Parameters

**Description:** The recalibration system ran on 663 V8-era bets (no SNR scalar, no integrity scalar, no seed scalar). The resulting ha=2.419 and sd_mult=1.0 reflect a model that compresses edges by approximately 2.8pp relative to the default configuration. V9 adds additional Kelly scalar layers on top of already-compressed sizing. The combined effect is that the system is structurally conservative beyond the design intent: a game with a genuine 3% edge, CONFIRMED integrity, and full SNR still bets at approximately 60-75% of the originally calibrated Kelly fraction.

**Likelihood:** High (structural fact, not a hypothesis).

**Impact:** Medium pre-tournament (missed bets, not wrong bets - capital is preserved). Potentially high during tournament if a sustained run of correct PASS verdicts on genuinely good games causes operator to lose confidence in the system.

**Mitigation:**
- Accept the conservatism for tournament phase. Capital preservation is the primary constraint per IDENTITY.md.
- Track V9-era settled bets. Trigger recalibration at 50 settled V9 bets (whichever comes first: 50 bets or end of tournament).
- Do NOT adjust MIN_BET_EDGE or calibration parameters during the tournament. Lock the model before March 18.
- Document for operator: the system may PASS on games where a V8 model would have bet. This is by design.

---

## Section 3: Team Readiness Matrix

| Agent | Role | Readiness (1-10) | Gaps |
|-------|------|-----------------|------|
| Claude Code | Master Architect: model math, Kelly scaling, risk policy, code guardian | 9/10 | O-9 tiered escalation not yet implemented. One session of engineering needed before March 18. |
| Gemini CLI | DevOps Strike Lead: Railway deploy, env vars, DB migrations, CI | 7/10 | Trust restoring after G-11/G-12 execution. Scope restricted to single-file tasks. BALLDONTLIE_API_KEY Railway task pending. Needs clean execution record through March 15. |
| Kimi CLI | Deep Intelligence: long-context analysis, tournament intel, high-stakes second opinion | 8/10 | K-4 spec complete and implemented. Ready for tiered escalation second opinions when O-9 is wired. No blockers. |
| OpenClaw | Integrity Execution: real-time DDGS + LLM validation, coordinator routing | 7/10 | O-6 spot-check not yet executed. O-8 baseline script not yet created. O-9 wiring depends on Claude. Hardware (Ollama/qwen2.5:3b) not load-tested at tournament volume. |

---

## Section 4: Pre-Tournament Checklist Status

Current status copied from HANDOFF.md Section 10:

| # | Item | Status | Owner | Verification |
|---|------|--------|-------|--------------|
| 1 | System Status Review Complete | DONE | Claude | Assessment doc at reports/2026-03-16-project-state-assessment.md |
| 2 | Team Readiness Confirmed | YELLOW | Claude | All agents below 8/10 readiness threshold. Gaps documented above. |
| 3 | A-26 T2 Spec Reviewed + Implemented | DONE | Claude | 464 tests passing, commit 7ee0207 |
| 4 | BallDontLie API Key Set | OPEN | Gemini | Railway env var BALLDONTLIE_API_KEY needed before March 16 |
| 5 | Seed-Spread Scalar Defaults Verified | DONE | Claude | 0.75/0.75/0.80 confirmed + env var overridable |
| 6 | OpenClaw Tiered Escalation Wired | OPEN | Claude | O-9 not yet implemented in analysis.py |
| 7 | Pre-Tournament Baseline Script Ready | OPEN | OpenClaw | O-8 script not yet created |
| 8 | Line Movement Monitor Deployed | OPEN | Gemini | O-10 not yet in production |
| 9 | Railway Deploy Pipeline Tested | OPEN | Gemini | Deploy time not yet verified |
| 10 | Rollback Plan Documented | DONE | Claude | Documented in Section 6 of this report |

Current green items: 1, 3, 5, 10 (4 of 10). Target: 10 of 10 by March 18.

---

## Section 5: Critical Path (March 7 to March 18)

```
NOW (March 7 — this session):
  - BallDontLie API bug FIXED (Task 1 complete, EMAC-045)
    Endpoint corrected to /bracket
    Field corrected to name
    Season offset corrected to year-1
  - Assessment report created (this document)
  - Rollback plan documented
  - HANDOFF.md updated to EMAC-045

March 7-9 (Claude — highest priority engineering):
  - O-9: Wire tiered escalation in analysis.py
    Connect coordinator.py routing to Pass 2 loop
    Test: game with recommended_units >= 1.5 triggers Kimi route
    Test: VOLATILE verdict triggers Kimi route
    Estimated: 2-3 hours. Must complete before March 18.

March 7-10 (Gemini — parallel track):
  - Add BALLDONTLIE_API_KEY to Railway env vars
    Verify env var appears in Railway UI
    Confirm no syntax issues in value (plain string, no spaces)
    Hard deadline: March 14

March 10-15 (Regular season window — monitoring mode):
  - Daily check: nightly analysis runs without errors
  - OpenClaw: O-6 spot-check (verify integrity_verdict fields in prod predictions)
  - OpenClaw: Create O-8 pre-tournament baseline script
    Test with any available team data
    Output format: JSON per team with seed, risk_level, summary
  - Gemini: Verify Railway deploy pipeline timing (< 10 min target)
  - Claude: Weekly A-27 calibration review (Friday March 13)
    Check MAE vs V8 baseline
    Document in memory/2026-03-13-calibration.md

March 14 (HARD GATE):
  - BALLDONTLIE_API_KEY must be in Railway by this date
  - O-9 must be merged and deployed
  - 464+ tests must still pass

March 16 (Bracket Day):
  - 6 PM ET: Bracket revealed
  - 8 PM ET: BallDontLie bracket data available (API typically lags 2h)
  - Trigger: python -c "from backend.services.tournament_data import fetch_tournament_bracket; print(fetch_tournament_bracket(2026))"
  - Verify: seed_map contains >= 60 teams
  - Trigger manual analysis via POST /admin/run-analysis
  - Verify in logs: seed_home and seed_away non-null for tournament games
  - Verify: seed_spread_kelly_scalar != 1.0 for at least one matchup
  - OpenClaw: Execute O-8 pre-tournament baseline (all 68 teams)
    Output: data/pre_tournament_intel.json
    Generate markdown health report

March 17 (Pre-tournament verification day):
  - Review O-8 baseline output — flag high-risk teams
  - Confirm all 10 checklist items green
  - Lock model configuration (no parameter changes after this point)
  - Confirm Railway health: GET /health returns 200

March 18 (First Four):
  - Tournament mode active
  - All 10 checklist items green (required)
  - O-9 tiered escalation firing for high-stakes games
  - O-10 line movement monitor running (if Gemini completed)
  - OpenClaw processing integrity sweep concurrently with analysis

SINGLE POINT OF FAILURE TASKS:
  - O-9 wiring (Claude): if this misses, no tiered escalation for tournament
  - BALLDONTLIE_API_KEY (Gemini): if this misses, seed scalars are dark

PARALLELIZABLE RIGHT NOW:
  - Claude: O-9 engineering
  - Gemini: BALLDONTLIE_API_KEY + Railway deploy pipeline test
  - OpenClaw: O-6 spot-check + O-8 script creation

CONTINGENCY (T-24h, March 17):
  - If O-9 not complete: Claude implements minimum viable version (no coordinator.py, just threshold check + Kimi call)
  - If BALLDONTLIE_API_KEY missing: Run tournament without seed scalars. System is safe. Document as known limitation.
  - If O-8 not complete: Enter tournament without pre-loaded baseline. OpenClaw runs reactive checks only.
  - If Railway deploy pipeline slow: Gemini identifies bottleneck, Claude reviews before bracket day
```

---

## Section 6: Rollback Plan

The following procedures allow reverting to a known-safe state at any point during the tournament.

### Safe Pre-Tournament Commit Reference

The last verified-stable commit before A-26 T2 implementation is commit `8edff01` (HANDOFF EMAC-044 update). The A-26 T2 implementation itself was committed at `7ee0207`.

For full code rollback (nuclear option): `git revert HEAD` or `git checkout 8edff01 -- backend/services/tournament_data.py backend/betting_model.py backend/services/analysis.py`

### Disabling Seed Scalars Without Code Changes (Preferred)

Set `BALLDONTLIE_API_KEY=` (empty string) in Railway environment variables. With no API key, `fetch_bracket_data()` returns `{}` immediately, and all seed lookups return `None`. The scalar in `betting_model.py` falls back to `1.0x` when seeds are unavailable. System behaves identically to pre-A-26-T2 state.

Verification: After setting empty key, trigger analysis and confirm log line: "BALLDONTLIE_API_KEY not set -- skipping seed fetch"

### Disabling Tournament SD Bump Without Code Changes

Set `TOURNAMENT_MODE_SD_BUMP=1.0` in Railway environment variables. This overrides the 1.15x neutral-site SD multiplier to a no-op 1.0x. Standard deviation returns to base calculation without tournament adjustment.

### Disabling O-9 Tiered Escalation (If Wired and Causing Issues)

Set `INTEGRITY_ESCALATION_ENABLED=false` in Railway environment variables (or whatever env guard is implemented in O-9). If the guard is not implemented, the fastest path is: `git checkout 7ee0207 -- backend/services/analysis.py` to restore the pre-O-9 analysis.py.

### Restoring 438-Test Clean State

If the test suite regresses during tournament prep: `git checkout 3281b51 -- backend/services/tournament_data.py`. This restores the file to the pre-bugfix state. Then reapply the three targeted fixes from EMAC-045 (endpoint, field name, season offset).

### Emergency Full Rollback Sequence

```
1. git stash (save any local changes)
2. git checkout 8edff01
3. git push --force-with-lease origin main
4. Railway auto-deploys from main
5. Verify: GET /health returns 200 and model_version is still v9.0
6. All 438 pre-A26T2 tests pass
```

Note: Force push is required because this is moving backward in history. Only use in true emergency. Coordinate with Gemini before executing.

---

## Appendix: Files Modified This Session (EMAC-045)

| File | Change |
|------|--------|
| backend/services/tournament_data.py | Endpoint /march_madness_bracket -> /bracket; season year-1 offset; field full_name -> name |
| tests/test_tournament_data.py | Mock JSON updated to use name field; assertions updated to match shorter names |
| reports/2026-03-16-project-state-assessment.md | This document (created) |
| HANDOFF.md | Updated to EMAC-045 (see Task 3) |
