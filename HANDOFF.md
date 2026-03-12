# OPERATIONAL HANDOFF (EMAC-068)

> Ground truth as of March 12, 2026 (test-validated). Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full roadmap: `docs/MLB_FANTASY_ROADMAP.md` · CBB plan: `tasks/cbb_enhancement_plan.md`

---

## 0. STANDING DECISIONS

- **Gemini CLI is Research-Only.** No production code. Deliverables go to `docs/` as markdown.
- **All production code: Claude Code only.**
- **GUARDIAN (Mar 18 - Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services during tournament window. EMAC-068 pre-tournament fixes are now COMPLETE and were the final allowed changes.

---

## EMAC-068: Pre-Tournament Fixes (COMPLETE — March 12, 2026)

Three targeted changes made before the March 18 guardian window:

### Fix 1 — EVANMIYA_DOWN_SE_ADDEND default 0.30 -> 0.00
- **File:** `backend/betting_model.py`, `_compute_margin_se()`, line 763
- **Effect:** `margin_se` drops from 1.80 back to 1.50 in normal 2-source operation.
  Conservative CI narrows; CONSIDER -> BET promotions expected on genuine edges.
- **Env override still available:** Set `EVANMIYA_DOWN_SE_ADDEND=0.30` in Railway to restore
  the penalty if EvanMiya is restored and then goes down unexpectedly.

### Fix 2 — FORCE_ env var overrides in analysis.py
- **File:** `backend/services/analysis.py`, lines 757-793
- **Effect:** Setting `FORCE_HOME_ADVANTAGE` or `FORCE_SD_MULTIPLIER` in Railway now
  bypasses the DB-calibrated values immediately (no DB operation required).
- **Normal mode:** Leave both blank. DB calibration continues as before.
- **Log signal:** `[FORCED]` suffix appears in the "Model initialised" log line when active.

### Fix 3 — .env.example documentation
- **File:** `.env.example`
- `EVANMIYA_DOWN_SE_ADDEND` default updated to `0.00`
- New section added at bottom: `FORCE_HOME_ADVANTAGE=`, `FORCE_SD_MULTIPLIER=`

### Test suite
- 2 tests updated in `tests/test_betting_model.py`:
  - `test_evanmiya_down_raises_se`: updated expected value 1.80 -> 1.50
  - `test_fully_degraded_raises_se`: updated expected value 2.10 -> 1.80
- All other SE tests verified safe under new math (monotone ordering still holds, cap still holds)
- Run: `python -m pytest tests/ -q --ignore=tests/test_integrity_sweep.py`
  **Confirmed:** 647 pass, 3 pre-existing DB-auth failures (validated March 12, 2026)

---

## 1. SYSTEM STATUS

### CBB Model — V9.1 (CALIBRATION DRIFT — see Section 4)

| Component | Status |
|-----------|--------|
| Railway API | ✅ Healthy |
| PostgreSQL | ✅ Connected (365 teams) |
| Scheduler | ✅ 10 jobs running |
| Discord | ✅ 16 channels operational |
| V9.1 Model | ⚠️ Over-conservative (SE fix applied EMAC-068; ha/sd_mult post-tournament) |
| Test suite | ✅ 647/650 pass (3 pre-existing DB-auth failures) |
| Dedup fix | ✅ COMPLETE — `run_tier` NULL matching fixed in analysis.py |

### Fantasy Baseball — DRAFT-READY

| Component | Status | File |
|-----------|--------|------|
| Yahoo OAuth + Draft Board + Live Tracker | ✅ COMPLETE | `11_Fantasy_Baseball.py`, `12_Live_Draft.py` |
| Draft Tracker backend + Discord alerts | ✅ COMPLETE | `draft_tracker.py`, `discord_notifier.py` |
| Bet settlement fix + re-settlement | ✅ COMPLETE | `bet_tracker.py` — `_resolve_home_away()` |

---

## 2. UPCOMING DEADLINES

| Date | Event | Owner | Action Required |
|------|-------|-------|----------------|
| **Mar 16 ~9 PM ET** | O-8 Baseline Execution | OpenClaw | `python scripts/openclaw_baseline.py --year 2026` |
| **Mar 17 ~7 PM ET** | O-9 Pre-tournament sweep | OpenClaw | See Section 6 |
| **Mar 18** | First Four begins | All | Tournament monitoring mode |
| **Mar 20** | Fantasy Keeper Deadline | User | Set keepers in Yahoo UI |
| **Mar 23 7:30am ET** | Fantasy Draft Day | User | Run `12_Live_Draft.py` |
| **Apr 7** | Tournament window closes | All | Guardian lifts; V9.2 work begins |

---

## 3. ROOT CAUSE: WHY THE MODEL HAS A POOR WIN RECORD

**Short answer:** V9.1 stacks 3 Kelly compression layers that were not present when `sd_mult` and `ha` were calibrated. The model now requires far more raw edge to emit a BET verdict than it was designed for, so it under-bets or emits CONSIDER instead of BET on genuine edges.

### The Compression Stack

| Layer | Value | Effect |
|-------|-------|--------|
| Fractional Kelly divisor | ÷2.0 | Half-Kelly baseline |
| SNR scalar | ×0.5–1.0 (avg ~0.7 with 2-source) | Effective divisor ~2.86 |
| Integrity scalar | ×0.5–1.0 (avg ~0.85) | Effective divisor ~3.37 |
| Conservative CI edge | Lower 2.5th pct, not point estimate | Requires ~6–8% raw edge for 2.5% conservative edge |
| `ha = 2.419` post-recalibration | 21.7% below 3.09 baseline | Understates home team margin |
| `sd_mult = 1.0` post-recalibration | 17.6% wider SD than default | Wider CI → fewer edges clear threshold |

**Net effect:** A game with genuine 4% model edge might emit `edge_conservative = 0.8%` → CONSIDER (not BET). The V9.1 scalars were added AFTER the 663-bet calibration dataset, so they compound on already-conservative params.

### Contributing Factors

1. **2-source mode** — EvanMiya intentionally excluded. ~~`EVANMIYA_DOWN_SE_ADDEND = 0.30`~~ → fixed to `0.00` in EMAC-068. `margin_se` now 1.50 (was 1.80). This factor is resolved.
2. **No CLV feedback loop** — We don't know if we're actually beating the closing line. Without this we can't distinguish "model edge is real but too compressed" from "model edge is noise."
3. **Possession simulator unvalidated** — `possession_sim.py` (947 lines) integrated but accuracy vs CLV never measured.

---

## 4. ACTIVE MISSIONS

### Claude Code — EMAC-067 (post-tournament, Apr 7+)

Pre-tournament: only the dedup fix was safe (done). Full recalibration must wait until after Apr 7.

**After Apr 7 — in order:**
1. **V9.2 recalibration** — implement Kimi's K-11/K-12 recommendations (see below). Adjust `MIN_BET_EDGE`, `BASE_MARGIN_SE`, and reset `ha`/`sd_mult` to V9-appropriate values. Target: BET rate improves from ~2% to ~8–12%.
2. **EvanMiya replacement** — wire Gemini's G-R7 findings to restore 3-source composite.
3. **Possession simulator A/B** — implement Kimi's K-13 recommendation (remove or keep).

### Kimi CLI — Critical Intelligence Missions

#### K-11: CLV Performance Attribution (START IMMEDIATELY)
```
MISSION K-11: Real CLV and edge bucket analysis

Read the database via scripts that query BetLog + ClosingLine tables.
Look at scripts/resettle_bets.py as a reference for DB connection pattern.

QUESTIONS TO ANSWER:
1. What is our mean CLV (closing line value) across all settled bets?
   - CLV > 0 means we beat the closing line (genuine edge exists)
   - CLV < 0 means market corrected against us (model finds noise, not signal)
2. By edge bucket (0-3%, 3-6%, 6%+): win rate and CLV in each bucket?
3. By conference: which conferences are profitable? Which are losses?
4. By game type: neutral site vs home game — win rate difference?
5. How many BET verdicts per week over the last 60 days? Is frequency too low?
6. What is our actual win rate vs expected win rate for each edge bucket?

Also look at reports/BETTING_HISTORY_AUDIT_MARCH_2026.md for prior audit findings.

DELIVERABLE: reports/K11_CLV_ATTRIBUTION_MARCH_2026.md
Due: March 16
```

#### K-12: V9.1 Recalibration Parameter Recommendation
```
MISSION K-12: Recalibration parameters for V9.2

CONTEXT:
- V9.1 added SNR scalar (avg ~0.70 in 2-source mode) and integrity scalar (~0.85)
- These were NOT present in the 663-bet calibration dataset
- Current params: sd_mult=1.0, ha=2.419 (calibrated for V8 with Kelly divisor=2.0)
- Effective V9.1 Kelly divisor: 2.0 / 0.70 / 0.85 = ~3.36
- Result: model over-conservative, emits CONSIDER on genuine BET opportunities

DERIVE: What should the V9.2 parameters be?
1. Given typical SNR=0.70 and integrity=0.85 in 2-source mode, what sd_mult
   preserves the SAME betting frequency as the V8 calibration?
   Hint: V8 sd_mult=0.85 with Kelly divisor=2.0.
   V9.2 should target sd_mult such that Kelly divisor 2.0 × SNR × integrity
   produces the same effective sizing as before.
2. Is ha=2.419 correct or is it an overcorrection? Compare to KenPom's published
   home court advantage estimates (~3.0-3.5 for D1 average).
3. What MIN_BET_EDGE value (currently 2.5%) makes sense given the wider CI?
   If the model needs 6% raw edge to produce 2.5% conservative edge, we may
   want to lower MIN_BET_EDGE to 1.5% or raise the margin_se ceiling.
4. ~~EVANMIYA_DOWN_SE_ADDEND penalty~~ — already resolved in EMAC-068 (default set to 0.00). Skip this question.

DELIVERABLE: reports/K12_RECALIBRATION_SPEC_V92.md
Include: exact parameter values to change, justification, expected betting frequency impact
Due: March 17
```

#### K-13: Possession Simulator Validation
```
MISSION K-13: Should possession_sim.py stay or go?

backend/possession_sim.py (947 lines) is integrated into the analysis pipeline.
Its accuracy vs the ratings-path has never been measured against actual outcomes.

TASK:
1. Read possession_sim.py — what does it contribute to margin calculation?
2. Compare: games where possession sim was used vs not (check analysis logs or
   model output fields for sim_used flag if it exists)
3. Run any offline backtests possible with the data we have
4. Recommendation: keep (with evidence it helps), tune (specific params), or remove

DELIVERABLE: reports/K13_POSSESSION_SIM_AUDIT.md
Due: March 18 (before tournament — if it's adding noise, we remove it pre-tournament)
```

### Gemini CLI — Research Missions

| Mission | Task | Deliverable | Priority |
|---------|------|-------------|----------|
| **G-R7** | **EvanMiya replacement — what 3rd rating source can we add?** Research: ESPN BPI, Sagarin, T-Rank (torvik.com/trank), Massey Ratings. Which is free/scrapeable? Which correlates best with CBB outcomes? | `docs/THIRD_RATING_SOURCE.md` | **HIGH — do first** |
| G-R1 | Steamer 2026 full download | `docs/PROJECTION_DATA_SOURCES.md` | Medium |
| G-R2 | Daily MLB lineup sources | `docs/LINEUP_CONFIRMATION_SOURCES.md` | Medium |
| G-R3 | Closer situations monitor | `docs/CLOSER_SITUATION_SOURCES.md` | Medium |
| G-R4 | Statcast bulk data | `docs/STATCAST_API_GUIDE.md` | Low |
| G-R5 | Yahoo Fantasy API XML format | `docs/YAHOO_API_REFERENCE.md` | Low |
| G-16 | Verify O-10 line monitor post-deploy | Report to HANDOFF | Medium |

### OpenClaw — Mission O-9: Pre-Tournament Sweep

**Run on March 17, 2026 ~7 PM ET:**
```bash
ls data/pre_tournament_baseline_2026.json  # If missing: python scripts/openclaw_baseline.py --year 2026
python scripts/test_discord.py             # Verify Discord bot
# GET /admin/odds-monitor/status           # Expect games_tracked > 0
```
For each First Four matchup: run `check_integrity_heuristic()`. Flag ABORT or VOLATILE here.

---

## 5. HANDOFF PROMPT — NEXT CLAUDE SESSION (post-Apr 7)

```
CONTEXT (April 7+, post-tournament):
- Guardian window lifted. CBB model work can resume.
- V9.1 has a calibration mismatch — over-conservative due to SNR+integrity scalar stacking.
- Kimi delivered K-11 (CLV attribution), K-12 (recalibration spec), K-13 (possession sim).
- Gemini delivered G-R7 (3rd rating source research).

MISSION EMAC-068: V9.2 Recalibration
1. Read reports/K12_RECALIBRATION_SPEC_V92.md — implement new sd_mult, ha, MIN_BET_EDGE
2. Read reports/K11_CLV_ATTRIBUTION_MARCH_2026.md — validate recalibration direction
3. Read docs/THIRD_RATING_SOURCE.md — wire in new 3rd source to restore 3-source composite
4. Read reports/K13_POSSESSION_SIM_AUDIT.md — keep or remove possession_sim.py
5. Run full test suite. Bump model_version to 'v9.2'.

TARGET: BET frequency increases from ~2% to ~8-12% of games analyzed.
Winning record requires genuine CLV > 0 (beat closing line) — verify this with K-11.

GUARDIAN: Tournament window is over. Normal dev protocols resume.
Run `python -m pytest tests/ -q` before any commit.
```

---

## 6. QUICK REFERENCE

```bash
python -m pytest tests/ -q
python scripts/preflight_check.py
python scripts/test_discord.py
railway logs --follow
streamlit run dashboard/app.py
```

---

## 7. HIVE WISDOM

| Lesson | Source |
|--------|--------|
| V9.1 Kelly stack: SNR×integrity×fractional = effective divisor ~3.4× — far too conservative for V8 params | EMAC-067 |
| EvanMiya SE penalty (+0.30) should be REMOVED if EvanMiya is intentionally excluded, not broken | EMAC-067 |
| CLV > 0 = genuine edge. CLV < 0 = no edge, no amount of model tuning will fix it | EMAC-067 |
| ha=2.419 (post-recalib) vs 3.09 baseline — 21.7% reduction may be overcorrection | K-3 audit |
| sd_mult=1.0 (post-recalib) = 17.6% wider SD than default 0.85 | K-3 audit |
| Bet settlement: use _resolve_home_away() — never raw string compare | EMAC-064 |
| Yahoo roster pre-draft returns players:[] (empty array) — handle gracefully | EMAC-063 |
| Prediction dedup: run_tier NULL causes duplicate rows — use or_() filter | EMAC-067 |
| Conference HCA: Big Ten 3.6 pts vs SWAC 1.5 pts = significant road differential | P2 |
| Sharp money: steam >=1.5 pts in <30 min = high confidence signal | P1 |
| BartTorvik public CSV needs no auth (cloudscraper only) | P0 |
| EvanMiya intentionally dropped — 2-source mode robust by design | P0 |
| Discord token must be in Railway Variables, not just .env | D-1 |
| Avoid non-ASCII chars in output strings (CP-1252 Windows terminal issue) | Python |

---

**Document Version:** EMAC-068
**Last Updated:** March 12, 2026
**Status:** All pre-tournament fixes COMPLETE and test-validated (647/650). Fantasy draft-ready. Root cause of poor win record identified (V9.1 calibration mismatch). Kimi assigned K-11/K-12/K-13. Gemini assigned G-R7. Guardian window opens Mar 18. Next Claude session: post-Apr 7 V9.2 recalibration.


---

## NEW: OPCL-001 OpenClaw Discord Enhancement � IMPLEMENTED

**Status:** Phase 1 COMPLETE ?  
**Date:** March 12, 2026  
**Files Created:**

### New Modules
| File | Purpose | Lines |
|------|---------|-------|
| ackend/services/openclaw_briefs.py | Morning brief generation | ~400 |
| ackend/services/openclaw_telemetry.py | Quiet system monitoring | ~450 |
| scripts/openclaw_scheduler.py | Cron integration | ~150 |
| 	ests/test_openclaw_briefs.py | Unit tests | ~200 |
| 	ests/test_openclaw_telemetry.py | Unit tests | ~250 |

### Discord Functions Added
- send_openclaw_morning_brief(embed) ? #openclaw-briefs
- send_openclaw_telemetry(embed) ? #openclaw-health  
- send_openclaw_live_alert(...) ? #openclaw-escalations

### Usage

**Morning Brief (Daily 7 AM ET):**
`ash
python scripts/openclaw_scheduler.py --morning-brief
`

**Telemetry Check (Every 30 min, quiet mode):**
`ash
python scripts/openclaw_scheduler.py --telemetry-check
`

**Force daily summary:**
`ash
python scripts/openclaw_scheduler.py --telemetry-check --force-summary
`

**Test mode (no Discord send):**
`ash
python -m backend.services.openclaw_briefs --test
python -m backend.services.openclaw_telemetry --test
`

### Design Philosophy: Clean & Quiet

- **Morning Brief:** One comprehensive message per day at 7 AM
- **Telemetry:** Only alerts when anomalies detected; daily summary optional
- **No Noise:** Healthy systems don't generate messages
- **Actionable:** Every message contains useful intelligence

### Testing Checklist (Before March 18)

- [ ] Run python scripts/openclaw_scheduler.py --morning-brief --test
- [ ] Run python scripts/openclaw_scheduler.py --telemetry-check --test
- [ ] Verify Discord embeds look correct
- [ ] Add to Railway scheduler or cron
- [ ] Monitor first few runs for errors

### Impact Target

| Metric | Before | After |
|--------|--------|-------|
| Daily intelligence | 0 | 1 morning brief |
| System visibility | Manual | Automated telemetry |
| Alert latency | Hours | <30 minutes |
| Operator time saved | � | ~15 min/day |

**Next:** Phase 2 (Live Monitor) for March 19-25 tournament

