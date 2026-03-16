# OPERATIONAL HANDOFF (EMAC-069)

> Ground truth as of March 13, 2026. Operator: Claude Code (Master Architect).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Full roadmap: `docs/MLB_FANTASY_ROADMAP.md` · CBB plan: `tasks/cbb_enhancement_plan.md`

---

## 0. STANDING DECISIONS

- **Gemini CLI is Research-Only.** No production code. Deliverables go to `docs/` as markdown.
- **All production code: Claude Code only.**
- **GUARDIAN (Mar 18 - Apr 7):** Do NOT touch `betting_model.py`, `analysis.py`, or CBB services. All pre-tournament fixes are COMPLETE — no further changes before Apr 7.

---

## 1. SYSTEM STATUS

### CBB Model — V9.1 (recalibration queued for Apr 7)

| Component | Status |
|-----------|--------|
| Railway API | ✅ Healthy |
| PostgreSQL | ✅ Connected (365 teams) |
| Scheduler | ✅ 10 jobs running |
| Discord | ⚠️ Channels operational but key jobs NOT sending notifications (see Section 8) |
| Test suite | ✅ 683/686 pass (3 pre-existing DB-auth failures) |
| V9.1 Model | ⚠️ Over-conservative — MIN_BET_EDGE lowered to 1.8%; ha/sd_mult/SNR queued Apr 7 |
| Haslametrics scraper | ✅ Built — `backend/services/haslametrics.py`, 12 tests. Wire in Apr 7. |
| Tournament bracket | ✅ Data available but NO Discord notification sent (Mar 16) |

### Fantasy Baseball — DRAFT-READY

| Component | Status | File |
|-----------|--------|------|
| Yahoo OAuth + Draft Board + Live Tracker | ✅ COMPLETE | `11_Fantasy_Baseball.py`, `12_Live_Draft.py` |
| Draft Tracker backend + Discord alerts | ✅ COMPLETE | `draft_tracker.py`, `discord_notifier.py` |
| Bet settlement fix | ✅ COMPLETE | `bet_tracker.py` — `_resolve_home_away()` |

---

## 2. UPCOMING DEADLINES

| Date | Event | Owner | Action Required |
|------|-------|-------|----------------|
| **Mar 17 ~7 PM ET** | O-9 Pre-tournament sweep | OpenClaw | See Section 4 |
| **Mar 18** | First Four begins | All | Tournament monitoring mode — GUARDIAN active |
| **Mar 20** | Fantasy Keeper Deadline | User | Set keepers in Yahoo UI |
| **Mar 23 7:30am ET** | Fantasy Draft Day | User | Run `12_Live_Draft.py` |
| **Apr 7** | Tournament window closes | Claude Code | Guardian lifts — execute Section 5 mission |

---

## 3. PRE-TOURNAMENT WORK LOG

All work below is shipped and test-validated. Full details in `reports/` and git history.

| Mission | What was done | Status |
|---------|---------------|--------|
| EMAC-068 Fix 1 | `EVANMIYA_DOWN_SE_ADDEND` default 0.30→0.00; `margin_se` back to 1.50 | ✅ |
| EMAC-068 Fix 2 | `FORCE_HOME_ADVANTAGE` / `FORCE_SD_MULTIPLIER` env var overrides in analysis.py | ✅ |
| EMAC-068 Phase 1 | `MIN_BET_EDGE` default 2.5%→1.8% in betting_model.py, analysis.py, line_monitor.py | ✅ |
| OPCL-001 | OpenClaw morning brief + telemetry Discord modules (24 tests pass) | ✅ |
| EMAC-069 | Haslametrics scraper `backend/services/haslametrics.py` (12 tests pass) | ✅ |
| K-11 | CLV attribution — positive CLV confirmed (+0.5–1.0 pts). Full report: `reports/K11_CLV_ATTRIBUTION_MARCH_2026.md` | ✅ |
| K-12 | V9.2 recalibration spec — sd_mult→0.80, ha→2.85, SNR floor→0.75. Full report: `reports/K12_RECALIBRATION_SPEC_V92.md` | ✅ |
| K-13 | Possession sim audit — KEEP verdict (push-aware Kelly, 24 tests). K-14 A/B monitoring post-Apr 7. Full report: `reports/K13_POSSESSION_SIM_AUDIT.md` | ✅ |
| G-R7 | Haslametrics recommended as EvanMiya replacement. Scraper already built. Full spec: `docs/THIRD_RATING_SOURCE.md` | ✅ |
| G-R1–R5 | MLB research complete (Steamer, lineups, closers, Statcast, Yahoo API). Docs in `docs/` | ✅ |

**Why the model has been over-conservative:** V9.1 stacks SNR scalar (~0.70) × integrity scalar (~0.85) × fractional Kelly (÷2.0) = effective divisor ~3.37×, applied on top of V8 params that were calibrated at ÷2.0. MIN_BET_EDGE fix (Phase 1) partially addressed this. Full fix is Phase 2 (Apr 7+).

---

## 4. ACTIVE TASKS

### OpenClaw — before March 18

- [ ] `python scripts/openclaw_scheduler.py --morning-brief --test` — verify Discord embeds
- [ ] `python scripts/openclaw_scheduler.py --telemetry-check --test`
- [ ] Add openclaw_scheduler to Railway cron (daily 7 AM + every 30 min)
- [ ] **O-9 sweep (Mar 17 ~7 PM ET):**
  ```bash
  ls data/pre_tournament_baseline_2026.json  # if missing: python scripts/openclaw_baseline.py --year 2026
  python scripts/test_discord.py
  # GET /admin/odds-monitor/status  — expect games_tracked > 0
  ```
  For each First Four matchup: run `check_integrity_heuristic()`. Flag ABORT or VOLATILE.

### Claude Code — April 7+

Execute in order. Run `pytest tests/ -q` before each commit.

1. **V9.2 Phase 2 params** — in `betting_model.py` / `analysis.py`:
   - `sd_mult` 1.0 → 0.80
   - `ha` 2.419 → 2.85
   - `SNR_KELLY_FLOOR` 0.50 → 0.75
   - Reference: `reports/K12_RECALIBRATION_SPEC_V92.md`

2. **Wire Haslametrics** — scraper already built at `backend/services/haslametrics.py`:
   - Add `from backend.services.haslametrics import get_haslametrics_ratings` to `ratings.py`
   - Assign EvanMiya's former 32.5% weight to Haslametrics in `CBBEdgeModel.weights`
   - Reference: `docs/THIRD_RATING_SOURCE.md`

3. **K-14 pricing engine tracking** — in `analysis.py` + DB migration:
   - Add `pricing_engine` column to `Prediction` model (values: `"markov"` / `"gaussian"`)
   - Write field per-prediction in analysis pipeline
   - Reference: `reports/K13_POSSESSION_SIM_AUDIT.md`

4. **Bump version + validate** — set `model_version = 'v9.2'`, run full test suite, confirm BET rate improvement.

---

## 5. NEXT CLAUDE SESSION PROMPT (post-Apr 7)

```
CONTEXT: Guardian window lifted. CBB model work resumes. All intelligence is in.

STATE:
- V9.1 is over-conservative (effective Kelly divisor ~3.37x vs intended ~2.0x)
- MIN_BET_EDGE already lowered to 1.8% (Phase 1, pre-tournament)
- Haslametrics scraper already built: backend/services/haslametrics.py (12 tests pass)
- K-11 confirms genuine positive CLV — recalibration is directionally correct

MISSION EMAC-070: V9.2 Recalibration + Haslametrics
1. betting_model.py / analysis.py: sd_mult 1.0→0.80, ha 2.419→2.85, SNR_KELLY_FLOOR 0.50→0.75
   Read reports/K12_RECALIBRATION_SPEC_V92.md for exact justification
2. ratings.py: wire backend/services/haslametrics.py as 3rd source (32.5% weight, replaces EvanMiya)
   Read docs/THIRD_RATING_SOURCE.md for integration spec
3. analysis.py + models.py: add pricing_engine field to Prediction, write "markov"/"gaussian" per game
   Read reports/K13_POSSESSION_SIM_AUDIT.md for K-14 spec
4. Bump model_version to 'v9.2'. Run pytest tests/ -q. Confirm BET rate increase.

TARGET: BET rate 3% → 8-12%. CLV already positive (K-11) — just need to unblock the bets.
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
| V9.1 effective Kelly divisor ~3.37× — calibrated params were for ÷2.0 | EMAC-067 |
| CLV > 0 = genuine edge. No amount of tuning fixes CLV < 0 | K-11 |
| Haslametrics uses play-by-play garbage-time filter — cleaner than EvanMiya | G-R7 |
| MIN_BET_EDGE 2.5% was too high given wide CI — 1.8% is the right pre-v9.2 value | K-12 |
| possession_sim: push-aware Kelly is worth keeping; add A/B monitoring not removal | K-13 |
| Bet settlement: use `_resolve_home_away()` — never raw string compare | EMAC-064 |
| Yahoo roster pre-draft returns `players:[]` (empty array) — handle gracefully | EMAC-063 |
| Prediction dedup: `run_tier` NULL causes duplicate rows — use `or_()` filter | EMAC-067 |
| Discord token must be in Railway Variables, not just .env | D-1 |
| Avoid non-ASCII chars in output strings (CP-1252 Windows terminal issue) | Python |

---

## 8. DISCORD ISSUES — IMMEDIATE FIX NEEDED (March 16, 2026)

### 🔴 Critical Finding: Notifications NOT Being Sent

**Full audit:** `reports/DISCORD_TOURNAMENT_AUDIT_MARCH_2026.md`

| Feature | Expected | Actual | Fix Required |
|---------|----------|--------|--------------|
| Morning Brief | Daily 7 AM ET to #openclaw-briefs | Job runs but only logs — NO DISCORD SEND | Fix `_morning_briefing_job()` in main.py |
| Tournament Bracket | Alert when released (Mar 16) | No notification sent | Create bracket notifier |
| End-of-Day Results | Daily 11 PM ET | Not scheduled | Add `_end_of_day_results_job()` to scheduler |

### Root Causes

1. **Morning Brief Job Only Logs** — `backend/main.py` function `_morning_briefing_job()` queries DB and generates narrative but never calls `send_morning_brief()` or any Discord function.

2. **Tournament Bracket Silent** — `backend/services/tournament_data.py` fetches bracket data but no Discord notification is triggered on release.

3. **Scheduler Not Wired** — `scripts/openclaw_scheduler_improved.py` exists but is NOT scheduled in Railway cron. The improvements (bet embeds, morning brief, end-of-day) are not running.

### Claude Fixes Required (Pre-Tournament, Mar 17)

**File: `backend/main.py`**

```python
# Fix 1: Morning briefing must send Discord
def _morning_briefing_job():
    # ... existing code ...
    from backend.services.openclaw_briefs_improved import generate_and_send_morning_brief_improved
    generate_and_send_morning_brief_improved()  # ADD THIS LINE

# Fix 2: Add end-of-day results job
def _end_of_day_results_job():
    """Send end-of-day results to Discord at 11 PM ET."""
    from backend.services.discord_bet_embeds import create_daily_results_embed
    from backend.services.discord_notifier import send_to_channel
    # Query BetLog for today's results, send embed to #cbb-bets

# Fix 3: Add tournament bracket notifier
def _tournament_bracket_job():
    """Send bracket release notification."""
    # Check if bracket newly available, send First Four matchups to Discord
```

**Schedule the new jobs:**
```python
# In scheduler setup (around line 188 in main.py):
scheduler.add_job(
    _end_of_day_results_job,
    CronTrigger(hour=23, minute=0, timezone=timezone),
    id="end_of_day_results",
    name="End of Day Results",
    replace_existing=True,
)
```

### Test Commands

```bash
# Test morning brief
python scripts/openclaw_scheduler_improved.py --morning-brief

# Test end of day
python scripts/openclaw_scheduler_improved.py --end-of-day

# Test all channels
python scripts/openclaw_scheduler_improved.py --test
```

---

## 9. ORIGINAL DISCORD IMPROVEMENTS (March 13, 2026)

### Problem Report
User reported:
1. Notification ID `1481976070243876954` was useless — didn't show actual picks
2. No morning briefing being sent

### Root Causes
1. **Bet notifications** showed summary only ("X bets found") without listing the actual picks
2. **Morning brief** had TODOs for data collection — never wired to real database
3. **Line monitor alerts** were too verbose with multiple scenarios

### Improvements Delivered

#### 8.1 New Discord Bet Embeds (`backend/services/discord_bet_embeds.py`)
- **Summary embed**: Shows ALL bets in ONE message with:
  - Numbered list: `1. Duke -4.5 (3.2% edge, 1.25u)`
  - Total exposure
  - Clear formatting
- **Detailed embeds**: For high-stakes bets (≥1.0u) with full analysis
- **BET NOW alerts**: For line movement opportunities
- **Daily results**: End-of-day P&L summary

#### 8.2 Improved Morning Brief (`backend/services/openclaw_briefs_improved.py`)
- Actually queries the database for real data
- Shows:
  - Today's bet count with avg edge and total units
  - Yesterday's results (wins/losses/P&L)
  - Tournament countdown
  - High-stakes highlight

#### 8.3 Improved Scheduler (`scripts/openclaw_scheduler_improved.py`)
New tasks:
- `--morning-brief`: Daily 7 AM ET brief
- `--daily-picks`: Send all picks after nightly analysis
- `--end-of-day`: 11 PM ET results summary
- `--line-monitor`: BET NOW alerts for line moves
- `--test`: Verify all channels working

### Deployment Instructions

```bash
# Test all channels
python scripts/openclaw_scheduler_improved.py --test

# Morning brief (cron: 0 7 * * *)
python scripts/openclaw_scheduler_improved.py --morning-brief

# Daily picks after analysis
python scripts/openclaw_scheduler_improved.py --daily-picks

# End of day results (cron: 0 23 * * *)
python scripts/openclaw_scheduler_improved.py --end-of-day

# Line monitor every 30 min during game days
python scripts/openclaw_scheduler_improved.py --line-monitor
```

### Files Modified/Created
- ✅ `backend/services/discord_bet_embeds.py` — NEW improved embed generators
- ✅ `backend/services/openclaw_briefs_improved.py` — NEW working morning brief
- ✅ `backend/services/discord_notifier.py` — Updated to use new embeds
- ✅ `scripts/openclaw_scheduler_improved.py` — NEW functional scheduler

---

**Document Version:** EMAC-070
**Last Updated:** March 16, 2026
**Status:** ⚠️ CRITICAL: Discord notifications NOT being sent (morning brief, tournament bracket, end-of-day). Fixes required pre-Mar 18 (see Section 8). Pre-tournament fixes done (683/686 tests). MIN_BET_EDGE=1.8% active. Guardian opens Mar 18. Next Claude session: Fix Discord jobs (Section 8), then Apr 7+ V9.2 Phase 2.
