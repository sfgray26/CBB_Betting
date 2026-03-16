# Discord & Tournament Audit Report — March 16, 2026

**Auditor:** Kimi CLI  
**Date:** March 16, 2026  
**Status:** ⚠️ CRITICAL GAPS IDENTIFIED

---

## 🔴 CRITICAL FINDINGS

### 1. Morning Briefing Job Only Logs — Does NOT Send to Discord

**File:** `backend/main.py`, lines 401-435

```python
def _morning_briefing_job():
    """Morning slate briefing at 7 AM ET — logs today's prediction slate summary."""
    # ... queries database ...
    narrative = generate_morning_briefing_narrative(n_bets, n_considered, top_bet)
    logger.info("Morning Briefing: %d BET, %d CONSIDER...", n_bets, n_considered, narrative)
    # ❌ NO DISCORD SEND HERE!
```

**Problem:** The job generates a morning briefing but only logs it. It never calls `send_morning_brief()` or any Discord function.

**Impact:** No morning briefs are being sent to Discord despite the job running.

---

### 2. Tournament Bracket Released — No Discord Notification

**File:** `backend/services/tournament_data.py` exists but...

**Problem:** There's no Discord notification when the bracket is released. The system fetches bracket data but doesn't alert users.

**Expected behavior:** When NCAA releases the bracket (March 16, 2026 ~6 PM ET), Discord should send:
- Bracket overview
- First Four matchups
- Model's initial thoughts on each region

**Current behavior:** Nothing. The bracket data is cached but no notification is sent.

---

### 3. OpenClaw Scheduler Not Wired to Railway Cron

**File:** `scripts/openclaw_scheduler_improved.py` exists but...

**Problem:** HANDOFF.md says "Cron wiring pending (user action)" — the scheduler exists but isn't actually scheduled to run in Railway.

**Current scheduled jobs in Railway (main.py):**
- `nightly_analysis` — daily at 6 AM ET ✅
- `update_outcomes` — every 2 hours ✅
- `capture_closing_lines` — every 30 min ✅
- `line_monitor` — every 30 min ✅
- `daily_snapshot` — 4:30 AM ET ✅
- `settle_games_daily` — 4:00 AM ET ✅
- `fetch_ratings` — 8:00 AM ET ✅
- `odds_monitor` — every 5 min ✅
- `morning_briefing` — 7:00 AM ET ⚠️ (but only logs, doesn't send Discord)

**Missing:**
- `openclaw_scheduler_improved.py` tasks are NOT scheduled
- No tournament bracket release notification
- No end-of-day results notification

---

### 4. Bracket Data Available But Not Used for Intelligence

**File:** `backend/services/tournament_data.py` — fetches bracket from BallDontLie API

**Current use:** Cached but only used for seed lookups during analysis

**Missing intelligence:**
- No "bracket released" Discord announcement
- No region-by-region breakdown
- No upset potential analysis
- No Cinderella team identification

---

## 📊 SYSTEM STATUS SUMMARY

| Component | Expected | Actual | Status |
|-----------|----------|--------|--------|
| Morning Brief Discord | Daily 7 AM | Never sends | 🔴 BROKEN |
| Tournament Bracket Alert | When released (Mar 16) | No alert | 🔴 MISSING |
| End-of-Day Results | Daily 11 PM | Never sends | 🔴 MISSING |
| BET NOW Line Alerts | Real-time | Working | 🟡 OK |
| Daily Picks | After analysis | Working | 🟡 OK |
| Nightly Analysis | 6 AM ET | Working | ✅ OK |

---

## 🛠️ REQUIRED FIXES

### Fix 1: Morning Briefing Job Must Send Discord

**File:** `backend/main.py`, function `_morning_briefing_job()`

Add at the end of the function:
```python
from backend.services.openclaw_briefs_improved import generate_and_send_morning_brief_improved
generate_and_send_morning_brief_improved()
```

### Fix 2: Tournament Bracket Release Notification

**New file:** `backend/services/tournament_bracket_notifier.py`

Create a function that:
1. Detects when bracket data is first available
2. Sends Discord embed with:
   - First Four matchups
   - Each region's top seeds
   - Model's initial edge detection (if any)

### Fix 3: Wire OpenClaw Scheduler to Railway Cron

**Options:**
A. Add scheduler tasks to `main.py` scheduler
B. Set up Railway cron jobs to run `openclaw_scheduler_improved.py`

Recommended: Option A — add to existing scheduler in `main.py`:
```python
# End of day results — 11 PM ET
scheduler.add_job(
    _end_of_day_results_job,
    CronTrigger(hour=23, minute=0, timezone=timezone),
    id="end_of_day_results",
    name="End of Day Results",
    replace_existing=True,
)
```

### Fix 4: End-of-Day Results Job

**File:** `backend/main.py` — new function

```python
def _end_of_day_results_job():
    """Send end-of-day results to Discord at 11 PM ET."""
    from backend.services.discord_bet_embeds import create_daily_results_embed
    from backend.services.discord_notifier import send_to_channel
    # ... query BetLog for today's results ...
    # ... send to Discord #cbb-bets ...
```

---

## 📋 HANDOFF.MD UPDATES NEEDED

Section 1 (System Status):
- Discord: Change from "✅ Morning brief + telemetry live" to "⚠️ Morning brief job runs but doesn't send Discord; bracket notifications missing"

Section 4 (Active Tasks):
- Add: "Fix morning briefing job to actually send Discord"
- Add: "Add tournament bracket release notification"
- Add: "Wire end-of-day results to scheduler"

Section 8 (Discord Improvements):
- Add note that improvements are deployed but not scheduled

---

## 🎯 IMMEDIATE ACTION ITEMS

1. **Fix `_morning_briefing_job()` in `main.py`** to send Discord
2. **Create `_end_of_day_results_job()`** and schedule it
3. **Create tournament bracket notifier** for March 16 bracket release
4. **Test all Discord notifications** with `--test` flag
5. **Update HANDOFF.md** with accurate status

---

*Audit completed: March 16, 2026*  
*Next review: After fixes deployed*
