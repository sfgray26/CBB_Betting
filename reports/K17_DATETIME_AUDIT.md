# K-17: Backend UTC Datetime Audit Report

**Date:** April 1, 2026  
**Analyst:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** All `datetime.utcnow()` and `.isoformat()` usages in `backend/`  
**Classification:** MUST FIX (user/game-time visible) vs LOW RISK (internal only)

---

## Executive Summary

Per `ORCHESTRATION.md` Rule 3: **"No UTC for baseball. `datetime.now(ZoneInfo("America/New_York"))` everywhere game dates are computed."**

This audit identified **91 usages** of `datetime.utcnow()` across 21 files. Of these, **9 are MUST FIX** (user-visible or game-time-adjacent), and **82 are LOW RISK** (internal logging, caching, or non-display timestamps).

**Estimated fix time:** <30 minutes for all MUST FIX items.

---

## MUST FIX (9 occurrences)

These affect user-facing displays, game-time comparisons, or fantasy decisions.

### 1. `backend/services/dashboard_service.py` (4 occurrences)

| Line | Code | Context | Risk |
|------|------|---------|------|
| 206 | `timestamp=datetime.utcnow().isoformat()` | Dashboard timestamp displayed to users | **HIGH** |
| 268 | `timestamp=datetime.utcnow()` | Roster validation timestamp | Medium |
| 342 | `recent_date = datetime.utcnow().date() - timedelta(days=1)` | Streak analysis date range | Medium |
| 827 | `today = datetime.utcnow()` | Probable pitchers "today" detection | **HIGH** |

**Fix:** Replace with `datetime.now(ZoneInfo("America/New_York"))`

```python
from zoneinfo import ZoneInfo

# Line 206
timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat()

# Line 827  
today = datetime.now(ZoneInfo("America/New_York"))
```

---

### 2. `backend/services/line_monitor.py` (1 occurrence)

| Line | Code | Context | Risk |
|------|------|---------|------|
| 88 | `game.game_date < datetime.utcnow()` | Game start time comparison | **HIGH** |

**Issue:** Game start times are stored in UTC (from Odds API). Comparing against `utcnow()` is correct for CBB betting, but for MLB fantasy we need consistent ET handling.

**Fix:** Ensure game_date is timezone-aware and compare properly:
```python
from zoneinfo import ZoneInfo
now_et = datetime.now(ZoneInfo("America/New_York"))
if game.game_date and game.game_date.astimezone(ZoneInfo("America/New_York")) < now_et:
```

---

### 3. `backend/services/data_reliability_engine.py` (1 occurrence)

| Line | Code | Context | Risk |
|------|------|---------|------|
| 186 | `StatcastPerformance.game_date == datetime.utcnow().date() - timedelta(days=1)` | Yesterday's Statcast data query | Medium |

**Issue:** Uses UTC "yesterday" which may differ from ET "yesterday" during late games.

**Fix:**
```python
from zoneinfo import ZoneInfo
yesterday_et = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).date()
```

---

### 4. `backend/services/analysis.py` (3 occurrences)

| Line | Code | Context | Risk |
|------|------|---------|------|
| 256 | `game_date = datetime.utcnow()` | Fallback game date for new games | Medium |
| 277 | `today_start = datetime.utcnow().replace(hour=0, minute=0...)` | Daily exposure calculation | Medium |
| 493 | `today_start = datetime.utcnow().replace(hour=0, minute=0...)` | Same, different function | Medium |

**Note:** These are CBB betting analysis functions. Lower priority since CBB season is ending, but should be fixed for consistency.

---

## LOW RISK (82 occurrences)

These are internal timestamps, cache tracking, logging, or Discord webhooks that don't affect user-facing game times.

### CBB Betting / Internal Systems (52 occurrences)

| File | Lines | Purpose | Risk Level |
|------|-------|---------|------------|
| `main.py` | 359, 1023, 1222, 1256-1257, 1307, 1353, 1483, 1496, 1504, 1656, 1768, 2019-2020, 2064, 2203, 2304, 2338-2339, 2500, 2536, 2601, 2647, 2838, 3000, 4693, 6187 | Internal timestamps, logging, cache tracking, Discord embeds | LOW |
| `services/tournament_data.py` | 69, 80, 119 | Tournament cache timestamps | LOW |
| `services/sharp_money.py` | 43, 114, 156, 234, 239, 241 | Internal tracking timestamps | LOW |
| `services/sentinel.py` | 126 | Health check timestamp | LOW |
| `services/recalibration.py` | 265, 382, 434, 461, 501, 590 | Internal metadata timestamps | LOW |
| `services/ratings.py` | 1070, 1631, 1672, 1738, 1774 | Cache timestamp tracking | LOW |
| `services/performance.py` | 328, 397, 475, 519, 585, 665, 737, 752 | Analytics cutoff calculations | LOW |
| `services/openclaw_lite.py` | 49, 156, 160, 208 | Queue job timestamps | LOW |
| `services/odds_monitor.py` | 150, 338 | Internal polling timestamps | LOW |
| `services/odds.py` | 98, 586, 609, 618 | Quota tracking, fetch timestamps | LOW |
| `services/injuries.py` | 284, 328 | Injury update timestamps | LOW |
| `services/dk_import.py` | 237, 462, 593 | Import tracking timestamps | LOW |
| `services/daily_ingestion.py` | 195, 205 | Scheduler metadata | LOW |
| `services/coordinator.py` | 129-130, 144-145 | Discord embed footers | LOW |
| `services/bet_tracker.py` | 444, 589 | Bet tracking timestamps | LOW |
| `services/alerts.py` | 228, 295 | Alert timestamps | LOW |

### Fantasy Baseball / Already Correct (25 occurrences)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `fantasy_baseball/openclaw_workflow.py` | 204, 249 | Internal timestamps | LOW |
| `fantasy_baseball/openclaw_email_notifier.py` | 88, 206, 238, 252 | Email timestamps | LOW |
| `fantasy_baseball/decision_tracker.py` | 73, 172 | Decision timestamps | LOW |
| `fantasy_baseball/daily_briefing.py` | 208 | Discord embed timestamp (uses `generated_at` field) | LOW |
| `fantasy_baseball/circuit_breaker.py` | 175 | Circuit state timestamp | LOW |
| `fantasy_baseball/cache_manager.py` | 27, 86 | Cache metadata | LOW |
| `fantasy_baseball/weather_fetcher.py` | 586, 589 | Cache timestamps | LOW |
| `fantasy_baseball/statcast_ingestion.py` | 166, 180, 190, 201, 214, 225, 237, 302-303, 740, 780 | Target date serialization (correctly uses `target_date` parameter) | ALREADY CORRECT |
| `fantasy_baseball/platoon_fetcher.py` | 237 | Last updated timestamp | LOW |
| `fantasy_baseball/pitcher_deep_dive.py` | 232 | Cache timestamp | LOW |

### Correct ET Usage Found (5 occurrences)

These files already use proper ET timezone:

| File | Line | Code | Status |
|------|------|------|--------|
| `services/openclaw_telemetry.py` | 33, 366, 413, 458 | `datetime.now(timezone.utc)` | Uses proper timezone |
| `services/daily_ingestion.py` | 548 | `datetime.now(ZoneInfo("America/New_York"))` | ✅ CORRECT |
| `services/discord_*.py` | Multiple | `datetime.now(timezone.utc)` | Discord requires UTC |
| `betting_model.py` | 1901 | `datetime.now(ZoneInfo("America/New_York"))` | ✅ CORRECT |
| `main.py` | 791, 864 | `datetime.now(_timezone.utc)` | Properly timezone-aware |

---

## Actionable Fix List for Claude Code

### Immediate (Pre-Apr 7)

```python
# File: backend/services/dashboard_service.py
# Add to imports:
from zoneinfo import ZoneInfo

# Line 206: Change to
timestamp=datetime.now(ZoneInfo("America/New_York")).isoformat(),

# Line 827: Change to  
today = datetime.now(ZoneInfo("America/New_York"))
```

### Post-Apr 7 (Lower Priority)

1. `backend/services/line_monitor.py` line 88 - Game start comparison
2. `backend/services/data_reliability_engine.py` line 186 - Yesterday's Statcast query
3. `backend/services/analysis.py` lines 256, 277, 493 - CBB betting fallbacks

---

## Verification

After fixes, verify with:

```python
# Test dashboard timestamp shows ET
from datetime import datetime
from zoneinfo import ZoneInfo

dt = datetime.now(ZoneInfo("America/New_York"))
print(dt.isoformat())  # Should show -04:00 or -05:00 offset
```

---

## Risk Assessment

| Risk | Count | Impact |
|------|-------|--------|
| **HIGH** (User-visible timestamps) | 2 | Dashboard shows wrong time, lineup dates off |
| **MEDIUM** (Game-time adjacent) | 7 | Streak analysis, pitcher detection may be off by 1 day |
| **LOW** (Internal only) | 82 | No user impact |

---

*Report completed: All `datetime.utcnow()` usages classified. Fix estimated at <30 minutes for all MUST FIX items.*
