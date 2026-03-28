# CLAUDE CODE — LEAD ARCHITECT EMERGENCY BRIEFING
## Production Crisis: Fantasy Baseball Platform Down (March 28, 2026)

> **Status:** CRITICAL — Active season (2 days in), users cannot view rosters or matchups  
> **Role:** You are the acting Lead Architect with full authority to direct Kimi and Gemini  
> **Time Sensitivity:** Every hour of downtime = users missing lineup changes for live games

---

## EXECUTIVE SUMMARY

**The Problem:** `/api/fantasy/roster` endpoint is returning HTTP 500, causing cascading failures across 5+ pages. Users cannot see their teams, matchups, or make lineup decisions during live games.

**Root Causes Identified (Forensic Audit Complete):**
1. **P0 CRITICAL:** `RosterPlayerOut` Pydantic validation error — `status` field receiving `False` (bool) instead of string
2. **P0 CRITICAL:** Yahoo API roster fetch failure — likely OAuth/token or response parsing error
3. **HIGH:** UTC timezone parser dropping West Coast games (9pm+ EDT)
4. **HIGH:** Matchup scoreboard team mapping empty despite 5 matchups returned

---

## IMMEDIATE ACTION REQUIRED (Next 2 Hours)

### Fix 1: Roster Endpoint HTTP 500 [P0]

**Error:**
```
pydantic_core.ValidationError: status: Input should be a valid string 
[type=string_type, input_value=False, input_type=bool]
```

**Location:** `backend/main.py` around line 5303

**Root Cause:** The `_parse_player` method in `yahoo_client_resilient.py` is setting `status: False` (bool) when Yahoo returns a falsy value, but `RosterPlayerOut` schema expects `Optional[str]`.

**Your Fix:**
```python
# In backend/main.py where RosterPlayerOut is constructed
# Around line 5303, find where status is being set

# WRONG (current):
status=p.get("status"),  # p.get("status") returns False when missing

# CORRECT:
status=p.get("status") if p.get("status") else None,
# OR ensure _parse_player in yahoo_client_resilient.py never returns bool for status
```

**Also Check:** `yahoo_client_resilient.py` `_parse_player` method — ensure status extraction returns `None` or string, never `False`:
```python
"status": meta.get("status") or None,  # Convert falsy to None
```

---

### Fix 2: Yahoo Roster Fetch Failure [P0]

**Symptom:** `Failed to fetch` on `/api/fantasy/roster`, logs show empty team list

**Likely Causes (in order):**
1. OAuth token expired/invalid — check `YAHOO_REFRESH_TOKEN` env var in Railway
2. Yahoo API response parsing failure in `_parse_player` or `_team_section`
3. Network/circuit breaker open

**Diagnostic Commands (Run via Gemini):**
```bash
# Check if token is valid
railway run python -c "
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
try:
    c = YahooFantasyClient()
    print('Auth OK, league:', c.league_key)
    print('My team:', c.get_my_team_key())
except Exception as e:
    print('Auth failed:', e)
"
```

**Your Fix Strategy:**
1. Add defensive logging to `_get_yahoo_roster` method
2. Wrap roster fetch in try/except with detailed error messages
3. If OAuth issue, implement automatic token refresh fallback

---

### Fix 3: UTC Timezone Parser [HIGH]

**Symptom:** West Coast games (9:10pm, 9:40pm EDT) showing `—` / "no game today"

**Affected Players:**
- Geraldo Perdomo (@ LAD, 9:10pm EDT) — falsely flagged "no game"
- Steven Kwan (@ SEA, 9:40pm EDT) — benched incorrectly
- Gavin Williams (@ SEA, 9:40pm EDT) — "no start" flag incorrect
- Edwin Díaz (vs AZ, 9:10pm EDT) — game time not resolved

**Root Cause:** Timezone conversion producing next-day UTC (e.g., 9:40pm EDT = 01:40 UTC+1), parser rejecting as "future" date

**Your Fix:**
```python
# In game time parsing logic (likely in daily_lineup_optimizer.py or smart_lineup_selector.py)
# When comparing game_date to "today", use EDT/EST local time, not UTC

from datetime import datetime
import pytz

def is_game_today(game_time_iso: str) -> bool:
    # Parse as UTC, convert to Eastern
    utc = datetime.fromisoformat(game_time_iso.replace('Z', '+00:00'))
    eastern = pytz.timezone('America/New_York')
    game_edt = utc.astimezone(eastern)
    
    # Compare dates in EDT (handles 9:40pm -> 1:40am next day UTC correctly)
    today_edt = datetime.now(eastern).date()
    return game_edt.date() == today_edt
```

---

### Fix 4: Matchup Scoreboard Empty Team Mapping [HIGH]

**Symptom:** `WARNING - My team key 469.l.72586.t.7 not found in scoreboard teams: []`

**Context:** Scoreboard returns 5 matchups, but team list is empty

**Location:** `backend/main.py` around line 5483

**Root Cause:** The `_extract_team_stats` function is not extracting `team_key` from Yahoo's nested response structure

**Your Fix:**
```python
# In backend/main.py, improve the _extract_team_stats function
# Current implementation may be too shallow for Yahoo's nested arrays

def _extract_team_stats(team_entry) -> tuple[str, str, dict]:
    """Extract (team_key, team_name, stats) from Yahoo team entry."""
    t_meta = {}
    stats_raw = []
    
    # Yahoo returns team data as deeply nested lists
    # Example: [[{"team_key": "..."}, {"name": "..."}], [{"team_stats": {...}}]]
    
    def flatten(entry, depth=0):
        if depth > 5:
            return
        if isinstance(entry, list):
            for item in entry:
                flatten(item, depth + 1)
        elif isinstance(entry, dict):
            # Extract key fields
            for key in ["team_key", "name", "team_id", "is_owned_by_current_login"]:
                if key in entry:
                    t_meta[key] = entry[key]
            # Extract stats
            if "team_stats" in entry:
                stats_list = entry["team_stats"].get("stats", [])
                if isinstance(stats_list, list):
                    stats_raw.extend(stats_list)
            # Recurse
            for v in entry.values():
                flatten(v, depth + 1)
    
    flatten(team_entry)
    
    # Build stats dict
    stats_dict = {}
    for s in stats_raw:
        if isinstance(s, dict):
            stat = s.get("stat", {})
            if isinstance(stat, dict):
                sid = str(stat.get("stat_id", ""))
                key = stat_id_map.get(sid, sid)
                val = stat.get("value", "")
                if key:
                    stats_dict[key] = val
    
    return (
        t_meta.get("team_key", ""),
        t_meta.get("name", ""),
        stats_dict,
    )
```

---

## SECONDARY FIXES (Next 4-6 Hours)

### Fix 5: Park Factor = 1.000 for All [HIGH]
**Issue:** All batters show Park Factor 1.000 (neutral)
**Cause:** JOIN between `games` and `park_factors` tables broken post-migration
**Fix:** Audit `park_factors` table for data, verify JOIN key (likely `stadium_id` or `venue_id`)

### Fix 6: Ownership % = 0.0% for All [CRITICAL]
**Issue:** All players showing 0.0% owned
**Cause:** Ownership table truncated or foreign key severed
**Fix:** Check `percent_owned` field in player records, verify Yahoo ownership subresource parsing

### Fix 7: IL Status Not Propagating [HIGH]
**Issue:** Blake Snell (IL15) showing `UNKNOWN` instead of `IL`
**Cause:** `injury_status` field not mapping to `RosterPlayerOut`
**Fix:** Add `injury_status` to schema and ensure `_parse_player` extracts it

### Fix 8: RP Misclassification as SP [HIGH]
**Issue:** Edwin Díaz, Jason Adam (RPs) appearing in Starting Pitchers table
**Cause:** SP query not filtering on position type
**Fix:** Add `WHERE position_type = 'SP'` to starting pitchers query

---

## DELEGATION MATRIX

### Kimi CLI (Your Deputy)
**Assign:**
- Frontend CSS fix for invisible injury names (P1-BUG-06)
- Dashboard "Last updated: Invalid Date" null guard (P1-BUG-01)
- Waiver wire ADD/DROP card rendering fix (P3-BUG-03)
- Error boundary user-friendly messages (P4-BUG-04)

**Do NOT assign:**
- Backend schema changes
- Database queries
- Yahoo API fixes

### Gemini CLI (Ops Support)
**Run immediately:**
```bash
# Verify OAuth token validity
railway run python -c "from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient; c = YahooFantasyClient(); print(c.get_my_team_key())"

# Check env vars
railway variables | grep -i yahoo

# Check if OR-Tools installed
railway run pip show ortools
```

---

## VALIDATION CHECKLIST

After fixes, verify:
- [ ] `/api/fantasy/roster` returns 200 with player list (not 500)
- [ ] No `pydantic.ValidationError` in logs
- [ ] West Coast games (9pm+ EDT) resolve correctly
- [ ] Matchup page shows correct team names (not "My Team" vs "TBD")
- [ ] Park factors vary by stadium (not all 1.000)
- [ ] Ownership % shows real values (not all 0.0%)

---

## CONTEXT FILES

Critical files to read:
- `backend/main.py` — Lines 5280-5320 (roster endpoint), 5370-5485 (matchup endpoint)
- `backend/fantasy_baseball/yahoo_client_resilient.py` — `_parse_player`, `_get_yahoo_roster`, `_extract_team_stats`
- `backend/schemas.py` — `RosterPlayerOut` schema
- `reports/yahoo-client-hotfix-march28.md` — Kimi's prior fixes (read to avoid conflict)

---

## SUCCESS CRITERIA

**P0 Fixes (Must have by EOD):**
1. Roster endpoint returns 200 OK with valid JSON
2. No Pydantic validation errors in production logs
3. Users can view their team roster
4. Matchup page resolves opponent correctly

**P1 Fixes (24-48 hours):**
5. West Coast game times resolve correctly
6. Park factors show stadium-specific values
7. Ownership percentages display correctly
8. IL status shows for injured players

---

## COMMUNICATION PROTOCOL

**Every 2 hours, update:**
1. Which P0 fixes are deployed
2. Current blocker (if any)
3. Delegation status (Kimi/Gemini tasks)

**If stuck >30 minutes:**
- Escalate to human immediately
- Do not spin on OAuth issues — get fresh tokens from user if needed

---

## CLOSING

**The platform is down for users right now.** They cannot see their rosters during live games. This is not a polish issue — this is a production outage.

**Your priorities in order:**
1. Fix the 500 error on `/api/fantasy/roster` (Pydantic bool→str issue)
2. Fix Yahoo API connection (OAuth/token)
3. Fix West Coast timezone parser
4. Fix matchup team mapping

**Do not** work on UI polish, CSS, or feature additions until P0 fixes are live.

— Emergency Directive, March 28, 2026
