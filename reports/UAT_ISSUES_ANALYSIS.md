# UAT Issues Analysis & Research Report

**Date:** March 25, 2026  
**Analyst:** Kimi CLI  
**Scope:** Research and root cause analysis for CBB Edge UAT findings

---

## Executive Summary

| Issue | Severity | Root Cause Hypothesis | Complexity |
|-------|----------|----------------------|------------|
| Bet History | Medium | Data filter logic incorrect | Low |
| Calibration | High | Missing brier score calculation | Medium |
| Today's Bets | Low | UI enhancement needed | Low |
| Odds Monitor | High | Portfolio data fetch failing | Medium |
| Daily Lineup | High | Park factor/projection defaults | Medium |
| Waiver Wire | Critical | Yahoo API endpoint change | High |
| My Roster | Medium | Data mapping issues | Low |
| Current Matchup | Medium | Season not started / API issue | Medium |
| Risk Dashboard | Medium | Config validation issue | Low |
| Feature Flags | Low | UI cleanup | Low |

---

## Issue 1: Bet History - Shows All Recommendations

### Problem
Bet history lists every single bet recommendation instead of only showing bets the user actually placed.

### Analysis
This is likely a **filter logic issue** in the database query or frontend display.

### Possible Causes
1. **Missing filter** in SQL query - `WHERE user_placed = true` or similar
2. **Frontend displaying raw recommendations** instead of filtering by user's placed bets
3. **Data model issue** - placed_bet flag not being set correctly

### Recommended Fix
```sql
-- Current (incorrect)
SELECT * FROM bet_recommendations ORDER BY date;

-- Should be
SELECT * FROM bet_recommendations 
WHERE user_placed = true 
ORDER BY date DESC;
```

### Files to Check
- Backend: Betting history endpoint
- Frontend: Bet history component
- Database: bet_recommendations table schema

---

## Issue 2: Calibration - Shows Nothing

### Problem
Calibration section displays no brier score or calibrated bets data.

### Analysis
Brier score calculation may not be implemented or the data pipeline is broken.

### What is Brier Score?
```
Brier Score = (predicted_probability - actual_outcome)²

Example:
- Model predicts 75% win probability
- Actual: Win (1)
- Brier = (0.75 - 1)² = 0.0625

Perfect score: 0
Random guessing: 0.25
```

### Possible Causes
1. **Calculation not implemented** - Missing brier_score() function
2. **Missing data** - No outcome tracking for past bets
3. **Query returning empty** - Filter too restrictive
4. **Frontend not displaying** - Data exists but not rendered

### Recommended Investigation
```python
# Check if calculation exists
def calculate_brier_score(predictions, outcomes):
    return np.mean((predictions - outcomes) ** 2)

# Verify data exists
SELECT COUNT(*) FROM bets WHERE outcome IS NOT NULL;
```

---

## Issue 3: Today's Bets - Missing "Placed" Checkbox

### Problem
Under "Consider" section, user needs ability to mark that they placed the bet.

### Analysis
This is a **UI/UX enhancement** request, not a bug.

### Implementation Options
1. **Add checkbox** in the "Consider" row
2. **Auto-detect** from bet slip integration (if available)
3. **Manual confirmation** button

### Database Change Needed
```sql
ALTER TABLE bet_recommendations 
ADD COLUMN user_placed BOOLEAN DEFAULT FALSE,
ADD COLUMN placed_at TIMESTAMP,
ADD COLUMN placed_amount DECIMAL(10,2);
```

---

## Issue 4: Odds Monitor - Portfolio Status Not Working

### Problem
Drawdown gauge, total exposure, and portfolio status not displaying data.

### Analysis
This suggests the **portfolio data fetch is failing** or returning empty.

### Possible Causes
1. **API endpoint down** - Odds API not responding
2. **Authentication expired** - API key invalid
3. **Data aggregation failing** - Calculation error in backend
4. **Empty portfolio** - No active bets to display

### Debug Steps
```javascript
// Check browser console for errors
// Check network tab for failed API calls
// Verify: GET /api/portfolio/status returns 200 with data
```

---

## Issue 5: Daily Lineup - All Players Show 4.50/1.000/4.625

### Problem
- Implied runs: 4.50 for every player
- Park factor: 1.000 for every player
- Score: 4.625 for every player

### Analysis
These are **default/fallback values** being used when actual calculation fails.

### Root Cause Hypothesis
The MLB projection service is not calculating player-specific values and falling back to league averages.

### Investigation Path
1. **Check MLB analysis service** - Is it running?
2. **Verify FanGraphs data** - Are player stats being fetched?
3. **Check pybaseball cache** - Is data available?
4. **Review projection formula** - Is it using defaults?

### Code to Check
```python
# In mlb_analysis.py - check if these are defaults
LEAGUE_AVG_ERA = 4.25
DEFAULT_PARK_FACTOR = 1.0
DEFAULT_RUNS = 4.25 * 1.0 + 0.25  # ~4.50

# Score calculation
score = (z_score + 5) / 10  # Normalizing? Should vary per player
```

---

## Issue 6: Daily Lineup - Yahoo API Error 422

### Error
```
422: set_lineup failed: 400
<?xml version="1.0" encoding="UTF-8"?>
<error xml:lang="en-us" yahoo:uri="...">
  <description>game_ids don't match for player key in</description>
</error>
```

### Analysis
This is a **Yahoo Fantasy API error** indicating a player is being set in a lineup for the wrong game/team.

### Root Causes
1. **Player traded** - Player now on different team than in your roster
2. **Game ID mismatch** - API expects different game_id for player
3. **Position eligibility** - Player not eligible for that position in that game
4. **Injured list** - Player on IL but roster slot doesn't support it

### Recommended Fix
```python
# Before setting lineup, verify player eligibility
# 1. Check if player is still on expected team
# 2. Verify game_id matches the player's actual game
# 3. Confirm position eligibility for that game

# Pseudo-code
def validate_lineup_changes(changes):
    for change in changes:
        player = get_player(change.player_key)
        if player.current_team != change.expected_team:
            raise Error(f"Player {player.name} changed teams")
        if not player.is_eligible_for_position(change.position):
            raise Error(f"Position mismatch")
```

---

## Issue 7: Waiver Wire - Failed to Load (503 Error)

### Error
```json
{
  "error": {
    "description": "Invalid subresource stats requested",
    "detail": ""
  }
}
```

### Analysis
Yahoo Fantasy API **endpoint or subresource parameter has changed**.

### Yahoo API Documentation Findings
From Yahoo Fantasy Sports API docs, valid player subresources include:
- `metadata` (default)
- `stats` 
- `ownership`
- `percent_owned`
- `draft_analysis`

The error "Invalid subresource stats requested" suggests the API no longer accepts `stats` as a subresource in this context, or the format has changed.

### Root Cause
The waiver wire query is likely using:
```
/fantasy/v2/league/{league_key}/players?out=metadata,stats
```

But Yahoo may have changed this to require separate calls, or the `stats` subresource is no longer valid for available players.

### Recommended Fix
```python
# Option 1: Remove stats from initial query, fetch separately
players = yahoo_api.get_available_players(out='metadata')  # No stats
for player in players:
    player.stats = yahoo_api.get_player_stats(player.key)  # Separate call

# Option 2: Use different endpoint
# Try /players;status=A instead of waiver endpoint
```

---

## Issue 8: My Roster - Z-Score Blank, Undroppable "Yes" for All

### Problem
- Z-score blank for some players
- Undroppable shows "Yes" for every player (incorrect)

### Analysis
**Data mapping issue** - likely hardcoded value or query error.

### Root Causes
1. **Z-score calculation failing** - Missing data for some players
2. **Undroppable hardcoded** - Default value not being overridden
3. **API response parsing** - Yahoo field name changed

### Check
```python
# In roster display code
print(player.undroppable)  # Should vary, likely shows "true" for all
print(player.z_score)      # None for some players

# Yahoo API field may be "is_undroppable" not "undroppable"
```

---

## Issue 9: Current Matchup - Not Showing Category Stats

### Problem
- Category stats not showing
- No opponent name or team name displayed

### Analysis
Two possibilities:
1. **Season hasn't started** - No matchup data available yet
2. **API call failing** - Matchup endpoint not working

### Expected Behavior
Even with 0 stats, should show:
- Opponent team name
- Category headers (R, HR, RBI, etc.)
- All zeros for current week

### Investigation
```javascript
// Check if API returns empty or errors
GET /api/matchup/current
// Should return: { opponent: "Team Name", categories: {...}, stats: {...} }
```

---

## Issue 10: Risk Dashboard - Settlement Lookback, Drawdown 0%

### Problems
1. Settlement lookback locked at "2 days", can't change
2. Portfolio drawdown showing 0.0%

### Analysis
**Configuration and calculation issues**.

### Settlement Lookback
Likely a **UI validation bug**:
```javascript
// Current (buggy)
<input type="number" min="20" value={lookback} />
// ^ min=20 prevents values starting with other digits

// Should be
<input type="number" min="1" max="30" value={lookback} />
```

### Drawdown 0%
Likely **calculation issue**:
```python
# Drawdown = (peak - current) / peak
# If peak not tracked correctly, always shows 0
```

---

## Issue 11: Feature Flags - Remove Bracket Simulator

### Request
Remove "Bracket Simulator" feature from UI completely (currently grayed out showing "Soon").

### Analysis
Simple **UI cleanup** task.

### Implementation
```jsx
// Remove from navigation
// Remove route definition
// Optionally: Delete component files if not needed later
```

---

## Recommended Priority Order

### Critical (Fix First)
1. **Waiver Wire** - API endpoint broken, core feature unusable
2. **Daily Lineup (Yahoo 422)** - Lineup setting failing

### High (Fix This Week)
3. **Calibration** - Core model quality metric missing
4. **Odds Monitor** - Portfolio tracking not working
5. **Daily Lineup (Projections)** - All default values

### Medium (Fix Soon)
6. **Bet History** - Filter logic fix
7. **My Roster** - Data mapping issues
8. **Current Matchup** - Verify after season starts
9. **Risk Dashboard** - Config fixes

### Low (Cleanup)
10. **Today's Bets checkbox** - UI enhancement
11. **Feature Flags** - Remove Bracket Simulator

---

## Files to Investigate (Claude Code Task)

### Backend
- `backend/fantasy_baseball/yahoo_api.py` - Waiver wire, roster, matchup calls
- `backend/services/mlb_analysis.py` - Projection calculations
- `backend/betting_model.py` - Calibration/brier score
- `backend/api/portfolio.py` - Odds monitor data

### Frontend
- `frontend/components/BetHistory.tsx` - Filter logic
- `frontend/components/DailyLineup.tsx` - Projection display
- `frontend/components/WaiverWire.tsx` - API error handling
- `frontend/components/Roster.tsx` - Z-score/undroppable display
- `frontend/components/RiskDashboard.tsx` - Form validation

### Database
- Check `bet_recommendations` table for placed flag
- Verify `player_projections` has calculated values
- Check `portfolio_snapshots` for drawdown data

---

## Next Steps

1. **Immediate**: Fix Waiver Wire API endpoint (blocking feature)
2. **This Week**: Fix Daily Lineup Yahoo 422 error
3. **Parallel**: Investigate Calibration calculation
4. **Before Apr 7**: Fix all High priority issues

---

**Report ID:** KIMI-UAT-2026-0325
**Status:** Research Complete - Ready for Implementation
