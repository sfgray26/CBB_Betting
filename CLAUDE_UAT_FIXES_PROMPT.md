# Prompt for Claude Code: UAT Critical Fixes

> **Related:** `CLAUDE_ARCHITECT_PROMPT_MARCH28.md` (production issues) | `CLAUDE_TEAM_COORDINATION_PROMPT.md` (work assignments)

Copy-paste into Claude Code to fix the UAT issues:

---

## PROMPT START

Fix the UAT issues identified in `reports/UAT_ISSUES_ANALYSIS.md`. Work in priority order.

### Priority 1: CRITICAL (Fix First)

#### Issue 6: Daily Lineup Yahoo API Error 422
**Error:** `game_ids don't match for player key`

```python
# In backend/fantasy_baseball/yahoo_api.py or similar
# Before setting lineup, validate player game_id matches

# Current code (problematic):
# yahoo_api.set_lineup(player_key, position)

# Fixed code:
def set_lineup_safe(team_key, players):
    for player in players:
        # Verify player is still on expected team
        current_team = get_player_team(player['player_key'])
        if current_team != player['expected_team']:
            logger.warning(f"Player {player['name']} changed teams")
            continue
            
        # Verify game_id for today's date
        game_id = get_player_game_id(player['player_key'], date.today())
        if not game_id:
            logger.warning(f"No game found for {player['name']}")
            continue
            
        # Now set lineup
        set_lineup(player['player_key'], player['position'])
```

**Investigate:**
- Check if any players were traded recently
- Verify player eligibility for position
- Confirm game_id matches between roster and lineup API

---

#### Issue 7: Waiver Wire API 503 Error
**Error:** `Invalid subresource stats requested`

**Root Cause:** Yahoo API no longer accepts `stats` subresource in players query.

```python
# Current (broken):
# /fantasy/v2/league/{league_id}/players?out=metadata,stats

# Fix Option 1: Remove stats from initial query
players = yahoo_api.get_league_players(
    league_key,
    status='A',
    out='metadata'  # Remove 'stats'
)

# Fetch stats separately
for player in players:
    player['stats'] = yahoo_api.get_player_stats(player['player_key'])

# Fix Option 2: Use waiver-specific endpoint
waivers = yahoo_api.get_waivers(league_key)  # Uses /waivers endpoint
```

**Files to check:**
- `backend/fantasy_baseball/yahoo_api.py`
- `backend/fantasy_baseball/pybaseball_loader.py`

---

### Priority 2: HIGH (Fix This Week)

#### Issue 2: Calibration - No Brier Score
**Problem:** Calibration section shows empty.

```python
# Add to backend/betting_model.py or create calibration.py

import numpy as np
from typing import List, Tuple

def calculate_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """
    Calculate Brier score for model calibration.
    
    Args:
        predictions: List of predicted probabilities (0-1)
        outcomes: List of actual outcomes (0 or 1)
    
    Returns:
        Brier score (0 = perfect, 0.25 = random, lower is better)
    """
    return np.mean((np.array(predictions) - np.array(outcomes)) ** 2)

def get_calibration_data(days: int = 30) -> dict:
    """Get calibration metrics for recent bets."""
    query = """
        SELECT predicted_prob, actual_outcome 
        FROM bet_recommendations 
        WHERE outcome IS NOT NULL 
          AND created_at >= NOW() - INTERVAL '%s days'
    """
    results = db.execute(query, (days,))
    
    if not results:
        return {"error": "No completed bets in time period"}
    
    predictions = [r[0] for r in results]
    outcomes = [r[1] for r in results]
    
    brier = calculate_brier_score(predictions, outcomes)
    
    # Calculate calibration by probability bins
    bins = defaultdict(lambda: {'predicted': [], 'actual': []})
    for pred, outcome in zip(predictions, outcomes):
        bin_key = round(pred * 10) / 10  # 0.1, 0.2, etc.
        bins[bin_key]['predicted'].append(pred)
        bins[bin_key]['actual'].append(outcome)
    
    calibration_bins = []
    for bin_key, data in sorted(bins.items()):
        avg_predicted = np.mean(data['predicted'])
        avg_actual = np.mean(data['actual'])
        calibration_bins.append({
            'bin': bin_key,
            'predicted': avg_predicted,
            'actual': avg_actual,
            'count': len(data['actual'])
        })
    
    return {
        'brier_score': round(brier, 4),
        'sample_size': len(results),
        'calibration_bins': calibration_bins
    }
```

**Add endpoint:**
```python
@app.get("/api/calibration")
def get_calibration():
    return get_calibration_data(days=30)
```

---

#### Issue 5: Daily Lineup - Default Values (4.50, 1.000, 4.625)
**Problem:** All players show identical values (defaults).

**Investigate:**
```python
# Check mlb_analysis.py projections

# 1. Verify FanGraphs/pybaseball data is loading
pitcher_stats = load_pitcher_stats()
print(f"Loaded {len(pitcher_stats)} pitcher xERA values")  # Should be > 100

# 2. Check if xERA defaults to LEAGUE_AVG_ERA (4.25)
def _pitcher_xera(self, name, stats):
    xera = stats.get(name.lower())
    if not xera or xera == 0:
        return LEAGUE_AVG_ERA  # This is the default!
    return xera

# 3. Check team wRC+ aggregation
team_stats = self._load_team_stats()
print(f"Teams with wRC+: {len(team_stats)}")  # Should be 30
```

**Fix:** Ensure pybaseball cache is fresh and players are being matched correctly.

```bash
# Force refresh pybaseball data
python scripts/fetch_statcast.py --force-refresh

# Verify cache files exist
ls data/cache/pybaseball_*.json
```

---

### Priority 3: MEDIUM (Fix Soon)

#### Issue 1: Bet History - Show Only Placed Bets
```sql
-- Find current query (likely missing filter)

-- Current (shows all):
SELECT * FROM bet_recommendations ORDER BY date DESC;

-- Fixed (shows only placed):
SELECT * FROM bet_recommendations 
WHERE user_placed = true 
ORDER BY date DESC;
```

**Add filter if missing:**
```python
@app.get("/api/bets/history")
def get_bet_history(placed_only: bool = True):
    query = db.query(BetRecommendation)
    if placed_only:
        query = query.filter(BetRecommendation.user_placed == True)
    return query.order_by(BetRecommendation.date.desc()).all()
```

---

#### Issue 8: My Roster - Z-Score & Undroppable
```python
# Check data mapping in roster fetch

# Undroppable issue - verify field name
def parse_player(player_data):
    return {
        'name': player_data['name']['full'],
        'z_score': player_data.get('z_score'),  # May be missing
        'is_undroppable': player_data.get('is_undroppable', False),  # Check field name
        # Yahoo API may use 'undroppable' or 'is_undroppable'
    }
```

---

#### Issue 10: Risk Dashboard
**Settlement lookback validation fix:**
```jsx
// Current (buggy - min="20" prevents changing to 1, 3, etc.)
<input type="number" min="20" value={lookback} />

// Fixed
<input type="number" min="1" max="30" value={lookback} 
       onChange={(e) => setLookback(parseInt(e.target.value))} />
```

**Drawdown calculation fix:**
```python
# Check portfolio calculation
# Drawdown = (peak_value - current_value) / peak_value

# If showing 0%, likely not tracking peak correctly
class PortfolioTracker:
    def __init__(self):
        self.peak_value = 0
        self.current_value = 0
    
    def update(self, new_value):
        self.current_value = new_value
        if new_value > self.peak_value:
            self.peak_value = new_value
    
    @property
    def drawdown(self):
        if self.peak_value == 0:
            return 0
        return (self.peak_value - self.current_value) / self.peak_value
```

---

### Priority 4: LOW (Cleanup)

#### Issue 3: Today's Bets - Add "Placed" Checkbox
```jsx
// In Today's Bets component
{recommendations.map(rec => (
  <tr key={rec.id}>
    <td>{rec.game}</td>
    <td>{rec.recommendation}</td>
    <td>
      <input 
        type="checkbox" 
        checked={rec.user_placed}
        onChange={(e) => markAsPlaced(rec.id, e.target.checked)}
      />
    </td>
  </tr>
))}
```

**Add API endpoint:**
```python
@app.post("/api/bets/{bet_id}/placed")
def mark_bet_placed(bet_id: int, placed: bool = True):
    bet = db.query(BetRecommendation).get(bet_id)
    bet.user_placed = placed
    bet.placed_at = datetime.now() if placed else None
    db.commit()
    return {"success": True}
```

---

#### Issue 11: Remove Bracket Simulator
```jsx
// Remove from navigation
// Before:
{features.bracketSimulator && (
  <NavItem disabled>Bracket Simulator <Badge>Soon</Badge></NavItem>
)}

// After: Remove completely
```

---

## Verification Checklist

After fixing each issue, verify:

- [ ] **Waiver Wire**: Loads available players without 503 error
- [ ] **Daily Lineup (422)**: Can set Yahoo lineup without error
- [ ] **Calibration**: Shows brier score and calibration chart
- [ ] **Daily Lineup (Values)**: Players show varying implied runs/park factors
- [ ] **Bet History**: Only shows bets marked as placed
- [ ] **Risk Dashboard**: Can change settlement lookback, drawdown shows non-zero
- [ ] **My Roster**: Z-scores populated, undroppable varies by player
- [ ] **Bracket Simulator**: Removed from UI

## PROMPT END

---

## Usage

1. **Start with Priority 1** (Critical fixes)
2. **Test each fix** before moving to next priority
3. **Update HANDOFF.md** as issues are resolved
4. **Re-run UAT** after all fixes complete

## Expected Timeline

- **Priority 1**: 1-2 days (API issues)
- **Priority 2**: 2-3 days (Calibration, projections)
- **Priority 3**: 2-3 days (Data fixes, validation)
- **Priority 4**: 1 day (UI cleanup)

**Total: 6-9 days for all fixes**
