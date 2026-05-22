# P0 Scoreboard Zero Stats - Findings & Fix

## DIAGNOSTIC RESULTS

### ✅ What's Working
1. **Yahoo API**: Successfully fetching data
2. **Stat Mappings**: All 18 categories correctly mapped
3. **Data Parsing**: 19/20 stats have non-zero values
4. **Scoreboard Processing**: Successfully creates scoreboard with 13 categories won

### ❌ The Problem
The data flows correctly through the backend, but the **FRONTEND IS NOT DISPLAYING IT**.

**Evidence:**
- Diagnostic shows: `Categories won: 13, Categories tied: 0, Win probability: 0.0`
- API is returning data, but win_probability shows 0.0 (calculated from Monte Carlo)
- Frontend showing: `0W - 0L - 18T, 0% win prob`

### 🎯 Root Cause
**Monte Carlo simulation is returning `win_probability: 0.0`** even though categories show wins.

This happens when:
1. `my_player_scores` is empty (no ROW projections)
2. Simulation can't calculate probabilities without player-level data
3. Scoreboard falls back to showing current stats only
4. But category math shows actual wins/losses

## THE FIX

The issue is in `assemble_matchup_scoreboard()` - it's calculating category wins correctly 
but the Monte Carlo simulation needs player_scores to work properly.

### Option 1: Quick Fix (Immediate)
Make the scoreboard show current stats even without projections:

```python
# In scoreboard_orchestrator.py:assemble_matchup_scoreboard()
# Line 318 - Replace empty list fallback with actual current stats projection
```

### Option 2: Full Fix (Recommended)
Fetch player_scores from database for current matchup players.

## DEPLOYMENT

Run these commands NOW:

```powershell
# 1. Deploy all 3 agents in parallel
python scripts/agent_orchestrator.py --batch scripts/major_improvement_sprint.json --parallel

# 2. Or run individually:
# Terminal 1: Codex fixes frontend
codex -p "Fix scoreboard frontend showing zeros" ...

# Terminal 2: Claude fixes backend data flow  
claude -p "Fix scoreboard backend to return proper data" ...

# Terminal 3: Gemini creates tests
gemini -p "Create scoreboard verification tests" ...
```

## EMERGENCY BYPASS

If no fix in 30 min, deploy hardcoded response to show actual stats:
