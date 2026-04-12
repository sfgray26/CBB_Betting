> **Note:** This is a copy. The canonical version is in the repository root: `CLAUDE_FANTASY_ROADMAP_PROMPT.md`

---

# Prompt for Claude Code: Fantasy Baseball Elite Platform Implementation

> **Related:** `CLAUDE_TEAM_COORDINATION_PROMPT.md` (work coordination) | `CLAUDE_ARCHITECT_PROMPT_MARCH28.md` (emergency issues)

**Context:** Review `reports/FANTASY_BASEBALL_GAP_ANALYSIS.md` and implement the roadmap.

---

## PROMPT START

Implement the elite fantasy baseball platform roadmap in phases. Work systematically through each phase, updating HANDOFF.md as you complete features.

## PHASE A: Critical Fixes (Week 1-2)

### A1: Fix Yahoo API Reliability

**Problem:** UAT shows 422 errors on lineup set, 503 errors on waiver wire

**Implementation:**
```python
# backend/fantasy_baseball/yahoo_api_client.py

from tenacity import retry, stop_after_attempt, wait_exponential
import backoff

class ReliableYahooAPI:
    """Yahoo API client with retry logic and fallback."""
    
    def __init__(self):
        self.max_retries = 3
        self.base_delay = 2
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        retry=retry_if_exception_type((YahooAPIError, TimeoutError))
    )
    async def set_lineup(self, team_key, players):
        """Set lineup with validation and retry."""
        # Validate player game_ids match
        for player in players:
            game_id = await self.get_player_game_id(player['key'])
            if game_id != player['expected_game_id']:
                logger.warning(f"Game ID mismatch for {player['name']}")
                # Try to get current game_id
                player['game_id'] = game_id
        
        try:
            return await self._set_lineup_api(team_key, players)
        except YahooAPIError as e:
            if "game_ids don't match" in str(e):
                # Refresh player data and retry once
                await self.refresh_player_cache()
                raise  # Let retry handle it
            raise
    
    @retry(stop=stop_after_attempt(3))
    async def get_waiver_players(self, league_key):
        """Get waiver players without stats subresource."""
        # Fix: Use only 'metadata' not 'metadata,stats'
        return await self._get_players(
            league_key, 
            status='W',
            out='metadata'  # Fixed: removed 'stats'
        )
```

**Tests:**
```python
# tests/test_yahoo_api_reliability.py

@pytest.mark.asyncio
async def test_set_lineup_retry_on_422():
    api = ReliableYahooAPI()
    # Mock first call fails, second succeeds
    ...

@pytest.mark.asyncio  
async def test_waiver_players_no_stats_subresource():
    api = ReliableYahooAPI()
    players = await api.get_waiver_players('mlb.l.72586')
    assert 'stats' not in api.last_request_params['out']
```

**Deliverables:**
- [ ] `ReliableYahooAPI` class with retry logic
- [ ] Fix waiver wire 503 error (remove stats subresource)
- [ ] Fix lineup 422 error (validate game_ids)
- [ ] Unit tests with mocked failures
- [ ] Update existing code to use new client

---

### A2: Complete UAT Phase 2 Fixes

From `CLAUDE_UAT_FIXES_PROMPT.md`, complete:

**Calibration (Brier Score):**
```python
# backend/services/calibration.py

import numpy as np
from typing import List, Tuple
from datetime import datetime, timedelta

def calculate_brier_score(predictions: List[float], outcomes: List[int]) -> float:
    """Calculate Brier score for model calibration."""
    return np.mean((np.array(predictions) - np.array(outcomes)) ** 2)

def get_calibration_metrics(days: int = 30) -> dict:
    """Get calibration metrics for recent bets."""
    query = """
        SELECT predicted_prob, actual_outcome, created_at
        FROM bet_recommendations 
        WHERE outcome IS NOT NULL 
          AND created_at >= NOW() - INTERVAL '%s days'
    """
    
    results = db.execute(query, (days,))
    
    if not results:
        return {"error": "No completed bets"}
    
    predictions = [r[0] for r in results]
    outcomes = [r[1] for r in results]
    
    brier = calculate_brier_score(predictions, outcomes)
    
    # Calculate by probability bins
    bins = defaultdict(lambda: {'predicted': [], 'actual': []})
    for pred, outcome in zip(predictions, outcomes):
        bin_key = round(pred * 10) / 10
        bins[bin_key]['predicted'].append(pred)
        bins[bin_key]['actual'].append(outcome)
    
    calibration_data = []
    for bin_key, data in sorted(bins.items()):
        calibration_data.append({
            'bin': bin_key,
            'predicted_rate': np.mean(data['predicted']),
            'actual_rate': np.mean(data['actual']),
            'sample_size': len(data['actual'])
        })
    
    return {
        'brier_score': round(brier, 4),
        'sample_size': len(results),
        'calibration_by_bin': calibration_data,
        'interpretation': 'Lower is better. 0 = perfect, 0.25 = random'
    }

@app.get("/api/calibration")
async def get_calibration(days: int = 30):
    return get_calibration_metrics(days)
```

**MLB Projection Defaults:**
```python
# Debug why all players show 4.50/1.000/4.625

# Check pybaseball cache
import json
from pathlib import Path

cache_dir = Path("data/cache")
pitcher_cache = cache_dir / "pybaseball_pitching_2025.json"
batter_cache = cache_dir / "pybaseball_batting_2025.json"

# Verify cache exists and has data
if not pitcher_cache.exists():
    print("ERROR: Pitcher cache missing!")
    # Trigger refresh
    
# Check if xERA values are present
with open(pitcher_cache) as f:
    data = json.load(f)
    for name, stats in list(data['players'].items())[:5]:
        print(f"{name}: xERA = {stats.get('xera', 'MISSING')}")

# If xERA is missing or 0, that's the problem
```

**Fix:**
```python
# In mlb_analysis.py, add validation

def _load_pitcher_stats(self) -> Dict[str, float]:
    cache = load_pybaseball_pitchers(year=2025)
    
    # Validate cache has actual data
    valid_count = sum(1 for p in cache.values() if p.xera and p.xera > 0)
    if valid_count < 50:  # Arbitrary threshold
        logger.warning(f"Only {valid_count} valid pitcher xERA values!")
        # Trigger cache refresh or use fallback
    
    return cache
```

**Deliverables:**
- [ ] Brier score calculation endpoint
- [ ] Calibration visualization in UI
- [ ] Debug and fix MLB projection defaults
- [ ] Fix Odds Monitor portfolio fetch
- [ ] All UAT Phase 2 issues resolved

---

## PHASE B: Foundation Complete (Week 3-4)

### B1: Enhanced Dashboard

**Layout:**
```tsx
// frontend/components/Dashboard.tsx

export default function Dashboard() {
  return (
    <div className="dashboard-grid">
      <Panel title="Today's Lineup" priority="high">
        <LineupGaps />
        <HotColdStreaks />
        <ProbablePitchers />
      </Panel>
      
      <Panel title="Waiver Wire" priority="high">
        <WaiverTargets />
        <StreamerSuggestions />
      </Panel>
      
      <Panel title="Injuries & News" priority="medium">
        <InjuryFlags />
        <RecentNews />
      </Panel>
      
      <Panel title="Your Matchup" priority="medium">
        <CurrentMatchup />
        <CategoryStandings />
      </Panel>
    </div>
  );
}

// Lineup gaps detection
function LineupGaps() {
  const roster = useRoster();
  const lineup = useTodayLineup();
  
  const gaps = useMemo(() => {
    const required = ['C', '1B', '2B', '3B', 'SS', 'OF', 'OF', 'OF', 'UTIL'];
    const filled = lineup.players.map(p => p.position);
    return required.filter(pos => !filled.includes(pos));
  }, [lineup]);
  
  if (gaps.length === 0) return null;
  
  return (
    <Alert type="warning">
      Lineup gaps: {gaps.join(', ')}
      <Button href="/lineup">Fix Lineup</Button>
    </Alert>
  );
}

// Hot/cold streaks
function HotColdStreaks() {
  const players = useRoster();
  
  const streaks = useMemo(() => {
    return players.map(p => ({
      ...p,
      last7: calculateStats(p, days=7),
      last14: calculateStats(p, days=14),
      trend: calculateTrend(p)
    }));
  }, [players]);
  
  const hot = streaks.filter(p => p.trend === 'up').slice(0, 3);
  const cold = streaks.filter(p => p.trend === 'down').slice(0, 3);
  
  return (
    <div className="streaks">
      <StreakList title="🔥 Hot" players={hot} />
      <StreakList title="❄️ Cold" players={cold} />
    </div>
  );
}
```

**Backend:**
```python
# backend/api/dashboard.py

@app.get("/api/dashboard")
async def get_dashboard():
    user = get_current_user()
    
    return {
        'lineup_gaps': detect_lineup_gaps(user.team_key),
        'hot_streaks': get_hot_players(user.team_key, days=7, limit=3),
        'cold_streaks': get_cold_players(user.team_key, days=7, limit=3),
        'waiver_targets': get_waiver_suggestions(user.team_key, limit=5),
        'injury_flags': get_injury_risks(user.roster),
        'matchup_preview': get_matchup_preview(user.team_key),
        'probable_pitchers': get_probable_pitchers(user.league_key, days=7)
    }
```

---

### B2: User Preferences System

```python
# backend/models.py

class UserPreferences(Base):
    __tablename__ = 'user_preferences'
    
    user_id = Column(UUID, primary_key=True)
    
    # Notification settings
    notifications = Column(JSONB, default={
        'lineup_deadline': True,
        'injury_alerts': True,
        'waiver_suggestions': True,
        'trade_offers': True,
        'channels': ['email', 'push']
    })
    
    # Dashboard layout
    dashboard_layout = Column(JSONB, default={
        'panels': [
            {'id': 'lineup', 'position': 'top-left', 'size': 'large'},
            {'id': 'waiver', 'position': 'top-right', 'size': 'medium'},
            {'id': 'injuries', 'position': 'bottom-left', 'size': 'small'},
            {'id': 'matchup', 'position': 'bottom-right', 'size': 'medium'}
        ]
    })
    
    # Projection preferences
    projection_weights = Column(JSONB, default={
        'zips': 0.25,
        'steamer': 0.25,
        'depth_charts': 0.20,
        'atc': 0.15,
        'the_bat': 0.15
    })
```

---

## PHASE C: Analytical Core (Week 5-8)

### C1: Trade Analyzer

```python
# backend/services/trade_analyzer.py

class TradeAnalyzer:
    def analyze_trade(self, your_players, their_players, league_context):
        """
        Analyze trade impact on ROS standings.
        
        Returns:
            - Projected category changes
            - Win probability shift
            - Recommendation (accept/counter/decline)
        """
        
        # Current team projection
        current_ros = self.project_team_performance(
            league_context.your_roster
        )
        
        # Post-trade projection
        new_roster = league_context.your_roster.copy()
        new_roster.remove(your_players)
        new_roster.add(their_players)
        
        post_trade_ros = self.project_team_performance(new_roster)
        
        # Category impact
        category_changes = {}
        for cat in league_context.categories:
            before = current_ros.category_projections[cat]
            after = post_trade_ros.category_projections[cat]
            category_changes[cat] = {
                'before': before,
                'after': after,
                'change': after - before,
                'percent_change': (after - before) / before * 100
            }
        
        # Win probability
        current_wp = self.calculate_win_probability(
            current_ros, league_context.standings
        )
        post_trade_wp = self.calculate_win_probability(
            post_trade_ros, league_context.standings
        )
        
        return {
            'category_impact': category_changes,
            'win_probability_shift': post_trade_wp - current_wp,
            'recommendation': self.generate_recommendation(
                category_changes, post_trade_wp - current_wp
            )
        }
```

---

## PHASE D: Intelligence Layer (Week 9-12)

### D1: Standings Projection

```python
# backend/services/standings_projector.py

class StandingsProjector:
    def __init__(self, mcmc_simulator):
        self.simulator = mcmc_simulator
    
    def project_standings(self, league_key, simulations=10000):
        """
        Run Monte Carlo simulation of remaining season.
        
        Returns:
            - Final standings distribution
            - Category clinch probabilities
            - Playoff odds
        """
        league = get_league(league_key)
        remaining_games = league.schedule.remaining_games
        
        results = []
        for _ in range(simulations):
            # Simulate each remaining game
            final_standings = self.simulate_remaining_season(
                league.teams, remaining_games
            )
            results.append(final_standings)
        
        return {
            'playoff_probability': calculate_playoff_odds(results),
            'category_clinch_prob': calculate_clinch_odds(results),
            'expected_final_rank': np.mean([r.your_rank for r in results]),
            'best_case': np.percentile([r.your_rank for r in results], 10),
            'worst_case': np.percentile([r.your_rank for r in results], 90)
        }
```

---

## Implementation Notes

### Priority Order
1. **Phase A (Critical):** Fix UAT issues first
2. **Phase B (Foundation):** Make dashboard usable
3. **Phase C (Analytics):** Add differentiation
4. **Phase D (Intelligence):** Proactive features

### Testing Strategy
- Unit tests for all calculations
- Integration tests with Yahoo sandbox
- Manual testing weekly with real league

### Documentation
- Update HANDOFF.md after each phase
- Add inline code comments
- Create user-facing help docs

## PROMPT END

---

## Usage

**Phase by Phase:**
```bash
# Week 1-2
claude "Implement Phase A from CLAUDE_FANTASY_ROADMAP_PROMPT.md"

# Week 3-4  
claude "Implement Phase B - Enhanced Dashboard"

# etc.
```

**Or All At Once:**
```bash
claude "Read CLAUDE_FANTASY_ROADMAP_PROMPT.md and implement systematically. Update HANDOFF.md after each phase completion."
```

