# OpenClaw Autonomy Spec v4.0 — MLB Betting Model Addendum

> **Status:** SCOPE CHANGE — MLB betting model now required  
> **Context:** CBB ends Apr 7, MLB already active, transition overlap Mar 28-Apr 7  
> **Deliverable:** MLB betting model + OpenClaw MLB patterns  

---

## 1. The Transition Challenge

### Current Reality (March 24, 2026)

| Sport | Status | Ends/Starts |
|-------|--------|-------------|
| **CBB Betting** | Active | Ends Apr 7 (championship) |
| **MLB Season** | **ALREADY STARTED** | Began Mar 28 |
| **Fantasy Baseball** | Active | Ongoing |

**Critical Gap:** Only CBB betting model exists. MLB betting model NOT BUILT.

**Time Constraint:** ~14 days to build MLB betting model before CBB ends.

---

## 2. MLB Betting Model — Core Components

### 2.1 Sport-Specific Configuration

```python
# backend/core/sport_config.py

class MLBConfig:
    """MLB-specific betting configuration."""
    
    SPORT_KEY = "baseball_mlb"
    
    # Market types (different from CBB spread)
    MARKETS = {
        'runline': 'spreads',      # MLB equivalent of spread (-1.5, +1.5)
        'total': 'totals',          # Over/under (7.5, 8.5, etc.)
        'moneyline': 'h2h',         # Straight winner
    }
    
    # Kelly sizing (different from CBB)
    BASE_SD = 4.0  # Runs (vs 11.0 points for CBB)
    HOME_ADVANTAGE = 0.25  # Slight home field in MLB
    MAX_KELLY_FRACTION = 0.05
    FRACTIONAL_KELLY_DIVISOR = 4
    
    # Key differentiators from CBB
    STARTING_PITCHER_WEIGHT = 0.40  # 40% of prediction
    BULLPEN_WEIGHT = 0.25           # 25% of prediction
    OFFENSE_WEIGHT = 0.25           # 25% of prediction
    DEFENSE_WEIGHT = 0.10           # 10% of prediction
```

### 2.2 Data Sources

| Source | CBB Uses | MLB Uses | Status |
|--------|----------|----------|--------|
| **KenPom** | Primary (45%) | ❌ N/A | MLB: Use FanGraphs |
| **BartTorvik** | Secondary (40%) | ❌ N/A | MLB: Use Baseball-Reference |
| **EvanMiya** | Tertiary (15%) | ❌ N/A | MLB: Use Statcast |
| **FanGraphs** | ❌ | **Primary (40%)** | Need API/scraper |
| **Baseball-Reference** | ❌ | **Secondary (35%)** | Need scraper |
| **Statcast** | ❌ | **Tertiary (25%)** | pybaseball integration |

**Critical Gap:** Need to build MLB data ingestion layer.

### 2.3 MLB Nightly Analysis Pipeline

```python
# backend/services/mlb_analysis.py (NEW)

async def run_mlb_nightly_analysis(date: date) -> List[Prediction]:
    """
    MLB equivalent of run_daily_analysis() for CBB.
    
    Steps:
    1. Fetch MLB schedule for date
    2. Get starting pitcher matchups
    3. Pull team offense/defense stats
    4. Calculate runline/total projections
    5. Compare to market odds
    6. Generate Kelly-sized picks
    """
    
    # 1. Fetch games
    games = await fetch_mlb_schedule(date)
    
    # 2. For each game, get key data
    for game in games:
        pitchers = await fetch_starting_pitchers(game)
        offense = await fetch_team_offense(game.home_team, game.away_team)
        bullpen = await fetch_bullpen_stats(game.home_team, game.away_team)
        
        # 3. Calculate projections
        runline_proj = calculate_runline_projection(pitchers, offense, bullpen)
        total_proj = calculate_total_projection(pitchers, offense, bullpen, game.venue)
        
        # 4. Compare to market
        market_line = await get_mlb_odds(game.id)
        edge = calculate_edge(runline_proj, market_line)
        
        # 5. Generate pick if edge exists
        if edge > MIN_EDGE_THRESHOLD:
            picks.append(create_mlb_pick(game, edge, kelly_size))
    
    return picks
```

---

## 3. OpenClaw MLB Patterns (New)

OpenClaw must detect MLB-specific vulnerabilities:

### 3.1 MLB Pattern Dimensions

| Dimension | CBB Pattern | MLB Pattern | Detection Method |
|-----------|-------------|-------------|------------------|
| **Pitcher Form** | N/A | Last 3 starts ERA vs season | Pitcher logs |
| **Bullpen Fatigue** | N/A | Pitches in last 3 days | Box scores |
| **Park Factor** | N/A | Coors Field (hitter) vs Petco (pitcher) | Statcast data |
| **Weather** | N/A | Wind speed/direction at Wrigley | Weather API |
| **Travel** | Conference play | Cross-country, no off-day | Schedule analysis |
| **Day/Night** | N/A | Team splits day vs night | Splits data |
| **Left/Right** | N/A | Team vs LHP/RHP splits | Splits data |

### 3.2 MLB-Specific OpenClaw Alerts

```python
MLB_VULNERABILITY_PATTERNS = {
    'pitcher_fatigue': {
        'detection': 'Starter >100 pitches in previous 2 starts',
        'impact': 'ERA +1.5 in next start',
        'severity': 'HIGH',
    },
    'bullpen_overuse': {
        'detection': 'Bullpen >150 pitches in last 3 days',
        'impact': 'Late-game runs +2.0',
        'severity': 'MEDIUM',
    },
    'coors_field': {
        'detection': 'Game at Coors Field',
        'impact': 'Total runs +2.5 vs league avg',
        'severity': 'HIGH',  # Always factor in
    },
    'wind_at_wrigley': {
        'detection': 'Wind >15 mph blowing out at Wrigley',
        'impact': 'Home runs +3x',
        'severity': 'HIGH',
    },
    'travel_no_rest': {
        'detection': 'Team played on West Coast yesterday, game today on East Coast',
        'impact': 'Win rate -15%',
        'severity': 'MEDIUM',
    },
}
```

---

## 4. Sport Transition Timeline

### Phase 0: Immediate (Mar 24-28)
**Goal:** Build MLB data layer

- [ ] FanGraphs scraper/API integration
- [ ] Baseball-Reference scraper  
- [ ] Starting pitcher matchup fetcher
- [ ] Bullpen stats calculator
- [ ] Park factor database

### Phase 1: Parallel Operation (Mar 28-Apr 7)
**Goal:** Both CBB and MLB models active

- [ ] MLB nightly analysis pipeline (separate from CBB)
- [ ] MLB OpenClaw integrity checks
- [ ] MLB-specific Discord alerts
- [ ] A/B test: CBB vs MLB model performance

**Daily Schedule:**
```
3:00 AM ET — CBB analysis (existing)
9:00 AM ET — MLB analysis (NEW)
7:00 AM ET — Morning brief (CBB + MLB)
```

### Phase 2: CBB Wind Down (Apr 1-7)
**Goal:** Graceful CBB → MLB transition

- [ ] CBB model pauses after championship
- [ ] Archive CBB data/models
- [ ] Full MLB focus

### Phase 3: Full MLB (Apr 8+)
**Goal:** MLB-only betting model

- [ ] CBB code archived
- [ ] MLB model optimized
- [ ] OpenClaw MLB patterns fully trained

---

## 5. Resource Allocation (Critical)

**Claude Code Priority Split:**

| Priority | Task | Timeline | Owner |
|----------|------|----------|-------|
| **P0** | Waiver wire fixes | 2-3 days | Claude |
| **P0** | MLB data layer (FanGraphs/BR) | 3-4 days | Claude |
| **P0** | MLB nightly analysis pipeline | 4-5 days | Claude |
| **P1** | MLB OpenClaw patterns | Parallel | Claude |
| **P2** | Full OpenClaw autonomy (v4.0) | Post-Apr 7 | Kimi/Claude |

**Reality Check:** With ~14 days until CBB ends, Claude must parallelize:
- Fix waiver wire (immediate revenue protection)
- Build MLB betting model (future revenue)

---

## 6. Risk: Incomplete Transition

**If MLB model isn't ready by Apr 7:**
- ❌ No betting picks for 2-3 weeks (lost revenue)
- ❌ No model continuity
- ❌ Loss of betting edge during critical early season

**Mitigation:**
- Start MLB build NOW (parallel to waiver wire)
- Use simplified model initially (pitcher ERA + park factor only)
- Gradually add complexity (bullpen, weather, splits)

---

## 7. Updated HANDOFF.md Actions

### Immediate (Next 48 Hours)
1. **Claude Code:** Scope MLB data sources (FanGraphs API vs scraper)
2. **Claude Code:** Build minimal MLB betting model (MVP)
3. **Kimi CLI:** Research MLB betting best practices, model architectures

### This Week
4. **Claude Code:** Parallel development — waiver wire + MLB model
5. **Gemini CLI:** Verify MLB odds API access (The Odds API)
6. **Kimi CLI:** Design MLB-specific OpenClaw patterns

### Next Week (Mar 31-Apr 7)
7. **Claude Code:** Full MLB pipeline operational
8. **OpenClaw:** MLB integrity checks live
9. **All:** Overlap testing (CBB + MLB)

---

**Document Version:** MLB-ADDENDUM-v1  
**Last Updated:** March 24, 2026  
**Status:** URGENT — MLB model required before Apr 7
