# Hybrid Fantasy-Betting Edge Framework
## Leveraging Odds APIs for H2H One Win Fantasy Baseball

**Date:** April 8, 2026  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**For:** Claude Code (Principal Architect)  
**Status:** Strategic Framework - Ready for Implementation Review

---

## Executive Summary

Pure fantasy managers optimize for **statistical projections**. Pure bettors optimize for **market prices**. The **hybrid edge** comes from exploiting the gap between:

1. **Fantasy projection systems** (Steamer, ZiPS, ATC) — slow-moving, fundamentals-based
2. **Betting markets** (The Odds API aggregation) — fast-moving, wisdom-of-crowds
3. **Situational factors** (weather/park) — often mispriced by both

**Core Thesis:** When fantasy projections and betting markets diverge significantly, one is wrong. The hybrid player identifies which—and exploits it.

---

## Part 1: The Information Arbitrage Framework

### 1.1 The Gap Model

```
FANTASY PROJECTION ──────GAP────── BETTING MARKET
      (Slow)                         (Fast)
   Fundamentals                  Wisdom of Crowds
   3-6 month horizon             Real-time updates
```

**The Edge:** Markets react to news (lineup changes, weather, injuries) in **minutes**. Fantasy projections update in **days**.

**Example Workflow:**
1. Weather alert: Wind blowing out 20mph at Wrigley (3 hours before game)
2. Betting market: Cubs total moves from 8.5 to 9.5 (+12% run expectation)
3. Fantasy projections: Still showing yesterday's baseline (no weather adjustment)
4. **Edge:** Start Cubs hitters / Fade Cubs pitchers before your opponent notices

### 1.2 Closing Line Value (CLV) Applied to Fantasy

**Traditional betting CLV:** Did you beat the closing line?
**Fantasy CLV:** Did you roster/start players before the market adjusted?

| Signal | Betting Market | Fantasy Action |
|--------|---------------|----------------|
| Line drops from -150 to -200 (heavy favorite action) | Sharp money on Team A | Start Team A's hitters (higher win probability = more R/RBI) |
| Total moves from 8 to 9.5 (wind out) | Over money | Stack both teams' hitters |
| Pitcher props move (K line drops from 6.5 to 5.5) | Market fading pitcher | Start opposing hitters |
| Player prop moves (HR odds shorten from +450 to +350) | HR speculation | Start that hitter |

---

## Part 2: Hybrid Data Architecture

### 2.1 New Data Models

```python
class MarketFantasyDivergence(BaseModel):
    """The core edge detection model"""
    player_id: str
    game_id: str
    
    # Fantasy baseline
    fantasy_projection: float  # From your system (Steamer/ZiPS-based)
    fantasy_category_expectation: Dict[Category, float]
    
    # Market signals
    implied_run_total: float  # From The Odds API (over/under)
    implied_team_win_prob: float  # From moneyline
    player_prop_signals: Dict[str, float]  # K prop, HR odds, etc.
    
    # The divergence
    market_fantasy_delta: float  # Percentage difference
    divergence_direction: Literal["MARKET_BULLISH", "MARKET_BEARISH", "ALIGNED"]
    
    # Situational context
    weather_adjustment_applied: bool
    park_factor_applied: bool
    lineup_confirmed: bool
    
    # Confidence
    signal_strength: Literal["STRONG", "MODERATE", "WEAK"]
    time_to_game_hours: float
    
class HybridEdgeAlert(BaseModel):
    """Actionable alerts for H2H weekly lineup decisions"""
    alert_type: Literal["MARKET_FADE", "MARKET_BOOST", "SITUATIONAL_EDGE", "LINEUP_CONFIRMED"]
    priority: int  # 1-10 (10 = must-act)
    
    player: Player
    current_fantasy_projection: float
    market_adjusted_projection: float
    
    # Why the edge exists
    market_evidence: str  # "Total moved from 8 to 9.5; HR odds shortened"
    fantasy_lag_reason: str  # "Projection from 3 days ago; no weather update"
    
    # Recommended action
    action: Literal["START", "SIT", "PICKUP", "DROP"]
    confidence: float  # 0-1
    
    # Urgency
    valid_until: datetime  # When lineup locks or market fully adjusts

class WeeklyMarketContext(BaseModel):
    """Week-level view for H2H One Win strategy"""
    week: int
    
    # Market sentiment
    high_scoring_environments: List[Game]  # Totals > 9.5
    low_scoring_environments: List[Game]   # Totals < 7.5
    heavily_favored_teams: List[Team]      # Win prob > 70%
    underdog_value_spots: List[Team]       # Win prob < 35%
    
    # Category implications
    hr_friendly_games: List[Game]          # Wind out, Coors, etc.
    sb_friendly_games: List[Game]          # Pitchers with high steal rates
    
    # Your roster alignment
    exposure_to_high_total_games: float    # % of roster in high-scoring env
    exposure_to_favored_teams: float       # % of roster on heavy favorites
```

### 2.2 Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID EDGE DETECTION LAYER                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   FANTASY    │      │   MARKET     │      │  SITUATIONAL │  │
│  │  PROJECTIONS │      │   DATA       │      │   DATA       │  │
│  │              │      │  (Odds API)  │      │              │  │
│  │ • Steamer    │      │              │      │ • Weather    │  │
│  │ • ZiPS       │      │ • Totals     │      │ • Park       │  │
│  │ • ATC        │      │ • ML         │      │ • Lineups    │  │
│  │ • ATC        │      │ • Props      │      │ • Rest       │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                     │          │
│         └─────────────────────┼─────────────────────┘          │
│                               ▼                                │
│                   ┌──────────────────────┐                     │
│                   │  DIVERGENCE ENGINE   │                     │
│                   │                      │                     │
│                   │ Compare:             │                     │
│                   │ - Fantasy vs Market  │                     │
│                   │ - Speed of update    │                     │
│                   │ - Confidence levels  │                     │
│                   └──────────┬───────────┘                     │
│                              ▼                                 │
│                   ┌──────────────────────┐                     │
│                   │   EDGE DETECTION     │                     │
│                   │                      │                     │
│                   │ Signals:             │                     │
│                   │ • Market > Fantasy   │                     │
│                   │ • Weather gap        │                     │
│                   │ • Prop movement      │                     │
│                   └──────────┬───────────┘                     │
│                              ▼                                 │
│                   ┌──────────────────────┐                     │
│                   │  H2H LINEUP OPTIMIZER│                     │
│                   │  (6-3 Win Target)    │                     │
│                   └──────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Specific Edge Patterns

### 3.1 Pattern 1: The Weather-Market Disconnect

**The Gap:** Betting markets adjust totals within minutes of weather forecasts. Fantasy projections update... never (static).

**Detection Algorithm:**
```python
def detect_weather_market_gap(game, fantasy_projections):
    """
    Returns edge when market has adjusted but fantasy hasn't
    """
    market_total = get_odds_api_total(game.id)
    baseline_total = game.season_avg_total  # ~8.5 for MLB
    
    # Market has moved significantly
    if market_total > baseline_total + 1.0:  # Total now 9.5+
        weather = get_weather_forecast(game.stadium)
        
        # Verify it's weather-driven (not just team quality)
        if weather.wind_speed > 15 and weather.wind_direction == "OUT":
            for player in game.home_team.hitters + game.away_team.hitters:
                fantasy_proj = fantasy_projections[player.id]
                
                # Fantasy projection hasn't adjusted
                if fantasy_proj.hr == baseline_hr_projection:
                    yield HybridEdgeAlert(
                        player=player,
                        alert_type="SITUATIONAL_EDGE",
                        action="START",
                        market_evidence=f"Total moved to {market_total}; wind {weather.wind_speed}mph out",
                        fantasy_lag_reason="Static projection; no weather adjustment"
                    )
```

**H2H One Win Application:**
- **HR/TB/RBI categories:** Start hitters in high-total games
- **ERA/WHIP categories:** Fade pitchers in those same games
- **Timing:** Get this edge before opponent (who uses static projections) sets lineup

### 3.2 Pattern 2: The Prop-to-Fantasy Translation

**The Gap:** Player prop markets are hyper-efficient (react to news in seconds). Fantasy leagues don't see prop markets at all.

**Translation Table:**
| Prop Market Movement | Fantasy Implication | Confidence |
|---------------------|---------------------|------------|
| K prop drops from 6.5 to 5.5 | Pitcher has reduced stuff/injury | High |
| HR odds shorten from +500 to +350 | Power surge, good matchup, or wind | Medium |
| Hits over/under moves up | Batting eye improved, good pitcher matchup | Medium |
| SB odds shorten | Green light to run / catcher has weak arm | High |
| Total bases over moves up | Expecting extra-base hits | Medium |

**Implementation:**
```python
class PropToFantasyTranslator:
    """Convert betting prop signals to fantasy category expectations"""
    
    def translate_k_prop(self, pitcher, line_movement):
        """Strikeout prop movement → K category expectation"""
        if line_movement.direction == "DOWN" and line_movement.magnitude > 1.0:
            # Market fading pitcher's K ability
            return CategoryAdjustment(
                category=Category.K,  # Pitcher Ks (good)
                adjustment=-0.5,     # Expect fewer
                confidence=0.7,
                reason=f"K prop dropped from {line_movement.open} to {line_movement.current}"
            )
    
    def translate_hr_odds(self, hitter, odds_change):
        """HR odds shortening → HR/TB/OPS boost"""
        if odds_change.implied_prob_change > 0.05:  # +5% HR probability
            return {
                Category.HR: 0.3,    # +0.3 HR expected
                Category.TB: 0.8,    # +0.8 total bases
                Category.OPS: 0.050  # +50 OPS points
            }
```

### 3.3 Pattern 3: The Moneyline-to-Category Correlation

**Framework:** Team win probability correlates with counting stats

| Win Probability | Expected Impact | Fantasy Categories |
|-----------------|-----------------|-------------------|
| >70% (heavy favorite) | More plate appearances (less walk-off risk), higher run support | R, RBI, W (pitcher) |
| <35% (heavy underdog) | Fewer PA, less run support, more save opps for reliever | Saves (RP), Losses (pitcher) |
| Close game (50/50) | More late-game situations, possible extra innings | All categories amplified |

**Implementation:**
```python
def calculate_win_prob_category_boost(team, moneyline_odds):
    """
    Convert moneyline odds to expected category deltas
    """
    # Convert odds to implied probability
    if moneyline_odds < 0:  # Favorite
        win_prob = abs(moneyline_odds) / (abs(moneyline_odds) + 100)
    else:  # Underdog
        win_prob = 100 / (moneyline_odds + 100)
    
    # Category adjustments based on win probability
    adjustments = {}
    
    if win_prob > 0.65:  # Heavy favorite
        adjustments = {
            Category.R: 0.15,      # More runs scored
            Category.RBI: 0.10,    # More RBI opportunities
            Category.W: 0.08,      # Pitcher win probability
            Category.NSV: -0.05    # Fewer save opportunities
        }
    elif win_prob < 0.35:  # Heavy underdog
        adjustments = {
            Category.R: -0.10,
            Category.RBI: -0.08,
            Category.L: 0.10,      # Pitcher loss probability
            Category.NSV: 0.08     # More save opportunities (if winning)
        }
    
    return adjustments
```

---

## Part 4: H2H One Win Optimization with Hybrid Data

### 4.1 The 6-3 Compass with Market Overlay

Traditional compass: "Am I set up for 6-3 based on projections?"
**Hybrid compass:** "Am I set up for 6-3 based on projections PLUS market information?"

```
┌─────────────────────────────────────────────┐
│ WEEKLY HYBRID COMPASS                       │
├─────────────────────────────────────────────┤
│                                             │
│ BASELINE (Fantasy Projections):             │
│ • HR: 65% win prob                          │
│ • RBI: 58% win prob                         │
│ • R: 52% win prob                           │
│                                             │
│ MARKET ADJUSTMENTS:                         │
│ • 3 players in high-total games (+10% HR)   │
│ • 2 pitchers facing teams with high ML odds │
│ • Wind out at Wrigley (K prop moved)        │
│                                             │
│ ADJUSTED PROBABILITIES:                     │
│ • HR: 72% (+7%) 🟢 STRONG                   │
│ • RBI: 65% (+7%) 🟢 LIKELY                  │
│ • R: 58% (+6%) 🟡 SWING                     │
│                                             │
│ MARKET-BOOSTED WIN PROB: 74% (+6%)          │
│                                             │
│ [View Market Signals] [Adjust Lineup]       │
└─────────────────────────────────────────────┘
```

### 4.2 Category-Specific Market Signals

| Your Category | Market Signal to Watch | Edge Detection |
|---------------|----------------------|----------------|
| **HR** | HR player prop odds, totals | Prop shortening + high total = stack |
| **SB** | SB odds, catcher framing stats | SB odds shorten + weak catcher arm = start |
| **R/RBI** | Team totals, moneylines | Favored team with high total = run boost |
| **AVG/OPS** | Hits props, batting order position | Lineup spot upgrade (market sees it first) |
| **K (pitcher)** | K props, moneylines | K prop up + big favorite = dominant start |
| **W** | Moneylines, bullpen strength | Heavy favorite + weak opponent = high win prob |
| **ERA/WHIP** | Totals, opponent implied runs | Low total + big favorite = quality start |
| **QS** | Totals, moneylines, park factors | Low total, pitcher-friendly park = QS likely |
| **NSV** | Moneylines, save props | Close game probability, setup man usage |

### 4.3 Weekly Streaming with Market Intel

**Traditional streaming:** Pick up players with good matchups based on opponent ERA.

**Hybrid streaming:** Pick up players where **market is more bullish than fantasy community**.

```python
def find_hybrid_streaming_targets(free_agents, upcoming_games):
    """
    Find waiver wire adds where market signals are strong
    """
    targets = []
    
    for player in free_agents:
        # Get player's upcoming games
        games = [g for g in upcoming_games if player.team in [g.home, g.away]]
        
        # Get market data
        market_totals = [get_total(g) for g in games]
        market_win_probs = [get_implied_win_prob(g, player.team) for g in games]
        
        # Check for prop movement
        prop_signals = get_player_prop_movement(player)
        
        # Score the opportunity
        opportunity_score = 0
        
        # High totals = HR/RBI opportunities
        if mean(market_totals) > 9.0:
            opportunity_score += 2 if player.position in ['LF', 'CF', 'RF', '1B'] else 0
        
        # Favorable moneylines = counting stats
        if mean(market_win_probs) > 0.60:
            opportunity_score += 1
        
        # Prop market bullish
        if prop_signals and prop_signals.bullish_score > 0.7:
            opportunity_score += 2
        
        # Fantasy ownership low (not widely rostered)
        if player.roster_pct < 30 and opportunity_score >= 3:
            targets.append({
                'player': player,
                'score': opportunity_score,
                'market_signals': {
                    'avg_total': mean(market_totals),
                    'avg_win_prob': mean(market_win_probs),
                    'prop_signals': prop_signals
                },
                'recommendation': 'HIGH_PRIORITY_ADD'
            })
    
    return sorted(targets, key=lambda x: x['score'], reverse=True)
```

---

## Part 5: Risk Management & Validation

### 5.1 When NOT to Trust the Market

Markets are wrong sometimes. Framework for **filtering false signals**:

| Market Signal | Filter | Risk |
|--------------|--------|------|
| Line movement | <2 hours before game | Steam chasing (fake sharp action) |
| Prop movement | No injury/news catalyst | Reverse line movement trap |
| High total | No weather confirmation | Weather forecast error |
| Heavy favorite | Pitcher is fatigued (high recent IP) | Overvalued win prob |

### 5.2 Confidence Scoring

```python
class SignalConfidence:
    def calculate(self, divergence):
        score = 0.5  # Baseline
        
        # Time to game (urgency)
        if divergence.time_to_game_hours < 2:
            score += 0.2  # Market more reliable close to game
        elif divergence.time_to_game_hours > 24:
            score -= 0.1  # Lots can change
        
        # Market consensus
        if divergence.bookmaker_agreement > 0.8:  # 80% of books agree
            score += 0.2
        elif divergence.bookmaker_agreement < 0.5:
            score -= 0.2  # Disagreement = uncertainty
        
        # Lineup confirmation
        if divergence.lineup_confirmed:
            score += 0.1
        else:
            score -= 0.1  # Risk of lineup scratch
        
        # Weather confidence
        if divergence.weather_confidence == "HIGH":
            score += 0.1
        
        return min(max(score, 0), 1)  # Clamp 0-1
```

### 5.3 Backtesting the Hybrid Approach

**Validation framework:**
1. **Historical simulation:** For 2024 season, apply hybrid signals to weekly lineups
2. **Benchmark:** Compare vs. pure fantasy projections
3. **Metric:** Win rate improvement in H2H One Win format
4. **Statistical test:** Is the improvement significant (p < 0.05)?

**Expected results:**
- Pure fantasy: ~50% win rate (baseline)
- Hybrid approach: 55-60% win rate (if edge is real)
- Market-only: 52-53% (markets efficient, but not tailored to fantasy categories)

---

## Part 6: Implementation Roadmap

### Phase 1: Market Data Integration (Week 1-2)
- [ ] Integrate The Odds API for totals and moneylines
- [ ] Store historical line movements (for backtesting)
- [ ] Create `MarketFantasyDivergence` detection job

### Phase 2: Prop Market Layer (Week 3-4)
- [ ] Add player prop feeds (K props, HR odds)
- [ ] Build `PropToFantasyTranslator`
- [ ] Create alert system for significant prop moves

### Phase 3: Hybrid Optimization Engine (Week 5-6)
- [ ] Enhance H2HOneWinSimulator with market signals
- [ ] Integrate weather/park factors (from K-29 spec)
- [ ] Build confidence scoring system

### Phase 4: UI/UX Integration (Week 7-8) - DELEGATED
- [ ] "Market Signals" badge on player cards
- [ ] Weekly Hybrid Compass view
- [ ] Streaming recommendations with market intel

---

## Part 7: Unique Selling Proposition (USP)

### What This Enables (That Competitors Don't)

| Feature | ESPN/Yahoo | Pure Betting Apps | Your Hybrid App |
|---------|-----------|-------------------|-----------------|
| Fantasy projections | ✅ | ❌ | ✅ + market overlay |
| Betting odds | ❌ | ✅ | ✅ + fantasy context |
| Weather integration | ❌ | Partial | ✅ Full physics model |
| Prop-to-fantasy translation | ❌ | ❌ | ✅ Unique |
| 6-3 win optimization | ❌ | ❌ | ✅ H2H One Win specific |
| Market speed + fantasy depth | ❌ | ❌ | ✅ Best of both |

**The Pitch:**
> "Most fantasy apps use projections from last week. Most betting apps don't understand fantasy categories. We combine real-time market intelligence with fantasy optimization—so you start players **before** the market adjusts, not after."

---

## Decision Required from Claude

### Option A: Full Hybrid Integration (Recommended)
**Scope:** Fantasy + The Odds API + Weather/Park + Prop markets  
**Timeline:** 8 weeks  
**Risk:** Complexity, API costs  
**Reward:** Sustainable competitive advantage

### Option B: Fantasy + Weather Only (MVP)
**Scope:** Park factors + weather (no betting markets)  
**Timeline:** 4 weeks  
**Risk:** Lower  
**Reward:** Solid edge, less differentiation

### Option C: Market Monitoring Only
**Scope:** Display odds as "additional info" only  
**Timeline:** 2 weeks  
**Risk:** Minimal  
**Reward:** Feature tick, not true edge

**Recommendation:** Option A for true differentiation, Option B if resource-constrained. The hybrid approach is defensible—competitors would need to build both fantasy AND betting infrastructure.

---

**Document Status:** Ready for architecture review and implementation planning.
