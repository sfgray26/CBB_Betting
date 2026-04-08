# Weather & Park Factors Integration Specification
## Fantasy Baseball H2H One Win App

**Date:** April 8, 2026  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**For:** Claude Code (Principal Architect)  
**Status:** Research Complete - Awaiting Architecture Review  

---

## Executive Summary

Weather and park factors represent a **15-30% variance** in baseball outcomes—the largest systematic, exploitable edge in fantasy baseball. Academic research (Dr. Alan Nathan, SABR, Climate Central) confirms:

- **Temperature:** +3-4 feet per 10°F (1°F ≈ 1% HR probability change)
- **Altitude:** +6 feet per 1,000 ft (Coors Field = +30 feet vs sea level)
- **Climate Change:** +500 home runs since 2010 due to warming

**Critical for H2H One Win Format:** Daily lineup optimization requires game-specific environmental adjustments, not seasonal averages. A player at Coors on an 85°F day has **~40% higher home run probability** than the same player at Oracle on a 55°F day—but most fantasy apps treat these as equivalent.

---

## 1. Research Synthesis: Key Findings

### 1.1 Physics Foundations (Dr. Alan Nathan, University of Illinois)

**Trajectory Calculator Model:**
```
Distance = f(exit_velocity, launch_angle, spin_rate, 
              temperature, pressure, humidity, altitude)
```

**Validated Rules of Thumb:**
| Variable | Effect | Data Source |
|----------|--------|-------------|
| Temperature | +3.3 ft per 10°F | Nathan (2019) |
| Altitude | +6 ft per 1,000 ft | Nathan / SABR (2014) |
| Pressure | +2 ft per 0.3 inHg | Weather Applied Metrics |
| Humidity | +1 ft per 50% (net effect) | WAM (2025) |

**Magnus Force Implications:**
- Thinner air (altitude/heat) = less curveball break
- Coors Field: Curveball loses ~4 inches of break
- Pitchers adjust: Fewer curves, more changeups (second-order effect)

### 1.2 Statistical Models

**Traditional Park Factor:**
```
PF = (Home RS + Home RA) / (Road RS + Road RA) × 100
```
- **Limitation:** 3-year sample required to stabilize
- **Bias:** Team quality, divisional schedule

**Modern Batted-Ball Modeling (Ballpark Pal):**
```
HR_Prob = Physics_Model(exit_velo, launch_angle, environment)
```
- **Advantage:** Isolates environment from player talent
- **Precision:** 1M+ batted balls aggregated

### 1.3 Climate Change Research (Climate Central, 2019)

- **Current Impact:** +58 home runs/year (2010-2019) due to warming
- **2050 Projection:** +182 home runs/year if warming continues
- **Fantasy Implication:** Late-season games (July-September) have systematically higher run environments than April games

---

## 2. Application to H2H One Win Format

### 2.1 Why This Matters More for H2H One Win

**Format Constraint:** You need **consistent 6-3 wins**, not maximum points.

**Weather/Park as Variance Control:**
| Strategy | High Variance | Low Variance |
|----------|--------------|--------------|
| Start pitcher at Coors Field | Explosive downside (ERA 6.00+) | — |
| Start pitcher at Oracle/Petco | — | Predictable 3.50 ERA |
| Stack hitters at Coors | Boom potential (3+ HR) | — |
| Spread hitters across neutral parks | — | Steady 2-4 category wins |

**Key Insight:** Weather/park data allows **risk stratification**—critical for format requiring consistency.

### 2.2 Category-Specific Impacts

Your 9 hitting categories affected by environment:

| Category | Park/Weather Impact | Magnitude |
|----------|---------------------|-----------|
| **HR** | Extreme (Coors +30%, Oracle -18%) | ±30% |
| **TB** | High (HR-dependent) | ±25% |
| **RBI** | Moderate (scoring opportunity) | ±15% |
| **R** | Moderate (team-dependent) | ±15% |
| **H** | Moderate (BABIP effects) | ±10% |
| **AVG** | Low-Moderate | ±8% |
| **OPS** | High (SLG-driven) | ±20% |
| **NSB** | Low (independent of ball flight) | ±5% |
| **K (batter)** | Moderate (plate discipline) | ±10% |

**Pitching Categories:**
| Category | Park/Weather Impact | Magnitude |
|----------|---------------------|-----------|
| **ERA** | Extreme | ±30% |
| **WHIP** | High | ±20% |
| **K** | Moderate (swinging strikes) | ±10% |
| **K/9** | Moderate | ±10% |
| **W** | Low (team-dependent) | ±8% |
| **QS** | Moderate (IP-dependent) | ±15% |
| **HR (pitcher)** | Extreme | ±35% |
| **NSV** | Low (situation-dependent) | ±5% |

---

## 3. Required Data Architecture

### 3.1 New Data Models

```python
class ParkFactor(BaseModel):
    """Seasonal park factors by category"""
    park_id: str  # e.g., "COL", "SFG"
    season: int
    
    # Standard park factors (100 = neutral)
    run_factor: float  # Coors: ~128, Oracle: ~82
    hr_factor: float
    double_factor: float
    triple_factor: float
    walk_factor: float
    strikeout_factor: float
    
    # Physics baseline
    elevation_ft: int  # Coors: 5280
    latitude: float    # Affects temperature baseline
    
    # Derived
    air_density_baseline: float  # Calculated from elevation
    
class WeatherForecast(BaseModel):
    """Game-specific weather (updated every 6 hours)"""
    game_id: str
    game_datetime: datetime
    
    # Forecast (0-6 hours: high confidence)
    temperature_f: float
    humidity_pct: float
    pressure_inhg: float
    wind_speed_mph: float
    wind_direction: str  # "Out to LF", "In from RF", etc.
    
    # Confidence
    forecast_horizon_hours: float
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    
    # Derived physics
    air_density_game: float  # Calculated
    distance_adjustment_ft: float  # vs. standard conditions
    
class EnvironmentAdjustedProjection(BaseModel):
    """Player projection adjusted for game environment"""
    player_id: str
    game_id: str
    
    # Baseline (from Steamer/ZiPS)
    baseline_hr_prob: float
    baseline_era: float
    
    # Adjustments
    park_factor_multiplier: float
    weather_multiplier: float
    combined_multiplier: float
    
    # Adjusted projections
    adjusted_hr_prob: float
    adjusted_era: float
    
    # Confidence interval (wider for weather uncertainty)
    ci_lower: float
    ci_upper: float
    
class ParkWeatherAlert(BaseModel):
    """Extreme environment alerts for daily lineup decisions"""
    alert_type: Literal["COORS_FIELD", "EXTREME_HEAT", "WIND_OUT", 
                        "MARINE_LAYER", "PITCHER_FRIENDLY"]
    severity: Literal["MILD", "MODERATE", "EXTREME"]
    
    affected_players: List[str]
    category_impacts: Dict[Category, float]  # e.g., {"HR": 1.35}
    
    recommendation: str
    # e.g., "Stack COL hitters; avoid starting pitchers"
```

### 3.2 Data Sources & Integration

| Data Type | Source | Frequency | Reliability |
|-----------|--------|-----------|-------------|
| **Park Factors** | FanGraphs, Baseball-Reference | Seasonal (updated monthly) | High |
| **Weather Forecasts** | NOAA, Weather API (OpenWeatherMap) | Every 6 hours | Medium-High |
| **Live Weather** | Stadium sensors (if available) | Real-time | High |
| **Air Density Calc** | Derived (physics model) | Per forecast | High |
| **Batted Ball Data** | Statcast (via baseballsavant) | Daily | High |

**Integration Strategy:**
```python
# Cron job: Every 6 hours
async def update_weather_forecasts():
    games = get_upcoming_games(next_48_hours=True)
    for game in games:
        forecast = await fetch_weather_api(game.stadium)
        air_density = calculate_air_density(
            temp=forecast.temp,
            pressure=forecast.pressure,
            humidity=forecast.humidity,
            elevation=game.stadium.elevation
        )
        distance_adj = calculate_distance_adjustment(air_density)
        
        await save_weather_forecast(game, forecast, air_density, distance_adj)
```

### 3.3 Physics Calculation Engine

**Air Density Calculation (Ideal Gas Law):**
```python
def calculate_air_density(temp_f: float, pressure_inhg: float, 
                          humidity_pct: float, elevation_ft: float) -> float:
    """
    Returns air density in kg/m³
    Standard sea level: ~1.225 kg/m³
    Coors Field: ~1.000 kg/m³ (18% less)
    """
    # Convert units
    temp_k = (temp_f - 32) * 5/9 + 273.15
    pressure_pa = pressure_inhg * 3386.39
    
    # Saturation vapor pressure (Tetens equation)
    e_s = 0.6108 * math.exp((17.27 * (temp_k - 273.15)) / (temp_k - 35.85))
    
    # Actual vapor pressure
    e = (humidity_pct / 100) * e_s
    
    # Partial pressure of dry air
    p_d = pressure_pa - e
    
    # Air density
    R_d = 287.05  # J/(kg·K)
    R_v = 461.5   # J/(kg·K)
    
    density = (p_d / (R_d * temp_k)) + (e / (R_v * temp_k))
    
    return density
```

**Distance Adjustment (Nathan Model):**
```python
def calculate_distance_adjustment(air_density: float, 
                                  baseline_density: float = 1.225) -> float:
    """
    Returns feet added/subtracted from fly ball distance
    Based on Alan Nathan's research: ~6 ft per 1,000 ft elevation
    """
    density_ratio = air_density / baseline_density
    
    # Empirical: 10% density reduction ≈ 5% distance increase
    # For 400 ft fly ball: 5% = 20 ft
    distance_pct_change = (1 - density_ratio) * 0.5
    
    # For typical home run trajectory (400 ft)
    distance_change_ft = 400 * distance_pct_change
    
    return distance_change_ft
```

---

## 4. UI/UX Integration

### 4.1 Feature: Environment-Adjusted Projections

**Location:** Player cards, lineup optimizer, waiver wire

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ Nolan Arenado                           │
│ Today vs LAD @ COL (Coors Field)         │
│                                         │
│ Baseline HR: 0.18 (18% per game)        │
│ 🌤️ 78°F, Light Wind Out                 │
│ Environment: +28% distance              │
│                                         │
│ ADJUSTED HR: 0.23 (+28%) 🔥             │
│ Confidence: HIGH (3 hours to game)      │
└─────────────────────────────────────────┘
```

### 4.2 Feature: Daily Weather Dashboard

**Location:** Main dashboard, daily lineup page

```
┌─────────────────────────────────────────┐
│ TODAY'S ENVIRONMENT ALERTS              │
├─────────────────────────────────────────┤
│                                         │
│ 🔥 EXTREME (3 games)                    │
│ • COL @ ARI (Chase Field) - 102°F       │
│   Stack hitters; avoid pitchers         │
│                                         │
│ 🌬️ WIND OUT (2 games)                   │
│ • CHC vs MIL (Wrigley) - 18 mph out     │
│   HR boost: +15%                        │
│                                         │
│ 🌫️ MARINE LAYER (2 games)               │
│ • SFG vs SD (Oracle) - Cool, damp       │
│   Pitcher-friendly: suppress offense    │
│                                         │
│ [View All 15 Games]                     │
└─────────────────────────────────────────┘
```

### 4.3 Feature: Park Factor Explorer

**Location:** Research tab, player pages

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ PARK FACTOR EXPLORER                    │
├─────────────────────────────────────────┤
│                                         │
│ Coors Field (COL)                       │
│ Elevation: 5,280 ft | Air Density: Low  │
│                                         │
│ Run Factor:  128 ██████████████▌ +28%   │
│ HR Factor:   132 ██████████████▌ +32%   │
│ 2B Factor:   110 ███████████░░░ +10%    │
│ 3B Factor:   145 █████████████████ +45% │
│                                         │
│ Your Players at Coors This Week:        │
│ • Arenado (3 games) - Start all         │
│ • Marquez (1 start) - Strong fade       │
└─────────────────────────────────────────┘
```

### 4.4 Feature: Lineup Optimizer with Environment

**Integration:** Daily lineup optimizer (existing)

**Enhancement:**
```python
# Current optimizer (existing)
optimal_lineup = optimize_lineup(
    roster=roster,
    categories=categories,
    opponent=opponent
)

# Enhanced optimizer (proposed)
optimal_lineup = optimize_lineup(
    roster=roster,
    categories=categories,
    opponent=opponent,
    environment_adjustments=True,  # NEW
    weather_forecasts=forecasts,   # NEW
    risk_tolerance="CONSERVATIVE"  # NEW (for H2H One Win)
)
```

### 4.5 Feature: Weekly Matchup Environment Preview

**For H2H One Win format:** Show upcoming week environment

```
┌─────────────────────────────────────────┐
│ WEEK 15 ENVIRONMENT OUTLOOK             │
├─────────────────────────────────────────┤
│                                         │
│ Your Hitters:                           │
│ • 8 games at Coors/Chase/GABP (hitters) │
│ • 4 games at Oracle/Petco (pitchers)    │
│ Category Impact: HR +12%, R +8%         │
│                                         │
│ Your Pitchers:                          │
│ • 2 starts at Coors (AVOID)             │
│ • 3 starts at pitcher-friendly parks    │
│ Recommendation: Stream 1 SP for Coors   │
│                                         │
│ [View Detailed Breakdown]               │
└─────────────────────────────────────────┘
```

---

## 5. Implementation Phases

### Phase 1: Data Layer (Week 1)
- [ ] Create `ParkFactor` model and table
- [ ] Create `WeatherForecast` model and table
- [ ] Build air density calculation engine
- [ ] Integrate weather API (OpenWeatherMap or NOAA)
- [ ] Populate park factor data (FanGraphs scrape or API)

### Phase 2: Projection Engine (Week 2)
- [ ] Build `EnvironmentAdjustedProjection` calculator
- [ ] Integrate with existing projection system
- [ ] Add confidence intervals for weather uncertainty
- [ ] Unit tests: Validate against Nathan physics models

### Phase 3: API Layer (Week 3)
- [ ] `GET /api/v1/games/{id}/weather`
- [ ] `GET /api/v1/parks/{id}/factors`
- [ ] `GET /api/v1/players/{id}/environment-adjusted`
- [ ] `GET /api/v1/league/{id}/weather-alerts`

### Phase 4: UI Integration (Week 4) - DELEGATED TO KIMI
- [ ] Environment badge on player cards
- [ ] Daily weather dashboard
- [ ] Park factor explorer
- [ ] Weekly matchup environment preview

---

## 6. Validation & Testing

### 6.1 Physics Validation
Compare calculated air density to known values:
| Location | Elevation | Expected Density | Tolerance |
|----------|-----------|------------------|-----------|
| Sea Level | 0 ft | 1.225 kg/m³ | ±0.01 |
| Denver | 5,280 ft | ~1.000 kg/m³ | ±0.01 |
| Phoenix | 1,100 ft | ~1.150 kg/m³ | ±0.01 |

### 6.2 Historical Backtest
Validate projections against 2023-2024 actuals:
- Coors Field hitters: Did HR rate increase 25-30% as projected?
- Extreme heat games (>95°F): Did HR increase 10-15%?
- Marine layer games: Did runs decrease 15-20%?

### 6.3 Edge Case Tests
- [ ] Roof closed (Tropicana, Minute Maid): No weather effect
- [ ] Retractable roof (Chase, Globe Life): Conditional effect
- [ ] Extreme altitude (Coors only): Maximum adjustment
- [ ] Dome teams on road: No park factor adjustment for home games

---

## 7. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Weather API downtime | Medium | Medium | Cache last forecast; show "stale" indicator |
| Park factor sample size (new parks) | Low | Medium | Use 3-year rolling; flag small samples |
| Roof status incorrect | Medium | High | Manual override capability; verify with stadium feeds |
| Physics model drift | Low | High | Annual validation vs. Statcast data |
| User overreaction to weather | High | Low | Education tooltips; confidence intervals |

---

## 8. Cross-References

### Related Documents
- `reports/2026-04-08-fantasy-baseball-ui-ux-research.md` - Main UI/UX spec
- `reports/2026-04-08-yahoo-nsb-audit.md` - K-28 completion
- `HANDOFF.md` - Current operational state

### Academic Sources
- Nathan, A. M. (2019). "Effect of Temperature on Home Run Production." *University of Illinois.*
- Nathan, A. M. (2020). "Fly Ball Carry and the Home Run Surge." *The Hardball Times.*
- Climate Central. (2019). "Baseball Season Heating Up."
- SABR. (2014). "High Altitude Offense." *Baseball Research Journal.*
- Weather Applied Metrics. (2025). "Modeling the Impact of Atmosphere in Baseball."

---

## Decision Required from Claude

### Option A: Full Integration (Recommended)
**Scope:** Park factors + weather forecasts + physics-based adjustments  
**Timeline:** 4 weeks (Phases 1-4)  
**Impact:** Significant competitive advantage for users; complex implementation  
**Data Requirements:** Weather API subscription, park factor data source

### Option B: Park Factors Only (MVP)
**Scope:** Seasonal park factors only (no weather)  
**Timeline:** 1-2 weeks  
**Impact:** 80% of value with 20% of effort; weather is high-variance anyway  
**Data Requirements:** Static park factor table

### Option C: Post-MVP Feature
**Scope:** Defer to Phase 2 of UI roadmap  
**Timeline:** After H2H One Win core features complete  
**Impact:** Lower priority than NSB efficiency, IP Bank, etc.

**Recommendation:** Option B for MVP, Option A for v2. Park factors are stable and high-impact; weather adds complexity with diminishing returns.

---

**Document Status:** Ready for Claude Code review and architecture decision.
