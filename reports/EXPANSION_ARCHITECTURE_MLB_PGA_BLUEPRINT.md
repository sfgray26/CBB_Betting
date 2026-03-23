# Expansion Architecture & Research Blueprint: MLB + PGA
## High-Stakes Betting Syndicate Quant Framework

**Date:** March 2026  
**Platform:** Python/FastAPI/PostgreSQL/Railway  
**Constraint:** The Odds API — 20,000 requests/month  
**Primary Market Maker:** DraftKings  

---

# 1. The Quantitative Paradigms (How to Win)

## 1.1 MLB: The Pitch-Level Edge

### Core Predictive Metrics Hierarchy

```
Tier 1 (Non-Negotiable)
├── SIERA (Skill-Interactive ERA) > ERA/FIP for future prediction
├── Stuff+ (100 = league avg, 110+ = elite, 120+ = dominant)
├── Location+ (pitch quality independent of results)
├── xwOBA / xERA (expected results, luck-neutral)
└── Park Factors (3-year rolling, handedness-split)

Tier 2 (Situational Alpha)
├── Umpire K% / BB% tendencies (sub-1% edge per game)
├── Weather vectors (wind in/out, temp >65°F boost offense)
├── Platoon splits (wOBA vs LHP/RHP, minimum 100 PA)
├── Rest advantage (travel, bullpen usage last 3 days)
└── Catcher framing (runs saved/100 pitches)

Tier 3 (Micro-Edges)
├── Spin efficiency (seam-shifted wake effects)
├── Release point consistency (injury predictor)
├── Batter timing disruption (pitch tempo variation)
└── Baserunning aggression (taking extra bases)
```

### Market-Specific Modeling Approaches

#### **Moneylines: Starting Pitcher Dominance Model**

The starting pitcher explains ~70% of moneyline variance. Architecture:

```python
class MLBPitcherScore:
    """
    Composite pitcher rating for moneyline prediction.
    Weights derived from 2020-2024 backtesting.
    """
    stuff_plus: float          # 35% weight — physical pitch quality
    location_plus: float       # 25% weight — command/precision  
    recent_siera: float        # 20% weight — last 3 starts, 5-start rolling
    pitch_mix_entropy: float   # 10% weight — unpredictability
    days_rest: int             # 5% weight — 5 days optimal, <4 or >7 penalized
    park_adjusted: bool        # 5% weight — home/road context
    
    def win_probability(self, opponent_score: 'MLBPitcherScore', 
                       home_bullpen: float, away_bullpen: float) -> float:
        """
        Elo-style rating system with bullpen decay.
        Late-game leverage (innings 7-9) weighted 1.5x.
        """
        pass
```

**Key Insight:** Markets overreact to ERA and recent wins. Model SIERA differential for alpha.

#### **Totals: Park-Weather-Platoon Intersection Model**

```python
class MLBTotalModel:
    """
    Total runs prediction with environmental overlays.
    """
    # Base components
    starting_pitcher_siera: float      # Per-inning expected runs
    bullpen_siera_last_7: float        # Recent form matters more than season
    lineup_woba_vs_pitcher_type: float # Platoon-adjusted
    
    # Environmental overlays (multiplicative)
    park_factor: float                 # 0.85 (pitcher parks) to 1.25 (Coors)
    temperature_factor: float          # >65°F: 1.02 per 5°F above
    wind_factor: float                 # Out >10mph: 1.08, In >10mph: 0.92
    humidity_factor: float             # High humidity: slight suppression
    
    # Umpire overlay (additive run expectation)
    umpire_k_factor: float             # +/- 0.15 runs based on zone size
    
    def projected_total(self) -> float:
        base = (self.starting_pitcher_siera * 5.5 + 
                self.bullpen_siera_last_7 * 3.5)  # SP: 5.5 IP avg
        offense = self.lineup_woba_vs_pitcher_type * 4.2  # wOBA to runs
        
        env_multiplier = (self.park_factor * self.temperature_factor * 
                         self.wind_factor * self.humidity_factor)
        
        umpire_adjustment = self.umpire_k_factor
        
        return (base + offense) * env_multiplier + umpire_adjustment
```

**Market Inefficiency:** Books adjust totals for wind direction too aggressively. Fade the public on "wind blowing out" narratives — model the vector components.

#### **Player Props: Statcast-Powered Over/Unders**

```python
class MLBPlayerPropModel:
    """
    Pitcher/hitter props with platoon and park context.
    """
    # Pitcher props (K's, outs, earned runs)
    k_per_bf_model: float              # Strikeouts per batter faced
    whip_projection: float
    innings_distribution: List[float]  # Probability of 4IP, 5IP, 6IP...
    
    # Hitter props (hits, HRs, RBIs, total bases)
    xwoba_vs_pitcher_type: float
    barrel_rate_last_30: float
    hard_hit_pct: float
    sprint_speed: float                # For doubles/triples props
    
    # Context
    platoon_advantage: bool
    park_factor_position: str          # "LF", "RF", "CF" for HR props
    lineup_spot: int                   # 1-9, affects PA expectation
    
    def prop_probability(self, market_line: float, stat_type: str) -> float:
        """
        Monte Carlo simulation of 10,000 plate appearances.
        Returns probability of over.
        """
        pass
```

**Prop-Specific Edges:**
- **Strikeouts:** Model whiff% (swinging + called) rather than raw K%. Pitchers with elite Stuff+ but poor results are buy-low K prop targets.
- **Home Runs:** Barrel% × Park Factor for handedness. Target hitters with >12% barrel rate in HR-friendly parks.
- **Hits:** xBA (expected batting average) > actual BA indicates positive regression.

---

## 1.2 PGA: Strokes Gained Decomposition

### Core Metrics Hierarchy

```
Tier 1 (Predictive Foundation)
├── Strokes Gained: Tee-to-Green (SG:T2G)
│   ├── SG: Off-the-Tee (driving distance + accuracy)
│   ├── SG: Approach (150-200 yard performance)
│   └── SG: Around-the-Green (short game)
├── Strokes Gained: Putting (least predictive week-to-week)
└── Baseline vs. Recent Form (6-month vs. last-4-events weighting)

Tier 2 (Course Fit)
├── Distance requirements ("Bomb & Gouge" vs. "Accuracy Premium")
├── Green complexity (Bermuda vs. Bentgrass, grain direction)
├── Rough penalty severity (U.S. Open vs. regular tour stop)
└── Scoring environment (birdie-fest vs. grind-it-out)

Tier 3 (Tactical Edges)
├── Wave advantages (morning vs. afternoon draw bias)
├── Rest/recency (weeks off, WD history)
├── Course history (long-term familiarity)
└── Pressure performance (back-9 Sunday scoring average)
```

### Market-Specific Modeling Approaches

#### **Outrights: Field Simulation with Variance**

```python
class PGAOutrightModel:
    """
    Tournament winner simulation with correlated variance.
    
    Key insight: Golf has high variance. A golfer with 10% "true" win probability
    will still lose 90% of the time. Bankroll management is critical.
    """
    
    # Player skill estimate (baseline + recent form blend)
    baseline_sg_total: float           # 12-month rolling average
    recent_sg_total: float             # Last 4 events, weighted recency
    form_blend: float = 0.65           # 65% recent, 35% baseline (tunable)
    
    # Course fit adjustment
    distance_premium: float            # + for bombers, - for accuracy players
    putting_surface_adjustment: float  # + for Bermuda specialists on Bermuda
    scrambling_requirement: float      # + for elite around-green players
    
    # Variance modeling (critical for proper pricing)
    sg_volatility: float               # Standard deviation of historical rounds
    round_correlation: float = 0.25    # Performance correlates across 4 rounds
    
    def simulate_tournament(self, field: List['PGAOutrightModel'], 
                           n_sims: int = 50000) -> Dict[str, float]:
        """
        Monte Carlo simulation accounting for:
        - Round-to-round correlation (good rounds cluster)
        - Course fit non-linearities
        - Weather draw bias (if applicable)
        
        Returns: Dict[player_name] = win_probability
        """
        pass
```

**Variance is the Product:** Golf betting isn't about picking winners — it's about finding mispriced variance. A golfer at 50/1 with 2% true probability is a +EV bet. A golfer at 10/1 with 8% true probability is -EV.

#### **Top 10/20s: Finish Position Modeling**

```python
class PGAFinishPositionModel:
    """
    Top N finish modeling uses the same simulation as outrights,
    but aggregates outcomes across positions.
    """
    
    def top_n_probability(self, simulations: List[List[Tuple[str, int]]], 
                          n: int) -> Dict[str, float]:
        """
        From tournament simulations, calculate Top N probability.
        
        Key edge: Markets often misprice the cut line. If wind is expected
        to increase Friday afternoon, morning wave has lower cut probability.
        """
        pass
```

**Market Inefficiency:** Books copy each other's Top 20 lines. Target specific course-fit profiles (e.g., accuracy players at narrow courses) when the market prices generically.

#### **Head-to-Head Matchups: Skill Differential + Correlation**

```python
class PGAMatchupModel:
    """
    H2H tournament matchups — most exploitable golf market.
    
    The correlation structure matters: if two golfers play similar styles,
    their performances correlate. This affects variance of the differential.
    """
    
    player_a_skill: float
    player_b_skill: float
    skill_differential: float
    
    # Correlation factors (both positive and negative correlations exist)
    similar_play_style: float          # +correlation (both distance players)
    opposite_wave_draws: bool          # -correlation (different conditions)
    
    # Variance of the differential
    def matchup_variance(self) -> float:
        """
        Var(A - B) = Var(A) + Var(B) - 2*Cov(A,B)
        
        Lower variance = skill differential more likely to determine outcome.
        """
        pass
    
    def win_probability(self) -> float:
        """
        Normal CDF of skill differential / sqrt(matchup_variance).
        """
        pass
```

**Matchup-Specific Edges:**
- **Skill gap overlooked:** Books price based on name recognition, not current SG data
- **Wave draw asymmetry:** If Player A has Thursday PM/Friday AM draw, they face harder conditions
- **Rest advantage:** Player coming off missed cut (rested) vs. Player who played playoff event (fatigue)

---

# 2. Data Sourcing & Pipeline Architecture

## 2.1 Data Source Inventory

### MLB Data Sources

| Source | Cost | Data Type | Refresh Rate | Integration Method |
|--------|------|-----------|--------------|-------------------|
| **Baseball Savant (Statcast)** | Free | Pitch-level (4M+ rows/season) | Daily | `pybaseball` library, CSV chunks |
| **Fangraphs** | Free | SIERA, WAR, projections | Daily | Scraping / CSV export |
| **Baseball-Reference** | Free | Historical box scores | Daily | `baseballreference` PyPI |
| **Odds API** | $0 (capped) | Real-time odds | 5-min polling | REST API |
| **Weather.gov API** | Free | Game-time conditions | Hourly | REST API |
| **Umpire Scorecards** | Free | Umpire tendencies | Weekly | Scraping |

### PGA Data Sources

| Source | Cost | Data Type | Refresh Rate | Integration Method |
|--------|------|-----------|--------------|-------------------|
| **DataGolf API** | ~$100/mo | SG data, rankings, projections | Real-time | REST API |
| **PGA Tour API** | Free (limited) | Shot-level, scoring | Real-time | REST API (registration required) |
| **Official World Golf Rank** | Free | Rankings, form | Weekly | Scraping |
| **Odds API** | $0 (capped) | Real-time odds | 5-min polling | REST API |
| **Weather APIs** | Free/Paid | Course conditions | Hourly | Visual Crossing / OpenWeather |

## 2.2 PostgreSQL Schema Architecture

### Design Philosophy: Partitioning for Scale

MLB generates ~4 million pitch records per season. PGA generates ~300,000 shot records. Without partitioning, queries become unusable within weeks.

### MLB Schema

```sql
-- ============================================
-- MLB CORE TABLES
-- ============================================

-- Games table (partitioned by season)
CREATE TABLE mlb_games (
    id BIGSERIAL,
    game_id VARCHAR(50) PRIMARY KEY,
    season INTEGER NOT NULL,
    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
    home_team VARCHAR(5) NOT NULL,
    away_team VARCHAR(5) NOT NULL,
    venue_id VARCHAR(10),
    temperature INTEGER,
    wind_speed INTEGER,
    wind_direction VARCHAR(10),
    umpire_hp_id VARCHAR(20),
    
    -- Results (populated post-game)
    home_score INTEGER,
    away_score INTEGER,
    innings INTEGER DEFAULT 9,
    completed BOOLEAN DEFAULT FALSE,
    
    -- For partitioning
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (season);

-- Create partitions per season
CREATE TABLE mlb_games_2025 PARTITION OF mlb_games
    FOR VALUES FROM (2025) TO (2026);
CREATE TABLE mlb_games_2026 PARTITION OF mlb_games
    FOR VALUES FROM (2026) TO (2027);

-- Pitches table (partitioned by game_date, massive scale)
CREATE TABLE mlb_pitches (
    id BIGSERIAL,
    pitch_id VARCHAR(100) PRIMARY KEY,
    game_id VARCHAR(50) REFERENCES mlb_games(game_id),
    season INTEGER NOT NULL,
    game_date TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- At-bat context
    inning INTEGER NOT NULL,
    inning_half VARCHAR(10) NOT NULL, -- 'top' or 'bottom'
    batter_id VARCHAR(20) NOT NULL,
    pitcher_id VARCHAR(20) NOT NULL,
    
    -- Pitch characteristics (Statcast)
    pitch_type VARCHAR(5), -- 'FF', 'SL', 'CU', etc.
    release_speed FLOAT,
    release_pos_x FLOAT,
    release_pos_z FLOAT,
    pfx_x FLOAT, -- horizontal movement
    pfx_z FLOAT, -- vertical movement
    plate_x FLOAT, -- horizontal location
    plate_z FLOAT, -- vertical location
    zone INTEGER, -- 1-9 strike zone grid
    
    -- Outcome
    description TEXT,
    type VARCHAR(10), -- 'S' strike, 'B' ball, 'X' in play
    bb_type VARCHAR(20), -- 'ground_ball', 'fly_ball', etc.
    
    -- Batted ball (if applicable)
    launch_speed FLOAT,
    launch_angle FLOAT,
    hit_distance_sc INTEGER,
    hc_x FLOAT, -- hit coordinate x
    hc_y FLOAT, -- hit coordinate y
    
    -- Run expectancy
    runs_scored_on_pitch INTEGER DEFAULT 0,
    run_expectancy_before FLOAT,
    run_expectancy_after FLOAT,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (game_date);

-- Monthly partitions for pitches (manageable chunks)
CREATE TABLE mlb_pitches_2025_04 PARTITION OF mlb_pitches
    FOR VALUES FROM ('2025-04-01') TO ('2025-05-01');
CREATE TABLE mlb_pitches_2025_05 PARTITION OF mlb_pitches
    FOR VALUES FROM ('2025-05-01') TO ('2025-06-01');
-- ... continue for all months

-- Player stats materialized view (refreshed daily)
-- This aggregates pitch-level data into usable player metrics
CREATE MATERIALIZED VIEW mlb_player_stats AS
WITH pitcher_stats AS (
    SELECT 
        pitcher_id,
        season,
        COUNT(*) AS total_pitches,
        AVG(release_speed) AS avg_velo,
        COUNT(DISTINCT game_id) AS games,
        -- Stuff+ components would go here
        AVG(CASE WHEN type = 'S' THEN 1 ELSE 0 END) AS strike_pct
    FROM mlb_pitches
    WHERE game_date >= NOW() - INTERVAL '12 months'
    GROUP BY pitcher_id, season
),
batter_stats AS (
    SELECT
        batter_id,
        season,
        COUNT(*) AS total_pitches_seen,
        AVG(launch_speed) AS avg_exit_velo,
        AVG(CASE WHEN launch_angle BETWEEN 8 AND 32 THEN 1 ELSE 0 END) AS sweet_spot_pct,
        -- xwOBA calculation would go here
        AVG(CASE WHEN bb_type = 'home_run' THEN 1 ELSE 0 END) AS hr_rate
    FROM mlb_pitches
    WHERE launch_speed IS NOT NULL
    GROUP BY batter_id, season
)
SELECT * FROM pitcher_stats
UNION ALL
SELECT * FROM batter_stats;

-- Create indexes for query performance
CREATE INDEX idx_mlb_pitches_pitcher_date ON mlb_pitches(pitcher_id, game_date DESC);
CREATE INDEX idx_mlb_pitches_batter_date ON mlb_pitches(batter_id, game_date DESC);
CREATE INDEX idx_mlb_pitches_game ON mlb_pitches(game_id);
```

### PGA Schema

```sql
-- ============================================
-- PGA CORE TABLES
-- ============================================

-- Tournaments
CREATE TABLE pga_tournaments (
    id SERIAL PRIMARY KEY,
    tournament_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    course_name VARCHAR(200),
    course_id VARCHAR(50),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    season INTEGER NOT NULL,
    
    -- Course characteristics
    par INTEGER,
    yardage INTEGER,
    greens_type VARCHAR(50), -- 'Bermuda', 'Bentgrass', etc.
    rough_difficulty FLOAT, -- 1.0 = normal, 1.2 = U.S. Open
    
    -- Weather summary
    avg_wind_speed FLOAT,
    wave_bias_factor FLOAT, -- + for AM bias, - for PM bias
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Players
CREATE TABLE pga_players (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) UNIQUE NOT NULL,
    dg_id INTEGER, -- DataGolf ID for cross-reference
    name VARCHAR(100) NOT NULL,
    country VARCHAR(5),
    
    -- Skill baselines (updated weekly)
    baseline_sg_total FLOAT,
    baseline_sg_ott FLOAT, -- Off-the-tee
    baseline_sg_app FLOAT, -- Approach
    baseline_sg_atg FLOAT, -- Around-the-green
    baseline_sg_putt FLOAT, -- Putting
    
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Shots (partitioned by tournament, ~300k per tournament)
CREATE TABLE pga_shots (
    id BIGSERIAL,
    shot_id VARCHAR(100) PRIMARY KEY,
    tournament_id VARCHAR(50) REFERENCES pga_tournaments(tournament_id),
    player_id VARCHAR(50) REFERENCES pga_players(player_id),
    
    -- Round context
    round_num INTEGER NOT NULL,
    hole_num INTEGER NOT NULL,
    par INTEGER,
    yardage INTEGER,
    
    -- Shot characteristics
    shot_type VARCHAR(20), -- 'tee', 'fairway', 'rough', 'sand', 'green'
    start_distance INTEGER, -- yards from hole
    end_distance INTEGER,
    
    -- Strokes Gained
    sg_stroke FLOAT, -- Actual strokes - Expected strokes
    
    -- Proximity (for approach shots)
    proximity_feet FLOAT,
    
    -- Score
    score_relative_to_par INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) PARTITION BY RANGE (tournament_id);

-- Round scores (for simulation backtesting)
CREATE TABLE pga_round_scores (
    id SERIAL PRIMARY KEY,
    tournament_id VARCHAR(50) REFERENCES pga_tournaments(tournament_id),
    player_id VARCHAR(50) REFERENCES pga_players(player_id),
    round_num INTEGER NOT NULL,
    score INTEGER NOT NULL,
    sg_total FLOAT,
    sg_ott FLOAT,
    sg_app FLOAT,
    sg_atg FLOAT,
    sg_putt FLOAT,
    
    UNIQUE(tournament_id, player_id, round_num)
);

-- Strokes Gained time-series materialized view
CREATE MATERIALIZED VIEW pga_sg_trends AS
WITH recent_rounds AS (
    SELECT 
        player_id,
        tournament_id,
        round_num,
        sg_total,
        sg_ott,
        sg_app,
        sg_atg,
        sg_putt,
        ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY 
            (SELECT start_date FROM pga_tournaments t WHERE t.tournament_id = rs.tournament_id),
            round_num DESC
        ) AS recency_rank
    FROM pga_round_scores rs
)
SELECT 
    player_id,
    -- Last 4 events (8-16 rounds)
    AVG(CASE WHEN recency_rank <= 12 THEN sg_total END) AS sg_total_l4,
    AVG(CASE WHEN recency_rank <= 12 THEN sg_ott END) AS sg_ott_l4,
    AVG(CASE WHEN recency_rank <= 12 THEN sg_app END) AS sg_app_l4,
    AVG(CASE WHEN recency_rank <= 12 THEN sg_atg END) AS sg_atg_l4,
    AVG(CASE WHEN recency_rank <= 12 THEN sg_putt END) AS sg_putt_l4,
    
    -- Last 12 months
    AVG(sg_total) AS sg_total_12m,
    COUNT(*) AS rounds_count
FROM recent_rounds
WHERE recency_rank <= 50  -- Cap at ~1 year of data
GROUP BY player_id;
```

### Odds & Betting Schema (Unified Across Sports)

```sql
-- ============================================
-- UNIFIED BETTING TABLES (Extends existing CBB schema)
-- ============================================

-- Extend games table concept for all sports
ALTER TYPE sport_type ADD VALUE 'mlb';
ALTER TYPE sport_type ADD VALUE 'pga';

-- Odds history (time-series for CLV tracking)
CREATE TABLE odds_history (
    id BIGSERIAL PRIMARY KEY,
    
    -- Universal identifiers
    sport VARCHAR(10) NOT NULL, -- 'cbb', 'mlb', 'pga'
    event_id VARCHAR(100) NOT NULL,
    
    -- For MLB: game_id
    -- For PGA: tournament_id_player_id (composite)
    
    market_type VARCHAR(20) NOT NULL, -- 'moneyline', 'spread', 'total', 'outright', 'matchup'
    
    -- Market line
    line_value FLOAT, -- spread value, total, or odds for ML
    odds_home FLOAT, -- American odds
    odds_away FLOAT,
    
    -- Bookmaker
    bookmaker VARCHAR(50) NOT NULL,
    is_sharp BOOLEAN DEFAULT FALSE, -- Pinnacle, Circa, etc.
    
    -- Timestamp (critical for CLV)
    captured_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Partition by sport for query efficiency
CREATE INDEX idx_odds_history_sport_time ON odds_history(sport, captured_at DESC);
CREATE INDEX idx_odds_history_event ON odds_history(event_id, market_type);

-- Predictions table extension
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS sport VARCHAR(10) DEFAULT 'cbb';

-- MLB-specific prediction fields
CREATE TABLE mlb_predictions (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    
    -- Starting pitchers
    home_sp_id VARCHAR(20),
    away_sp_id VARCHAR(20),
    home_sp_siera FLOAT,
    away_sp_siera FLOAT,
    home_sp_stuff_plus FLOAT,
    away_sp_stuff_plus FLOAT,
    
    -- Bullpen context
    home_bullpen_era_7d FLOAT,
    away_bullpen_era_7d FLOAT,
    
    -- Environmental
    park_factor FLOAT,
    wind_in_out FLOAT, -- positive = wind out
    temperature FLOAT,
    
    -- Projections
    projected_home_score FLOAT,
    projected_away_score FLOAT,
    projected_total FLOAT,
    
    -- Player props (JSON for flexibility)
    player_prop_projections JSONB DEFAULT '{}'
);

-- PGA-specific prediction fields
CREATE TABLE pga_predictions (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id) ON DELETE CASCADE,
    tournament_id VARCHAR(50) REFERENCES pga_tournaments(tournament_id),
    
    -- Simulation parameters
    n_simulations INTEGER DEFAULT 50000,
    variance_model VARCHAR(20), -- 'normal', 't-distribution', 'empirical'
    
    -- Course fit adjustments applied
    distance_premium_weight FLOAT DEFAULT 0.0,
    putting_surface_adjustment FLOAT DEFAULT 0.0,
    
    -- Results by player (stored as JSONB for 150+ player fields)
    outright_probabilities JSONB NOT NULL DEFAULT '{}',
    top_10_probabilities JSONB NOT NULL DEFAULT '{}',
    top_20_probabilities JSONB NOT NULL DEFAULT '{}',
    matchup_probabilities JSONB NOT NULL DEFAULT '{}'
);
```

## 2.3 Ingestion Pipeline Architecture

### MLB Daily Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MLB DAILY INGESTION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

04:00 UTC (11 PM ET prev day)     12:00 UTC (7 AM ET)
    │                                   │
    ▼                                   ▼
┌─────────────────────┐          ┌─────────────────────┐
│ 1. PYBASEBALL FETCH │          │ 2. LINEUP CONFIRM   │
│    - Previous day   │          │    - Rotowire scraper│
│      pitch data     │          │    - Starting SPs    │
│    - 7-day chunks   │          │    - Weather update  │
│    - Upsert to DB   │          │    - Umpire assign   │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
           ▼                                ▼
┌─────────────────────┐          ┌─────────────────────┐
│ 3. MATERIALIZED VIEW│          │ 4. FANGraphs UPDATE │
│    REFRESH          │          │    - SIERA, WAR     │
│    - player_stats   │          │    - Park factors   │
│    - 10-min window  │          │    - 30-day trends  │
└──────────┬──────────┘          └──────────┬──────────┘
           │                                │
           └────────────┬───────────────────┘
                        ▼
           ┌─────────────────────┐
           │ 5. PREDICTION MODEL │
           │    - Daily batch    │
           │    - SP matchups    │
           │    - Prop calcs     │
           └──────────┬──────────┘
                      ▼
           ┌─────────────────────┐
           │ 6. ODDS API POLL    │
           │    - Every 5 min    │
           │    - CLV tracking   │
           └─────────────────────┘
```

### PGA Weekly Ingestion Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PGA WEEKLY INGESTION PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

Monday (Tournament Week)
    │
    ▼
┌─────────────────────┐
│ 1. FIELD EXTRACTION │
│    - PGA Tour API   │
│    - DataGolf field │
│    - Player matching│
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. COURSE PROFILING │
│    - Yardage, par   │
│    - Greens type    │
│    - Historical     │
│      scoring avg    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. PLAYER FORM SYNC │
│    - Last 4 events  │
│    - SG breakdowns  │
│    - Rest/schedule  │
└──────────┬──────────┘
           ▼
Wednesday Night
           │
           ▼
┌─────────────────────┐
│ 4. DRAW ANALYSIS    │
│    - Wave splits    │
│    - Weather bias   │
│    - Tee time adj   │
└──────────┬──────────┘
           ▼
Thursday-Sunday
           │
           ▼
┌─────────────────────┐
│ 5. LIVE SCORING     │
│    - Shotlink feed  │
│    - Real-time SG   │
│    - Model update   │
└─────────────────────┘
```

---

# 3. Quota-Optimized API Strategy

## 3.1 The 20,000 Request Budget Allocation

```
Monthly Budget: 20,000 requests
================================

MLB (March-October, ~180 days)
├── Daily pre-game odds: 180 days × 1 request = 180
├── Intraday line checks: 180 days × 2 requests = 360
├── Derivative markets: 180 days × 0.5 requests = 90
└── MLB subtotal: ~630 requests/month during season

PGA (52 weeks/year)
├── Pre-tournament outrights: 52 × 1 = 52
├── Live round updates: 52 tournaments × 4 days × 2 = 416
├── H2H matchup odds: 52 × 3 = 156
└── PGA subtotal: ~624 requests/month

CBB (Nov-March, existing usage)
└── Preserve existing: ~800 requests/month

RESERVE BUDGET: 20,000 - (630 + 624 + 800) = 17,946
→ Allocate 10,000 for CLV capture, line movement, +EV alerts
→ Keep 7,946 as emergency reserve
```

## 3.2 CLV Capture Strategy by Sport

### MLB CLV Timing

```python
class MLBCLVStrategy:
    """
    CLV is highest in MLB during:
    1. Lineup announcements (1-2 hours before game)
    2. Weather updates (if wind changes)
    3. Bullpen availability news
    """
    
    POLL_SCHEDULE = {
        # Morning scan (identify overnight moves)
        "06:00_ET": "light_scan",      # 1 request, track major line moves
        
        # Lineup confirmation window (HIGHEST VALUE)
        "17:00_ET": "deep_scan",       # All games tonight
        "18:00_ET": "deep_scan",       # 1 hour before first pitch
        "19:00_ET": "closing_scan",    # Final lines, most predictive
        
        # Late games separate scan
        "21:00_ET": "west_coast_scan", # 10 PM ET games
    }
    
    def smart_poll(self, games_today: List[MLBGame]) -> List[OddsSnapshot]:
        """
        Adaptive polling: Focus quota on games with:
        - High model confidence (>3% edge)
        - Pitcher announcement pending
        - Weather uncertainty
        """
        priority_games = self.prioritize(games_today)
        
        # Request budget per scan: ~15 games × 1 request = 15 requests
        # 4 scans/day × 15 = 60 requests/day during MLB season
        # × 180 days = 10,800 requests (fits in reserve)
        pass
```

### PGA CLV Timing

```python
class PGACLvStrategy:
    """
    PGA CLV patterns are different:
    - Pre-tournament: Lines settle 24-48 hours before tee off
    - Thursday AM: Wave announcement creates movement
    - Live betting: Rapid SG updates create edges
    """
    
    TOURNAMENT_SCHEDULE = {
        # Monday-Tuesday: Field announcement, light monitoring
        "early_week": "skip",  # Use DataGolf for pricing
        
        # Wednesday: Draw announcement (CRITICAL)
        "wednesday_18:00_ET": "wave_scan",  # Draw timing known
        "wednesday_22:00_ET": "closing_outrights",  # Pre-tournament close
        
        # Thursday-Sunday: Live rounds
        "round_start": "live_sync",  # Every 2 hours during rounds
        "round_end": "settle_scan",  # Final results for CLV calc
    }
    
    def tournament_poll_budget(self, tournament: PGATournament) -> int:
        """
        Budget ~20 requests per tournament:
        - Pre-tournament: 3 requests
        - Per round: 4 requests × 4 rounds = 16
        - Total: 19 requests/tournament
        × 52 tournaments = 988 requests/year
        """
        return 20
```

## 3.3 Circuit Breaker & Rate Limiting

Extend your existing `OddsAPIClient` with sport-aware quota management:

```python
class QuotaManagedOddsClient(OddsAPIClient):
    """
    Sport-aware quota allocation with emergency reserves.
    """
    
    MONTHLY_QUOTA = 20000
    EMERGENCY_RESERVE = 5000
    
    SPORT_BUDGETS = {
        "cbb": 800,      # Existing usage
        "mlb": 4000,     # ~22 requests/day during season
        "pga": 2000,     # ~38 requests/tournament
    }
    
    def __init__(self):
        super().__init__()
        self.usage_by_sport = {"cbb": 0, "mlb": 0, "pga": 0}
        self.month_start = datetime.now().replace(day=1)
    
    def can_request(self, sport: str, priority: str = "normal") -> bool:
        """
        Check if request should be allowed based on budget.
        
        Priority levels:
        - "background": Only if well under budget
        - "normal": Standard budget check
        - "clv_critical": Use emergency reserve for closing line value
        """
        used = sum(self.usage_by_sport.values())
        remaining = self.MONTHLY_QUOTA - used
        
        if priority == "clv_critical" and remaining > 100:
            return True
            
        if self.usage_by_sport[sport] >= self.SPORT_BUDGETS[sport]:
            return False
            
        return remaining > self.EMERGENCY_RESERVE
    
    def get_sport_odds(self, sport: str, markets: str, priority: str = "normal"):
        """
        Fetch odds with quota tracking.
        """
        if not self.can_request(sport, priority):
            logger.warning(f"Quota exhausted for {sport}, skipping request")
            return None
            
        # Map sport to API key
        sport_keys = {
            "mlb": "baseball_mlb",
            "pga": "golf_pga",  # Note: Verify Odds API coverage
            "cbb": "basketball_ncaab",
        }
        
        url = f"{BASE_URL}/sports/{sport_keys[sport]}/odds"
        # ... existing request logic
        
        self.usage_by_sport[sport] += 1
        return response
```

---

# 4. Phased Expansion Roadmap

## Phase 1: Foundation (Weeks 1-4)

### Week 1-2: Database Schema & Migration
```python
# Priority: Create partitioned tables first
"""
1. Run schema migrations for MLB tables
2. Create partition management scripts
3. Set up monthly partition auto-creation
4. Test ingestion with small dataset (1 week of 2025 data)
"""

# Migration script structure
migrations/"
├── 001_add_mlb_tables.sql
├── 002_add_pga_tables.sql
├── 003_add_unified_odds_schema.sql
└── 004_add_sport_type_enum.sql
```

### Week 3-4: MLB Data Pipeline MVP
```python
# Core services to build
backend/services/
├── mlb/
│   ├── __init__.py
│   ├── data_ingestion.py      # pybaseball integration
│   ├── statcast_sync.py       # Pitch-level data
│   ├── pitcher_ratings.py     # SIERA, Stuff+ calculations
│   ├── park_factors.py        # Weather, venue adjustments
│   └── lineup_monitor.py      # Starting pitcher confirmation
```

**Deliverable:** Automated daily ingestion of previous day's MLB data

## Phase 2: MLB Modeling (Weeks 5-8)

### Week 5-6: Core Models
```python
backend/services/mlb/
├── models/
│   ├── __init__.py
│   ├── moneyline_model.py     # SP-centric predictions
│   ├── totals_model.py        # Park-weather intersection
│   ├── props_model.py         # K's, HRs, hits
│   └── ensemble.py            # Model combination
```

**Key Algorithm:**
- SIERA calculation from box scores (existing library)
- Stuff+ estimation from Statcast pitch characteristics
- Park factor adjustments (handedness-specific)

### Week 7-8: Odds Integration & Testing
```python
backend/services/mlb/
├── odds_integration.py        # MLB-specific odds parsing
├── clv_tracker.py             # Closing line value for MLB
├── bet_recommender.py         # +EV bet identification
└── paper_trade_logger.py      # Simulation tracking
```

**Deliverable:** MLB paper trading system with Discord alerts

## Phase 3: PGA Foundation (Weeks 9-12)

### Week 9-10: DataGolf Integration
```python
backend/services/pga/
├── __init__.py
├── datagolf_client.py         # API wrapper
├── field_sync.py              # Tournament field extraction
├── sg_calculator.py           # Strokes Gained from shot data
└── course_profiler.py         # Course fit analysis
```

### Week 11-12: PGA Modeling
```python
backend/services/pga/
├── models/
│   ├── __init__.py
│   ├── outright_simulator.py  # Monte Carlo tournament sim
│   ├── matchup_model.py       # H2H probability
│   ├── top_n_model.py         # Top 10/20 probabilities
│   └── form_calculator.py     # Baseline vs. recent blend
```

**Deliverable:** PGA outright and matchup predictions

## Phase 4: Unification & Scale (Weeks 13-16)

### Week 13-14: Unified Interface
```python
# Extend existing analysis.py patterns
backend/services/
├── unified_analysis.py        # Sport-agnostic analysis orchestrator
├── sport_router.py            # Route to MLB/PGA/CBB models
└── quota_manager.py           # 20k request budget allocation

dashboard/pages/
├── 13_MLB_Analysis.py         # MLB-specific dashboard
├── 14_PGA_Analysis.py         # PGA-specific dashboard
└── 15_Unified_Bet_Tracker.py  # Cross-sport P&L
```

### Week 15-16: Decoupled Workers

```yaml
# railway.yaml — Decoupled worker configuration
services:
  - name: web
    # Existing FastAPI dashboard
    
  - name: mlb-ingestion-worker
    command: python -m backend.workers.mlb_daily_sync
    schedule: "0 4 * * *"  # 4 AM UTC daily
    
  - name: pga-ingestion-worker  
    command: python -m backend.workers.pga_tournament_sync
    # Triggered manually or by tournament schedule
    
  - name: odds-polling-worker
    command: python -m backend.workers.odds_poll_scheduler
    # Smart polling based on game times
    
  - name: live-mlb-worker
    command: python -m backend.workers.mlb_live_tracker
    # Active only during MLB games
    
  - name: live-pga-worker
    command: python -m backend.workers.pga_live_scoring
    # Active Thursday-Sunday during tournaments
```

## Phase 5: Live Operations (Week 17+)

### Decoupling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DECOUPLED WORKER ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘

Main FastAPI Instance (Railway)
├── Serves dashboard, API endpoints
├── Reads from PostgreSQL
├── No heavy computation
└── Always running

Worker Instances (Railway Cron/Background)
├── MLB Daily Worker
│   ├── Runs at 4 AM UTC
│   ├── Fetches previous day Statcast
│   ├── Updates materialized views
│   └── Exits when complete
│
├── MLB Live Worker
│   ├── Active 11 AM - 2 AM ET (game hours)
│   ├── Polls every 5 minutes during games
│   ├── Sends Discord alerts for +EV
│   └── Sleeps between games
│
├── PGA Tournament Worker
│   ├── Triggered: Monday (field) → Sunday (final)
│   ├── Wednesday: Heavy load (draw analysis)
│   ├── Thursday-Sunday: Live scoring sync
│   └── Exits Monday post-tournament
│
├── Odds Quota Manager
│   ├── Centralized quota tracking
│   ├── Allocates requests by sport/priority
│   └── Prevents 429 exhaustion
│
└── Discord Notification Worker
    ├── Batches alerts across sports
    ├── Rate-limits to avoid spam
    └── Maintains channel separation
```

### Critical Decoupling Pattern

```python
# backend/workers/mlb_daily_sync.py
"""
MLB daily ingestion worker — runs as independent Railway process.
Does NOT block other workers.
"""

import sys
from datetime import datetime, timedelta

def main():
    """
    Fetch yesterday's MLB data, update stats, exit.
    Railway cron triggers this once daily.
    """
    yesterday = datetime.now() - timedelta(days=1)
    
    try:
        # 1. Fetch Statcast data (pybaseball)
        pitches = fetch_statcast_day(yesterday)
        
        # 2. Bulk insert to partitioned table
        bulk_insert_pitches(pitches)
        
        # 3. Refresh materialized views
        refresh_mlb_player_stats()
        
        # 4. Update Fangraphs ratings
        sync_fangraphs_data()
        
        # 5. Generate predictions for today's games
        generate_mlb_predictions()
        
        logger.info(f"MLB daily sync complete for {yesterday.date()}")
        return 0
        
    except Exception as e:
        logger.error(f"MLB sync failed: {e}")
        send_alert_to_discord(f"🚨 MLB sync failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

```python
# backend/workers/pga_tournament_sync.py
"""
PGA tournament worker — triggered by tournament schedule.
Runs independently of MLB workers.
"""

def main():
    tournament = get_current_or_upcoming_tournament()
    
    if tournament is None:
        logger.info("No active tournament, exiting")
        return 0
    
    day_of_week = datetime.now().weekday()
    
    try:
        if day_of_week == 0:  # Monday
            sync_field(tournament)
        elif day_of_week == 2:  # Wednesday
            sync_draw_and_weather(tournament)
            generate_outright_predictions(tournament)
        elif day_of_week in [3, 4, 5, 6]:  # Thu-Sun
            sync_live_scoring(tournament)
            update_predictions(tournament)
        
        return 0
        
    except Exception as e:
        logger.error(f"PGA sync failed: {e}")
        return 1
```

---

# Appendix A: Key Metrics Formulas

## SIERA (Skill-Interactive ERA)
```python
def calculate_siera(so, bb, gb, fb, pu, ip):
    """
    Fangraphs SIERA formula (simplified).
    More predictive than ERA, FIP, or xFIP.
    """
    so_pa = so / (ip * 2.9)  # SO per PA approximation
    bb_pa = bb / (ip * 2.9)
    gb_fb_ratio = gb / fb if fb > 0 else 1.5
    
    siera = (6.145 - 18.015*so_pa + 11.428*bb_pa - 
             1.697*gb_fb_ratio + 6.603*(so_pa**2) - 
             1.547*(bb_pa**2))
    return max(siera, 2.0)  # Floor at 2.00
```

## Stuff+ Estimation
```python
def calculate_stuff_plus(velo, spin, break_x, break_z, pitch_type):
    """
    Estimate Stuff+ from Statcast pitch characteristics.
    100 = league average, 110 = 10% better, etc.
    """
    # Type-specific baselines
    baselines = {
        'FF': {'velo': 93.5, 'spin': 2200, 'break_x': -5, 'break_z': 9},
        'SL': {'velo': 84.0, 'spin': 2100, 'break_x': 3, 'break_z': 2},
        # ... etc
    }
    
    base = baselines.get(pitch_type, baselines['FF'])
    
    # Z-scores for each component
    velo_z = (velo - base['velo']) / 2.5
    spin_z = (spin - base['spin']) / 200
    break_z = ((break_x - base['break_x']) ** 2 + 
               (break_z - base['break_z']) ** 2) ** 0.5 / 3
    
    # Weighted combination
    stuff_plus = 100 + (velo_z * 8 + spin_z * 4 + break_z * 6)
    return stuff_plus
```

## Strokes Gained
```python
def calculate_strokes_gained(start_distance, end_distance, strokes_to_hole_out):
    """
    SG = Expected strokes from start - Actual strokes - Expected strokes from end
    """
    expected_start = strokes_gained_baseline[start_distance]
    expected_end = strokes_gained_baseline[end_distance] if end_distance > 0 else 0
    
    actual_strokes = 1  # The shot just taken
    
    sg = expected_start - actual_strokes - expected_end
    return sg
```

---

# Appendix B: Environment Configuration

```bash
# .env additions for MLB + PGA expansion

# ============================================
# MLB CONFIGURATION
# ============================================
MLB_DATA_ENABLED=true
MLB_SEASON=2026
STATCAST_CHUNK_DAYS=7
MLB_LINEUP_SOURCE=rotowire  # or 'mlb_api'

# Materialized view refresh schedule
MLB_STATS_REFRESH_HOUR=4
MLB_STATS_REFRESH_MINUTE=30

# ============================================
# PGA CONFIGURATION
# ============================================
PGA_DATA_ENABLED=true
PGA_DATAGOLF_API_KEY=your_key_here
PGA_TOUR_API_KEY=your_key_here

# Tournament sync schedule (UTC)
PGA_FIELD_SYNC_DAY=monday
PGA_FIELD_SYNC_HOUR=14
PGA_DRAW_SYNC_DAY=wednesday
PGA_DRAW_SYNC_HOUR=20

# ============================================
# UNIFIED QUOTA MANAGEMENT
# ============================================
ODDS_API_QUOTA_Monthly=20000
ODDS_API_RESERVE_MLB=4000
ODDS_API_RESERVE_PGA=2000
ODDS_API_RESERVE_CBB=800
ODDS_API_EMERGENCY_RESERVE=5000

# Smart polling intervals (minutes)
MLB_ODDS_POLL_INTERVAL=5
PGA_ODDS_POLL_INTERVAL=30  # Less frequent for golf
```

---

**Document Version:** 1.0  
**Last Updated:** March 2026  
**Next Review:** Post-MLB Opening Day

**Key Success Metrics:**
- MLB: >5% ROI on paper trades by May 31
- PGA: >8% ROI on matchups by first major
- System: Zero quota exhaustion events
- Infrastructure: <2s query time on partitioned tables
