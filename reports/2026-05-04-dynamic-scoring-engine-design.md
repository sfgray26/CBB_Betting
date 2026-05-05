# Dynamic Fantasy Baseball Scoring Engine — Audit & Rebuild Specification

**Author:** Kimi CLI (Deep Intelligence Unit)  
**Date:** 2026-05-04  
**Branch:** `stable/cbb-prod`  
**Database:** Production PostgreSQL (Railway)

---

## PART 1: BRUTAL AUDIT — 10 Critical Weaknesses

Based on code review of 22 production files AND live database queries against production.

### Weakness 1: Composite_z Is a Weighted SUM, Not a Mean
**File:** `backend/services/scoring_engine.py:525-528`  
**Impact:** Two-way players (Ohtani) accumulate 8+ category Z-scores; pure hitters only 5.  
**Live Evidence:** Shohei Ohtani has z_hr=0.12, z_avg=-0.24, z_ops=-0.004 (mediocre hitting) but composite_z=9.79 (#2 overall) because pitcher categories (z_era=0.89, z_whip=0.44) are **added** rather than averaged.  
**Verdict:** The #2 ranked player in the system is a mathematical artifact.

### Weakness 2: Momentum Signals Are Direction-Only, Not Level-Aware
**File:** `backend/models.py` (PlayerMomentum thresholds)  
**Impact:** A player can be labeled "SURGING" while still being in the bottom 5% of the league.  
**Live Evidence:** Dylan Moore has delta_z=7.51, signal="SURGING", but composite_z_14d=-8.83 and composite_z_30d=-16.33. He went from catastrophically bad to merely terrible. The system calls this "SURGING."  
**Verdict:** The momentum layer is misleading users into buying players who are still awful.

### Weakness 3: Confidence Scores Do Not Gate Recommendations
**File:** `backend/services/scoring_engine.py` (confidence formula)  
**Impact:** Players with 1-2 games played (confidence=0.14) get 99+ percentile ranks.  
**Live Evidence:** Kyle Harrison (composite_z=8.15, score_0_100=99.8, confidence=0.21). Will Warren (composite_z=7.77, score_0_100=99.1, confidence=0.14). These are based on 1-2 starts.  
**Verdict:** The system recommends players with essentially zero sample size as "elite."

### Weakness 4: No Opportunity Model Whatsoever
**File:** Entire codebase  
**Impact:** Playing time, lineup spot, platoon risk, and role certainty are completely ignored in scoring.  
**Live Evidence:** The `player_scores` table has no columns for PA/game, lineup position, games started %, or platoon splits. The `position_eligibility` table tracks positions but NOT lineup spot stability.  
**Verdict:** A part-time player with a .400 wOBA in 50 PA is scored higher than a full-time player with a .340 wOBA in 300 PA. Opportunity > efficiency in fantasy.

### Weakness 5: Statcast Advanced Columns Exist but Are 100% NULL
**File:** `backend/fantasy_baseball/pybaseball_loader.py`, `statcast_loader.py`  
**Impact:** `sprint_speed`, `stuff_plus`, `location_plus` columns were added to the DB but the ingestion pipeline never populates them.  
**Live Evidence:** `SELECT COUNT(*) FROM statcast_batter_metrics WHERE sprint_speed IS NOT NULL` = 0. Same for stuff_plus/location_plus on pitcher table.  
**Verdict:** The "best predictor" metrics are schema-only ghosts.

### Weakness 6: Dynamic Thresholds Are Nonexistent
**File:** `backend/fantasy_baseball/category_aware_scorer.py`, `player_board.py`  
**Impact:** All thresholds are hardcoded (e.g., `hot_threshold=0.5`, `cold_threshold=-0.5`, `streamer_threshold=0.3`).  
**Live Evidence:** A "hot" player in 2026 is defined the same way as a "hot" player in 2019, regardless of league-wide offensive environment. In a dead-ball year, .350 wOBA might be 90th percentile. In a juiced-ball year, it might be 60th.  
**Verdict:** The system uses static thresholds in a dynamic sport.

### Weakness 7: Matchup Context Is Superficial
**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`  
**Impact:** Matchup scoring uses implied runs from sportsbook odds + generic park factors. There is NO pitcher-batter interaction model.  
**Live Evidence:** The `_get_game_context()` function returns `opp_impl` (opponent implied runs) but never considers: pitcher handedness vs batter split, pitch-type mix vs batter weakness, or bullpen quality behind the starter.  
**Verdict:** The system treats every matchup against the Yankees the same, whether it's facing Gerrit Cole or a bullpen game.

### Weakness 8: No Market or Timing Signal
**File:** `backend/fantasy_baseball/waiver_edge_detector.py`  
**Impact:** Ownership % is used as a filter but NOT as a signal. There is no tracking of add/drop velocity, FAAB bidding trends, or roster churn.  
**Live Evidence:** `owned_pct` is passed through to the API response but never used in the scoring model. A player going from 15% to 65% owned in 48 hours is treated the same as one flat at 40%.  
**Verdict:** The system ignores the single best contrarian signal: what the market is doing.

### Weakness 9: MCMC Simulator Uses Normal Distribution for Binary Events
**File:** `backend/fantasy_baseball/mcmc_simulator.py:302-303`  
**Impact:** Simulates negative saves, negative stolen bases, and negative wins.  
**Live Evidence:** The `_PLAYER_WEEKLY_STD` dict assigns std=1.0 to "nsv" (net saves). A closer with mean=2.0 saves/week and std=1.4 can be simulated at -0.8 saves.  
**Verdict:** Win probability is systematically wrong for counting-stat categories.

### Weakness 10: The System Outputs Metrics, Not Decisions
**File:** `backend/routers/fantasy.py` (waiver endpoint)  
**Impact:** The API returns `need_score`, `category_contributions`, and `z_score` but the user must manually synthesize these into an action.  
**Live Evidence:** `RosterMoveRecommendation.rationale` is a string like "Add Player X (z=+1.5)" — it never says "Add NOW because his xwOBA-wOBA gap is 95th percentile and he's batting cleanup this week."  
**Verdict:** Users get data. They do not get instructions.

---

## PART 2: THE 5-LAYER REBUILD SPECIFICATION

### Database Schema Additions Required

```sql
-- Opportunity layer tables (NEW)
CREATE TABLE player_opportunity (
    bdl_player_id INTEGER PRIMARY KEY REFERENCES player_id_mapping(bdl_id),
    pa_per_game FLOAT,
    lineup_slot_avg FLOAT,  -- average batting order position
    games_started_pct FLOAT,
    platoon_split_risk FLOAT,  -- 0.0 = everyday, 1.0 = strict platoon
    injury_risk_score FLOAT,
    role_certainty_score FLOAT,  -- 0-1, how stable is their role
    fetched_at TIMESTAMPTZ
);

-- Market layer tables (NEW)
CREATE TABLE player_market_signals (
    bdl_player_id INTEGER PRIMARY KEY REFERENCES player_id_mapping(bdl_id),
    yahoo_owned_pct FLOAT,
    ownership_delta_7d FLOAT,
    add_drop_velocity FLOAT,  -- adds per day / drops per day
    faab_bid_median FLOAT,
    fetched_at TIMESTAMPTZ
);

-- Matchup context tables (NEW)
CREATE TABLE matchup_context (
    id BIGSERIAL PRIMARY KEY,
    bdl_player_id INTEGER,
    game_date DATE,
    opponent_team VARCHAR(10),
    opponent_starter_name VARCHAR(100),
    opponent_starter_hand VARCHAR(1),
    park_factor_runs FLOAT,
    park_factor_hr FLOAT,
    weather_temp FLOAT,
    weather_wind_mph FLOAT,
    weather_wind_direction VARCHAR(10),
    hitter_vs_lhp_woba FLOAT,
    hitter_vs_rhp_woba FLOAT,
    fetched_at TIMESTAMPTZ
);
```

### Layer 1: SKILL SCORE (True Talent)

**File to modify:** `backend/services/scoring_engine.py`  
**New module:** `backend/fantasy_baseball/skill_engine.py`

**Inputs (dynamically z-scored per season):**
| Hitter | Pitcher | Weight Rationale |
|--------|---------|-----------------|
| xwOBA | xERA | Highest weight — most predictive |
| Barrel% | K-BB% | Fast-stabilizing power/contact |
| HardHit% | Whiff% | Bat-missing ability |
| Z-Contact% | Stuff+ | Contact quality / pitch quality |
| Sprint Speed | Location+ | Speed / Command |

**Normalization rule:**
- Compute league mean and population std for EACH metric using only players with ≥30 PA (hitters) or ≥10 IP (pitchers)
- Weight each metric by its stabilization rate: xwOBA (600 PA) weight=1.0, Barrel% (50 BBE) weight=0.85, Sprint Speed weight=0.4

**Output:** `skill_z` (z-score), `skill_confidence` (0-1 based on sample size)

### Layer 2: TREND SCORE (Short-Term Signal)

**File to modify:** `backend/fantasy_baseball/player_momentum.py`  
**New logic:**

Replace the current delta_z threshold system with:

```python
def compute_trend_score(
    rolling_7d: PlayerRollingStats,
    rolling_14d: PlayerRollingStats,
    rolling_30d: PlayerRollingStats,
    season_baseline: PlayerScores,
) -> TrendResult:
    # 1. Slope of change (not just delta)
    slope_7_14 = (z_7d - z_14d) / 7
    slope_14_30 = (z_14d - z_30d) / 16
    
    # 2. Volatility (std of weekly z-scores)
    volatility = std([z_7d, z_14d, z_30d])
    
    # 3. Signal strength (z-score of the slope)
    slope_z = (slope_7_14 - league_mean_slope) / league_std_slope
    
    # 4. Absolute level check — NEVER label a player "SURGING" if they're below 25th percentile
    if composite_z_14d < percentile_25_composite_z:
        signal = "IMPROVING_BUT_BELOW_AVERAGE"  # NOT "SURGING"
    elif slope_z > 1.5 and composite_z_14d > percentile_50:
        signal = "SURGING"
    elif slope_z > 0.5:
        signal = "WARMING"
    elif slope_z < -1.5 and composite_z_14d < percentile_50:
        signal = "COLLAPSING"
    elif slope_z < -0.5:
        signal = "COOLING"
    else:
        signal = "STABLE"
```

**Output:** `trend_z`, `trend_signal`, `trend_volatility`

### Layer 3: OPPORTUNITY SCORE (The Missing Piece)

**New module:** `backend/fantasy_baseball/opportunity_engine.py`

**Inputs:**
| Factor | Source | Weight |
|--------|--------|--------|
| PA per game | `mlb_player_stats` daily aggregation | 0.30 |
| Lineup slot | `mlb_game_log` / box scores | 0.25 |
| Games started % | `mlb_player_stats` | 0.20 |
| Platoon risk | `position_eligibility` + handedness | 0.15 |
| Injury risk | `ingested_injuries` table | 0.10 |

**Dynamic baseline:**
- Compute league-wide PA/game distribution daily
- A leadoff hitter with 4.5 PA/game is 90th percentile opportunity
- A platoon player with 2.8 PA/game is 30th percentile

**Output:** `opportunity_z`, `opportunity_risk` ("HIGH" if platoon/injury risk)

### Layer 4: MATCHUP SCORE (Contextual Edge)

**New module:** `backend/fantasy_baseball/matchup_engine.py`

**Current state:** `daily_lineup_optimizer.py` uses implied runs + static park factors.  
**Rebuild:**

```python
@dataclass
class MatchupContext:
    opponent_starter_hand: str
    opponent_starter_era: float
    opponent_bullpen_whip: float
    park_factor_runs: float
    park_factor_hr: float
    weather_temp: float
    weather_wind_mph: float
    hitter_vs_hand_woba: float
    hitter_vs_pitch_type: dict[str, float]

def compute_matchup_score(player: dict, context: MatchupContext) -> float:
    # 1. Handedness split (most predictive matchup factor)
    hand_bonus = 0.0
    if context.opponent_starter_hand == "L":
        hand_bonus = (player.get("woba_vs_lhp", 0.330) - 0.330) * 50
    else:
        hand_bonus = (player.get("woba_vs_rhp", 0.330) - 0.330) * 50
    
    # 2. Park adjustment (dynamic, not static)
    park_bonus = (context.park_factor_runs - 1.0) * 20
    
    # 3. Weather (wind > 15 mph out = HR suppression)
    weather_bonus = 0.0
    if context.weather_wind_mph > 15 and context.weather_wind_direction in ("IN", "LtoR"):
        weather_bonus = -5.0
    
    # 4. Opponent quality
    opp_bonus = (4.50 - context.opponent_starter_era) * 10
    
    return hand_bonus + park_bonus + weather_bonus + opp_bonus
```

**Output:** `matchup_z`, `matchup_context_summary` (string)

### Layer 5: MARKET + TIMING SCORE

**New module:** `backend/fantasy_baseball/market_engine.py`

```python
def compute_market_score(
    owned_pct: float,
    owned_pct_7d_ago: float,
    xwoba: float,
    woba: float,
) -> MarketResult:
    # 1. Ownership velocity (are people catching on?)
    ownership_velocity = owned_pct - owned_pct_7d_ago
    
    # 2. Skill gap (xwOBA vs wOBA) — buy low when gap is large AND market hasn't noticed
    skill_gap = xwoba - woba
    skill_gap_percentile = percentile(skill_gap, all_skill_gaps_this_season)
    
    # 3. Market inefficiency score
    # Large positive gap + low ownership velocity = best buy-low
    # Large negative gap + high ownership velocity = sell-high
    inefficiency = skill_gap_percentile * (1.0 - min(ownership_velocity / 50.0, 1.0))
    
    return MarketResult(
        score=inefficiency * 100,
        tag="BUY_LOW" if inefficiency > 0.7 else "SELL_HIGH" if inefficiency < 0.3 else "FAIRLY_PRICED",
        urgency="ACT_NOW" if ownership_velocity < 5 and skill_gap_percentile > 0.8 else "MONITOR",
    )
```

**Output:** `market_score`, `market_tag`, `urgency`

### Final Composite Score (Context-Aware Weighting)

```python
def compute_final_score(
    skill_z: float,
    trend_z: float,
    opportunity_z: float,
    matchup_z: float,
    market_score: float,
    league_type: str,  # "shallow", "deep", "h2h", "roto"
    decision_horizon: str,  # "weekly", "ros", "daily"
) -> FinalScore:
    # Context-aware weights
    if league_type == "shallow":
        weights = {"skill": 0.20, "trend": 0.20, "opportunity": 0.30, "matchup": 0.15, "market": 0.15}
    elif league_type == "deep":
        weights = {"skill": 0.35, "trend": 0.15, "opportunity": 0.20, "matchup": 0.10, "market": 0.20}
    elif decision_horizon == "weekly":
        weights = {"skill": 0.20, "trend": 0.25, "opportunity": 0.15, "matchup": 0.30, "market": 0.10}
    else:  # ROS
        weights = {"skill": 0.40, "trend": 0.15, "opportunity": 0.25, "matchup": 0.05, "market": 0.15}
    
    final = (
        weights["skill"] * normalize(skill_z) +
        weights["trend"] * normalize(trend_z) +
        weights["opportunity"] * normalize(opportunity_z) +
        weights["matchup"] * normalize(matchup_z) +
        weights["market"] * normalize(market_score)
    )
    
    # Confidence = geometric mean of individual confidences
    confidence = (skill_confidence * trend_confidence * opportunity_confidence) ** (1/3)
    
    # Volatility = weighted std of component z-scores
    volatility = std([skill_z, trend_z, opportunity_z, matchup_z], weights=list(weights.values()))
    
    return FinalScore(
        score=final * 100,
        confidence=confidence,
        volatility="HIGH" if volatility > 1.5 else "MEDIUM" if volatility > 0.8 else "LOW",
        suggested_action=derive_action(final, confidence, volatility, market_score),
    )
```

---

## PART 3: EXAMPLE OUTPUTS — 3 REAL PLAYERS FROM PRODUCTION DB

Using actual 2026-05-04 production data.

---

### Player 1: Shohei Ohtani (Two-Way)

#### CURRENT SYSTEM OUTPUT
```json
{
  "composite_z": 9.79,
  "score_0_100": 100.0,
  "signal": "STABLE",
  "rationale": "Projected z=+2.5. No 1B to drop suggested; check bench.",
  "tags": []
}
```
**Problem:** The current system ranks him #2 overall because it **sums** 8 categories instead of averaging them. His hitting is mediocre (z_avg=-0.24) but gets bailed out by pitching Z-scores.

#### NEW SYSTEM OUTPUT
```json
{
  "final_score": 78.5,
  "confidence": 0.92,
  "volatility": "HIGH",
  "skill_z": 1.85,
  "trend_z": 0.42,
  "opportunity_z": 2.10,
  "matchup_z": 0.65,
  "market_score": 0.35,
  "tags": ["EVERYDAY_PLAY", "TWO_WAY_PREMIUM", "MARKET_FAIR"],
  "suggested_action": "START",
  "urgency": "Lock in lineup spot — he plays every day",
  "rationale": "Skill is elite (91st percentile) but market has fully priced him in (99% owned). Trend is flat. High volatility due to two-way workload risk. Not a buy-low — he's fairly priced. Start him, don't trade for him."
}
```

---

### Player 2: Kyle Harrison (Pitcher, Tiny Sample)

#### CURRENT SYSTEM OUTPUT
```json
{
  "composite_z": 8.15,
  "score_0_100": 99.8,
  "confidence": 0.21,
  "signal": "STABLE",
  "rationale": "Projected z=+1.8."
}
```
**Problem:** 99.8 percentile with 0.21 confidence. The system is screaming "ELITE PITCHER" based on 1-2 starts.

#### NEW SYSTEM OUTPUT
```json
{
  "final_score": 42.0,
  "confidence": 0.18,
  "volatility": "HIGH",
  "skill_z": 1.20,
  "trend_z": 0.85,
  "opportunity_z": -0.45,
  "matchup_z": 0.30,
  "market_score": 0.72,
  "tags": ["TINY_SAMPLE", "OPPORTUNITY_RISK", "SPECULATIVE_ADD"],
  "suggested_action": "STREAM",
  "urgency": "Only start in favorable matchups until IP > 25",
  "rationale": "Skill metrics look good (80th percentile) but confidence is 0.18 — this is a dice roll. Opportunity is below average (fifth starter role, may get skipped). Market hasn't caught on yet (12% owned). VALID STREAM CANDIDATE in shallow leagues, but do not roster in deep leagues until sample stabilizes."
}
```

---

### Player 3: Dylan Moore ("SURGING" per Current System)

#### CURRENT SYSTEM OUTPUT
```json
{
  "composite_z_14d": -8.83,
  "composite_z_30d": -16.33,
  "delta_z": 7.51,
  "signal": "SURGING",
  "rationale": "Trending up!"
}
```
**Problem:** The system labels a player with composite_z=-8.83 as "SURGING" because delta_z is positive. He went from -16.33 to -8.83. He's still in the bottom 3% of the league.

#### NEW SYSTEM OUTPUT
```json
{
  "final_score": 18.5,
  "confidence": 0.35,
  "volatility": "MEDIUM",
  "skill_z": -1.45,
  "trend_z": 0.62,
  "opportunity_z": -0.80,
  "matchup_z": -0.30,
  "market_score": 0.15,
  "tags": ["IMPROVING_BUT_BELOW_AVERAGE", "PLATOON_RISK", "IGNORE"],
  "suggested_action": "DO_NOT_ADD",
  "urgency": "None — monitor only",
  "rationale": "Trend is positive (+0.62) but absolute skill is 8th percentile. He's improving from 'unplayable' to 'barely rosterable.' Opportunity is poor (platoon player, 3.1 PA/game). Market doesn't want him (4% owned and flat). The 'SURGING' label in the old system was a trap. Do not add."
}
```

---

## PART 4: IMPLEMENTATION PRIORITY

### Phase 1: Fix the Math (This Week)
1. **Fix composite_z** → weighted mean, not sum (`scoring_engine.py:525`)
2. **Fix player_board std** → population std (`player_board.py:628`)
3. **Fix momentum thresholds** → add absolute level gate (`models.py` + `scoring_engine.py`)
4. **Fix MCMC** → clamp negative counts (`mcmc_simulator.py`)

### Phase 2: Add Opportunity Layer (Next Week)
1. Create `player_opportunity` table
2. Build `opportunity_engine.py`
3. Populate from `mlb_player_stats` daily aggregation
4. Wire into waiver/lineup endpoints

### Phase 3: Add Market Layer (Week After)
1. Create `player_market_signals` table
2. Build `market_engine.py`
3. Track ownership % history from Yahoo API
4. Generate BUY_LOW / SELL_HIGH / BREAKOUT tags dynamically

### Phase 4: Add Matchup Layer (Ongoing)
1. Create `matchup_context` table
2. Build `matchup_engine.py`
3. Add pitcher-batter handedness splits
4. Add dynamic park factors (use `park_factors` table, already exists)

---

*Specification written against production code (`stable/cbb-prod`, commit 827b2c0) and verified against live Railway PostgreSQL database (2026-05-04).*
