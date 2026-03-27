# Fantasy Baseball System — Architecture Analysis

> **Date:** March 26, 2026  
> **Purpose:** Compare designed vs. implemented, identify gaps, prioritize work

---

## 1. The Vision (From Research Docs)

### Core Philosophy
Transform fantasy baseball from "draft helper with waiver lists" into an **institutional-grade quantitative asset management system** combining:
- Sabermetrics (Statcast, xwOBA, barrel%)
- Quantitative finance (mean-variance optimization, Sharpe ratios)
- Machine learning (Bayesian updating, RL, GNNs)
- Multi-agent orchestration

### Key Algorithms Designed

| Component | Algorithm | Status | Owner |
|-----------|-----------|--------|-------|
| Projection Updating | Bayesian conjugate update with shrinkage | ❌ NOT BUILT | Claude |
| Multi-Source Projections | Inverse-MAE weighted ensemble | ❌ NOT BUILT | Claude |
| Weekly Outcomes | MCMC (Gibbs sampling, 10k sims) | ❌ NOT BUILT | Claude |
| Roster Optimization | Mean-variance quadratic programming | ❌ NOT BUILT | Claude |
| Real-Time Decisions | Contextual bandit (LinUCB) | ❌ NOT BUILT | Claude |
| Long-Term Strategy | Deep Q-Network (DQN) | ❌ NOT BUILT | Claude |
| Daily Lineup | Graph Neural Network | ❌ NOT BUILT | Claude |

### Data Pipeline Architecture (Designed)

```
┌─────────────────────────────────────────────────────────────┐
│                  DATA INGESTION LAYER                       │
│  Yahoo API  →  Statcast  →  FanGraphs  →  MLB Stats API     │
│   (live)       (daily)       (daily)        (real-time)     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              PROJECTION ENGINE LAYER                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Steamer    │  │   Bayesian   │  │    MLE       │      │
│  │   (Prior)    │  │   Updater    │  │  (Minors)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│              DECISION ENGINE LAYER                          │
│  MCMC Simulator → Portfolio Optimizer → GNN Lineup Setter   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Current Reality (What's Actually Implemented)

### ✅ What's Working

| Component | Implementation | Quality |
|-----------|---------------|---------|
| Yahoo OAuth | `yahoo_client.py` | ✅ Production-ready |
| Static Projections | Steamer CSV from March 9 | ⚠️ 17 days stale |
| Daily Lineup (Basic) | Implied runs × park factor | ⚠️ Simplistic (40% variance) |
| Waiver Wire | Edge detector with category deficits | ✅ Good foundation |
| Player Board | Hardcoded + CSV fallback | ⚠️ Static data |
| Team Abbr Mapping | Normalization layer | ✅ Recently fixed |

### ❌ Critical Gaps

| Gap | Impact | Evidence |
|-----|--------|----------|
| **No live Statcast ingestion** | Can't identify breakout candidates | "Hot" players rely on 17-day-old data |
| **No Bayesian updating** | Projections don't learn from 2026 season | Early-season surprises ignored |
| **No pitcher quality integration** | Matchup quality ignored | Research: 20-30% performance shift |
| **No platoon split data** | LHP/RHP matchups ignored | Kyle Schwarber: .920 vs RHP, .650 vs LHP |
| **No rolling form** | Hot/cold streaks ignored | 7-day/14-day performance not weighted |
| **No MCMC simulation** | Can't say "70% chance to win HR" | Single-point projections only |
| **No pattern detection** | Miss pitcher fatigue, bullpen overuse | Logs show 0 probable pitchers fetched |

---

## 3. The Real Problem

### What I Was Doing (Wrong)
- Fixing timezone bugs
- Correcting team abbreviations
- Patching scorer logic

### What Needs to Be Built (Right)
- **Live data pipeline** that pulls from Baseball Savant daily
- **Bayesian updater** that continuously adjusts projections
- **Matchup quality engine** with pitcher xERA integration
- **Pattern detector** for MLB-specific vulnerabilities
- **MCMC simulator** for probabilistic weekly projections

---

## 4. Priority Implementation Roadmap

### Phase 1: Foundation (Next 2 Weeks)

| Task | Effort | Owner | Deliverable |
|------|--------|-------|-------------|
| **Statcast Daily Ingestion** | 8h | Claude | `statcast_ingestion.py` pulls yesterday's data |
| **Bayesian Projection Updater** | 12h | Claude | `bayesian_updater.py` with shrinkage priors |
| **Pitcher Quality Integration** | 6h | Claude | Matchup multiplier based on xERA |
| **Platoon Split Loader** | 4h | Claude | FanGraphs splits scraper |
| **MCMC Simulator (Basic)** | 10h | Claude | 10k sim weekly matchup engine |

**Phase 1 Success Criteria:**
- Projections update daily with new Statcast data
- Matchup quality affects lineup scores
- Can run 10k sims and output "70% chance to win HR"

### Phase 2: Intelligence (Weeks 3-4)

| Task | Effort | Owner | Deliverable |
|------|--------|-------|-------------|
| **Ensemble Projector** | 8h | Claude | Weight Steamer + ZiPS + Yahoo ROS |
| **MLB Pattern Detector** | 10h | OpenClaw | Pitch fatigue, bullpen overuse alerts |
| **Rolling Form Tracker** | 6h | Claude | 7/14/30 day rolling averages |
| **Contextual Bandit** | 12h | Claude | LinUCB for waiver decisions |

### Phase 3: Optimization (Weeks 5-6)

| Task | Effort | Owner | Deliverable |
|------|--------|-------|-------------|
| **Portfolio Optimizer** | 14h | Claude | Mean-variance roster optimization |
| **GNN Lineup Setter** | 16h | Claude | Optimal daily lineup selection |
| **Weekly Strategy Engine** | 10h | Claude | Category allocation, punt analysis |

---

## 5. Immediate Next Steps (This Week)

### 5.1 Live Data Pipeline (Critical)

```python
# New file: backend/fantasy_baseball/live_data_pipeline.py

class StatcastIngestionAgent:
    """
    Daily ingestion of Statcast data.
    Run at 6 AM ET after previous night's games complete.
    """
    
    def ingest_yesterday(self) -> List[PlayerPerformance]:
        """
        Pull yesterday's games from Baseball Savant.
        Calculate: xwOBA, barrel%, exit velocity, hard hit%
        """
        
    def update_projections(self, performances: List[PlayerPerformance]):
        """
        Bayesian update: posterior = prior × likelihood
        Shrinkage factor tells us how much to trust new data.
        """

class ProbablePitcherAgent:
    """
    Fetch today's probable pitchers from MLB Stats API.
    Cross-reference with Statcast for xERA, K%, BB%.
    """
```

### 5.2 Matchup Quality Engine

```python
# Enhance: backend/fantasy_baseball/daily_lineup_optimizer.py

def calculate_matchup_multiplier(
    batter_woba: float,
    pitcher_xera: float,
    league_avg_era: float = 4.00
) -> float:
    """
    Calculate matchup quality multiplier.
    
    3.00 ERA pitcher vs .350 wOBA batter:
        multiplier = 1.0 + (4.0 - 3.0) * 0.05 = 1.05
    
    5.00 ERA pitcher vs same batter:
        multiplier = 1.0 + (4.0 - 5.0) * 0.05 = 0.95
    """
    return 1.0 + (league_avg_era - pitcher_xera) * 0.05
```

### 5.3 MCMC Simulator (Foundation)

```python
# New file: backend/fantasy_baseball/mcmc_simulator.py

class WeeklyMCMCSimulator:
    """
    Run 10,000 simulations of weekly matchup.
    Output: Win probabilities per category.
    """
    
    def simulate(
        self,
        my_roster: List[Player],
        opponent_roster: List[Player],
        n_sims: int = 10000
    ) -> MatchupSimulation:
        """
        Returns:
            - category_win_probabilities: Dict[str, float]
            - expected_categories_won: float
            - worst_case_scenario: float (10th percentile)
            - best_case_scenario: float (90th percentile)
        """
```

---

## 6. Success Metrics

| Metric | Current | Target | How to Measure |
|--------|---------|--------|----------------|
| Projection MAE | Unknown | < 0.30 wOBA | Compare to actuals after 30 games |
| Category Win Rate | Unknown | > 55% | Track H2H results weekly |
| Top 20% Hit Rate | Unknown | > 35% | % of recommended players finishing top 20% |
| Data Freshness | 17 days | < 24 hours | Time since last Statcast pull |
| Recommendation Latency | N/A | < 5 seconds | API response time |

---

## 7. Files to Create/Modify

### New Files (This Week)
1. `backend/fantasy_baseball/statcast_ingestion.py` — Daily Statcast pulls
2. `backend/fantasy_baseball/bayesian_updater.py` — Projection updating
3. `backend/fantasy_baseball/mcmc_simulator.py` — Weekly matchup sims
4. `backend/fantasy_baseball/pitcher_quality.py` — xERA matchup data

### Modify (This Week)
1. `daily_lineup_optimizer.py` — Add matchup multiplier
2. `smart_lineup_selector.py` — Integrate MCMC
3. `main.py` — Add endpoints for live data

---

**Bottom Line:** The bug fixes were necessary but insufficient. The real work is building the live data pipeline and quantitative engine that makes this an elite system. Let's focus on Phase 1 starting immediately.
