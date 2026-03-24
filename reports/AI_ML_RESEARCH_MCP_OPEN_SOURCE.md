# AI/ML Research: MCP & Open Source Projects for CBB Edge

**Author:** Kimi CLI (Deep Intelligence Unit)  
**Date:** March 23, 2026  
**Purpose:** Research advanced AI/ML features and open-source projects applicable to Fantasy Baseball and CBB betting systems

---

## Part 1: Understanding MCP (Model Context Protocol)

### What is MCP?

**MCP (Model Context Protocol)** is an open-source JSON-RPC standard introduced by Anthropic in November 2024. It solves the "N×M integration problem" by providing a universal interface for AI applications to connect with external tools, data sources, and APIs.

**Analogy:** MCP is to AI what USB-C is to hardware — a single standard that replaces fragmented, custom integrations.

### Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MCP HOST                                │
│          (Your AI application - Claude, Cursor, etc.)       │
└──────────────────────────┬──────────────────────────────────┘
                          │
┌──────────────────────────▼──────────────────────────────────┐
│                     MCP CLIENT                              │
│          (Manages connection to MCP servers)                │
└──────────────────────────┬──────────────────────────────────┘
                          │
┌──────────────────────────▼──────────────────────────────────┐
│                     MCP SERVERS                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ File System │  │  Database   │  │   APIs      │         │
│  │   Server    │  │   Server    │  │  (Yahoo,    │         │
│  │             │  │             │  │  Statcast)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Why MCP Matters for Your Application

| Problem Without MCP | Solution With MCP |
|---------------------|-------------------|
| Custom Yahoo API integration | Standardized `mcp-yahoo-fantasy` server |
| Hardcoded Statcast calls | Reusable `mcp-statcast` server |
| Fragile Discord webhooks | `mcp-discord` routing layer |
| No audit trail | Built-in logging per MCP transaction |
| Vendor lock-in to one LLM | Any MCP-compatible client can use your servers |

### MCP Components for Fantasy Baseball

```python
# MCP Server Example: Fantasy Baseball Data Provider
class FantasyBaseballMCPServer:
    """
    Exposes Yahoo Fantasy, Statcast, and projection data
    via standardized MCP protocol
    """
    
    @mcp.tool()
    async def get_player_projections(player_id: str) -> dict:
        """Get ROS projections for a player"""
        return await projection_service.get(player_id)
    
    @mcp.tool()
    async def get_free_agents(league_id: str, position: str) -> list:
        """List available free agents"""
        return await yahoo_client.get_free_agents(league_id, position)
    
    @mcp.resource("statcast://{player_id}/exit_velocity")
    async def get_exit_velocity(player_id: str) -> float:
        """Get average exit velocity from Statcast"""
        return await statcast_client.get_avg_ev(player_id)
```

### Relevant MCP Servers to Consider

| Server | Purpose | GitHub |
|--------|---------|--------|
| `mcp-postgres` | Database queries | github.com/modelcontextprotocol/servers |
| `mcp-discord` | Discord bot integration | Community contributed |
| `mcp-fetch` | Web scraping | Official Anthropic |
| `mcp-filesystem` | File operations | Official Anthropic |

---

## Part 2: Open Source Sports Betting & ML Projects

### Tier 1: High-Relevance Projects

#### 1. **NBA-Machine-Learning-Sports-Betting** (kyleskom)
- **URL:** github.com/kyleskom/NBA-Machine-Learning-Sports-Betting
- **Language:** Python
- **Stars:** 1.2k+
- **Key Features:**
  - XGBoost + Neural Network models
  - Real-time odds fetching from multiple sportsbooks
  - Kelly Criterion bankroll management
  - ~69% accuracy on moneylines, ~55% on totals
  - Flask web app for predictions
- **Applicable to CBB:** High (same sport betting concepts)
- **Key Files to Study:**
  - `main.py` - Kelly criterion implementation
  - `XGBoost_Model/` - Feature engineering patterns
  - `Flask/` - Web app architecture

#### 2. **ML for Sports Betting** (conorwalsh99) - Academic Grade
- **URL:** github.com/conorwalsh99/ml-for-sports-betting
- **Language:** Python
- **Published:** Machine Learning with Applications journal (2024)
- **Key Features:**
  - **Calibration-focused model selection** (not just accuracy)
  - Random Forest + Gradient Boost comparison
  - Brier Score evaluation
  - Poetry dependency management
- **Academic Paper:** "Should model selection be based on accuracy or calibration?"
- **Applicable to CBB:** Very High - calibration is your edge
- **Key Insight:** "Well-calibrated models produce probability estimates that match the true likelihood of events"

#### 3. **baseballforecaster** (baileymorton989)
- **URL:** github.com/baileymorton989/baseballforecaster
- **Language:** Python
- **Key Features:**
  - **Monte Carlo simulation for fantasy drafts**
  - Mean-shift clustering for player groups
  - Risk-adjusted scoring
  - Monte Carlo Search Tree for drafting
- **Applicable to Fantasy Baseball:** Direct relevance
- **Code Patterns to Extract:**
  ```python
  # Their MC simulation approach
  forecaster.monte_carlo_forecast()  # Season simulation
  drafter.draft()  # Search tree optimization
  ```

### Tier 2: Moderate-Relevance Projects

#### 4. **Sports_Prediction_and_Betting_Model** (Ali-m89)
- **URL:** github.com/Ali-m89/Sports_Prediction_and_Betting_Model
- **ROI:** 1.95% over 4,431 MLB matches
- **Features:** Random Forest, Kelly Criterion, Brier Score tracking
- **Key Learning:** Their feature engineering from shared opponents

#### 5. **mlb_game_predictor** (laplaces42)
- **URL:** github.com/laplaces42/mlb_game_predictor
- **Models:** Ridge Classifier, Linear Regression
- **Data:** FanGraphs + MLB-StatsAPI
- **Value:** Good example of multi-source data integration

#### 6. **baseball-analytics** (eric8395)
- **URL:** github.com/eric8395/baseball-analytics
- **Features:**
  - PyBaseball integration
  - 300+ Statcast features
  - Gradient Boost (R² = 0.78 for batting)
  - Streamlit deployment
- **Applicable:** Salary prediction models can adapt to fantasy value

### Tier 3: Reference Projects

#### 7. **fantasy-baseball** (cdchan)
- **URL:** github.com/cdchan/fantasy-baseball
- **Focus:** ESPN/Yahoo H2H categories leagues
- **Approach:** Cumulative winning probability added (WPA)
- **Value:** Valuation model using rest-of-season projections

#### 8. **fantasy_baseball** (lbenz730)
- **URL:** github.com/lbenz730/fantasy_baseball
- **Features:**
  - Win probability models (XGBoost)
  - Shiny app for ESPN leagues
  - Model calibration tracking

---

## Part 3: Probabilistic Programming Libraries

### NumPyro (Pyro + JAX)

**URL:** github.com/pyro-ppl/numpyro

**Why It Matters:**
- **JAX-powered:** Automatic differentiation, JIT compilation, GPU/TPU support
- **Fast MCMC:** NUTS sampler for Bayesian inference
- **Baseball example included:** Efron-Morris batting average (hierarchical model)

**Key Example: Baseball Hierarchical Model**
```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

def baseball_model(at_bats, hits):
    """
    Partial pooling model for batting averages.
    Shares information across players (hierarchical).
    """
    # Hyperparameters
    mu_p = numpyro.sample("mu_p", dist.Beta(1, 1))
    sigma_p = numpyro.sample("sigma_p", dist.Uniform(0, 0.5))
    
    # Player-level parameters (partially pooled)
    with numpyro.plate("players", len(at_bats)):
        p = numpyro.sample("p", dist.Beta(mu_p, sigma_p))
        numpyro.sample("obs", dist.Binomial(at_bats, p), obs=hits)

# Run inference
nuts_kernel = NUTS(baseball_model)
mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500)
mcmc.run(rng_key, at_bats, hits)
```

**For Your Fantasy Baseball System:**
- Bayesian projection updating (our Phase 2 spec)
- Uncertainty quantification for player values
- Hierarchical models for position/age effects

### PyMC + Bambi

**URL:** github.com/pymc-devs/pymc

**Key Resource: Bayesian MARCEL Implementation**
- Blog: pymc-labs.com/blog-posts/bayesian-marcel
- **Innovation:** Probabilistic version of Tom Tango's MARCEL projection system
- **Features:**
  - Dirichlet-distributed weights (learns optimal season weighting)
  - Triangular aging curves with estimated peak age
  - Hierarchical partial pooling
  - Full uncertainty quantification

**Code Pattern to Adopt:**
```python
import pymc as pm
import bambi as bmb

# Bambi for high-level specification
model = bmb.Model(
    "hard_hit_rate ~ 1 + (1|player_id) + age + age_peak",
    data=player_data,
    family="beta"
)

# PyMC for low-level control
with pm.Model() as marcel:
    # Dirichlet weights for 3-year history
    weights = pm.Dirichlet("weights", a=np.array([1, 1, 1]))
    
    # Age adjustment
    age_effect = pm.Triangular("age_effect", lower=20, upper=40, c=28)
    
    # Hierarchical player effects
    player_mu = pm.Beta("player_mu", mu=0.35, sigma=0.2)
    player_sigma = pm.Uniform("player_sigma", 0, 0.5)
    
    # ... projection logic
```

### PyMC Sports Analytics Tutorial

**URL:** github.com/fonnesbeck/hierarchical_models_sports_analytics

**Presenter:** Chris Fonnesbeck (Principal Quant Analyst, Philadelphia Phillies)

**Key Topics:**
- Hierarchical models for small sample sizes
- Partial pooling for player evaluation
- Real-time Bayesian updating during games
- Model comparison (LOO, posterior predictive checks)

**Why This Is Gold:**
- Built by MLB practitioner
- Uses actual baseball data
- Covers exactly what you need: updating beliefs in-season

---

## Part 4: Specific Algorithms to Implement

### From Research to Implementation

| Algorithm | Source | Difficulty | Priority | Files to Create |
|-----------|--------|------------|----------|-----------------|
| **Bayesian MARCEL** | PyMC Labs blog | Medium | P1 | `backend/fantasy_baseball/bayesian_marcel.py` |
| **Hierarchical Player Pooling** | Fonnesbeck tutorial | High | P1 | `backend/fantasy_baseball/hierarchical_pooling.py` |
| **MCMC Weekly Simulator** | baseballforecaster | High | P2 | `backend/fantasy_baseball/mcmc_weekly.py` |
| **Kelly Criterion** | NBA-ML repos | Low | P0 | Already in betting_model.py |
| **XGBoost Ranker** | kyleskom repo | Medium | P2 | `backend/fantasy_baseball/xgboost_ranker.py` |
| **Calibration Tracking** | Walsh paper | Medium | P2 | `backend/fantasy_baseball/calibration.py` |

### NumPyro Implementation: Bayesian Projection Updater

```python
# backend/fantasy_baseball/bayesian_updater.py
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import NUTS, MCMC

class BayesianProjectionUpdater:
    """
    Continuously update projections as new games arrive.
    Based on NumPyro's baseball example + MARCEL concepts.
    """
    
    def __init__(self, prior_mean: dict, prior_variance: dict):
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        
    def model(self, games_played: int, observed_stats: dict):
        """
        Update player projection with new evidence.
        
        Args:
            games_played: Number of games in sample
            observed_stats: Dict of {stat_name: observed_value}
        """
        for stat, prior_mu in self.prior_mean.items():
            # Prior precision (confidence in preseason projection)
            prior_precision = 1.0 / self.prior_variance[stat]
            
            # Likelihood precision (based on sample size)
            # More games = more confidence in observed data
            sample_variance = observed_stats.get(f"{stat}_var", 0.01)
            likelihood_precision = games_played / max(sample_variance, 0.001)
            
            # Posterior (conjugate normal update)
            posterior_precision = prior_precision + likelihood_precision
            posterior_mean = (
                (prior_precision * prior_mu + 
                 likelihood_precision * observed_stats[stat]) / 
                posterior_precision
            )
            
            # Sample from posterior
            numpyro.sample(
                f"projected_{stat}",
                dist.Normal(posterior_mean, 1.0/jnp.sqrt(posterior_precision))
            )
    
    def update(self, rng_key, games_played: int, observed_stats: dict) -> dict:
        """
        Run MCMC to get updated projections with uncertainty.
        """
        nuts_kernel = NUTS(self.model)
        mcmc = MCMC(nuts_kernel, num_samples=1000, num_warmup=500)
        mcmc.run(rng_key, games_played, observed_stats)
        
        # Extract posterior means as updated projections
        samples = mcmc.get_samples()
        updated = {}
        for stat in self.prior_mean.keys():
            updated[stat] = {
                "mean": float(jnp.mean(samples[f"projected_{stat}"])),
                "std": float(jnp.std(samples[f"projected_{stat}"])),
                "ci_95": (
                    float(jnp.percentile(samples[f"projected_{stat}"], 2.5)),
                    float(jnp.percentile(samples[f"projected_{stat}"], 97.5))
                )
            }
        return updated
```

---

## Part 5: Recommended Implementation Path

### Phase 1: Foundation (Week 1-2)

1. **Install Dependencies**
```bash
pip install numpyro jax jaxlib pymc bambi arviz
```

2. **Study These Repos First**
- `baseballforecaster` - Monte Carlo patterns
- `nba-ml-sports-betting` - Kelly criterion + feature engineering
- Walsh paper - Calibration importance

3. **Implement Universal Projections**
```python
# From our spec
from backend.fantasy_baseball.projections import get_or_create_projection
```

### Phase 2: Bayesian Layer (Week 3-4)

1. **Port Bayesian MARCEL**
- Adapt PyMC Labs blog code to your data structure
- Add Dirichlet weight learning
- Implement age curves

2. **Hierarchical Pooling**
- Position-level effects
- Team-level effects (park factors)
- Player-level shrinkage

### Phase 3: Advanced ML (Week 5-6)

1. **XGBoost Ranker**
- Feature: projected stats + recent performance + context
- Target: ROS fantasy value
- Calibration: Platt scaling or isotonic regression

2. **MCMC Weekly Simulator**
- Simulate 10,000 week outcomes
- Category correlation matrix
- Win probability estimates

### Phase 4: MCP Integration (Week 7-8)

1. **Create MCP Servers**
```bash
# Structure
backend/mcp_servers/
├── yahoo_fantasy_server.py
├── statcast_server.py
├── projections_server.py
└── discord_router_server.py
```

2. **Standardize Tool Calling**
```python
# Instead of direct API calls
from mcp_client import call_tool

result = await call_tool(
    "yahoo-fantasy",  # server name
    "get_free_agents",  # tool name
    {"league_id": "123", "position": "2B"}
)
```

---

## Part 6: Key Papers & Resources

### Academic Papers

1. **"Machine learning for sports betting: Should model selection be based on accuracy or calibration?"**
   - Walsh & Joshi, Machine Learning with Applications (2024)
   - DOI: 10.1016/j.mlwa.2024.100539
   - **Key Takeaway:** Calibration > Accuracy for betting

2. **"Forecasting Outcomes of Major League Baseball Games Using Machine Learning"**
   - Andrew Cui, Wharton Thesis (2020)
   - **Key Takeaway:** Ridge/Linear models outperform complex NN for MLB

### Books & Courses

1. **Fantasy Football Analytics Textbook** (Isaac Petersen)
   - URL: isaactpetersen.github.io/Fantasy-Football-Analytics-Textbook/
   - **Why:** Statistical Rethinking applied to fantasy sports
   - Free, open-source, Python/R code

2. **Statistical Rethinking with NumPyro**
   - McElreath's textbook ported to NumPyro
   - **Key:** Hierarchical models, Bayesian workflow

3. **PyMC Hierarchical Modeling Workshop** (Alex Andorra)
   - Athlyticz course (Nov 2025)
   - **Focus:** Sports analytics with PyMC

---

## Part 7: Multi-Agent Workflow with MCP

### Proposed Architecture

```yaml
# .clawhub/mcp-config.yaml
servers:
  yahoo-fantasy:
    command: python backend/mcp_servers/yahoo_server.py
    env:
      YAHOO_CLIENT_ID: ${YAHOO_CLIENT_ID}
      YAHOO_CLIENT_SECRET: ${YAHOO_CLIENT_SECRET}
  
  statcast:
    command: python backend/mcp_servers/statcast_server.py
    env:
      MLB_API_KEY: ${MLB_API_KEY}
  
  projections:
    command: python backend/mcp_servers/projections_server.py
    resources:
      - "projections://{player_id}/ros"
      - "projections://{player_id}/weekly"
  
  discord-router:
    command: python backend/mcp_servers/discord_server.py
    tools:
      - send_alert
      - send_digest

client:
  timeout: 30
  retry: 3
  cache: redis://localhost:6379
```

### Agent-Specific MCP Usage

| Agent | MCP Servers Used | Primary Tools |
|-------|-----------------|---------------|
| **Yahoo Data Fetcher** | yahoo-fantasy | `get_roster`, `get_free_agents`, `get_player_stats` |
| **Statcast Analytics** | statcast | `get_exit_velocity`, `get_barrels`, `get_pitch_data` |
| **Projection Ensemble** | projections | `get_ros_projection`, `update_bayesian`, `ensemble_forecast` |
| **Weekly Strategy** | projections, yahoo-fantasy | `simulate_matchup`, `optimize_lineup` |
| **Discord Router** | discord-router | `send_alert`, `send_digest`, `escalate` |

---

## Appendix: Quick Reference Commands

### Clone Key Repos for Study
```bash
# High priority
git clone https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting.git
git clone https://github.com/baileymorton989/baseballforecaster.git
git clone https://github.com/conorwalsh99/ml-for-sports-betting.git

# Bayesian methods
git clone https://github.com/pyro-ppl/numpyro.git
git clone https://github.com/fonnesbeck/hierarchical_models_sports_analytics.git

# Fantasy reference
git clone https://github.com/cdchan/fantasy-baseball.git
```

### Install ML Stack
```bash
# Core
pip install numpyro jax jaxlib

# Bayesian
pip install pymc bambi arviz

# ML
pip install xgboost lightgbm scikit-learn

# Probabilistic ML
pip install tensorflow-probability
```

---

## Summary

**MCP** provides the infrastructure for multi-agent coordination. **NumPyro/PyMC** provide the Bayesian machinery for sophisticated projections. The **open-source repos** provide battle-tested patterns for:
- Feature engineering from sports data
- Kelly criterion bankroll management
- Model calibration and evaluation
- Monte Carlo simulation

**Your Fantasy Baseball system can become the first MCP-native, Bayesian-powered, institutional-grade roster management platform.**

**Next Steps:**
1. Review the Bayesian MARCEL blog post (PyMC Labs)
2. Study the baseballforecaster MC simulation approach
3. Implement universal projections (Phase 1 spec)
4. Add NumPyro-based Bayesian updater (Phase 2 spec)
