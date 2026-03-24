# Fantasy Baseball Elite System — Technical Specification v2.0

**Status:** Draft — Awaiting Claude Code Implementation  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**Date:** 2026-03-23  
**Priority:** P0 (Post-March Madness Pivot)  

---

## Executive Summary

Transform the Fantasy Baseball module from a "draft helper with waiver lists" into an **institutional-grade quantitative asset management system**. This system treats fantasy baseball as a multi-agent, multi-timeframe portfolio optimization problem combining sabermetrics, quantitative finance, and machine learning.

---

## Part 1: Algorithmic Innovations (Cutting-Edge)

### 1.1 Bayesian Projection Updating

**Problem:** Static projections (Steamer/ZiPS) become stale quickly. Early season performance is noisy but contains signal.

**Solution:** Bayesian updating with shrinkage priors.

```python
class BayesianProjectionUpdater:
    """
    Continuously update projections as new data arrives.
    Balances prior (Steamer) with likelihood (actual performance).
    """
    
    def __init__(self, prior_projection: PlayerProjection):
        self.prior_mean = prior_projection  # Steamer ROS
        self.prior_precision = 1 / prior_projection.variance  # Confidence in prior
        self.observations: List[GamePerformance] = []
        
    def update(self, new_games: List[GamePerformance]):
        """
        Update projection using conjugate normal updating.
        
        posterior_precision = prior_precision + likelihood_precision
        posterior_mean = (prior_precision * prior_mean + likelihood_precision * sample_mean) / posterior_precision
        """
        # Calculate sample statistics from new games
        sample_mean = np.mean([g.woba for g in new_games])
        sample_variance = np.var([g.woba for g in new_games])
        n = len(new_games)
        
        # Precision-weighted update (shrinkage toward prior)
        likelihood_precision = n / sample_variance if sample_variance > 0 else 0
        posterior_precision = self.prior_precision + likelihood_precision
        
        posterior_mean = (
            (self.prior_precision * self.prior_mean.woba) + 
            (likelihood_precision * sample_mean)
        ) / posterior_precision
        
        # Shrinkage factor tells us how much we've learned
        shrinkage = self.prior_precision / posterior_precision
        
        return UpdatedProjection(
            mean=posterior_mean,
            variance=1 / posterior_precision,
            shrinkage=shrinkage,  # Close to 1.0 = trust prior; Close to 0.0 = trust data
            sample_size=n,
        )
```

**Key Insight:** Early season (n=50 PA), shrinkage ≈ 0.85 (trust prior). Late season (n=400 PA), shrinkage ≈ 0.30 (trust performance).

---

### 1.2 Ensemble Projection System

**Problem:** Single projection sources have systematic biases.

**Solution:** Weighted ensemble with dynamic weight adjustment based on historical accuracy.

```python
@dataclass
class ProjectionSource:
    name: str  # "steamer", "zips", "thebat", "yahoo_ros", "mle_derived"
    projection: PlayerProjection
    historical_mae: float  # Mean absolute error last 3 years
    recency_weight: float  # How much to weight recent predictions
    last_updated: datetime

class EnsembleProjector:
    """
    Combines multiple projection sources with accuracy-weighted averaging.
    Weights update continuously based on prediction accuracy.
    """
    
    def __init__(self):
        self.source_weights: Dict[str, float] = {
            "steamer": 0.30,
            "zips": 0.25,
            "thebat": 0.25,
            "yahoo_ros": 0.15,
            "mle_derived": 0.05,
        }
        self.accuracy_history: Dict[str, List[float]] = defaultdict(list)
    
    def project(self, player: Player, sources: List[ProjectionSource]) -> EnsembleProjection:
        """
        Weighted average with inverse-MAE weighting.
        Sources with lower historical error get higher weight.
        """
        valid_sources = [s for s in sources if s.projection is not None]
        
        if not valid_sources:
            return self.fallback_projection(player)
        
        # Calculate dynamic weights based on historical accuracy
        total_inverse_mae = sum(1 / s.historical_mae for s in valid_sources)
        dynamic_weights = {
            s.name: (1 / s.historical_mae) / total_inverse_mae
            for s in valid_sources
        }
        
        # Weighted average of projections
        weighted_stats = {}
        for stat in ["hr", "rbi", "sb", "avg", "ops"]:
            weighted_stats[stat] = sum(
                dynamic_weights[s.name] * getattr(s.projection, stat)
                for s in valid_sources
            )
        
        # Ensemble uncertainty = weighted variance between sources
        projection_variance = sum(
            dynamic_weights[s.name] * (getattr(s.projection, stat) - weighted_stats[stat]) ** 2
            for s in valid_sources
        )
        
        return EnsembleProjection(
            stats=weighted_stats,
            variance=projection_variance,
            source_weights=dynamic_weights,
            confidence=self.calculate_confidence(valid_sources),
        )
    
    def update_accuracy(self, source_name: str, predicted: float, actual: float):
        """
        Online learning: update source accuracy after each projection matures.
        Uses exponential decay weighting (recent predictions matter more).
        """
        error = abs(predicted - actual)
        self.accuracy_history[source_name].append(error)
        
        # Exponential moving average of MAE
        alpha = 0.1  # Decay factor
        if len(self.accuracy_history[source_name]) > 1:
            old_mae = self.source_accuracy[source_name]
            new_mae = alpha * error + (1 - alpha) * old_mae
            self.source_accuracy[source_name] = new_mae
```

---

### 1.3 Markov Chain Monte Carlo (MCMC) for Weekly Projections

**Problem:** Single-point projections miss the distribution of possible outcomes.

**Solution:** MCMC simulation for full outcome distributions.

```python
class WeeklyMCMCSimulator:
    """
    Simulate thousands of possible weekly outcomes.
    Enables probabilistic statements: "70% chance we win HR category"
    """
    
    def __init__(self, n_sims: int = 10000):
        self.n_sims = n_sims
    
    def simulate_weekly_matchup(
        self,
        my_roster: List[Player],
        opponent_roster: List[Player],
        schedule: WeekSchedule,
    ) -> MatchupSimulation:
        """
        Run 10,000 simulations of the week's games.
        Account for:
        - Playing time uncertainty (platoon, rest days)
        - Performance variance (good/bad week scenarios)
        - Injury probability
        """
        results = []
        
        for _ in range(self.n_sims):
            sim_result = self._single_simulation(my_roster, opponent_roster, schedule)
            results.append(sim_result)
        
        # Calculate win probabilities per category
        category_win_probs = {}
        for cat in CATEGORIES:
            my_wins = sum(1 for r in results if r[cat]["me"] > r[cat]["opp"])
            category_win_probs[cat] = my_wins / self.n_sims
        
        # Calculate distribution of total category wins
        total_wins_dist = [sum(r[cat]["win"] for cat in CATEGORIES) for r in results]
        
        return MatchupSimulation(
            category_win_probabilities=category_win_probs,
            total_wins_distribution=total_wins_dist,
            expected_categories_won=np.mean(total_wins_dist),
            worst_case_scenario=np.percentile(total_wins_dist, 10),
            best_case_scenario=np.percentile(total_wins_dist, 90),
        )
    
    def _single_simulation(self, my_roster, opponent_roster, schedule):
        """
        One simulation iteration.
        Sample from each player's performance distribution.
        """
        my_stats = defaultdict(float)
        opp_stats = defaultdict(float)
        
        for player in my_roster:
            # Sample from player's performance distribution
            games_this_week = self._sample_games_played(player, schedule)
            for game in games_this_week:
                performance = self._sample_performance(player)
                for cat, val in performance.items():
                    my_stats[cat] += val
        
        # Same for opponent...
        
        return {"me": my_stats, "opp": opp_stats}
```

---

### 1.4 Reinforcement Learning for Roster Management

**Problem:** Static rules don't adapt to league dynamics.

**Solution:** RL agent learns optimal roster moves through simulation.

```python
class RosterManagementEnvironment:
    """
    Gym-like environment for RL training.
    State = current roster + free agents + matchup
    Action = roster moves
    Reward = matchup wins over season
    """
    
    def __init__(self, league_context: LeagueContext):
        self.league = league_context
        self.current_week = 1
        
    def reset(self):
        """Initialize with draft roster."""
        self.my_roster = self.league.get_my_draft_roster()
        self.free_agents = self.league.get_free_agents()
        self.record = (0, 0, 0)
        return self._get_state()
    
    def step(self, action: RosterMove) -> Tuple[State, float, bool]:
        """
        Execute roster move, simulate week, return reward.
        """
        # Execute the move
        if action.type == "ADD_DROP":
            self.my_roster.remove(action.drop_player)
            self.my_roster.add(action.add_player)
        
        # Simulate week's games
        opponent = self.league.get_weekly_opponent(self.current_week)
        result = self._simulate_matchup(self.my_roster, opponent)
        
        # Calculate reward (category wins)
        categories_won = sum(1 for cat, outcome in result.items() if outcome == "WIN")
        reward = categories_won  # Could weight by category scarcity
        
        # Update record
        if categories_won > len(CATEGORIES) / 2:
            self.record = (self.record[0] + 1, self.record[1], self.record[2])
        elif categories_won < len(CATEGORIES) / 2:
            self.record = (self.record[0], self.record[1] + 1, self.record[2])
        else:
            self.record = (self.record[0], self.record[1], self.record[2] + 1)
        
        self.current_week += 1
        done = self.current_week > 22  # End of season
        
        return self._get_state(), reward, done
    
    def _get_state(self) -> State:
        """
        Encode current situation as feature vector.
        """
        return State(
            roster_features=self._encode_roster(self.my_roster),
            matchup_features=self._encode_matchup(self.current_opponent),
            free_agent_features=self._encode_top_free_agents(self.free_agents, k=20),
            season_context=self._encode_season_context(self.record, self.current_week),
        )

class DQNAgent:
    """
    Deep Q-Network for learning optimal roster moves.
    Trained offline on historical league data + simulations.
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        self.q_network = self._build_network(state_dim, action_dim)
        self.target_network = self._build_network(state_dim, action_dim)
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
    def select_action(self, state: State, epsilon: float = 0.1) -> RosterMove:
        """
        Epsilon-greedy action selection.
        90% of time: choose best move according to Q-network
        10% of time: explore random move
        """
        if random.random() < epsilon:
            return self._random_valid_move(state)
        
        q_values = self.q_network.predict(state.to_tensor())
        best_action_idx = np.argmax(q_values)
        return self._idx_to_action(best_action_idx)
    
    def train(self, batch_size: int = 32):
        """
        Train on batch of experiences from replay buffer.
        """
        batch = self.replay_buffer.sample(batch_size)
        
        for state, action, reward, next_state, done in batch:
            # Q-learning update
            target_q = reward + (0.99 * np.max(self.target_network.predict(next_state)) if not done else 0)
            current_q = self.q_network.predict(state)[action]
            
            loss = (target_q - current_q) ** 2
            self.q_network.backpropagate(loss)
```

---

### 1.5 Contextual Bandits for Real-Time Decisions

**Problem:** RL is overkill for simple "add player X or Y?" decisions. Need fast, contextual recommendations.

**Solution:** Contextual bandits (simpler than full RL, adapts quickly).

```python
class LinUCBContextualBandit:
    """
    Linear Upper Confidence Bound algorithm.
    Balances exploration (trying new players) with exploitation (known good players).
    """
    
    def __init__(self, n_features: int, alpha: float = 1.0):
        self.alpha = alpha  # Exploration parameter
        self.A = {}  # Covariance matrices per arm (player)
        self.b = {}  # Reward vectors per arm
        
    def select_player(self, available_players: List[Player], context: Context) -> Player:
        """
        Select player with highest UCB score.
        """
        context_vector = self._featurize_context(context)
        
        best_player = None
        best_score = -np.inf
        
        for player in available_players:
            if player.id not in self.A:
                # New player: initialize
                self.A[player.id] = np.eye(len(context_vector))
                self.b[player.id] = np.zeros(len(context_vector))
            
            # Calculate UCB score
            A_inv = np.linalg.inv(self.A[player.id])
            theta = A_inv @ self.b[player.id]  # Estimated reward parameters
            
            expected_reward = theta @ context_vector
            uncertainty = self.alpha * np.sqrt(context_vector @ A_inv @ context_vector)
            ucb_score = expected_reward + uncertainty
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_player = player
        
        return best_player
    
    def update(self, player: Player, context: Context, reward: float):
        """
        Update after observing actual performance.
        """
        context_vector = self._featurize_context(context)
        self.A[player.id] += np.outer(context_vector, context_vector)
        self.b[player.id] += reward * context_vector
```

---

### 1.6 Graph Neural Networks for Lineup Optimization

**Problem:** Lineups have structure (batting order, handedness, park factors).

**Solution:** GNN that models player interactions.

```python
class LineupGNN(nn.Module):
    """
    Graph Neural Network for optimal daily lineup selection.
    Nodes = players, edges = lineup position relationships.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.conv1 = GATConv(in_channels=PLAYER_FEATURE_DIM, out_channels=hidden_dim)
        self.conv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)  # Should start? (binary)
        
    def forward(self, lineup_graph: Data):
        """
        lineup_graph contains:
        - x: player features (recent performance, matchup, handedness)
        - edge_index: lineup position connections (1→2→3...)
        - edge_attr: edge features (batting order position)
        """
        x = self.conv1(lineup_graph.x, lineup_graph.edge_index, lineup_graph.edge_attr)
        x = F.relu(x)
        x = self.conv2(x, lineup_graph.edge_index, lineup_graph.edge_attr)
        
        # Predict probability each player should start
        start_probs = torch.sigmoid(self.classifier(x))
        return start_probs
```

---

## Part 2: Multi-Agent Orchestration Architecture

### 2.1 Agent Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FANTASY BASEBALL ORCHESTRATION                           │
│                         (Master Controller)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                    DATA INGESTION LAYER                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Yahoo Agent  │  │ Statcast     │  │ FanGraphs    │              │  │
│  │  │ (Real-time)  │  │ Agent        │  │ Agent        │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                  PROJECTION ENGINE LAYER                            │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Ensemble     │  │ Bayesian     │  │ MLE          │              │  │
│  │  │ Agent        │  │ Update Agent │  │ Convert Agent│              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                 DECISION ENGINE LAYER                               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Weekly       │  │ Roster       │  │ Streamer     │              │  │
│  │  │ Strategy     │  │ Construction │  │ Optimization │              │  │
│  │  │ Agent        │  │ Agent (GNN)  │  │ Agent        │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                   EXECUTION LAYER                                   │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │  │
│  │  │ Waiver Wire  │  │ Lineup       │  │ Trade        │              │  │
│  │  │ Executor     │  │ Setter       │  │ Negotiator   │              │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘              │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Specifications

#### Agent: Yahoo Data Fetcher
**Owner:** OpenClaw (real-time execution)  
**Swimlane:** External API calls, rate limiting

```python
class YahooDataAgent:
    """
    Fetches real-time data from Yahoo Fantasy API.
    Handles rate limiting, caching, error retry.
    """
    
    responsibilities = [
        "Fetch roster data every 5 minutes during games",
        "Poll free agent list every 15 minutes",
        "Detect player status changes (active, bench, IL)",
        "Stream live scoring updates",
    ]
    
    def fetch_with_backoff(self, endpoint: str) -> Response:
        """Exponential backoff for rate limiting."""
        
    def cache_strategy(self) -> CachePolicy:
        """TTL-based caching with stale-while-revalidate."""
```

#### Agent: Statcast Analytics
**Owner:** Kimi CLI (heavy compute, research)  
**Swimlane:** Statcast data processing, trend detection

```python
class StatcastAnalyticsAgent:
    """
    Processes Statcast data for underlying skill metrics.
    Identifies buy-low/sell-high candidates via xStats divergence.
    """
    
    responsibilities = [
        "Calculate rolling xWOBA, xERA windows",
        "Detect swing/hard-hit rate changes",
        "Identify velocity drops (injury risk)",
        "Generate 'hidden gem' alerts",
    ]
    
    def detect_breakout_candidates(self, min_pa: int = 50) -> List[PlayerAlert]:
        """
        Find players with improved underlying skills not reflected in results.
        xSLG > SLG by >50 points = buy low opportunity.
        """
        
    def calculate_injury_risk_score(self, player: Player) -> float:
        """
        Based on velocity drops, spin rate changes, usage patterns.
        """
```

#### Agent: Projection Ensemble
**Owner:** Claude Code (mathematical rigor)  
**Swimlane:** Algorithm implementation, statistical modeling

```python
class ProjectionEnsembleAgent:
    """
    Maintains ensemble projections with dynamic weighting.
    Runs Bayesian updates as new data arrives.
    """
    
    responsibilities = [
        "Weight multiple projection sources optimally",
        "Apply Bayesian shrinkage to noisy early-season data",
        "Generate confidence intervals for all projections",
        "Identify projection-market inefficiencies",
    ]
    
    def run_bayesian_update(self, player: Player, new_games: List[Game]):
        """Update posterior distribution with new evidence."""
        
    def detect_projection_divergence(self) -> List[MarketInefficiency]:
        """Where do we disagree with Yahoo ownership %?"""
```

#### Agent: Weekly Strategy Planner
**Owner:** Claude Code (strategy, game theory)  
**Swimlane:** Matchup analysis, category prioritization

```python
class WeeklyStrategyAgent:
    """
    Determines which categories to target/punt each week.
    Uses game theory to exploit opponent weaknesses.
    """
    
    responsibilities = [
        "Analyze opponent roster strengths/weaknesses",
        "Determine optimal category allocation",
        "Identify 'stealable' categories",
        "Plan streaming SP schedule",
    ]
    
    def develop_weekly_battle_plan(self, opponent: Team) -> WeeklyStrategy:
        """
        Returns: Focus categories, ignore categories, streamer targets.
        """
        
    def simulate_scenarios(self, n_sims: int = 10000) -> ScenarioAnalysis:
        """
        Monte Carlo simulation of week outcomes.
        """
```

#### Agent: Roster Construction (Portfolio Optimizer)
**Owner:** Claude Code (optimization, quant methods)  
**Swimlane:** Mean-variance optimization, constraint solving

```python
class RosterConstructionAgent:
    """
    Portfolio theory applied to roster construction.
    Optimizes for return/variance tradeoff.
    """
    
    responsibilities = [
        "Calculate covariance matrix for all players",
        "Optimize roster for mean-variance efficiency",
        "Suggest adds/drops that improve Sharpe ratio",
        "Manage injury risk through diversification",
    ]
    
    def optimize_roster(self, risk_tolerance: float) -> RosterOptimization:
        """
        Quadratic programming solution for optimal roster.
        """
        
    def calculate_roster_sharpe_ratio(self) -> float:
        """
        Excess return per unit of variance.
        Higher is better.
        """
```

#### Agent: Streamer Selector
**Owner:** OpenClaw (fast execution, simple rules)  
**Swimlane:** Two-start SPs, matchup analysis, quick decisions

```python
class StreamerSelectorAgent:
    """
    Identifies optimal weekly streamers.
    Considers: opponent offense, park factors, two-start weeks.
    """
    
    responsibilities = [
        "Identify two-start SPs each week",
        "Rank streamers by matchup quality",
        "Monitor weather/postponement risk",
        "Execute add/drop for streaming",
    ]
    
    def rank_weekly_streamers(self, position: str) -> List[PlayerRank]:
        """
        For SP: two-start value, opponent OPS, park factors.
        For hitters: platoon advantage, opposing SP handedness.
        """
```

### 2.3 Orchestration Protocol

```yaml
# .clawhub/fantasy-baseball-orchestration.yaml

workflow:
  name: "Weekly Fantasy Management"
  trigger: "cron(0 6 * * 1)"  # Monday 6am
  
  phases:
    - name: "Data Refresh"
      agents: [yahoo, statcast]
      parallel: true
      timeout: 300
      
    - name: "Projection Update"
      agents: [ensemble]
      depends_on: ["Data Refresh"]
      timeout: 120
      
    - name: "Strategy Planning"
      agents: [weekly_strategy]
      depends_on: ["Projection Update"]
      timeout: 60
      
    - name: "Roster Optimization"
      agents: [roster_construction, streamer_selector]
      depends_on: ["Strategy Planning"]
      parallel: true
      timeout: 180
      
    - name: "Decision Review"
      agents: [kimi_validator]  # High-stakes check
      depends_on: ["Roster Optimization"]
      condition: "recommended_moves.value > 50_units"
      timeout: 300
      
    - name: "Execution"
      agents: [waiver_executor, lineup_setter]
      depends_on: ["Decision Review"]
      auto_execute: false  # Require human approval
      timeout: 60

escalation:
  - condition: "injury_detected.priority == 'high'"
    action: "immediate_notification"
    agents: [all]
    
  - condition: "breakout_candidate.confidence > 0.85"
    action: "priority_alert"
    agents: [kimi_validator, claude_code]
```

### 2.4 Agent Communication Protocol

```python
@dataclass
class AgentMessage:
    """
    Standardized inter-agent communication.
    """
    message_id: UUID
    sender: str  # Agent name
    recipients: List[str]
    message_type: Literal["DATA", "ALERT", "REQUEST", "RESPONSE"]
    payload: dict
    priority: int  # 1-5, 5 = highest
    timestamp: datetime
    requires_ack: bool = False

class MessageBus:
    """
    Pub/sub message bus for agent coordination.
    """
    
    def publish(self, message: AgentMessage):
        """Broadcast to all subscribers."""
        
    def request(self, recipient: str, request: dict, timeout: float) -> Response:
        """Synchronous request-response pattern."""
```

---

## Part 3: Implementation Phases

### Phase 1: Foundation (Weeks 1-2)

| Component | Owner | Deliverable |
|-----------|-------|-------------|
| Universal projection system | Claude | `get_or_create_projection()` with Yahoo ROS fallback |
| Basic roster recommendations | Claude | `generate_roster_recommendations()` v1 |
| Yahoo data enhancement | OpenClaw | `percent_owned` extraction, batch player stats |
| MCMC weekly simulator | Claude | 10k sim matchup engine |

### Phase 2: Intelligence (Weeks 3-4)

| Component | Owner | Deliverable |
|-----------|-------|-------------|
| Bayesian updater | Claude | Posterior projection updating |
| Ensemble projector | Claude | Weighted multi-source projections |
| Statcast integration | Kimi | Trend detection, buy-low alerts |
| Contextual bandit | Claude | LinUCB for add/drop decisions |

### Phase 3: Optimization (Weeks 5-6)

| Component | Owner | Deliverable |
|-----------|-------|-------------|
| Portfolio optimizer | Claude | Mean-variance roster optimization |
| Weekly strategy engine | Claude | Category allocation, punt analysis |
| GNN lineup setter | Claude | Optimal daily lineup selection |
| Multi-agent orchestration | OpenClaw | Agent coordination, message bus |

### Phase 4: Automation (Weeks 7-8)

| Component | Owner | Deliverable |
|-----------|-------|-------------|
| RL agent training | Claude | DQN for roster management |
| Auto-execution | OpenClaw | Low-risk move automation |
| Alert system | OpenClaw | Real-time opportunity notifications |
| Performance analytics | Kimi | Backtesting, accuracy measurement |

---

## Part 4: Success Metrics & Monitoring

### Model Performance Metrics

```python
@dataclass
class SystemPerformance:
    """
    Track effectiveness of recommendations.
    """
    # Projection accuracy
    projection_mae: float  # Mean absolute error
    projection_bias: float  # Systematic over/under
    projection_calibration: float  # Predicted vs actual quantiles
    
    # Recommendation quality
    add_success_rate: float  # % of recommended adds that outperform drops
    streamer_hit_rate: float  # % of streamers that win their week
    category_win_rate: float  # % of categories won vs expected
    
    # Decision value
    avg_z_score_added: float  # Value captured per transaction
    opportunity_cost: float  # Value missed by not acting
    
    # System health
    recommendation_latency_ms: float  # Time to generate advice
    projection_coverage_pct: float  # % of players with projections
    data_freshness_minutes: float  # Age of latest data
```

### Alert Thresholds

```yaml
alerts:
  - name: "Projection Staleness"
    condition: "data_freshness > 60_minutes"
    severity: warning
    
  - name: "High Confidence Opportunity"
    condition: "edge_score > 2.0 AND confidence > 0.9"
    severity: info
    action: notify_immediate
    
  - name: "Injury Risk Detected"
    condition: "injury_risk_score > 0.7"
    severity: critical
    action: suggest_alternative
    
  - name: "Model Drift"
    condition: "projection_mae > 0.5_stds_above_baseline"
    severity: warning
    action: trigger_retraining
```

---

## Part 5: Document Evolution

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-03-23 | Initial roadmap, universal projections, actionable recommendations | Kimi CLI |
| 2.0 | 2026-03-23 | Added MCMC, RL, GNN, multi-agent architecture, Bayesian updating | Kimi CLI |
| 2.1 | TBD | Implementation notes, API contracts, test specifications | Claude Code |
| 2.2 | TBD | Performance benchmarks, tuning parameters | Kimi CLI |

---

## Appendix A: Algorithm Cheat Sheet

| Problem | Algorithm | Complexity | When to Use |
|---------|-----------|------------|-------------|
| Projection updating | Bayesian conjugate update | O(1) | Every game played |
| Multi-source projections | Inverse-MAE weighted ensemble | O(n) | Daily refresh |
| Weekly outcome distribution | MCMC (Gibbs sampling) | O(k×n) | Pre-matchup planning |
| Roster optimization | Mean-variance quadratic program | O(n³) | Weekly/daily |
| Real-time decisions | Contextual bandit (LinUCB) | O(d²) | Waiver wire, streaming |
| Long-term strategy | Deep Q-Network (DQN) | O(network) | Season-long learning |
| Lineup optimization | Graph Neural Network | O(E×d) | Daily lineups |

---

**Next Action:** Claude Code to review, prioritize Phase 1 implementation, create API contracts, and begin development of `get_or_create_projection()` and MCMC simulator.
