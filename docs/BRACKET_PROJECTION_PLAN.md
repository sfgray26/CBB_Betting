# NCAA Tournament Bracket Projection System

**Author:** Kimi CLI (Quant + CBB Expert Mode)  
**Date:** March 12, 2026  
**Document:** BRACKET-001  
**Status:** Implementation Plan  

---

## Executive Summary

Once the Selection Committee releases the bracket (Selection Sunday, March 16, ~6 PM ET), we have ~48 hours to generate comprehensive tournament projections before the First Four tips off (March 18). 

This plan outlines a **quantitative bracket simulation system** that:
1. Simulates the tournament 50,000+ times using CBB Edge V9.1
2. Generates win probabilities for every possible matchup
3. Identifies value bets vs. market futures
4. Produces Cinderella/upset probability rankings
5. Optimizes bracket pool strategy (EV maximization)

---

## 1. Data Requirements (Post-Selection Sunday)

### 1.1 Seed & Bracket Structure (Manual Input)
```python
bracket_2026 = {
    "south": {
        "region_seed_1": "Auburn",      # 1-seed
        "region_seed_2": "Michigan St", # 2-seed
        # ... 1-16 seeds
        "first_four": ["Team A", "Team B"]  # Play-in games
    },
    "east": {...},
    "west": {...},
    "midwest": {...}
}
```

### 1.2 Team Ratings Database (Already Available)
- **KenPom**: Adjusted efficiency margins (existing)
- **BartTorvik**: T-Rank ratings (existing)
- **CBB Edge Composite**: V9.1 model ratings (existing)
- **Conference HCA**: Round-specific adjustments (K-10 implemented)

### 1.3 Market Data (To Collect)
- **Futures odds**: Champion, Final Four, Elite Eight (by Sunday night)
- **Game lines**: First Four lines (Tuesday AM), R64 lines (Tuesday PM)
- **Seed history**: Historical upset rates by seed differential

### 1.4 Team-Specific Factors (Automated Lookups)
```python
team_profile = {
    "team_name": "Duke",
    "kp_adj_em": 25.3,           # From existing DB
    "bt_rating": 18.7,           # From existing DB
    "v9_composite": 22.1,        # CBB Edge rating
    "conference": "ACC",         # For HCA calc
    "pace": 68.5,                # Possessions/game
    "three_pt_rate": 0.42,       # 3PA/FGA
    "def_style": "pressure",     # For matchup sim
    "injuries": [...],           # From OpenClaw
    "recent_form": "W5",         # Last 5 games
    "tournament_exp": 0.75,      # Returner minutes %
}
```

---

## 2. Modeling Architecture

### 2.1 Game-Level Prediction Engine

**Base Model**: CBB Edge V9.1 (already implemented)
```python
def predict_tournament_game(team_a, team_b, round_num, is_neutral=True):
    """
    Predict single tournament game outcome.
    
    Args:
        team_a: TeamProfile object
        team_b: TeamProfile object  
        round_num: 0=First Four, 1=R64, 2=R32, 3=S16, 4=E8, 5=F4, 6=Champ
        is_neutral: True for all tournament games (except Dayton First Four)
    
    Returns:
        win_prob_a: Probability team A wins (0-1)
        margin_pred: Predicted margin (team A perspective)
        confidence: Model confidence (0-1)
    """
    # 1. Base ratings differential
    rating_diff = team_a.v9_rating - team_b.v9_rating
    
    # 2. Tournament-specific adjustments (NEW)
    # Round-specific variance (higher in early rounds)
    round_variance_multiplier = {
        0: 1.15,  # First Four (play-in chaos)
        1: 1.12,  # R64 (upset city)
        2: 1.08,  # R32 (still volatile)
        3: 1.05,  # S16
        4: 1.02,  # E8
        5: 1.00,  # F4
        6: 1.00,  # Championship
    }.get(round_num, 1.0)
    
    # 3. Fatigue/Rest (K-8 implemented)
    fatigue_adj = calculate_fatigue_advantage(team_a, team_b)
    
    # 4. Conference HCA (K-10 implemented, neutral = 0)
    hca_adj = 0 if is_neutral else get_conference_hca_delta(team_a.conference)
    
    # 5. Style matchup factors (NEW)
    style_adj = calculate_style_matchup(team_a, team_b)
    # e.g., 3pt-heavy teams vs. teams that close out poorly
    
    # 6. Tournament experience (NEW)
    exp_adj = (team_a.tournament_exp - team_b.tournament_exp) * 0.5  # Small effect
    
    # Combined prediction
    total_diff = rating_diff + fatigue_adj + hca_adj + style_adj + exp_adj
    
    # Convert to win probability (logistic function)
    # sd_scaled for round variance
    base_sd = 11.0 * round_variance_multiplier  # V9 base SD
    win_prob_a = 1 / (1 + math.exp(-total_diff / base_sd))
    
    return win_prob_a, total_diff, base_sd
```

### 2.2 Style Matchup Engine (NEW COMPONENT)
```python
def calculate_style_matchup(offense, defense):
    """
    Calculate stylistic advantages beyond raw ratings.
    
    Examples:
    - 3pt-heavy offense vs. poor 3pt defense = +edge
    - Slow tempo team vs. fast-break team = neutralization
    - Press defense vs. inexperienced guards = +edge
    """
    edges = []
    
    # 3pt shooting vs. 3pt defense
    if offense.three_pt_pct > 0.37 and defense.opp_three_pt_pct > 0.34:
        edges.append(("3pt_advantage", +0.8))
    
    # Tempo mismatch
    tempo_diff = abs(offense.pace - defense.pace)
    if tempo_diff > 6:
        # Slower team forces their pace
        if offense.pace < defense.pace:
            edges.append(("tempo_control", +0.5))
    
    # Rebounding battle
    reb_diff = offense.orb_pct - defense.drb_pct
    if reb_diff > 5:
        edges.append(("offensive_rebounding", +0.6))
    
    # Turnover generation vs. protection
    to_diff = defense.tov_pct_forced - offense.tov_pct
    if to_diff > 3:
        edges.append(("turnover_pressure", +0.7))
    
    return sum(e[1] for e in edges)
```

### 2.3 Historical Seed-Based Priors (NEW COMPONENT)
```python
SEED_UPSET_RATES = {
    # Historical upset rates by seed differential (2000-2024)
    # Key: (higher_seed, lower_seed) -> upset_prob
    (1, 16): 0.013,   # 1-seed vs 16-seed (1.3% upset rate)
    (2, 15): 0.067,   # 6.7%
    (3, 14): 0.153,   # 15.3%
    (4, 13): 0.216,   # 21.6%
    (5, 12): 0.352,   # 35.2% - famous 12-5 upsets
    (6, 11): 0.389,   # 38.9%
    (7, 10): 0.394,   # 39.4%
    (8, 9):  0.487,   # 48.7% - essentially coin flip
}

def blend_model_with_seed_history(model_prob, seed_a, seed_b):
    """
    Blend V9.1 prediction with historical seed upset rates.
    
    Weight: 70% model, 30% historical (early rounds only)
    """
    if seed_a < seed_b:
        hist_prob = 1 - SEED_UPSET_RATES.get((seed_a, seed_b), 0.5)
    else:
        hist_prob = SEED_UPSET_RATES.get((seed_b, seed_a), 0.5)
    
    # Weight more toward history for extreme mismatches (1v16, 2v15)
    seed_diff = abs(seed_a - seed_b)
    if seed_diff >= 13:
        weight_model = 0.6
    elif seed_diff >= 10:
        weight_model = 0.7
    else:
        weight_model = 0.8
    
    return weight_model * model_prob + (1 - weight_model) * hist_prob
```

---

## 3. Monte Carlo Simulation Engine

### 3.1 Full Tournament Simulation
```python
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import concurrent.futures

@dataclass
class TournamentResult:
    """Results from one tournament simulation."""
    champion: str
    final_four: List[str]
    runner_up: str
    championship_margin: float
    upsets: List[Tuple[str, str, int]]  # (winner, loser, round)
    cinderella: str  # Lowest seed to S16

def simulate_single_tournament(bracket, random_seed=None) -> TournamentResult:
    """
    Simulate one full tournament run.
    
    Args:
        bracket: Full bracket structure with team ratings
        random_seed: For reproducibility
    
    Returns:
        TournamentResult with full outcome
    """
    if random_seed:
        np.random.seed(random_seed)
    
    # Track teams through bracket
    current_round = bracket.copy()
    final_four = []
    upsets = []
    cinderella_seed = 99
    
    # Simulate each round
    for round_num in range(7):  # 0=First Four, 1=R64, ..., 6=Championship
        next_round = {}
        
        for region, teams in current_round.items():
            if len(teams) == 1:
                # Winner of region
                final_four.append(teams[0])
                continue
            
            # Pair teams (1v16, 8v9, etc.)
            matchups = pair_teams(teams)
            winners = []
            
            for team_a, team_b in matchups:
                win_prob = predict_tournament_game(team_a, team_b, round_num)
                
                # Simulate outcome
                if np.random.random() < win_prob:
                    winner, loser = team_a, team_b
                else:
                    winner, loser = team_b, team_a
                
                winners.append(winner)
                
                # Track upsets (lower seed beating higher seed)
                if winner.seed > loser.seed:
                    upsets.append((winner.name, loser.name, round_num))
                    if round_num >= 3 and winner.seed < cinderella_seed:
                        cinderella_seed = winner.seed
                        cinderella = winner.name
            
            next_round[region] = winners
        
        current_round = next_round
    
    # Championship game
    if len(final_four) == 4:
        # Simulate F4 matchups (S1 vs S2, E1 vs E2)
        # Then championship
        ff_winners = simulate_final_four(final_four)
        champion, runner_up, margin = simulate_championship(ff_winners)
    else:
        champion = current_round.get("champion", [None])[0]
        runner_up = None
        margin = 0
    
    return TournamentResult(
        champion=champion,
        final_four=[t.name for t in final_four],
        runner_up=runner_up.name if runner_up else None,
        championship_margin=margin,
        upsets=upsets,
        cinderella=cinderella if cinderella_seed < 99 else None
    )

def run_monte_carlo(bracket, n_sims=50000, n_workers=8) -> Dict:
    """
    Run full Monte Carlo simulation.
    
    Args:
        bracket: Complete bracket with team data
        n_sims: Number of simulations (default 50k)
        n_workers: Parallel workers
    
    Returns:
        Aggregated results with probabilities
    """
    results = {
        "championship": defaultdict(int),
        "final_four": defaultdict(int),
        "elite_eight": defaultdict(int),
        "sweet_sixteen": defaultdict(int),
        "cinderella_prob": defaultdict(float),
        "upset_counts": [],
        "avg_championship_margin": [],
    }
    
    def simulate_batch(batch_size):
        batch_results = []
        for _ in range(batch_size):
            result = simulate_single_tournament(bracket)
            batch_results.append(result)
        return batch_results
    
    # Parallel simulation
    batch_size = n_sims // n_workers
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(simulate_batch, batch_size) 
                   for _ in range(n_workers)]
        
        for future in concurrent.futures.as_completed(futures):
            batch = future.result()
            for result in batch:
                # Aggregate results
                results["championship"][result.champion] += 1
                for team in result.final_four:
                    results["final_four"][team] += 1
                results["upset_counts"].append(len(result.upsets))
                results["avg_championship_margin"].append(result.championship_margin)
    
    # Convert to probabilities
    for key in ["championship", "final_four", "elite_eight", "sweet_sixteen"]:
        for team in results[key]:
            results[key][team] /= n_sims
    
    return results
```

### 3.2 Output: Win Probability Matrix
```python
def generate_win_probability_matrix(all_teams, round_num) -> pd.DataFrame:
    """
    Generate head-to-head win probability matrix for a given round.
    
    Used for:
    - Futures betting (what's the chance X makes Final Four?)
    - Bracket pool optimization
    - Cinderella identification
    """
    n_teams = len(all_teams)
    matrix = np.zeros((n_teams, n_teams))
    
    for i, team_a in enumerate(all_teams):
        for j, team_b in enumerate(all_teams):
            if i != j:
                win_prob, _, _ = predict_tournament_game(team_a, team_b, round_num)
                matrix[i, j] = win_prob
    
    return pd.DataFrame(matrix, 
                       index=[t.name for t in all_teams],
                       columns=[t.name for t in all_teams])
```

---

## 4. Betting & Futures Analysis

### 4.1 Futures Value Detection
```python
def analyze_futures_value(sim_results, market_odds):
    """
    Identify value in championship and Final Four futures.
    
    Args:
        sim_results: Output from run_monte_carlo()
        market_odds: Dict of {team: american_odds}
    
    Returns:
        List of value bets with EV
    """
    value_bets = []
    
    for team, prob in sim_results["championship"].items():
        if team not in market_odds:
            continue
        
        fair_odds = probability_to_american_odds(prob)
        market_price = market_odds[team]
        
        # Calculate EV
        ev = calculate_ev(prob, market_price)
        
        if ev > 0.1:  # 10%+ edge
            value_bets.append({
                "team": team,
                "market": market_price,
                "fair": fair_odds,
                "model_prob": prob,
                "ev_pct": ev * 100,
                "recommendation": "BET" if ev > 0.15 else "CONSIDER"
            })
    
    return sorted(value_bets, key=lambda x: x["ev_pct"], reverse=True)

def calculate_ev(prob, american_odds):
    """Calculate expected value of a bet."""
    if american_odds > 0:
        profit = american_odds / 100
        stake = 1
    else:
        profit = 1
        stake = abs(american_odds) / 100
    
    win_amount = profit * stake
    lose_amount = -stake
    
    ev = (prob * win_amount) + ((1 - prob) * lose_amount)
    return ev / stake  # Return as percentage of stake
```

### 4.2 Round-by-Round Survivor Probabilities
```python
def generate_survival_probs(bracket, n_sims=10000) -> pd.DataFrame:
    """
    Probability of each team surviving each round.
    
    Returns DataFrame: teams x rounds with survival probability
    """
    rounds = ["R64", "R32", "S16", "E8", "F4", "Champ"]
    survival_counts = defaultdict(lambda: defaultdict(int))
    
    for _ in range(n_sims):
        result = simulate_single_tournament(bracket)
        
        # Track how far each team went
        for round_idx, teams in enumerate(get_round_results(result)):
            for team in teams:
                survival_counts[team][round_idx] += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(index=list(survival_counts.keys()), columns=rounds)
    for team in survival_counts:
        for round_idx, count in survival_counts[team].items():
            df.loc[team, rounds[round_idx]] = count / n_sims
    
    return df
```

---

## 5. Bracket Pool Strategy

### 5.1 ESPN/Yahoo Pool Optimization
```python
def optimize_bracket_pool(sim_results, pool_size_estimate=100):
    """
    Generate optimal bracket for pool play.
    
    Strategy: Pick favorites in early rounds (high probability),
    differentiate in later rounds (high payout)
    
    Args:
        sim_results: Monte Carlo results
        pool_size_estimate: Expected number of entries
    
    Returns:
        Optimized bracket picks
    """
    bracket = {}
    
    # R64 & R32: Pick mostly favorites (survival mode)
    # Exceptions: Pick 1-2 calculated upsets where public overvalues favorite
    
    # S16 & beyond: Pick based on EV vs. public picking rates
    # If public picks Team A 70% but model says 55%, pick Team B
    
    return bracket
```

### 5.2 Contrarian Picks Identification
```python
def identify_contrarian_picks(sim_results, public_picking_rates):
    """
    Find picks where model and public diverge significantly.
    
    High-value contrarian opportunities:
    - 12-seed over 5-seed (public picks 5, model favors 12)
    - Strong mid-majors (public underrates)
    - Injured favorites (public overvalues name brand)
    """
    contrarian = []
    
    for matchup, public_pct in public_picking_rates.items():
        team_a, team_b = matchup
        model_prob_a = sim_results["head_to_head"][(team_a, team_b)]
        
        # If model says 45% win but public only picks them 20%
        if abs(model_prob_a - public_pct) > 0.20:
            contrarian.append({
                "matchup": matchup,
                "model_prob": model_prob_a,
                "public_pct": public_pct,
                "edge": model_prob_a - public_pct,
                "recommendation": team_a if model_prob_a > public_pct else team_b
            })
    
    return contrarian
```

---

## 6. Cinderella & Upset Prediction

### 6.1 Cinderella Probability Rankings
```python
def cinderella_probabilities(sim_results, threshold_seed=11):
    """
    Probability of each double-digit seed reaching Sweet 16 or beyond.
    
    Returns teams sorted by "Cinderella score" (P(S16) * seed multiplier)
    """
    cinderellas = []
    
    for team in sim_results["sweet_sixteen"]:
        seed = get_team_seed(team)
        if seed >= threshold_seed:
            s16_prob = sim_results["sweet_sixteen"][team]
            e8_prob = sim_results["elite_eight"].get(team, 0)
            
            # Cinderella score = seed * P(S16) * 10
            # Higher seed = more impressive = higher score
            score = seed * s16_prob * 10
            
            cinderellas.append({
                "team": team,
                "seed": seed,
                "p_sweet16": s16_prob,
                "p_elite8": e8_prob,
                "cinderella_score": score
            })
    
    return sorted(cinderellas, key=lambda x: x["cinderella_score"], reverse=True)
```

### 6.2 Upset Heat Map
```python
def generate_upset_heatmap(bracket) -> pd.DataFrame:
    """
    Heat map of upset probability for each R64/R32 matchup.
    
    Color code: Red = high upset probability, Green = safe favorite
    """
    matchups = get_all_r64_matchups(bracket)
    upset_probs = []
    
    for favorite, underdog in matchups:
        win_prob = predict_tournament_game(favorite, underdog, round_num=1)
        upset_prob = 1 - win_prob
        
        upset_probs.append({
            "matchup": f"{favorite.name} ({favorite.seed}) vs {underdog.name} ({underdog.seed})",
            "upset_probability": upset_prob,
            "risk_level": "HIGH" if upset_prob > 0.35 else "MED" if upset_prob > 0.20 else "LOW"
        })
    
    return pd.DataFrame(upset_probs).sort_values("upset_probability", ascending=False)
```

---

## 7. Implementation Timeline

### Selection Sunday (March 16)

**6:00 PM ET** — Bracket Revealed
- [ ] Manually input bracket structure (30 min)
- [ ] Pull latest team ratings from DB (automated)
- [ ] Collect opening futures odds from sportsbooks (manual)

**8:00 PM ET** — Initial Simulations
- [ ] Run 10,000 simulation batch (15 min)
- [ ] Generate first-cut win probabilities
- [ ] Identify obvious value bets

**10:00 PM ET** — Full Analysis
- [ ] Run 50,000 simulation batch (1 hour)
- [ ] Generate all reports (futures, cinderellas, upset heat map)
- [ ] Post initial analysis to Discord #cbb-tournament

### Monday (March 17)

**9:00 AM ET** — Morning Update
- [ ] Re-run sims with updated injury info from OpenClaw
- [ ] Post morning brief with First Four value plays

**6:00 PM ET** — First Four Preview
- [ ] Game-by-game breakdown of play-in games
- [ ] Final Four futures value update

### Tuesday (March 18) — First Four

**7:00 PM ET** — First Four Begins
- [ ] Live monitoring (if live betting enabled)
- [ ] Update bracket projections after each play-in game

---

## 8. Output Files

### 8.1 Generated Reports

| File | Description | Format |
|------|-------------|--------|
| `bracket_projections_2026.json` | Full simulation results | JSON |
| `championship_probs.csv` | Championship probabilities | CSV |
| `final_four_probs.csv` | Final Four probabilities | CSV |
| `futures_value_plays.csv` | Value bets vs. market | CSV |
| `cinderella_rankings.csv` | Double-digit seed S16 probabilities | CSV |
| `upset_heatmap_r64.csv` | First round upset probabilities | CSV |
| `bracket_pool_optimal.pdf` | Visual bracket recommendation | PDF |

### 8.2 Discord Delivery

```python
async def post_bracket_analysis():
    """Post bracket analysis to Discord."""
    
    # 1. Championship probabilities
    await send_to_channel("cbb-tournament", embed=champ_probs_embed())
    
    # 2. Cinderella watch
    await send_to_channel("cbb-tournament", embed=cinderella_embed())
    
    # 3. Value futures
    await send_to_channel("openclaw-briefs", embed=futures_value_embed())
    
    # 4. Upset alert (high-probability upsets only)
    high_upsets = get_high_probability_upsets(threshold=0.30)
    if high_upsets:
        await send_to_channel("cbb-alerts", embed=upset_alert_embed(high_upsets))
```

---

## 9. Technical Implementation

### 9.1 New Files to Create

```
backend/tournament/
├── __init__.py
├── bracket_simulator.py      # Monte Carlo engine
├── matchup_predictor.py      # Game-level predictions
├── futures_analyzer.py       # Value bet identification
├── cinderella_tracker.py     # Upset/Cinderella detection
└── pool_optimizer.py         # Bracket pool strategy

scripts/
├── run_bracket_sims.py       # CLI for running simulations
├── generate_bracket_report.py # Report generation

outputs/
└── tournament_2026/          # Generated reports
    ├── sim_results.json
    ├── *.csv
    └── *.pdf
```

### 9.2 Dependencies

```python
# requirements.txt additions
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0  # For visualizations
seaborn>=0.12.0    # For heat maps
fpdf2>=2.7.0       # For PDF reports
```

### 9.3 Performance Targets

- **10,000 sims**: <2 minutes (single-threaded)
- **50,000 sims**: <5 minutes (8 workers)
- **Memory**: <2GB for full simulation
- **Storage**: <50MB per tournament run

---

## 10. Key Metrics to Track

### Model Performance
- Brier score on game predictions
- Calibration of win probabilities
- ROI on futures value bets
- Bracket pool performance (top 10% finish rate)

### Post-Tournament Analysis
- Which upsets did we predict?
- Which favorites disappointed?
- Model accuracy by round
- Futures value bet hit rate

---

## Summary

This bracket projection system will:

1. **Simulate** 50,000+ tournaments using CBB Edge V9.1
2. **Identify** value in futures markets
3. **Predict** Cinderella runs and upsets
4. **Optimize** bracket pool entries
5. **Deliver** actionable intelligence to Discord

**Next Steps:**
1. Implement `bracket_simulator.py` core engine
2. Create Selection Sunday data input pipeline
3. Build report generation system
4. Test with 2025 historical bracket
5. Deploy for 2026 tournament

**Expected Output:** Best-in-class tournament projections with quantified uncertainty and actionable betting edges.
