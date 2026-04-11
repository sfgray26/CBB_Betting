# VORP Implementation Guide for Fantasy Baseball

> **Research Date:** April 11, 2026  
> **Researcher:** Kimi CLI (Deep Intelligence)  
> **Task ID:** K-38  
> **Status:** COMPLETE

---

## 1. Executive Summary

This guide provides a comprehensive methodology for implementing **Value Over Replacement Player (VORP)** in fantasy baseball contexts. VORP was invented by Keith Woolner (Baseball Prospectus) and measures a player's contribution above what a "replacement level" player at the same position would provide.

### Key Findings

| Concept | Finding |
|---------|---------|
| **VORP Definition** | Player's value minus replacement level value at same position |
| **Replacement Level** | ~80% of positional average for most positions |
| **Position Adjustment** | Catchers: 75%, 1B/DH: 85%, All others: 80% |
| **Fantasy Application** | Sum of Z-scores above replacement Z-score |
| **Unit** | Runs (real baseball) or Z-score points (fantasy) |

### Quick Formula

```
Fantasy VORP = Player's Total Z-Score - Replacement Level Z-Score (by position)
```

---

## 2. VORP Fundamentals

### 2.1 Definition

> "VORP stands for Value Over Replacement Player, which is broken down as the number of runs contributed beyond what a replacement-level player at the same position would contribute if given the same percentage of team plate appearances."
> — Baseball Prospectus

**Key Insight:** VORP captures **marginal value** — the advantage a player provides over the best available alternative on the waiver wire.

### 2.2 Real Baseball vs Fantasy VORP

| Aspect | Real Baseball VORP | Fantasy VORP |
|--------|-------------------|--------------|
| **Unit** | Runs created/prevented | Z-score points |
| **Basis** | Runs Created (RC) or Linear Weights | Category Z-scores |
| **Replacement Definition** | Freely available talent (~80% of avg) | Worst starter at position |
| **Positions** | Actual defensive position | Fantasy position eligibility |
| **Defense** | Not included in VORP | Not applicable |

### 2.3 Why Use VORP in Fantasy?

1. **Position Scarcity Adjustment** - Properly values scarce positions (C, 2B, SS)
2. **Fair Comparison** - Compares players across positions fairly
3. **Draft Strategy** - Identifies true position scarcity vs. perceived scarcity
4. **Trade Evaluation** - Objective measure for trade decisions

---

## 3. Replacement Level Definition

### 3.1 Woolner's Original Definition

> "A replacement player is roughly 80 percent as good as an average major league hitter at his position."
> — Keith Woolner

### 3.2 Position-Specific Replacement Levels

From Baseball Prospectus/TangoTiger research:

| Position | Replacement Level | Rationale |
|----------|------------------|-----------|
| **Catcher** | 75% of positional average | Defensive demands, fewer PA |
| **First Base** | 85% of positional average | Offensive position, deep pool |
| **Designated Hitter** | 85% of positional average | Offense-only |
| **Second Base** | 80% of positional average | Standard scarcity |
| **Third Base** | 80% of positional average | Standard scarcity |
| **Shortstop** | 80% of positional average | Standard scarcity |
| **Outfield** | 80% of positional average | Standard scarcity |

### 3.3 Calculating Replacement Level Z-Scores

**Step 1:** Calculate total Z-scores for all players globally  
**Step 2:** Sort by position  
**Step 3:** Identify the Nth player at each position (where N = number of starters in your league)  
**Step 4:** That player's Z-score = Replacement Level Z-score

**Example (12-team league, 1 catcher):**
- 12 catchers drafted
- 12th best catcher's Z-score = -6.0
- Replacement level Z = -6.0
- All catchers get +6.0 adjustment

---

## 4. Mathematical Formulation

### 4.1 Fantasy VORP Formula

```
VORP_player = (Z_player - Z_replacement_position) × Playing_Time_Factor

Where:
- Z_player = Player's total Z-score across all categories
- Z_replacement_position = Z-score of replacement level at player's position
- Playing_Time_Factor = (Player's projected PA or IP) / (League average PA/IP)
```

### 4.2 Simplified Formula (Constant Playing Time)

```
VORP = Z_player - Z_replacement

Example:
Player X (Catcher): Z = +0.5, Replacement Z = -6.0
VORP = 0.5 - (-6.0) = +6.5

Player Y (1B): Z = +2.0, Replacement Z = -3.0
VORP = 2.0 - (-3.0) = +5.0

Result: Catcher X is MORE valuable despite lower raw Z-score
```

### 4.3 Multi-Eligible Players

For players eligible at multiple positions, use the **position with the lowest replacement level** (scarcest position):

```python
def get_player_vorp(player: Dict, position_replacement: Dict) -> float:
    """
    Calculate VORP for a player.
    
    For multi-eligible players, use the position with lowest replacement level.
    """
    raw_z = player['total_z_score']
    eligible_positions = player['eligible_positions']
    
    # Find the position with the lowest replacement level (most scarce)
    best_position = min(eligible_positions, 
                       key=lambda pos: position_replacement[pos])
    
    replacement_z = position_replacement[best_position]
    
    return raw_z - replacement_z
```

---

## 5. Historical Replacement Level Z-Scores

### 5.1 Research Data (2016-2019)

From FanGraphs research by Ariel Cohen:

| Position | 2016 | 2017 | 2018 | 2019 | Trend |
|----------|------|------|------|------|-------|
| **Catcher (2C league)** | -6.33 | -6.26 | -6.71 | -7.09 | Declining |
| **Catcher (1C league)** | -4.60 | -5.10 | -5.43 | -6.97 | Declining |
| **First Base** | ~-3.0 | ~-3.0 | ~-3.0 | ~-3.5 | Stable |
| **Second Base** | ~-3.5 | ~-3.5 | ~-3.5 | ~-4.0 | Stable |
| **Shortstop** | ~-4.0 | ~-4.0 | ~-4.0 | ~-4.0 | Stable |
| **Third Base** | ~-3.0 | ~-3.0 | ~-3.5 | ~-3.5 | Stable |
| **Outfield** | ~-2.5 | ~-2.5 | ~-3.0 | ~-3.0 | Stable |

### 5.2 Key Insights

1. **Catcher replacement level declining** - Position is getting weaker
2. **Catcher adjustment should be -6 to -7 in 2-catcher leagues**
3. **Catcher adjustment should be -4 to -5 in 1-catcher leagues**
4. **Other positions more stable** around -3 to -4

### 5.3 Recommended 2026 Values

For a 12-team H2H league with standard positions (1 C, 1 1B, 1 2B, 1 3B, 1 SS, 3 OF):

| Position | Replacement Z | Adjustment |
|----------|--------------|------------|
| **C** | -5.5 | +5.5 |
| **1B** | -3.0 | +3.0 |
| **2B** | -4.0 | +4.0 |
| **3B** | -3.5 | +3.5 |
| **SS** | -4.0 | +4.0 |
| **OF** | -2.5 | +2.5 |

---

## 6. Implementation Guide

### 6.1 Complete Implementation

```python
import numpy as np
import pandas as pd
from typing import Dict, List

class VORPCalculator:
    """
    Calculate Value Over Replacement Player for fantasy baseball.
    """
    
    # Default replacement levels by position (Z-scores)
    DEFAULT_REPLACEMENT_LEVELS = {
        'C': -5.5,
        '1B': -3.0,
        '2B': -4.0,
        '3B': -3.5,
        'SS': -4.0,
        'OF': -2.5,
        'DH': -3.0,
        'SP': -2.0,
        'RP': -1.5,
    }
    
    def __init__(self, replacement_levels: Dict[str, float] = None):
        """
        Initialize calculator.
        
        Args:
            replacement_levels: Dict of position -> replacement Z-score
                               If None, uses defaults
        """
        self.replacement_levels = replacement_levels or self.DEFAULT_REPLACEMENT_LEVELS
    
    def calculate_player_vorp(self, 
                             player_z: float, 
                             eligible_positions: List[str]) -> Dict:
        """
        Calculate VORP for a single player.
        
        Args:
            player_z: Player's total Z-score
            eligible_positions: List of eligible positions
        
        Returns:
            Dict with VORP and metadata
        """
        # Filter to positions we have replacement levels for
        valid_positions = [p for p in eligible_positions 
                          if p in self.replacement_levels]
        
        if not valid_positions:
            return {
                'vorp': player_z,
                'raw_z': player_z,
                'position_used': 'UTIL',
                'replacement_z': 0.0
            }
        
        # Use the position with LOWEST replacement level (scarcest)
        best_position = min(valid_positions, 
                           key=lambda p: self.replacement_levels[p])
        
        replacement_z = self.replacement_levels[best_position]
        vorp = player_z - replacement_z
        
        return {
            'vorp': vorp,
            'raw_z': player_z,
            'position_used': best_position,
            'replacement_z': replacement_z,
            'position_adjustment': -replacement_z
        }
    
    def calculate_all_vorp(self, players_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VORP for all players in a DataFrame.
        
        Args:
            players_df: DataFrame with columns:
                       - 'player_id', 'name', 'total_z', 'positions'
        
        Returns:
            DataFrame with added VORP columns
        """
        results = []
        
        for _, player in players_df.iterrows():
            vorp_data = self.calculate_player_vorp(
                player['total_z'],
                player['positions']
            )
            
            results.append({
                'player_id': player['player_id'],
                'name': player['name'],
                **vorp_data
            })
        
        return pd.DataFrame(results)
```

### 6.2 Example Usage

```python
# Sample player data
players = pd.DataFrame({
    'player_id': [1, 2, 3, 4, 5],
    'name': ['Elite Catcher', 'Average 1B', 'Elite SS', 'Replacement C', 'Replacement 1B'],
    'total_z': [1.0, 2.0, 2.5, -5.5, -3.0],
    'positions': [['C'], ['1B'], ['SS'], ['C'], ['1B']]
})

# Calculate VORP
calculator = VORPCalculator()
results = calculator.calculate_all_vorp(players)

print(results[['name', 'raw_z', 'position_used', 'replacement_z', 'vorp']])
# Output:
#           name  raw_z position_used  replacement_z  vorp
#  Elite Catcher    1.0            C           -5.5   6.5
#     Average 1B    2.0           1B           -3.0   5.0
#       Elite SS    2.5           SS           -4.0   6.5
#  Replacement C   -5.5            C           -5.5   0.0
# Replacement 1B   -3.0           1B           -3.0   0.0
```

### 6.3 Determining Custom Replacement Levels

```python
def calculate_replacement_levels(player_pool: pd.DataFrame,
                                 league_size: int = 12,
                                 roster_spots: Dict[str, int] = None) -> Dict[str, float]:
    """
    Calculate replacement levels based on your specific league.
    
    Args:
        player_pool: DataFrame with all players and their Z-scores
        league_size: Number of teams
        roster_spots: Dict of position -> number of starters
    
    Returns:
        Dict of position -> replacement Z-score
    """
    default_roster = {
        'C': 1, '1B': 1, '2B': 1, '3B': 1, 'SS': 1,
        'OF': 3, 'DH': 0, 'SP': 5, 'RP': 2
    }
    roster = roster_spots or default_roster
    
    replacement_levels = {}
    
    for position, spots in roster.items():
        if spots == 0:
            continue
            
        # Filter to players eligible at this position
        eligible = player_pool[
            player_pool['positions'].apply(lambda x: position in x)
        ]
        
        # Sort by Z-score
        sorted_players = eligible.sort_values('total_z', ascending=False)
        
        # Replacement = Nth best player where N = league_size * spots
        replacement_idx = min(league_size * spots - 1, len(sorted_players) - 1)
        
        if replacement_idx >= 0:
            replacement_z = sorted_players.iloc[replacement_idx]['total_z']
            replacement_levels[position] = replacement_z
        else:
            replacement_levels[position] = -3.0  # Default fallback
    
    return replacement_levels
```

---

## 7. Worked Examples

### 7.1 Example 1: Comparing Across Positions

**Scenario:** You have the 2nd pick. Two players available:
- **Player A:** Elite Catcher, Z = +1.5
- **Player B:** Very Good 1B, Z = +3.0

**Calculation:**
```
Player A VORP = 1.5 - (-5.5) = +7.0
Player B VORP = 3.0 - (-3.0) = +6.0

Winner: Player A (Catcher) despite lower raw Z-score
```

### 7.2 Example 2: Multi-Eligible Player

**Scenario:** Mookie Betts eligible at 2B and OF
- Betts Z-score = +4.0
- 2B replacement = -4.0
- OF replacement = -2.5

**Calculation:**
```python
# Use the scarcer position (2B)
vorp = 4.0 - (-4.0) = +8.0

# NOT: 4.0 - (-2.5) = +6.5 (this would undervalue him)
```

### 7.3 Example 3: Identifying Replacement Level

**12-team league, 2 catchers each:**
- 24 catchers drafted
- 24th best catcher has Z = -6.2
- **Replacement level = -6.2**
- All catchers get +6.2 adjustment

**Result:**
| Catcher Rank | Raw Z | Adjustment | VORP |
|--------------|-------|------------|------|
| 1 | +2.0 | +6.2 | +8.2 |
| 5 | -1.0 | +6.2 | +5.2 |
| 10 | -3.0 | +6.2 | +3.2 |
| 15 | -4.5 | +6.2 | +1.7 |
| 24 | -6.2 | +6.2 | 0.0 |
| 30 | -8.0 | +6.2 | -1.8 (negative VORP = waiver wire) |

---

## 8. Practical Applications

### 8.1 Draft Strategy

**VORP-Based Draft Approach:**

1. **Pre-Draft:** Calculate VORP for all players
2. **During Draft:** Select highest VORP player available
3. **Position Scarcity Auto-Handled:** No need to "reach" for positions

**Example Draft Board (by VORP):**
```
Rank | Player       | Pos | Raw Z | VORP
-----|--------------|-----|-------|------
1    | Elite C      | C   | +2.0  | +7.5
2    | Elite SS     | SS  | +3.0  | +7.0
3    | Elite 2B     | 2B  | +3.0  | +7.0
4    | Very Good 1B | 1B  | +4.0  | +7.0
5    | Good C       | C   | +0.5  | +6.0
```

### 8.2 Trade Evaluation

**Scenario:** Trade offer - Your Elite Catcher (VORP +7.5) for their Elite 1B (VORP +7.0) + Good OF (VORP +3.0)

**Analysis:**
```
You Give:    +7.5
You Get:     +7.0 + 3.0 = +10.0
Net Gain:    +2.5 VORP

Verdict: Accept (but check roster fit)
```

### 8.3 Waiver Wire Decisions

**Rule:** Only pick up players with positive VORP

```python
def should_pickup(player_vorp: float, current_bench_vorp: float) -> bool:
    """
    Determine if a waiver wire player should be picked up.
    
    Args:
        player_vorp: VORP of available player
        current_bench_vorp: VORP of your worst bench player
    
    Returns:
        True if should pickup
    """
    return player_vorp > current_bench_vorp
```

---

## 9. When NOT to Use VORP

### 9.1 Limitations

| Situation | Why VORP Fails | Alternative |
|-----------|---------------|-------------|
| **H2H Categories** | Weekly matchup dynamics | Category-specific analysis |
| **Streaming positions** | Constant turnover | Recent performance only |
| **Playoff push** | Must-win weeks | High-variance plays |
| **Category punting** | Irrelevant categories ignored | Punt-adjusted Z-scores |
| **Injury replacement** | Short-term fill | Streamer rankings |

### 9.2 VORP Blind Spots

1. **Doesn't account for:**
   - Category balance on your team
   - Upcoming schedule strength
   - Player health/playing time risk
   - Platoon splits

2. **Assumes:**
   - All categories weighted equally
   - Constant playing time
   - Accurate Z-score projections

---

## 10. Advanced: Converting VORP to Dollar Values

### 10.1 Auction Dollar Conversion

```python
def vorp_to_dollar(vorp_values: List[float], 
                   total_budget: float = 260,
                   num_teams: int = 12,
                   roster_size: int = 23) -> List[float]:
    """
    Convert VORP values to auction dollars.
    
    Formula: $1 per VORP unit, distributed across budget
    """
    # Only positive VORP players get paid
    positive_vorp = [max(0, v) for v in vorp_values]
    total_vorp = sum(positive_vorp)
    
    # Calculate $ per VORP unit
    # Every team must spend $1 per roster spot
    min_cost = num_teams * roster_size
    available_budget = total_budget * num_teams - min_cost
    
    dollar_per_vorp = available_budget / total_vorp
    
    # Convert
    dollar_values = []
    for vorp in vorp_values:
        if vorp <= 0:
            dollar_values.append(1.0)  # $1 minimum
        else:
            dollar_values.append(1.0 + vorp * dollar_per_vorp)
    
    return dollar_values
```

### 10.2 Example Conversion

| Player | VORP | Calculation | Dollar Value |
|--------|------|-------------|--------------|
| Elite C | +7.5 | $1 + 7.5 × $3 | $23.50 |
| Elite SS | +7.0 | $1 + 7.0 × $3 | $22.00 |
| Average 1B | +4.0 | $1 + 4.0 × $3 | $13.00 |
| Replacement | 0.0 | $1 minimum | $1.00 |
| Below Replacement | -1.0 | $1 minimum | $1.00 |

---

## 11. Implementation Checklist

### Pre-Season Setup

- [ ] Determine league roster requirements
- [ ] Calculate custom replacement levels
- [ ] Verify position eligibility rules
- [ ] Test VORP calculation on known players

### In-Season Usage

- [ ] Recalculate Z-scores weekly
- [ ] Update replacement levels monthly
- [ ] Adjust for significant injuries
- [ ] Re-evaluate position eligibility changes

### Validation Tests

- [ ] Replacement level player at each position = ~$1
- [ ] Top player VORP reasonable (~+8 to +10)
- [ ] Catcher adjustment > 1B adjustment
- [ ] Multi-eligible players use scarcest position

---

## 12. Summary

### Key Formulas

```
Fantasy VORP = Player_Z - Replacement_Z (by position)

Replacement Level:
- Catcher: 75% of positional average (Z ≈ -5.5 to -7)
- 1B/DH: 85% of positional average (Z ≈ -3)
- All others: 80% of positional average (Z ≈ -3.5 to -4)

Multi-Eligible: Use position with LOWEST replacement level
```

### Recommended Default Replacement Z-Scores

| Position | 1-Catcher League | 2-Catcher League |
|----------|------------------|------------------|
| **C** | -4.5 | -6.5 |
| **1B** | -3.0 | -3.0 |
| **2B** | -4.0 | -4.0 |
| **3B** | -3.5 | -3.5 |
| **SS** | -4.0 | -4.0 |
| **OF** | -2.5 | -2.5 |

---

## Sources

1. **Keith Woolner VORP:** https://www.baseballprospectus.com/news/article/6231/
2. **TangoTiger Wiki:** https://tangotiger.net/wiki_archive/VORP.html
3. **Catcher Adjustment:** https://fantasy.fangraphs.com/the-catcher-positional-adjustment-using-z-scores/
4. **Position Scarcity:** https://www.fantasypros.com/2026/01/position-scarcity-strategy-draft-advice-2026-fantasy-baseball/
5. **Replacement Level Theory:** https://tht.fangraphs.com/my-approach-to-player-valuation/

---

*Report generated: April 11, 2026*  
*Research confidence: HIGH (authoritative sources + established methodology)*
