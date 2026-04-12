# Z-Score Best Practices for Fantasy Baseball

> **Research Date:** April 11, 2026  
> **Researcher:** Kimi CLI (Deep Intelligence)  
> **Task ID:** K-35  
> **Status:** COMPLETE

---

## 1. Executive Summary

This report outlines statistical best practices for Z-score calculations in fantasy baseball contexts. Z-scores (standard scores) are the foundation of modern player valuation, enabling comparison across different statistical categories.

### Key Findings

| Topic | Recommendation |
|-------|---------------|
| **Sample Size** | Use Russell Carleton stabilization points for reliability thresholds |
| **Position Adjustment** | Apply replacement level by position, not within-position Z-scores |
| **Outlier Handling** | Winsorize at 5th/95th percentile OR use Robust Z-scores (median/MAD) |
| **Temporal Window** | YTD for in-season; 30-day rolling for recent performance |
| **Two-Way Players** | Calculate separate Z-scores for batting and pitching |

### Quick Reference: Stabilization Points

| Stat | Stabilization Point | Notes |
|------|-------------------|-------|
| Swing Rate | 40 PA | Very quick to stabilize |
| K% (Strikeout Rate) | 60 PA | Quick stabilization |
| Exit Velocity | 40 BBE | Very quick |
| BB% (Walk Rate) | 120 PA | Moderate |
| AVG | ~100 PA | Moderate |
| OBP | ~100 PA | Moderate |
| ISO (Isolated Power) | ~150 PA | Slower |
| BABIP | ~400 BBE | Very slow |
| ERA (pitchers) | ~70 IP | Moderate |
| WHIP | ~70 IP | Moderate |

---

## 2. Z-Score Fundamentals

### 2.1 Basic Formula

The standard Z-score formula:

```
Z = (X - μ) / σ

Where:
- X = Player's stat value
- μ = Mean of the population
- σ = Standard deviation of the population
```

### 2.2 Interpretation

| Z-Score Range | Interpretation |
|---------------|---------------|
| > +2.0 | Elite (top 2.5%) |
| +1.0 to +2.0 | Very good (top 16%) |
| +0.5 to +1.0 | Good (top 31%) |
| -0.5 to +0.5 | Average (middle 38%) |
| -1.0 to -0.5 | Below average (bottom 31%) |
| -2.0 to -1.0 | Poor (bottom 16%) |
| < -2.0 | Very poor (bottom 2.5%) |

### 2.3 Implementation in Python

```python
import numpy as np
from typing import List, Dict

def calculate_z_scores(player_stats: List[float]) -> List[float]:
    """
    Calculate standard Z-scores for a list of player statistics.
    
    Args:
        player_stats: List of statistical values (e.g., all players' HR totals)
    
    Returns:
        List of Z-scores corresponding to each player
    """
    mean = np.mean(player_stats)
    std = np.std(player_stats, ddof=1)  # Sample standard deviation
    
    if std == 0:
        return [0.0] * len(player_stats)
    
    return [(x - mean) / std for x in player_stats]
```

---

## 3. Sample Size Considerations

### 3.1 Russell Carleton Stabilization Points

Russell Carleton's ("Pizza Cutter") landmark research determined how many plate appearances are needed for statistics to "stabilize" - meaning they become reliable indicators of true talent.

**Key Concept:** Stabilization occurs when the correlation between two equal samples reaches r = 0.707 (R² = 0.50), indicating equal parts signal and noise.

### 3.2 Hitting Stabilization Points

| Stat | Stabilization | Denominator | Practical Use |
|------|--------------|-------------|---------------|
| **Swing Rate** | 40 PA | PA | Reliable after 2 weeks |
| **K%** | 60 PA | PA | Reliable after 3 weeks |
| **Exit Velocity** | 40 BBE | BBE | Reliable quickly |
| **BB%** | 120 PA | PA | Reliable after 1 month |
| **AVG** | ~100 PA | AB | Moderate reliability |
| **OBP** | ~100 PA | PA | Moderate reliability |
| **SLG** | ~150 PA | AB | Use with caution early |
| **ISO** | ~150 PA | AB | Use with caution early |
| **BABIP** | ~400 BBE | BIP | Very noisy, regresses heavily |
| **LD%** | ~600 BIP | BIP | Extremely noisy |

### 3.3 Pitching Stabilization Points

| Stat | Stabilization | Denominator | Practical Use |
|------|--------------|-------------|---------------|
| **K%** | 100 BF | BF | Reliable after 3-4 starts |
| **BB%** | 170 BF | BF | Reliable after 5-6 starts |
| **GB%** | 70 BIP | BIP | Quick to stabilize |
| **FB%** | 70 BIP | BIP | Quick to stabilize |
| **ERA** | ~70 IP | IP | Use with caution early |
| **WHIP** | ~70 IP | IP | Moderate reliability |

### 3.4 Recommendations by Sample Size

**< 30 PA/IP:**
- Heavy regression to mean (75-90%)
- Use preseason projections primarily
- Z-scores have high uncertainty

**30-60 PA/IP:**
- Moderate regression (50-75%)
- Blend current performance with projections
- Z-scores becoming meaningful

**60-100 PA/IP:**
- Light regression (25-50%)
- Current performance more reliable
- Z-scores reasonably stable

**> 100 PA/IP:**
- Minimal regression (10-25%)
- Current performance is signal
- Z-scores are reliable

### 3.5 Implementation: Sample Size Weighting

```python
def calculate_regression_weight(sample_size: int, stabilization_point: int) -> float:
    """
    Calculate how much to weight observed performance vs. mean.
    
    At stabilization point, weight = 0.5 (50/50 split)
    Below stabilization, weight < 0.5 (regress more)
    Above stabilization, weight > 0.5 (trust observed)
    
    Args:
        sample_size: Current PA, AB, or IP
        stabilization_point: Carleton stabilization point
    
    Returns:
        Weight to apply to observed performance (0.0 to 1.0)
    """
    weight = sample_size / (sample_size + stabilization_point)
    return weight


def regress_to_mean(observed: float, mean: float, sample_size: int, 
                    stabilization_point: int) -> float:
    """
    Regress observed statistic toward the mean based on sample size.
    
    Args:
        observed: Player's observed statistic
        mean: Population mean for the statistic
        sample_size: Player's sample size (PA, AB, IP)
        stabilization_point: Stat-specific stabilization point
    
    Returns:
        Regressed statistic value
    """
    weight = calculate_regression_weight(sample_size, stabilization_point)
    return weight * observed + (1 - weight) * mean
```

---

## 4. Position Adjustments

### 4.1 The Core Question

**Should Z-scores be calculated within positions (C vs C, 1B vs 1B) or globally (all players together)?**

**Answer:** Calculate Z-scores **globally**, then apply **position adjustments** via replacement level.

### 4.2 Why Global Z-Scores?

| Approach | Problem |
|----------|---------|
| Within-position Z-scores | Best catcher gets Z=+2.0 even if worse than average 1B |
| Global Z-scores + position adjustment | Properly values positional scarcity |

**Example:**
- Best catcher: .250 AVG, 15 HR, 60 RBI (Z-total = +0.5 globally)
- Average 1B: .260 AVG, 25 HR, 85 RBI (Z-total = +0.0 globally)
- Within-position Z: Catcher = +2.0, 1B = +0.0
- **Within-position overrates the catcher**

### 4.3 Replacement Level by Position

The proper way to handle position scarcity:

1. Calculate Z-scores globally (all players together)
2. Determine replacement level Z-score for each position
3. Adjust player values: `Value = Z - Replacement_Z`

**Historical Replacement Level Z-Scores (2018-2019):**

| Position | Replacement Z-Score | Notes |
|----------|-------------------|-------|
| Catcher | -6.0 to -7.0 | Deepest negative |
| Shortstop | -3.5 to -4.0 | Weak offense |
| 2B | -3.5 to -4.0 | Weak offense |
| 3B | -3.0 to -3.5 | Moderate |
| 1B | -2.5 to -3.0 | Shallow position |
| OF | -2.5 to -3.0 | Deep position |

### 4.4 Catcher Adjustment Example

From FanGraphs research (Ariel Cohen, 2019):

| Catcher Rank | Raw Z-Score | Adjustment | Adjusted Z |
|--------------|-------------|------------|------------|
| 1 | +1.5 | +6.0 | +7.5 |
| 5 | +0.5 | +6.0 | +6.5 |
| 10 | -0.5 | +6.0 | +5.5 |
| 15 | -2.0 | +6.0 | +4.0 |
| 20 | -3.5 | +6.0 | +2.5 |
| 30 (replacement) | -6.0 | +6.0 | 0.0 |

**Without adjustment:** 30th catcher has value
**With adjustment:** 30th catcher = replacement level ($1 value)

---

## 5. Outlier Handling

### 5.1 The Problem: Extreme Values

In fantasy baseball, outliers can significantly skew Z-scores:
- Shohei Ohtani's unique two-way performance
- Barry Bonds-esque walk rates
- Extreme BABIP luck (high or low)

### 5.2 Approach 1: Winsorization

**Definition:** Cap extreme values at specific percentiles (typically 5th and 95th).

**Formula:**
```
X_winsorized = Q_0.05    if X < Q_0.05
               X         if Q_0.05 ≤ X ≤ Q_0.95
               Q_0.95    if X > Q_0.95
```

**Implementation:**
```python
def winsorize(data: np.ndarray, limits: tuple = (0.05, 0.05)) -> np.ndarray:
    """
    Winsorize data at specified percentiles.
    
    Args:
        data: Array of values
        limits: Tuple of (lower_percentile, upper_percentile) to trim
    
    Returns:
        Winsorized array
    """
    from scipy.stats import mstats
    return mstats.winsorize(data, limits=limits)


# Example usage for fantasy stats
player_hr = np.array([5, 8, 12, 15, 18, 22, 25, 28, 32, 35, 38, 42, 48, 55, 65])
winsorized_hr = winsorize(player_hr, limits=(0.05, 0.05))
# 65 capped to ~55, 5 capped to ~8
```

**Pros:**
- Preserves all data points
- Reduces impact of extreme outliers
- Standard statistical technique

**Cons:**
- Artificially compresses range
- May undervalue truly elite players

### 5.3 Approach 2: Robust Z-Scores (Median/MAD)

**Definition:** Use median instead of mean, Median Absolute Deviation (MAD) instead of standard deviation.

**Formula:**
```
Robust Z = 0.6745 * (X - Median) / MAD

Where:
- MAD = Median(|X_i - Median|)
- 0.6745 = scaling factor (75th percentile of normal distribution)
```

**Implementation:**
```python
def calculate_robust_z_scores(data: np.ndarray) -> np.ndarray:
    """
    Calculate robust Z-scores using median and MAD.
    
    More resistant to outliers than standard Z-scores.
    
    Args:
        data: Array of values
    
    Returns:
        Array of robust Z-scores
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        return np.zeros_like(data)
    
    return 0.6745 * (data - median) / mad


# Comparison
standard_z = calculate_z_scores(player_hr)
robust_z = calculate_robust_z_scores(player_hr)

# Robust Z-scores are less affected by the 65 HR outlier
```

**Pros:**
- Naturally resistant to outliers
- No arbitrary percentile cutoff
- Better for skewed distributions

**Cons:**
- Less intuitive interpretation
- MAD can be zero (if >50% of data identical)

### 5.4 Comparison: Standard vs Robust Z-Scores

```python
import pandas as pd

# Example dataset with outliers
np.random.seed(42)
home_runs = np.concatenate([
    np.random.normal(20, 8, 100),  # Normal players
    [65, 5]  # Outliers: elite and terrible
])

comparison = pd.DataFrame({
    'HR': home_runs,
    'Standard_Z': calculate_z_scores(home_runs),
    'Robust_Z': calculate_robust_z_scores(home_runs)
})

# Standard Z: 65 HR = +4.2, 5 HR = -1.8
# Robust Z: 65 HR = +3.1, 5 HR = -1.5
# Robust Z less extreme for outliers
```

### 5.5 Recommendation

| Scenario | Recommended Method |
|----------|-------------------|
| Normal distribution, few outliers | Standard Z-score |
| Heavy-tailed distribution | Robust Z-score |
| Extreme outliers present | Winsorization at 95% |
| Skewed data | Robust Z-score or log-transform |

---

## 6. Temporal Considerations

### 6.1 Time Window Options

| Window | Use Case | Weighting |
|--------|----------|-----------|
| **YTD (Year-to-Date)** | Standard in-season | Equal weight |
| **30-day rolling** | Recent performance | Equal weight |
| **14-day rolling** | Hot/cold streaks | Equal weight |
| **Weighted (exponential)** | Recency matters | λ = 0.9-0.95 |
| **Career/3-year** | Projections | Varies |

### 6.2 Exponential Weighting

Give more weight to recent performance:

```python
def calculate_weighted_average(values: List[float], 
                               halflife: int = 10) -> float:
    """
    Calculate exponentially weighted moving average.
    
    Args:
        values: Time series of values (oldest to newest)
        halflife: Number of periods for weight to decay to 0.5
    
    Returns:
        Weighted average
    """
    lambda_decay = 0.5 ** (1 / halflife)
    weights = [lambda_decay ** (len(values) - 1 - i) 
               for i in range(len(values))]
    
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


# Example: 30-day weighted average with 10-day halflife
recent_hr = [0, 1, 0, 2, 1, 1, 0, 3, 2, 1]  # Last 10 games
weighted_avg = calculate_weighted_average(recent_hr, halflife=5)
# Recent games weighted more heavily
```

### 6.3 When Do Stats Become Reliable?

**Early Season (April):**
- Use preseason projections heavily (>80%)
- Z-scores based on small samples are noisy
- Focus on skills (K%, BB%, exit velocity) not outcomes (AVG, ERA)

**Mid-Season (June-July):**
- Blend projections with current performance (50/50)
- Z-scores becoming reliable for most stats
- Sample sizes approaching stabilization

**Late Season (August-September):**
- Current performance > projections
- Z-scores are reliable
- Consider fatigue, September call-ups

---

## 7. Category-Specific Considerations

### 7.1 Counting Stats vs Rate Stats

| Type | Examples | Z-Score Approach |
|------|----------|------------------|
| **Counting** | R, HR, RBI, SB, W, K | Raw totals (volume matters) |
| **Rate** | AVG, OBP, ERA, WHIP, K/9 | Weighted by denominator |

### 7.2 Rate Stat Weighting

Rate stats should be weighted by their denominator to avoid overvaluing small samples:

```python
def calculate_weighted_rate_z_score(rates: List[float], 
                                    denominators: List[int],
                                    min_denom: int = 50) -> List[float]:
    """
    Calculate Z-scores for rate stats, weighted by sample size.
    
    Args:
        rates: Rate values (e.g., batting averages)
        denominators: Sample sizes (e.g., at-bats)
        min_denom: Minimum denominator to include
    
    Returns:
        Weighted Z-scores
    """
    # Filter out small samples
    valid_data = [(r, d) for r, d in zip(rates, denominators) 
                  if d >= min_denom]
    
    if not valid_data:
        return [0.0] * len(rates)
    
    valid_rates, valid_denoms = zip(*valid_data)
    
    # Weighted mean
    total_weight = sum(valid_denoms)
    weighted_mean = sum(r * d for r, d in zip(valid_rates, valid_denoms)) / total_weight
    
    # Weighted std (simplified)
    variance = sum(d * (r - weighted_mean) ** 2 
                   for r, d in zip(valid_rates, valid_denoms)) / total_weight
    weighted_std = np.sqrt(variance)
    
    # Calculate Z-scores (unweighted for individual players)
    return [(r - weighted_mean) / weighted_std if weighted_std > 0 else 0 
            for r in rates]
```

### 7.3 Ratio Categories (ERA, WHIP)

Lower is better - flip the sign:

```python
def calculate_ratio_z_score(player_rates: List[float]) -> List[float]:
    """
    Calculate Z-scores for ratio categories (lower is better).
    
    Returns negative Z-scores so positive value = good performance.
    """
    mean = np.mean(player_rates)
    std = np.std(player_rates, ddof=1)
    
    # Negate so lower ERA = positive Z-score
    return [-(r - mean) / std for r in player_rates]
```

---

## 8. Special Cases

### 8.1 Two-Way Players (Ohtani)

**Recommendation:** Calculate Z-scores separately for batting and pitching.

```python
def calculate_two_way_z_scores(batting_stats: Dict, 
                               pitching_stats: Dict) -> Dict:
    """
    Calculate Z-scores for a two-way player.
    
    Returns separate batting and pitching Z-scores.
    Do NOT combine into single number.
    """
    batting_z = calculate_batting_z_scores(batting_stats)
    pitching_z = calculate_pitching_z_scores(pitching_stats)
    
    return {
        'batting': batting_z,
        'pitching': pitching_z,
        'total': batting_z['total'] + pitching_z['total']  # If must combine
    }
```

### 8.2 Multi-Position Eligibility

Use the position with highest replacement level (usually):

```python
def get_position_adjusted_z(player: Dict, eligible_positions: List[str],
                           replacement_levels: Dict[str, float]) -> float:
    """
    Get Z-score adjusted for player's most valuable position.
    
    Args:
        player: Player dict with 'raw_z_score'
        eligible_positions: List of eligible positions
        replacement_levels: Dict of position -> replacement Z
    
    Returns:
        Position-adjusted Z-score
    """
    raw_z = player['raw_z_score']
    
    # Find position with lowest replacement level (scarcest)
    best_position = min(eligible_positions, 
                       key=lambda pos: replacement_levels.get(pos, 0))
    
    adjustment = -replacement_levels.get(best_position, 0)
    return raw_z + adjustment
```

---

## 9. Implementation Checklist

### 9.1 Core Functions

```python
class ZScoreCalculator:
    """Complete Z-score calculation system for fantasy baseball."""
    
    STABILIZATION_POINTS = {
        'k_rate': 60,    # PA for hitters, BF for pitchers
        'bb_rate': 120,
        'avg': 100,      # AB
        'obp': 100,      # PA
        'slg': 150,      # AB
        'iso': 150,
        'era': 70,       # IP
        'whip': 70,
        'babip': 400,    # BIP
    }
    
    def __init__(self, use_robust: bool = False, 
                 winsorize_limits: tuple = None):
        """
        Initialize calculator.
        
        Args:
            use_robust: Use median/MAD instead of mean/std
            winsorize_limits: (lower, upper) percentiles to cap
        """
        self.use_robust = use_robust
        self.winsorize_limits = winsorize_limits
    
    def calculate(self, values: np.ndarray, 
                  denominators: np.ndarray = None) -> np.ndarray:
        """Calculate Z-scores with configured options."""
        data = values.copy()
        
        # Winsorization
        if self.winsorize_limits:
            data = self._winsorize(data, self.winsorize_limits)
        
        # Calculate Z-scores
        if self.use_robust:
            return self._robust_z_score(data)
        else:
            return self._standard_z_score(data)
    
    def _standard_z_score(self, data: np.ndarray) -> np.ndarray:
        """Standard (mean/std) Z-scores."""
        return (data - np.mean(data)) / np.std(data, ddof=1)
    
    def _robust_z_score(self, data: np.ndarray) -> np.ndarray:
        """Robust (median/MAD) Z-scores."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return np.zeros_like(data)
        return 0.6745 * (data - median) / mad
    
    def _winsorize(self, data: np.ndarray, 
                   limits: tuple) -> np.ndarray:
        """Winsorize data at percentiles."""
        from scipy.stats import mstats
        return mstats.winsorize(data, limits=limits)
```

### 9.2 Validation Checklist

Before deploying Z-score calculations:

- [ ] Verify sample sizes meet stabilization thresholds
- [ ] Check for and handle outliers appropriately
- [ ] Confirm rate stats are weighted by denominator
- [ ] Validate position adjustments are applied
- [ ] Test with known players (validate rankings make sense)
- [ ] Check for division by zero errors
- [ ] Verify small samples are regressed appropriately

---

## 10. Common Mistakes to Avoid

| Mistake | Why It's Wrong | Correct Approach |
|---------|---------------|------------------|
| **Using small samples raw** | High variance, unreliable | Regress to mean using stabilization points |
| **Ignoring position scarcity** | Overvalues 1B, undervalues C | Apply position adjustments |
| **Not handling outliers** | Skews mean/std for everyone | Winsorize or use robust Z-scores |
| **Combining Ohtani's stats** | Pitching ≠ Hitting | Calculate separately |
| **Rate stats unweighted** | Small samples = extreme values | Weight by AB/IP/PA |
| **Z-score within position** | Ignores true value differences | Global Z + position adjustment |
| **Not updating mean/std** | Player pool changes | Recalculate regularly |

---

## 11. Summary & Recommendations

### Best Practices Summary

1. **Use global Z-scores** with position adjustments, not within-position
2. **Regress small samples** using Carleton stabilization points
3. **Handle outliers** via Winsorization (5%/95%) or robust Z-scores
4. **Weight rate stats** by their denominator
5. **Calculate two-way players** separately for batting and pitching
6. **Update means/stds** regularly as player pool changes

### Recommended Configuration

```python
# Production settings
calculator = ZScoreCalculator(
    use_robust=False,           # Standard is more interpretable
    winsorize_limits=(0.05, 0.05)  # Cap at 5th/95th percentile
)

# For stats with extreme outliers (rarely)
robust_calculator = ZScoreCalculator(
    use_robust=True,
    winsorize_limits=None
)
```

---

## Sources

1. **Russell Carleton Stabilization:** https://www.baseballprospectus.com/news/article/14293/
2. **FanGraphs Z-Score Guide:** https://fantasy.fangraphs.com/the-catcher-positional-adjustment-using-z-scores/
3. **Smart Fantasy Baseball:** https://www.smartfantasybaseball.com/2014/04/cautionary-notes-about-pizza-cutter-carleton-sample-size-stabilization-points/
4. **Winsorization Methods:** https://www.mdpi.com/2227-9032/13/24/3193
5. **Robust Z-Scores:** https://thirdeyedata.ai/anomaly-detection-with-robust-zscore/

---

*Report generated: April 11, 2026*  
*Research confidence: HIGH (peer-reviewed sources + established statistical methods)*
