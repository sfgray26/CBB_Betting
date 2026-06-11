# UAT Evaluation Criteria - Quant Trading & Sabermetrics Perspective

## Persona Definition

**Background**: Quantitative analyst with expertise in:
- Statistical arbitrage and edge detection
- Sabermetrics (wOBA, xFIP, SIERA, WAR, etc.)
- Probability theory and expected value calculations
- Market efficiency and price discovery
- Backtesting and model validation

**Expectations**: Rigorously validated data, transparent methodology, statistically sound projections, and measurable alpha generation.

## Evaluation Dimensions

### 1. Edge Calculation & Signal Quality (Weight: 25%)

#### 1.1 Closing Line Value (CLV)
- [ ] **P0**: CLV calculations match market data
- [ ] **P0**: Beat line percentage accurately tracked
- [ ] **P1**: CLV by bet type (spread, total, ML) broken out
- [ ] **P2**: t-statistic and confidence intervals shown

**Quantitative Tests**:
```python
# CLV Backtest Validation
def validate_clv_accuracy():
    """
    CLV should predict win probability better than opening line
    """
    sample_bets = get_historical_bets(n=1000)
    
    # Test: Positive CLV should correlate with higher win rate
    positive_clv = [b for b in sample_bets if b.clv > 0]
    negative_clv = [b for b in sample_bets if b.clv < 0]
    
    win_rate_positive = calculate_win_rate(positive_clv)
    win_rate_negative = calculate_win_rate(negative_clv)
    
    assert win_rate_positive > win_rate_negative, \
        "Positive CLV should have higher win rate"
    assert win_rate_positive > 0.5, \
        "Positive CLV should beat 50% baseline"
    
    return {
        "positive_clv_win_rate": win_rate_positive,
        "negative_clv_win_rate": win_rate_negative,
        "edge": win_rate_positive - 0.5
    }
```

**Scoring**:
- 10: CLV is highly predictive (correlation > 0.3)
- 8: CLV has measurable edge (correlation 0.15-0.3)
- 6: CLV shows weak signal (correlation 0.05-0.15)
- 4: CLV noise (correlation near 0)
- 0: CLV misleading (negative correlation)

#### 1.2 Expected Value (EV) Calculations
- [ ] **P0**: EV formula is transparent and correct
- [ ] **P0**: Kelly criterion or fractional Kelly used
- [ ] **P1**: EV accounts for juice/vig
- [ ] **P2**: Model uncertainty incorporated

**Validation Check**:
```
EV = (Win_Prob × Profit) - (Loss_Prob × Stake)

Test Case: -110 odds, 55% win probability
EV = (0.55 × 100) - (0.45 × 110) = 55 - 49.50 = +5.50
Expected Output: +5.0% EV
Actual Output: [Document during test]
```

### 2. Statistical Model Quality (Weight: 25%)

#### 2.1 Projection Accuracy
- [ ] **P0**: Historical backtests show positive ROI
- [ ] **P0**: Brier score or log loss calculated
- [ ] **P1**: Calibration plots available (predicted vs actual)
- [ ] **P2**: Model updates reflect new information

**Backtest Requirements**:
```python
# Minimum Backtest Standards
MIN_SAMPLE_SIZE = 500  # bets
MIN_TIME_PERIOD = "6_months"
REQUIRED_METRICS = [
    "total_roi",
    "roi_by_quartile",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate_by_confidence",
    "calibration_error"
]

# Sharpe Ratio Benchmark
assert sharpe_ratio > 1.0, "Risk-adjusted returns inadequate"
assert max_drawdown < 0.30, "Drawdown too severe"
```

**Scoring Matrix**:
| Metric | Poor | Acceptable | Good | Excellent |
|--------|------|------------|------|-----------|
| ROI | < 0% | 0-5% | 5-10% | > 10% |
| Sharpe | < 0.5 | 0.5-1.0 | 1.0-2.0 | > 2.0 |
| Max DD | > 50% | 30-50% | 20-30% | < 20% |
| Brier Score | > 0.25 | 0.20-0.25 | 0.15-0.20 | < 0.15 |

#### 2.2 Feature Engineering
- [ ] **P1**: Advanced sabermetrics included (xWOBA, xFIP, etc.)
- [ ] **P1**: Park factors properly applied
- [ ] **P2**: Weather adjustments for pitchers
- [ ] **P2**: Umpire tendencies considered

**Sabermetrics Checklist**:
```
Required Metrics:
✓ wOBA (Weighted On-Base Average)
✓ xWOBA (Expected wOBA)
✓ Barrel% (Batted ball quality)
✓ xFIP (Expected Fielding Independent Pitching)
✓ SIERA (Skill-Interactive ERA)
✓ WAR (Wins Above Replacement)
✓ wRC+ (Weighted Runs Created Plus)

Optional Advanced Metrics:
○ HardHit%
○ Chase%
○ Whiff%
○ Spin Rate (for pitchers)
```

### 3. Data Quality & Integrity (Weight: 20%)

#### 3.1 Source Verification
- [ ] **P0**: Data sources documented
- [ ] **P0**: Staleness indicators present
- [ ] **P1**: Data validation rules active
- [ ] **P2**: Outlier detection implemented

**Data Quality Metrics**:
```python
# Data Freshness SLA
MAX_DATA_AGE = {
    "player_stats": timedelta(hours=1),
    "lineups": timedelta(minutes=15),
    "probable_pitchers": timedelta(hours=2),
    "odds": timedelta(minutes=5),
    "injuries": timedelta(minutes=30)
}

# Validation Checks
def validate_data_quality():
    issues = []
    
    # Check for impossible values
    if any(player.avg > 0.500 for player in hitters):
        issues.append("Impossible batting average detected")
    
    # Check for stale data
    if (now() - last_update) > MAX_DATA_AGE["player_stats"]:
        issues.append("Player stats stale")
    
    return issues
```

#### 3.2 Completeness
- [ ] **P0**: All active players have projections
- [ ] **P0**: No missing data for starters
- [ ] **P1**: Prospect call-ups handled quickly
- [ ] **P2**: Minor league stats available

### 4. Risk Management (Weight: 15%)

#### 4.1 Bankroll Management
- [ ] **P0**: Position sizing recommendations given
- [ ] **P0**: Unit system clearly defined
- [ ] **P1**: Kelly fraction adjustable
- [ ] **P2": Correlation between bets considered

**Kelly Criterion Validation**:
```python
def calculate_kelly_fraction(edge, odds):
    """
    Kelly = (bp - q) / b
    where:
    b = odds received (decimal - 1)
    p = probability of win
    q = probability of loss (1 - p)
    """
    b = odds - 1
    p = edge
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly)  # Never bet negative

# Test: Verify Kelly recommendations
for bet in sample_bets:
    recommended = bet.kelly_fraction
    calculated = calculate_kelly_fraction(bet.edge, bet.odds)
    assert abs(recommended - calculated) < 0.01, "Kelly mismatch"
```

#### 4.2 Portfolio Construction
- [ ] **P2**: Exposure limits by position/sport
- [ ] **P2**: Correlation matrix displayed
- [ ] **P3**: Monte Carlo simulation of outcomes

### 5. Model Transparency (Weight: 10%)

#### 5.1 Explainability
- [ ] **P1**: Model inputs visible
- [ ] **P1**: Confidence intervals shown
- [ ] **P2**: Feature importance displayed
- [ ] **P3**: SHAP values or LIME explanations

**Explainability Requirements**:
```
Every recommendation must show:
1. Primary driver (e.g., "70% based on pitcher matchup")
2. Secondary factors (e.g., "20% park factor, 10% weather")
3. Confidence level (e.g., "High confidence - 200+ PA sample")
4. Caveats (e.g., "Small sample vs LHP")
```

#### 5.2 Audit Trail
- [ ] **P2**: Model versions tracked
- [ ] **P2**: Prediction history logged
- [ ] **P3": A/B test results available

### 6. Market Efficiency Analysis (Weight: 5%)

#### 6.1 Market Movement
- [ ] **P2**: Line movement tracking
- [ ] **P2**: Sharp money indicators
- [ ] **P3**: Reverse line movement detection

#### 6.2 Arbitrage Detection
- [ ] **P3**: Cross-book arbitrage opportunities
- [ ] **P3**: Derivative pricing anomalies

## Quantitative Validation Suite

### Automated Statistical Tests

```python
# 1. Calibration Test
def test_calibration():
    """
    If model says 60% win prob, actual win rate should be ~60%
    """
    bins = defaultdict(list)
    for bet in historical_bets:
        prob_bin = round(bet.model_prob * 10) / 10  # 0.1 increments
        bins[prob_bin].append(bet.won)
    
    for prob_bin, outcomes in bins.items():
        actual = np.mean(outcomes)
        assert abs(actual - prob_bin) < 0.05, \
            f"Calibration off at {prob_bin}: actual {actual}"

# 2. Sharpe Ratio Calculation
def calculate_sharpe(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

# 3. Maximum Drawdown
def calculate_max_drawdown(equity_curve):
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

# 4. Value at Risk (VaR)
def calculate_var(returns, confidence=0.95):
    return np.percentile(returns, (1 - confidence) * 100)
```

## Scoring Matrix

| Dimension | Weight | Score (1-10) | Weighted Score |
|-----------|--------|--------------|----------------|
| Edge Calculation & Signal Quality | 25% | [ ] | [ ] |
| Statistical Model Quality | 25% | [ ] | [ ] |
| Data Quality & Integrity | 20% | [ ] | [ ] |
| Risk Management | 15% | [ ] | [ ] |
| Model Transparency | 10% | [ ] | [ ] |
| Market Efficiency Analysis | 5% | [ ] | [ ] |
| **TOTAL** | **100%** | | **[ ]** |

## Red Flags (Automatic Failure)

- [ ] Backtest shows negative ROI with >500 bets
- [ ] Sharpe ratio < 0.5 over 3+ months
- [ ] Calibration error > 10%
- [ ] Data inconsistencies detected
- [ ] Kelly recommendations > 25% of bankroll
- [ ] Model outputs don't match documented methodology

## Test Execution

```bash
# Full Quant Evaluation with statistical validation
python scripts/uat_quant_evaluation.py \
  --base-url https://your-app.railway.app \
  --api-key $API_KEY \
  --backtest-days 90 \
  --sample-size 1000 \
  --confidence 0.95 \
  --output reports/uat_quant_$(date +%Y%m%d).md
```

## Example Output

```
========================================
QUANT & SABERMETRICS UAT REPORT
Date: 2026-05-17
Tester: Kimi + Statistical Validation
Sample Size: 1,247 bets
Time Period: 2026-02-15 to 2026-05-17
========================================

OVERALL SCORE: 7.8/10 ⚠️ (Acceptable with reservations)

STATISTICAL SUMMARY:
- ROI: +6.8% ✅（Excellent)
- Sharpe Ratio: 1.34 ✅ (Good)
- Max Drawdown: 18.2% ✅ (Good)
- Brier Score: 0.198 ✅ (Acceptable)
- Calibration Error: 4.3% ✅ (Good)

DIMENSION SCORES:
1. Edge Calculation: 9/10 ✅
   - CLV predictive power: r=0.28
   - EV calculations validated
   - Note: Some +EV bets not flagged

2. Model Quality: 8/10 ✅
   - Backtest: +6.8% ROI over 1,247 bets
   - Sharpe: 1.34
   - Issue: Underperforms in high-vig markets

3. Data Quality: 6/10 ⚠️
   - ⚠️ Staleness: 3hr delay on injury updates
   - ❌ Missing xWOBA for 12% of hitters
   - ⚠️ Park factors outdated (2024 data)

4. Risk Management: 9/10 ✅
   - Kelly fractions appropriate (5-15%)
   - Position sizing logic sound

5. Transparency: 7/10 ✅
   - Model inputs visible
   - Confidence intervals shown
   - Missing: Feature importance breakdown

CRITICAL ISSUES:
1. P1: Injury data 3hr stale (affects lineup decisions)
2. P1: Missing xWOBA impacts hitter valuations
3. P2: Park factors need 2025 update

RECOMMENDATIONS:
1. Implement real-time injury feed (P1)
2. Backfill xWOBA data (P1)
3. Update park factors (P2)
4. Add feature importance visualization (P3)

ALPHA GENERATION POTENTIAL: 7.5/10
The system shows measurable edge but data quality issues
are eroding 15-20% of potential returns.
```
