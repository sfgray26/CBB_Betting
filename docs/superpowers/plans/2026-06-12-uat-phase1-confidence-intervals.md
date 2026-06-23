# UAT Phase 1: Critical Data Fixes — Confidence Intervals on Projections

**Date:** 2026-06-12
**Priority:** P0 (Blocks optimal usage)
**Baseball IQ Impact:** 6.5/10 → 7.5/10

---

## Problem Statement

The Match Score shows huge variance without context: `16.04 ± 6.4` (±40%!). Managers cannot distinguish between:

- **High-confidence recommendations:** Consistent player, small variance
- **Low-confidence recommendations:** Volatile player, large variance

**UAT Finding:** "Match Score '16.04 ± 6.4' has huge variance (±40%). No explanation of uncertainty: Is ±6.4 due to sample size, model sensitivity, or injury risk?"

---

## Solution: Store and Display Projection Confidence Intervals

### Design

1. **Store percentiles in player_projections:**
   - `projection_p05`: 5th percentile (worst case)
   - `projection_p50`: median (expected)
   - `projection_p95`: 95th percentile (best case)

2. **Compute confidence interval width:**
   - `ci_width = projection_p95 - projection_p05`
   - `ci_pct = ci_width / projection_p50`

3. **Display with uncertainty bands:**
   - `"1.3 pts (range: 0.8-1.9 pts)"` for narrow CI
   - `"2.4 pts (range: 0.5-4.3 pts, high variance)"` for wide CI

4. **Add explanatory tooltips:**
   - `"Range based on recent variance (σ=1.2) and sample size (n=14 games)"`
   - `"95% confidence interval — 5% chance of <0.8 pts, 5% chance of >1.9 pts"`

---

## Implementation Tasks

### Task 2.1: Update Projection Schema

**File:** `backend/models.py` (PlayerProjection model)

Add percentile fields:
```python
projection_p05 = Column(Float, nullable=True)  # 5th percentile (worst case)
projection_p25 = Column(Float, nullable=True)  # 25th percentile (Q1)
projection_p50 = Column(Float, nullable=True)  # median (expected) - RENAME projection
projection_p75 = Column(Float, nullable=True)  # 75th percentile (Q3)
projection_p95 = Column(Float, nullable=True)  # 95th percentile (best case)
ci_variance = Column(Float, nullable=True)     # Coefficient of variation (σ/μ)
ci_sample_size = Column(Integer, nullable=True) # Number of games in sample
```

### Task 2.2: Compute Percentiles from Simulation Results

**File:** `backend/services/projection_engine.py` (compute_player_projections)

Modify to use existing Monte Carlo simulation results:
```python
def compute_projection_percentiles(simulation_results: List[float]) -> dict:
    """
    Compute percentiles from Monte Carlo simulation results.

    Returns:
      {
        'p05': np.percentile(results, 5),
        'p25': np.percentile(results, 25),
        'p50': np.percentile(results, 50),
        'p75': np.percentile(results, 75),
        'p95': np.percentile(results, 95),
        'ci_variance': np.std(results) / np.mean(results),
        'ci_sample_size': len(results),
      }
    """
```

### Task 2.3: Update Waiver Wire Response Schema

**File:** `backend/contracts.py` (WaiverPlayerOut)

Add CI fields:
```python
projection_p05: Optional[float] = None
projection_p50: Optional[float] = None  # Expected value (was 'projection')
projection_p95: Optional[float] = None
ci_variance: Optional[float] = None
ci_sample_size: Optional[int] = None
ci_explanation: Optional[str] = None
```

### Task 2.4: Update Frontend Display

**File:** `frontend/components/waiver/player-row.tsx`

Add CI badge:
```tsx
{player.ci_variance != null && player.ci_variance > 0.3 && (
  <Tooltip content={
    `High variance: ${player.ci_pct}% CI based on ${
      player.ci_sample_size
    } games. Recent performance volatile.`
  }>
    <Badge variant="warning">⚠ High Variance</Badge>
  </Tooltip>
)}

{player.projection_p50 != null && player.projection_p05 != null && player.projection_p95 != null && (
  <div className="text-xs text-text-tertiary">
    {player.projection_p50.toFixed(1)} pts
    <span className="text-text-muted">
      ({player.projection_p05.toFixed(1)}-{player.projection_p95.toFixed(1)})
    </span>
  </div>
)}
```

### Task 2.5: Add Low-Confidence Warnings

**File:** `frontend/components/waiver/waiver-wire.tsx`

Add warning banner if >30% of recommendations have high variance:
```tsx
{highVarianceCount > recommendations.length * 0.3 && (
  <Alert variant="warning">
    <AlertCircle className="h-4 w-4" />
    <AlertTitle>High Variance Alert</AlertTitle>
    <AlertDescription>
      {highVarianceCount} of {recommendations.length} recommendations have high variance
      (CI width > 30%). These players may be volatile. Consider roster stability.
    </AlertDescription>
  </Alert>
)}
```

---

## Testing Strategy

### Unit Tests
1. Test percentile computation: verify np.percentile works correctly
2. Test CI variance: verify σ/μ formula
3. Test threshold: CI > 0.3 = high variance

### Integration Tests
1. Run waiver wire endpoint: verify percentiles returned
2. Test high variance warning: if 40% have CI > 0.3, show banner

### Regression Tests
1. Ensure existing projection field works (backward compat)
2. Verify no performance regression (Monte Carlo runs at 5 AM)

---

## Rollout Plan

1. **Add migration** for new percentile fields
2. **Update projection engine** to compute percentiles from simulation
3. **Run manual sync** to populate percentile fields
4. **Deploy backend** with CI fields
5. **Deploy frontend** with CI display
6. **Monitor** for high-variance clusters (log if >50% recommendations have CI > 0.3)

---

## Success Metrics

- **Percentile coverage:** 100% of projections have p05, p50, p95
- **CI width distribution:** 60% have CI < 0.2, 30% have CI 0.2-0.3, 10% have CI > 0.3
- **User comprehension:** Survey shows >70% understand CI meaning

---

## Dependencies

- **Monte Carlo simulation results** (already exists in simulation_results table)
- **Related:** P0-2 (Model Stability) adds volatility detection

---

## Time Estimate

- Schema update + migration: 2 hours
- Percentile computation: 2 hours
- Schema updates: 2 hours
- Frontend display: 4 hours
- Testing: 3 hours

**Total: 13 hours (2 days)**