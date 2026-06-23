# UAT Phase 1: Critical Data Fixes — Momentum Classification Engine

**Date:** 2026-06-12
**Priority:** P0 (Blocks optimal usage)
**Baseball IQ Impact:** 6.5/10 → 8.0/10

---

## Problem Statement

The current momentum classification (HOT/COLD) uses a static 14d vs 30d window comparison that misses trend reversals. This causes critical misclassifications:

- **Jarren Duran marked COLD** despite 20-for-66, 6 HRs in last 15 games (SCORCHING HOT)
- **Chandler Simpson marked HOT** despite returning from 3-game absence (volatility risk)
- **7-day rolling window** captures bad days within longer good stretches

**Root Cause:** `delta_z = composite_z_14d - composite_z_30d` compares two static windows. If a player slumped in Days 1-7 but surged in Days 8-14, the 14d composite_z is still depressed.

---

## Solution: EMA-Based Momentum with Trend Reversal

### New Algorithm

1. **Compute EMA-weighted composite_z:**
   - Last 3 days: 50% weight
   - Days 4-14: 50% weight
   - Formula: `EMA_14d = 0.5 * composite_z_3d + 0.5 * composite_z_11d`

2. **Detect trend reversal:**
   - `recent_trend = EMA_3d - EMA_11d`
   - If `recent_trend > 0.3` → Positive surge detected
   - If `recent_trend < -0.3` → Negative collapse detected

3. **Override logic:**
   - If `recent_trend > 0.3` AND current signal in [STABLE, COLD] → Upgrade to HOT
   - If `recent_trend < -0.3` AND current signal in [STABLE, HOT] → Downgrade to COLD

4. **Add trend indicator:**
   - Store `trend_direction` field: 'improving', 'stable', 'declining'
   - Store `trend_strength` field: 0.0-1.0 scale

---

## Implementation Tasks

### Task 1.1: Update Momentum Schema

**File:** `backend/models.py` (PlayerMomentum model)

Add new fields:
```python
trend_direction = Column(String(10), nullable=True)  # 'improving', 'stable', 'declining'
trend_strength = Column(Float, nullable=True)        # 0.0-1.0 scale
ema_14d = Column(Float, nullable=True)               # EMA-weighted composite_z_14d
```

### Task 1.2: Implement EMA Computation

**File:** `backend/services/momentum_engine.py`

Add new function:
```python
def compute_ema_weighted_momentum(
    score_14d,
    score_3d,  # NEW: 3-day window score
    score_30d,
    cohort_z_scores: list[float] | None = None,
    cohort_deltas: list[float] | None = None,
) -> MomentumResult:
    """
    Compute momentum with EMA weighting and trend reversal detection.
    
    Formula:
      EMA_14d = 0.5 * composite_z_3d + 0.5 * (composite_z_14d - composite_z_3d)
      recent_trend = composite_z_3d - (composite_z_14d - composite_z_3d)
    
    Trend reversal:
      If recent_trend > 0.3 and signal in [STABLE, COLD] → Upgrade to HOT
      If recent_trend < -0.3 and signal in [STABLE, HOT] → Downgrade to COLD
    """
```

### Task 1.3: Update Daily Ingestion Pipeline

**File:** `backend/services/daily_ingestion.py` (_compute_player_momentum)

Modify to:
1. Query player_scores WHERE window_days = 3 (NEW)
2. Query player_scores WHERE window_days = 14
3. Query player_scores WHERE window_days = 30
4. Call `compute_ema_weighted_momentum()` instead of `compute_player_momentum()`
5. Upsert new fields: trend_direction, trend_strength, ema_14d

### Task 1.4: Update Frontend Display

**File:** `frontend/lib/types.ts` (PlayerCard, MomentumDisplay)

Add trend indicators:
```typescript
interface MomentumDisplay {
  signal: 'SURGING' | 'HOT' | 'STABLE' | 'COLD' | 'COLLAPSING'
  trendDirection: 'improving' | 'stable' | 'declining'
  trendStrength: number  // 0.0-1.0
  // Display: HOT ↑ (improving, 0.85 strength)
}
```

Add icons:
- ↑ for improving (green)
- → for stable (gray)
- ↓ for declining (red)

### Task 1.5: Add Trend Reversal Warning

**File:** `frontend/components/waiver/player-row.tsx`

Add tooltip logic:
```tsx
{player.trend_direction === 'improving' && (
  <Tooltip content="Trend reversal detected! Last 3 days surge overrides prior 7-day slump">
    <TrendingUp className="h-3 w-3 text-green-500" />
  </Tooltip>
)}
```

---

## Testing Strategy

### Unit Tests
1. Test EMA computation: verify 0.5 * 3d + 0.5 * 11d formula
2. Test trend reversal: if 3d = +0.8, 11d = -0.4 → recent_trend = 1.2 → upgrade to HOT
3. Test negative reversal: if 3d = -0.8, 11d = +0.4 → recent_trend = -1.2 → downgrade to COLD

### Integration Tests
1. Run pipeline with Jarren Duran data: verify signal = HOT (not COLD)
2. Run pipeline with slump scenario: verify signal = COLD with declining trend

### Regression Tests
1. Ensure existing hot/cold players not affected if trend stable
2. Verify cohort-relative thresholds still apply (no regression)

---

## Rollout Plan

1. **Add migration** for new fields (trend_direction, trend_strength, ema_14d)
2. **Deploy backend** with EMA computation
3. **Run manual sync** to populate new fields for historical data
4. **Deploy frontend** with trend indicators
5. **Monitor** for anomalous signals (log warnings if surge > 0.5)

---

## Success Metrics

- **Trend reversal accuracy:** >85% of players with recent_trend > 0.3 show HOT in next 7 days
- **False positive rate:** <10% of upgrades to HOT are not sustained
- **Jarren Duran test case:** Signal = HOT (not COLD)

---

## Dependencies

- **None** — can implement independently
- **Related:** P0-2 (Model Stability) adds confidence intervals

---

## Time Estimate

- Schema update + migration: 2 hours
- EMA computation implementation: 3 hours
- Pipeline update: 2 hours
- Frontend display: 3 hours
- Testing: 4 hours

**Total: 14 hours (2-3 days)**