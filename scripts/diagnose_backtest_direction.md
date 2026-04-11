# Task 8: direction_correct Field Analysis

**Date:** April 9, 2026
**Purpose**: Analysis of direction_correct field in backtest_results table

---

## Executive Summary

**Finding**: The `direction_correct` field is an **intentionally unimplemented placeholder feature**, not a bug.

**Status**: Field exists in schema and is populated (as NULL), but the backtesting harness explicitly sets it to NULL with no computation logic.

**Recommendation**: Document as "reserved for future use" - requires business logic definition and architectural design.

---

## Current State

### Table Schema
**Table**: `backtest_results` (models.py lines 1447-1493)
**Column**: `direction_correct = Column(Boolean, nullable=True)` (line 1484)
**Current Value**: 100% NULL

### Backtest Architecture

**BacktestInput** (backend/services/backtesting_harness.py lines 43-68):
```python
@dataclass
class BacktestInput:
    # Projections (from simulation_results p50 columns)
    proj_hr_p50: Optional[float]
    proj_rbi_p50: Optional[float]
    proj_sb_p50: Optional[float]
    proj_avg_p50: Optional[float]
    proj_k_p50: Optional[float]
    proj_era_p50: Optional[float]
    proj_whip_p50: Optional[float]

    # Actuals (aggregated from mlb_player_stats over the backtest window)
    actual_hr: Optional[float]
    actual_rbi: Optional[float]
    actual_sb: Optional[float]
    actual_avg: Optional[float]
    actual_k: Optional[float]
    actual_era: Optional[float]
    actual_whip: Optional[float]
```

**BacktestResult** (backend/services/backtesting_harness.py lines 72-100):
```python
@dataclass
class BacktestResult:
    # Per-stat MAE and RMSE
    mae_hr: Optional[float]
    rmse_hr: Optional[float]
    # ... (other stats)
    
    # Composite accuracy (mean of non-None stat MAEs)
    composite_mae: Optional[float]
    
    # Direction accuracy placeholder (populated by caller for multi-player sets)
    direction_correct: Optional[bool]  # ← Line 100
```

**Key Comment** (Line 99-100):
```python
# Direction accuracy placeholder (populated by caller for multi-player sets)
direction_correct: Optional[bool]
```

---

## Root Cause Analysis

### Intentional Design Choice

**Evidence from Code**:

1. **Explicit NULL Assignment** (backtesting_harness.py line 228):
```python
def evaluate_player(inp: BacktestInput) -> BacktestResult:
    """
    ...
    direction_correct is left as None here; the caller may populate it
    after ranking a multi-player cohort.
    """
    # ... compute MAE/RMSE ...
    return BacktestResult(
        # ... all fields ...
        direction_correct=None,  # ← Explicitly set to NULL
    )
```

2. **Placeholder Comment** (Line 178-179):
```python
direction_correct is left as None here; the caller may populate it
after ranking a multi-player cohort.
```

3. **No Caller Population** (daily_ingestion.py lines 2637-2661):
```python
for res in results:
    stmt = pg_insert(BacktestResultORM.__table__).values(
        # ... all fields ...
        direction_correct=res.direction_correct,  # ← Always NULL
        # ...
    )
```

**Analysis**: The backtesting harness was designed with `direction_correct` as a placeholder field that would be populated "after ranking a multi-player cohort" (per comment), but:
- No ranking logic exists in the codebase
- No caller populates this field
- The field has never been implemented

---

## Why "Direction" Cannot Be Computed

### Missing Data: No Margin Fields

The original task spec assumed `predicted_margin` and `actual_margin` fields would exist:
```python
direction_correct = (predicted_margin > 0 and actual_margin > 0) or 
                     (predicted_margin < 0 and actual_margin < 0)
```

**Reality**: These fields **do not exist** in the current schema:
- ❌ No `predicted_margin` column
- ❌ No `actual_margin` column
- ❌ No `projected_margin` column
- ❌ No `forecast_margin` column

### Available Data: Projections vs Actuals per Stat

The backtest only has:
- **Per-stat projections**: proj_hr_p50, proj_rbi_p50, proj_avg_p50, etc.
- **Per-stat actuals**: actual_hr, actual_rbi, actual_avg, etc.
- **Error metrics**: MAE, RMSE (absolute and squared error)

### What Would "Direction" Mean?

Without margin fields, "direction_correct" is ambiguous. Possible interpretations:

**Option 1**: Sign of projection-actual deviation
- **Logic**: `(proj - actual) > 0` → over-projected, `< 0` → under-projected
- **Problem**: This measures "bias direction", not "accuracy direction"
- **Business value**: Unclear - doesn't indicate if projection was "correct"

**Option 2**: Above/below league average
- **Logic**: Both proj and actual are above/below league mean
- **Problem**: No league averages stored in backtest_results
- **Data gap**: Would require league-wide aggregation

**Option 3**: Composite rank direction
- **Logic**: Did we correctly rank players within cohort?
- **Problem**: No ranking logic exists
- **Implementation gap**: Would require cohort ranking infrastructure

**Option 4**: Per-stat direction aggregation
- **Logic**: For each stat, was the sign of (proj - actual) correct? Average across stats.
- **Problem**: Unclear business logic - how to handle mixed results?
- **Complexity**: Requires defining what "direction" means per stat type

---

## Implementation Requirements

To properly implement `direction_correct`, we would need:

### Phase 1: Business Logic Definition
1. **Define "direction"** in business terms:
   - What are we measuring? (Accuracy? Bias? Rank?)
   - What is the success criterion?
   - How do we aggregate across multiple stats?

2. **Define computation formula**:
   - Mathematical definition of direction_correct
   - Edge cases: NULL values, zero margins, ties
   - Aggregation rules for multi-stat results

### Phase 2: Schema Changes
1. **Add margin columns** (if using margin-based approach):
   ```sql
   ALTER TABLE backtest_results 
   ADD COLUMN predicted_margin FLOAT,
   ADD COLUMN actual_margin FLOAT;
   ```

2. **Or add league averages** (if using above/below approach):
   ```sql
   ALTER TABLE backtest_results 
   ADD COLUMN league_avg_proj FLOAT,
   ADD COLUMN league_avg_actual FLOAT;
   ```

### Phase 3: Computation Logic
1. **Implement direction calculation**:
   - Add logic in `evaluate_player()` or new function
   - Handle NULL values appropriately
   - Test with real data

2. **Update backtesting pipeline**:
   - Integrate direction computation into `_run_backtesting()`
   - Ensure it runs for all players
   - Add verification/monitoring

### Phase 4: Backfill & Validation
1. **Backfill historical data**:
   - Compute direction_correct for existing rows
   - Validate results make business sense
   - Monitor distribution (should be roughly 50/50 if random)

### Estimated Effort: 16-24 hours
- Business logic definition: 2-4 hours
- Schema migration: 2-4 hours
- Implementation: 8-12 hours
- Testing & validation: 4-4 hours

---

## Comparison with Task 7 (ops/whip)

| Aspect | Task 7 (ops/whip) | Task 8 (direction_correct) |
|--------|-------------------|---------------------------|
| **Root Cause** | BDL API doesn't provide fields | Intentionally unimplemented placeholder |
| **Solution Complexity** | Simple computation from existing data | Requires business logic definition |
| **Schema Changes** | None required | Likely required (margins or averages) |
| **Data Available** | Yes (obp, slg, walks_allowed, hits_allowed, ip) | Partially (projections & actuals, but no margins) |
| **Formula Clarity** | Clear (OPS = OBP + SLG, WHIP = (BB+H)/IP) | Ambiguous (undefined) |
| **Effort Estimate** | 2-3 hours | 16-24 hours |
| **Risk Level** | Low (well-defined computation) | High (requires business logic) |

---

## Options

### Option 1: Implement Full Direction Logic ✅ RECOMMENDED
- **Effort**: 16-24 hours
- **Approach**: Define business logic, add schema, implement computation
- **Pros**: Completes the intended feature
- **Cons**: Requires significant design work

### Option 2: Document as Unimplemented 🔶 ACCEPTABLE
- **Effort**: 30-60 minutes
- **Approach**: Document current state, mark as "reserved for future use"
- **Pros**: Honest about current state, no speculation
- **Cons**: Feature remains incomplete

### Option 3: Remove Field ❌ NOT RECOMMENDED
- **Effort**: 2-3 hours
- **Approach**: Drop direction_correct column from schema
- **Pros**: Removes ambiguity
- **Cons**: Loses placeholder for future implementation

---

## Recommendation

**Document as Unimplemented** (Option 2)

**Justification**:
1. **Not a bug**: Field is intentionally set to NULL per design
2. **Architecture gap**: Requires business logic definition that doesn't exist
3. **Low priority**: Backtest accuracy metrics (MAE, RMSE) are already computed and working
4. **Risk**: Implementing without clear business logic could introduce misleading metrics
5. **Effort**: 16-24 hours vs 30 minutes (high opportunity cost)

**Next Steps**:
1. Create `docs/analytics-roadmap.md` documenting direction_correct as unimplemented
2. Document requirements for future implementation:
   - Business logic definition
   - Schema changes needed
   - Computation approach
   - Effort estimate
3. Commit documentation
4. Move to Task 9 (similar documentation task)

---

## Evidence Summary

**Code Inspection**:
- ✅ BacktestInput has proj_* and actual_* fields (lines 51-66)
- ✅ BacktestResult has direction_correct field (line 100)
- ✅ evaluate_player() sets direction_correct=None explicitly (line 228)
- ✅ Comment says "Direction accuracy placeholder" (line 99)
- ✅ No margin fields exist in schema
- ✅ No caller populates direction_correct

**Schema Verification**:
```sql
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'backtest_results' 
ORDER BY ordinal_position;

-- Result: No margin columns found
```

**Conclusion**: This is an intentionally unimplemented feature, not a data quality bug.

---

**Prepared by**: Claude Code (Master Architect)
**Preparation Date**: April 9, 2026 6:15 PM EDT
**Status**: Ready for Task 8 completion via documentation
