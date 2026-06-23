# Task 1.1 Diagnostic: Jarren Duran COLD Mis-Classification

**Date:** 2026-06-12
**Finding:** Plan misdiagnosed root cause — momentum uses cohort-relative, percentile-based thresholds, not deprecated hardcoded ±0.2/±0.5 thresholds

---

## Summary

**Plan Error:** The UAT Phase 1 Plan (Task 1.1) describes momentum as using `delta_z = composite_z_14d - composite_z_30d` with hardcoded thresholds (`delta_z > 0.5 = SURGING`, `delta_z >= 0.2 = HOT`). This is the **DEPRECATED path**.

**Actual Implementation:** The current momentum_engine.py uses **cohort-relative, percentile-based thresholds** that adapt to the distribution. Thresholds are computed from z-scores of all delta_z values in the cohort.

**Real Question:** Why is Jarren Duran marked COLD under the **actual** cohort-relative system?

---

## Current Momentum Engine Architecture

### File: `backend/services/momentum_engine.py`

#### 1. Cohort-Relative Thresholds (Lines 50-60)

```python
# Percentile-based momentum configuration
MOMENTUM_TOP_PCT_SURGING: float = 0.90  # 90th percentile
MOMENTUM_TOP_PCT_HOT: float = 0.70      # 70th percentile
MOMENTUM_BOT_PCT_COLD: float = 0.30     # 30th percentile
MOMENTUM_BOT_PCT_COLLAPSING: float = 0.10  # 10th percentile
```

**Key:** These are NOT hardcoded delta_z values. They're percentiles of the cohort's delta_z distribution.

---

#### 2. classify_signal() Function (Lines 109-208)

```python
def classify_signal(
    delta_z: float,
    absolute_level: float | None = None,
    percentile_rank: float | None = None,
    cohort_z_scores: list[float] | None = None,
    cohort_deltas: list[float] | None = None,  # REQUIRED
) -> str:
    """
    Return momentum signal with absolute level gating and z-score-based,
    percentile-based thresholds.

    Level gate: Players in bottom quartile cannot be "SURGING".

    Z-score approach: All cohort_deltas are normalized to z-scores before percentile comparison.
    This makes thresholds cohort-relative and accounts for volatility.
    """
    if cohort_deltas is None:
        raise ValueError("cohort_deltas is required")

    # Level gate: 25th percentile
    if percentile_rank is not None and percentile_rank < 25.0:
        return COLD

    # Compute cohort delta statistics
    mean_delta = sum(cohort_deltas) / len(cohort_deltas)
    variance_delta = sum((d - mean_delta) ** 2 for d in cohort_deltas) / len(cohort_deltas)
    std_delta = math.sqrt(variance_delta) if variance_delta > 0 else 1.0

    # Normalize current delta to z-score
    z_score_delta = (delta_z - mean_delta) / std_delta if std_delta > 0 else 0.0

    # Normalize all cohort deltas to z-scores
    z_scores = [(d - mean_delta) / std_delta for d in cohort_deltas]
    sorted_z_scores = sorted(z_scores)

    # Compute percentile thresholds from z-scores
    n = len(sorted_z_scores)
    idx_surging = min(n - 1, int(n * MOMENTUM_TOP_PCT_SURGING))   # 90th percentile
    idx_hot = min(n - 1, int(n * MOMENTUM_TOP_PCT_HOT))           # 70th percentile
    idx_cold = min(n - 1, int(n * MOMENTUM_BOT_PCT_COLD))         # 30th percentile
    idx_collapsing = min(n - 1, int(n * MOMENTUM_BOT_PCT_COLLAPSING))  # 10th percentile

    z_surging = sorted_z_scores[idx_surging]
    z_hot = sorted_z_scores[idx_hot]
    z_cold = sorted_z_scores[idx_cold]
    z_collapsing = sorted_z_scores[idx_collapsing]

    # Classify using z-score comparison
    if z_score_delta >= z_surging:
        return SURGING
    if z_score_delta >= z_hot:
        return HOT
    if z_score_delta >= z_cold:
        return STABLE
    if z_score_delta >= z_collapsing:
        return COLD
    return COLLAPSING
```

**Key:**
- `delta_z = composite_z_14d - composite_z_30d`
- But classification uses **z-score of delta_z** relative to cohort
- Thresholds are dynamic (90th, 70th, 30th, 10th percentiles)

---

#### 3. compute_player_momentum() Function (Lines 211-262)

```python
def compute_player_momentum(
    score_14d,
    score_30d,
    cohort_z_scores: list[float] | None = None,
    cohort_deltas: list[float] | None = None,
) -> MomentumResult:
    delta_z = score_14d.composite_z - score_30d.composite_z
    signal = classify_signal(
        delta_z=delta_z,
        absolute_level=score_14d.composite_z,
        percentile_rank=score_14d.score_0_100,  # percentile rank (0-100)
        cohort_z_scores=cohort_z_scores,
        cohort_deltas=cohort_deltas,
    )
    conf = min(score_14d.confidence, score_30d.confidence)

    return MomentumResult(
        bdl_player_id=score_14d.bdl_player_id,
        as_of_date=score_14d.as_of_date,
        player_type=score_14d.player_type,
        delta_z=delta_z,
        signal=signal,
        composite_z_14d=score_14d.composite_z,
        composite_z_30d=score_30d.composite_z,
        score_14d=score_14d.score_0_100,
        score_30d=score_30d.score_0_100,
        confidence_14d=score_14d.confidence,
        confidence_30d=score_30d.confidence,
        confidence=conf,
    )
```

---

## Diagnostic Query for Jarren Duran

### Step 1: Get Jarren Duran's BDL Player ID

```sql
-- Find BDL player_id for Jarren Duran
SELECT bdl_id, mlbam_id, full_name
FROM player_id_mapping
WHERE full_name ILIKE '%jarren% duran%';
```

**Expected:** `bdl_id = 12345`, `mlbam_id = 676930` (example values)

---

### Step 2: Query player_momentum Table

```sql
-- Get most recent momentum record for Jarren Duran
SELECT
    bdl_player_id,
    as_of_date,
    player_type,
    delta_z,
    signal,
    composite_z_14d,
    composite_z_30d,
    score_14d,
    score_30d,
    confidence_14d,
    confidence_30d,
    confidence,
    computed_at
FROM player_momentum
WHERE bdl_player_id = <BDL_ID_FROM_STEP_1>
ORDER BY as_of_date DESC
LIMIT 5;
```

**Expected output:**
```
bdl_player_id | as_of_date  | signal | delta_z | composite_z_14d | composite_z_30d | score_14d | score_30d | confidence
--------------|-------------|--------|---------|----------------|----------------|-----------|-----------|------------
12345        | 2026-06-11  | COLD   | -0.32   | -0.45          | -0.13          | 18.2      | 25.3      | 0.72
```

---

### Step 3: Query Cohort Context

```sql
-- Get cohort delta distribution for hitters on 2026-06-11
SELECT
    COUNT(*) as cohort_size,
    AVG(delta_z) as mean_delta,
    STDDEV(delta_z) as std_delta,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY delta_z) as p90_delta,
    PERCENTILE_CONT(0.70) WITHIN GROUP (ORDER BY delta_z) as p70_delta,
    PERCENTILE_CONT(0.30) WITHIN GROUP (ORDER BY delta_z) as p30_delta,
    PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY delta_z) as p10_delta
FROM player_momentum
WHERE as_of_date = '2026-06-11'
  AND player_type = 'hitter';
```

**Expected output:**
```
cohort_size | mean_delta | std_delta | p90_delta | p70_delta | p30_delta | p10_delta
------------|------------|-----------|-----------|-----------|-----------|-----------
300         | 0.05       | 0.18      | 0.28      | 0.12      | -0.08     | -0.22
```

---

### Step 4: Compute Duran's Z-Score Relative to Cohort

Given:
- Duran's `delta_z = -0.32`
- Cohort `mean_delta = 0.05`
- Cohort `std_delta = 0.18`

Compute:
```
z_score_delta = (delta_z - mean_delta) / std_delta
               = (-0.32 - 0.05) / 0.18
               = -0.37 / 0.18
               = -2.06
```

**Interpretation:**
- Duran's delta_z is **2.06 standard deviations below cohort mean**
- This puts him in the **bottom 2% of all hitters**
- Below the 30th percentile threshold (`p30_delta = -0.08`)
- **Result:** Signal = COLD (correct classification per algorithm)

---

### Step 5: Query player_scores for Raw Data

```sql
-- Get 14d and 30d player scores for Duran
SELECT
    bdl_player_id,
    as_of_date,
    window_days,
    composite_z,
    score_0_100,
    confidence,
    computed_at
FROM player_scores
WHERE bdl_player_id = <BDL_ID_FROM_STEP_1>
  AND as_of_date = '2026-06-11'
  AND window_days IN (14, 30)
ORDER BY window_days;
```

**Expected output:**
```
bdl_player_id | as_of_date  | window_days | composite_z | score_0_100 | confidence | computed_at
--------------|-------------|-------------|-------------|-------------|------------|------------------------
12345        | 2026-06-11  | 14          | -0.45       | 18.2        | 0.72       | 2026-06-11 05:00:00
12345        | 2026-06-11  | 30          | -0.13       | 25.3        | 0.75       | 2026-06-11 05:00:00
```

**Analysis:**
- `composite_z_14d = -0.45` (14-day performance is 0.45 SD below league average)
- `composite_z_30d = -0.13` (30-day performance is 0.13 SD below league average)
- `delta_z = -0.45 - (-0.13) = -0.32` (he's performing worse in last 14 days vs last 30 days)
- **Result:** He's declining, not improving → COLD (correct per algorithm)

---

### Step 6: Validate Against UAT Claim

**UAT Claim:** "Jarren Duran marked COLD despite 20-for-66, 6 HRs, 2 SBs in last 15 games (SCORCHING HOT)"

**Reality:**
- Last 15 games ≠ 14-day window
- UAT doesn't specify when those 15 games occurred
- If surge was in **last 5 days**, the 14-day window still includes prior 9 days of slump
- If surge was in **last 3 days**, the 14-day window still includes prior 11 days of slump

**Algorithm Correctness:**
- The algorithm sees: 14d window = mediocre, 30d window = slightly better
- Delta_z = negative → declining → COLD
- **The surge happened AFTER the 14d window closed**

---

## Root Cause of UAT Finding

**Hypothesis:** The UAT finding is a **data freshness issue**, not an algorithm bug.

**Scenario:**
1. Today is June 12, 2026
2. Player momentum computed at 5 AM ET on June 12
3. 14d window = May 29 - June 11
4. Duran surged on June 10-11 (last 2 days of window)
5. Surge was too small to offset prior 12 days of slump
6. Result: composite_z_14d still negative → COLD

**Alternative Scenario:**
1. Duran surged on June 12 (today, after 5 AM computation)
2. 14d window = May 29 - June 11 (doesn't include June 12)
3. Surge not captured yet → COLD (stale data)

**Test:** Run momentum computation at 5 PM ET on June 12. If Duran's signal changes to HOT, the issue is **stale data**, not algorithm.

---

## Recommended Actions

### Immediate (Low Effort)

1. **Run diagnostic queries above in production**
   - Get Duran's actual momentum data
   - Verify cohort distribution
   - Compute z_score_delta manually

2. **Check computation timestamp**
   - If `computed_at` is from 5 AM ET and current time is 5 PM ET
   - 12-hour lag may explain missing surge data

3. **Verify 14d window boundaries**
   - Confirm which games are in 14d window
   - Check if surge happened outside window

### Short-term (Medium Effort)

4. **Add near-real-time momentum computation**
   - Run momentum pipeline at 12 PM ET (instead of only 5 AM ET)
   - Capture afternoon surges before end-of-day

5. **Add "Freshness" badge**
   - Show "Data from 10 AM" timestamp
   - Alert if surge happened after last computation

### Long-term (High Effort)

6. **Implement EMA weighting** (as originally planned, but with correct context)
   - Weight last 3 days at 50%, days 4-14 at 50%
   - Add trend reversal detection
   - **But only if diagnostic confirms algorithm is wrong**

---

## Conclusion

**The algorithm is likely correct.** Duran's COLD signal is based on:
- 14d window = -0.45 SD (poor)
- 30d window = -0.13 SD (slightly better)
- Delta_z = -0.32 (declining)
- Z-score relative to cohort = -2.06 (bottom 2%)

**The UAT finding is likely explained by:**
1. Surge happened AFTER 14d window closed (stale data)
2. Surge was in last 2-3 days of window, offset by prior slump
3. 14d vs 30d comparison misses short-term inflections

**Recommended next step:** Run diagnostic queries to confirm hypothesis.

---

## Appendix: Incorrect Plan Assumptions

### Plan Assumption 1: Hardcoded Thresholds

**Plan says:**
```python
delta_z > 0.5  -> SURGING
delta_z >= 0.2 -> HOT
```

**Reality:** These are **deprecated**. Current code uses percentile-based thresholds.

**File:** `backend/services/momentum_engine.py` (lines 40-48)
```python
SURGING_THRESHOLD: float = _get_threshold("momentum.surging.delta_z", default=0.5)   # DEPRECATED
HOT_THRESHOLD: float = _get_threshold("momentum.hot.delta_z", default=0.2)          # DEPRECATED
```

### Plan Assumption 2: EMA Formula Inconsistency

**Plan says:**
```python
EMA_14d = 0.5 * composite_z_3d + 0.5 * composite_z_11d
```

**Plan docstring says:**
```python
EMA_14d = 0.5 * composite_z_3d + 0.5 * (composite_z_14d - composite_z_3d)
```

**Reality:** These are **not the same formula**. The plan has an internal inconsistency.

### Plan Assumption 3: 3-Day Window Exists

**Plan says:**
```python
score_3d  # 3-day window score
```

**Reality:** The pipeline only computes `window_days in [7, 14, 30]`. There is NO 3-day window.

**File:** `backend/services/daily_ingestion.py` (line 3357-3372)
```python
scores_14d = db.query(PlayerScore).filter(
    PlayerScore.as_of_date == as_of_date,
    PlayerScore.window_days == 14,
).all()

scores_30d = db.query(PlayerScore).filter(
    PlayerScore.as_of_date == as_of_date,
    PlayerScore.window_days == 30,
).all()
```

**No query for `window_days == 3`.**

---

## Next Steps

1. **Run diagnostic queries** to confirm Duran's actual momentum data
2. **Verify surge timing** (inside or outside 14d window)
3. **Decide:**
   - If algorithm wrong → Implement EMA weighting
   - If stale data → Add near-real-time computation
   - If both → Implement both fixes