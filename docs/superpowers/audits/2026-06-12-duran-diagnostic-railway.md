# Duran Momentum Diagnostic — Railway Execution

**Date:** 2026-06-12
**Script:** `scripts/diagnostic_momentum_cold.py`
**Status:** Ready to run

---

## Diagnostic Script Created

**File:** `scripts/diagnostic_momentum_cold.py`

**What it does:**
1. Finds BDL player IDs for Jarren Duran and Chandler Simpson
2. Retrieves last 5 momentum records for each
3. Computes cohort distribution (mean, std, percentiles)
4. Calculates z_score_delta for each player
5. Compares to percentile thresholds (90th, 70th, 30th, 10th)
6. Fetches 14d vs 30d player_scores
7. Checks data freshness (last computation timestamp)

**Output:**
- Momentum data for both players
- Cohort distribution (300+ hitters)
- Z-score delta analysis
- Root cause determination (stale data vs algorithm bug)

---

## Railway Execution Command

```bash
cd /app && python scripts/diagnostic_momentum_cold.py
```

**Or via Railway CLI:**
```powershell
railway run python scripts/diagnostic_momentum_cold.py
```

---

## Expected Output Format

```
================================================================================
MOMENTUM DIAGNOSTIC: COLD MISCLASSIFICATION
================================================================================
As of: 2026-06-12

Step 1: Finding BDL player IDs...
--------------------------------------------------------------------------------
Found: Jarren Duran -> BDL ID: 12345
Found: Chandler Simpson -> BDL ID: 67890

Step 2: Momentum records for target players...
--------------------------------------------------------------------------------

📊 Jarren Duran (BDL ID: 12345)
--------------------------------------------------------------------------------
  Date:      2026-06-11
  Signal:    COLD
  Delta Z:   -0.320
  14d Z:     -0.450 (score: 18.2, conf: 0.72)
  30d Z:     -0.130 (score: 25.3, conf: 0.75)
  Confidence: 0.72
  Computed:  2026-06-11 05:00:00

Step 3: Cohort distribution analysis (hitters)...
--------------------------------------------------------------------------------
Most recent momentum date: 2026-06-11

Cohort Size:      312
Delta Z Mean:     0.0500
Delta Z StdDev:   0.1800
Delta Z Range:    [-0.4500, 0.6800]

Percentiles (delta_z):
  90th percentile (SURGING threshold):  0.2800
  70th percentile (HOT threshold):      0.1200
  30th percentile (COLD threshold):     -0.0800
  10th percentile (COLLAPSING threshold): -0.2200

Step 4: Z-score delta analysis for target players...
--------------------------------------------------------------------------------

Jarren Duran:
  Delta Z:         -0.3200
  Z-score delta:   -2.0556 (standard deviations from mean)

  → Z-score < 10th percentile → Should be COLLAPSING ✅

Step 5: Player scores (14d vs 30d) for target players...
--------------------------------------------------------------------------------

Jarren Duran:
--------------------------------------------------------------------------------
  Window:     14 days
  Composite Z: -0.450
  Score:      18.2
  Confidence: 0.72
  Computed:   2026-06-11 05:00:00

  Window:     30 days
  Composite Z: -0.130
  Score:      25.3
  Confidence: 0.75
  Computed:   2026-06-11 05:00:00

  Computed Delta Z: -0.3200

Step 6: Data freshness check...
--------------------------------------------------------------------------------
Last momentum computation: 2026-06-11 05:00:00
Current time: 2026-06-12 17:30:00
Lag: 36.5 hours

⚠️  WARNING: Data is stale (>12 hours old)
   Recent surges may not be captured.

================================================================================
DIAGNOSTIC COMPLETE
================================================================================
```

---

## Root Cause Determination

### If Lag > 12 Hours: Stale Data
- Last momentum computation was >12 hours ago
- Recent surge (last 1-2 days) not captured in 14d window
- **Fix:** Add near-real-time momentum computation (12 PM ET run)

### If Lag < 6 Hours: Algorithm Correct
- Last computation was recent
- Player's delta_z truly is below 30th percentile
- **Fix:** None — algorithm working correctly

### If Lag 6-12 Hours: Partial Staleness
- Some recent performance captured, but not all
- **Fix:** Add "Data from X AM" freshness badge

---

## After Running the Script

1. **Share the output** — I'll analyze the results
2. **Determine root cause** — Stale data vs algorithm bug
3. **Decide next action** — Near-real-time computation or EMA weighting

---

## Quick Railway Execution

```powershell
railway run python scripts/diagnostic_momentum_cold.py
```

**Or open Railway shell and run:**
```bash
cd /app && python scripts/diagnostic_momentum_cold.py
```

---

**Ready to run?** Execute the command and share the output.