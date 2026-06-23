# Chandler Simpson Investigation — Railway Execution

**Date:** 2026-06-12
**Script:** `scripts/investigate_chandler_simpson.py`
**Status:** Ready for Railway execution

---

## Investigation Hypotheses

Chandler Simpson is marked **COLD** but has:
- **Delta Z:** +0.083
- **Expected Signal:** HOT (delta_z > HOT threshold +0.0568)
- **14d Score:** 4.5
- **30d Score:** 2.0

**Why COLD?** Possible causes:
1. **Stale persisted signal** — Signal computed earlier, delta_z changed but signal not updated
2. **Confidence gate** — Confidence too low, forced to COLD
3. **Level gate** — score_14d < 25th percentile, forced downgrade
4. **Dirty BDL mapping** — Using wrong BDL ID (802415 instead of 653217)

---

## What the Script Checks

1. **BDL ID mappings** — All three (653217, 802415, None)
2. **Momentum data** — Both BDL IDs, last 10 records
3. **Level gate** — Is score_14d < 25th percentile?
4. **Confidence gate** — Is confidence < 0.60?
5. **Manual classification** — Simulate classify_signal() to check expected signal
6. **Stale signal check** — Did signal stay COLD while delta_z changed?
7. **Player scores** — 14d vs 30d windows, compute delta_z manually

---

## Railway Execution Command

```bash
cd /app && python scripts/investigate_chandler_simpson.py
```

---

## Expected Output Format

```
================================================================================
CHANDLER SIMPSON INVESTIGATION
================================================================================
As of: 2026-06-12

Step 1: BDL ID mappings for Chandler Simpson...
--------------------------------------------------------------------------------
  BDL ID: 653217, Name: Chandler Simpson, MLBAM: 679823, Primary: true
  BDL ID: 802415, Name: Chandler Simpson, MLBAM: 679823, Primary: false
  BDL ID: None, Name: Chandler Simpson, MLBAM: None, Primary: false

Step 2: Momentum data for both BDL IDs...
--------------------------------------------------------------------------------

📊 BDL ID: 653217
--------------------------------------------------------------------------------
  Date:        2026-06-12
  Signal:      COLD
  Delta Z:     0.0826
  14d Z:       -0.8030 (score: 4.5, conf: 0.35)
  30d Z:       -0.8860 (score: 2.0, conf: 0.25)
  Confidence:  0.35
  Computed:    2026-06-12 09:00:21

📊 BDL ID: 802415
--------------------------------------------------------------------------------
  ⚠️  No momentum records for BDL ID: 802415

Step 3: Level gate check (score_14d < 25th percentile?)...
--------------------------------------------------------------------------------
Cohort Size:              423
25th Percentile (score_14d): 28.5
Mean (score_14d):          50.0
StdDev (score_14d):        28.0

Chandler Simpson (BDL 653217):
  score_14d:    4.5
  25th pct:     28.5
  confidence:   0.35
  signal:       COLD

❌ LEVEL GATE TRIGGERED: score_14d (4.5) < 25th percentile (28.5)
   This forces downgrade to COLD regardless of delta_z.

❌ CONFIDENCE GATE TRIGGERED: confidence (0.35) < 0.60
   This may force downgrade to COLD.

Step 4: Manual signal classification simulation...
--------------------------------------------------------------------------------
Cohort Delta Z Thresholds:
  90th percentile (SURGING):  0.2097
  70th percentile (HOT):      0.0568
  30th percentile (COLD):     -0.1494
  10th percentile (COLLAPSING): -0.4655
  Mean: -0.0804, StdDev: 0.2707

Chandler Simpson (BDL 653217):
  delta_z:       0.0826
  z_score_delta: 0.6037
  score_14d:     4.5
  confidence:    0.35
  Actual signal: COLD

Expected signal (delta_z only): HOT

❌ SIGNAL MISMATCH: Actual COLD != Expected HOT
   Root cause: Level gate or confidence gate overriding delta_z classification

Step 5: Stale signal check...
--------------------------------------------------------------------------------
Last 5 records for Chandler Simpson (BDL 653217):
  2026-06-12: delta_z=0.0826, signal=COLD, computed=2026-06-12 09:00:21
  2026-06-11: delta_z=-0.0154, signal=COLD, computed=2026-06-11 05:00:00
  2026-06-10: delta_z=-0.1234, signal=COLD, computed=2026-06-10 05:00:00

❌ POTENTIAL STALE SIGNAL: Signal stayed COLD while delta_z changed
   Previous delta_z: -0.0154
   Latest delta_z:    0.0826

Step 6: Player scores (14d vs 30d) for Chandler Simpson...
--------------------------------------------------------------------------------

📊 BDL ID: 653217
--------------------------------------------------------------------------------
  Window:     14 days
  Composite Z: -0.8030
  Score:      4.5
  Confidence: 0.35
  Computed:   2026-06-12 09:00:21

  Window:     30 days
  Composite Z: -0.8860
  Score:      2.0
  Confidence: 0.25
  Computed:   2026-06-12 09:00:00

  Computed Delta Z: -0.6830

⚠️  Computed Delta Z (-0.6830) != Stored Delta Z (0.0826)
   This suggests a data inconsistency.

================================================================================
INVESTIGATION COMPLETE
================================================================================
```

---

## Root Cause Determination

### Level Gate (score_14d < 25th percentile)
If score_14d (4.5) < 25th percentile (28.5):
- **Root cause:** Level gate forcing downgrade to COLD
- **Fix:** None — level gate is by design to prevent labeling weak players as HOT

### Confidence Gate (confidence < 0.60)
If confidence (0.35) < 0.60:
- **Root cause:** Confidence gate forcing downgrade to COLD
- **Fix:** None — confidence gate is by design to prevent labeling uncertain players as HOT

### Stale Signal
If signal stayed COLD while delta_z changed:
- **Root cause:** Signal not updated when delta_z changed
- **Fix:** Ensure signal recomputed on every momentum run

### Data Inconsistency
If computed delta_z (-0.6830) != stored delta_z (0.0826):
- **Root cause:** Bug in delta_z computation or storage
- **Fix:** Investigate compute_player_momentum() function

---

## After Running the Script

1. **Share the output** — I'll analyze the results
2. **Determine root cause** — Level gate, confidence gate, stale signal, or data inconsistency
3. **Decide next action** — Fix bug or document expected behavior

---

## Quick Railway Execution

```powershell
railway run python scripts/investigate_chandler_simpson.py
```

**Or open Railway shell and run:**
```bash
cd /app && python scripts/investigate_chandler_simpson.py
```

---

**Ready to run?** Execute the command and share the output.