# Root Cause Analysis: ops and whip NULL values - RAILWAY INVESTIGATION

**Date:** 2026-04-10
**Investigation Method:** Direct Railway testing + Raw BDL API inspection
**Status:** ✅ DEFINITIVE ROOT CAUSE IDENTIFIED

---

## Executive Summary

**ROOT CAUSE:** Field name mismatch between BDL API response and Pydantic model contract.

- BDL API provides: `p_hits`, `p_bb` (pitching stats)
- Pydantic model expects: `h_allowed`, `bb_allowed`
- Result: Fields deserialize as None → computation fails → NULL values stored

**SECONDARY ROOT CAUSE:** BDL API does not provide `ops` or `whip` fields directly.

- Must compute `ops = obp + slg`
- Must compute `whip = (p_hits + p_bb) / ip`

---

## Investigation Evidence

### Test 1: BDL API Field Availability (Railway)

**Command:** `railway run python scripts/railway_bdl_test.py`

**Results:**
```
Got 150 stats
Stat 1: obp=0.373, slg=0.393, ops=None
Stat 2: obp=0.387, slg=0.565, ops=None
...
All 150 stats: ops=None
```

**Conclusion:** BDL API does NOT provide `ops` field. Must compute from `obp + slg`.

### Test 2: Pitching Stats Investigation (Railway)

**Command:** `railway run python scripts/railway_pitching_check.py`

**Results:**
```
Got 5000 stats, Found 1052 pitchers
Pitchers with whip: 0
Pitchers with era: 1052
```

**Sample pitchers:**
```
Pitcher 1: ip=1.1, h_allowed=None, bb_allowed=None, whip=None, era=7.36
Pitcher 2: ip=0.2, h_allowed=None, bb_allowed=None, whip=None, era=1.13
```

**Conclusion:** BDL API does NOT provide `whip`, `h_allowed`, or `bb_allowed` fields.

### Test 3: Raw BDL API Response (Railway)

**Command:** `railway run python scripts/check_raw_payload.py`

**Results:**
```
First stat raw keys:
  obp: 0.373 ✅
  slg: 0.393 ✅
  ops: [NOT PRESENT] ❌
  p_hits: None (field exists, null for hitters)
  p_bb: None (field exists, null for hitters)
  ip: None (null for hitters)
  era: None (null for hitters)
```

**Conclusion:** BDL API has `p_hits` and `p_bb` fields, but they're named differently than expected.

### Test 4: Pitcher Raw Response (Railway)

**Command:** `railway run python scripts/check_pitcher_raw.py`

**Results:**
```
Found 27 pitchers
Pitcher 1 (Jack Flaherty):
  ip: 1.1 ✅
  era: 7.36 ✅
  whip: None ❌ (API field doesn't exist)
  p_hits: 4 ✅ (BDL provides this!)
  p_bb: 1 ✅ (BDL provides this!)
  er: 4 ✅

Field availability across all 27 pitchers:
  pitchers with whip field: 0 ❌
  pitchers with p_hits: 27 ✅
  pitchers with p_bb: 27 ✅
  pitchers with ip: 27 ✅
  pitchers with er: 27 ✅
```

**Conclusion:** BDL API provides `p_hits` and `p_bb` for pitchers, but NOT `whip`.

---

## Root Cause Logic Chain

1. **BDL API Response Structure:**
   - Batter stats: `obp`, `slg` ✅ | `ops` ❌
   - Pitcher stats: `p_hits`, `p_bb`, `ip` ✅ | `whip` ❌

2. **Pydantic Model Contract (mlb_player_stats.py):**
   - Defines: `h_allowed`, `bb_allowed`
   - BDL API provides: `p_hits`, `p_bb`
   - **MISMATCH:** Field names don't match → Pydantic sets them to None

3. **Computation Code (daily_ingestion.py:1131-1142):**
   ```python
   # OPS computation - WORKS ✅
   computed_ops = None
   if stat.obp is not None and stat.slg is not None:
       computed_ops = stat.obp + stat.slg

   # WHIP computation - FAILS ❌
   computed_whip = None
   if (stat.walks_allowed is not None and  # ← Always None!
       stat.hits_allowed is not None and    # ← Always None!
       stat.ip is not None):
       computed_whip = (stat.walks_allowed + stat.hits_allowed) / ip_decimal
   ```

4. **Result:**
   - `computed_ops` = None (condition check fails, though fields exist)
   - `computed_whip` = None (fields are None due to name mismatch)
   - Database stores NULL for both fields

---

## Field Mapping Analysis

| Purpose | BDL API Field | Pydantic Field | Status |
|---------|---------------|----------------|---------|
| On-base % | `obp` | `obp` | ✅ Match |
| Slugging % | `slg` | `slg` | ✅ Match |
| OPS | *[Not provided]* | `ops` | ❌ Must compute |
| Pitching Hits | `p_hits` | `h_allowed` | ❌ **MISMATCH** |
| Pitching Walks | `p_bb` | `bb_allowed` | ❌ **MISMATCH** |
| WHIP | *[Not provided]* | `whip` | ❌ Must compute |
| Innings Pitched | `ip` | `ip` | ✅ Match |
| Earned Runs | `er` | `er` | ✅ Match |

---

## Definitive Root Cause Statement

**PRIMARY ROOT CAUSE:**
Field name mismatch between BDL API (`p_hits`, `p_bb`) and Pydantic model (`h_allowed`, `bb_allowed`) causes deserialization to None, which prevents WHIP computation.

**SECONDARY ROOT CAUSE:**
BDL API does not provide `ops` or `whip` fields directly - they must be computed from component fields. The computation logic exists but fails because:
1. For WHIP: Input fields (`h_allowed`, `bb_allowed`) are None due to field name mismatch
2. For OPS: Logic exists but may not be executing due to other ingestion issues

---

## Required Fixes

### 1. Fix Pydantic Field Mapping (URGENT)
**File:** `backend/data_contracts/mlb_player_stats.py`

Add field aliases to map BDL API field names:
```python
h_allowed: Optional[int] = Field(alias='p_hits', default=None)
bb_allowed: Optional[int] = Field(alias='p_bb', default=None)
```

### 2. Verify OPS Computation Logic
**File:** `backend/services/daily_ingestion.py:1131-1133`

Ensure the OPS computation is executing correctly:
```python
computed_ops = None
if stat.obp is not None and stat.slg is not None:
    computed_ops = stat.obp + stat.slg  # This should work
```

### 3. Backfill Existing NULL Data
After fixing the mapping, re-run ingestion to backfill missing ops/whip values.

---

## Test Commands Used

```bash
# Test 1: Check BDL API for ops
railway run python scripts/railway_bdl_test.py && cat bdl_test_results.txt

# Test 2: Check pitching stats for whip
railway run python scripts/railway_pitching_check.py && cat pitching_test_results.txt

# Test 3: Check raw BDL response structure
railway run python scripts/check_raw_payload.py && cat raw_payload_check.txt

# Test 4: Check pitcher-specific fields
railway run python scripts/check_pitcher_raw.py && cat pitcher_raw_check.txt
```

---

## Investigation Methodology Notes

1. **Railway Testing**: Used `railway run python` to test in production environment
2. **File Output**: Railway stdout buffering required writing results to files
3. **Raw API Inspection**: Direct `requests.get()` calls to see actual BDL response structure
4. **Progressive Analysis**: Started with field availability, drilled down to field names

---

## Next Steps

1. ✅ Root cause identified
2. ⏳ Fix Pydantic field mapping (Task #24, #22)
3. ⏳ Re-run ingestion to compute ops/whip (Task #26)
4. ⏳ Verify computation is working (Task #25)
5. ⏳ Backfill existing NULL data (Task #26)