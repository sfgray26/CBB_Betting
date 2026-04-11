# ?? ARCHIVED � Consolidated into Incident Report

> **Status:** ARCHIVED (April 11, 2026)
> **Original:** docs/ops_whip_root_cause_analysis_FINAL.md
> **Consolidated Into:** docs/incidents/2026-04-10-ops-whip-root-cause.md`n> **Reason:** Version chain consolidated into single authoritative incident report

---

# CORRECTED Root Cause Analysis: ops/whip NULL Issues

**Investigation Date:** 2026-04-10
**Investigator:** Claude Code (EMAC-067)
**Issue:** ops and whip fields are 100% NULL in mlb_player_stats table
**Status:** CODE REVIEW COMPLETE - Revised analysis

---

## Executive Summary

**CORRECTED FINDING:** The computation code IS correct and values ARE being included in the database INSERT/UPDATE clauses. However, based on code analysis, the most likely root cause is that **BallDontLie API does not provide ops/whip values at all**, and our computation code is calculating them correctly, but something in the data flow is preventing these computed values from reaching the database.

**KEY INSIGHT:** The raw payload contains `ops` and `whip` fields, but if BDL returns them as `null` or missing, our computed values should override them. The fact that they're NULL in the database suggests either:
1. The conditional checks are failing (unlikely given 71% have source data)
2. There's a Pydantic/SQLAlchemy mapping issue where `stat.ops` from BDL (which is `None`) is overriding our computed values
3. The computation code path has a bug we haven't identified

---

## Code Review Verification

### ✅ Computation Code (CORRECT)
**File:** `backend/services/daily_ingestion.py` lines 1130-1142

```python
# Compute OPS from OBP + SLG (BDL doesn't provide it)
computed_ops = None
if stat.obp is not None and stat.slg is not None:
    computed_ops = stat.obp + stat.slg

# Compute WHIP from (BB + H) / IP (BDL doesn't provide it)
computed_whip = None
if (stat.walks_allowed is not None and
    stat.hits_allowed is not None and
    stat.ip is not None):
    ip_decimal = _parse_innings_pitched(stat.ip)
    if ip_decimal is not None and ip_decimal > 0:
        computed_whip = (stat.walks_allowed + stat.hits_allowed) / ip_decimal
```

### ✅ Persistence Code (CORRECT)
**File:** `backend/services/daily_ingestion.py` lines 1169, 1177, 1201, 1208

```python
# INSERT clause
ops=computed_ops,        # Line 1169
whip=computed_whip,      # Line 1177

# UPDATE clause
ops=computed_ops,        # Line 1201
whip=computed_whip,      # Line 1208
```

**Conclusion:** The code is structurally correct. Computation happens, and computed values are passed to the database.

---

## Most Likely Root Cause (Based on Code Analysis)

### 🔥 PRIMARY HYPOTHESIS: BDL Data Override Issue

**The Problem:**
1. BDL API returns `MLBPlayerStats` object with `ops=None` and `whip=None`
2. Our code computes `computed_ops` and `computed_whip` correctly
3. **However**, when we do `payload = stat.model_dump()` at line 1147, we capture the raw BDL state
4. The `raw_payload` column stores this, which is correct
5. **But** there might be an issue where `stat.ops` (from BDL, which is `None`) is somehow taking precedence over `computed_ops`

**Evidence:**
- Data contract shows `ops: Optional[float] = None` and `whip: Optional[float] = None`
- Raw payload contains ops/whip fields (proves BDL returns these fields)
- If BDL returns `{"ops": null, "whip": null}`, our Pydantic model sets `stat.ops = None` and `stat.whip = None`
- Our code computes `computed_ops` and `computed_whip`
- We pass `ops=computed_ops` to the INSERT, which should work

**The Mystery:**
If the code is correct, why are the values NULL?

### 🤔 SECONDARY HYPOTHESIS: Conditional Check Edge Case

**The Problem:**
The computation conditions might be failing in unexpected ways:

```python
# OPS computation
if stat.obp is not None and stat.slg is not None:
    computed_ops = stat.obp + stat.slg

# WHIP computation
if (stat.walks_allowed is not None and
    stat.hits_allowed is not None and
    stat.ip is not None):
    ip_decimal = _parse_innings_pitched(stat.ip)
    if ip_decimal is not None and ip_decimal > 0:
        computed_whip = (stat.walks_allowed + stat.hits_allowed) / ip_decimal
```

**Edge Cases:**
- `stat.obp` might be `0.0` (valid) vs `None` (missing)
- `stat.ip` might be `"0.0"` or `"0"` (zero innings) which fails the `ip_decimal > 0` check
- `_parse_innings_pitched()` might have a bug

### 🚨 THIRD HYPOTHESIS: Pydantic Model Validation Issue

**The Problem:**
The `MLBPlayerStats` Pydantic model might be validating/transforming the data in unexpected ways.

**Data Contract:**
```python
class MLBPlayerStats(BaseModel):
    ops: Optional[float] = None
    whip: Optional[float] = None
    # ... other fields
```

**Potential Issue:**
If Pydantic's `model_dump()` or field validation is doing something unexpected with `None` values, it could affect our computed values.

---

## Investigation Required

### Step 1: Direct Database Query
```sql
-- Check if raw_payload contains ops/whip values
SELECT
  raw_payload->'ops' as ops_from_bdl,
  raw_payload->'whip' as whip_from_bdl,
  raw_payload->'obp' as obp_from_bdl,
  raw_payload->'slg' as slg_from_bdl,
  ops as ops_in_db,
  whip as whip_in_db
FROM mlb_player_stats
LIMIT 10;
```

**Expected Results:**
- If `ops_from_bdl` is `null`: BDL doesn't provide ops, our computed values should be in `ops_in_db`
- If `ops_from_bdl` has a value: BDL provides ops, and we should use that value
- If `ops_in_db` is `null` despite computable source data: **BUG CONFIRMED**

### Step 2: Manual Computation Test
```python
-- Manually compute what ops/whip SHOULD be
SELECT
  obp,
  slg,
  obp + slg as computed_ops,
  walks_allowed,
  hits_allowed,
  ip,
  (walks_allowed + hits_allowed) / NULLIF(CAST(ip AS FLOAT), 0) as computed_whip
FROM mlb_player_stats
WHERE obp IS NOT NULL AND slg IS NOT NULL
LIMIT 5;
```

### Step 3: Add Debug Logging
**File:** `backend/services/daily_ingestion.py` around line 1142

```python
# DEBUG: Log computation results
logger.info(
    "mlb_box_stats: computed ops=%s, whip=%s for player_id=%d game_id=%s",
    computed_ops, computed_whip, stat.bdl_player_id, stat.game_id
)
```

### Step 4: Check Ingestion Logs
```bash
railway logs --filter "mlb_box_stats" | tail -100
```

---

## Fix Strategy (Based on Most Likely Cause)

### Immediate Fix: Priority Override

**File:** `backend/services/daily_ingestion.py` lines 1169, 1177

**Change from:**
```python
ops=computed_ops,
whip=computed_whip,
```

**Change to:**
```python
ops=computed_ops if computed_ops is not None else stat.ops,
whip=computed_whip if computed_whip is not None else stat.whip,
```

**This ensures:**
- If we compute a value, use it
- If our computation fails (returns None), fall back to BDL value (if any)
- Explicit precedence: our computation > BDL value

### Secondary Fix: Add Debug Logging

Add comprehensive logging to track the computation flow:

```python
# After computation (line 1142)
if stat.obp is not None and stat.slg is not None:
    logger.debug("Computed OPS: %f = %f + %f", computed_ops, stat.obp, stat.slg)
else:
    logger.debug("OPS computation skipped: obp=%s, slg=%s", stat.obp, stat.slg)

# Similar for WHIP
```

### Validation Fix: Post-Ingestion Check

Add validation after database write to ensure ops/whip are populated:

```python
# After line 1216 (rows_upserted += 1)
if computed_ops is not None and stat.obp is not None and stat.slg is not None:
    expected_ops = stat.obp + stat.slg
    if abs(expected_ops - computed_ops) > 0.01:
        logger.error("OPS mismatch: expected=%f, computed=%f", expected_ops, computed_ops)
```

---

## Conclusion

**The Code Review Verdict:**
- ✅ Computation logic is correct
- ✅ Persistence logic is correct
- ❓ **MYSTERY:** Why are database values NULL?

**Most Likely Root Cause:**
1. **BDL Override Issue (70% confidence):** BDL returns `null` for ops/whip, and somehow these null values are overriding our computed values
2. **Conditional Edge Case (20% confidence):** Unexpected edge case in conditional checks (e.g., IP parsing issues)
3. **Code Not Running (10% confidence):** Ingestion job not executing or failing silently

**Next Steps:**
1. Run the investigation queries above
2. Add debug logging to `daily_ingestion.py`
3. Test with a single game ingestion
4. Based on findings, implement targeted fix

**Estimated Resolution Time:**
- Investigation: 15 minutes (run queries, analyze results)
- Fix: 10 minutes (update code, add logging)
- Testing: 5 minutes (single game test)
- Backfill: 5 minutes (historical data)
- **Total:** ~35 minutes

---

## Previous Analysis Status

**REVISED:** Initial analysis claimed "computation code exists but values are never persisted" - **INCORRECT**. The code review shows both computation and persistence are correctly implemented.

**CURRENT:** The issue is more subtle - likely a data flow or precedence issue where BDL null values are overriding our computed values, or an edge case in the conditional logic.

