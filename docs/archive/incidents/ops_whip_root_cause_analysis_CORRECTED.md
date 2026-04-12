# ?? ARCHIVED — Consolidated into Incident Report

> **Status:** ARCHIVED (April 11, 2026)
> **Original:** docs/ops_whip_root_cause_analysis_CORRECTED.md
> **Consolidated Into:** docs/incidents/2026-04-10-ops-whip-root-cause.md`n> **Reason:** Version chain consolidated into single authoritative incident report

---

# CORRECTED Root Cause Analysis: ops/whip NULL Issues

**Investigation Date:** 2026-04-10
**Investigator:** Claude Code (EMAC-067)
**Issue:** ops and whip fields are 100% NULL in mlb_player_stats table
**Status:** INITIAL ANALYSIS INCORRECT - Revised analysis below

---

## Executive Summary

**CORRECTED ROOT CAUSE:** The computation code IS correct and values ARE being persisted to the database. The INSERT and UPDATE clauses in `daily_ingestion.py` (lines 1169, 1177, 1201, 1208) correctly include `ops=computed_ops` and `whip=computed_whip`.

**REAL INVESTIGATION NEEDED:** Since the code is correct, we must determine:
1. Is this code path actually being executed?
2. Are `computed_ops` and `computed_whip` always None at the point of database write?
3. Are the conditional checks (`if stat.obp is not None and stat.slg is not None`) failing?
4. Is production using different code than this file?
5. Is there a transaction rollback happening?

---

## Code Review Findings

### The Code IS Correct

**File:** `backend/services/daily_ingestion.py` lines 1130-1220

**Computation Logic (Lines 1130-1142):**
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

**Persistence Logic (Lines 1169, 1177, 1201, 1208):**
```python
# INSERT clause (line 1169):
ops=computed_ops,

# UPDATE clause (line 1201):
ops=computed_ops,

# INSERT clause (line 1177):
whip=computed_whip,

# UPDATE clause (line 1208):
whip=computed_whip,
```

**Conclusion:** The code correctly computes and persists ops/whip values.

---

## Previous Analysis Error

**INCORRECT CLAIM:** "The computation code exists but values are never persisted."

**CORRECTION:** The values ARE being persisted - both `ops=computed_ops` and `whip=computed_whip` are present in the INSERT (lines 1169, 1177) and UPDATE (lines 1201, 1208) clauses.

---

## Revised Investigation Path

### Hypothesis 1: Conditional Check Failure

The computation is guarded by conditional checks:
- **OPS:** `if stat.obp is not None and stat.slg is not None:`
- **WHIP:** `if (stat.walks_allowed is not None and stat.hits_allowed is not None and stat.ip is not None):`

**Question:** Are these conditions failing for 100% of records?

**Test needed:**
```sql
SELECT
  COUNT(*) as total,
  COUNT(obp) as has_obp,
  COUNT(slg) as has_slg,
  COUNT(obp) + COUNT(slg) as has_both_for_ops,
  COUNT(walks_allowed) as has_bb,
  COUNT(hits_allowed) as has_h,
  COUNT(ip) as has_ip
FROM mlb_player_stats;
```

### Hypothesis 2: BDL Provides ops/whip (Override Issue)

**Data Contract Analysis:**
- `MLBPlayerStats.ops` is `Optional[float]` - BDL MAY provide it
- `MLBPlayerStats.whip` is `Optional[float]` - BDL MAY provide it

**Question:** If BDL API returns `ops` and `whip` in the response, are they `None` or do they have values that are different from our computations?

**Raw Payload Analysis:**
```
Rows with 'ops' in raw_payload:  5,197 (100%)
Rows with 'whip' in raw_payload: 5,197 (100%)
```

**Critical Question:** What are the VALUES of `ops` and `whip` in the raw_payload?
- If `raw_payload['ops']` is `null`, BDL doesn't provide it and we should compute it
- If `raw_payload['ops']` has a value, BDL provides it and we should use that value

### Hypothesis 3: Code Path Not Executing

**Question:** Is the `mlb_box_stats` job actually running?

**Test needed:**
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
result = db.execute(text('SELECT MAX(ingested_at) FROM mlb_player_stats')).fetchone()
print(f'Latest ingestion: {result[0]}')
db.close()
"
```

### Hypothesis 4: Transaction Rollback

**Question:** Is there a transaction rollback happening after the INSERT?

**Code review:** Lines 1213-1228 show proper error handling with `db.begin_nested()` and explicit exception handling. No obvious rollback issue.

### Hypothesis 5: Production vs Local Code Mismatch

**Question:** Is production running a different version of the code?

**Test needed:**
```bash
railway run python -c "
import inspect
from backend.services.daily_ingestion import DailyIngestionScheduler
source = inspect.getsource(DailyIngestionScheduler.mlb_box_stats)
print('Lines 1130-1142:')
print(source.split('computed_ops')[1].split('computed_cs')[0])
"
```

---

## Required Investigation Steps

### Step 1: Check Raw Payload Values
```sql
SELECT
  raw_payload->'obp' as obp,
  raw_payload->'slg' as slg,
  raw_payload->'ops' as ops_from_bdl,
  raw_payload->'whip' as whip_from_bdl,
  ops as ops_in_db,
  whip as whip_in_db
FROM mlb_player_stats
LIMIT 10;
```

### Step 2: Test Conditional Logic
```python
# Simulate the conditional checks
import statistics
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text('''
  SELECT
    COUNT(*) as total,
    COUNT(obp) as has_obp,
    COUNT(slg) as has_slg,
    COUNT(CASE WHEN obp IS NOT NULL AND slg IS NOT NULL THEN 1 END) as ops_computable,
    COUNT(walks_allowed) as has_bb,
    COUNT(hits_allowed) as has_h,
    COUNT(ip) as has_ip,
    COUNT(CASE WHEN walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL AND ip IS NOT NULL THEN 1 END) as whip_computable
  FROM mlb_player_stats
''')).fetchone()
print(f"Total: {result.total}")
print(f"Has OBP: {result.has_obp}")
print(f"Has SLG: {result.has_slg}")
print(f"OPS computable: {result.ops_computable}")
print(f"WHIP computable: {result.whip_computable}")
db.close()
```

### Step 3: Verify BDL API Response
```python
from backend.services.balldontlie import BallDontLieClient

bdl = BallDontLieClient()
stats = bdl.get_mlb_stats(limit=1)
if stats:
    stat = stats[0]
    print(f"BDL response - OBP: {stat.obp}, SLG: {stat.slg}, OPS: {stat.ops}")
    print(f"BDL response - BB: {stat.bb_allowed}, H: {stat.h_allowed}, IP: {stat.ip}, WHIP: {stat.whip}")
```

---

## Initial Diagnosis (Pending Verification)

**Most Likely Root Cause:** One of these three scenarios:

1. **Conditional Failure:** All records fail the `if stat.obp is not None and stat.slg is not None` check (unlikely given 71% have source data)

2. **BDL Override:** BDL API returns `ops=null` and `whip=null`, so even though we compute values, something in the Pydantic model or SQLAlchemy mapping is overriding our computed values with `None` from the BDL response

3. **Code Not Running:** The `mlb_box_stats` ingestion job is not executing or is encountering an error before reaching the computation code

**Least Likely:** Transaction rollback or production code mismatch (error handling looks solid, Railway deployment should be consistent)

---

## Next Actions

1. **Immediate:** Run the investigation steps above to determine which hypothesis is correct
2. **Based on findings:** Implement targeted fix
3. **Verification:** Re-run the investigation endpoint to confirm ops/whip are populated

---

## Previous Analysis (Incorrect - Do Not Use)

The initial analysis claimed "computation code exists but values are never persisted." This was **INCORRECT**. The code review shows that `ops=computed_ops` and `whip=computed_whip` are correctly included in both INSERT and UPDATE clauses.

The real issue is elsewhere - likely in the conditional logic, BDL API response handling, or code execution flow.

