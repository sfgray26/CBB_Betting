# Root Cause Analysis: ops/whip NULL Issues

**Investigation Date:** 2026-04-10
**Investigator:** Claude Code (EMAC-067)
**Issue:** ops and whip fields are 100% NULL in mlb_player_stats table despite computation code existing

---

## Executive Summary

**ROOT CAUSE IDENTIFIED:** The computation code exists and receives valid source data, but the computed values are never persisted back to the database.

**Key Finding:** The investigation reveals that both obp/slg and walks_allowed/hits_allowed data exist in the database, and raw_payload contains all necessary fields. However, the computation code in `daily_ingestion.py` (lines 1130-1142) performs calculations but never commits these values to the database.

---

## Investigation Results

### Investigation 1: Source Data Availability
```
Total rows:                    5,197
Has obp:                       3,690 (71%)
Has slg:                       3,690 (71%)
Has ops:                       0 (0%)
Has BOTH obp and slg:         3,690 (71%)
```

**Finding:** Source data (obp, slg) exists for 71% of records. The BDL API successfully provides these metrics.

---

### Investigation 2: Sample Data Validation
```json
{
  "bdl_player_id": 370,
  "obp": 0.267,
  "slg": 0.467,
  "ops": null,
  "game_date": "2026-04-09"
}
```

**Finding:** Sample rows confirm obp/slg are populated with valid float values, but ops remains NULL despite having source data.

---

### Investigation 3: Raw Payload Analysis
```
Rows with 'obp' in raw_payload:  5,197 (100%)
Rows with 'slg' in raw_payload:  5,197 (100%)
Rows with 'ops' in raw_payload:  5,197 (100%)
```

**Finding:** BDL API returns ALL necessary fields (obp, slg, ops) in the raw payload for 100% of records. This eliminates "BDL doesn't provide the data" as a hypothesis.

---

### Investigation 4: WHIP Component Analysis
```
Total rows:                    5,197
Has walks_allowed:            1,485 (29%)
Has hits_allowed:             1,485 (29%)
Has whip:                        0 (0%)
Has BOTH components:          1,485 (29%)
```

**Finding:** WHIP source data exists for 29% of records (pitching stats are less common than batting stats), but whip field is completely NULL.

---

### Investigation 5: ERA Anomaly
```json
{
  "bdl_player_id": 1638,
  "era": 162.0,
  "earned_runs": 6.0,
  "innings_pitched": 0.1,
  "game_date": "2026-03-28"
}
```

**Finding:** ERA > 100 issue confirmed. Player with 0.1 IP and 6 ER results in ERA = (6.0 / 0.1) * 9 = 162.0. This is mathematically correct but should be handled with minimum IP threshold.

---

### Investigation 6: Orphaned Position Eligibility
```
Orphaned position_eligibility records: 477
```

**Finding:** 477 position_eligibility records reference yahoo_player_keys that don't exist in player_id_mapping table. This is a separate data quality issue.

---

## Root Cause Diagnosis

### The Problem

The code in `daily_ingestion.py` (lines 1130-1142) computes ops and whip correctly:

```python
# Compute ops (on-base + slugging)
if obp is not None and slg is not None:
    stats.ops = obp + slg

# Compute whip (walks + hits per inning)
if stats.walks_allowed is not None and stats.hits_allowed is not None and stats.innings_pitched is not None and stats.innings_pitched > 0:
    stats.whip = (stats.walks_allowed + stats.hits_allowed) / stats.innings_pitched
```

**However, these computations are never persisted!**

The code flow:
1. BDL data is fetched and stored in `raw_payload`
2. Individual fields are extracted from `raw_payload` into normalized columns
3. **ops/whip are computed in local memory**
4. Records are written to database
5. **Step 3 computations are lost because they happen after the DB write**

### Why This Happened

The computation code was likely added during a refactoring session but the database persistence logic was never updated to include the computed fields. The SQLAlchemy model expects ops/whip to be explicitly set, but the ingestion code doesn't include them in the final `db.add()` or `db.merge()` call.

---

## Fix Strategy

### Immediate Fix (Priority 1)

**File:** `backend/services/daily_ingestion.py`
**Location:** Lines 1130-1142 (computation section)

**Solution:** Move the computation logic to happen BEFORE the database write, and ensure computed values are included in the model instance before persistence.

```python
# Current (broken) flow:
# 1. Extract fields from raw_payload
# 2. Write to database
# 3. Compute ops/whip (discarded!)

# Correct flow:
# 1. Extract fields from raw_payload
# 2. Compute ops/whip
# 3. Write to database (including computed fields)
```

### Secondary Fixes (Priority 2)

1. **ERA Anomaly:** Add minimum innings pitched threshold (e.g., 0.2 IP) before computing ERA to avoid division artifacts
2. **Orphaned Position Records:** Run backfill to link orphaned position_eligibility records to player_id_mapping
3. **Data Validation:** Add post-ingestion validation to ensure computed fields are populated

---

## Impact Assessment

### Current State
- **ops field:** 0/5,197 records (0%)
- **whip field:** 0/5,197 records (0%)
- **Source data available:** 3,690 ops candidates (71%), 1,485 whip candidates (29%)

### Post-Fix Expected State
- **ops field:** 3,690/5,197 records (71%) - batting stats only
- **whip field:** 1,485/5,197 records (29%) - pitching stats only
- **Data quality:** Improved by eliminating manual computation gaps

---

## Verification Plan

1. **Pre-Fix Backup:** Export current mlb_player_stats table to CSV
2. **Apply Fix:** Update daily_ingestion.py computation/persistence logic
3. **Test Run:** Execute single day ingestion to verify ops/whip are populated
4. **Backfill:** Run historical backfill for all existing records
5. **Validation:** Re-run investigation endpoint to confirm 0 NULLs for eligible records

---

## Additional Findings

### ERA Computation Artifact
The ERA = 162.0 for player 1638 is mathematically correct but represents a statistical artifact:
- 6 earned runs in 0.1 innings
- ERA = (6.0 / 0.1) * 9 = 162.0
- Recommendation: Add minimum IP threshold (e.g., 0.2) for ERA display purposes

### Orphaned Position Records
477 position_eligibility records reference non-existent yahoo_player_keys:
- Likely caused by incomplete player_id_mapping backfill
- Recommendation: Cross-reference with BDL player registry and backfill missing mappings

---

## Conclusion

The root cause is **not** missing BDL data or API limitations. The computation code exists and receives valid source data, but the results are never persisted to the database. This is a classic "compute but don't save" bug that can be fixed by ensuring computed values are attached to the SQLAlchemy model instance before the database commit.

**Estimated Fix Time:** 15 minutes (code change + testing)
**Estimated Backfill Time:** 5 minutes (one-time script execution)
**Total Data Recovery:** 5,175 records (3,690 ops + 1,485 whip)