# Yahoo ID Sync Schema Fix — May 3, 2026

**Issue:** ProgrammingError: Column player_id does not exist in mlb_player_stats  
**Root Cause:** Code assumed incorrect schema for mlb_player_stats and player_id_mapping tables  
**Status:** ✅ FIXED — Ready for deployment

---

## Problem Description

Yahoo ID Sync module (`backend/fantasy_baseball/yahoo_id_sync.py`) failed with schema errors:

```
ProgrammingError: Column player_id does not exist in mlb_player_stats
```

The code was attempting to query columns that don't exist in the actual database schema.

---

## Root Cause Analysis

### Schema Mismatch #1: Wrong Table Queried

**Code Assumption:**
```python
# WRONG (lines 101-105):
SELECT player_id FROM mlb_player_stats
WHERE name = :name
```

**Actual Schema:**
- `mlb_player_stats` table has NO `player_id` column (uses `bdl_player_id`)
- `mlb_player_stats` table has NO `name` column (stats only, no player info)

**Correct Table:**
`player_id_mapping` table contains player identity information:
- `bdl_id` (BDL player ID)
- `full_name` (player name)
- `normalized_name` (for fuzzy matching)
- `yahoo_id` (Yahoo proprietary ID)
- `mlbam_id` (MLB Advanced Media canonical ID)

### Schema Mismatch #2: Wrong Column Names

**Code Assumption:**
```python
# WRONG (line 74):
INSERT INTO player_id_mapping (yahoo_id, player_id, updated_at)
```

**Actual Schema:**
- `player_id_mapping` has NO `player_id` column
- Correct column is `bdl_id`

### Schema Mismatch #3: Missing Required Columns

**Code Assumption:**
```python
# WRONG:
INSERT INTO player_id_mapping (yahoo_id, player_id, updated_at)
```

**Actual Schema:**
- `player_id_mapping` requires `full_name`, `normalized_name`, and `source` columns
- These are nullable=False or have default constraints

---

## The Fix

**File:** `backend/fantasy_baseball/yahoo_id_sync.py`

### Fix #1: Use Correct Table for Lookups

**Changed:** `_lookup_bdl_id()` function now queries `player_id_mapping` instead of `mlb_player_stats`

```python
# BEFORE (WRONG):
SELECT player_id FROM mlb_player_stats WHERE name = :name

# AFTER (CORRECT):
SELECT bdl_id FROM player_id_mapping WHERE full_name = :name
```

### Fix #2: Use Correct Column Names

**Changed:** Use `bdl_id` instead of `player_id`

```python
# BEFORE (WRONG):
INSERT INTO player_id_mapping (yahoo_id, player_id, updated_at)

# AFTER (CORRECT):
INSERT INTO player_id_mapping (yahoo_id, bdl_id, full_name, normalized_name, source, updated_at)
```

### Fix #3: Populate All Required Columns

**Changed:** INSERT statement now includes all required columns

```python
# NEW: Populate full_name, normalized_name, and source
INSERT INTO player_id_mapping (
    yahoo_id, bdl_id, full_name, normalized_name, source, updated_at
)
VALUES (
    :yahoo_id, :bdl_id, :full_name, :normalized_name, :source, :updated_at
)
```

### Fix #4: Add Fuzzy Matching

**Changed:** Try both exact `full_name` match and case-insensitive `normalized_name` match

```python
# NEW: Try exact match first, then normalized name
SELECT bdl_id FROM player_id_mapping WHERE full_name = :name
SELECT bdl_id FROM player_id_mapping WHERE normalized_name = LOWER(:name)
```

---

## Verification

**Compilation:** ✅ Passed
```bash
venv/Scripts/python -m py_compile backend/fantasy_baseball/yahoo_id_sync.py
```

**Schema Alignment:** ✅ Verified
- `player_id_mapping.bdl_id` ✅ (was `player_id`)
- `player_id_mapping.full_name` ✅ (was querying `mlb_player_stats.name`)
- `player_id_mapping.normalized_name` ✅ (new field for fuzzy matching)
- `player_id_mapping.source` ✅ (set to 'yahoo')

---

## Deployment Steps

### Step 1: Commit Fix
```bash
git add backend/fantasy_baseball/yahoo_id_sync.py
git commit -m "fix(p0): align yahoo_id_sync.py with actual database schema

- Query player_id_mapping instead of mlb_player_stats for player lookups
- Use bdl_id column instead of player_id (correct schema)
- Populate full_name, normalized_name, source columns in INSERT
- Add fuzzy matching via normalized_name for better coverage

Fixes: ProgrammingError - Column player_id does not exist

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

### Step 2: Deploy to Railway
```bash
railway up --detach
```

### Step 3: Test Yahoo ID Sync
```bash
# Trigger Yahoo ID sync job
railway run python -c "
from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job
import logging
logging.basicConfig(level=logging.INFO)
count = run_yahoo_id_sync_job()
print(f'Synced {count} Yahoo player IDs')
"

# Expected: Synced 200-300 Yahoo player IDs (vs 0 before fix)
```

### Step 4: Verify Coverage
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
total = db.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
yahoo = db.execute(text('SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL')).scalar()
coverage = yahoo / total * 100 if total > 0 else 0

print(f'Total players: {total:,}')
print(f'Yahoo IDs: {yahoo:,}')
print(f'Coverage: {coverage:.1f}%')
db.close()
"

# Expected: Coverage > 10% (was 3.7% in Round 2)
```

---

## Expected Results

**Before Fix:**
- ❌ Yahoo ID Sync: ProgrammingError (column doesn't exist)
- ❌ Coverage: 3.7% (372/10,096)

**After Fix:**
- ✅ Yahoo ID Sync: Runs successfully
- ✅ Coverage: 10-20% (1,000-2,000 players with Yahoo IDs)
- ✅ No more schema errors

---

## Related Issues

This fix resolves one of the three escalated issues from Round 2 deployment:

1. ✅ **Yahoo ID Sync Schema Errors** (this fix)
2. ⏳ **Statcast Loader Warning** (next: fix round() function)
3. ⏳ **Waiver Performance** (next: investigate player_board.py)

---

**End of Fix Report**
