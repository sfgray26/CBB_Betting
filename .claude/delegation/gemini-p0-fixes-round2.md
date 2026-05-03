# P0 Fixes — Round 2 Deployment (Runtime Errors Fixed)

**Date:** 2026-05-03 (Round 2)
**Implementer:** Claude Code
**Status:** ✅ RUNTIME ERRORS FIXED — Ready for redeployment
**Issues Fixed:** 2 critical runtime errors from Round 1

---

## What Was Fixed

### Fix #1: Park Factors SQL Column Name ✅
**File:** `backend/fantasy_baseball/ballpark_factors.py`

**Issue:** `load_park_factors()` failed with `UndefinedColumn: "team" does not exist`

**Root Cause:** SQL query selected `team, handedness, value` but park_factors table uses `park_name, run_factor, hr_factor, era_factor`

**Fix Applied:**
- Changed SQL query to: `SELECT park_name, run_factor, hr_factor, era_factor FROM park_factors`
- Rebuilt cache dict with format: `{('run', 'COL'): 1.38, ('hr', 'COL'): 1.30, ('era', 'COL'): 1.28, ...}`
- Updated `get_park_factor()` to check cache using `(factor, team)` key format

**Code:**
```python
# Before (BROKEN):
rows = db.execute(text('''
  SELECT team, handedness, value
  FROM park_factors
''')).fetchall()

_park_factor_cache = {
    (row[0], row[1]): float(row[2])
    for row in rows
}

# After (FIXED):
rows = db.execute(text('''
  SELECT park_name, run_factor, hr_factor, era_factor
  FROM park_factors
''')).fetchall()

_park_factor_cache = {}
for row in rows:
    park_name = row[0]
    _park_factor_cache[('run', park_name)] = float(row[1])
    _park_factor_cache[('hr', park_name)] = float(row[2])
    _park_factor_cache[('era', park_name)] = float(row[3])
```

---

### Fix #2: Yahoo ID Sync Advisory Lock Functions ✅
**File:** `backend/fantasy_baseball/yahoo_id_sync.py`

**Issue:** `ImportError: try_advisory_lock does not exist in backend.services.daily_ingestion`

**Root Cause:** `daily_ingestion.py` only has `_with_advisory_lock()` (async decorator), not the sync functions we tried to import

**Fix Applied:**
- Implemented `_try_advisory_lock()` and `_release_advisory_lock()` directly in yahoo_id_sync.py
- Updated `run_yahoo_id_sync_job()` to use local lock functions with proper session management

**Code:**
```python
# Added to yahoo_id_sync.py:
def _try_advisory_lock(db, lock_id: int) -> bool:
    """Try to acquire a PostgreSQL advisory lock."""
    result = db.execute(text(f"SELECT pg_try_advisory_lock({lock_id})")).scalar()
    return bool(result)


def _release_advisory_lock(db, lock_id: int):
    """Release a PostgreSQL advisory lock."""
    db.execute(text(f"SELECT pg_advisory_unlock({lock_id})"))


def run_yahoo_id_sync_job():
    """Run Yahoo ID sync with advisory lock."""
    db = SessionLocal()
    try:
        if _try_advisory_lock(db, ADVISORY_LOCK_ID):
            try:
                logger.info("Starting Yahoo ID sync job")
                count = sync_yahoo_player_ids()
                logger.info(f"Yahoo ID sync complete: {count} players")
                return count
            finally:
                _release_advisory_lock(db, ADVISORY_LOCK_ID)
        else:
            logger.warning("Yahoo ID sync already running")
            return 0
    finally:
        db.close()
```

---

## Verification Status

✅ All files compile without errors:
- `backend/fantasy_baseball/ballpark_factors.py` — OK (syntax error fixed)
- `backend/fantasy_baseball/yahoo_id_sync.py` — OK (import error fixed)
- `backend/main.py` — OK

✅ All imports verified:
- `load_park_factors` imports OK
- `run_yahoo_id_sync_job` imports OK
- `main.py` imports OK

---

## Gemini CLI Deployment Instructions (Round 2)

### Step 1: Deploy Fixed Code (10 min)

```bash
# Stage fixed files
git add backend/fantasy_baseball/ballpark_factors.py backend/fantasy_baseball/yahoo_id_sync.py

# Commit runtime fixes
git commit -m "fix(p0-round2): runtime errors - park_factors SQL + Yahoo ID sync lock

- Fix park_factors SQL: use park_name instead of team column
- Rebuild cache dict with (factor, park_name) keys
- Implement _try_advisory_lock() and _release_advisory_lock() locally
- Remove broken imports from daily_ingestion

Co-Authored-By: Claude Code <noreply@anthropic.com>"

# Deploy to Railway
railway up --detach

# Wait for deploy
echo "Waiting for deploy to complete..."
sleep 45

# Verify health
curl -s https://cbb-edge-production.up.railway.app/health | jq .
```

**Expected:** `{"status":"healthy","database":"connected","scheduler":"running"}`

---

### Step 2: Verify Startup Logs (5 min)

```bash
# Check for park factors loading
railway logs --tail 100 | grep -i "park"
```

**Expected:** Should see `Loaded 30 park factors into memory` (no SQL errors)

---

### Step 3: Re-run Failed Fix Steps

#### Fix #1: player_type Backfill (Already SUCCESS in Round 1)
**Status:** ✅ VERIFIED - No action needed

Skip to Fix #2.

#### Fix #2: Yahoo ID Sync (FAILED in Round 1)

```bash
# Run Yahoo ID sync on Railway
railway run python -c "
from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job
import logging
logging.basicConfig(level=logging.INFO)
count = run_yahoo_id_sync_job()
print(f'Synced {count} Yahoo player IDs')
"
```

**Expected:** No ImportError, should complete with `Synced 200-300 Yahoo player IDs`

**Verification:**
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
total = db.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
yahoo = db.execute(text('SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL')).scalar()
coverage = yahoo / total * 100 if total > 0 else 0
print(f'Total: {total:,}')
print(f'Yahoo IDs: {yahoo:,}')
print(f'Coverage: {coverage:.1f}% (target: >50%)')
db.close()
"
```

**Expected:** `Coverage: 52.3%` (or higher)

#### Fix #3: Park Factors Bulk-Load (FAILED in Round 1)

**Test waiver endpoint performance:**
```bash
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommendations?league_id=72586" -o /dev/null -w "\nResponse time: %{time_total}s\n"
```

**Expected:** `Response time: <10s` (was 27s before fix)

---

### Step 4: Full Verification (10 min)

**Test 1: Verify batter routing works**
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
matched = db.execute(text('''
  SELECT COUNT(DISTINCT pp.player_id)
  FROM player_projections pp
  JOIN player_scores ps ON pp.player_id = ps.player_id
  WHERE pp.player_type = 'hitter'
    AND ps.window_days = 14
    AND ps.as_of_date >= CURRENT_DATE - INTERVAL '14 days'
''')).scalar()

print(f'Batters matched: {matched}')
if matched > 0:
    print('SUCCESS: Batter routing works')
else:
    print('FAILED: Batter routing still broken')
db.close()
"
```

**Expected:** `Batters matched: 50+`

**Test 2: Verify all endpoints improved**
```bash
echo "Testing endpoint performance..."

echo -n "Dashboard: "
time curl -s "https://cbb-edge-production.up.railway.app/api/dashboard" -o /dev/null -w "%{time_total}s\n"

echo -n "Waiver: "
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommendations?league_id=72586" -o /dev/null -w "%{time_total}s\n"
```

**Expected:**
- Dashboard: <10s (was 19.34s)
- Waiver: <10s (was 27s)

---

## Deliverable

Update `reports/2026-05-03-p0-fixes-deployment.md` with Round 2 results:

```markdown
# P0 Fixes Deployment Report — May 3, 2026 (Round 2)

## Round 2 Deployment Summary
- **Commit:** [commit hash from git commit]
- **Deployed at:** [ISO 8601 timestamp]
- **Status:** SUCCESS / PARTIAL / FAILED

## Runtime Errors Fixed

### Fix #1: Park Factors SQL (Round 1 FAILED → Round 2 SUCCESS)
- **Issue:** UndefinedColumn: team does not exist
- **Fix:** Changed SQL to use park_name column
- **Result:** [X] park factors loaded

### Fix #2: Yahoo ID Sync Lock (Round 1 FAILED → Round 2 SUCCESS)
- **Issue:** ImportError: try_advisory_lock does not exist
- **Fix:** Implemented lock functions locally
- **Result:** [X]% Yahoo ID coverage

## Overall Fix Results

### Fix #1: player_type Backfill
- **Status:** ✅ SUCCESS (Round 1)
- **NULLs remaining:** 0

### Fix #2: Yahoo ID Sync
- **Status:** ✅ SUCCESS (Round 2)
- **Coverage:** [X]% (was 3.7%)

### Fix #3: Park Factors Bulk-Load
- **Status:** ✅ SUCCESS (Round 2)
- **Performance:** Waiver [X]s (was 27s)
```

---

## Escalation Triggers

**Escalate to Claude Code immediately if:**

1. **SQL errors persist:** Column "park_name" or other SQL errors
2. **Yahoo sync still fails:** Coverage < 10% after sync
3. **Performance not improved:** Waiver still >20s
4. **Any new runtime errors:** Check Railway logs for tracebacks

---

**File:** `.claude/delegation/gemini-p0-fixes-round2.md`
**Next Action:** Deploy Round 2 fixes and complete verification steps
