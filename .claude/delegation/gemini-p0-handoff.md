# P0 Fixes Implementation Complete — Gemini CLI Deployment Handoff

**Date:** 2026-05-03
**Implementer:** Claude Code
**Status:** ✅ IMPLEMENTATION COMPLETE — Ready for Railway deployment
**Baseline:** HEAD `9d991d8` · 2482 pass / 3–4 skip / 0 fail

---

## What Was Built

### Fix #1: player_type Backfill Script
**File:** `scripts/backfill_player_type.py` (102 lines)

**Purpose:** Backfill player_type from positions JSONB to fix batter routing
**Impact:** Converts 441 NULL rows (71%) to 'hitter' or 'pitcher'

**Key Logic:**
```sql
UPDATE player_projections
SET player_type = CASE
  WHEN positions ? ANY(array['SP','RP','P']) THEN 'pitcher'
  ELSE 'hitter'
END
WHERE player_type IS NULL
```

**Verification:** Script prints before/after distribution and confirms 0 NULLs remain

---

### Fix #2: Yahoo ID Sync Module
**File:** `backend/fantasy_baseball/yahoo_id_sync.py` (134 lines)

**Purpose:** Sync Yahoo player IDs from fantasy API to player_id_mapping table
**Impact:** Increases Yahoo ID coverage from 3.7% to target >50%

**Key Functions:**
- `sync_yahoo_player_ids()` — Fetches from mlb.l.72586, maps to BDL player_id
- `_lookup_bdl_id()` — Exact name match, then name+team match
- `run_yahoo_id_sync_job()` — Uses advisory lock 100_034

**Integration:** Wired into main.py as daily 6 AM ET scheduler job

---

### Fix #3: Park Factors Bulk-Loading
**File:** `backend/fantasy_baseball/ballpark_factors.py` (MODIFIED)

**Changes:**
1. Added imports: `from functools import lru_cache` and `Dict, Tuple` from typing
2. Added global cache: `_park_factor_cache: Dict[Tuple[str, str], float] = {}`
3. Added `load_park_factors()` function — loads all park_factors from DB into memory dict
4. Added `@lru_cache(maxsize=32)` decorator to `get_park_factor()`
5. Modified `get_park_factor()` to check cache first, then DB, then hardcoded constant

**Impact:** Eliminates N+1 queries — waiver endpoint should drop from 27s to <10s

---

### Fix #4: Startup Event + Scheduler Job
**File:** `backend/main.py` (MODIFIED)

**Changes:**
1. Added imports (line ~147):
   - `from backend.fantasy_baseball.ballpark_factors import load_park_factors`
   - `from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job`

2. Added startup call (line ~428):
   ```python
   # Load park factors into memory before starting scheduler
   try:
       load_park_factors()
   except Exception as e:
       logger.warning(f"Could not load park factors on startup: {e}")
   ```

3. Added Yahoo ID sync wrapper function (line ~1814):
   ```python
   def _yahoo_id_sync_job_wrapper():
       """Daily 6:00 AM Yahoo player ID sync."""
       try:
           count = run_yahoo_id_sync_job()
           logger.info(f"Yahoo ID sync completed: {count} players synced")
       except Exception as e:
           logger.exception(f"Yahoo ID sync job failed: {e}")
   ```

4. Added scheduler job (line ~426):
   ```python
   scheduler.add_job(
       _yahoo_id_sync_job_wrapper,
       CronTrigger(hour=6, minute=0, timezone=timezone),
       id="yahoo_id_sync",
       name="Yahoo Player ID Sync",
       replace_existing=True,
   )
   ```

---

## Verification Status

✅ All files compile without errors:
- `scripts/backfill_player_type.py` — OK
- `backend/fantasy_baseball/yahoo_id_sync.py` — OK
- `backend/fantasy_baseball/ballpark_factors.py` — OK
- `backend/main.py` — OK

✅ All imports verified:
- `yahoo_id_sync` imports OK
- `ballpark_factors` imports OK
- `main.py` imports OK (with venv/Scripts/python)

---

## Gemini CLI Deployment Instructions

### Phase 1: Deploy Code to Railway (10 min)

```bash
# Stage all changes
git add scripts/backfill_player_type.py
git add backend/fantasy_baseball/yahoo_id_sync.py
git add backend/fantasy_baseball/ballpark_factors.py
git add backend/main.py

# Commit (deployable unit)
git commit -m "fix(p0): backfill player_type NULLs + Yahoo ID sync + park_factors bulk-load

- Backfill player_type from positions JSONB (441 rows)
- Create yahoo_id_sync scheduler job (lock 100_034)
- Bulk-load park_factors to fix N+1 queries
- Add startup event to load park_factors on boot

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

### Phase 2: Execute Backfill (10 min)

```bash
# Run player_type backfill on Railway
railway run python scripts/backfill_player_type.py
```

**Expected output:**
```
Current player_type distribution:
  NULL       441
  pitcher    176

Backfilling player_type from positions JSONB...
Updated 441 rows

New player_type distribution:
  hitter     441
  pitcher    176

SUCCESS: All NULLs backfilled
```

**Verification:**
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
nulls = db.execute(text('SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL')).scalar()
print(f'Remaining NULLs: {nulls}')
db.close()
"
```

**Expected:** `Remaining NULLs: 0`

---

### Phase 3: Execute Yahoo ID Sync (15 min)

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

**Expected:** `Synced 200-300 Yahoo player IDs`

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

**Expected:** `Coverage: 52.3%` (or higher, was 3.7%)

---

### Phase 4: Verify Park Factors Performance (5 min)

**Check startup logs:**
```bash
railway logs --tail 50 | grep -i "park"
```

**Expected:** Should see `Loaded 30 park factors into memory`

**Test waiver endpoint performance:**
```bash
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommendations?league_id=72586" -o /dev/null -w "\nResponse time: %{time_total}s\n"
```

**Expected:** `Response time: <10s` (was 27s before fix)

---

### Phase 5: End-to-End Verification (10 min)

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

**Test 2: Verify performance improved**
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

Create `reports/2026-05-03-p0-fixes-deployment.md` with results:

```markdown
# P0 Fixes Deployment Report — May 3, 2026

## Deployment Summary
- **Commit:** [commit hash from git commit]
- **Deployed at:** [ISO 8601 timestamp]
- **Status:** SUCCESS / PARTIAL / FAILED

## Fix Results

### Fix #1: player_type Backfill
- **Before:** 441 NULL rows (71%)
- **After:** [X] NULL rows remaining
- **Status:** SUCCESS / PARTIAL

### Fix #2: Yahoo ID Sync
- **Before:** 3.7% coverage (372/10,096)
- **After:** [X]% coverage ([X]/10,096)
- **Status:** SUCCESS / PARTIAL

### Fix #3: Park Factors Bulk-Load
- **Before:** Waiver 27s, Dashboard 19s
- **After:** Waiver [X]s, Dashboard [X]s
- **Status:** SUCCESS / PARTIAL

## Verification Tests
- [ ] Batter routing works ([X] batters matched)
- [ ] Yahoo IDs present in mapping (sample of 10)
- [ ] Performance improved (all endpoints <10s)

## Issues Found
- [List any issues]
```

---

## Escalation Triggers

**Escalate to Claude Code immediately if:**

1. **Compilation errors:** Any file fails to compile
2. **Backfill fails:** `Remaining NULLs: > 0`
3. **Yahoo sync fails:** Coverage < 10% after sync
4. **Deploy breaks production:** /health returns not healthy
5. **Performance worse:** Any endpoint >30s after deploy
6. **Tests fail:** Batter routing still 0 matches

---

**File:** `.claude/delegation/gemini-p0-handoff.md`
**Next Action:** Deploy to Railway and execute all verification steps
