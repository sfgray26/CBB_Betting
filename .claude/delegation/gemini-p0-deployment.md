# Task: Deploy & Verify P0 Fixes on Railway

**Agent:** Gemini CLI (DevOps Strike Lead)
**Objective:** Deploy all P0 fixes to Railway production and verify they work
**Timebox:** 1 hour
**Deliverables:** All fixes deployed, verification queries pass

---

## Mission

Claude Code has created the implementation. Your job: deploy to Railway and verify all fixes work.

DO NOT modify any code. Your role is deployment, compilation checks, and verification only.

---

## Phase 1: Pre-Deployment Compilation (10 min)

### Step 1: Verify Claude's code compiles

```bash
# Check all files compile
python -m py_compile scripts/backfill_player_type.py
python -m py_compile backend/fantasy_baseball/yahoo_id_sync.py
python -m py_compile backend/fantasy_baseball/ballpark_factors.py
python -m py_compile backend/main.py

# Verify imports
python -c "from backend.fantasy_baseball.yahoo_id_sync import sync_yahoo_player_ids; print('✅ yahoo_id_sync OK')"
python -c "from backend.fantasy_baseball.ballpark_factors import load_park_factors; print('✅ ballpark_factors OK')"
python -c "from backend.main import app; print('✅ main.py OK')"
```

**Expected:** All ✅ marks, no errors

**If errors:** Escalate to Claude Code with exact error message

---

## Phase 2: Deploy to Railway (10 min)

### Step 1: Deploy code changes

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

**If unhealthy:** Check Railway logs: `railway logs`

---

## Phase 3: Execute Backfill (10 min)

### Step 1: Run player_type backfill on production

```bash
# Execute backfill script on Railway
railway run python scripts/backfill_player_type.py
```

**Expected output:**
```
Current player_type distribution:
  NULL       441
  pitcher    176

Backfilling player_type from positions JSONB...
✅ Updated 441 rows

New player_type distribution:
  hitter     441
  pitcher    176

✅ SUCCESS: All NULLs backfilled
```

**Step 2: Verify 0 NULLs remain**

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

**If NOT 0:** Escalate to Claude Code

---

## Phase 4: Execute Yahoo ID Sync (15 min)

### Step 1: Run manual Yahoo ID sync

```bash
# Run sync job on Railway
railway run python -c "
from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job
import logging
logging.basicConfig(level=logging.INFO)
count = run_yahoo_id_sync_job()
print(f'Synced {count} Yahoo player IDs')
"
```

**Expected:** `Synced 200-300 Yahoo player IDs`

**Step 2: Verify coverage increased

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

**If < 10%:** Escalate to Claude Code (sync failed)

---

## Phase 5: Verify Park Factors Loaded (5 min)

### Step 1: Check startup logs

```bash
# Check recent logs for park factor loading
railway logs --tail 50 | grep -i "park"
```

**Expected:** Should see `✅ Park factors loaded into memory`

**Step 2: Test waiver endpoint performance

```bash
# Time the waiver recommendations endpoint
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommendations?league_id=72586" -o /dev/null -w "\nResponse time: %{time_total}s\n"
```

**Expected:** `Response time: <10s` (was 27s before fix)

**If still >20s:** Park factors cache not working, escalate

---

## Phase 6: End-to-End Verification (10 min)

### Test 1: Verify batter routing works

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
# Test batter accuracy query (should return > 0 matches now)
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
    print('✅ Batter routing works')
else:
    print('❌ Batter routing still broken')
db.close()
"
```

**Expected:** `Batters matched: 50+` (was 0)

### Test 2: Verify Yahoo IDs in mapping

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
# Sample 10 players with yahoo_id
rows = db.execute(text('''
  SELECT pim.yahoo_id, pp.player_name
  FROM player_id_mapping pim
  JOIN player_projections pp ON pim.player_id = pp.player_id
  WHERE pim.yahoo_id IS NOT NULL
  LIMIT 10
''')).fetchall()

print(f'Sample players with Yahoo IDs:')
for r in rows:
    print(f'  {r[1]:30} yahoo_id={r[0]}')
db.close()
"
```

**Expected:** Should see 10 players with yahoo_id values

### Test 3: Verify performance improved

```bash
echo "Testing endpoint performance..."

echo -n "Dashboard: "
time curl -s "https://cbb-edge-production.up.railway.app/api/dashboard" -o /dev/null -w "%{time_total}s\n"

echo -n "Waiver: "
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommendations?league_id=72586" -o /dev/null -w "%{time_total}s\n"

echo -n "Optimizer: "
time curl -s -X POST "https://cbb-edge-production.up.railway.app/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}' \
  -o /dev/null -w "%{time_total}s\n"
```

**Expected:**
- Dashboard: <10s (was 19.34s → 9.95s after caching)
- Waiver: <10s (was 27s)
- Optimizer: <1s (already fast)

---

## Deliverable: Create Report

Create `reports/2026-05-03-p0-fixes-deployment.md`:

```markdown
# P0 Fixes Deployment Report — May 3, 2026

## Deployment Summary
- **Commit:** [commit hash from git commit]
- **Deployed at:** [ISO 8601 timestamp]
- **Status:** ✅ SUCCESS / ⚠️ PARTIAL / ❌ FAILED

## Fix Results

### Fix #1: player_type Backfill
- **Before:** 441 NULL rows (71%)
- **After:** [X] NULL rows remaining
- **Status:** ✅ SUCCESS / ⚠️ PARTIAL

### Fix #2: Yahoo ID Sync
- **Before:** 3.7% coverage (372/10,096)
- **After:** [X]% coverage ([X]/10,096)
- **Status:** ✅ SUCCESS / ⚠️ PARTIAL

### Fix #3: Park Factors Bulk-Load
- **Before:** Waiver 27s, Dashboard 19s
- **After:** Waiver [X]s, Dashboard [X]s
- **Status:** ✅ SUCCESS / ⚠️ PARTIAL

## Verification Tests
- [ ] Batter routing works ([X] batters matched)
- [ ] Yahoo IDs present in mapping (sample of 10)
- [ ] Performance improved (all endpoints <10s)

## Issues Found
- [List any issues that require escalation]
- [Include error messages if applicable]

## Recommendations
- [If all fixes successful: Ready for feature work]
- [If partial: What needs to be re-deployed?]
- [If failed: What blockers need Claude Code intervention?]

---
**Reported by:** Gemini CLI (DevOps)
**Verified at:** [Timestamp]
```

---

## Escalation Triggers

**Escalate to Claude Code immediately if:**

1. **Compilation errors:** Any file fails to compile
   - Action: Stop deployment, send exact error message

2. **Backfill fails:** `Remaining NULLs: > 0`
   - Action: Check positions JSONB data, may need different backfill logic

3. **Yahoo sync fails:** Coverage < 10% after sync
   - Action: Check Yahoo API connection, verify league_key format

4. **Deploy breaks production:** /health returns not healthy
   - Action: Railway rollback: `railway rollback`

5. **Performance worse:** Any endpoint >30s after deploy
   - Action: Rollback, investigate park_factors cache

6. **Tests fail:** Batter routing still 0 matches
   - Action: Check if backfill actually ran, verify data

---

## Success Criteria

- [ ] All files compile without errors
- [ ] Deploy succeeds (railway up completes)
- [ ] /health returns healthy status
- [ ] player_type NULLs: 0 (was 441)
- [ ] Yahoo ID coverage: >50% (was 3.7%)
- [ ] Waiver response time: <10s (was 27s)
- [ ] Dashboard response time: <10s (was 19s)
- [ ] Batter routing test: >0 matches (was 0)
- [ ] Report saved to `reports/2026-05-03-p0-fixes-deployment.md`

**DO NOT:**
- ❌ Modify any code (that's Claude's job)
- ❌ Run ad-hoc SQL queries without verification
- ❌ Skip verification steps (all gates must pass)

**DO:**
- ✅ Follow the checklist in order
- ✅ Document all results in report
- ✅ Escalate immediately if blockers found
- ✅ Test rollback if deploy breaks production

---

**File:** `.claude/delegation/gemini-p0-deployment.md`
**Timebox:** 1 hour
**Priority:** P0 — Blocks all feature work
