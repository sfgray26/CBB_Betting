# CODEX DEPLOYMENT INSTRUCTIONS — P0 PR #98

> **Date:** 2026-06-10
> **PR:** https://github.com/sfgray26/CBB_Betting/pull/98
> **Branch:** `stable/cbb-prod`
> **Agent:** Codex (DevOps Lead)
> **Priority:** P0 (Critical UAT blocker deployment)

---

## CONTEXT

### What We're Deploying

PR #98 implements 4 critical UAT fixes from the June 10 UAT analysis:

1. **Real-Time Availability Guard (P0)**
   - `DailyAvailabilityOverride` DB table for day-off blacklist (Caballero pattern)
   - Admin API: `POST/DELETE /api/admin/availability-override`
   - Blacklisted players suppressed to score 0 in waiver rankings
   - Color-coded injury badges (red=IL, amber=DTD)

2. **Roster Constraint Awareness Layer (P0)**
   - IL slot capacity check before ADD_DROP recommendations
   - FAAB balance check for free-agent-only mode
   - `constraint_warning` field in `RosterMoveRecommendation`
   - Frontend warning strip with icon

3. **Dashboard IL Crisis Detection (P0)**
   - Detects 3+ rostered players with confirmed injury status in active slots
   - Overrides false-positive "Lineup Gaps: none" with ROSTER EMERGENCY
   - Action link to `/war-room/roster`
   - Emergency-styled UI (red border, bold text)

4. **Win Probability Context Labels (P1)**
   - War Room: `Week N · IN-FLIGHT` (gold badge)
   - Weekly Preview: `Week N · PREVIEW` (blue badge)
   - Resolves confusing 6% vs 100% win probability discrepancy

### Test Results
- `pytest tests/test_availability_guard.py tests/test_dashboard_il_crisis.py -v` — **13/13 pass**
- `pytest tests/ -q` — **3073 pass, 4 pre-existing failures, zero regressions**
- All syntax checks clean across 5 backend files

### UAT Critical Findings Addressed

| UAT Finding | Fix | Severity |
|-------------|-----|----------|
| José Caballero recommended on day off | `DailyAvailabilityOverride` table | 🔴 P0 |
| Casey Mize "must add" with full IL | IL slot capacity check | 🔴 P0 |
| IL crisis shows "no gaps" | Dashboard crisis detection | 🔴 P0 |
| Win prob confusion (6% vs 100%) | Week context badges | 🔴 P0 |

---

## PRE-DEPLOYMENT CHECKLIST

### 1. Setup Railway SSH Key (BLOCKING)

**Current State:** `railway ssh-keys list` returns "No SSH keys registered"

**Required Action:**
```powershell
# In PowerShell
cd /mnt/c/Users/sfgra/repos/Fixed/cbb-edge

# Step 1: Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "sfgray26+railway@github.com" -f ~/.ssh/railway_ed25519

# Step 2: Copy public key
cat ~/.ssh/railway_ed25519.pub

# Step 3: Add to Railway
railway ssh-keys add ~/.ssh/railway_ed25519.pub

# Step 4: Verify
railway ssh-keys list
```

**Expected Output:**
```
┌──────────────────────────────────┬────────────┬───────────────┐
│ ID                               │ NAME       │ FINGERPRINT   │
├──────────────────────────────────┼────────────┼───────────────┤
│ <key-id>                         │ railway... │ <fingerprint> │
└──────────────────────────────────┴────────────┴───────────────┘
```

**Time Estimate:** 5-10 minutes

### 2. Verify Production Config

```powershell
# Check Railway environment variables
railway variables | grep -i "predictive_stats_v1_enabled\|freshness_threshold"

# Verify feature flag is OFF (P1 not deployed yet)
# Should return: predictive_stats_v1_enabled=false
```

### 3. Check Current Deployment Status

```powershell
# Verify current production is healthy
railway status

# Check recent logs for errors
railway logs --lines 50

# Verify health endpoint is responding
railway run curl -s https://observant-benevolence-production.up.railway.app/health | jq
```

**Expected:** Status should be `active`, health endpoint returns `{"status":"healthy"}`

---

## DEPLOYMENT STEPS

### Step 1: Merge PR to Production Branch

```powershell
# Switch to stable/cbb-prod (should already be there)
git checkout stable/cbb-prod

# Pull latest from origin
git pull origin stable/cbb-prod

# Verify you're on the correct commit
git log --oneline -1
# Should show commit message for P0 fixes
```

### Step 2: Trigger Railway Deployment

```powershell
# Push to Railway (triggers automatic deploy)
railway up

# Monitor deployment logs
railway logs --follow
```

**Expected Behavior:**
- Railway will build Docker image
- Run migrations (new `DailyAvailabilityOverride` table)
- Deploy to production
- Health check will run automatically

**Time Estimate:** 5-8 minutes

### Step 3: Verify Deployment Success

```powershell
# Check deployment status
railway status

# Should show: status=active, health=healthy
```

**Log Indicators of Success:**
```
✓ Build completed
✓ Database migrations applied
✓ Health check passed
✓ Service is healthy
```

**Log Indicators of Failure:**
```
✗ Build failed
✗ Migration error
✗ Health check failed
✗ Service unhealthy
```

---

## POST-DEPLOYMENT SMOKE TESTS

### Test 1: Health Endpoint

```powershell
railway run curl -s https://observant-benevolence-production.up.railway.app/health | jq
```

**Expected:**
```json
{
  "status": "healthy",
  "timestamp": "2026-06-10T...",
  "services": {
    "database": "connected",
    "yahoo_api": "connected",
    "schedule_feed": "active"
  }
}
```

### Test 2: New API Endpoints

```powershell
# Test daily availability override endpoint (should return empty list initially)
railway run curl -s https://observant-benevolence-production.up.railway.app/api/admin/availability-override | jq

# Expected: []
```

### Test 3: Database Table Created

```powershell
# Verify DailyAvailabilityOverride table exists
railway run python -c "
from backend.database import get_db
from sqlalchemy import inspect

engine = get_db()
inspector = inspect(engine)
tables = inspector.get_table_names()
print('DailyAvailabilityOverride exists:', 'daily_availability_override' in tables)
"
```

**Expected:** `DailyAvailabilityOverride exists: True`

### Test 4: Waiver Wire Response Has New Fields

```powershell
# Check that waiver responses include availability_note and constraint_warning
railway run curl -s "https://observant-benevolence-production.up.railway.app/api/fantasy/waiver" | jq '.[0].availability_note'
```

**Expected:** Either `null` or a string value (not missing field)

### Test 5: Dashboard Gaps Endpoint

```powershell
# Verify IL crisis detection works
railway run curl -s "https://observant-benevolence-production.up.railway.app/api/fantasy/dashboard" | jq '.lineup_gaps'
```

**Expected:** Array of gaps (may be empty if no crisis, but field must exist)

### Test 6: Week Context Badges

```powershell
# War Room should return week number
railway run curl -s "https://observant-benevolence-production.up.railway.app/api/fantasy/matchup" | jq '.week'

# Weekly Preview should return week number
railway run curl -s "https://observant-benevolence-production.up.railway.app/api/fantasy/preview" | jq '.week_number'
```

**Expected:** Valid week integers (e.g., `12`, `13`)

---

## ROLLBACK PROCEDURE

### If Deployment Fails

```powershell
# 1. Identify the last successful commit
git log --oneline stable/cbb-prod -5

# 2. Reset to the last known good commit
git reset --hard <last-good-commit-hash>

# 3. Force push to Railway
railway up

# 4. Monitor rollback
railway logs --follow
```

### If Smoke Tests Fail

```powershell
# 1. Check error logs
railway logs --lines 100 | grep -i "error\|exception\|failed"

# 2. If critical errors, rollback immediately (see above)

# 3. Report to HANDOFF.md with specific error messages
```

---

## HANDOFF TO HERMES

After deployment (successful or failed), update HANDOFF.md:

```markdown
### 2026-06-10 — P0 Deployment — [SUCCESS|FAILED]

**Deployer:** Codex
**PR:** #98
**Status:** [SUCCESS|FAILED]
**Timestamp:** 2026-06-10 [TIME]

**Deployed Changes:**
- Task 1: Real-Time Availability Guard
- Task 2: Roster Constraint Awareness
- Task 3: Dashboard IL Crisis Detection
- Task 4: Win Probability Context Labels

**Smoke Test Results:**
- [x] Health endpoint: [PASS|FAIL]
- [x] DailyAvailabilityOverride table: [PASS|FAIL]
- [x] Waiver wire new fields: [PASS|FAIL]
- [x] Dashboard IL crisis detection: [PASS|FAIL]
- [x] Week context badges: [PASS|FAIL]

**Issues Found:**
- [None|Specific issues documented]

**Next Steps:**
- [If SUCCESS] Route UAT regression testing
- [If FAILED] Rollback + investigate root cause
```

---

## ESCALATION PATHS

### If You're Blocked

| Situation | Escalate To | How |
|-----------|-------------|-----|
| Railway SSH key setup fails | Claude Code | Handoff with error message |
| Migration fails | Claude Code | Migration script + error log |
| Health check fails | Claude Code | Health endpoint response + logs |
| Smoke test fails | Claude Code | Test output + logs |
| Rollback required | Hermes | HANDOFF.md update + rollback summary |

### What Codex Should NOT Do

❌ Do NOT edit any file in `backend/`, `frontend/`, or `tests/` during deployment
❌ Do NOT write DB migration scripts (Claude writes, Codex runs)
❌ Do NOT modify application logic
❌ Do NOT deploy to Railway without completing SSH key setup

---

## CONSTRAINTS (Per AGENTS.md)

**Codex Restrictions:**
- ✅ `railway logs --follow` — permitted
- ✅ Railway dashboard env var changes — permitted
- ✅ Running pre-approved scripts — permitted
- ✅ CI/CD pipeline changes — permitted
- ✅ Triggering Railway redeploys (`railway up`) — permitted
- ✅ Infrastructure-as-code configs — permitted
- ✅ Operational `.md` file updates — permitted
- ❌ Editing `backend/`, `frontend/`, `tests/` — **NOT permitted**
- ❌ Writing DB migration scripts — **NOT permitted** (Claude writes, Codex runs)
- ❌ Modifying Python/TypeScript application logic — **NOT permitted**

---

## TIME ESTIMATES

| Phase | Estimate |
|-------|----------|
| SSH key setup | 5-10 minutes |
| Pre-deployment checks | 5 minutes |
| Railway deployment | 5-8 minutes |
| Smoke tests | 10-15 minutes |
| **Total** | **25-38 minutes** |

---

## SUCCESS CRITERIA

Deployment is successful when:
- ✅ Railway status shows `active` and `healthy`
- ✅ All 6 smoke tests pass
- ✅ No errors in deployment logs
- ✅ Health endpoint returns `{"status":"healthy"}`
- ✅ HANDOFF.md updated with deployment summary

---

## READY TO PROCEED

**Prerequisites Met:**
- [ ] PR #98 reviewed and approved
- [ ] Test results verified (3073 passed, zero regressions)
- [ ] AGENTS.md constraints understood
- [ ] Rollback procedure reviewed

**When Ready:**
1. Setup Railway SSH key
2. Complete pre-deployment checklist
3. Execute deployment steps
4. Run smoke tests
5. Update HANDOFF.md

---

*Document created by Hermes (Session Orchestrator) — 2026-06-10*
*Refer to AGENTS.md lines 52-91 for Codex swimlane and permissions*