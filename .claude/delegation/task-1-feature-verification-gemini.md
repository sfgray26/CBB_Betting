# Task 1: Fantasy Feature Verification (Gemini CLI)

**Priority:** P0 — Foundation for all work
**Assigned:** Gemini CLI (DevOps Strike Lead)
**Escalation:** Claude Code if endpoints fail or require backend fixes
**Timebox:** 2 hours

---

## Mission

Verify that the 38 deployed fantasy endpoints actually work end-to-end. We cannot build on features that may be broken.

---

## Exact Commands

### Step 1: Get production health

```bash
railway status
curl https://cbb-edge-production.up.railway.app/health
```

Expected: `{"status":"healthy","database":"connected","scheduler":"running"}`

### Step 2: Test core fantasy endpoints

```bash
# Test 1: Lineup endpoint (already verified, but re-check)
curl "https://cbb-edge-production.up.railway.app/api/fantasy/lineup?date=2026-05-02" | head -c 500

# Test 2: Roster optimization (PRIORITY - timing out)
time curl -X POST "https://cbb-edge-production.up.railway.app/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}'

# Test 3: Matchup preview (uses MCMC win_prob)
curl "https://cbb-edge-production.up.railway.app/api/fantasy/matchup/preview?date=2026-05-02" | jq '.win_prob'

# Test 4: Waiver recommendations
curl "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommend?league_id=72586" | head -c 500

# Test 5: Player projections
curl "https://cbb-edge-production.up.railway.app/api/fantasy/projections?player_type=hitter&limit=5" | jq '.[0] | {player_name, cat_scores}'
```

### Step 3: Document results

Create file: `reports/2026-05-03-fantasy-endpoint-verification.md`

```markdown
# Fantasy Endpoint Verification — May 3, 2026

## Health Check
- Status: [healthy/database_connected/scheduler_running]
- Timestamp: [ISO 8601]

## Endpoint Results

| Endpoint | Status | Response Time | Notes |
|----------|--------|---------------|-------|
| GET /api/fantasy/lineup | [✅/❌] | [Xms] | [Notes] |
| POST /api/fantasy/roster/optimize | [✅/❌/TIMEOUT] | [Xms] | [Notes] |
| GET /api/fantasy/matchup/preview | [✅/❌] | [Xms] | [Notes] |
| GET /api/fantasy/waiver/recommend | [✅/❌] | [Xms] | [Notes] |
| GET /api/fantasy/projections | [✅/❌] | [Xms] | [Notes] |

## Critical Findings
- [List any errors, timeouts, or unexpected behavior]
- [Note if optimizer timeout (>30s) is confirmed]
- [Check if win_prob varies (not constant 0.763)]

## Recommendations
- [What needs fixing before proceeding]
```

---

## Success Criteria

- [ ] All 5 endpoints tested with documented results
- [ ] Response times recorded
- [ ] Any failures or timeouts flagged for escalation
- [ ] Report saved to `reports/2026-05-03-fantasy-endpoint-verification.md`

---

## Escalation Triggers

**Escalate to Claude Code immediately if:**
1. `/api/fantasy/roster/optimize` times out (>30s) — **P0 blocker**
2. `/api/fantasy/matchup/preview` returns `win_prob=0.763` constant — **regression**
3. Any endpoint returns 500 error — **backend bug**
4. Database connection fails — **production outage**

**Do NOT attempt to fix backend code yourself** — your role is verification and reporting only.

---

## Reporting Format

After completion, create a GitHub issue with title:
```
[Verification] Fantasy Endpoint Health Check — May 3, 2026
```

Body: Paste the contents of `reports/2026-05-03-fantasy-endpoint-verification.md`

Tag @claude-code (me) for review.
