# Phase 4 Deployment Verification — DevOps Task

**Assigned to:** Gemini CLI (DevOps Lead)
**Date:** 2026-05-05
**Architect:** Claude Code (Master Architect)
**Estimated time:** 30 minutes
**Priority:** P0 — Required before market signals go live

---

## Context

Phase 4 (Market Signals Engine) was just completed by the Architect. This adds Yahoo ownership-based market intelligence to the fantasy platform. All code is committed and tests pass locally, but we need DevOps verification before production activation.

**What was built:**
- Pure computation module: `backend/services/market_engine.py` (ownership_velocity, market_score, tag classification)
- Daily ingestion job: 8:30 AM ET scheduler task (lock 100_038)
- Tiebreaker integration: market_score as tertiary sort key in `waiver_edge_detector.py`
- Database schema: `player_market_signals` table (migration script exists)
- Tests: 43 tests total (29 market_engine + 14 waiver_edge) — all passing locally

**Architecture principles:**
- Market score is a TIEBREAKER, not a primary signal (max 10% weight)
- Feature flags control activation: `market_signals_enabled` = false initially
- Pure computation isolated from DB queries (clean separation)

---

## Your Tasks

### Task 1: Run Migration on Railway

Connect to Railway and run the migration script:

```bash
railway ssh
python scripts/migration_player_market_signals.py
```

**Expected output:** "PR 4.1 migration ready. Verified: player_market_signals exists."

**Verify table created:**
```sql
SELECT table_name FROM information_schema.tables
WHERE table_schema = 'public' AND table_name = 'player_market_signals';
```

**Verify indexes:**
```sql
SELECT indexname FROM pg_indexes
WHERE tablename = 'player_market_signals';
```

Should see: `idx_market_signals_player_date`, `idx_market_signals_date_score`

---

### Task 2: Seed Feature Flags

The feature flags control whether market signals affect scoring. We seed them as FALSE initially (log-only mode).

```sql
INSERT INTO threshold_config (key, value, description)
VALUES
  ('market_signals_enabled', 'false', 'PR 4: Market score tiebreaker in waiver recommendations'),
  ('opportunity_enabled', 'false', 'PR 3: Opportunity adjustment in player_scores')
ON CONFLICT (key) DO NOTHING;
```

**Verify seeded:**
```sql
SELECT key, value, description FROM threshold_config WHERE key LIKE '%_enabled';
```

Expected: 2 rows with `value = 'false'`

---

### Task 3: Verify Daily Job Scheduler

The market signals job runs daily at 8:30 AM ET. Verify it's registered correctly.

**Check lock ID in code:**
```bash
grep "market_signals_update" backend/services/daily_ingestion.py | head -5
```

Should see lock ID mapping: `"market_signals_update": 100_038`

**Check job schedule:**
```bash
grep -A10 "Market signals: daily 8:30 AM ET" backend/services/daily_ingestion.py
```

Should see: `CronTrigger(hour=8, minute=30, timezone=tz)`

**Verify advisory lock available:**
```sql
SELECT pg_try_advisory_lock(100_038);
-- Should return 't' (true = available)
SELECT pg_advisory_unlock(100_038);
```

---

### Task 4: Run Tests on Railway

Tests pass locally on Windows. Verify they also pass on Railway (Linux environment).

```bash
railway ssh
python -m pytest tests/test_market_engine.py tests/test_waiver_edge.py -v --tb=short
```

**Expected:** All 43 tests passing
- `test_market_engine.py`: 29 tests (velocity, deltas, add/drop ratio, tags, market_score)
- `test_waiver_edge.py`: 14 tests (waiver recommendations with new tiebreaker)

If tests fail, capture full error output with `--tb=long`.

---

## Success Criteria

Before reporting complete, verify ALL of:

- [ ] Migration script ran without errors
- [ ] `player_market_signals` table exists with correct schema
- [ ] Indexes created: `idx_market_signals_player_date`, `idx_market_signals_date_score`
- [ ] Feature flags seeded: `market_signals_enabled`, `opportunity_enabled` (both = false)
- [ ] Lock ID 100_038 is available (pg_try_advisory_lock returns 't')
- [ ] Daily job scheduled for 8:30 AM ET (grep confirmation)
- [ ] All 43 tests pass on Railway

---

## Report Template

After completion, report back with:

```markdown
## Phase 4 Deployment Verification Report

**Date:** 2026-05-05
**Engineer:** Gemini CLI
**Commit:** 8ee4c8c

### Migration Status
- player_market_signals table: [CREATED/FAILED]
- idx_market_signals_player_date: [CREATED/MISSING]
- idx_market_signals_date_score: [CREATED/MISSING]

### Feature Flags (threshold_config)
- market_signals_enabled: [false/true] [SEEDED/MISSING]
- opportunity_enabled: [false/true] [SEEDED/MISSING]

### Advisory Lock Verification
- Lock ID 100_038: [AVAILABLE/TAKEN/ERROR]

### Scheduler Verification
- Job registered: [YES/NO]
- Schedule: [8:30 AM ET/INCORRECT]
- Grep output: [paste relevant lines]

### Test Results (Railway)
- test_market_engine.py: [X/29] passing
- test_waiver_edge.py: [X/14] passing
- Total: [X/43] passing
- Failures: [list any failing tests]

### Issues Found
[None or detailed description of any errors]

### Deviation from Instructions
[None or explain what you did differently]

### Recommendation
- [ ] PROCEED TO PRODUCTION — All checks passed
- [ ] NEEDS FIXES — See Issues Found above

### Next Steps
1. Market signals job will run at 8:30 AM ET tomorrow
2. Monitor logs for successful execution: `railway logs --filter "market_signals_update"`
3. After 48h of data, Architect will verify signal quality before setting feature flag to true
```

---

## Troubleshooting

**If migration fails:**
- Check DATABASE_URL points to correct Postgres (Postgres-ygnV, not CBB)
- Verify migration file exists: `ls scripts/migration_player_market_signals.py`
- Check file permissions: `chmod +x scripts/migration_player_market_signals.py`

**If tests fail on Railway:**
- Capture full traceback: `python -m pytest tests/test_market_engine.py --tb=long`
- Check Python version: `python --version` (should be 3.11+)
- Verify dependencies: `pip list | grep -i pytest`

**If feature flags already exist:**
- The `ON CONFLICT DO NOTHING` clause is safe
- Verify current values: `SELECT key, value FROM threshold_config WHERE key LIKE '%_enabled';`

**If lock ID is taken:**
- Check if job is already running: `SELECT * FROM pg_locks WHERE classid = 100_038;`
- If no job running, force unlock: `SELECT pg_advisory_unlock(100_038);`

---

## Questions?

If anything is unclear or you encounter unexpected errors, STOP and ask the Architect for clarification. Do not guess — this is production infrastructure.

**DO NOT:**
- Modify the migration script
- Change feature flag values to true
- Delete or rename the player_market_signals table
- Run the migration more than once (it's idempotent via ON CONFLICT, but no need)

**ASK FIRST:**
- If you see any error messages during migration
- If tests fail with unexpected errors
- If the table already exists with different schema
- If you're unsure about any step
