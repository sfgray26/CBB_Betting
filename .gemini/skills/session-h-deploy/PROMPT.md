You are Gemini CLI on `cbb-edge`. Read `HANDOFF.md` Sections 2 and 3 before starting. You are doing DevOps operations only — no code changes.

## Railway Auth Reminder
The `RAILWAY_API_TOKEN` in `.env` is a workspace token. If you see "Unauthorized" on CLI commands, run:
```bash
. ./scripts/load-env.ps1 -Force
```
Project-level commands (`railway up`, `railway logs`, `railway variables`, `railway run`) all work with this token. Account-level commands (`railway list`) will fail — this is expected.

---

## Session H Deployment

Claude Code has committed Session H (ff7b5a6 and follow-ups). Execute the deployment using the `session-h-deploy` skill:

### Step 1: Pre-Deploy Validation
- Syntax-check all modified `.py` files (`python -m py_compile`)
- Run `python -m pytest tests/test_waiver_integration.py tests/test_mlb_analysis.py -q --tb=short`
- If tests fail → STOP and report to Claude Code

### Step 2: Deploy
```bash
railway up
```
Wait for build. Then verify health:
```bash
curl -s --max-time 20 https://fantasy-app-production-5079.up.railway.app/health
```
Expected: `{"status":"healthy","database":"connected","scheduler":"running"}`

### Step 3: Drop bdl_stat_id column
```bash
railway run python scripts/migrations/drop_bdl_stat_id.py
```
Expected: `bdl_stat_id dropped (or did not exist).`

### Step 4: V31 rolling backfill (allow up to 10 minutes)
```bash
railway run python scripts/backfill_v31_rolling.py --execute
```
Watch for progress every 500 rows. Report final "N rows updated" count.

### Step 5: V32 z-score backfill (run ONLY after Step 4 completes)
```bash
railway run python scripts/backfill_v32_zscores.py --execute
```
Same pattern. Report final count.

### Step 6: Re-trigger valuation cache
```bash
curl -s -X POST --max-time 30 https://fantasy-app-production-5079.up.railway.app/admin/refresh-valuation-cache
```
Report response body.

### Step 7: Spot-check queries
Run via `@postgres` MCP:

```sql
SELECT COUNT(*) AS w_runs_populated FROM player_rolling_stats WHERE w_runs IS NOT NULL;
SELECT COUNT(*) AS z_r_populated FROM player_scores WHERE z_r IS NOT NULL;
SELECT scarcity_rank, COUNT(*) AS cnt FROM position_eligibility GROUP BY scarcity_rank ORDER BY scarcity_rank;
SELECT quality_score IS NULL AS is_null, COUNT(*) FROM probable_pitchers GROUP BY quality_score IS NULL;
SELECT COUNT(*) AS rows FROM information_schema.columns WHERE table_name='mlb_player_stats' AND column_name='bdl_stat_id';
```

Expected:
- `w_runs_populated` > 60,000
- `z_r_populated` > 60,000
- `scarcity_rank` has non-null values (1-12)
- `quality_score IS NULL` = false has count > 0 (no nulls)
- `bdl_stat_id` column count = 0

---

## REPORT BACK
For each step: result / row counts / any errors. Do NOT escalate to code changes — if any step fails, report the exact error and stop. Claude will fix it.
