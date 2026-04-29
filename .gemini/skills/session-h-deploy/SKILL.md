---
name: session-h-deploy
description: Deploy Session H commits and run all post-deploy backfills and verifications. Use ONLY when Claude Code says Session H is ready to deploy.
---

# Session H Deployment & Backfill

## Prerequisites

- Claude Code has confirmed Session H code is complete and committed
- `scripts/backfill_v31_rolling.py`, `scripts/backfill_v32_zscores.py`, and `scripts/migrations/drop_bdl_stat_id.py` all exist
- Railway workspace token is loaded (run `. ./scripts/load-env.ps1 -Force` if you see "Unauthorized")

## Workflow — Execute in EXACT Order

### Step 1: Pre-Deploy Validation
```bash
# Syntax check modified files
for f in $(git diff --name-only HEAD | grep '\.py$'); do
  python -m py_compile "$f" || { echo "SYNTAX ERROR in $f"; exit 1; }
done

# Run critical test subset
python -m pytest tests/test_waiver_integration.py tests/test_mlb_analysis.py -q --tb=short
```
- If tests fail → STOP, report output to Claude Code

### Step 2: Deploy
```bash
railway up
```
- Wait for build to complete (~2-5 minutes)

### Step 3: Verify deployment health
```bash
curl -s --max-time 20 https://fantasy-app-production-5079.up.railway.app/health
```
- Expected: `{"status":"healthy","database":"connected","scheduler":"running"}`
- If not healthy → run `railway logs | head -50`, report errors to Claude Code

### Step 4: Drop bdl_stat_id column
```bash
railway run python scripts/migrations/drop_bdl_stat_id.py
```
- Expected output: `bdl_stat_id dropped (or did not exist).`

### Step 5: V31 rolling backfill (allow up to 10 minutes)
```bash
railway run python scripts/backfill_v31_rolling.py --execute
```
- Watch for progress logs every 500 rows
- Report final "N rows updated" count

### Step 6: V32 z-score backfill (run ONLY after Step 5 completes)
```bash
railway run python scripts/backfill_v32_zscores.py --execute
```
- Same pattern. Report final count.

### Step 7: Re-trigger valuation cache
```bash
curl -s -X POST --max-time 30 https://fantasy-app-production-5079.up.railway.app/admin/refresh-valuation-cache
```
- Report the response body

### Step 8: Spot-check queries
Run via `@postgres` MCP or `railway run python -c` with psycopg2:

```sql
SELECT COUNT(*) AS w_runs_populated FROM player_rolling_stats WHERE w_runs IS NOT NULL;
SELECT COUNT(*) AS z_r_populated FROM player_scores WHERE z_r IS NOT NULL;
SELECT scarcity_rank, COUNT(*) AS cnt FROM position_eligibility GROUP BY scarcity_rank ORDER BY scarcity_rank;
SELECT quality_score IS NULL AS is_null, COUNT(*) FROM probable_pitchers GROUP BY quality_score IS NULL;
SELECT COUNT(*) AS rows FROM information_schema.columns WHERE table_name='mlb_player_stats' AND column_name='bdl_stat_id';
```

**Expected results:**
- `w_runs_populated` > 60,000
- `z_r_populated` > 60,000
- `scarcity_rank` has non-null values (1-12)
- `quality_score IS NULL` = false has count > 0 (no nulls)
- `bdl_stat_id` column count = 0 (column dropped)

## Rules

- **NEVER** skip Step 1 (pre-deploy validation)
- **NEVER** run Step 6 before Step 5 completes (V32 depends on V31 data)
- If ANY step fails → STOP, capture exact error output, and report to Claude Code
- Do NOT retry failed steps without Claude Code approval
- Do NOT modify scripts or code — only run what Claude Code prepared

## Report Format

For each step, report:
1. Step number and name
2. Result (PASS / FAIL)
3. Key output or row counts
4. Any errors (verbatim)
