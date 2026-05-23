# 2026-05-22 Codex Deploy Report

## Summary

Wave 1 / Wave 2 code was deployed to Railway from the current local workspace on `stable/cbb-prod`.

Deployment result:

- Backend `Fantasy-App`: `6e9a5dc9-57d0-4806-889c-ca795874b9c9` — `SUCCESS`
- Frontend `observant-benevolence`: `8442b83b-67c9-4798-88a8-367588a50928` — `SUCCESS`

The rollout completed, both containers started, and the backend `/health` endpoint is healthy.

Budget verification is only partially successful:

- `week_label` = `Week 9` ✅
- `ip_accumulated` = `19.2` ✅
- `acquisitions_used` = `0` ❌

The incorrect acquisitions count is explained by a production backend warning after deploy: local fallback counting fails because table `roster_acquisitions` does not exist in production.

## Step 1: Branch / Merge Verification

Current branch:

- `stable/cbb-prod`

Important finding:

- local `main` exists, but it is **not** the freshest branch
- `main` tops out at `9790cab`
- `stable/cbb-prod` contains newer Wave work including:
  - `976a08f` `Merge agent/kimi/wave2-dashboard-waiver-ui: within-league transaction feed`
  - `a87a12c` `Merge agent/copilot/wave1-il-filter-eta-fix: IL filter + ETA fix`
  - `751bff8` `fix(projector): align _MLB_OPENING_DAY to first Yahoo matchup Monday`
  - `56e0bb4` `fix(budget): add Yahoo current_week sync guard with WARNING on mismatch`
  - `9e471db` `fix(budget): use first Yahoo matchup Monday as week epoch`

Conclusion:

- `main` does **not** appear to contain all latest Wave deploy content
- the deploy used the newer `stable/cbb-prod` workspace, which is the correct choice for “latest code”

## Step 2: Commit Log Verification

Recent commit evidence includes more than 6 relevant Wave commits / merges:

1. `976a08f` — Merge agent/kimi/wave2-dashboard-waiver-ui
2. `9d1bf1f` — within-league transaction feed for waiver intelligence
3. `a87a12c` — Merge agent/copilot/wave1-il-filter-eta-fix
4. `c6c734c` — IL hard gate + IL-type-aware ETA fix
5. `751bff8` — opening day / week boundary fix
6. `56e0bb4` — Yahoo current_week sync guard
7. `9e471db` — use first Yahoo matchup Monday as week epoch
8. `f4df2ac` — UI consistency fix

Note:

- `git log --pretty` shows author as `Simon Gray` on these commits, but the merge subjects and branch names confirm Kimi / Copilot wave work is present

## Step 3–4: Railway Deploy

Commands executed:

- backend upload to `Fantasy-App`
- frontend upload to `observant-benevolence`

Accepted deployment IDs:

- backend: `6e9a5dc9-57d0-4806-889c-ca795874b9c9`
- frontend: `8442b83b-67c9-4798-88a8-367588a50928`

Final Railway status:

- backend: `SUCCESS`
- frontend: `SUCCESS`

## Step 5: Health Check

Requested frontend URL check:

- `https://observant-benevolence-production.up.railway.app/health`

Observed result:

- returned the frontend login HTML shell, not a structured health JSON payload

Direct backend health check:

- `https://fantasy-app-production-5079.up.railway.app/health`

Observed result:

```json
{"status":"healthy","database":"connected","scheduler":"running"}
```

Conclusion:

- frontend is serving traffic
- backend health endpoint is healthy

## Step 6: Budget API Verification

Production budget response:

```json
{
  "budget": {
    "acquisitions_used": 0,
    "acquisitions_remaining": 8,
    "acquisition_limit": 8,
    "acquisition_warning": false,
    "il_used": 3,
    "il_total": 3,
    "ip_accumulated": 19.2,
    "ip_minimum": 18.0,
    "ip_pace": "AHEAD",
    "ip_data_available": true,
    "week_label": "Week 9",
    "weeks_remaining": 16,
    "days_in_week_remaining": 3,
    "acquisitions_this_season": 239
  },
  "freshness": {
    "primary_source": "yahoo",
    "is_stale": false
  }
}
```

Verification against requested criteria:

- `week_label shows 'Week 9'` — pass
- `ip_accumulated is non-zero (~19.2)` — pass
- `acquisitions_used shows correct count` — fail

## Step 7: Railway Logs, First 5 Minutes

Frontend:

- startup clean
- no error evidence surfaced in the checked logs

Backend:

- startup completed cleanly
- scheduler started
- async job queue processor ran successfully immediately after boot

However, budget endpoint logs exposed a production issue:

```text
budget: local acquisition query failed: (psycopg2.errors.UndefinedTable) relation "roster_acquisitions" does not exist
```

Related budget logs:

- Yahoo transactions fetched: `239`
- Yahoo weekly count computed: `0`
- local fallback query failed because `roster_acquisitions` table is missing

Interpretation:

- deploy itself succeeded
- the acquisition-count fix is still not functionally complete in production because the local fallback depends on a table absent from the deployed database

## Final Status

What is deployed:

- latest local backend workspace to Railway backend
- latest local frontend workspace to Railway frontend

What is verified working:

- backend deployment success
- frontend deployment success
- backend `/health` healthy
- budget week boundary fixed (`Week 9`)
- budget IP tracking fixed (`19.2`)

What is still broken:

- `acquisitions_used` remains `0`
- production logs show missing table `roster_acquisitions`, which breaks the intended local fallback path

## Recommended Next Action

Hand off to Claude for backend/database correction:

- either create / deploy the missing `roster_acquisitions` table
- or remove that dependency and make weekly acquisition counting work directly from Yahoo transactions in production
