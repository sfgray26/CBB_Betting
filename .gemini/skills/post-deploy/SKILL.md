---
name: post-deploy
description: Verify a deployment succeeded and the system is healthy after railway up. Use immediately after any deploy.
---

# Post-Deploy Verification

## Workflow

### Step 1: Check Railway status
```bash
railway status
```

### Step 2: Watch logs for startup errors (first 50 lines)
```bash
railway logs | head -50
```

### Step 3: Run health check
```bash
bash .gemini/skills/health-check/scripts/check-health.sh
```

### Step 4: Check critical endpoints
```bash
curl -s -o /dev/null -w "%{http_code}" https://fantasy-app-production-5079.up.railway.app/health
curl -s -o /dev/null -w "%{http_code}" https://fantasy-app-production-5079.up.railway.app/admin/scheduler/status
```

## Success Criteria

- Railway shows "running"
- Logs show "Uvicorn running" without ImportError
- Health check returns OK
- Both endpoints return 200

## Failure Handling

- If restart loop → `railway logs --follow` and escalate to Claude Code
- If ImportError → likely missing dependency, escalate to Claude Code
- If 500 on health → check logs for Python traceback, escalate to Claude Code
