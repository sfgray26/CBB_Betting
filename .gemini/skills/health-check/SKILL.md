---
name: health-check
description: Check the health of the CBB Edge production system on Railway. Use when asked to verify the system is running, check if Railway is healthy, confirm an endpoint works, or diagnose why something is down.
---

# System Health Check

## When to Use This Skill

Activate when:
- "is the system healthy"
- "check Railway status"
- "is the API up"
- "did the restart fix it"
- "verify production is working"

## Workflow

### Full health check (recommended)
```bash
bash .gemini/skills/health-check/scripts/check-health.sh
```

### Single component check
```bash
bash .gemini/skills/health-check/scripts/check-health.sh --component=railway
bash .gemini/skills/health-check/scripts/check-health.sh --component=api
bash .gemini/skills/health-check/scripts/check-health.sh --component=scheduler
```

## What's Being Checked

| Component | Check | Healthy When |
|-----------|-------|-------------|
| Railway status | `railway status` | Shows "running" or "active" |
| API root | `GET /health` | Returns 200 |
| Scheduler | `GET /admin/scheduler/status` | Jobs listed with next_run |
| DB connection | `GET /admin/db-status` | Returns 200 (if endpoint exists) |

## Interpreting Results

**System healthy:**
- Railway shows service as running
- API returns 200 on health endpoint
- Scheduler shows 8+ jobs scheduled

**System degraded:**
- Railway restart loop → check logs (railway-logs skill)
- API 500 → check logs for Python error → escalate to Claude Code
- Scheduler missing jobs → check env vars (env-check skill)

## After Health Check

- If healthy → report status in chat
- If degraded → run env-check, then railway-logs, then escalate
- Update HANDOFF.md §16.4 with findings
