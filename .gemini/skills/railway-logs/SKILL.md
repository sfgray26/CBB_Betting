---
name: railway-logs
description: View, filter, and diagnose Railway deployment logs. Use when the user asks to check logs, diagnose errors, monitor startup, or watch live output from the CBB Edge production service.
---

# Railway Log Viewer

## When to Use This Skill

Activate when the user says things like:
- "check the logs"
- "what's happening in production"
- "is there an error in Railway"
- "watch the startup"
- "did the migration apply"

## Commands

### Live follow (default — stop with Ctrl+C)
```bash
railway logs --follow
```

### Filter for errors and warnings only
```bash
bash .gemini/skills/railway-logs/scripts/filter-logs.sh --errors
```

### Filter for a specific keyword
```bash
bash .gemini/skills/railway-logs/scripts/filter-logs.sh --grep="<keyword>"
```

### Check startup sequence (first 100 lines)
```bash
railway logs | head -100
```

## What to Look For

**Healthy startup looks like:**
- `"Uvicorn running on"` — backend started
- `"Scheduler started"` — APScheduler active
- `"DailyIngestionOrchestrator"` — if ENABLE_INGESTION_ORCHESTRATOR=true
- `"MLB nightly analysis enabled"` — if ENABLE_MLB_ANALYSIS=true

**Problem indicators:**
- `"restart"` or `"crash"` — container restart loop
- `"INTEGRITY_SWEEP_ENABLED"` errors — if env var not set to false
- `"ImportError"` or `"ModuleNotFoundError"` — dependency issue
- `"DATABASE_URL"` errors — DB connection problem

## Escalation

- If you see Python errors → escalate to Claude Code
- If you see DB connection failures → check DATABASE_URL via env-check skill
- If container is restarting → check logs for root cause before touching env vars
