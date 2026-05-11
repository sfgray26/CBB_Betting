---
name: railway-devops
description: Provides expert guidance and guardrails for Railway deployment and production database operations.
---

# Railway DevOps Skill

This skill enforces the strict operational rules for CBB Edge production environments on Railway, preventing unresolvable hostname errors and ensuring reliable log filtering.

## When to Use
Activate this skill when:
- Running Python scripts that connect to the production database.
- Tailing or filtering Railway logs.
- Verifying environment variables.
- Investigating `OperationalError: could not translate host name` failures.

## Core Workflow

### 1. Database Command Rule (CRITICAL)
Production `DATABASE_URL` uses internal Railway hostnames that do NOT resolve locally on Windows.

| Scenario | Command Pattern | Why |
|----------|-----------------|-----|
| **Local / No DB** | `railway run <cmd>` | Injects env vars only. Safe for logs/local logic. |
| **Production / DB Access** | `railway ssh <cmd>` | Runs inside the container where DNS resolves. |

### 2. Log Tailing & Filtering
Use the following patterns to monitor the production service:
```bash
# Tail all logs
railway logs --follow

# Filter logs for a specific job (e.g., player_id_mapping)
railway run python scripts/devops/railway_logs_filter.py --job player_id_mapping --lines 50
```

### 3. Environment Variable Audit
```bash
# List all variables to check for missing keys
railway variables

# Set a critical gate variable
railway variables set INTEGRITY_SWEEP_ENABLED=false
```

## Guidelines
- **No Local DB Connections:** Never attempt to connect to the internal production DB hostname directly from a local terminal. Always use `railway ssh`.
- **Fail-Safe Triage:** If a command fails with a "host name" error, retry with `railway ssh`.
- **Selective Logging:** Prefer `railway_logs_filter.py` over raw `railway logs` for specific job debugging.

## Examples
"Run the database health check script."
-> *Action:* `railway ssh python scripts/devops/db_health.py`

"Check the logs for the ingestion orchestrator."
-> *Action:* `railway run python scripts/devops/railway_logs_filter.py --job ingestion_orchestrator`
