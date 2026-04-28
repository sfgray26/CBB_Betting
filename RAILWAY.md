# RAILWAY.md — Authoritative Railway Operations Guide

> **Maintainer:** DevOps Lead (Gemini CLI)
> **Status:** ACTIVE / MANDATORY
> **Last Updated:** April 27, 2026

This document serves as the single source of truth for all Railway-related operations. All swarm agents MUST follow these protocols to prevent execution failures and minimize token waste.

---

## ⚠️ THE GOLDEN RULE (SSH vs RUN)

The production `DATABASE_URL` points to an **internal Railway hostname** (e.g., `postgres-ygnv.railway.internal`). This hostname **DOES NOT RESOLVE** on local Windows machines.

| Command | Execution Environment | Use Case |
|---------|-----------------------|----------|
| `railway run <cmd>` | **LOCAL** (Windows) | Tailing logs, local scripts NOT touching the DB. |
| `railway ssh <cmd>` | **PRODUCTION** (Container) | **MANDATORY** for any script opening a DB connection (migrations, queries, health checks). |

### ❌ WRONG (Fails on local machine)
```bash
railway run python scripts/devops/db_health.py
# Error: OperationalError: could not translate host name
```

### ✅ CORRECT (Works inside Railway)
```bash
railway ssh python scripts/devops/db_health.py
```

---

## 1. Database Operations

Any operation involving `psycopg2`, `SQLAlchemy`, or raw SQL queries MUST run inside the Railway environment.

### Run arbitrary SQL
```bash
railway ssh python scripts/devops/db_query.py "SELECT COUNT(*) FROM player_id_mapping"
```

### Full DB health snapshot
```bash
railway ssh python scripts/devops/db_health.py
```

### Run a migration
```bash
railway ssh python scripts/migrations/apply_openclaw_migration.py
```

---

## 2. Log Monitoring & Filtering

Log tailing is safe for `railway run` because it uses the Railway API, not direct DB connections.

### Tail all logs (live)
```bash
railway logs --follow
```

### Filter logs by job name
Use the pre-approved filter script for focused debugging.
```bash
railway run python scripts/devops/railway_logs_filter.py --job player_id_mapping --lines 50
```

---

## 3. Environment Variables

### List current variables
```bash
railway variables
```

### Set a variable
```bash
railway variables set ENABLE_MLB_ANALYSIS=true
```

---

## 4. Advanced: Postgres MCP Pro

Industrial-strength database tools are available via a pre-configured CLI wrapper talking to the Railway public proxy.

```bash
# Analyze full DB health (duplicate indexes, vacuum status, cache hits)
venv_mcp\Scripts\python scripts/devops/postgres_mcp_cli.py analyze_db_health

# Explain a slow query
venv_mcp\Scripts\python scripts/devops/postgres_mcp_cli.py explain_query --query "SELECT * FROM player_id_mapping WHERE bdl_id = 208"
```

---

## 5. Common Troubleshooting

| Issue | Resolution |
|-------|------------|
| `could not translate host name` | You used `railway run` instead of `railway ssh`. Retry with `ssh`. |
| `Permission denied (publickey)` | Ensure your SSH key is added to the Railway dashboard. |
| `UniqueViolation` in logs | An ingestion job is attempting to insert duplicate keys. Run `scripts/devops/db_query.py` to inspect existing data. |
| `TypeError: offset-naive and offset-aware` | Datetime mismatch in logic. Ensure all timestamps use `ZoneInfo("UTC")`. |

---

## 6. Pre-Approved DevOps Scripts

- `scripts/devops/db_query.py`: Run arbitrary SQL.
- `scripts/devops/db_health.py`: Summary of row counts and data freshness.
- `scripts/devops/railway_logs_filter.py`: Filter logs by job name.
- `scripts/backfill_numeric_player_names.py`: Resolve orphan player IDs.
