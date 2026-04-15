# Delegation K-33 — Railway MCP Server & DevOps Tooling Rollout

> **To:** Gemini CLI (DevOps Lead) / Kimi CLI (Intelligence Unit) / Claude Code (Architect)  
> **From:** Kimi CLI  
> **Date:** April 15, 2026  
> **Authority:** AGENTS.md — Gemini owns Railway ops; Kimi owns research & tooling setup.

---

## Summary

Reviewed https://docs.railway.com/ai/mcp-server and implemented a two-track empowerment strategy for the DevOps lead:

1. **MCP Track** (Kimi CLI enabled immediately): Railway MCP server installed in `~/.kimi/mcp.json`
2. **Script Track** (Gemini CLI ready immediately): Pre-approved Python scripts in `scripts/devops/` for DB queries, health checks, and log filtering

---

## 1. Railway MCP Server

**Package:** `@railway/mcp-server` (official, v0.1.8)

**Installed for:** Kimi CLI via `~/.kimi/mcp.json`

**Tools exposed:**
- `check-railway-status` — CLI health check
- `list-projects` / `list-services`
- `deploy` / `deploy-template`
- `get-logs` — log retrieval
- `list-variables` / `set-variables`
- `create-environment` / `link-environment`
- `generate-domain`

**How to use (Kimi):**
> "List all Railway services for this project" — Kimi will invoke the MCP tool directly.

**Limitations:**
- Requires Railway CLI to already be authenticated (`railway.ps1` is installed at `C:\Users\sfgra\AppData\Roaming\npm\railway.ps1`)
- MCP is only connected to Kimi CLI at this time. Claude Code and Gemini CLI do not yet have MCP client configs in this environment.

**Future:** Once Claude Code / Gemini CLI support MCP, copy the `railway` block from `~/.kimi/mcp.json` into their respective MCP configs.

---

## 2. DevOps Scripts (Gemini-Approved)

Because Gemini CLI's permitted actions are shell-based (`railway run`, `railway logs`, etc.), three helper scripts were created to give Gemini "full db access" without needing MCP:

### `scripts/devops/db_query.py`
Run arbitrary SQL and get JSON.

```bash
railway run python scripts/devops/db_query.py "SELECT COUNT(*) FROM player_id_mapping"
```

### `scripts/devops/db_health.py`
Snapshot of row counts, freshness, and anomalies.

```bash
railway run python scripts/devops/db_health.py
```

### `scripts/devops/railway_logs_filter.py`
Tail Railway logs with grep filtering (cross-shell safe).

```bash
railway run python scripts/devops/railway_logs_filter.py --job player_id_mapping --lines 50
```

**Validation:** All scripts pass `py_compile`.

**Updated context:** `GEMINI.md` now includes these commands in the "New DevOps Scripts" section.

---

## 3. PostgreSQL MCP (Deferred)

A PostgreSQL MCP server (`mcp-postgres` v1.2.1) was evaluated but **not enabled** because the correct production `DATABASE_URL` could not be verified from the local shell:
- Local PostgreSQL (`127.0.0.1:5432`) rejected the password in `.env`
- The Railway proxy URL in `.env` appears stale/corrupted (`# WEIGHT_KENPOM=1.0postgresql://...`)

**Recommendation:** Once a verified `DATABASE_URL` is available, add the following to `~/.kimi/mcp.json`:

```json
"postgres": {
  "command": "npx",
  "args": ["-y", "mcp-postgres"],
  "env": {
    "DATABASE_URL": "<verified-production-url>"
  }
}
```

This would give Kimi native SQL query tools alongside the Railway tools.

---

## 4. Updated Files

| File | Change |
|------|--------|
| `~/.kimi/mcp.json` | Added `@railway/mcp-server` |
| `scripts/devops/db_query.py` | New — arbitrary SQL runner |
| `scripts/devops/db_health.py` | New — DB health snapshot |
| `scripts/devops/railway_logs_filter.py` | New — log tail + filter |
| `GEMINI.md` | Added pre-approved script commands |
| `HANDOFF.md` | Added `## RAILWAY MCP SERVER & DEVOPS TOOLING` section |

---

## Prompt to Give Gemini

> **New DevOps tools are live.** You now have three pre-approved scripts in `scripts/devops/` for direct DB access and log filtering. Update your operational playbook to prefer these over raw bash where possible:
>
> ```bash
> # DB query
> railway run python scripts/devops/db_query.py "SELECT COUNT(*) FROM player_id_mapping"
>
> # Health snapshot
> railway run python scripts/devops/db_health.py
>
> # Filtered logs
> railway run python scripts/devops/railway_logs_filter.py --job player_id_mapping --lines 50
> ```
>
> Continue executing your current mission (`memory/delegation_g-32_player_id_mapping_migration.md`). After completion, update `HANDOFF.md` with results.
