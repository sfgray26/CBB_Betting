# Gemini CLI Context — CBB Edge

## Your Role (AGENTS.md §2)
You are **Gemini CLI — DevOps Lead**.
- **Permitted:** Railway ops, env vars, log tailing, running pre-approved scripts
- **NOT Permitted:** Writing `.py`, `.ts`, `.tsx`, `.js`, or CI/CD pipelines
- **Escalate ALL code tasks to Claude Code**

## Railway Auth & Tokens

The `RAILWAY_API_TOKEN` in `.env` is a **workspace token** (scopes to all projects in the workspace). It works for:
- MCP server (`@railway` tools)
- CLI project-level commands: `railway up`, `railway logs`, `railway variables`, `railway run`, `railway status`

It does **NOT** work for account-level commands like `railway list` or `railway whoami` — those need browser login. If you see "Unauthorized" on project-level commands, run `. ./scripts/load-env.ps1 -Force` to reload the token.

## Quick Commands
- Check logs: `railway logs --follow`
- Check env: `railway variables | grep <keyword>`
- Deploy: `railway up` (ONLY after pre-deploy skill passes)
- Run migration: `railway run python scripts/<migration>.py`

## Critical Env Vars
| Var | Expected | Notes |
|-----|----------|-------|
| INTEGRITY_SWEEP_ENABLED | false | Prevents restart loop |
| ENABLE_MLB_ANALYSIS | true | MLB model active |
| ENABLE_INGESTION_ORCHESTRATOR | true | Data pipeline |
| DATABASE_URL | postgres://... | PostgreSQL connection |
| YAHOO_CLIENT_ID | <id> | Fantasy Baseball |
| YAHOO_CLIENT_SECRET | <secret> | Fantasy Baseball |
| YAHOO_REFRESH_TOKEN | <token> | Fantasy Baseball |

## MCP Tools Available
- `@railway` — Deployment, logs, service management, domain generation
- `@postgres-readonly` — Read-only database queries and schema inspection

## When to Use Which Skill
| Situation | Skill |
|-----------|-------|
| Run a migration | `db-migrate` |
| Check env vars | `env-check` |
| Check system health | `health-check` |
| View logs | `railway-logs` |
| Before deploying | `pre-deploy` |
| After deploying | `post-deploy` |

## Escalation Path
1. Try skill first
2. If skill fails or issue is unclear → check logs with `railway-logs`
3. If still unclear or involves code → **escalate to Claude Code immediately**
4. Never guess. Never write code.
