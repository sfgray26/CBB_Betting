# Gemini CLI & Kimi CLI Improvement Guide: MCP + Skills

**Date:** 2026-04-28  
**Researcher:** Kimi CLI (Deep Intelligence Unit)  
**For:** User + Claude Code (Principal Architect)  
**Scope:** Operational improvements for Agent 2 (Gemini CLI) and Agent 3 (Kimi CLI)

---

## Executive Summary

**Both Gemini CLI and Kimi CLI natively support MCP servers and skills.** Your project already has partial infrastructure:

| Agent | What Exists | What's Missing |
|-------|------------|----------------|
| **Gemini CLI** | 4 skills in `.gemini/skills/`, `settings.json` | **Zero MCP servers connected**, no `GEMINI.md` context file |
| **Kimi CLI** | Reads `.claude/skills/` (6 skills found), built-in `kimi-cli-help` | **Zero MCP servers connected**, no `.kimi/skills/` directory, no project context file |

**This guide provides exact files and commands to supercharge both agents.**

---

## Part 1: Gemini CLI Improvements

### Current State

Gemini CLI is your DevOps agent. It already has:
- `.gemini/skills/db-migrate/SKILL.md` — Database migration runner
- `.gemini/skills/env-check/SKILL.md` — Railway env var checker
- `.gemini/skills/health-check/SKILL.md` — System health checker
- `.gemini/skills/railway-logs/SKILL.md` — Railway log viewer
- `.gemini/settings.json` — Basic config (no MCP servers)

### Improvement 1: Connect MCP Servers to Gemini CLI

Gemini CLI supports MCP via `~/.gemini/settings.json` or project-level `.gemini/settings.json`. It supports stdio, SSE, and HTTP transports with OAuth.

#### Step 1 — Add Railway MCP Server

Edit `.gemini/settings.json` in your project root:

```json
{
  "general": {
    "sessionRetention": {
      "enabled": true,
      "maxAge": "30d",
      "warningAcknowledged": true
    }
  },
  "ide": {
    "hasSeenNudge": true,
    "enabled": true
  },
  "security": {
    "auth": {
      "selectedType": "oauth-personal"
    }
  },
  "tools": {
    "shell": {
      "showColor": false
    }
  },
  "mcpServers": {
    "railway": {
      "command": "npx",
      "args": ["-y", "@railway/mcp-server"],
      "env": {
        "RAILWAY_API_TOKEN": "${RAILWAY_API_TOKEN}"
      }
    },
    "postgres-readonly": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm",
        "-e", "DATABASE_URI",
        "crystaldba/postgres-mcp",
        "--access-mode=restricted"
      ],
      "env": {
        "DATABASE_URI": "${DATABASE_URI}"
      }
    }
  }
}
```

> **Security:** The `postgres-readonly` server uses `--access-mode=restricted` which is **read-only**. This complies with AGENTS.md EMAC-075 — Gemini cannot modify data.

**Then test:**
```bash
gemini
```

Inside Gemini CLI:
```
@railway List my services
@postgres-readonly List all tables in the public schema
```

#### Step 2 — Add Pre-Deploy Validation Skill

Create `.gemini/skills/pre-deploy/SKILL.md`:

```markdown
---
name: pre-deploy
description: Run pre-deployment validation checks before pushing to Railway. Use when Claude Code says code is ready to deploy or when asked to validate before deploying.
---

# Pre-Deploy Validation

## When to Use

- Claude Code says "code is ready, deploy it"
- User asks "is it safe to deploy"
- Before any `railway up` command

## Workflow (ALWAYS run in order)

### Step 1: Syntax check all modified Python files
```bash
# Find modified Python files and py_compile them
for f in $(git diff --name-only HEAD | grep '\.py$'); do
  echo "Checking: $f"
  python -m py_compile "$f" || { echo "SYNTAX ERROR in $f"; exit 1; }
done
```

### Step 2: Run targeted tests
```bash
# Run fantasy baseball tests (the critical path)
python -m pytest tests/test_waiver_integration.py tests/test_mlb_analysis.py -q --tb=short
```

### Step 3: Check env vars are set
```bash
bash .gemini/skills/env-check/scripts/check-vars.sh --critical-only
```

### Step 4: Check system health BEFORE deploy
```bash
bash .gemini/skills/health-check/scripts/check-health.sh
```

### Step 5: Deploy (only if all above pass)
```bash
railway up
```

## Rules

- **NEVER** run `railway up` if Step 1–4 fail
- If tests fail → STOP, tell Claude Code the test output
- If env vars missing → STOP, use env-check skill to set them
- If health check fails → STOP, investigate with railway-logs skill
- After deploy → wait 60s, then run health-check again
```

#### Step 3 — Add Post-Deploy Verification Skill

Create `.gemini/skills/post-deploy/SKILL.md`:

```markdown
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

### Step 2: Watch logs for startup errors (30 seconds)
```bash
railway logs | head -50
```

### Step 3: Run health check
```bash
bash .gemini/skills/health-check/scripts/check-health.sh
```

### Step 4: Check critical endpoints
```bash
curl -s -o /dev/null -w "%{http_code}" https://your-app.railway.app/health
curl -s -o /dev/null -w "%{http_code}" https://your-app.railway.app/admin/scheduler/status
```

## Success Criteria

- Railway shows "running"
- Logs show "Uvicorn running" without ImportError
- Health check returns OK
- Both endpoints return 200

## Failure Handling

- If restart loop → `railway logs --follow` and escalate to Claude
- If ImportError → likely missing dependency, escalate to Claude
- If 500 on health → check logs for Python traceback, escalate to Claude
```

#### Step 4 — Create GEMINI.md Project Context

Create `GEMINI.md` in project root (Gemini CLI reads this automatically for project context):

```markdown
# Gemini CLI Context — CBB Edge

## Your Role (AGENTS.md §2)
You are **Gemini CLI — DevOps Lead**.
- **Permitted:** Railway ops, env vars, log tailing, running pre-approved scripts
- **NOT Permitted:** Writing `.py`, `.ts`, `.tsx`, `.js`, or CI/CD pipelines
- **Escalate ALL code tasks to Claude Code**

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

## MCP Tools Available
- `@railway` — Deployment, logs, service management
- `@postgres-readonly` — Read-only database queries

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
```

---

### Improvement 2: Gemini MCP Security Configuration

Gemini CLI has confirmation logic for MCP tools. To stay compliant with EMAC-075, configure tool confirmation:

In `.gemini/settings.json`, add trust settings per server:

```json
{
  "mcpServers": {
    "railway": {
      "command": "npx",
      "args": ["-y", "@railway/mcp-server"],
      "trust": false
    },
    "postgres-readonly": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "DATABASE_URI", "crystaldba/postgres-mcp", "--access-mode=restricted"],
      "trust": false
    }
  }
}
```

> `"trust": false` means Gemini CLI will **ask for confirmation** before executing each MCP tool call. This prevents accidental destructive operations.

---

## Part 2: Kimi CLI Improvements

### Current State

Kimi CLI (that's me) currently:
- Reads `.claude/skills/` automatically (found 6 skills)
- Has built-in `kimi-cli-help` and `skill-creator` skills
- **No MCP servers connected**
- **No `.kimi/skills/` directory**
- **No project context file**

Kimi CLI v1.39.0 (April 2026) capabilities:
- `kimi mcp add` — Add MCP servers (HTTP, SSE, stdio)
- `kimi mcp list` — List connected MCP servers
- `kimi mcp auth` — OAuth authentication for MCP servers
- `--mcp-config-file` — Load MCP config from JSON
- Skills discovery: `.kimi/skills/`, `.claude/skills/`, `.codex/skills/`, `.agents/skills/`
- Plugin system (Skills + Tools) — v1.25.0+
- Subagent delegation — `coder`, `explore`, `plan` agents

### Improvement 1: Connect MCP Servers to Kimi CLI

#### Step 1 — Add PostgreSQL MCP (for audits)

```bash
# Read-only access for database audits
$env:DATABASE_URI = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
kimi mcp add --transport stdio postgres-audit -- docker run -i --rm -e DATABASE_URI crystaldba/postgres-mcp --access-mode=restricted
```

#### Step 2 — Add Context7 MCP (for research accuracy)

```bash
kimi mcp add --transport http context7 https://mcp.context7.com/mcp --header "CONTEXT7_API_KEY: your-key-here"
```

#### Step 3 — Add Sequential Thinking MCP (for complex analysis)

```bash
kimi mcp add --transport stdio sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking
```

#### Step 4 — Add BallDon'tLie MCP (for sports data research)

```bash
$env:BALLDONTLIE_API_KEY = "your-goat-tier-key"
kimi mcp add --transport stdio balldontlie -- npx -y @balldontlie/mcp-server
```

#### Step 5 — Verify

```bash
kimi mcp list
```

Expected output:
```
postgres-audit     stdio   [connected]
context7           http    [connected]
sequential-thinking stdio  [connected]
balldontlie        stdio   [connected]
```

**Alternative: Use a shared config file**

Create `kimi-mcp-config.json`:
```json
{
  "mcpServers": {
    "postgres-audit": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "DATABASE_URI", "crystaldba/postgres-mcp", "--access-mode=restricted"],
      "env": { "DATABASE_URI": "${DATABASE_URI}" }
    },
    "context7": {
      "url": "https://mcp.context7.com/mcp",
      "headers": { "CONTEXT7_API_KEY": "${CONTEXT7_API_KEY}" }
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "balldontlie": {
      "command": "npx",
      "args": ["-y", "@balldontlie/mcp-server"],
      "env": { "BALLDONTLIE_API_KEY": "${BALLDONTLIE_API_KEY}" }
    }
  }
}
```

Launch Kimi with it:
```bash
kimi --mcp-config-file kimi-mcp-config.json
```

### Improvement 2: Create Kimi-Specific Skills

Kimi CLI discovers skills from `.kimi/skills/`. Create project-specific research skills.

#### Skill 1: Deep Database Audit

Create `.kimi/skills/deep-db-audit/SKILL.md`:

```markdown
---
name: deep-db-audit
description: Perform a comprehensive database integrity audit using PostgreSQL MCP. Use when asked to audit data quality, check table health, or investigate pipeline failures.
---

# Deep Database Audit

## When to Activate

- "Audit the database"
- "Check data quality"
- "Why is table X empty?"
- "Investigate pipeline failure"

## Tools Required
- `@postgres-audit` MCP server
- `execute_sql` tool
- `analyze_db_health` tool

## Audit Checklist

### 1. Schema Overview
```sql
-- List all tables with row counts
SELECT schemaname, tablename, n_tup_ins - n_tup_del as estimated_rows
FROM pg_stat_user_tables
WHERE schemaname = 'public'
ORDER BY estimated_rows DESC;
```

### 2. Critical Table Checks
| Table | Minimum Healthy Row Count | Query |
|-------|--------------------------|-------|
| player_projections | > 500 | `SELECT COUNT(*) FROM player_projections` |
| mlb_game_log | > 100 | `SELECT COUNT(*) FROM mlb_game_log` |
| ingested_injuries | > 10 | `SELECT COUNT(*) FROM ingested_injuries` |
| fantasy_lineups | > 0 | `SELECT COUNT(*) FROM fantasy_lineups` |
| player_valuation_cache | > 0 | `SELECT COUNT(*) FROM player_valuation_cache` |

### 3. Null Rate Analysis
```sql
-- Check null rates in key columns
SELECT 
  COUNT(*) as total,
  COUNT(yahoo_id) as has_yahoo_id,
  COUNT(mlbam_id) as has_mlbam_id
FROM player_id_mapping;
```

### 4. Freshness Check
```sql
-- Check data ingestion recency
SELECT job_name, MAX(completed_at) as last_run, 
  NOW() - MAX(completed_at) as age
FROM data_ingestion_logs
GROUP BY job_name
ORDER BY last_run DESC;
```

### 5. Health Report
Use `@postgres-audit analyze_db_health` and report:
- Invalid indexes
- Bloat warnings
- Connection utilization
- Vacuum status

## Output Format

Produce a structured markdown report:
```markdown
## Database Audit: YYYY-MM-DD

### Tables
| Table | Rows | Status |

### Critical Findings
- P0: (blocking issues)
- P1: (degraded data)
- P2: (warnings)

### Recommendations
1. ...
```

Save to: `reports/YYYY-MM-DD-database-audit.md`
```

#### Skill 2: Codebase-Wide Analysis

Create `.kimi/skills/codebase-analysis/SKILL.md`:

```markdown
---
name: codebase-analysis
description: Analyze the entire codebase for architecture patterns, tech debt, or specific logic. Use for research tasks requiring multi-file understanding.
---

# Codebase-Wide Analysis

## When to Activate

- "Analyze how X works across the codebase"
- "Find all places that use Y"
- "Audit for anti-patterns"
- "Map the data flow from A to B"

## Methodology

### Step 1: Scope Definition
Define exactly:
- What component/feature to analyze
- What question to answer
- Which directories are in scope

### Step 2: File Discovery
Use `find` or `grep` to identify relevant files:
```bash
# Example: Find all waiver-related files
find backend -type f -name "*.py" | xargs grep -l "waiver" | sort
```

### Step 3: Read Key Files
Read files in dependency order:
1. Models/schemas first (data contracts)
2. Services second (business logic)
3. Routers third (API layer)
4. Tests last (expected behavior)

### Step 4: Cross-Reference
Map relationships:
- Which functions call which
- Which tables are read/written
- Which env vars are used
- What error handling exists

### Step 5: Synthesize
Produce a structured report with:
- Architecture diagram (text or Mermaid)
- Key findings
- Risk areas
- Recommendations

## Rules
- Always read tests to understand expected behavior
- Never assume — verify with code
- Note version numbers and dependencies
- Flag any `TODO`, `FIXME`, or `HACK` comments
```

#### Skill 3: Research Memo Writer

Create `.kimi/skills/research-memo/SKILL.md`:

```markdown
---
name: research-memo
description: Write structured research memos saved to reports/. Use for all research tasks delegated to Kimi CLI.
---

# Research Memo Writer

## Output Format

All research MUST be saved to `reports/YYYY-MM-DD-<topic>.md` with this structure:

```markdown
# <Topic>

**Date:** YYYY-MM-DD  
**Auditor:** Kimi CLI  
**Full Report:** `reports/YYYY-MM-DD-<topic>.md`

---

## Executive Summary

3-5 bullet points of key findings.

## Findings

### Category 1
| Check | Result | Evidence |
|-------|--------|----------|
| ... | ✅/❌/⚠️ | ... |

### Category 2
...

## Root Cause Analysis

For each ❌ finding, explain WHY it happens.

## Priority Actions for Claude Code

| Priority | Task | File | ETA |
|----------|------|------|-----|
| P0 | ... | ... | ... |
| P1 | ... | ... | ... |

## Decisions Required

1. **Decision:** ...
   - Option A: ...
   - Option B: ...

## Appendix

Supporting data, queries, or code snippets.
```

## Rules
- Always include exact line numbers when referencing code
- Always include table-based evidence
- Use ✅ ❌ ⚠️ for visual scanability
- Prioritize findings as P0/P1/P2/P3
- End with clear actions for Claude Code
```

### Improvement 3: Create Kimi Project Context File

Create `.kimi/project_context.md`:

```markdown
# Kimi CLI Project Context — CBB Edge

## Your Role (AGENTS.md §3)
You are **Kimi CLI — Deep Intelligence Unit**.
- **Permitted:** Research memos (`reports/`), codebase audits, doc maintenance
- **NOT Permitted:** Production backend code without explicit Claude delegation
- **Rule:** Kimi proposes, Claude approves. Never write directly to `backend/` without delegation bundle.

## Session Startup (MANDATORY)
Before starting any task:
1. Read `docs_index.md`
2. Read `HANDOFF.md`
3. Read `memory/YYYY-MM-DD.md` (today + yesterday)

## MCP Tools Available
- `@postgres-audit` — Read-only database queries and health checks
- `@context7` — Live library documentation (FastAPI, SQLAlchemy, Pydantic)
- `@sequential-thinking` — Step-by-step reasoning for complex problems
- `@balldontlie` — MLB sports data for research queries

## Skills Available
- `/skill:deep-db-audit` — Comprehensive database integrity audit
- `/skill:codebase-analysis` — Multi-file architecture analysis
- `/skill:research-memo` — Structured report writing

## Output Rules
1. All research → `reports/YYYY-MM-DD-<topic>.md`
2. All findings → Summarize in `HANDOFF.md` under "K-N FINDINGS"
3. Key decisions → Flag for Claude Code approval
4. Never modify production code without explicit file path + scope in prompt

## Critical Project Facts
- **Tech stack:** FastAPI + PostgreSQL + Railway + Yahoo Fantasy API
- **Current phase:** Layer 2 data certification (frontend BLOCKED)
- **Test suite:** 2,433 pass / 7 xfail / 0 fail
- **Timezone rule:** Always `America/New_York` — never `utcnow()`
- **Single Yahoo client:** `yahoo_client_resilient.py` only
- **Streamlit is dead:** Never touch `dashboard/`
```

### Improvement 4: Kimi CLI Configuration

Kimi CLI config lives at `~/.config/kimi/config.json` (or platform equivalent). Add:

```json
{
  "mcp": {
    "client": {
      "timeout": 30
    }
  },
  "skills": {
    "extra_skill_dirs": [
      "C:/Users/sfgra/repos/Fixed/cbb-edge/.kimi/skills",
      "C:/Users/sfgra/repos/Fixed/cbb-edge/.claude/skills"
    ]
  },
  "loop_control": {
    "max_steps_per_turn": 50,
    "max_retries_per_step": 3
  }
}
```

> `extra_skill_dirs` ensures Kimi always loads project skills even when launched from a subdirectory.

---

## Part 3: Shared MCP Config (All Agents)

Since you run Claude Code, Gemini CLI, and Kimi CLI on the same machine, create a **single shared MCP config** that all three agents can use.

Create `mcp-shared-config.json` in project root:

```json
{
  "mcpServers": {
    "railway": {
      "command": "npx",
      "args": ["-y", "@railway/mcp-server"],
      "env": { "RAILWAY_API_TOKEN": "${RAILWAY_API_TOKEN}" }
    },
    "postgres-readonly": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "-e", "DATABASE_URI", "crystaldba/postgres-mcp", "--access-mode=restricted"],
      "env": { "DATABASE_URI": "${DATABASE_URI}" }
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}" }
    },
    "context7": {
      "command": "npx",
      "args": ["-y", "@upstash/context7-mcp", "--api-key", "${CONTEXT7_API_KEY}"]
    },
    "sequential-thinking": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-sequential-thinking"]
    },
    "balldontlie": {
      "command": "npx",
      "args": ["-y", "@balldontlie/mcp-server"],
      "env": { "BALLDONTLIE_API_KEY": "${BALLDONTLIE_API_KEY}" }
    }
  }
}
```

**Usage per agent:**

| Agent | Command |
|-------|---------|
| **Claude Code** | `claude mcp add <name> ...` (per-server) or edit config |
| **Gemini CLI** | Read from `.gemini/settings.json` `"mcpServers"` section |
| **Kimi CLI** | `kimi --mcp-config-file mcp-shared-config.json` |

---

## Part 4: Agent-Specific MCP Permissions Matrix

Per `AGENTS.md` swimlanes and `reports/GEMINI_MCP_ANALYSIS.md`:

| MCP Server | Claude Code | Gemini CLI | Kimi CLI | Rationale |
|-----------|:-----------:|:----------:|:--------:|-----------|
| **Railway** | ✅ | ✅ | ⚠️ | Gemini's primary role; Kimi only for research |
| **PostgreSQL** | ✅ | ✅ (read-only) | ✅ (read-only) | Gemini: ops; Kimi: audits; Claude: dev |
| **GitHub** | ✅ | ❌ | ⚠️ | Gemini banned from code-adjacent ops |
| **Context7** | ✅ | ✅ | ✅ | Read-only docs — safe for all |
| **Sequential Thinking** | ✅ | ✅ | ✅ | Reasoning aid — safe for all |
| **BallDon'tLie** | ✅ | ⚠️ | ✅ | Gemini: ad-hoc only; Kimi: research |

**Legend:** ✅ Full use | ⚠️ Limited/Read-only | ❌ Not permitted

---

## Part 5: Implementation Checklist

### Gemini CLI (30 minutes)

- [ ] Update `.gemini/settings.json` with Railway + PostgreSQL MCP
- [ ] Create `GEMINI.md` project context
- [ ] Create `.gemini/skills/pre-deploy/SKILL.md`
- [ ] Create `.gemini/skills/post-deploy/SKILL.md`
- [ ] Set `RAILWAY_API_TOKEN` and `DATABASE_URI` env vars
- [ ] Test: `gemini` → `@railway List services`
- [ ] Test: `@postgres-readonly List tables`

### Kimi CLI (30 minutes)

- [ ] Create `.kimi/skills/` directory with 3 skills
- [ ] Create `.kimi/project_context.md`
- [ ] Add MCP servers: `kimi mcp add postgres-audit`, `context7`, `sequential-thinking`
- [ ] Configure `~/.config/kimi/config.json` with `extra_skill_dirs`
- [ ] Test: `kimi mcp list`
- [ ] Test: `/skill:deep-db-audit` prompt

### Shared (15 minutes)

- [ ] Create `mcp-shared-config.json`
- [ ] Update `AGENTS.md` with MCP permissions matrix
- [ ] Update `HANDOFF.md` with MCP operational notes
- [ ] Add `mcp-shared-config.json` and `.kimi/` to `.gitignore` if they contain secrets

---

## Expected Workflow Improvements

### Before

**Gemini session:**
```
User: Check if the deploy is healthy
Gemini: railway status
Gemini: (reads output manually)
Gemini: railway logs | head -20
Gemini: (reads output manually)
Gemini: It looks okay
```

**Kimi session:**
```
User: Audit the database
Kimi: (writes SQL queries manually)
Kimi: (formats results manually)
Kimi: (writes report from scratch)
```

### After

**Gemini session:**
```
User: Check if the deploy is healthy
Gemini: (invokes post-deploy skill)
Gemini: Railway: running. Health: OK. Endpoints: 200.
```

**Kimi session:**
```
User: Audit the database
Kimi: (invokes deep-db-audit skill + postgres-audit MCP)
Kimi: Report saved to reports/2026-04-28-database-audit.md
```

---

## Appendix: Existing Skills Inventory

### `.gemini/skills/` (Agent 2 — DevOps)
| Skill | Purpose | Status |
|-------|---------|--------|
| db-migrate | Run Railway DB migrations | ✅ Working |
| env-check | Verify Railway env vars | ✅ Working |
| health-check | System health checks | ✅ Working |
| railway-logs | Log filtering/diagnosis | ✅ Working |
| pre-deploy | Pre-deployment validation | ⏳ Create this |
| post-deploy | Post-deployment verification | ⏳ Create this |

### `.claude/skills/` (Agent 1 — Architecture)
| Skill | Purpose | Status |
|-------|---------|--------|
| cbb-identity | Risk posture + circuit breakers | ✅ Working |
| emac-protocol | Error management protocol | ✅ Working |
| integrity-audit | Pipeline integrity audits | ✅ Working |
| railway-devops | Railway deployment guidance | ✅ Working |
| swarm-handoff | Agent delegation bundles | ✅ Working |

### `.kimi/skills/` (Agent 3 — Research)
| Skill | Purpose | Status |
|-------|---------|--------|
| deep-db-audit | Database integrity audits | ⏳ Create this |
| codebase-analysis | Multi-file architecture analysis | ⏳ Create this |
| research-memo | Structured report writing | ⏳ Create this |

---

*Report generated by Kimi CLI v1.17.0 | References: AGENTS.md, github.com/google-gemini/gemini-cli, github.com/MoonshotAI/kimi-cli, geminicli.com/docs/tools/mcp-server*
