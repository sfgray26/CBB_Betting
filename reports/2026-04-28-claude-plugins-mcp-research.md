# Claude Plugins & MCP Servers Research for CBB Edge

**Date:** 2026-04-28  
**Researcher:** Kimi CLI (Deep Intelligence Unit)  
**Audience:** Claude Code (Principal Architect) + User  
**Status:** Actionable — Ready for Implementation  

---

## Executive Summary

Your project already has **partial MCP infrastructure in place**:
- `venv_mcp/` — isolated Python environment with `mcp` SDK installed
- `scripts/devops/postgres_mcp_cli.py` — CLI wrapper for `crystaldba/postgres-mcp` (Docker-based)
- `.claude/settings.local.json` — contains `mcp__kimi-code__*` permissions but **zero enabled plugins**
- Prior research: `reports/2026-04-08-balldontlie-mlb-mcp-research.md` (BDL MCP), `reports/GEMINI_MCP_ANALYSIS.md` (Gemini scope analysis)

**This report identifies 7 Claude plugins/MCP servers that will materially improve your multi-agent workflow,** with exact installation steps scoped to your FastAPI + PostgreSQL + Railway + Yahoo Fantasy stack.

| Tier | Plugin/MCP | Primary Beneficiary | Effort | Impact |
|------|-----------|---------------------|--------|--------|
| **1** | Railway MCP Server | Gemini CLI (DevOps) | 15 min | 🔥 High |
| **1** | PostgreSQL MCP Pro | Claude Code + Kimi | 15 min | 🔥 High |
| **1** | GitHub MCP Server | Claude Code | 10 min | 🔥 High |
| **2** | Context7 MCP | Claude Code | 10 min | ✅ High |
| **2** | Sequential Thinking MCP | Claude Code | 5 min | ✅ Medium |
| **2** | BallDon'tLie MCP | Claude Code (AI features) | 10 min | ✅ Medium |
| **3** | Claude-Mem (Episodic Memory) | All agents | 20 min | ⚡ Future |

---

## What Are Claude Plugins vs MCP Servers?

| Term | What It Is | How You Use It |
|------|-----------|----------------|
| **Claude Plugin** | Installable package for Claude Code (official or community) | `/plugin install <name>` — adds slash commands, skills, hooks |
| **MCP Server** | Standardized tool server using Model Context Protocol | `claude mcp add <name> <command>` — gives Claude tools for external systems |
| **Claude Skill** | Markdown-based reusable prompt template | Drop into `.claude/skills/` — project-specific workflows |

**For your project, MCP servers are the higher-leverage investment** because they connect Claude to live systems (Railway, PostgreSQL, GitHub) rather than just adding prompt templates.

---

## TIER 1: Must-Have — Implement This Week

---

### 1. Railway MCP Server 🚂

**What it does:** Official Railway MCP server that exposes deployment, log, env var, and service management tools to Claude via natural language.

**Why your project needs it:**
- Gemini CLI's primary job is Railway DevOps (logs, deploys, env vars, migrations)
- Currently Gemini types raw `railway` CLI commands; MCP gives Claude *structured* access to the same operations
- Fits the `AGENTS.md` DevOps swimlane perfectly — read-only + operational tools, no code writes

**Capabilities exposed:**
| Tool | What It Does | AGENTS.md Compliant? |
|------|-------------|---------------------|
| `check-railway-status` | Verify CLI auth | ✅ |
| `list-projects` | Show Railway projects | ✅ |
| `list-services` | Show services in linked project | ✅ |
| `list-variables` | Read env vars (not write) | ✅ |
| `get-logs` | Tail build or deploy logs | ✅ |
| `deploy` | Trigger redeploy | ✅ |
| `generate-domain` | Get Railway domain URL | ✅ |

**Security note:** The official Railway MCP intentionally excludes destructive operations. It does NOT expose `set-variables` without safeguards.

#### Exact Setup Instructions

**Step 1 — Install the MCP server:**

```bash
# In Claude Code terminal
claude mcp add railway-mcp-server -- npx -y @railway/mcp-server
```

**Step 2 — Set your Railway token:**

```bash
# Windows PowerShell
$env:RAILWAY_API_TOKEN = "your-railway-token-here"

# Or add to your user environment variables permanently
# Claude Code will pick it up from the shell environment
```

> To get your token: `railway login` then find it in Railway dashboard → Account Settings → Tokens

**Step 3 — Verify in Claude Code:**

```
/mcp
```

You should see `railway-mcp-server` listed as connected.

**Step 4 — Usage examples:**

```
"Check the health of my Railway project"
"Show me the last 50 lines of deploy logs for the Fantasy-App service"
"List all environment variables in production"
"Trigger a redeploy of the current service"
```

#### Alternative: Railway Claude Code Plugin (Official)

Railway also publishes an official **Claude Code plugin** (not just MCP):

```bash
# In Claude Code
/plugin add railway
/plugin install use-railway
```

This installs:
- `use-railway` skill (route-first agent for deployments, troubleshooting, networking)
- Auto-approve hook for Railway CLI commands (no Y/N prompt spam)

**Recommendation:** Install **both** — the MCP server for structured tool calls, and the plugin for the `use-railway` skill reference.

---

### 2. PostgreSQL MCP Pro 🐘

**What it does:** Gives Claude natural-language access to your PostgreSQL database — schema inspection, SQL execution, performance analysis, and index recommendations.

**Why your project needs it:**
- You already built `scripts/devops/postgres_mcp_cli.py` as a manual CLI wrapper
- Kimi CLI runs database audits constantly; this lets Claude Code and Kimi both query the DB conversationally
- Enables live diagnostics: "Why is player_projections missing 353 names?" → Claude runs the query

**Capabilities exposed:**
| Tool | What It Does |
|------|-------------|
| `execute_sql` | Run read-only SQL queries |
| `explain_query` | Analyze query execution plans |
| `analyze_db_health` | Check invalid indexes, bloat, vacuum status |
| `get_top_queries` | Find slowest queries from pg_stat_statements |
| `analyze_query_indexes` | Recommend indexes for specific queries |
| `list_schemas` / `list_objects` | Schema exploration |

#### Exact Setup Instructions

**Option A: Docker (Recommended — matches your existing CLI script)**

Your existing `scripts/devops/postgres_mcp_cli.py` already uses this pattern. To add it as a persistent MCP server in Claude Code:

```bash
claude mcp add postgres-mcp -- docker run -i --rm -e DATABASE_URI crystaldba/postgres-mcp --access-mode=restricted
```

> **Important:** Set `DATABASE_URI` as an environment variable in your shell before starting Claude Code, or the MCP server will have no connection string.

**Windows PowerShell (permanent session):**
```powershell
$env:DATABASE_URI = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
claude mcp add postgres-mcp -- docker run -i --rm -e DATABASE_URI crystaldba/postgres-mcp --access-mode=restricted
```

**Option B: uvx (Faster, no Docker overhead)**

```bash
# Install uv if you don't have it
pip install uv

# Add MCP server via uvx
claude mcp add postgres-mcp -- uvx postgres-mcp --access-mode restricted
```

With `DATABASE_URI` exported in your environment.

**Option C: Official Anthropic Postgres MCP (Simplest)**

```bash
claude mcp add postgres -- npx -y @modelcontextprotocol/server-postgres "postgresql://user:pass@host:5432/dbname"
```

> ⚠️ **Security Warning:** The official Anthropic server supports **writes by default**. For production safety, prefer `crystaldba/postgres-mcp` with `--access-mode=restricted` (read-only).

**Usage examples:**

```
"List all tables in the public schema"
"How many rows are in player_projections where name is just a number?"
"Explain why SELECT * FROM mlb_game_log WHERE game_date > '2026-04-01' is slow"
"Analyze database health and flag any bloated indexes"
"Recommend indexes for queries against player_id_mapping.yahoo_id"
```

---

### 3. GitHub MCP Server 🐙

**What it does:** Official GitHub MCP server — issues, PRs, commits, branches, code search, file creation.

**Why your project needs it:**
- Your workflow involves Claude Code writing code → Gemini deploying → manual git operations
- GitHub MCP lets Claude create branches, draft PRs, review diffs, and check issues without leaving the terminal
- Enables the `PR Review Toolkit` plugin workflow (see Tier 3)

#### Exact Setup Instructions

**Step 1 — Create a GitHub Personal Access Token:**

1. Go to https://github.com/settings/tokens
2. Click **Generate new token (classic)**
3. Scopes needed: `repo`, `read:org`, `read:user`
4. Copy the token (starts with `ghp_`)

**Step 2 — Install MCP server:**

```bash
# Windows PowerShell
$env:GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_your_token_here"
claude mcp add github -- npx -y @modelcontextprotocol/server-github
```

**Step 3 — Usage:**

```
"List open issues in this repo tagged as bug"
"Create a new branch called fix/lineup-scores and commit my changes"
"Show me the diff for the last 3 commits"
"Draft a PR description for the current branch"
```

---

## TIER 2: High-Value — Implement After Tier 1

---

### 4. Context7 MCP 📚

**What it does:** Fetches **real-time, version-specific documentation** for any library and injects it into Claude's context. No more hallucinated APIs or outdated patterns.

**Why your project needs it:**
- Your stack uses fast-moving libraries: FastAPI, SQLAlchemy 2.0, Pydantic v2, Next.js
- Claude's training data has a cutoff; Context7 fetches the *current* docs
- Prevents bugs like deprecated `datetime.utcnow()` or old SQLAlchemy patterns

#### Exact Setup Instructions

```bash
# Get a free API key at https://context7.com/dashboard (optional, increases rate limits)
# Then install:
claude mcp add context7 -- npx -y @upstash/context7-mcp --api-key YOUR_API_KEY

# Or without API key (lower rate limits):
claude mcp add context7 -- npx -y @upstash/context7-mcp
```

**Auto-invoke without typing "use context7":**

Add to your existing `CLAUDE.md` (or create `.claude/rules/context7.md`):

```markdown
Always use Context7 MCP when I need library documentation, code generation, 
or setup steps for FastAPI, SQLAlchemy, Pydantic, Next.js, or React.
```

**Usage examples:**

```
"How do I use SQLAlchemy 2.0 async session with FastAPI? use context7"
"Show me the correct way to define Pydantic v2 models with computed fields"
"What's the current best practice for Next.js 15 app router middleware?"
```

---

### 5. Sequential Thinking MCP 🧠

**What it does:** A lightweight MCP server that forces Claude to think step-by-step through complex problems before acting. Reduces architectural mistakes.

**Why your project needs it:**
- Your codebase is complex (2,400+ tests, multi-module fantasy baseball + CBB betting)
- Claude sometimes jumps to implementation before fully understanding the data flow
- This MCP adds a structured reasoning layer for tasks like "Refactor the waiver edge detector"

#### Exact Setup Instructions

```bash
claude mcp add sequential-thinking -- npx -y @modelcontextprotocol/server-sequential-thinking
```

**Usage:**

```
"Think through the steps needed to fix the negative lineup_score bug before writing any code"
```

Claude will use the `sequentialthinking` tool to build a reasoning chain, then present the plan for approval.

---

### 6. BallDon'tLie MCP Server ⚾

**What it does:** Natural language interface to BallDon'tLie's MLB API (19 endpoints including games, stats, odds, injuries, lineups).

**Why your project needs it:**
- You already have GOAT tier ($39.99/mo, 600 req/min)
- Your K-31 research mapped 10 pain points BDL can solve (lineups, injuries, splits, rolling stats)
- MCP is ideal for **conversational AI features** (lineup assistant, daily briefing bot)
- **Not for high-frequency backend ingestion** — keep direct API for that

#### Exact Setup Instructions

```bash
# In Claude Code
$env:BALLDONTLIE_API_KEY = "your-goat-tier-key"
claude mcp add balldontlie -- npx -y @balldontlie/mcp-server
```

**Or use the hosted endpoint (no local install):**

```bash
claude mcp add --header "Authorization: your-goat-tier-key" --transport http balldontlie https://mcp.balldontlie.io/mcp
```

**Usage examples:**

```
"Get today's MLB games with betting odds"
"Show me the confirmed lineup for the Yankees game today"
"Compare Aaron Judge's stats over the last 14 days vs his season average"
"List all players on the IL with return dates this week"
```

**Architecture note:** Use this for **AI assistant features only** (chatbot, briefing generation). For your ingestion pipelines (`daily_ingestion.py`), continue using the direct `balldontlie` Python SDK for performance.

---

## TIER 3: Strategic — Consider for Q3 2026

---

### 7. Claude-Mem (Episodic Memory) 💾

**What it does:** Persistent memory across Claude Code sessions. Captures actions, compresses them, and injects relevant context into future sessions.

**Why your project needs it:**
- Your project has massive context (100k+ lines, 2,400 tests, 6 months of handoff history)
- Currently you rely on `HANDOFF.md` and `memory/` files for session continuity
- Claude-Mem automates this — it remembers your coding preferences, past decisions, and project patterns

#### Exact Setup Instructions

```bash
# Requires Node.js
npm install -g claude-mem

# Initialize memory store
claude-mem init

# Add to Claude Code MCP
claude mcp add claude-mem -- claude-mem server
```

**Configuration in `.claude/settings.local.json`:**

Add these permission entries (you already have similar patterns):
```json
"mcp__plugin_episodic-memory_episodic-memory__read",
"mcp__plugin_episodic-memory_episodic-memory__search"
```

**Caution:** This is still early-stage. Test in a non-critical session first.

---

### 8. Code Review Plugin (Official Anthropic) 🔍

**What it does:** Multi-agent PR review that runs 5 specialized agents in parallel (bug detection, type design, test review, error handling, simplification).

**Why consider it:**
- Your test suite is 2,400+ tests — this automates regression detection before merge
- The `AGENTS.md` compliance agent could be a custom addition

**Installation:**
```bash
/plugin install code-review@claude-plugins-official
```

**Usage:**
```
/review --pr 123 --agents=all
```

---

## Complete Configuration File

Here's what your **Claude Code MCP configuration** should look like after installing Tiers 1–2.

Save this to your Claude Code config (location varies by OS):
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json` or project-level `.mcp.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "railway": {
      "command": "npx",
      "args": ["-y", "@railway/mcp-server"],
      "env": {
        "RAILWAY_API_TOKEN": "${RAILWAY_API_TOKEN}"
      }
    },
    "postgres": {
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
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_PERSONAL_ACCESS_TOKEN}"
      }
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
      "env": {
        "BALLDONTLIE_API_KEY": "${BALLDONTLIE_API_KEY}"
      }
    }
  }
}
```

> **Note:** Environment variable substitution (`${VAR}`) works in Claude Code if the variables are exported in your shell before launching `claude`.

---

## Security & AGENTS.md Compliance

### Critical Rules

| Rule | Rationale |
|------|-----------|
| **Never commit API tokens** to the repo | Use env vars or `.env` (already in `.gitignore`) |
| **Use `--access-mode=restricted` for Postgres** | Prevents accidental DROP TABLE or data mutation |
| **Set `trust: false` for all Gemini-facing MCPs** | Gemini CLI must confirm each tool call per EMAC-075 |
| **Read-only by default** | Railway MCP already excludes destructive ops; verify others |
| **Audit log all MCP executions** | Add to `HANDOFF.md` when MCP tools are used for operational changes |

### Gemini CLI Scope (EMAC-075)

Per `AGENTS.md` and `reports/GEMINI_MCP_ANALYSIS.md`:

| MCP Server | Allowed for Gemini? | Notes |
|-----------|---------------------|-------|
| Railway MCP | ✅ Yes | Read logs, list vars, deploy — already in Gemini playbook |
| PostgreSQL MCP | ⚠️ Read-only only | `--access-mode=restricted` REQUIRED |
| GitHub MCP | ❌ No | PR creation is code-adjacent; defer to Claude |
| Context7 | ✅ Yes | Docs lookup is read-only |
| BDL MCP | ⚠️ Limited | Ad-hoc queries only; no bulk ingestion |

### CVE Awareness

Claude Code had two MCP-related CVEs in 2025–2026:
- **CVE-2025-59536:** Malicious `.claude/settings.json` hooks executed before trust dialog
- **CVE-2026-21852:** API key exfiltration via `ANTHROPIC_BASE_URL` manipulation

**Both are patched in Claude Code 2.0.65+.** Ensure your Claude Code is up to date:
```bash
claude --version
# Should be 2.0.65 or later
```

---

## Implementation Roadmap

### Week 1: Foundation (Tier 1)

| Day | Task | Owner | Verification |
|-----|------|-------|-------------|
| 1 | Install Railway MCP + Plugin | User/Gemini | `claude mcp list` shows "railway" |
| 1 | Install PostgreSQL MCP Pro | User | Run: "List all tables in public schema" |
| 2 | Install GitHub MCP | User | Run: "List open issues in this repo" |
| 2 | Update `.claude/settings.local.json` with MCP permissions | Claude | Review + test |
| 3 | Document in `AGENTS.md` which agents can use which MCPs | Claude | PR review |

### Week 2: Enhancement (Tier 2)

| Day | Task | Owner | Verification |
|-----|------|-------|-------------|
| 4 | Install Context7 MCP | User | Run: "FastAPI dependency injection docs" |
| 5 | Install Sequential Thinking | User | Run: "Think through lineup score fix" |
| 5 | Install BallDon'tLie MCP | User | Run: "Get today's MLB games" |
| 6 | Test multi-MCP workflows | Claude | Example: DB query → GitHub issue creation |
| 7 | Update `HANDOFF.md` with MCP operational notes | Kimi/Claude | Audit compliance |

### Week 3+: Optimization (Tier 3)

| Task | Owner | Trigger |
|------|-------|---------|
| Evaluate Claude-Mem for session persistence | User | After 10+ Claude Code sessions |
| Evaluate Code Review plugin | Claude | When PR volume increases |
| Build custom `backend/mcp_servers/` for project-specific tools | Claude | When generic MCPs don't cover domain needs |

---

## Cost Analysis

| Plugin/MCP | Cost | Notes |
|-----------|------|-------|
| Railway MCP | Free | Requires Railway account |
| Railway Plugin | Free | Included with Railway |
| PostgreSQL MCP Pro | Free | Open source (crystaldba) |
| GitHub MCP | Free | Requires GitHub token |
| Context7 | Free tier: 1,000 queries/mo | Pro: $10/mo for higher limits |
| Sequential Thinking | Free | Open source |
| BallDon'tLie MCP | Free (uses your existing GOAT tier quota) | No extra cost |
| Claude-Mem | Free | Open source |
| **Total** | **$0–$10/mo** | Assuming Context7 free tier |

---

## Expected Workflow Improvements

### Before MCP
```
User: "Why is the waiver endpoint returning 503?"
Gemini: (types) railway logs --follow
Gemini: (reads manually) "Looks like Yahoo API error"
Gemini: (types) railway variables | grep YAHOO
Gemini: (reports back to user)
User: (asks Claude to fix)
Claude: (fixes code)
Gemini: (types) railway up
```

### After MCP
```
User: "Why is the waiver endpoint returning 503?"
Claude: (calls Railway MCP) get_logs → finds Yahoo error
Claude: (calls Railway MCP) list_variables → confirms env var
Claude: (fixes code)
Claude: (calls Railway MCP) deploy → pushes fix
```

**Time saved:** 5–10 minutes per incident × 20 incidents/month = **~2 hours/month**

---

## Appendix: Existing Project MCP Assets

| Asset | Location | Status |
|-------|----------|--------|
| `postgres_mcp_cli.py` | `scripts/devops/postgres_mcp_cli.py` | ✅ Working — Docker-based CLI wrapper |
| `venv_mcp` | Project root | ✅ Has `mcp` SDK installed |
| BDL MCP research | `reports/2026-04-08-balldontlie-mlb-mcp-research.md` | ✅ Complete |
| Gemini MCP analysis | `reports/GEMINI_MCP_ANALYSIS.md` | ✅ Complete |
| K-33 delegation prompt | `.claude/prompts/CLAUDE_K33_MCP_DELEGATION_PROMPT.md` | ⏳ Pending execution |

---

## Next Steps

1. **Approve this report** — User confirms which Tiers to implement
2. **Execute Week 1** — Install Tier 1 MCPs (30 minutes total)
3. **Update AGENTS.md** — Claude documents MCP permissions per agent role
4. **Update HANDOFF.md** — Log MCP adoption in operational state
5. **Re-audit in 2 weeks** — Kimi validates MCP usage patterns and time savings

---

*Report generated by Kimi CLI v1.17.0 | References: AGENTS.md, HANDOFF.md, reports/GEMINI_MCP_ANALYSIS.md, docs.railway.com/ai/mcp-server*
