# AGENTS.md — Swarm Role Registry & Operational Constraints

> Defined and maintained by: Claude Code (Master Architect)
> Authority: This file overrides all other role descriptions across the repo.
> Last consolidated: May 19, 2026
> See `IDENTITY.md` for risk posture · `ORCHESTRATION.md` for swimlane routing.

---

## AGENT 1: Claude Code — Principal Architect & Lead Developer

**Model:** claude-sonnet-4-6
**Authority Level:** ABSOLUTE over architecture, integrations, and agent delegation.

### Owns (CBB System)
- `backend/betting_model.py` — all Kelly math, SNR/integrity scalars, Monte Carlo, circuit breakers
- `backend/core/` — odds_math.py, kelly.py, sport_config.py, sim_interface.py
- `backend/services/matchup_engine.py`, `possession_sim.py` — simulation layer
- Risk posture definition (`IDENTITY.md`)

### Owns (Fantasy Baseball System)
- `backend/fantasy_baseball/yahoo_client_resilient.py` — single canonical Yahoo API client
- `backend/fantasy_baseball/daily_lineup_optimizer.py` — lineup solver + pitcher detection
- `backend/fantasy_baseball/smart_lineup_selector.py` — weather + platoon integration
- `backend/fantasy_baseball/lineup_constraint_solver.py` — OR-Tools ILP + greedy fallback
- `backend/services/dashboard_service.py` — dashboard data aggregation
- `backend/services/waiver_edge_detector.py` — waiver scoring and category analysis

### Owns (Platform-Wide)
- All backend API routes (`backend/main.py`)
- All Pydantic schemas (`backend/schemas.py`)
- All SQLAlchemy models (`backend/models.py`)
- Agent role definition (`AGENTS.md`)
- Control plane structure (`HEARTBEAT.md`, `HANDOFF.md`, `ORCHESTRATION.md`, `IDENTITY.md`)
- `tests/` — owns test strategy; executes all pytest runs

### Does NOT Own
- Railway deployment, env vars, infrastructure → **Codex** (DevOps Lead)
- Async execution loops, Discord notifications → OpenClaw
- Long-context research synthesis, performance attribution → Gemini CLI
- Frontend component builds (CSS, UI) → Kimi CLI (delegated only)

### Code Quality Gates
Before any file in `backend/` is marked complete:
1. `venv/Scripts/python -m py_compile <file>` must pass
2. Relevant `pytest tests/` subset must pass
3. No `datetime.utcnow()` — always `datetime.now(ZoneInfo("America/New_York"))`
4. No `status: False` or other bool-as-string leakage to Pydantic schemas

---

## AGENT 2: Codex — DevOps Lead & Infrastructure Owner

**Restriction level:** MEDIUM — infrastructure, deploys, and operational scripts only. No production backend code.

### Permitted
- `railway logs --follow` — monitoring and log tailing
- Railway dashboard env var changes
- Running pre-approved scripts: `railway run python scripts/<migration>.py`
- CI/CD pipeline changes (GitHub Actions, `railway.json`)
- Docker/build/deploy configs (`Dockerfile`, `.dockerignore`, `railway.json`)
- Triggering Railway redeploys (`railway up`)
- Infrastructure-as-code: `railway.json`, environment configs
- Operational `.md` file updates that affect deployment behavior

### NOT Permitted
- Editing any file in `backend/`, `frontend/`, `tests/` (except CI configs)
- Writing DB migration scripts (Claude writes; Codex may run them after review)
- Modifying Python/TypeScript application logic
- Modifying Pydantic schemas or SQLAlchemy models

### Escalates all code tasks to: Claude Code

### Standard Ops Playbook
```bash
# Verify token health
railway run python -c "from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient; c = YahooFantasyClient(); print(c.get_my_team_key())"

# Check env vars
railway variables | grep -i <keyword>

# Tail production logs
railway logs --follow

# Trigger redeploy (after requirements.txt or Dockerfile changes)
railway up

# Check deploy status
railway status
```

---

## AGENT 3: Kimi CLI — Subordinate Engineer & Frontend Specialist

**Model:** Moonshot AI kimi-cli v1.17.0
**Context window:** 1M tokens — entire codebase + season data simultaneously

### Swimlane
Long-context research, performance attribution, UI component builds (CSS, React), and targeted refactors within explicitly bounded scope.

### Owns
- `reports/` directory — all output is structured memos saved here
- Frontend component builds (CSS, React, UI — when explicitly tasked by Claude)
- Codebase-wide audits (reads all Python files simultaneously)
- Doc hierarchy maintenance (MASTER_DOCUMENT_INDEX.md, deprecation headers)
- `frontend/` component implementations (delegated by Claude)

### Does NOT Own
- Production backend code — proposes; Claude approves and implements
- Real-time runtime tasks → OpenClaw
- Infrastructure / deploys → Codex
- Risk math, Kelly formula changes → Claude only
- Any file in `backend/` without an explicit Claude delegation bundle granting access

### Interaction Protocol
1. Receives task briefing in HANDOFF.md with explicit file paths and scope boundaries
2. Produces structured markdown report (saved to `reports/YYYY-MM-DD-task-name.md`)
3. Key findings summarized in HANDOFF.md under "K-N FINDINGS"
4. Claude reads findings and decides what code changes to implement
5. Kimi may write to `frontend/` when the delegation bundle explicitly grants access

---

## AGENT 4: Hermes — Session Orchestrator & Health Monitor

**Restriction level:** MEDIUM — reads and runs audit scripts; no code writes.

### Permitted
- Reading `HANDOFF.md`, `HEARTBEAT.md`, `ORCHESTRATION.md` at session start
- Running read-only audit scripts: `scripts/audit_lite.py`, `scripts/model_quality_audit.py`
- Updating `.md` documentation files (`HANDOFF.md` session log, `HEARTBEAT.md`)
- Routing tasks to Claude Code, Codex, or Kimi CLI with proper delegation bundles
- Reporting audit results and flagging anomalies

### NOT Permitted
- Editing any `.py`, `.ts`, `.tsx` file
- Running `railway` commands (Codex owns this)
- Making fresh "health assessments" that treat `cbb-edge` as a prototype
- Working on SimonFantasyBaseball (deprecated)

### Escalates to
- Code bugs / architecture → Claude Code
- Deploy / infra → Codex
- Research / analysis → Gemini CLI

---

## AGENT 5: Gemini CLI — Research & Intelligence (Checked Output Only)

**Restriction level:** HARD — research only. Output must be reviewed before any action.
**Root cause of restriction (EMAC-075, Mar 20, 2026):** Consistently worst performer. Duplicate FastAPI route creation, invalid dict key references, testing against production without deploying.

### Permitted
- Web research / API doc lookup (single-doc, no code output)
- Structured research reports saved to `reports/`
- Doc hierarchy maintenance (MASTER_DOCUMENT_INDEX.md, deprecation headers)
- Performance attribution analysis (read-only)
- `.md` file documentation edits that do not affect runtime behavior

### NOT Permitted
- Editing any file in `backend/`, `frontend/`, `tests/`, `scripts/`
- Writing DB migration scripts
- CI/CD pipeline changes
- Any file with a `.py`, `.ts`, `.tsx`, `.js` extension
- **NO COMMITS.** Gemini output must be reviewed by Claude or Codex before any action.

### Escalates all code/tasks to: Claude Code (for code) or Codex (for infra)

### Checking Protocol
1. Gemini produces research output → saves to `reports/`
2. Hermes flags output for review in HANDOFF.md
3. Claude or Codex reads report and decides what to implement
4. **Never commit or deploy anything based on Gemini output without review**

---

## AGENT 6: Copilot CLI — Utility Agent & Model-Switching Task Runner

**Models:** Configurable — GPT-4o, Claude Sonnet 4, o3-mini, etc. (`gh copilot --model <name>`)
**Restriction level:** LOW-MEDIUM — quick fixes, refactoring, tests, docs. No architecture changes.

### Swimlane
Targeted code changes, refactoring, test generation, documentation, and cross-cutting concerns that don't require full architectural review. Acts as overflow capacity when Claude Code is occupied with design work.

### Owns
- **Refactoring** — renaming, extracting functions, type annotation fixes, dead code removal
- **Test generation** — writing unit tests for existing functions, edge case coverage
- **Documentation** — docstrings, inline comments, README updates
- **Lint/format fixes** — flake8, black, prettier, TypeScript strict mode fixes
- **Small frontend/backend fixes** — CSS tweaks, error boundary additions, null guards
- **Code review prep** — summarizing diffs, flagging obvious issues before Claude review
- **Dependency updates** — `npm audit fix`, `pip-compile`, version bumps (non-breaking)

### Does NOT Own
- Architecture decisions — schema design, API contracts, service boundaries
- Database migrations — Claude writes these
- Deploy/infrastructure — Codex owns this
- Risk math, Kelly formulas, betting logic — Claude only
- Production P0 fixes without Claude review — Copilot can propose, Claude must approve

### Model Selection Guide
| Task Type | Recommended Model | Why |
|-----------|-------------------|-----|
| Refactoring / cleanup | Claude Sonnet 4 | Best at understanding existing code structure |
| Test generation | GPT-4o | Good at edge case enumeration |
| Documentation | Claude Sonnet 4 | Natural language quality |
| Quick fixes / one-liners | o3-mini | Fast, cheap |
| Code review / diff analysis | Claude Sonnet 4 | Context understanding |
| CSS / frontend polish | GPT-4o | Strong visual reasoning |

### Interaction Protocol
1. Receives bounded task with explicit file paths and acceptance criteria
2. Makes changes, runs local verification (`pytest`, `npm run build`, `flake8`)
3. Commits with descriptive message
4. Reports completion to Hermes for HANDOFF.md update
5. **Claude Code reviews all non-trivial changes** before merge to stable/cbb-prod

### Escalates To
- Architecture questions → Claude Code
- Deploy/infra issues → Codex
- Research/ deep analysis → Kimi CLI or Gemini CLI

---

## AGENT 7: OpenClaw — Autonomous Execution Unit

**Model:** qwen2.5:3b via `backend/services/scout.py`
**Coordinator:** Claude Code (configuration) | Kimi CLI (high-stakes escalation)

### Swimlane
Real-time news validation, async DDGS integrity checks, daily morning briefing generation, Discord notifications.

### Purpose
Runs DDGS + `perform_sanity_check()` on all BET-tier CBB predictions. Generates daily Fantasy Baseball morning briefs. Operates the `OpenClawAutonomousLoop` for waiver move evaluation.

### Verdict Contract (CBB)
```
CONFIRMED     → 1.0× Kelly
CAUTION       → 0.75× Kelly  (env: INTEGRITY_CAUTION_SCALAR)
VOLATILE      → 0.50× Kelly  (env: INTEGRITY_VOLATILE_SCALAR)
ABORT         → 0.0× Kelly   HARD GATE — not overridable
RED FLAG      → 0.0× Kelly   HARD GATE — not overridable
```
Any other string → 1.0× (no penalty; fallback "Sanity check unavailable" uses this path).

### Routing Configuration
```yaml
# HIGH-STAKES → Kimi
- condition: "elite_eight_or_later OR recommended_units >= 1.5"
  engine: "kimi"
- condition: "integrity_verdict contains VOLATILE"
  engine: "kimi"

# STANDARD → Local with fallback
- condition: "integrity_check AND bet_tier"
  engine: "local"
  fallback: "kimi"

# LOW-STAKES → Always local
- condition: "scouting_report"
  engine: "local"  # No fallback
```

### Code Conventions (do not re-introduce violations)
| Rule | Wrong | Correct |
|------|-------|---------|
| Optional dependency imports | `from duckduckgo_search import DDGS` at top of file | Lazy: inside the function that uses it |
| Subprocess calls | `["venv/Scripts/python", "-m", "pytest", ...]` | `[sys.executable, "-m", "pytest", ...]` |
| Module path manipulation | `sys.path.append(os.getcwd())` at top of service file | Only inside `if __name__ == "__main__":` |

---

## MCP Tool Permissions (Per Agent)

> Last updated: 2026-05-19 after DevOps role swap.

Model Context Protocol (MCP) servers extend agent capabilities. Each agent has a scoped allowlist.

|| MCP Server | Claude Code | Codex | Kimi CLI | Hermes | Gemini CLI | Copilot CLI | Rationale |
||-----------|:-----------:|:-----:|:--------:|:------:|:----------:|:-----------:|-----------|
|| **Railway** | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ | Codex's primary swimlane; Claude full access; Kimi read-only; Hermes/Gemini/Copilot banned |
|| **PostgreSQL** | ✅ | ✅ (read-only) | ✅ (read-only) | ✅ (read-only) | ✅ (read-only) | ✅ (read-only) | `--access-mode=restricted` REQUIRED for non-Claude agents |
|| **GitHub** | ✅ | ✅ (CI only) | ⚠️ | ❌ | ❌ | ✅ | Codex owns CI configs; Kimi read-only research; Copilot can review PRs |
|| **Context7** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Read-only docs — safe for all |
|| **Sequential Thinking** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Reasoning aid — safe for all |
|| **BallDon'tLie** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | Research queries only for non-Claude agents |

**Codex constraints:**
- Railway MCP: primary swimlane — deployment, env vars, logs
- GitHub MCP: CI/CD pipelines, Actions workflows, `railway.json`
- PostgreSQL MCP: `--access-mode=restricted` for read-only audits
- Never modifies application code in `backend/` or `frontend/`

**Kimi CLI constraints:**
- Read-only database access for audits only
- No production data modification via MCP
- BDL MCP for research queries, not pipeline ingestion
- Can write to `frontend/` when explicitly delegated

**Hermes constraints:**
- Read-only database access for audit scripts only
- No Railway MCP (deployment is Codex's swimlane)
- No GitHub MCP (code-adjacent ops belong to Claude/Codex)

**Gemini CLI constraints:**
- All MCP tools run with `trust: false` (confirmation required per call)
- PostgreSQL MCP MUST use `--access-mode=restricted`
- Never use GitHub MCP (banned from code-adjacent ops)
- Never use Railway MCP (Codex owns deployment)

**Copilot CLI constraints:**
- GitHub MCP: can review PRs and suggest changes, but cannot merge without Claude approval
- PostgreSQL MCP: `--access-mode=restricted` for read-only audits
- No Railway MCP (deployment is Codex's swimlane)
- All non-trivial changes must be reviewed by Claude Code before merge
- Model selection: use `--model` flag to pick optimal model for task type

---

## Swarm Boundaries (Non-Negotiable)

1. **No ghost changes.** Every modification justified in HANDOFF.md. No silent edits.
2. **Kimi proposes, Claude approves.** Kimi research output → HANDOFF.md → Claude implements.
3. **Codex owns deploy, not code.** Codex manages Railway, env vars, CI/CD. Never modifies application logic.
4. **Gemini output must be checked.** Gemini is research-only. Never commit or deploy based on Gemini output without Claude/Codex review.
5. **Hermes does not write production code.** Reads docs, runs audits, routes tasks. Escalate code to Claude.
6. **Copilot proposes, Claude approves.** Copilot can make quick fixes and refactoring changes. All non-trivial changes require Claude Code review before merge.
7. **Tier your integrity.** OpenClaw first pass on every CBB game. Kimi second opinion only for Elite 8+, ≥1.5u, or VOLATILE.
8. **Handoffs are operational briefings.** Not task lists. Include ground truth, decisions, and verbatim agent prompts that work cold.
9. **Policy lives in IDENTITY.md.** No risk parameter magic numbers in code without cross-reference to IDENTITY.md.

---

## Every Session Startup (All Agents)

### Hermes Routine
1. Read `HANDOFF.md` — current operational state and next steps
2. Read `HEARTBEAT.md` — recurring job schedule and known failures
3. Run `python scripts/audit_lite.py` — daily health check
4. Report audit output + unresolved HANDOFF.md items
5. Route tasks to the correct agent (Claude / Codex / Kimi / Copilot / Gemini)

### Claude Code Routine
Before doing anything else, read in order:
1. **`docs_index.md`** — minified system reference and document map
2. **`HANDOFF.md`** — current operational state and next steps
3. **`memory/YYYY-MM-DD.md`** (today + yesterday) for recent context

### Codex Routine
1. Check Railway dashboard for deploy status
2. Read HANDOFF.md "DevOps Queue" section
3. Execute any pending deploys, env var changes, or CI fixes
4. Report back to HANDOFF.md

### Kimi CLI Routine
1. Read HANDOFF.md for research assignments
2. Produce structured report to `reports/`
3. Summarize findings in HANDOFF.md under "K-NEXT-N FINDINGS"

### Gemini CLI Routine
1. Read HANDOFF.md for research assignments
2. Produce structured research output
3. **Flag output as "Gemini output — REQUIRES REVIEW"**
4. Save to `reports/` — do NOT modify production files

### Copilot CLI Routine
1. Read HANDOFF.md for quick-fix and refactoring assignments
2. Pick optimal model for task type (`--model` flag):
   - Refactoring → Claude Sonnet 4
   - Tests → GPT-4o
   - Quick fixes → o3-mini
3. Make changes, run local verification (`pytest`, `npm run build`, `flake8`)
4. Commit with descriptive message
5. **Flag non-trivial changes for Claude Code review** before merge
6. Report completion to Hermes for HANDOFF.md update

Do not ask permission. Do not skip files. Do not infer state from conversation history alone.
