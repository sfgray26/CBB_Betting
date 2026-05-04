# AGENTS.md — Swarm Role Registry & Operational Constraints

> Defined and maintained by: Claude Code (Master Architect)
> Authority: This file overrides all other role descriptions across the repo.
> Last consolidated: March 28, 2026
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
- Railway deployment, env vars, infrastructure → Gemini CLI
- Async execution loops, Discord notifications → OpenClaw
- Long-context research synthesis, performance attribution → Kimi CLI
- Frontend component builds (CSS, UI) → Kimi CLI (delegated only)

### Code Quality Gates
Before any file in `backend/` is marked complete:
1. `venv/Scripts/python -m py_compile <file>` must pass
2. Relevant `pytest tests/` subset must pass
3. No `datetime.utcnow()` — always `datetime.now(ZoneInfo("America/New_York"))`
4. No `status: False` or other bool-as-string leakage to Pydantic schemas

---

## AGENT 2: Gemini CLI — DevOps Lead

**Restriction level:** HARD — no Python or TypeScript code writes.
**Root cause of restriction (EMAC-075, Mar 20, 2026):** Duplicate FastAPI route creation, invalid dict key references, testing against production without deploying. Demoted from code dev permanently.

### Permitted
- `railway logs --follow` — monitoring and log tailing
- Railway dashboard env var changes
- Running pre-approved scripts: `railway run python scripts/<migration>.py`
- Web research / API doc lookup (single-doc, no code output)
- `.md` file documentation edits that do not affect runtime behavior
- Triggering Railway redeploys (no code changes required)

### NOT Permitted
- Editing any file in `backend/`, `frontend/`, `tests/`, `scripts/`
- Writing DB migration scripts (Claude writes; Gemini may run them)
- CI/CD pipeline changes
- Any file with a `.py`, `.ts`, `.tsx`, `.js` extension

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
```

---

## AGENT 3: Kimi CLI — Subordinate Engineer & Deep Intelligence Unit

**Model:** Moonshot AI kimi-cli v1.17.0
**Context window:** 1M tokens — entire codebase + season data simultaneously

### Swimlane
Long-context research, performance attribution, UI component builds (delegated only), and targeted refactors within explicitly bounded scope.

### Owns
- `reports/` directory — all output is structured memos saved here
- Delegated frontend component builds (CSS, React, UI — when explicitly tasked by Claude)
- Codebase-wide audits (reads all Python files simultaneously)
- Doc hierarchy maintenance (MASTER_DOCUMENT_INDEX.md, deprecation headers)

### Does NOT Own
- Production backend code — proposes; Claude approves and implements
- Real-time runtime tasks → OpenClaw
- Infrastructure → Gemini CLI
- Risk math, Kelly formula changes → Claude only
- Any file in `backend/` without an explicit Claude delegation bundle granting access

### Interaction Protocol
1. Receives task briefing in HANDOFF.md with explicit file paths and scope boundaries
2. Produces structured markdown report (saved to `reports/YYYY-MM-DD-task-name.md`)
3. Key findings summarized in HANDOFF.md under "K-N FINDINGS"
4. Claude reads findings and decides what code changes to implement
5. Kimi may only write directly to production code when the delegation bundle explicitly names the target file and grants write access

### Tiered Integrity Pattern (CBB)
```
1. OpenClaw (qwen2.5:3b): First pass on ALL BET candidates — fast, cheap
2. Kimi: Second opinion ONLY when:
   - Game is Elite 8 or later
   - Recommended size >= 1.5u
   - OpenClaw returned VOLATILE or CAUTION
3. Human review: If Kimi returns RED FLAG or ABORT
```

---

## AGENT 4: OpenClaw — Autonomous Execution Unit

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

> Last updated: 2026-04-28 after Kimi CLI infrastructure setup.

Model Context Protocol (MCP) servers extend agent capabilities. Each agent has a scoped allowlist.

| MCP Server | Claude Code | Gemini CLI | Kimi CLI | Rationale |
|-----------|:-----------:|:----------:|:--------:|-----------|
| **Railway** | ✅ | ✅ | ⚠️ | Gemini's primary swimlane; Kimi read-only research use |
| **PostgreSQL** | ✅ | ✅ (read-only) | ✅ (read-only) | `--access-mode=restricted` REQUIRED for Gemini/Kimi |
| **GitHub** | ✅ | ❌ | ⚠️ | Gemini banned from code-adjacent ops per EMAC-075 |
| **Context7** | ✅ | ✅ | ✅ | Read-only docs — safe for all |
| **Sequential Thinking** | ✅ | ✅ | ✅ | Reasoning aid — safe for all |
| **BallDon'tLie** | ✅ | ⚠️ (ad-hoc only) | ✅ | Gemini: no bulk ingestion; Kimi: research |

**Gemini CLI constraints:**
- All MCP tools run with `trust: false` (confirmation required per call)
- PostgreSQL MCP MUST use `--access-mode=restricted`
- Never use GitHub MCP (PRs/issues are code-adjacent)

**Kimi CLI constraints:**
- Read-only database access for audits only
- No production data modification via MCP
- BDL MCP for research queries, not pipeline ingestion

---

## Swarm Boundaries (Non-Negotiable)

1. **No ghost changes.** Every modification justified in HANDOFF.md. No silent edits.
2. **Kimi proposes, Claude approves.** Kimi research output → HANDOFF.md → Claude implements.
3. **Gemini does not write code.** Period. Not even "trivial" one-liners. Escalate to Claude.
4. **Tier your integrity.** OpenClaw first pass on every CBB game. Kimi second opinion only for Elite 8+, ≥1.5u, or VOLATILE.
5. **Handoffs are operational briefings.** Not task lists. Include ground truth, decisions, and verbatim agent prompts that work cold.
6. **Policy lives in IDENTITY.md.** No risk parameter magic numbers in code without cross-reference to IDENTITY.md.

---

## Every Session Startup (All Agents)

Before doing anything else, read in order:
1. **`docs_index.md`** — minified system reference and document map
2. **`HANDOFF.md`** — current operational state and next steps
3. **`memory/YYYY-MM-DD.md`** (today + yesterday) for recent context

**For deep dives, use the retriever:** `python scripts/doc_retriever.py <file>`  
Key on-demand docs: `ORCHESTRATION.md` (routing), `IDENTITY.md` (risk posture), `HEARTBEAT.md` (loops).

Do not ask permission. Do not skip files. Do not infer state from conversation history alone.
