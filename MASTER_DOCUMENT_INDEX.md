# CBB Edge — Master Documentation Index

> **Navigation Hub:** Find the right document for your role and task.
> 
> **Last Updated:** March 28, 2026

---

## 🚨 EMERGENCY / ACTIVE INCIDENT

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[CLAUDE_ARCHITECT_PROMPT_MARCH28.md](CLAUDE_ARCHITECT_PROMPT_MARCH28.md)** | P0 Production Crisis Response | ⚠️ **ACTIVE EMERGENCY** — Roster 500 errors, matchup failures |

**Status:** This document supersedes all others during the active incident.

---

## 👥 Agent Role Definitions (READ FIRST)

| Document | Purpose | Owner |
|----------|---------|-------|
| **[AGENTS.md](AGENTS.md)** | Your workspace guide — read every session | All Agents |
| **[ORCHESTRATION.md](ORCHESTRATION.md)** | Multi-agent coordination rules | All Agents |
| **[IDENTITY.md](IDENTITY.md)** | Risk posture, Kelly math, circuit breakers | All Agents |
| **[SOUL.md](SOUL.md)** | System identity and ethos | All Agents |

---

## 🔄 Handoff & Session Continuity

| Document | Purpose | Updated By |
|----------|---------|------------|
| **[HANDOFF.md](HANDOFF.md)** | ⚠️ **CRITICAL:** Current operational state | Each agent after work |
| **[HEARTBEAT.md](HEARTBEAT.md)** | Periodic check-in reminders | System |
| `memory/YYYY-MM-DD.md` | Daily raw logs | Each agent |
| [MEMORY.md](MEMORY.md) | Curated long-term memory (main sessions only) | Claude |

**Rule:** Always read HANDOFF.md first. Always update it before finishing.

---

## 🎯 Claude Code Prompts (Lead Architect)

### Active Work Coordination
| Document | Purpose | Status |
|----------|---------|--------|
| **[CLAUDE_TEAM_COORDINATION_PROMPT.md](CLAUDE_TEAM_COORDINATION_PROMPT.md)** | Assign P0 gaps to Kimi/Gemini | ✅ **Authoritative** |
| ~~[CLAUDE_COORDINATION_PROMPT.md](CLAUDE_COORDINATION_PROMPT.md)~~ | *Deprecated — see above* | ⚠️ Superseded |

### Return Briefings
| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[CLAUDE_ARCHITECT_PROMPT_MARCH28.md](CLAUDE_ARCHITECT_PROMPT_MARCH28.md)** | 🚨 Emergency return (P0 crisis) | **Use now until resolved** |
| [CLAUDE_RETURN_PROMPT.md](CLAUDE_RETURN_PROMPT.md) | Standard return briefing (March 27) | Use after crisis resolved |

### Specialized Work
| Document | Purpose |
|----------|---------|
| [CLAUDE_FANTASY_ROADMAP_PROMPT.md](CLAUDE_FANTASY_ROADMAP_PROMPT.md) | Implement elite fantasy roadmap |
| [CLAUDE_UAT_FIXES_PROMPT.md](CLAUDE_UAT_FIXES_PROMPT.md) | User acceptance testing fixes |
| [CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md](CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md) | OpenClaw automation |

---

## 🤖 Other Agent Prompts

| Document | Agent | Purpose |
|----------|-------|---------|
| [CLAUDE_GEMINI_SKILLS_PROMPT.md](CLAUDE_GEMINI_SKILLS_PROMPT.md) | Gemini CLI | Railway ops, env vars, monitoring |
| [CLAUDE_LOCAL_LLM_PROMPT.md](CLAUDE_LOCAL_LLM_PROMPT.md) | OpenClaw | Local LLM execution |

---

## 📊 Reference Documentation

| Document | Purpose |
|----------|---------|
| [QUICKREF.md](QUICKREF.md) | Quick command reference |
| [TOOLS.md](TOOLS.md) | Tool integrations and credentials |
| [USER.md](USER.md) | Who we're helping |
| [README.md](README.md) | Project overview |
| [INSTALL.md](INSTALL.md) | Setup instructions |

---

## 📈 Research & Analysis (Kimi CLI Output)

| Location | Purpose |
|----------|---------|
| `reports/` | Research memos, gap analyses, audit reports |
| `reports/validation/` | Validation reports by feature |
| `reports/yahoo-client-hotfix-march28.md` | Latest Kimi fixes |

---

## 🗂️ Archive (Historical)

| Location | Contents |
|----------|----------|
| `docs/archive/` | Deprecated documentation |
| `memory/` | Daily session logs |
| `.agent_tasks/done/` | Completed phase tasks |

---

## 🔍 Quick Find by Task

### I need to...

**...start a new session:**
1. Read [AGENTS.md](AGENTS.md)
2. Read [HANDOFF.md](HANDOFF.md)
3. Read [ORCHESTRATION.md](ORCHESTRATION.md)

**...fix the roster/matchup errors (March 28):**
- Use [CLAUDE_ARCHITECT_PROMPT_MARCH28.md](CLAUDE_ARCHITECT_PROMPT_MARCH28.md)

**...coordinate work between agents:**
- Use [CLAUDE_TEAM_COORDINATION_PROMPT.md](CLAUDE_TEAM_COORDINATION_PROMPT.md)

**...implement fantasy baseball features:**
- Use [CLAUDE_FANTASY_ROADMAP_PROMPT.md](CLAUDE_FANTASY_ROADMAP_PROMPT.md)

**...check Railway logs or env vars:**
- Assign to Gemini CLI per [ORCHESTRATION.md](ORCHESTRATION.md)

**...analyze the whole codebase:**
- Assign to Kimi CLI per [ORCHESTRATION.md](ORCHESTRATION.md)

**...understand the betting model:**
- Read [IDENTITY.md](IDENTITY.md) Section "Risk Posture"

---

## 📝 Documentation Maintenance

**When adding new documents:**
1. Add to this index
2. Mark protected files clearly
3. Deprecate old versions with header notices
4. Link to related documents

**Protected Files (DO NOT MODIFY without Architect approval):**
- AGENTS.md
- ORCHESTRATION.md
- IDENTITY.md
- HEARTBEAT.md
- HANDOFF.md (update, don't delete)

---

*This index maintained by Documentation Refactoring Agent.*
*Last refactored: March 28, 2026*
