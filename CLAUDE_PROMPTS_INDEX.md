# Claude Prompts Index

> **Master index for all Claude Code prompts and return briefings.**  
> **Last Updated:** April 11, 2026  
> **Navigation:** Use this to find the right prompt for your task.

---

## 🚨 Emergency / Crisis Response

| Prompt | Purpose | Status |
|--------|---------|--------|
| **[CLAUDE_ARCHITECT_PROMPT_MARCH28.md](CLAUDE_ARCHITECT_PROMPT_MARCH28.md)** | Production crisis: Roster 500 errors, matchup failures | ⚠️ Active if crisis ongoing |

**Rule:** If there's an active production incident, start here. Otherwise use standard prompts below.

---

## 🔄 Standard Return Briefings

| Prompt | Purpose | When to Use |
|--------|---------|-------------|
| **[CLAUDE_RETURN_PROMPT.md](CLAUDE_RETURN_PROMPT.md)** | Full context + immediate directives | Standard return after absence |
| **[CLAUDE.md](CLAUDE.md)** | Quick project orientation | Every session start |

**Rule:** Always read `CLAUDE.md` + `HANDOFF.md` first. Use `CLAUDE_RETURN_PROMPT.md` if you need full context after being away.

---

## 👥 Team Coordination

| Prompt | Purpose | Authority |
|--------|---------|-----------|
| **[CLAUDE_TEAM_COORDINATION_PROMPT.md](CLAUDE_TEAM_COORDINATION_PROMPT.md)** | Assign P0 gaps to Kimi/Gemini | ✅ **Authoritative** |
| ~~[CLAUDE_COORDINATION_PROMPT.md](.claude/prompts/archive/CLAUDE_COORDINATION_PROMPT.md)~~ | *Deprecated* | See above |

**Rule:** Use TEAM_COORDINATION for all multi-agent task assignments.

---

## 🎯 Specialized Work Prompts

### Development & Implementation
| Prompt | Purpose | Status |
|--------|---------|--------|
| **[CLAUDE_FANTASY_ROADMAP_PROMPT.md](CLAUDE_FANTASY_ROADMAP_PROMPT.md)** | Elite fantasy baseball roadmap implementation | Active |
| **[CLAUDE_UI_UX_ARCHITECT_PROMPT.md](CLAUDE_UI_UX_ARCHITECT_PROMPT.md)** | Frontend design system implementation | Active |
| **[CLAUDE_UAT_FIXES_PROMPT.md](CLAUDE_UAT_FIXES_PROMPT.md)** | User acceptance testing fixes | Active |
| ~~[CLAUDE_PROMPT_K26_MATCHUP_FIX.md](.claude/prompts/archive/CLAUDE_PROMPT_K26_MATCHUP_FIX.md)~~ | *Archived* | K-26 complete |

### Multi-Agent Delegation
| Prompt | Purpose | Target Agent |
|--------|---------|--------------|
| **[CLAUDE_K34_K38_KIMI_DELEGATION.md](CLAUDE_K34_K38_KIMI_DELEGATION.md)** | Data quality research bundle (K-34 to K-38) | Kimi CLI |
| **[CLAUDE_K33_MCP_DELEGATION_PROMPT.md](CLAUDE_K33_MCP_DELEGATION_PROMPT.md)** | MCP server implementation research | Kimi CLI |
| **[CLAUDE_KIMI_DELEGATION.md](CLAUDE_KIMI_DELEGATION.md)** | General Kimi task delegation | Kimi CLI |
| **[CLAUDE_GEMINI_SKILLS_PROMPT.md](CLAUDE_GEMINI_SKILLS_PROMPT.md)** | Railway operations, env vars, monitoring | Gemini CLI |
| **[CLAUDE_LOCAL_LLM_PROMPT.md](CLAUDE_LOCAL_LLM_PROMPT.md)** | OpenClaw local LLM execution | OpenClaw |
| **[CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md](CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md)** | OpenClaw automation workflows | OpenClaw |

---

## 📂 Archive Location

Historical prompts moved to `.claude/prompts/archive/`:

| Prompt | Archived | Reason |
|--------|----------|--------|
| [CLAUDE_COORDINATION_PROMPT.md](.claude/prompts/archive/CLAUDE_COORDINATION_PROMPT.md) | Apr 2026 | Superseded by TEAM_COORDINATION |
| [CLAUDE_PROMPT_K26_MATCHUP_FIX.md](.claude/prompts/archive/CLAUDE_PROMPT_K26_MATCHUP_FIX.md) | Apr 2026 | K-26 task completed |

---

## 🗂️ Related Documentation

| Document | Purpose |
|----------|---------|
| **[AGENTS.md](AGENTS.md)** | Agent role definitions and swimlanes |
| **[ORCHESTRATION.md](ORCHESTRATION.md)** | Multi-agent coordination rules |
| **[HANDOFF.md](HANDOFF.md)** | Current operational state |
| **[MASTER_DOCUMENT_INDEX.md](MASTER_DOCUMENT_INDEX.md)** | All documentation navigation |

---

## 📝 Maintenance

**To update this index:**
1. Add new prompts to appropriate section
2. Mark deprecated prompts with ~~strikethrough~~ and move to archive
3. Update "Last Updated" date
4. Ensure all root-level CLAUDE*.md files are listed here

**Naming convention:**
- Active: `CLAUDE_<PURPOSE>.md` in root
- Archived: Move to `.claude/prompts/archive/`

---

*Index created as part of documentation cleanup (April 2026). See DOCUMENTATION_CLEANUP_PLAN.md for details.*
