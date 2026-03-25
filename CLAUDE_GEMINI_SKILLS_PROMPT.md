# Prompt for Claude Code: Gemini Skills & Custom Commands

Copy-paste the following into Claude Code:

---

## PROMPT START

Read HANDOFF.md section 16.5 (Future Enhancement: Gemini Skills & Custom Commands) and implement a Gemini skills system for the DevOps Lead role.

### Background
Gemini CLI is configured as our DevOps Lead for Railway operations. Currently it uses shell commands defined in GEMINI.md. Gemini CLI supports Agent Skills and Custom Commands that could provide a better UX.

### Current State
- `GEMINI.md` - Basic context with shell commands
- `.gemini/settings.json` - Tool restrictions (blocks write_file, apply_diff, git)
- `scripts/gemini_recovery.sh` - Session recovery script
- 4 pending Railway ops that Gemini needs to execute

### Requirements

Create a skills system in `.gemini/skills/` that provides:

1. **railway-logs** skill
   - Command: `/railway-logs [--follow] [--errors] [--last=1h]`
   - Wraps: `railway logs --follow` with filtering
   - Output: Colorized, filtered logs

2. **db-migrate** skill  
   - Command: `/db-migrate [--dry-run] [--migration=<name>]`
   - Wraps: `railway run python scripts/migrations/...`
   - Verifies: Migration applied successfully
   - Updates: HANDOFF.md migration status

3. **env-check** skill
   - Command: `/env-check [--critical-only]`
   - Checks: All required env vars per HANDOFF.md
   - Compares: Current vs expected values
   - Reports: Missing/incorrect variables

4. **health-check** skill
   - Command: `/health-check [--component=<name>]`
   - Checks: Railway status, DB connectivity, API endpoints
   - Reports: System health summary

### Constraints (Per AGENTS.md)
- Gemini CANNOT write code (enforced by settings.json)
- Skills should only wrap shell commands and read files
- No application logic changes
- Document everything in HANDOFF.md

### Deliverables
1. `.gemini/skills/railway-logs/skill.json` - Skill definition
2. `.gemini/skills/railway-logs/handler.sh` - Implementation
3. `.gemini/skills/db-migrate/skill.json` - Skill definition  
4. `.gemini/skills/db-migrate/handler.sh` - Implementation
5. `.gemini/skills/env-check/skill.json` - Skill definition
6. `.gemini/skills/env-check/handler.sh` - Implementation
7. `.gemini/skills/health-check/skill.json` - Skill definition
8. `.gemini/skills/health-check/handler.sh` - Implementation
9. Update HANDOFF.md §16.5 to document the new skills
10. Test that `/help` shows the new commands

### Reference
- Gemini CLI docs: https://geminicli.com/docs/features/agent-skills/
- Current config: GEMINI.md, .gemini/settings.json
- HANDOFF.md §16 for DevOps Lead context

Evaluate if this adds enough value over shell commands before implementing. If not, document why in HANDOFF.md §16.5.

## PROMPT END

---

## Usage

1. Start Claude Code in the project directory
2. Paste the prompt above
3. Claude will evaluate and either:
   - Implement the skills system, OR
   - Document why shell commands are sufficient

## Expected Outcome

Either:
- ✅ New `.gemini/skills/` directory with 4 working skills
- ✅ HANDOFF.md updated with skill documentation
- ✅ Gemini CLI can run `/railway-logs`, `/db-migrate`, etc.

OR:
- 📋 Documented decision in HANDOFF.md §16.5 that skills add insufficient value
- 📋 Recommendation to stick with current shell-based workflow
