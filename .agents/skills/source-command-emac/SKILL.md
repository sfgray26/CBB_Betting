---
name: "source-command-emac"
description: "Initiates the next EMAC mission by reading HANDOFF.md and delegating to the appropriate sub-agent."
---

# source-command-emac

Use this skill when the user asks to run the migrated source command `emac`.

## Command Template

1. Read `HANDOFF.md`.
2. Identify the "DELEGATION BUNDLE" assigned to Codex (Master Architect).
3. If tasks are found, invoke the `cbb-architect` agent using the Task tool to execute them.
4. If the tasks involve running tests, invoke a Bash tool to execute `pytest tests/ -q`.
