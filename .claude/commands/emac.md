---
description: Initiates the next EMAC mission by reading HANDOFF.md and delegating to the appropriate sub-agent.
---

1. Read `HANDOFF.md`.
2. Identify the "DELEGATION BUNDLE" assigned to Claude Code (Master Architect).
3. If tasks are found, invoke the `cbb-architect` agent using the Task tool to execute them.
4. If the tasks involve running tests, invoke a Bash tool to execute `pytest tests/ -q`.
