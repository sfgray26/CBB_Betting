# .claude/delegation/
This directory stores structured Delegation Bundles (JSON/YAML) for system-mediated handoffs between swarm agents.

## Schema
```json
{
  "task_id": "YYYYMMDD-unique-name",
  "assignee": "Kimi | Gemini | OpenClaw",
  "branch": "optional/feature-branch-name",
  "scope": ["list/of/files/allowed"],
  "goal": "Clear technical objective",
  "criteria": "Acceptance criteria (e.g., tests pass)",
  "status": "PENDING | IN_PROGRESS | READY_FOR_REVIEW | COMPLETED"
}
```
