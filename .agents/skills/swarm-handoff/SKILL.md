---
name: swarm-handoff
description: Orchestrates system-mediated agent handoffs via structured Delegation Bundles and Git branching.
---

# Swarm Handoff Skill

This skill automates the creation, management, and review of system-mediated handoff Delegation Bundles between agents in the CBB Edge swarm.

## When to Use
Activate this skill when:
- Delegating a technical task to a sub-agent (Kimi, OpenClaw, Gemini).
- Creating a new task contract in `.Codex/delegation/`.
- Reviewing a feature branch delivered by a sub-agent.
- Finalizing a handoff cycle in `HANDOFF.md`.

## Core Workflow

### 1. Task Delegation (Scaffolding)
When starting a new delegation, create a JSON bundle in `.Codex/delegation/YYYYMMDD-task-name.json` following this schema:

```json
{
  "task_id": "YYYYMMDD-unique-name",
  "assignee": "Kimi | Gemini | OpenClaw",
  "branch": "optional/feature-branch-name",
  "scope": ["list/of/files/allowed"],
  "goal": "Clear technical objective",
  "criteria": "Acceptance criteria (e.g., tests pass)",
  "status": "PENDING"
}
```

### 2. Branch Management
Ensure the assignee uses the correct naming convention for their branch:
- `kimi/feature-name`
- `openclaw/feature-name`
- `gemini/devops-task`

### 3. Differential Review
Once the assignee sets the status to `READY_FOR_REVIEW`, use the following git commands to inspect changes:

```bash
# Fetch latest branches
git fetch origin

# Review differences against main
git diff main...assignee/feature-branch

# Review specifically targeted files
git diff main...assignee/feature-branch -- path/to/file.py
```

### 4. Integration & Cleanup
After approval:
1. Merge the branch: `git merge assignee/feature-branch`.
2. Update `.Codex/delegation/task.json` status to `COMPLETED`.
3. Update `HANDOFF.md` to document the deployment.
4. Delete the feature branch: `git branch -d assignee/feature-branch`.

## Guidelines
- **Explicit Scope:** Never leave the `scope` array empty. List every file the agent is authorized to touch.
- **Contract Integrity:** The Delegation Bundle is the absolute source of truth for the task's boundaries.
- **Traceability:** Every commit in the feature branch must correspond to the `task_id`.
- **No Direct Merges:** Sub-agents (except Codex) are forbidden from merging their own branches to `main`.

## Examples

### Creating a Delegation for Kimi
"Delegate the MLB player stats hydration to Kimi. Authorized files: backend/services/mlb_hydration.py. Goal: Ensure all season stats are populated."
-> *Action:* Create `.Codex/delegation/20260427-mlb-hydration.json` and update `HANDOFF.md`.

### Reviewing an OpenClaw Search Task
"Review the latest OpenClaw branch for narrative checks."
-> *Action:* Run `git diff main...openclaw/narrative-audit` and inspect the markdown findings.
