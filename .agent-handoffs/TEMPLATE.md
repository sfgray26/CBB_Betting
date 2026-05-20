# Agent Task Handoff

## Task ID
{{task_id}} — from Hermes Kanban

## Agent
{{agent_name}} (claude | codex)

## Objective
{{goal}}

## Context
- **Project**: CBB Edge — Fantasy baseball (MLB)
- **Branch**: {{branch_name}}
- **Worktree**: {{worktree_path}}
- **Hermes session**: {{hermes_session_id}}

## Files to Modify
{{file_list}}

## Code Patterns (from Hermes skill)
{{patterns}}

## Tests Required
{{test_plan}}

## Acceptance Criteria
- [ ] All tests pass
- [ ] No regressions in existing tests
- [ ] Handoff summary written to .agent-handoffs/COMPLETED_{{task_id}}.md

## Hermes Context
{{relevant_hermes_memory}}

## Commands to Run

