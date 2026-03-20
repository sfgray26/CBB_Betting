# Agent Task Pool

Task locking mechanism inspired by Anthropic's C Compiler project (16 agents, 2,000 sessions).

## How It Works

1. Each `.md` file in this directory is a **task unit** — one agent, one file.
2. When an agent starts a task, it claims the file by adding `CLAIMED_BY: <agent> <timestamp>`.
3. If another agent sees `CLAIMED_BY`, it picks a different task.
4. When complete, agent moves file to `.agent_tasks/done/` and updates HANDOFF.md.

## Task Size Rule

Each task must be completable in 1–3 Claude sessions.
- Too big: "Rewrite entire frontend" → Split it.
- Just right: "Build /bracket page" or "Fix Discord morning brief job".

## Status Tags

- `STATUS: OPEN` — available to claim
- `STATUS: CLAIMED` — in progress (check CLAIMED_BY)
- `STATUS: BLOCKED` — waiting on dependency (see BLOCKED_BY)
- `STATUS: DONE` — move to done/ folder
- `STATUS: GUARDIAN` — locked until stated date (do NOT touch)
