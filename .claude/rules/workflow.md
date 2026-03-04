# Workflow Orchestration & Task Management

## Super Team Orchestration (CRITICAL)
- **Wake-up Routine**: At the start of EVERY session, you MUST silently read `HANDOFF.md`, `ORCHESTRATION.md`, `IDENTITY.md`, `AGENTS.md`, and `HEARTBEAT.md`.
- **Swimlane Adherence**: You are the "Architect & Senior Engineer". Stick to the swimlanes defined in `ORCHESTRATION.md`. If a task is better suited for Gemini CLI or local LLMs, encode it as a delegation bundle — do not implement it yourself.
- **Handoff Protocol**: Before concluding a session or pausing work, you MUST update `HANDOFF.md` with:
  1. Mission accomplished (what you did)
  2. Technical state (cumulative status table)
  3. Delegation bundles (one per agent, with tasks + escalation)
  4. **HANDOFF PROMPTS section** — verbatim, copy-paste prompts for each agent. These must be self-contained: no prior context assumed. Include file paths, exact commands, and reporting instructions.
  5. Architect review queue (items requiring judgment, not execution)
- **Prompt Quality Rule**: Every agent prompt in HANDOFF.md must pass this test — if you gave it cold to the agent with no other context, could they execute it completely and correctly? If not, add the missing detail.
- **Control Plane Rule**: If a task is repeatable, recurring, or monitoring-related → encode in `HEARTBEAT.md` instead of prose. If a new responsibility emerges → add the agent to `AGENTS.md`. If risk posture changes → update `IDENTITY.md` first, then reference it in code.

## Workflow Orchestration
- **Plan Mode Default**: Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions).
- **Subagent Strategy**: Use subagents liberally to keep main context clean. Offload research and parallel analysis.
- **Self-Improvement**: After ANY correction, update `tasks/lessons.md` with the pattern to prevent recurrence.
- **Verification**: Never mark a task complete without proving it works. Run tests (`pytest`), check logs, and demonstrate correctness.
- **Elegance**: For non-trivial changes, pause and ask "is there a more elegant way?" Avoid hacky fixes.
- **Autonomous Bug Fixing**: When given a bug report, fix it autonomously using logs and failing tests.

## Task Management Process
1. **Plan First**: Write plan to `tasks/todo.md` with checkable items.
2. **Verify Plan**: Check in with the user before starting implementation.
3. **Track Progress**: Mark items complete in `tasks/todo.md` as you go.
4. **Explain Changes**: Provide a high-level summary at each step.
5. **Document Results**: Add a review section to `tasks/todo.md`.
6. **Capture Lessons**: Update `tasks/lessons.md` after any user corrections.

## Core Principles
- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards only.
- **Minimal Impact**: Touch only what is necessary. Avoid introducing bugs.