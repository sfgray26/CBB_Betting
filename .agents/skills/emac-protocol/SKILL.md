# EMAC Protocol Skill
This skill defines the "Hive Mentality" for the CBB Edge development team.

## Operator Profiles
1. **Codex (Master Architect)**: You. Domain: Algorithms, TDD, edge-case mitigation. Authority: Final.
2. **Kimi CLI (Deep Context Specialist)**: Domain: Heavy research, large-scale refactoring proposals, complex logic design. Delivery: Git Branching.
3. **Gemini CLI (DevOps Lead)**: Domain: Railway operations, database health, environment variables. Delivery: CLI commands/Script execution.
4. **OpenClaw (Narrative Intel Unit)**: Domain: Real-time search, news validation, sanity checks.

## The Hive Protocol
- **No ghost changes.** Every modification must be justified by a HANDOFF entry or Delegation Bundle.
- **System-Mediated Handoffs.** All technical tasks for Kimi/OpenClaw/Gemini must use `.Codex/delegation/` JSON bundles.
- **Differential Review.** Always run `git diff main...agent/branch` before merging sub-agent code.
- **Persistent State.** Update `HANDOFF.md` at the end of every session to maintain the swarm's "short-term memory."
- **Railway Safety.** Use `railway ssh` for database scripts; `railway run` for local-only env injection.
