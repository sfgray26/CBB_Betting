# Claude Code System Prompt Update — Documentation Optimization

> **Effective immediately:** Replace the static "read all docs at startup" behavior with the optimized retrieval workflow below.

---

## New Startup Sequence

At the beginning of **every session**, read in this exact order:

1. **`docs_index.md`** — This is the single source of truth. It contains the minified agent roles, critical rules, risk posture, current focus, and a document map.
2. **`HANDOFF.md`** — Current operational state and next steps.
3. **`memory/YYYY-MM-DD.md`** (today + yesterday) — Recent session context.

**Do NOT** automatically read `AGENTS.md`, `ORCHESTRATION.md`, `IDENTITY.md`, or `HEARTBEAT.md` in full at session start. Those are now on-demand documents.

---

## On-Demand Retrieval Rule

If a task requires deeper context from a specific domain, use the retrieval tool instead of loading the entire file into your context window:

```bash
python scripts/doc_retriever.py <relative-path>
```

**Common retrieval patterns:**
- Swimlane routing details → `python scripts/doc_retriever.py ORCHESTRATION.md`
- Full risk posture / circuit breakers → `python scripts/doc_retriever.py IDENTITY.md`
- Operational heartbeat loops → `python scripts/doc_retriever.py HEARTBEAT.md`
- Full agent role definitions → `python scripts/doc_retriever.py AGENTS.md`
- Historical HANDOFF context → `python scripts/doc_retriever.py HANDOFF_ARCHIVE.md`
- Recent audit findings → `python scripts/doc_retriever.py reports/2026-04-15-comprehensive-application-audit.md`

---

## Token Budget Rule

Only read a full document if the task at hand explicitly requires it. For example:
- If you're fixing a betting model bug, read `IDENTITY.md` (risk posture) on demand.
- If you're coordinating agents, read `ORCHESTRATION.md` on demand.
- If you're doing routine backend work, `docs_index.md` + `HANDOFF.md` is usually sufficient.

---

## What Changed & Why

We were burning ~70KB of tokens per session loading `AGENTS.md` + `HANDOFF.md` + `ORCHESTRATION.md` + `IDENTITY.md` + `HEARTBEAT.md` unconditionally. `HANDOFF.md` alone was 40KB. We have now:

- Compressed `HANDOFF.md` from ~40KB to ~7KB by archiving historical sections to `HANDOFF_ARCHIVE.md`
- Created `docs_index.md` (~5KB) as a lightweight master reference
- Built `scripts/doc_retriever.py` for dynamic on-demand fetching
- Updated `AGENTS.md` to point to `docs_index.md` first

**Follow this new workflow strictly to minimize token burn.**
