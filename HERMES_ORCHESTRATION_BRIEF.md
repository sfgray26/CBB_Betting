# Hermes Orchestration Brief — CBB Edge Fantasy Baseball Modules

## Your Role
You are the project lead and multi-agent orchestrator for the CBB Edge codebase.
Your job is to:
1. Read the attached UAT report
2. Decompose the findings into a clean, prioritised backlog of discrete tasks
3. Assign each task to the most appropriate specialist agent
4. Identify dependencies and sequencing so agents are unblocked
5. Output **ONLY** the backlog and sprint plan — do **NOT** begin implementation

---

## Input Files
Read these before decomposing:
- **UAT Report:** `C:\Users\sfgra\repos\Fixed\cbb-edge\UAT_REPORT_FANTASY_BASEBALL_WEEK_10_2026-05-31.md`
- **Codebase root:** `C:\Users\sfgra\repos\Fixed\cbb-edge`

---

## Available Agents
Assign tasks from this pool. Do not invent agents.
- **DataAgent** — data pipelines, API integrations, feed ingestion, caching, sync logic
- **BackendAgent** — server-side business logic, scoring algorithms, projection engines, aggregation math
- **FrontendAgent** — UI components, state management, loading/error states, UX flows
- **QAAgent** — regression tests, sanity monitors, data-integrity assertions, cross-check scripts
- **FullStackAgent** — tasks that span backend logic AND the UI component that consumes it (use sparingly)

## Agent Delegation (how you spawn them)
When you are ready to dispatch tasks (not in this turn), use the appropriate bridge script:
- **Claude Code** (deep reasoning, multi-file): `~/hermes-delegate-to-claude.sh "task description"`
- **Codex** (fast implementation, tests, UI): `~/hermes-delegate-to-codex.sh "task description"`
- **Gemini** (research, data investigation): `~/hermes-delegate-to-gemini.sh "task description"`
- **Z.AI** (visual analysis, NL): `~/hermes-delegate-to-zai.sh "task description"`

Map agent types to bridges:
- DataAgent → Gemini
- BackendAgent → Claude Code
- FrontendAgent → Codex
- QAAgent → Codex (test scripts) or Claude Code (complex assertions)
- FullStackAgent → Claude Code

---

## Output Format
For each task, output exactly this block:

```
### TASK-[N]: [Short title]
- **Module:** [page/feature]
- **Severity:** P0 / P1 / P2
- **Agent:** [from available list]
- **Depends on:** [TASK-N, or "none"]
- **Problem:** One sentence — what is broken or missing
- **Acceptance Criteria:** Bullet list of what "done" looks like, testable and verifiable
- **Notes for agent:** Hints, constraints, known file paths, formulas, or edge cases
```

---

## Sprint Plan Output
After the backlog, output:
1. **Phase list** — Foundation → Optimizer → Gatekeeping/UX → Waiver/Streaming → Roster/Budget/Preview → Data Integrity/QA
2. **Parallel vs. Sequential** — for each phase, which tasks run in parallel and which are blocked
3. **Gaps / Ambiguities** — any items in this brief that need owner clarification before agents begin

---

## Definition of Done (sprint level)
The sprint closes when:
- No P0 remains open
- All schedule-dependent modules show real game data (no "No Game" everywhere)
- Optimizer produces a lineup where the top-3 OPS hitter is not benched without injury/no-game cause
- Waiver Wire shows a recommended drop for every suggested add
- Weekly Preview shows a real opponent and internally consistent win probability
- QAAgent's cross-check script passes on a 15-player sample

---

## Guardrails / Out of Scope
- Do **NOT** mutate the user's actual Yahoo lineup or roster (no "Apply" clicks in production)
- Do **NOT** change the underlying z-score valuation framework unless the root cause demands it
- Do **NOT** add new data providers; work with existing stats/ownership/feeds
- Do **NOT** implement new features beyond what's in the UAT report
- If a task requires clarification from the owner, flag it as **BLOCKED** and move on

---

## Pre-numbered Task Inventory
Use these task IDs in your output. You may split, merge, or renumber if the decomposition demands it, but start from this inventory.

### Phase 1 — Foundation (schedule + data unification)
- **TASK-1:** Fix schedule/games feed — every roster player shows "No Game" (DataAgent) — **P0**
- **TASK-2:** Unify matchup data source across `/war-room` and `/war-room/roster` (DataAgent + BackendAgent) — **P0**
- **TASK-3:** Unify IP data source across Roster and Budget pages (DataAgent) — **P1**

### Phase 2 — Optimizer Scoring
- **TASK-4:** Replace `proxy_projection` flat 58.0 fallback with real projection or exclusion flag (BackendAgent) — **P0**
- **TASK-5:** Normalize hitter and pitcher scores to a common unit before cross-position ranking (BackendAgent) — **P0**
- **TASK-6:** Add optimizer regression test: top-3 OPS hitter never benched without injury/no-game reason (QAAgent) — **P0**

### Phase 3 — Gatekeeping & UX
- **TASK-7:** Gate optimizer on valid schedule; show "No game data for [date]" and disable button if empty (FullStackAgent) — **P0**
- **TASK-8:** Remove debug date-mismatch banner from optimizer UI (FrontendAgent) — **P1**
- **TASK-9:** Add "Apply All" confirmation modal with before/after diff and move-cap warning (FrontendAgent) — **P1**

### Phase 4 — Waiver & Streaming
- **TASK-10:** Compute and display recommended drop for each waiver add, plus net category delta (BackendAgent) — **P0**
- **TASK-11:** Fix "Loading recommendations…" spinner persistence after data arrives (FrontendAgent) — **P1**
- **TASK-12:** Add tag legend + validate contradictory tag combos (e.g., `COLD` + `BUY_LOW`) (FrontendAgent) — **P1**
- **TASK-13:** Rebuild Streaming Station ranking on next-N-days projected category contribution (FullStackAgent) — **P0**
- **TASK-14:** Wire "2-Start SPs Only" filter to probable-pitcher schedule data (FrontendAgent) — **P1**

### Phase 5 — Roster, Budget, Preview
- **TASK-15:** Fix team AVG aggregation: compute as ΣH / ΣAB, exclude 0-AB players and pitchers (BackendAgent) — **P1**
- **TASK-16:** Consolidate IP display: both pages show identical pending/syncing state (FrontendAgent) — **P1**
- **TASK-17:** Label and verify "SEASON ADDS 272" calculation and source (BackendAgent) — **P1**
- **TASK-18:** Fix Week 11 opponent lookup in Weekly Preview (DataAgent + BackendAgent) — **P0**
- **TASK-19:** Suppress win% when opponent is unknown; never default to 100% or 0% (FrontendAgent) — **P0**

### Phase 6 — Data Integrity & QA (run in parallel from sprint start)
- **TASK-20:** Verify player-ID join between stats provider and ownership provider (DataAgent) — **P0/P1**
- **TASK-21:** Add sanity monitor: flag any player with OPS > .820 and ownership < 10% as data-integrity alert (QAAgent) — **P1**
- **TASK-22:** Audit all components for debug strings / console.log / dev-mode renders in production (QAAgent) — **P1**

### Deferred / P2 (can parallelise once P0s are resolved)
- **TASK-23:** Replace raw score provenance tags (`player_scores`, `proxy_projection`) with confidence indicator (FrontendAgent) — **P2**
- **TASK-24:** Show hidden players on hover when "Hide Owned Players" is active (FrontendAgent) — **P2**
- **TASK-25:** Extend "⚠️ Small Sample" flag to hitters with low AB (FrontendAgent) — **P2**
- **TASK-26:** Add "realistically available" ownership filter to Waiver (FrontendAgent) — **P1**

---

## Mandatory Sequencing Rules
1. **Phase 1 must finish before Phase 2, 4, and 5** — schedule feed is a hard dependency for optimizer gating, streaming, and preview opponent lookup
2. **TASK-4 must finish before TASK-6** — the safety assertion is meaningless until the fallback is fixed
3. **TASK-2 must finish before TASK-15** — correct matchup data needed to verify roster aggregation logic
4. **TASK-20 should start immediately** and run in parallel with Phase 1 — it does not block other tasks but its findings may force re-prioritisation
5. **Phase 3 can overlap with Phase 2** once TASK-4 is complete (TASK-7 depends on TASK-4, TASK-8 and TASK-9 do not)

---

## What to Flag for Owner Clarification
Include any of these in your "Gaps / Ambiguities" section if still unclear after reading the UAT report and codebase:
- What is the exact schema / mapping table for the player-ID join between stats and ownership providers?
- Is "SEASON ADDS 272" a league cap, a count used, or a league-wide total? What is the expected formula?
- What is the correct formula for pitcher "score" in the optimizer today, so we can normalize it against hitters?
- Does the 8-move weekly acquisition cap apply to lineup moves, waiver adds, or both?
- Where is the schedule/games feed currently ingested? (file path, API endpoint, or cron job)
