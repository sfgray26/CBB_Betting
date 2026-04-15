# CBB Edge — Task Tracker
*Updated: 2026-04-15 | Architect: Claude Code | Mission: Restore strict layer-by-layer execution with Layer 2 as the only active workstream*

> Canonical source: `HANDOFF.md`
> This file is the execution board. If this tracker and HANDOFF disagree, HANDOFF wins.

---

## Operating Rule

Only Layer 2 work is active.

No new work is authorized on:
- derived stats expansion
- lineup optimization
- waiver breadth
- simulation / Monte Carlo
- frontend / UX
- Yahoo automation beyond what is strictly required for Layer 2 validation

---

## Layer Status Board

| Layer | Name | Status | Rule |
|------|------|--------|------|
| 0 | Immutable Decision Contracts | HOLD | Change only if Layer 2 forces it |
| 1 | Pure Stateless Intelligence | HOLD | No new work until Layer 2 passes |
| 2 | Data and Adaptation | ACTIVE | Only authorized workstream |
| 3 | Derived Stats and Scoring | FROZEN | Blocked by Layer 2 hard gate |
| 4 | Decision Engines and Simulation | FROZEN | Blocked by Layer 2 hard gate |
| 5 | APIs and Service Presentation | LIMITED | Layer 2 truth only |
| 6 | Frontend and UX | FROZEN | No new work |

---

## Active Priority Queue

### 2A. Deployment Truth (ACTIVE)
**Spec:** `HANDOFF.md` Layer 2A | **Priority:** highest

| Task | Owner | Done? |
|------|-------|-------|
| Redeploy latest repo state to Railway | Gemini | [ ] |
| Confirm stale production behavior is gone | Gemini | [ ] |

**Review:** Production must match repo before any Layer 2 validation result can be trusted.

### 2B. Observability and Auditability (ACTIVE)
**Spec:** `HANDOFF.md` Layer 2B | **Priority:** highest

| Task | Owner | Done? |
|------|-------|-------|
| Verify `data_ingestion_logs` persists durable rows from real job runs | Gemini + Claude | [ ] |
| Verify latest ingestion rows contain timestamps, statuses, and job metadata | Gemini + Claude | [ ] |
| Verify health endpoints degrade correctly when critical tables are empty | Gemini + Claude | [ ] |

**Review:** Without durable audit rows and truthful health semantics, the data platform is operationally blind.

### 2C. Critical Raw Table Health (ACTIVE)
**Spec:** `HANDOFF.md` Layer 2C | **Priority:** highest

| Task | Owner | Done? |
|------|-------|-------|
| Verify `mlb_player_stats` freshness and continuity | Claude | [ ] |
| Verify `statcast_performances` freshness and continuity | Claude | [ ] |
| Verify `probable_pitchers` populates usable rows in production | Gemini + Claude | [ ] |
| If `probable_pitchers` is empty, classify failure as deploy gap, source outage, or fallback miss-rate | Gemini + Claude | [ ] |

**Review:** Raw table health must be explicit and evidenced, not inferred from downstream behavior.

### 2D. Canonical Context Persistence (PENDING AFTER 2A-2C)
**Spec:** `HANDOFF.md` Layer 2D | **Priority:** next after production truth is established

| Task | Owner | Done? |
|------|-------|-------|
| Design DB-backed environment snapshot layer | Claude | [ ] |
| Persist weather context canonically | Claude | [ ] |
| Persist park-factor context canonically | Claude | [ ] |
| Define downstream read path to consume persisted context | Claude | [ ] |

**Review:** Environment context counts as Layer 2 data, not Layer 4 optimization.

### 2E. Validation and Exit Report (PENDING AFTER 2A-2D)
**Spec:** `HANDOFF.md` Layer 2E | **Priority:** gate to unlock Layer 3+

| Task | Owner | Done? |
|------|-------|-------|
| Complete Layer 2 acceptance checklist | Claude | [ ] |
| Add short Layer 2 completion note to HANDOFF | Claude | [ ] |
| Explicitly authorize or deny Layer 3 resumption | Claude | [ ] |

**Review:** Layer 2 is only complete when it is explicitly signed off, not when it feels close.

---

## Frozen Backlog

These items are valid future work but remain blocked:

| Workstream | Status | Why Frozen |
|-----------|--------|------------|
| Derived stats expansion | BLOCKED | Layer 2 incomplete |
| Lineup optimizer improvements | BLOCKED | Layer 2 incomplete |
| Waiver breadth and decision attribution | BLOCKED | Layer 2 incomplete |
| Simulation / Monte Carlo enhancements | BLOCKED | Layer 2 incomplete |
| Provider rationalization | BLOCKED | Not critical until Layer 2 is healthy |
| Frontend / UX work | BLOCKED | Face comes after brain |

---

## Recent Completed Cleanup

| Item | Status |
|------|--------|
| Re-centered roadmap on strict layer order | [x] |
| Rewrote HANDOFF around Layer 2 hard gate | [x] |
| Removed mixed downstream priorities from active board | [x] |

---

## Note To Future Sessions

If you are about to work on anything above Layer 2, stop and re-read `HANDOFF.md` first.

Last Updated: 2026-04-15
