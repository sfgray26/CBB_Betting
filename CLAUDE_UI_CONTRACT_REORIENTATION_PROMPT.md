# Claude Code: UI Contract Reorientation & HANDOFF.md Update

> **Purpose:** Incorporate the UI Specification Contract Audit into project memory and update HANDOFF.md to reflect the new plan. Do NOT start coding. This is a planning and documentation session only.
>
> **Date:** April 17, 2026
> **Trigger:** A comprehensive UI specification has been locked as the authoritative contract. A field-level audit found the backend is 17% ready. HANDOFF.md must be updated to reflect this new reality without losing existing production health context.

---

## 1. READ FIRST (mandatory, in order)

1. `HANDOFF.md` — current operating brief (you wrote this on April 16)
2. `reports/2026-04-17-ui-specification-contract-audit.md` — the new audit (this is the ground truth for this session)
3. `AGENTS.md` — your role boundaries
4. `IDENTITY.md` — risk posture
5. `backend/utils/fantasy_stat_contract.json` — the 18 canonical categories

Do not proceed until all five files are read. Do not infer their contents from memory.

---

## 2. SITUATION ASSESSMENT

**What changed:** A complete, field-level UI specification was written as an authoritative contract. The rule is: *the UI spec is the contract; the backend serves it; not the other way around.* This reverses the current HANDOFF.md posture, which says "do not start frontend implementation until the backend decision pipeline is trusted."

**The audit found:**
- 110 required fields across 6 pages + global header + cross-cutting requirements
- **19 READY (17%), 27 PARTIAL (25%), 64 MISSING (58%)**
- The Matchup Scoreboard (the load-bearing page) has 3 of 16 fields ready
- The Trade Analyzer has 0 of 9 fields ready
- The #1 blocker is the complete absence of Rest-of-Week (ROW) projections — this blocks 18 fields
- Rolling stats cover 9 of 18 categories (missing R, TB, W, L, HRA, K(pitching), QS, NSV)
- Projections cover 8 of 18 categories
- No Layer 0 contracts exist for: scoreboard response shape, canonical player row, constraint budget, category status tags
- The existing Layer 3 work (L3A-L3F) was correctly built but narrowly scoped — it addresses scoring observability, not the data completeness required by the UI spec

**What hasn't changed:**
- Layer 2 is still certified complete — do NOT reopen it
- Production health is still good — probable_pitchers, park_factors, mlb_player_stats, statcast_performances all healthy
- The 7-layer architecture is correct — the audit maps every gap to a specific layer
- L3E (Market-Implied Probabilities) remains deferred — unchanged

---

## 3. WHAT HANDOFF.md MUST SAY AFTER YOUR UPDATE

The update must accomplish these specific things. Do not add or subtract from this list.

### 3A. Update the status line at the top

Change from:
> Status: Layer 2 certified complete. Layer 3B consolidation complete. Layer 3D observability complete. API endpoint live with auth. Decision pipeline observability complete. L3F (Decision Output Read Surface) complete. L3E (Market-Implied Probabilities) is deferred as a future enhancement. Do not reopen Layer 2 except for regressions.

To something that reflects:
> Status: Layer 2 certified complete. UI Specification Contract Audit complete — backend is 17% ready to serve the locked UI spec. 9-phase gated implementation plan adopted. Active workstream: Phase 0 (Layer 0 contract definitions). L3E deferred. Do not reopen Layer 2 except for regressions.

### 3B. Add a new section: "UI Contract Authority"

Place it after "Mission Accomplished" and before "Core Doctrine." Contents:

- State that `reports/2026-04-17-ui-specification-contract-audit.md` is the authoritative UI-to-backend mapping
- State that the UI spec defines 110 required fields across 6 pages, and the backend must serve all of them
- State the readiness: 17% READY, 25% PARTIAL, 58% MISSING
- State the top 5 blockers (from the audit's Ranked Blocker List, Section 5):
  1. ROW projection pipeline does not exist (blocks 18 fields)
  2. Rolling stats cover 9/18 categories (27+ cell gaps)
  3. Projections cover 8/18 categories (10+ cell gaps)
  4. Per-player games-remaining-this-week missing (blocks ROW pipeline)
  5. Acquisition count not tracked (blocks 5 fields)
- State the 6 canonical pages: Matchup Scoreboard, My Roster, Waiver Wire, Probable Pitchers/Streaming, Trade Analyzer, Season Dashboard
- State page priority: P1 = Scoreboard + Roster, P2 = Waiver + Streaming, P3 = Trade + Season

### 3C. Update the Layer Status table

| Layer | Status change |
|-------|--------------|
| 0 | Change from "Stable" to **"ACTIVE — Phase 0: define 6 missing contracts"** |
| 1 | Keep "Available" but add note: "5 pure functions needed: delta-to-flip, ratio risk, IP pace, acquisition budget, category-count delta, TB calculator" |
| 2 | Keep "Certified Complete" but add note: "7 data gap tasks identified for Phase 1 (acquisition counter, IP extractor, games-remaining, standings parsing, opposing-SP lookup, ownership % fix, playing-today status)" |
| 3 | Change from "Active" to **"ACTIVE — Phase 2: expand rolling stats + projections to 18 categories, build ROW projection pipeline"** — note that L3A/L3B/L3D/L3F remain complete |
| 4 | Change from "Hold until..." to **"GATED — partial lift after Phase 2 gate passes. Phase 3 wires H2H Monte Carlo, MCMC, and lineup solver to projected data."** |
| 5 | Change from "Maintenance" to **"GATED — Phase 4: build scoreboard, budget, and roster endpoints after Phase 3 gate passes"** |
| 6 | Change from "Maintenance" to **"GATED — Phase 5: build P1 pages after Phase 4 gate passes. 15 components salvageable, 9 CBB pages to archive."** |

### 3D. Replace the "Active Workstream" section

The current active workstream section lists L3A-L3F completion details. Archive those into the "Layer 3 Status" section (they're historical accomplishments). Replace with:

**Active Workstream: Phase 0 — Layer 0 Contract Definitions**

Tasks (from audit Section 6, Phase 0):
- P0-1: Define `LeagueConfig` constant (acquisition_limit=8, ip_minimum=18.0, il_slots=3, full roster shape, matchup_period_days=7)
- P0-2: Define `CategoryStatusTag` enum (LOCKED_WIN, LOCKED_LOSS, BUBBLE, LEANING_WIN, LEANING_LOSS)
- P0-3: Define `ConstraintBudget` Pydantic model
- P0-4: Define `MatchupScoreboardRow` and `MatchupScoreboardResponse` Pydantic models
- P0-5: Define `CanonicalPlayerRow` Pydantic model (22 fields, 18-category dicts)
- P0-6: Define `FreshnessMetadata` Pydantic model

Gate 0 criteria: all contracts compile, all reference 18 categories from `fantasy_stat_contract.json`, no contract allows optional fields for data the UI requires unconditionally.

### 3E. Replace the "Immediate Priority Queue" section

The current queue is all "Complete." Replace with the 9-phase plan summary:

| Phase | Layer Focus | Key Deliverable | Gate Criteria | Status |
|-------|------------|----------------|---------------|--------|
| 0 | L0 | 6 Pydantic contracts | All compile, all reference 18 categories | **NEXT** |
| 1 | L2 | 7 data gap closures | Acquisition counter + games-remaining + IP extractor verified against Yahoo | Blocked by Phase 0 |
| 2 | L3 | 18-category rolling stats + projections + ROW pipeline | ROW projections stable for full matchup week | Blocked by Phase 1 |
| 3 | L1 + L4 | Pure functions + engine wiring | H2H Monte Carlo with projected finals produces non-degenerate results | Blocked by Phase 2 |
| 4 | L5 | P1 page APIs (scoreboard, budget, roster, optimize) | All endpoints return complete data per contract | Blocked by Phase 3 |
| 5 | L6 | Matchup Scoreboard + My Roster pages | Pages render with live data, mobile-optimized | Blocked by Phase 4 |
| 6 | L3-L5 | P2 page backends (waiver v2, streaming) | Endpoints return complete data | Blocked by Phase 5 |
| 7 | L6 | Waiver Wire + Streaming pages | Pages render with live data | Blocked by Phase 6 |
| 8-9 | L3-L6 | P3 pages (Trade + Season Dashboard) | Complete | Blocked by Phase 7 |

### 3F. Update "Architect Review Queue"

Remove items that are now addressed by the phased plan. Add:
- The audit has 11 open questions (Section 8 of the audit report) that should be resolved during Phase 0 or Phase 1. Key ones: Yahoo API rate limits (Q1), W/L/SV/HLD/QS availability from Yahoo (Q2-Q3), FAAB vs priority waivers (Q4), opponent projection approach (Q5), matchup week boundary definition (Q7), acceptable scoreboard response time (Q8).
- The existing `contracts.py` has `UncertaintyRange`, `LineupOptimizationRequest`, `PlayerValuationReport`, and `ExecutionDecision` — Phase 0 new contracts should live alongside these, not replace them.
- L3E (Market-Implied Probabilities) remains deferred and unchanged. Do not conflate it with Phase 0-2 work.

### 3G. Keep these sections UNCHANGED

- "Mission Accomplished" — still accurate
- "Current Production Truth" — still accurate
- "Layer 2 Certification Record" — still accurate
- "Delegation Bundles" — no active delegation needed yet
- "Frontend Readiness Brief" — superseded by the phased plan but keep for historical reference; add a note at the top saying "NOTE: Superseded by the 9-phase gated plan. See 'Active Workstream' and 'Immediate Priority Queue' for the current plan."

### 3H. Update the "Last Updated" line

To: `April 17, 2026 (UI Specification Contract Audit incorporated. 9-phase gated implementation plan adopted. Phase 0 is next.)`

---

## 4. WHAT YOU MUST NOT DO

1. **Do not start coding.** No Python files, no migrations, no new endpoints. This session is documentation only.
2. **Do not delete historical completion records.** L3A, L3B, L3D, L3F achievements are real — move them into the Layer 3 status section as completed sub-items, don't erase them.
3. **Do not change the layer architecture.** The 7-layer model is correct. The audit maps cleanly onto it. The phases are sequenced through the layers.
4. **Do not reopen Layer 2.** The 7 data gap tasks in Phase 1 are NEW capabilities (acquisition counting, games-remaining), not regressions in existing L2 functionality.
5. **Do not touch L3E.** Market-Implied Probabilities remains deferred. The audit doesn't reference it. Don't conflate it.
6. **Do not create new files** besides updating HANDOFF.md. The audit report already exists and is the reference document.
7. **Do not change the audit report.** `reports/2026-04-17-ui-specification-contract-audit.md` is frozen as written.

---

## 5. VERIFICATION CHECKLIST

After updating HANDOFF.md, verify:

- [ ] The status line at the top references the UI Contract Audit and Phase 0
- [ ] A new "UI Contract Authority" section exists with readiness percentages and top 5 blockers
- [ ] The Layer Status table shows L0 as ACTIVE, L4/L5/L6 as GATED
- [ ] The Active Workstream section describes Phase 0 with 6 specific contract tasks
- [ ] The Immediate Priority Queue shows the 9-phase plan with gates
- [ ] L3A/L3B/L3D/L3F completion records are preserved (moved, not deleted)
- [ ] L3E remains explicitly deferred
- [ ] The Frontend Readiness Brief has a supersession note
- [ ] "Last Updated" reflects April 17, 2026 and the plan adoption
- [ ] No code changes were made
- [ ] The file `reports/2026-04-17-ui-specification-contract-audit.md` is referenced as the authoritative audit

---

## 6. SUCCESS CRITERIA FOR THIS SESSION

The single deliverable is an updated `HANDOFF.md` that:
1. A cold reader can understand the current state, the destination, and the sequenced path between them
2. Phase 0 tasks are concrete enough to start executing immediately in the next session
3. No ambiguity about what's active vs. gated vs. deferred
4. Historical accomplishments preserved
5. The 110-field UI contract is acknowledged as the requirements document that drives all downstream work

---

*End of prompt. This prompt is self-contained. No prior conversation context is assumed.*
