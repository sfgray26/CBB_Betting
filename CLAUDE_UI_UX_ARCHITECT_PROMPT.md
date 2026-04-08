# Optimized Prompt for Claude Code
## Fantasy Baseball UI/UX Architecture Review & Roadmap

**Context:** You are Claude Code, Principal Architect & Lead Developer per `AGENTS.md`. You have absolute authority over architecture, integrations, and agent delegation. You own all backend code, API routes, Pydantic schemas, and SQLAlchemy models.

**Your Task:** Review the attached UI/UX research document, audit the current fantasy baseball codebase, and produce a technical implementation roadmap.

---

## INPUT DOCUMENTS (Read in Order)

### 1. PRIMARY DOCUMENT (Required First)
**File:** `reports/2026-04-08-fantasy-baseball-ui-ux-research.md`
**Status:** Fresh research from Kimi CLI (Deep Intelligence Unit)
**Contents:** 
- H2H One Win league format specifications (10-team, 18 categories, position-specific OF)
- 8 required UI features with detailed data model specifications
- Data architecture requirements (Monte Carlo, caching, WebSocket)
- Validation checklist that must pass before UI design
- Anti-features (what NOT to build)

### 2. CONTEXT DOCUMENTS (Read for Ground Truth)
**Read these to understand current system state:**

| File | Purpose |
|------|---------|
| `AGENTS.md` | Your authority boundaries, code ownership, quality gates |
| `HANDOFF.md` | What was completed and what's next |
| `ORCHESTRATION.md` | Swimlane routing rules |
| `IDENTITY.md` | Risk posture and system identity |
| `HEARTBEAT.md` | Active operational loops |
| `CLAUDE.md` | Project context and architecture patterns |

### 3. CODEBASE AUDIT TARGETS (Analyze These)
**Fantasy Baseball Backend:**
- `backend/fantasy_baseball/` - All fantasy baseball modules
- `backend/main.py` - API routes (you own these)
- `backend/schemas.py` - Pydantic schemas
- `backend/models.py` - SQLAlchemy models
- `backend/services/dashboard_service.py` - Current dashboard aggregation
- `backend/services/waiver_edge_detector.py` - Waiver logic
- `backend/fantasy_baseball/yahoo_client_resilient.py` - Yahoo API client
- `backend/fantasy_baseball/daily_lineup_optimizer.py` - Lineup optimization

**Existing Reports (Cross-Reference):**
- `reports/FANTASY_BASEBALL_2026_PROJECTIONS_COMPLETE.md`
- `reports/FANTASY_BASEBALL_GAP_ANALYSIS.md`
- `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md`
- `docs/MLB_FANTASY_ROADMAP.md`

---

## YOUR ANALYSIS FRAMEWORK

### Phase 1: Document Synthesis (Do First)
Answer these questions explicitly:

1. **Strategic Alignment:** Does the Kimi research align with the H2H One Win format constraints identified in existing roadmap docs?

2. **Data Model Gaps:** Compare the 7 UI-optimized view models in Section 2.2 against current `backend/schemas.py`. What new models are needed? What existing models need extension?

3. **Compute Requirements:** Can the current backend infrastructure support:
   - Monte Carlo simulation (10,000 runs, <200ms)
   - Real-time WebSocket updates for 6 trigger events
   - Position scarcity index recalculation (<1 min latency)
   - IP Bank tracking with lineup change triggers

4. **Yahoo API Limitations:** Does the current `yahoo_client_resilient.py` fetch all required stats?
   - NSB (Net Stolen Bases) requires CS data
   - QS (Quality Starts) - often not in default endpoints
   - K/9 - requires IP denominator
   - Probable starters for two-start detection

### Phase 2: Current State Audit (Do Second)

**For each of the 8 UI features in Section 3, determine:**

| Feature | Current State | Gap Analysis | Complexity |
|---------|---------------|--------------|------------|
| Weekly Compass | Exists? Partial? Missing? | What's needed? | Hours |
| Position Scarcity | ... | ... | ... |
| Two-Start Command Center | ... | ... | ... |
| NSB Efficiency | ... | ... | ... |
| IP Bank | ... | ... | ... |
| Waiver Priority | ... | ... | ... |
| IL Shuffle | ... | ... | ... |
| Matchup Difficulty | ... | ... | ... |

**Database Schema Audit:**
- Does the current schema support CF/LF/RF position granularity?
- Can it store Monte Carlo simulation results?
- Is there a waiver priority history table?
- Can it track acquisition budget (8 moves/week)?

**API Endpoint Audit:**
- Which required endpoints exist?
- Which need creation?
- Which need modification?

### Phase 3: Risk & Dependency Analysis (Do Third)

Identify:
1. **Blocking Dependencies:** What must be built first?
2. **External API Risks:** Yahoo API rate limits, data availability
3. **Performance Bottlenecks:** Where will Monte Carlo or scarcity calculations strain?
4. **Data Integrity Risks:** Sources of truth for NSB, QS, probable pitchers

---

## DELIVERABLE: Technical Implementation Roadmap

### Required Output Structure

```markdown
# Fantasy Baseball UI/UX Implementation Roadmap
**Version:** 1.0
**Author:** Claude Code (Principal Architect)
**Date:** [Current Date]

## Executive Summary
- One paragraph synthesis
- Go/No-Go decision on proceeding to UI phase
- Critical risks identified

## Phase 1: Data Layer Hardening (Weeks 1-2)
### 1.1 Schema Extensions
- [ ] New table: `position_scarcity_snapshots`
- [ ] New table: `weekly_compass_cache`
- [ ] Extend `players` table: `eligibility_matrix` JSONB
- [ ] New table: `two_start_opportunities`
- [ ] Extend `rosters` table: `ip_bank_projection`
- [ ] New table: `acquisition_budget_tracking`

### 1.2 Model Layer (Pydantic)
- [ ] `WeeklyCompass` view model
- [ ] `PositionScarcityIndex` view model
- [ ] `TwoStartOpportunity` view model
- [ ] `NSBEfficiencyProfile` view model
- [ ] `IPLedger` view model
- [ ] `WaiverResourceBudget` view model
- [ ] `ILShufflePlan` view model

### 1.3 Validation Suite
- [ ] Unit test: NSB calculation (SB - CS)
- [ ] Unit test: Scarcity index with multi-eligibility
- [ ] Unit test: IP bank precision
- [ ] Unit test: One win probability (5-4 = 1W, 9-0 = 1W)
- [ ] Integration test: Sunday scramble concurrency
- [ ] Integration test: Waiver priority cascade

**Exit Criteria:** All tests pass, PR approved

## Phase 2: Compute Layer (Weeks 3-4)
### 2.1 Monte Carlo Engine
- [ ] Implement `WinProbabilityEngine` class
- [ ] 10,000 simulation runs, <200ms target
- [ ] Category distribution outputs
- [ ] Confidence interval calculations

### 2.2 Real-Time Infrastructure
- [ ] WebSocket manager for live updates
- [ ] Event triggers: lineup_change, waiver_claim, stat_update, etc.
- [ ] Redis pub/sub integration
- [ ] <400ms latency guarantee (Doherty Threshold)

### 2.3 Background Jobs
- [ ] Daily: `scarcity_index_recalc` (6 AM)
- [ ] Daily: `two_start_sp_identification` (6 AM)
- [ ] Hourly: `win_probability_monte_carlo`
- [ ] Continuous: `waiver_priority_sync`

**Exit Criteria:** Load testing passes (50 concurrent users, <400ms)

## Phase 3: API Layer Stabilization (Weeks 5-6)
### 3.1 REST Endpoints
- [ ] `GET /api/v1/league/{id}/weekly-compass`
- [ ] `GET /api/v1/league/{id}/scarcity-index`
- [ ] `GET /api/v1/league/{id}/two-start-opportunities`
- [ ] `GET /api/v1/team/{id}/ip-ledger`
- [ ] `GET /api/v1/team/{id}/waiver-budget`
- [ ] `POST /api/v1/team/{id}/il-shuffle-plan`

### 3.2 WebSocket Channels
- [ ] `ws://.../league/{id}/live`
- [ ] Event types: compass_update, scarcity_change, ip_bank_alert

### 3.3 Error Handling
- [ ] `DataFreshnessIndicator` on all responses
- [ ] Graceful degradation strategies
- [ ] Retry logic with exponential backoff

**Exit Criteria:** All endpoints <300ms, 100% test coverage

## Phase 4: Data Pipeline Integrity (Week 7)
### 4.1 Yahoo API Extensions
- [ ] NSB data: Fetch CS (caught stealing) stats
- [ ] QS data: Quality starts endpoint or calculation
- [ ] K/9: Ensure IP denominator available
- [ ] Probable pitchers: Integration or alternative source

### 4.2 Sync Validation
- [ ] Idempotency checks
- [ ] Checksum validation against Yahoo totals
- [ ] Stale data detection (>15 min = warning)

**Exit Criteria:** Data accuracy >99.5%, sync latency <5 min

## Phase 5: UI Component Library (Week 8+) - DELEGATED
**Note:** Per AGENTS.md, Kimi CLI proposes UI components; you approve.

### 5.1 Component Specifications
- Provide exact prop interfaces for each component
- Define data contracts (what the UI receives)
- Specify loading states and error boundaries

### 5.2 Handoff to Kimi
- Create `HANDOFF.md` entry with component specs
- Define acceptance criteria
- Schedule review checkpoint

## Risk Register
| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Yahoo API lacks CS data | Medium | High | Fallback: Manual input or alternative source |
| Monte Carlo too slow | Low | Medium | Pre-compute, cache aggressively |
| WebSocket scale issues | Low | High | Load test at 2x expected capacity |
| Position eligibility churn | High | Medium | Real-time sync, not batch |

## Dependencies & Blockers
- [ ] Yahoo API credentials validated for extended stats
- [ ] Redis instance provisioned (if not exists)
- [ ] WebSocket infrastructure decision (Socket.io vs native)

## Appendix: Kimi Research Alignment
- Confirm which recommendations are accepted/rejected/modified
- Document deviations with rationale
```

---

## CONSTRAINTS & QUALITY GATES

### From AGENTS.md (Non-Negotiable)
- [ ] `venv/Scripts/python -m py_compile <file>` must pass for all new files
- [ ] Relevant `pytest tests/` subset must pass
- [ ] No `datetime.utcnow()` — use `datetime.now(ZoneInfo("America/New_York"))`
- [ ] No `status: False` bool-as-string leakage in Pydantic schemas
- [ ] You do NOT write frontend code (Kimi proposes, you approve)

### From IDENTITY.md (Risk Posture)
- All data integrity checks must be explicit
- Graceful degradation over crashing
- No "ghost changes" — all modifications justified in HANDOFF.md

### Code Conventions
- Use `[sys.executable, "-m", "pytest", ...]` not hardcoded paths
- Lazy imports for optional dependencies (DDGS, etc.)
- `sys.path.append()` only inside `if __name__ == "__main__":`

---

## OUTPUT INSTRUCTIONS

1. **Write the roadmap to:** `reports/2026-04-[XX]-fantasy-baseball-ui-roadmap.md`
2. **Update HANDOFF.md** with:
   - Summary of findings
   - Decision log (accept/reject/modify Kimi recommendations)
   - Next steps and owner assignments
3. **Create GitHub Issues** (if applicable) for each Phase 1-2 task
4. **Schedule checkpoint:** Propose review date with user for Phase 1 completion

---

## REMINDER: Your Role

You are the **Principal Architect**. Your job is not to implement everything yourself, but to:
- Design the architecture
- Define the contracts
- Ensure data integrity
- Delegate implementation (to yourself or others) with clear specifications
- Maintain quality gates

**Do not proceed to UI design until Phase 1-2 validation checklist is complete.**

The UI is a reflection of a rigorous data layer. Build the foundation first.
