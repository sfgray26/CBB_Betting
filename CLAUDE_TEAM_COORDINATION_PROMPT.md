# Team Coordination Prompt — Optimizing Parallel Workstreams

> **Context:** Token budget constraints require spreading fantasy baseball work across Kimi, Gemini, and OpenClaw. Need Claude's architectural guidance on dependencies, parallelization, and risk management.

## Current State: 5 P0 Critical Gaps Identified

| # | Gap | Files | Complexity | Dependencies |
|---|-----|-------|------------|--------------|
| 1 | **IL Roster Support** | `yahoo_client.py`, `schemas.py`, `waiver_edge_detector.py`, `main.py`, frontend | Medium | None — self-contained |
| 2 | **Closer Data Bug** | `player_board.py`, `projections_loader.py` | Low | None — data fix only |
| 3 | **No Closers Available Detection** | `waiver_edge_detector.py`, `main.py`, frontend | Medium | Depends on #1 (IL support) for accurate roster counting |
| 4 | **Missing Z-Scores** | `player_board.py`, `get_or_create_projection()` | Low | None — add fallback/placeholder z-scores |
| 5 | **Daily Lineup Optimizer** | `daily_lineup_optimizer.py` (rewrite), `main.py`, `schemas.py`, frontend | **HIGH** | None for backend, but frontend depends on API contract |

## User's Immediate Needs (Next 7 Days)

1. **IL Support** — Season starting, must understand roster spots
2. **Working Lineup Optimizer** — Daily competitive disadvantage without it
3. **Closer Detection** — Currently flying blind on saves strategy

## Team Capacity & Constraints

### Claude (Architect)
- **Availability:** Limited (token constrained)
- **Best For:** Architecture decisions, complex logic, integration points, code review
- **Avoid:** Data entry, documentation-only, repetitive patterns

### Kimi (Deep Intelligence)
- **Availability:** High for read/analysis, lower for write
- **Best For:** Algorithm design, testing strategies, code audits, specification writing
- **Constraint:** Proposes; Claude implements production code (per AGENTS.md)

### Gemini (Ops)
- **Availability:** High, but CODE-RESTRICTED (per EMAC-075)
- **Best For:** Railway ops, env vars, running migrations, log monitoring
- **Cannot:** Write Python/TypeScript, modify service logic

### OpenClaw (Execution)
- **Availability:** Automated/scheduled
- **Best For:** Real-time monitoring, data validation, scheduled checks
- **Constraint:** Local LLM (qwen2.5:3b) — good for detection, not architecture

## The Question for Claude

**How do we parallelize these 5 P0 gaps to minimize time-to-value while respecting dependencies and token constraints?**

Specifically:

### 1. Work Assignment Strategy
Which gaps should each agent own, considering:
- **Parallelization potential** (can work happen simultaneously?)
- **Review requirements** (does Claude need to review everything?)
- **Risk level** (high-risk changes need architect oversight)
- **User impact priority** (what delivers value fastest?)

### 2. Dependency Management
- Gap #3 (No Closers Detection) depends on #1 (IL Support) for accurate roster math
- Gap #5 (Lineup Optimizer) is large and self-contained — can Kimi design while Claude builds #1?
- Gap #2 and #4 are data fixes — can these be batched?

### 3. Token Optimization Strategies
Options to consider:

**Option A: Kimi writes specs, Claude implements**
- Kimi produces detailed implementation specs for gaps #1, #3, #5
- Claude reviews specs (low token), implements (high token but focused)
- Kimi writes tests (lower token than implementation)

**Option B: Parallel tracks with interfaces**
- Claude defines API contracts/schemas for all 5 gaps upfront (one session)
- Kimi works on algorithm designs (#5 optimizer logic)
- Gemini handles data/migrations (#2 closer projections CSV)
- Claude implements core logic in focused sessions

**Option C: MVP-first prioritization**
- Claude does #1 (IL Support) fully — unblocks roster management
- Claude does minimal #5 (Lineup Optimizer) — basic position constraints only
- Kimi takes #3 (closer detection) to full spec with trade/punt logic
- Defer advanced lineup features (park factors, weather, etc.)

**Option D: Frontend/backend decoupling**
- Claude defines API contracts for all gaps
- Frontend work starts immediately with mock data
- Backend implementation proceeds in parallel

### 4. Specific Architectural Questions

1. **For IL Support (#1):** Should we add `selected_position` to the base Yahoo response, or create a separate `get_lineup_positions()` call? Which is less breaking?

2. **For Lineup Optimizer (#5):** Is this a constraint satisfaction problem (use OR-Tools/pulp) or a simple greedy algorithm? Given 9 slots and ~16 players, brute force is feasible. What's the right approach?

3. **For Closer Detection (#3):** Should this be a special case in `waiver_edge_detector.py`, or a separate `CloserStrategyAdvisor` service? User needs both "what's available" AND "what should I do" (trade/punt/monitor).

4. **Test Strategy:** Given time pressure, which gaps MUST have tests before deployment vs. can be tested post-launch?

### 5. Suggested Immediate Actions

**For Claude (next session):**
- [ ] Confirm priority order of the 5 gaps
- [ ] Define API contracts/schemas for gaps #1, #3, #5
- [ ] Decide: MVP lineups vs. full optimizer
- [ ] Assign #2 and #4 (data fixes) to appropriate agent

**For Kimi (can start immediately):**
- [ ] Write detailed algorithm spec for Daily Lineup Optimizer (#5)
- [ ] Design test cases for all 5 gaps
- [ ] Audit current waiver logic for other gaps

**For Gemini (can start immediately):**
- [ ] Verify Railway env vars for MLB analysis
- [ ] Prepare migration strategy for schema changes
- [ ] Check Odds API quota/status

**For OpenClaw (scheduled):**
- [ ] Daily check: Are closers available on waiver wire?
- [ ] Alert if lineup optimizer returns invalid lineups

## Success Criteria

**Week 1 (Opening Day):**
- IL support working (accurate roster spot counting)
- Basic lineup optimizer (fills 9 valid slots)
- Closer gap detected and surfaced to user

**Week 2-3:**
- Full waiver strategy for saves (trade/punt/monitor)
- Advanced lineup optimization (park factors, matchups)
- All z-scores present

## Request for Claude

**Please provide:**

1. **Recommended work assignment** — Who does what, in what order?
2. **Dependency graph** — What can be parallel, what must be sequential?
3. **API contract definitions** — For gaps #1, #3, #5 (so frontend can work in parallel)
4. **Token-efficient implementation plan** — How to minimize Claude's coding time while maximizing quality?
5. **Risk assessment** — Which gaps are "ship fast and iterate" vs. "must be right first time"?

---

**Background:** User just picked up JoJo Romero and Marcus Semien, successfully used IL spots. Immediate need is working daily lineup optimizer before Opening Day. Token budget is tight — need to maximize parallel workstreams.
