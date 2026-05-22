# P1 Bug Fix Orchestration Plan
## Multi-Agent Coordination for Critical Bug Resolution

**Date:** 2026-05-18  
**Scope:** 5 P1 Critical Bugs  
**Approach:** Parallel multi-agent execution with dependency management  
**Estimated Duration:** 3-5 days

---

## Executive Summary

This plan uses Hermes' multi-agent delegation to fix all 5 P1 bugs in parallel where possible, with careful dependency management to avoid conflicts.

### Strategic Approach
1. **Phase 1:** Math/parser fixes (isolated, no conflicts)
2. **Phase 2:** Solver unification (structural, high impact)
3. **Phase 3:** Data contract fixes (requires Phase 1)
4. **Phase 4:** Enable disabled features (requires stable base)

---

## The 5 P1 Bugs

### Bug 1: Roster Optimize Endpoint Uses Wrong Solver
- **File:** `fantasy.py:3447-3477`
- **Issue:** Greedy allocator instead of scarcity-aware solver
- **Impact:** Wrong lineup slots

### Bug 2: Inverted Implied Runs Sign  
- **File:** `daily_lineup_optimizer.py:420-436`
- **Issue:** Sign inverted for negative home spreads
- **Impact:** Pollutes batter/pitcher rankings

### Bug 3: Silent Empty Roster on Missing Count Field
- **File:** `yahoo_client_resilient.py:693-701`
- **Issue:** Returns empty list if count field missing
- **Impact:** Cascading "no roster" failures

### Bug 4: Disabled Pitcher Handedness Signal
- **File:** `matchup_engine.py:227-267`
- **Issue:** hand=None always, 35% of matchup score disabled
- **Impact:** Less discriminating matchup scores

### Bug 5: Unsafe Live Projection Fallback
- **File:** `projection_assembly_service.py:503-530`
- **Issue:** ilike("%%") matches any DB row
- **Impact:** Wrong projections for Yahoo-only players

---

## Multi-Agent Execution Plan

### Dependency Graph

```
Phase 1 (Parallel - Day 1)
├── Bug 2: Inverted Sign Fix [Agent A]
├── Bug 5: SQL Safety Fix [Agent B]
└── Bug 3: Empty Roster Fix [Agent C]

Phase 2 (Day 2-3, after Phase 1)
└── Bug 1: Solver Unification [Agent D + E]
    └── Depends on: Phase 1 complete

Phase 3 (Day 4)
└── Bug 4: Enable Handedness [Agent F]
    └── Depends on: Phase 2 stable

Phase 4 (Day 5)
└── Integration Testing [All Agents]
```

---

## Detailed Agent Assignments

### 🎯 AGENT A - Math Fix Specialist
**Bug:** #2 Inverted Implied Runs Sign  
**Estimated Time:** 2-3 hours  
**Risk:** Low

**Task:**
```python
# Current (WRONG):
def _implied_runs(self, home_spread: float, total: float) -> Tuple[float, float]:
    home_runs = (total / 2) + (home_spread / 2)  # <- Sign error here
    away_runs = total - home_runs
    return home_runs, away_runs

# Fix: When home team is favored (negative spread), 
# they should get MORE runs, not fewer
```

**Test Cases Required:**
1. Home fav -3, total 8 → home 5.5, away 2.5
2. Home dog +3, total 8 → home 2.5, away 5.5
3. Pick'em 0, total 8 → home 4, away 4

**Deliverable:** PR with fix + unit tests

---

### 🎯 AGENT B - Database Safety Specialist
**Bug:** #5 Unsafe Live Projection Fallback  
**Estimated Time:** 2-3 hours  
**Risk:** Low

**Task:**
```python
# Current (DANGEROUS):
if not player_id:
    query = query.where(PlayerProjection.player_name.ilike("%%"))
    # ^ Returns ALL rows!

# Fix: Return None or raise exception
if not player_id:
    logger.warning(f"No player_id provided for projection lookup")
    return None
```

**Deliverable:** PR with parameterized query fix

---

### 🎯 AGENT C - API Resilience Specialist
**Bug:** #3 Silent Empty Roster  
**Estimated Time:** 3-4 hours  
**Risk:** Medium

**Task:**
```python
# Current (SILENT FAILURE):
count = roster_data.get('count')
if count == 0:
    return []  # Silent empty list

# Fix: Explicit validation + logging
if 'count' not in roster_data:
    logger.warning("Yahoo payload missing 'count' field - attempting recovery")
    # Try to infer from players array length
    players = roster_data.get('players', [])
    count = len(players)
```

**Edge Cases:**
- Missing count field entirely
- count=0 but players array populated
- count mismatch with actual players

**Deliverable:** PR with validation + recovery logic

---

### 🎯 AGENT D + E - Architecture Refactoring Team
**Bug:** #1 Wrong Solver in Endpoint  
**Estimated Time:** 1-2 days  
**Risk:** HIGH

**Complexity:** This requires understanding TWO implementations:
1. Current greedy allocator in endpoint (lines 3447-3477)
2. Tested scarcity-aware solver (elsewhere in codebase)

**Approach:**
1. **Agent D** (Day 1): Map both implementations
   - Document current endpoint logic
   - Find and document scarcity-aware solver
   - Create comparison matrix

2. **Agent E** (Day 2): Execute migration
   - Replace greedy with scarcity-aware
   - Ensure backward compatibility
   - Add feature flag for rollback

**Key Questions to Answer:**
- Where is the tested scarcity-aware solver located?
- What are the input/output contracts?
- Are there any endpoint-specific requirements?

**Deliverable:** PR with unified solver + migration guide

---

### 🎯 AGENT F - Feature Enablement Specialist
**Bug:** #4 Disabled Pitcher Handedness  
**Estimated Time:** 4-6 hours  
**Risk:** Medium

**Task:**
```python
# Current (DISABLED):
def calculate_matchup_score(self, batter, pitcher):
    hand_bonus = 0  # Always 0!
    # ... rest of calculation

# Fix: Wire up handedness lookup
hand_bonus = self._get_hand_split_bonus(
    batter_handedness=batter.bats,
    pitcher_handedness=pitcher.throws
)
```

**Research Required:**
- Where should pitcher handedness come from?
- Is it in ProbablePitcherSnapshot?
- What's the actual 35% calculation?

**Deliverable:** PR with enabled feature + data source integration

---

## Execution Sequence

### Day 1: Parallel Phase 1
```bash
# Launch 3 agents simultaneously
hermes delegate_task "Fix Bug #2 - Inverted implied runs sign" &
hermes delegate_task "Fix Bug #5 - Unsafe SQL fallback" &
hermes delegate_task "Fix Bug #3 - Silent empty roster" &

# Wait for all 3 to complete
wait
```

### Day 2-3: Solver Unification (Critical Path)
```bash
# Agent D maps implementations
hermes delegate_task "Map greedy vs scarcity-aware solvers"

# After mapping complete:
hermes delegate_task "Migrate endpoint to use scarcity-aware solver"
```

### Day 4: Feature Enablement
```bash
# Only after solver is stable
hermes delegate_task "Enable pitcher handedness signal"
```

### Day 5: Integration Testing
```bash
# All agents participate in testing
hermes delegate_task "Run integration tests for all 5 fixes"
```

---

## Quality Gates

### Before Each PR
- [ ] Unit tests pass
- [ ] No regression in existing tests
- [ ] Code review by peer agent
- [ ] Documentation updated

### Integration Checkpoints
- [ ] Day 1 end: Phase 1 bugs fixed
- [ ] Day 3 end: Solver migration complete
- [ ] Day 4 end: Handedness enabled
- [ ] Day 5 end: All tests pass

---

## Risk Mitigation

### High-Risk: Solver Unification (Bug #1)
**Mitigation:**
- Feature flag for instant rollback
- Parallel implementation period
- Gradual rollout (canary)
- Full regression test suite

### Medium-Risk: Handedness Enablement (Bug #4)
**Mitigation:**
- Validate data source availability
- A/B test before full enable
- Monitor matchup score distribution

### Low-Risk: Others (Bugs #2, #3, #5)
**Mitigation:**
- Standard code review
- Unit test coverage

---

## Communication Protocol

### Daily Standups (Async via Discord)
Each agent posts:
```
Agent [A-F] Update:
- Completed: [task]
- Blocked by: [dependency or none]
- Next: [next task]
- ETA: [time estimate]
```

### Escalation Path
1. **Technical issues** → Post in #system-logs with @admin
2. **Conflicts** → Head agent (you) decides
3. **Scope creep** → Must be approved

---

## Success Criteria

### Bug Fixes
- [ ] Bug #1: Endpoint uses scarcity-aware solver
- [ ] Bug #2: Implied runs sign correct for all cases
- [ ] Bug #3: Roster recovery when count missing
- [ ] Bug #4: Handedness signal active (hand != None)
- [ ] Bug #5: Safe fallback (no ilike("%%"))

### Quality Metrics
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] No new warnings/errors in logs
- [ ] Performance unchanged or improved

### Documentation
- [ ] Migration guide for solver change
- [ ] Updated API documentation
- [ ] Runbook for future issues

---

## Rollback Plan

If critical issues arise:

1. **Immediate (Feature Flags)**
   ```python
   USE_SCARCITY_SOLVER = os.getenv("USE_SCARCITY_SOLVER", "false").lower() == "true"
   ```

2. **Short-term (Git Revert)**
   ```bash
   git revert HEAD  # Last PR
   railway deploy
   ```

3. **Long-term (Hotfix Branch)**
   ```bash
   git checkout -b hotfix/p1-rollback
   # Revert specific commits
   # Deploy emergency fix
   ```

---

## Kanban Board Setup

### Columns
1. **Backlog** → Ready to assign
2. **In Progress** → Agent working
3. **Code Review** → Peer review
4. **Testing** → Integration testing
5. **Done** → Merged to stable

### Swimlanes
- **Phase 1** (Parallel): Bugs #2, #3, #5
- **Phase 2** (Critical): Bug #1
- **Phase 3** (Enhancement): Bug #4
- **Phase 4** (Validation): Integration testing

---

## Next Steps

### Immediate (Today)
1. ✅ Review and approve this plan
2. 🎯 Create kanban board with these tasks
3. 🚀 Launch Phase 1 agents (3 parallel)

### This Week
4. Monitor Phase 1 progress
5. Trigger Phase 2 when Phase 1 complete
6. Execute Phase 3 and 4

### Success Measurement
7. Run UAT after all fixes merged
8. Verify Elite FM and Quant scores improve
9. Document lessons learned

---

## Ready to Execute?

**Choose your path:**

**Option A: Launch All Now**
```bash
# Create kanban and start all Phase 1 agents
hermes kanban create --from docs/P1_BUG_FIX_ORCHESTRATION_PLAN.md
```

**Option B: Phased Launch**
```bash
# Start with lowest risk (Bug #2 - Math fix)
hermes delegate_task "Fix Bug #2 - Inverted implied runs sign in daily_lineup_optimizer.py"
```

**Option C: Review First**
- Ask questions about the plan
- Modify agent assignments
- Adjust timeline

---

**Plan Version:** 1.0  
**Last Updated:** 2026-05-18  
**Status:** Ready for Execution
