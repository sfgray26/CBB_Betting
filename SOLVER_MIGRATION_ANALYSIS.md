# Solver Migration Analysis: Greedy Allocator → Scarcity-Aware Solver

**Bug #1: Roster Optimize Endpoint Uses Wrong Solver**

**Date:** May 18, 2026  
**Analyst:** Agent D - Architecture Mapper  
**Status:** Analysis Complete - Ready for Implementation

---

## Executive Summary

The `/api/fantasy/roster/optimize` endpoint (lines 3474-3750 in `backend/routers/fantasy.py`) uses a custom-built greedy allocator that claims to be "scarcity-aware" but significantly diverges from the tested, production-ready `LineupConstraintSolver` in `backend/fantasy_baseball/lineup_constraint_solver.py`.

The endpoint's current implementation (added in "Bugfix May 15") attempts a two-phase scarcity-first approach but lacks:
1. Proper natural-position tie-breaking
2. OR-Tools ILP optimization (mathematically optimal solutions)
3. Post-greedy swap-improvement pass
4. Consistent slot configuration with the tested solver

**Recommendation:** Migrate the endpoint to use the `LineupConstraintSolver` class (or `DailyLineupOptimizer.solve_lineup()`) for consistent, tested, and mathematically optimal lineup assignments.

---

## 1. Current Implementation (Greedy Allocator)

### Location
- **File:** `backend/routers/fantasy.py`
- **Function:** `optimize_roster()` (lines 3474-3750)
- **Core Logic:** Lines 3602-3703

### How It Works

#### Phase 1: Scarcity-Adjusted Player Sorting (lines 3607-3626)
```python
def _scarcity_score(player):
    base_score = player["lineup_score"]
    positions = [p.upper() for p in (player.get("eligible_positions") or [])]
    
    # Bonus for scarce position eligibility (C=+9, SS=+8, 2B=+7, etc.)
    scarcity_bonus = 0
    for i, scarce_pos in enumerate(["C", "SS", "2B", "3B", "1B"]):
        if scarce_pos in positions:
            scarcity_bonus = max(scarcity_bonus, 10 - i)
    
    # Bonus for multi-position flexibility
    hitting_positions = set(positions) & _HITTER_POSITIONS
    if len(hitting_positions) >= 3:
        scarcity_bonus += 3
    
    return base_score + scarcity_bonus

# Sort by effective score (base + scarcity bonus)
player_data.sort(key=_scarcity_score, reverse=True)
```

#### Phase 2: Two-Pass Slot Assignment (lines 3637-3703)

**Pass 1: Scarce Hitting Slots (C, SS, 2B, 3B, 1B)**
- Iterates through scarce positions in order
- For each slot, finds first unassigned player who can fill it
- Uses `_can_fill_slot()` for eligibility check
- Only considers players in scarcity-sorted order

**Pass 2: Remaining Slots (OF, Util, SP, RP, P)**
- Iterates through remaining players in scarcity order
- For each player, finds first available slot they can fill
- Slots processed in `SCARCITY_PRIORITY` order

**Pass 3: Bench Assignment**
- Remaining unassigned players go to bench (up to 5 slots)

### Input/Output Contract

**Inputs:**
- `RosterOptimizeRequest` with optional `target_date`
- Yahoo roster data via `client.get_roster(team_key)`
- Player scores from `player_scores` table (14-day rolling)

**Outputs:**
- `RosterOptimizeResponse` with:
  - `starters`: List of `PlayerSlotAssignment`
  - `bench`: List of bench assignments
  - `unrostered`: Players not fitting in lineup
  - `total_lineup_score`: Sum of all starter scores

### Key Issues

1. **No Natural-Position Tie-Breaking**: The scarcity bonus is applied during sorting, not during slot assignment. Two players with the same effective score may be assigned arbitrarily.

2. **No Post-Assignment Optimization**: After initial assignment, there's no swap-improvement pass to correct suboptimal placements.

3. **Limited Scoring Model**: Uses only `player_scores.score_0_100` without the full talent-prior/matchup-modifier approach in `DailyLineupOptimizer`.

4. **No Locked Slot Support**: Cannot handle user-locked positions.

---

## 2. Target Implementation (Scarcity-Aware Solvers)

### Option A: LineupConstraintSolver (Recommended)

**Location:** `backend/fantasy_baseball/lineup_constraint_solver.py`

#### Architecture

**Two Solver Paths:**
1. **ILP Solver (OR-Tools CP-SAT):** Mathematically optimal solution
2. **Greedy Fallback:** Scarcity-first approach when OR-Tools unavailable

#### Key Features

**Slot Configuration (lines 83-93):**
```python
SLOT_CONFIG = [
    (PositionSlot.CATCHER,    ["C"],                        1),   # scarcest
    (PositionSlot.SHORTSTOP,  ["SS"],                       2),
    (PositionSlot.SECOND_BASE, ["2B"],                      3),
    (PositionSlot.THIRD_BASE, ["3B"],                       4),
    (PositionSlot.FIRST_BASE, ["1B"],                       5),
    (PositionSlot.OUTFIELD_1, ["OF", "LF", "CF", "RF"],     6),
    (PositionSlot.OUTFIELD_2, ["OF", "LF", "CF", "RF"],     7),
    (PositionSlot.OUTFIELD_3, ["OF", "LF", "CF", "RF"],     8),
    (PositionSlot.UTILITY,    ["C", "1B", "2B", "3B", "SS", 
                               "OF", "LF", "CF", "RF", "DH"], 9),
]
```

**ILP Optimization (lines 122-234):**
- Uses OR-Tools CP-SAT solver for constraint satisfaction
- Constraints:
  - Each slot filled by exactly 1 player
  - Each player in at most 1 slot
  - Player must be eligible for assigned slot
  - Locked slots enforced as hard constraints
- Objective: Maximize total score + natural-position bonus

**Natural-Position Bonus (lines 181-192):**
```python
# Bonus = 10 * (10 - scarcity_rank)
# C→+90, SS→+80, 2B→+70, 3B→+60, 1B→+50, OF→+40/30/20, Util→0
for pid in player_ids:
    player_eligs = eligibility.get(pid, [])
    for slot, eligible_positions, scarcity_rank in self.SLOT_CONFIG:
        if slot == PositionSlot.UTILITY:
            continue
        if any(pos in player_eligs for pos in eligible_positions):
            bonus = 10 * (10 - scarcity_rank)
            objective_terms.append(bonus * x[(pid, slot)])
```

**Greedy Fallback (lines 236-330):**
- Mirrors ILP logic without OR-Tools dependency
- Same natural-position bonus applied during candidate selection
- Handles locked slots

### Option B: DailyLineupOptimizer.solve_lineup()

**Location:** `backend/fantasy_baseball/daily_lineup_optimizer.py` (lines 799-1007)

#### Additional Features

**70/30 Talent-Matchup Scoring (lines 615-650):**
```python
# TALENT PRIOR (70%): per-game normalized ROS projections + live composite_z
# MATCHUP MODIFIER (30%): daily environment (run environment, park, etc.)
talent_prior = (
    proj.get("hr", 0) * 2.0 / _GAMES_ROS
    + proj.get("r", 0) * 0.3 / _GAMES_ROS
    + proj.get("rbi", 0) * 0.3 / _GAMES_ROS
    + proj.get("nsb", 0) * 0.5 / _GAMES_ROS
    + proj.get("avg", 0.0) * 5.0
) * 10 + cz_val * 1.0

lineup_score = talent_prior * 0.7 + matchup_modifier * 0.3 + 6.0
```

**Post-Greedy Swap-Improvement (lines 932-1006):**
```python
# After greedy assignment, check if any bench player scores higher
# than an active slot player they could replace
while swap_improved:
    swap_improved = False
    for active in slot_results:
        for bench in slot_results:
            if bench.lineup_score > active.lineup_score:
                # Check eligibility and swap if beneficial
```

**Off-Day Filtering:**
- Uses MLB odds API to detect teams without games
- Deprioritizes players whose teams are off
- Warning generated when off-day player starts

### Input/Output Contract

**LineupConstraintSolver.solve() Inputs:**
- `players`: List of dicts with 'player_id', 'name'
- `player_scores`: Dict mapping player_id to `EliteScore`
- `eligibility`: Dict mapping player_id to eligible positions
- `locked_slots`: Optional {slot: player_id} overrides

**LineupConstraintSolver.solve() Outputs:**
- `OptimizedLineup` dataclass:
  - `assignments`: List of `PlayerSlotAssignment`
  - `total_score`: Sum of assigned scores
  - `is_optimal`: True if OR-Tools found optimal solution
  - `solver_type`: "OR-Tools CP-SAT" or "Greedy (Scarcity-First)"
  - `unassigned_players`: List of unassigned player IDs

---

## 3. Side-by-Side Comparison

| Aspect | Current (Greedy) | Target (LineupConstraintSolver) | Target (DailyLineupOptimizer) |
|--------|------------------|----------------------------------|-------------------------------|
| **Solver Type** | Custom 2-phase greedy | OR-Tools ILP + Greedy fallback | Greedy + Swap-improvement |
| **Optimality** | Not guaranteed | Mathematically optimal (ILP) | Improved but not guaranteed |
| **Scarcity Order** | C, SS, 2B, 3B, 1B, OF, Util | C, SS, 2B, 3B, 1B, OF×3, Util | C, SS, 2B, 3B, 1B, OF×3, Util |
| **Natural Position Bonus** | ❌ No (sorting only) | ✅ Yes (+90 to +20) | ✅ Yes (via scarcity rank) |
| **Tie-Breaking** | Arbitrary | By natural position | By scarcity rank |
| **Post-Assignment Optimization** | ❌ No | ❌ No | ✅ Swap-improvement pass |
| **Locked Slots** | ❌ No | ✅ Yes | ❌ No |
| **Scoring Model** | player_scores.score_0_100 | EliteScore.total_score | 70/30 Talent-Matchup model |
| **Off-Day Handling** | ❌ No | ❌ No | ✅ Yes (odds-based) |
| **Test Coverage** | ❌ None dedicated | ✅ test_lineup_constraint_solver.py | ✅ test_lineup_scarcity_solver.py |
| **OR-Tools Required** | No | Optional (falls back to greedy) | No |

---

## 4. Key Differences in Slot Assignment Logic

### Example Scenario: Catcher Tie

**Roster:**
- Player A: C only, score=10.0
- Player B: C+SS, score=10.0
- Player C: SS only, score=9.0

**Current Greedy Behavior:**
1. Both A and B get +9 scarcity bonus (C eligibility)
2. Sorted order: A (10+9=19), B (10+9=19), C (9)
3. C slot: A fills it (first in list)
4. SS slot: B fills it (C already filled)
5. **Result:** A→C, B→SS ✅ (lucky)

**Problem:** If B appears first in the list (arbitrary), B→C and A→Util (❌ wrong).

**LineupConstraintSolver Behavior:**
1. ILP creates assignment variables for all player-slot combinations
2. Objective includes score + natural-position bonus
3. A at C = 10 + 90 = 100, B at C = 10 + 90 = 100
4. A at Util = 10 + 0 = 10, B at SS = 10 + 80 = 90
5. Total with A→C, B→SS = 100 + 90 = 190
6. Total with B→C, A→Util = 100 + 10 = 110
7. **Result:** A→C, B→SS ✅ (mathematically optimal)

---

## 5. Migration Challenges

### Challenge 1: Data Model Differences

**Current:**
- Uses raw Yahoo roster dicts with `player_key`, `name`, `positions`
- Scores from `player_scores` table directly

**Target:**
- Requires `EliteScore` objects with `total_score`, `reasoning`
- May need projection data for full scoring model

**Mitigation:**
- Create adapter to build `EliteScore` from existing `player_scores` data
- Or use `DailyLineupOptimizer.rank_batters()` to generate scores

### Challenge 2: Slot Representation

**Current:**
- String slots: "C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "P", "BN"
- Separate handling for pitchers

**Target (LineupConstraintSolver):**
- `PositionSlot` enum: CATCHER, FIRST_BASE, etc.
- Only batter slots (C, 1B, 2B, 3B, SS, OF1, OF2, OF3, Util)
- No pitcher handling

**Mitigation:**
- Use `DailyLineupOptimizer.solve_lineup()` which handles both
- Or call solver for batters, separate logic for pitchers

### Challenge 3: Response Format

**Current:**
- `RosterOptimizeResponse` with `PlayerSlotAssignment` (custom model)
- Includes `freshness` metadata

**Target:**
- `OptimizedLineup` with different `PlayerSlotAssignment` (different model)

**Mitigation:**
- Map solver output to existing response format
- Maintain backward compatibility

### Challenge 4: Database Session Handling

**Current:**
- Uses FastAPI `Depends(get_db)` for session

**Target:**
- `DailyLineupOptimizer` creates its own sessions via `SessionLocal()`

**Mitigation:**
- Pass db session to solver methods (may require refactoring)
- Or accept separate session management in optimizer

---

## 6. Recommended Migration Approach

### Phase 1: Minimal Change (Recommended for Bug Fix)

Replace the custom greedy logic in `optimize_roster()` with a call to `DailyLineupOptimizer.solve_lineup()`:

```python
# Current (lines 3602-3703)
# Replace with:

from backend.fantasy_baseball.daily_lineup_optimizer import DailyLineupOptimizer

optimizer = DailyLineupOptimizer()
slot_results, warnings = optimizer.solve_lineup(
    roster=raw_players,
    projections=[],  # Or fetch projections if available
    game_date=target_date,
    db=db,  # May need to modify solve_lineup to accept db
)

# Map slot_results to RosterOptimizeResponse format
```

**Pros:**
- Minimal code changes
- Leverages tested, production-ready code
- Gets swap-improvement pass and off-day handling

**Cons:**
- Still uses greedy (not ILP) optimization
- Requires modifying `solve_lineup()` to accept external db session

### Phase 2: Full Integration (Future Enhancement)

Integrate `LineupConstraintSolver` for ILP optimization when OR-Tools available:

```python
from backend.fantasy_baseball.lineup_constraint_solver import LineupConstraintSolver

solver = LineupConstraintSolver()
if solver.use_ortools:
    # Build EliteScore objects from player data
    elite_scores = {
        pid: EliteScore(total_score=score, ...)
        for pid, score in player_scores_map.items()
    }
    result = solver.solve(players, elite_scores, eligibility)
else:
    # Fall back to DailyLineupOptimizer
```

**Pros:**
- Mathematically optimal solutions
- Natural-position tie-breaking
- Locked slot support

**Cons:**
- More complex integration
- Requires OR-Tools dependency
- Need to map between data models

---

## 7. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Regression in lineup quality** | Low | High | A/B test with same rosters; verify C/SS assignment |
| **Response format changes break UI** | Medium | Medium | Maintain existing response format mapping |
| **Performance degradation** | Low | Low | Both solvers are O(n²) or better; ILP has 5s timeout |
| **Database session conflicts** | Medium | Medium | Pass db session explicitly; test transaction handling |
| **Missing pitcher handling** | High | High | Ensure `DailyLineupOptimizer` handles pitchers or add separate logic |
| **Off-day filtering too aggressive** | Medium | Medium | Configurable threshold; warnings in response |

---

## 8. Test Strategy

### Unit Tests to Add

1. **Tie-Breaking Test:** Two C-eligible players with equal scores → natural C player gets C slot
2. **Scarcity Order Test:** C slot filled before 1B even with lower score
3. **Multi-Position Test:** 1B/SS player fills SS when Catcher is scarce
4. **Swap-Improvement Test:** Better player on bench gets swapped into active slot
5. **Off-Day Test:** Player with no game gets warning but can still start if no alternative

### Integration Tests

1. **End-to-End API Test:** Call `/api/fantasy/roster/optimize` with known roster, verify expected assignments
2. **Database Session Test:** Ensure no session leaks or transaction issues
3. **Yahoo API Compatibility:** Verify response format matches frontend expectations

---

## 9. Files to Modify

### Primary Changes

1. **`backend/routers/fantasy.py`** (lines 3602-3703)
   - Replace custom greedy logic with call to tested solver
   - Maintain existing input/output contracts

### Potential Supporting Changes

2. **`backend/fantasy_baseball/daily_lineup_optimizer.py`**
   - Add `db` parameter to `solve_lineup()` for external session injection
   - Or refactor to use passed session instead of `SessionLocal()`

3. **`backend/fantasy_baseball/lineup_constraint_solver.py`**
   - May need adapter to work with Yahoo roster format directly

---

## 10. Decision Matrix

| Criteria | Current | DailyLineupOptimizer | LineupConstraintSolver |
|----------|---------|----------------------|------------------------|
| **Fixes the bug** | ❌ | ✅ | ✅ |
| **Minimal changes** | N/A | ✅ | ❌ |
| **Tested** | ❌ | ✅ | ✅ |
| **Optimal solutions** | ❌ | ❌ | ✅ (with OR-Tools) |
| **Handles pitchers** | ✅ | ✅ | ❌ (batters only) |
| **Locked slots** | ❌ | ❌ | ✅ |
| **Future-proof** | ❌ | ✅ | ✅ |

**Recommendation:** Use `DailyLineupOptimizer.solve_lineup()` for immediate bug fix (Phase 1). Consider `LineupConstraintSolver` integration for future enhancement with ILP optimization.

---

## Appendix A: Code References

### Current Implementation
- File: `backend/routers/fantasy.py`
- Lines: 3474-3750 (function), 3602-3703 (core logic)
- Key functions: `_scarcity_score()`, slot assignment loops

### Target Implementations
- `LineupConstraintSolver`: `backend/fantasy_baseball/lineup_constraint_solver.py` (441 lines)
- `DailyLineupOptimizer.solve_lineup()`: `backend/fantasy_baseball/daily_lineup_optimizer.py` lines 799-1007

### Tests
- `tests/test_lineup_constraint_solver.py` (159 lines)
- `tests/test_lineup_scarcity_solver.py` (scarcity-specific tests)
- `tests/test_fantasy_h2h_validations.py` (scarcity index tests)

---

## Appendix B: Scarcity Ranking Comparison

| Position | Current Rank | LineupConstraintSolver | _POSITION_SCARCITY |
|----------|--------------|------------------------|-------------------|
| C | 1 | 1 | 1 |
| SS | 2 | 2 | 2 |
| 2B | 3 | 3 | 3 |
| 3B | 4 | 4 | 4 |
| 1B | 5 | 5 | 10 |
| OF | 6 | 6-8 | 12 |
| Util | 7 | 9 | N/A |
| SP | 8 | N/A | 6 |
| RP | 9 | N/A | 7 |
| P | 10 | N/A | N/A |

*Note: Current implementation and LineupConstraintSolver agree on batter scarcity order, which is the critical piece for this bug fix.*
