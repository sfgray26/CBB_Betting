# ⚠️ ARCHIVED — See CLAUDE_TEAM_COORDINATION_PROMPT.md

> **Status:** ARCHIVED (April 11, 2026)  
> **Original Location:** `CLAUDE_COORDINATION_PROMPT.md` (root)  
> **Archive Reason:** Superseded by CLAUDE_TEAM_COORDINATION_PROMPT.md  
> 
> This document has been archived as part of repository documentation cleanup.
> It has been consolidated into `CLAUDE_TEAM_COORDINATION_PROMPT.md`,
> which contains the same P0 gap analysis with more detailed workstream assignments.
> 
> **Use this instead:** `CLAUDE_TEAM_COORDINATION_PROMPT.md` (now in root)
> 
> ---
> *Original content preserved below for historical reference.*

---

*The content below is preserved for historical reference only.*

# Claude Coordination Prompt: P0 Fantasy Baseball Gaps

**Context:** 5 P0 critical gaps identified. Token constraints require parallel work across team. Need architectural decisions on work distribution.

## The 5 P0 Gaps (Priority Order)

### 1. Daily Lineup Optimizer (CRITICAL - User's #1 Complaint)
**Problem:** Currently just ranks players by score, takes top 9. Ignores position requirements (C/1B/2B/3B/SS/OF/Util), off-days, matchups.

**Impact:** User must manually set lineups; benched Willi Castro (most flexible player) while starting spare 1B.

**Files:**
- `backend/fantasy_baseball/daily_lineup_optimizer.py` (complete rewrite needed)
- `backend/main.py` (endpoint update)
- `backend/schemas.py` (response format)

**Complexity:** HIGH - Constraint satisfaction problem with multi-position eligibility

---

### 2. IL Roster Support (CRITICAL - Season Started)
**Problem:** Yahoo updated to show IL status. Players on IL don't count against roster spots. App doesn't understand this.

**Impact:** Wrong drop suggestions, incorrect roster math.

**Files:**
- `backend/fantasy_baseball/yahoo_client.py` (extract selected_position)
- `backend/schemas.py` (add field)
- `backend/services/waiver_edge_detector.py` (exclude IL from drops)

**Complexity:** MEDIUM - API change + logic update

---

### 3. Closer Data Bug (HIGH)
**Problem:** Edwin Diaz showing 0 projected saves (should be 32+). Missing z-scores for some players.

**Impact:** Can't evaluate RP value, waiver recommendations broken.

**Files:**
- `backend/fantasy_baseball/player_board.py` (fix projections)
- `backend/fantasy_baseball/projections_loader.py` (fallback z-scores)

**Complexity:** LOW - Data fixes

---

### 4. No Closers Available Detection (HIGH)
**Problem:** Waiver wire has zero closers, system doesn't alert user or suggest alternatives (trade/punt/monitor).

**Impact:** User wasted time searching for non-existent closers.

**Files:**
- `backend/services/waiver_edge_detector.py` (detection logic)
- `backend/main.py` (API response flags)

**Complexity:** MEDIUM - New detection logic + alternative strategies

---

### 5. Missing Z-Scores (MEDIUM)
**Problem:** Some players (Emmanuel Clase) have no z-score, breaking waiver calculations.

**Impact:** Waiver recommendations fail for affected players.

**Files:**
- `backend/fantasy_baseball/player_board.py` (add fallback)

**Complexity:** LOW - Add placeholder/fallback logic

---

## Team Capacity & Constraints

| Agent | Best For | Constraints |
|-------|----------|-------------|
| **Claude (you)** | Architecture, complex logic, integration | Limited tokens - prioritize high-impact |
| **Kimi** | Specs, test design, audits, research | Proposes; you implement production code |
| **Gemini** | Railway ops, env vars, migrations | NO CODE - per EMAC-075 restriction |
| **OpenClaw** | Automated execution, monitoring | Local LLM - good for patterns, not architecture |

---

## Key Questions for You (Claude)

### 1. Work Distribution Strategy

**Option A: Kimi specs, Claude implements**
- Kimi writes detailed specs for gaps #1, #2, #4 (algorithm, API contracts, tests)
- You review specs (low token), implement (focused coding)
- Best for complex items requiring careful design

**Option B: Parallel tracks with interfaces**
- You define API contracts for all 5 gaps upfront (one session)
- Kimi works on test cases and edge case identification
- You implement core logic in focused sessions
- Best for maximizing parallel work

**Option C: MVP-first prioritization**
- You do #2 (IL Support) fully - unblocks roster management immediately
- You do minimal #1 (Lineup Optimizer) - basic position constraints only
- Kimi takes #4 (closer detection) to full spec with trade/punt logic
- Defer advanced lineup features (park factors, weather)
- Best for fastest user value

**Which strategy do you recommend?**

---

### 2. Dependency Management

Dependencies identified:
- Gap #4 (No Closers Detection) depends on #2 (IL Support) for accurate roster counting
- Gap #1 (Lineup Optimizer) is self-contained but frontend depends on API contract
- Gap #3 and #5 are data-only, no dependencies

**Question:** Should we sequence these or can any be parallel?

---

### 3. Daily Lineup Optimizer Architecture

This is the user's #1 pain point. Two approaches:

**Approach A: Constraint Satisfaction (OR-Tools/pulp)**
- Formal optimization with constraints
- Handles multi-position eligibility correctly
- More complex, needs library

**Approach B: Greedy Algorithm with Backtracking**
- Fill mandatory positions first (C, SS)
- Use multi-position players to cover gaps
- Simpler, easier to debug

**Which approach do you recommend?** Consider:
- 9 slots to fill, ~16 players on roster
- Must handle: C, 1B, 2B, 3B, SS, OF×3, Util
- Castro can play 2B/3B/LF/RF (key flex)
- Need to check off-days (team plays today?)

---

### 4. API Contract Decisions

Need your decisions on:

**For IL Support:**
```python
# Option A: Add to base player response
class RosterPlayerOut:
    selected_position: Optional[str]  # "IL", "BN", "C", etc.

# Option B: Separate endpoint
GET /api/fantasy/roster/positions  # Returns position mapping
```

**For Lineup Optimizer:**
```python
# Option A: Slot-based response
{
  "slots": {
    "C": {"player_id": "...", "name": "Diaz"},
    "1B": {"player_id": "...", "name": "Alonso"},
    ...
  }
}

# Option B: Player list with assigned slot
{
  "batters": [
    {"player_id": "...", "name": "Alonso", "slot": "1B", "status": "START"},
    ...
  ]
}
```

**Which contracts do you prefer?**

---

### 5. Token Optimization

Given limited tokens, which gaps should you personally implement vs. delegate spec writing to Kimi?

**Suggestion:**
- **You implement:** #2 (IL Support) - needed first, clear scope
- **You implement:** #1 (Lineup Optimizer) - core architecture, complex logic
- **Kimi specs:** #4 (Closer Detection) - strategy logic, trade/punt algorithms
- **You or Kimi:** #3, #5 (Data fixes) - straightforward

**Do you agree with this split?**

---

## Immediate Decisions Needed

Please provide:

1. **Priority order** of the 5 gaps (user says lineup optimizer is #1 pain)
2. **Work assignment** - Who does what (Claude/Kimi/Gemini/OpenClaw)?
3. **API contracts** - For gaps #1, #2, #4 (so frontend can work in parallel)
4. **Implementation approach** - For lineup optimizer (constraint vs greedy)
5. **First task** - What should I start on immediately?

---

## User's Immediate Needs (Next 7 Days)

1. Working lineup optimizer for Opening Day (March 28)
2. IL support (affects daily roster management)
3. Closer strategy guidance (currently flying blind)

**Focus on these first.**

---

## Background Context

**User's roster for reference:**
- C: Diaz (only option)
- 1B: Alonso, Pasquantino, Torkelson (3 - too many)
- 2B: Semien (new), Castro (flex), Westburg (IL)
- 3B: Chapman, Castro (flex)
- SS: Perdomo (only)
- OF: Soto, Nimmo, Buxton, Crow-Armstrong, Frelick, Suzuki (IL)
- SP: 6 healthy + Snell (IL)
- RP: Diaz (active), Adam (DTD), Romero (new)

**Key insight:** Castro's multi-position eligibility (2B/3B/LF/RF) is critical for optimization.

---

**When you respond, please:**
1. Answer the 5 questions above
2. Provide API contracts for gaps #1, #2, #4
3. State your recommended work distribution
4. Identify your first implementation task

This will unblock parallel work across the team.
