# Fantasy Baseball UI/UX Research Document
## H2H One Win League - Strategic Design Specifications

**Date:** 2026-04-08  
**Author:** Kimi CLI (Deep Intelligence Unit)  
**For:** Claude Code (Principal Architect)  
**Status:** Research Complete - Awaiting Data Layer Implementation

---

## Executive Summary

This document synthesizes strategic fantasy baseball insights, UI/UX design principles, and strict data layer architecture requirements for a **10-team H2H One Win** fantasy baseball league. It serves as the definitive reference for UI design decisions **after** the data layer and validation systems are complete.

**Key Insight:** H2H One Win format fundamentally changes the optimization target from "maximize points" to "maximize consistency of 6-3 wins." The UI must function as a "consistency coach," not a scoreboard.

---

## Part 1: League Context & Strategic Implications

### 1.1 League Format Specifications

```yaml
Format: Head-to-Head (H2H) One Win
Teams: 10
Roster Positions:
  Batters: C, 1B, 2B, 3B, SS, LF, CF, RF, Util
  Pitchers: SP, SP, RP, RP, P, P, P
  Bench: 5 slots
  IL: 3 slots
  NA: 1 slot

Batting Categories (9):
  - Runs (R)
  - Hits (H)
  - Home Runs (HR)
  - Runs Batted In (RBI)
  - Strikeouts (K)
  - Total Bases (TB)
  - Batting Average (AVG)
  - On-base + Slugging (OPS)
  - Net Stolen Bases (NSB)

Pitching Categories (9):
  - Wins (W)
  - Losses (L)
  - Home Runs (HR)
  - Strikeouts (K)
  - Earned Run Average (ERA)
  - Walks + Hits per IP (WHIP)
  - Strikeouts per Nine Innings (K/9)
  - Quality Starts (QS)
  - Net Saves (NSV)

Transaction Rules:
  - Waiver Time: 1 day
  - Waiver Type: Continual rolling list
  - Max Acquisitions/Week: 8
  - Min IP/Week: 18
  - Trade Review: Commissioner (2 days)
  - IL Adds: No direct-to-IL (must add to bench first)
```

### 1.2 Strategic Implications for UI Design

| Format Quirk | Strategic Impact | UI Requirement |
|--------------|------------------|----------------|
| **One Win** (5-4 = 1W, 9-0 = 1W) | Consistency > Extremity; target reliable 6-3 not boom/bust 8-1 | Show "Win Probability" not projected record |
| **Position-specific OF** (LF/CF/RF) | CF scarcity creates artificial premium | Position topology visualization with scarcity index |
| **18 IP minimum** | Two-start SPs are essential; failing minimum = auto-loss | "IP Bank" with countdown; two-start command center |
| **NSB (Net Steals)** | CS hurts; efficiency > volume | Show success rate %, not just raw SB |
| **Rolling waivers** | Priority is finite resource; hoard for breakouts | Waiver priority advisor; "burn priority" warnings |
| **No direct-to-IL** | Roster management friction | "IL Shuffle" helper; bench fluidity planner |
| **8 max acquisitions** | Constrained streaming; batch moves | Acquisition budget visualization |

### 1.3 The "6-3 Compass" Design Philosophy

**Core UX Thesis:** The app should answer one question instantly: *"Am I reliably positioned to win 6 categories this week?"*

**Traditional App Approach (Wrong for This Format):**
- Shows projected score: "You're winning 7-2"
- Encourages punting 2-3 categories aggressively
- Optimizes for maximum margin

**Optimal Approach (Right for This Format):**
- Shows win probability distribution
- Highlights "swing categories" (within ±10%)
- Optimizes for minimum variance in 6+ category wins

---

## Part 2: Required Data Models (UI-Optimized)

### 2.1 Core Entities

```python
class PlayerCore(BaseModel):
    """Foundation player data"""
    player_id: str
    name: str
    positions: List[Position]  # LF, CF, RF - not generic "OF"
    eligibility_matrix: Dict[str, bool]  # {"LF": True, "CF": True, "RF": False}
    current_stats: PlayerStats
    projections: Projections  # ROS (rest of season)
    injury_status: InjuryStatus
    
class LeagueContext(BaseModel):
    """Your format burned into data layer"""
    format: ScoringType = ScoringType.H2H_ONE_WIN
    categories: List[Category]  # All 18 specific cats
    roster_slots: List[RosterSlot]  # 23 active with position requirements
    min_ip: int = 18
    max_acquisitions: int = 8
    waiver_type: WaiverType = WaiverType.ROLLING_LIST
    allows_direct_il: bool = False  # Critical for UI flow
```

### 2.2 UI-Aggregated View Models

These models should be **pre-computed** by the backend—never calculated client-side:

```python
class WeeklyCompass(BaseModel):
    """Primary dashboard widget"""
    win_probability: float  # 0-1, P(winning week)
    locked_categories: List[Category]  # >85% win probability
    swing_categories: List[SwingCategory]  # 40-60% win probability
    vulnerable_categories: List[Category]  # <30% win probability
    projected_record: Tuple[int, int]  # (wins, losses) if played 100 times
    confidence_score: ConfidenceLevel  # HIGH/MEDIUM/LOW based on variance
    recommendation: str  # "Solidify QS" or "Consider punting NSB"

class PositionScarcityIndex(BaseModel):
    """For roster topology visualization"""
    position: Position  # CF, LF, RF, etc.
    scarcity_score: float  # 0-100 (0 = extinct, 100 = abundant)
    league_rostered_count: int  # How many rostered across 10 teams
    league_rostered_pct: float  # Of available MLB players
    replacement_level_value: float  # Waiver wire best available
    multi_eligibility_premium: bool  # True if CF-eligible (scarcity hedge)
    trend_direction: Trend  # INCREASING, STABLE, DECREASING

class TwoStartOpportunity(BaseModel):
    """Two-start SP command center data"""
    pitcher_id: str
    name: str
    week: int
    game_1: MatchupRating  # Park factors, opponent OPS, etc.
    game_2: Optional[MatchupRating]
    total_ip_projection: float  # Usually 11-13 for two-start
    categories_addressed: List[Category]  # W, QS, K, K/9 likely
    acquisition_method: Literal["ROSTERED", "WAIVER", "FREE_AGENT"]
    waiver_priority_cost: Optional[int]  # If on waivers
    faab_cost_estimate: Optional[int]  # If your league switches to FAAB

class NSBEfficiencyProfile(BaseModel):
    """Net Stolen Bases optimization"""
    player_id: str
    raw_sb: int
    raw_cs: int
    nsb: int  # SB - CS
    success_rate: float  # SB / (SB + CS)
    league_avg_success_rate: float
    efficiency_grade: Literal["A", "B", "C", "D", "F"]
    sit_recommendations: List[Situation]  # vs LHP, in certain parks, etc.

class IPLedger(BaseModel):
    """18 IP minimum tracker"""
    current_ip: float  # Sum of scheduled starter IP projections
    projected_ip: float  # Including probable two-starts
    deficit: float  # Negative if under 18
    status: Literal["SAFE", "WARNING", "CRITICAL"]
    paths_to_minimum: List[AcquisitionOption]  # Streamer suggestions

class WaiverResourceBudget(BaseModel):
    """8 moves/week tracker"""
    moves_used: int
    moves_remaining: int
    moves_by_day: Dict[str, int]  # {"Mon": 2, "Tue": 0, ...}
    recommended_reserve: int  # Save 1-2 for Sunday closer claims
    effective_priority: int  # Current position in rolling list
    breakouts_available: List[Player]  # Worth using priority on
    streamers_available: List[Player]  # Use back of line

class ILShufflePlan(BaseModel):
    """No direct-to-IL workaround"""
    target_player: Player  # Injured player on waivers/FA
    required_steps: List[ShuffleStep]
    # 1. Drop bench player X
    # 2. Add target to bench
    # 3. Move target to IL
    # 4. (Optional) Reclaim dropped player if desired
    temp_bench_slot_needed: bool
    preview: RosterState  # Before/after visualization
```

---

## Part 3: Required UI Features & UX Patterns

### 3.1 The "Weekly Compass" Dashboard

**Purpose:** Primary entry point; answers "am I set up for 6-3?"

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ WEEK 12 COMPASS            [68% 6-PACK] │
├─────────────────────────────────────────┤
│                                         │
│  LOCKED           SWING           VULN  │
│  ───────         ───────         ─────  │
│  • HR ✓          • QS ?          • NSB ✗│
│  • RBI ✓         • K ?                   │
│  • TB ✓                                 │
│                                         │
│  [View Lineup]  [Optimize]  [Compare]   │
└─────────────────────────────────────────┘
```

**UX Principles:**
- **Hick's Law:** Only 3 actions; primary "Optimize" is largest
- **Jakob's Law:** Uses health bar metaphor for "68% 6-Pack"
- **Feedback Loop:** Updates in real-time as lineups change

**Data Requirements:**
- Monte Carlo simulation of remaining week (10,000 sims)
- Category-by-category probability distributions
- Opponent roster strength comparison

### 3.2 Roster Topology with Scarcity Overlay

**Purpose:** Surface CF scarcity and multi-eligibility value

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ ROSTER TOPOLOGY                         │
├─────────────────────────────────────────┤
│                                         │
│ CF: ★★☆ (Scarce)      [Bellinger] ✓✓   │
│     8/45 available      [Multi: LF,RF] │
│                                         │
│ LF: ★☆☆ (Deep)        [Schwarber]      │
│     35/60 available                     │
│                                         │
│ RF: ★★☆ (Moderate)    [Judge]          │
│     28/55 available                     │
│                                         │
│ [Find CF Options]  [Trade for CF]       │
└─────────────────────────────────────────┘
```

**Strategic Insight:** CF is the scarcest position in this format. Users should NEVER drop CF-eligible players without warning.

### 3.3 Two-Start Command Center

**Purpose:** Ensure 18 IP minimum; maximize volume category coverage

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ TWO-START OPPORTUNITIES - WEEK 12       │
├─────────────────────────────────────────┤
│                                         │
│ ROSTERED:                               │
│ • Cole (NYY)  Wed vs KC [+0.8] ⚾⚾    │
│   Proj: 13 IP | Adds: W, QS, K          │
│                                         │
│ AVAILABLE:                              │
│ • Eflin (TB)  Mon vs OAK, Sat vs DET    │
│   [CLAIM - Uses Waiver #3]              │
│   Proj: 12 IP | Matchup: +1.2, +0.5     │
│                                         │
│ IP BANK: 14/18 ⚠️  [Need 4 more]        │
│ [Find Streamers]  [View Probables]      │
└─────────────────────────────────────────┘
```

**Critical Features:**
- **IP Bank countdown** (never fail 18 IP)
- **Matchup difficulty ratings** (park-adjusted)
- **Acquisition cost transparency** (waiver priority impact)

### 3.4 NSB Efficiency Dashboard

**Purpose:** Surface that NSB is efficiency-based, not volume

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ NET STOLEN BASES EFFICIENCY             │
├─────────────────────────────────────────┤
│                                         │
│ Your NSB: +8 (League Avg: +12) ⚠️       │
│                                         │
│ Player Breakdown:                       │
│ • Carroll  22 SB / 3 CS  (88%) 🟢 A     │
│ • Alberts  15 SB / 8 CS  (65%) 🔴 D     │
│ • Rodriguez 8 SB / 0 CS  (100%) 🟢 A+   │
│                                         │
│ RECOMMENDATION:                         │
│ Sit Alberts vs LHP (58% career CS)      │
│ [View Weekly Matchups]                  │
└─────────────────────────────────────────┘
```

**Data Requirements:**
- SB/CS splits by opponent pitcher hand (LHP/RHP)
- Park factors for SB (some parks easier to steal)
- Real-time success rate calculations

### 3.5 Waiver Priority Intelligence

**Purpose:** Prevent burning priority on streamers

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ WAIVER RESOURCES                        │
├─────────────────────────────────────────┤
│                                         │
│ PRIORITY: 3/10  [Save for breakouts]    │
│ MOVES: 3/8 remaining this week          │
│                                         │
│ 🔴 BREAKOUT ALERT (Use Priority #3)     │
│    Colton Cowser called up              │
│    CF eligible [Addresses scarcity]     │
│    [CLAIM - WILL BURN PRIORITY]         │
│                                         │
│ 🟡 STREAMER OPTIONS (Back of line OK)   │
│    Eflin (2-start), Hicks vs OAK        │
│    [Add - Keeps Priority #3]            │
└─────────────────────────────────────────┘
```

**Behavioral Economics:**
- **Loss Aversion:** "Burn Priority" warning creates friction
- **Smart Defaults:** Streamers default to "add to back of line"

### 3.6 IL Shuffle Helper

**Purpose:** Navigate "no direct-to-IL" constraint

**Visual Design:**
```
┌─────────────────────────────────────────┐
│ ADD TO IL: Muncy (LAD - 10-day)         │
├─────────────────────────────────────────┤
│                                         │
│ REQUIRED STEPS:                         │
│                                         │
│ 1. ☐ DROP: [Select Bench Player ▼]      │
│    Options: Jones, Miller, Smith        │
│                                         │
│ 2. ☐ ADD Muncy to bench                 │
│                                         │  
│ 3. ☐ MOVE Muncy to IL (slot available)  │
│                                         │
│ [PREVIEW ROSTER]  [EXECUTE 3 STEPS]     │
└─────────────────────────────────────────┘
```

**UX Principle:** **Forgiveness**—preview mode shows exact outcome before confirming

### 3.7 Acquisition Budget Tracker

**Purpose:** Manage 8 moves/week constraint

**Visual Design:**
```
Moves: ████████░░ 6/8 used

[MON] Claimed: Eflin (2-ST)    [Priority: 3→8]
[WED] Streamed: Hicks @ OAK    [Priority: 8→9]
[FRI] Added: Duran (breakout)  [Priority: 9→10]

RESERVE: 2 moves recommended
- 1 for Sunday closer claim
- 1 for injury replacement buffer
```

### 3.8 Matchup Difficulty Lens

**Purpose:** Show opponent strengths relative to your build

```
┌─────────────────────────────────────────┐
│ WEEK 12 vs @GRONKSMADNESS               │
├─────────────────────────────────────────┤
│                                         │
│ Their Strength: Power (HR, TB, OPS)     │
│ Your Strength: Speed (R, NSB, AVG)      │
│                                         │
│ STRATEGY ADJUSTMENT:                    │
│ • Lean into pitching ratios (ERA/WHIP)  │
│ • Avoid HR-heavy streamers              │
│ • Prioritize SB efficiency over volume  │
│                                         │
│ [VIEW OPTIMAL LINEUP]                   │
└─────────────────────────────────────────┘
```

---

## Part 4: Data Architecture Requirements

### 4.1 The "Accuracy First" Principle

**Critical Constraint:** The UI cannot be more accurate than the data layer. Every feature above requires:

| Feature | Data Dependency | Accuracy Check |
|---------|-----------------|----------------|
| Win Probability | Monte Carlo convergence | 10,000+ sims, <5% variance |
| Position Scarcity | Real-time roster counts | Sync every waiver claim |
| Two-Start SPs | Probable starter schedules | Updated daily at 6 AM |
| NSB Efficiency | CS data (not all sources have) | Validate against MLB official |
| IP Bank | IP projections + probable starts | Re-calc on lineup changes |
| Waiver Priority | Exact rolling list order | Post-claim re-ranking |

### 4.2 Compute Layer Specifications

**Real-Time Updates (WebSocket):**
```python
# Events that trigger UI refresh
TRIGGER_EVENTS = [
    "lineup_change",
    "waiver_claim_processed",
    "stat_update_mid_game",
    "opponent_roster_change",
    "injury_status_change",
    "probable_pitcher_announcement"
]

# Latency Budget
MAX_LATENCY_MS = 400  # Doherty Threshold
TARGET_LATENCY_MS = 200
```

**Batch Computations (Daily/Hourly):**
```python
DAILY_JOBS = [
    "scarcity_index_recalc",      # 6 AM
    "two_start_sp_identification", # 6 AM
    "projection_model_update",     # 12 AM
    "waiver_priority_snapshot"     # Continuous
]

HOURLY_JOBS = [
    "win_probability_monte_carlo",  # When lineups approach lock
    "matchup_difficulty_ratings"    # Park factor updates
]
```

### 4.3 Caching Strategy

| Data Type | Cache Layer | TTL | Invalidation Trigger |
|-----------|-------------|-----|---------------------|
| Weekly Compass | Redis Hot | 5 min | Lineup/stat change |
| Scarcity Index | Redis Hot | 1 min | Waiver claim |
| Two-Start SPs | Redis Warm | 24 hr | Probable pitcher update |
| Player Stats | Redis Hot | 15 min | Stats API sync |
| NSB Efficiency | Redis Warm | 30 min | Daily stats sync |
| Waiver Priority | Redis Hot | 30 sec | Claim processing |

### 4.4 Error States & Graceful Degradation

```python
class DataFreshnessIndicator(BaseModel):
    status: Literal["fresh", "stale", "error", "calculating"]
    last_updated: datetime
    next_update: Optional[datetime]
    fallback_strategy: str  # "Showing cached from 5 min ago"
    retry_after: Optional[int]

# UI renders:
# 🟢 Fresh (live data)
# 🟡 Stale (cached, >15 min old)
# 🔴 Error (retry button)
# ⏳ Calculating (skeleton screen)
```

### 4.5 Validation Requirements

**Unit Tests (Must Pass Before UI Design):**
```python
def test_nsb_calculation():
    """NSB = SB - CS, not raw SB"""
    player = create_player(sb=5, cs=3)
    assert player.nsb == 2
    
def test_scarcity_index_accuracy():
    """CF scarcity reflects multi-eligibility"""
    # 10 teams × 1.2 CF average = 12 CFs rostered
    # But Bellinger (CF/LF/RF) counts for all three
    index = calculate_scarcity(Position.CF)
    assert index.multi_eligibility_adjusted == True
    
def test_ip_bank_precision():
    """IP projections must sum correctly"""
    roster = create_roster(sp=[cole, glasnow, eflin])
    bank = calculate_ip_bank(roster)
    assert bank.projected_ip == sum(sp.ip_projection for sp in roster.sp)
    
def test_one_win_probability():
    """5-4 and 9-0 both = 1 win"""
    sim_result = monte_carlo_sim(categories_won=5)
    assert sim_result.week_win == True
    
    sim_result = monte_carlo_sim(categories_won=9)
    assert sim_result.week_win == True  # Same value, not 9x better
```

**Integration Tests:**
```python
def test_sunday_scramble_concurrency():
    """50 users simultaneously checking IP bank at 11:55 PM"""
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = executor.map(simulate_user_session, range(50))
    
    assert all(r.latency < 400 for r in results)
    assert all(r.data_accuracy == 1.0 for r in results)

def test_waiver_priority_cascade():
    """Priority #1 claims → moves to #10, everyone shifts up"""
    initial_priority = get_priority(user_id="user_1")  # 3
    process_claim(user_id="user_3", priority=1)  # User 3 uses #1
    
    new_priority = get_priority(user_id="user_1")  # Should be 2
    assert new_priority == 2
```

---

## Part 5: Mobile-First Interaction Patterns

### 5.1 Thumb-Zone Gestures

| Action | Gesture | Zone | Safety |
|--------|---------|------|--------|
| Move to IL/Bench | Swipe Left (easy) | Middle | Reversible |
| Drop Player | Swipe Right + Confirm | Harder | Destructive |
| View Matchup Splits | Long Press | Any | Non-destructive |
| Quick Add to Lineup | Tap + Checkmark | Bottom | Reversible |
| Execute IL Shuffle | Bottom Sheet | Bottom | Preview first |

### 5.2 The "Daily Briefing" Notification

**Trigger:** 8:00 AM on game days (user timezone)

**Content:**
```
⚾ Today: 3 SPs starting, 2 vs weak offenses

🚨 Action: Consider sitting Giolito vs HOU
   (career 5.80 ERA vs Astros)

[Tap to Set Lineup] → deep link to lineup page
```

**Fogg Behavior Model:**
- **Motivation:** Game day urgency
- **Ability:** One-tap action
- **Trigger:** Contextual notification

### 5.3 Progressive Disclosure Pattern

```
Level 1 (Dashboard):     "68% 6-Pack Win"
Level 2 (Tap):           Category breakdown with bars
Level 3 (Tap category):  Player-by-player contribution
Level 4 (Tap player):    Full stat splits, matchup history
```

### 5.4 Loading States & Skeletons

```
┌─────────────────────────────────────────┐
│ WEEKLY COMPASS                          │
├─────────────────────────────────────────┤
│                                         │
│ ████████████░░░░  Loading...            │
│                                         │
│ • Category data: ✓                      │
│ • Opponent roster: ✓                    │
│ • Monte Carlo sim: ⏳ (8,200/10,000)    │
│                                         │
│ Est: 2 seconds remaining                │
└─────────────────────────────────────────┘
```

---

## Part 6: Implementation Checklist for Claude

### Phase 1: Data Layer Hardening (Before Any UI)

- [ ] **Schema Validation:** All models in Section 2.2 are queryable via API
- [ ] **Accuracy Suite:** Unit tests in Section 4.5 pass
- [ ] **Sync Pipeline:** Yahoo API sync includes NSB, QS, K/9 (not all default)
- [ ] **Scarcity Engine:** CF/LF/RF counts update within 1 min of waiver claims
- [ ] **Probable Pitchers:** Two-start detection accurate to 95%+
- [ ] **Monte Carlo:** Win probability converges with <5% variance

### Phase 2: API Contract Stabilization

- [ ] **REST Endpoints:** All "View Models" return <300ms
- [ ] **WebSocket:** Real-time updates for 6 key events (Section 4.2)
- [ ] **Error Contracts:** Every endpoint returns `DataFreshnessIndicator`
- [ ] **Caching:** Redis hit rate >85% for hot data
- [ ] **Versioning:** API versioned (v1, v2) for future evolution

### Phase 3: UI Component Library (After Data Layer)

**Build in this order:**
1. **Weekly Compass** (the "6-3" visualization) - highest value
2. **IP Bank** (18 IP countdown) - critical constraint
3. **Two-Start Command Center** - drives streaming decisions
4. **Position Scarcity** (CF alert) - format-specific edge
5. **Waiver Priority Advisor** - resource management
6. **NSB Efficiency** - category-specific optimization
7. **IL Shuffle Helper** - friction reduction

### Phase 4: Stress Testing

- [ ] **Sunday Scramble:** 50 concurrent users, <400ms response
- [ ] **Waiver Processing:** Priority cascade accurate across 10 teams
- [ ] **IP Bank Edge:** Alerts trigger at 14, 16, 17, 17.5 IP
- [ ] **Monte Carlo Load:** 10,000 sims complete <200ms

---

## Part 7: Anti-Features (What NOT to Build)

| Traditional Feature | Why Skip It | Replacement |
|---------------------|-------------|-------------|
| Projected Points Total | One Win format doesn't use raw points | Win Probability % |
| Category Rankings | Punting is high-variance in this format | Category Lock/Swing/Vulnerable |
| Generic "Add Player" Button | Ignores waiver priority cost | Contextual "Claim (Burns Priority #3)" |
| Raw SB Leaderboard | NSB penalizes caught stealings | NSB Efficiency Grade |
| Generic OF Filter | LF/CF/RF have different scarcity | Position Topology Visualization |
| Simple IL Toggle | You can't add directly to IL | IL Shuffle Workflow |
| Unlimited Streaming UI | 8 max acquisitions/week | Budget Tracker with Reserve |

---

## Appendix A: Cross-References

### Related Documents
- `AGENTS.md` - Role definitions (Claude owns backend, Kimi proposes UI)
- `HANDOFF.md` - Current operational state
- `ORCHESTRATION.md` - Swimlane routing
- `IDENTITY.md` - Risk posture

### External Frameworks Referenced
- **Hick's Law** - Decision paralysis reduction
- **Fitts's Law** - Touch target sizing
- **Jakob's Law** - Platform convention adherence
- **Fogg Behavior Model** - B = MAT (Behavior = Motivation × Ability × Trigger)
- **Doherty Threshold** - <400ms response time
- **Tesler's Law** - Conservation of complexity
- **Kelly Criterion** - Resource allocation (waiver priority as bankroll)
- **Monte Carlo Simulation** - Win probability calculation

---

## Document Control

**Version:** 1.0  
**Author:** Kimi CLI  
**Review Required By:** Claude Code (Principal Architect)  
**Approval Gate:** Data Layer Validation Complete (Section 6 Checklist)  
**Next Action:** Proceed to data layer implementation; do not begin UI design until Phase 1-2 complete.

---

*"The UI is a reflection of a rigorous data layer, not a calculator that gets out of sync."*
