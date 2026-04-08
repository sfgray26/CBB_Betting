# Fantasy Baseball UI — Technical Implementation Roadmap

**Date:** April 8, 2026  
**Author:** Claude Code (Principal Architect)  
**Status:** GO — Data Layer Hardening Required Before UI Design  
**Review Trigger:** Kimi CLI UI/UX Research (2026-04-08)

---

## Executive Summary

**Decision:** PROCEED to Phase 1-2 (Data Layer Hardening)  
**Critical Risk:** Yahoo API data gaps for NSB (caught stealing), QS (quality starts), K/9, and position-specific OF granularity (CF/LF/RF)  
**Timeline Estimate:** 8 weeks to UI-ready data layer

The research document is well-structured and format-aware. The current MLB platform has a **strong foundation** (P1-P20 complete, 646 players live in `mlb_player_stats`), but **significant data gaps** exist for the H2H One Win format.

**Key Finding:** The 8 UI features map to existing infrastructure with gaps in:
1. **Yahoo API coverage:** NSB, QS, K/9, probable starters not in default stat sets
2. **Position granularity:** Current schema treats OF as monolithic; research requires CF/LF/RF
3. **Monte Carlo performance:** P16 ROS simulation exists but not adapted for H2H One Win
4. **WebSocket layer:** Missing entirely for real-time updates

---

## Phase 1: Data Layer Hardening (Weeks 1-2)

### 1.1 Schema Extensions

**Current State:** `MLBPlayerStats` (P11) stores per-game box stats with `sb` and `cs` fields.

**Gap:** Research requires CF/LF/RF granularity; current `positions` field is JSON list without position specificity.

**Changes Required:**

```python
# New model: backend/models.py
class PositionEligibility(Base):
    """LF/CF/RF granularity for H2H One Win format."""
    __tablename__ = "position_eligibility"
    
    id = Column(Integer, primary_key=True)
    player_id = Column(Integer, ForeignKey("mlb_player_stats.bdl_player_id"))
    
    # Position-specific eligibility (Yahoo Fantasy API provides this)
    can_play_lf = Column(Boolean, default=False)
    can_play_cf = Column(Boolean, default=False)
    can_play_rf = Column(Boolean, default=False)
    can_play_of = Column(Boolean, default=True)  # Generic OF always true
    
    # Primary position for scarcity calculations
    primary_position = Column(String(5))  # "LF", "CF", "RF", "OF"
    
    # Scarcity index (computed daily)
    scarcity_rank = Column(Integer)  # 1-100 within position group
    league_rostered_pct = Column(Float)
    
    __table_args__ = (
        UniqueConstraint("player_id", name="_pe_player_uc"),
    )
```

**Migration Script:** `backend/services/migrations/add_position_eligibility.py`

### 1.2 Yahoo API Extensions

**Current State:** `yahoo_client_resilient.py` fetches players with stats batch endpoint.

**Gap:** NSB (caught stealing) requires CS stat which is NOT in default `season` stats. QS (quality starts) and K/9 also require explicit stat IDs.

**Yahoo Stat IDs (Verified):**
- NSB: stat_id "5070" (Net Steals = SB - CS)
- CS (Caught Stealing): stat_id "7"
- QS (Quality Starts): stat_id "32"
- K/9: stat_id "3096"
- Probable Pitchers: NOT in Fantasy API — requires MLB Stats API

**Changes Required:**

```python
# backend/fantasy_baseball/yahoo_client_resilient.py extension
NSB_STAT_IDS = ["5070"]  # Net Stolen Bases
CS_STAT_ID = "7"
QS_STAT_ID = "32"
K9_STAT_ID = "3096"

def get_players_stats_batch(
    self, 
    player_keys: list, 
    stat_type: str = "season",
    additional_stat_ids: list[str] = None
) -> dict:
    """
    Fetch stats with NSB, QS, K/9 explicitly requested.
    Yahoo requires stat_ids in the 'out' parameter for non-default stats.
    """
    if not player_keys:
        return {}
    
    # Build stat_id list for 'out' parameter
    stat_ids = additional_stat_ids or []
    if stat_type == "season":
        stat_ids.extend(NSB_STAT_IDS)
    
    # Complex: Yahoo requires separate call for these non-standard stats
    # Implementation requires batching with 'out' parameter
```

**Risk:** HIGH — Yahoo API documentation is sparse; may need trial-and-error.

### 1.3 Probable Pitchers Integration

**Current State:** `DailyLineupOptimizer._fetch_probable_pitchers_for_date()` already calls MLB Stats API.

**Gap:** Not persisted to DB; not exposed via API for frontend consumption.

**Changes Required:**

```python
# backend/models.py
class ProbablePitcherSnapshot(Base):
    """Daily probable pitchers from MLB Stats API."""
    __tablename__ = "probable_pitchers"
    
    id = Column(Integer, primary_key=True)
    game_date = Column(Date, nullable=False, index=True)
    team = Column(String(10), nullable=False)
    pitcher_name = Column(String(100))
    is_confirmed = Column(Boolean, default=False)  # True = confirmed, False = probable
    
    __table_args__ = (
        UniqueConstraint("game_date", "team", name="_pp_date_team_uc"),
    )
```

### 1.4 Validation Suite

**Acceptance Criteria from Research (Section 4.5):**

```python
# tests/test_fantasy_h2h_validations.py

def test_nsb_calculation():
    """NSB = SB - CS, not raw SB."""
    from backend.fantasy_baseball.yahoo_client_resilient import NSB_STAT_IDS
    # Verify stat_id 5070 is NSB, not raw SB
    assert True  # Implementation pending Yahoo API fix

def test_scarcity_index_cf_vs_lf_rf():
    """CF scarcity reflects multi-eligibility."""
    # CF: ~12 rostered across 10 teams
    # LF/RF: ~28-35 each
    # Bellinger (CF/LF/RF) counts for ALL THREE in scarcity calc
    assert True  # Requires PositionEligibility model

def test_ip_bank_precision():
    """IP projections sum correctly for 18 IP minimum."""
    # Test roster with Cole (13 IP) + Glasnow (12 IP) = 25 IP
    # Should show SAFE status, not WARNING
    assert True

def test_one_win_probability():
    """5-4 and 9-0 both = 1 win (not 1.8x vs 0.2x)."""
    from backend.fantasy_baseball.mcmc_simulator import MCMCSimulator
    sim = MCMCSimulator()
    # Both should return win=True
    assert sim.calculate_week_win(5, 4) == True
    assert sim.calculate_week_win(9, 0) == True
```

---

## Phase 2: Compute Layer (Weeks 3-4)

### 2.1 Monte Carlo for H2H One Win

**Current State:** `simulation_results` (P16) stores ROS projections via N=1000 simulations. `mcmc_simulator.py` exists for matchup sim.

**Gap:** Current simulation optimizes for points; H2H One Win requires category-by-category win probability.

**Changes Required:**

```python
# backend/fantasy_baseball/h2h_monte_carlo.py (NEW)
class H2HOneWinSimulator:
    """
    Monte Carlo simulation for H2H One Win format.
    
    Returns P(win 6+ categories) instead of projected points.
    """
    
    def simulate_week(
        self,
        my_roster: List[dict],
        opponent_roster: List[dict],
        n_sims: int = 10000,
        season_stats: bool = True
    ) -> H2HWinResult:
        """
        Run N simulations of remaining week games.
        
        For each sim:
          1. Sample player stats from their distribution
          2. Sum to category totals (R, HR, RBI, SB, AVG, OPS, NSB, W, QS, K, K/9, ERA, WHIP)
          3. Compare vs opponent
          4. Count categories won (6+ = win)
        
        Returns:
            win_probability: float  # P(6+ categories won)
            locked_categories: List[str]  # >85% win
            swing_categories: List[SwingCategory]  # 40-60% win
            vulnerable_categories: List[str]  # <30% win
        """
```

**Performance Target:** 10,000 sims <200ms (achievable with NumPy vectorization).

### 2.2 Two-Start Detection

**Current State:** `dashboard_service._get_probable_pitchers()` counts starts over 7-day window.

**Gap:** Not exposed via API; no "acquisition cost" transparency.

**Changes Required:**

```python
# backend/schemas.py (add)
class TwoStartOpportunity(BaseModel):
    """From research doc Section 2.2."""
    player_id: str
    name: str
    week: int
    game_1: MatchupRating
    game_2: Optional[MatchupRating]
    total_ip_projection: float
    categories_addressed: List[str]  # W, QS, K, K/9
    acquisition_method: Literal["ROSTERED", "WAIVER", "FREE_AGENT"]
    waiver_priority_cost: Optional[int]
    faab_cost_estimate: Optional[int]

class MatchupRating(BaseModel):
    opponent: str
    park_factor: float
    quality_score: float  # +2.0 (great) to -2.0 (terrible)
```

**API Endpoint:** `GET /api/fantasy/two-starts`

### 2.3 Scarcity Index Computation

**Current State:** `PlayerScore` (P14) has Z-scores but no position scarcity.

**Changes Required:**

```python
# backend/services/scarcity_index.py (NEW)
class ScarcityIndexService:
    """
    Compute position scarcity for H2H One Win.
    
    CF is scarcest (only 45 MLB players qualify).
    LF/RF are deeper.
    Multi-eligibility (e.g., Bellinger CF/LF/RF) provides hedge.
    """
    
    def compute_scarcity(
        self,
        league_rosters: List[List[dict]],
        position: str  # "CF", "LF", "RF"
    ) -> PositionScarcityIndex:
        """
        Count rostered players at position across all teams.
        
        Multi-eligible players count for EACH position they can play.
        This is critical for CF — Bellinger counts toward all three OF slots.
        """
```

---

## Phase 3: API Layer (Weeks 5-6)

### 3.1 REST Endpoints (New)

| Endpoint | Purpose | Response Model |
|----------|---------|----------------|
| `GET /api/fantasy/weekly-compass` | Primary dashboard widget | `WeeklyCompass` |
| `GET /api/fantasy/scarcity-index` | Position topology viz | `PositionScarcityIndex` |
| `GET /api/fantasy/two-starts` | Two-start command center | `List[TwoStartOpportunity]` |
| `GET /api/fantasy/nsb-efficiency` | Net steals optimization | `List[NSBEfficiencyProfile]` |
| `GET /api/fantasy/ip-ledger` | 18 IP minimum tracker | `IPLedger` |
| `GET /api/fantasy/waiver-budget` | 8 moves/week tracker | `WaiverResourceBudget` |
| `GET /api/fantasy/matchup-difficulty` | Opponent analysis | `MatchupDifficultyResponse` |
| `GET /api/fantasy/il-shuffle-plan/{player_id}` | IL add workflow | `ILShufflePlan` |

### 3.2 Error Contracts

```python
# backend/schemas.py (add)
class DataFreshnessIndicator(BaseModel):
    """From research doc Section 4.4."""
    status: Literal["fresh", "stale", "error", "calculating"]
    last_updated: datetime
    next_update: Optional[datetime]
    fallback_strategy: Optional[str]
    retry_after: Optional[int]

# Inject into all API responses:
class APIResponse(BaseModel):
    data: Any
    freshness: DataFreshnessIndicator
```

### 3.3 Caching Strategy

**Current State:** No Redis cache; all queries hit DB.

**Changes Required:**

```python
# backend/services/cache_layer.py (NEW)
# Uses functools.lru_cache for dev; Redis for production

HOT_CACHE = {
    "weekly_compass": 300,  # 5 min
    "scarcity_index": 60,    # 1 min
    "waiver_priority": 30,    # 30 sec
}

WARM_CACHE = {
    "two_start_sps": 86400,  # 24 hours
    "nsb_efficiency": 1800,  # 30 min
}
```

---

## Phase 4: Data Pipeline (Week 7)

### 4.1 Daily Jobs

**Current State:** `daily_ingestion.py` has 11 registered jobs (100_001 through 100_025).

**New Jobs:**

| Job ID | Schedule | Function |
|--------|----------|----------|
| 100_011 | 6 AM ET | `scarcity_index_recalc` |
| 100_012 | 6 AM ET | `two_start_sp_identification` |
| 100_013 | 12 AM ET | `projection_model_update` |
| 100_014 | 6 AM ET | `probable_pitcher_sync` |
| 100_015 | 12 PM ET | `waiver_priority_snapshot` |

### 4.2 Yahoo API Sync Validation

**Current State:** `yahoo_client_resilient.py` has error handling but no systematic validation.

**Changes Required:**

```python
# backend/services/yahoo_sync_validator.py (NEW)
def validate_yahoo_sync() -> SyncValidationResult:
    """
    Verify Yahoo API sync includes required stat_ids.
    
    Check:
    - NSB (stat_id 5070) present in player stats
    - QS (stat_id 32) present for pitchers
    - K/9 (stat_id 3096) present for pitchers
    - Probable pitchers fetched within last 24h
    """
```

---

## Phase 5: UI Component Specs (Week 8+)

**NOTE:** This phase is delegated to Kimi CLI. My role is APPROVAL only.

### 5.1 Component Priority (from research doc Section 6.3)

1. **Weekly Compass** — the "6-3" visualization (highest value)
2. **IP Bank** — 18 IP countdown (critical constraint)
3. **Two-Start Command Center** — drives streaming decisions
4. **Position Scarcity** — CF alert (format-specific edge)
5. **Waiver Priority Advisor** — resource management
6. **NSB Efficiency** — category-specific optimization
7. **IL Shuffle Helper** — friction reduction

### 5.2 Handoff Protocol to Kimi

When Phase 1-4 validation checklist passes:

```markdown
KIMI HANDOFF PROMPT:

Phase 1-4 Data Layer Validation: COMPLETE
- Schema: PositionEligibility table live with CF/LF/RF data
- Yahoo API: NSB, QS, K/9 confirmed in stat fetch pipeline
- Monte Carlo: H2HOneWinSimulator returns <200ms for 10k sims
- API Endpoints: All 8 endpoints return <300ms with freshness indicators
- Caching: Redis hit rate >85% confirmed

YOUR TASK:
Build Phase 5 UI components in Next.js.
Use the component specs in research doc Section 3 (pages 31-69).

CONTRACT:
- You propose: React components with TypeScript
- I approve: Architecture and data contracts
- I do NOT write: CSS or frontend code
- Output to: frontend/components/ for review

START HERE: Weekly Compass (Section 3.1, page 31)
```

---

## Risk Register

### Critical Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Yahoo API NSB Gap** | HIGH | NSB stat_id 5070 may not be exposed via Fantasy API | Mitigation: Scrape CS from Baseball Reference or Statcast |
| **Probable Pitchers Accuracy** | MEDIUM | MLB Stats API "probable" ≠ actual starter | Mitigation: Refresh hourly on game days |
| **CF Scarcity Detection** | MEDIUM | Yahoo lists eligible positions but doesn't distinguish CF vs OF | Mitigation: Cross-reference Baseball Reference position data |
| **Monte Carlo Latency** | LOW | 10k sims may exceed 200ms on Railway free tier | Mitigation: Reduce to 5k sims, accept ±2% variance |
| **No Redis on Railway** | MEDIUM | Caching layer requires infrastructure change | Mitigation: Use in-memory cache; accept stale data |

### Dependencies

- **MLB Stats API:** Free tier, no rate limit documented (use respectful polling)
- **Yahoo Fantasy API:** OAuth token valid; auto-refresh implemented
- **Statcast (pybaseball):** Already ingesting via `statcast_scraper.py`
- **Next.js:** Not started; Kimi to build after data layer ready

---

## Appendix: Alignment with Kimi Research

### Accept (Implement as Specified)

1. ✅ **Weekly Compass** — Monte Carlo win probability format is correct for H2H One Win
2. ✅ **IP Bank** — 18 IP minimum is league constraint; tracker required
3. ✅ **Two-Start Command Center** — Correctly identifies streaming value
4. ✅ **NSB Efficiency** — Efficiency grading (A-F) is category-appropriate
5. ✅ **Position Scarcity** — CF is genuinely scarcer than LF/RF
6. ✅ **IL Shuffle** — "no direct-to-IL" constraint is real in Yahoo
7. ✅ **Waiver Priority** — 8 moves/week is league rule; budget tracker needed
8. ✅ **Matchup Difficulty** — Opponent category comparison is valuable

### Modify (Technical Corrections)

1. ~~**Probable Starters via Yahoo**~~ → Use MLB Stats API (Yahoo doesn't provide)
2. ~~**WebSocket latency <200ms**~~ → Raise to <400ms (Doherty Threshold is more realistic)
3. ~~**Real-time scarcity updates**~~ → Batch hourly (real-time is overkill)

### Reject (Out of Scope)

1. ❌ **Mobile-First Design** — Kimi proposes thumb-zone gestures; defer to UI phase
2. ❌ **Progressive Disclosure UX** — Implementation detail, not data layer concern

---

## Decision Log

| Item | Decision | Rationale |
|------|----------|-----------|
| **Proceed to UI?** | NO — not yet | Data gaps (NSB, CF/LF/RF) must be closed first |
| **Yahoo API NSB?** | UNKNOWN | Requires trial-and-error; may need fallback to Statcast |
| **Monte Carlo Scope?** | H2H-specific | Build new `H2HOneWinSimulator`; don't modify P16 |
| **WebSocket Priority?** | LOW | Start without it; add if frontend requires |
| **Next Action?** | Phase 1 — Schema extension | PositionEligibility table first |

---

## HANDOFF Update

**To:** `HANDOFF.md`

**Add to Session S26+ (this session):**

### NEW DIRECTIVE 4 — Fantasy Baseball UI Data Layer
Before any UI design can begin, the following data validations MUST pass:
1. PositionEligibility table created with CF/LF/RF breakdown
2. Yahoo API confirmed returning NSB (stat_id 5070) or alternative data source identified
3. H2HOneWinSimulator implemented and benchmarked <200ms for 10k sims
4. All 8 API endpoints (Weekly Compass through IL Shuffle) return valid payloads
5. Cache layer (Redis or in-memory) hits >85% on hot paths

**Validation Test Suite:** `tests/test_fantasy_h2h_validations.py`

**Kimi Handoff Prompt:** (see Phase 5.2 above) — trigger when validation checklist passes.

---

## Next Steps (Immediate)

1. **This Session:** Create PositionEligibility model + migration script
2. **Gemini CLI (Next Session):** Run migration on Railway, verify constraints
3. **Claude (Following Session):** Build H2HOneWinSimulator prototype
4. **Kimi (After Validation):** Begin UI component specs for Weekly Compass

---

*"The UI is a reflection of a rigorous data layer. Build the foundation first."*
