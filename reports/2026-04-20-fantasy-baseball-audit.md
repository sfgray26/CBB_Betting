# Fantasy Baseball Application Audit Report
**Lead Quantitative Architect & Backend Systems Analyst**
**Date**: April 20, 2026
**Scope**: Phase 1-3 Comprehensive Audit

---

## Executive Summary

### Critical Findings

| Issue | Severity | Impact |
|-------|----------|--------|
| **Deployment Gap** | CRITICAL | April 20 UAT fixes completed locally but NOT deployed to Railway |
| **Hollow Waiver Intelligence** | HIGH | All `owned_pct: 0.0`, empty `category_contributions` in production |
| **Universal Drop Bug** | HIGH | 22/22 waiver recommendations target same player (Garrett Crochet) |
| **Incomplete Category Coverage** | MEDIUM | 3/18 categories missing from scoring engine (SB, L, HR_P, NSV) |
| **Greenfield Categories** | MEDIUM | 4/18 categories return 0.0 (W, L, HR_P, NSV) |
| **Broken Schedule Lookup** | MEDIUM | All players show `has_game: false`, `games_count: 0` |

### Production vs. Local State

**Production (Railway)**: Serving pre-April 20 code with:
- Hollow waiver intelligence (no ownership percentages, no category breakdowns)
- All drop recommendations pointing to single player
- Unrealistic ROS projections (0.00 ERA, 91 HR, 204 RBI)

**Local Environment**: April 20 UAT remediation completed but NOT deployed:
- Fixed Yahoo API parameter issues (K-20, K-24)
- Coverage-aware drop protection
- IL slot awareness
- Projection freshness gates

### Root Cause Analysis

1. **Data Enrichment Failure**: `cat_scores` missing from Yahoo payloads; projection enrichment not wired into waiver endpoint
2. **Deployment Pipeline**: No automated deployment from local fixes to Railway
3. **Schema Mismatch**: Postman response models expect fields that Yahoo API doesn't provide natively

---

## API & Data Ingestion Audit

### Yahoo Fantasy API Integration

**File**: `backend/fantasy_baseball/yahoo_client_resilient.py`

**Status**: OAuth 2.0 functional, but critical parameter issues identified:

| Issue | Reference | Fix Status |
|-------|-----------|------------|
| `out=metadata` strips `percent_rostered` | K-20 | Fixed locally |
| `out=stats` causes 400 error | K-24 | Fixed locally |
| Stats batch failure non-fatal | Integration test | Fixed locally |

**Current Implementation**:
```python
# FIXED: get_free_agents() now omits 'out' param
def get_free_agents(self, count=25, position=None):
    params = {"status": "A", "count": count}
    if position:
        params["position"] = position
    # NO 'out' param - was causing 400 or stripping percent_rostered
```

### Data Flow Diagram

```
Yahoo API → YahooFantasyClient → dashboard_service.py → API Response
                          ↓
                  player_board.py (projections)
                          ↓
                  waiver_edge_detector.py (enrichment)
                          ↓
                  mcmc_simulator.py (win probability)
```

**Broken Links**:
1. Yahoo → `cat_scores`: Yahoo doesn't provide category z-scores
2. `player_board` → Production: 200+ hardcoded players, not dynamically loaded
3. Schedule lookup: MLB Stats API not integrated for `has_game`/`games_count`

### Postman Response Analysis

**Sample**: `waiver_200.json` (25 free agents)

| Field | Expected | Actual | Root Cause |
|-------|----------|--------|------------|
| `owned_pct` | 0-100 | 0.0 (all) | K-20 regression, not fixed in prod |
| `category_contributions` | {} with values | {} (empty) | Scoring engine not wired |
| `matchup_opponent` | Team name | "TBD" | Scoreboard parsing fragile |
| `starts_this_week` | 0-2 | 0 (all) | Schedule lookup broken |
| `statcast_signals` | Array of signals | [] (empty) | Not populated |

**Sample**: `decisions_200.json` (31 decisions)

| Issue | Details |
|-------|---------|
| Universal drop target | All 22 waiver decisions drop Garrett Crochet (bdl_id 555) |
| Unrealistic projections | "Projects 0.00 ERA ROS", "Projects 91.2 HR ROS" |

### Endpoint Coverage

| Endpoint | Path | Status |
|----------|------|--------|
| Lineup Recommendations | `/api/fantasy/lineup/{lineup_date}` | Live, lacks category awareness |
| Waiver Recommendations | `/api/fantasy/waiver` | Live, hollow data |
| Decisions | `/api/fantasy/decisions` | Live, buggy drop logic |
| Dashboard Waiver Targets | `/api/dashboard/waiver-targets` | Live |

---

## Projection Model Diagnostics

### Scoring Engine (`scoring_engine.py`)

**Coverage**: 15/18 categories computed

**Missing Categories**:
- `SB` (Stolen Bases) - mapped to `NSB` but not computed
- `L` (Pitching Losses) - greenfield
- `HR_P` (Home Runs Allowed) - greenfield
- `NSV` (Net Saves) - greenfield

**Z-Score Computation**:
```python
def compute_league_zscores(players, category_code):
    # Uses type-appropriate player pools
    # Caps z-scores at ±3.0 to dampen outlier distortion
    # LOWER_IS_BETTER handled via inversion
```

**Issue**: Population standard deviation calculation assumes normal distribution; fantasy stats are often skewed.

### ROW Projector (`row_projector.py`)

**Algorithm**: 60% rolling (14d) + 40% season rate blending

**Greenfield Categories** (return 0.0):
```python
_GREENFIELD_CATEGORIES = frozenset({"W", "L", "HR_P", "NSV"})
```

**Hardcoded Values**:
```python
_DEFAULT_SEASON_DAYS = 100  # Should use actual days_into_season
_STANDARD_WINDOW_DAYS = 14
_ROLLING_WEIGHT = 0.60
_SEASON_WEIGHT = 0.40
```

**Ratio Formulas**:
```python
AVG  = sum(H) / sum(AB)
OPS  = (sum(H+BB)/sum(AB+BB)) + (sum(TB)/sum(AB))
ERA  = 27 * sum(ER) / sum(IP_outs)
WHIP = 3 * sum(H+BB) / sum(IP_outs)
K/9  = 27 * sum(K) / sum(IP_outs)
```

### Player Board (`player_board.py`)

**Current State**:
- 200+ hardcoded Steamer/ZiPS consensus players
- `get_or_create_projection()` returns board-compatible dict
- Falls back to position-average z_score proxy for unknown players
- Runtime cache: `_projection_cache`

**Issue**: Hardcoded player base becomes stale; not refreshed from live projection sources.

### MCMC Simulator (`mcmc_simulator.py`)

**Implementation**: Vectorized numpy sampling, 1000 simulations in <50ms

**Per-Player Weekly SD** (z-score units):
```python
_PLAYER_WEEKLY_STD = {
    "r": 0.70, "h": 0.55, "hr_b": 0.65, "rbi": 0.70,
    "k_b": 0.50, "tb": 0.65, "nsb": 0.90,
    "avg": 0.40, "ops": 0.40,
    "w": 0.85, "l": 0.85, "hr_p": 0.75,
    "k_p": 0.75, "qs": 0.80, "nsv": 1.00,
    "era": 0.65, "whip": 0.55, "k_9": 0.40,
}
```

**Position Multipliers**:
```python
_POSITION_MULT = {
    "C": 1.30, "2B": 1.10, "SS": 1.10, "3B": 1.10,
    "RP": 1.50,  # Relievers most volatile
    "SP": 1.20,
}
```

**Win Threshold**: Dynamic (n_cats / 2), defaults to 10 for 18 categories

### Decision Engine (`decision_engine.py`)

**Lineup Scoring** (simple, non-category-aware):
```python
score_0_100 = normalized_value * 100
momentum = (recent_14d - season_avg) * 10
proj_bonus = sum(proj_categories.values()) * 5

final_score = 0.6 * score_0_100 + 0.3 * momentum + 0.1 * proj_bonus
```

**Issue**: No category awareness; doesn't optimize for category deficits.

**Greedy Algorithm**:
1. Sort all candidates by score
2. For each slot, pick best available eligible player
3. No backtracking or global optimization

### Waiver Edge Detector (`waiver_edge_detector.py`)

**Protected Drop Logic**:
```python
def is_protected_drop_candidate(player):
    if z_score >= 4.0: return True  # Elite threshold
    if tier <= 2 or adp <= 30: return True
    if owned_pct >= 92 and z_score >= 1.5: return True
    return long_term_hold_floor(player) >= 2.25
```

**Long-Term Hold Floor**:
```python
_TIER_HOLD_FLOORS = {
    1: 4.5, 2: 3.5, 3: 2.75, 4: 2.0, 5: 1.25,
}
```

**Issue**: Universal drop bug indicates `_weakest_droppable_at()` returning same player regardless of FA position.

---

## Proposed Architecture/Refactoring Plan

### Phase 1: Immediate Deployment (Priority 0)

**Action**: Deploy April 20 UAT fixes to Railway

**Bundle Includes**:
1. K-20 fix: Remove `out=metadata` from Yahoo API calls
2. K-24 fix: Remove `out=stats` from base players call
3. Coverage-aware drop protection
4. IL slot awareness
5. Projection freshness gates

**Verification Commands**:
```bash
# Before deploy
railway run python -m pytest tests/test_waiver_integration.py -q

# After deploy
curl https://<production-url>/api/fantasy/waiver | jq '.top_available[:3] | .[].owned_pct'
# Should see non-zero values
```

### Phase 2: Data Enrichment Pipeline (Priority 1)

**Objective**: Wire `cat_scores` into waiver endpoint

**Approach**:
```
Yahoo FA List → player_board.get_or_create_projection()
                              ↓
                    enrich with cat_scores
                              ↓
                    WaiverEdgeDetector._score_fa_against_deficits()
```

**Schema Contract**:
```python
@dataclass
class EnrichedPlayer:
    name: str
    positions: list[str]
    cat_scores: dict[str, float]  # REQUIRED for MCMC
    z_score: float
    tier: int
    adp: float
    owned_pct: float
    percent_owned: float  # Alias for compatibility
```

### Phase 3: Greenfield Category Implementation (Priority 2)

**Target Categories**: W, L, HR_P, NSV

**Data Sources**:
| Category | Source | Field |
|----------|--------|-------|
| W | MLB Stats API | `stats.pitching.wins` |
| L | MLB Stats API | `stats.pitching.losses` |
| HR_P | MLB Stats API | `stats.pitching.homeRunsAllowed` |
| NSV | MLB Stats API | `stats.pitching.saves - blownSaves` |

**Integration Point**: `backend/services/balldontlie.py` expansion

### Phase 4: Schedule Lookup Fix (Priority 2)

**Current**: `has_game: false` for all players

**Required**: MLB Stats API Schedule integration

**Implementation**:
```python
def fetch_player_schedule(player_id: str, team: str, date_range: tuple) -> dict:
    """Return {has_game: bool, games_count: int, opponent: str}"""
    # Query MLB Stats API /schedule endpoint
    # Filter by team and date range
    # Parse probables for pitchers
```

### Phase 5: Category-Aware Lineup Optimization (Priority 3)

**Current**: Greedy with simple score formula

**Proposed**: OR-Tools Constraint Solver with category deficits

**Objective Function**:
```python
maximize: sum(category_deficit_weights[cat] * player_cat_scores[player][cat])
           for player in lineup
           for cat in categories
```

### Phase 6: Projection Refresh Pipeline (Priority 3)

**Current**: 200+ hardcoded players in `player_board.py`

**Proposed**:
```
FanGraphs RoS API → projections_loader → CSV cache → player_board
```

**Freshness SLA**: 12 hours for ensemble blend

---

## Technical Debt Inventory

| ID | Description | File | Impact |
|----|-------------|------|--------|
| TD-001 | Hardcoded _DEFAULT_SEASON_DAYS = 100 | row_projector.py | Rate accuracy degrades as season progresses |
| TD-002 | Z-capping at ±3.0 arbitrary | scoring_engine.py | Distorts true outlier impact |
| TD-003 | No backtracking in lineup optimizer | decision_engine.py | Suboptimal lineups |
| TD-004 | Silent fallback to 0.0 for missing stats | Multiple | Hidden data quality issues |
| TD-005 | Race condition in FA cache | waiver_edge_detector.py | Stale data possible |

---

## Testing Recommendations

### Unit Test Coverage (Current)

```
tests/test_waiver_integration.py      565 lines (K-20, K-24, coverage protection)
tests/test_projections_bridge.py      184 lines (Steamer CSV bridge)
tests/test_mcmc_simulator_v2.py       (MCMC correctness)
tests/test_row_simulation_bridge.py   (ROW projection)
```

### Additions Needed

1. **Integration**: `test_waiver_end_to_end.py` - Verify Yahoo → API response with populated cat_scores
2. **Regression**: `test_universal_drop_bug.py` - Ensure distinct drop candidates
3. **Schedule**: `test_schedule_lookup.py` - MLB Stats API integration
4. **Greenfield**: `test_greenfield_categories.py` - W, L, HR_P, NSV population

---

## Appendices

### A. Category Code Mapping

| Canonical | Lowercase | Legacy Keys | LOWER_IS_BETTER |
|-----------|-----------|-------------|-----------------|
| R | r | runs | No |
| H | h | hits | No |
| HR_B | hr_b | hr | No |
| RBI | rbi | rbi | No |
| K_B | k_b | strikeouts_bat | **Yes** |
| TB | tb | total_bases | No |
| NSB | nsb | net_stolen_bases | No |
| AVG | avg | avg | No |
| OPS | ops | ops | No |
| W | w | wins | No |
| L | l | losses | **Yes** |
| HR_P | hr_p | home_runs_allowed | **Yes** |
| K_P | k_p | k_pit, strikeouts_pit | No |
| QS | qs | quality_starts | No |
| ERA | era | era | **Yes** |
| WHIP | whip | whip | **Yes** |
| K_9 | k_9 | k9 | No |
| NSV | nsv | net_saves | No |

### B. Deployment Checklist

- [ ] Run full test suite: `pytest tests/ -q --tb=short`
- [ ] Verify projection freshness gate passes
- [ ] Check Yahoo API credentials valid
- [ ] Test waiver endpoint returns non-zero `owned_pct`
- [ ] Verify distinct drop candidates across positions
- [ ] Railway deploy with health check confirmation

### C. Monitoring Endpoints

| Endpoint | Purpose | Alert Threshold |
|----------|---------|-----------------|
| `/api/fantasy/decisions/status` | Pipeline health | Last update > 24h |
| `/api/dashboard/waiver-targets` | Data quality | All `owned_pct = 0` |
| Health check | Service availability | 5xx rate > 5% |

---

**Report Prepared By**: Claude Code (Opus 4.7)
**Review Required**: Simon Gray
**Next Review**: Post-deployment verification
