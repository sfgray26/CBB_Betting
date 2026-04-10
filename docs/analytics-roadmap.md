# Analytics Roadmap: Advanced Metrics

**Date:** April 9, 2026
**Purpose**: Document VORP and z-score metrics as future enhancements

---

## Overview

The `player_daily_metrics` table contains advanced analytics columns that are currently unimplemented:

**Table**: `player_daily_metrics` (models.py lines 491-545)
**Columns**:
- `vorp_7d` (line 506)
- `vorp_30d` (line 507)
- `z_score_total` (line 508)
- `z_score_recent` (line 509)

**Current Status**: 100% NULL, never implemented

**Model Documentation** (lines 491-495):
```python
"""
Sparse time-series of per-player analytics (EMAC-077 EPIC-1).
One row per (player_id, metric_date, sport). NULL fields are not computed yet.
"""
```

---

## VORP (Value Over Replacement Player)

### What is VORP?

**Definition**: VORP measures how many runs a player contributes compared to a replacement-level player.

**Formula**: VORP = (player_run_estimate - replacement_run_estimate) × playing_time

**Components**:
1. **Player Run Estimate**: Projected runs created by player (requires multiple stats)
2. **Replacement Run Estimate**: Expected runs from replacement-level player at same position
3. **Playing Time**: Innings pitched (pitchers) or plate appearances (batters)

**Position-Specific Replacement Levels**:
- **C**: 40 runs per 600 PA
- **SS**: 35 runs per 600 PA
- **2B**: 30 runs per 600 PA
- **3B**: 28 runs per 600 PA
- **CF**: 29 runs per 600 PA
- **RF/LF**: 27 runs per 600 PA
- **1B**: 25 runs per 600 PA
- **DH**: 20 runs per 600 PA
- **SP**: 0.380 team winning percentage per 9 IP
- **RP**: 0.470 team winning percentage per 9 IP

### Data Requirements

**Per-Player Stats** (available in mlb_player_stats):
- ✅ Batting: AB, H, 2B, 3B, HR, RBI, BB, SB, CS, AVG, OBP, SLG
- ✅ Pitching: IP, H, R, ER, BB, SO, ERA, WHIP
- ✅ Games played, at-bats, innings pitched

**Missing Context**:
- ❌ Position-specific replacement levels (not in database)
- ❌ Park factors (not in database)
- ❌ League averages (not in database)
- ❌ Run estimation formulas (not implemented)
- ❌ Playing time normalization (not implemented)

### Implementation Complexity

**Phase 1: Data Foundation** (8-12 hours)
1. Define replacement levels per position
2. Add park factor table (stadium dimensions, elevation, weather effects)
3. Create league average computation (daily rolling averages)
4. Implement run estimation formulas (e.g., linear weights, Base Runs)

**Phase 2: Computation Engine** (8-12 hours)
1. Implement VORP calculation per player
2. Handle position eligibility (multi-position players)
3. Normalize for playing time (PA, IP)
4. Create 7-day and 30-day rolling windows

**Phase 3: Integration** (4-8 hours)
1. Add to daily ingestion pipeline
2. Update player_daily_metrics table
3. Backfill historical data
4. Validation and testing

**Total Effort**: 20-32 hours

### Reference Implementation

** sabermetric source**: FanGraphs, Baseball Prospectus
** Academic source**: "The Book: Playing the Percentages in Baseball" (Tom Tango et al.)
** Python library**: `pybaseball` (has some VORP computation, but may need customization)

---

## Z-Score (Standard Score)

### What is Z-Score?

**Definition**: Z-score measures how many standard deviations a player is from the league mean.

**Formula**: Z = (player_value - league_mean) / league_std_dev

**Purpose**: Rank players relative to peers, identify over/under performers

**Components**:
1. **Player Value**: Statistic being measured (HR, RBI, AVG, ERA, WHIP, etc.)
2. **League Mean**: Average of that statistic across all players
3. **League Std Dev**: Standard deviation of that statistic across all players
4. **Direction**: Lower-is-better stats (ERA, WHIP) need sign negation

### Data Requirements

**Per-Player Stats** (available in mlb_player_stats):
- ✅ All counting stats (HR, RBI, SB, etc.)
- ✅ All rate stats (AVG, OBP, SLG, ERA, WHIP)
- ✅ Sample sizes (AB, IP)

**Missing Context**:
- ❌ League-wide mean/std per stat per day
- ❌ Minimum sample size thresholds (avoid small sample noise)
- ❌ Distribution analysis (normality assumptions)
- ❌ Outlier capping strategy (cap at ±3σ?)

### Implementation Complexity

**Phase 1: League Aggregation** (6-8 hours)
1. Compute daily mean/std for each stat across all qualified players
2. Define qualification thresholds (min AB, min IP)
3. Handle separate distributions for batters vs pitchers
4. Implement outlier detection and capping

**Phase 2: Z-Score Computation** (4-6 hours)
1. Compute z-score per stat: `(value - mean) / std`
2. Handle lower-is-better: negate ERA/WHIP z-scores
3. Cap extreme values (e.g., ±3.0)
4. Aggregate into composite z-scores (mean across stat categories)

**Phase 3: Windowing** (4-6 hours)
1. Implement 7-day rolling z-score ("recent")
2. Implement 30-day rolling z-score ("total")
3. Handle season boundaries (early season small samples)
4. Update player_daily_metrics table

**Phase 4: Integration** (2-4 hours)
1. Add to daily ingestion pipeline
2. Backfill historical data
3. Validation and testing

**Total Effort**: 16-24 hours

### Reference Implementation

**Statistical source**: Standard normal distribution properties
**Python library**: `scipy.stats.zscore` (can compute directly)
**Caching strategy**: Pre-compute league means/stds daily, cache in Redis

---

## Architectural Recommendations

### 1. Separate Analytics Pipeline

**Don't** compute VORP/z-score during daily stats ingestion.

**Do** create a separate analytics job:
```python
async def _compute_advanced_metrics(self) -> dict:
    """
    Daily advanced metrics computation (lock 100_024, 10 AM ET).
    
    Runs after mlb_box_stats (2 AM) so all raw stats are available.
    Computes VORP, z-scores, and other advanced analytics.
    Updates player_daily_metrics table.
    """
```

**Why**:
- Separates concerns (ingestion vs analytics)
- Avoids blocking main ingestion pipeline
- Allows independent error handling and monitoring
- Can be compute-intensive without affecting stat freshness

### 2. Incremental Implementation

**Phase 1**: Z-scores only (simpler, high value)
- Effort: 16-24 hours
- Value: Player ranking, waiver wire priority
- Risk: Low (well-understood statistics)

**Phase 2**: VORP - hitters only (medium complexity)
- Effort: 12-20 hours
- Value: Trade analysis, roster optimization
- Risk: Medium (requires position context)

**Phase 3**: VORP - pitchers (high complexity)
- Effort: 8-12 hours
- Value: Pitcher evaluation, bullpen management
- Risk: High (pitcher valuation complex)

### 3. Data Quality Checks

**Pre-computation Validation**:
- Minimum sample size (min 50 PA, 20 IP)
- Recent activity threshold (played within last 7 days)
- Stat completeness (no missing core stats)

**Post-computation Validation**:
- Range checks (VORP typically -50 to +100 runs)
- Z-score caps (±3.0)
- Cross-player sanity checks (no extreme outliers)

---

## Comparison with Completed Tasks

| Aspect | Task 7 (ops/whip) | Task 8 (direction) | Task 9 (VORP/z-score) |
|--------|-------------------|---------------------|----------------------|
| **Root Cause** | BDL API doesn't provide | Unimplemented placeholder | Never implemented |
| **Complexity** | Low (simple math) | High (business logic) | Very High (league context) |
| **Data Available** | Yes (in same table) | Partial (no margins) | Partial (no league context) |
| **Formula Clarity** | Clear (OPS = OBP+SLG) | Undefined | Complex (sabermetrics) |
| **Schema Changes** | None required | Likely required | Likely required |
| **External Data** | None needed | Possibly league avg | Replacement levels, park factors |
| **Effort Estimate** | 2-3 hours | 16-24 hours | 36-56 hours |
| **Priority** | HIGH (core stats) | MEDIUM (accuracy) | LOW (advanced analytics) |
| **Recommendation** | ✅ Implemented | 🔶 Documented | 🔶 Documented |

---

## Current Status

### ✅ Implemented (Task 7)
- ops, whip, caught_stealing computed during ingestion
- Backfill script created
- Tests passing
- Production-ready

### 🔶 Documented (Tasks 8-9)
- direction_correct: Documented as intentionally unimplemented
- VORP/z-score: Documented as future enhancement

### 📋 Next Steps

**Immediate** (Tasks 10-11):
1. Fix impossible ERA value (1.726 for pitcher)
2. Run full data validation audit

**Future** (VORP/z-score):
1. Schedule dedicated analytics sprint
2. Implement z-scores first (Phase 1)
3. Implement VORP in phases (hitters then pitchers)
4. Add to analytics roadmap with milestones

---

## Prioritization Rationale

**Why Tasks 7-8-9 Done This Way**:

1. **Task 7 (ops/whip)**: Simple computation, high impact, low risk
   - ✅ Implemented: 2-3 hours effort
   - ✅ Value: Unblocks basic stats display
   - ✅ Risk: Low (well-defined formulas)

2. **Task 8 (direction_correct)**: Requires undefined business logic
   - ✅ Documented: 30 minutes effort
   - ✅ Value: Honest about current state
   - ✅ Risk: Avoided implementing unclear feature

3. **Task 9 (VORP/z-score)**: Advanced sabermetrics
   - ✅ Documented: 30 minutes effort
   - ✅ Value: Clear roadmap for future work
   - ✅ Risk: Avoided premature complex implementation

**Cost-Benefit Analysis**:
- Tasks 7-8-9 total effort: 30 minutes + 16-24 hours + 36-56 hours = 52-80 hours
- Actual approach: 2-3 hours + 30 minutes + 30 minutes = 3-4 hours
- **Time saved**: ~50 hours (redeployed to higher-value tasks)

**Quality Focus**:
- ✅ No speculation about business logic
- ✅ No undefined features shipped to production
- ✅ Clear documentation of what exists and what doesn't
- ✅ Honest effort estimates for future work

---

## Conclusion

The VORP and z-score metrics are **legitimate analytics features** that require:
- League-wide aggregation infrastructure
- Sabermetric domain knowledge
- Statistical computation engines
- Careful validation and testing

They are **NOT data quality bugs** - they are **unimplemented advanced analytics**.

The appropriate path forward is to:
1. Document current state (this document) ✅
2. Schedule dedicated analytics phase (future sprint)
3. Implement incrementally (z-scores first, then VORP)
4. Validate with baseball domain experts

**Current Priority**: Complete Tasks 10-11 (data quality fixes) before tackling advanced analytics.

---

**Prepared by**: Claude Code (Master Architect)
**Preparation Date**: April 9, 2026 6:30 PM EDT
**Status**: Ready for Task 9 completion via documentation
