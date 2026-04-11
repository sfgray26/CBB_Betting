# Tasks 10-11 Data Quality Fixes Preparation

**Date:** April 10, 2026
**Purpose**: Deep analysis and quality checklist for Tasks 10-11 (Final Data Quality Fixes)

---

## Task 10: Fix Impossible ERA Value

### Current Understanding

**Table**: `mlb_player_stats` (models.py lines 1015-1086)
**Issue**: One row has ERA > 100 (specifically 1.726, which is actually quite low - need to verify actual value)
**Status**: Identified in validation audit but not yet investigated

**Critical Questions**:
1. What is the actual impossible ERA value?
2. Is it a data entry error or calculation bug?
3. Is the ERA actually 1.726 (which is valid) or 172.6 (impossible)?
4. If it's 1.726, why is it flagged as impossible?

### Investigation Plan

**Step 1: Identify the problematic row**
- Query all ERA values to find the actual maximum
- Check if there are values > 100 (truly impossible) or if 1.726 is the max (which would be valid)

**Step 2: Examine raw stats**
- For the identified row(s), check:
  - earned_runs
  - innings_pitched
  - Calculation: ERA = (earned_runs / innings_pitched) × 9

**Step 3: Determine root cause**
- If ERA = 172.6 (or similar): Calculation error or bad raw data
- If ERA = 1.726: This is valid (excellent ERA), why was it flagged?

**Step 4: Fix strategy**
- If calculation bug: Fix the ERA calculation logic
- If bad raw data: Update or delete the row
- If false positive: Remove from validation rules or adjust threshold

### Quality Checklist for Task 10

**Step 1: Find all extreme ERA values**
- [ ] Query: `SELECT bdl_player_id, era, earned_runs, innings_pitched FROM mlb_player_stats WHERE era IS NOT NULL ORDER BY era DESC LIMIT 20`
- [ ] Expected: Should show the highest ERA values
- [ ] Determine: Are any truly impossible (> 100)?

**Step 2: Verify ERA calculation**
- [ ] For the highest ERA value, manually recalculate:
  ```python
  era_calculated = (earned_runs / innings_pitched) * 9
  ```
- [ ] Compare with stored ERA value
- [ ] Determine if calculation matches

**Step 3: Check data source**
- [ ] If calculation doesn't match, check BDL API response in raw_payload
- [ ] Verify if BDL provided bad ERA or if our calculation is wrong
- [ ] Check if innings_pitched parsing is incorrect (e.g., "6.2" format)

**Step 4: Implement fix**
- [ ] If calculation bug: Fix the ERA calculation in ingestion logic
- [ ] If bad source data: NULL out the incorrect ERA value
- [ ] Update validation rules if threshold is wrong

**Step 5: Backfill/Correct**
- [ ] For all affected rows, recalculate ERA from raw stats
- [ ] Update rows with corrected values
- [ ] Verify no rows have impossible ERA after fix

**Step 6: Add validation test**
- [ ] Create test to ensure ERA values are within valid range (0-100)
- [ ] Add to test suite to prevent future regressions

**Step 7: Verify fix**
- [ ] Query: `SELECT COUNT(*) FROM mlb_player_stats WHERE era < 0 OR era > 100`
- [ ] Expected: 0 rows

### Quality Gates

**Data Accuracy**:
- [ ] All ERA values are within valid range (0-100)
- [ ] ERA calculations are correct: (ER / IP) × 9
- [ ] No impossible values after fix

**Root Cause Identified**:
- [ ] Clear understanding of why impossible ERA occurred
- [ ] Prevention measure in place (test or validation)

---

## Task 11: Run Full Data Validation Audit

### Current Understanding

**Purpose**: Final validation to ensure all data quality issues are resolved
**Baseline**: Original audit had 0 CRITICAL, 131 WARNING, 64 INFO
**Target**: 0 CRITICAL, <50 WARNING (after fixing Tasks 1-10)

### Validation Audit Plan

**Step 1: Re-run the validation audit**
- Use `scripts/db_validation_audit.py` (or create if doesn't exist)
- Generate fresh report with current state

**Step 2: Compare with baseline**
- Compare CRITICAL/WARNING/INFO counts
- Identify which issues were resolved
- Identify any new issues introduced

**Step 3: Verify specific fixes**
- Task 1-2: Player identity resolution (yahoo_key, bdl_player_id populated)
- Task 7: ops/whip/caught_stealing computed
- Task 10: Impossible ERA fixed
- Other improvements from Tasks 1-9

**Step 4: Document final state**
- Create before/after comparison report
- Highlight remaining WARNING items
- Prioritize any remaining issues for future work

### Quality Checklist for Task 11

**Step 1: Run validation audit**
- [ ] Execute: `python scripts/db_validation_audit.py`
- [ ] Capture output to file
- [ ] Parse CRITICAL/WARNING/INFO counts

**Step 2: Analyze results**
- [ ] Compare with original audit (if available)
- [ ] Categorize remaining warnings by severity
- [ ] Identify any new issues introduced during remediation

**Step 3: Verify specific fixes**
- [ ] Check: `SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL` → Should be > 0
- [ ] Check: `SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NOT NULL` → Should be > 0
- [ ] Check: `SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NOT NULL` → Should be > 0
- [ ] Check: `SELECT COUNT(*) FROM mlb_player_stats WHERE era < 0 OR era > 100` → Should be 0

**Step 4: Create final report**
- [ ] Document before/after metrics
- [ ] List all completed fixes
- [ ] List remaining issues (if any)
- [ ] Provide recommendations for next phase

**Step 5: Commit report**
- [ ] Git add/commit final validation report
- [ ] Update HANDOFF.md with completion status

### Quality Gates

**Validation Completeness**:
- [ ] All tables audited
- [ ] All critical issues resolved (0 CRITICAL)
- [ ] Warning count significantly reduced

**Documentation**:
- [ ] Before/after comparison clear
- [ ] All fixes documented
- [ ] Remaining issues prioritized

---

## Kimi Research Delegation Opportunities

Based on the analytics roadmap and current codebase state, here are high-value research tasks Kimi can tackle to accelerate future roadmap items:

### Research Task 1: BDL API Endpoint Capabilities Analysis

**Objective**: Document all BDL API endpoints and their capabilities for MLB data

**Why This Matters**:
- Current codebase uses partial BDL integration
- Full endpoint documentation will accelerate future MLB features
- Reduces trial-and-error in API integration

**Research Questions**:
1. What are ALL available `/mlb/v1/` endpoints in BDL GOAT MLB?
2. Which endpoints provide odds data (to replace OddsAPI for MLB)?
3. Which endpoints provide probable pitcher data?
4. What are the rate limits and best practices?
5. Are there any undocumented endpoints or parameters?

**Deliverable**: `research/bdl-api-capabilities-2026-04-10.md`

**Time Estimate**: 2-3 hours

**Impact**: HIGH — Unblocks multiple MLB features (odds, probable pitchers, advanced stats)

---

### Research Task 2: MLB Stats API vs BDL API Comparison

**Objective**: Compare MLB Stats API and BDL API for data completeness and reliability

**Why This Matters**:
- Current codebase uses both APIs (mlb_analysis.py uses MLB Stats, daily_ingestion.py uses BDL)
- Understanding strengths/weaknesses will inform consolidation strategy
- Identifies which API to use for which data type

**Research Questions**:
1. What data fields are unique to each API?
2. Which API has more reliable uptime?
3. Which API has more accurate data?
4. What are the rate limits for each?
5. Should we consolidate to one API or use both strategically?

**Deliverable**: `research/mlb-api-comparison-2026-04-10.md`

**Time Estimate**: 3-4 hours

**Impact**: HIGH — Informs architecture decisions for MLB data pipeline

---

### Research Task 3: VORP Implementation Research for MLB

**Objective**: Research VORP (Value Over Replacement Player) calculation for MLB

**Why This Matters**:
- Documented in analytics-roadmap.md as 20-32 hour implementation
- Kimi can research sabermetric sources and formulas
- Accelerates future VORP implementation

**Research Questions**:
1. What is the authoritative VORP formula for MLB?
2. Where do we get position-specific replacement levels?
3. What park factors do we need and where to get them?
4. Are there Python libraries that calculate VORP?
5. What data do we currently have vs what's needed for VORP?

**Deliverable**: `research/vorp-implementation-guide-2026-04-10.md`

**Time Estimate**: 4-5 hours

**Impact**: MEDIUM — Advanced analytics feature, not blocking current work

---

### Research Task 4: Z-Score Calculation Best Practices for Baseball

**Objective**: Research z-score calculation for baseball player analytics

**Why This Matters**:
- Documented in analytics-roadmap.md as 16-24 hour implementation
- Kimi can research statistical best practices and league aggregation
- Accelerates future z-score implementation

**Research Questions**:
1. How do we calculate league-wide mean/std for baseball stats?
2. What's the minimum sample size for reliable z-scores?
3. How do we handle lower-is-better stats (ERA, WHIP)?
4. Do we cap z-scores at ±3σ?
5. How often should we recalculate league averages?

**Deliverable**: `research/zscore-implementation-guide-2026-04-10.md`

**Time Estimate**: 2-3 hours

**Impact**: MEDIUM — Advanced analytics feature, not blocking current work

---

### Research Task 5: Fantasy Baseball Scoring Systems Analysis

**Objective**: Research common fantasy baseball scoring systems and their stat requirements

**Why This Matters**:
- Current platform is fantasy-first (priority 1)
- Understanding scoring systems informs which stats to prioritize
- Helps with lineup optimizer and waiver recommendations

**Research Questions**:
1. What are the most common fantasy baseball scoring systems?
2. Which stats are used in each system (rotisserie, head-to-head, points)?
3. How do we calculate fantasy points from raw stats?
4. What are the stat weights for different systems?
5. Which scoring system should we support first?

**Deliverable**: `research/fantasy-scoring-systems-2026-04-10.md`

**Time Estimate**: 3-4 hours

**Impact**: HIGH — Directly impacts fantasy product features

---

## Execution Strategy

### Phase 1: Research (Kimi Delegation)
1. Delegate Research Tasks 1-5 to Kimi CLI
2. Kimi works in parallel on research while we execute Tasks 10-11
3. Research results inform next roadmap phase

### Phase 2: Planning (Tasks 10-11)
1. Create detailed implementation plan for Task 10
2. Create detailed implementation plan for Task 11
3. Review with user before implementation

### Phase 3: Execution (Tasks 10-11)
1. Execute Task 10 with quality-first approach
2. Execute Task 11 with comprehensive validation
3. Update HANDOFF.md with completion status

### Phase 4: Handoff
1. Review all research outputs from Kimi
2. Prioritize next phase based on research findings
3. Create delegation bundles for next agent (if needed)

---

## Quality Preparation Summary

### Readiness Assessment: **HIGH** ✅

I have thoroughly analyzed:
1. ✅ Task 10 requirements (impossible ERA diagnosis and fix)
2. ✅ Task 11 requirements (full validation audit)
3. ✅ Kimi research opportunities for future acceleration
4. ✅ Execution strategy with parallel research

### Key Insights

**Task 10 (Impossible ERA)**: Investigation required
- Need to verify actual ERA value (1.726 seems valid, may be false positive)
- Must examine raw stats to determine root cause
- Could be calculation bug, bad source data, or validation rule error

**Task 11 (Validation Audit)**: Final checkpoint
- Comprehensive validation of all fixes
- Before/after comparison
- Foundation for next development phase

**Kimi Research**: High-value parallel work
- 5 research tasks identified
- Total time: 14-19 hours
- Impact: Unblocks multiple MLB features and advanced analytics

### Preparation for Implementation

When ready to proceed, I will:
1. Delegate research tasks to Kimi CLI
2. Execute Task 10 with thorough investigation
3. Execute Task 11 with comprehensive validation
4. Review Kimi's research outputs
5. Update HANDOFF.md with final status

### No Scope Creep Commitment

I will NOT:
- ❌ Implement VORP/z-score computation (future work)
- ❌ Skip investigation for Task 10 (must find root cause)
- ❌ Rush validation audit for Task 11 (must be thorough)
- ❌ Add features beyond data quality fixes

I WILL:
- ✅ Follow investigation steps exactly
- ✅ Find root cause of impossible ERA
- ✅ Run comprehensive validation audit
- ✅ Delegate research to Kimi for acceleration
- ✅ Document all findings thoroughly
- ✅ Maintain quality-first approach

---

**Prepared by**: Claude Code (Master Architect)
**Preparation Date**: April 10, 2026
**Status**: Ready for Tasks 10-11 execution + Kimi research delegation
