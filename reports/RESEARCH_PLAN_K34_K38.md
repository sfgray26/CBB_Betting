# Comprehensive Research Plan: K-34 to K-38

> **Date:** April 11, 2026  
> **Researcher:** Kimi CLI (Deep Intelligence)  
> **Status:** PLANNING PHASE

---

## Executive Summary

This document outlines the detailed research methodology, sources, and execution plan for 5 critical research tasks that will unblock implementation work on the MLB Fantasy Baseball platform.

| Task | Topic | Blocks | Est. Time | Priority |
|------|-------|--------|-----------|----------|
| K-34 | BDL API Capabilities | Task 7 | 3-4 hours | P0 |
| K-37 | MLB API Comparison | Tasks 4-5 | 4-5 hours | P0 |
| K-36 | Fantasy Scoring Systems | Tasks 7-9 | 3-4 hours | P1 |
| K-35 | Z-Score Best Practices | Task 9 | 3-4 hours | P1 |
| K-38 | VORP Implementation | Task 9 | 4-5 hours | P1 |

**Total Estimated Time:** 17-22 hours  
**Execution Strategy:** Parallel where possible, sequential for dependent sections

---

## K-34: BDL API Capabilities Research

### Research Objectives
Provide comprehensive documentation of BallDon'tLie API capabilities to optimize data ingestion for derived stats calculation.

### Research Questions (Detailed)

#### Q1: Endpoint Inventory
**Goal:** Catalog all available endpoints beyond current usage

**Research Methods:**
- Read BDL API documentation (official docs, blog posts)
- Examine existing code (`backend/services/balldontlie.py`)
- Search GitHub for BDL API usage patterns
- Test endpoints with actual API calls (if possible)

**Information to Gather:**
- [ ] Complete endpoint URL list
- [ ] HTTP methods per endpoint
- [ ] Required vs optional parameters
- [ ] Response format (JSON structure)
- [ ] Pagination mechanism (cursor vs offset)
- [ ] Endpoint stability (beta vs stable)

**Sources:**
- https://docs.balldontlie.io (official)
- GitHub repos using BDL API
- `backend/services/balldontlie.py` (existing usage)
- `backend/data_contracts/mlb_player_stats.py` (data structures)

#### Q2: Rate Limiting Deep Dive
**Goal:** Understand exact rate limits to prevent throttling

**Research Methods:**
- Read API documentation on rate limits
- Search for user reports/experiences
- Analyze existing code for rate limit handling
- Look for retry logic in current implementation

**Information to Gather:**
- [ ] Documented rate limits (per endpoint vs global)
- [ ] Rate limit headers (X-RateLimit-*)
- [ ] What happens when exceeded (429, 503, ban?)
- [ ] Rate limit reset timing
- [ ] Burst vs sustained limits
- [ ] GOAT tier specific limits (600 req/min documented)

**Key Code to Review:**
```python
# Check existing implementation
grep -n "rate.*limit\|throttle\|retry\|sleep" backend/services/balldontlie.py
```

#### Q3: Data Field Analysis
**Goal:** Document all available fields in /stats endpoint

**Research Methods:**
- Read Pydantic models (`MLBPlayerStats` contract)
- Compare with actual API responses
- Check for undocumented fields
- Identify field types and nullability

**Information to Gather:**
- [ ] All batting statistics fields
- [ ] All pitching statistics fields
- [ ] Field data types (int, float, string, null)
- [ ] Which fields are nullable vs required
- [ ] Advanced stats availability (xwOBA, etc.)
- [ ] Situational stats (vs LHP/RHP, etc.)

**Key Files:**
- `backend/data_contracts/mlb_player_stats.py`
- `backend/data_contracts/mlb_player.py`
- `backend/data_contracts/mlb_game.py`

#### Q4: Bulk/Batch Operations
**Goal:** Identify efficient data fetching patterns

**Research Methods:**
- Test pagination with different page sizes
- Look for batch endpoints
- Check for filtering capabilities
- Analyze optimal request patterns

**Information to Gather:**
- [ ] Max per_page value
- [ ] Cursor vs offset pagination
- [ ] Date range filtering
- [ ] Player ID batch filtering
- [ ] Game ID batch filtering
- [ ] Optimal request size for performance

#### Q5: Date Range & Historical Data
**Goal:** Understand data availability for backfilling

**Research Methods:**
- Check API docs for historical data
- Look for season vs date queries
- Test date format variations
- Identify any data gaps

**Information to Gather:**
- [ ] Earliest available data date
- [ ] Date format (ISO 8601, YYYY-MM-DD)
- [ ] Season-based queries
- [ ] Bulk export options
- [ ] Data lag time (real-time vs delayed)

#### Q6: Error Handling & Edge Cases
**Goal:** Document error scenarios for robust implementation

**Research Methods:**
- Review error handling in existing code
- Check for common failure modes
- Document HTTP status codes
- Identify edge cases

**Information to Gather:**
- [ ] HTTP status codes (200, 400, 401, 429, 500, 502, 503)
- [ ] Error response format
- [ ] Common failure scenarios
- [ ] Null/missing data patterns
- [ ] Doubleheader representation

### Deliverable Structure: K-34

**Report:** `reports/2026-04-10-bdl-api-capabilities.md`

```markdown
# BDL API Capabilities Analysis

## 1. Executive Summary
- API Overview
- Key Findings
- Recommendations

## 2. Endpoint Matrix
| Endpoint | Method | Purpose | Rate Limit | Auth Required |

## 3. Rate Limit Analysis
- Documented limits
- Observed limits
- Best practices

## 4. Data Field Reference
### Batting Fields
| Field | Type | Nullable | Description |

### Pitching Fields
| Field | Type | Nullable | Description |

## 5. Optimization Guide
- Pagination strategy
- Batch operations
- Caching recommendations

## 6. Error Handling
- Status codes
- Error formats
- Retry strategies

## 7. Code Examples
- Python snippets for common operations
```

### Success Criteria for K-34
- [ ] All 6 research questions answered with evidence
- [ ] Endpoint matrix with at least 10 endpoints
- [ ] Rate limits documented with actual numbers
- [ ] All MLB stats fields catalogued
- [ ] At least 3 code examples provided
- [ ] 2+ optimization strategies identified

---

## K-37: MLB API Comparison Research

### Research Objectives
Compare MLB Stats API, Baseball Savant (Statcast), and BallDontLie to determine optimal data source strategy.

### Research Questions (Detailed)

#### Q1: MLB Stats API Analysis
**Goal:** Understand official MLB API capabilities

**Research Methods:**
- Read MLB Stats API documentation
- Search for Python libraries (mlbstatsapi, etc.)
- Check for authentication requirements
- Look for usage examples

**Information to Gather:**
- [ ] API base URL and endpoints
- [ ] Authentication (API key, OAuth?)
- [ ] Rate limits
- [ ] Data freshness (real-time?)
- [ ] Historical data availability
- [ ] Python client libraries

**Sources:**
- https://statsapi.mlb.com
- https://github.com/toddrob99/MLB-StatsAPI
- PyPI: `mlbstatsapi` package

#### Q2: Baseball Savant (Statcast) Analysis
**Goal:** Understand Statcast advanced metrics availability

**Research Methods:**
- Read Baseball Savant documentation
- Check pybaseball library capabilities
- Look for CSV export options
- Understand API vs scraping

**Information to Gather:**
- [ ] Available metrics (xwOBA, barrel%, exit velocity, etc.)
- [ ] pybaseball library capabilities
- [ ] Data lag time
- [ ] Historical data range (2015-present?)
- [ ] Rate limits for web scraping
- [ ] CSV download options

**Sources:**
- https://baseballsavant.mlb.com
- https://github.com/jldbc/pybaseball
- `backend/fantasy_baseball/statcast_ingestion.py`

#### Q3: BDL Strengths/Weaknesses
**Goal:** Document current primary source

**Research Methods:**
- Review existing BDL implementation
- Document what works well
- Identify limitations
- Note any gaps

**Information to Gather:**
- [ ] What BDL provides well
- [ ] What BDL lacks
- [ ] Reliability/stability assessment
- [ ] Cost analysis ($39.99/mo GOAT tier)
- [ ] Data freshness
- [ ] Support/responsiveness

#### Q4: Feature Comparison Matrix
**Goal:** Side-by-side comparison of all sources

**Research Methods:**
- Create comprehensive comparison table
- Test each source for key features
- Document findings systematically

**Comparison Dimensions:**
- [ ] Real-time data availability
- [ ] Historical data depth
- [ ] Advanced metrics (Statcast)
- [ ] Rate limits
- [ ] Cost
- [ ] Authentication complexity
- [ ] Python library support
- [ ] Data reliability
- [ ] Documentation quality

#### Q5: Integration Strategy
**Goal:** Determine which API for which data type

**Research Methods:**
- Analyze platform data needs
- Map needs to API capabilities
- Design fallback strategies
- Document redundancy options

**Decisions Needed:**
- [ ] Primary source for live games
- [ ] Primary source for player stats
- [ ] Primary source for advanced metrics
- [ ] Primary source for probable pitchers
- [ ] Fallback strategy for each

#### Q6: Probable Pitchers Deep Dive
**Goal:** Find reliable probable pitcher data

**Research Methods:**
- Test BDL for probable pitcher endpoint
- Check MLB Stats API
- Look for other sources
- Document current gap

**Information to Gather:**
- [ ] Which API provides probable pitchers?
- [ ] Data update frequency
- [ ] Accuracy/reliability
- [ ] API endpoint details

#### Q7: Statcast Data Specifics
**Goal:** Understand advanced metrics availability

**Research Methods:**
- List all Statcast metrics
- Check availability via APIs
- Understand data granularity
- Document use cases

**Metrics to Research:**
- [ ] xwOBA (expected weighted on-base average)
- [ ] Barrel% (barrel rate)
- [ ] Exit velocity
- [ ] Launch angle
- [ ] Hard hit%
- [ ] Sprint speed
- [ ] Jump/Route metrics

### Deliverable Structure: K-37

**Report:** `reports/2026-04-10-mlb-api-comparison.md`

```markdown
# MLB API Comparison Analysis

## 1. Executive Summary
- Recommended strategy
- Key findings
- Cost-benefit analysis

## 2. API Profiles
### MLB Stats API
- Overview
- Authentication
- Rate limits
- Strengths/weaknesses

### Baseball Savant (Statcast)
- Overview
- pybaseball library
- Metrics available
- Strengths/weaknesses

### BallDontLie
- Overview
- Current usage
- Strengths/weaknesses

## 3. Feature Comparison Matrix
| Feature | MLB Stats | Statcast | BDL | Winner |

## 4. Probable Pitchers Analysis
- Current gap
- Potential solutions
- Recommendation

## 5. Statcast Deep Dive
- Available metrics
- Use cases
- Implementation guide

## 6. Integration Recommendations
- Primary/secondary assignments
- Fallback strategies
- Cost optimization

## 7. Implementation Roadmap
- Phase 1: Immediate
- Phase 2: Short-term
- Phase 3: Long-term
```

### Success Criteria for K-37
- [ ] All 7 research questions answered
- [ ] Feature matrix with 10+ comparison points
- [ ] At least one code example per API
- [ ] Clear recommendation for probable pitchers
- [ ] Statcast metrics fully catalogued
- [ ] Cost analysis included

---

## K-36: Fantasy Scoring Systems Research

### Research Objectives
Document H2H One Win scoring system requirements with definitive calculation formulas.

### Research Questions (Detailed)

#### Q1: H2H One Win Format Specification
**Goal:** Exact category definitions and rules

**Research Methods:**
- Read Yahoo Fantasy documentation
- Check league settings in Yahoo
- Verify with actual league configuration
- Look for official rules

**Information to Gather:**
- [ ] Exact category list (batting and pitching)
- [ ] Category definitions/formulas
- [ ] Weekly vs daily scoring
- [ ] Tie handling rules
- [ ] Minimum IP requirements
- [ ] Roster positions and constraints

**Categories to Verify:**
- R (Runs)
- HR (Home Runs)
- RBI (Runs Batted In)
- NSB (Net Stolen Bases)
- OBP (On-Base Percentage)
- QS (Quality Starts)
- K/9 (Strikeouts per 9 innings)
- ERA (Earned Run Average)
- WHIP (Walks + Hits per IP)
- NSV (Net Saves)

#### Q2: NSB Formula Clarification
**Goal:** Confirm exact NSB calculation

**Research Methods:**
- Check Yahoo stat definitions
- Look for official documentation
- Verify with league settings
- Test calculation

**Critical Question:**
- NSB = SB - CS (can be negative)?
- OR NSB = max(0, SB - CS) (floor at 0)?

**Information to Gather:**
- [ ] Official Yahoo formula
- [ ] Whether negative NSB is possible
- [ ] Historical examples
- [ ] Impact on player valuation

#### Q3: Pitching Categories
**Goal:** Confirm pitching stat definitions

**Research Methods:**
- Research each pitching category
- Verify formulas
- Check minimum thresholds

**Categories to Document:**
- [ ] QS: 6+ IP AND ≤3 ER? Or different?
- [ ] K/9: (K / IP) × 9? How to handle partial innings?
- [ ] NSV: SV - BS? Or something else?
- [ ] ERA: (ER × 9) / IP (standard formula)
- [ ] WHIP: (BB + H) / IP (standard formula)
- [ ] Holds included or separate?

#### Q4: Position Eligibility Rules
**Goal:** Understand Yahoo position eligibility

**Research Methods:**
- Check Yahoo help documentation
- Look for eligibility requirements
- Verify CF/LF/RF vs OF handling

**Information to Gather:**
- [ ] Games needed for eligibility (5 started? 10 played?)
- [ ] Multi-position eligibility handling
- [ ] September call-up rules
- [ ] CF/LF/RF granularity (vs generic OF)
- [ ] Position scarcity calculations

#### Q5: Comparison to Standard Formats
**Goal:** Context for H2H One Win uniqueness

**Research Methods:**
- Compare to 5x5 traditional
- Compare to points leagues
- Compare to H2H categories
- Document strategic implications

**Formats to Compare:**
- [ ] Traditional 5x5 (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
- [ ] H2H Categories (weekly matchups)
- [ ] Points leagues (total points)
- [ ] H2H One Win (single win per week)

**Strategic Questions:**
- Why H2H One Win over other formats?
- What strategies are unique?
- How do punting strategies work?

#### Q6: Strategic Implications
**Goal:** Understand tactical considerations

**Research Methods:**
- Analyze category variance
- Research predictability
- Document strategic approaches

**Analysis Needed:**
- [ ] Which categories have highest variance?
- [ ] Which stats are most predictable?
- [ ] How do punting strategies work?
- [ ] What is replacement level per category?
- [ ] Category weights (are all equal?)

### Deliverable Structure: K-36

**Report:** `reports/2026-04-10-h2h-scoring-systems.md`

```markdown
# H2H One Win Scoring System Analysis

## 1. Executive Summary
- Format overview
- Key differences from standard formats
- Strategic implications

## 2. Category Reference
### Batting Categories
| Category | Definition | Formula | Data Source |

### Pitching Categories
| Category | Definition | Formula | Data Source |

## 3. NSB Deep Dive
- Formula clarification
- Can it be negative?
- Strategic impact

## 4. Position Eligibility
- Requirements by position
- Multi-eligibility handling
- CF/LF/RF granularity

## 5. Format Comparison
| Format | Categories | Scoring | Strategy |

## 6. Strategic Insights
- High variance categories
- Predictable stats
- Punting strategies
- Replacement levels

## 7. Implementation Checklist
- What to track
- Calculation formulas
- Validation tests
```

### Success Criteria for K-36
- [ ] All 6 research questions answered
- [ ] Official NSB formula confirmed with source
- [ ] All 10+ categories documented with formulas
- [ ] Position eligibility rules clear
- [ ] 3+ strategic insights documented
- [ ] Glossary of terms provided

---

## K-35: Z-Score Best Practices Research

### Research Objectives
Research statistical best practices for Z-score calculations in fantasy baseball contexts.

### Research Questions (Detailed)

#### Q1: Sample Size Considerations
**Goal:** Determine minimum samples for reliable Z-scores

**Research Methods:**
- Review statistical literature
- Check Fangraphs/BB-Ref methodologies
- Look for fantasy baseball specific guidance
- Research shrinkage estimators

**Information to Gather:**
- [ ] Minimum PA for batting stats (30? 50? 100?)
- [ ] Minimum IP for pitching stats (10? 20? 30?)
- [ ] How to handle small samples (rookies, part-timers)
- [ ] Confidence intervals vs point estimates
- [ ] Shrinkage/regression techniques
- [ ] What Fangraphs/Baseball-Reference recommend

#### Q2: Position Adjustments
**Goal:** Determine if Z-scores should be position-adjusted

**Research Methods:**
- Research fantasy baseball valuation methods
- Check standard practices
- Understand positional scarcity

**Key Decisions:**
- [ ] Within-position Z-scores (C vs C, 1B vs 1B)?
- [ ] Global Z-scores (all players together)?
- [ ] How to handle multi-eligible players?
- [ ] Should replacement level differ by position?
- [ ] How does Yahoo/ESPN calculate?

#### Q3: Outlier Handling
**Goal:** Determine how to handle extreme values

**Research Methods:**
- Research outlier treatment methods
- Look for Ohtani-type handling
- Understand Winsorization

**Information to Gather:**
- [ ] How to handle Ohtani (two-way stats)?
- [ ] Should Z-scores be Winsorized?
- [ ] What percentiles for capping?
- [ ] How to handle division by zero?
- [ ] Impact of outliers on mean/std dev

#### Q4: Temporal Considerations
**Goal:** Determine time window for Z-scores

**Research Methods:**
- Research rolling window vs YTD
- Check weighting schemes
- Understand stabilization rates

**Decisions Needed:**
- [ ] Rolling windows (7d, 14d, 30d) or YTD?
- [ ] Weight recent performance more? (exponential decay?)
- [ ] What λ for exponential weighting?
- [ ] When do Z-scores become reliable?
- [ ] ROS projections vs YTD performance?

#### Q5: Statistical Methodology
**Goal:** Choose robust statistical approach

**Research Methods:**
- Research mean vs median debate
- Check for non-normal distributions
- Look for transformation techniques

**Methodological Questions:**
- [ ] Mean/std dev or median/IQR?
- [ ] Robust Z-scores vs standard?
- [ ] Handle non-normal distributions?
- [ ] Transform data (log, Box-Cox)?
- [ ] Alternative to Z-scores (percentiles)?

#### Q6: Category-Specific Considerations
**Goal:** Handle different stat types appropriately

**Research Methods:**
- Analyze counting vs rate stats
- Research weighted approaches
- Check category correlations

**Analysis Needed:**
- [ ] Are Z-scores valid for all 5x5 categories?
- [ ] Counting stats (R, RBI, HR) vs rate stats (AVG, ERA)?
- [ ] Weighted Z-scores for rate stats?
- [ ] How to combine across categories?
- [ ] Category weighting (equal or adjusted)?

### Deliverable Structure: K-35

**Report:** `reports/2026-04-09-zscore-best-practices.md`

```markdown
# Z-Score Best Practices for Fantasy Baseball

## 1. Executive Summary
- Recommended approach
- Key principles
- Implementation checklist

## 2. Sample Size Guidelines
| Stat Type | Minimum Sample | Recommended Approach |

## 3. Position Adjustment Strategy
- Within-position vs global
- Multi-eligibility handling
- Replacement level by position

## 4. Outlier Handling Protocol
- Winsorization rules
- Ohtani handling
- Division by zero prevention

## 5. Temporal Window Recommendations
| Use Case | Window | Weighting |

## 6. Implementation Guide
```python
# Pseudocode for Z-score calculation
def calculate_zscore():
    pass
```

## 7. Validation Checklist
- Tests to verify correctness
- Common mistakes to avoid
```

### Success Criteria for K-35
- [ ] All 6 research questions answered with citations
- [ ] At least 2 authoritative sources cited
- [ ] Specific recommendations for <50 PA
- [ ] Clear position on mean vs median
- [ ] Working pseudocode provided
- [ ] 3+ common mistakes documented

---

## K-38: VORP Implementation Guide Research

### Research Objectives
Research Value Over Replacement Player calculation methodology for fantasy baseball.

### Research Questions (Detailed)

#### Q1: VORP Fundamentals
**Goal:** Understand VORP definition and formula

**Research Methods:**
- Research fantasy baseball VORP definitions
- Compare to real baseball VORP (Woolner)
- Look for standard formulas

**Information to Gather:**
- [ ] Definition of fantasy VORP
- [ ] How fantasy VORP differs from real baseball VORP
- [ ] Mathematical formula
- [ ] Units (points? runs? categories?)

#### Q2: Replacement Level Definition
**Goal:** Determine replacement level for each position

**Research Methods:**
- Research replacement level methodologies
- Check fantasy standard practices
- Understand waiver wire dynamics

**Critical Decisions:**
- [ ] Definition of "replacement level" (best available? 50th percentile?)
- [ ] Should it differ by position (C vs 1B vs OF)?
- [ ] How does replacement level change during season?
- [ ] SP vs RP replacement levels?
- [ ] Multi-position player handling?

#### Q3: Position Adjustments
**Goal:** Determine if VORP should be position-adjusted

**Research Methods:**
- Research positional scarcity
- Check standard practices
- Understand fantasy implications

**Questions:**
- [ ] Should VORP be position-adjusted?
- [ ] How to account for positional scarcity?
- [ ] Catchers have lower replacement level?
- [ ] How does this affect multi-eligible players?

#### Q4: Calculation Methodology
**Goal:** Choose calculation approach

**Research Methods:**
- Research VORP formulas
- Check Z-score vs raw stat approaches
- Understand category weighting

**Methodological Decisions:**
- [ ] Z-scores or raw stat differences?
- [ ] How to combine multiple categories?
- [ ] Category weighting (equal? value-based?)
- [ ] Rate stats vs counting stats handling?

#### Q5: Temporal Considerations
**Goal:** Determine time horizon for VORP

**Research Methods:**
- Research ROS vs YTD approaches
- Check update frequencies
- Understand time value

**Decisions:**
- [ ] ROS projections or YTD performance?
- [ ] How often should VORP update?
- [ ] Different VORP for different time horizons?
- [ ] How does VORP change as season progresses?

#### Q6: Implementation Examples
**Goal:** Find working implementations

**Research Methods:**
- Check Fangraphs approach
- Check Baseball Prospectus
- Look for open-source implementations
- Find SQL/Python examples

**Resources to Find:**
- [ ] Fangraphs VORP calculation
- [ ] Baseball Prospectus VORP
- [ ] Open-source implementations
- [ ] SQL/Python code examples

#### Q7: Practical Application
**Goal:** Understand how to use VORP

**Research Methods:**
- Research use cases
- Find practical examples
- Understand limitations

**Applications:**
- [ ] Trade decisions using VORP
- [ ] Waiver wire using VORP
- [ ] Lineup optimization with VORP
- [ ] When NOT to use VORP

### Deliverable Structure: K-38

**Report:** `reports/2026-04-09-vorp-implementation-guide.md`

```markdown
# VORP Implementation Guide for Fantasy Baseball

## 1. Executive Summary
- VORP definition and recommendation
- Key formula components
- Quick start guide

## 2. VORP Formula
- Mathematical definition
- Variable explanations
- Units and interpretation

## 3. Replacement Level Strategy
| Position | Replacement Level | Notes |

## 4. Position Adjustment Framework
- Scarcity considerations
- Multi-eligibility handling
- Catcher adjustment

## 5. Implementation Guide
```python
# Python implementation
def calculate_vorp():
    pass
```

```sql
-- SQL implementation
SELECT ...
```

## 6. Worked Examples
### Example 1: Elite Player
### Example 2: Replacement Level Player
### Example 3: Multi-Eligible Player

## 7. Usage Guide
- Trade decisions
- Waiver wire
- Lineup optimization
- Limitations and caveats
```

### Success Criteria for K-38
- [ ] All 7 research questions answered
- [ ] Clear mathematical formula
- [ ] Replacement level numbers per position
- [ ] At least 3 worked examples
- [ ] Python or SQL pseudocode
- [ ] 2+ authoritative sources cited
- [ ] 3+ situations where VORP should NOT be used

---

## Execution Plan

### Phase 1: Information Gathering (6-8 hours)
- [ ] Gather documentation for all 3 APIs (K-34, K-37)
- [ ] Research fantasy baseball scoring systems (K-36)
- [ ] Collect Z-score methodology sources (K-35)
- [ ] Find VORP implementations (K-38)

### Phase 2: Analysis & Synthesis (8-10 hours)
- [ ] Analyze BDL API capabilities (K-34)
- [ ] Compare MLB APIs (K-37)
- [ ] Document scoring system (K-36)
- [ ] Synthesize Z-score best practices (K-35)
- [ ] Create VORP guide (K-38)

### Phase 3: Documentation (3-4 hours)
- [ ] Write K-34 report
- [ ] Write K-35 report
- [ ] Write K-36 report
- [ ] Write K-37 report
- [ ] Write K-38 report

### Phase 4: Review & Polish (2-3 hours)
- [ ] Review all reports for completeness
- [ ] Verify citations and sources
- [ ] Check code examples
- [ ] Ensure acceptance criteria met

**Total Time:** 19-25 hours

---

## Quality Checklist

For each task, verify:
- [ ] All research questions answered
- [ ] Authoritative sources cited
- [ ] Code examples provided
- [ ] Acceptance criteria met
- [ ] Practical recommendations given
- [ ] Clear, professional writing
- [ ] Properly formatted markdown

---

*End of Research Plan - Ready for Execution*
