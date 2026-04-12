> **Note:** This is a copy. The canonical version is in the repository root: $(Split-Path System.Collections.Hashtable.Path -Leaf)`n
---

# Kimi Delegation: K-34 to K-38 Research Bundle# Kimi CLI Delegation Bundle — K-34 through K-38

> **Delegation Date:** April 9, 2026 9:20 PM EDT  
> **Delegating Agent:** Claude Code (Master Architect)  
> **Status:** READY FOR KIMI CLI EXECUTION

---

## 🎯 DELEGATION OVERVIEW

**Context:** The MLB Fantasy Baseball platform has critical data quality issues. Before implementing fixes (Tasks 1-11), we need **comprehensive research** on APIs, statistical methodologies, and best practices. These 5 research tasks (K-34 to K-38) will provide the foundation for all subsequent implementation work.

**Priority Order:**
1. K-34: BDL API Capabilities (blocks Task 7)
2. K-36: Fantasy Scoring Systems (blocks Tasks 7-9)
3. K-37: MLB API Comparison (blocks Tasks 4-5)
4. K-35: Z-Score Best Practices (blocks Task 9)
5. K-38: VORP Implementation (blocks Task 9)

**Expected Timeline:** 2-3 hours per task (can run in parallel where dependencies allow)

---

## K-34: BallDon'tLie API Capabilities Research

### Mission
Conduct comprehensive investigation into BallDon'tLie (BDL) MLB API capabilities, endpoints, rate limits, data fields, and optimization strategies to maximize data ingestion efficiency.

### Background
The system currently uses BDL API with GOAT tier (600 req/min) but may not be leveraging all available endpoints or optimizing for rate limits. We need complete API documentation to implement efficient bulk ingestion for Tasks 7 (derived stats) and backfilling.

### Research Questions (Answer All)

#### 1. Endpoint Inventory
- What MLB endpoints exist beyond `/games`, `/odds`, `/player_injuries`, `/players`, `/stats`?
- Are there endpoints for: player season stats, team stats, standings, schedule?
- What is the complete endpoint URL structure and HTTP methods?

#### 2. Rate Limiting Deep Dive
- What are the exact rate limits per endpoint (not just global 600 req/min)?
- Are there different limits for read vs write operations?
- What is the rate limit reset window (per second, minute, hour)?
- What happens when limits are exceeded (429 response, IP ban, tier downgrade)?
- Are there burst allowances or smoothing algorithms?

#### 3. Data Field Analysis
For `/mlb/v1/stats` endpoint specifically:
- What batting fields are available? (we know: ab, r, h, double, triple, hr, rbi, bb, so, sb, cs, avg, obp, slg, ops)
- What pitching fields are available? (we know: ip, h_allowed, r_allowed, er, bb_allowed, k, whip, era)
- Are advanced stats available? (xwOBA, barrel%, exit velocity, hard hit%, launch angle)
- Are situational stats available? (vs LHP/RHP, home/away, clutch, RISP)
- What is the data lag time (real-time, 15-min delay, next-day)?

#### 4. Bulk/Batch Operations
- Are there bulk endpoints that return multiple players/games in one request?
- What is the maximum `per_page` value for pagination?
- What is the optimal pagination strategy (cursor vs offset)?
- Are there batch update endpoints for historical data?

#### 5. Date Range & Historical Data
- How far back does historical data go?
- Are there date range filters for all endpoints?
- What date formats are accepted (ISO 8601, YYYY-MM-DD)?
- Are there season-based queries vs date-based?
- Is there a bulk export feature for historical data?

#### 6. Error Handling & Edge Cases
- What HTTP status codes does BDL return?
- What error response formats should we expect?
- How are missing/null fields represented?
- What happens for players with no stats (call-ups, recent trades)?
- How are doubleheaders represented in the data?

### Deliverable

**File:** `reports/2026-04-09-bdl-api-capabilities.md`

**Required Sections:**
1. **Executive Summary** (1 paragraph)
2. **Endpoint Matrix** (table: endpoint, method, purpose, rate limit, data lag)
3. **Rate Limit Deep Dive** (detailed section with practical examples)
4. **Data Field Reference** (complete field list for /stats endpoint)
5. **Optimization Recommendations** (3-5 specific strategies for our use case)
6. **Code Examples** (Python snippets for optimal pagination, error handling)
7. **Known Limitations** (what BDL cannot provide that we need)

### Acceptance Criteria
- [ ] All 6 research questions answered with evidence
- [ ] At least one working Python code example for pagination
- [ ] Rate limit table with actual numbers (not generic "600 req/min")
- [ ] Identification of at least 2 unused endpoints that could benefit the pipeline
- [ ] Clear statement of what data BDL does NOT provide (for gap analysis)

### Dependencies
- Blocks: Task 7 (compute derived stats) - need to understand available source data
- Related: K-37 (MLB API Comparison) - coordinate on field availability

---

## K-35: Z-Score Best Practices Research

### Mission
Research statistical best practices for Z-score calculations in fantasy baseball, including sample size handling, position adjustments, outlier treatment, and temporal considerations.

### Background
Z-scores are critical for player valuation in fantasy baseball. The current implementation may have methodological gaps. We need rigorous statistical guidance for Task 9 (compute VORP/z-score in player_daily_metrics).

### Research Questions (Answer All)

#### 1. Sample Size Considerations
- What is the minimum sample size (plate appearances, innings pitched) for reliable Z-scores?
- How should Z-scores be calculated for players with small samples (rookies, part-time players)?
- Should we use confidence intervals or shrinkage estimators for small samples?
- What does Fangraphs, Baseball-Reference, or other authorities recommend?

#### 2. Position Adjustments
- Should Z-scores be calculated within position groups (C vs 1B vs OF) or across all players?
- How to handle multi-position eligible players (e.g., Bellinger CF/LF/RF)?
- Should replacement level differ by position?
- How do standard fantasy platforms (Yahoo, ESPN) calculate position-adjusted values?

#### 3. Outlier Handling
- How should extreme outliers be treated (e.g., Ohtani's pitching + hitting)?
- Should Z-scores be Winsorized (capped at certain percentiles)?
- How to handle players with infinite or undefined rates (division by zero)?
- What is the impact of outliers on league mean/std dev?

#### 4. Temporal Considerations
- Should Z-scores use rolling windows (7-day, 14-day, 30-day) or season-to-date?
- How to weight recent performance vs early season data?
- Should we use weighted averages (exponential decay)? If so, what λ?
- How do Z-scores stabilize over time (when do they become reliable)?

#### 5. Statistical Methodology
- Should we use mean/std dev or median/IQR for robustness?
- Are there alternatives to Z-scores that work better for fantasy (percentiles, robust Z-scores)?
- How to handle non-normal distributions (rate stats are often skewed)?
- Should we transform data (log, Box-Cox) before Z-score calculation?

#### 6. Category-Specific Considerations
- Are Z-scores equally valid for all 5x5 categories?
- How to handle counting stats (R, RBI, HR) vs rate stats (AVG, ERA, WHIP)?
- Should rate stats use weighted Z-scores based on sample size?
- How to combine Z-scores across categories into single value?

### Deliverable

**File:** `reports/2026-04-09-zscore-best-practices.md`

**Required Sections:**
1. **Executive Summary** (1 paragraph with key recommendation)
2. **Sample Size Guidelines** (table: stat type, min PA/IP, recommended approach)
3. **Position Adjustment Strategy** (recommended approach with justification)
4. **Outlier Handling Protocol** (specific rules with examples)
5. **Temporal Window Recommendations** (which windows for which use cases)
6. **Implementation Guide** (pseudocode for Z-score calculation)
7. **Validation Checklist** (how to verify Z-scores are correct)

### Acceptance Criteria
- [ ] All 6 research questions answered with citations
- [ ] At least 2 authoritative sources cited (Fangraphs, academic paper, book)
- [ ] Specific recommendations for small sample handling (<50 PA)
- [ ] Clear position on mean vs median debate
- [ ] Working pseudocode or Python example
- [ ] Red flags: at least 3 common mistakes to avoid

### Dependencies
- Blocks: Task 9 (compute VORP/z-score)
- Related: K-36 (Fantasy Scoring) - coordinate on category weights

---

## K-36: Fantasy Scoring Systems Research

### Mission
Document H2H One Win scoring system requirements, compare to standard 5x5 and points leagues, and provide definitive calculation formulas for all categories.

### Background
The system implements H2H One Win format but there may be ambiguity in category definitions (especially NSB). We need exact specifications for Tasks 7-9.

### Research Questions (Answer All)

#### 1. H2H One Win Format Specification
- What are the EXACT category definitions? (Batting: R, HR, RBI, NSB, OBP? Pitching: QS, K/9, ERA, WHIP, NSV?)
- How is "One Win" calculated? (1 win per category won? 1 win total per week?)
- What happens in ties? (0.5 wins each? push?)
- Are there minimum IP requirements? (weekly or daily?)

#### 2. NSB (Net Stolen Bases) Clarification
- Formula: SB - CS or max(0, SB - CS)?
- Can NSB be negative?
- Does Yahoo calculate this in real-time or overnight?
- How does this affect player valuation?

#### 3. Pitching Categories
- QS (Quality Starts): 6+ IP AND 3 or fewer ER? Or different definition?
- K/9: strikeouts per 9 innings? How to handle partial innings?
- NSV (Net Saves): SV - BS? Or something else?
- Are holds included?

#### 4. Position Eligibility Rules
- How many games at a position for eligibility? (Yahoo: 5 games started, 10 games played?)
- Do multi-eligible players count toward scarcity at all positions?
- How does Yahoo handle September call-ups for eligibility?
- What is the CF/LF/RF breakdown vs generic OF?

#### 5. Comparison to Standard Formats
- 5x5 traditional: R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP
- H2H Categories: same categories, weekly head-to-head
- Points leagues: how do category values translate?
- Why H2H One Win vs other formats?

#### 6. Strategic Implications
- Which categories have highest variance? (scarcity impact)
- Which stats are most predictable? (reliability)
- How do punting strategies work in H2H One Win?
- What is the "replacement level" for each category?

### Deliverable

**File:** `reports/2026-04-09-h2h-scoring-systems.md`

**Required Sections:**
1. **Executive Summary** (format definition in one paragraph)
2. **Category Reference** (table: category, definition, formula, data source)
3. **NSB Deep Dive** (history, formula, strategic impact)
4. **Position Eligibility Matrix** (requirements by position)
5. **Format Comparison** (H2H One Win vs 5x5 vs Points)
6. **Strategic Insights** (3-5 tactical implications for lineup optimization)
7. **Implementation Checklist** (what we need to track for each category)

### Acceptance Criteria
- [ ] All 6 research questions answered
- [ ] Official Yahoo documentation or authoritative source for NSB formula
- [ ] Clear position eligibility rules with game counts
- [ ] Category formulas verified against at least 2 sources
- [ ] Strategic value: at least 3 actionable insights for H2H One Win
- [ ] Glossary of terms for non-fantasy-baseball readers

### Dependencies
- Blocks: Tasks 7, 8, 9 (all scoring-related tasks)
- Related: K-35 (Z-Score) - coordinate on category weights

---

## K-37: MLB API Comparison Research

### Mission
Compare MLB Stats API, Baseball Savant (Statcast), and BallDontLie for data completeness, accuracy, latency, and cost to determine optimal data source strategy.

### Background
The system currently uses multiple data sources but there may be redundancy or gaps. We need a comprehensive comparison to fix Tasks 4 (probable_pitchers) and 5 (statcast_performances).

### Research Questions (Answer All)

#### 1. MLB Stats API (statsapi.mlb.com)
- What data does the official MLB API provide?
- Is it free? What are the rate limits?
- What is the authentication mechanism?
- What data formats are available (JSON, XML, CSV)?
- How real-time is the data?

#### 2. Baseball Savant (Statcast)
- What advanced metrics are available? (xwOBA, barrel%, exit velocity, launch angle, sprint speed)
- How to access programmatically? (API, CSV downloads, pybaseball)
- What is the data lag? (real-time, end-of-game, next-day?)
- What are the rate limits for Statcast searches?
- Is there a Python library (pybaseball)? How reliable is it?

#### 3. BallDontLie (current primary source)
- What are BDL's strengths vs weaknesses?
- What data does BDL aggregate from other sources?
- How does BDL handle real-time updates?
- What is the cost structure (GOAT tier: $39.99/mo, 600 req/min)?

#### 4. Feature Comparison Matrix
Create detailed comparison of:
- Game schedules and probable pitchers
- Box scores (batting and pitching)
- Advanced stats (Statcast metrics)
- Injury reports
- Real-time vs delayed data
- Historical data availability
- Rate limits and cost

#### 5. Integration Strategy
- Which API should be primary for which data type?
- Where should we use redundancy (multiple sources)?
- Where is single-source sufficient?
- What are the fallback strategies if primary source fails?

#### 6. Probable Pitchers Specific
- MLB Stats API: how to get probable pitchers?
- BDL: which endpoint provides probable pitchers?
- Baseball Savant: any probable pitcher data?
- What is the accuracy and update frequency?

#### 7. Statcast Data Specific
- Which API provides xwOBA, barrel%, exit velocity?
- How far back does Statcast data go (2015?)?
- What is the granularity (per PA, per game, aggregated)?
- Can we get historical Statcast data via API or only current?

### Deliverable

**File:** `reports/2026-04-09-mlb-api-comparison.md`

**Required Sections:**
1. **Executive Summary** (recommended API strategy in 1 paragraph)
2. **API Profiles** (detailed section for each API)
3. **Feature Comparison Matrix** (comprehensive table)
4. **Probable Pitchers Analysis** (which API, how to access)
5. **Statcast Deep Dive** (advanced metrics availability)
6. **Integration Recommendations** (primary/secondary/fallback for each data type)
7. **Cost-Benefit Analysis** (ROI of each API subscription)
8. **Implementation Roadmap** (migration plan if needed)

### Acceptance Criteria
- [ ] All 7 research questions answered
- [ ] Feature comparison matrix with at least 10 comparison points
- [ ] At least one working code example per API (if accessible)
- [ ] Clear recommendation for probable_pitchers data source
- [ ] Clear recommendation for Statcast data source
- [ ] Identification of any data gaps requiring manual intervention

### Dependencies
- Blocks: Tasks 4 (probable_pitchers), 5 (statcast_performances)
- Related: K-34 (BDL API) - coordinate on BDL capabilities

---

## K-38: VORP Implementation Guide Research

### Mission
Research Value Over Replacement Player (VORP) calculation methodology for fantasy baseball, including replacement level definition, position adjustments, and implementation formulas.

### Background
VORP is a key metric for player valuation. We need rigorous methodology for Task 9 (compute VORP in player_daily_metrics).

### Research Questions (Answer All)

#### 1. VORP Fundamentals
- What is the definition of VORP in fantasy baseball?
- How does fantasy VORP differ from real baseball VORP (Keith Woolner)?
- What is the mathematical formula? (Runs Above Replacement? Points Above Replacement?)

#### 2. Replacement Level Definition
- How is "replacement level" defined for fantasy?
- Is it the best available waiver wire player? 50th percentile? 75th percentile?
- Should replacement level differ by position (C vs 1B vs OF)?
- How does replacement level change during the season?
- What is the replacement level for pitchers (SP vs RP)?

#### 3. Position Adjustments
- Should VORP be position-adjusted?
- How to handle multi-eligible players?
- Should catchers have lower replacement level due to scarcity?
- How to account for positional scarcity in VORP?

#### 4. Calculation Methodology
- Should VORP use Z-scores or raw stat differences?
- How to combine multiple categories into single VORP?
- Should VORP be category-weighted? (HR more valuable than SB?)
- How to handle rate stats vs counting stats?

#### 5. Temporal Considerations
- Should VORP use rest-of-season projections or year-to-date performance?
- How should VORP update as season progresses?
- Should there be different VORP for different time horizons (ROS, weekly, daily)?

#### 6. Implementation Examples
- How does Fangraphs calculate fantasy VORP?
- How does Baseball Prospectus calculate VORP?
- Are there open-source implementations we can reference?
- What SQL/Python formulas are needed?

#### 7. Practical Application
- How to use VORP for trade decisions?
- How to use VORP for waiver wire decisions?
- How to use VORP for lineup optimization?
- What are VORP limitations? (when NOT to use it)

### Deliverable

**File:** `reports/2026-04-09-vorp-implementation-guide.md`

**Required Sections:**
1. **Executive Summary** (VORP definition and recommendation)
2. **VORP Formula** (mathematical definition with variables)
3. **Replacement Level Strategy** (how to calculate for each position)
4. **Position Adjustment Framework** (scarcity considerations)
5. **Implementation Guide** (Python/SQL pseudocode)
6. **Worked Examples** (3-5 player examples showing calculation)
7. **Usage Guide** (how to interpret and apply VORP)
8. **Limitations & Caveats** (when VORP breaks down)

### Acceptance Criteria
- [ ] All 7 research questions answered
- [ ] Clear mathematical formula for VORP
- [ ] Replacement level benchmark numbers for each position
- [ ] At least 3 worked examples with actual numbers
- [ ] Python or SQL pseudocode for implementation
- [ ] Citations from at least 2 authoritative sources
- [ ] Red flags: at least 3 situations where VORP should NOT be used

### Dependencies
- Blocks: Task 9 (compute VORP/z-score)
- Related: K-35 (Z-Score), K-36 (Scoring) - coordinate on calculations

---

## 🔄 COORDINATION NOTES

### Parallel Execution Opportunities
- K-34, K-35, K-36 can run in parallel (no dependencies)
- K-37 depends on K-34 (BDL capabilities) but can start with partial info
- K-38 depends on K-35 (Z-scores) and K-36 (scoring systems)

### Cross-Reference Requirements
- K-34 and K-37 should coordinate on BDL capabilities
- K-35, K-36, and K-38 should coordinate on statistical methodology
- All reports should reference each other where relevant

### Quality Gates
Before marking complete, each task must:
1. Answer ALL research questions
2. Provide actionable recommendations
3. Include working code examples where applicable
4. Cite authoritative sources
5. Be reviewed for technical accuracy

---

## 📋 HANDOFF CHECKLIST

When complete, Kimi should:
- [ ] Save all 5 reports to `reports/` directory
- [ ] Update HANDOFF.md with findings summaries
- [ ] Flag any blocking issues discovered
- [ ] Recommend priority order for implementation
- [ ] Suggest any additional research needed

---

**End of Delegation Bundle — Ready for Kimi CLI Execution**

