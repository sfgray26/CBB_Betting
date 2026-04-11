# K-37 Execution Plan: MLB API Comparison Research

## Research Methodology

### Phase 1: MLB Stats API Analysis (2 hours)

#### Search Strategy

**Query 1: Official MLB Stats API**
```
statsapi.mlb.com documentation API endpoints
```

**Query 2: Python Libraries**
```
mlbstatsapi python library documentation
```

**Query 3: MLB API Usage**
```
"statsapi.mlb.com" python examples endpoints
```

**Query 4: GitHub Examples**
```
github.com MLB Stats API python implementation
```

#### Sources to Check

1. **Official Resources**
   - https://statsapi.mlb.com
   - API documentation (if available)
   - GitHub: toddrob99/MLB-StatsAPI

2. **Python Libraries**
   - PyPI: mlbstatsapi
   - GitHub examples
   - Documentation quality

3. **Comparison Articles**
   - MLB Stats API vs BDL
   - Pros/cons of each
   - Migration guides

### Phase 2: Baseball Savant/Statcast Analysis (2 hours)

#### Search Strategy

**Query 1: Statcast Overview**
```
baseballsavant.mlb.com Statcast metrics API data
```

**Query 2: pybaseball Library**
```
pybaseball python library Statcast documentation
```

**Query 3: Available Metrics**
```
Statcast metrics xwOBA barrel exit velocity sprint speed
```

**Query 4: Data Access**
```
baseballsavant CSV export API scraper python
```

#### Sources to Check

1. **Official Savant**
   - https://baseballsavant.mlb.com
   - Statcast glossary
   - Data export tools

2. **pybaseball Library**
   - https://github.com/jldbc/pybaseball
   - Documentation and examples
   - Metrics availability

3. **Statcast Metrics**
   - Complete metric list
   - Calculations and definitions
   - Use cases for fantasy

4. **Current Codebase**
   - `backend/fantasy_baseball/statcast_ingestion.py`
   - Existing Statcast integration

### Phase 3: Probable Pitchers Deep Dive (1 hour)

#### Search Strategy

**Query 1: MLB Probable Pitchers**
```
"probable pitchers" MLB API statsapi endpoint
```

**Query 2: BDL Probable Pitchers**
```
"balldontlie" probable pitchers starting pitchers endpoint
```

**Query 3: Alternative Sources**
```
ESPN probable pitchers API CSV data source
```

**Query 4: Yahoo API**
```
Yahoo Fantasy API probable pitchers endpoint
```

#### Analysis Focus

- Identify which APIs provide probable pitcher data
- Document update frequency
- Assess reliability
- Recommend primary source

### Phase 4: Comparison & Synthesis (1 hour)

Create feature matrix comparing:
- Data freshness
- Historical depth
- Advanced metrics
- Rate limits
- Cost
- Documentation quality
- Python library support

### Phase 5: Documentation (1 hour)

Write the report with clear recommendations.

---

## Expected Findings (Hypothesis)

1. **MLB Stats API**
   - Free, official data
   - No rate limit documentation
   - Complex endpoint structure
   - Limited Python library support

2. **Baseball Savant**
   - Advanced metrics (xwOBA, etc.)
   - pybaseball library is good
   - Data delay (next day typically)
   - Scraping vs API

3. **BDL**
   - Paid ($39.99/mo GOAT tier)
   - Good for live data
   - No probable pitchers
   - Clear rate limits

4. **Recommendation**
   - BDL for live games
   - Savant for advanced metrics
   - Stats API as potential alternative

---

## Acceptance Criteria Checklist

- [ ] Feature matrix with 10+ comparison points
- [ ] At least one code example per API
- [ ] Clear recommendation for probable pitchers
- [ ] Statcast metrics fully catalogued
- [ ] Cost analysis included

*Ready to execute*
