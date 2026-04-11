# ?? ARCHIVED — Research Complete

> **Status:** ARCHIVED (April 11, 2026)
> **Original Location:** reports/EXECUTION_PLAN_K35.md
> **Archive Reason:** K-K35 research completed and delivered
> **Deliverables:** See reports/YYYY-MM-DD-*.md files for K-K35 output

---

# K-35 Execution Plan: Z-Score Best Practices Research

## Research Methodology

### Phase 1: Statistical Foundation (1 hour)

#### Search Strategy

**Query 1: Z-Score Definition**
```
Z-score fantasy baseball valuation Fangraphs methodology
```

**Query 2: Sample Size Statistical Guidelines**
```
"regression to the mean" baseball stats stabilization points
```

**Query 3: Fantasy Baseball Statistics**
```
fantasy baseball Z-score calculation sample size best practices
```

#### Sources to Check

1. **Fangraphs**
   - Fantasy valuation articles
   - Statistical methodology guides
   - Stabilization points reference

2. **Baseball Prospectus**
   - Statistical methodology
   - Fantasy analysis articles

3. **Academic Sources**
   - Regression in baseball statistics
   - Shrinkage estimators

### Phase 2: Sample Size Research (1 hour)

#### Stabilization Points to Research

| Stat | Stabilization Point | Source |
|------|-------------------|--------|
| AVG | ~100 PA | Russell Carleton |
| OBP | ~100 PA | Russell Carleton |
| SLG | ~150 PA | Russell Carleton |
| ISO | ~150 PA | Russell Carleton |
| K% | ~60 PA | Russell Carleton |
| BB% | ~120 PA | Russell Carleton |
| ERA | ~70 IP | Russell Carleton |
| WHIP | ~70 IP | Russell Carleton |
| K/9 | ~70 IP | Russell Carleton |
| HR/FB | ~400 FB | Russell Carleton |

**Questions:**
- What happens below stabilization?
- Should we regress to mean?
- How to handle small samples?

### Phase 3: Position Adjustment Research (1 hour)

#### Search Strategy

**Query 1: Position-Adjusted Z-Scores**
```
"position adjusted" Z-score fantasy baseball valuation catcher
```

**Query 2: Multi-Position Eligibility**
```
fantasy baseball Z-score multi-position eligibility calculation
```

**Query 3: Replacement Level**
```
replacement level fantasy baseball Z-score calculation
```

#### Key Decisions

- Within-position vs global Z-scores?
- How to handle multi-eligible players?
- Should catchers get adjustment?

### Phase 4: Outlier Handling (1 hour)

#### Search Strategy

**Query 1: Winsorization**
```
Winsorization fantasy baseball Z-scores outlier handling
```

**Query 2: Robust Statistics**
```
"robust Z-score" median IQR fantasy baseball
```

**Query 3: Two-Way Players**
```
Shohei Ohtani two-way fantasy baseball valuation Z-score
```

#### Approaches to Research

1. **Winsorization**
   - Cap at 95th/5th percentile?
   - Effect on mean/std dev

2. **Robust Z-Scores**
   - Median instead of mean
   - IQR instead of std dev

3. **Ohtani Handling**
   - Separate batting/pitching?
   - Combine somehow?
   - Special case logic?

### Phase 5: Temporal Considerations (1 hour)

#### Search Strategy

**Query 1: Rolling Windows**
```
fantasy baseball "rolling" Z-score time window recent performance
```

**Query 2: Weighted Averages**
```
"exponentially weighted" fantasy baseball Z-score recency
```

**Query 3: Projections vs YTD**
```
ROS projections vs YTD Z-score fantasy baseball
```

#### Approaches to Compare

- YTD (year-to-date)
- Rolling 14d/30d/60d
- Exponentially weighted
- ROS projections

### Phase 6: Methodology Comparison (1 hour)

Compare approaches:
- Standard vs robust Z-scores
- Mean vs median
- Different weighting schemes
- Best practices from experts

### Phase 7: Documentation (1 hour)

Write comprehensive guide.

---

## Expected Findings (Hypothesis)

1. **Sample Size**
   - <30 PA/IP: Heavy regression needed
   - 30-100: Moderate regression
   - >100: Reliable

2. **Position Adjustment**
   - Global Z-scores are standard
   - VORP handles position scarcity

3. **Outliers**
   - Winsorization at 95th/5th percentile recommended
   - Ohtani: separate valuations

4. **Temporal**
   - Weighted approach for recency
   - Î»=0.9 to 0.95 for daily updates

---

## Acceptance Criteria Checklist

- [ ] At least 2 authoritative sources cited
- [ ] Specific recommendations for <50 PA
- [ ] Clear position on mean vs median
- [ ] Working pseudocode provided
- [ ] 3+ common mistakes documented

*Ready to execute*

