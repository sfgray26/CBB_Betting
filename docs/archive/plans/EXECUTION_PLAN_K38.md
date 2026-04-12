# ?? ARCHIVED — Research Complete

> **Status:** ARCHIVED (April 11, 2026)
> **Original Location:** reports/EXECUTION_PLAN_K38.md
> **Archive Reason:** K-K38 research completed and delivered
> **Deliverables:** See reports/YYYY-MM-DD-*.md files for K-K38 output

---

# K-38 Execution Plan: VORP Implementation Guide Research

## Research Methodology

### Phase 1: VORP Fundamentals (1 hour)

#### Search Strategy

**Query 1: VORP Definition**
```
"Value Over Replacement Player" fantasy baseball definition formula
```

**Query 2: Baseball Prospectus**
```
"Baseball Prospectus" VORP definition calculation Woolner
```

**Query 3: Fangraphs VORP**
```
Fangraphs "Value Over Replacement Player" fantasy baseball
```

**Query 4: Real vs Fantasy VORP**
```
VORP "real baseball" vs "fantasy baseball" difference
```

#### Sources to Check

1. **Baseball Prospectus**
   - Keith Woolner's original VORP
   - Real baseball formula (runs based)

2. **Fangraphs**
   - Fantasy VORP approach
   - Player valuation articles

3. **Fantasy Resources**
   - Razzball VORP explanation
   - Other fantasy sites

### Phase 2: Replacement Level (1.5 hours)

#### Search Strategy

**Query 1: Replacement Level Definition**
```
fantasy baseball "replacement level" definition calculation waiver wire
```

**Query 2: Positional Replacement Level**
```
"replacement level" catcher vs first base fantasy baseball
```

**Query 3: League Size Impact**
```
fantasy baseball replacement level "12 team" "10 team" league size
```

#### Key Questions

1. **Definition**
   - What is replacement level?
   - Best available on waivers?
   - 50th percentile?
   - Mathematical definition?

2. **Position-Specific**
   - Same replacement level for all positions?
   - Catchers have lower replacement level?
   - How to calculate per position?

3. **League Size**
   - How many teams matter?
   - Roster sizes impact?
   - Multi-eligibility impact?

4. **Temporal**
   - Does replacement level change?
   - In-season adjustments?
   - Streaming impact?

### Phase 3: Calculation Methodology (1.5 hours)

#### Search Strategy

**Query 1: VORP Formula**
```
fantasy baseball VORP formula calculation example
```

**Query 2: Z-Score VORP**
```
fantasy baseball VORP "Z-score" calculation
```

**Query 3: Category Weighting**
```
fantasy baseball VORP "category weight" equal value
```

#### Approaches to Compare

1. **Z-Score Approach**
   - Sum of Z-scores above replacement
   - Standard in fantasy

2. **Raw Stats Approach**
   - Difference in raw counting stats
   - Issue with rate stats

3. **Dollar Value Approach**
   - Convert to auction dollars
   - Complex but valuable

4. **Category Weighting**
   - Equal weights?
   - Value-based weights?
   - Historical performance weights?

### Phase 4: Implementation Examples (1 hour)

#### Search Strategy

**Query 1: Python VORP**
```
fantasy baseball VORP python code implementation
```

**Query 2: SQL VORP**
```
fantasy baseball VORP SQL calculation query
```

**Query 3: Excel VORP**
```
fantasy baseball VORP excel spreadsheet calculation
```

#### Code to Find

- Python implementations
- SQL implementations
- Spreadsheet formulas
- Open-source tools

### Phase 5: Position Adjustment (1 hour)

#### Search Strategy

**Query 1: Positional Scarcity**
```
fantasy baseball "positional scarcity" VORP catcher adjustment
```

**Query 2: Multi-Position VORP**
```
fantasy baseball VORP "multi-eligible" position calculation
```

**Query 3: Replacement Position**
```
which position for VORP multi-eligible fantasy baseball
```

#### Decisions Needed

1. **Positional Scarcity**
   - Should VORP account for position scarcity?
   - Or is it separate from replacement level?

2. **Multi-Eligible Players**
   - Use best position?
   - Use primary position?
   - Weighted average?

### Phase 6: Practical Application (1 hour)

#### Use Cases to Research

1. **Trade Decisions**
   - How to use VORP in trades
   - Trading surplus for need
   - Position balancing

2. **Waiver Wire**
   - Picking up replacement level players
   - Identifying undervalued players
   - Streaming decisions

3. **Lineup Optimization**
   - Daily lineup using VORP
   - Platoon advantages
   - Matchup-based decisions

4. **Limitations**
   - When NOT to use VORP
   - Categories VORP misses
   - Situational factors

### Phase 7: Documentation (1 hour)

Write comprehensive guide.

---

## Expected Findings (Hypothesis)

1. **VORP Formula**
   - Sum of Z-scores above replacement
   - Or raw stat difference Ã— category weight

2. **Replacement Level**
   - Best available on waiver wire
   - Differs by position (C < 1B)
   - Dynamic throughout season

3. **Position Adjustment**
   - Replacement level handles scarcity
   - No additional position multiplier needed

4. **Implementation**
   - Calculate mean/std for universe
   - Calculate replacement level stats
   - VORP = Player Z - Replacement Z

---

## Acceptance Criteria Checklist

- [ ] Clear mathematical formula
- [ ] Replacement level numbers per position
- [ ] At least 3 worked examples
- [ ] Python or SQL pseudocode
- [ ] 2+ authoritative sources cited
- [ ] 3+ situations where VORP should NOT be used

*Ready to execute*

