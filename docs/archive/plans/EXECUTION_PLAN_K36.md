# ?? ARCHIVED — Research Complete

> **Status:** ARCHIVED (April 11, 2026)
> **Original Location:** reports/EXECUTION_PLAN_K36.md
> **Archive Reason:** K-K36 research completed and delivered
> **Deliverables:** See reports/YYYY-MM-DD-*.md files for K-K36 output

---

# K-36 Execution Plan: H2H Scoring Systems Research

## Research Methodology

### Phase 1: Yahoo Fantasy Documentation (1 hour)

#### Search Strategy

**Query 1: Yahoo H2H One Win**
```
site:help.yahoo.com "H2H One Win" fantasy baseball
```

**Query 2: Yahoo Scoring Categories**
```
Yahoo Fantasy Baseball "H2H One Win" categories scoring
```

**Query 3: NSB Definition**
```
"net stolen bases" Yahoo Fantasy definition formula
```

**Query 4: NSV Definition**
```
"net saves" Yahoo Fantasy definition formula
```

#### Sources to Check

1. **Yahoo Help**
   - help.yahoo.com fantasy baseball section
   - Scoring settings documentation
   - Category definitions

2. **Fantasy Resources**
   - Razzball scoring guides
   - Fangraphs fantasy articles
   - Reddit r/fantasybaseball

### Phase 2: Category Definitions (1 hour)

#### Research Each Category

**Batting Categories:**
1. R (Runs)
2. HR (Home Runs)
3. RBI (Runs Batted In)
4. NSB (Net Stolen Bases)
5. OBP (On-Base Percentage)

**Pitching Categories:**
1. QS (Quality Starts)
2. K/9 (Strikeouts per 9 innings)
3. ERA (Earned Run Average)
4. WHIP (Walks + Hits per IP)
5. NSV (Net Saves)

**Questions to Answer:**
- Exact formula for each
- What constitutes a Quality Start?
- NSB = SB - CS? Can it be negative?
- NSV = SV - BS? Or something else?
- Minimum innings pitched requirements

### Phase 3: Position Eligibility (1 hour)

#### Search Strategy

**Query 1: Yahoo Position Eligibility**
```
site:help.yahoo.com "position eligibility" fantasy baseball games played
```

**Query 2: Outfield Positions**
```
Yahoo Fantasy "CF" "LF" "RF" "OF" eligibility separate
```

**Query 3: Multi-Position**
```
Yahoo Fantasy baseball multi-position eligibility rules
```

#### Information to Find

- Games/starts needed for eligibility
- CF/LF/RF vs OF granularity
- Multi-position handling
- September call-up rules
- Position assignment logic

### Phase 4: Format Comparison (1 hour)

Research standard fantasy formats:
1. Traditional 5x5 (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
2. H2H Categories (weekly matchups)
3. Points Leagues
4. H2H One Win

Document differences, advantages, and strategic implications.

### Phase 5: Strategic Analysis (1 hour)

Research strategic considerations:
- Which categories have highest variance?
- Which stats are most predictable?
- Punting strategies in H2H One Win
- Replacement level by category

### Phase 6: Documentation (1 hour)

Write comprehensive report.

---

## Expected Findings (Hypothesis)

1. **Categories**
   - R, HR, RBI, NSB, OBP (batting)
   - QS, K/9, ERA, WHIP, NSV (pitching)

2. **NSB Formula**
   - NSB = SB - CS
   - Can be negative
   - Penalty for getting caught

3. **QS Definition**
   - 6+ IP AND â‰¤3 ER (likely)

4. **Position Eligibility**
   - 5 games started OR 10 games played
   - Generic OF, not CF/LF/RF

---

## Acceptance Criteria Checklist

- [ ] Official NSB formula confirmed with source
- [ ] All 10+ categories documented with formulas
- [ ] Position eligibility rules clear
- [ ] 3+ strategic insights documented
- [ ] Glossary of terms provided

*Ready to execute*

