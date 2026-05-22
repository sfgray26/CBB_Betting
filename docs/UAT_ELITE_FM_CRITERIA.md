# UAT Evaluation Criteria - Elite Fantasy Manager Perspective

## Persona Definition

**Background**: 10+ years fantasy baseball experience, multiple league championships, deep understanding of:
- Category management and roster construction
- Streaming strategies and matchup optimization
- Prospect valuation and waiver wire dynamics
- Injury management and lineup decisions

**Expectations**: Tools that provide actionable insights, not just data dumps. Values efficiency, accuracy, and competitive advantage.

## Evaluation Dimensions

### 1. Roster Management (Weight: 25%)

#### 1.1 Lineup Optimization
- [ ] **P0**: Starting lineup correctly identifies optimal players
- [ ] **P0**: Position eligibility handled accurately (UTIL, OF slots)
- [ ] **P1**: Two-start pitcher detection working
- [ ] **P1**: Platoon splits considered for daily lineups
- [ ] **P2**: Weather postponements trigger automatic benching

**Test Cases**:
```
Scenario: User has 3 starting pitchers on Tuesday
Expected: System identifies all 3 as "START" recommendations
Actual: [To be filled during UAT]
Score: [0-10]
```

#### 1.2 Roster Construction Analysis
- [ ] **P1**: Category balance visualization clear
- [ ] **P1**: Strengths/weaknesses accurately identified
- [ ] **P2**: Suggested trades improve category balance
- [ ] **P2**: IL slot management recommendations

**Scoring Rubric**:
- 10: Elite-level insights, would use in championship week
- 8: Good recommendations, occasional misses
- 6: Decent data, needs manual interpretation
- 4: Basic info, significant gaps
- 2: Unreliable, can't trust for decisions
- 0: Broken or misleading

### 2. Waiver Wire Intelligence (Weight: 20%)

#### 2.1 Player Recommendations
- [ ] **P0**: FAAB bids reflect actual player value
- [ ] **P0**: Priority rankings match expert consensus
- [ ] **P1**: Matchup-based streaming picks are logical
- [ ] **P1**: Category-specific targets identified correctly
- [ ] **P2**: Handcuff/stash recommendations accurate

**Key Questions**:
1. Would I actually pick up the #1 recommended player?
2. Are the bid suggestions reasonable for my league?
3. Does it account for my specific category needs?

#### 2.2 Add/Drop Decision Support
- [ ] **P1**: Drop recommendations don't suggest cutting keepers
- [ ] **P1**: ROS (Rest of Season) value considered
- [ ] **P2**: Playoff schedule strength factored in

**Test Scenario**:
```
Given: User is weak in SB, strong in HR
When: Viewing waiver wire
Then: Should prioritize SB contributors over HR hitters
Actual Behavior: [Document during testing]
```

### 3. Matchup Analysis (Weight: 20%)

#### 3.1 Weekly Matchup Preview
- [ ] **P0**: Opponent roster visible and current
- [ ] **P0**: Category-by-category projections displayed
- [ ] **P1**: Win probability seems realistic
- [ ] **P2**: Suggested moves to win specific categories

**Scoring Criteria**:
- Projections within 10% of actual = Full points
- Projections 10-20% off = Half points
- Projections >20% off = Zero points

#### 3.2 Daily Streaming Decisions
- [ ] **P1**: Today's best pickups clearly identified
- [ ] **P1**: Tomorrow's streamers previewed
- [ ] **P2**: Quality start probability shown
- [ ] **P2**: Batter vs pitcher matchups analyzed

### 4. Data Quality & Timeliness (Weight: 15%)

#### 4.1 Roster Sync
- [ ] **P0**: Yahoo roster syncs within 15 minutes
- [ ] **P0**: Injury status updates reflected quickly
- [ ] **P1**: Lineup changes from Yahoo propagated
- [ ] **P2**: Probable pitchers updated daily

#### 4.2 Statistical Accuracy
- [ ] **P0**: Player stats match Yahoo/official sources
- [ ] **P1**: Projections use current-season data
- [ ] **P2**: Recent performance weighted appropriately

**Data Validation Tests**:
```python
# Automated check - compare to Yahoo
players_to_check = [
    "Shohei Ohtani",
    "Ronald Acuña Jr.",
    "Mookie Betts"
]
for player in players_to_check:
    verify_stats_match(player, source="yahoo")
```

### 5. User Experience (Weight: 10%)

#### 5.1 Navigation Efficiency
- [ ] **P1**: Can set lineup in < 2 minutes
- [ ] **P1**: Waiver search is fast and filterable
- [ ] **P2**: Mobile experience usable for quick checks

#### 5.2 Information Density
- [ ] **P1**: High-value info above the fold
- [ ] **P2**: Customizable views for different needs
- [ ] **P2**: Contextual help/tooltips where needed

### 6. Competitive Advantage (Weight: 10%)

#### 6.1 Unique Insights
- [ ] **P2**: Identifies opportunities competitors miss
- [ ] **P2**: Early warnings on declining players
- [ ] **P3**: Advanced stats (xWOBA, barrel%, etc.) available

#### 6.2 Edge Identification
- [ ] **P2**: Clear explanation of WHY a move is recommended
- [ ] **P3**: Confidence levels on projections
- [ ] **P3**: Alternative strategies presented

## Scoring Matrix

| Dimension | Weight | Score (1-10) | Weighted Score |
|-----------|--------|--------------|----------------|
| Roster Management | 25% | [ ] | [ ] |
| Waiver Wire Intelligence | 20% | [ ] | [ ] |
| Matchup Analysis | 20% | [ ] | [ ] |
| Data Quality | 15% | [ ] | [ ] |
| User Experience | 10% | [ ] | [ ] |
| Competitive Advantage | 10% | [ ] | [ ] |
| **TOTAL** | **100%** | | **[ ]** |

## Red Flags (Automatic Failure)

- [ ] Wrong lineup optimization costs user a category
- [ ] Waiver recommendations include injured/out players
- [ ] Stats don't match official sources
- [ ] Roster sync fails silently
- [ ] Core feature crashes or times out

## Test Execution Commands

```bash
# Full Elite FM evaluation
python scripts/uat_elite_fm_evaluation.py \
  --base-url https://your-app.railway.app \
  --api-key $API_KEY \
  --verbose \
  --output reports/uat_elite_fm_$(date +%Y%m%d).md
```

## Example Output

```
========================================
ELITE FANTASY MANAGER UAT REPORT
Date: 2026-05-17
Tester: Kimi + Devtools MCP
========================================

OVERALL SCORE: 8.4/10 ✅ (Pass)

CRITICAL FINDINGS:
- Roster Management: 9/10 ✅
  * Lineup optimization working well
  * Two-start pitcher detection accurate
  - Minor: Weather delays not auto-updating

- Waiver Wire: 8/10 ✅
  * FAAB recommendations reasonable
  * Good category-specific targeting
  - Issue: #3 ranked player was actually injured

- Matchup Analysis: 7/10 ⚠️
  * Category projections mostly accurate
  - Issue: Win probability seems inflated by 15%

RECOMMENDED ACTIONS:
1. Fix injury status sync (P1)
2. Calibrate win probability model (P2)
3. Add weather auto-update (P2)
```
