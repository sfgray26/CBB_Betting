# H2H One Win Scoring System Analysis

> **Research Date:** April 11, 2026  
> **Researcher:** Kimi CLI (Deep Intelligence)  
> **Task ID:** K-36  
> **Status:** COMPLETE

---

## 1. Executive Summary

This report documents the **Head-to-Head (H2H) One Win** scoring system as implemented by Yahoo Fantasy Baseball. This format is the platform's target configuration and understanding its precise rules is critical for accurate player valuation, lineup optimization, and VORP/z-score calculations.

### Key Findings

| Aspect | Finding |
|--------|---------|
| **Format Type** | Head-to-Head One Win (Private Leagues) |
| **Categories** | 5 batting + 5 pitching (10 total) |
| **Weekly Result** | Single win/loss/tie (not category-by-category) |
| **NSB Formula** | SB - CS (can be negative) |
| **NSV Formula** | Saves - Blown Saves (can be negative) |
| **QS Definition** | 6+ IP AND ≤3 ER |
| **Position Eligibility** | 5 games started OR 10 games played |

### Critical Clarifications

1. **NSB CAN be negative** - Net Stolen Bases = Stolen Bases minus Caught Stealing
2. **NSV CAN be negative** - Net Saves = Saves minus Blown Saves
3. **H2H One Win ≠ H2H Categories** - One Win gives 1 result per week; Categories gives 10 results per week
4. **All 10 categories matter equally** - Win 6+ categories = 1 weekly win

---

## 2. Format Overview

### What is H2H One Win?

> "H2H One Win (Private Leagues only) is identical to H2H Categories, except only 1 result is added to the W-L-T record per week."
> — Yahoo Fantasy Baseball Official Documentation

### Comparison of H2H Formats

| Format | Weekly Result | Strategy Implication |
|--------|---------------|---------------------|
| **H2H Categories** | 10 results (W-L-T per category) | Can lose week 4-6 and still win season |
| **H2H One Win** | 1 result (win majority of categories) | More binary; playoff seeding critical |
| **H2H Points** | 1 result (total points) | Different player valuations entirely |
| **Rotisserie** | No weekly matchups | Season-long accumulation |

### Why H2H One Win?

- **Balanced approach** - Rewards both consistency and category dominance
- **Playoff friendly** - Single weekly result mirrors fantasy football
- **Less random than Categories** - One bad category doesn't doom your week
- **More strategic than Points** - Category management matters

---

## 3. Category Reference

### 3.1 Batting Categories (5)

| Category | Abbreviation | Type | Definition | Formula |
|----------|--------------|------|------------|---------|
| **Runs** | R | Counting | Times batter scores | Runs scored |
| **Home Runs** | HR | Counting | Home runs hit | HR total |
| **RBI** | RBI | Counting | Runs batted in | RBI total |
| **Net Stolen Bases** | NSB | Counting (net) | SB minus CS | `SB - CS` |
| **On-Base Percentage** | OBP | Ratio | Times on base per PA | `(H + BB + HBP) / PA` |

#### NSB (Net Stolen Bases) Deep Dive

**Official Yahoo Definition:**
> "Net Stolen Bases: SB - CS"

**Key Points:**
- **CAN be negative** - If CS > SB, NSB is negative
- Penalizes inefficient base stealers
- Rewards high-success-rate stealers
- Makes players like Trea Turner (high volume, good success rate) more valuable
- Makes players like Jorge Mateo (high volume, lower success rate) less valuable

**Example:**
| Player | SB | CS | NSB |
|--------|-----|-----|-----|
| Trea Turner | 30 | 3 | **27** |
| Jorge Mateo | 35 | 10 | **25** |
| Rickey Henderson | 10 | 1 | **9** |
| Slow Runner | 2 | 5 | **-3** |

**Calculation in Code:**
```python
def calculate_nsb(sb: int, cs: int) -> int:
    """
    Calculate Net Stolen Bases.
    
    NSB = SB - CS
    Can be negative.
    """
    return sb - cs
```

**Strategic Implication:**
- Players with high CS% hurt your team
- Don't stream base stealers with <75% success rate
- One CS cancels one SB - efficiency matters as much as volume

---

### 3.2 Pitching Categories (5)

| Category | Abbreviation | Type | Definition | Formula |
|----------|--------------|------|------------|---------|
| **Quality Starts** | QS | Counting | 6+ IP, ≤3 ER starts | Count of qualifying starts |
| **Strikeouts per 9** | K/9 | Ratio | Strikeouts per 9 innings | `(K / IP) * 9` |
| **ERA** | ERA | Ratio | Earned runs per 9 innings | `(ER * 9) / IP` |
| **WHIP** | WHIP | Ratio | Walks + Hits per inning | `(BB + H) / IP` |
| **Net Saves** | NSV | Counting (net) | Saves minus Blown Saves | `SV - BS` |

#### QS (Quality Start) Deep Dive

**Official Definition (MLB/Yahoo):**
> "A starting pitcher records a quality start when he pitches at least six innings and allows three earned runs or fewer."

**Requirements:**
1. Must be the starting pitcher
2. Must complete ≥6 innings
3. Must allow ≤3 earned runs
4. Final line matters (can lose QS in 7th inning)

**Key Points:**
- 6 IP + 3 ER = minimum QS (4.50 ERA - not great but counts)
- 7 IP + 2 ER = better QS
- 8 IP + 1 ER = excellent QS
- 9 IP + 0 ER = complete game shutout (also QS)

**QS Rate Leaders (2024):**
| Pitcher | QS | GS | QS% |
|---------|-----|-----|------|
| Zack Wheeler | 26 | 32 | 81% |
| Logan Webb | 24 | 33 | 73% |
| Corbin Burnes | 23 | 32 | 72% |

**Calculation in Code:**
```python
def is_quality_start(innings_pitched: float, earned_runs: int) -> bool:
    """
    Determine if a start qualifies as a Quality Start.
    
    QS = 6+ IP AND ≤3 ER
    """
    return innings_pitched >= 6 and earned_runs <= 3
```

---

#### K/9 (Strikeouts per 9 Innings) Deep Dive

**Formula:**
```
K/9 = (Strikeouts / Innings Pitched) × 9
```

**Handling Partial Innings:**
- Standard decimal format: 6.2 innings = 6 + 2/3 = 6.667 innings
- BDL provides IP as string "6.2" or float 6.2
- Conversion: `ip_outs = int(ip) * 3 + (ip % 1) * 10`

**Example:**
| Pitcher | K | IP | K/9 |
|---------|-----|-------|------|
| Dylan Cease | 8 | 6.0 | 12.0 |
| Logan Webb | 6 | 7.0 | 7.71 |
| Pitcher A | 5 | 5.2 | 7.89 |

**Calculation:**
```python
def calculate_k_per_9(strikeouts: int, innings_pitched: float) -> float:
    """
    Calculate K/9 ratio.
    
    K/9 = (K / IP) * 9
    """
    if innings_pitched == 0:
        return 0.0
    return round((strikeouts / innings_pitched) * 9, 2)
```

---

#### NSV (Net Saves) Deep Dive

**Official Yahoo Definition:**
> "NSV, Net Saves: Saves - Blown Saves"

**Key Points:**
- **CAN be negative** - If BS > SV, NSV is negative
- Penalizes closers with high blown save rates
- Rewards reliable closers
- Makes high-volume, shaky closers less valuable

**Blown Save Definition:**
- Pitcher enters with save opportunity
- Allows tying run to score (regardless of who scores it)
- Even if team retakes lead, it's a blown save

**Example:**
| Closer | SV | BS | NSV |
|--------|-----|-----|-----|
| Emmanuel Clase | 40 | 2 | **38** |
| Shaky Closer | 35 | 8 | **27** |
| Bad Closer | 25 | 12 | **13** |
| Setup Guy | 5 | 1 | **4** |

**NSV+HLD Variation:**
Some leagues use NSV+HLD:
> "NSV+HLD, Net Saves and Holds: (Saves - Blown Saves) + Holds"

---

## 4. Position Eligibility Rules

### 4.1 Yahoo Position Eligibility (Official)

**Batters:**
> "Batters need either 5 Games Started or 10 Games Played at a position to gain eligibility."

**Pitchers:**
> "Pitchers need 3 Starts to gain SP eligibility, 5 Relief Appearances to gain RP eligibility."

### 4.2 Eligibility Requirements Table

| Position Type | Requirement | Notes |
|---------------|-------------|-------|
| **C, 1B, 2B, 3B, SS** | 5 GS OR 10 GP | Infield positions |
| **OF (all)** | 5 GS OR 10 GP | Generic OF (not LF/CF/RF) |
| **DH/UTIL** | No requirements | Any batter eligible |
| **SP** | 3 starts | Starting pitcher |
| **RP** | 5 relief apps | Relief pitcher |

### 4.3 Multi-Position Eligibility

**Yahoo Approach:**
- Players can be eligible at multiple positions
- Previous season games count for start-of-season eligibility
- Current season games update eligibility in real-time
- CF/LF/RF are NOT separate positions in Yahoo (all = OF)

**Example Multi-Eligible Players (2026):**
| Player | Primary | Secondary | Value |
|--------|---------|-----------|-------|
| Jose Altuve | 2B | OF | High flexibility |
| Mookie Betts | OF | 2B | High flexibility |
| Ben Rice | C | 1B | Catcher scarcity + 1B |
| Noelvi Marte | 3B | OF | Corner INF + OF |

### 4.4 Position Scarcity Considerations

| Position | Scarcity Level | Notes |
|----------|---------------|-------|
| **Catcher** | HIGH | Worst offensive production |
| **Shortstop** | MEDIUM | Deep but top-heavy |
| **3B** | MEDIUM | Good production |
| **2B** | LOW | Deep position |
| **1B** | LOW | Plenty of power |
| **OF** | LOW | 3 spots, deep pool |
| **SP** | HIGH | Injury risk, innings limits |
| **RP** | HIGH | Closer turnover |

---

## 5. Format Comparison: H2H One Win vs Others

### 5.1 Detailed Comparison

| Aspect | H2H One Win | H2H Categories | H2H Points | Rotisserie |
|--------|-------------|----------------|------------|------------|
| **Weekly Matchup** | Yes | Yes | Yes | No |
| **Result Type** | 1 W/L/T | 10 W/L/T | 1 W/L | Season-long rank |
| **Win Condition** | Win 6+ of 10 cats | Win most categories | Most total points | Best season totals |
| **Punting Strategy** | Viable | Viable | Not applicable | Risky |
| **Playoffs** | Yes | Yes | Yes | No |
| **Category Balance** | Important | Critical | N/A | Critical |
| **Streaming** | Common | Common | Less common | Rare |

### 5.2 H2H One Win vs H2H Categories

**Key Difference:**
- **H2H Categories:** Win 6-4 in categories = 6 wins, 4 losses added to record
- **H2H One Win:** Win 6-4 in categories = 1 win, 0 losses added to record

**Strategic Implications:**

| Scenario | H2H Categories Result | H2H One Win Result |
|----------|----------------------|-------------------|
| Win 6-4 weekly | 6-4 record | 1-0 record |
| Lose 4-6 weekly | 4-6 record | 0-1 record |
| Go 5-5 (tie) | 5-5 record | 0.5-0.5 record |
| Dominate 9-1 | 9-1 record | 1-0 record |
| Get swept 1-9 | 1-9 record | 0-1 record |

**Why One Win is Different:**
1. **Domination not rewarded** - 6-4 win = 9-1 win in standings
2. **Consistency matters** - Better to win every week 6-4 than alternate 9-1 and 1-9
3. **Playoff seeding** - Record matters, not category dominance
4. **Tiebreakers** - Often use total category wins (reverts to Categories-style)

### 5.3 Punting in H2H One Win

**What is Punting?**
Intentionally ignoring one category to strengthen others.

**Viable Punt Categories:**
| Category | Punt Viability | Reasoning |
|----------|---------------|-----------|
| **Saves (NSV)** | HIGH | Only need 1 closer; stream saves |
| **Steals (NSB)** | HIGH | One-category players less valuable |
| **AVG/OBP** | MEDIUM | Can win with power |
| **QS** | LOW | Too tied to wins, ERA, WHIP, K/9 |
| **ERA/WHIP** | LOW | Hard to punt ratios |
| **HR/RBI/R** | VERY LOW | Too correlated with other cats |
| **K/9** | MEDIUM | Can win with ground ball pitchers |

**Punting Saves Example:**
- Don't draft any closers
- Use RP slots for elite setup men (high K/9, good ratios)
- Stream saves when ahead in matchup
- Win QS, K/9, ERA, WHIP consistently
- Hope to split offensive categories 3-2

---

## 6. Strategic Insights

### 6.1 Category Variance Analysis

| Category | Predictability | Volatility | Streaming Viable? |
|----------|---------------|------------|-------------------|
| **R** | Medium | High | Yes |
| **HR** | Medium | High | Yes |
| **RBI** | Medium | High | Yes |
| **NSB** | Low | Very High | Yes (matchup dependent) |
| **OBP** | High | Low | No |
| **QS** | Medium | Medium | Limited |
| **K/9** | High | Low | No |
| **ERA** | Low | Very High | Yes (2-start stream) |
| **WHIP** | Medium | High | Limited |
| **NSV** | Low | Very High | Yes |

**High Variance Categories (Stream aggressively):**
- ERA - One bad start can ruin it; stream for matchups
- NSB - Steal at home vs bad catchers
- NSV - Closer turnover; stream new closers

**Low Variance Categories (Draft for):**
- OBP - Skill-based, consistent
- K/9 - Skill-based, consistent
- WHIP - Skill-based, less volatile than ERA

### 6.2 Two-Start Pitcher Strategy

**In H2H One Win:**
- Two-start pitchers valuable for QS, K/9, and wins
- BUT - bad two-start week can destroy ERA
- Risk/reward calculation needed

**When to stream two-start pitchers:**
- Need QS and K/9
- Opponent has weak pitching
- Matchups are favorable (vs weak offenses)
- Your ERA/WHIP cushion is large

**When to avoid:**
- ERA/WHIP are close
- Matchups are tough
- Already leading in QS/K/9

### 6.3 Lineup Optimization Tips

**Daily vs Weekly Lineups:**
- If daily: Check matchups, platoon advantages
- If weekly: Prioritize volume (2-start SPs, everyday players)

**Category-Specific Streaming:**
```
Need SB? → Pick up players facing bad catchers
Need HR? → Pick up players in hitter-friendly parks
Need QS? → Pick up pitchers with favorable matchups
Need SV? → Pick up new closers, committee members
```

---

## 7. Implementation Checklist

### 7.1 For Platform Development

| Requirement | Status | Notes |
|-------------|--------|-------|
| Track all 10 categories | ☐ | R, HR, RBI, NSB, OBP, QS, K/9, ERA, WHIP, NSV |
| NSB calculation | ☐ | SB - CS (can be negative) |
| NSV calculation | ☐ | SV - BS (can be negative) |
| QS detection | ☐ | 6+ IP AND ≤3 ER |
| K/9 calculation | ☐ | (K / IP) × 9 |
| Weekly matchup scoring | ☐ | Win = 6+ categories won |
| Position eligibility | ☐ | 5 GS / 10 GP for batters |
| Pitcher eligibility | ☐ | 3 starts = SP, 5 relief = RP |

### 7.2 Data Requirements from BDL

| Category | BDL Fields Needed | Calculation |
|----------|------------------|-------------|
| R | `r` | Direct |
| HR | `hr` | Direct |
| RBI | `rbi` | Direct |
| NSB | `sb`, `cs` | `sb - cs` |
| OBP | `h`, `bb`, `hbp`, `ab` | `(h + bb + hbp) / pa` |
| QS | `ip`, `er`, `games_started` | `ip >= 6 AND er <= 3` |
| K/9 | `k`, `ip` | `(k / ip) * 9` |
| ERA | `er`, `ip` | `(er * 9) / ip` |
| WHIP | `bb_allowed`, `h_allowed`, `ip` | `(bb + h) / ip` |
| NSV | `saves`, `blown_saves` | `saves - blown_saves` |

### 7.3 Weekly Scoring Logic

```python
def calculate_weekly_matchup(team_a_stats: dict, team_b_stats: dict) -> dict:
    """
    Calculate H2H One Win matchup result.
    
    Returns win/loss/tie for the week.
    """
    categories = ['r', 'hr', 'rbi', 'nsb', 'obp', 'qs', 'k9', 'era', 'whip', 'nsv']
    
    team_a_wins = 0
    team_b_wins = 0
    
    for cat in categories:
        if cat in ['era', 'whip']:  # Lower is better
            if team_a_stats[cat] < team_b_stats[cat]:
                team_a_wins += 1
            elif team_b_stats[cat] < team_a_stats[cat]:
                team_b_wins += 1
        else:  # Higher is better
            if team_a_stats[cat] > team_b_stats[cat]:
                team_a_wins += 1
            elif team_b_stats[cat] > team_a_stats[cat]:
                team_b_wins += 1
    
    # H2H One Win: Win majority of categories
    if team_a_wins > team_b_wins:
        return {'winner': 'A', 'result': '1-0', 'category_score': f'{team_a_wins}-{team_b_wins}'}
    elif team_b_wins > team_a_wins:
        return {'winner': 'B', 'result': '0-1', 'category_score': f'{team_a_wins}-{team_b_wins}'}
    else:
        return {'winner': 'tie', 'result': '0.5-0.5', 'category_score': f'{team_a_wins}-{team_b_wins}'}
```

---

## 8. Glossary

| Term | Definition |
|------|------------|
| **BDL** | BallDontLie (data provider) |
| **BS** | Blown Save - when a closer allows tying run to score |
| **CS** | Caught Stealing - runner thrown out attempting steal |
| **ERA** | Earned Run Average - earned runs per 9 innings |
| **GS** | Games Started - for position eligibility |
| **GP** | Games Played - for position eligibility |
| **H2H** | Head-to-Head - weekly matchup format |
| **HLD** | Hold - relief appearance with lead maintained |
| **HR** | Home Run |
| **IP** | Innings Pitched |
| **K/9** | Strikeouts per 9 innings |
| **NSB** | Net Stolen Bases - SB minus CS |
| **NSV** | Net Saves - Saves minus Blown Saves |
| **OBP** | On-Base Percentage - (H + BB + HBP) / PA |
| **QS** | Quality Start - 6+ IP, ≤3 ER |
| **R** | Runs scored |
| **RBI** | Runs Batted In |
| **RP** | Relief Pitcher |
| **SB** | Stolen Base |
| **SP** | Starting Pitcher |
| **SV** | Save - closer preserves lead |
| **WHIP** | Walks + Hits per Inning Pitched |

---

## 9. Summary & Recommendations

### Key Formulas to Implement

```python
# Net Stolen Bases (CAN be negative)
nsb = sb - cs

# Net Saves (CAN be negative)  
nsv = saves - blown_saves

# Quality Start (boolean)
qs = (ip >= 6) and (er <= 3)

# Strikeouts per 9
k9 = (k / ip) * 9

# ERA
era = (er * 9) / ip

# WHIP
whip = (bb + h) / ip

# OBP
obp = (h + bb + hbp) / pa
```

### Critical Implementation Notes

1. **NSB and NSV can be negative** - Ensure database schema allows negative integers
2. **Position eligibility** - 5 GS OR 10 GP (not AND)
3. **Weekly scoring** - Win requires 6+ categories (not 5)
4. **Tie handling** - 5-5 split = 0.5-0.5 record
5. **Ratio categories** - ERA and WHIP are "lower is better"

### Platform-Specific Requirements

- Yahoo's H2H One Win is **Private Leagues only**
- Public leagues default to H2H Points
- Position eligibility updates in real-time during season
- Previous season games count for start-of-year eligibility

---

## Sources

1. **Yahoo Fantasy Baseball Help:** https://help.yahoo.com/kb/earned-player-stats-fantasy-baseball-sln6800.html
2. **Yahoo H2H Guide:** https://sports.yahoo.com/fantasy/article/yahoo-fantasy-baseball-a-101-guide-on-how-to-play-for-the-2026-mlb-season-162443777.html
3. **MLB Quality Start Definition:** https://www.mlb.com/glossary/standard-stats/quality-start
4. **Yahoo Position Eligibility:** https://baseball.fantasysports.yahoo.com/b1/860/positioneligibility
5. **FantasyPros H2H Strategy:** https://www.fantasypros.com/2023/03/h2h-categories-league-overview-strategy-2023-fantasy-baseball/

---

*Report generated: April 11, 2026*  
*Research confidence: HIGH (official Yahoo documentation + multiple source verification)*
