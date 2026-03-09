# Fantasy Baseball App — Master Roadmap & Task Tracker

> League: Treemendous (Yahoo ID 72586) · 12 teams · H2H One Win · 18 categories · Snake Draft
> Keeper Deadline: Fri Mar 20 3:00am EDT | Draft: Mon Mar 23 7:30am EDT | 90-sec clock
> Tech: Extend CBB_Betting (FastAPI + PostgreSQL + Streamlit + Yahoo API)

---

## CRITICAL PATH TIMELINE

```
Today  Mar 9 ──► Mar 12 ──► Mar 15 ──► Mar 20 ──► Mar 23 ──► Mar 24 ──► Sep 27
       Phase 0    Data       Draft     KEEPER      DRAFT     In-Season   Playoffs
       Setup      Pipeline   Engine    DEADLINE    DAY       Begins      End
```

---

## PHASE 0 — Foundation & Data Pipeline (Mar 9–12) 🔴 CRITICAL

### 0.1 Project Scaffolding
- [ ] Create `fantasy_baseball/` package inside CBB_Betting backend
- [ ] New PostgreSQL schema tables (see schema section below)
- [ ] Yahoo Fantasy OAuth 2.0 integration (`yahoo_fantasy_api` library)
- [ ] Environment variables: `YAHOO_CLIENT_ID`, `YAHOO_CLIENT_SECRET`, `YAHOO_REFRESH_TOKEN`
- [ ] Add `fantasy_baseball/` routes to FastAPI main.py
- [ ] New Streamlit section: `dashboard/pages/6_Fantasy_Baseball.py`

### 0.2 Data Sources Integration
- [ ] **Yahoo Fantasy API**: roster, standings, waiver wire, player stats
- [ ] **FanGraphs Projections** (Steamer/ZiPS via scrape or CSV ingest): PA/IP projections
- [ ] **Baseball Reference**: historical stats for keeper valuation
- [ ] **Rotowire/ESPN**: injury feed (scrape headlines + IL moves)
- [ ] **The Odds API**: player props + game lines (for injury/lineup intelligence)
- [ ] Player universe table: ~800 MLB players with position eligibility, ADP, projections

### 0.3 Category-Weighted Value Engine
- [ ] Build `CategoryValueEngine` — 18-category H2H One Win optimizer
- [ ] Compute z-scores for each category across projected player pool
- [ ] Apply category weights based on scarcity (SB, NSV heavily scarce)
- [ ] **CRITICAL**: Batter K is NEGATIVE — high-K sluggers triple-penalized (K, AVG, OPS); prioritize contact + power players
- [ ] **CRITICAL**: Pitcher L is NEGATIVE — penalize back-end starters
- [ ] Positional scarcity adjustments (C is always shallow; SS/2B deep in 2026)

---

## PHASE 1 — Keeper Evaluation Engine (Mar 9–19) 🔴 CRITICAL

### 1.1 Keeper Analysis Module (`fantasy_baseball/keeper_engine.py`)
- [ ] Fetch current roster from Yahoo API
- [ ] Pull 2025 actuals + 2026 Steamer/ZiPS projections for each rostered player
- [ ] Calculate keeper "cost" vs "value":
  - Cost = round they'd be drafted at in a fresh draft (ADP-based)
  - Value = projected z-score total across all 18 categories
  - Net Value = Value surplus over draft cost
- [ ] Rank keepers by surplus value — output top N recommendations
- [ ] Flag injury risks, age regression candidates, breakout upside players
- [ ] Deadline alert: March 20 3:00 AM EDT

### 1.2 Keeper Decision Dashboard
- [ ] Streamlit page: keeper comparison table (sortable by surplus value)
- [ ] Visual: player projection bar charts per category contribution
- [ ] Side-by-side comparison: keep vs. draft replacement at same round

---

## PHASE 2 — Draft Intelligence Engine (Mar 12–22) 🔴 CRITICAL

### 2.1 Pre-Draft Ranking System
- [ ] Merge ADP (Yahoo, NFBC, FantasyPros consensus) with projected z-scores
- [ ] Compute Value Over Replacement Player (VORP) per position slot
- [ ] Roster construction optimizer: 18-category balance analysis
  - Target: competitive in 14+ of 18 categories each week
  - Identify category pairs that correlate (TB/HR/RBI cluster; SB is independent)
- [ ] Generate tiered big board (not just a flat list) — tier breaks are draft strategy
- [ ] Export: ranked CSV + JSON for live draft tool

### 2.2 Live Draft Assistant (Mar 22 MUST BE READY) 🔥
- [ ] **90-second clock is the hard constraint** — all recommendations must fire in <10 seconds
- [ ] `DraftAssistant` class: tracks live picks, updates available player pool in real-time
- [ ] Input: current pick number, roster state, remaining available players
- [ ] Output: top-5 recommended picks with rationale + projected category impact
- [ ] Snake draft logic: knows your next pick position (e.g., pick 7 in round 1, pick 16 in round 2)
- [ ] "Auto-reach" alerts: warns when a player is being drafted significantly above their tier
- [ ] Real-time roster balance tracker: shows current category strengths/gaps as draft progresses
- [ ] Streamlit live draft UI: shows recommendations, remaining needs, auto-updates on pick entry
- [ ] CLI fallback mode: terminal output for quick access during draft

### 2.3 Draft Strategy Framework
- [ ] Round 1–3: Elite hitters with multi-category contribution (HR+R+RBI+TB, elite AVG/OPS)
- [ ] Round 4–6: Ace SPs (W, K, K/9, QS, ERA, WHIP — hits 6 pitching cats)
- [ ] Round 7–10: Stolen base assets (NSB is scarce — stock up mid-rounds)
- [ ] Round 11–15: Streaming-friendly SP options (verify: do they minimize L category?)
- [ ] Round 16–20: Closer/NSV pipeline, high-K/9 relievers
- [ ] Implement "handcuff" logic for high-injury-risk stars

---

## PHASE 3 — In-Season Management (Mar 24 – Sep 27)

### 3.1 Weekly Lineup Optimizer
- [ ] Pull scheduled games, probable pitchers, ballpark factors
- [ ] Optimize batter starts: maximize category contribution vs. opposing pitcher
- [ ] SP starts: project W probability + QS probability + K/9 + ERA/WHIP risk
- [ ] RP/Closer: NSV optimization — prioritize high-save-opportunity arms
- [ ] Min innings pitched enforcer: track weekly IP toward 18-inning minimum
- [ ] Respect 8 max acquisition/week limit in waiver recommendations

### 3.2 Waiver Wire Intelligence Engine
- [ ] Daily ingestion: Yahoo waiver wire availability + player stat updates
- [ ] `WaiverRankingEngine`: rank free agents by category scarcity + immediate need
- [ ] H2H matchup-specific waiver logic: what category gap does this week's opponent expose?
- [ ] Continual rolling list waiver position tracker (know your waiver priority)
- [ ] Streaming SP recommendations: games this week, home/away, bullpen rest days
- [ ] Alert: players dropped by other teams above a value threshold

### 3.3 Trade Analyzer
- [ ] Input: proposed trade (give X, receive Y)
- [ ] Output: net category impact analysis (which cats improve, which weaken)
- [ ] Rest-of-season (ROS) projections-based valuation
- [ ] "Win now vs. future" toggle for playoff push vs. rebuild modes
- [ ] Trade deadline: August 6, 2026 — alert system as deadline approaches

### 3.4 Category Tracking & Standings Projector
- [ ] Live weekly category tracker: current H2H matchup status (winning/losing which cats)
- [ ] Season-long standings projector: simulate rest of season based on current rosters
- [ ] Playoff probability calculator (top 4 qualify)
- [ ] Strength of schedule analysis: remaining opponents' projected category outputs

### 3.5 Injury Management
- [ ] Monitor IL moves daily (ESPN scrape + Odds API line moves as signal)
- [ ] Auto-generate replacement recommendations when a player hits IL
- [ ] Track IL slot usage (3 IL slots, 1 NA slot)
- [ ] Return-from-IL alerts: activate from IL before they become available to waivers

### 3.6 Performance Analytics
- [ ] Weekly report: actual vs. projected performance per player
- [ ] Category win rate tracker: rolling 4-week trend per category
- [ ] Identify "punting" opponents: teams who intentionally concede certain categories
- [ ] Track opponent tendencies for trade leverage intelligence

---

## PHASE 4 — Playoff Push (Weeks 25–26, Sep 14–27)

### 4.1 Playoff Mode
- [ ] Switch optimizer to 2-week cumulative optimization (not weekly matchup)
- [ ] Identify players with most games scheduled in playoff weeks
- [ ] NSV and NSB become even more important in short series
- [ ] Trade deadline passed — focus on waiver wire pickups only
- [ ] Category punting analysis: if you have a commanding lead in 14 cats, can you stream for 4 more?

---

## AGENT ASSIGNMENT MATRIX

### Claude (Sonnet 4.6) — Architect & Senior Engineer
**Role**: System design, complex algorithm implementation, code architecture
- CategoryValueEngine design and implementation
- Draft Assistant core logic (VORP, tier calculations, snake pick optimization)
- Trade Analyzer category-delta engine
- Integration architecture (Yahoo API ↔ PostgreSQL ↔ Streamlit)
- All production-grade Python code
- Database schema design

### Gemini CLI — Big Data Analyst
**Superpower**: ~1M token context window → process entire player databases in one shot
- **Pre-draft**: Ingest full 800-player pool CSV + all projection systems → unified ranking
- **Weekly analysis**: Full 12-team league state analysis (all rosters + all stats)
- **Trade analysis**: Process full ROS projections for all players involved in multi-player deals
- **Category correlations**: Analyze full season data to find category correlation patterns
- **Prompt template**: Feed entire FanGraphs Steamer projections CSV + league settings → ranked output
- **Delegation trigger**: Any task requiring >50K tokens of player data context

### Kimi CLI — Research & Deep Analysis
**Superpower**: Long context + strong analytical reasoning + fast iteration
- **Keeper evaluation**: Deep dive per-player analysis with full career trajectory context
- **Opponent scouting**: Analyze each opponent's roster tendencies, trade patterns
- **Streaming pitcher research**: Process weekly start quality reports
- **Rules deep-dive**: Interpret Yahoo scoring edge cases (e.g., exactly how NSV is calculated)
- **Draft prep memos**: Generate category-by-category position scarcity reports
- **Delegation trigger**: Research tasks requiring synthesis of multiple documents

### Qwen via OpenClaw (Local) — Real-Time Decisioner
**Superpower**: No rate limits, no latency, private, fast inference — runs DURING LIVE DRAFT
- **Live draft assistant**: Sub-5-second pick recommendations during 90-second clock
- **Daily lineup**: Morning lineup decisions (fast, repeated, low-latency)
- **Waiver wire quick checks**: "Should I add Player X?" rapid yes/no with rationale
- **IP tracker**: Real-time innings count warnings during the week
- **Alert triaging**: Classify injury news headlines (start/sit impact)
- **Delegation trigger**: Any task that repeats daily, requires <5s response, or runs on personal data

### The Odds API — Market Intelligence Layer
**Role**: Not an LLM, but a signal source for all agents
- **Injury intelligence**: Sharp line moves signal news before official announcements
- **Pitcher usage**: Game total movement signals bullpen usage/SP removal risk
- **Player props**: K/9, hits, RBI props signal Vegas projection vs. your model
- **Weather/Vegas consensus**: Game cancellation/postponement signals
- Integration: Feed into DraftAssistant and WaiverRankingEngine as confidence modifiers

---

## DATABASE SCHEMA (New Tables)

```sql
-- Core player registry
fantasy_players (
    id, yahoo_player_id, name, mlb_team, positions[],
    batting_hand, age, service_years, updated_at
)

-- Season projections (multiple systems)
player_projections (
    id, player_id, projection_system, season,
    -- Batting
    pa, ab, r, h, hr, rbi, sb, cs, bb, k, tb, avg, obp, slg, ops,
    -- Pitching
    g, gs, ip, w, l, sv, bs, qs, k9, era, whip, hr9,
    -- Derived
    z_score_total, category_ranks JSONB,
    created_at
)

-- League roster state (synced from Yahoo)
league_rosters (
    id, yahoo_league_id, team_id, team_name, player_id,
    roster_slot, acquisition_type, acquired_date, synced_at
)

-- Keeper decisions
keeper_decisions (
    id, player_id, team_id, season, round_cost,
    projected_surplus_value, decision, decided_at, notes
)

-- Draft board
draft_board (
    id, player_id, overall_rank, position_rank, tier,
    adp_yahoo, adp_nfbc, adp_consensus,
    vorp, projected_z_total,
    is_available BOOLEAN, drafted_by_team_id, draft_pick_number,
    created_at, updated_at
)

-- Live draft picks
draft_picks (
    id, draft_id, pick_number, round, team_id,
    player_id, pick_time, was_recommended BOOLEAN
)

-- Weekly lineup decisions
lineup_decisions (
    id, team_id, week_number, player_id, roster_slot,
    game_date, decision_rationale, actual_stats JSONB
)

-- Waiver wire actions
waiver_actions (
    id, team_id, week_number, player_added_id, player_dropped_id,
    waiver_priority_used, category_target, result, created_at
)

-- H2H matchup tracking
matchup_results (
    id, week_number, team1_id, team2_id,
    categories_won_team1, categories_won_team2, categories_tied,
    final_result, team1_stats JSONB, team2_stats JSONB
)

-- Trade log
trades (
    id, week_number, team1_id, team2_id,
    team1_gives JSONB, team2_gives JSONB,
    category_delta JSONB, accepted BOOLEAN, processed_date
)
```

---

## KEY STRATEGIC INSIGHTS (This League)

### Category Quirks vs. Standard Leagues
| Category | Standard Value | This League Adjustment |
|----------|---------------|----------------------|
| Batter K | Negative (strikeouts bad) | **NEGATIVE per Yahoo but verify** — standard penalty; confirm in league settings |
| Pitcher L | Often ignored | **NEGATIVE** — stacks losses against you |
| NSV | Standard saves | Save - BS = net; avoid volatile closers |
| NSB | Standard SB | Caught stealing hurts; target high SB% |
| K/9 | Strikeout ratio | Premium on SP with high K rates |
| TB | Total Bases | Rewards SLG, doubles/HR hitters |
| H | Hits | Unusual standalone cat — rewards high-contact AVG |
| QS | Quality Starts | Rewards workhorse SPs over mediocre volume SPs |

### H2H One Win Strategy
- You need to WIN MORE CATEGORIES than opponent each week
- Winning 10-8 = same W as winning 18-0
- **Don't over-invest** in any single category (diminishing returns)
- Target: be in top-6 of all 18 categories (consistently competitive)
- Identify which categories opponents are weak in and stream to attack

### Scarcity Stack (2026 Market)
- **NSB**: Stolen base leaders are rare; prioritize rounds 3-7
- **NSV**: Elite closers = late 1st/early 2nd round; tier cliff is steep
- **K/9**: High-K SPs scarce; stack with K hitters for double K benefit
- **C**: Catcher is always the weakest position — ADP discount is real

---

## DEVELOPMENT SPRINT TIMELINE

### Week 1 (Mar 9–15): Infrastructure + Data
- [ ] Day 1-2: Yahoo OAuth setup, DB schema, player data ingestion
- [ ] Day 3-4: CategoryValueEngine + z-score calculator
- [ ] Day 5-6: Keeper evaluation engine + dashboard
- [ ] Day 7: Testing + Keeper report generation

### Week 2 (Mar 16–22): Draft Engine
- [ ] Day 8-9: Pre-draft rankings + VORP + tier builder
- [ ] Day 10-11: Live draft assistant (core logic + 90-sec constraint)
- [ ] Day 12-13: Streamlit draft UI + CLI fallback
- [ ] Day 14 (Mar 22 EOD): Full dry run — simulate mock draft

### Week 3+ (Mar 24+): In-Season
- [ ] Week 3: Lineup optimizer + waiver engine
- [ ] Week 4: Trade analyzer + matchup tracker
- [ ] Week 5: Category standings projector
- [ ] Ongoing: Weekly iteration based on performance

---

## REVIEW SECTION
*(To be filled after each phase)*

---
_Last updated: 2026-03-09_
