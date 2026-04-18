# UI Specification Contract Audit and Gated Implementation Plan

> Date: April 17, 2026 | Author: Claude Code (Master Architect)
> Scope: The UI specification is the authoritative contract. The backend serves it. This audit maps every required field and interaction to current layer readiness, identifies all gaps, and produces a gated build sequence.

> **League format (immutable):** 10-team H2H One Win, 18 categories (Batting: R, H, HR, RBI, K(B), TB, AVG, OPS, NSB; Pitching: W, L, HRA, K, ERA, WHIP, K/9, QS, NSV), 18-IP minimum, 8 weekly acquisitions, 1-day rolling waivers, roster C/1B/2B/3B/SS/LF/CF/RF/Util/SP×2/RP×2/P×3/BN×4/IL×3/NA.

---

## 1. Master Data Contract

Every field the UI specification requires, organized by entity. Source layer indicates where the field originates. Freshness indicates how stale it can be before the UI must flag it.

### 1A. Global Header Entity

Required on every page.

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| GH-1 | Current matchup week number | int | L2 (Yahoo scoreboard) | Daily | This-week |
| GH-2 | My season record (W-L-T) | string | L2 (Yahoo standings) | Daily | Season |
| GH-3 | Opponent team name | string | L2 (Yahoo scoreboard) | Weekly | This-week |
| GH-4 | Projected matchup outcome (e.g. "10-8 W") | string | L4 (H2H Monte Carlo over L3 projections) | Daily | This-week |
| GH-5 | Day of week + days remaining in matchup period | string | L1 (pure date math over league schedule) | Real-time | This-week |
| GH-6 | Acquisitions used this week | int | L2 (Yahoo transactions, filtered by week) | Real-time | This-week |
| GH-7 | Acquisitions remaining this week | int | L1 (8 − GH-6) | Real-time | This-week |
| GH-8 | Acquisition limit (8) | int | L0 (league config constant) | Static | Season |
| GH-9 | Visual flag when acquisitions ≥ 6 | bool | L1 (GH-6 ≥ 6) | Real-time | This-week |
| GH-10 | IL slots used | int | L2 (Yahoo roster, count IL/IL10/IL60) | Daily | Today |
| GH-11 | IL slots total (3) | int | L0 (league config constant) | Static | Season |
| GH-12 | Weekly IP accumulated | float | L2 (Yahoo matchup scoreboard, stat_id 50) | Daily | This-week |
| GH-13 | IP minimum (18) | float | L0 (league config constant) | Static | Season |
| GH-14 | IP pace flag (behind/on-track/ahead) | enum | L1 (GH-12 vs GH-13 vs days remaining) | Daily | This-week |
| GH-15 | League name | string | L2 (Yahoo league metadata) | Static | Season |

### 1B. Category Status Tag Entity

Used everywhere categories appear.

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| CS-1 | Category name (one of 18) | string | L0 (fantasy_stat_contract) | Static | — |
| CS-2 | Status tag | enum: LOCKED_WIN, LOCKED_LOSS, BUBBLE, LEANING_WIN, LEANING_LOSS | L3/L4 (classification over projected finals + Monte Carlo) | Daily | This-week |
| CS-3 | Consistent color/shape/placement | UI convention | L6 | — | — |

### 1C. Canonical Player Row Entity

Used everywhere a player appears. Every field is required for every player on every page.

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| PR-1 | Player name | string | L2 (Yahoo / BDL) | Static | — |
| PR-2 | Team abbreviation | string | L2 (Yahoo / BDL) | Static | — |
| PR-3 | Position eligibility (all slots, not just primary) | list[string] | L2 (Yahoo eligible_positions) | Daily | — |
| PR-4 | Today's status: playing / not playing / probable / IL / minors | enum | L2 (Yahoo roster status + MLB schedule for today) | Real-time | Today |
| PR-5 | If pitcher + probable today: opponent | string | L2 (probable_pitchers table) | Daily | Today |
| PR-6 | If pitcher + probable today: home/away | string | L2 (probable_pitchers or schedule) | Daily | Today |
| PR-7 | If pitcher + probable today: projected K | float | L3 (player projection × matchup context) | Daily | Today |
| PR-8 | If pitcher + probable today: projected ERA impact on team weekly ERA | float | L3/L4 (ratio risk calculation) | Daily | Today |
| PR-9 | If hitter + playing today: opponent | string | L2 (MLB schedule) | Daily | Today |
| PR-10 | If hitter + playing today: home/away | string | L2 (MLB schedule) | Daily | Today |
| PR-11 | If hitter + playing today: opposing SP handedness | string | L2 (probable_pitchers) | Daily | Today |
| PR-12 | If hitter + playing today: projected impact | float | L3 (elite_lineup_scorer or projection) | Daily | Today |
| PR-13 | Season stats in all 18 league categories | dict[cat, float] | L2 (Yahoo player stats or mlb_player_stats aggregated) | Daily | Season |
| PR-14 | Last 7-day rolling stats in league categories | dict[cat, float] | L3 (player_rolling_stats, window=7) | Daily | Last-7 |
| PR-15 | Last 15-day rolling stats in league categories | dict[cat, float] | L3 (player_rolling_stats, window=14 — closest available) | Daily | Last-15 |
| PR-16 | Last 30-day rolling stats in league categories | dict[cat, float] | L3 (player_rolling_stats, window=30) | Daily | Last-30 |
| PR-17 | Rest-of-season projection in league categories | dict[cat, float] | L3 (player_projection table) | Daily | ROS |
| PR-18 | Rest-of-week projection in league categories | dict[cat, float] | L3 (ROW projection pipeline) | Daily | This-week |
| PR-19 | Ownership % across the league | float | L2 (Yahoo percent_owned) | Daily | — |
| PR-20 | Injury status | string (Healthy / IL10 / IL60 / DTD / OUT) | L2 (Yahoo injury metadata) | Daily | — |
| PR-21 | Injury return timeline | string | L2 (Yahoo injury notes or news) | Daily | — |
| PR-22 | Freshness timestamp (hover/tap accessible) | datetime | L2→L5 (propagated from source) | Per-update | — |

### 1D. Matchup Scoreboard Entity (Page 1)

One row per category × 18 categories.

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| MS-1 | Category name | string | L0 | Static | — |
| MS-2 | My current value | float | L2 (Yahoo scoreboard) | Daily | This-week |
| MS-3 | Opponent current value | float | L2 (Yahoo scoreboard) | Daily | This-week |
| MS-4 | Current margin (+/−) | float | L1 (MS-2 − MS-3, sign-aware for lower-is-better) | Daily | This-week |
| MS-5 | My projected final value | float | L3 (current + ROW projection aggregated for my roster) | Daily | This-week |
| MS-6 | Opponent projected final value | float | L3 (current + opponent ROW projection) | Daily | This-week |
| MS-7 | Projected final margin | float | L1 (MS-5 − MS-6) | Daily | This-week |
| MS-8 | Status tag (CS-2) | enum | L3/L4 | Daily | This-week |
| MS-9 | Flip probability % | float | L4 (Monte Carlo over projected finals) | Daily | This-week |
| MS-10 | Delta-to-flip | string (e.g. "Need +3 HR" or "Keep ERA < 3.85") | L1 (pure math over projected margins + opponent) | Daily | This-week |
| MS-11 | Games remaining (for counting stats) | int | L2 (schedule) | Daily | This-week |
| MS-12 | IP remaining vs minimum (for pitching categories) | float | L1 (GH-13 − GH-12) | Daily | This-week |
| MS-13 | Categories currently won / lost / tied (header) | int / int / int | L1 (count over MS-4 signs) | Daily | This-week |
| MS-14 | Categories projected to win / lose / tie (header) | int / int / int | L1 (count over MS-7 signs) | Daily | This-week |
| MS-15 | Overall matchup win probability (header) | float | L4 (Monte Carlo) | Daily | This-week |
| MS-16 | Historical overlay: this category in last 4 weeks | list[float] | L2 (Yahoo past matchup results) | Weekly | Last-4-weeks |

### 1D-I. Matchup Scoreboard Interactions

| # | Interaction | Data Required |
|---|------------|---------------|
| MI-1 | Click category row → drill into contributing players, remaining games, bench/waiver options that could improve | Player rows (1C) filtered to category contributors + waiver candidates scored by marginal value for that category |
| MI-2 | Sort by: flip probability, margin, category type | L1 (client-side sort over MS-* fields) |
| MI-3 | Filter: bubbles only / pitching only / hitting only | L1 (client-side filter over CS-2 and category type) |
| MI-4 | Historical overlay for selected category | MS-16 data |

### 1E. My Roster Entity (Page 2)

One row per roster slot (25 slots total: C/1B/2B/3B/SS/LF/CF/RF/Util/SP×2/RP×2/P×3/BN×4/IL×3/NA).

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| RO-1 | Slot label | string | L0 (league config) | Static | — |
| RO-2 | Canonical player row (PR-1 through PR-22) for player in slot | entity | L2/L3 | Daily | — |
| RO-3 | Today's start/sit state | enum (starting / benched) | L2 (current selected_position) | Real-time | Today |
| RO-4 | Projected category contribution today (all 18 cats) | dict[cat, float] | L3 (today's game projection per player) | Daily | Today |
| RO-5 | Flag: injured / bye / minors / not starting | enum | L2 (Yahoo status) | Real-time | Today |
| RO-6 | Flag: player in sub-optimal slot | bool | L4 (lineup optimizer comparison) | Daily | Today |
| RO-7 | Total projected category contribution today from active lineup (summary) | dict[cat, float] | L3 (sum of RO-4 for active slots) | Daily | Today |
| RO-8 | Slots with issues count + quick-jump | int + list | L1 (count slots where RO-5 is non-normal) | Real-time | Today |
| RO-9 | IL utilization detail: who's in IL, expected return, wasted-on-DTD flag | list[{player, return_date, wasted}] | L2 (Yahoo roster + injury data) | Daily | — |
| RO-10 | BN utilization detail: who's benched and why | list[{player, reason}] | L3/L4 (bench reasoning from optimizer) | Daily | Today |

### 1E-I. My Roster Interactions

| # | Interaction | Data Required |
|---|------------|---------------|
| RI-1 | Swap players between slots | L5 API: roster move endpoint (Yahoo API POST) |
| RI-2 | Move to/from IL | L5 API: roster move endpoint |
| RI-3 | Drop player with matchup impact preview | L4: expected-category-wins delta calculation |
| RI-4 | "Optimize lineup" → suggested swaps with category-wins delta | L4 (lineup optimizer + H2H sim) |
| RI-5 | Filter: today's active / pitchers with starts this week / hitters playing today | L1/L2 (client-side filter) |
| RI-6 | Views: Today / Week (full schedule) / Projections (ROW + ROS) | L2 (schedule), L3 (projections) |

### 1F. Waiver Wire Entity (Page 3)

One row per available player.

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| WV-1 | Canonical player row (PR-1 through PR-22) | entity | L2/L3 | Daily | — |
| WV-2 | Marginal value score: projected expected-category-wins gained if added | float | L4 (MCMC sim: win_prob_after − win_prob_before, expressed as category count delta) | Daily | This-week |
| WV-3 | Bubble categories this player would help flip (listed explicitly) | list[string] | L3/L4 (category classification + player category contribution) | Daily | This-week |
| WV-4 | Waiver claim position / FA status | string | L2 (Yahoo waiver/FA data) | Real-time | — |
| WV-5 | Drop candidate suggestion: who to drop if adding this player | string (player name) | L4 (drop candidate logic) | Daily | — |
| WV-6 | Drop impact: expected-category-wins change from the drop | float | L4 (MCMC sim) | Daily | This-week |
| WV-7 | Trending: % of leagues adding/dropping in last 24h | float | L2 (Yahoo transaction trends) | Daily | — |
| WV-8 | Games remaining this week | int | L2 (schedule) | Daily | This-week |

### 1F-I. Waiver Wire Interactions

| # | Interaction | Data Required |
|---|------------|---------------|
| WI-1 | Add player → auto-surface drop candidates with impact | L4 (drop logic + MCMC) |
| WI-2 | Add to watchlist | L5 (local state or DB) |
| WI-3 | Preview impact: "If I add X and drop Y, projected record becomes..." | L4 (full MCMC re-simulation) |
| WI-4 | Acquisition counter visible and decrements on claim | GH-6, GH-7 |
| WI-5 | Warning when acquisitions ≥ 6 | GH-9 |

### 1F-II. Waiver Wire Filters (all required)

| # | Filter | Source |
|---|--------|--------|
| WF-1 | Position (C/1B/2B/3B/SS/OF/LF/CF/RF/Util/SP/RP/P/multi-eligible) | L2 (Yahoo eligible_positions) |
| WF-2 | Handedness | L2 (Yahoo player metadata) |
| WF-3 | Team | L2 |
| WF-4 | Games remaining this week: 0, 1-2, 3-4, 5+ | L2 (schedule) |
| WF-5 | Probable pitcher status | L2 (probable_pitchers) |
| WF-6 | Availability: FA / on waivers / claim deadline | L2 (Yahoo) |
| WF-7 | Rostered % | L2 (Yahoo percent_owned) |
| WF-8 | Category impact filter: "help my bubble categories" and per-category | L3/L4 |
| WF-9 | Hide my players | L1 (client-side exclusion) |
| WF-10 | Hide injured / IL-bound | L2 (Yahoo injury data) |

### 1G. Probable Pitchers / Streaming Entity (Page 4)

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| PP-1 | Canonical player row (PR-1 through PR-22) | entity | L2/L3 | Daily | — |
| PP-2 | Next start date | date | L2 (probable_pitchers) | Daily | This-week |
| PP-3 | Opponent team | string | L2 (probable_pitchers) | Daily | Today |
| PP-4 | Park + home/away | string | L2 (schedule + park_factors) | Daily | Today |
| PP-5 | Opponent team stats: K rate, AVG, OPS | dict | L2/L3 (aggregated team stats) | Daily | Season |
| PP-6 | Opponent hand splits vs pitcher handedness | dict | L2/L3 (team platoon splits) | Weekly | Season |
| PP-7 | Park factor for HR, R | float | L2 (park_factors table) | Static | Season |
| PP-8 | Weather flag (if relevant, not dome) | string | L2 (weather_fetcher real-time) | Real-time | Today |
| PP-9 | Projected line: IP, K, ERA, WHIP, W prob, QS prob | dict | L3 (pitcher projection + matchup context) | Daily | Today |
| PP-10 | Ratio risk score: P(this start damages my ERA/WHIP) | float | L4 (ratio risk quantifier + current matchup state) | Daily | This-week |
| PP-11 | Category leverage score: contribution to bubble categories | float | L3/L4 (category classification + pitcher projected stats) | Daily | This-week |
| PP-12 | Start/skip recommendation with reasoning | {recommendation, reasoning} | L4 (composite of PP-10, PP-11, matchup state) | Daily | This-week |

### 1G-I. Streaming Interactions

| # | Interaction | Data Required |
|---|------------|---------------|
| PI-1 | Pin start to lineup planner | L5 API + UI state |
| PI-2 | Compare two streamers side-by-side with category-wins delta | L4 (MCMC: sim with pitcher A vs sim with pitcher B) |
| PI-3 | "Should I start this pitcher?" explicit recommendation | PP-12 |

### 1H. Trade Analyzer Entity (Page 5)

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| TR-1 | My category pace analysis (18 cats, season-long) | dict[cat, {pace, rank}] | L3 (season stats aggregated + league comparison) | Weekly | Season |
| TR-2 | Suggested trade targets: complementary owners | list[{owner, strengths, weaknesses}] | L3/L4 (cross-owner category analysis) | Weekly | Season |
| TR-3 | Suggested player packages | list[{send, receive, rationale}] | L4 (optimization over category surpluses/deficits) | Weekly | Season |
| TR-4 | Trade evaluator input: players out / players in | user input | — | — | — |
| TR-5 | This-week matchup impact (per-category delta) | dict[cat, float] | L4 (MCMC: current vs post-trade) | Daily | This-week |
| TR-6 | ROS category pace impact (per-category delta) | dict[cat, float] | L3 (projection aggregation change) | Daily | ROS |
| TR-7 | Roster construction impact (position scarcity, stacking) | analysis | L4 | Daily | ROS |
| TR-8 | Accept/counter/reject recommendation | {verdict, reasoning} | L4 | Daily | — |
| TR-9 | Counter-offer builder suggestions | list[{modification, reasoning}] | L4 | Daily | — |

### 1I. Season Dashboard Entity (Page 6)

| # | Field | Type | Source Layer | Freshness | Horizon |
|---|-------|------|-------------|-----------|---------|
| SD-1 | Season-to-date category rankings (my rank in each of 18 among 10 teams) | dict[cat, int] | L2/L3 (Yahoo standings + league stats) | Weekly | Season |
| SD-2 | Week-by-week matchup results with category breakdowns | list[{week, opponent, cats_won, cats_lost, per_cat_result}] | L2 (Yahoo past matchups) | Weekly | Season |
| SD-3 | Projected final standings | list[{team, projected_record}] | L4 (season-long Monte Carlo) | Weekly | Season |
| SD-4 | Structural category diagnosis (consistently lose → why) | list[{cat, diagnosis, cause}] | L3/L4 (pattern analysis) | Weekly | Season |
| SD-5 | Decision log: adds/drops/trades with EV-vs-actual | list[{action, ev_at_time, actual_outcome}] | L2/L3 (decision_tracker + actuals) | Daily | Season |
| SD-6 | Upcoming 4 opponents with category strengths/weaknesses | list[{opponent, strong_cats, weak_cats}] | L2/L3 (schedule + opponent category data) | Weekly | Season |

### 1J. Cross-Cutting Requirements

| # | Requirement | Type | Source Layer |
|---|------------|------|-------------|
| XC-1 | Freshness timestamp on every data point (hover/tap) | datetime per field | L2→L5 propagation |
| XC-2 | Stale data visual flag (per data-type threshold) | bool per field | L5 (freshness check) |
| XC-3 | Projection uncertainty: point estimate + 10th/50th/90th percentile | {p10, p50, p90} per projection | L3 (UncertaintyRange contract exists in contracts.py) |
| XC-4 | Horizon label on every stat/projection: today / ROW / ROS | enum per field | L5 (response metadata) |
| XC-5 | Canonical player row identical across all pages | UI enforcement | L6 |
| XC-6 | Category status tags identical across all pages | UI enforcement | L6 |
| XC-7 | Constraint counters identical across all pages | UI enforcement | L6 |
| XC-8 | Mobile parity: all pages fully functional on mobile | UI enforcement | L6 |
| XC-9 | Matchup Scoreboard + My Roster optimized mobile-first | UI enforcement | L6 |

---

## 2. Layer Readiness Audit

Status key: **READY** = exists, reliable, correctly shaped. **PARTIAL** = exists but incomplete/wrong shape/stale. **MISSING** = does not exist.

### 2A. Global Header

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| GH-1 | Week number | READY | [category_tracker.py](backend/fantasy_baseball/category_tracker.py) line 32: `MatchupStatus.week` |
| GH-2 | My season record | PARTIAL | `get_standings()` at [yahoo_client_resilient.py](backend/fantasy_baseball/yahoo_client_resilient.py) line 376 returns raw data; `_parse_team()` at line 1028 does NOT extract W-L-T fields |
| GH-3 | Opponent team name | READY | Parsed from matchup at [category_tracker.py](backend/fantasy_baseball/category_tracker.py) line 78 |
| GH-4 | Projected matchup outcome | MISSING | Requires L3 ROW projections → L4 Monte Carlo aggregation. Neither pipeline exists. |
| GH-5 | Day of week + days remaining | READY | Pure date math; league week boundaries derivable from Yahoo API |
| GH-6 | Acquisitions used this week | MISSING | `get_transactions()` at [yahoo_client_resilient.py](backend/fantasy_baseball/yahoo_client_resilient.py) line 1013 fetches raw transactions; NO counting, NO week-boundary filtering, NO API endpoint |
| GH-7 | Acquisitions remaining | MISSING | Depends on GH-6 |
| GH-8 | Acquisition limit (8) | MISSING | Not codified as a league config constant anywhere |
| GH-9 | Acquisition warning flag | MISSING | Depends on GH-6 |
| GH-10 | IL slots used | PARTIAL | `count_il_slots_used()` in [waiver_edge_detector.py](backend/services/waiver_edge_detector.py) line 33 counts IL players, but not exposed via API |
| GH-11 | IL slots total (3) | MISSING | Not codified as a league config constant |
| GH-12 | Weekly IP accumulated | PARTIAL | Yahoo scoreboard returns IP as stat_id 50; [category_tracker.py](backend/fantasy_baseball/category_tracker.py) fetches it but doesn't isolate IP specifically |
| GH-13 | IP minimum (18) | MISSING | Not codified anywhere in codebase. Zero references to "18" as innings minimum. |
| GH-14 | IP pace flag | MISSING | Depends on GH-12 and GH-13 |
| GH-15 | League name | READY | Available from Yahoo league metadata |

### 2B. Category Status Tags

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| CS-1 | Category name (18) | READY | [fantasy_stat_contract.json](backend/utils/fantasy_stat_contract.json) has all 18 |
| CS-2 | Status tag classification | PARTIAL | H2H Monte Carlo at [h2h_monte_carlo.py](backend/fantasy_baseball/h2h_monte_carlo.py) lines 184-206 produces `locked_categories` (>85%), `swing_categories` (40-60%), `vulnerable_categories` (<30%). **Gaps:** (a) taxonomy mismatch — spec requires LOCKED_WIN/LOCKED_LOSS/BUBBLE/LEANING_WIN/LEANING_LOSS but code produces locked/swing/vulnerable without win/loss directionality; (b) requires projected finals as input, which don't exist; (c) not wired to any API endpoint that serves UI |
| CS-3 | Consistent styling | N/A | L6 concern — no backend dependency |

### 2C. Canonical Player Row

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| PR-1 | Name | READY | Yahoo + BDL + [player_id_mapping](backend/models.py) line 1090 |
| PR-2 | Team | READY | Same sources |
| PR-3 | Position eligibility (all slots) | READY | `eligible_positions` from Yahoo [yahoo_client_resilient.py](backend/fantasy_baseball/yahoo_client_resilient.py) line 1071 |
| PR-4 | Today's status | PARTIAL | Yahoo `selected_position` indicates IL/BN; playing-today requires cross-referencing MLB schedule (team has game today + player is in lineup). **No pipeline** joins these. |
| PR-5 | Pitcher probable today: opponent | READY | `probable_pitchers` table has 94 rows, includes opponent |
| PR-6 | Pitcher probable today: home/away | READY | Derivable from probable_pitchers + schedule |
| PR-7 | Pitcher today: projected K | PARTIAL | [elite_lineup_scorer.py](backend/fantasy_baseball/elite_lineup_scorer.py) computes pitcher scores but not isolated K projection. K/9 projection exists in `player_projection.k_per_nine`; converting to projected K requires estimated IP for the start. No such function exists. |
| PR-8 | Pitcher today: projected ERA impact on team weekly ERA | MISSING | No function computes "if this pitcher throws X IP at Y ERA, my team weekly ERA changes from A to B". This is the ratio risk quantifier — completely absent. |
| PR-9 | Hitter today: opponent | PARTIAL | Derivable from MLB schedule, but no joined pipeline exists |
| PR-10 | Hitter today: home/away | PARTIAL | Same — derivable, not joined |
| PR-11 | Hitter today: opposing SP handedness | PARTIAL | `probable_pitchers` table stores pitcher; handedness available via Yahoo or BDL player metadata. No joined lookup exists. |
| PR-12 | Hitter today: projected impact | PARTIAL | [elite_lineup_scorer.py](backend/fantasy_baseball/elite_lineup_scorer.py) lines 314-335 compute `EliteScore` per hitter-pitcher-context. Works but requires orchestration to call for each rostered hitter against their day's SP. Not pre-computed or cached. |
| PR-13 | Season stats in all 18 categories | PARTIAL | Yahoo returns season stats by stat_id. **Gaps:** stat_id mapping exists in [fantasy_stat_contract.json](backend/utils/fantasy_stat_contract.json), but no backend function assembles a clean dict of all 18 league categories per player. Season stats for pitchers (W, L, SV, HLD, QS) are in Yahoo but not parsed into our schema. |
| PR-14 | 7-day rolling stats in 18 categories | PARTIAL | [PlayerRollingStats](backend/models.py) line 1143 has window_days=7 with batting: w_avg, w_ops, w_home_runs, w_rbi, w_hits, w_net_stolen_bases, w_strikeouts_bat. **Missing from rolling stats:** R (runs), TB (total bases — computable from components), and ALL pitching decision stats: W, L, SV, HLD, QS. Only ERA, WHIP, K/9 are tracked for pitchers. That's 6 of 9 batting categories covered and 3 of 9 pitching categories covered = **9 of 18 categories in rolling stats.** |
| PR-15 | 15-day rolling stats | PARTIAL | Window=14 exists (closest). Same coverage gap as PR-14. |
| PR-16 | 30-day rolling stats | PARTIAL | Window=30 exists. Same coverage gap. |
| PR-17 | ROS projection in 18 categories | PARTIAL | [PlayerProjection](backend/models.py) line 681 has: avg, ops, hr, r, rbi, sb, era, whip, k_per_nine. `cat_scores` JSONB can hold any key but pipeline only populates a subset. **Missing projections for:** H, K(B), TB, NSB, W, L, HRA, SV/NSV, HLD, QS = **10 of 18 categories not projected.** |
| PR-18 | ROW projection in 18 categories | MISSING | No rest-of-week projection pipeline exists. [daily_briefing.py](backend/fantasy_baseball/daily_briefing.py) line 87 uses naive `current * 7` — not a real per-player ROW projection. |
| PR-19 | Ownership % | PARTIAL | Yahoo returns `percent_owned` parsed at [main.py](backend/main.py) line 5581. Known issue: was returning 0.0% for all players (March 2026). Needs verification. |
| PR-20 | Injury status | READY | Yahoo provides injury metadata; parsed in `_parse_player()` at [yahoo_client_resilient.py](backend/fantasy_baseball/yahoo_client_resilient.py) line 1044 |
| PR-21 | Injury return timeline | PARTIAL | Yahoo injury notes exist but return timeline parsing is best-effort, not structured |
| PR-22 | Freshness timestamp | MISSING | No API response currently carries per-field freshness metadata. `computed_at` exists on some tables but is not propagated to Pydantic response schemas. |

### 2D. Matchup Scoreboard

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| MS-1 | Category name | READY | L0 contract |
| MS-2 | My current value | READY | [category_tracker.py](backend/fantasy_baseball/category_tracker.py) line 153-173 |
| MS-3 | Opponent current value | READY | Same code path |
| MS-4 | Current margin | READY | L1 pure math over MS-2 and MS-3 (sign-aware for lower-is-better via `lowerIsBetter` list in [fantasy_stat_contract.json](backend/utils/fantasy_stat_contract.json)) |
| MS-5 | My projected final | MISSING | Requires ROW projection pipeline (PR-18) aggregated to team level |
| MS-6 | Opponent projected final | MISSING | Requires opponent ROW projection — opponent roster is fetchable (READY), but projection pipeline doesn't exist |
| MS-7 | Projected final margin | MISSING | Depends on MS-5 and MS-6 |
| MS-8 | Status tag | MISSING | Depends on MS-5/MS-6/MS-9 + classification function. H2H Monte Carlo exists but can't run without projected inputs. |
| MS-9 | Flip probability | MISSING | Depends on projected finals + Monte Carlo |
| MS-10 | Delta-to-flip | MISSING | No pure function exists. Needs: for counting stats, simple arithmetic; for rate stats, algebraic inversion (e.g., "at current IP, ERA must stay below X to maintain lead"). |
| MS-11 | Games remaining | MISSING | No per-team games-remaining-this-week computation. MLB schedule data available via `statsapi.schedule()` but not wired. |
| MS-12 | IP remaining vs minimum | MISSING | Depends on GH-12 and GH-13 |
| MS-13 | Categories currently won/lost/tied | READY | Derivable from MS-2/MS-3 (the CategoryTracker already does this comparison) |
| MS-14 | Categories projected to win/lose/tie | MISSING | Depends on MS-5/MS-6 |
| MS-15 | Overall win probability | MISSING | Depends on projected finals + Monte Carlo. H2H engine exists (READY in isolation) but has no projected-finals input. |
| MS-16 | Historical overlay (last 4 weeks) | PARTIAL | Yahoo scoreboard can be queried per-week. No caching or pre-computation. |

### 2E. My Roster

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| RO-1 | Slot label | READY | League config + Yahoo `selected_position` |
| RO-2 | Canonical player row | PARTIAL | See 2C — 9/18 rolling categories, 8/18 projections, no ROW |
| RO-3 | Start/sit state | READY | Yahoo `selected_position` (active slot vs BN/IL) |
| RO-4 | Projected category contribution today (18 cats) | MISSING | No function produces "Player X's projected contribution to each of 18 categories for today's game". The [elite_lineup_scorer.py](backend/fantasy_baseball/elite_lineup_scorer.py) produces a composite score, not per-category breakdown. |
| RO-5 | Issue flags | PARTIAL | Injury status from Yahoo (READY). Not-starting and bye require schedule cross-reference (PARTIAL). Minors status needs Yahoo NA-eligible check (PARTIAL). |
| RO-6 | Sub-optimal slot flag | MISSING | Requires running optimizer and comparing to current assignment. No diff function exists. |
| RO-7 | Total projected contribution today | MISSING | Depends on RO-4 |
| RO-8 | Issue slot count + quick-jump | PARTIAL | Count is pure math over RO-5. Quick-jump is L6. |
| RO-9 | IL utilization detail | PARTIAL | Who's in IL: READY. Expected return: PARTIAL (PR-21). Wasted-on-DTD: MISSING (no logic distinguishes DTD vs real IL). |
| RO-10 | BN reasoning | MISSING | No function explains why a player is benched (optimizer doesn't produce explanations) |

### 2F. Waiver Wire

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| WV-1 | Canonical player row | PARTIAL | Same gaps as 2C |
| WV-2 | Marginal value (expected-category-wins gained) | PARTIAL | [waiver_edge_detector.py](backend/services/waiver_edge_detector.py) computes `need_score` (deficit-weighted z-score). MCMC enrichment at lines 95-110 computes `win_prob_gain`. **Gap:** `win_prob_gain` is overall win probability delta, not category-count delta ("you gain 1.3 expected categories"). |
| WV-3 | Bubble categories helped | MISSING | MCMC returns `category_win_probs` dict (per-category) but no post-processing identifies which BUBBLE categories flip. The H2H classification isn't wired here. |
| WV-4 | Waiver/FA status | PARTIAL | Yahoo API returns `ownership_type` but not parsed to clean enum |
| WV-5 | Drop candidate suggestion | READY | `_weakest_safe_to_drop()` at [main.py](backend/main.py) line 5918 |
| WV-6 | Drop impact (category-wins delta) | MISSING | Drop candidate logic doesn't compute category-level impact |
| WV-7 | Trending adds/drops | PARTIAL | Yahoo `percent_owned` available. 24h trend delta not computed. |
| WV-8 | Games remaining this week | MISSING | Same as MS-11 |

### 2G. Probable Pitchers / Streaming

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| PP-1 | Canonical player row | PARTIAL | Same gaps as 2C |
| PP-2 | Next start date | READY | `probable_pitchers` table populated |
| PP-3 | Opponent | READY | In probable_pitchers record |
| PP-4 | Park + home/away | READY | park_factors table (27 parks) + schedule |
| PP-5 | Opponent team offensive stats | PARTIAL | Team-level aggregated stats not pre-computed. Would need to aggregate from mlb_player_stats by team. |
| PP-6 | Opponent hand splits vs pitcher handedness | MISSING | No team-level platoon split data. Individual platoon splits are hardcoded baselines in [elite_lineup_scorer.py](backend/fantasy_baseball/elite_lineup_scorer.py) lines 54-58 (not per-team). |
| PP-7 | Park factor for HR, R | READY | [ballpark_factors.py](backend/fantasy_baseball/ballpark_factors.py) `get_park_factor()` with DB → constant → 1.0 fallback |
| PP-8 | Weather flag | READY | [weather_fetcher.py](backend/fantasy_baseball/weather_fetcher.py) returns GameWeather with hitter_friendly_score, hr_factor, game_risk |
| PP-9 | Projected line (IP, K, ERA, WHIP, W prob, QS prob) | MISSING | No single function produces a projected game line. ERA/WHIP projectable from player_projection; K projectable from k_per_nine × estimated_ip. **W probability and QS probability do not exist anywhere.** |
| PP-10 | Ratio risk score | MISSING | No function: "P(this start flips my ERA/WHIP from win to loss given current matchup state)" |
| PP-11 | Category leverage score | MISSING | Depends on category classification (CS-2) which isn't wired |
| PP-12 | Start/skip recommendation | MISSING | No recommendation engine. Would compose PP-10 + PP-11 + matchup context. |

### 2H. Trade Analyzer

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| TR-1 | Category pace analysis | MISSING | No season-pace computation across 18 categories against league average |
| TR-2 | Suggested trade targets | MISSING | No cross-owner category analysis |
| TR-3 | Suggested player packages | MISSING | No trade optimization engine |
| TR-4 | Trade input (players in/out) | N/A (user input) | — |
| TR-5 | This-week matchup impact | MISSING | Depends on full MCMC with roster modification |
| TR-6 | ROS category pace impact | MISSING | Depends on TR-1 |
| TR-7 | Roster construction impact | MISSING | No position scarcity / stacking analysis |
| TR-8 | Accept/counter/reject | MISSING | No trade evaluation engine |
| TR-9 | Counter-offer builder | MISSING | No counter-offer logic |

### 2I. Season Dashboard

| # | Field | Status | Evidence |
|---|-------|--------|----------|
| SD-1 | Category rankings (league-wide) | PARTIAL | Yahoo standings has team stats; no aggregation to per-category rank |
| SD-2 | Week-by-week results | PARTIAL | Yahoo scoreboard queryable per-week, but no caching or structured aggregation |
| SD-3 | Projected final standings | MISSING | No season-long Monte Carlo |
| SD-4 | Structural category diagnosis | MISSING | No pattern analysis over historical category performance |
| SD-5 | Decision log with EV-vs-actual | PARTIAL | [decision_tracker.py](backend/fantasy_baseball/decision_tracker.py) records decisions to JSONL. `resolve_decision()` at line 133 tracks outcomes. But `DecisionResult` DB table is empty — job doesn't run. |
| SD-6 | Upcoming 4 opponents | PARTIAL | Yahoo schedule available; opponent category strengths not pre-computed |

### 2J. Cross-Cutting

| # | Requirement | Status | Evidence |
|---|------------|--------|----------|
| XC-1 | Freshness timestamps per data point | MISSING | No API response carries freshness metadata. DB tables have `computed_at` / `updated_at` but not propagated to Pydantic response schemas. |
| XC-2 | Stale data visual flag | MISSING | No staleness thresholds defined per data type |
| XC-3 | Projection uncertainty (p10/p50/p90) | PARTIAL | `UncertaintyRange` contract defined at [contracts.py](backend/contracts.py) lines 29-36 with `lower_80`, `upper_80`, `lower_95`, `upper_95`. No L3 pipeline populates these ranges. H2H Monte Carlo produces distributions but doesn't extract percentiles per category. |
| XC-4 | Horizon labeling | MISSING | No response metadata distinguishes today/ROW/ROS |
| XC-5-9 | UI consistency + mobile | N/A | L6 concerns — no backend dependency |

---

## 3. Gap Analysis by Layer

### Layer 0 — Immutable Decision Contracts

**What's READY:**
- 18 category definitions in [fantasy_stat_contract.json](backend/utils/fantasy_stat_contract.json) with stat_id mapping, labels, lowerIsBetter list
- `UncertaintyRange` contract in [contracts.py](backend/contracts.py) lines 29-36
- `LineupOptimizationRequest`, `PlayerValuationReport`, `ExecutionDecision` contracts in contracts.py

**What's MISSING:**
| Gap | Impact | Fields Blocked |
|-----|--------|---------------|
| **League configuration constants** — acquisition limit (8), IP minimum (18), IL slots (3), roster shape (25 slots with exact labels), waiver type, matchup period length | GH-8, GH-11, GH-13, RO-1 | All constraint displays |
| **MatchupScoreboardRow contract** — per-category shape: {category, my_current, opp_current, my_projected, opp_projected, margin, status_tag, flip_prob, delta_to_flip, games_remaining} | MS-1 through MS-16 | Entire scoreboard page |
| **CategoryStatusTag enum** — LOCKED_WIN, LOCKED_LOSS, BUBBLE, LEANING_WIN, LEANING_LOSS | CS-2 | Every category display |
| **CanonicalPlayerRow contract** — standardized shape for all 22 PR-fields | PR-1 through PR-22 | Every page |
| **ConstraintBudget contract** — {acquisitions_used, acquisitions_remaining, acquisitions_limit, ip_accumulated, ip_minimum, ip_pace_flag, il_used, il_total} | GH-6 through GH-14 | Global header |
| **FreshnessMetadata contract** — {field_name, source, fetched_at, computed_at, staleness_threshold, is_stale} | XC-1, XC-2, PR-22 | Cross-cutting freshness |

### Layer 1 — Pure Stateless Intelligence

**What's READY:**
- `lowerIsBetter` comparison logic (inverted sign for ERA, WHIP, L, K(B), HRA)
- Date math for day-of-week / days-remaining (pure stdlib)
- De-vigging / odds math in [odds_math.py](backend/core/odds_math.py)

**What's MISSING:**
| Gap | Impact | Fields Blocked |
|-----|--------|---------------|
| **Delta-to-flip calculator** — given current + projected margins, compute stat delta needed to flip a category. Counting stats: simple subtraction. Rate stats: algebraic inversion (e.g., for ERA: max_allowed_ER = ((target_ERA × (current_IP + remaining_IP)) − current_ER) / 9) | MS-10 | Scoreboard delta-to-flip |
| **Ratio risk quantifier** — P(adding this pitcher's projected start flips my ERA/WHIP from win to loss). Inputs: current team ERA/IP, projected pitcher ERA/IP, opponent ERA. | PP-10, PR-8 | Streaming ratio risk, pitcher ERA impact |
| **IP pace classifier** — given current IP, minimum IP, days remaining → behind/on-track/ahead | GH-14 | Global header IP flag |
| **Acquisition budget math** — 8 − count(adds this week) | GH-7, GH-9 | Global header acquisition display |
| **Category-count delta extractor** — from two H2H simulation results (before/after a move), compute the expected change in categories won (not just overall win prob) | WV-2, WV-6, RI-3, RI-4, PI-2, TR-5 | Marginal value displays on Waiver, Roster, Streaming, Trade pages |
| **Total Bases calculator** — H + 2B + 2×3B + 3×HR (simple but must be explicit) | PR-14/15/16 TB component | Rolling stats TB |

### Layer 2 — Data and Adaptation

**What's READY:**
- Yahoo scoreboard fetch (current matchup stats for both teams): [category_tracker.py](backend/fantasy_baseball/category_tracker.py) lines 70-173
- Yahoo roster fetch (my roster with slots, positions, IL): [yahoo_client_resilient.py](backend/fantasy_baseball/yahoo_client_resilient.py) line 561
- Opponent roster fetch: `get_roster(opponent_team_key)` line 561
- Yahoo transactions fetch: `get_transactions()` line 1013
- Probable pitchers: 94 rows, populated daily
- Park factors: 27 parks, DB-backed with fallback
- Weather: real-time via [weather_fetcher.py](backend/fantasy_baseball/weather_fetcher.py)
- Player ID mapping: yahoo_key ↔ mlbam_id ↔ bdl_id
- Statcast daily stats: 7408 rows in statcast_performances
- MLB player stats: 7249 rows in mlb_player_stats

**What's PARTIAL:**
| Gap | Current State | Fix Required |
|-----|--------------|-------------|
| **Season record (W-L-T)** | `get_standings()` returns raw Yahoo data; `_parse_team()` doesn't extract W-L-T | Extend `_parse_team()` to extract outcome_totals |
| **IP accumulated this week** | Yahoo scoreboard stat_id 50 is fetched by CategoryTracker but not isolated as a named field | Add explicit `ip_accumulated` extraction |
| **IL slot count** | `count_il_slots_used()` exists in waiver_edge_detector.py but not exposed via API | Wire to a constraint-budget endpoint |
| **Ownership %** | Parsed but was returning 0.0% (March 2026 bug) | Verify and fix Yahoo parsing |
| **Player injury data** | Injury status parsed; return timeline is free-text, not structured | Best-effort — may need news API enrichment |
| **Opponent team offensive stats** | Raw game-level data in mlb_player_stats; not aggregated to team-level season stats | Need team-stats aggregation query |

**What's MISSING:**
| Gap | Impact | Fields Blocked | Dependencies |
|-----|--------|---------------|-------------|
| **Acquisition count this week** | No counting logic applied to `get_transactions()` output | GH-6, GH-7, GH-9, WI-4, WI-5 | None — raw data available, logic needed |
| **Per-team games-remaining-this-week** | MLB schedule available via `statsapi.schedule()` at [mlb_analysis.py](backend/services/mlb_analysis.py) line 176, but no function maps team → remaining games in current matchup week | MS-11, WV-8, WF-4 | League week boundaries needed from Yahoo |
| **Per-player playing-today status** | No joined pipeline: (player → team → team has game today → player in lineup). Components exist separately. | PR-4 | schedule data + roster data join |
| **Hitter opposing-SP lookup** | No pipeline joins (hitter → today's game → opponent probable pitcher → pitcher handedness) | PR-11 | probable_pitchers + schedule join |
| **Past matchup results (weekly historical)** | Yahoo scoreboard can be queried per-week but no batch-fetch or cache | MS-16, SD-2 | Yahoo API call per historical week |
| **Team-level platoon splits** | No team aggregation of batting stats vs LHP/RHP | PP-6 | Would need FanGraphs/Statcast splits data |
| **Transaction trend data (24h adds/drops)** | Yahoo percent_owned available but delta not computed | WV-7 | Requires storing yesterday's ownership for diff |

### Layer 3 — Derived Stats and Scoring

**What's READY:**
- Player rolling stats (7/14/30-day windows, exponential decay): [PlayerRollingStats](backend/models.py) line 1143
- Player z-scores via [PlayerScore](backend/models.py) line 1218 (composite scoring with confidence)
- Bayesian projections via `BayesianProjectionUpdater` running every 6h
- Category deficits: [waiver_edge_detector.py](backend/services/waiver_edge_detector.py) lines 169-178
- Elite lineup scoring: [elite_lineup_scorer.py](backend/fantasy_baseball/elite_lineup_scorer.py) (multi-factor batter/pitcher scoring)
- Category needs from matchup: [category_tracker.py](backend/fantasy_baseball/category_tracker.py) `get_category_needs()`

**What's PARTIAL:**
| Gap | Current State | Fix Required |
|-----|--------------|-------------|
| **Rolling stats cover 9/18 categories** | Batting: avg, ops, hr, rbi, h, nsb, k(b) = 7 of 9 (missing R, TB). Pitching: era, whip, k/9 = 3 of 9 (missing W, L, HRA, K(pit), QS, NSV). | Add R and TB to batting rolling window. Add W, L, HRA, K, QS, NSV to pitching — requires these stats in the source ingestion. |
| **Projections cover 8/18 categories** | Batting: avg, ops, hr, r, rbi, sb. Pitching: era, whip, k_per_nine. Missing: H, K(B), TB, NSB, W, L, HRA, SV/NSV, HLD, QS. | Expand projection pipeline to all 18. Some (H, TB, K(B)) are derivable from existing components. Others (W, QS, NSV) require new models. |
| **Category classification (locked/swing/vulnerable)** | H2H Monte Carlo produces these at [h2h_monte_carlo.py](backend/fantasy_baseball/h2h_monte_carlo.py) lines 184-206, but (a) not directional (no win/loss), (b) requires projected finals which don't exist, (c) not wired to any API | Add directionality, wire to scoreboard |
| **UncertaintyRange** | Contract exists in contracts.py. No pipeline populates it. H2H sim produces distributions but doesn't extract percentiles. | Add percentile extraction to simulation output |

**What's MISSING:**
| Gap | Impact | Fields Blocked | Dependencies |
|-----|--------|---------------|-------------|
| **Rest-of-week (ROW) projection pipeline** | THE critical missing capability. Must produce per-player projected stat accumulation for remaining days of matchup week across all 18 categories. | MS-5, MS-6, MS-7, MS-8, MS-9, MS-14, MS-15, PR-18, RO-4, RO-7, GH-4, WV-2, WV-3, PP-11 | L2: per-player games remaining. L3: per-game projection × games remaining × context. Must handle both counting stats (additive) and rate stats (IP-weighted). |
| **Per-player per-category daily contribution projection** | "Player X is projected to contribute +0.3 HR, +1.2 TB, −0.015 to team AVG today" | RO-4, RO-7 | Requires per-game projection model for all 18 categories |
| **Opponent ROW projection** | Same as my ROW but for opponent roster. Opponent roster is fetchable (L2 READY). Needs same projection logic. | MS-6 | Opponent roster (READY) + ROW pipeline (MISSING) |
| **Team-level ROW aggregation** | Sum per-player ROW projections to team totals, handling rate stats correctly (IP-weighted ERA/WHIP/K9, AB-weighted AVG/OPS) | MS-5, MS-6 | Per-player ROW (MISSING) |
| **Pitcher projected game line** | IP, K, ERA, WHIP, W probability, QS probability for a single start | PP-9 | Partial: IP/K/ERA/WHIP derivable from projections × estimated IP. W probability and QS probability are new models needed. |
| **Pitcher W probability model** | P(pitcher earns a win in this start) | PP-9 | Requires: team implied runs, opponent implied runs, pitcher projection, bullpen quality. Complex model. |
| **Pitcher QS probability model** | P(pitcher goes ≥ 6 IP with ≤ 3 ER) | PP-9 | Requires: pitcher IP distribution, ER distribution. Could bootstrap from historical Statcast. |
| **Season category pace** | Per-category rate over season, compared to league average, projected to end-of-season | TR-1, SD-1, SD-4 | Yahoo season stats (READY) + league stats aggregation (PARTIAL) |
| **Freshness propagation** | Every derived stat must carry `computed_at` + source freshness | XC-1, XC-2, PR-22 | Requires schema changes across all L3 output tables |

### Layer 4 — Decision Engines and Simulation

**What's READY:**
- H2H Monte Carlo simulation: [h2h_monte_carlo.py](backend/fantasy_baseball/h2h_monte_carlo.py) — produces win prob, per-category win probs, locked/swing/vulnerable classification. Works if given pre-aggregated roster projections.
- Lineup optimization (batting only): OR-Tools ILP solver in [lineup_constraint_solver.py](backend/fantasy_baseball/lineup_constraint_solver.py) — solves 9 batting slots.
- Smart lineup selection: [smart_lineup_selector.py](backend/fantasy_baseball/smart_lineup_selector.py) — multi-factor scoring with category needs awareness.
- Drop candidate selection: `_weakest_safe_to_drop()` in [main.py](backend/main.py) line 5918.
- MCMC roster-move simulation: [mcmc_simulator.py](backend/fantasy_baseball/mcmc_simulator.py) `simulate_roster_move()` — returns win_prob delta.

**What's PARTIAL:**
| Gap | Current State | Fix Required |
|-----|--------------|-------------|
| **Lineup solver covers batting only** | [lineup_constraint_solver.py](backend/fantasy_baseball/lineup_constraint_solver.py) lines 40-80 handles C/1B/2B/3B/SS/OF×3/Util. No pitcher slot optimization (SP×2/RP×2/P×3). | Extend solver to full 25-slot roster. |
| **MCMC returns win-prob delta, not category-count delta** | `simulate_roster_move()` returns `win_prob_gain: float`. The UI spec requires "expected category wins gained" (e.g., "+1.3 categories"). | Post-process `category_win_probs` from before/after sims to compute per-category flip count delta. |
| **Drop candidate uses z-score, not category impact** | `_weakest_safe_to_drop()` picks lowest z_score. Doesn't consider which categories the dropped player is propping up. | Weight drop selection by category leverage (don't drop a player holding up a BUBBLE category). |

**What's MISSING:**
| Gap | Impact | Fields Blocked | Dependencies |
|-----|--------|---------------|-------------|
| **Ratio risk quantifier** (L4 orchestration over L1 pure math + L3 projections + L2 matchup state) | PP-10, PP-12, PR-8 | Streaming page, pitcher ERA impact | L1 ratio-risk pure function (MISSING) + L3 pitcher projection (PARTIAL) + L2 matchup state (READY) |
| **Category leverage scorer** | PP-11 | Streaming page category contribution | Depends on CS-2 classification (PARTIAL) |
| **Start/skip recommendation engine** | PP-12 | Streaming recommendations | Composes ratio risk + category leverage + matchup state. All inputs are missing or partial. |
| **Trade evaluation engine** | TR-5 through TR-9 | Entire trade analyzer page | Requires: MCMC with roster modification, ROS projections for all 18 cats, position scarcity model |
| **Trade finder / package suggestion** | TR-2, TR-3 | Trade finder proactive view | Requires: cross-owner category surplus/deficit analysis, multi-player optimization |
| **Season-long Monte Carlo** | SD-3 | Projected final standings | Requires: full league roster projections, remaining schedule |
| **Structural category diagnosis** | SD-4 | Season dashboard diagnosis | Requires: season-long pattern analysis over historical weekly results |
| **Full-roster optimizer** | RI-4 (optimize button) | My Roster page optimize button | Extends current batting-only ILP to 25-slot roster |

### Layer 5 — APIs and Service Presentation

**What's READY:**
- `GET /api/fantasy/decisions` — decision output with auth ([routers/fantasy.py](backend/routers/fantasy.py))
- `GET /api/fantasy/players/{id}/scores` — player scores by window
- `GET /api/fantasy/lineup/{date}` — daily lineup
- `GET /api/fantasy/waiver` — waiver targets + category deficits
- `GET /api/fantasy/waiver/recommendations` — ranked roster moves
- `GET /api/fantasy/roster` — team roster with z-scores
- `GET /api/fantasy/matchup` — raw matchup category data
- `POST /api/fantasy/matchup/simulate` — H2H Monte Carlo simulation
- `GET /api/fantasy/briefing/{date}` — daily morning briefing
- `GET /api/dashboard` — dashboard aggregation
- Auth via `verify_api_key` on all fantasy endpoints

**What's PARTIAL:**
| Gap | Current State | Fix Required |
|-----|--------------|-------------|
| **Matchup endpoint returns raw data, not scoreboard shape** | `GET /api/fantasy/matchup` returns category-by-category stats but no projections, no status tags, no margins, no flip probability | New endpoint or expansion: `GET /api/fantasy/scoreboard` returning MatchupScoreboardRow contract |
| **Waiver endpoint doesn't reference bubble categories** | Returns `CategoryDeficitOut` but no LOCKED/BUBBLE labels | Wire CS-2 status tags into waiver response |
| **Lineup endpoint doesn't return per-category contribution** | Returns player with `lineup_score` composite but not per-category breakdown | Add `category_contributions: dict[str, float]` to lineup response |
| **Dashboard matchup preview is stubbed** | `_get_matchup_preview()` in [dashboard_service.py](backend/services/dashboard_service.py) returns None | Implement or replace with scoreboard endpoint |

**What's MISSING:**
| Gap | Impact | Endpoint Needed |
|-----|--------|----------------|
| **Constraint budget endpoint** | GH-6 through GH-14 | `GET /api/fantasy/budget` → ConstraintBudget contract |
| **Scoreboard endpoint** | MS-1 through MS-16 | `GET /api/fantasy/scoreboard` → 18 MatchupScoreboardRow objects + header summary |
| **Full canonical player row endpoint** | PR-1 through PR-22 | `GET /api/fantasy/players/{id}/full` → CanonicalPlayerRow contract, or embed in all responses |
| **Streaming recommendation endpoint** | PP-1 through PP-12 | `GET /api/fantasy/streaming` → list of scored pitcher starts |
| **Trade evaluation endpoint** | TR-4 through TR-9 | `POST /api/fantasy/trade/evaluate` → trade impact analysis |
| **Trade finder endpoint** | TR-1 through TR-3 | `GET /api/fantasy/trade/suggestions` → suggested packages |
| **Season dashboard endpoint** | SD-1 through SD-6 | `GET /api/fantasy/season` → season dashboard aggregation |
| **Freshness-decorated responses** | XC-1, XC-2 | Every endpoint must include freshness metadata |
| **Uncertainty-decorated projections** | XC-3 | Every projection field must offer {p10, p50, p90} |
| **Horizon-labeled responses** | XC-4 | Every stat/projection field must carry horizon tag |
| **Roster mutation endpoints** | RI-1, RI-2 | `POST /api/fantasy/roster/move` — swap slots, move to/from IL (proxying Yahoo API) |
| **Waiver claim endpoint** | WI-1 | `POST /api/fantasy/waiver/claim` — add player + drop, decrement acquisition counter |

---

## 4. Layer 4 Gate Review

The architecture gates Layer 4: "Hold until the first Layer 3 objective is stable." The UI spec makes clear which L4 engines are required for which pages. Here's the review:

### Required L4 Engines by Page

| L4 Engine | Pages That Require It | Can Page Function Without It? |
|-----------|----------------------|-------------------------------|
| **H2H Monte Carlo** (existing) | Matchup Scoreboard (MS-9, MS-15), My Roster (RI-3, RI-4), Waiver Wire (WV-2, WI-3), Streaming (PI-2), Trade (TR-5) | **Scoreboard:** No — flip probabilities and overall win prob are core fields. **Roster:** Degraded — optimize button won't show category-wins delta. **Waiver:** Degraded — can show need_score but not expected-wins delta. **Streaming:** No — comparison requires simulation. **Trade:** No — evaluation is meaningless without simulation. |
| **Lineup optimizer** (existing, batting-only) | My Roster (RI-4, RO-6) | Degraded — can show roster but no optimize button, no sub-optimal-slot flag |
| **MCMC roster-move simulator** (existing) | Waiver Wire (WV-2, WI-3), My Roster (RI-3) | No — marginal value is the primary ranking signal for waivers |
| **Ratio risk quantifier** (MISSING) | Streaming (PP-10, PP-12), My Roster (PR-8) | **Streaming:** No — ratio risk is the defining feature of the streaming page. **Roster:** Degraded — pitcher ERA impact not shown. |
| **Category leverage scorer** (MISSING) | Streaming (PP-11), Waiver Wire (WV-3) | No — without leverage scoring, streaming is generic rankings |
| **Start/skip recommender** (MISSING) | Streaming (PP-12) | No — the page's purpose is the recommendation |
| **Trade evaluation engine** (MISSING) | Trade Analyzer (all TR-fields) | No — the entire page is this engine's output |
| **Season-long Monte Carlo** (MISSING) | Season Dashboard (SD-3) | Degraded — can show history but not projections |

### Gate Lift Recommendation

**The Layer 4 gate MUST partially lift for Priority 1 pages (Scoreboard + Roster).** Here's why:

1. The H2H Monte Carlo simulation is the only way to produce flip probabilities (MS-9) and overall win probability (MS-15). These are core scoreboard fields, not nice-to-haves.
2. However, the H2H Monte Carlo simulation already exists and works. The gate was set to protect against unstable L3 inputs. Once L3 ROW projections are stable, the gate criterion is met.

**Lift criteria:**
- ROW projection pipeline produces stable, plausible output for all 18 categories for at least one complete matchup week
- ROW projections pass automated sanity checks (no negative counting stats, rate stats within plausible bounds, projection magnitudes proportional to games remaining)
- H2H Monte Carlo fed with ROW-projected finals produces non-degenerate results (not all categories locked, win probability between 0.1 and 0.9 for typical matchups)

**Engines to lift:**
- Phase 1 (with Scoreboard + Roster): H2H Monte Carlo, MCMC roster-move simulator, lineup optimizer
- Phase 2 (with Waiver + Streaming): Ratio risk quantifier, category leverage scorer, start/skip recommender
- Phase 3 (with Trade + Season): Trade engine, season-long Monte Carlo

---

## 5. Ranked Blocker List

Ranked by: how many Priority 1-2 page fields are blocked. Matchup Scoreboard and My Roster are P1. Waiver Wire and Streaming are P2.

| Rank | Blocker | Layer | P1 Fields Blocked | P2 Fields Blocked | Total Fields Blocked |
|------|---------|-------|--------------------|--------------------|--------------------|
| **1** | **ROW projection pipeline does not exist** | L3 | MS-5, MS-6, MS-7, MS-8, MS-9, MS-10, MS-14, MS-15, PR-18, RO-4, RO-7, GH-4 = **12** | WV-2, WV-3, PP-9, PP-10, PP-11, PP-12 = **6** | **18** |
| **2** | **Rolling stats cover only 9 of 18 categories** | L3 | PR-14, PR-15, PR-16 (each missing 9 cats) = **3 fields × 9 cat gaps = 27 cell gaps** | Same for all waiver/streaming player rows | **27+ cell gaps** |
| **3** | **Projections cover only 8 of 18 categories** | L3 | PR-17 (missing 10 cats), feeds MS-5/MS-6 | Same | **10+ cell gaps** |
| **4** | **Per-player games-remaining-this-week missing** | L2 | MS-11, feeds ROW pipeline (Blocker 1 depends on this) | WV-8, WF-4 | **3 + ROW dependency** |
| **5** | **Acquisition count not tracked** | L2 | GH-6, GH-7, GH-9 = **3** | WI-4, WI-5 = **2** | **5** |
| **6** | **IP minimum not codified; IP pace not computed** | L0/L1/L2 | GH-12 (partial), GH-13, GH-14, MS-12 = **4** | — | **4** |
| **7** | **No L0 contracts for scoreboard, player row, budget, status tags** | L0 | Entire scoreboard shape, entire player row shape, entire budget shape | All pages | **Structural** — all API design depends on these |
| **8** | **Category classification not directional or wired to APIs** | L3/L4/L5 | CS-2, MS-8 = **2** | WV-3, PP-11 = **2** | **4** |
| **9** | **Expected-category-wins delta function missing** | L1/L4 | RI-3, RI-4 = **2** | WV-2, WV-6, PI-2 = **3** | **5** |
| **10** | **No ratio risk quantifier** | L1/L4 | PR-8 = **1** | PP-10, PP-12 = **2** | **3** |
| **11** | **No delta-to-flip calculator** | L1 | MS-10 = **1** | — | **1** |
| **12** | **No freshness propagation in API responses** | L5 | XC-1, XC-2, PR-22 = **3** | Same across all pages | **3 × all pages** |
| **13** | **No pitcher W/QS probability model** | L3 | — | PP-9 = **1** | **1** |
| **14** | **No trade evaluation engine** | L4 | — | — | Blocks entire Trade page (P3) |
| **15** | **No season-long Monte Carlo** | L4 | — | — | Blocks SD-3 (P3) |

---

## 6. Sequenced Implementation Plan with Gates

### Phase 0: Layer 0 — Lock Contracts (No code that runs; only type definitions)

| Task | Description | Deliverable |
|------|------------|-------------|
| P0-1 | **Define LeagueConfig constant** — acquisition_limit=8, ip_minimum=18.0, il_slots=3, roster_slots=[C,1B,2B,3B,SS,LF,CF,RF,Util,SP,SP,RP,RP,P,P,P,BN,BN,BN,BN,IL,IL,IL,NA], matchup_period_days=7 | New `league_config.py` or addition to `contracts.py` |
| P0-2 | **Define CategoryStatusTag enum** — LOCKED_WIN, LOCKED_LOSS, BUBBLE, LEANING_WIN, LEANING_LOSS with threshold definitions (e.g., LOCKED >90% sim prob, BUBBLE 35-65%, LEANING outside BUBBLE but not LOCKED) | Enum in contracts module |
| P0-3 | **Define ConstraintBudget contract** — Pydantic model with all GH-6 through GH-14 fields | Contract in contracts module |
| P0-4 | **Define MatchupScoreboardRow contract** — Pydantic model matching all MS-1 through MS-12 fields per category + MatchupScoreboardResponse with header (MS-13 through MS-16) | Contract in contracts module |
| P0-5 | **Define CanonicalPlayerRow contract** — Pydantic model matching all PR-1 through PR-22 fields, including nested dicts for season/rolling/projected stats keyed by the 18 category names | Contract in contracts module |
| P0-6 | **Define FreshnessMetadata contract** — Pydantic model for per-field freshness annotations, staleness thresholds per data type | Contract in contracts module |

**Gate 0:** All contracts compile. All contracts reference the 18 categories from `fantasy_stat_contract.json`. Contracts are reviewed and frozen. No implementation code yet — these are type definitions only.

**Gate 0 checklist:**
- [ ] `LeagueConfig` has all 5 constraint constants
- [ ] `CategoryStatusTag` enum has all 5 values with threshold definitions
- [ ] `ConstraintBudget` has all fields from GH-6 through GH-14
- [ ] `MatchupScoreboardRow` has all fields from MS-1 through MS-12
- [ ] `MatchupScoreboardResponse` has header fields MS-13 through MS-16
- [ ] `CanonicalPlayerRow` has all fields from PR-1 through PR-22
- [ ] Every dict field keyed by category uses the 18 canonical category names
- [ ] `py_compile` passes on all new files
- [ ] Review confirms no contract allows partial/optional fields for data that the UI spec requires unconditionally

---

### Phase 1: Layer 2 — Close Data Gaps

| Task | Description | Deliverable | Test Criteria |
|------|------------|-------------|---------------|
| P1-1 | **Build acquisition counter** — Parse `get_transactions()` response, filter by current matchup week boundaries (from Yahoo or league config), count add operations attributed to my team | Function `count_acquisitions_this_week() → int` | Returns correct count when verified against Yahoo league page. Unit tests for 0 moves, mid-week, and boundary cases. |
| P1-2 | **Build IP pace extractor** — From Yahoo scoreboard response, isolate stat_id 50 (IP) for my team in current matchup | Function `get_ip_accumulated() → float` | Returns IP matching Yahoo scoreboard. |
| P1-3 | **Build per-team schedule for current matchup week** — Using `statsapi.schedule()`, compute games remaining per team for current matchup period | Function `get_team_games_remaining(team_abbr, week_start, week_end) → int` | Returns correct count for 3+ teams verified against MLB schedule. |
| P1-4 | **Build per-player games-remaining** — Join roster (player → team) with team schedule (team → games remaining) | Function `get_player_games_remaining(roster, week_start, week_end) → dict[player_id, int]` | All rostered players have game counts. Pitchers cross-referenced with probable_pitchers for start counts. |
| P1-5 | **Extend standings parsing** — Extract W-L-T from `get_standings()` response | `_parse_team()` returns `{"wins": int, "losses": int, "ties": int}` | Matches Yahoo standings page. |
| P1-6 | **Build opposing-SP lookup** — Join (hitter → today's game → opponent probable pitcher → pitcher handedness) | Function `get_opposing_sp(player_team, game_date) → {name, handedness, team}` | Returns correct SP for 5+ test cases. |
| P1-7 | **Verify ownership % parsing** — Confirm Yahoo `percent_owned` is non-zero for rostered players | Fix `_parse_player()` if needed | Non-zero ownership for >80% of rostered players. |

**Gate 1:** All P1 tasks pass. `count_acquisitions_this_week()` returns real data. `get_player_games_remaining()` returns non-zero for active players mid-week. `get_ip_accumulated()` returns a float > 0 when matchup is in progress. `get_opposing_sp()` returns a real pitcher for today's games.

**Gate 1 checklist:**
- [ ] `count_acquisitions_this_week()` verified against Yahoo (manual check)
- [ ] `get_ip_accumulated()` matches Yahoo scoreboard IP
- [ ] `get_team_games_remaining()` correct for ≥3 teams
- [ ] `get_player_games_remaining()` non-zero for active players mid-week
- [ ] `_parse_team()` extracts W-L-T correctly
- [ ] `get_opposing_sp()` returns real pitcher for today's slate
- [ ] ownership % non-zero for >80% of rostered players
- [ ] All unit tests pass (`pytest tests/ -q --tb=short`)

---

### Phase 2: Layer 3 — Complete Category Coverage + ROW Projections

**Depends on:** Gate 1 (games-remaining data required)

| Task | Description | Deliverable | Test Criteria |
|------|------------|-------------|---------------|
| P2-1 | **Expand rolling stats to 18 categories** — Add R (runs), TB (total bases, computed from H+2B+2×3B+3×HR) to batting rolling windows. Add W, L, HRA, K(pit), QS, NSV to pitching rolling windows. Source from `mlb_player_stats` where possible; W/L/QS/SV may require additional box-score ingestion. | Updated `PlayerRollingStats` model + migration + rolling-window computation | All 18 categories present in 7/14/30-day windows for players with sufficient history. |
| P2-2 | **Expand projections to 18 categories** — Extend `BayesianProjectionUpdater` to project all 18 league categories. Derivable: H (from AVG×AB), TB (from SLG×AB), K(B) (from K-rate×PA), NSB (from SB−CS projection). New models needed: W, L, QS, NSV, HRA, HLD. For initially-missing pitching decision stats, use league-average rates scaled by pitcher quality (K/9, ERA) as priors. | Updated `PlayerProjection` model with all 18 categories in `cat_scores` JSONB | `cat_scores` non-null for all 18 categories for ≥80% of players with ≥20 PA/IP. |
| P2-3 | **Build ROW projection pipeline** — For each player: `per_game_projection × games_remaining`. Counting stats are additive. Rate stats (AVG, OPS, ERA, WHIP, K/9) must be projected as weighted averages over projected counting components (e.g., projected ERA = projected ER / projected IP × 9). Handle zero-games-remaining (return 0 for counting, current-rate for rate stats). | Function `compute_row_projections(roster, week_start, week_end) → dict[player_id, dict[cat, float]]` | Projections for all rostered players across all 18 categories. Counting stats scale proportionally with games remaining. Rate stats remain in plausible bounds. |
| P2-4 | **Build team-level ROW aggregation** — Sum per-player ROW projections to team totals. Handle rate stats correctly: ERA = sum(ER)/sum(IP)×9, AVG = sum(H)/sum(AB), etc. | Function `aggregate_team_row(player_rows) → dict[cat, float]` | Team aggregation matches manual calculation for test roster. Rate stats are IP-weighted or AB-weighted as appropriate. |
| P2-5 | **Build opponent ROW projection** — Fetch opponent roster, apply same per-player ROW pipeline | Function `compute_opponent_row(opponent_team_key, week_start, week_end) → dict[cat, float]` | Produces plausible opponent projected finals. |
| P2-6 | **Build projected-finals aggregator** — Current matchup totals + ROW projections for both teams | Function `get_projected_finals(week) → {my: dict[cat, float], opp: dict[cat, float]}` | Projected finals = current + ROW for all 18 categories. |
| P2-7 | **Build category classification** — Feed projected finals + current state into H2H Monte Carlo. For each category, classify using CategoryStatusTag enum thresholds. Add directionality (WIN/LOSS) based on projected margin sign. | Function `classify_categories(projected_finals) → list[{cat, status_tag, flip_prob}]` | At least one BUBBLE category exists for typical mid-week matchup. Status tags use L0 enum. Flip probabilities between 0 and 1. |
| P2-8 | **Build delta-to-flip calculator** — Pure L1 function. Counting stats: `delta = opp_projected − my_projected + 1` (need to exceed, not tie). Rate stats: algebraic inversion for ERA/WHIP/AVG/OPS given current and projected IP/AB. | Function `compute_delta_to_flip(my_proj, opp_proj, cat, is_lower_better, current_ip_or_ab) → string` | Returns "Need +3 HR" for counting stat trailing by 2. Returns "Keep ERA below 3.85" for rate stat currently winning. |
| P2-9 | **Add freshness timestamps to all L3 outputs** — Every derived table and API response includes `computed_at`, `source_freshness`, `is_stale` | Updated Pydantic schemas with `FreshnessMetadata` nested object | All L3 responses carry freshness timestamps. |

**Gate 2:** ROW projections are stable and plausible.

**Gate 2 checklist:**
- [ ] Rolling stats cover all 18 categories for ≥80% of players with ≥20 PA/IP
- [ ] Projections cover all 18 categories for ≥80% of players
- [ ] `compute_row_projections()` returns 18 non-null categories per player
- [ ] `aggregate_team_row()` produces plausible team totals
- [ ] `get_projected_finals()` returns projected finals for both teams
- [ ] `classify_categories()` produces non-degenerate status tags (not all LOCKED)
- [ ] `compute_delta_to_flip()` returns sensible strings for counting and rate stats
- [ ] All projections pass sanity checks: no negative counting stats, rate stats in [0, 10] for ERA, [0, 3] for WHIP, [0, 0.500] for AVG
- [ ] End-to-end: scoreboard pipeline runs in < 10 seconds
- [ ] All unit tests pass

---

### Phase 3: Layer 1 + Layer 4 — Pure Functions and Engine Wiring (Partial Gate Lift)

**Depends on:** Gate 2

| Task | Description | Deliverable | Test Criteria |
|------|------------|-------------|---------------|
| P3-1 | **Build IP pace classifier** — Pure L1 function. Given current IP, minimum IP, days remaining, compute pace flag. | `classify_ip_pace(ip_current, ip_min, days_remaining) → enum(BEHIND, ON_TRACK, AHEAD)` | BEHIND when extrapolated IP < minimum. ON_TRACK within ±10%. AHEAD otherwise. |
| P3-2 | **Build category-count delta extractor** — From two H2H sim results (before/after a move), compute expected change in categories won. | `compute_category_delta(sim_before, sim_after) → {expected_cats_gained: float, cats_flipped: list[str]}` | Delta correctly identifies categories that flip. Sum matches overall win-prob change. |
| P3-3 | **Build ratio risk quantifier** — Pure L1 function + L4 orchestration. Given pitcher projection (expected IP, expected ERA), current team IP/ER totals, opponent ERA projection: P(team ERA flips from win to loss). | `compute_ratio_risk(pitcher_era_proj, pitcher_ip_proj, team_current_er, team_current_ip, opp_era_proj) → {flip_probability: float, expected_era_after: float}` | Returns plausible flip probabilities. High for bad pitcher in close ERA matchup. Low for ace. |
| P3-4 | **Wire H2H Monte Carlo to projected finals** — Modify `POST /api/fantasy/matchup/simulate` to accept projected finals OR compute them internally. | Updated endpoint that can auto-compute projected finals when given a week parameter. | Simulation uses projected finals, not just current state. |
| P3-5 | **Wire MCMC roster-move to category-count delta** — Modify `simulate_roster_move()` to return `expected_cats_gained` and `cats_flipped` in addition to `win_prob_gain`. | Updated MCMC output with category-count fields | Waiver recommendations show "+1.3 expected categories" |
| P3-6 | **Extend lineup solver to full roster** — Add pitcher slot constraints (SP×2, RP×2, P×3) and BN/IL/NA handling to ILP solver. | Updated [lineup_constraint_solver.py](backend/fantasy_baseball/lineup_constraint_solver.py) handling 25 slots | Solver assigns all 25 slots. Pitcher eligibility respected (SP vs RP vs P). |

**Gate 3:** L4 engines produce correct outputs when fed L3 projected data.

**Gate 3 checklist:**
- [ ] `classify_ip_pace()` returns correct flag for 5 test scenarios
- [ ] `compute_category_delta()` returns correct delta for known before/after sim outputs
- [ ] `compute_ratio_risk()` returns plausible probabilities for 5 pitcher scenarios
- [ ] H2H Monte Carlo with projected finals produces non-degenerate results
- [ ] MCMC roster-move returns category-count delta
- [ ] Lineup solver assigns all 25 slots without constraint violations
- [ ] All unit tests pass

---

### Phase 4: Layer 5 — API Endpoints for P1 Pages

**Depends on:** Gate 3

| Task | Description | Deliverable | Test Criteria |
|------|------------|-------------|---------------|
| P4-1 | **Build `GET /api/fantasy/scoreboard`** — Returns `MatchupScoreboardResponse` with 18 `MatchupScoreboardRow` objects + header summary. Auto-computes projected finals, runs classification, computes delta-to-flip. | New endpoint in fantasy router | Returns all MS-1 through MS-16 fields. Auth via verify_api_key. 10+ test cases. Response time < 15 seconds. |
| P4-2 | **Build `GET /api/fantasy/budget`** — Returns `ConstraintBudget` with acquisitions, IP pace, IL usage. | New endpoint | Returns all GH-6 through GH-14 fields. |
| P4-3 | **Extend `GET /api/fantasy/roster` to return CanonicalPlayerRow** — Each player includes all PR-1 through PR-22 fields, including 18-category season/rolling/projected stats, today's status, freshness. | Extended response schema | All 22 fields populated per player. 18 categories per stat window. |
| P4-4 | **Build `POST /api/fantasy/roster/move`** — Proxy to Yahoo API for slot swaps, IL moves, start/sit changes. Returns updated roster + matchup impact. | New endpoint | Slot swap reflected in subsequent GET /roster. Impact preview returned. |
| P4-5 | **Build `POST /api/fantasy/roster/optimize`** — Run full-roster optimizer, return suggested changes with per-change category-wins delta. | New endpoint | Returns list of suggested swaps with expected_cats_gained. |
| P4-6 | **Extend `POST /api/fantasy/roster/drop`** (or add as parameter) — Drop player with matchup impact preview before execution. | New or extended endpoint | Returns projected matchup record before and after drop. |
| P4-7 | **Add freshness + horizon metadata to all responses** — Every response includes `FreshnessMetadata` and horizon labels per field. | Updated Pydantic response models | `data_freshness` field present on all responses. Horizon labels on all projection fields. |

**Gate 4:** P1 page APIs are complete and stable.

**Gate 4 checklist:**
- [ ] `GET /api/fantasy/scoreboard` returns 18 rows with all fields populated
- [ ] `GET /api/fantasy/budget` returns correct constraint values
- [ ] `GET /api/fantasy/roster` returns CanonicalPlayerRow for all roster slots
- [ ] `POST /api/fantasy/roster/move` successfully swaps players
- [ ] `POST /api/fantasy/roster/optimize` returns suggested changes with category deltas
- [ ] All responses include freshness metadata
- [ ] All projection fields are horizon-labeled
- [ ] ≥10 tests per endpoint, all passing
- [ ] Response times: scoreboard < 15s, budget < 2s, roster < 10s, optimize < 30s

---

### Phase 5: Layer 6 — Build P1 Pages (Scoreboard + Roster)

**Depends on:** Gate 4

| Task | Description |
|------|------------|
| P5-1 | **Archive CBB pages** — Move 9 CBB pages to `/legacy/` route group, gate behind `SHOW_CBB_LEGACY=false` feature flag. Remove from primary navigation. |
| P5-2 | **Rebuild sidebar navigation** — Matchup Scoreboard (primary), My Roster, Waiver Wire, Streaming, Trade Analyzer, Season Dashboard. Admin stays. |
| P5-3 | **Build global persistent header** — Week/record/opponent/projected-outcome, day/days-remaining, acquisition counter with warning, IL slots, IP pace with flag. Consumes `GET /api/fantasy/budget` + `GET /api/fantasy/scoreboard` header. |
| P5-4 | **Build CategoryStatusTag component** — Consistent color/shape/placement. Used across all pages. |
| P5-5 | **Build CanonicalPlayerRow component** — 22 fields, identical across all pages. Season/rolling/projected stats expandable or tabbed. Freshness on hover. |
| P5-6 | **Build Matchup Scoreboard page** — 18 category rows, sortable/filterable, click-to-drill, historical overlay. Consumes `GET /api/fantasy/scoreboard`. |
| P5-7 | **Build My Roster page** — 25 slots in roster order, canonical player rows, start/sit toggle, optimize button, issue flags, IL/BN utilization panels, Today/Week/Projections views. Consumes `GET /api/fantasy/roster` + `POST /api/fantasy/roster/optimize`. |
| P5-8 | **Mobile optimization** — Scoreboard and Roster mobile-first. Touch targets, responsive layout, swipeable interactions. |

**Gate 5:** P1 pages render correctly with live data from production APIs.

**Gate 5 checklist:**
- [ ] CBB pages hidden from default navigation
- [ ] Sidebar shows canonical page hierarchy
- [ ] Global header visible on every page with live constraint data
- [ ] CategoryStatusTag component used consistently
- [ ] CanonicalPlayerRow component used on both Scoreboard drill-in and Roster page
- [ ] Scoreboard shows 18 category rows with all MS-fields populated
- [ ] Scoreboard sort/filter/drill interactions work
- [ ] Roster shows 25 slots in correct order with live player data
- [ ] Roster optimize button returns and displays suggestions
- [ ] Both pages fully functional on mobile viewport
- [ ] All 8 operating principles verifiable against the two pages:
  - [1] Projected values visible, not just current
  - [2] All values contextualized to this user's matchup
  - [3] Category triage visible (LOCKED/BUBBLE/LEANING)
  - [4] Ratio risk indicated for pitching categories
  - [5] Budget constraints visible (moves, IP, IL)
  - [6] Single-screen decision context
  - [7] Not required for P1 — acceptable to defer
  - [8] "Day X of Y" and horizon labels visible

---

### Phase 6: Layers 3-5 — Close Blockers for P2 Pages (Waiver + Streaming)

**Depends on:** Gate 5 (P1 pages stable)

| Task | Description |
|------|------------|
| P6-1 | **Wire bubble categories to waiver scoring** — Waiver endpoint returns which BUBBLE categories each FA would help flip. |
| P6-2 | **Build marginal-value ranking for waivers** — Default sort by expected-category-wins gained, not need_score. |
| P6-3 | **Build drop-impact analysis** — For each add, auto-compute and return the category-wins impact of dropping each viable candidate. |
| P6-4 | **Build `GET /api/fantasy/waiver` v2** — Returns CanonicalPlayerRow + WV-2 through WV-8 fields + all WF-1 through WF-10 filters as query params. |
| P6-5 | **Build pitcher projected game line** — IP, K, ERA, WHIP, W probability, QS probability per start. W/QS models may initially use simplified approaches (e.g., team win probability × historical pitcher-of-record rate). |
| P6-6 | **Build `GET /api/fantasy/streaming`** — Returns scored pitcher starts with PP-1 through PP-12 fields. |
| P6-7 | **Build `POST /api/fantasy/waiver/claim`** — Execute add/drop via Yahoo API, decrement local acquisition counter, return updated budget. |

**Gate 6 checklist:**
- [ ] Waiver endpoint returns expected-category-wins-gained as primary sort
- [ ] Waiver endpoint returns bubble categories per FA
- [ ] Waiver endpoint supports all 10 filter types
- [ ] Drop impact analysis returns per-category deltas
- [ ] Streaming endpoint returns ratio risk score and category leverage score
- [ ] Streaming endpoint returns start/skip recommendation
- [ ] Waiver claim endpoint executes via Yahoo API and decrements acquisition counter
- [ ] ≥10 tests per endpoint, all passing

---

### Phase 7: Layer 6 — Build P2 Pages (Waiver + Streaming)

**Depends on:** Gate 6

Build Waiver Wire page and Probable Pitchers / Streaming page per spec. Same pattern as Phase 5: consume APIs, use CanonicalPlayerRow and CategoryStatusTag components, ensure mobile parity, verify against operating principles.

---

### Phase 8: Layers 3-5 — Close Blockers for P3 Pages (Trade + Season)

**Depends on:** Gate 7 (P2 pages stable)

| Task | Description |
|------|------------|
| P8-1 | Build season category pace computation (18 cats, league-wide ranking) |
| P8-2 | Build cross-owner category surplus/deficit analysis |
| P8-3 | Build trade evaluation engine (per-category delta for this-week + ROS) |
| P8-4 | Build trade finder / package suggestion engine |
| P8-5 | Build counter-offer builder |
| P8-6 | Build season-long Monte Carlo for projected standings |
| P8-7 | Build structural category diagnosis engine |
| P8-8 | Build decision log with EV-vs-actual tracking |
| P8-9 | Build corresponding Layer 5 endpoints |

**Gate 8:** Trade evaluation endpoint and season dashboard endpoint return correct data.

---

### Phase 9: Layer 6 — Build P3 Pages (Trade + Season)

**Depends on:** Gate 8

Build Trade Analyzer page and Season Dashboard page per spec.

---

## 7. Existing UI Salvage Assessment

### Components — SALVAGEABLE

| Component | File | Verdict | Rationale |
|-----------|------|---------|-----------|
| Card/CardHeader/CardTitle/CardValue/CardContent | [frontend/components/ui/card.tsx](frontend/components/ui/card.tsx) | **KEEP** | Generic, well-built, matches design system |
| Button (variants) | [frontend/components/ui/button.tsx](frontend/components/ui/button.tsx) | **KEEP** | Standard Tailwind component |
| Badge | [frontend/components/ui/badge.tsx](frontend/components/ui/badge.tsx) | **REFACTOR** | Currently has BET/CONSIDER/PASS variants (CBB). Needs LOCKED_WIN/LOCKED_LOSS/BUBBLE/LEANING_WIN/LEANING_LOSS variants. |
| Alert | [frontend/components/ui/alert.tsx](frontend/components/ui/alert.tsx) | **KEEP** | Generic |
| Input/Label/Select/Tabs/Switch/Slider | frontend/components/ui/*.tsx | **KEEP** | Generic form controls |
| Separator/Progress/Skeleton | frontend/components/ui/*.tsx | **KEEP** | Generic |
| DataTable | [frontend/components/ui/data-table.tsx](frontend/components/ui/data-table.tsx) | **KEEP** | Sortable table — will be used for scoreboard rows |
| KpiCard | [frontend/components/ui/kpi-card.tsx](frontend/components/ui/kpi-card.tsx) | **KEEP** | Useful for header summary stats |
| ErrorBoundary | [frontend/components/ui/error-boundary.tsx](frontend/components/ui/error-boundary.tsx) | **KEEP** | Generic |
| Tooltip | [frontend/components/shared/tooltip.tsx](frontend/components/shared/tooltip.tsx) | **KEEP** | Needed for freshness timestamps |
| Providers (TanStack Query) | [frontend/components/providers.tsx](frontend/components/providers.tsx) | **KEEP** | Correct pattern |

### Infrastructure — SALVAGEABLE

| Item | File | Verdict | Rationale |
|------|------|---------|-----------|
| API client (`apiFetch<T>()`) | [frontend/lib/api.ts](frontend/lib/api.ts) | **KEEP + EXTEND** | Centralized fetch with auth. Needs new endpoint functions. |
| Auth flow | [frontend/app/login/page.tsx](frontend/app/login/page.tsx) | **KEEP** | Works correctly |
| Query client config | [frontend/lib/query-client.ts](frontend/lib/query-client.ts) | **KEEP** | TanStack Query setup |
| Layout (sidebar + header) | [frontend/app/(dashboard)/layout.tsx](frontend/app/(dashboard)/layout.tsx) | **REFACTOR** | Structure is fine. Sidebar content must be rebuilt. Header must add persistent constraint display. |
| Tailwind config + dark theme | [frontend/tailwind.config.ts](frontend/tailwind.config.ts) | **KEEP** | Zinc palette, design system tokens |
| Utils (cn()) | [frontend/lib/utils.ts](frontend/lib/utils.ts) | **KEEP** | Standard classname merger |

### Pages — Assessment

| Page | File | Verdict | Rationale |
|------|------|---------|-----------|
| `/login` | [frontend/app/login/page.tsx](frontend/app/login/page.tsx) | **KEEP** | Works, spec-neutral |
| `/dashboard` | [frontend/app/(dashboard)/dashboard/page.tsx](frontend/app/(dashboard)/dashboard/page.tsx) | **DISCARD** | Hot/cold streaks as primary surface = anti-pattern #2. Matchup preview stubbed. Does not conform to any canonical page. |
| `/decisions` | [frontend/app/(dashboard)/decisions/page.tsx](frontend/app/(dashboard)/decisions/page.tsx) | **DISCARD** | Does not map to any page in canonical hierarchy. Decision data is an output of the system, not a standalone page. Decision information should be embedded in Roster (recommended lineup changes) and Waiver (recommended adds). |
| `/performance` | [frontend/app/(dashboard)/performance/page.tsx](frontend/app/(dashboard)/performance/page.tsx) | **ARCHIVE** | CBB. Move to /legacy/. |
| `/clv` | [frontend/app/(dashboard)/clv/page.tsx](frontend/app/(dashboard)/clv/page.tsx) | **ARCHIVE** | CBB |
| `/bet-history` | [frontend/app/(dashboard)/bet-history/page.tsx](frontend/app/(dashboard)/bet-history/page.tsx) | **ARCHIVE** | CBB |
| `/calibration` | [frontend/app/(dashboard)/calibration/page.tsx](frontend/app/(dashboard)/calibration/page.tsx) | **ARCHIVE** | CBB |
| `/today` | [frontend/app/(dashboard)/today/page.tsx](frontend/app/(dashboard)/today/page.tsx) | **ARCHIVE** | CBB |
| `/live-slate` | [frontend/app/(dashboard)/live-slate/page.tsx](frontend/app/(dashboard)/live-slate/page.tsx) | **ARCHIVE** | CBB |
| `/odds-monitor` | [frontend/app/(dashboard)/odds-monitor/page.tsx](frontend/app/(dashboard)/odds-monitor/page.tsx) | **ARCHIVE** | CBB |
| `/bracket` | [frontend/app/(dashboard)/bracket/page.tsx](frontend/app/(dashboard)/bracket/page.tsx) | **ARCHIVE** | CBB (already hidden) |
| `/alerts` | [frontend/app/(dashboard)/alerts/page.tsx](frontend/app/(dashboard)/alerts/page.tsx) | **KEEP** | System alerts are useful. May need refactoring to reference fantasy concepts. |
| `/admin` | [frontend/app/(dashboard)/admin/page.tsx](frontend/app/(dashboard)/admin/page.tsx) | **KEEP** | Admin utilities are useful for debugging. Not user-facing but valuable for operator. |
| **Sidebar** | [frontend/components/layout/sidebar.tsx](frontend/components/layout/sidebar.tsx) | **REBUILD** | 9 CBB links + 1 fantasy link. Must be replaced with canonical hierarchy. Keep the drawer/responsive mechanism. |

### Types — Assessment

| Item | File | Verdict | Rationale |
|------|------|---------|-----------|
| CBB types (BetLog, ClvBetEntry, etc.) | [frontend/lib/types.ts](frontend/lib/types.ts) | **ARCHIVE** | Move to `types-legacy.ts` |
| Fantasy types (DashboardData, StreakPlayer, etc.) | [frontend/lib/types.ts](frontend/lib/types.ts) | **DISCARD + REBUILD** | Current types don't match CanonicalPlayerRow, MatchupScoreboardRow, or ConstraintBudget contracts. Must be regenerated from L0 contracts. |
| DecisionResultOut, DecisionExplanationOut | [frontend/lib/types.ts](frontend/lib/types.ts) | **KEEP** | Matches L5 API. Used in /decisions page which is being discarded, but the types remain valid for embedding decision data in Roster/Waiver pages. |

### Summary

- **Keep:** 15 UI components, API client, auth flow, query client, layout mechanism, Tailwind config, utils, login page, alerts page, admin page
- **Refactor:** Badge (new variants), sidebar (new links), layout header (add constraint display)
- **Archive (CBB):** 9 pages, CBB type definitions
- **Discard:** /dashboard page, /decisions page, current fantasy TypeScript types
- **Rebuild from scratch:** Matchup Scoreboard, My Roster, Waiver Wire, Streaming, Trade Analyzer, Season Dashboard, sidebar content, global header, CanonicalPlayerRow component, CategoryStatusTag component, all fantasy TypeScript types

---

## 8. Open Questions

| # | Question | Why It Matters | Impact If Unresolved |
|---|----------|---------------|---------------------|
| **Q1** | **Yahoo API rate limits for scoreboard/transactions/roster calls.** Each scoreboard page load may trigger 3+ Yahoo API calls (scoreboard + my roster + opponent roster + transactions). What is the rate limit? | Determines whether we need aggressive caching or can call Yahoo on every page load. | May need a 60-second cache layer between L5 endpoints and Yahoo API. Affects response freshness guarantees. |
| **Q2** | **Are W, L, SV, HLD, QS available in Yahoo player season stats?** Yahoo stat_id mapping shows these exist (23=W, 24=L, 32=SV, 48=HLD, 29=QS). Need to confirm the Yahoo `get_players_stats_batch()` response actually includes them for pitchers. | If yes: season stats for all 18 categories come from Yahoo (PR-13 READY). If no: need to source from MLB boxscores. | Affects P2-1 implementation path. |
| **Q3** | **Can Yahoo provide per-game box-score stats (W/L/QS/SV per individual game)?** Or only season aggregates? Needed for building rolling-window W/L/QS/SV/NSV. | Determines whether rolling stats for pitching decision categories require a separate box-score data source (e.g., MLB Stats API or Statcast). | Affects P2-1 scope and possibly P1 (new ingestion job). |
| **Q4** | **Does the league use FAAB or priority-based waivers?** The spec says "1-day rolling waivers" and "8 weekly acquisitions cap." If FAAB, the budget display needs a dollar amount. If priority-based, the display needs waiver claim position. | Affects ConstraintBudget contract (P0-3) and Waiver Wire page (WV-4). | Add FAAB balance OR waiver priority to contract and UI. |
| **Q5** | **For opponent rest-of-week projections: is per-player projection acceptable, or is a pace-based estimate sufficient initially?** Per-player opponent projection requires fetching their full roster, running projections for each player, and aggregating. Pace-based: extrapolate current daily rate × remaining days. | Per-player is more accurate but significantly more complex and slower (additional Yahoo API calls for opponent roster). Pace-based is a ~20-minute build. | Affects P2-5 scope. Recommend pace-based initially, upgrade later. |
| **Q6** | **What time zone defines "today" for schedule and playing-status lookups?** The codebase uses `America/New_York` per CLAUDE.md. But if a West Coast game starts at 10 PM ET, is that "today" until it ends or until midnight ET? | Affects PR-4 (playing today) and PP-2 (next start) definitions at day boundaries. | Define in LeagueConfig (P0-1). |
| **Q7** | **What defines the matchup week boundary?** Is it Monday-Sunday? Does Yahoo provide explicit week start/end dates? Or does it vary? | Affects GH-5 (days remaining), P1-1 (acquisition counting window), P1-3 (games-remaining window). | Must query Yahoo league settings for week boundaries. |
| **Q8** | **What's the acceptable response time for the scoreboard endpoint?** Computing projected finals requires: Yahoo API calls + projection computation + Monte Carlo simulation. Initial estimate: 10-20 seconds cold, 2-5 seconds cached. | If < 5s required: must pre-compute and cache on a schedule. If 15s acceptable: can compute on-demand with loading state. | Affects whether scoreboard data is computed on-request or by a scheduled job. |
| **Q9** | **How should the canonical player row handle players who appear on both my roster and the opponent's roster discussion (trade context)?** The spec says every player row is identical across pages. In trade evaluation, the same player might appear as "sending" and "receiving" — are there contextual variants? | Affects CanonicalPlayerRow contract completeness. | May need a `context` field (my_roster / opponent / free_agent / trade_target) to control which fields are relevant. |
| **Q10** | **Is the `HLD` (Holds) category confirmed as one of the 9 pitching categories?** The user's spec says "Pitching: W, L, HR, K, ERA, WHIP, K/9, QS, NSV" — that's 9 pitching categories and does NOT include HLD. But `fantasy_stat_contract.json` includes HLD (stat_id 48) in the stat labels. | If HLD is a display-only stat (like IP), it appears on the scoreboard but isn't a scoring category. If it IS a scoring category, the user's spec has only 8 pitching categories and is missing one. | Clarify the exact 18 scoring categories. Current count from user spec: 9 batting + 9 pitching = 18. HLD may be display-only. |
| **Q11** | **Is `K(B)` (Batting Strikeouts) lower-is-better confirmed?** The `lowerIsBetter` list in `fantasy_stat_contract.json` includes `K(B)` (stat_id 42). This means MORE strikeouts is BAD for the hitter. | Affects category classification and delta-to-flip logic for hitting K(B). | Confirm league scoring direction. |

---

## Appendix: Field Count Summary

| Page | Total Required Fields | READY | PARTIAL | MISSING |
|------|----------------------|-------|---------|---------|
| Global Header (GH) | 15 | 3 | 4 | 8 |
| Category Status Tag (CS) | 3 | 1 | 1 | 1 |
| Canonical Player Row (PR) | 22 | 5 | 11 | 6 |
| Matchup Scoreboard (MS) | 16 | 3 | 1 | 12 |
| My Roster (RO) | 10 | 2 | 3 | 5 |
| Waiver Wire (WV) | 8 | 1 | 2 | 5 |
| Streaming (PP) | 12 | 4 | 1 | 7 |
| Trade Analyzer (TR) | 9 | 0 | 0 | 9 |
| Season Dashboard (SD) | 6 | 0 | 3 | 3 |
| Cross-Cutting (XC) | 9 | 0 | 1 | 4 (+ 4 L6-only) |
| **TOTAL** | **110** | **19 (17%)** | **27 (25%)** | **64 (58%)** |

**Readiness: 17% of required fields are ready. 58% are missing entirely.**

The load-bearing page (Matchup Scoreboard) has 3 of 16 fields ready (19%). Its defining features — projected finals, status tags, flip probability, delta-to-flip — are all MISSING.

---

Last updated: April 17, 2026
