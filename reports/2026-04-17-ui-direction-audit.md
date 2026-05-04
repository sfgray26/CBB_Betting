# UI Specification Contract Audit and Gated Implementation Plan

> Date: April 17, 2026 | Author: Claude Code (Master Architect)
> Scope: The UI specification is the authoritative contract. The backend serves it. This audit maps every required field and interaction to current layer readiness, identifies all gaps, and produces a gated build sequence.

> **League format (immutable):** 10-team H2H One Win, 18 categories (Batting: R, H, HR, RBI, K(B), TB, AVG, OPS, NSB; Pitching: W, L, HRA, K, ERA, WHIP, K/9, QS, NSV), 18-IP minimum, 8 weekly acquisitions, 1-day rolling waivers, roster C/1B/2B/3B/SS/LF/CF/RF/Util/SP×2/RP×2/P×3/BN×4/IL×3/NA.

---

## 1. Master Data Contract

### 1.1 Page Inventory

The frontend is a Next.js 15 App Router application with 13 pages across two product domains.

**Fantasy Baseball Pages (2 of 13):**

| Page | Route | API Endpoints | Information Hierarchy |
|------|-------|---------------|----------------------|
| Daily Decisions | `/decisions` | `GET /api/fantasy/decisions`, `/api/fantasy/decisions/status` | Pipeline health badge → Lineup coverage map → Decision cards (player, type, confidence, value gain, reasoning, expandable explanation) |
| Dashboard | `/dashboard` | `GET /api/dashboard` | Lineup gap count → Hot/cold streaks → Waiver targets → Injury flags → Matchup preview → Probable pitchers |

**CBB Betting Pages (9 of 13) — All Frozen:**

| Page | Route | Domain |
|------|-------|--------|
| Performance | `/performance` | CBB — ROI, win rate, profit charts |
| CLV Analysis | `/clv` | CBB — closing line value distribution |
| Bet History | `/bet-history` | CBB — historical bet log |
| Calibration | `/calibration` | CBB — model confidence calibration |
| Today's Bets | `/today` | CBB — daily bet recommendations |
| Live Slate | `/live-slate` | CBB — in-progress games |
| Odds Monitor | `/odds-monitor` | CBB — odds movement tracking |
| Bracket | `/bracket` | CBB — tournament simulator (hidden) |
| Alerts | `/alerts` | Shared — system alerts |

**Admin/Shared Pages (2 of 13):**

| Page | Route | Domain |
|------|-------|--------|
| Admin | `/admin` | Both — risk dashboard, scheduler, feature flags |
| Login | `/login` | Both — API key auth |

### 1.2 Navigation Structure

The sidebar ([frontend/components/layout/sidebar.tsx](frontend/components/layout/sidebar.tsx)) has 5 sections:

1. **Analytics** — Dashboard, Performance, CLV, Bet History, Calibration, Alerts (4 of 6 are CBB)
2. **Trading** — Today's Bets, Live Slate, Odds Monitor (all CBB)
3. **Tournament** — Bracket Simulator (CBB, hidden)
4. **Fantasy** — Daily Decisions (only 1 link)
5. **Admin** — Risk Dashboard

The navigation structure is **dominated by CBB betting** (9 links) with fantasy baseball as a single afterthought link. The sidebar grouping reflects the application's history, not its current product direction.

### 1.3 Data Flow Summary

The fantasy pages consume:

- `/decisions` page: `DecisionResult` + `DecisionExplanation` tables via `GET /api/fantasy/decisions` — these tables are **likely empty** since the decision recording job is not wired to the lineup/waiver endpoints.
- `/dashboard` page: `DashboardData` aggregation from `dashboard_service.py` — lineup gaps, streaks (from `player_daily_metric` rolling windows), waiver targets, injury flags, and matchup preview (`_get_matchup_preview()` returns None/empty).

### 1.4 Anti-Patterns Detected

| Anti-Pattern | Severity | Evidence |
|-------------|----------|----------|
| **Generic player rankings divorced from roster context** | HIGH | `/dashboard` hot/cold streaks are absolute z-score rankings, not marginal to the user's category needs. Waiver targets use `priority_score` based on ownership %, not roster-relative value. |
| **Recency/hot-streak leaderboards as primary surfaces** | HIGH | `/dashboard` prominently features "Hot Streaks" and "Cold Streaks" sections — pure recency bias signals with no category leverage context. |
| **Hiding acquisition counter** | CRITICAL | The 8-move weekly limit is not tracked, displayed, or enforced anywhere in the system. Zero awareness in backend or frontend. |
| **Hiding IP pace against 18-IP floor** | CRITICAL | No innings tracking, no minimum innings awareness anywhere. A manager could stream and bench all pitchers with no warning about the IP floor. |
| **Missing day-of-week context** | HIGH | `/decisions` shows `as_of_date` but no awareness of "Day 3 of 7", remaining games this week, or how decision urgency changes through the week. |
| **Season-long stats dominating weekly decision surfaces** | MEDIUM | `/dashboard` matchup preview (when it works) uses current matchup totals without projecting rest-of-week. Streaks use 7/14/30-day windows, not matchup-scoped windows. |
| **Point-estimate projections without uncertainty** | MEDIUM | `contracts.py` defines `UncertaintyRange` with 80%/95% CI ([contracts.py](backend/contracts.py) lines 29-36), but no frontend page displays uncertainty ranges. All values are point estimates. |
| **Waiver/trade features not referencing bubble categories** | HIGH | `/dashboard` waiver targets don't show which categories are BUBBLE or how the target addresses them. Category deficit data exists in the backend (`CategoryDeficitOut` in schemas.py) but isn't surfaced prominently. |

---

## 2. Alignment Scoring

### Scoring Key
- **A** = Fully aligned, implemented correctly
- **B** = Partially aligned, needs work
- **C** = Misaligned, needs redesign
- **F** = Absent or contradicts principle
- **N/A** = Page doesn't exist yet

### 2.1 Existing Pages vs. Eight Principles

| Principle | `/decisions` | `/dashboard` | Notes |
|-----------|:---:|:---:|-------|
| 1. EV over outcomes | B | F | Decisions show confidence + value gain (good). Dashboard shows hot/cold streaks — pure recency bias. |
| 2. Marginal value over absolute | C | F | Decisions have `value_gain` field but it's not computed relative to matchup bubbles. Dashboard rankings are absolute. |
| 3. Category leverage / marginal wins | F | F | Neither page shows LOCKED/BUBBLE/LEANING classification. No category triage visible anywhere. |
| 4. Asymmetric risk management | F | F | No ERA/WHIP risk visibility. No ratio protection warnings. |
| 5. Inventory management under constraint | F | F | No acquisition counter. No IP pace. No IL budget display. |
| 6. Process discipline / low cognitive load | B | C | Decisions page is clean and actionable. Dashboard is a grab bag without clear decision flow. |
| 7. Portfolio thinking / correlation | F | F | No stacking info, no archetype concentration, no category correlation. |
| 8. Time-horizon clarity | F | C | Decisions show date but no week-position context. Dashboard mixes 7/14/30-day windows without matchup framing. |

### 2.2 Existing Pages vs. Canonical Page Hierarchy

| Canonical Page | Current Status | Evidence |
|---------------|----------------|----------|
| 1. **Matchup Scoreboard** | **DOES NOT EXIST** | No page shows per-category rows of me vs. opponent with status flags. The `/dashboard` matchup preview is stubbed (`_get_matchup_preview()` returns empty). |
| 2. **Projections Engine** (data layer) | **DOES NOT EXIST** | No rest-of-week projection pipeline. Current projections are per-game or season-level, not matchup-week scoped. |
| 3. **Roster Management** | **PARTIAL** | `/decisions` shows lineup decisions but without category-win context. No optimizer UI. `POST /api/fantasy/lineup/async-optimize` exists but no frontend page calls it. |
| 4. **Waiver / Free Agent** | **PARTIAL** | `/dashboard` shows top 5 waiver targets with priority_score and tier, but no marginal value, no bubble-category linkage, no acquisition counter. |
| 5. **Probable Pitchers / Streaming** | **PARTIAL** | `/dashboard` shows probable pitchers and two-start pitchers, but no ratio risk scoring, no category leverage filtering. |
| 6. **Trade Analyzer** | **DOES NOT EXIST** | No trade evaluation page or endpoint. |
| 7. **Season Dashboard** | **DOES NOT EXIST** | No 26-week trajectory, no structural analysis. |

### 2.3 Summary

The current UI is a CBB betting dashboard with a fantasy baseball sidebar link bolted on. The two fantasy pages (`/decisions` and `/dashboard`) are information displays, not decision-support tools. They violate 6 of 8 operating principles and do not implement any of the canonical pages.

The load-bearing page (Matchup Scoreboard) does not exist. Its primary data dependency (rest-of-week projections) does not exist.

---

## 3. Layer Readiness Matrix

### 3.1 Matchup Scoreboard Requirements

| Requirement | Layer | Status | Evidence |
|------------|-------|--------|----------|
| **Category enum (18 cats)** | L0 | READY | [fantasy_stat_contract.py](backend/fantasy_baseball/fantasy_stat_contract.py): 9 batting + 9 pitching categories canonically defined |
| **CategoryNeed contract** | L0 | PARTIAL | `CategoryNeed` exists as dataclass in [category_tracker.py](backend/fantasy_baseball/category_tracker.py) but not in canonical `contracts.py` |
| **LOCKED/BUBBLE/LEANING classification** | L0 | PARTIAL | `H2HOneWinSimResponse` in [schemas.py](backend/schemas.py) lines 755-800 defines `locked_categories`, `swing_categories`, `vulnerable_categories` — but taxonomy doesn't match (SWING ≠ BUBBLE, VULNERABLE ≠ LEANING). Missing LEANING classification. |
| **Matchup state contract** | L0 | MISSING | No contract for "current me/opp totals + projected final + status per category" as a single shape |
| **De-vigging / odds math** | L1 | READY | [odds_math.py](backend/core/odds_math.py) has pure functions |
| **Category win probability computation** | L1 | READY | [h2h_monte_carlo.py](backend/fantasy_baseball/h2h_monte_carlo.py) lines 49-302: simulation produces per-category win probabilities |
| **Delta-to-flip calculation** | L1 | MISSING | No pure function computes "how much does category X need to change to flip from LOSS to WIN" |
| **Current matchup category totals (me vs. opp)** | L2 | READY | [category_tracker.py](backend/fantasy_baseball/category_tracker.py) lines 70-173: `get_category_needs()` fetches live Yahoo scoreboard data for both teams |
| **Player schedule (games remaining this week)** | L2 | MISSING | Only pitcher starts tracked via [two_start_detector.py](backend/fantasy_baseball/two_start_detector.py). No batter schedule awareness. No "games remaining" for any player. |
| **Rest-of-week projections** | L3 | MISSING | No pipeline projects stat accumulation for remaining days. [daily_briefing.py](backend/fantasy_baseball/daily_briefing.py) line 87 uses naive `current * 7` — not a real projection. |
| **Projected final category totals** | L3 | MISSING | Cannot compute without rest-of-week projections. Current state only. |
| **Category classification over projected finals** | L3/L4 | PARTIAL | H2H Monte Carlo can classify if given pre-aggregated projections, but there's no pipeline to produce those projections for input. |
| **API: single-query matchup state** | L5 | MISSING | `GET /api/fantasy/matchup` returns raw category data but not the complete scoreboard shape (no status flags, no projections, no delta-to-flip). |

### 3.2 Projections Engine Requirements

| Requirement | Layer | Status | Evidence |
|------------|-------|--------|----------|
| **Bayesian per-player projections (season)** | L3 | READY | `BayesianProjectionUpdater` runs every 6h via [daily_ingestion.py](backend/services/daily_ingestion.py). `player_projection` table populated. |
| **Per-player games-remaining-this-week** | L2 | MISSING | No schedule → player mapping for batters. Only pitchers via probable_pitchers table. |
| **Per-player rest-of-week stat projection** | L3 | MISSING | Need: `projected_per_game_stats × games_remaining × park_factors × matchup_quality`. None of this exists as a pipeline. |
| **Team-level aggregated rest-of-week projection** | L3 | MISSING | Depends on per-player ROW projections. |
| **Opponent roster + projection** | L2/L3 | PARTIAL | Yahoo scoreboard has opponent totals-to-date. No opponent roster-level projection exists. |

### 3.3 Roster Management Requirements

| Requirement | Layer | Status | Evidence |
|------------|-------|--------|----------|
| **Lineup optimization (slot assignment)** | L4 | READY | OR-Tools ILP solver in [lineup_constraint_solver.py](backend/fantasy_baseball/lineup_constraint_solver.py) + [daily_lineup_optimizer.py](backend/fantasy_baseball/daily_lineup_optimizer.py) |
| **Category-aware lineup optimization** | L4 | PARTIAL | [smart_lineup_selector.py](backend/fantasy_baseball/smart_lineup_selector.py) has `category_fit_bonus` (20% weight) from CategoryTracker, but it's not driven by BUBBLE classification |
| **Expected category wins gained/lost per move** | L4 | MISSING | No function computes marginal category win delta for a lineup swap |
| **Lineup optimization API** | L5 | READY | `POST /api/fantasy/lineup/async-optimize` exists |

### 3.4 Waiver / Free Agent Requirements

| Requirement | Layer | Status | Evidence |
|------------|-------|--------|----------|
| **Category deficit computation** | L3 | READY | [waiver_edge_detector.py](backend/services/waiver_edge_detector.py) lines 169-178: computes per-category deficits |
| **FA scoring by deficit-weighted value** | L3 | READY | Same file — `need_score = sum(fa_cat × max(0, deficit))` |
| **Marginal win probability (before/after)** | L4 | PARTIAL | MCMC enrichment exists (lines 95-110) but optional and disconnected from bubble classification |
| **Acquisition counter** | L2 | MISSING | Yahoo API `get_transactions()` exists but no counting/tracking logic. Weekly 8-move limit completely untracked. |
| **Waiver API with bubble context** | L5 | PARTIAL | `GET /api/fantasy/waiver` returns `CategoryDeficitOut` but without BUBBLE/LOCKED labels |

### 3.5 Probable Pitchers / Streaming Requirements

| Requirement | Layer | Status | Evidence |
|------------|-------|--------|----------|
| **Probable pitcher schedule** | L2 | READY | `probable_pitchers` table with 94 rows. Two-start detection working. |
| **Ratio risk scoring (ERA/WHIP exposure)** | L1/L3 | MISSING | No function quantifies "if I start this pitcher, what's the probability my ERA/WHIP category flips from WIN to LOSS" |
| **Category leverage filter** | L3 | MISSING | No pipeline filters streamers by which pitching categories are BUBBLE |
| **IP pace tracking** | L2 | MISSING | No awareness of 18-IP minimum or current IP accumulated |

### 3.6 Constraint Budget Tracking

| Requirement | Layer | Status | Evidence |
|------------|-------|--------|----------|
| **Weekly acquisition count** | L2 | MISSING | No code anywhere |
| **IP accumulated this week** | L2 | MISSING | No code anywhere |
| **IL slot utilization** | L2 | PARTIAL | `count_il_slots_used()` in [waiver_edge_detector.py](backend/services/waiver_edge_detector.py) line 33. Counts current IL usage but no capacity display. |
| **Roster slot availability** | L2 | READY | Yahoo roster data includes position eligibility and slot assignments |

---

## 4. Blocking Gaps (Prioritized)

### CRITICAL BLOCKERS — Nothing above can be built honestly without these

| # | Gap | Layer | Impact | Why It Blocks |
|---|-----|-------|--------|---------------|
| **B1** | **No rest-of-week projection pipeline** | L3 | Matchup Scoreboard, Projections Engine, all downstream | Without projected final totals, category status flags (LOCKED/BUBBLE/LEANING) are meaningless. The scoreboard would show current state only — a glorified Yahoo scoreboard with no edge. This is the single most important missing capability. |
| **B2** | **No per-player games-remaining-this-week** | L2 | Projections Engine | ROW projections require knowing how many games each batter/pitcher has left. Pitcher starts exist (probable_pitchers). Batter schedule is completely absent. MLB schedule data is available via BDL or statsapi but not wired. |
| **B3** | **No acquisition counter (8-move weekly limit)** | L2 | Waiver page, all transaction decisions | A manager making waiver decisions without seeing "3 of 8 moves used" is flying blind. This is a constraint the entire decision framework must respect. |
| **B4** | **No IP pace tracking (18-IP minimum)** | L2 | Pitcher streaming, roster management | Starting or benching pitchers without IP floor awareness risks an auto-loss of all pitching categories. |

### HIGH-PRIORITY GAPS — Required for the scoreboard to be useful

| # | Gap | Layer | Impact |
|---|-----|-------|--------|
| **G1** | **No matchup state contract** (single-shape: current + projected + status per category) | L0 | Scoreboard API design depends on this. Without a canonical contract, frontend will invent ad-hoc shapes. |
| **G2** | **Category classification not wired to decision routes** | L3→L5 | H2H Monte Carlo produces locked/swing/vulnerable but this output is stranded — not fed to lineup, waiver, or any visible API. |
| **G3** | **No delta-to-flip computation** | L1 | For BUBBLE categories, the scoreboard must show "you need +2 HR to flip this category". No pure function exists. |
| **G4** | **No ratio risk quantification** | L1 | "Starting Pitcher X has a 30% chance of flipping your ERA from WIN to LOSS" — no function exists. |
| **G5** | **Navigation structure is CBB-dominated** | L6 | 9 CBB links vs. 1 fantasy link. Sidebar must be restructured around the canonical page hierarchy. |

### MEDIUM-PRIORITY GAPS — Required for pages 3-5

| # | Gap | Layer | Impact |
|---|-----|-------|--------|
| **M1** | **Marginal value computation** | L4 | Waiver ranking by "win probability gain" requires marginal value, not absolute player value. MCMC enrichment exists but is optional and disconnected. |
| **M2** | **Opponent roster projection** | L2/L3 | Full matchup simulation needs projected opponent accumulation, not just current totals. |
| **M3** | **Decision recording pipeline not running** | L3/L5 | `/decisions` page consumes `DecisionResult` table which is likely empty. The page works structurally but has no data. |
| **M4** | **Dashboard `_get_matchup_preview()` returns None** | L5 | [dashboard_service.py](backend/services/dashboard_service.py) matchup preview is not implemented. |

### LOW-PRIORITY — Nice to have, can be stubbed

| # | Gap | Layer | Impact |
|---|-----|-------|--------|
| **L1** | UncertaintyRange not surfaced in any frontend | L5/L6 | Contracts define 80%/95% CI but UI shows point estimates only |
| **L2** | Trade analyzer endpoint/page | L4/L5/L6 | Canonical page #6 — acceptable to defer |
| **L3** | Season dashboard (26-week trajectory) | L3/L6 | Canonical page #7 — acceptable to defer |

---

## 5. Halt-or-Proceed Recommendation

### Verdict: **HALT UI DEVELOPMENT. Immediately.**

**Rationale:**

The current UI direction is unsalvageable as-is. Two of the thirteen pages serve fantasy baseball, and both violate the majority of the operating principles. The load-bearing page (Matchup Scoreboard) does not exist. Its load-bearing data dependency (rest-of-week projections) does not exist. The entire navigation structure is organized around a frozen CBB betting product.

Building more UI on top of this foundation would compound the problem:

1. **No projections engine** → any scoreboard built now would be a Yahoo scoreboard clone with no edge value. Users would see current totals and nothing else. This directly violates Principles 1 (EV over outcomes), 3 (category leverage), and 8 (time-horizon clarity).

2. **No constraint tracking** → any waiver or lineup page built now would hide the two scarcest resources (moves remaining, IP pace), leading the user to make uninformed decisions. This directly violates Principles 5 (inventory management) and 6 (process discipline).

3. **Missing integration** → the H2H Monte Carlo simulation works in isolation but isn't connected to any decision route. Building a UI that calls disconnected endpoints would create the illusion of decision support without the substance.

**The correct action is:**
- Stop all Layer 6 work.
- Execute the Layer 2/3 remediation below.
- Resume UI only after the gates are met.

**What is salvageable:**
- The component library (card, badge, button, table, skeleton, etc.) is solid and design-system-aligned.
- The TanStack Query data-fetching pattern is correct.
- The auth flow works.
- The `/decisions` page structure is reasonable once it has data.
- The dark theme + Zinc palette is ready.

**What should be discarded:**
- The 9 CBB betting pages should be archived (moved to a `/legacy/` route group or hidden behind a feature flag). They are dead product surface.
- The current sidebar navigation must be rebuilt around the canonical page hierarchy.
- The `/dashboard` page in its current form (streaks/waiver targets as primary surfaces) should not be the landing page. The Matchup Scoreboard should be.

---

## 6. Sequenced Remediation Plan with Gates

### Phase 0: Constraint Foundation (Layer 2)
**Duration:** Must complete before any other phase.

| Task | Layer | Description | Acceptance Criteria |
|------|-------|-------------|---------------------|
| **P0-1** | L2 | **Build weekly acquisition counter** — Parse Yahoo `get_transactions()` response, count adds this matchup week, expose via API | `GET /api/fantasy/budget` returns `{"moves_used": 3, "moves_remaining": 5, "moves_limit": 8, "week": 4}`. Verified against Yahoo league settings. Unit tests pass. |
| **P0-2** | L2 | **Build IP pace tracker** — Query Yahoo scoreboard for current week IP, compute against 18-IP minimum | Same endpoint returns `{"ip_accumulated": 12.1, "ip_minimum": 18.0, "ip_pace": "BEHIND", "ip_needed": 5.2}`. Unit tests pass. |
| **P0-3** | L2 | **Build per-player schedule for current matchup week** — Use MLB schedule data (BDL or statsapi) to compute games remaining per rostered player | New function `get_player_games_remaining(week)` returns `Dict[player_id, int]`. Tested for mid-week accuracy. |

**Gate 0:** All three P0 tasks pass with tests. `GET /api/fantasy/budget` returns real data from Yahoo.

---

### Phase 1: Projection Spine (Layer 3)
**Depends on:** Gate 0 (schedule data required)

| Task | Layer | Description | Acceptance Criteria |
|------|-------|-------------|---------------------|
| **P1-1** | L0 | **Define MatchupScoreboardContract** — Canonical Pydantic model for the scoreboard shape: per-category row with `{category, my_current, opp_current, my_projected_final, opp_projected_final, status_flag, delta_to_flip}` | Contract defined in `contracts.py`. Status flag enum: `LOCKED_WIN`, `LOCKED_LOSS`, `BUBBLE_AHEAD`, `BUBBLE_BEHIND`, `LEANING_WIN`, `LEANING_LOSS`. Reviewed and frozen. |
| **P1-2** | L1 | **Build delta-to-flip pure function** — Given current state + projected state + simulation probabilities, compute the stat delta needed to flip a category | `compute_delta_to_flip(my_proj, opp_proj, category_type)` returns float. Unit tests for counting stats (HR: need +2) and rate stats (AVG: need +0.015). |
| **P1-3** | L1 | **Build ratio risk quantifier** — Given a pitcher's projection + current team ratios, compute probability of ratio category flip | `compute_ratio_risk(pitcher_proj, current_era, current_ip, opp_era)` returns `{flip_probability: float, expected_era_impact: float}`. Unit tests pass. |
| **P1-4** | L3 | **Build rest-of-week projection pipeline** — For each rostered player: `per_game_projection × games_remaining × context_factors`. Produce projected rest-of-week stat accumulation. | `get_row_projections(roster, week)` returns `Dict[player_id, Dict[category, projected_value]]`. Tests verify mid-week vs. start-of-week behavior. |
| **P1-5** | L3 | **Build projected-final-totals aggregator** — Sum current matchup totals + ROW projections for both teams | `get_projected_finals(week)` returns `{my_projected: Dict[cat, float], opp_projected: Dict[cat, float]}`. |
| **P1-6** | L3 | **Build category classification over projections** — Feed projected finals into H2H Monte Carlo, extract per-category status flags | `classify_categories(projected_finals)` returns `List[CategoryStatus]` with status flags matching the L0 contract enum. |

**Gate 1:** `classify_categories()` produces valid output for a real mid-week matchup. Status flags are non-trivial (at least one BUBBLE exists). All unit tests pass. The projection pipeline runs end-to-end in < 5 seconds.

---

### Phase 2: Scoreboard API (Layer 5)
**Depends on:** Gate 1

| Task | Layer | Description | Acceptance Criteria |
|------|-------|-------------|---------------------|
| **P2-1** | L5 | **Build `GET /api/fantasy/scoreboard`** — Single endpoint returning the full `MatchupScoreboardContract` shape | Returns 18 category rows, each with current/projected/status/delta. Includes header: week, day-of-week, record, projected record. Auth via `verify_api_key`. |
| **P2-2** | L5 | **Wire constraint budget into scoreboard response** — Include `moves_used/remaining`, `ip_accumulated/minimum`, `il_used/available` in response header | Budget fields present and non-null. |
| **P2-3** | L5 | **Wire H2H sim into scoreboard** — Include `overall_win_probability` in response header | Win probability computed from projected finals, not current state. |

**Gate 2:** `GET /api/fantasy/scoreboard` returns a complete, internally consistent response for the current matchup week. Manual inspection confirms category flags are plausible. 10 tests minimum covering happy path, mid-week, and edge cases (no games remaining, IP at minimum, all categories locked).

---

### Phase 3: UI Resume — Matchup Scoreboard Only (Layer 6)
**Depends on:** Gate 2

| Task | Layer | Description | Acceptance Criteria |
|------|-------|-------------|---------------------|
| **P3-1** | L6 | **Archive CBB pages** — Move all CBB betting pages under a `/legacy/` route group or gate behind `SHOW_CBB_LEGACY` feature flag. Remove from primary navigation. | CBB pages inaccessible from default nav. Feature flag can re-enable. |
| **P3-2** | L6 | **Rebuild sidebar** — Matchup Scoreboard as primary link. Supporting pages ordered per canonical hierarchy. | Sidebar: Scoreboard → Roster → Waivers → Pitchers → (future: Trade, Season). Admin stays. |
| **P3-3** | L6 | **Build Matchup Scoreboard page** — 18 category rows. Header with week/day/record/budget. Color-coded status flags. Delta-to-flip for bubbles. | Page renders with real data from `GET /api/fantasy/scoreboard`. Visually inspected against DESIGN.md. |
| **P3-4** | L6 | **Integrate constraint budget display** — Persistent header or footer showing moves remaining, IP pace, IL slots | Budget always visible when on any fantasy page. |

**Gate 3:** Matchup Scoreboard renders correctly with live data. All 8 operating principles are verifiable against the scoreboard:
1. Projected values visible (not just current)
2. Status flags relative to THIS matchup (not generic)
3. Category triage visible (LOCKED/BUBBLE/LEANING)
4. Ratio risk indicated for pitching categories
5. Budget constraints visible (moves, IP, IL)
6. Single-screen decision context (low cognitive load)
7. No correlation/portfolio features yet (acceptable)
8. "Day X of 7" visible in header

---

### Phase 4: Supporting Pages (Layer 4-6)
**Depends on:** Gate 3. May also require lifting the Layer 4 gate.

| Task | Layer | Description | Gate |
|------|-------|-------------|------|
| **P4-1** | L4/L5/L6 | **Roster Management page** — Lineup optimizer driven by scoreboard context. Show expected category wins gained/lost per move. | Requires marginal-value function (M1 gap). Layer 4 gate may need to lift for category-win optimization. |
| **P4-2** | L5/L6 | **Waiver / Free Agent page** — Ranked by marginal value to bubble categories. Acquisition counter always visible. | Requires MCMC enrichment wired to bubble classification (G2 gap). |
| **P4-3** | L5/L6 | **Probable Pitchers / Streaming page** — Filtered by ratio risk and category leverage for current matchup. | Requires ratio risk function (P1-3) and category leverage filter. |
| **P4-4** | L4/L5/L6 | **Trade Analyzer** | Deferred. Requires season-pace model. |
| **P4-5** | L3/L6 | **Season Dashboard** | Deferred. Requires 26-week trajectory data. |

**Gate 4 (for each sub-page):** Page renders with real data. Anti-pattern checklist passes. Operating principles verified for the page's scope.

---

### Decision on Layer 4 Gate

The Layer 4 gate ("Hold until the first Layer 3 objective is stable") can partially lift once Gate 1 is met:

- **May lift for:** Lineup optimization with category-win weighting (P4-1), MCMC-enriched waiver recommendations (P4-2), ratio risk computation (P1-3).
- **Should NOT lift for:** Broad simulation expansion, possession simulation, Monte Carlo redesign, or any new decision engine architecture.

The criterion: Layer 3 ROW projections are producing stable, plausible output for at least one full matchup week before Layer 4 optimization code consumes them.

---

## 7. Open Questions for the Human

| # | Question | Why It Matters |
|---|----------|---------------|
| **Q1** | **Is the Yahoo API the sole source for current-week matchup stats (my totals vs. opponent totals)?** Or is there a database-backed copy? I see `CategoryTracker` fetching live from Yahoo, which means every scoreboard load triggers a Yahoo API call. If rate-limited, we may need to cache matchup state. | Determines whether P0 tasks need a caching layer. |
| **Q2** | **Should the 9 CBB betting pages be completely removed from the codebase, or archived behind a feature flag?** They're dead product but may have sentimental/reference value. I recommended feature-flagging them. | Affects P3-1 scope. |
| **Q3** | **What is the Yahoo API rate limit for scoreboard/transaction calls?** The scoreboard fetching in CategoryTracker and the lineup optimizer both hit Yahoo. If the scoreboard page polls frequently, we need to know the ceiling. | Determines polling interval for the scoreboard page. |
| **Q4** | **Does the league use FAAB (free agent acquisition budget) or priority-based waivers?** You mentioned "1-day rolling waivers" and "8 weekly acquisitions cap". If FAAB, the budget display needs a dollar amount, not just a move count. | Affects P0-1 design. |
| **Q5** | **For opponent rest-of-week projections: is a naive approach acceptable initially?** We could project opponent accumulation based on their current daily pace × remaining days, without modeling their actual roster. A more sophisticated approach (fetching opponent roster and projecting each player) is significantly more complex. | Affects P1-5 scope. |
| **Q6** | **The `player_projection` table is populated by `BayesianProjectionUpdater` every 6 hours. Are these season-level projections, or can they be interpreted as per-game projections?** I need to understand the unit of projection before building ROW projections on top of them. | Critical for P1-4 design. |
| **Q7** | **Is there MLB schedule data already ingested into the database, or does it need to be fetched fresh?** `mlb_player_stats` has game-level rows. The schedule may be derivable from existing data, or it may need a new ingestion job. | Affects P0-3 implementation path. |

---

## Appendix: File Reference Index

| File | Relevance |
|------|-----------|
| [frontend/components/layout/sidebar.tsx](frontend/components/layout/sidebar.tsx) | Navigation structure — CBB-dominated |
| [frontend/app/(dashboard)/decisions/page.tsx](frontend/app/(dashboard)/decisions/page.tsx) | Only dedicated fantasy page |
| [frontend/app/(dashboard)/dashboard/page.tsx](frontend/app/(dashboard)/dashboard/page.tsx) | Dashboard — anti-patterns present |
| [frontend/lib/types.ts](frontend/lib/types.ts) | Frontend type definitions |
| [frontend/lib/api.ts](frontend/lib/api.ts) | API client — all endpoint contracts |
| [backend/fantasy_baseball/category_tracker.py](backend/fantasy_baseball/category_tracker.py) | Current matchup state — READY |
| [backend/fantasy_baseball/h2h_monte_carlo.py](backend/fantasy_baseball/h2h_monte_carlo.py) | Category classification — READY but disconnected |
| [backend/fantasy_baseball/mcmc_simulator.py](backend/fantasy_baseball/mcmc_simulator.py) | Matchup simulation — no ROW |
| [backend/fantasy_baseball/daily_lineup_optimizer.py](backend/fantasy_baseball/daily_lineup_optimizer.py) | Lineup optimization — no category leverage |
| [backend/fantasy_baseball/smart_lineup_selector.py](backend/fantasy_baseball/smart_lineup_selector.py) | Smart scoring — partial category awareness |
| [backend/fantasy_baseball/elite_lineup_scorer.py](backend/fantasy_baseball/elite_lineup_scorer.py) | Batter scoring — vs pitcher, not vs roster |
| [backend/fantasy_baseball/two_start_detector.py](backend/fantasy_baseball/two_start_detector.py) | Pitcher schedule — READY |
| [backend/fantasy_baseball/valuation_worker.py](backend/fantasy_baseball/valuation_worker.py) | Player valuation — no marginal value |
| [backend/fantasy_baseball/decision_tracker.py](backend/fantasy_baseball/decision_tracker.py) | Decision recording — exists, not running |
| [backend/services/dashboard_service.py](backend/services/dashboard_service.py) | Dashboard aggregation — matchup preview stubbed |
| [backend/services/waiver_edge_detector.py](backend/services/waiver_edge_detector.py) | Waiver scoring — deficit-weighted, no bubble labels |
| [backend/contracts.py](backend/contracts.py) | Layer 0 contracts — UncertaintyRange defined but unused |
| [backend/schemas.py](backend/schemas.py) | Pydantic schemas — H2HOneWinSimResponse has classification |
| [backend/models.py](backend/models.py) | ORM models — PlayerProjection, DecisionResult exist |
| [backend/fantasy_baseball/yahoo_client_resilient.py](backend/fantasy_baseball/yahoo_client_resilient.py) | Yahoo API — matchup + transactions, no move counting |
| [backend/services/daily_ingestion.py](backend/services/daily_ingestion.py) | Scheduler — projection updates every 6h |
| [DESIGN.md](DESIGN.md) | Visual design authority — Revolut-inspired |
