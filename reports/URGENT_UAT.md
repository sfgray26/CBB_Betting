# Fantasy Lineup Tool — UAT Handoff Report

**Application:** `https://observant-benevolence-production.up.railway.app/fantasy/lineup`
**Audit Date:** 2026-03-30
**Stack:** Next.js (SSR), Python/Pydantic backend, Railway deployment
**Auditor Persona:** Lead QA Automation Architect + Elite Quantitative Fantasy Baseball Analyst

---

## 1. EXECUTIVE SUMMARY

The application has a catastrophic data-pipeline failure: at least 5 of 13 batters are displayed as having "no game today" when their MLB teams (HOU, ARI, CLE, SF) are confirmed to be playing on March 30, 2026 — rendering the entire lineup optimization output untrustworthy. A Pydantic schema mismatch (`injury_status` receiving `bool` instead of `string`) has disabled the constraint solver entirely, meaning the "Optimize Lineup" button is running degraded logic and producing invalid slot assignments (e.g., a natural 2B slotted at 3B). As a quantitative tool, it surfaces only two raw inputs (Implied Runs, Park Factor) with zero predictive or underlying metrics, no opposing-pitcher matchup data, no injury status indicators, and no explanation of its composite Score formula — making it fundamentally inadequate for competitive high-stakes fantasy baseball.

---

## 2. CRITICAL TECHNICAL BUGS (P0 / P1)

### P0 — Data Pipeline: Players Incorrectly Flagged as "No Game Today"

- [ ] **Bug:** Yainer Diaz (HOU), Geraldo Perdomo (ARI), Steven Kwan (CLE), Matt Chapman (SF), and Luis Arraez (SF) all display `OPP: —` and `TIME: —`, with the warning banner stating they have no game. **Real-world verification confirms all five teams have games on 2026-03-30** (HOU vs BOS, ARI vs DET, CLE vs SEA, SF @ SD).
- **Expected:** All players with scheduled games show opponent, game time, and valid Implied Runs / Park Factor values derived from the actual matchup.
- **Actual:** These players receive a hardcoded fallback of `Implied Runs: 4.50`, `Park Factor: 1.000`, `Score: 4.500` — which is a sentinel/default value that masks the data failure entirely.
- **Impact:** Two of these "no-game" players (Diaz, Perdomo) are still recommended as **START**, meaning the tool is telling the user to start players it believes have no game. This is the single most damaging bug in the application.
- **Root Cause (Likely):** The schedule/odds data source is either stale, failing silently, or the team-abbreviation mapping is broken for these specific teams. The SSR data fetch returns no game data, and the frontend has no fallback error state — it just renders defaults.
- **Fix Directive:** Audit the backend schedule-fetching service. Add explicit error handling: if a player's team has no matched game, flag the player with a distinct `NO_DATA` state (not a default score). The frontend must render this state as an orange "⚠ Schedule Data Missing" badge, never as a START recommendation.

### P0 — Backend: Pydantic Validation Error Crashing Constraint Solver

- [ ] **Bug:** The warning banner displays a raw Pydantic error: `"Constraint solver unavailable: 1 validation error for LineupPlayerOut injury_status Input should be a valid string [type=string_type, input_value=True, input_type=bool]"`
- **Expected:** The constraint solver runs successfully, or if it fails, a user-friendly error is displayed and a graceful degradation path is used.
- **Actual:** A raw Python stack trace fragment is rendered directly in the UI. The constraint solver is completely bypassed, meaning lineup optimization is running in a degraded/greedy mode.
- **Location:** The `LineupPlayerOut` Pydantic model (likely in a file like `schemas/lineup.py` or `models/lineup.py`) defines `injury_status: str`, but somewhere upstream the field is being set to `True` (a Python `bool`).
- **Fix Directive:** Find the data source that populates `injury_status`. It is likely an API response or database field returning a boolean (e.g., `is_injured: True`) that is being passed directly without coercion. Fix by either: (a) changing the Pydantic field to `injury_status: str | bool` with a validator that coerces bools to strings like `"ACTIVE"` / `"IL"`, or (b) fixing the upstream data transformation to always pass a string enum (`"HEALTHY"`, `"IL10"`, `"IL60"`, `"DTD"`, `"OUT"`, `"SUSPENDED"`).

### P0 — Slot Assignment Logic: Positional Mismatch

- [ ] **Bug:** Willi Castro is listed as `2B` (his natural position) but assigned to the `3B` slot. The roster does not contain a natural 3B starter — Matt Chapman (3B) is benched as "no game."
- **Expected:** If the constraint solver were functional, it would not assign a 2B-only player to a 3B slot unless Yahoo eligibility explicitly allows it. The UI should indicate multi-position eligibility if applicable.
- **Actual:** Because the constraint solver is crashed (see Pydantic bug above), the fallback greedy algorithm is assigning players to slots without proper positional validation.
- **Fix Directive:** Ensure the greedy/fallback optimizer still respects Yahoo positional eligibility. Add a `eligible_positions: list[str]` field to the player schema and validate assignments against it.

### P1 — Pitchers: SP Score 0.000 and "UNKNOWN" Action State

- [ ] **Bug:** Garrett Crochet (BOS), Gavin Williams (CLE), and Blake Snell (LAD) all show `SP Score: 0.000` and `Action: UNKNOWN`. The warning says "3 SP(s) have no start today."
- **Expected:** Pitchers without a confirmed start should display a clear `NOT STARTING` or `REST DAY` label. If a pitcher is on the IL (e.g., Blake Snell was reported as likely starting the 2026 season on the IL), the injury status should be prominently displayed.
- **Actual:** `0.000` is a misleading score (it's not that they're bad — they just have no data). `UNKNOWN` is not actionable.
- **Fix Directive:** Replace `0.000` with `—` or `N/A` for pitchers without starts. Replace `UNKNOWN` with a specific state: `NO START`, `IL`, or `SKIP`. Add an `injury_status` badge next to pitcher names.

### P1 — Production Localhost Link in Sidebar

- [ ] **Bug:** The sidebar contains a link to `http://localhost:8501` labeled "Streamlit Dashboard ↗". This is a development artifact deployed to production.
- **Expected:** Either remove the link entirely or point it to a production Streamlit URL.
- **Actual:** Clicking this in production will fail or potentially expose a local development service.
- **Location:** Sidebar navigation component (likely `components/Sidebar.tsx` or `app/(dashboard)/layout.tsx`).
- **Fix Directive:** Gate this link behind an environment variable (`SHOW_STREAMLIT_LINK=true`) or remove it. If the Streamlit dashboard is a real feature, deploy it and use the production URL.

### P1 — Date Picker: Potential Timezone Off-by-One

- [ ] **Bug:** The `<input type="date">` DOM value reads `2026-03-29` while the UI renders `03/30/2026`. This discrepancy suggests a timezone conversion issue — the date may be stored as UTC midnight (which would be March 29 in some US timezones) but displayed after local conversion.
- **Expected:** The date picker value and the data displayed should always agree. Since the MLB schedule is EDT/CT-anchored, the app should normalize all dates to US Eastern Time.
- **Actual:** If the server is in UTC and the client is in a US timezone, there is a window where yesterday's data could be fetched instead of today's.
- **Fix Directive:** Normalize all date handling to `America/New_York` timezone. Use `date-fns-tz` or `luxon` for timezone-aware date manipulation. The backend should accept and return dates in ET explicitly.

---

## 3. UX & UI TECHNICAL DEBT

### 3.1 Accessibility (a11y) Failures

- [ ] **Tables lack semantic attributes:** Neither the Batters nor Pitchers table has `<caption>`, `aria-label`, or `scope` attributes on `<th>` elements. Screen readers cannot parse the table structure.
- [ ] **No `aria-live` regions:** The warning banner, score updates, and "Last updated: Xs ago" counter all change dynamically but have no `aria-live="polite"` or `aria-live="assertive"` regions.
- [ ] **Color-only differentiation:** The Score column uses green text (#4ade80-ish) for values, but there is no secondary indicator (icon, pattern, bold weight) for colorblind users.
- [ ] **Action buttons lack accessible names:** The START/BENCH buttons do not include the player name in their accessible label (e.g., `aria-label="Start Byron Buxton"`).

### 3.2 Responsive Design

- [ ] **No mobile breakpoint:** At 390px viewport width, the sidebar remains fully visible and the main content area is clipped. The data tables overflow horizontally. There is no hamburger menu or collapsible sidebar behavior despite a "Open navigation" button existing in the header.
- [ ] **Table density:** At any viewport under ~1200px, the 9-column batter table becomes unreadable. Implement a card-based layout at mobile breakpoints or allow horizontal scroll with sticky player name column.

### 3.3 Layout & Visual Polish

- [ ] **Warning banner renders raw error strings:** The Pydantic validation error should never be shown to users. Parse backend errors and display user-friendly messages only.
- [ ] **No visual distinction between starters and bench:** The only difference is the START/BENCH button text. Add row background shading (e.g., subtle green tint for starters, gray for bench) and a divider between the groups.
- [ ] **Score column lacks context:** A score of `5.390` means nothing without context. Add a color gradient, percentile rank, or tooltip showing the calculation breakdown.
- [ ] **Missing player headshot or team logo:** The table is pure text. Even small 24px team logos would dramatically improve scanability.
- [ ] **"Optimize Lineup" button shows "Optimizing..." indefinitely during SSR hydration:** The button text changes to "Optimizing..." while the page loads, creating a false impression that optimization is running when it is actually just SSR hydration.

---

## 4. ELITE QUANTITATIVE FEATURE GAPS

### 4.1 Missing Metrics (Priority Order)

| Priority | Metric | Why It Matters | Data Source |
|----------|--------|----------------|-------------|
| **P0** | Opposing Starting Pitcher (Name, Hand, ERA, xFIP) | The single most important variable for batter projections. Currently not shown at all. | FanGraphs API, Rotowire |
| **P0** | Player Injury Status (IL10/IL60/DTD/OUT) | Without this, the tool recommends starting injured players. The Pydantic crash proves the data exists but is broken. | Yahoo Fantasy API, ESPN API |
| **P0** | Platoon Splits (vs. LHP / vs. RHP) | A batter's value changes dramatically based on pitcher handedness. | FanGraphs Splits |
| **P1** | xwOBA / xSLG (Statcast expected metrics) | Underlying quality metrics that separate true talent from luck. Far more predictive than counting stats. | Baseball Savant API |
| **P1** | CSW% (Called Strike + Whiff Rate) | Best single metric for pitcher dominance. | Baseball Savant |
| **P1** | SIERA (Skill-Interactive ERA) | More predictive than ERA for pitcher evaluation. | FanGraphs |
| **P1** | BvP (Batter vs. Pitcher history) | While sample sizes are small, 20+ PA matchup data is meaningful. | Baseball Reference, Statcast |
| **P2** | Weather & Wind (mph, direction) | A 15 mph wind blowing out at Wrigley can add 2+ runs to a game. | Weather API (OpenWeather), Ballpark Pal |
| **P2** | Umpire Home Plate Tendencies | Some umpires have strike zones that favor pitchers by 0.5+ runs per game. | UmpScorecard.com API |
| **P2** | Confirmed Lineup Position (batting order) | A player hitting 2nd gets ~0.5 more PA per game than one hitting 8th. | Rotowire Daily Lineups API |
| **P2** | Recent Form (Last 7/14/30 day rolling stats) | Hot/cold streaks matter for daily decisions. | FanGraphs Game Logs |
| **P3** | Stolen Base Opportunity (SB attempts, success rate, catcher pop time) | Category-specific edge for roto/H2H. | Statcast Sprint Speed + Catcher data |

### 4.2 Optimization Logic Deficiencies

- [ ] **No category-aware optimization:** The current Score appears to be a single composite number. In H2H category leagues, you need to optimize across 10+ categories simultaneously (R, HR, RBI, SB, AVG/OBP for batters; W, K, ERA, WHIP, SV+HLD for pitchers). The optimizer should accept the user's league scoring format and adjust accordingly.
- [ ] **No weekly matchup context:** In H2H leagues, you don't just optimize for today — you optimize for the full weekly matchup. If you're already winning ERA but losing K's, you should start high-K pitchers even with worse ERA projections.
- [ ] **No games-remaining awareness:** If a player has 4 games remaining this week vs. a bench player with 6, the bench player may provide more cumulative value. The tool shows no weekly game counts.
- [ ] **No "Start/Sit confidence" indicator:** Users need to know the margin. Is Byron Buxton a "strong start" or a "coin flip start"? Show confidence intervals or percentile ranges.
- [ ] **No multi-league support:** High-stakes players are often in 5-15 leagues simultaneously. The tool appears to manage a single roster.

### 4.3 Missing Pitcher-Specific Features

- [ ] **No opposing team implied total against the pitcher:** The "OPP Implied" column exists for pitchers, but batters have no equivalent "Opposing SP" column showing who they face.
- [ ] **No pitch-mix or Stuff+ data:** Modern pitcher evaluation requires understanding arsenal quality.
- [ ] **No QS probability / win probability:** Pitchers scored by `SP Score` alone without showing the component factors.

---

## 5. ARCHITECTURAL & CODING DIRECTIVES

### 5.1 Backend: Fix the Pydantic Schema

```
File: schemas/lineup.py (or wherever LineupPlayerOut is defined)

Current (broken):
  injury_status: str

Fix Option A (coerce at schema level):
  from pydantic import field_validator

  injury_status: str = "HEALTHY"

  @field_validator("injury_status", mode="before")
  @classmethod
  def coerce_injury_status(cls, v):
      if isinstance(v, bool):
          return "IL" if v else "HEALTHY"
      if v is None:
          return "HEALTHY"
      return str(v)

Fix Option B (fix upstream data source):
  In the data-fetching layer (e.g., yahoo_api.py, data_pipeline.py),
  ensure the raw API response's injury field is mapped to a string enum
  BEFORE being passed to the Pydantic model.
```

### 5.2 Backend: Fix Schedule Data Pipeline

```
Investigate the game-schedule data source. The following teams have games
on 2026-03-30 but are returning no data:
  - HOU (vs BOS, 7:10 PM ET)
  - ARI (vs DET, 9:10 PM ET)
  - CLE (vs SEA)
  - SF (@ SD, 9:40 PM ET)

Likely root causes (check in this order):
1. Team abbreviation mismatch (e.g., "AZ" vs "ARI", "CWS" vs "CHW")
2. Schedule API returning UTC dates causing off-by-one filtering
3. Stale cache not refreshing after Opening Day schedule updates
4. API rate-limiting or silent failures without error logging

Add structured logging to the schedule-fetch function:
  logger.info(f"Fetched {len(games)} games for {date}")
  logger.warning(f"No game found for team {team_abbr} on {date}")
```

### 5.3 Frontend: Component Refactoring

```
Components that need refactoring:

1. WarningBanner (new component)
   - Parse backend warnings into structured objects
   - Render user-friendly messages only
   - NEVER render raw error strings, stack traces, or Pydantic errors
   - Add severity levels: info (blue), warning (amber), error (red)

2. PlayerRow (refactor existing)
   - Add injury_status badge (colored pill: green=healthy, red=IL, yellow=DTD)
   - Add opposing pitcher name + hand for batters
   - Add confidence indicator for start/sit recommendation
   - Add aria-label with full player context

3. ScoreCell (refactor existing)
   - Replace raw number with:
     a. Color gradient (red → yellow → green based on percentile)
     b. Tooltip showing: Score = f(Implied Runs, Park Factor, ...)
     c. Percentile rank badge (e.g., "87th")
   - Render "—" for players with no game data (not 4.500)

4. PitcherTable (refactor existing)
   - Add opponent team name column
   - Add pitcher handedness indicator
   - Replace "UNKNOWN" action with "NO START" or "IL" based on status
   - Replace 0.000 SP Score with "—"

5. Sidebar (fix)
   - Remove or env-gate the localhost:8501 Streamlit link
   - Add responsive collapse behavior at < 768px
```

### 5.4 Data Layer: New Integrations Required

```
Priority integrations (in build order):

1. Yahoo Fantasy API — injury status, positional eligibility, league settings
   → Fixes the Pydantic crash AND enables multi-position slot validation

2. FanGraphs API — xwOBA, SIERA, CSW%, platoon splits, game logs
   → Powers the advanced metrics columns

3. Baseball Savant (Statcast) — xwOBA, barrel%, sprint speed, pitch data
   → Requires scraping or the public CSV dumps

4. Rotowire/ESPN — confirmed daily lineups, batting order position
   → Critical for knowing if a player is actually in the real-life lineup

5. Weather API — wind speed/direction, temperature, precipitation
   → Park-factor adjustment multiplier

6. UmpScorecard — home plate umpire assignment + zone tendencies
   → Pitcher-specific adjustment factor
```

### 5.5 Testing Directives

```
Add the following test cases immediately:

Unit Tests:
- test_injury_status_coercion: Pass bool, None, and string to LineupPlayerOut
- test_no_game_player_excluded_from_starts: Players with no game data must
  not receive START recommendations
- test_slot_assignment_respects_eligibility: 2B-only player cannot fill 3B slot
- test_score_fallback_is_flagged: Score=4.500 + PF=1.000 must be flagged
  as "default/missing data", not treated as a real projection

Integration Tests:
- test_schedule_returns_all_active_teams: For any given MLB game day,
  verify all 30 teams return game data (or explicit "off day" status)
- test_optimize_endpoint_returns_valid_json: The /api/optimize (or equivalent)
  endpoint must never return Pydantic error strings in the response body

E2E Tests:
- test_warning_banner_no_raw_errors: Assert that the warning banner
  never contains "validation error", "Traceback", or "pydantic"
- test_mobile_viewport_renders: At 390px width, assert sidebar is collapsed
  and tables are scrollable
```

---

## APPENDIX A: Full Roster Audit (2026-03-30)

### Batters

| Player | Pos | Team | App Shows | Real-World Status | Verdict |
|--------|-----|------|-----------|-------------------|---------|
| Byron Buxton | CF | MIN | vs KC, 4:11 PM, Score 5.390, START | MIN vs KC confirmed | ✅ Correct |
| Pete Alonso | 1B | BAL | vs TEX, 6:36 PM, Score 5.145, START | BAL vs TEX confirmed | ✅ Correct |
| Jordan Westburg | 2B | BAL | vs TEX, 6:36 PM, Score 5.145, START | BAL vs TEX confirmed | ✅ Correct |
| Willi Castro | 2B | COL | vs TOR, 7:08 PM, Score 5.100, 3B slot | COL vs TOR confirmed | ⚠️ Wrong slot (2B in 3B) |
| Yainer Diaz | C | HOU | No game (—), Score 4.500, START | HOU vs BOS 7:10 PM | ❌ Game exists, data missing |
| Geraldo Perdomo | SS | ARI | No game (—), Score 4.500, START | ARI vs DET 9:10 PM | ❌ Game exists, data missing |
| Steven Kwan | LF | CLE | No game (—), Score 4.500, BENCH | CLE vs SEA confirmed | ❌ Game exists, data missing |
| Matt Chapman | 3B | SF | No game (—), Score 4.500, BENCH | SF @ SD 9:40 PM | ❌ Game exists, data missing |
| Luis Arraez | 1B | SF | No game (—), Score 4.500, BENCH | SF @ SD 9:40 PM | ❌ Game exists, data missing |
| Pete Crow-Armstrong | CF | CHC | vs LAA, 7:41 PM, Score 4.080, START | CHC vs LAA confirmed | ✅ Correct |
| Seiya Suzuki | LF | CHC | vs LAA, 7:41 PM, Score 4.080, START | CHC vs LAA confirmed | ✅ Correct |
| Vinnie Pasquantino | 1B | KC | vs MIN, 4:11 PM, Score 3.920, Util | KC vs MIN confirmed | ✅ Correct |
| Marcus Semien | 2B | NYM | vs STL, 7:46 PM, Score 3.750, BENCH | NYM vs STL confirmed | ✅ Correct (but see note) |

**Note on Marcus Semien:** Benched with a 3.750 score while Yainer Diaz (4.500 default) and Geraldo Perdomo (4.500 default) are started. If the data pipeline were working, Diaz and Perdomo would likely have real scores that could be higher or lower — the current START recommendations for them are based on fake data.

### Starting Pitchers

| Pitcher | Team | App Shows | Real-World Status | Verdict |
|---------|------|-----------|-------------------|---------|
| Eury Perez | MIA | 6:41 PM, OppImp 4.75, Score 5.500, START | Likely starting for MIA | ✅ Reasonable |
| Shota Imanaga | CHC | 7:41 PM, OppImp 5.50, Score 4.900, START | CHC vs LAA confirmed | ✅ Reasonable |
| Cristopher Sanchez | PHI | 6:41 PM, OppImp 5.25, Score 4.650, START | Likely starting for PHI | ✅ Reasonable |
| Garrett Crochet | BOS | No start (—), Score 0.000, UNKNOWN | Struggled in spring (7.36 ERA), status unclear | ⚠️ Needs injury/status data |
| Gavin Williams | CLE | No start (—), Score 0.000, UNKNOWN | CLE has a game; may not be in rotation spot | ⚠️ Needs clarity |
| Blake Snell | LAD | No start (—), Score 0.000, UNKNOWN | Reported likely to start season on IL (shoulder) | ❌ Should show IL status |

---

## APPENDIX B: Score Formula Reverse-Engineering

From observed data, the Score does NOT equal `Implied Runs × Park Factor`:

| Player | Implied Runs | Park Factor | IR × PF | Actual Score | Delta |
|--------|-------------|-------------|---------|--------------|-------|
| Byron Buxton | 4.00 | 0.980 | 3.920 | 5.390 | +1.470 |
| Pete Alonso | 3.75 | 0.980 | 3.675 | 5.145 | +1.470 |
| Jordan Westburg | 3.75 | 0.980 | 3.675 | 5.145 | +1.470 |
| Willi Castro | 3.50 | 1.020 | 3.570 | 5.100 | +1.530 |
| Pete Crow-Armstrong | 5.50 | 1.020 | 5.610 | 4.080 | -1.530 |
| Vinnie Pasquantino | 5.50 | 0.980 | 5.390 | 3.920 | -1.470 |

**Observation:** There appears to be an inverse relationship for some players — higher implied runs correlate with LOWER scores (Crow-Armstrong, Pasquantino). The Score likely incorporates a factor that inverts the opponent's implied runs (i.e., your team's implied runs are good, but your opponent's implied runs are bad for you if you're a batter on the other side). This formula should be explicitly documented and visible to users via a tooltip.

---

*Generated: 2026-03-30 | For ingestion by Claude Code autonomous coding agent*
