# Production API Probe Report
**Date:** 2026-04-15
**Target:** `https://fantasy-app-production-5079.up.railway.app`
**Endpoints Hit:** 14 (12× 200, 2× 404, 1× 422)
**Probe Script:** `postman_collections/api_probe.py`
**Response Archive:** `postman_collections/responses/*.json`

---

## Executive Summary

The production API is **functionally alive but data-blind** in critical paths. While the draft board and decisions APIs return rich structured data, the lineup engine cannot see the schedule, waiver intelligence is hollow, and matchup data is essentially empty. **Nine distinct data gaps** render the daily operational loop unreliable. The most urgent issues are (1) Odds API schedule lookup failure causing all players to bench, and (2) projection math producing impossible values (0.00 ERA, 91.2 HR ROS).

---

## 1. Endpoint Status Matrix

| Endpoint | Status | Assessment |
|----------|--------|------------|
| `GET /` | 200 | OK — API info page |
| `GET /health` | 200 | OK — DB reachable |
| `GET /api/fantasy/draft-board?limit=200` | 200 | **Rich data** — Steamer projections, z-scores, risk adjustments |
| `GET /api/fantasy/roster` | 200 | **Partial** — 6 proxy players with empty cat_scores |
| `GET /api/fantasy/lineup/2026-04-20` | 200 | **Broken** — all players benched, no schedule data |
| `GET /api/fantasy/waiver` | 200 | **Hollow** — category_contributions empty, all intelligence fields null |
| `GET /api/fantasy/waiver/recommendations` | 200 | **Weak** — only 1 rec, always drops Crochet, win_prob ~1.0 |
| `GET /api/fantasy/matchup` | 200 | **Empty** — my team stats all 0, missing ERA/WHIP/K9 |
| `GET /api/fantasy/decisions` | 200 | **Rich but buggy** — good factor breakdowns, impossible projections |
| `GET /api/fantasy/briefing/2026-04-20` | 200 | **Broken** — categories=[], starters=[], all benched |
| `GET /admin/pipeline-health` | 200 | **Healthy tables** — 7/7 tables OK, data through 2026-04-19 |
| `GET /admin/scheduler/status` | 200 | OK |
| `GET /api/fantasy/player-scores?period=season` | **404** | Route missing or param invalid |
| `GET /admin/validate-system` | **404** | Route missing |
| `POST /api/fantasy/roster/optimize` | **404** | Route missing |
| `POST /api/fantasy/matchup/simulate` | **422** | Missing required request body |

---

## 2. Critical Data Gap #1: Lineup Engine — Total Schedule Blindness

**Affected Endpoints:** `/api/fantasy/lineup/{date}`, `/api/fantasy/briefing/{date}`

**Symptoms:**
- All 14 batters have `position: "?"`, `implied_runs: 4.5`, `park_factor: 1.0`, `lineup_score: -4.375`
- `has_game: false` for **every** player
- All batters moved to `BENCH` with reason: `"no game data for [TEAM] on 2026-04-20 (Odds API coverage gap)"`
- Only 1 pitcher (Arrighetti) has `status: "START"`; all others `NO_START`
- `games_count: 0`, `no_games_today: true`
- Briefing `categories: []`, `starters: []`, all on bench

**Root Cause Hypothesis:** The Odds API schedule lookup (`_get_schedule_for_date` in `daily_ingestion.py` or `matchup_engine.py`) returns empty for 2026-04-20. This could be:
1. The Odds API genuinely has no April 2026 MLB data (offseason / simulated season mismatch)
2. The date parameter is being formatted incorrectly for the API call
3. The API key or endpoint for schedules has changed
4. The schedule data is stored in `probable_pitchers` (which has 226 rows through 2026-04-24) but the lineup API queries a different source

**Impact:** The entire daily lineup optimization loop is non-functional. Users cannot get meaningful start/sit recommendations because the system thinks nobody plays.

---

## 3. Critical Data Gap #2: Proxy Player Crisis

**Affected Endpoints:** `/api/fantasy/roster`, `/api/fantasy/lineup`, `/api/fantasy/decisions`

**Symptoms:**
- 6 of 23 roster players are `is_proxy: true` with empty `cat_scores`: {}

| Player | Yahoo Key | z_score | Issue |
|--------|-----------|---------|-------|
| Moisés Ballesteros | 469.p.60120 | -1.5 | No BDL mapping |
| Hyeseong Kim | 469.p.64858 | -0.8 | No BDL mapping |
| Munetaka Murakami | 469.p.66369 | -0.8 | No BDL mapping |
| Cam Smith | 469.p.64330 | -0.8 | No BDL mapping |
| Spencer Arrighetti | 469.p.60188 | -0.5 | No BDL mapping |
| Enyel De Los Santos | 469.p.11064 | -0.3 | No BDL mapping |

**Root Cause:** These are young/international players not in the BDL database. `PlayerIDMapping` has no `bdl_id` for their `yahoo_key`. The `z_score` values are hardcoded (-0.8 or -1.5) when `cat_scores` is empty, not computed.

**Impact:** Proxy players are systematically undervalued. In the decisions API, Kim and Murakami are paradoxically recommended as starters with high scores (91.2, 89.5) despite being proxies — this suggests the decision engine uses a different scoring path than the roster API.

---

## 4. Critical Data Gap #3: Waiver Intelligence — Empty Fields

**Affected Endpoint:** `/api/fantasy/waiver`

**Symptoms (present in ALL 25 returned players):**
- `category_contributions: {}` — completely empty
- `owned_pct: 0.0` — should show Yahoo ownership percentage
- `starts_this_week: 0` — never populated
- `hot_cold: null` — never populated
- `projected_saves: 0.0` — never populated
- `projected_points: 0.0` — never populated
- `statcast_signals: []` — never populated
- `stats` values are **strings**, not numbers (e.g., `"IP": "24.1"`, `"K": "1"`)

**Root Cause:** The waiver endpoint in `main.py` likely returns raw Yahoo free agent data wrapped in a lightweight schema. The enrichment pipeline (category contributions, ownership %, hot/cold signals, statcast) is not wired up.

**Impact:** The waiver wire page shows raw stats but no intelligence. Users cannot see which categories a free agent helps, how hot they are, or their ownership rate.

---

## 5. Critical Data Gap #4: Waiver Recommendations — Monoculture

**Affected Endpoint:** `/api/fantasy/waiver/recommendations`

**Symptoms:**
- Only **1** recommendation returned (Seth Lugo → drop Spencer Arrighetti)
- In the decisions API, **all 22 waiver decisions** recommend dropping the same player: **Garrett Crochet** (bdl_id 555)
- `win_prob_before: 1.0`, `win_prob_after: 0.999` — win probability delta is effectively zero
- `regression_delta: 0.0` for all
- `category_targets: []` for all
- `stats: {}` for the add_player in recommendations

**Root Cause:** The `optimize_waivers()` function in `decision_engine.py` uses `_composite_value()` and world-with/world-without math. If Garrett Crochet has the lowest composite value on the roster (or a bug makes him appear as 0.000), he becomes the universal drop target. The win probability model may be saturated (always returns ~1.0) due to missing opponent data.

**Impact:** Waiver recommendations are not credible when they always suggest the same drop. The system appears to have no nuanced roster evaluation.

---

## 6. Critical Data Gap #5: Matchup Stats — Near-Empty

**Affected Endpoint:** `/api/fantasy/matchup`

**Symptoms:**
- My team: 0 in R, H, HR, RBI, IP, W, GS, K, QS, K/BB, K(B), NSV, OBP
- My team: **missing** AVG, OPS, ERA, WHIP, K/9 entirely
- Opponent: minimal stats (NSB: "1/5", H: 1, W: 1, AVG: ".200", OPS: ".400")
- Opponent: **missing** ERA, WHIP, K/9, R, HR, RBI, IP, GS, K, QS, K/BB, K(B), NSV, OBP

**Root Cause:** The matchup endpoint aggregates Yahoo team stats for the current week. Week 5 (2026-04-20) has just started, so accumulated stats are near-zero. However, the **schema inconsistency** (some categories present, others missing) suggests the category mapping in the matchup service is incomplete or the Yahoo API response is being filtered incorrectly.

**Impact:** Users cannot see a meaningful category-by-category comparison with their opponent.

---

## 7. Critical Data Gap #6: Projection Math Produces Impossible Values

**Affected Endpoint:** `/api/fantasy/decisions`

**Symptoms (sampled from 31 decisions):**
- Daniel Lynch IV: `"Projects 0.00 WHIP ROS"` — impossible
- Alex Vesia: `"Projects 0.00 ERA ROS"` — impossible
- Andrew Alvarez: `"Projects 0.00 ERA ROS"` — impossible
- Mickey Moniak: `"Projects 39.2 HR ROS"` — from 14 games, projects 39 HR? Possible but check math
- Dalton Rushing: `"Projects 91.2 HR ROS"` — from 6 games, projects 91 HR — **impossible**
- Dalton Rushing: `"Projects 204.4 RBI ROS"` — **impossible**
- Andrew Alvarez: `"Projects 130 K ROS"` — from 1 game, projects 130 K

**Root Cause:** The projection math in `decision_engine.py` or `scoring_engine.py` uses a naive rate × remaining-games formula without caps or regression. For players with tiny samples (1-6 games), the rate is extrapolated across ~140 remaining games, producing absurd totals. The `0.00 ERA/WHIP` suggests division-by-zero protection is missing or the raw stat is exactly 0.00.

**Impact:** Confidence in the decision engine is undermined when it claims a player will hit 91 HR or have 0.00 ERA. The narrative text is exposed directly to users.

---

## 8. Critical Data Gap #7: Missing Route Implementations

**Affected Endpoints:**
- `GET /api/fantasy/player-scores?period=season` → 404
- `GET /admin/validate-system` → 404
- `POST /api/fantasy/roster/optimize` → 404

**Root Cause:** These routes are referenced in documentation or the Postman collection but not implemented in `backend/main.py`.

**Impact:** Frontend or external tools expecting these endpoints will fail. The `roster/optimize` POST endpoint is particularly important as it would trigger the lineup optimizer on demand.

---

## 9. Positive Findings

| Area | Finding |
|------|---------|
| **Draft Board** | Rich Steamer projections with park-adjusted and risk-adjusted z-scores. All 9 batter categories present. |
| **Decisions Explanations** | Excellent factor-level breakdowns with weights, labels (ELITE/STRONG/AVERAGE/WEAK), and narratives. |
| **Pipeline Health** | 7/7 tables healthy. `player_rolling_stats`: 46,302 rows. `player_scores`: 46,148 rows. `statcast_performances`: 9,436 rows. Data through 2026-04-19. |
| **Waiver Recommendations** | `category_win_probs` has meaningful values across all 18 categories. MCMC enabled flag present. |
| **Roster Mapping** | `selected_position` correctly reflects Yahoo slot assignments. Injury notes present for Soto, Snell, Westburg. |

---

## 10. Decisions API Deep Dive

**Composition:** 31 total decisions
- 9 lineup decisions (filling active slots)
- 22 waiver decisions (ADD_DROP recommendations)

**Lineup Decisions:**
- SP: Arrighetti (score 96.0), Gavin Williams (90.7)
- RP: Enyel De Los Santos (80.0)
- 1B: Pete Alonso (implied), Munetaka Murakami (89.5)
- 2B: Hyeseong Kim (91.2)
- 3B: Jordan Walker (94.6 at Util)
- SS: Geraldo Perdomo
- OF: Pete Crow-Armstrong, Seiya Suzuki
- Util: Vinnie Pasquantino

**Waiver Decisions Pattern:**
- Drop target: **Garrett Crochet (bdl_id 555)** in 100% of recommendations
- Add targets: Alex Vesia, Daniel Lynch IV, Dalton Rushing, Mickey Moniak, Andrew Alvarez, Will Warren, Tony Santillan, Aaron Ashby, etc.
- Value gains range from +0.14 (Lynch IV) to +2.25 (Moniak)
- Confidence range: 0.75–0.90

---

## 11. Waiver Pool Data Quality

**Sample of returned free agents (25 players):**

| Name | Position | need_score | Key Stats |
|------|----------|------------|-----------|
| Seth Lugo | SP | 1.034 | IP 24.1, ERA 1.48, WHIP 0.99, K/9 7.77, NSV 3 |
| Michael Wacha | SP | -0.081 | IP 27.0, ERA 1.00, WHIP 0.78, NSV 4 |
| Alex Vesia | RP | -0.272 | IP 8.2, ERA 0.00, WHIP 0.58, K/9 10.38 |
| Rico Garcia | RP | -0.300 | IP 10.0, ERA 0.00, WHIP 0.30 |
| Aaron Ashby | RP | -0.300 | IP 14.0, ERA 3.21, WHIP 1.36, K/9 14.14 |
| Landen Roupp | SP | -0.500 | IP 22.2, ERA 2.38, WHIP 0.97, NSV 3 |
| Ildemaro Vargas | 1B | -0.800 | NSB 21/57, R 11, H 21, HR 2, AVG .368, OPS .986 |
| Mickey Moniak | LF | -0.800 | NSB 15/55, R 10, H 15, HR 6, AVG .273, OPS .960 |
| Josh Bell | 1B | -0.800 | NSB 20/76, R 17, H 20, HR 3, AVG .263, OPS .799 |
| Will Warren | SP | -1.078 | IP 25.1, ERA 2.49, WHIP 1.11, K/9 11.01 |
| Dalton Rushing | C | -1.500 | NSB 10/22, R 8, H 10, HR 5, AVG .455, OPS 1.705 |

**Observations:**
- Pitcher stats show `K` (wins? strikeouts as pitcher?) as small integers (0–5). This is likely **wins**, not Ks.
- Batter stats include `IP` and `W` which are pitcher fields — schema pollution.
- `OBP` in pitcher stats appears to be walks allowed or something unrelated.
- NSB format is `"SB/PA"` for batters (e.g., "21/57").

---

## 12. Recommendations by Priority

### P0 — Fix Before Next Game Day
1. **Fix schedule lookup** — Investigate why Odds API returns no games for 2026-04-20. Check `probable_pitchers` table (226 rows, latest 2026-04-24) as alternative source. The lineup engine should fallback to this table.
2. **Cap projection extrapolation** — Add maximum plausible ROS totals (e.g., 60 HR, 200 RBI, 0.50 ERA minimum) to prevent impossible projections in decision narratives.
3. **Fix universal drop target bug** — Investigate why Garrett Crochet is the drop target in 100% of waiver decisions. Check `_composite_value()` return for bdl_id 555.

### P1 — Fix This Week
4. **Populate waiver intelligence fields** — Wire up `category_contributions`, `owned_pct`, `hot_cold`, `statcast_signals`, `projected_saves` in the waiver endpoint.
5. **Implement missing routes** — Add `/api/fantasy/player-scores`, `/admin/validate-system`, and `POST /api/fantasy/roster/optimize` to `main.py`.
6. **Fix matchup category schema** — Ensure all 18 v2 categories are present for both teams, with 0 as default rather than missing keys.

### P2 — Fix Before Playoffs
7. **Proxy player enrichment** — Backfill BDL IDs for the 6 proxy players, or implement a fallback scoring path that uses Yahoo stats directly when BDL mapping is missing.
8. **Statcast integration** — `statcast_performances` has 9,436 rows. Surface `xwOBA`, `barrel_pct`, `hard_hit_pct` in waiver and lineup APIs.
9. **Projection pipeline** — `PlayerProjection` ORM has 0 rows. Build or connect the Steamer/ZiPS ingestion pipeline so `projections_bridge.py` can function.

---

## Appendix: Response File Inventory

| File | Endpoint | Status | Size |
|------|----------|--------|------|
| `draft_board_200.json` | `/api/fantasy/draft-board?limit=200` | 200 | ~1.2 MB |
| `roster_200.json` | `/api/fantasy/roster` | 200 | 571 lines |
| `lineup_200.json` | `/api/fantasy/lineup/2026-04-20` | 200 | 400 lines |
| `waiver_200.json` | `/api/fantasy/waiver` | 200 | 743 lines |
| `waiver_recommendations_200.json` | `/api/fantasy/waiver/recommendations` | 200 | 69 lines |
| `matchup_200.json` | `/api/fantasy/matchup` | 200 | 47 lines |
| `decisions_200.json` | `/api/fantasy/decisions` | 200 | ~1,800 lines |
| `briefing_200.json` | `/api/fantasy/briefing/2026-04-20` | 200 | 87 lines |
| `pipeline_health_200.json` | `/admin/pipeline-health` | 200 | 56 lines |
| `scheduler_status_200.json` | `/admin/scheduler/status` | 200 | small |
| `root_200.json` | `/` | 200 | small |
| `health_200.json` | `/health` | 200 | small |
| `player_scores_404.json` | `/api/fantasy/player-scores?period=season` | 404 | small |
| `validate_system_404.json` | `/admin/validate-system` | 404 | small |
| `roster_optimize_404.json` | `/api/fantasy/roster/optimize` | 404 | small |
| `matchup_simulate_422.json` | `/api/fantasy/matchup/simulate` | 422 | small |
