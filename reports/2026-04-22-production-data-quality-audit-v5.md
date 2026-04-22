# Production Data Quality Audit — Fresh Probe v5 (April 22, 10:16 UTC)
**Date**: April 22, 2026  
**Probe Time**: 10:16:29 UTC  
**Method**: Fresh API probe against `https://fantasy-app-production-5079.up.railway.app`  
**Response Files**: `postman_collections/responses/*20260422_101629.json` (14 files)  

---

## Executive Summary

This audit is based on a **fresh probe executed at 10:16 UTC on April 22**, with all prior response files deleted first to guarantee no stale data. The most significant changes since the 09:01 probe are that the **lineup endpoint has been partially fixed** (batters now have `lineup_status` and `eligible_positions`), **briefing category names are now populated**, **pipeline `overall` is no longer null**, and the **OpenClaw morning job has been removed from the scheduler**. However, **waiver recommendations remain broken** with the same TypeError, and several data quality issues persist.

---

## Endpoint Status Matrix (April 22, 10:16 UTC)

| Endpoint | Status | Size | Change Since 04/22 09:01 |
|----------|--------|------|-------------------------|
| `GET /` | 200 | 129 B | No change |
| `GET /health` | 200 | 82 B | No change |
| `GET /api/fantasy/draft-board?limit=200` | 200 | 200 KB | No change |
| `GET /api/fantasy/roster` | 200 | 63 KB | No change |
| `GET /api/fantasy/lineup/2026-04-22` | 200 | 14 KB | **Pitcher data improved** |
| `GET /api/fantasy/waiver` | 200 | 21 KB | No change |
| `GET /api/fantasy/waiver/recommendations` | **503** | 96 B | No change |
| `GET /api/fantasy/matchup` | 200 | 953 B | Live stats updated |
| `GET /api/fantasy/player-scores?period=season` | 404 | 29 B | No change |
| `GET /api/fantasy/decisions` | 200 | 26 KB | No change |
| `GET /api/fantasy/briefing/2026-04-22` | 200 | 5.7 KB | **Category names fixed** |
| `GET /admin/pipeline-health` | 200 | 1.2 KB | **overall now True** |
| `GET /admin/scheduler/status` | 200 | 1.4 KB | **OpenClaw job removed** |
| `GET /admin/validate-system` | 404 | 29 B | No change |
| `POST /api/fantasy/roster/optimize` | 422 | 218 B | No change |
| `POST /api/fantasy/matchup/simulate` | 422 | 218 B | No change |

---

## Regression 1: Waiver Recommendations 503 with TypeError

**Endpoint**: `GET /api/fantasy/waiver/recommendations`  
**Status**: 503

**Error Message**:
```json
{"detail": "Unexpected error: '>' not supported between instances of 'float' and 'tuple'"}
```

**Root Cause (>85% confidence)**: A Python comparison bug in the waiver recommendation sorting or ranking logic. A `tuple` (likely a multi-element sort key or priority tuple) is being compared directly against a `float` instead of extracting the scalar value. Typical pattern: `sorted(candidates, key=lambda x: x.need_score)` where `need_score` is sometimes a tuple `(score, tiebreaker)` instead of a float.

**Impact**: Waiver recommendations are completely unavailable. The base `/waiver` endpoint returns 200 with player stats, but the intelligence layer that ranks and explains moves is broken.

---

## Issue 1: Roster — Persistent Null Fields

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

### Fields That Are 100% Null (23/23)

| Field | Null Count | Severity | Notes |
|-------|-----------|----------|-------|
| `game_context` | 23/23 | High | No opponent, venue, weather, or start time for any player |
| `ros_projection` | 23/23 | High | No rest-of-season projections |
| `row_projection` | 23/23 | High | No rest-of-week projections |
| `rolling_15d` | 23/23 | Medium | 15-day rolling window absent for all players |
| `injury_return_timeline` | 23/23 | Medium | No injury timeline data |
| `ownership_pct` | 23/23 | Medium | All players show `0.0` — not a null but universally zero |
| `freshness.fetched_at` | 23/23 | Low | All `null`; `computed_at` is populated |

### Partially Null Fields

| Field | Null Count | Notes |
|-------|-----------|-------|
| `injury_status` | 20/23 | Only 3 players have injury data |
| `mlbam_id` | 8/23 | 15 players resolved (65%) |
| `bdl_player_id` | 4/23 | 19 players resolved (83%) |
| `rolling_7d` | 3/23 | 20 players have data (87%) |
| `rolling_14d` | 3/23 | 20 players have data (87%) |
| `rolling_30d` | 2/23 | 21 players have data (91%) |

### Illogical Values

**Blake Snell has completely null season stats**: Despite being a starting pitcher (`eligible_positions: ["SP", "P", "IL"]`), all 18 `season_stats.values` fields are `null`:
```json
{"TB": null, "NSB": null, "WHIP": null, "RBI": null, "OPS": null,
 "W": null, "K_P": null, "AVG": null, "R": null, "ERA": null,
 "K_B": null, "HR_B": null, "K_9": null, "QS": null, "H": null,
 "HR_P": null, "L": null, "NSV": null}
```

**Severity**: High for null projections, Medium for ownership_pct and rolling_15d.

---

## Issue 2: Lineup — Pitchers Still Missing Key Fields

**Endpoint**: `GET /api/fantasy/lineup/2026-04-22`  
**Status**: 200

**Improvements since 09:01**: Batters now have `lineup_status` and `eligible_positions` populated. Pitchers now have `has_game` (8/10 true) and `is_two_start` populated.

**Remaining Null Fields**:

| Field | Role | Null Count / Total |
|-------|------|-------------------|
| `lineup_status` | Pitcher | **10/10** |
| `eligible_positions` | Pitcher | **10/10** |
| `bdl_player_id` | Batter | 13/13 |
| `bdl_player_id` | Pitcher | 10/10 |
| `mlbam_id` | Batter | 13/13 |
| `mlbam_id` | Pitcher | 10/10 |
| `weather` | Batter | 13/13 |
| `weather` | Pitcher | 10/10 |
| `game_time` | Batter | 4/13 |
| `game_time` | Pitcher | 3/10 |

**Lineup Warnings**: 12 warnings returned, including:
- 4 batters with no game scheduled (Hyeseong Kim, Munetaka Murakami, Geraldo Perdomo, Brandon Nimmo)
- 3 players moved to BENCH due to "Odds API coverage gap"
- 7 SPs have no start today
- "Only 0 active pitcher slots filled -- consider streaming a SP"

**Severity**: Medium. Pitchers cannot be displayed with lineup status or eligible positions.

---

## Issue 3: Waiver Intelligence Mostly Empty

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200

**Observation**: 25 available players.

| Field | Empty/Null / Total |
|-------|-------------------|
| `owned_pct` | 25/25 = 0.0 |
| `starts_this_week` | 24/25 = 0 |
| `statcast_signals` | 25/25 = `[]` |
| `hot_cold` | 24/25 = null |
| `status` | 25/25 = null |
| `injury_note` | 25/25 = null |
| `injury_status` | 25/25 = null |
| `statcast_stats` | 25/25 = null |
| `category_contributions` | 23/25 = `{}` |

Only **2/25** players have non-empty `category_contributions`:
- Seth Lugo: `{"W": 0.686, "WHIP": 0.019, "QS": 0.981}`
- Will Warren: `{"W": 0.097, "QS": 0.562}`

**Severity**: Medium. The endpoint returns Yahoo player stats but almost no intelligence layer data.

---

## Issue 4: Decisions Missing Core Fields

**Endpoint**: `GET /api/fantasy/decisions`  
**Status**: 200

**Observation**: 14 lineup decisions, **0 waiver decisions**.

| Field | Null / Total |
|-------|-------------|
| `value_gain` | 14/14 |
| `risk_narrative` | 14/14 |
| `track_record_narrative` | 14/14 |
| `drop_player_id` | 14/14 |
| `drop_player_name` | 14/14 |

**as_of_date**: All 14 decisions are dated `2026-04-21` (not updated to 04-22).

**Severity**: Medium. No value gain estimates, risk assessment, or track record for any decision.

---

## Issue 5: Three Pitchers with ERA=0.00

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200

**Observation**: Three waiver pitchers have `ERA: "0.00"` with non-zero innings pitched:

| Player | ERA | IP |
|--------|-----|-----|
| Louis Varland | 0.00 | 13.0 |
| Tony Santillan | 0.00 | 10.0 |
| Huascar Brazobán | 0.00 | 10.2 |

A 0.00 ERA is mathematically possible if no earned runs have been allowed, but it is noteworthy for three pitchers with meaningful innings.

**Severity**: Low. Data is factually correct if they haven't allowed earned runs.

---

## Issue 6: Draft Board Age=0 for 92.5%

**Endpoint**: `GET /api/fantasy/draft-board?limit=200`  
**Status**: 200

**Observation**: 185 of 200 players have `age: 0`.

**Severity**: Low.

---

## Issue 7: Persistent Endpoint Errors

| Endpoint | Status | Detail |
|----------|--------|--------|
| `/api/fantasy/player-scores?period=season` | 404 | `{"detail": "Not Found"}` |
| `/admin/validate-system` | 404 | `{"detail": "Not Found"}` |
| `/api/fantasy/roster/optimize` | 422 | Pydantic validation: missing body |
| `/api/fantasy/matchup/simulate` | 422 | Pydantic validation: missing body |

**Severity**: Medium. Two endpoints permanently 404; two POST endpoints require request bodies.

---

## Improvements Since 04/22 09:01

### Lineup Endpoint Partially Fixed
- Batter `lineup_status`: now 0/13 null (was 13/13 null)
- Batter `eligible_positions`: now 0/13 null (was 13/13 null)
- Pitcher `has_game`: now 8/10 true (was 0/10 null)
- Pitcher `is_two_start`: now 0/10 null (was 10/10 null)

### Briefing Category Names Fixed
- Now returns readable names: `Runs`, `Hits`, `Home Runs`, `RBI`, `Stolen Bases`, `Batting Average`, `Wins`, `K`, `Saves`, `ERA`, `WHIP`
- Was all `null` at 09:01

### Pipeline Health Fixed
- `overall`: now `true` (was `null`)
- `overall_healthy`: `true`
- `healthy_count`: 7, `unhealthy_count`: 0

### OpenClaw Job Removed from Scheduler
- `openclaw_morning` job no longer appears in the scheduler job list (was present at 09:01)
- Confirms the cleanup from the prior session took effect

### Waiver Category Contributions Improved
- Seth Lugo now has `QS: 0.981` (was missing)
- Will Warren now has `QS: 0.562` (was missing)

### Matchup Live Data Updated
- My `K_B`: 13 (was 19), `TB`: 19 (was 13), `NSV`: 1 (was 0)
- Opp `K_B`: 8 (was 28), `TB`: 28 (was 8)

---

## Pipeline Health (Latest)

| Table | Rows | Latest Date | Change Since 09:01 |
|-------|------|-------------|-------------------|
| player_rolling_stats | 51,515 | 2026-04-21 | No change |
| player_scores | 51,335 | 2026-04-21 | No change |
| statcast_performances | 9,720 | 2026-04-20 | No change |
| probable_pitchers | 251 | 2026-04-26 | No change |
| simulation_results | 16,342 | 2026-04-20 | No change |
| mlb_player_stats | 10,061 | 2026-04-21 | No change |
| data_ingestion_logs | **2,287** | 2026-04-22 | **+17** |

All 7 tables report `healthy: true`.

---

## Scheduler Status

**Observation**: 9 jobs active (was 10), `running: true`. The `openclaw_morning` job has been removed. All jobs show `next_run: null` which indicates the scheduler status endpoint does not serialize next run times.

**Jobs**:
- `job_queue_processor`
- `capture_closing_lines`
- `update_outcomes`
- `nightly_decision_resolution`
- `settle_games_daily`
- `daily_snapshot`
- `statcast_daily_ingestion`
- `fetch_pybaseball`
- `mlb_nightly_analysis`

---

## Confidence Statement

All quantitative claims (counts, ratios, exact values, status codes) are derived from **direct inspection of saved JSON response files** written during the April 22 10:16 UTC probe. All prior response files were deleted before the probe executed. Root causes marked with ">85% confidence" are based on explicit error messages or clear code pattern matches.

---

**Report Author**: Kimi CLI  
**Probe Time**: 2026-04-22 10:16:29 UTC  
**Data Source**: `postman_collections/responses/*20260422_101629.json`
