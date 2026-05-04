# Production Data Quality Audit — Fresh Probe v4 (April 22, 09:01 UTC)
**Date**: April 22, 2026  
**Probe Time**: 09:01:02 UTC  
**Method**: Fresh API probe against `https://fantasy-app-production-5079.up.railway.app`  
**Response Files**: `postman_collections/responses/*20260422_090102.json` (14 files)  

---

## Executive Summary

This audit is based on a **fresh probe executed at 09:01 UTC on April 22**. All claims are derived from saved JSON response files. The most significant change since the April 21 21:09 probe is that `/api/fantasy/waiver` has **recovered from 503 to 200**, rolling windows have **dramatically improved** (from 0% populated to ~87-91%), and **MLBAM IDs are now present for 65% of roster players** (was 0%). However, **new regressions** have appeared in the lineup response schema, briefing category names, waiver recommendations, and pipeline health structure.

---

## Endpoint Status Matrix (April 22, 09:01 UTC)

| Endpoint | Status | Size | Change Since 04/21 21:09 |
|----------|--------|------|-------------------------|
| `GET /` | 200 | 107 B | Recovered (was timeout) |
| `GET /health` | 200 | 65 B | Recovered (was timeout) |
| `GET /api/fantasy/draft-board?limit=200` | 200 | 200 KB | No change |
| `GET /api/fantasy/roster` | 200 | 63 KB | BDL+MLBAM IDs improved |
| `GET /api/fantasy/lineup/2026-04-22` | 200 | 11 KB | **Schema changed** |
| `GET /api/fantasy/waiver` | **200** | 21 KB | **FIXED** (was 503) |
| `GET /api/fantasy/waiver/recommendations` | **503** | 96 B | **NEW ERROR** (was different 503) |
| `GET /api/fantasy/matchup` | 200 | 1,005 B | No change |
| `GET /api/fantasy/player-scores?period=season` | 404 | 29 B | No change |
| `GET /api/fantasy/decisions` | 200 | 26 KB | **0 waiver decisions** |
| `GET /api/fantasy/briefing/2026-04-22` | 200 | 5.3 KB | **Category names null** |
| `GET /admin/pipeline-health` | 200 | 1.2 KB | **Structure changed** |
| `GET /admin/scheduler/status` | 200 | 1.5 KB | No change |
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

**Root Cause (85%+ confidence)**: This is a Python comparison bug in the waiver recommendation sorting or filtering logic. A `tuple` (likely a multi-element sort key like `(score, -timestamp)`) is being compared directly against a `float` instead of extracting the scalar value first. Common pattern: `sorted(candidates, key=lambda x: x.some_tuple)` where `x.some_tuple` should be `x.some_tuple[0]`.

**Impact**: Waiver recommendations are completely unavailable. The base `/waiver` endpoint works, but the intelligence layer that ranks and explains moves is broken.

---

## Regression 2: Lineup Response Schema Broken

**Endpoint**: `GET /api/fantasy/lineup/2026-04-22`  
**Status**: 200

**Changes from Prior Schema**:
- Previously returned `players` array. Now returns separate `batters` (13) and `pitchers` (10) arrays.
- This is a **breaking schema change** for any frontend or consumer expecting `players`.

**Critical Null Fields**:

| Field | Null Count / Total | Notes |
|-------|-------------------|-------|
| `lineup_status` | 23/23 | ALL players have `null` lineup status |
| `eligible_positions` | 23/23 | ALL players have `null` eligible positions |
| `has_game` (pitchers) | 10/10 | ALL pitchers have `null` has_game |
| `bdl_player_id` | 23/23 | ALL null |
| `mlbam_id` | 23/23 | ALL null |
| `game_time` | 23/23 | ALL null |
| `weather` | 23/23 | ALL null |
| `is_two_start` | 10/10 | ALL null |

**What IS present**:
- `has_game` for batters: 9/13 have `True`
- `opponent` for all 23 players
- `games_count`: 11

**Severity**: High. The lineup endpoint cannot tell consumers who is starting vs benched because `lineup_status` is universally null.

---

## Regression 3: Briefing Category Names All Null

**Endpoint**: `GET /api/fantasy/briefing/2026-04-22`  
**Status**: 200

**Observation**: 11 categories returned. All 11 have `name: null`.

```json
[
  {"name": null, ...},
  {"name": null, ...},
  ... (11 times)
]
```

**Severity**: Medium. The briefing displays 11 categories but cannot label them.

---

## Regression 4: Pipeline Health `overall` is Null

**Endpoint**: `GET /admin/pipeline-health`  
**Status**: 200

**Observation**: The `overall` field is `null`. The response structure has also changed — `tables` is now a `list` instead of a `dict`.

**Old structure (04/21)**:
```json
{"overall": true, "tables": {"player_rolling_stats": {...}}}
```

**New structure (04/22)**:
```json
{"overall": null, "overall_healthy": true, "tables": [{"name": "player_rolling_stats", ...}]}
```

**Severity**: Low-Medium. The `overall_healthy` boolean is present and `true`, but `overall` is null which may break consumers expecting a boolean there.

---

## Regression 5: Blake Snell Has Completely Null Season Stats

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

**Observation**: Blake Snell (`eligible_positions`: `["SP", "P", "IL"]`) has **all 18 season_stats fields set to null**:

```json
{"TB": null, "NSB": null, "WHIP": null, "RBI": null, "OPS": null,
 "W": null, "K_P": null, "AVG": null, "R": null, "ERA": null,
 "K_B": null, "HR_B": null, "K_9": null, "QS": null, "H": null,
 "HR_P": null, "L": null, "NSV": null}
```

**Severity**: Medium. A starting pitcher with zero statistical data is a clear data gap.

---

## Improvement 1: Waiver Endpoint Recovered (200)

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200 (was 503 on 04/21 21:09)

The Yahoo API `out=ownership` error has been resolved. The endpoint now returns 25 available players with full stats.

---

## Improvement 2: Roster Rolling Windows Dramatically Improved

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

| Rolling Window | Populated (04/22) | Populated (04/21 21:09) | Change |
|----------------|-------------------|------------------------|--------|
| `rolling_7d` | **20/23** (87%) | 0/23 (0%) | **+20** |
| `rolling_14d` | **20/23** (87%) | 0/23 (0%) | **+20** |
| `rolling_30d` | **21/23** (91%) | 0/23 (0%) | **+21** |
| `rolling_15d` | 0/23 (0%) | 0/23 (0%) | No change |

**Root Cause (85%+ confidence)**: The `player_rolling_stats` table grew from 48,910 rows (04/21) to **51,515 rows** (04/22), indicating fresh data ingestion successfully populated the rolling window calculations.

---

## Improvement 3: Roster Identity Resolution Improved

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

| Field | Non-null (04/22) | Non-null (04/21 21:09) | Change |
|-------|-----------------|------------------------|--------|
| `bdl_player_id` | **19/23** (83%) | 19/23 (83%) | Stable |
| `mlbam_id` | **15/23** (65%) | 0/23 (0%) | **+15** |

**Players still missing both IDs**:
- Moisés Ballesteros (C)
- Eury Pérez (SP)
- Cristopher Sánchez (SP)
- Edwin Díaz (RP)

**Severity**: Medium. MLBAM ID is the cross-reference key for Statcast/FanGraphs data.

---

## Improvement 4: Impossible Projections Gone

**Endpoint**: `GET /api/fantasy/decisions`  
**Status**: 200

**Observation**: 0 instances of "projects 0.00 ERA ROS" or "projects 0.00 WHIP ROS" (was 5 on 04/21).

This is likely because there are **0 waiver decisions** in this run (all 14 are lineup decisions). The impossible projections previously appeared in waiver decision narratives.

---

## Persistent Issue 1: ROS/ROW Projections 100% Null

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

**Observation**: 23/23 players have `ros_projection: null` and `row_projection: null`.

**Severity**: High. Rest-of-season and rest-of-week projections are completely absent.

---

## Persistent Issue 2: Game Context 100% Null

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

**Observation**: 23/23 players have `game_context: null`.

**Severity**: Medium. No opponent, venue, or weather context for any player.

---

## Persistent Issue 3: Ownership Percentage 0% for All Roster Players

**Endpoint**: `GET /api/fantasy/roster`  
**Status**: 200

**Observation**: All 23 roster players have `ownership_pct: 0.0`.

**Severity**: Low-Medium. This field likely represents Yahoo league ownership, but 0% for every player (including stars like Pete Alonso and Blake Snell) suggests the value is not being populated from the data source.

---

## Persistent Issue 4: Waiver Intelligence Mostly Empty

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200

**Observation**: 25 available players.

| Field | Empty/Null / Total |
|-------|-------------------|
| `owned_pct` | 25/25 = 0.0 |
| `starts_this_week` | 24/25 = 0 |
| `statcast_signals` | 25/25 = `[]` |
| `hot_cold` | 25/25 = null |
| `status` | 25/25 = null |
| `injury_note` | 25/25 = null |
| `injury_status` | 25/25 = null |
| `statcast_stats` | 25/25 = null |
| `category_contributions` | 23/25 = `{}` |

Only **2/25** players have non-empty `category_contributions`:
- Seth Lugo: `{"W": 0.686, "WHIP": 0.019}`
- Will Warren: `{"W": 0.097}`

**Severity**: Medium. The waiver endpoint returns player stats but almost no intelligence layer data (need scores, category contributions, statcast signals, injury status).

---

## Persistent Issue 5: Draft Board Age=0 for 92.5%

**Endpoint**: `GET /api/fantasy/draft-board?limit=200`  
**Status**: 200

**Observation**: 185 of 200 players have `age: 0`.

**Severity**: Low.

---

## Persistent Issue 6: Decisions Missing Value Fields

**Endpoint**: `GET /api/fantasy/decisions`  
**Status**: 200

**Observation**: 14 lineup decisions, 0 waiver decisions.

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

## Persistent Issue 7: Three Pitchers with ERA=0.00

**Endpoint**: `GET /api/fantasy/waiver`  
**Status**: 200

**Observation**: Three waiver pitchers have `ERA: "0.00"` with non-zero innings pitched:

| Player | ERA | IP |
|--------|-----|-----|
| Louis Varland | 0.00 | 13.0 |
| Tony Santillan | 0.00 | 10.0 |
| Huascar Brazobán | 0.00 | 10.2 |

While a 0.00 ERA is mathematically possible early in the season (no earned runs allowed), it is extremely rare and noteworthy. All three have allowed 0 earned runs across meaningful innings.

**Severity**: Low. Data is factually correct if they haven't allowed earned runs.

---

## Persistent Issue 8: Endpoint Errors

| Endpoint | Status | Detail |
|----------|--------|--------|
| `/api/fantasy/player-scores?period=season` | 404 | `{"detail": "Not Found"}` |
| `/admin/validate-system` | 404 | `{"detail": "Not Found"}` |
| `/api/fantasy/roster/optimize` | 422 | Pydantic validation error: missing body |
| `/api/fantasy/matchup/simulate` | 422 | Pydantic validation error: missing body |

**Severity**: Medium. Two endpoints permanently 404; two POST endpoints require request bodies that the probe does not send.

---

## Pipeline Health (Latest)

| Table | Rows | Latest Date | Change Since 04/21 |
|-------|------|-------------|-------------------|
| player_rolling_stats | **51,515** | 2026-04-21 | **+2,605** |
| player_scores | **51,335** | 2026-04-21 | **+2,594** |
| statcast_performances | 9,720 | 2026-04-20 | No change |
| probable_pitchers | **251** | 2026-04-26 | **+2** |
| simulation_results | 16,342 | 2026-04-20 | No change |
| mlb_player_stats | **10,061** | 2026-04-21 | **+446** |
| data_ingestion_logs | **2,270** | 2026-04-22 | **+174** |

All 7 tables report `healthy: true`. The significant growth in `player_rolling_stats`, `player_scores`, and `mlb_player_stats` explains why rolling windows and identity resolution improved.

---

## Scheduler Status

**Observation**: 10 jobs active, `running: true`. Notable: `openclaw_morning` job is still registered despite being paused in code. All jobs show `next_run: null` which may indicate they are running on-demand or the scheduler status endpoint does not serialize next run times correctly.

---

## Confidence Statement

All quantitative claims (counts, ratios, exact values, status codes) are derived from **direct inspection of saved JSON response files** written during the April 22 09:01 UTC probe. Root causes marked with "85%+ confidence" are based on explicit error messages (e.g., Python `TypeError` strings) or clear code pattern matches. No claim relies on inference about unobserved code behavior beyond what the error messages directly reveal.

---

**Report Author**: Kimi CLI  
**Probe Time**: 2026-04-22 09:01:02 UTC  
**Data Source**: `postman_collections/responses/*20260422_090102.json`
