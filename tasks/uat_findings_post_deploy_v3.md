# UAT Validation Report

**Generated:** 2026-04-20 20:41 ET
**Total Checks:** 77
**PASS:** 70  |  **FAIL:** 6  |  **WARN:** 1  |  **SKIP:** 0

## Summary by Endpoint

| Endpoint | HTTP | Time (ms) | PASS | FAIL | WARN |
|----------|------|-----------|------|------|------|
| GET /health | 200 | 343 | 2 | 0 | 0 |
| GET /api/fantasy/roster | 500 | 1140 | 0 | 1 | 0 |
| GET /api/fantasy/matchup | 200 | 1702 | 10 | 0 | 0 |
| GET /api/fantasy/scoreboard | 200 | 1188 | 7 | 0 | 0 |
| GET /api/fantasy/budget | 200 | 1358 | 12 | 1 | 0 |
| POST /api/fantasy/roster/optimize | 500 | 672 | 0 | 1 | 0 |
| GET /api/fantasy/waiver | 200 | 3671 | 7 | 2 | 0 |
| GET /api/fantasy/lineup/2026-04-20 | 200 | 3687 | 8 | 0 | 1 |
| GET /api/fantasy/briefing/2026-04-20 | 200 | 3593 | 5 | 0 | 0 |
| GET /api/fantasy/players/1/scores | 500 | 156 | 0 | 1 | 0 |
| GET /api/fantasy/decisions | 200 | 187 | 17 | 0 | 0 |
| GET /api/fantasy/decisions/status | 200 | 203 | 2 | 0 | 0 |

## FAILURES (must fix)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/roster | http_status | 200 | 500 |  |
| 2 | GET /api/fantasy/budget | ip_pace_valid | one of {'BEHIND', 'COMPLETE', 'ON_TRACK'} | 'behind' |  |
| 3 | POST /api/fantasy/roster/optimize | http_status | 200 | 500 |  |
| 4 | GET /api/fantasy/waiver | matchup_opponent | non-TBD name | 'TBD' |  |
| 5 | GET /api/fantasy/waiver | category_deficits_present | >0 categories | 0 |  |
| 6 | GET /api/fantasy/players/1/scores | http_status | 200 or 404 | 500 |  |

## WARNINGS (should investigate)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/lineup/2026-04-20 | lineup_warnings | empty ideally | 30 warnings: ['No games found for this date -- Odds API may not have data yet (requested: 2026-04-20). Lineup ranked by projections only.', 'Moisés Ballesteros: Starting but no game today'] | Lineup produced warnings |

## Full Check Details

### GET /health
HTTP 200 — 343ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | status_field | healthy | healthy |

### GET /api/fantasy/roster
HTTP 500 — 1140ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 500 |

### GET /api/fantasy/matchup
HTTP 200 — 1702ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | my_team_present | dict | dict |
| PASS | opponent_present | dict | dict |
| PASS | my_team_has_stats | >0 stats | 15 stats |
| PASS | opponent_has_stats | >0 stats | 15 stats |
| PASS | stat_keys_recognized | all recognized | all recognized |
| PASS | all_stat_values_numeric | all numeric | 0 non-numeric: [] |
| PASS | week_present | int | 5 |
| PASS | my_team_name | non-empty | 'Lindor Truffles' |
| PASS | opponent_team_name | non-empty | "Bartolo's Colon" |

### GET /api/fantasy/scoreboard
HTTP 200 — 1188ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | row_count | 18 | 18 |
| PASS | categories_match_canonical | exact 18: ['AVG', 'ERA', 'H', 'HR_B', 'HR_P', 'K_9', 'K_B', 'K_P', 'L', 'NSB', 'NSV', 'OPS', 'QS', 'R', 'RBI', 'TB', 'W', 'WHIP'] | missing=[] extra=[] |
| PASS | rows_have_required_fields | all fields present | 0 issues: [] |
| PASS | budget_present | dict | dict |
| PASS | win_probability_range | 0-1 | 0.0 |
| PASS | freshness_present | dict | dict |

### GET /api/fantasy/budget
HTTP 200 — 1358ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | budget.acquisitions_used_present | present | 0 |
| PASS | budget.acquisitions_remaining_present | present | 8 |
| PASS | budget.acquisition_limit_present | present | 8 |
| PASS | budget.il_used_present | present | 3 |
| PASS | budget.il_total_present | present | 3 |
| PASS | budget.ip_accumulated_present | present | 0.0 |
| PASS | budget.ip_minimum_present | present | 90.0 |
| PASS | budget.ip_pace_present | present | 'behind' |
| PASS | acquisitions_used_reasonable | 0 <= used <= 8 | 0 |
| PASS | il_used_reasonable | 0 <= 3 <= 3 | il_used=3, il_total=3 |
| **FAIL** | ip_pace_valid | one of {'BEHIND', 'COMPLETE', 'ON_TRACK'} | 'behind' |
| PASS | freshness_present | dict | dict |

### POST /api/fantasy/roster/optimize
HTTP 500 — 672ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 500 |

### GET /api/fantasy/waiver
HTTP 200 — 3671ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | top_available_count | >0 | 10 |
| PASS | all_have_names | 0 nameless | 0 nameless |
| PASS | all_have_player_id | 0 without ID | 0 without: [] |
| PASS | need_score_numeric | all numeric | 0 non-numeric |
| PASS | positions_valid | all valid | 0 invalid: [] |
| **FAIL** | matchup_opponent | non-TBD name | 'TBD' |
| **FAIL** | category_deficits_present | >0 categories | 0 |
| PASS | pagination_present | dict | dict |

### GET /api/fantasy/lineup/2026-04-20
HTTP 200 — 3687ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | batters_count | >0 | 14 |
| PASS | batters_have_names | 0 nameless | 0 nameless |
| PASS | batters_have_teams | 0 teamless | 0 teamless |
| PASS | batter_scores_numeric | all numeric | 0 non-numeric |
| PASS | active_batters_have_slots | 0 without slot | 0 active without slot |
| PASS | pitchers_present | list | list |
| PASS | games_count_present | int >= 0 | 0 |
| WARN | lineup_warnings | empty ideally | 30 warnings: ['No games found for this date -- Odds API may not have data yet (requested: 2026-04-20). Lineup ranked by projections only.', 'Moisés Ballesteros: Starting but no game today'] |

### GET /api/fantasy/briefing/2026-04-20
HTTP 200 — 3593ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | strategy_present | non-empty | 'standard' |
| PASS | categories_present | >0 | 11 |
| PASS | starters_present | >0 | 9 |
| PASS | overall_confidence | 0-1 float | 0.7 |

### GET /api/fantasy/players/1/scores
HTTP 500 — 156ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 or 404 | 500 |

### GET /api/fantasy/decisions
HTTP 200 — 187ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | count_matches | count=10 | count=10 |
| PASS | decision[0].bdl_player_id | present | 356 |
| PASS | decision[0].decision_type | present | 'lineup' |
| PASS | decision[0].confidence | present | 0.96 |
| PASS | decision[1].bdl_player_id | present | 539 |
| PASS | decision[1].decision_type | present | 'lineup' |
| PASS | decision[1].confidence | present | 0.946 |
| PASS | decision[2].bdl_player_id | present | 933897 |
| PASS | decision[2].decision_type | present | 'lineup' |
| PASS | decision[2].confidence | present | 0.912 |
| PASS | decision[3].bdl_player_id | present | 879 |
| PASS | decision[3].decision_type | present | 'lineup' |
| PASS | decision[3].confidence | present | 0.907 |
| PASS | decision[4].bdl_player_id | present | 69 |
| PASS | decision[4].decision_type | present | 'waiver' |
| PASS | decision[4].confidence | present | 0.9033 |

### GET /api/fantasy/decisions/status
HTTP 200 — 203ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | verdict_valid | one of {'partial', 'missing', 'stale', 'healthy'} | 'healthy' |

## Raw Response Samples (Failed Endpoints)

### GET /api/fantasy/roster
```json
{
  "detail": "Internal server error",
  "type": "ProgrammingError"
}
```

### GET /api/fantasy/budget
```json
{
  "budget": {
    "acquisitions_used": 0,
    "acquisitions_remaining": 8,
    "acquisition_limit": 8,
    "acquisition_warning": false,
    "il_used": 3,
    "il_total": 3,
    "ip_accumulated": 0.0,
    "ip_minimum": 90.0,
    "ip_pace": "behind",
    "as_of": "2026-04-20T20:41:17.881743-04:00"
  },
  "freshness": {
    "primary_source": "yahoo",
    "fetched_at": "2026-04-20T20:41:16.690935-04:00",
    "computed_at": "2026-04-20T20:41:16.690935-04:00",
    "staleness_threshold_minutes": 60,
    "is_stale": false
  }
}
```

### POST /api/fantasy/roster/optimize
```json
{
  "detail": "Internal server error",
  "type": "ProgrammingError"
}
```

### GET /api/fantasy/waiver
```json
{
  "week_end": "2026-04-26",
  "matchup_opponent": "TBD",
  "category_deficits": [],
  "top_available": [
    {
      "player_id": "469.p.10326",
      "name": "Seth Lugo",
      "team": "KC",
      "position": "SP",
      "need_score": 1.034,
      "category_contributions": {},
      "owned_pct": 0.0,
      "starts_this_week": 0,
      "statcast_signals": [],
      "projected_saves": 0.0,
      "projected_points": 0.0,
      "hot_cold": null,
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "stats": {
        "IP": "24.1",
        "K_P": "1",
        "QS": "1",
        "38": "0",
        "K_B": "21",
        "ERA": "1.48",
        "WHIP": "0.99",
        "K_9": "7.77",
        "NSV": "3",
        "OBP": "0"
      },
      "statcast_stats": null
    },
    {
      "player_id": "469.p.9329",
      "name": "Michael Wacha",
      "team": "KC",
      "position": "SP",
      "need_score": -0.081,
      "category_contributions": {},
      "owned_pct": 0.0,
      "starts_this_week": 0,
      "statcast_signals": [],
      "projected_saves": 0.0,
      "projected_points": 0.0,
      "hot_cold": null,
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "stats": {
        "IP": "27.0",
        "K_P": "2",
        "QS": "0",
        "38": "2",
        "K_B": "23",
        "ERA": "1.00",
        "WHIP": "0.78",
        "K_9": "7.67",
        "NSV": "4",
        "OBP": "0"
      },
      "statcast_stats": null
    },
    {
      "player_id": "469.p.11453",
      "name": "Rico Garcia",
      "team": "BAL",
      "position": "RP",
      "need_score": -0.3,
      "category_contributions": {},
      "owned_pct": 0.0,
      "starts_this_week": 0,
      "statcast_signals": [],
      "projected_saves": 0.0,
      "projected_points": 0.0,
      "hot_cold": null,
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "stats": {
        "IP": "10.0",
        "K_P": "2",
        "QS": "0",
        "38": "0",
        "K_B": "10",
        "ERA": "0.00",
        "WHIP": "0.30",
        "K_9": "9.00",
        "NSV": "0",
        "OBP": "1"
      },
      "statcast_stats": null
    },
    {
      "player_id": "469.p.11489",
      "name": "Aaron Ashby",
      "team": "MIL",
      "position": "RP",
      "need_score": -0.3,
      "category_contributions": {},
      "owned_pct": 0.0,
      "starts_this_week": 0,
      "statcast_signals": [],
      "projected_saves": 0.0,
      "projected_points": 0.0,
      "hot_cold": null,
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "stats": {
        "IP": "14.0",
        "K_P": "5",
        "QS": "0",
        "38": "1",
        "K_B": "22",
        "ERA": "3.21",
        "WHIP": "1.36",
        "K_9": "14.14",
        "NSV": "0",
        "OBP": "0"
      },
      "statcast_stats": null
    },
    {
      "player_id": "469.p.10864",
      "name": "Tony Santillan",
      "team": "CIN",
      "position": "RP",
... (truncated)
```

### GET /api/fantasy/players/1/scores
```json
{
  "detail": "Internal server error",
  "type": "InternalError"
}
```
