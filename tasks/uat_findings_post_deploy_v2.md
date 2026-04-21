# UAT Validation Report

**Generated:** 2026-04-20 18:39 ET
**Total Checks:** 80
**PASS:** 69  |  **FAIL:** 8  |  **WARN:** 3  |  **SKIP:** 0

## Summary by Endpoint

| Endpoint | HTTP | Time (ms) | PASS | FAIL | WARN |
|----------|------|-----------|------|------|------|
| GET /health | 200 | 281 | 2 | 0 | 0 |
| GET /api/fantasy/roster | 500 | 422 | 0 | 1 | 0 |
| GET /api/fantasy/matchup | 200 | 811 | 9 | 1 | 0 |
| GET /api/fantasy/scoreboard | 500 | 438 | 0 | 1 | 0 |
| GET /api/fantasy/budget | 200 | 530 | 12 | 1 | 0 |
| POST /api/fantasy/roster/optimize | 200 | 281 | 9 | 0 | 2 |
| GET /api/fantasy/waiver | 200 | 1561 | 7 | 2 | 0 |
| GET /api/fantasy/lineup/2026-04-20 | 200 | 1282 | 8 | 0 | 1 |
| GET /api/fantasy/briefing/2026-04-20 | 200 | 1140 | 5 | 0 | 0 |
| GET /api/fantasy/players/1/scores | 500 | 109 | 0 | 1 | 0 |
| GET /api/fantasy/decisions | 200 | 217 | 17 | 0 | 0 |
| GET /api/fantasy/decisions/status | 500 | 110 | 0 | 1 | 0 |

## FAILURES (must fix)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/roster | http_status | 200 | 500 |  |
| 2 | GET /api/fantasy/matchup | all_stat_values_numeric | all numeric | 5 non-numeric: ['my.AVG=None', 'my.OPS=None', 'opp.ERA=None', 'opp.WHIP=None', 'opp.K_9=None'] |  |
| 3 | GET /api/fantasy/scoreboard | http_status | 200 | 500 |  |
| 4 | GET /api/fantasy/budget | ip_pace_valid | one of {'BEHIND', 'COMPLETE', 'ON_TRACK'} | 'behind' |  |
| 5 | GET /api/fantasy/waiver | matchup_opponent | non-TBD name | 'TBD' |  |
| 6 | GET /api/fantasy/waiver | category_deficits_present | >0 categories | 0 |  |
| 7 | GET /api/fantasy/players/1/scores | http_status | 200 or 404 | 500 |  |
| 8 | GET /api/fantasy/decisions/status | http_status | 200 | 500 |  |

## WARNINGS (should investigate)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | POST /api/fantasy/roster/optimize | scores_not_all_default | varied scores | all=50.0 | All players have same score — player_scores may be stale |
| 2 | POST /api/fantasy/roster/optimize | score_source_quality | <50% stale/default | 14/14 stale | Most players using fallback scores |
| 3 | GET /api/fantasy/lineup/2026-04-20 | lineup_warnings | empty ideally | 30 warnings: ['No games found for this date -- Odds API may not have data yet (requested: 2026-04-20). Lineup ranked by projections only.', 'Moisés Ballesteros: Starting but no game today'] | Lineup produced warnings |

## Full Check Details

### GET /health
HTTP 200 — 281ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | status_field | healthy | healthy |

### GET /api/fantasy/roster
HTTP 500 — 422ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 500 |

### GET /api/fantasy/matchup
HTTP 200 — 811ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | my_team_present | dict | dict |
| PASS | opponent_present | dict | dict |
| PASS | my_team_has_stats | >0 stats | 15 stats |
| PASS | opponent_has_stats | >0 stats | 15 stats |
| PASS | stat_keys_recognized | all recognized | all recognized |
| **FAIL** | all_stat_values_numeric | all numeric | 5 non-numeric: ['my.AVG=None', 'my.OPS=None', 'opp.ERA=None', 'opp.WHIP=None', 'opp.K_9=None'] |
| PASS | week_present | int | 5 |
| PASS | my_team_name | non-empty | 'Lindor Truffles' |
| PASS | opponent_team_name | non-empty | "Bartolo's Colon" |

### GET /api/fantasy/scoreboard
HTTP 500 — 438ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 500 |

### GET /api/fantasy/budget
HTTP 200 — 530ms

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
HTTP 200 — 281ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | success | true | True |
| PASS | starters_count | >0 | 14 |
| PASS | all_slots_valid | all valid | invalid: [] |
| PASS | no_slot_overflow | all within capacity | overflows: [] |
| PASS | all_starters_have_scores | all scored | 0 without score: [] |
| WARN | scores_not_all_default | varied scores | all=50.0 |
| WARN | score_source_quality | <50% stale/default | 14/14 stale |
| PASS | bench_present | list | list |
| PASS | total_lineup_score | numeric > 0 | 700.0 |
| PASS | freshness_present | dict | dict |

### GET /api/fantasy/waiver
HTTP 200 — 1561ms

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
HTTP 200 — 1282ms

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
HTTP 200 — 1140ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | strategy_present | non-empty | 'standard' |
| PASS | categories_present | >0 | 11 |
| PASS | starters_present | >0 | 9 |
| PASS | overall_confidence | 0-1 float | 0.7 |

### GET /api/fantasy/players/1/scores
HTTP 500 — 109ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 or 404 | 500 |

### GET /api/fantasy/decisions
HTTP 200 — 217ms

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
HTTP 500 — 110ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 500 |

## Raw Response Samples (Failed Endpoints)

### GET /api/fantasy/roster
```json
{
  "detail": "Internal server error",
  "type": "ImportError"
}
```

### GET /api/fantasy/matchup
```json
{
  "week": 5,
  "my_team": {
    "team_key": "469.l.72586.t.7",
    "team_name": "Lindor Truffles",
    "stats": {
      "NSB": "0",
      "R": 0,
      "H": 0,
      "HR_B": 0,
      "RBI": 0,
      "W": 0,
      "AVG": null,
      "OPS": null,
      "K_P": 0,
      "QS": 0,
      "K_B": 0,
      "ERA": "9.00",
      "WHIP": "3.00",
      "K_9": "0.00",
      "NSV": 0
    }
  },
  "opponent": {
    "team_key": "469.l.72586.t.2",
    "team_name": "Bartolo's Colon",
    "stats": {
      "NSB": "1",
      "R": 0,
      "H": 1,
      "HR_B": 0,
      "RBI": 0,
      "W": 1,
      "AVG": ".200",
      "OPS": ".400",
      "K_P": 0,
      "QS": 0,
      "K_B": 0,
      "ERA": null,
      "WHIP": null,
      "K_9": null,
      "NSV": 0
    }
  },
  "is_playoffs": false,
  "message": null
}
```

### GET /api/fantasy/scoreboard
```json
{
  "detail": "Internal server error",
  "type": "ValueError"
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
    "as_of": "2026-04-20T18:39:50.627531-04:00"
  },
  "freshness": {
    "primary_source": "yahoo",
    "fetched_at": "2026-04-20T18:39:50.265818-04:00",
    "computed_at": "2026-04-20T18:39:50.265818-04:00",
    "staleness_threshold_minutes": 60,
    "is_stale": false
  }
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
      "player_id": "469.p.11920",
      "name": "Alex Vesia",
      "team": "LAD",
      "position": "RP",
      "need_score": -0.272,
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
        "IP": "8.2",
        "K_P": "0",
        "QS": "0",
        "38": "0",
        "K_B": "10",
        "ERA": "0.00",
        "WHIP": "0.58",
        "K_9": "10.38",
        "NSV": "0",
        "OBP": "2"
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
  
... (truncated)
```

### GET /api/fantasy/players/1/scores
```json
{
  "detail": "Internal server error",
  "type": "ProgrammingError"
}
```

### GET /api/fantasy/decisions/status
```json
{
  "detail": "Internal server error",
  "type": "TypeError"
}
```
