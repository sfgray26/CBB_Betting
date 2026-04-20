# UAT Validation Report

**Generated:** 2026-04-20 17:51 ET
**Total Checks:** 70
**PASS:** 55  |  **FAIL:** 13  |  **WARN:** 2  |  **SKIP:** 0

## Summary by Endpoint

| Endpoint | HTTP | Time (ms) | PASS | FAIL | WARN |
|----------|------|-----------|------|------|------|
| GET /health | 200 | 280 | 2 | 0 | 0 |
| GET /api/fantasy/roster | 200 | 11906 | 7 | 3 | 0 |
| GET /api/fantasy/matchup | 200 | 1532 | 8 | 1 | 1 |
| GET /api/fantasy/scoreboard | 404 | 108 | 0 | 1 | 0 |
| GET /api/fantasy/budget | 404 | 108 | 0 | 1 | 0 |
| POST /api/fantasy/roster/optimize | 404 | 109 | 0 | 1 | 0 |
| GET /api/fantasy/waiver | 200 | 4031 | 7 | 2 | 0 |
| GET /api/fantasy/lineup/2026-04-20 | 200 | 7797 | 8 | 0 | 1 |
| GET /api/fantasy/briefing/2026-04-20 | 200 | 3436 | 2 | 3 | 0 |
| GET /api/fantasy/players/1/scores | 200 | 188 | 4 | 0 | 0 |
| GET /api/fantasy/decisions | 200 | 202 | 17 | 0 | 0 |
| GET /api/fantasy/decisions/status | 500 | 125 | 0 | 1 | 0 |

## FAILURES (must fix)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/roster | freshness_present | dict | NoneType |  |
| 2 | GET /api/fantasy/roster | all_have_status | 0 missing | 21 missing |  |
| 3 | GET /api/fantasy/roster | players_with_stats | >50% | 0% (0/23) |  |
| 4 | GET /api/fantasy/matchup | all_stat_values_numeric | all numeric | 2 non-numeric: ["my.NSB='0/0'", "opp.NSB='1/5'"] |  |
| 5 | GET /api/fantasy/scoreboard | http_status | 200 | 404 |  |
| 6 | GET /api/fantasy/budget | http_status | 200 | 404 |  |
| 7 | POST /api/fantasy/roster/optimize | http_status | 200 | 404 |  |
| 8 | GET /api/fantasy/waiver | matchup_opponent | non-TBD name | 'TBD' |  |
| 9 | GET /api/fantasy/waiver | category_deficits_present | >0 categories | 0 |  |
| 10 | GET /api/fantasy/briefing/2026-04-20 | categories_present | >0 | 0 |  |
| 11 | GET /api/fantasy/briefing/2026-04-20 | starters_present | >0 | 0 |  |
| 12 | GET /api/fantasy/briefing/2026-04-20 | overall_confidence | 0-1 float | 95 |  |
| 13 | GET /api/fantasy/decisions/status | http_status | 200 | 500 |  |

## WARNINGS (should investigate)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/matchup | stat_keys_recognized | all recognized | unknown: {'K/BB', 'OBP'} | Non-canonical stat keys in matchup |
| 2 | GET /api/fantasy/lineup/2026-04-20 | lineup_warnings | empty ideally | 30 warnings: ['No games found for this date -- Odds API may not have data yet (requested: 2026-04-20). Lineup ranked by projections only.', 'Moisés Ballesteros: Starting but no game today'] | Lineup produced warnings |

## Full Check Details

### GET /health
HTTP 200 — 280ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | status_field | healthy | healthy |

### GET /api/fantasy/roster
HTTP 200 — 11906ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | team_key_present | non-empty string | '469.l.72586.t.7' |
| PASS | players_count | >0 | 23 |
| PASS | count_matches_players | count=23 | count=23 |
| **FAIL** | freshness_present | dict | NoneType |
| PASS | no_null_player_names | 0 nulls | 0 nulls |
| PASS | no_null_player_keys | 0 nulls | 0 nulls |
| PASS | all_positions_valid | all valid | 0 invalid: [] |
| **FAIL** | all_have_status | 0 missing | 21 missing |
| **FAIL** | players_with_stats | >50% | 0% (0/23) |

### GET /api/fantasy/matchup
HTTP 200 — 1532ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | my_team_present | dict | dict |
| PASS | opponent_present | dict | dict |
| PASS | my_team_has_stats | >0 stats | 14 stats |
| PASS | opponent_has_stats | >0 stats | 16 stats |
| WARN | stat_keys_recognized | all recognized | unknown: {'K/BB', 'OBP'} |
| **FAIL** | all_stat_values_numeric | all numeric | 2 non-numeric: ["my.NSB='0/0'", "opp.NSB='1/5'"] |
| PASS | week_present | int | 5 |
| PASS | my_team_name | non-empty | 'Lindor Truffles' |
| PASS | opponent_team_name | non-empty | "Bartolo's Colon" |

### GET /api/fantasy/scoreboard
HTTP 404 — 108ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 404 |

### GET /api/fantasy/budget
HTTP 404 — 108ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 404 |

### POST /api/fantasy/roster/optimize
HTTP 404 — 109ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 404 |

### GET /api/fantasy/waiver
HTTP 200 — 4031ms

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
HTTP 200 — 7797ms

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
HTTP 200 — 3436ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | strategy_present | non-empty | 'standard' |
| **FAIL** | categories_present | >0 | 0 |
| **FAIL** | starters_present | >0 | 0 |
| **FAIL** | overall_confidence | 0-1 float | 95 |

### GET /api/fantasy/players/1/scores
HTTP 200 — 188ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | score_0_100_range | 0-100 | 83.8 |
| PASS | composite_z_present | numeric | 0.6561257422379267 |
| PASS | category_scores_present | dict | dict |

### GET /api/fantasy/decisions
HTTP 200 — 202ms

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
HTTP 500 — 125ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| **FAIL** | http_status | 200 | 500 |

## Raw Response Samples (Failed Endpoints)

### GET /api/fantasy/roster
```json
{
  "team_key": "469.l.72586.t.7",
  "players": [
    {
      "player_key": "469.p.60120",
      "name": "Mois\u00e9s Ballesteros",
      "team": "CHC",
      "positions": [
        "C",
        "Util"
      ],
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "z_score": -1.5,
      "is_undroppable": false,
      "is_proxy": true,
      "cat_scores": {},
      "selected_position": "C"
    },
    {
      "player_key": "469.p.10918",
      "name": "Pete Alonso",
      "team": "BAL",
      "positions": [
        "1B",
        "Util"
      ],
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "z_score": 6.023,
      "is_undroppable": false,
      "is_proxy": false,
      "cat_scores": {
        "r": 1.002,
        "h": 0.597,
        "hr": 2.33,
        "rbi": 1.553,
        "k_bat": -0.569,
        "tb": 1.297,
        "avg": -0.159,
        "ops": 1.045,
        "nsb": -1.073
      },
      "selected_position": "1B"
    },
    {
      "player_key": "469.p.64858",
      "name": "Hyeseong Kim",
      "team": "LAD",
      "positions": [
        "2B",
        "SS",
        "CF",
        "Util"
      ],
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "z_score": -0.8,
      "is_undroppable": false,
      "is_proxy": true,
      "cat_scores": {},
      "selected_position": "2B"
    },
    {
      "player_key": "469.p.66369",
      "name": "Munetaka Murakami",
      "team": "CWS",
      "positions": [
        "1B",
        "3B",
        "Util"
      ],
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "z_score": -0.8,
      "is_undroppable": false,
      "is_proxy": true,
      "cat_scores": {},
      "selected_position": "3B"
    },
    {
      "player_key": "469.p.11417",
      "name": "Geraldo Perdomo",
      "team": "AZ",
      "positions": [
        "SS",
        "Util"
      ],
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "z_score": 2.37,
      "is_undroppable": false,
      "is_proxy": false,
      "cat_scores": {
        "r": 0.542,
        "h": 0.38,
        "hr": -0.958,
        "rbi": -0.592,
        "k_bat": 1.086,
        "tb": 0.041,
        "avg": 0.311,
        "ops": 0.006,
        "nsb": 1.553
      },
      "selected_position": "SS"
    },
    {
      "player_key": "469.p.12024",
      "name": "Jordan Walker",
      "team": "STL",
      "positions": [
        "LF",
        "RF",
        "Util"
      ],
      "status": null,
      "injury_note": null,
      "injury_status": null,
      "z_score": 0.643,
      "is_undroppable": false,
      "is_proxy": false,
      "cat_scores": {
        "r": 0.128,
        "h": 0.299,
        "hr": 0.412,
        "rbi": 0.508,
        "k_bat": -0.857,
        "tb": 0.347,
        "avg": 0.203,
        "ops": 0.02,
        "nsb": -0.417
      },
      "selected_position": "LF"
    },
    {
      "player_key": "469.p.12157",
      "n
... (truncated)
```

### GET /api/fantasy/matchup
```json
{
  "week": 5,
  "my_team": {
    "team_key": "469.l.72586.t.7",
    "team_name": "Lindor Truffles",
    "stats": {
      "NSB": "0/0",
      "R": 0,
      "H": 0,
      "HR": 0,
      "RBI": 0,
      "IP": "0.0",
      "W": 0,
      "GS": 0,
      "K": 0,
      "QS": 0,
      "K/BB": 0,
      "K(B)": 0,
      "NSV": 0,
      "OBP": 0
    }
  },
  "opponent": {
    "team_key": "469.l.72586.t.2",
    "team_name": "Bartolo's Colon",
    "stats": {
      "NSB": "1/5",
      "R": 0,
      "H": 1,
      "HR": 0,
      "RBI": 0,
      "IP": "0.0",
      "W": 1,
      "AVG": ".200",
      "OPS": ".400",
      "GS": 0,
      "K": 0,
      "QS": 0,
      "K/BB": 0,
      "K(B)": 0,
      "NSV": 0,
      "OBP": 0
    }
  },
  "is_playoffs": false,
  "message": null
}
```

### GET /api/fantasy/scoreboard
```json
{
  "detail": "Not Found"
}
```

### GET /api/fantasy/budget
```json
{
  "detail": "Not Found"
}
```

### POST /api/fantasy/roster/optimize
```json
{
  "detail": "Not Found"
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
        "K": "1",
        "QS": "1",
        "K/BB": "0",
        "K(B)": "21",
        "ERA": "1.48",
        "WHIP": "0.99",
        "K/9": "7.77",
        "NSV": "3",
        "OBP": "0"
      }
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
        "K": "2",
        "QS": "0",
        "K/BB": "2",
        "K(B)": "23",
        "ERA": "1.00",
        "WHIP": "0.78",
        "K/9": "7.67",
        "NSV": "4",
        "OBP": "0"
      }
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
        "K": "0",
        "QS": "0",
        "K/BB": "0",
        "K(B)": "10",
        "ERA": "0.00",
        "WHIP": "0.58",
        "K/9": "10.38",
        "NSV": "0",
        "OBP": "2"
      }
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
        "K": "2",
        "QS": "0",
        "K/BB": "0",
        "K(B)": "10",
        "ERA": "0.00",
        "WHIP": "0.30",
        "K/9": "9.00",
        "NSV": "0",
        "OBP": "1"
      }
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
   
... (truncated)
```

### GET /api/fantasy/briefing/2026-04-20
```json
{
  "date": "2026-04-20",
  "generated_at": "2026-04-20T21:51:28.791337",
  "strategy": "standard",
  "risk_profile": "BALANCED",
  "overall_confidence": 95,
  "summary": {
    "total_decisions": 23,
    "easy_decisions": 14,
    "tough_decisions": 0,
    "monitor_count": 0
  },
  "categories": [],
  "starters": [],
  "bench": [
    {
      "emoji": "\ud83e\ude91",
      "name": "Mois\u00e9s Ballesteros",
      "team": "CHC",
      "vs": "TBD",
      "recommendation": "BENCH",
      "confidence": 95,
      "rating": "NEUTRAL",
      "factors": [
        "No game today"
      ]
    },
    {
      "emoji": "\ud83e\ude91",
      "name": "Pete Alonso",
      "team": "BAL",
      "vs": "TBD",
      "recommendation": "BENCH",
      "confidence": 95,
      "rating": "NEUTRAL",
      "factors": [
        "No game today"
      ]
    },
    {
      "emoji": "\ud83e\ude91",
      "name": "Hyeseong Kim",
      "team": "LAD",
      "vs": "TBD",
      "recommendation": "BENCH",
      "confidence": 95,
      "rating": "NEUTRAL",
      "factors": [
        "No game today"
      ]
    },
    {
      "emoji": "\ud83e\ude91",
      "name": "Munetaka Murakami",
      "team": "CWS",
      "vs": "TBD",
      "recommendation": "BENCH",
      "confidence": 95,
      "rating": "NEUTRAL",
      "factors": [
        "No game today"
      ]
    },
    {
      "emoji": "\ud83e\ude91",
      "name": "Geraldo Perdomo",
      "team": "ARI",
      "vs": "TBD",
      "recommendation": "BENCH",
      "confidence": 95,
      "rating": "NEUTRAL",
      "factors": [
        "No game today"
      ]
    }
  ],
  "monitor": [],
  "alerts": [
    "Mois\u00e9s Ballesteros: Starting but no game today",
    "Pete Alonso: Starting but no game today",
    "Hyeseong Kim: Starting but no game today"
  ],
  "_meta": {
    "decisions_recorded": true,
    "decisions_count": 23
  }
}
```

### GET /api/fantasy/decisions/status
```json
{
  "detail": "Internal server error",
  "type": "AttributeError"
}
```
