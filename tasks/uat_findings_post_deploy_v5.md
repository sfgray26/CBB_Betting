# UAT Validation Report

**Generated:** 2026-04-21 18:21 ET
**Total Checks:** 99
**PASS:** 95  |  **FAIL:** 3  |  **WARN:** 1  |  **SKIP:** 0

## Summary by Endpoint

| Endpoint | HTTP | Time (ms) | PASS | FAIL | WARN |
|----------|------|-----------|------|------|------|
| GET /health | 200 | 312 | 2 | 0 | 0 |
| GET /api/fantasy/roster | 200 | 2217 | 9 | 1 | 0 |
| GET /api/fantasy/matchup | 200 | 1469 | 10 | 0 | 0 |
| GET /api/fantasy/scoreboard | 200 | 1094 | 7 | 0 | 0 |
| GET /api/fantasy/budget | 200 | 1233 | 13 | 0 | 0 |
| POST /api/fantasy/roster/optimize | 200 | 14030 | 11 | 0 | 0 |
| GET /api/fantasy/waiver | 200 | 19265 | 7 | 2 | 0 |
| GET /api/fantasy/lineup/2026-04-21 | 200 | 9515 | 8 | 0 | 1 |
| GET /api/fantasy/briefing/2026-04-21 | 200 | 3782 | 5 | 0 | 0 |
| GET /api/fantasy/players/1/scores | 200 | 187 | 4 | 0 | 0 |
| GET /api/fantasy/decisions | 200 | 219 | 17 | 0 | 0 |
| GET /api/fantasy/decisions/status | 200 | 202 | 2 | 0 | 0 |

## FAILURES (must fix)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/roster | players_with_stats | >50% | 0% (0/23) |  |
| 2 | GET /api/fantasy/waiver | matchup_opponent | non-TBD name | 'TBD' |  |
| 3 | GET /api/fantasy/waiver | category_deficits_present | >0 categories | 0 |  |

## WARNINGS (should investigate)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/lineup/2026-04-21 | lineup_warnings | empty ideally | 2 warnings: ['7 SP(s) have no start today: Eury Pérez, Spencer Arrighetti, Gavin Williams', 'Only 0 active pitcher slots filled -- consider streaming a SP.'] | Lineup produced warnings |

## Full Check Details

### GET /health
HTTP 200 — 312ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | status_field | healthy | healthy |

### GET /api/fantasy/roster
HTTP 200 — 2217ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | team_key_present | non-empty string | '469.l.72586.t.7' |
| PASS | players_count | >0 | 23 |
| PASS | count_matches_players | count=23 | count=23 |
| PASS | freshness_present | dict | dict |
| PASS | no_null_player_names | 0 nulls | 0 nulls |
| PASS | no_null_player_keys | 0 nulls | 0 nulls |
| PASS | all_positions_valid | all valid | 0 invalid: [] |
| PASS | all_have_status | 0 missing | 0 missing |
| **FAIL** | players_with_stats | >50% | 0% (0/23) |

### GET /api/fantasy/matchup
HTTP 200 — 1469ms

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
HTTP 200 — 1094ms

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
HTTP 200 — 1233ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | budget.acquisitions_used_present | present | 0 |
| PASS | budget.acquisitions_remaining_present | present | 8 |
| PASS | budget.acquisition_limit_present | present | 8 |
| PASS | budget.il_used_present | present | 2 |
| PASS | budget.il_total_present | present | 3 |
| PASS | budget.ip_accumulated_present | present | 0.0 |
| PASS | budget.ip_minimum_present | present | 90.0 |
| PASS | budget.ip_pace_present | present | 'BEHIND' |
| PASS | acquisitions_used_reasonable | 0 <= used <= 8 | 0 |
| PASS | il_used_reasonable | 0 <= 2 <= 3 | il_used=2, il_total=3 |
| PASS | ip_pace_valid | one of {'ON_TRACK', 'COMPLETE', 'BEHIND'} | 'BEHIND' |
| PASS | freshness_present | dict | dict |

### POST /api/fantasy/roster/optimize
HTTP 200 — 14030ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | success | true | True |
| PASS | starters_count | >0 | 13 |
| PASS | all_slots_valid | all valid | invalid: [] |
| PASS | no_slot_overflow | all within capacity | overflows: [] |
| PASS | all_starters_have_scores | all scored | 0 without score: [] |
| PASS | scores_not_all_default | varied scores | 13 distinct values |
| PASS | score_source_quality | <50% stale/default | 0/13 stale |
| PASS | bench_present | list | list |
| PASS | total_lineup_score | numeric > 0 | 1032.7 |
| PASS | freshness_present | dict | dict |

### GET /api/fantasy/waiver
HTTP 200 — 19265ms

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

### GET /api/fantasy/lineup/2026-04-21
HTTP 200 — 9515ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | batters_count | >0 | 13 |
| PASS | batters_have_names | 0 nameless | 0 nameless |
| PASS | batters_have_teams | 0 teamless | 0 teamless |
| PASS | batter_scores_numeric | all numeric | 0 non-numeric |
| PASS | active_batters_have_slots | 0 without slot | 0 active without slot |
| PASS | pitchers_present | list | list |
| PASS | games_count_present | int >= 0 | 15 |
| WARN | lineup_warnings | empty ideally | 2 warnings: ['7 SP(s) have no start today: Eury Pérez, Spencer Arrighetti, Gavin Williams', 'Only 0 active pitcher slots filled -- consider streaming a SP.'] |

### GET /api/fantasy/briefing/2026-04-21
HTTP 200 — 3782ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | strategy_present | non-empty | 'standard' |
| PASS | categories_present | >0 | 11 |
| PASS | starters_present | >0 | 9 |
| PASS | overall_confidence | 0-1 float | 0.35 |

### GET /api/fantasy/players/1/scores
HTTP 200 — 187ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | score_0_100_range | 0-100 | 58.5 |
| PASS | composite_z_present | numeric | 0.5032497088603167 |
| PASS | category_scores_present | dict | dict |

### GET /api/fantasy/decisions
HTTP 200 — 219ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | count_matches | count=10 | count=10 |
| PASS | decision[0].bdl_player_id | present | 879 |
| PASS | decision[0].decision_type | present | 'lineup' |
| PASS | decision[0].confidence | present | 0.98 |
| PASS | decision[1].bdl_player_id | present | 539 |
| PASS | decision[1].decision_type | present | 'lineup' |
| PASS | decision[1].confidence | present | 0.968 |
| PASS | decision[2].bdl_player_id | present | 770 |
| PASS | decision[2].decision_type | present | 'waiver' |
| PASS | decision[2].confidence | present | 0.9643 |
| PASS | decision[3].bdl_player_id | present | 4624 |
| PASS | decision[3].decision_type | present | 'lineup' |
| PASS | decision[3].confidence | present | 0.927 |
| PASS | decision[4].bdl_player_id | present | 356 |
| PASS | decision[4].decision_type | present | 'lineup' |
| PASS | decision[4].confidence | present | 0.92 |

### GET /api/fantasy/decisions/status
HTTP 200 — 202ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | verdict_valid | one of {'stale', 'missing', 'healthy', 'partial'} | 'healthy' |

## Raw Response Samples (Failed Endpoints)

### GET /api/fantasy/roster
```json
{
  "team_key": "469.l.72586.t.7",
  "players": [
    {
      "player_name": "Mois\u00e9s Ballesteros",
      "team": "CHC",
      "eligible_positions": [
        "C",
        "Util"
      ],
      "status": "playing",
      "game_context": null,
      "season_stats": null,
      "rolling_7d": null,
      "rolling_14d": null,
      "rolling_15d": null,
      "rolling_30d": null,
      "ros_projection": null,
      "row_projection": null,
      "ownership_pct": 0.0,
      "injury_status": null,
      "injury_return_timeline": null,
      "freshness": {
        "primary_source": "yahoo",
        "fetched_at": null,
        "computed_at": "2026-04-21T18:20:16.380652-04:00",
        "staleness_threshold_minutes": 60,
        "is_stale": false
      },
      "yahoo_player_key": "469.p.60120",
      "bdl_player_id": null,
      "mlbam_id": null
    },
    {
      "player_name": "Pete Alonso",
      "team": "BAL",
      "eligible_positions": [
        "1B",
        "Util"
      ],
      "status": "playing",
      "game_context": null,
      "season_stats": null,
      "rolling_7d": null,
      "rolling_14d": null,
      "rolling_15d": null,
      "rolling_30d": null,
      "ros_projection": null,
      "row_projection": null,
      "ownership_pct": 0.0,
      "injury_status": null,
      "injury_return_timeline": null,
      "freshness": {
        "primary_source": "yahoo",
        "fetched_at": null,
        "computed_at": "2026-04-21T18:20:16.380652-04:00",
        "staleness_threshold_minutes": 60,
        "is_stale": false
      },
      "yahoo_player_key": "469.p.10918",
      "bdl_player_id": null,
      "mlbam_id": null
    },
    {
      "player_name": "Hyeseong Kim",
      "team": "LAD",
      "eligible_positions": [
        "2B",
        "SS",
        "CF",
        "Util"
      ],
      "status": "playing",
      "game_context": null,
      "season_stats": null,
      "rolling_7d": null,
      "rolling_14d": null,
      "rolling_15d": null,
      "rolling_30d": null,
      "ros_projection": null,
      "row_projection": null,
      "ownership_pct": 0.0,
      "injury_status": null,
      "injury_return_timeline": null,
      "freshness": {
        "primary_source": "yahoo",
        "fetched_at": null,
        "computed_at": "2026-04-21T18:20:16.380652-04:00",
        "staleness_threshold_minutes": 60,
        "is_stale": false
      },
      "yahoo_player_key": "469.p.64858",
      "bdl_player_id": null,
      "mlbam_id": null
    },
    {
      "player_name": "Munetaka Murakami",
      "team": "CWS",
      "eligible_positions": [
        "1B",
        "3B",
        "Util"
      ],
      "status": "playing",
      "game_context": null,
      "season_stats": null,
      "rolling_7d": null,
      "rolling_14d": null,
      "rolling_15d": null,
      "rolling_30d": null,
      "ros_projection": null,
      "row_projection": null,
      "ownership_pct": 0.0,
      "injury_status": null,
      "injury_return_timeline": null,
      "freshness": {
  
... (truncated)
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
        "IP": "31.1",
        "K_P": "1",
        "QS": "1",
        "K_B": "28",
        "ERA": "1.15",
        "WHIP": "0.93",
        "K_9": "8.04",
        "NSV": "4",
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
        "IP": "11.0",
        "K_P": "2",
        "QS": "0",
        "K_B": "11",
        "ERA": "0.00",
        "WHIP": "0.36",
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
      "player_id": "469.p.10985",
      "name": "Tyler Kinley",
      "team": "ATL",
      "position": "RP",
      "need_score": -0.3,
      "category_contributions": {},
      "owned_pc
... (truncated)
```
