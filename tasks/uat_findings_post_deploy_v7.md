# UAT Validation Report

**Generated:** 2026-04-21 21:12 ET
**Total Checks:** 91
**PASS:** 89  |  **FAIL:** 0  |  **WARN:** 2  |  **SKIP:** 0

## Summary by Endpoint

| Endpoint | HTTP | Time (ms) | PASS | FAIL | WARN |
|----------|------|-----------|------|------|------|
| GET /health | 200 | 297 | 2 | 0 | 0 |
| GET /api/fantasy/roster | 200 | 1890 | 10 | 0 | 0 |
| GET /api/fantasy/matchup | 200 | 1641 | 10 | 0 | 0 |
| GET /api/fantasy/scoreboard | 200 | 1218 | 7 | 0 | 0 |
| GET /api/fantasy/budget | 200 | 1265 | 13 | 0 | 0 |
| POST /api/fantasy/roster/optimize | 200 | 608 | 11 | 0 | 0 |
| GET /api/fantasy/waiver | 503 | 2391 | 0 | 0 | 1 |
| GET /api/fantasy/lineup/2026-04-21 | 200 | 4250 | 8 | 0 | 1 |
| GET /api/fantasy/briefing/2026-04-21 | 200 | 3952 | 5 | 0 | 0 |
| GET /api/fantasy/players/1/scores | 200 | 202 | 4 | 0 | 0 |
| GET /api/fantasy/decisions | 200 | 187 | 17 | 0 | 0 |
| GET /api/fantasy/decisions/status | 200 | 202 | 2 | 0 | 0 |

## WARNINGS (should investigate)

| # | Endpoint | Check | Expected | Actual | Detail |
|---|----------|-------|----------|--------|--------|
| 1 | GET /api/fantasy/waiver | http_status | 200 | 503 | Yahoo auth issue |
| 2 | GET /api/fantasy/lineup/2026-04-21 | lineup_warnings | empty ideally | 12 warnings: ['Hyeseong Kim: No game scheduled — monitor before lock', 'Munetaka Murakami: No game scheduled — monitor before lock'] | Lineup produced warnings |

## Full Check Details

### GET /health
HTTP 200 — 297ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | status_field | healthy | healthy |

### GET /api/fantasy/roster
HTTP 200 — 1890ms

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
| PASS | players_with_stats | >50% | 100% |

### GET /api/fantasy/matchup
HTTP 200 — 1641ms

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
HTTP 200 — 1218ms

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
HTTP 200 — 1265ms

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
| PASS | ip_pace_valid | one of {'COMPLETE', 'BEHIND', 'ON_TRACK'} | 'BEHIND' |
| PASS | freshness_present | dict | dict |

### POST /api/fantasy/roster/optimize
HTTP 200 — 608ms

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
HTTP 503 — 2391ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| WARN | http_status | 200 | 503 |

### GET /api/fantasy/lineup/2026-04-21
HTTP 200 — 4250ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | batters_count | >0 | 13 |
| PASS | batters_have_names | 0 nameless | 0 nameless |
| PASS | batters_have_teams | 0 teamless | 0 teamless |
| PASS | batter_scores_numeric | all numeric | 0 non-numeric |
| PASS | active_batters_have_slots | 0 without slot | 0 active without slot |
| PASS | pitchers_present | list | list |
| PASS | games_count_present | int >= 0 | 9 |
| WARN | lineup_warnings | empty ideally | 12 warnings: ['Hyeseong Kim: No game scheduled — monitor before lock', 'Munetaka Murakami: No game scheduled — monitor before lock'] |

### GET /api/fantasy/briefing/2026-04-21
HTTP 200 — 3952ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | strategy_present | non-empty | 'standard' |
| PASS | categories_present | >0 | 11 |
| PASS | starters_present | >0 | 9 |
| PASS | overall_confidence | 0-1 float | 0.25 |

### GET /api/fantasy/players/1/scores
HTTP 200 — 202ms

| Status | Check | Expected | Actual |
|--------|-------|----------|--------|
| PASS | http_status | 200 | 200 |
| PASS | score_0_100_range | 0-100 | 58.5 |
| PASS | composite_z_present | numeric | 0.5032497088603167 |
| PASS | category_scores_present | dict | dict |

### GET /api/fantasy/decisions
HTTP 200 — 187ms

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
| PASS | verdict_valid | one of {'healthy', 'missing', 'partial', 'stale'} | 'healthy' |
