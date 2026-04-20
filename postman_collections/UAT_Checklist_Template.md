# UAT Checklist - CBB Edge MLB Platform

> Use this checklist while testing the Postman collection.
> Document findings in the Notes column.

**Environment:** {{baseUrl}} (Production/Railway)
**Date:** __________________
**Tester:** __________________

## Setup

- [ ] Postman collection imported
- [ ] Environment configured with baseUrl
- [ ] API key set in environment variables
- [ ] Verified Railway app is running

---

## 0. Health Check

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/` | Returns API name/version | | |
| GET `/health` | Status: healthy | | |

---

## 1. Fantasy Baseball - Draft Board

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/draft-board` | Returns 200+ players | | |
| Players have `player_id` | Yes | | |
| Players have `name` | Yes | | |
| Players have `position` | Yes | | |
| Players have `tier` | Yes | | |
| Players have projection stats | Yes (R, HR, RBI, SB, etc.) | | |
| `position=OF` filter works | Only OF returned | | |
| `player_type=batter` filter works | Only batters returned | | |
| `tier_max=2` filter works | Only T1-T2 returned | | |

**Findings:**


---

## 2. Fantasy Baseball - Roster Management

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/roster` | Returns roster with slots | | |
| Has `slots` array | Yes (all 16 slots) | | |
| Has `players` array | Yes | | |
| Has `metadata` | Yes | | |
| `slots` have `player_id` or null | Yes | | |
| `slots` have `eligible_positions` | Yes | | |
| Player has `projected_rest_of_season` | Yes | | |
| POST `/api/fantasy/roster/move` | Returns validation result | | |
| Move validation catches invalid slot | Yes | | |
| POST `/api/fantasy/roster/optimize` | Returns recommendations | | |
| Returns `recommended_moves` | Yes | | |
| Each move has `type`, `player_id`, `rationale` | Yes | | |

**Findings:**


---

## 3. Fantasy Baseball - Lineup Management

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/lineup/{date}` | Returns lineup | | |
| Has `date` | Yes | | |
| Has `slots` array (16 slots) | Yes | | |
| Each slot has `slot_type` | Yes | | |
| Each slot has `player` or null | Yes | | |
| Has `metadata` with `game_count` | Yes | | |
| Has `recommended` boolean | Yes | | |
| GET `/api/fantasy/saved-lineup/{date}` | Returns saved lineup | | |
| POST `/api/fantasy/lineup` | Saves lineup | | |
| PUT `/api/fantasy/lineup/apply` | Preview or applies | | |
| `confirm=false` dry-run works | Yes | | |

**Findings:**


---

## 4. Fantasy Baseball - Waiver Wire

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/waiver` | Returns waiver targets | | |
| Has `players` array | Yes | | |
| Player has `marginal_value` | Yes | | |
| Player has `roster_percent` | Yes | | |
| Player has `add_priority` | Yes | | |
| `position=SP` filter works | Only pitchers returned | | |
| GET `/api/fantasy/waiver/recommendations` | Returns AI recommendations | | |
| Has `rationale` text | Yes | | |
| POST `/api/fantasy/waiver/add` | Returns simulation result | | |
| Shows projected impact | Yes | | |

**Findings:**


---

## 5. Fantasy Baseball - Matchup

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/matchup` | Returns matchup data | | |
| Has `my_team` object | Yes | | |
| Has `opponent_team` object | Yes | | |
| Has `categories` array (18 cats) | Yes | | |
| Each category has `code` | Yes | | |
| Each category has `my_score` | Yes | | |
| Each category has `opponent_score` | Yes | | |
| Each category has `winner` (my/opponent/tie) | Yes | | |
| Each category has `projection` | Yes | | |
| Has `win_probability` | Yes | | |
| POST `/api/fantasy/matchup/simulate` | Returns MC simulation | | |
| Returns `my_win_probability` | Yes | | |
| Has `category_win_probs` | Yes | | |

**Findings:**


---

## 6. Scoreboard

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/scoreboard` | Returns scoreboard | | |
| Has `my_team` with total_score | Yes | | |
| Has `opponent` with total_score | Yes | | |
| Has `categories` array (18) | Yes | | |
| Categories are correct 18-cat set | R/HR/RBI/SB/BB/K(Net)/TB/NSB/OPS/W/L/SV/NSV/HLD/K(QS)/ERA/WHIP | | |
| Each cat has `winner` field | Yes | | |
| Each cat has `projection` field | Yes | | |
| Has `metadata` with `week` | Yes | | |
| Has `metadata` with `scoring_period_left` | Yes | | |

**Findings:**


---

## 7. Budget

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/budget` | Returns budget info | | |
| Has `remaining_budget` | Yes | | |
| Has `starting_budget` | Yes | | |
| Has `acquisitions_this_week` | Yes | | |
| Has `max_acquisitions` | Yes | | |
| Has `week_number` | Yes | | |
| Values are reasonable | budget <= $100, acquisitions <= 10 | | |

**Findings:**


---

## 8. Daily Briefing

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/briefing/{date}` | Returns briefing | | |
| Has `probable_pitchers` array | Yes | | |
| Each pitcher has `team`, `opponent` | Yes | | |
| Each pitcher has `hand`, `proj_ip` | Yes | | |
| Has `injuries` array | Yes | | |
| Has `weather_forecasts` | Yes | | |
| Has `recommendations` | Yes | | |

**Findings:**


---

## 9. Player Scores

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/players/{id}/scores` | Returns player scores | | |
| Has `player_id`, `name` | Yes | | |
| Has `category_breakdown` array | Yes | | |
| Covers all 18 categories | Yes | | |
| Has `z_score` for each category | Yes | | |
| `period=season` works | Yes | | |
| `period=week_7` works | Yes | | |

**Findings:**


---

## 10. Decisions

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/decisions` | Returns decisions | | |
| Has `decisions` array | Yes | | |
| Each decision has `decision_type` | lineup/waiver | | |
| Each decision has `timestamp` | Yes | | |
| Each decision has `explanation` | Yes | | |
| GET `/api/fantasy/decisions/status` | Returns pipeline status | | |
| Has `last_updated` | Yes | | |
| Has `projection_freshness` | Yes | | |

**Findings:**


---

## 11. Admin - Ingestion Status

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/admin/ingestion/status` | Returns status | | |
| Has `projection_freshness` | Yes | | |
| Has `violations` array or empty | Yes | | |
| Shows all job statuses | Yes | | |
| No freshness violations | Yes (if healthy) | | |

**Findings:**


---

## 12. Admin - Yahoo Diagnostics

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/api/fantasy/yahoo-diag` | Returns diag | | |
| Has `connected` boolean | Yes | | |
| Has `last_sync` | Yes | | |
| GET `/admin/yahoo/test` | Tests connection | | |
| GET `/admin/yahoo/roster` | Returns parsed roster | | |
| Player mappings are present | Yes | | |

**Findings:**


---

## 13. Admin - Database Audit

| Endpoint | Expected | Actual | Notes |
|----------|----------|--------|-------|
| GET `/admin/audit-tables` | Returns audit | | |
| Has `tables` array | Yes | | |
| Each table has `row_count` | Yes | | |
| Each table has `last_updated` | Yes | | |
| Critical tables have data | player_daily_metric, projection_cache_entry, fantasy_roster | | |

**Findings:**


---

## Data Quality Checks

| Check | Expected | Actual | Notes |
|-------|----------|--------|-------|
| No null player_ids in roster | Yes | | |
| All players have position data | Yes | | |
| Projections are non-null where expected | Yes | | |
| Scores are numeric (not null/NaN) | Yes | | |
| Dates are consistent (ET timezone) | Yes | | |
| Category codes match 18-cat spec | Yes | | |

**Findings:**


---

## Performance Checks

| Check | Expected | Actual | Notes |
|-------|----------|--------|-------|
| `/health` responds < 500ms | Yes | | |
| `/scoreboard` responds < 2s | Yes | | |
| `/api/fantasy/roster` responds < 2s | Yes | | |
| `/api/fantasy/matchup` responds < 3s | Yes | | |

**Findings:**


---

## Summary

**Critical Issues (Blockers):**


**High Priority Issues:**


**Medium Priority Issues:**


**Nice to Have / Polish:**


**Overall Assessment:**
- [ ] Ready for UI development
- [ ] Needs fixes before UI
- [ ] Major data quality issues

---

## Post-UAT Actions

- [ ] Create GitHub issues for all findings
- [ ] Prioritize by P1 (blockers) / P2 (important) / P3 (polish)
- [ ] Update HANDOFF.md with UAT results
