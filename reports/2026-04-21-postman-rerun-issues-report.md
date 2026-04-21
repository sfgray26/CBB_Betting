# 2026-04-21 Postman Rerun Issues Report

## Scope

This report summarizes the current state of the checked-in Postman response captures under `postman_collections/responses/` as of 2026-04-21, with emphasis on:

- outright endpoint failures
- semantically broken `200 OK` responses
- null-heavy and default-heavy payloads
- gaps between the captures and the current `HANDOFF.md`

Note: no new aggregate Postman assertion report was found in the repo for this rerun. The user-reported "14 failed tests" is therefore treated as external run context, not something independently recoverable from the checked-in artifacts.

## Executive Summary

The current captures still show a split system:

1. Core health/admin infrastructure is mostly alive.
2. Several fantasy read/write endpoints are still missing or broken.
3. Several fantasy endpoints return `200` but are functionally degraded because they emit placeholders, neutral defaults, empty arrays, or null-heavy structures instead of live context.
4. Pipeline-health endpoints report healthy backing tables, which strongly suggests read-path drift, fallback misuse, or route ownership issues rather than raw ingestion collapse.

## Highest-Priority Findings

### P0: Roster endpoint is still broken in the latest timestamped capture

Evidence:
- `postman_collections/responses/api_fantasy_roster_20260420_181810.json`

Observed payload:

```json
{
  "detail": "Internal server error",
  "type": "ImportError"
}
```

Impact:
- Core fantasy roster surface is unavailable.
- Any downstream route depending on roster hydration is suspect.

Interpretation:
This report summarizes the current live UAT rerun state as of 2026-04-21 using:

- `tasks/uat_findings_post_deploy_v5.md` as the newest post-deploy validation artifact
- `postman_collections/responses/` as historical capture context

The focus is:

- hard endpoint failures
- semantically broken `200 OK` responses
- null-heavy and default-heavy payloads
- the gap between the newest rerun evidence and the current `HANDOFF.md`

Note: no standalone Newman/Postman assertion JSON was found for the rerun. The user-reported "14 failed tests" is not directly recoverable from the checked-in Postman response JSONs.

The newest available rerun shows the system is materially healthier than the older Apr 20 captures suggested.

Current rerun summary from `tasks/uat_findings_post_deploy_v5.md`:

1. 95 PASS
2. 3 FAIL
3. 1 WARN
4. All validated primary fantasy endpoints in that report returned `200`

The remaining issues are no longer dominated by route crashes. They are now dominated by data-gap and enrichment failures:

1. `GET /api/fantasy/roster` returns players, but all 23 visible roster rows are effectively unenriched.
2. `GET /api/fantasy/waiver` still lacks matchup and category-needs context.
3. `GET /api/fantasy/lineup` is live again, but pitcher-start coverage still emits warnings that deserve follow-up.

Evidence:
### P0: Roster enrichment is structurally present but functionally empty

Evidence:

- `tasks/uat_findings_post_deploy_v5.md`
- `postman_collections/responses/roster_200.json`

Current rerun failure:

- `players_with_stats | >50% | 0% (0/23)`

Observed payload pattern in the rerun sample:

- `season_stats = null`
- `rolling_7d = null`
- `rolling_14d = null`
- `rolling_15d = null`
- `rolling_30d = null`
- `ros_projection = null`
- `row_projection = null`
- `game_context = null`
- `bdl_player_id = null` for visible sample rows
- `mlbam_id = null` for visible sample rows
- `ownership_pct = 0.0`

Impact:

- The roster route is up, but it is not providing actionable player context.
- Any consumer depending on rolling stats, projections, identity mapping, or ownership will degrade downstream.

Interpretation:

- This is not an availability bug anymore.
- It is now a join/enrichment failure and should be treated as the current top live defect.
Observed payload:
### P0: Waiver context is still schedule-blind and need-blind

Evidence:

- `tasks/uat_findings_post_deploy_v5.md`
- `postman_collections/responses/waiver_200.json`

Current rerun failures:

- `matchup_opponent | non-TBD name | 'TBD'`
- `category_deficits_present | >0 categories | 0`

Observed payload pattern:

- `matchup_opponent = "TBD"`
- `category_deficits = []`
- `starts_this_week = 0`
- `projected_points = 0.0`
- `projected_saves = 0.0`
- `category_contributions = {}`
- many player-level nullable fields remain empty (`hot_cold`, `status`, `injury_note`, `injury_status`, `statcast_stats`)

Impact:

- The waiver engine is live but still missing the matchup context and need-state that should drive recommendations.
- This makes the endpoint less useful even when its HTTP contract passes.

Interpretation:

- This is a read-path/context assembly defect, not a transport or routing defect.
{
### P1: Lineup is no longer collapsed, but pitcher-start detection still needs work

Evidence:

- `tasks/uat_findings_post_deploy_v5.md`

Current rerun warning:

- `7 SP(s) have no start today: Eury Pérez, Spencer Arrighetti, Gavin Williams`
- `Only 0 active pitcher slots filled -- consider streaming a SP.`

Impact:

- Not a blocking failure, but it is still a product-quality risk on pitcher recommendations.

Interpretation:

- The earlier blanket lineup default collapse appears fixed.
- The remaining problem is narrower and likely sits in pitcher availability/start detection.

Impact:
### 1. Roster nulls are the most important live data-gap class

The roster endpoint is now available but nearly all enrichment is absent.

Why this matters more than a simple 500:

- the UI can render a roster page that looks alive while still carrying no usable stats
- the null pattern is systemic across the whole roster sample
- downstream lineup, waiver, and decision surfaces are likely to inherit this poverty of context

This is the highest-value remaining live defect.
- Also indicates a code/data-model drift in waiver logic.
### 2. Waiver defaults are the clearest remaining context failure

The waiver endpoint still carries the same structural smell as before, but now as a narrowed live defect set:

- `matchup_opponent = "TBD"`
- `category_deficits = []`
- `starts_this_week = 0`
- projected context remains zeroed
- category contributions remain empty

This suggests the waiver engine is alive but not connected to matchup state and schedule state.
### P0: Lineup endpoint returns complete fallback-collapse data
### 3. Historical lineup defaults are strong evidence of the prior failure mode, but are no longer current truth

Older captures still show the fully collapsed lineup pattern:

- `implied_runs = 4.5`
- `park_factor = 1.0`
- `lineup_score = -4.375`
- `position = "?"`
- `opponent = ""`
- `start_time = null`
- `has_game = false`

That remains useful regression history, but it should not be treated as the primary current defect because the newest rerun shows lineup is live again.
- `postman_collections/responses/api_fantasy_lineup_2026_04_20_20260420_181810.json`
### 4. Historical briefing collapse appears improved in the newest rerun

Older captures showed:

- `categories = []`
- `starters = []`
- `vs = "TBD"`
- contradictory no-game alerts

The newest rerun now shows:

- `categories_present | >0 | 11`
- `starters_present | >0 | 9`
- `overall_confidence | 0-1 float | 0.35`

Conclusion:

- Briefing should move off the top of the active defect list unless the next rerun regresses it again.

### 5. Waiver recommendations had a harmful-move contradiction historically and should remain on watch

Historical evidence from `waiver_recommendations_200.json` showed:

- `action = ADD_DROP`
- `win_prob_before = 1.0`
- `win_prob_after = 0.999`
- `win_prob_gain = -0.001`

This is no longer the best current truth artifact, but it remains a high-value regression watchpoint.
- `lineup_score = -4.375`
### 6. Proxy players remain a structural data gap

Historical roster captures still show proxy players with:

- `is_proxy = true`
- empty `cat_scores`
- sentinel-like z-scores

This is likely a known upstream data-gap class rather than a new rerun regression, but it still affects product quality.
- `has_game = false`
### 7. Decisions endpoint works structurally but still exposes suspect projection extremes

Evidence:

- `postman_collections/responses/decisions_200.json`

Observed examples:

- projected 0.00 ERA ROS narratives
- projected 91.2 HR ROS narrative
- projected 204.4 RBI ROS narrative

Interpretation:

- The endpoint works structurally, but the projection/extrapolation layer still needs plausibility caps.
Impact:
The following surfaces appear structurally healthy in the newest rerun and/or checked-in captures:

- `GET /health`
- `GET /api/fantasy/matchup`
- `GET /api/fantasy/scoreboard`
- `GET /api/fantasy/budget`
- `POST /api/fantasy/roster/optimize`
- `GET /api/fantasy/lineup/2026-04-21`
- `GET /api/fantasy/briefing/2026-04-21`
- `GET /api/fantasy/players/1/scores`
- `GET /api/fantasy/decisions`
- `GET /api/fantasy/decisions/status`
- `pipeline_health_200.json`
- `admin_pipeline_health_20260420_181810.json`

This matters because it argues against any claim that the platform is still in the same state as the earlier 404/500-heavy captures.

The biggest architectural contradiction in the newest evidence is:

1. `pipeline_health` indicates key tables are healthy and current.
2. But the live rerun still shows:
   - roster rows with no stats/projections at all
   - waiver context defaults like `TBD` and empty category deficits

Conclusion:

The dominant failure mode is not ingestion collapse.
It is much more likely to be one or more of:

- identity-mapping misses in consumer assembly
- consumer-layer gating that discards available context
- roster enrichment joins not landing on canonical player IDs
- waiver context builders not receiving matchup/category inputs
- fallback paths still winning over richer persisted data
- This is not natural sparse data.
### What HANDOFF.md gets right

`HANDOFF.md` is directionally correct in that it records the recent repair slices and flags schedule fallback, proxy players, and projection plausibility as active concerns.

### Where HANDOFF.md is stale

`HANDOFF.md` is now behind the newest UAT artifact in three ways:

1. It still says the Apr 21 Postman P0/P1 fixes are local and uncommitted, but `tasks/uat_findings_post_deploy_v5.md` strongly implies a newer deployed validation run already happened.
2. It does not mention the newest live failures that now matter most: roster stats missing on all 23 players, and waiver still returning `TBD` plus empty deficits.
3. It still centers older pre-deploy failures more than the current post-deploy data-gap failures.

### What HANDOFF.md should say next

The next HANDOFF update should add:

1. a post-deploy rerun section keyed to `tasks/uat_findings_post_deploy_v5.md`
2. explicit closure of the earlier route-availability failures that no longer reproduce
3. a narrowed open-defect list centered on roster enrichment and waiver context assembly

## Recommended Priority Order

1. Fix roster enrichment so at least the majority of roster players populate stats, projections, and canonical IDs.
2. Fix waiver matchup/category context so `matchup_opponent` and `category_deficits` are real inputs, not placeholders.
3. Tighten pitcher-start detection and lineup warnings.
4. Keep historical regressions on watch: lineup blanket defaults, briefing contradictions, negative-win-prob waiver recs, raw numeric stat IDs.

## Bottom Line

Based on the newest available rerun evidence:

- The app is no longer dominated by endpoint crashes.
- The main remaining defects are roster enrichment nulls and waiver context/data-gap failures.
- Historical route crashes and blanket lineup collapse now look mostly fixed, but they should stay on regression watch until the next rerun confirms they remain closed.

The next rerun should be compared against this report as the current baseline.

### P0: Briefing endpoint is downstream-corrupted by lineup/schedule blindness

Evidence:
- `postman_collections/responses/briefing_200.json`

Observed payload characteristics:

- `categories = []`
- `starters = []`
- visible bench entries have `vs = "TBD"`
- factors are uniformly `"No game today"`
- alerts still claim some players were starting

Impact:
- Briefing is internally contradictory and not trustworthy.

Interpretation:
- This looks like a shared upstream issue with lineup/game-context resolution, not an isolated briefing bug.

## Secondary Failures / Contract Issues

### P1: Player scores route is still missing

Evidence:
- `postman_collections/responses/player_scores_404.json`
- `postman_collections/responses/api_fantasy_player_scores_period_season_20260420_181810.json`
- `postman_collections/responses/summary.json`

Observed payload:

```json
{
  "detail": "Not Found"
}
```

Impact:
- Consumer read surface for `player_scores` is unavailable despite healthy backing-table claims.

### P1: Validate-system admin route is still missing

Evidence:
- `postman_collections/responses/validate_system_404.json`
- `postman_collections/responses/admin_validate_system_20260420_181810.json`
- `postman_collections/responses/summary.json`

Impact:
- Some admin/UAT expectations are still pointing at a missing or renamed route.

### P1: Roster optimize route state is inconsistent across captures

Evidence:
- `postman_collections/responses/roster_optimize_404.json`
- `postman_collections/responses/api_fantasy_roster_optimize_20260420_181810.json`

Observed behavior:

- one capture shows `404 Not Found`
- the timestamped capture shows `422` because the request body is missing

Interpretation:
- This looks like mixed route-availability and collection-request-shape drift.
- The 422 is not necessarily a backend defect; it may mean the route exists and the request is malformed.

### P1: Matchup simulate is failing due to missing request body

Evidence:
- `postman_collections/responses/matchup_simulate_422.json`
- `postman_collections/responses/summary.json`

Observed payload says `my_roster` is missing.

Interpretation:
- Most likely a collection/request-construction problem rather than an application defect.

## Null / Default / Data-Gap Analysis

### 1. Lineup defaults are clearly broken, not merely sparse

The lineup payload is the strongest evidence of hard semantic degradation.

Why it is broken rather than merely incomplete:

- the same values repeat for every batter
- the placeholders are neutral defaults (`4.5`, `1.0`, `?`, empty opponent)
- every player is forced into `has_game = false`
- every player ends up benched

This pattern should be treated as a P0 product defect.

### 2. Briefing emptiness is not acceptable emptiness

In `briefing_200.json`, the empty arrays are not benign:

- `categories = []`
- `starters = []`
- `monitor = []`

Those would only be acceptable if the system explicitly indicated there were no decisions or no schedule context. Instead, the same payload says `decisions_recorded = true` and `decisions_count = 23`.

That contradiction makes the empty structures product-breaking.

### 3. Waiver surface is dominated by fallback values

Evidence:
- `postman_collections/responses/waiver_200.json`
- `postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_20260420_181810.json`

Repeated gap/default patterns:

- `matchup_opponent = "TBD"`
- `category_deficits = []`
- `owned_pct = 0.0`
- `starts_this_week = 0`
- `projected_points = 0.0`
- `projected_saves = 0.0`
- `hot_cold = null`
- `status = null`
- `injury_note = null`
- `injury_status = null`
- `category_contributions = {}` for many candidates
- `statcast_stats = null`

Interpretation:
- some of these nulls are acceptable individually
- the combination is not
- together they show the waiver engine lacks matchup context, ownership context, and start-count context simultaneously

### 4. Waiver recommendations are logically contradictory even when 200

Evidence:
- `postman_collections/responses/waiver_recommendations_200.json`

Observed contradiction:

- recommendation action is `ADD_DROP`
- `win_prob_before = 1.0`
- `win_prob_after = 0.999`
- `win_prob_gain = -0.001`

Interpretation:
- The system recommends a move that its own probability model says is harmful.
- This is a logic defect, not a display issue.

### 5. Public stat contract leakage still appears in waiver output

Evidence:
- `postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_20260420_181810.json`

Observed:

- keys like `K_P`, `K_B`, `K_9`
- raw numeric-looking key `"38"`

Interpretation:
- If `K_P` / `K_B` / `K_9` are the intended v2 public canonicals, they are acceptable.
- The bare `"38"` key is not acceptable in a public API response.
- This indicates at least partial stat-normalization leakage.

### 6. Matchup endpoint is only partially normalized

Evidence:
- `postman_collections/responses/matchup_200.json`
- `postman_collections/responses/api_fantasy_matchup_20260420_181810.json`

Observed:

- older capture uses fractional `NSB` values like `0/0`, `1/5`
- newer capture normalizes `NSB` to plain values
- newer capture still has `AVG = null`, `OPS = null`, `ERA = null`, `WHIP = null`, `K_9 = null` for one or both teams

Interpretation:
- some nulls may be legitimate early-week emptiness
- but the contract is still inconsistent across captures

### 7. Roster proxy players remain a structural data gap

Evidence:
- `postman_collections/responses/roster_200.json`

Observed patterns:

- `is_proxy = true` for multiple players
- `cat_scores = {}` on those proxy rows
- sentinel-like z-scores such as `-0.8` and `-1.5`

Impact:
- downstream optimizers and recommendation layers are at risk of treating synthetic/incomplete rows as real scored entities

Interpretation:
- This is a known data-gap class, not necessarily a new regression.

### 8. Decisions endpoint mostly works but still exposes suspect projection extremes

Evidence:
- `postman_collections/responses/decisions_200.json`

Observed examples:

- projected 0.00 ERA ROS narratives
- projected 91.2 HR ROS narrative
- projected 204.4 RBI ROS narrative

Interpretation:
- The endpoint works structurally, but the projection/extrapolation layer still needs plausibility caps.

## What Looks Healthy

The following surfaces appear structurally healthy in the checked-in captures:

- `health_root_200.json`
- `health_check_200.json`
- `draft_board_200.json`
- `decisions_200.json` (structure healthy, content still has plausibility issues)
- `pipeline_health_200.json`
- `admin_pipeline_health_20260420_181810.json`
- `scheduler_status_200.json`

This matters because it argues against a total platform outage.

## Most Important Contradiction

The biggest architectural contradiction in the captures is:

1. `pipeline_health` says `player_scores`, `probable_pitchers`, `simulation_results`, and `player_rolling_stats` are healthy and current.
2. But consumer endpoints still show:
   - missing `player_scores` route
   - lineup schedule blindness
   - briefing collapse
   - waiver context defaults

Conclusion:

The dominant failure mode is probably not ingestion.
It is much more likely to be one or more of:

- route drift
- wrong handler ownership
- stale serializer contracts
- fallback code taking precedence over persisted data
- consumer-layer gating that incorrectly discards available context

## HANDOFF.md Review

### What HANDOFF.md gets right

`HANDOFF.md` is broadly aligned with the repo state in three ways:

1. It correctly frames the latest slices as a sequence of Apr 20 and Apr 21 fantasy repairs.
2. It explicitly calls out that the latest Postman P0/P1 fixes are local and uncommitted.
3. Its open-defect list already includes several issues that still appear in the captures:
   - `/admin/backfill/player-id-mapping` runtime uncertainty
   - schedule fallback mode ambiguity
   - proxy-player data gaps
   - impossible projection extrapolations

### Where HANDOFF.md is ahead of the captures

The report in `HANDOFF.md` is more optimistic than the currently checked-in Postman captures.

Examples:

1. It says the roster ImportError path is fixed locally, but the latest timestamped capture still shows `ImportError`.
2. It says the waiver recommendations `RiskProfile.acquisition` error is already fixed in a prior session, but the latest timestamped capture still shows that error.
3. It says lineup/admin regressions were repaired, but the checked-in lineup/briefing captures still show pre-deploy all-default behavior.

Interpretation:

- This is not necessarily a contradiction in code state.
- It most likely means the Postman captures are still pre-deploy or pre-rerun relative to the local fixes described in `HANDOFF.md`.

### What HANDOFF.md should mention next

If a fresh rerun confirms the same issues, `HANDOFF.md` should be updated to add:

1. a fresh post-rerun defect section, separate from the Apr 20 captures
2. an explicit note distinguishing pre-deploy captures from post-deploy captures
3. a line on whether the user-reported "14 failed tests" came from assertion failures vs HTTP failures vs schema checks

## Recommended Next Report Structure For The Next Rerun

When the new Postman rerun finishes, the next issue report should group findings into:

1. **Hard endpoint failures**
   - non-200 responses
   - route missing
   - import/runtime errors

2. **Semantically broken 200s**
   - lineup fallback collapse
   - briefing contradictions
   - waiver negative win-prob recommendations

3. **Null/default-heavy but possibly recoverable gaps**
   - proxy players
   - matchup ratio nulls
   - missing statcast enrichments
   - zero ownership / zero starts_this_week

4. **Collection/request defects**
   - malformed POST bodies
   - outdated route paths

That split will make the next triage cycle much cleaner.

## Bottom Line

Based on the checked-in captures available right now:

- P0 defects still visible in artifacts: roster ImportError, waiver recommendations exception, lineup fallback collapse, briefing collapse.
- P1 defects still visible in artifacts: missing `player_scores` route, admin route drift, waiver contract leakage, recommendation logic contradictions.
- null/data-gap work is still substantial, but the biggest product blockers are not nulls alone — they are broken consumers and fallback misuse.

The next rerun should be evaluated against this report as a baseline.