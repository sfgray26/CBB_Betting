# Production Data Quality Audit — Reviewed

Date: 2026-04-21

This is a reviewed version of `reports/2026-04-21-production-data-quality-audit-fresh.md` with each major claim classified as:

- Confirmed
- Confirmed but overstated
- Speculative / needs code-level proof

Primary evidence base:

- `postman_collections/responses/*20260421_190158.json`
- `tasks/uat_findings_post_deploy_v5.md`
- current `HANDOFF.md`

## Executive Verdict

The fresh April 21 audit is directionally correct and substantially more accurate than the April 20 audit. The biggest state change is real: the system is no longer dominated by route crashes. The remaining defects are mostly data-enrichment and recommendation-quality defects.

However, several statements in the fresh audit should be tightened before treating it as a source-of-truth handoff:

1. The issue counts are inconsistent.
2. "Roster fully functional" is too strong.
3. Some null-stat interpretations are plausible but not fully proven from response JSON alone.

## Reviewed Claims

### 1. Roster endpoint recovered from 500

Status: Confirmed

Evidence:

- `api_fantasy_roster_20260421_190158.json` returns 200 with a populated player list.

Conclusion:

- The old ImportError failure is fixed as an availability issue.

### 2. Waiver recommendations recovered from 503

Status: Confirmed

Evidence:

- `api_fantasy_waiver_recommendations_20260421_190158.json` returns 200 with a recommendation payload including win probability fields.

Conclusion:

- The old `RiskProfile.acquisition` crash is no longer present in the fresh probe.

### 3. Lineup schedule blindness fixed

Status: Confirmed

Evidence:

- `api_fantasy_lineup_2026_04_20_20260421_190158.json` shows `games_count = 10`.
- The same file shows 8 batters with `status = "START"` and 5 with `status = "BENCH"`.

Conclusion:

- The old all-benched, no-games-found state is no longer current.

### 4. Matchup null ratio stats fixed

Status: Confirmed

Evidence:

- `api_fantasy_matchup_20260421_190158.json` contains populated `AVG`, `OPS`, `ERA`, `WHIP`, and `K_9` values for both teams.

Conclusion:

- The April 20 null-ratio problem no longer reproduces in the fresh probe.

### 5. Raw stat ID `38` leak is gone

Status: Confirmed

Evidence:

- No `"38"` key appears in `api_fantasy_waiver_position_ALL_player_type_ALL_20260421_190158.json`.

Conclusion:

- The old untranslated stat-ID leak appears fixed in the fresh waiver output.

### 6. Waiver category_contributions partially improved

Status: Confirmed

Evidence:

- `api_fantasy_waiver_position_ALL_player_type_ALL_20260421_190158.json` shows `category_contributions` populated for Seth Lugo, Michael Wacha, and Will Warren.

Conclusion:

- The field is no longer 100% empty, but still sparse.

### 7. Decisions staleness fixed

Status: Confirmed

Evidence:

- `api_fantasy_decisions_20260421_190158.json` uses `as_of_date = "2026-04-20"` throughout the fresh probe.

Conclusion:

- The April 20 stale-date issue appears fixed.

### 8. Roster rolling windows 100% null

Status: Confirmed

Evidence:

- `api_fantasy_roster_20260421_190158.json` shows `rolling_7d`, `rolling_14d`, `rolling_15d`, `rolling_30d`, `ros_projection`, `row_projection`, and `game_context` as null in the visible roster rows.
- `tasks/uat_findings_post_deploy_v5.md` independently flags `players_with_stats = 0% (0/23)` for the pre-fix live rerun.

Conclusion:

- This remains a live data-quality problem.

### 9. Waiver intelligence remains mostly hollow

Status: Confirmed

Evidence:

- In `api_fantasy_waiver_position_ALL_player_type_ALL_20260421_190158.json`, `owned_pct` remains `0.0`, `starts_this_week` remains `0`, `statcast_signals` remains empty, `statcast_stats` remains null, and most category_contributions remain empty.

Conclusion:

- Improvement is partial, not broad.

### 10. Universal drop bug persists

Status: Confirmed

Evidence:

- `api_fantasy_decisions_20260421_190158.json` contains 24 waiver decisions.
- All 24 visible waiver decisions use `drop_player_name = "Seiya Suzuki"`.

Conclusion:

- The target changed from Garrett Crochet to Seiya Suzuki, but the universal-drop pattern remains.

### 11. `K_P` field is still effectively mislabeled

Status: Confirmed symptom, speculative root cause

Evidence:

- `api_fantasy_waiver_position_ALL_player_type_ALL_20260421_190158.json` shows Seth Lugo with `IP = "31.1"`, `K_P = "1"`, and `K_9 = "8.04"`.

Conclusion:

- It is safe to say the `K_P` field is not carrying strikeout counts as a consumer would expect.
- It is not safe to claim the exact internal mapping bug from JSON alone.

### 12. Batters have pitcher stats in waiver output

Status: Confirmed

Evidence:

- `api_fantasy_waiver_position_ALL_player_type_ALL_20260421_190158.json` shows batter rows like Dalton Rushing carrying `IP` and `W` fields.

Conclusion:

- This is real schema pollution in the public response.

### 13. Impossible projection narratives persist

Status: Confirmed

Evidence:

- The fresh audit cites 5 remaining impossible projections; the saved decisions response continues to include small-sample zero-rate style narratives.

Conclusion:

- The plausibility-cap problem remains active.

### 14. Draft board age=0 for 92.5% of players

Status: Confirmed

Evidence:

- The fresh draft-board response still contains 185 `age: 0` entries out of 200 players.

Conclusion:

- The data gap is real.

### 15. Briefing uses legacy category names

Status: Confirmed

Evidence:

- `api_fantasy_briefing_2026_04_20_20260421_190158.json` still uses categories `HR`, `SB`, `K`, and `SV`.

Conclusion:

- This remains a live contract mismatch for v2 consumers.

### 16. Player-scores and validate-system routes return 404

Status: Confirmed

Evidence:

- `api_fantasy_player_scores_period_season_20260421_190158.json` returns `{"detail":"Not Found"}`.
- `admin_validate_system_20260421_190158.json` returns `{"detail":"Not Found"}`.

Conclusion:

- The routes are absent in the fresh probe.

### 17. Roster BDL/MLBAM IDs null for all players

Status: Confirmed

Evidence:

- The fresh roster response shows `bdl_player_id = null` and `mlbam_id = null` for every visible row, and grep confirms that pattern throughout the file.

Conclusion:

- This is a real identity-enrichment gap.

### 18. Roster season_stats missing `K_B`, `TB`, and `NSB`

Status: Confirmed nulls, overstated interpretation

Evidence:

- The fresh roster response shows batter season-stats payloads where `K_B`, `TB`, and `NSB` are null while other batting fields are present.

Conclusion:

- The nulls are real.
- The stronger claim that all of these should definitely be populated for every batter is not fully proven from response JSON alone.
- Best wording: likely incomplete stat mapping or incomplete Yahoo category coverage.

## Count Corrections

The fresh audit file has an internal inconsistency:

- It says: `Twelve issues remain active, and two new issues were discovered`.
- The summarized version said: `9 Issues Still Active` and `2 New Issues`.

Treat the counts as needing cleanup before reuse.

## Recommended Clean Summary

Use this wording if you need a short, defensible state summary:

1. The April 21 deploy fixed the major route-availability regressions: roster, waiver recommendations, lineup schedule context, matchup null ratios, stat ID `38` leakage, and stale decisions dates.
2. The main remaining defects are enrichment-quality defects: null rolling windows, null canonical IDs, hollow waiver intelligence, universal-drop behavior, miskeyed waiver stat fields, batter/pitcher stat pollution, legacy briefing categories, and implausible projection narratives.
3. The roster endpoint is recovered but not fully healthy as a data product.

## Priority Order For Claude

1. Roster enrichment and identity mapping
2. Waiver intelligence hydration and universal-drop logic
3. Stat-key/schema cleanup in waiver output
4. Projection plausibility caps
5. Briefing contract modernization