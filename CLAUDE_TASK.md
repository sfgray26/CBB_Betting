You are working on the CBB Edge fantasy baseball backend (FastAPI, SQLAlchemy, PostgreSQL, deployed on Railway). This is a follow-up to a UAT session that fixed 4 critical regressions and now needs the remaining issues resolved.

## CONTEXT FILES (read these first)
- HANDOFF.md — contains the full task backlog with root cause analysis for each issue
- UAT_REPORT_FANTASY_BASEBALL_WEEK_10_2026-05-31.md — the original UAT report
- backend/routers/fantasy.py — all /api/fantasy/* routes (very large, search for relevant functions)
- backend/fantasy_baseball/daily_lineup_optimizer.py — schedule/game context logic
- backend/services/scoreboard_orchestrator.py — scoreboard assembly

## ISSUES TO FIX (in priority order)

### TASK-A: Fix fetched_at null in freshness metadata
**File:** backend/services/scoreboard_orchestrator.py
**Problem:** The scoreboard response returns "fetched_at": null. The "computed_at" is set but "fetched_at" (when Yahoo data was actually fetched) is missing. A manager needs to know data freshness.
**Fix:** In assemble_matchup_scoreboard() or wherever freshness metadata is built, record the actual timestamp of the Yahoo API call (not just when the response was assembled). Store it in the freshness dict and include it in the JSON response.
**Test:** GET /api/fantasy/scoreboard?week=10 should return fetched_at as a valid ISO timestamp, not null.

### TASK-B: Add waiver drop recommendations
**File:** backend/routers/fantasy.py (waiver recommendations endpoint)
**Problem:** The waiver wire shows suggested adds but never recommends who to drop. In a league with limited acquisitions, every add is meaningless without a corresponding drop.
**Fix:** In the waiver recommendations handler, after computing the best add for each gap category, compute the worst rosterable player to drop. The drop candidate should be the lowest-scored player (by lineup_score or ROS z-score) at a position that becomes redundant after the add. Show net category delta (what categories improve, what might regress). Add a "recommended_drop" field to each waiver recommendation.
**Test:** GET /api/fantasy/waiver/recommendations should include a recommended_drop player_id/name for each recommendation, with a net_category_delta dict.
**Constraint:** Do NOT change the existing scoring framework or z-score valuation. Work within the existing player_board and category_aware_scorer modules.

### TASK-C: Fix weekly preview opponent lookup
**File:** backend/routers/fantasy.py (weekly preview endpoint) and/or backend/fantasy_baseball/daily_briefing.py
**Problem:** The weekly preview shows "Unknown" opponent and empty category table. The opponent lookup for the next week is failing.
**Fix:** The preview should use the same opponent resolution logic that the scoreboard already uses successfully. Find where get_matchup_scoreboard() resolves the opponent and mirror that path in the preview handler. If the Yahoo API doesn't expose next-week matchups yet, fall back to computing it from the league schedule (week cycle + team position).
**Test:** GET /api/fantasy/matchup-preview should return a real opponent_name (not "Unknown") and a non-empty category projection table.

### TASK-D: Rebuild streaming station on schedule data
**File:** backend/routers/fantasy.py (streaming endpoint)
**Problem:** The streaming station is a duplicate of the waiver wire — it ranks by season z-score, not upcoming schedule. A manager looking for a two-start SP wants to see who starts twice with favorable matchups.
**Fix:** Change the streaming endpoint to:
1. Filter for pitchers with 2 starts in the next 7 days (using ProbablePitcherSnapshot or the MLB Stats API schedule fallback already added)
2. Rank by projected category contribution (K, W, QS, NSV) in those starts
3. Include opponent quality (implied runs allowed) and park factor for each start
4. Include a "schedule_score" that combines 2-start bonus + favorable matchups
**Test:** GET /api/fantasy/streaming should return pitchers sorted by schedule_score, with start_date and opponent for each projected start.

### TASK-E: Fix rate stat aggregation bug
**File:** backend/routers/fantasy.py (roster endpoint or scoreboard)
**Problem:** The UAT report flagged a .237 AVG as implausible for a team with known good hitters. This suggests rate stats (AVG, OPS, ERA, WHIP) are being aggregated incorrectly (summing instead of recomputing from raw components).
**Fix:** Verify that AVG = total_H / total_AB (not sum of individual AVGs). OPS = total_OBP + total_SLG (or compute from total H, BB, AB, TB). ERA = total_ER / total_IP * 9. WHIP = (total_H + total_BB) / total_IP. Apply this fix wherever team-level rate stats are aggregated (roster endpoint, scoreboard, matchup preview).
**Test:** The roster endpoint and scoreboard should show rate stats that match recomputation from raw counting stats, not naive summation.

### TASK-F: Add move confirmation diff
**File:** backend/routers/fantasy.py (optimize_roster response)
**Problem:** The optimizer returns a full lineup assignment but the frontend "Apply All N Moves" button shows no confirmation diff. A manager can't see what actually changes before committing.
**Fix:** In the optimize_roster response, include a "diff" section that shows: (1) players moved from bench to start, (2) players moved from start to bench, (3) new acquisitions (if waiver integration), (4) net category impact of the proposed lineup vs current. The diff should be a first-class field in the API response so the frontend can render it.
**Test:** POST /api/fantasy/roster/optimize response should include a "proposed_diff" field with changes and net_category_impact.

### TASK-G: Fix data accuracy — 15-player cross-check
**File:** backend/fantasy_baseball/player_board.py, backend/fantasy_baseball/projections_loader.py
**Problem:** The UAT report flagged Josh Jung (.307 AVG, 6 HR, .839 OPS) at 4% ownership. This suggests a player-ID join error (data from one player attributed to another) or stale ownership feed.
**Fix:**
1. In the player board / projection loader, verify that ownership % is pulled from the correct Yahoo API field (not accidentally from a different player's record).
2. Run a spot-check: pick 15 known players and verify their ownership %, AVG, and OPS against Yahoo's live data. Flag any discrepancies >5% for ownership or >0.020 for AVG.
3. If player-ID joins are the issue, add a defensive check: when two players have nearly identical names, require a secondary match on team or position before merging data.
**Test:** The player board should pass a sanity check where no player has ownership <10% while batting .300+ with .850+ OPS (unless they're a prospect/unknown). Log any mismatches as warnings.

## WORKING RULES
- Do NOT change the z-score valuation framework.
- Do NOT add new data providers.
- Do NOT mutate the user's actual Yahoo lineup (no "Apply" clicks in production).
- Run py_compile on every changed file.
- Run relevant pytest tests after each fix. The pre-existing asyncio failures in test_ingestion_orchestrator.py are expected — ignore them.
- Commit each task separately with a descriptive message.
- If a task is blocked or requires user clarification, mark it BLOCKED and move to the next task.
- Report back which tasks were completed, which are blocked, and test results for each.
