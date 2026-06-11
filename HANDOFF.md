# HANDOFF.md — Crisis Session: Fantasy Baseball Regression (2026-05-31)

> **Date:** 2026-05-31 | **Status:** 🚨 CRITICAL REGRESSION DETECTED
> **Branch:** `stable/cbb-prod` | **HEAD:** `fe7b84b` (2026-05-25)
> **Last known good:** 2026-05-22 (Wave 5A complete)
> **Regression window:** May 23–31, 2026 (9 days)

---

## 🚨 REGRESSION ALERT — ALL MODULES BROKEN

**Source:** UAT Report `UAT_REPORT_FANTASY_BASEBALL_WEEK_10_2026-05-31.md`
**Verdict:** The War Room suite is functionally non-trustworthy. An elite fantasy manager cannot use it for roster decisions.
**Root cause assessment:** Empty or broken schedule/games feed cascading through all modules, plus silent fallbacks fabricating confident output.

### Critical System State (as of 2026-05-31 12:09 PM EDT)

| Module | P0 Issues | P1 Issues | Functional Status |
|--------|-----------|-----------|-------------------|
| **Lineup Optimizer** | 4 | 2 | ❌ BROKEN — flat 58.0 fallbacks, benches Soto, no schedule gating |
| **My Roster** | 1 | 1 | ❌ BROKEN — matchup shows 0W-0L-18T, AVG .237 implausible |
| **Waiver Wire** | 1 | 3 | ❌ BROKEN — no drop recommendations, infinite loading, contradictory tags |
| **Streaming Station** | 1 | 1 | ❌ BROKEN — duplicate of waiver, wrong ranking axis |
| **Budget** | 0 | 2 | ⚠️ DEGRADED — IP inconsistency, unlabeled metric |
| **Weekly Preview** | 2 | 0 | ❌ BROKEN — Unknown opponent, contradictory win% |
| **Data Integrity** | 1 | 2 | ❌ BROKEN — ownership stale, Josh Jung anomaly |

---

## Immediate Action Plan

### Phase 1 — Foundation (Schedule + Data Unification) — BLOCKING ALL ELSE

**Status:** NOT STARTED
**Critical Path:** Unblocks Optimizer, Streaming, Preview
**Agent:** DataAgent → Gemini (research + feed investigation) + BackendAgent → Claude Code (implementation)

### Phase 2 — Optimizer Scoring

**Status:** NOT STARTED
**Critical Path:** Makes lineup recommendations trustworthy
**Agent:** BackendAgent → Claude Code (implementation) + QAAgent → Claude Code (tests)

### Phase 3 — Gatekeeping & UX

**Status:** NOT STARTED
**Critical Path:** Prevents bad decisions
**Agent:** FullStackAgent → Claude Code (gating) + FrontendAgent → Codex (UI fixes)

### Phase 4 — Waiver & Streaming

**Status:** NOT STARTED
**Critical Path:** Makes add/drop decisions possible
**Agent:** BackendAgent → Claude Code (drop recommendations) + FullStackAgent → Claude Code (streaming rebuild) + FrontendAgent → Codex (UI)

### Phase 5 — Roster, Budget, Preview

**Status:** NOT STARTED
**Critical Path:** Completes War Room suite
**Agent:** BackendAgent → Claude Code (roster aggregation, preview opponent) + FrontendAgent → Codex (UI polish)

### Phase 6 — Data Integrity & QA — PARALLEL FROM START

**Status:** NOT STARTED
**Critical Path:** Catches systemic issues early
**Agent:** QAAgent → Claude Code (player-ID join verification) + QAAgent → Claude Code (sanity monitors) + QAAgent → Claude Code (debug string audit)

---

## Known Regression Symptom Cluster

1. **"No Game" everywhere** — schedule/games feed empty
2. **Flat 58.0 proxy_projection fallback** — optimizer noise sorting
3. **Top hitters benched without cause** — Juan Soto benched (OPS .974)
4. **Pitchers ranked 96–99 vs hitters 87** — scoring scale mismatch
5. **Matchup zeros (0W-0L-18T)** — data source not unified
6. **Infinite loading on waiver** — frontend Promise chain broken
7. **Unknown opponent in Preview** — Week 11 lookup failing
8. **Contradictory win% (100% vs "losing K")** — fallback default artifact
9. **Ownership 4% for .307/.839 OPS hitter** — player-ID join broken
10. **Debug strings leaking** — date banner, score provenance tags

---

## Sprint Closure Criteria (Definition of Done)

The sprint closes when:
- [ ] No P0 remains open
- [ ] All schedule-dependent modules show real game data (no "No Game" everywhere)
- [ ] Optimizer produces a lineup where the top-3 OPS hitter is not benched without injury/no-game cause
- [ ] Waiver Wire shows a recommended drop for every suggested add
- [ ] Weekly Preview shows a real opponent and internally consistent win probability
- [ ] QAAgent's cross-check script passes on a 15-player sample

---

## Escalation Paths

| Situation | Escalate To | How |
|-----------|-------------|-----|
| Schedule feed investigation | DataAgent → Gemini | Route with UAT report + "No Game everywhere" evidence |
| Backend scoring algorithms | BackendAgent → Claude Code | Route with optimizer output showing 58.0 fallbacks |
| Frontend loading states | FrontendAgent → Codex | Route with "Loading recommendations…" persistence evidence |
| Data integrity joins | QAAgent → Claude Code | Route with Josh Jung anomaly (ESPN vs CBB Edge) |
| Railway deployment needed | Codex | Route with full PR + rollback plan |

---

## Guardrails (Do NOT Cross)

- Do **NOT** mutate user's actual Yahoo lineup (no "Apply" clicks in production)
- Do **NOT** change z-score valuation framework
- Do **NOT** add new data providers
- Do **NOT** implement features beyond UAT report scope
- If task requires owner clarification → flag as **BLOCKED** and move on

---

## Session Log

### 2026-05-31 13:20 — CRISIS SESSION START
- **Trigger:** User attached HERMES_ORCHESTRATION_BRIEF.md + UAT_REPORT_FANTASY_BASEBALL_WEEK_10_2026-05-31.md
- **Assessment:** Significant regression since Wave 5A (2026-05-22). All War Room modules broken.
- **Action:** HERMES updating HANDOFF.md, then producing task backlog and sprint plan per orchestration brief.
- **Next:** Delegate Phase 1 tasks (schedule feed) immediately to DataAgent + BackendAgent.

### 2026-05-31 — TASK-1, TASK-4, TASK-5 — Claude Code (cbb-architect)

**TASK-1: Schedule/Games Feed (root cause of "No Game" everywhere)**

Root cause identified: `DailyLineupOptimizer.fetch_mlb_odds()` queries `MLBOddsSnapshot JOIN MLBGameLog` first. When no odds are in the DB (BDL odds ingestion not yet run), it falls to `_load_schedule_fallback_games()` which reads `ProbablePitcherSnapshot`. When THAT table is also empty for today (ingestion jobs not yet fired at startup time), the method returned `[]`. With zero games in `team_odds`, the lineup router's no-game gate (`if _b.status == "START" and not _b.opponent: _b.status = "BENCH"`) benches every roster player.

Fix: Added Tier 2 fallback inside `_load_schedule_fallback_games()` — when `ProbablePitcherSnapshot` is empty for the date, the method now calls the MLB Stats API (`/api/v1/schedule?sportId=1&date=DATE&gameType=R`) to get today's games directly. Park-factor-adjusted neutral implied runs are synthesized (same formula as the snapshot tier). This ensures game context is always available.

File: `backend/fantasy_baseball/daily_lineup_optimizer.py` — method `_load_schedule_fallback_games`

**TASK-4: Flat 58.0 proxy_projection Fallback**

Root cause identified: `_projection_fallback_score()` in `backend/routers/fantasy.py` capped `is_proxy=True` players (no Steamer/Statcast data found) at a score of 58.0. Since the 0-100 scale for real players runs ~50-95, capping proxy players at 58 caused them to rank ahead of real-scored players near the floor. This produced noise-based tie-breaking.

Fix: Changed the `is_proxy` branch to return `(0.0, "no_score")` instead of `(min(score, 58.0), "proxy_projection")`. Players with `no_score` stay at 0.0 after normalization and will only fill slots if no real-scored player is eligible. The `is_not_proxy` path is unchanged (uses z_score from `player_board`).

File: `backend/routers/fantasy.py` — function `_projection_fallback_score`

**TASK-5: Hitter/Pitcher Score Scale Mismatch**

Root cause confirmed: `score_0_100` in `player_scores` is a within-cohort percentile rank. A pitcher at 97 is the 97th percentile among pitchers; a hitter at 87 is 87th among hitters. These are not comparable across position types. The `optimize_roster` endpoint used `score_0_100` directly as `lineup_score` for both groups, making the displayed scores misleading and potentially causing the ILP solver to misrank players when a two-way player straddled both groups.

Fix: Added `_normalize_group_scores()` function inside `optimize_roster()`. After building `player_data`, the function splits players into `_hitter_group` and `_pitcher_group`, applies min-max normalization to the `lineup_score` within each group (excluding `no_score` players), and updates `player_data` in-place. The ILP solver and greedy pitcher sort then operate on position-normalized 0-100 scores. A top-3 hitter at 87 normalizes to ~100 within their group, preventing any cross-scale comparison error.

File: `backend/routers/fantasy.py` — function `optimize_roster`

**Status:** All three fixes implemented. Syntax checks passed locally (both files exit 0).

**Test results:** 3052 passed, 3 skipped, 4 pre-existing failures (in `test_row_projector.py` / `test_row_projector_fixes.py` — confirmed pre-existing on `fe7b84b` before our changes). **Zero regressions introduced.**

---

### 2026-05-31 — Phase 1 Complete — Pending Deploy

**Files changed (unstaged):**
- `backend/fantasy_baseball/daily_lineup_optimizer.py` — TASK-1 Tier 2 MLB Stats API fallback
- `backend/routers/fantasy.py` — TASK-4 no_score branch + TASK-5 group normalization

**Delegation to Gemini CLI (DevOps):**
```
railway run python -m py_compile backend/fantasy_baseball/daily_lineup_optimizer.py
railway run python -m py_compile backend/routers/fantasy.py
```
Then commit and deploy `stable/cbb-prod` to Railway.

**Smoke tests after deploy:**
- `GET /api/fantasy/lineup/2026-05-31` — at least one player shows a real opponent (not "No Game")
- `POST /api/fantasy/roster/optimize` — no `lineup_score = 58.0`; no `score_source = "proxy_projection"`
- DB check: `SELECT team, opponent, game_date FROM probable_pitchers WHERE game_date = CURRENT_DATE LIMIT 10`

**Remaining P0 tasks (Claude Code):**
- **TASK-7:** Gate optimizer on valid schedule — blocked on TASK-1 deploy
- **TASK-10:** Drop recommendations for waiver adds (BackendAgent)
- **TASK-13:** Streaming Station rebuild on projected category contribution (FullStackAgent)
- **TASK-18:** Week 11 opponent lookup fix (DataAgent + BackendAgent)

---

### 2026-05-31 — TASK-2 — Unify matchup data source (Claude Code)

**Root cause:** `/api/fantasy/scoreboard` (used by `/war-room/roster`) called `get_matchup_stats(week)` which internally calls `get_scoreboard()` but has fragile nested-struct team-key matching. When it failed to find the user's team, it returned `{"my_stats": {}, "opp_stats": {}}` → `assemble_matchup_scoreboard` received all zeros → 0W-0L-18T on the roster page.

Meanwhile `/api/fantasy/matchup` (used by `/war-room`) called `get_scoreboard()` directly and used `_iter_scoreboard_matchup_teams()` + `_flatten_scoreboard_team_entry()` which handles all Yahoo response shape variants correctly. This worked fine, showing the live 9-7 score.

**Fix:** Replaced the `get_matchup_stats()` call in `/api/fantasy/scoreboard` with `client.get_scoreboard(week=week)` + inline `_parse_stats_to_float()` helper that uses `_iter_scoreboard_matchup_teams()` and the same `_YAHOO_STAT_FALLBACK` stat_id map. Both endpoints now read from the same Yahoo data source via the same parsing path.

**File:** `backend/routers/fantasy.py` — `get_matchup_scoreboard()` (lines ~6426-6510 replaced)

**Syntax check:** Exit 0.

**Smoke test needed after deploy:**
- `GET /api/fantasy/scoreboard` — should return real category wins/losses, not all-tied
- `/war-room/roster` matchup strip should match `/war-room` live score

---

---

### 2026-06-01 — TASK-2 Follow-up — Pending Deploy

**Files changed (unstaged):**
- `backend/fantasy_baseball/yahoo_client_resilient.py` — flexible Yahoo team-key matching when resolving the user's matchup
- `backend/routers/fantasy.py` — replace hardcoded scoreboard constraint values with parsed IP and date-derived remaining days

**Validation needed before deploy:**
- `py_compile` for both files
- production `/health`
- production `/api/fantasy/scoreboard` smoke check with real, non-hardcoded constraint values

*Last updated: 2026-06-01 (Codex — follow-up backend fixes queued for deploy)*
