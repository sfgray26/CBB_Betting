# HANDOFF.md — MLB Platform Operating Brief

> **Date:** 2026-04-21 | **Architect:** Claude Code (Master Architect)
> **Status:** Apr 20 UAT remediation and Apr 21 Lineup/Admin repair are **committed** (see commits below). Apr 21 Postman P0/P1 regression fixes are **local and uncommitted** — awaiting Gemini deploy. Phase 4.5a Quality Remediation is **COMPLETE**. Phase 4.5b UAT and Phase 5 (frontend) remain gated.

---

## 1. Mission Accomplished — Latest Session (2026-04-21, Postman P0/P1)

- Root-caused and fixed the highest-priority fantasy API regressions surfaced by the April 20 Postman captures in `postman_collections/responses/`.
- Four P0 defects plus three P1 contract defects, each with a regression test. No refactors or speculative cleanup.
- Full suite after fixes: **2285 passed, 1 failed (pre-existing, out of scope), 3 skipped**.

See §3 for the rolled-up session log covering this session and the two prior ones (already committed).

---

## 2. Current State

### 2.1 Deploy State

| Slice | State | Commit(s) |
|-------|-------|-----------|
| Apr 20 UAT Remediation (schemas, fantasy router hardening, waiver edge detector, scoreboard orchestrator, player mapper) | **Committed** | `a2e2e56`, `791f6fa`, `3347937` |
| Apr 21 Lineup/Admin Repair (probable-pitcher fallback, smart-selector positions, admin compatibility) | **Committed** | `2749276`, `9147f83`, `80889dc`, `8ca2ebe` |
| Apr 21 Postman P0/P1 Regression Fixes (briefing MONITOR routing, waiver MCMC gate, stat-contract filter, roster ImportError hoist) | **Local / uncommitted** | — |

Railway auto-deploy state for committed slices is not confirmed in this session. Gemini should verify `/admin/version` reflects the latest commit on deploy.

### 2.2 Phase Plan Progress

| Phase | Focus | Status |
|-------|-------|--------|
| 0 | Contracts + stat_contract package | **COMPLETE** |
| 1 | V1→V2 migration + 7 data gap closures | **COMPLETE** |
| 2 | 18-category rolling stats + ROW projector | **COMPLETE** (4 greenfield cats — W, L, HR_P, NSV — deferred, awaiting upstream Yahoo data) |
| 3 | Pure functions + H2H Monte Carlo + MCMC v2 alignment | **COMPLETE** |
| 4 | P1 API endpoints (scoreboard, budget, roster, move, optimize) | **COMPLETE** |
| 4.5a | Quality Remediation — wire live data, fix LOWER_IS_BETTER, expand simulation coverage | **COMPLETE** (P4 statcast integration remains optional) |
| 4.5b | UAT — manual sanity-check of live API data | **IN PROGRESS** — Postman captures drove this session's fixes; rerun required after deploy |
| 5 | Frontend P1 pages (Matchup Scoreboard + My Roster) | **GATED** on 4.5b |
| 6 | P2 backends (waiver v2, streaming) | Gated on Phase 5 |
| 7 | P2 frontend (Waiver Wire + Streaming) | Gated on Phase 6 |
| 8–9 | P3 pages (Trade + Season Dashboard) | Gated on Phase 7 |

### 2.3 Open Defects (Prioritized)

| # | Severity | Defect | Owner / Next Action |
|---|----------|--------|--------------------|
| 1 | P1 | Pre-existing NSB composite test failure (`test_composite_z_excludes_z_sb_when_both_populated`) — reproduces on clean HEAD | Architect — decide whether composite math excludes `z_sb` when both SB and CS populated, or update test expectation |
| 2 | P2 | Unknown Yahoo stat_ids (e.g. `"38"`) silently dropped from waiver output | Architect — decide whether to log as warning so `YAHOO_ID_INDEX` can be enriched over time |
| 3 | P2 | `/admin/backfill/player-id-mapping` runtime behavior — reported "no response" in prod, not reproduced locally | Gemini — confirm on next deploy: responds / times out / hangs |
| 4 | P2 | Schedule-blind lineup fallback emits synthetic implied runs without a mode flag | Architect — decide whether to surface "schedule fallback mode" so UI can distinguish sportsbook-derived vs neutral context |
| 5 | P2 | Proxy players (6 of 23 roster — Ballesteros, Kim, Murakami, Smith, Arrighetti, De Los Santos) carry hardcoded `z=-0.8`, empty `cat_scores`, genuinely absent from Steamer/ZiPS | Claude — future pipeline work to synthesize proxy projections or route through Yahoo season stats |
| 6 | P3 | Impossible projection extrapolations surfaced in Decisions API (0.00 ERA ROS, 91.2 HR ROS, 204.4 RBI ROS) | Claude — add plausible ROS caps in projection math |

---

## 3. Recent Session Log

Three consecutive-day sessions; rolled up here. File-by-file breakdown is in `git log -p` on the listed commits.

### 3.1 Apr 21 — Postman P0/P1 Regression Fixes (THIS SESSION, uncommitted)

| Defect | Fix |
|--------|-----|
| Waiver recs 500 on `RiskProfile.acquisition` | Already fixed in commit `9147f83` (prior session). This session added regression test only. |
| Roster endpoint 500 ImportError | Hoisted `player_mapper` imports to module-top in `backend/routers/fantasy.py` so the ImportError fails at deploy time. |
| Briefing `vs="TBD"` + contradictory START+"no game today" alerts | `daily_briefing.py` now prefers `opposing_pitcher.team` over scheduler fallback and routes no-game players to MONITOR. `smart_lineup_selector.py` reworded warning from "Starting but no game today" to "No game scheduled — monitor before lock". |
| Waiver recs pass through negative `win_prob_gain` | Added MCMC gate in `get_waiver_recommendations`: when the simulator ran (`mcmc_enabled=True`) with `win_prob_gain < 0`, the candidate is skipped. |
| Waiver stat-contract leak `"38": "0"` | `_to_waiver_player` now drops any key still all-digits after `sid_map.get(k, k)`. Canonical codes (`K_P`, `K_B`, `K_9`, `NSV`, `QS`, `OBP`, `ERA`, `WHIP`, `IP`) preserved — they are stat_contract v2 canonicals, not leakage. |

**Files:** `backend/routers/fantasy.py`, `backend/fantasy_baseball/daily_briefing.py`, `backend/fantasy_baseball/smart_lineup_selector.py`, `tests/test_waiver_edge.py`, **new** `tests/test_daily_briefing_no_game_contract.py`, **new** `tests/test_waiver_recommendations_gates.py`.

**Validation:** `py_compile` clean. Targeted pytest 20 passed. Full suite 2285 passed / 1 pre-existing fail / 3 skipped.

### 3.2 Apr 21 — Lineup/Admin Regression Repair (committed: `2749276`, `9147f83`, `80889dc`, `8ca2ebe`)

- `DailyLineupOptimizer.fetch_mlb_odds()` now falls back to persisted `probable_pitchers` snapshots and synthesizes game context when live Odds API coverage is absent. Lineup APIs no longer return blanket `implied_runs=4.5`, `park_factor=1.0`, `has_game=false`.
- `smart_lineup_selector.solve_smart_lineup()` now returns `positions`; both lineup routes stop emitting blanket `position="?"`.
- Admin compatibility fixed: `/admin/odds-monitor/status` degrades cleanly without Odds API config, `/admin/yahoo/test` emits `connected: true`, `/admin/audit-tables` returns a `tables` field.

### 3.3 Apr 20 — UAT Remediation (committed: `a2e2e56`, `791f6fa`, `3347937`)

UAT run against Railway production: 53 PASS / 15 FAIL.

- **Critical discovery:** `ConfigDict` was missing from `backend/schemas.py` — crashed the entire fantasy router on import, causing scoreboard/budget/optimize/decisions to 404 and explaining many observed failures.
- Waiver edge detector hardened: projection enrichment so `need_score` no longer collapses to 0.0; long-term hold-value floors and protected-drop logic so core assets (Juan Soto, Eury Perez) are not treated as routine drop fodder.
- `/api/fantasy/roster/optimize` identity resolution fixed: resolve via canonical `yahoo_key` variants first, projection-driven fallback scores replace flat 50.0.
- `/api/fantasy/roster` stats enrichment added via `get_players_stats_batch()`.
- Duplicated inline `/api/fantasy/waiver/recommendations` route in `main.py` removed — canonical router is the single owner.
- `backend/services/scoreboard_orchestrator.py` ratio-component crash fixed: preserve default `games_remaining`, keep season-only players in ROW inputs.
- `backend/services/player_mapper.py` import drift fixed (`PlayerIdMap` → `PlayerIDMapping`).
- `/api/fantasy/players/{id}/scores` hardened against legacy table shapes.
- `/api/fantasy/decisions/status` SQL normalization.

Post-fix suite: **2245 passed, 3 skipped, 0 failed**.

---

## 4. Delegation Bundles

### 4.1 Gemini CLI — Deploy Postman Regression Fixes

**Task:** commit, push, deploy, and validate the Apr 21 Postman P0/P1 regression fixes. Gemini is restricted to deploy, logs, and validation — no Python edits.

See §5.1 for the copy-paste prompt.

### 4.2 Kimi CLI

No active delegation. Phase 2 research prompts (K1 rolling stats audit, K2 ROW projection spec) remain available if Phase 2b greenfield work is reopened.

---

## 5. HANDOFF PROMPTS

### 5.1 For Gemini CLI — Postman Regression Deploy

```
Deploy the April 21 Postman P0/P1 regression fixes to Railway.

Do NOT edit any Python files. Gemini CLI is restricted to deploy, logs, and validation only.

Uncommitted files in this slice:
  backend/routers/fantasy.py
  backend/fantasy_baseball/daily_briefing.py
  backend/fantasy_baseball/smart_lineup_selector.py
  tests/test_waiver_edge.py
  tests/test_daily_briefing_no_game_contract.py  (new)
  tests/test_waiver_recommendations_gates.py     (new)
  HANDOFF.md
  tasks/todo.md

Purpose of this slice:
  1. Roster endpoint should no longer 500 on player_mapper ImportError — imports are
     hoisted to module top so the error surfaces at deploy time, not per-request.
  2. Briefing endpoint should report opponents as real team codes (PHI, NYM, etc.)
     when probable-pitcher context is available, and route no-game players to MONITOR
     instead of emitting contradictory START + "no game today" alerts.
  3. Waiver recommendations should never surface an MCMC-gated move with a negative
     simulated win_prob_gain.
  4. Waiver stats block should not contain bare numeric Yahoo stat_ids such as "38" —
     only canonical stat_contract v2 codes.

Pre-deploy validation:
  1. venv/Scripts/python -m py_compile \
       backend/routers/fantasy.py \
       backend/fantasy_baseball/daily_briefing.py \
       backend/fantasy_baseball/smart_lineup_selector.py
  2. venv/Scripts/python -m pytest \
       tests/test_waiver_recommendations_gates.py \
       tests/test_daily_briefing_no_game_contract.py \
       tests/test_waiver_edge.py \
       tests/test_waiver_integration.py \
       tests/test_dashboard_service_waiver_targets.py \
       -q --tb=short
     Expect: 72 passed.

Commit + push:
  3. git add backend/routers/fantasy.py backend/fantasy_baseball/daily_briefing.py \
            backend/fantasy_baseball/smart_lineup_selector.py \
            tests/test_waiver_edge.py tests/test_daily_briefing_no_game_contract.py \
            tests/test_waiver_recommendations_gates.py HANDOFF.md tasks/todo.md
  4. git commit -m "fix(fantasy): Postman P0/P1 regressions — briefing MONITOR routing, MCMC gate, stat-id filter, roster import hoist"
  5. git push origin stable/cbb-prod
  6. Confirm Railway auto-deploys. Wait until /admin/version reflects the new commit SHA.

Post-deploy endpoint validation (capture to postman_collections/responses/ with today's date suffix):
  7. GET /api/fantasy/roster              — expect 200, no ImportError in logs
  8. GET /api/fantasy/briefing/2026-04-22 — expect opponents != "TBD" where probable pitcher known;
                                             no START card paired with a "no game today" alert
  9. GET /api/fantasy/waiver?position=ALL — expect NO "38" key in any player's stats dict;
                                             canonical codes (K_P, K_B, K_9, NSV, QS, OBP, IP, ERA, WHIP)
                                             retained where populated
 10. GET /api/fantasy/waiver/recommendations
                                           — expect no recommendation with mcmc.win_prob_gain < 0
                                             when mcmc_enabled=true
 11. Smoke test (should not regress):
       GET /api/fantasy/lineup/2026-04-22
       GET /admin/audit-tables
       GET /admin/odds-monitor/status
       GET /admin/yahoo/test

Report back:
  - HTTP status + 1-line body summary for each of the four primary endpoints (7–10)
  - Explicit yes/no on each of the four expected behaviors above
  - Any surprise regressions in the smoke-test endpoints

If any live endpoint still fails, provide the exact status code and JSON body.
Do not patch code — report back to Claude Code for diagnosis.
```

### 5.2 For Kimi CLI

No active prompt. K1 / K2 research briefs remain on the backlog; re-open only if Phase 2b greenfield data sourcing becomes active.

---

## 6. Architect Review Queue

Consolidated from prior sessions and this one. Items require judgment, not execution.

### 6.1 Code / Scope Decisions

1. **NSB composite math** — pre-existing `test_composite_z_excludes_z_sb_when_both_populated` failure. Decide: fix the composite_z aggregation (exclude `z_sb` when both SB and CS populated) or update the test expectation. Root cause is in NSB aggregation, not waiver/briefing work.
2. **Unknown Yahoo stat_ids** — currently silently dropped from waiver output. Decide whether to log a warning when seen so `YAHOO_ID_INDEX` can be enriched over time as Yahoo adds new stat types.
3. **Schedule fallback mode flag** — synthetic implied-run estimates from probable-pitcher fallback are currently indistinguishable from sportsbook-derived context. Decide whether to surface an explicit "schedule fallback mode" flag on the lineup API so the UI can label the difference.
4. **`/admin/backfill/player-id-mapping` job model** — decide whether to remain a synchronous long-running endpoint or move onto the existing job queue so Postman/UAT stops depending on request timeouts.
5. **Projection extrapolation caps** — Decisions API surfaces impossible ROS figures (0.00 ERA, 91 HR, 204 RBI). Rate extrapolation is uncapped. Claude to add plausible caps; decide the cap policy per category.
6. **Proxy player pipeline** — 6 of 23 roster carry `is_proxy: true` with hardcoded `z=-0.8`. Decide whether to synthesize proxy projections from rolling stats, route through Yahoo season stats, or accept the placeholder for non-top-tier assets.
7. **Statcast x-stats integration (Phase 4.5a Priority 4)** — wire `statcast_performances` data (xwOBA, barrel%, exit_velocity) into player scoring for luck-adjusted projections. Deferred pending data-quality review. Scope: scoring_engine or decision_engine consumption — TBD.

### 6.2 UI Contract Open Questions (from 2026-04-17 audit)

1. **Q1** — Yahoo API rate limits for scoreboard/transactions/roster calls (affects caching strategy).
2. **Q2–Q3** — Are W, L, SV, HLD, QS available in Yahoo player season stats? (Affects rolling stats source for 4 greenfield categories.)
3. **Q4** — Does the league use FAAB or priority-based waivers? (Affects `ConstraintBudget` contract.)
4. **Q5** — For opponent ROW projections: per-player or pace-based? (Affects P2-5 scope.)
5. **Q7** — What defines the matchup week boundary? (Affects acquisition counting and games-remaining windows.)
6. **Q8** — Acceptable scoreboard response time? (Determines on-demand vs pre-compute strategy.)
7. **Q9** — How should canonical player row handle trade context (same player as sending / receiving)?

**Resolved:** Q6 (America/New_York timezone confirmed), Q10 (HLD is supporting, not scoring), Q11 (K_B is `lower_is_better`).

---

## 7. Reference

### 7.1 UI Contract Readiness Snapshot

From `reports/2026-04-17-ui-specification-contract-audit.md` — **authoritative** mapping between the locked UI spec and backend readiness.

| Status | Count | % |
|--------|-------|---|
| READY | 19 | 17% |
| PARTIAL | 27 | 25% |
| MISSING | 64 | 58% |

**Top 5 Blockers (Phase 5 gate context):**

1. ROW projection pipeline does not exist → blocks 18 fields across Matchup Scoreboard, My Roster, Waiver, Streaming.
2. Rolling stats cover only 9 of 18 categories → missing R, TB from batting; W, L, HRA, K(pit), QS, NSV from pitching.
3. Projections cover only 8 of 18 categories → missing H, K(B), TB, NSB, W, L, HRA, SV/NSV, HLD, QS.
4. Per-player games-remaining-this-week missing → blocks scoreboard games-remaining and waiver filters.
5. Acquisition count not tracked → Yahoo transactions fetched but not counted / week-filtered.

### 7.2 Layer Status (condensed)

| Layer | Status |
|-------|--------|
| L0 — Contracts | **COMPLETE** — stat_contract package + 6 UI contracts |
| L1 — Pure functions | **COMPLETE** — IP pace, category math, ratio risk, delta-to-flip, constraint helpers |
| L2 — Data & adaptation | **CERTIFIED COMPLETE** — regressions only; do not reopen unless production evidence degrades |
| L3 — Derived stats & scoring | **COMPLETE** — L3A/B/D/F all live; 15/18 Z-scores; 4 greenfield cats deferred |
| L3E — Market-implied probability | **DEFERRED** — future enhancement backlog; conflicts with CLAUDE.md OddsAPI hard-stop |
| L4 — Decision engines & simulation | **COMPLETE** — H2H Monte Carlo + MCMC aligned to v2; ROW→Simulation bridge |
| L5 — APIs & presentation | **COMPLETE** — 5 P1 endpoints (scoreboard, budget, roster, move, optimize) + orchestrator + mapper |
| L6 — Frontend | **GATED** — Phase 5 after 4.5b UAT passes |

### 7.3 Kimi Framework Audit Verdicts (2026-04-18)

Claude verified each Kimi claim. Preserved here as historical record — informs decision-engine roadmap.

| # | Claim | Verdict |
|---|-------|---------|
| 1 | `smart_lineup_selector.py` orphaned | **REFUTED** — wired into API routes; architectural gap (not in daily pipeline), not orphaning |
| 2 | `decision_engine.py` primitive scoring (60/30/10 weighted, no category awareness) | **CONFIRMED** |
| 3 | `simulation_engine.py` only 7 stats | **CONFIRMED → REMEDIATED** — now covers 15 of 18 (Phase 4.5a P3); W, L, HR_P, NSV still require upstream data |
| 4 | `statcast_performances` never consumed by scoring/decisions/simulation | **PARTIALLY CONFIRMED** — read only by `data_reliability_engine.py`. Phase 4.5a P4 deferred |
| 5 | `h2h_monte_carlo.py` not in production | **REFUTED** — wired into `scoreboard_orchestrator.py`, called by `GET /api/fantasy/scoreboard` |
| 6 | `category_tracker.py` orphaned | **REFUTED** |
| 7 | `elite_lineup_scorer.py` orphaned | **REFUTED** |
| 8 | Waiver pool empty | **REFUTED** — 25 free agents fetched per cycle |

**Decision pipeline root cause:** sophisticated modules exist but each lives in a separate API route; daily automated pipeline and the API optimize endpoint bypass the full 18-category system. Phase 4.5a addressed the worst of this; remaining consolidation is future work.

### 7.4 Deferred / Archived

- **L3E Market-implied probability** — complete spec exists, requires policy gate before becoming active (conflicts with `CLAUDE.md` OddsAPI hard-stop).
- **Frontend Readiness Brief** — superseded by the 9-phase gated plan; see `DESIGN.md`, `FRONTEND_MIGRATION.md`, `reports/2026-04-17-ui-specification-contract-audit.md` when Phase 5 opens.
- **`HANDOFF_ARCHIVE.md`** — pre-April 17 historical context.
- **Phase 2b greenfield categories** — W, L, HR_P, NSV (pitching) awaiting upstream Yahoo data exposure.

### 7.5 Canonical References

| Document | Purpose |
|----------|---------|
| `CLAUDE.md` | Project orientation, hard stops, provider rules |
| `.claude/rules/workflow.md` | Handoff protocol, task management process |
| `AGENTS.md` | Agent team roles and swimlanes |
| `ORCHESTRATION.md` | Agent delegation rules |
| `reports/2026-04-17-ui-specification-contract-audit.md` | UI contract authority — field-level mapping |
| `reports/2026-04-18-framework-audit.md` | Kimi framework audit (informs Phase 4.5a+) |
| `postman_collections/responses/` | Captured live responses — drives regression fixes |

---

*Last updated: 2026-04-21 — HANDOFF.md restructured for coherence and priority. Prior accreted session deltas consolidated; redundant Gemini prompts merged. Historical detail available in git log.*
