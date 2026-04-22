# Fantasy Baseball Platform — Executive Architectural Roadmap

> **Date:** 2026-04-21 21:30 UTC  
> **Architect:** Claude Code (Master Architect)  
> **Authority:** Replaces all contradictory completion claims across documentation  
> **Plan Reference:** `docs/plan/fantasy-recovery-2026-04/plan.yaml`

---

## ⚠️ CRITICAL PRODUCTION ALERT (2026-04-21 21:30 UTC)

**Waiver endpoints regressed to 503** as of 21:09 UTC probe. A deployment between 19:01-21:09 UTC reintroduced the Yahoo API `out=ownership` bug (K-20 fix reverted). **Waiver wire is completely unavailable.**

**Immediate Action Required:** Gemini CLI to investigate deployment history, identify current commit, redeploy correct version with K-20 fix. See [HANDOFF.md](../HANDOFF.md) §3.1 for emergency recovery steps.

**Positive Changes (v3 audit):** BDL IDs improved (0/23 → 19/23), injury data improved (0/23 → 3/23).

---

## Executive Summary

The fantasy baseball platform has achieved **route availability** (95/100 HTTP checks pass) but suffers **critical data quality failures**. APIs return 200 OK but with 0-33% enrichment on fields required for user decisions. **Phase 5 UI development is blocked** not by defects but by **missing features**: ROW projection pipeline does not exist, rolling stats cover only 9/18 categories, ROS projections cover only 8/18 categories.

**This is not a bug-fix situation. This is a missing-feature situation.**

The path forward requires **60-90 days** of phased development across 10 waves: document cleanup (complete), **emergency waiver recovery (IMMEDIATE)**, deploy uncommitted fixes, diagnose data ingestion failures, build ROW projection pipeline, expand rolling/ROS stats to 18/18 categories, implement games-remaining calculation, validate UI contract readiness (gate Phase 5), then execute frontend P1 pages.

---

## Ground Truth — Where We Really Are

### ✅ What's Working

- **Route Availability:** All P1 endpoints return 200 OK (roster, waiver, lineup, matchup, decisions)
- **Core Architecture:** Contracts, pure functions, Monte Carlo simulation, MCMC decision engine
- **Data Ingestion:** Yahoo OAuth client, league/roster/matchup fetching, schedule syncing
- **Test Coverage:** 309 fantasy tests passing, 0 regressions

### ❌ What's Broken

| Component | Issue | Evidence | Impact |
|-----------|-------|----------|--------|
| **⚠️ Waiver Endpoints** | **503 regression** (as of 21:09 UTC) | Yahoo API `out=ownership` error; K-20 fix reverted between 19:01-21:09 | **Waiver wire completely unavailable** — no free agents, no recommendations |
| **Roster Enrichment** | Rolling windows 100% null, MLBAM IDs 100% null | `rolling_7d/14d/15d/30d = null`, `ros_projection = null`, `row_projection = null`, `mlbam_id = null` for all 23 players | Cannot assess player trends or join to Statcast data |
| **Waiver Intelligence** | Mostly hollow (when working) | `owned_pct = 0.0` for all, `starts_this_week = 0` for all, `category_contributions` only 30% populated | Recommendations based on incomplete data |
| **Recommendation Quality** | Universal-drop bug | All 24 waiver decisions drop "Seiya Suzuki" (v3 audit) | Recommendations not credible |
| **Stat Schema** | Pollution | Batters carry pitcher stats (IP, W), K_P mislabeled | Data not trustworthy for scoring |
| **Briefing Categories** | Legacy v1 names | Uses HR/SB/K/SV instead of HR_B/NSB/K_P/K_B/NSV | Missing 11 v2 canonical categories |

**Recent Positive Changes (v3 audit):**
- BDL player IDs: 0/23 → 19/23 (83% populated)
- Injury data: 0/23 → 3/23 (13% populated)
- Stat ID "38" leak remains fixed (not in any response)

### ⚠️ What's Missing (Not Broken — Never Built)

| Feature | Status | Why It Blocks UI |
|---------|--------|------------------|
| **ROW Projection Pipeline** | MISSING | Matchup Scoreboard requires projected finals (MS-5, MS-6) to show "projected to win/lose". Without ROW, scoreboard is current-state-only. |
| **Rolling Stats (Full Coverage)** | 9/18 categories | My Roster shows recent trends. Missing R, TB, W, L, HRA, K_P, QS, NSV means pitcher decision categories invisible. |
| **ROS Projections (Full Coverage)** | 8/18 categories | Player rows show rest-of-season outlook. Missing H, K(B), TB, NSB, W, L, HRA, SV/NSV, HLD, QS blocks pitcher value assessment. |
| **Games Remaining This Week** | MISSING | Waiver wire filters by games-remaining. Matchup scoreboard shows games-remaining. No per-player calculation exists. |

---

## UI Readiness — The Real Blockers

**Current State (from UI Contract Audit):**
- **READY:** 19/110 fields (17%)
- **PARTIAL:** 27/110 fields (25%)
- **MISSING:** 64/110 fields (58%)

**Top 5 Blockers (Must-Fix for Phase 5):**

1. **B1 (HIGH EFFORT):** ROW projection pipeline does not exist → blocks 18 UI fields across Matchup Scoreboard
2. **B2 (MEDIUM EFFORT):** Rolling stats only 9/18 categories → blocks 12 UI fields for My Roster trends
3. **B3 (MEDIUM EFFORT):** ROS projections only 8/18 categories → blocks 10 UI fields for player value assessment
4. **B4 (LOW EFFORT):** PlayerIDMapping table empty → blocks game context joins (is this player starting today?)
5. **B5 (MEDIUM EFFORT):** Games-remaining-this-week calculation missing → blocks 8 UI fields for scoreboard + waiver filters

**Phase 5 UI cannot start until B1 (ROW pipeline) is built and B2 (rolling stats expansion) is completed.**

---

## Phased Recovery Plan — 10 Waves, 60-90 Days

### Wave 1: Document Restructure ✅ COMPLETE

**Status:** Complete (2026-04-21)
Emergency Waiver Recovery + Deploy Uncommitted Fixes (IMMEDIATE)

**Focus:** Fix waiver 503 regression FIRST, then deploy Apr 21 Postman P0/P1 + UAT v5 fixes

**Critical Priority 0 (Emergency):**
- Investigate Railway deployment between 19:01-21:09 UTC (2026-04-21)
- Identify current commit via `/admin/version`
- Verify K-20 fix presence (Yahoo client must NOT include `out=ownership`)
- Redeploy correct version (commit `8ca2ebe` or later)
- Validate waiver endpoints return 200, not 503

**Priority 1 (After waiver recovery):**
- Deploy Apr 21 uncommitted fixes (roster enrichment, waiver matchup/deficits, MCMC gate, briefing routing)
- Validate with fresh Postman captures

**Expected Outcome:**
- Waiver endpoints restored to 200 (from 503)
- Roster `players_with_stats` improves to >50% (from 0%)
- Waiver `matchup_opponent` changes from "TBD" to real team code
- Waiver `category_deficits` populates (was empty array)

**Owner:** Gemini CLI (deployment investigation and execution only, no code edits)

**Duration:** 1-2 days (URGENT)
- Roster `players_with_stats` improves from 0% to >50%
- Waiver `matchup_opponent` changes from "TBD" to real team code
- Waiver `category_deficits` populates (was empty array)

**Owner:** Gemini CLI (deployment only, no code edits)

**Duration:** 1-2 days

### Wave 3: Data Ingestion Diagnosis (Critical Path)

**Focus:** Diagnose and fix data pipeline failures

**Tasks:**
1. Diagnose PlayerIDMapping ingestion (why is table empty?)
2. Verify player_rolling_stats freshness (how stale?)
3. Fix ownership_pct ingestion (currently all 0.0)
4. Clean up stat schema (K_P mislabeling, batter/pitcher pollution)

**Expected Outcome:**
- PlayerIDMapping >90% populated (enables game context joins)
- Rolling stats <7 days stale
- Ownership % populated from Yahoo API
- Schema pollution eliminated

**Duration:** 5-7 days

### Wave 4: ROW Projection Pipeline (B1 Blocker — HIGH EFFORT)

**Focus:** Build rest-of-week projection calculation

**Tasks:**
1. Design ROW projection algorithm (rolling stats → games-remaining → ROW projections)
2. Implement `backend/services/row_projector.py` module
3. Wire into scoreboard orchestrator
4. Add UI contract fields MS-5, MS-6, MS-7
5. Validate with Postman captures

**Expected Outcome:**
- Scoreboard API returns populated ROW projections (18 categories)
- UI can show "Projected to win/lose X-Y" for matchup finals

**Dependencies:** Requires Wave 5 (rolling stats 18/18) for input data

**Duration:** 10-14 days

### Wave 5: Rolling Stats Expansion (B2 Blocker)

**Focus:** Expand rolling stats from 9/18 to 18/18 categories

**Tasks:**
1. Verify Yahoo API provides missing categories (R, TB, W, L, HRA, K_P, QS, NSV)
2. Expand `backend/services/rolling_stats.py` ingestion
3. Update database schema if needed
4. Expose via roster API (`rolling_7d/14d/30d` fields)
5. Regression test all 18 categories

**Expected Outcome:**
- Rolling stats cover all 18 categories
- My Roster can show trends for pitching decision categories

**Duration:** 7-10 days

**Note:** 4 categories (W, L, HR_P, NSV) may be deferred if Yahoo data unavailable. Document exact Yahoo API gap.

### Wave 6: ROS Projection Expansion (B3 Blocker)

**Focus:** Expand ROS projections from 8/18 to 18/18 categories

**Tasks:**
1. Add missing categories to projection ingestion
2. Add plausibility caps (no more "0.00 ERA ROS", "91 HR ROS")
3. Expose via roster API (`ros_projection` field)
4. Regression test all 18 categories

**Expected Outcome:**
- ROS projections cover all 18 categories
- Player value assessment no longer blind to pitcher categories

**Duration:** 5-7 days

### Wave 7: Games-Remaining Calculation (B5 Blocker)

**Focus:** Implement per-player games-remaining-this-week

**Tasks:**
1. Design calculation over schedule data
2. Implement in `backend/services/schedule.py` or new module
3. Wire into waiver filters (WF-4)
4. Wire into matchup scoreboard (MS-11)

**Expected Outcome:**
- Waiver wire can filter by "players with 4+ games this week"
- Scoreboard shows games-remaining per team

**Duration:** 5-7 days

### Wave 8: UI Readiness Gate (CRITICAL GATE — DO NOT SKIP)

**Focus:** Validate data quality meets UI minimum viable threshold

**Tasks:**
1. Run UI contract audit against live data
2. Verify ROW pipeline operational (non-null MS-5, MS-6, MS-7)
3. Verify rolling stats 18/18 categories
4. Verify ROS projections 18/18 categories
5. Target: READY >50%, MISSING <30%

**Gate Decision:**
- ✅ **PASS:** Phase 5 UI development can start
- ❌ **FAIL:** Identify gaps, fix, re-validate (do not proceed to Phase 5)

**Duration:** 2-3 days

### Wave 9-10: Phase 5 UI Execution (Post-Gate Only)

**Focus:** Build P1 frontend pages

**Wave 9:** Matchup Scoreboard P1  
**Wave 10:** My Roster P1

**Prerequisites:** Wave 8 gate must pass

**Duration:** 14-21 days (combined)

---

## Risk Assessment

### Critical Failure Modes

**1. ROW pipeline reveals Yahoo doesn't provide games-remaining data**
- **Likelihood:** Medium
- **Impact:** Critical (blocks Wave 4)
- **Mitigation:** Pre-research Yahoo scoreboard API structure before Wave 4; design fallback using MLB Stats API schedule

**2. Rolling stats expansion blocked by missing Yahoo upstream data**
- **Likelihood:** Medium (W, L, HR_P, NSV known issue)
- **Impact:** High (blocks 50% rolling stat coverage)
- **Mitigation:** Document exact Yahoo API gap in Wave 5; proceed with available categories; defer 4 categories to Phase 2b

**3. PlayerIDMapping empty because BDL ingestion never built**
- **Likelihood:** High
- **Impact:** High (blocks game context joins)
- **Mitigation:** Wave 3 must audit `daily_ingestion.py` for missing sync job; build if needed

**4. Data quality issues cascade — fixing one reveals 3 more**
- **Likelihood:** High
- **Impact:** Medium (extends timeline)
- **Mitigation:** Wave 3 diagnosis must produce comprehensive data lineage audit before Wave 4 begins

**5. Wave 8 gate fails due to cache invalidation issues**
- **Likelihood:** Medium
- **Impact:** High (blocks Phase 5)
- **Mitigation:** Wave 2 validation must verify enrichment survives 24-hour window

---

## Success Criteria — Definition of Done

### Wave 2 (Deploy Fixes)
- Fresh Postman captures show:
  - Roster `players_with_stats` >0% (was 0%)
  - Waiver `matchup_opponent` != "TBD" (was "TBD")
  - Waiver `category_deficits` length >0 (was 0)

### Wave 3 (Data Ingestion)
- PlayerIDMapping table >90% populated
- player_rolling_stats freshness <7 days stale
- Ownership % populated (not all 0.0)
- Schema pollution eliminated (no K_P=1 with K_9=8.04)

### Wave 4 (ROW Pipeline)
- Scoreboard API response contains non-null MS-5, MS-6, MS-7 fields
- ROW projections for 18 categories (or documented gaps)

### Wave 5 (Rolling Stats)
- Database query shows all 18 categories populated in player_rolling_stats
- Roster API returns populated `rolling_7d/14d/30d` fields

### Wave 6 (ROS Projections)
- Roster API returns all 18 categories in `ros_projection` field
- No impossible values (0.00 ERA, 91 HR, etc.)

### Wave 7 (Games-Remaining)
- Waiver API accepts `min_games_this_week` filter
- Scoreboard API shows games-remaining per team

### Wave 8 (UI Gate)
- UI contract audit shows READY >50%, MISSING <30%
- No critical blockers identified

---

## Timeline — Realistic Expectations

| Milestone | Optimistic | Realistic | Pessimistic |
|-----------|------------|-----------|-------------|
| Wave 2 Deploy | 1 day | 2 days | 3 days |
| Wave 3 Diagnosis | 3 days | 5 days | 7 days |
| Wave 4-7 Feature Dev | 25 days | 35 days | 45 days |
| Wave 8 Gate | 2 days | 3 days | 5 days |
| **Total to UI Gate** | **30 days** | **45 days** | **60 days** |
| Wave 9-10 UI Execution | 14 days | 21 days | 30 days |
| **Total to P1 UI Complete** | **44 days** | **66 days** | **90 days** |

**Start Date:** 2026-04-22 (Wave 2 deploy)  
**Realistic UI Enablement Date:** 2026-06-06 (Wave 8 gate pass)  
**Realistic P1 UI Complete:** 2026-06-27

---

## Next Action — Immediate (Wave 2)

**Owner:** Gemini CLI  
**Task:** Deploy Apr 21 uncommitted fixes to Railway  
**Prompt:** See `HANDOFF.md` §3.1  
**Expected Duration:** 1-2 days  
**Success Criteria:** Roster enrichment >0%, waiver matchup populated, category deficits populated

**After Wave 2:** Proceed to Wave 3 data ingestion diagnosis (Claude Code ownership)

---

## Key Architectural Decisions Pending

See `tasks/architect_review.md` for full decision queue. Critical items:

1. **NSB composite math** — resolve test failure before claiming Phase 2 complete
2. **Unknown stat_ids** — add /admin visibility for manual YAHOO_ID_INDEX enrichment
3. **Schedule fallback mode** — add `game_context.source` field for UI transparency
4. **Projection caps** — define plausible bounds per category (Wave 6 implementation)
5. **Proxy player pipeline** — decide synthesize projections vs accept hardcoded z=-0.8

---

## Documentation Hygiene — Maintain Going Forward

**HANDOFF.md** remains operational brief (≤150 lines):
- §1: Latest session only (what was just done)
- §2: Current state (deploy/phase/defects tables)
- §3: Delegation bundles (one per active agent, verbatim prompts)
- §4: References (links only, no detail)

**Session logs** go to `memory/YYYY-MM-DD.md` (never HANDOFF.md)  
**Architect decisions** go to `tasks/architect_review.md` (never inline in code)  
**Layer status** stays in `docs/ARCHITECTURE.md` (not HANDOFF.md)

**Historical detail** lives in git log (commit messages, PR descriptions) — do not duplicate in operational docs.

---

*Plan ID: `fantasy-recovery-2026-04` | Created: 2026-04-21 | Authority: Replaces all contradictory completion claims*
