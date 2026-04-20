# MLB Platform — Task Tracker
*Updated: 2026-04-16 | Architect: Claude Code | Mission: Layer 3A-C complete. Layer 3B park factor consolidation complete. Layer 3D observability complete. Decision pipeline observability complete. L3F (Decision Output Read Surface) complete. L3E deferred.*

> Canonical source: `HANDOFF.md`
> This file is the execution board for the current phase. If this tracker and HANDOFF disagree, HANDOFF wins.

## Current Session Override — 2026-04-20 Roster Optimize Scoring Repair

Status: IN PROGRESS locally.

Plan:
- [ ] Repair `/api/fantasy/roster/optimize` identity resolution so roster players map to `PlayerIDMapping` by canonical `yahoo_key` before looser fallbacks.
- [ ] Replace flat `50.0` optimize fallbacks with projection-driven fallback scores when `player_scores` rows are missing or stale.
- [ ] Add focused regression coverage for full-key Yahoo mappings and non-uniform fallback scoring.
- [ ] Validate with `py_compile` and targeted `pytest`.

Review:
- Root cause under investigation: the optimize route was querying `PlayerIDMapping.yahoo_id` from the numeric tail only, while the fantasy stack’s authoritative linkage is `yahoo_key` first.
- User-visible failure: starters and bench were collapsing to identical `lineup_score: 50.0` with `"Score 50.0 (default)"` reasoning even for elite roster players.

---

## Operating Rule

Only Layer 3 work is active.

Do not start:
- broad Layer 4 simulation or decision-engine expansion
- frontend feature work
- waiver breadth expansion
- optimizer redesign
- repo-wide park-factor cleanup outside the chosen scoring path
- new Layer 2 roadmap work unless a regression is observed in production

---

## Layer Status Board

| Layer | Name | Status | Rule |
|------|------|--------|------|
| 0 | Immutable Decision Contracts | STABLE | Change only if Layer 3 contract definition forces it |
| 1 | Pure Stateless Intelligence | AVAILABLE | Extend only as required by the scoring objective |
| 2 | Data and Adaptation | CERTIFIED COMPLETE | Regression fixes only |
| 3 | Derived Stats and Scoring | STABLE | L3A-L3F complete; L3E deferred pending policy gate |
| 4 | Decision Engines and Simulation | HOLD | Do not expand until Layer 3 output is stable |
| 5 | APIs and Service Presentation | LIMITED | Only expose validated Layer 3 output |
| 6 | Frontend and UX | HOLD | No new UI work until Layer 3 contract is stable |

---

## Active Objective

### 3A. Authoritative Player Scores Contract
**Spec:** `HANDOFF.md` L3A | **Priority:** highest | **Status:** COMPLETE (2026-04-16)

Goal: define and stabilize one authoritative Layer 3 scoring output before adding any new public API surface.

Target output:
- `player_scores` as the canonical Layer 3 scoring artifact

Primary files:
- `backend/services/scoring_engine.py`
- `backend/services/daily_ingestion.py`
- `backend/models.py`
- `backend/services/decision_engine.py`
- `backend/routers/fantasy.py`
- `backend/admin_scoring_diagnostics.py`

Tasks:

| Task | Owner | Done? |
|------|-------|-------|
| Define the canonical `player_scores` contract: required fields, supported windows, `as_of_date` semantics | Claude | [x] |
| Identify the current authoritative consumer path for `player_scores` | Claude | [x] |
| Verify whether any schema or response model changes are needed for a stable contract | Claude | [x] |
| Write explicit success criteria for what counts as a trustworthy Layer 3 score output | Claude | [x] |

Contract locked:
- Fields: composite_z, score_0_100, confidence, category breakdown (z_hr, z_rbi, z_nsb, z_avg, z_obp for hitters; z_era, z_whip, z_k_per_9 for pitchers)
- Windows: 7, 14, 30 (default 14)
- as_of_date: optional, defaults to latest available
- Auth: verify_api_key required (public API)

---

### 3B. Context Input Stabilization
**Spec:** `HANDOFF.md` L3A | **Priority:** highest | **Status:** COMPLETE (2026-04-16)**

Audit completed. Key findings:
- Authoritative path `compute_league_zscores()` is PURE Z-score scoring - NO park factors, NO weather
- Park factor fragmentation: 5+ hardcoded copies (ballpark_factors.py, mlb_analysis.py, daily_lineup_optimizer.py, two_start_detector.py, weather_ingestion.py)
- scoring_engine.py has DB helper but it's UNUSED by main scoring
- Weather infrastructure exists but is deferred (appropriate for rolling windows)

Tasks:

| Task | Owner | Done? |
|------|-------|-------|
| Define which park-factor source is authoritative for the Layer 3 scoring path | Claude | [x] |
| Audit the chosen scoring path for hardcoded park-factor leakage | Claude | [x] |
| Decide whether weather context is in-scope for the first scoring objective or explicitly deferred | Claude | [x] |
| Limit any park-factor migration to the chosen scoring path only | Claude | [x] |

**Scoped consolidation complete (2026-04-16):**
- Updated `ballpark_factors.py:get_park_factor()` to read from persisted ParkFactor table first
- Preserves PARK_FACTORS constant as fallback
- Preserves neutral 1.0 default for unknown teams
- Added 9 focused tests in `tests/test_ballpark_factors.py`
- Resolution order: DB → hardcoded constant → neutral

---

### 3C. Scoring Output Exposure
**Spec:** depends on 3A and 3B | **Status:** COMPLETE (2026-04-16)

Implemented `GET /api/fantasy/players/{bdl_player_id}/scores` endpoint.

Files modified:
- `backend/routers/fantasy.py` - endpoint implementation with verify_api_key auth
- `backend/schemas.py` - response models (PlayerScoresResponse, PlayerScoreOut, PlayerScoreCategoryBreakdown)
- `tests/test_player_scores_api.py` - 13 test cases including auth validation

Tasks:

| Task | Owner | Done? |
|------|-------|-------|
| Choose whether the first exposure is public API, internal service, or admin-only surface | Claude | [x] |
| If public API is chosen, define the exact response shape and supported query params | Claude | [x] |
| Expose only the authoritative `player_scores` output, not a mixed composite of unrelated artifacts | Claude | [x] |
| Validate that the exposed output matches the underlying scheduled job output | Claude | [x] |

Contract decisions locked:
- Public API (requires verify_api_key)
- Query params: `window_days` (7, 14, 30; default 14), `as_of_date` (optional; defaults to latest)
- Response shape: bdl_player_id, requested_window_days, as_of_date, score (with category breakdown)
- 400 for invalid window_days, 404 for missing scores

---

### 3D. Layer 3 Observability
**Spec:** support for 3A-3C | **Priority:** after first exposed output exists | **Status:** COMPLETE (2026-04-16)

Goal: make the Layer 3 scoring spine inspectable without opening a broad new admin surface.

Implemented endpoint: `GET /admin/diagnose-scoring/layer3-freshness`

Existing diagnostic surface:
- `backend/admin_scoring_diagnostics.py`

Tasks:

| Task | Owner | Done? |
|------|-------|-------|
| Decide whether existing scoring diagnostics are sufficient for the first release | Claude | [x] |
| Add Layer 3 freshness endpoint with verdict, row counts, and audit logs | Claude | [x] |
| Ensure observability reflects actual `player_scores` freshness and fill-rate health | Claude | [x] |

Outcome:
- Endpoint returns freshness verdict (healthy/stale/partial/missing)
- Row counts by window (7/14/30) for player_rolling_stats and player_scores
- Latest audit log entries for rolling_windows and player_scores jobs
- 13 comprehensive tests covering all verdict branches and null cases

**Decision Pipeline Observability (2026-04-16):**

Implemented endpoint: `GET /admin/diagnose-decision/pipeline-freshness`

Tasks:

| Task | Owner | Done? |
|------|-------|-------|
| Add decision pipeline freshness endpoint for P17-P19 stages | Claude | [x] |
| Include verdict, breakdown_by_type, row counts, computed_at timestamps | Claude | [x] |
| Add 8 comprehensive tests covering all verdict branches | Claude | [x] |

Outcome:
- Endpoint returns freshness verdict (healthy/stale/partial/missing)
- Provides breakdown_by_type (lineup/waiver) for DecisionResult and DecisionExplanation tables
- Shows latest computed_at timestamps and schedule expectations (~7 AM / ~9 AM)
- Total test count for admin_scoring_diagnostics.py: 34 tests passing

---

## Recommended Execution Sequence

1. Lock the `player_scores` contract.
2. Identify the authoritative scoring consumer path.
3. Stabilize park-factor authority within that path.
4. Decide whether weather is included now or explicitly deferred.
5. Expose one narrow score output.
6. Add only the minimal observability needed to trust it.

---

## Frozen Backlog

These items remain intentionally deferred:

| Workstream | Status | Why Frozen |
|-----------|--------|------------|
| Broad simulation work | HOLD | Layer 4 not active |
| Full decision-engine expansion | HOLD | Depends on stable Layer 3 output |
| Repo-wide park-factor cleanup | HOLD | Too broad for first Layer 3 objective |
| Weather-driven scoring expansion | HOLD unless explicitly chosen in 3B | Keep first objective narrow |
| Frontend scoring views | HOLD | API/contract must stabilize first |
| Waiver and optimizer redesign | HOLD | Not the current scoring-spine objective |

---

## Immediate Next Decision

**Completed (2026-04-16):**

1. What exact `player_scores` fields are canonical for first exposure?
   - Answer: `composite_z`, `score_0_100`, `confidence`, plus category breakdown (z_hr, z_rbi, z_nsb, z_avg, z_obp for hitters; z_era, z_whip, z_k_per_9 for pitchers)
2. Which windows are in-scope: 7, 14, 30, or a subset?
   - Answer: All three (7, 14, 30) are supported; 14 is default
3. Is weather part of the first scoring objective, or deferred?
   - Answer: DEFERRED - endpoint exposes existing `player_scores` data only; 3B audit confirms weather not used in scoring path
4. Is the first consumer public API or an internal/service contract?
   - Answer: Public API (uses verify_api_key auth) - IMPLEMENTED
5. (3B) Is context authority fragmented or consolidated?
   - Answer: FRAGMENTED - 5+ hardcoded park factor copies; scoring is pure (no park/weather used); scoped fix: ballpark_factors.py → DB-backed
   - **CONSOLIDATION COMPLETE (2026-04-16)**: Updated `ballpark_factors.py:get_park_factor()` to DB-backed read with fallback
6. (3D) Is Layer 3 observability sufficient?
   - Answer: YES - `/admin/diagnose-scoring/layer3-freshness` endpoint live with 13 tests passing
7. Is weather_forecasts populated or deferred?
   - Answer: DEFERRED - table exists but is EMPTY; request-time weather (weather_fetcher.py) serves immediate-decision needs; appropriate for rolling-window scoring

**Next steps:**
- All Layer 3 foundational work (3A-3D) is COMPLETE
- Decision pipeline observability (P17-P19) is COMPLETE
- **L3F (Decision Output Read Surface) is COMPLETE (2026-04-16)**
  - GET /api/fantasy/decisions endpoint live
  - Exposes DecisionResult/DecisionExplanation via read API
  - verify_api_key auth
  - Query params: decision_type (lineup/waiver), as_of_date, limit
  - 13 tests passing (test_decisions_api.py)
- **L3E (Market-Implied Probabilities) is DEFERRED** - preserved as backlog in HANDOFF.md
  - Requires explicit policy gate to activate (conflicts with CLAUDE.md hard-stop rules)
  - Full specification preserved for future reference
- P2 pending: Decide whether any Layer 5 response shape changes are needed after scoring output stabilizes

---

Last Updated: 2026-04-16 (21:00 UTC - L3F Decision Output Read Surface complete; GET /api/fantasy/decisions live with 13 tests passing; L3E remains deferred pending policy gate)