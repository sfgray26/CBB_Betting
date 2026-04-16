# MLB Platform — Task Tracker
*Updated: 2026-04-16 | Architect: Claude Code | Mission: Layer 3B audit complete. Scoring path confirmed pure. Park factor consolidation scoped.*

> Canonical source: `HANDOFF.md`
> This file is the execution board for the current phase. If this tracker and HANDOFF disagree, HANDOFF wins.

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
| 3 | Derived Stats and Scoring | ACTIVE | Only authorized engineering workstream |
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

**Next scoped step:** Update `ballpark_factors.py:get_park_factor()` to read from persisted ParkFactor table - ONE function in ONE file.

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
**Spec:** support for 3A-3C | **Priority:** after first exposed output exists

Goal: make the Layer 3 scoring spine inspectable without opening a broad new admin surface.

Existing diagnostic surface:
- `backend/admin_scoring_diagnostics.py`

Tasks:

| Task | Owner | Done? |
|------|-------|-------|
| Decide whether existing scoring diagnostics are sufficient for the first release | Claude | [ ] |
| If not sufficient, add one minimal health/summary diagnostic for Layer 3 jobs | Claude | [ ] |
| Ensure observability reflects actual `player_scores` freshness and fill-rate health | Claude | [ ] |

Review:
Prefer a minimal diagnostic addition over a broad admin expansion.

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
   - Answer: Deferred - endpoint exposes existing `player_scores` data only; 3B audit confirms weather not used in scoring path
4. Is the first consumer public API or an internal/service contract?
   - Answer: Public API (uses verify_api_key auth) - IMPLEMENTED
5. (3B) Is context authority fragmented or consolidated?
   - Answer: FRAGMENTED - 5+ hardcoded park factor copies; scoring is pure (no park/weather used); scoped fix: ballpark_factors.py → DB-backed

**Next steps:**
- Verify `player_scores` table is being populated by scheduled job in production
- Implement scoped park factor consolidation: `ballpark_factors.py:get_park_factor()` → DB-backed read
- Add observability for Layer 3 job health if needed

---

Last Updated: 2026-04-16 (16:00 UTC - Layer 3B context authority audit complete)