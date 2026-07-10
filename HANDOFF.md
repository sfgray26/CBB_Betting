# HANDOFF.md — Fantasy Baseball Platform (2026-06-25)

> **Date:** 2026-07-10 | **Status:** ✅ ROOT CAUSE FIXED — 100% PROJECTION COVERAGE, TABLE REPAIRED, CONSTRAINT INSTALLED
> **Branch:** `stable/cbb-prod` | **Commit:** a60be4a

---

## Current Mission State

### P0 Surgical Fixes — Route Shadowing + Mutation Auth ✅ DEPLOYED & VALIDATED (2026-07-10 PM)

**Fix 1 — Inline route removal (`d09e2d0`):** All inline `/api/fantasy/*`,
`/api/dashboard*`, and `/api/user/preferences` routes deleted from
`backend/main.py` (~3,270 lines). Routes are served exclusively by
`backend/routers/fantasy.py`. **Exception kept inline:**
`GET /api/fantasy/projections/canonical` has no router equivalent and the
frontend calls it (`frontend/lib/api.ts:378`) — migrate it to the router in a
future task, then delete the inline copy.
Validated in prod (deploy `00ab703a`): `GET /api/fantasy/roster` → 200 and
Railway logs show `backend.routers.fantasy - INFO - ROUTER_EXECUTED`.

**Fix 2 — Auth on mutation endpoints (`a60be4a`):** `POST /api/fantasy/roster/move`
and `POST /api/fantasy/roster/bulk-apply` now require `verify_api_key`.
Validated in prod (deploy `bf4544e1`): unauthenticated POST → 401 on both;
authenticated POST clears auth (move with invalid position → 200 `success:false`,
bulk-apply empty moves → 400) — validated without mutating the live roster.
Test fixtures in `tests/test_roster_move_api.py` and
`tests/test_roster_move_swap_logic.py` got the repo-standard
`dependency_overrides[verify_api_key]` pattern.

**Fix 3 — Frontend build:** No change needed. `matchup-strip.test.tsx` has no
`cat as any`; `tsc --noEmit` and `npm run build` both pass. There is no
`npm test` script in `frontend/package.json`. Frontend `/` and
`/war-room/roster` return 200 in prod.

**Cleanup queue:**
1. Remove TEMPORARY `logger.info("ROUTER_EXECUTED")` marker from
   `get_fantasy_roster` in `backend/routers/fantasy.py` (~line 3886) — it was
   added for Fix 1 validation, which is complete.
2. Pre-existing (NOT caused by these fixes): 6 tests in
   `tests/test_ballpark_factors.py` fail under full-suite runs but pass in
   isolation — order-dependent state pollution, reproduced at HEAD without any
   of these changes. Needs a dedicated debugging task.
3. `test_main_py_briefing_serializer_has_name_field` was removed from
   `tests/test_briefing_category_names.py` — it asserted the now-deleted
   mirrored serializer in main.py.

### Root-Cause Fix — Identity Resolution + player_id_mapping Repair ✅ DEPLOYED & VERIFIED (2026-07-10)

**Mission:** Replace the corruption workaround with a clean table, BDL-primary data,
and coverage monitoring (user spec 2026-07-10). All four validation gates passed.

**Root Cause 1 — Accent normalization bug (the "Cy Young fallback" bug):**
`_normalize_identity_name` used NFKD but never stripped combining marks, so
Yahoo's "Cristopher Sánchez" ≠ DB's "cristopher sanchez" and the resolver
REJECTED correct mapping rows. Sánchez/Nuñez had correct mappings and fresh
scores all along — the code refused to match them. Fixed in
`backend/routers/fantasy.py` + 2 inline copies in `daily_ingestion.py`; the
workaround's name fallback now also matches on normalized_name. Regression
tests: `tests/test_identity_name_normalization.py`.

**Root Cause 2 — Table corruption (bdl_id holding MLBAM values):**
- Class 1: bdl_id == own mlbam_id (403 rows) — merged into clean siblings.
- Class 2: bdl_id == sibling's mlbam_id, own mlbam NULL (Jordan Walker pattern,
  50 rows) — 47 merged, 3 documented skips (below).
- Accented normalized_name rows (persisted by the old bug) re-normalized.
- Max Muncy manually merged (two real players: LAD bdl=142, ATH bdl=241414).
- 152 NULL-bdl rows resolved via live BDL search; 240 unresolvable (minor
  leaguers BDL doesn't carry) left NULL by design.
- CHECK constraint `ck_pim_bdl_not_mlbam` installed — corruption vector blocked at DB level.
- Tool: `backend/scripts/repair_player_id_mapping.py` (dry-run default,
  --apply / --resolve-nulls / --add-constraint / --manual-merge).

**BDL integration (REST per CLAUDE.md; MCP surface shape):**
- `MLBSeasonStats` contract + `BallDontLieClient.get_mlb_season_stats()` for
  `/mlb/v1/season_stats` (true aggregate endpoint, previously unused).
- `backend/services/bdl_mcp_client.py`: `BDLPlayerResolver` with
  search_players / get_player_by_name / get_player_stats / get_projections,
  accent-insensitive matching, ambiguity-safe. BDL has NO forward projection
  endpoint — get_projections packages season-to-date aggregates, source="bdl".

**Coverage monitoring:**
- `backend/services/projection_coverage.py` — shared reconciliation (IL-aware).
- `GET /api/fantasy/projection-coverage` — green(100%)/yellow(90-99)/red(<90).
- Daily job `projection_coverage` (advisory lock **100_044**, 7:30 AM ET) WARNs
  on any roster player missing projections.
- Frontend `ProjectionCoverageWidget` on the dashboard grid (deployed 15:28 UTC).

**Production validation (2026-07-10):**
- BDL live: Sánchez bdl=40 (19 GS, 2.62 ERA, 137 K, 4.99 WAR), Nuñez bdl=164 (86 GP, .247, 33 SB)
- Mappings: Sánchez 469.p.11706→40, Nuñez 469.p.11785→164, Soto→1106, Crochet→555 ✅
- Coverage endpoint: **GREEN 100.0% (16/16 active)**, 6 IL excluded ✅
- Optimizer: Sánchez STARTER 87.03, Soto 99.39, Nuñez bench 66.22, fallbacks: NONE, degraded banner: GONE ✅
- Corruption audit: class1=0, class2=3 (documented skips) ✅

**Known residue (manual triage queue):**
1. `bad_id=547` Derek Hill — sibling row already owned by yahoo_key 469.p.64354 (two Yahoo keys claim one player)
2. `bad_id=554` Jacob Wilson — two real MLB players, needs operator --manual-merge like Muncy
3. `bad_id=84513` Blake Walston — sibling owned by yahoo_key 469.p.62838
4. 240 yahoo-keyed rows with NULL bdl_id — minor leaguers absent from BDL; coverage job will flag any that reach the roster
5. Full-suite pytest has 6 order-dependent flaky failures (ballpark/availability files pass in isolation) — pre-existing, not from this work

**Commits:** `1bde2b2`, `d3dc256`, `3dddcc4` — all deployed to Fantasy-App + observant-benevolence.

---

### Layer 1 Fix — Projection Pipeline Data Corruption ✅ DEPLOYED & VERIFIED (2026-07-07)

**Issue:** 36% of roster players had no projections in production despite workaround code.

**Root Cause of Bug:** The workaround checked `if alt_bdl_id in player_scores_map`, but `player_scores_map`
was pre-built with only the corrupted bdl_ids from `player_key_to_ids`, NOT the alternatives.
This caused the workaround to fail for ALL corrupted players.

**Fix Applied (commit f7bd87e):**
- Changed workaround to query DB directly for alternative bdl_id scores
- Uses same logic as player_scores_map: `as_of_date <= target_date`, `window_days == 14`
- Both fallback paths (mlbam_id and full_name) now query DB directly

**Players Recovered in Production:**
- Dillon Dingler: bdl_id=203 (score=52.0) ✅
- Pete Alonso: bdl_id=1635 (score=68.8) ✅
- Luke Keaschall: bdl_id=654344 (score=84.4) ✅
- Sam Antonacci: bdl_id=4839465 (score=64.3) ✅
- Carson Benge: bdl_id=4839085 (score=78.7) ✅
- Juan Soto: bdl_id=1106 (score=96.6) ✅
- Munetaka Murakami: bdl_id=4667586 (score=5.9) ✅
- Others recovered via fallback paths

**Production Verification:**
- Total roster: 19 players
- Has projections: 19 players
- Fallback: **0 players**
- **Fallback rate: 0.0%** ✅ PRODUCTION VERIFIED
- Optimizer returns 200 with valid lineup ✅

**Files Modified:**
- `backend/routers/fantasy.py`: Fixed workaround to query DB directly (commit f7bd87e)

**Log Output (Workaround Triggering):**
```
INFO - PlayerIDMapping corruption workaround: 469.p.11928 using alt bdl_id=203 (score=52.0) instead of 693307
INFO - PlayerIDMapping corruption workaround: 469.p.10918 using alt bdl_id=1635 (score=68.8) instead of 624413
INFO - PlayerIDMapping corruption workaround: 469.p.63023 using alt bdl_id=654344 (score=84.4) instead of 807712
...
```

---

### Layer 2 Fix — IL Exclusion & Position Eligibility Bugs ✅ DEPLOYED & VERIFIED (2026-07-07)

**Issue:** Optimizer produced dangerous lineups:
- BUG 1: Garrett Crochet (IL/Shoulder, SP/P only) placed in UTIL slot
- BUG 2: IL detection missed "Shoulder" injury note
- BUG 3: Players with no positions classified as hitters

**Root Causes:**
1. IL detection only checked status keywords (IL, DL, OUT, DTD), not injury body parts
2. Position classification allowed players with NO positions into hitter pool
3. No post-optimization safety check to prevent IL players in active slots

**Fix Applied (commit 8e3122d):**
1. **Expanded IL Detection:**
   - Added `INJURY_KEYWORDS` with body parts: Shoulder, Elbow, Knee, Arm, Finger, Wrist, Back, Hip, Hamstring, Quad, Ankle
   - Added injury terms: Strain, Sprain, Fracture, Surgery, Torn, Ruptured, Bruised, Sore, Inflammation
   - Changed pattern from `rf"\b{keyword}\b"` to `r"\b" + keyword + r"\b"` (fixed f-string backslash error)

2. **Fixed Position Classification:**
   - Changed hitter classification to require `bool(p.get("eligible_positions"))`
   - Only players with valid hitting positions go to hitter_data
   - Pitchers (SP, RP, P) explicitly excluded from UTIL consideration

3. **Added Post-Optimization Safety Check:**
   - Rejects lineups with IL players in active slots
   - Returns 500 error with clear message: "Safety check failed: X IL player(s) in active lineup"
   - Lists problematic players with their slots

**Production Verification:**
- Optimizer endpoint: 200 OK ✅
- Message: "5 IL players excluded from active slots" ✅
- Crochet in active lineup: **False** ✅
- IL players in active slots: **0** ✅
- All UTIL players have hitting positions: **True** ✅
- Total active players: 14

**Test Script:** `backend/scripts/test_il_exclusion.py`
```
[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]
Total active players: 14
UTIL players: 0
Crochet in active: False
IL players in active: 0
All UTIL valid: True
```

**Files Modified:**
- `backend/routers/fantasy.py`: Lines 5093-5109 (position classification), Lines 5343-5444 (IL detection), Lines 5295-5321 (safety check)
- `backend/scripts/test_il_exclusion.py`: New test script for validation

**Technical Notes:**
- Fixed nested f-string backslash error: `rf"\b{keyword}\b"` → `r"\b" + keyword + r"\b"`
- Fixed nested f-string in error message: Extracted player list string separately
- IL detection now covers status (IL, DL, OUT, DTD) AND injury notes (Shoulder, Elbow, etc.)

---

## Previous Mission State

---

## Previous Mission State

### DevOps Update — 2026-07-02 Roster Move Post-Write 500

- Production log root cause confirmed for `/api/fantasy/roster/move`: Yahoo lineup write succeeds, then the handler crashes during post-write cache invalidation with `AttributeError: 'YahooFantasyClient' object has no attribute 'clear_cache'`.
- Applied narrow backend guard in `backend/routers/fantasy.py`: cache clear is now best-effort after a successful Yahoo write and cannot convert the move into a 500.
- Local verification: `python -m py_compile backend/routers/fantasy.py` passed.
- Commit pushed to `stable/cbb-prod`: `3914d09 fix: prevent roster move post-write cache crash`.
- Backend Railway service `Fantasy-App` deployed successfully: `d1ad2f69-20a9-4d98-b097-c7521a3d9a2b`.

### DevOps Update — 2026-07-02 Roster Move Frontend Refetch

- Frontend roster page changed to call TanStack Query `refetchQueries({ queryKey: ['roster'] })` immediately after successful roster move instead of delayed invalidation.
- Added frontend debug logs for roster move `onSuccess` and `onError`.
- Local verification: `npm run build` in `frontend/` passed with existing image/workspace-root warnings only.
- Commit pushed to `stable/cbb-prod`: `1b269ac fix: refetch roster after successful move`.
- Frontend Railway service `observant-benevolence` deployed successfully from repo root: `8a2f39ef-c678-4cc9-b638-361e1c42ab4e`.
- Deployment note: two earlier manual deploy attempts failed because the snapshot did not include the top-level `frontend/` directory required by the service `rootDirectory=/frontend`; deploying from repo root resolved it.

### DevOps Update — 2026-07-02 Roster Move Infrastructure Complete ✅

**Status: CRITICAL 2 PASS** — All infrastructure working correctly. Moves fail only due to Yahoo lineup lock (games in progress), which is correct behavior.

**Final Fixes Applied:**
- Backend: Added `clear_all()` method to `YahooAPICache` class (alias for `clear()`)
  - `clear_cache()` was calling nonexistent `clear_all()`, causing AttributeError
  - Exception was caught and logged, so move succeeded but cache stayed stale
  - Commit: `64f64ca`
- Frontend: Added explicit console logging to `onSuccess` callback
  - Logs success message, state changes, invalidation, refetch, banner lifecycle
  - Helps debug any remaining issues

**Console Trace Validation (all ✅):**
1. `onSuccess` handler fires
2. Success banner sets and renders
3. Cache invalidation fires
4. Banner auto-clears after ~3s

**UX Polish Items (future work):**
1. **Error message cleanup:** Currently shows raw Yahoo XML (`<?xml version...`). Should extract user-friendly text like "Move failed: Lineup is locked (game in progress)".
2. **Banner display time:** Currently ~3s may be too fast if user is scrolled down. Consider longer display or dismiss-on-click pattern.

**Retest Recommendation:** Test the success path tomorrow before games start (during lineup-editing window) to validate the full success flow with `success:true`.

### DevOps Update — 2026-06-26 Loop 28 Ownership Refresh

- Commit pushed to `stable/cbb-prod`: `8c438c7 feat: refresh fantasy ownership data`.
- Backend Railway service `Fantasy-App` deployed successfully: `5f769a92-790f-4149-b07e-c8aa75249198`.
- Frontend auto-deploy was skipped because the overall GitHub CI suite failed, but the `frontend` job itself passed. Manual frontend Railway deploy completed successfully: `aaf40987-e355-4fb0-a5f3-e121730aa2f2`.
- Smoke checks:
  - Backend health: `200 {"status":"healthy","database":"connected","scheduler":"running"}`.
  - Frontend `/war-room/streaming`: `200`.
- GitHub CI status for `8c438c7`: workflow failed in backend `test` job at `Lint — bug gate (flake8 F-errors only)`. Public GitHub API exposed job/annotation metadata but not the protected log payload needed to see the exact flake8 lines; requires authenticated/admin log access or a local env with `flake8` installed.
- `CREDENTIALS.md` remains untracked and was intentionally excluded from commit/deploy.

### DevOps Update — 2026-06-30 Loop 28 Redeploy

- Railway auth restored by user; Codex retried deployment.
- Backend syntax validation passed for:
  - `backend/services/daily_ingestion.py`
  - `backend/schemas.py`
  - `backend/routers/fantasy.py`
- Backend Railway service `Fantasy-App` redeployed successfully: `798f77f2-cb6f-4b30-89bd-53cd8ad88a9a`.
- Frontend Railway service `observant-benevolence` redeployed successfully: `1f0fac56-c972-4482-84b3-07ee0b3ae52a`.
- Smoke checks:
  - Backend health: `200 {"status":"healthy","database":"connected","scheduler":"running"}`.
  - Frontend `/war-room/streaming`: `200`.
- Runtime logs checked:
  - Backend scheduler and MLB odds jobs executing successfully; `/health` logged `200`.
  - Frontend Next.js container started and reported ready.

### Completed Work

| Loop | Objective | Status | Key Deliverables |
|------|-----------|--------|------------------|
| **10** | Build Actionable Moves — Add/Drop Execution | ✅ COMPLETE | POST `/api/fantasy/roster/action` endpoint, two-phase commit, automatic rollback |
| **11** | Fix Need-Score Inconsistency | ✅ COMPLETE | Unified need_score service, base/boost separation, transparent Statcast adjustments |
| **12** | Frontend Action Button for Streaming | ✅ COMPLETE | Execute Add button, confirmation modal, Auto-Stream toggle (UI-only) |

---

## Cumulative Status Table

| Component | Status | Notes |
|-----------|--------|-------|
| Yahoo Add/Drop API | ✅ LIVE | `/api/fantasy/roster/action` with validation and rollback |
| Need-Score Service | ✅ LIVE | Unified calculation with transparent Statcast boost |
| Waiver Wire API | ✅ LIVE | Returns base need_score + statcast_boost + adjusted_need_score |
| Waiver Recommendations API | ✅ LIVE | Uses unified need_score service |
| Streaming Action Button | ✅ LIVE | Execute Add button with confirmation modal |
| Auto-Stream Toggle | ⏳ UI-ONLY | Backend execution planned for Loop Iteration 13 |
| Frontend Display | ⏳ PENDING | UI needs update to show statcast_boost field |

---

## Files Created (Last 3 Loops)

| File | Purpose | Lines |
|------|---------|-------|
| `backend/services/yahoo_actions.py` | Two-phase commit for roster actions | ~580 |
| `backend/services/need_score.py` | Unified need-score calculation | ~271 |
| `frontend/components/streaming/action-modal.tsx` | Confirmation modal for roster actions | ~260 |
| `tests/test_yahoo_actions.py` | Yahoo actions test suite | ~530 |
| `tests/test_need_score.py` | Need-score tests | ~200 |

---

## Files Modified (Last 3 Loops)

| File | Changes | Lines |
|------|---------|-------|
| `backend/routers/fantasy.py` | Added `/roster/action` endpoint, refactored `/waiver` and `/waiver/recommendations` | ~330 |
| `backend/schemas.py` | Added `RosterActionRequest/Response`, `statcast_boost`, `adjusted_need_score` | ~50 |
| `frontend/lib/types.ts` | Added `RosterActionRequest/Response` types | +30 |
| `frontend/lib/api.ts` | Added `rosterAction` endpoint function | +5 |
| `frontend/components/streaming/streaming-recommendations.tsx` | Added Execute Add button, Auto-Stream toggle | +60 |
| `frontend/components/streaming/streaming-recommendations.test.tsx` | Added modal and button tests | +140 |

---

## Test Results

### Loop Iteration 10: Yahoo Actions
```
venv/Scripts/python -m pytest tests/test_yahoo_actions.py -v --tb=short
11 passed in 0.65s
```

### Loop Iteration 11: Need-Score Service
```
venv/Scripts/python -m pytest tests/test_need_score.py -v --tb=short
16 passed in 3.07s
```

### Loop Iteration 12: Frontend Components
```
npm test -- streaming-recommendations.test.tsx
9 new tests added (modal + button states)
```

**Total Backend**: 27 tests, 100% pass rate
**Total Frontend**: 9 tests (basic render + action modal)

---

## Deployment Status

**Syntax Validation**: ✅ All files compile
```bash
# Backend
venv/Scripts/python -m py_compile backend/services/yahoo_actions.py
venv/Scripts/python -m py_compile backend/services/need_score.py
venv/Scripts/python -m py_compile backend/routers/fantasy.py
venv/Scripts/python -m py_compile backend/schemas.py

# Frontend
npx tsc --noEmit
```

**Railway Deployment**: ⏳ PENDING
- Backend endpoints need deployment to production
- Frontend changes need deployment
- Smoke tests required post-deploy

---

## End-to-End Validation Checklist

### Backend (After Railway Deploy)
- [ ] `POST /api/fantasy/roster/action` with ADD action succeeds
- [ ] `POST /api/fantasy/roster/action` with invalid data returns structured error
- [ ] Rollback succeeds when ADD succeeds but DROP fails
- [ ] `/api/fantasy/waiver` returns `statcast_boost` field
- [ ] `/api/fantasy/waiver/recommendations` uses unified need_score

### Frontend (After Deploy)
- [ ] Execute Add button appears on streaming recommendations
- [ ] Button is disabled for AVOID recommendations
- [ ] Button is disabled for LOW confidence recommendations
- [ ] Modal opens on button click
- [ ] Modal shows player details, warnings, drop candidate selection
- [ ] Success response shows transaction ID and refreshes data
- [ ] Error response shows structured error message
- [ ] Auto-Stream toggle shows pending actions queue

---

## Known Issues

### P1: Frontend Statcast Boost Display
**Issue**: Waiver Wire UI shows only `need_score`, not `statcast_boost` or `adjusted_need_score`
**Impact**: Users can't see Statcast adjustments
**Fix**: Update `frontend/components/waiver/waiver-wire.tsx` to display all three fields
**Agent**: FrontendAgent → Codex

### P2: Auto-Stream Backend Execution
**Issue**: Auto-Stream toggle is UI-only scaffolding, no backend execution
**Impact**: Users can queue actions but they won't execute automatically
**Fix**: Implement backend execution service in Loop Iteration 13
**Agent**: BackendAgent → Claude Code

### P3: Injury Penalty Applied to Base, Not Adjusted
**Issue**: `apply_injury_penalty()` modifies base need_score, then Statcast boost is added
**Expected**: Penalty should apply to final adjusted score
**Current**: Penalty on base, then boost added (double-counts benefit)
**Status**: Documented, not blocking

---

## Architectural Decisions

### ADR-010: Two-Phase Commit for Roster Actions
**Decision**: Use validate-then-execute pattern with automatic rollback
**Rationale**: Prevents orphaned roster state when partial failure occurs
**Trade-off**: Additional Yahoo API call for validation step

### ADR-011: Unified Need-Score Service
**Decision**: Centralize need-score calculation with transparent components
**Rationale**: Eliminates endpoint inconsistency, provides Statcast transparency
**Trade-off**: Additional service layer indirection

### ADR-012: Multi-Step Modal for Roster Actions
**Decision**: Use confirm → executing → success/error modal flow
**Rationale**: Provides clear feedback and handles all error states gracefully
**Trade-off**: Additional UI complexity vs inline actions

---

## Next Session Priorities

### 1. Railway Deployment (DevOps)
**Agent**: Codex
**Tasks**:
- Push to Railway (both backend and frontend)
- Run smoke tests on `/api/fantasy/roster/action`
- Verify need_score consistency across endpoints
- Test Execute Add button end-to-end

**Smoke Tests**:
```bash
# Backend
curl -X POST https://cbb-edge.railway.app/api/fantasy/roster/action \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $RAILWAY_API_KEY" \
  -d '{"action": "ADD", "add_player_id": "469.p.12345", "position": "BN"}'

curl https://cbb-edge.railway.app/api/fantasy/waiver | jq ".[0].statcast_boost"

# Frontend (manual test)
1. Navigate to Streaming Station
2. Find GOOD + HIGH confidence pitcher
3. Click "Execute Add"
4. Verify modal opens with correct details
5. Confirm and verify success response
```

### 2. Frontend Statcast Display (Frontend)
**Agent**: Codex
**Tasks**:
- Update Waiver Wire to show statcast_boost
- Update War Room to show adjusted_need_score
- Add tooltip: "Base: pure category score. Boost: Statcast adjustment. Adjusted: base + boost"

### 3. Auto-Stream Backend (Backend)
**Agent**: Claude Code
**Tasks**:
- Create backend service for Auto-Stream execution
- Add endpoint to queue/dequeue actions
- Implement scheduler for EXCELLENT + HIGH confidence pitchers
- Add tests for Auto-Stream service

---

## Delegation Bundles

### For Codex (DevOps + Frontend)

```
BUNDLE: Railway Deployment + Frontend Statcast Display

1. DEPLOY TO RAILWAY:
   # Backend
   git add backend/ tests/ loop_log.md
   git commit -m "feat(loops-10-11): yahoo actions + need-score unification"
   git push

   # Frontend (separate deploy if using Next.js deployment)
   git add frontend/ loop_log.md
   git commit -m "feat(loop-12): streaming action button + modal"
   git push

2. SMOKE TESTS (after deploy):
   # Backend smoke test
   curl -X POST https://cbb-edge.railway.app/api/fantasy/roster/action \
     -H "Content-Type: application/json" \
     -H "X-API-Key: $RAILWAY_API_KEY" \
     -d '{"action": "ADD", "add_player_id": "bdl.12345", "position": "P"}'

   # Verify statcast_boost field
   curl https://cbb-edge.railway.app/api/fantasy/waiver | jq ".[0] | {need_score, statcast_boost, adjusted_need_score}"

3. FRONTEND VALIDATION:
   - Navigate to /war-room/streaming
   - Click Execute Add on GOOD + HIGH confidence pitcher
   - Verify modal opens with player details
   - Confirm action and verify success response

4. FRONTEND STATCAST DISPLAY:
   - Edit frontend/components/waiver/waiver-wire.tsx
   - Add column for Statcast Boost
   - Add tooltip explaining base vs adjusted

REPORTING: Back to HANDOFF.md with deployment status and frontend changes made.
```

### For Claude Code (Backend)

```
BUNDLE: Auto-Stream Backend Execution

CONTEXT: Auto-Stream toggle is currently UI-only scaffolding. Users can queue actions but they don't execute automatically.

TASK: Create backend service for Auto-Stream execution.

FILES:
- backend/services/auto_stream.py (NEW) — Service for Auto-Stream execution
- backend/routers/fantasy.py — Add endpoints for queue/dequeue actions

REQUIREMENTS:
1. Queue endpoint: POST /api/fantasy/auto-stream/queue
   - Accepts pitcher bdl_id and drop priority list
   - Validates EXCELLENT + HIGH confidence
   - Queues action for execution

2. Dequeue endpoint: POST /api/fantasy/auto-stream/dequeue
   - Cancels pending action
   - Returns updated queue

3. Scheduler function:
   - Executes queued actions at optimal time (1 hour before game)
   - Uses existing /api/fantasy/roster/action endpoint
   - Logs success/failure

TESTS:
- tests/test_auto_stream.py with queue/dequeue/execution tests

REPORTING: Back to HANDOFF.md with service implementation and test results.
```

---

## Control Plane

### Active Monitor: Need-Score Consistency
**Check**: Base need_score identical for same player across `/waiver` and `/waiver/recommendations`
**Frequency**: Per deployment
**Owner**: QAAgent → Claude Code
**Action**: If mismatch found → regression → need_score.py audit

### Active Monitor: Yahoo Actions Rollback
**Check**: Verify rollback succeeds when ADD succeeds but DROP fails
**Frequency**: Per deployment
**Owner**: QAAgent → Claude Code
**Action**: If rollback fails → regression → yahoo_actions.py audit

### Active Monitor: Auto-Stream Queue Health
**Check**: Verify pending actions don't get stuck in queue
**Frequency**: Hourly (when Auto-Stream is enabled)
**Owner**: QAAgent → Claude Code
**Action**: If queue stuck → alert → auto_stream.py audit

---

## Risk Posture

| Risk | Mitigation | Status |
|------|------------|--------|
| Yahoo API rate limits | Circuit breaker in yahoo_client_resilient.py | ✅ Mitigated |
| Orphaned roster state | Two-phase commit with rollback | ✅ Mitigated |
| Need-score inconsistency | Unified service with transparent components | ✅ Mitigated |
| Statcast boost confusion | Separate field with documentation | ⏳ Partial (UI pending) |
| Injury penalty double-count | Documented, fix planned | ⚠️ Known |
| Auto-Stream queue stuck | Monitor + alert planned | ⚠️ Mitigation pending |
| Frontend network failure | Graceful error handling in modal | ✅ Mitigated |

---

**LAST UPDATED**: 2026-06-25 19:00 EDT
**NEXT REVIEW**: After Railway deployment
**OWNER**: Claude Code (Principal Architect)
