# Main.py Strangler-Fig Duplicate Routes Analysis

## Summary

**backend/main.py** (7,943 lines) contains inline route definitions that duplicate routes in the modular router files. The routers are mounted AFTER the inline routes, meaning **the inline routes take precedence** (FastAPI matches in registration order).

This creates strangler-fig debt where:
1. Routes exist in two places (inline in main.py + modular routers)
2. The inline routes are the ones actually being served
3. Changes to router files have no effect until inline routes are removed

## The Three Router Mounts (Lines 624-629)

```python
from backend.routers.edge import router as _edge_router
from backend.routers.fantasy import router as _fantasy_router
from backend.routers.admin import router as _admin_router
app.include_router(_edge_router)
app.include_router(_fantasy_router)
app.include_router(_admin_router)
```

## Duplicate Routes Identified

### Edge Router Duplicates (`backend/routers/edge.py`)

| Route | Line in main.py | Line in edge.py | Notes |
|-------|----------------|-----------------|-------|
| `GET /api/predictions/today` | 1877 | 208 | Core prediction endpoint |
| `GET /api/predictions/today/all` | 1925 | 253 | All predictions variant |
| `GET /api/predictions/game/{game_id}` | 2011 | (check) | Single game lookup |
| `GET /api/predictions/parlays` | 2080 | (check) | Parlay recommendations |
| `GET /api/performance/summary` | 2198 | 496 | Performance stats |
| `GET /api/performance/clv-analysis` | 2210 | 505 | CLV analysis |
| `GET /api/performance/calibration` | 2219 | 514 | Calibration metrics |
| `GET /api/performance/model-accuracy` | 2229 | 524 | Accuracy metrics |
| `GET /api/performance/timeline` | 2246 | 534 | Timeline data |
| `GET /api/performance/financial-metrics` | 2256 | 544 | Financial summary |
| `GET /api/performance/by-team` | 2266 | 554 | Team breakdown |
| `GET /api/performance/source-weights` | 2381 | 648 | Weight analysis |
| `GET /api/performance/alerts` | 2424 | 691 | Performance alerts |
| `POST /api/bets/log` | 2471 | 737 | Log a bet |
| `PUT /api/bets/{bet_id}/outcome` | 2513 | 779 | Update outcome |
| `POST /api/bets/{bet_id}/placed` | 2613 | 865 | Mark as placed |
| `GET /api/bets` | 2677 | 925 | List bets |
| `GET /api/performance/history` | 2823 | 1059 | Historical data |

### Admin Router Duplicates (`backend/routers/admin.py`)

| Route | Line in main.py | Line in admin.py | Notes |
|-------|----------------|------------------|-------|
| `POST /admin/run-analysis` | 2880 | 134 | Trigger analysis |
| `POST /admin/discord/test` | 2918 | 171 | Discord test |
| `POST /admin/discord/test-simple` | 2940 | 193 | Simple Discord test |
| `POST /admin/discord/send-todays-bets` | 2954 | 207 | Send daily bets |
| `POST /admin/recalibrate` | 3044 | 296 | Recalibrate models |
| `GET /admin/recalibration/audit` | 3076 | 326 | Recalibration audit |
| `GET /admin/debug/duplicate-bets` | 3147 | 383 | Debug duplicates |
| `GET /admin/debug/bets-last-24h` | 3216 | 441 | Recent bets debug |
| `POST /admin/cleanup/duplicate-bets` | 3258 | 477 | Cleanup duplicates |
| `POST /admin/force-update-outcomes` | 3348 | 558 | Force outcome update |
| `POST /admin/force-capture-lines` | 3362 | 572 | Force line capture |
| `DELETE /admin/bets/{bet_id}` | 3381 | 583 | Delete bet |
| `DELETE /admin/bets/orphaned/cleanup` | 3397 | 599 | Cleanup orphaned |
| `DELETE /admin/games/{game_id}` | 3421 | 622 | Delete game |
| `POST /admin/alerts/{alert_id}/acknowledge` | 3462 | 663 | Acknowledge alert |
| `GET /admin/scheduler/status` | 3478 | 679 | Scheduler status |
| `GET /admin/ingestion/status` | 3574 | 696 | Ingestion status |
| `GET /admin/portfolio/status` | 3993 | 1128 | Portfolio status |
| `GET /admin/audit-tables` | 710 | 1148 | Table audit |
| `GET /admin/odds-monitor/status` | 4017 | 1345 | Odds monitor |
| `GET /admin/oracle/flagged` | 4039 | 1352 | Oracle flagged |
| `GET /admin/ratings/status` | 4092 | 1400 | Ratings status |

### Fantasy Router Duplicates (`backend/routers/fantasy.py`)

The fantasy router likely duplicates all the `/api/fantasy/*` and Yahoo integration routes defined inline in main.py. These need to be checked individually.

## Temporary/Admin Routers (Should be Removed)

Lines 631-663 include temporary routers that should be removed after their tasks are complete:

| Router | Line | Status | Notes |
|--------|------|--------|-------|
| `_test_router` | 632 | Temporary | Sync job testing |
| `_db_verify_router` | 635 | Temporary | DB verification |
| `_yahoo_debug_router` | 638 | Temporary | Yahoo API debugging |
| `_yahoo_token_router` | 641 | Temporary | Token refresh |
| `_yahoo_parsing_test_router` | 642 | Temporary | Parsing tests |
| `_yahoo_structure_dump_router` | 643 | Temporary | Structure dump |
| `_era_diagnostic_router` | 646 | Task 10 | ERA diagnostics |
| `_validation_audit_router` | 649 | Task 11 | Validation audit |
| `_backfill_ops_whip_router` | 652 | Task 26 | OPS/WHIP backfill |
| `_statcast_diag_router` | 655 | Temporary | Statcast diagnostics |
| `_scoring_diag_router` | 659 | Temporary | Scoring diagnostics |
| `_constraint_migration_router` | 663 | Temporary | Constraint migration |

## Recommended Actions

### Phase 1: Clean Up Temporary Routers (1 hour)
Remove all temporary routers marked above that are no longer needed.

### Phase 2: Cut Over One Router at a Time (1-2 days per router)
For each router (edge, fantasy, admin):
1. Compare inline vs router implementations line-by-line
2. Ensure router version has all features/fixes from inline version
3. Delete inline routes
4. Verify router routes now serve requests
5. Run integration tests

### Phase 3: Verify CORS and Behavior (2-4 hours)
After cutover, verify:
- CORS headers are consistent
- Response schemas match
- Error handling is equivalent
- Performance is acceptable

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Router version missing inline fixes | High | High | Diff comparison before cutover |
| CORS differences | Medium | High | Test CORS headers post-cutover |
| Response schema differences | Medium | Medium | Schema validation tests |
| Performance regression | Low | Medium | Load testing |

## Files Involved

- `backend/main.py` (7,943 lines) - Contains inline routes to remove
- `backend/routers/edge.py` - Edge/prediction router
- `backend/routers/fantasy.py` - Fantasy baseball router  
- `backend/routers/admin.py` - Admin operations router
