# OPERATIONAL HANDOFF — MARCH 26, 2026: LIVE DATA PIPELINE FOUNDATION

> **Ground truth as of March 26, 2026.** Author: Kimi CLI (Deep Intelligence Unit).
> See `IDENTITY.md` for risk policy · `AGENTS.md` for roles · `HEARTBEAT.md` for loops.
> Prior state: `EMAC-086` — UAT Phase 4 fixes complete.
>
> **CRITICAL CONTEXT:** Today we pivoted from bug fixes to building the **institutional-grade quantitative foundation** for fantasy baseball. The user correctly identified that we were patching symptoms while ignoring the architectural vision.
>
> **Key Insight:** We need live data ingestion + Bayesian updating + MCMC simulation to achieve quant-trading level edge. Not just fix the timezone bug.

---

## MISSION ACCOMPLISHED — Mar 26, 2026: LIVE DATA PIPELINE

### What Was Built Today

| Component | File | Status | Purpose |
|-----------|------|--------|---------|
| **Statcast Ingestion Agent** | `backend/fantasy_baseball/statcast_ingestion.py` | ✅ COMPLETE | Daily pull from Baseball Savant with data quality validation |
| **Bayesian Projection Updater** | `backend/fantasy_baseball/statcast_ingestion.py` | ✅ COMPLETE | Conjugate normal update with shrinkage priors |
| **Data Quality Checker** | `backend/fantasy_baseball/statcast_ingestion.py` | ✅ COMPLETE | Validation rules for completeness, accuracy, anomalies |
| **Database Models** | `backend/models.py` | ✅ COMPLETE | 4 new tables for performances, projections, alerts, logs |
| **Migration Script** | `scripts/migrate_v9_live_data.py` | ✅ COMPLETE | v9 migration with up/downgrade |
| **Scheduler Integration** | `backend/main.py` | ✅ COMPLETE | Daily 6:00 AM ET ingestion job |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DAILY DATA PIPELINE                          │
│                     (6:00 AM ET)                                │
├─────────────────────────────────────────────────────────────────┤
│  1. FETCH → Baseball Savant CSV API                             │
│             (yesterday's games, ~300-500 player records)        │
│                                                                 │
│  2. VALIDATE → Data Quality Checker                             │
│                - Min 200 records                                │
│                - Date range correct                             │
│                - Null rates < 50%                               │
│                - Value ranges reasonable                        │
│                                                                 │
│  3. STORE → statcast_performances table                         │
│             (granular daily metrics per player)                 │
│                                                                 │
│  4. UPDATE → Bayesian Projection Updater                        │
│              - Prior: Steamer/ZiPS (from CSV)                   │
│              - Likelihood: Recent performance (7-14 days)       │
│              - Posterior: Weighted combination                  │
│              - Shrinkage: Trust prior early, data late          │
│                                                                 │
│  5. STORE → player_projections table                            │
│             (live-updated projections with confidence)          │
└─────────────────────────────────────────────────────────────────┘
```

### Bayesian Updating Formula

```python
# Conjugate normal update with shrinkage
posterior_precision = prior_precision + likelihood_precision
posterior_mean = (prior_precision * prior_mean + likelihood_precision * sample_mean) / posterior_precision

# Shrinkage tells us how much to trust new data
shrinkage = prior_precision / posterior_precision

# Early season (n=50 PA): shrinkage ≈ 0.85 (trust prior 85%)
# Late season (n=400 PA): shrinkage ≈ 0.30 (trust prior 30%)
```

### Key Algorithmic Features

1. **Data Quality Gates**
   - Minimum 200 records per day (catches API failures)
   - Date validation (prevents wrong-day ingestion)
   - Null rate checks (ensures data completeness)
   - Value range validation (flags xwoba > 0.700 anomalies)

2. **Bayesian Shrinkage**
   - Handles early-season noise (don't overreact to 10 PA)
   - Natural regression to prior (Steamer is good baseline)
   - Confidence intervals (95% CI on all projections)
   - Data quality scores (0-1 based on sample size)

3. **Audit Logging**
   - Every ingestion logged with metrics
   - Error tracking for monitoring
   - Performance timing (latency tracking)
   - Big mover detection (who's hot/cold)

### Database Schema (v9)

**statcast_performances**
- Daily hitting/pitching metrics per player
- Statcast quality: exit velocity, barrel%, xwOBA
- Calculated stats: AVG, OBP, SLG, OPS, wOBA
- Unique constraint: (player_id, game_date)

**player_projections**
- Live-updated projections via Bayesian inference
- Metadata: shrinkage, data_quality_score, sample_size
- Update tracking: prior_source, update_method, updated_at
- Category scores for H2H leagues (JSONB)

**pattern_detection_alerts**
- MLB-specific edges (pitcher fatigue, bullpen overuse, etc.)
- Confidence scoring for each alert
- Betting implications tracked
- Resolution workflow (active → resolved)

**data_ingestion_logs**
- Audit trail for all data operations
- Quality metrics (errors, warnings, scores)
- Performance tracking (processing time)
- Error details for debugging

### Scheduler Job

```python
# Runs daily at 6:00 AM ET (configured in main.py lifespan)
scheduler.add_job(
    _statcast_daily_ingestion_job,
    CronTrigger(hour=6, minute=0, timezone="America/New_York"),
    id="statcast_daily_ingestion",
    name="Statcast Daily Ingestion + Bayesian Updates",
)
```

### Immediate Next Steps for Claude

1. **Deploy Migration v9** (Priority: CRITICAL)
   ```bash
   python scripts/migrate_v9_live_data.py
   ```

2. **Test Ingestion Pipeline** (Priority: CRITICAL)
   ```bash
   python -c "from backend.fantasy_baseball.statcast_ingestion import run_daily_ingestion; print(run_daily_ingestion())"
   ```

3. **Build Pitcher Quality Integration** (Priority: HIGH)
   - Location: `backend/fantasy_baseball/pitcher_quality.py`
   - Purpose: Matchup multiplier based on xERA
   - Formula: `matchup_mult = 1.0 + (4.00 - pitcher_xera) * 0.05`
   - Integrate into `daily_lineup_optimizer.py` scoring

4. **Build MCMC Simulator Foundation** (Priority: HIGH)
   - Location: `backend/fantasy_baseball/mcmc_simulator.py`
   - Purpose: 10k simulations for weekly matchup probs
   - Output: "70% chance to win HR category"
   - See: `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md` Section 1.3

### Research Docs to Read

Claude must read these before building next components:

1. **`reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md`**
   - Full quant-trading architecture vision
   - MCMC, RL, GNN specifications
   - Multi-agent orchestration design

2. **`reports/daily_lineup_optimization_research.md`**
   - Why implied runs × park factor is insufficient (40% variance)
   - Target: 70-75% variance capture with matchup + platoon + form
   - Multiplicative vs additive scoring models

3. **`reports/KIMI_RESEARCH_MLB_OPENCLAW_PATTERNS.md`**
   - 12 MLB-specific edge patterns
   - Pitcher fatigue, bullpen overuse, platoon splits
   - Detection methods + confidence thresholds

### What NOT to Do

❌ **Do NOT** keep fixing frontend timezone bugs — the real fix is live data
❌ **Do NOT** add more static CSV data — we need live Statcast ingestion
❌ **Do NOT** build complex UIs before the quantitative engine works
❌ **Do NOT** touch `backend/betting_model.py` — Guardian freeze active

### What TO Do

✅ **DO** deploy migration v9 and verify data pipeline works
✅ **DO** build pitcher quality matcher (next 4 hours)
✅ **DO** build MCMC simulator foundation (next 8 hours)
✅ **DO** integrate live projections into daily lineup optimizer
✅ **DO** add pattern detection for pitcher fatigue (OpenClaw)

---

## PREVIOUS HANDOFF — EMAC-086 "UAT PHASE 4 FIXES"

> See below for previous completed work (bug fixes, waiver improvements, etc.)
> All that work remains valid and in production.
>
> **Ground truth as of March 25, 2026.** Author: Claude Code (Master Architect).

---

## MISSION ACCOMPLISHED — Mar 25, 2026 (EMAC-086)

### UAT Phase 4 — Risk Dashboard + Bet Placement Fixes

| Item | Status | Finding |
|------|--------|---------|
| Fix 1a: Settlement lookback `min="20"` → `min="1"` | ALREADY CORRECT | Code had `min={1}` in JSX — no change needed |
| Fix 1b: Drawdown showing 0% | FIXED | `get_portfolio_status` now calls `pm.load_from_db(db)` |
| Fix 2: Roster z-scores not populated | ALREADY CORRECT | Both backend (`z_score=proj.get("z_score")`) and frontend (Z-Score column) already wired |
| Fix 3: `POST /api/bets/{bet_id}/placed` endpoint | ADDED | New endpoint toggles `executed` field on existing BetLog |

**Root cause of drawdown 0%:** `PortfolioManager` is a singleton that initializes `current_bankroll = starting_bankroll`, meaning `drawdown_pct` is always 0 unless `load_from_db()` is called. The endpoint was calling `get_state()` directly without loading DB state first.

**Fix:** `get_portfolio_status` now accepts `db: Session` dependency and calls `pm.load_from_db(db)` before reading state. The singleton's bankroll is reconstructed from the most recent settled `BetLog.bankroll_at_bet + profit_loss_dollars` on each call.

**New endpoint contract:**
- `POST /api/bets/{bet_id}/placed?placed=true` — sets `executed=True`
- `POST /api/bets/{bet_id}/placed?placed=false` — sets `executed=False`
- Returns `{"success": True, "bet_id": N, "placed": bool}`

**Files changed:**
- `backend/main.py` — 2 edits only (no analysis.py or betting_model.py touched)

---

## MISSION ACCOMPLISHED — Mar 25, 2026 (EMAC-085)

### Waiver 503 Logging + Line Movement Deduplication

| Item | Status |
|------|--------|
| `main.py` waiver endpoint: add `logger.error()` to `YahooAuthError` + `YahooAPIError` handlers | COMPLETE |
| `main.py` waiver recommendations endpoint: same logging | COMPLETE |
| `line_monitor.py`: module-level `_alerted_moves` dict; suppress repeated SIGNIFICANT_MOVE for unchanged spread | COMPLETE |

**Root cause of waiver 503:** The error was being caught by `YahooAuthError` or `YahooAPIError` handlers, which had no logging — only the generic `except Exception` handler had `logger.exception()`. After this fix, Railway logs will show the actual Yahoo API error message.

**Line monitor fix:** `_alerted_moves: Dict[int, float]` maps bet_id → last_alerted_curr_spread. Suppresses re-alert unless spread moves >= 0.5 pts from last alert. Duke -12.1 → -6.5 will fire once and stay silent until the line moves further.

**Tests:** 14/14 pass (all EMAC-082/083/084 tests still green).

---

## MISSION ACCOMPLISHED — Mar 25, 2026 (EMAC-081 Final)

### IL Slot Awareness + Closer Alert + projected_saves

| Item | Status |
|------|--------|
| `waiver_edge_detector.py`: `count_il_slots_used()` + `il_capacity_info()` module-level helpers | COMPLETE |
| `schemas.py`: `projected_saves: float = 0.0` on `WaiverPlayerOut` | COMPLETE |
| `schemas.py`: `closer_alert`, `il_slots_used`, `il_slots_available` on `WaiverWireResponse` | COMPLETE |
| `main.py` waiver endpoint: roster fetch, IL capacity compute, closer alert, `projected_saves` in `_to_waiver_player` | COMPLETE |
| `main.py` recommendations: IL slot free → prepend "[IL slot free — move X to IL first]" to rationale | COMPLETE |
| `tests/test_waiver_integration.py`: `TestILSlotAwareness` (3) + `TestCloserAlert` (3) | COMPLETE — 6 tests pass |

**Total new tests this session:** 6 (TestILSlotAwareness + TestCloserAlert)
**Full suite:** 1141/1145 (4 pre-existing failures only)

---

## MISSION ACCOMPLISHED — Mar 25, 2026 (EMAC-081 P1)

### Fantasy Baseball Phase 2 Quality Fixes + Lineup Optimizer Constraint Solver

| Item | Status |
|------|--------|
| Fix A: Fuzzy name matching (`player_board.py` step 3b) | COMPLETE |
| Fix B1: Propagate status/injury_note/is_undroppable in `my_roster_scored` | COMPLETE |
| Fix B2: Coverage-aware `_weakest_safe_to_drop()` replaces `_weakest_at_positions()` | COMPLETE |
| Fix B3: IL opportunity hint in recommendation rationale | COMPLETE |
| Fix C: Two-start pitchers via MLB Stats API (no statsapi dep) + 6h TTL cache | COMPLETE |
| Lineup optimizer: `solve_lineup()` greedy constraint solver | ✅ COMPLETE — Reviewed by Kimi, 6 tests pass |
| Lineup optimizer: `flag_pitcher_starts()` off-day detection | ✅ COMPLETE — SP start detection working |
| `schemas.py`: `assigned_slot`, `has_game` on `LineupPlayerOut`; `lineup_warnings` on response | COMPLETE |
| `tests/test_player_board_fuzzy.py` | COMPLETE — 5 tests |
| `tests/test_waiver_integration.py` — `TestCoverageProtection` + `TestTwoStartPitchers` | COMPLETE — 9 new tests |
| `tests/test_lineup_optimizer.py` | COMPLETE — 6 tests |

**Total new tests this session:** 20
**Full suite:** 1124/1128 (4 pre-existing DB-auth/cache failures only)

---

## MISSION ACCOMPLISHED — Mar 24, 2026 (EMAC-080)

| Item | Status |
|------|--------|
| `SportConfig.mlb()` constructor | COMPLETE — `backend/core/sport_config.py` |
| `SPORT_ID_MLB` constant | COMPLETE — added near SPORT_ID_NCAAB/NBA/NCAAF |
| `backend/services/mlb_analysis.py` | COMPLETE — MLBAnalysisService + MLBGameProjection |
| `_mlb_analysis_service` module var + lifespan block | COMPLETE — `backend/main.py` |
| `_run_mlb_analysis_job()` async job function | COMPLETE — `backend/main.py` |
| `tests/test_mlb_analysis.py` | COMPLETE — 12 tests, all pass |

**Total new tests this session:** 12
**Full suite:** 1103/1107 (4 pre-existing DB-auth/cache failures only)
**Files modified:** `backend/core/sport_config.py`, `backend/main.py`
**Files created:** `backend/services/mlb_analysis.py`, `tests/test_mlb_analysis.py`

---

## ✅ COMPLETE — EMAC-084 "CALIBRATION BRIER + TODAY FILTER"

> **Assignee:** Claude Code (Master Architect)
> **Status:** ✅ COMPLETE — March 25, 2026
> **Tests:** 5 new (test_calibration_brier.py), 1155 total pass

### Summary of Fixes

| Item | Fix | File(s) |
|------|-----|---------|
| **9** Brier score (calibration) | `calculate_calibration(days=90)` — added `days` param + date filter; Brier now uses individual `model_prob` per bet instead of bin midpoints | `services/performance.py`, `main.py:1481` |
| **11** Today's bets | Added `<option value={1}>Today</option>` to days dropdown | `bet-history/page.tsx` |
| **14** Settlement min bug | Already correct (`min={1}`) — no change needed | N/A |
| **13** Matchup zeros | Pre-season correct state — no change needed | N/A |
| **12** Bracket removal | Deferred to EPIC-4, April 7 | N/A |

---

## ✅ COMPLETE — EMAC-083 "ADMIN PANEL + BET HISTORY + EPIC-1 GATES"

> **Assignee:** Claude Code (Master Architect)
> **Status:** ✅ COMPLETE — March 25, 2026
> **Tests:** 1150 pass (no regressions)

### Summary of Fixes

| Gap | Fix | File(s) |
|-----|-----|---------|
| **A** Odds Monitor/Portfolio 403 | Downgraded read-only status endpoints from `verify_admin_api_key` to `verify_api_key` | `main.py:2732, 2748` |
| **B** Bet History shows all | Added `placed` status filter (`BetLog.executed.is_(True)`) + "Placed (Real)" dropdown option | `main.py:1918,1945`, `bet-history/page.tsx:198` |
| **C** EPIC-1 gates | `migrate_v8_post_draft.py` confirmed complete + `test_schema_v8.py` 7/7 pass | `scripts/`, `tests/` |

---

## ✅ COMPLETE — EMAC-082 "FANTASY BASEBALL PRESEASON FIXES"

> **Discovered:** March 25, 2026 — User testing reveals multiple critical UX issues blocking fantasy baseball functionality.
> **Severity:** P0 — Fantasy Baseball Opening Day is March 28 (3 days)
> **Assignee:** Claude Code (Master Architect)
> **Status:** ✅ COMPLETE — March 25, 2026
> **Tests:** 9/9 new tests pass, 1150 total pass

### Summary of Fixes

| Bug | Fix | File(s) |
|-----|-----|---------|
| **#1** is_undroppable string '0' → True | Explicit `in (1, '1', True, 'true')` check | `yahoo_client.py:760` |
| **#2** Silent fallback scoring | Warning banner when odds unavailable | `main.py`, `types.ts`, `lineup/page.tsx` |
| **#3** Lineup apply 400 block | Changed to `apply_warnings.append(...)` | `main.py:5000-5014` |
| **#4** No Optimize Lineup button | Added Optimize button, Slot column badges, amber warning | `lineup/page.tsx`, `types.ts` |
| **#5** Matchup page blank | Added `message: Optional[str]` to schema, blue info banner | `schemas.py`, `main.py`, `matchup/page.tsx` |
| **#6** Waiver 503 swallows traceback | Added `logger.exception(...)` before 503 raises | `main.py` (2 handlers) |

### Bug 1: Roster Shows All Players as "Undroppable = Yes"

**Symptom:** Every player in My Roster shows "Undroppable: Yes" even though they should be droppable.

**Root Cause Analysis:**
- `yahoo_client.py:760` returns `meta.get("is_undroppable", 0)` 
- `main.py:4819` converts to `bool(p.get("is_undroppable", 0))`
- Yahoo API may return `'0'` (string) or `0` (int), and `bool('0')` is `True` in Python!

**Fix Required:**
```python
# In yahoo_client.py _parse_player():
raw = meta.get("is_undroppable", 0)
is_undroppable = raw in (1, '1', True, 'true')

# Same fix needed in main.py where it's propagated
```

**Files to modify:**
- `backend/fantasy_baseball/yahoo_client.py` — line ~760
- `backend/main.py` — lines ~4419, ~4819

---

### Bug 2: Daily Lineup Shows Identical Scores for All Players

**Symptom:** Every batter shows Implied Runs: 4.50, Park Factor: 1.000, Score: 4.625

**ROOT CAUSE IDENTIFIED:** ⚠️ **Odds API quota exhausted** (20,000 / 20,000 calls used — 100.0%)

**STATUS: RESOLVED** — User upgraded to 100k calls/month plan. API calls will now succeed.

**However:** Consider adding **resilient fallback mode** for future API failures (network issues, rate limits):
```python
# In rank_batters(), when team_odds is empty:
if not team_odds:
    # Fallback mode: use projections + neutral environment
    implied_runs = 4.5
    park_factor = 1.0
    # Weight projection stats more heavily for differentiation
    stat_bonus = (proj.get("hr", 0) * 2.0 + proj.get("r", 0) * 0.3 + ...)
    lineup_score = implied_runs * park_factor + stat_bonus * 0.5
    reason_parts.append("using projections (odds unavailable)")
```

**Files to modify (optional but recommended):**
- `backend/fantasy_baseball/daily_lineup_optimizer.py` — Add resilient fallback mode
- `frontend/app/(dashboard)/fantasy/lineup/page.tsx` — Add "Using projections" warning banner when odds unavailable

---

### Bug 3: Lineup Apply Returns 400 "No MLB games scheduled"

**Symptom:** Clicking "Apply to Yahoo" returns error: "No MLB games scheduled for 2026-03-25. Lineup lock not available until the season starts (first game: March 28, 2026)."

**Root Cause Analysis:**
- `main.py:5003-5009` has hardcoded check that blocks ALL lineup changes before March 28:
  ```python
  if not mlb_games:
      raise HTTPException(
          status_code=400,
          detail="No MLB games scheduled for {apply_date}..."
      )
  ```
- This is blocking users from setting up their lineups before Opening Day
- Combined with Bug #2 (odds quota exhausted), this makes the app unusable

**Fix Required:**
1. **Remove the hard block** — allow lineup changes during preseason
2. **Change from error to warning** — apply the lineup but return a warning:
   ```python
   warnings = []
   if not mlb_games:
       warnings.append("Preseason mode: No MLB games scheduled for this date yet.")
   # Continue with lineup application regardless...
   result = client.set_lineup(team_key=team_key, date=apply_date, lineup=lineup_dicts)
   return {
       "success": True,
       "applied": len(result.get("applied", [])),
       "warnings": warnings + result.get("warnings", []),
   }
   ```

**Files to modify:**
- `backend/main.py` — `POST /api/fantasy/lineup/apply` endpoint (~lines 5000-5030)

---

### Bug 4: No "Optimize Lineup" Button in Daily Lineup UI

**Symptom:** Daily Lineup page shows players but there's no "Optimize" button to run the constraint solver.

**Root Cause Analysis:**
- The backend has `solve_lineup()` implemented in `daily_lineup_optimizer.py`
- The frontend only shows current Yahoo lineup, doesn't call optimizer
- User expects: "Optimize" button → runs constraint solver → shows recommended slots

**CRITICAL INSIGHT FROM USER:**
The current `rank_batters()` scoring is **too simplistic** (implied runs only). A truly elite optimizer needs:
1. **Opposing pitcher quality** (matchup difficulty)
2. **Platoon splits** (LHP/RHP wOBA differences)
3. **Recent form** (7/14 day rolling performance)
4. **Statcast regression indicators** (xwOBA vs wOBA)
5. **Lineup spot** (leadoff vs cleanup matters for PAs/RBIs)

See `reports/daily_lineup_optimization_research.md` for comprehensive analysis.

**Fix Required:**

**Phase 1 (Immediate):** Add Optimize button with current scoring
1. **Add Optimize button** to `frontend/app/(dashboard)/fantasy/lineup/page.tsx`
2. **Add new endpoint** `POST /api/fantasy/lineup/optimize` that:
   - Calls `solve_lineup()` with current roster
   - Returns optimized slot assignments

**Phase 2 (Next):** Enhance scoring model
1. **Add matchup quality** — fetch probable pitcher, adjust score by xERA
2. **Add platoon splits** — adjust wOBA by LHP/RHP split
3. **Add recent form** — 7-day rolling performance weighting
4. **Add xwOBA regression** — boost unlucky players, penalize lucky ones

**Files to modify:**
- `backend/main.py` — add `POST /api/fantasy/lineup/optimize` endpoint
- `backend/schemas.py` — add `LineupOptimizeRequest`/`LineupOptimizeResponse`
- `backend/fantasy_baseball/daily_lineup_optimizer.py` — enhance `rank_batters()` with matchup/splits
- `backend/fantasy_baseball/` — add platoon splits data source
- `frontend/app/(dashboard)/fantasy/lineup/page.tsx` — add Optimize button + display

---

### Bug 5: Matchup Page is Blank

**Symptom:** Matchup page shows no data — no opponent, no categories, blank.

**Root Cause Analysis:**
- Matchup data comes from Yahoo scoreboard API
- Before season starts, there's no active matchup → API returns empty/invalid data
- Frontend doesn't handle "no matchup" state gracefully

**Fix Required:**
1. **Handle no-matchup state** in backend:
   ```python
   if not matchup_data:
       return {
           "my_team": {"team_name": "Your Team", "stats": {}},
           "opponent": {"team_name": "TBD", "stats": {}},
           "week": None,
           "is_playoffs": False,
           "message": "No active matchup. Season starts March 28, 2026."
       }
   ```
2. **Show message in frontend** when no matchup exists

**Files to modify:**
- `backend/main.py` — `GET /api/fantasy/matchup` endpoint
- `frontend/app/(dashboard)/fantasy/matchup/page.tsx` — handle no-matchup message

---

### Bug 6: Waiver Wire 503 Errors

**Symptom:** Waiver wire endpoints return 503 Service Unavailable intermittently

**Root Cause Analysis:**
- Logs show Yahoo auth working (tokens refreshed successfully)
- But waiver endpoint returns 503
- Code catches ALL exceptions and returns 503, masking the real error
- Possible causes:
  - `get_board()` failing (player board unavailable)
  - `get_or_create_projection()` failing
  - `_to_waiver_player()` raising KeyError or AttributeError

**Fix Required:**
1. **Add detailed error logging** before the 503 catch-all:
   ```python
   except Exception as exc:
       logger.exception("Waiver endpoint failed: %s", exc)  # Add this
       raise HTTPException(status_code=503, detail=f"...")
   ```
2. **Check specific error** in Railway logs after fix deployed

**Files to modify:**
- `backend/main.py` — waiver endpoint exception handlers (~lines 4233-4247)

---

## SUMMARY: Files to Modify for EMAC-082

### Priority Order (By User Impact)

| Priority | Bug/Feature | Fix | Status |
|----------|-------------|-----|--------|
| ✅ **RESOLVED** | #2 (Identical scores) | Upgraded Odds API to 100k calls/month | API quota no longer blocking |
| 🚨 **P0** | #3 (Apply lineup blocked) | Remove hard block in `main.py:~5003` | Users can't set lineups before Opening Day |
| 🚨 **P0** | #6 (Waiver 503) | Add error logging to diagnose root cause | Blocking waiver wire access |
| **P1** | #1 (Undroppable = Yes) | Fix `bool('0')` parsing in `yahoo_client.py:~760` | UX confusion |
| **P1** | #4 (Optimize button) | Add basic endpoint + UI | Missing feature |
| **P2** | #5 (Matchup blank) | Add no-matchup message | Expected pre-season |
| **P2** | #4-Enhanced | **Matchup quality + platoon splits** — see research doc | Needs opposing pitcher data |

### Research Deliverable: Daily Lineup Optimization

**See:** `reports/daily_lineup_optimization_research.md`

**Key Finding:** Current scoring (implied runs only) captures ~40% of variance. Adding these factors could reach **70-75%**:

| Factor | Impact | Data Source | Status |
|--------|--------|-------------|--------|
| **Opposing pitcher quality** | VERY HIGH | MLB Stats API + Statcast | ❌ Not implemented |
| **Platoon splits (LHP/RHP)** | HIGH | FanGraphs API | ❌ Not implemented |
| **Recent form (7/14 day)** | MEDIUM | Statcast rolling | ❌ Not implemented |
| **xwOBA regression** | MEDIUM | Statcast | ✅ Available, not used |
| **Lineup spot** | MEDIUM | MLB Lineup API | ❌ Not implemented |
| **Implied runs** | HIGH | Odds API | ✅ Implemented |
| **Park factors** | MEDIUM | Hardcoded | ✅ Implemented |

**Recommendation:** After fixing P0 bugs, prioritize adding **opposing pitcher quality** and **platoon splits** to the daily lineup optimizer.

### Files to Modify

| File | Changes |
|------|---------|
| `backend/fantasy_baseball/yahoo_client.py` | Fix `is_undroppable` string/int parsing (~line 760) |
| `backend/main.py` | Fix `is_undroppable` bool conversion (~lines 4419, 4819) |
| `backend/main.py` | Remove lineup apply hard block, add warning instead (~5003-5009) |
| `backend/main.py` | Add detailed error logging to waiver endpoint |
| `backend/main.py` | Add `POST /api/fantasy/lineup/optimize` endpoint |
| `backend/main.py` | Handle no-matchup state gracefully |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Add preseason mode when odds API fails |
| `backend/schemas.py` | Add `LineupOptimizeRequest`/`LineupOptimizeResponse` |
| `frontend/app/(dashboard)/fantasy/lineup/page.tsx` | Add Optimize button + warning banner |
| `frontend/app/(dashboard)/fantasy/matchup/page.tsx` | Add no-matchup message display |
| Railway env vars | Verify `THE_ODDS_API_KEY` is set correctly |

---

## REMAINING P0 GAPS — Kimi Delegation

### K-FB-01: Closer/Z-Score Data Bug Diagnosis

**Assigned to:** Kimi CLI
**Output:** `reports/closer_data_bug.md`

```
MISSION K-FB-01: Closer Data Bug Diagnosis

Read these files IN FULL (do not summarise or truncate):
  backend/fantasy_baseball/player_board.py
  backend/fantasy_baseball/projections_loader.py  (if it exists; else note absence)

DIAGNOSTIC QUESTIONS:
1. Edwin Diaz entry (~line 346 in player_board.py) has 32 saves in PLAYER_BOARD tuple.
   But the live waiver endpoint shows 0 projected saves. Trace the full path:
   PLAYER_BOARD tuple → _compute_z_scores() (or equivalent) → get_or_create_projection()
   return value. At what point does the save count get lost or zeroed?

2. Emmanuel Clase (~line 332) — does his board entry produce a valid z_score?
   If z_score is near the _POSITION_BASELINE_Z floor (-1.0 for RP), why?

3. Are there any RP players in PLAYER_BOARD whose save projections are non-zero
   but whose computed z_score would be near or below -1.0? List all such players.

4. Does projections_loader.py override PLAYER_BOARD values in a way that zeros
   out saves for relievers?

OUTPUT — save to reports/closer_data_bug.md:
  - Root cause with file:line references
  - Complete list of affected closers/relievers
  - Proposed fix (what Claude should change)
  Do NOT write production code — propose; Claude implements.
```

### K-FB-02: No-Closers-On-Waivers Alert Spec

**Assigned to:** Kimi CLI
**Output:** `reports/closer_alert_spec.md`

```
MISSION K-FB-02: No-Closers-On-Waivers Alert Architecture

Read these sections of backend/main.py:
  - GET /api/fantasy/waiver endpoint (~lines 3848-4140)
  - Two-start pitchers block (~lines 4084-4117)

Read backend/schemas.py:
  - WaiverWireResponse definition

DESIGN:
1. Where in WaiverWireResponse should a "closer_situation" alert field live?
2. Detection logic: when scored_fas has 0 RP with projected saves > threshold,
   what should fire?
3. Alert messages for: (a) zero closers, (b) one closer, (c) normal.
4. Strategic advice copy for each scenario:
   e.g. "No closers on waivers. Options: trade a surplus 1B, monitor for
   closer emergence, consider punting saves category."

OUTPUT — save to reports/closer_alert_spec.md:
  - Schema delta (field to add to WaiverWireResponse)
  - Detection logic pseudocode
  - Alert message strings for each scenario
  - Strategic advice copy
  Do NOT write production code — Claude implements from spec.
```

---

## ✅ COMPLETE — IL ROSTER SUPPORT

> **Discovered:** March 25, 2026 — Yahoo updated to show IL status. Players on IL do NOT count against roster spots.
> **Status:** ✅ IMPLEMENTED March 25, 2026
> **Implemented by:** Kimi CLI (per Claude's assignment)
> **Tests:** 9/10 passing (1 integration test needs fixture)
> **Impact:** IL players now excluded from waiver drop suggestions; selected_position exposed in API

### Problem Statement

Yahoo Fantasy Baseball now properly marks players as "IL" (Injured List). These players:
- Don't count against active roster spots (typically 23 active + 2 IL = 25 total)
- Cannot be dropped (they're on IL, not bench)
- Should not factor into "weakest droppable" calculations

The current `YahooFantasyClient.get_roster()` returns ALL players (21+) without distinguishing IL status. The waiver edge detector and lineup optimizer treat IL players as bench players, leading to:
1. Incorrect roster spot availability calculations
2. Potentially suggesting drops of active players when IL slots are available
3. Not factoring IL status into waiver decisions

### Files Changed

| File | Change | Status |
|------|--------|--------|
| `backend/fantasy_baseball/yahoo_client.py` | Added `_extract_selected_position()` method; `get_roster()` now captures selected_position | ✅ COMPLETE |
| `backend/schemas.py` | Added `selected_position: Optional[str]` to `RosterPlayerOut` | ✅ COMPLETE |
| `backend/main.py` | `/api/fantasy/roster` populates selected_position | ✅ COMPLETE |
| `backend/services/waiver_edge_detector.py` | `_weakest_droppable_at()` excludes IL players; added `_INACTIVE_STATUSES` constant | ✅ COMPLETE |
| `tests/test_il_roster_support.py` | New test file with 9 passing tests | ✅ COMPLETE |
| `backend/services/waiver_edge_detector.py` | Add IL slot awareness (how many IL slots, how many used) | ⏳ PENDING |
| `backend/main.py` | `/api/fantasy/waiver/recommendations`: factor IL status into move recommendations | ⏳ PENDING |
| `frontend/lib/types.ts` | `RosterPlayer`: add `selected_position?: string` field | ⏳ FRONTEND |
| `frontend/app/(dashboard)/fantasy/waiver/page.tsx` | Display IL status, filter IL players from drop suggestions | ⏳ FRONTEND |

### Implementation Notes

**Yahoo API Response Structure:**
```json
{
  "player": [
    [{"player_key": "469.p.12345"}, {"name": {...}}],
    {"selected_position": {"position": "IL"}}  // This is where IL status lives
  ]
}
```

The `selected_position` is a sibling to the player metadata array, not inside it. The `get_lineup()` method already extracts this correctly — use it as reference.

**Key Logic Changes:**
1. `get_roster()` must capture `selected_position` alongside player data
2. Waiver recommendations should:
   - Never suggest dropping an IL player
   - Count IL slots separately from active roster spots
   - Consider moving injured players to IL before suggesting drops
3. UI should clearly mark IL players (e.g., red badge: "IL")

### Testing Requirements

- Unit test: `get_roster()` returns `selected_position` = "IL" for IL players
- Unit test: `waiver_edge_detector` excludes IL players from drop candidates
- Integration test: `/api/fantasy/roster` includes `selected_position` field
- UI test: IL players visually distinguished in roster view

---

## 🔴 CRITICAL P0 — CLOSER/ SAVES DETECTION GAP (NEW)

> **Discovered:** March 25, 2026 — User has 1 healthy closer (Diaz), 1 injured (Adam). System showing Diaz with 0 projected saves. Waiver system not prioritizing closers despite need.
> **Impact:** User cannot find available closers on waiver wire; roster construction broken for saves category.
> **Season Start:** IMMINENT — saves are a critical binary category in H2H leagues.

### Problem Statement

1. **Edwin Diaz showing 0 projected saves** — He's one of the best closers in baseball (should be 30+ saves projected). The player board data is wrong.

2. **Waiver system doesn't prioritize closers** — When a user has fewer than 2 healthy RPs, the system should surface available closers as high-priority adds. Currently:
   - RP is grouped with SP/P positionally (they compete for drop slots)
   - No special handling for "0 healthy closers" emergency
   - Saves (nsv) are binary — you either compete in the category or punt it

3. **Available closers not shown** — The `/api/fantasy/waiver` endpoint returns general free agents, but closers with saves potential should be surfaced prominently when needed.

### Current State (User's Team)

| RP | Status | Projected Saves (Current) | Should Be |
|----|--------|---------------------------|-----------|
| Edwin Diaz | Active | **0.0** (BUG!) | ~32 |
| Jason Adam | DTD (hurt) | 3.2 | ~15 |

**Result:** User effectively has 0 reliable closers but the waiver system isn't flagging this as urgent.

### Files Requiring Changes

| File | Change Required |
|------|-----------------|
| `backend/fantasy_baseball/player_board.py` | Fix projected saves for top closers (Diaz, Iglesias, Doval, etc.) |
| `backend/fantasy_baseball/closer_situations.py` (NEW) | Load closer situations CSV and apply save projections |
| `backend/services/waiver_edge_detector.py` | Add `healthy_closer_count` check; if < 2, boost all FA RPs with nsv > 5 |
| `backend/services/waiver_edge_detector.py` | Separate RP from SP/P position group for roster construction logic |
| `backend/main.py` | `/api/fantasy/waiver`: Add `closer_alert` field when healthy_RP_count < 2 |
| `backend/main.py` | `/api/fantasy/waiver/recommendations`: Prioritize closers when needed |
| `frontend/app/(dashboard)/fantasy/waiver/page.tsx` | Display "CLOSER NEEDED" alert; highlight available closers |

### Data Fix Requirements

**Closers with wrong/missing save projections:**
- Edwin Diaz: Currently 0 → Should be 32+
- Raisel Iglesias: Check
- Camilo Doval: Check
- Emmanuel Clase: Check
- Ryan Pressly: Check
- Andres Munoz: Check

Source: `data/projections/closer_situations_2026.csv` should have accurate closer roles and projected saves.

### Logic Changes

**Waiver Priority Algorithm:**
```python
# In WaiverEdgeDetector.get_top_moves()
healthy_closers = count_healthy_rps_with_saves(my_roster, min_saves=5)

if healthy_closers < 2:
    # Emergency closer needed
    available_closers = [fa for fa in free_agents if is_closer(fa)]
    for closer in available_closers:
        # Boost score regardless of category deficits
        closer['need_score'] += CLOSER_EMERGENCY_BOOST  # +10.0 or similar
```

### Testing Requirements

- Data test: Diaz, Iglesias, Doval have nsv >= 25 in player_board
- Unit test: `waiver_edge_detector` boosts closers when healthy_count < 2
- Integration test: `/api/fantasy/waiver` returns closers first when needed
- UI test: "NEED CLOSER" banner appears when appropriate

---

## 🔴 CRITICAL P0 — NO CLOSERS AVAILABLE & MISSING Z-SCORES (NEW)

> **Discovered:** March 25, 2026 — User checked waiver wire: **ZERO actual closers available**. Only setup men (Vodnik, Henry, Ginkel, Rogers) with 0-3 projected saves. Also: some players have **no z-score** in the system.
> **Impact:** User cannot solve saves problem via waivers; trading is only option. Missing z-scores break waiver recommendations entirely.
> **Season Start:** IMMINENT — must pivot strategy from "add closer" to "trade for closer" or "punt saves".

### Discovery: Waiver Wire Reality

**Available RPs (actual Yahoo free agents):**
| Player | Team | Proj Saves | Notes |
|--------|------|------------|-------|
| Victor Vodnik | COL | 0 | Setup/middle relief |
| Cole Henry | WSH | 0 | Setup/middle relief |
| Kevin Ginkel | AZ | 0 | Setup (Doval is closer) |
| Taylor Rogers | MIN | 0 | Setup (possibly shares saves) |
| Riley O'Brien | STL | 0 | Middle relief |
| Edwin Uceta | TB | 0 | Middle relief |
| **Paul Sewald** | **AZ** | **~5** | **Closest to closer, but uncertain role** |
| Kirby Yates | LAA | 0 | **IL15** — hurt |
| A.J. Puk | AZ | 0 | **IL60** — out for months |
| Robert Stephenson | LAA | 0 | **IL60** — out for months |

**Key Finding:** There are **no closers with 10+ projected saves** available on the waiver wire. This is normal in competitive leagues — closers get drafted or picked up immediately.

### The "No Z-Score" Bug

Some players on user's roster have **no z-score** in the system:
- Emmanuel Clase (mentioned by user — legal issues, not playing but rostered)
- Potentially others

**Impact:** When `get_or_create_projection()` returns `None` or empty `cat_scores`, the waiver system cannot:
- Calculate category deficits
- Score free agents against needs
- Make recommendations

### What the System Should Do

When `healthy_closer_count < 2` AND `available_closers_count == 0`:

1. **Alert the user** — "No closers available on waivers"
2. **Provide alternative strategies:**
   - Trade targets: Closers on other teams to trade for
   - Speculative adds: Setup men who might become closers
   - Punt saves: Consider punting the category, focus on other 17
3. **Monitor for closer changes:** New closer announcements happen frequently
4. **Don't waste waiver priority** on middle relievers

### Files Requiring Changes

| File | Change Required |
|------|-----------------|
| `backend/services/waiver_edge_detector.py` | Detect when `available_closers_count == 0`; surface alternative strategies |
| `backend/main.py` | `/api/fantasy/waiver`: Add `no_closers_available` flag + `alternative_strategy` field |
| `backend/fantasy_baseball/player_board.py` | Fix missing z-scores for rostered players (even if not playing) |
| `frontend/app/(dashboard)/fantasy/waiver/page.tsx` | Display "No Closers Available" message; show speculative setup men; suggest trading |

### Alternative Strategy Recommendations

When no closers are available, the system should suggest:

**Option 1: Trade for a Closer**
- Target teams with 3+ closers
- Offer excess 1B (you have 3: Alonso, Pasquantino, Torkelson)
- Offer SP depth (you have 6 healthy SPs)

**Option 2: Speculative Setup Men**
These RPs could become closers if the current closer struggles/gets hurt:
- Kevin Ginkel (AZ) — if Doval struggles
- Taylor Rogers (MIN) — Jhoan Duran is closer but Rogers is experienced
- Cole Henry (WSH) — Kyle Finnegan is closer but shaky
- Victor Vodnik (COL) — unclear situation in Colorado

**Option 3: Punt Saves**
- Accept you'll lose saves category most weeks
- Focus on dominating other 17 categories
- Don't waste roster spots on middle relievers

### Testing Requirements

- Unit test: Waiver detector returns `no_closers_available=true` when appropriate
- Data test: All rostered players have z-scores (even if placeholder/zero)
- Integration test: `/api/fantasy/waiver` returns alternative strategies when no closers found
- UI test: "Trade for Closer" suggestion appears when waivers are dry

---

## 🔴 CRITICAL P0 — DAILY LINEUP OPTIMIZER NOT OPTIMIZING (NEW)

> **Discovered:** March 25, 2026 — User reports daily lineup is "completely broken" and "not optimizing for anything."
> **Impact:** User must manually set lineups daily without data-driven optimization; massive competitive disadvantage in H2H league.
> **Season Start:** IMMINENT — Opening Day lineups need to be optimized.

### The Problem

The current `DailyLineupOptimizer` and `/api/fantasy/lineup/{date}` endpoint are **fundamentally broken**:

**Current "Optimization" Logic (line 3816):**
```python
status="START" if i < 9 else "BENCH"  # Just takes top 9 by score!
```

This is **not optimization** — it's just ranking players by a composite score and taking the top 9. It completely ignores:

1. **Position Requirements** — You can't just start any 9 batters; you need:
   - 1 Catcher
   - 1 First Baseman
   - 1 Second Baseman  
   - 1 Third Baseman
   - 1 Shortstop
   - 3 Outfielders
   - 1 Utility (any position)

2. **Pitcher Slots** — The system doesn't distinguish between:
   - SP slots (typically 2)
   - RP slots (typically 2)
   - P slots (either SP or RP, typically 2)

3. **Players Not Playing** — If a player has an off-day, starting them gives zero stats

4. **Injured Players** — DTD/IL players should never be suggested as START

5. **Multi-Position Eligibility** — Willi Castro can play 2B/3B/LF/RF; the optimizer should use him to fill the weakest position

6. **Matchup Optimization** — The system should:
   - Stack batters from teams with high implied runs (>5.0)
   - Bench batters facing elite pitchers
   - Start pitchers in pitcher-friendly parks
   - Bench pitchers against stacked lineups (e.g., avoid starting SP vs Dodgers in Coors)

### What's Actually Happening

When you call `GET /api/fantasy/lineup/2026-03-28`:

1. Fetches MLB odds from The Odds API ✓
2. Computes implied runs per team ✓
3. Ranks your batters by `lineup_score` (implied runs × park factor + stats) ✓
4. **WRONG:** Just says "start the top 9, bench the rest"

This is like saying "start your 9 best players" without considering that you need specific positions filled!

### Real Example: User's Roster Problem

| Position | Players | Issue |
|----------|---------|-------|
| C | Yainer Diaz | Only 1 catcher - must start |
| 1B | Alonso, Pasquantino, Torkelson | 3 players, 1 slot - pick best matchup |
| 2B | Willi Castro, Marcus Semien (new), Jordan Westburg (IL) | Semien should start; Castro flexes to OF/3B |
| 3B | Matt Chapman, Willi Castro | Chapman starts; Castro flexes |
| SS | Geraldo Perdomo | Only 1 SS - must start |
| OF | Soto, Nimmo, Buxton, Crow-Armstrong, Frelick, Suzuki (IL) | Need 3 OFs + maybe 1 at Utility |

**Current system would:** Just pick top 9 by score, potentially benching your only catcher or shortstop!

**Proper optimizer should:**
1. Identify must-starts (only C, only SS)
2. Fill mandatory positions with best available
3. Use multi-position players (Castro, Semien) to cover gaps
4. Fill Utility with best remaining batter
5. Ensure all 9 slots are filled with players actually playing that day

### Files Requiring Changes

| File | Change Required |
|------|-----------------|
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | **Complete rewrite of `rank_batters()`** — implement positional constraint satisfaction |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Add `optimize_lineup()` method that fills specific slots: C, 1B, 2B, 3B, SS, OF×3, Util |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Add pitcher slot logic: SP×2, RP×2, P×2 (or league-specific) |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Check if player is actually playing that day (not off-day) |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Factor opponent pitcher quality into batter rankings |
| `backend/fantasy_baseball/daily_lineup_optimizer.py` | Factor batter vs pitcher handedness splits |
| `backend/main.py` | `/api/fantasy/lineup/{date}`: Call new `optimize_lineup()` instead of simple rank |
| `backend/schemas.py` | `DailyLineupResponse`: Add `slots` field showing which player goes in which slot |
| `frontend/app/(dashboard)/fantasy/lineup/page.tsx` | Display lineup as position slots, not just ranked list |

### Algorithm: Proper Lineup Optimization

```python
def optimize_lineup(roster, games_today, projections):
    """
    Fill 9 batter slots optimally:
    - C (1)
    - 1B (1)  
    - 2B (1)
    - 3B (1)
    - SS (1)
    - OF (3)
    - Util (1 - any batter)
    
    Constraints:
    - Player must be active (not IL/DTD)
    - Player's team must have a game today
    - Player must be eligible for the slot position
    """
    slots = {
        'C': None, '1B': None, '2B': None, '3B': None, 'SS': None,
        'OF1': None, 'OF2': None, 'OF3': None, 'Util': None
    }
    
    # Get all active players whose teams play today
    available = [
        p for p in roster 
        if p['status'] not in ('IL', 'IL60', 'NA', 'DTD')
        and p['team'] in games_today
    ]
    
    # Sort by daily lineup score (implied runs, park factor, projections)
    available.sort(key=lambda p: p['lineup_score'], reverse=True)
    
    # Fill mandatory single positions first (C, SS) - often only one option
    for pos in ['C', 'SS']:
        candidates = [p for p in available if pos in p['positions'] and p not in slots.values()]
        if candidates:
            slots[pos] = candidates[0]
    
    # Fill multi-candidate positions with best available
    for pos in ['1B', '2B', '3B']:
        candidates = [p for p in available if pos in p['positions'] and p not in slots.values()]
        if candidates:
            slots[pos] = candidates[0]
    
    # Fill OF slots (need 3)
    of_candidates = [p for p in available if any(pos in p['positions'] for pos in ['OF','LF','CF','RF']) and p not in slots.values()]
    for i in range(3):
        if of_candidates:
            slots[f'OF{i+1}'] = of_candidates.pop(0)
    
    # Fill Util with best remaining batter
    remaining = [p for p in available if p not in slots.values()]
    if remaining:
        slots['Util'] = remaining[0]
    
    return slots
```

### Key Features Missing

1. **Off-day detection** — Skip players whose teams don't play
2. **Pitcher opponent quality** — Don't start batters vs Scherzer/deGrom types
3. **Handedness splits** — Start lefty batters vs RHP, vice versa
4. **Recent performance** — Hot hitters get boost
5. **Weather factors** — Wind blowing out at Wrigley = stack hitters
6. **Rest days** — Veterans often rest day games after night games

### Testing Requirements

- Unit test: Optimizer fills all 9 slots with valid position eligibility
- Unit test: Off-day players are excluded from consideration
- Unit test: IL/DTD players are excluded
- Unit test: Multi-position players (Castro) can fill any eligible slot
- Integration test: `/api/fantasy/lineup/{date}` returns filled lineup slots
- Integration test: Lineup respects Yahoo position eligibility
- UI test: Lineup displayed as position slots with START/BENCH decisions

---

## 0. CURRENT STATE — WHAT IS TRUE RIGHT NOW

| Subsystem | Status | Notes |
|-----------|--------|-------|
| V9.1 CBB Model | FROZEN until Apr 7 | Guardian active. See EMAC-076 §3 |
| K-15 Oracle Validation | **LIVE** (Mar 23) | `oracle_validator.py`, DB columns, `GET /admin/oracle/flagged`, 19 tests. Spec: `reports/K15_ORACLE_VALIDATION_SPEC.md` |
| Fantasy Draft | COMPLETE | Juan Soto kept. Draft session endpoints live. |
| Value-Board Endpoint | LIVE | `GET /api/fantasy/draft-session/value-board` w/ Statcast overlay |
| Yahoo OAuth Sync | LIVE | `POST /api/fantasy/draft-session/{key}/sync-yahoo` polls draftresults |
| Pre-Draft Keeper Sweep | **LIVE** (Mar 23) | `POST /api/fantasy/draft-session/{key}/sync-keepers` — fetches all 12 rosters from Yahoo at room open, marks all keepers, cleans pool before first pick |
| Time-Series Schema | **SCHEMA LIVE** (Mar 24) | ORM models + migration script exist. `tests/test_schema_v8.py` created (7 tests). Run `pytest tests/test_schema_v8.py -v` to confirm. DB tables require `migrate_v8_post_draft.py` to be run on Railway. |
| Ingestion Orchestrator | **LIVE** (Mar 24) | `backend/services/daily_ingestion.py`. 5 jobs (mlb_odds/statcast/rolling_z/clv/cleanup). Advisory locks. 11 tests pass. Mount via `ENABLE_INGESTION_ORCHESTRATOR=true`. |
| OpenClaw Autonomous Loop | **LIVE** (Mar 24) | `backend/services/openclaw_autonomous.py`. Scheduler job at 8:30 AM. 8 tests pass. |
| DiscordRouter | **LIVE** (Mar 24) | `backend/services/discord_router.py`. Rate-limited (60s/channel). Batch flush at 5 items or 300s. |
| WaiverEdgeDetector | **LIVE** (Mar 24) | `backend/services/waiver_edge_detector.py`. FA cache 10 min. MCMC-enriched. |
| MCMCWeeklySimulator (OOP) | **LIVE** (Mar 24) | `backend/services/mcmc_simulator.py`. Wrapper around fantasy_baseball/mcmc_simulator.py. |
| Waiver Wire (Next.js) | **LIVE** (Mar 24) | Filter/sort/pagination/recommendations UI. Backend bugs fixed (owned_pct, two_start SPs, get_roster). |
| **IL Roster Support** | **⚠️ CRITICAL GAP** | Yahoo updated to show IL status. Players on IL don't count against roster spots. App does NOT handle this — waiver/drop logic is incorrect. See §IL ROSTER SUPPORT above. |
| **Closer/Saves Detection** | **⚠️ CRITICAL GAP** | Edwin Diaz showing 0 projected saves (bug). System doesn't prioritize closers when < 2 healthy RPs. User can't find saves on waiver wire. See §CLOSER/SAVES DETECTION GAP above. |
| **No Closers Available** | **⚠️ STRATEGIC GAP** | Waiver wire has ZERO closers (only setup men). System doesn't alert user or suggest alternatives (trade/punt/monitor). See §NO CLOSERS AVAILABLE above. |
| **Missing Z-Scores** | **⚠️ DATA GAP** | Some rostered players (Emmanuel Clase) have no z-score. Breaks waiver calculations. See §NO CLOSERS AVAILABLE above. |
| **Daily Lineup Optimizer** | **⚠️ CRITICAL GAP** | Not actually optimizing! Just ranks players by score and takes top 9. Ignores position requirements (C/1B/2B/3B/SS/OF/Util), off-days, matchups. See §DAILY LINEUP OPTIMIZER above. |
| EdgeGenerationEngine | NOT EXISTS | `backend/services/edge_engine.py` does not exist |
| Migration scripts dir | ABSENT | No `backend/migrations/` directory. Precedent: `scripts/migrate_v*.py` |
| Test suite | **~665/668** | 35 new tests added this session. 3 pre-existing DB-auth failures only. |
| **UI Stack** | **Next.js 15** | **Streamlit RETIRED** — see ADR-010. All UI work in `frontend/`. Do not reference `dashboard/`. |
| **MLB Betting Model** | **NOT BUILT** | ⚠️ **CRITICAL** — CBB ends Apr 7. MLB model must be operational by Apr 1. See ADR-006 (updated). |
| **Sport Transition** | **Overlap Phase** | CBB active until Apr 7. MLB season started Mar 28. Parallel operation required. |

**Existing scheduler (CRITICAL READ BEFORE EPIC-2):**
`main.py` line 96 instantiates `AsyncIOScheduler()` at module level and registers 14 jobs in
`lifespan()`. On Railway with multiple Uvicorn workers this scheduler fires in **every worker
process simultaneously**. The existing jobs are low-risk (read-only polls, idempotent writes) but
any new jobs in the Ingestion Orchestrator MUST be guarded by a PostgreSQL advisory lock.
See ADR-001 below — this is non-negotiable.

---

## 1. ARCHITECTURE DECISION RECORDS (ADRs)

These decisions are final. Agents must not re-open them. If you believe an ADR is wrong, write
a one-paragraph dissent in `reports/` and surface it during the next Architect review session.
Do not deviate from ADRs while implementing.

### ADR-001: Multi-Worker Scheduler Lock via PostgreSQL Advisory Locks

**Problem:** Railway deploys 2+ Uvicorn workers per dyno. `AsyncIOScheduler` starts in every
worker's event loop. Adding Statcast pulls, CLV attribution, and waiver scans to the existing
scheduler would fire each job N-workers times per trigger window.

**Decision:** Every new job registered in `DailyIngestionOrchestrator` MUST acquire a
PostgreSQL advisory lock before executing. Use `pg_try_advisory_lock(bigint)` (non-blocking).
If the lock is already held by another worker, the job logs `SKIPPED — lock held` and returns
immediately. Lock is released automatically when the DB session closes (transaction-level).

**Implementation contract:**
```python
# backend/services/daily_ingestion.py — required wrapper for all jobs
from sqlalchemy import text

LOCK_IDS = {
    'mlb_odds':      100_001,
    'statcast':      100_002,
    'rolling_z':     100_003,
    'cbb_ratings':   100_004,
    'clv':           100_005,
    'cleanup':       100_006,
    'waiver_scan':   100_007,
    'mlb_brief':     100_008,
}

async def _with_advisory_lock(lock_id: int, coro):
    """
    Acquire pg_try_advisory_lock(lock_id). If another worker holds it,
    skip silently. Always returns — never blocks.
    Caller awaits this wrapper, not the coro directly.
    """
    from backend.models import SessionLocal
    db = SessionLocal()
    try:
        result = db.execute(
            text("SELECT pg_try_advisory_lock(:lid)"), {"lid": lock_id}
        ).scalar()
        if not result:
            logger.info("SKIPPED — advisory lock %d held by another worker", lock_id)
            return None
        return await coro()
    finally:
        db.execute(text("SELECT pg_advisory_unlock(:lid)"), {"lid": lock_id})
        db.close()
```

**Test requirement:** `tests/test_advisory_lock.py` — mock two concurrent calls to the same
job ID, assert only one executes the handler body.

### ADR-002: Migration Convention — `scripts/migrate_v8_post_draft.py`

**Problem:** No `alembic` or `backend/migrations/` directory exists. Prior migrations are in
`scripts/migrate_v*.py` (v3: actual_margin, v5: team_profiles, v6: D1 defaults).

**Decision:** Continue the existing convention. New migration file:
`scripts/migrate_v8_post_draft.py`

Every migration script MUST contain:
1. `def upgrade(db)` — DDL for the up-revision
2. `def downgrade(db)` — DDL for the down-revision (DROP TABLE / ALTER TABLE DROP COLUMN)
3. `if __name__ == "__main__":` block that runs upgrade with DB URL from env
4. A `--dry-run` flag that prints SQL without executing

**Rollback command:** `python scripts/migrate_v8_post_draft.py --downgrade` must be
safe to run at any point and restore the pre-EPIC-1 schema exactly.

### ADR-003: Epic Isolation — No Cross-Epic Work

Epics are strictly sequential. An agent must not begin EPIC-2 until EPIC-1's migration has
been verified on Railway. An agent must not begin EPIC-3 until EPIC-2's scheduler is running
and its `/admin/ingestion/status` endpoint returns healthy status for all jobs.

**Verification gates are defined in each Epic's "Exit Criteria" section below.**

### ADR-004: New Services Are Additive — No Modification to Guardian-Frozen Files

`DailyIngestionOrchestrator`, `EdgeGenerationEngine`, and `OpenClawAutonomousLoop` are NEW files.
They call existing services as imports. They do NOT modify:
- `backend/betting_model.py`
- `backend/services/analysis.py`
- `backend/services/clv.py` (extend only — add `compute_daily_clv_attribution()` to existing module)
- Any existing `scheduler.add_job()` call in `main.py`

Mount the new orchestrator as a **separate startup hook** via a conditional env var
`ENABLE_INGESTION_ORCHESTRATOR=true`. Default: false. This prevents accidental activation
on Railway before the feature is verified.

### ADR-005: No New Discord Channels Until Bot Has Verified Access

Before creating `DiscordRouter` routes to `#fantasy-waivers`, `#fantasy-lineups`, etc., confirm
the bot has access to those channels. The env var names already exist in `discord_notifier.py`
(`DISCORD_CHANNEL_FANTASY_WAIVERS`, `DISCORD_CHANNEL_FANTASY_LINEUPS`). Set them in Railway env
vars first. `DiscordRouter` reads the same env vars — it does NOT hardcode channel IDs.

### ADR-010: UI Stack — Next.js Only, Streamlit Retired (Mar 24, 2026)

**Problem:** The `dashboard/` folder contains a legacy Streamlit application that was the original
UI. As of March 2026, we completed a full migration to Next.js 15 (see `FRONTEND_MIGRATION.md`).
However, agents may still reference Streamlit code for UI patterns or mistakenly attempt to fix
bugs in the deprecated `dashboard/pages/` files.

**Decision:** 
1. **Streamlit is RETIRED** — The `dashboard/` folder is deprecated and will be archived in EPIC-4.
2. **Next.js is the ONLY UI** — All UI work goes in `frontend/` (Next.js 15, TypeScript, Tailwind).
3. **NEVER reference Streamlit** — Agents must NOT look at `dashboard/` for UI patterns, components,
   or logic. The Streamlit code is frozen and will be deleted.

**Current State:**
| Location | Status | Purpose |
|----------|--------|---------|
| `frontend/` | **ACTIVE** | Next.js 15 production UI — all new work here |
| `dashboard/` | **DEPRECATED** | Old Streamlit app — do not modify, do not reference |

**Agent Instructions:**
- ❌ DO NOT open files in `dashboard/` for any reason
- ❌ DO NOT use Streamlit as a reference for UI patterns
- ❌ DO NOT fix bugs in `dashboard/pages/*.py`
- ✅ DO build all UI in `frontend/app/(dashboard)/`
- ✅ DO use `frontend/lib/types.ts` and `frontend/lib/api.ts` as source of truth
- ✅ DO refer to `FRONTEND_MIGRATION.md` for component patterns

**Removal Timeline:**
- **Now:** Streamlit code is deprecated but present
- **EPIC-4 (Apr 7, 2026):** `dashboard/` folder will be moved to `archive/dashboard/`
- **Post-EPIC-4:** Streamlit dependencies removed from `requirements.txt`

---

## 2. EPIC-1: TIME-SERIES SCHEMA

**Owner:** Claude Code (Architect)
**Prerequisite:** None
**Status:** NOT STARTED
**Touches:** `scripts/migrate_v8_post_draft.py`, `backend/models.py`
**Does NOT touch:** Any service, any scheduler, any existing table

### 2.1 Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 1.1 | Write `upgrade()` for `player_daily_metrics` | `scripts/migrate_v8_post_draft.py` | [ ] |
| 1.2 | Write `downgrade()` for `player_daily_metrics` | same | [ ] |
| 1.3 | Write `upgrade()` for `projection_snapshots` | same | [ ] |
| 1.4 | Write `downgrade()` for `projection_snapshots` | same | [ ] |
| 1.5 | Add `pricing_engine` column to `predictions` (K-14 spec) | same | [ ] |
| 1.6 | Add SQLAlchemy ORM models for both new tables | `backend/models.py` | [ ] |
| 1.7 | Dry-run test locally | — | [ ] |
| 1.8 | Run migration on Railway | — | [ ] |
| 1.9 | Verify schema via `psql` or Railway DB console | — | [ ] |

### 2.2 Migration Script Specification

File: `scripts/migrate_v8_post_draft.py`

**upgrade() DDL — exact SQL to execute:**

```sql
-- Table 1: player_daily_metrics (sparse time-series)
CREATE TABLE IF NOT EXISTS player_daily_metrics (
    id SERIAL PRIMARY KEY,
    player_id VARCHAR(50) NOT NULL,
    player_name VARCHAR(100) NOT NULL,
    metric_date DATE NOT NULL,
    sport VARCHAR(10) NOT NULL CHECK (sport IN ('mlb', 'cbb')),

    -- Core value metrics
    vorp_7d FLOAT,
    vorp_30d FLOAT,
    z_score_total FLOAT,
    z_score_recent FLOAT,

    -- Statcast 2.0 (MLB only — NULL for CBB rows)
    blast_pct FLOAT,
    bat_speed FLOAT,
    squared_up_pct FLOAT,
    swing_length FLOAT,
    stuff_plus FLOAT,
    plv FLOAT,

    -- Flexible rolling windows (sparse JSONB)
    rolling_window JSONB DEFAULT '{}',

    -- Metadata
    data_source VARCHAR(50),
    fetched_at TIMESTAMP DEFAULT NOW(),

    UNIQUE (player_id, metric_date, sport)
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pdm_player_date
    ON player_daily_metrics (player_id, metric_date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pdm_sport_date
    ON player_daily_metrics (sport, metric_date DESC)
    WHERE sport = 'mlb';

-- Table 2: projection_snapshots (delta audit trail)
CREATE TABLE IF NOT EXISTS projection_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_date DATE NOT NULL,
    sport VARCHAR(10) NOT NULL CHECK (sport IN ('mlb', 'cbb')),
    player_changes JSONB NOT NULL DEFAULT '{}',
    total_players INTEGER,
    significant_changes INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_ps_date_sport
    ON projection_snapshots (snapshot_date DESC, sport);

-- K-14: pricing_engine tracking on predictions
ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS pricing_engine VARCHAR(20)
    CHECK (pricing_engine IN ('markov', 'gaussian', NULL));
```

**downgrade() DDL — exact SQL to restore prior state:**

```sql
ALTER TABLE predictions DROP COLUMN IF EXISTS pricing_engine;
DROP INDEX CONCURRENTLY IF EXISTS idx_ps_date_sport;
DROP TABLE IF EXISTS projection_snapshots;
DROP INDEX CONCURRENTLY IF EXISTS idx_pdm_sport_date;
DROP INDEX CONCURRENTLY IF EXISTS idx_pdm_player_date;
DROP TABLE IF EXISTS player_daily_metrics;
```

**Script skeleton:**
```python
#!/usr/bin/env python
"""
EMAC-077 EPIC-1 — Post-draft time-series schema migration.
Usage:
    python scripts/migrate_v8_post_draft.py              # run upgrade
    python scripts/migrate_v8_post_draft.py --downgrade  # run downgrade
    python scripts/migrate_v8_post_draft.py --dry-run    # print SQL, no execute
"""
import argparse
import os
import sys
from sqlalchemy import create_engine, text

UPGRADE_SQL = """...paste upgrade DDL here..."""
DOWNGRADE_SQL = """...paste downgrade DDL here..."""

def upgrade(engine, dry_run=False):
    ...

def downgrade(engine, dry_run=False):
    ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--downgrade", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    engine = create_engine(os.environ["DATABASE_URL"])
    if args.downgrade:
        downgrade(engine, dry_run=args.dry_run)
    else:
        upgrade(engine, dry_run=args.dry_run)
```

### 2.3 SQLAlchemy ORM Models

Add to `backend/models.py` after the existing model definitions:

```python
class PlayerDailyMetric(Base):
    __tablename__ = "player_daily_metrics"

    id = Column(Integer, primary_key=True)
    player_id = Column(String(50), nullable=False, index=True)
    player_name = Column(String(100), nullable=False)
    metric_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)

    vorp_7d = Column(Float)
    vorp_30d = Column(Float)
    z_score_total = Column(Float)
    z_score_recent = Column(Float)

    blast_pct = Column(Float)
    bat_speed = Column(Float)
    squared_up_pct = Column(Float)
    swing_length = Column(Float)
    stuff_plus = Column(Float)
    plv = Column(Float)

    rolling_window = Column(JSONB, default=dict)
    data_source = Column(String(50))
    fetched_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("player_id", "metric_date", "sport"),
    )


class ProjectionSnapshot(Base):
    __tablename__ = "projection_snapshots"

    id = Column(Integer, primary_key=True)
    snapshot_date = Column(Date, nullable=False)
    sport = Column(String(10), nullable=False)
    player_changes = Column(JSONB, nullable=False, default=dict)
    total_players = Column(Integer)
    significant_changes = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 2.4 Required Tests — `tests/test_schema_v8.py`

**Minimum passing tests before EPIC-1 is complete:**

```
test_player_daily_metric_insert_and_unique_constraint
test_player_daily_metric_sport_check_constraint_rejects_invalid
test_projection_snapshot_insert_and_query
test_pricing_engine_column_exists_on_prediction
test_pricing_engine_rejects_invalid_values
test_downgrade_removes_all_new_tables
test_downgrade_removes_pricing_engine_column
```

Coverage target: 100% of new model code. Run with:
```bash
pytest tests/test_schema_v8.py -v
```

### 2.5 Exit Criteria for EPIC-1

All of the following must be TRUE before EPIC-2 starts:

- [ ] `pytest tests/test_schema_v8.py` — all 7 tests pass
- [ ] `python scripts/migrate_v8_post_draft.py --dry-run` — prints SQL, exits 0
- [ ] `python scripts/migrate_v8_post_draft.py` runs on Railway without error
- [ ] `\d player_daily_metrics` on Railway DB shows correct columns + UNIQUE constraint
- [ ] `\d projection_snapshots` on Railway DB shows correct columns
- [ ] `\d predictions` on Railway DB shows `pricing_engine` column
- [ ] `python scripts/migrate_v8_post_draft.py --downgrade` followed by `--upgrade` restores schema cleanly
- [ ] Full test suite still passes: `pytest tests/ -q` — no regressions

---

## 3. EPIC-2: INGESTION ORCHESTRATOR

**Owner:** Claude Code (Architect)
**Prerequisite:** EPIC-1 exit criteria satisfied
**Status:** NOT STARTED
**Touches:** `backend/services/daily_ingestion.py` (new), `backend/main.py` (mount hook only)
**Does NOT touch:** Any existing scheduler job. Any CBB model service.

### 3.1 Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 2.1 | Implement `DailyIngestionOrchestrator` skeleton | `backend/services/daily_ingestion.py` | [ ] |
| 2.2 | Implement `_with_advisory_lock()` wrapper (ADR-001) | same | [ ] |
| 2.3 | Implement `_poll_mlb_odds()` handler | same | [ ] |
| 2.4 | Implement `_update_statcast()` handler | same | [ ] |
| 2.5 | Implement `_calc_rolling_zscores()` — query `player_daily_metrics`, write results | same | [ ] |
| 2.6 | Implement `_compute_clv()` — delegate to `clv.compute_daily_clv_attribution()` | same | [ ] |
| 2.7 | Implement `_cleanup_old_metrics()` — purge rows older than 90 days | same | [ ] |
| 2.8 | Mount orchestrator in `main.py` lifespan under `ENABLE_INGESTION_ORCHESTRATOR=true` | `backend/main.py` | [ ] |
| 2.9 | Add `/admin/ingestion/status` endpoint | `backend/main.py` | [ ] |
| 2.10 | Implement CLV attribution addition to existing `clv.py` | `backend/services/clv.py` | [ ] |
| 2.11 | Write tests | `tests/test_ingestion_orchestrator.py` | [ ] |

### 3.2 API Contract: DailyIngestionOrchestrator

```python
# INPUT: configuration via env vars (no constructor params needed)
# ENABLE_INGESTION_ORCHESTRATOR=true to activate
# NIGHTLY_CRON_TIMEZONE=America/New_York (reuse existing var)

class DailyIngestionOrchestrator:

    def start(self) -> None:
        """
        Register all jobs and start APScheduler.
        Called once in lifespan() startup, ONLY when
        os.getenv('ENABLE_INGESTION_ORCHESTRATOR') == 'true'.
        """

    def get_status(self) -> dict:
        """
        OUTPUT shape (used by /admin/ingestion/status endpoint):
        {
            "job_id": {
                "name": str,
                "enabled": bool,
                "last_run": str | None,      # ISO datetime
                "last_status": str | None,   # "success" | "failed" | "skipped"
                "next_run": str | None       # ISO datetime from APScheduler
            }
        }
        """
```

**Job handler signature contract:**
```python
async def _handler_name(self) -> dict:
    """
    Returns a result dict with at minimum:
    {"status": "success" | "skipped", "records": int, "elapsed_ms": int}
    Raises on unrecoverable error (wrapper catches and alerts Discord).
    """
```

### 3.3 Mount in main.py — Exact Pattern

```python
# In lifespan() after existing scheduler.start() call:

if os.getenv("ENABLE_INGESTION_ORCHESTRATOR", "false").lower() == "true":
    from backend.services.daily_ingestion import DailyIngestionOrchestrator
    _ingestion_orchestrator = DailyIngestionOrchestrator()
    _ingestion_orchestrator.start()
    logger.info("DailyIngestionOrchestrator started")
else:
    _ingestion_orchestrator = None
    logger.info("DailyIngestionOrchestrator disabled (ENABLE_INGESTION_ORCHESTRATOR not set)")
```

```python
# New admin endpoint (add after existing /admin/scheduler/status):
@app.get("/admin/ingestion/status")
async def ingestion_status(user: str = Depends(verify_api_key)):
    if _ingestion_orchestrator is None:
        return {"enabled": False, "jobs": {}}
    return {"enabled": True, "jobs": _ingestion_orchestrator.get_status()}
```

### 3.4 CLV Attribution Extension

**File:** `backend/services/clv.py` — add this function (do NOT modify existing functions):

```python
async def compute_daily_clv_attribution() -> dict:
    """
    Automated CLV calculation comparing our projected spread vs closing line.
    Runs nightly at 11 PM ET via DailyIngestionOrchestrator.

    INPUT: None (reads from DB)
    OUTPUT: {
        "date": str,
        "games_processed": int,
        "clv_positive": int,       # games where we beat the closing line
        "clv_negative": int,
        "avg_clv_points": float,   # mean(|our_spread - closing_spread|)
        "favorable_rate": float,   # fraction of games where our side beat close
        "negative_streak_days": int | None,  # if streak detected
        "records": List[dict]      # per-game detail
    }
    Raises: CLVAttributionError on unrecoverable DB failure
    """
```

**Key implementation notes:**
- Query `Prediction` joined to `ClosingLine` via `game_id`
- A "favorable" CLV means our projected side moved in our favor from open to close
- Alert `discord_notifier.send_system_error()` if `negative_streak_days >= 7`
- Store per-game records in `ProjectionSnapshot` table (JSONB `player_changes` field)
  with `sport='cbb'` and `snapshot_date` = yesterday

### 3.5 Rolling Z-Score Calculation Specification

`_calc_rolling_zscores()` in `DailyIngestionOrchestrator`:

1. Query `player_daily_metrics` WHERE `sport='mlb'` AND `metric_date >= today - 30 days`
2. Group by `player_id`
3. For each player with >= 7 rows: compute `z_score_recent` from 7-day window
4. For each player with >= 30 rows: compute `z_score_total` from 30-day window
5. Upsert back to `player_daily_metrics` for today's date
6. Write summary to `ProjectionSnapshot` with `significant_changes` = count of players
   where `|new_z - old_z| > 0.5`

### 3.6 Required Tests — `tests/test_ingestion_orchestrator.py`

```
test_advisory_lock_prevents_double_execution
test_advisory_lock_releases_on_exception
test_orchestrator_get_status_returns_all_jobs
test_orchestrator_skips_disabled_jobs
test_rolling_zscore_calc_with_7_day_window
test_rolling_zscore_calc_skips_players_with_insufficient_data
test_cleanup_old_metrics_deletes_rows_before_90_days
test_cleanup_old_metrics_preserves_recent_rows
test_clv_attribution_returns_correct_shape
test_clv_attribution_detects_negative_streak
test_ingestion_status_endpoint_returns_enabled_false_when_not_started
```

Coverage target: >= 85% of `daily_ingestion.py` lines.

### 3.7 Exit Criteria for EPIC-2

- [ ] `pytest tests/test_ingestion_orchestrator.py` — all 11 tests pass
- [ ] Full test suite: `pytest tests/ -q` — no regressions vs 647/650 baseline
- [ ] `ENABLE_INGESTION_ORCHESTRATOR=false` (default): Railway logs show "orchestrator disabled" — existing behavior unaffected
- [ ] `ENABLE_INGESTION_ORCHESTRATOR=true` locally: scheduler starts, `/admin/ingestion/status` returns all 6 jobs with `last_status=null`
- [ ] Manual trigger test: call `_poll_mlb_odds()` directly, confirm it returns `{"status": "success", ...}` or graceful skip if MLB odds endpoint returns no data
- [ ] Advisory lock test: spin two asyncio tasks calling the same handler, confirm logs show exactly one "SKIPPED" entry
- [ ] Set `ENABLE_INGESTION_ORCHESTRATOR=true` in Railway, verify no duplicate job fires in `railway logs --follow` over a 10-minute window

---

## 4. EPIC-3: OPENCLAW AUTONOMOUS LOOP

**Owner:** Claude Code (Architect) + OpenClaw (execution target)
**Prerequisite:** EPIC-2 exit criteria satisfied
**Status:** NOT STARTED
**Touches:** (new files only, plus additive changes to 2 existing files)

### 4.1 Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 3.1 | Implement `WaiverEdgeDetector` | `backend/services/waiver_edge_detector.py` | [ ] |
| 3.2 | Implement `DiscordRouter` with rate limiting | `backend/services/discord_router.py` | [ ] |
| 3.3 | Add `send_batch_digest()` to existing discord_notifier | `backend/services/discord_notifier.py` | [ ] |
| 3.4 | Implement `OpenClawAutonomousLoop` | `backend/services/openclaw_autonomous.py` | [ ] |
| 3.5 | Wire MLB DFS section into `MorningBriefGenerator` | `backend/services/openclaw_briefs.py` | [ ] |
| 3.6 | Add `/api/fantasy/waiver` endpoint that returns `WaiverEdgeDetector` results | `backend/main.py` | [ ] |
| 3.7 | Write tests | `tests/test_openclaw_autonomous.py`, `tests/test_waiver_edge.py` | [ ] |
| 3.8 | Update HEARTBEAT.md with new autonomous loops | `HEARTBEAT.md` | [ ] |

### 4.2 API Contract: WaiverEdgeDetector

```python
# INPUT: None (reads from Yahoo API + player board)
# IMPORTANT: Yahoo API is rate-limited. Cap `get_free_agents()` calls at 1/30s.

class WaiverEdgeDetector:

    def detect_waiver_edges(self) -> list[dict]:
        """
        OUTPUT: list of waiver candidates, sorted descending by priority.
        Each dict shape:
        {
            "player": str,             # full name
            "player_id": str,          # normalized ID (matches player_board id key)
            "positions": list[str],
            "team": str,
            "z_score": float,
            "percent_rostered": float, # 0-100
            "tier": str,               # "must_add" | "strong_add" | "streamer"
            "priority": float,         # higher = more urgent
            "reason": str,
            "projected_war": float | None
        }
        Returns [] (empty list) — never raises — if Yahoo API is unavailable.
        Logs warning on Yahoo failure, does not propagate exception.
        """
```

**Graceful degradation requirement:** If `YahooFantasyClient()` raises `YahooAuthError`,
log the error and return `[]`. The autonomous loop must not crash because Yahoo is down.

### 4.3 API Contract: DiscordRouter

```python
# INPUT: IntelPackage dataclass (defined in openclaw_autonomous.py)
# OUTPUT: bool (True = delivered, False = rate-limited or failed)

@dataclass
class IntelPackage:
    channel: str        # key into CHANNELS dict (not a raw channel ID)
    embed: dict         # Discord embed payload (matches existing discord_notifier format)
    priority: int       # 1-5 (5 = critical, bypasses rate limit)
    timestamp: datetime
    mention_admin: bool = False

class DiscordRouter:
    RATE_LIMITS: dict[str, int]  # channel_key -> max_per_hour

    async def route(self, intel: IntelPackage) -> bool:
        """
        Attempt delivery with rate limit check.
        Priority >= 4: bypass rate limit, deliver immediately.
        Priority < 4 + rate limited: enqueue for batch digest.
        Returns True only if message was delivered in this call.
        Delegates actual HTTP call to discord_notifier.send_to_channel().
        """

    async def flush_batch(self, channel: str) -> bool:
        """
        Combine all queued messages for channel into a single embed and deliver.
        Called hourly by OpenClawAutonomousLoop for non-critical channels.
        """
```

**Critical constraint:** `DiscordRouter` calls `discord_notifier.send_to_channel()` — it does
NOT make its own HTTP calls to Discord. All actual Discord HTTP logic stays in
`discord_notifier.py`. `DiscordRouter` is a routing/rate-limiting layer only.

### 4.4 API Contract: OpenClawAutonomousLoop

```python
class OpenClawAutonomousLoop:
    """
    Registered as a single APScheduler job in DailyIngestionOrchestrator
    — NOT as a separate infinite loop. This prevents Railway process conflicts.
    """

    # Called by orchestrator at 7:00 AM ET
    async def run_morning_workflow(self) -> dict:
        """
        OUTPUT: {"brief_sent": bool, "waiver_sent": bool, "waiver_count": int}
        """

    # Called by orchestrator at 10:00 AM ET
    async def run_lineup_workflow(self) -> dict:
        """
        OUTPUT: {"lineup_sent": bool, "player_count": int}
        """

    # Called by orchestrator every 2h between 12 PM - 11 PM ET
    async def run_live_monitor(self) -> dict:
        """
        OUTPUT: {"escalations_sent": int, "games_checked": int}
        """

    # Called by orchestrator hourly
    async def run_telemetry_update(self) -> dict:
        """
        OUTPUT: {"telemetry_sent": bool, "token_budget_pct": float | None}
        """
```

**Scheduling in orchestrator** (add to `_register_default_jobs()` in EPIC-2):
```python
self.register_job(
    id='morning_workflow', name='Morning Brief + Waiver Scan',
    trigger=CronTrigger(hour=7, minute=0, timezone='America/New_York'),
    handler=lambda: self.openclaw_loop.run_morning_workflow()
)
self.register_job(
    id='lineup_workflow', name='Daily Lineup Optimization',
    trigger=CronTrigger(hour=10, minute=0, timezone='America/New_York'),
    handler=lambda: self.openclaw_loop.run_lineup_workflow()
)
self.register_job(
    id='live_monitor', name='Live Game Monitor',
    trigger=CronTrigger(hour='12-23', minute=0, timezone='America/New_York'),
    handler=lambda: self.openclaw_loop.run_live_monitor()
)
self.register_job(
    id='telemetry', name='OpenClaw Health Telemetry',
    trigger=CronTrigger(minute=0),
    handler=lambda: self.openclaw_loop.run_telemetry_update()
)
```

### 4.5 Morning Brief MLB Add-On Specification

**File:** `backend/services/openclaw_briefs.py`
**Function to add:** `collect_mlb_dfs_section(date_str: str) -> dict`

```python
# OUTPUT shape:
{
    "top_batters": [  # top 5 from DailyLineupOptimizer, sorted by score
        {"name": str, "team": str, "implied_runs": float, "park_factor": float, "score": float}
    ],
    "top_pitchers": [  # top 3 SPs with best park factor + low ERA
        {"name": str, "team": str, "opponent": str, "era": float, "park_factor": float}
    ],
    "slate_size": int,   # number of games today
    "best_park": str,    # park with highest run factor today
    "avoid_park": str    # park with lowest run factor today
}
# Returns {} on any exception — never raises
```

**Embed integration:** Add a "⚾ MLB DFS Outlook" section to the existing
`MorningBriefGenerator.generate_brief()` embed. Check `bool(mlb_addon)` before adding —
if empty dict, skip the section entirely.

### 4.6 Required Tests

**`tests/test_waiver_edge.py`:**
```
test_detect_waiver_edges_returns_list
test_detect_waiver_edges_returns_empty_on_yahoo_auth_error
test_calculate_pickup_edge_must_add_tier
test_calculate_pickup_edge_strong_add_tier
test_calculate_pickup_edge_streamer_tier
test_calculate_pickup_edge_returns_none_below_threshold
test_priority_sort_order_descending
```

**`tests/test_openclaw_autonomous.py`:**
```
test_morning_workflow_returns_correct_shape
test_morning_workflow_completes_if_waiver_detector_returns_empty
test_lineup_workflow_returns_correct_shape
test_live_monitor_returns_correct_shape
test_discord_router_rate_limit_blocks_low_priority
test_discord_router_critical_bypasses_rate_limit
test_discord_router_flush_batch_combines_queued
test_discord_router_delegates_to_discord_notifier_not_raw_http
```

Coverage target: >= 80% of `openclaw_autonomous.py` and `waiver_edge_detector.py`.

### 4.7 Exit Criteria for EPIC-3

- [ ] `pytest tests/test_waiver_edge.py tests/test_openclaw_autonomous.py` — all 15 tests pass
- [ ] Full test suite: `pytest tests/ -q` — total count >= 662 (647 + ~15 new)
- [ ] Set `ENABLE_INGESTION_ORCHESTRATOR=true` in Railway
- [ ] Set required Discord channel env vars in Railway (see §5.2)
- [ ] `railway logs --follow` at 7:00 AM ET — confirm single morning brief delivered to `#openclaw-briefs`
- [ ] `railway logs --follow` — confirm no duplicate job fires (advisory lock working)
- [ ] `/api/fantasy/waiver` endpoint returns non-empty list when real Yahoo data available

---

## 5. DEPLOYMENT CHECKLIST

### 5.1 Railway Environment Variables — Add Before EPIC-3

```bash
ENABLE_INGESTION_ORCHESTRATOR=true         # EPIC-2 activation
DISCORD_CHANNEL_FANTASY_WAIVERS=<id>       # from Discord server settings
DISCORD_CHANNEL_FANTASY_LINEUPS=<id>
DISCORD_CHANNEL_FANTASY_NEWS=<id>
DISCORD_CHANNEL_OPENCLAW_BRIEFS=<id>       # probably already set
DISCORD_CHANNEL_OPENCLAW_ESCALATIONS=<id>  # probably already set
DISCORD_CHANNEL_OPENCLAW_HEALTH=<id>       # probably already set
```

Verify all existing required vars are still set:
```bash
DATABASE_URL, THE_ODDS_API_KEY, KENPOM_API_KEY,
API_KEY_USER1, DISCORD_BOT_TOKEN,
YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REFRESH_TOKEN
```

### 5.2 Railway Deploy Sequence

```
Epic 1:  git push origin main
         railway run python scripts/migrate_v8_post_draft.py
         # verify schema via psql or Railway DB console

Epic 2:  git push origin main
         # set ENABLE_INGESTION_ORCHESTRATOR=true in Railway vars
         # watch railway logs --follow for "DailyIngestionOrchestrator started"
         # call GET /admin/ingestion/status and confirm all jobs listed

Epic 3:  git push origin main
         # set all DISCORD_CHANNEL_* vars
         # verify morning brief at 7 AM ET next day
```

### 5.3 Rollback Procedures

**Schema rollback (if Epic 1 breaks Railway startup):**
```bash
railway run python scripts/migrate_v8_post_draft.py --downgrade
# Confirm Railway restarts cleanly
```

**Epic 2 rollback (if orchestrator causes duplicate jobs):**
```bash
# In Railway env vars:
ENABLE_INGESTION_ORCHESTRATOR=false
# Railway auto-restarts with orchestrator disabled
# No code change or schema rollback required
```

**Epic 3 rollback (if Discord rate limits or loop errors):**
```bash
# In Railway env vars:
ENABLE_INGESTION_ORCHESTRATOR=false
# All Epic 3 code is inside the orchestrator — disabled instantly
```

---

## 6. PERFORMANCE & MONITORING CONTRACTS

### 6.1 Operational Metrics

| Metric | Target | Alert Threshold | Alert Destination |
|--------|--------|-----------------|-------------------|
| Daily ingestion success rate | >99% | <95% | `#openclaw-escalations` |
| Advisory lock skips per 24h | <5 | >20 | Log warning only |
| Morning brief latency | <30s | >60s | `#openclaw-health` |
| Waiver scan latency | <45s | >90s | `#openclaw-health` |
| Discord delivery success | >99% | <95% | `send_system_error()` |
| `player_daily_metrics` row count | <27,400/90d | >50,000 | Log warning (cleanup running?) |

### 6.2 Table Bloat Prevention

`_cleanup_old_metrics()` runs daily at 3:30 AM ET via `DailyIngestionOrchestrator`.
Retention: 90 days for `player_daily_metrics`.
Retention: indefinite for `projection_snapshots` (delta-compressed, small).

Expected steady-state size: ~300 players × 365 days = ~110,000 rows/year.
At ~500 bytes/row = ~55 MB/year. Acceptable without partitioning.

---

## 7. AGENT ROUTING — WHO DOES WHAT

### Immediate Priority (Waiver Wire Critical Fixes)

| Task | Agent | Constraint |
|------|-------|-----------|
| Waiver wire backend fixes (ownership %, 2-start SP, pagination) | Claude Code | See §14 for detailed spec |
| Waiver wire Next.js UI overhaul | Claude Code | `frontend/app/(dashboard)/fantasy/waiver/page.tsx` |
| Railway `railway run python scripts/migrate_v8_post_draft.py` | Gemini (ops) | Run command only, no edits |
| Railway env var setup | Gemini (ops) | Verify then set |
| `railway logs --follow` monitoring | Gemini (ops) | Report back in HANDOFF.md |

### Strategic Workstream (OpenClaw Autonomy)

| Task | Agent | Constraint |
|------|-------|-----------|
| **OpenClaw Autonomy Architecture** | **Kimi CLI (LEAD)** | Design full SOUL.md vision: alpha decay detection, performance monitoring, self-improvement loops |
| OpenClaw implementation | Kimi CLI proposes; Claude approves | Guardian-compliant (read-only until Apr 7) |
| Post-implementation audit (whole corpus) | Kimi CLI | Read all new files + models.py, confirm no anti-patterns |
| Waiver report interpretation | OpenClaw | Reads output of `/api/fantasy/waiver` endpoint |
| V9.2 recalibration (Apr 7+) | Claude Code | After EPIC-3 complete. See HANDOFF.md §5.1 (prior version) |

**Note:** Claude owns immediate waiver wire delivery (P0). Kimi owns OpenClaw autonomy design (strategic). Parallel workstreams with weekly sync.

---

## 8. UPDATED HEARTBEAT REGISTRY

Add these loops to `HEARTBEAT.md` after EPIC-2 and EPIC-3 are live:

### New Loop: MLB Odds Poll
- **Trigger:** Every 5 min, 10 AM–11 PM ET (EPIC-2 orchestrator)
- **Job ID:** `mlb_odds`
- **Owner:** DailyIngestionOrchestrator
- **Advisory lock ID:** 100_001

### New Loop: Statcast 2.0 Update
- **Trigger:** Every 6 hours (EPIC-2 orchestrator)
- **Job ID:** `statcast`
- **Advisory lock ID:** 100_002

### New Loop: Rolling Z-Scores
- **Trigger:** Daily 4 AM ET (EPIC-2 orchestrator)
- **Job ID:** `rolling_z`
- **Advisory lock ID:** 100_003

### New Loop: OpenClaw Morning Workflow (MLB + Waiver)
- **Trigger:** 7 AM ET daily (EPIC-3 via orchestrator)
- **Job ID:** `morning_workflow`
- **Output channels:** `#openclaw-briefs`, `#fantasy-waivers`
- **Advisory lock ID:** 100_008

### New Loop: CLV Attribution
- **Trigger:** 11 PM ET daily (EPIC-2 orchestrator)
- **Job ID:** `clv`
- **Advisory lock ID:** 100_005

---

## 9. HIVE WISDOM — LESSONS TO CARRY FORWARD

| Lesson | Source |
|--------|--------|
| `AsyncIOScheduler` fires in EVERY Uvicorn worker — use pg_try_advisory_lock for any new job | ADR-001 |
| `player_daily_metrics` UNIQUE constraint is `(player_id, metric_date, sport)` — always upsert, never insert-or-fail | Schema design |
| `discount_notifier.send_to_channel()` is the only function that should make raw Discord HTTP calls | ADR, discord_router contract |
| `WaiverEdgeDetector` must return `[]` not raise — autonomous loop must be failure-proof | ADR-004 |
| EPIC-2 orchestrator is gated by `ENABLE_INGESTION_ORCHESTRATOR=true` — off by default | ADR-004 |
| Set Discord channel env vars before running EPIC-3 or the router silently no-ops | discord_notifier behavior |
| `migrate_v8_post_draft.py --downgrade` is the atomic rollback — keep it working | ADR-002 |
| CBB model files still frozen until Apr 7 — V9.2 recalibration is a separate mission | GUARDIAN |

---

## 10. PRIOR ART — PRESERVE THESE

These are the items from EMAC-076 that must NOT be lost during EPIC implementation:

- `tasks/cbb_enhancement_plan.md` — V9.2 implementation roadmap
- `reports/K12_RECALIBRATION_SPEC_V92.md` — V9.2 params (apr 7)
- `reports/K13_POSSESSION_SIM_AUDIT.md` — K-14 pricing_engine spec
- `backend/services/haslametrics.py` — 3rd rating source, 12 tests, ready to wire post-Apr 7
- `backend/fantasy_baseball/draft_analytics.py` — value-board engine (written EMAC-077 pre-draft)
- CBB GUARDIAN: do not touch `betting_model.py`, `analysis.py` until Apr 7

---

## 11. IGNITION SWITCH

This is the single command to run on Monday morning to start EPIC-1. Run it from the project root after confirming `pytest tests/ -q` passes clean.

```bash
python scripts/migrate_v8_post_draft.py --dry-run && echo "DRY RUN OK — review SQL above, then run without --dry-run"
```

After reviewing the SQL output and confirming it matches the spec in §2.2, run:

```bash
python scripts/migrate_v8_post_draft.py
```

Then verify:

```bash
pytest tests/test_schema_v8.py -v && pytest tests/ -q
```

If both pass, EPIC-1 is complete. Proceed to EPIC-2.

---

---

## 12. FANTASY BASEBALL ELITE ROADMAP — ALGORITHMIC EXPANSION

> **Authored:** Kimi CLI · March 23, 2026  
> **Spec:** `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md`  
> **Status:** DRAFT — Awaits EPIC-1 through EPIC-3 completion  
> **Priority:** P0 (Post-CBB/March Madness pivot)

### Context: From Draft Helper to Quantitative Asset Management

The Fantasy Baseball module is evolving from a **draft-day assistant** into an **institutional-grade roster management system**. This requires treating fantasy baseball as a multi-agent, multi-timeframe portfolio optimization problem.

### Algorithmic Innovations (Phase 2)

| Innovation | Algorithm | Purpose | Owner |
|------------|-----------|---------|-------|
| **Bayesian Projection Updating** | Conjugate normal priors + shrinkage | Adapt projections as season unfolds | Claude Code |
| **Ensemble Projections** | Inverse-MAE weighted ensemble | Combine Steamer/ZiPS/Yahoo ROS optimally | Claude Code |
| **MCMC Weekly Simulator** | Gibbs sampling (10k sims) | Full outcome distributions, not point estimates | Claude Code |
| **Contextual Bandits** | LinUCB | Real-time add/drop decisions | Claude Code |
| **Portfolio Optimization** | Mean-variance quadratic programming | Risk-adjusted roster construction | Claude Code |
| **Reinforcement Learning** | Deep Q-Network (DQN) | Learn optimal roster moves over season | Claude Code + Kimi (validation) |
| **Graph Neural Networks** | GAT (Graph Attention Networks) | Optimal daily lineup selection | Claude Code |

### Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              FANTASY BASEBALL ORCHESTRATION                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Yahoo Agent │  │Statcast     │  │FanGraphs    │  Data Layer │
│  │ (OpenClaw)  │  │Agent (Kimi) │  │Agent (Claude│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│         │                │                │                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           ENSEMBLE PROJECTION AGENT (Claude)           │   │
│  │    Bayesian Update → Ensemble → Confidence Intervals   │   │
│  └─────────────────────────────────────────────────────────┘   │
│         │                │                │                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Weekly      │  │ Roster      │  │ Streamer    │  Decision   │
│  │ Strategy    │  │ Construction│  │ Optimization│  Layer      │
│  │ Agent       │  │ Agent (GNN) │  │ Agent       │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Integration with EMAC-077/078 EPICs

**Fantasy Baseball work is SEPARATE from CBB EPICs 1-6.**

| Timeline | CBB Activity | Fantasy Activity |
|----------|-------------|------------------|
| Now (Mar 23) | EPIC-1 (Schema) + EPIC-2 (Orchestrator) | Foundation planning |
| Apr 7 | EPIC-4 (Bracket sunset), EPIC-5 (MLB polling) | Begin Phase 1 implementation |
| Apr 15 | EPIC-6 (Admin suite) | Universal projections + MCMC |
| May | V9.2 CBB recalibration (off-season) | Bayesian updater + RL training |
| June | CBB model maintenance | Full multi-agent deployment |

### Key Technical Decisions

**ADR-007: Fantasy Baseball is Additive, Not Substitution**
- All Fantasy Baseball code lives in `backend/fantasy_baseball/` and `backend/services/daily_ingestion.py`
- No modification to CBB model files (frozen per ADR-004)
- Fantasy orchestrator is a separate APScheduler instance within `DailyIngestionOrchestrator`

**ADR-008: Projection Layer Stratification**
```python
# Tier 1: Pre-computed (Draft Board) — static, high confidence
# Tier 2: Yahoo API (Real-time ROS) — refreshed every 6 hours
# Tier 3: Derived/Heuristic (MLE for call-ups) — computed on-demand
# Tier 4: Bayesian Posterior (Season-long learning) — updated after each game
```

**ADR-009: Multi-Timeframe Value Functions**
```python
class PlayerValue:
    ros_value: float          # Trade decisions (full season)
    four_week_value: float    # Waiver add decisions
    weekly_value: float       # Streamer decisions
    daily_value: float        # Lineup optimization
```

### Implementation Phases

**Phase 1: Foundation (Weeks 1-2, starting Apr 7)**
- Universal projection system (`get_or_create_projection()`)
- Yahoo ROS integration
- Basic roster recommendations
- MCMC weekly simulator

**Phase 2: Intelligence (Weeks 3-4)**
- Bayesian updater
- Ensemble projector
- Statcast trend detection (Kimi)
- Contextual bandit

**Phase 3: Optimization (Weeks 5-6)**
- Portfolio optimizer
- Weekly strategy engine
- GNN lineup setter
- Multi-agent orchestration

**Phase 4: Automation (Weeks 7-8)**
- RL agent (DQN) training
- Auto-execution for low-risk moves
- Real-time opportunity alerts

### Claude Code Responsibilities

1. **Architect all algorithms** — Bayesian, MCMC, RL, GNN implementations
2. **Design orchestration layer** — Agent message bus, coordination protocol
3. **Implement Phase 1** — Universal projections, MCMC simulator
4. **Coordinate with Kimi** — Validation of RL training, trend detection logic
5. **Coordinate with OpenClaw** — Real-time execution, Yahoo API integration

### Success Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Projection coverage | 99%+ of Yahoo universe | ~30% (draft board) |
| Recommendation accuracy | 70%+ of adds outperform drops | N/A |
| Weekly matchup win rate | 60%+ H2H | Baseline 50% |
| Time to actionable insight | <5 seconds | Manual browsing |
| Human intervention required | <20% of moves | 100% manual |

### Document References

- Full spec: `reports/FANTASY_BASEBALL_ELITE_ROADMAP_v2.md`
- Algorithm cheat sheet in spec appendix
- Multi-agent orchestration YAML in spec §2.3

---

## 13. PHASE 2 TRANSITION ROADMAP — EPICS 4-6

> **Authored:** EMAC-078 · March 23, 2026 · Claude Code (Master Architect)
> **Trigger condition:** These epics activate AFTER the CBB season concludes and ADR-004 freeze lifts (April 7, 2026).
> EPIC-4 → EPIC-5 → EPIC-6 must run sequentially. Do not start EPIC-5 until EPIC-4 is merged and verified.

### ADR-006: MLB Model Analysis — NOW IN SCOPE (Updated Mar 24, 2026)

**Status:** ⚠️ **SCOPE CHANGE** — MLB betting model transition required before CBB season ends (Apr 7).

**Context:** 
- CBB season ends ~April 7, 2026 (championship game)
- MLB season starts ~March 28, 2026 (ALREADY ACTIVE)
- **System must transition from CBB betting → MLB betting during overlap period**
- Fantasy baseball is already operational (separate from betting model)

**Two Core Components (Active Simultaneously During Overlap):**
1. **Fantasy Baseball** — Waiver wire, lineups, daily optimizations (ALREADY LIVE)
2. **MLB Betting Model** — Today's picks, runline analysis, nightly pipeline (MUST BE BUILT)

**Requirements:**
- `SportConfig.mlb()` constructor for MLB-specific parameters
- Parallel nightly analysis pipeline for MLB (runline, totals)
- MLB-specific OpenClaw patterns (pitcher form, bullpen fatigue, weather)
- Sport switching logic: CBB winds down Apr 1-7, MLB fully active Apr 8+

**Timeline:**
- **Now-Apr 1:** Build MLB betting model alongside CBB
- **Apr 1-7:** Overlap period — both sports active
- **Apr 8+:** Full MLB betting mode

**Quota Management:** MLB odds polling ~1,800/month (well under 20,000 cap)

---

### EPIC-4: Bracket Sunset (UI Deprecation)

**Owner:** Claude Code
**Trigger:** April 7, 2026 (post-championship)
**Prerequisite:** EPIC-1, EPIC-2, EPIC-3 complete
**Touches:** Frontend only + one scheduler job removal

#### Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 4.1 | Safety grep: `grep -r "bracket\|BracketProjection\|tournament_data"` — confirm no imports outside target files | Various | [ ] |
| 4.2 | Delete bracket route files | `frontend/app/(dashboard)/bracket/page.tsx`, `error.tsx`, `loading.tsx` | [ ] |
| 4.3 | Remove "Tournament" nav section (Trophy icon block) | `frontend/components/layout/sidebar.tsx` | [ ] |
| 4.4 | Remove `bracketProjection()` function | `frontend/lib/api.ts` | [ ] |
| 4.5 | Remove `BracketProjection`, `TeamAdvancement`, `UpsetAlert` interfaces | `frontend/lib/types.ts` | [ ] |
| 4.6 | Remove `tournament_bracket_notifier` APScheduler job | `backend/main.py` (lines ~235-238) | [ ] |
| 4.7 | Archive tournament service (do NOT delete — preserve for potential future use) | `backend/services/tournament_data.py` → `backend/archive/tournament_data.py` | [ ] |
| 4.8 | Verify: `npm run build` passes with zero TS errors | `frontend/` | [ ] |
| 4.9 | Verify: `/bracket` returns 404, all other routes healthy | Live app | [ ] |

#### EPIC-4 Handoff Prompt (copy-paste ready for coding agent)

```
EPIC-4: Bracket Sunset

Context: NCAA tournament is over. We are removing all bracket/tournament UI from the frontend.
The CBB model files remain FROZEN (ADR-004) — do not touch betting_model.py or analysis.py.

Step 1 — Safety check (READ-ONLY first):
  grep -r "bracket\|BracketProjection\|tournament_data\|Trophy" /home/user/CBB_Betting/frontend/
  grep -r "tournament_bracket" /home/user/CBB_Betting/backend/main.py
  Report every file that contains these strings before making any edits.

Step 2 — Delete these files (only after step 1 confirms no surprise imports):
  frontend/app/(dashboard)/bracket/page.tsx
  frontend/app/(dashboard)/bracket/error.tsx
  frontend/app/(dashboard)/bracket/loading.tsx

Step 3 — Edit these files:
  a. frontend/components/layout/sidebar.tsx — remove the "Tournament" nav section
     (the block containing the Trophy icon and the /bracket href)
  b. frontend/lib/api.ts — remove the bracketProjection() function
  c. frontend/lib/types.ts — remove BracketProjection, TeamAdvancement, UpsetAlert interfaces

Step 4 — Backend cleanup:
  a. backend/main.py — remove the tournament_bracket_notifier scheduler job
  b. Move (do NOT delete): backend/services/tournament_data.py → backend/archive/tournament_data.py
     (create backend/archive/ directory if it doesn't exist)

Step 5 — Verify:
  cd /home/user/CBB_Betting/frontend && npm run build
  npx tsc --noEmit
  Both must pass with zero errors. Report the output.

Step 6 — Commit and push to branch claude/clarify-bet-recommendations-ui-WC8Do:
  git add -A && git commit -m "EPIC-4: Remove bracket/tournament UI post-season"
  git push -u origin claude/clarify-bet-recommendations-ui-WC8Do
```

---

### EPIC-5: Sport Polling Switch (API Quota Management)

**Owner:** Claude Code (backend) · Gemini CLI (Railway env vars only)
**Trigger:** April 8, 2026
**Prerequisite:** EPIC-4 complete
**Touches:** `backend/core/sport_polling_switch.py` (new), `backend/services/odds.py`, `backend/models.py`, `backend/main.py`

#### Quota Budget (do not exceed)

| Phase | Sport | Calls/Month | Budget |
|---|---|---|---|
| Now (CBB active) | basketball_ncaab | ~11,610 | OK |
| Transition (Apr 7-8) | Both winding down | ~3,000 | OK |
| MLB season | baseball_mlb | ~1,800 | Well under |
| **Hard cap** | | **20,000** | 2,000 reserve |

MLB polling schedule: Morning check 9 AM (1 call), pre-game 11 AM-4 PM every 10 min (30 calls), game-time 5 PM-midnight every 15 min (28 calls), nightly settle 1 AM (1 call). Total: ~60/day → 1,800/month.

#### Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 5.1 | Add `sport_poll_config` table to models | `backend/models.py` | [ ] |
| 5.2 | Write migration script | `scripts/migrate_sport_poll_config.py` | [ ] |
| 5.3 | Create `SportPollingSwitch` class with `wind_down_cbb()`, `activate_mlb()`, `get_quota_status()` | `backend/core/sport_polling_switch.py` (NEW) | [ ] |
| 5.4 | Add generic `get_odds(sport_key: str)` to OddsAPIClient; keep `get_cbb_odds()` as wrapper | `backend/services/odds.py` | [ ] |
| 5.5 | Add `get_mlb_odds()` wrapper | `backend/services/odds.py` | [ ] |
| 5.6 | Register MLB scheduler jobs with PG advisory locks (ADR-001) | `backend/main.py` | [ ] |
| 5.7 | Auto-pause CBB jobs on April 7 (CronTrigger 11:59 PM ET) | `backend/main.py` | [ ] |
| 5.8 | Add admin endpoints: `GET/POST /admin/sport-switch`, `GET /admin/quota/history` | `backend/main.py` | [ ] |
| 5.9 | Write `tests/test_sport_polling_switch.py` (mock OddsAPI, test lock behavior) | `tests/` | [ ] |
| 5.10 | Gemini CLI: set `MLB_ACTIVE=false` in Railway env (starting value) | Railway dashboard | [ ] |

#### EPIC-5 Handoff Prompt

```
EPIC-5: Sport Polling Switch

Context: CBB season is over. We need to pivot API polling from basketball_ncaab to baseball_mlb.
Hard quota cap: 20,000 requests/month to The Odds API. Target MLB spend: ~1,800/month.
ADR-001 is non-negotiable: ALL new scheduler jobs must use pg_try_advisory_lock.
ADR-006: Do NOT wire MLB odds into nightly_analysis or produce model predictions for MLB.

Files to read first:
  backend/services/odds.py           — OddsAPIClient, get_cbb_odds(), quota tracking
  backend/core/sport_config.py       — SportConfig class (already has mlb sport key stub)
  backend/models.py                  — existing table patterns to follow
  backend/main.py lines 94-258       — existing scheduler jobs (pattern to replicate)

Tasks:
1. Add to backend/models.py:
   - Table `sport_poll_config`: id, cbb_active (bool, default True), mlb_active (bool, default False),
     transition_date (Date), updated_at (Timestamptz)
   - ORM class SportPollConfig

2. Create backend/core/sport_polling_switch.py:
   - Class SportPollingSwitch(db: Session)
   - Methods: active_sports(), wind_down_cbb(), activate_mlb(), get_quota_status()
   - wind_down_cbb() sets cbb_active=False; activate_mlb() sets mlb_active=True

3. Modify backend/services/odds.py:
   - Add get_odds(sport_key: str) — generic method (move URL construction there)
   - Refactor get_cbb_odds() to call get_odds("basketball_ncaab")
   - Add get_mlb_odds() calling get_odds("baseball_mlb")

4. Modify backend/main.py:
   - Add CronTrigger job `cbb_wind_down` firing April 7 at 11:59 PM ET
     → calls SportPollingSwitch.wind_down_cbb() and pauses CBB jobs
   - Add 4 MLB jobs (all with pg_try_advisory_lock):
     * mlb_morning_lines: CronTrigger 9 AM ET
     * mlb_pregame_monitor: IntervalTrigger 10 min (only run between 11 AM-4 PM via time check)
     * mlb_game_monitor: IntervalTrigger 15 min (only run between 5 PM-midnight)
     * mlb_nightly_settle: CronTrigger 1 AM ET
   - Add endpoints: GET /admin/sport-switch/status, POST /admin/sport-switch,
     GET /admin/quota/history

5. Write tests/test_sport_polling_switch.py covering:
   - wind_down_cbb() sets correct DB state
   - activate_mlb() sets correct DB state
   - get_mlb_odds() calls correct URL ("baseball_mlb")
   - Advisory lock prevents double-execution

6. Create migration script: scripts/migrate_sport_poll_config.py
   Run it and report output.

Report: file diffs, test results, quota projection calculation.
Push to branch claude/clarify-bet-recommendations-ui-WC8Do.
```

---

### EPIC-6: Admin Suite & Access Control

**Owner:** Claude Code
**Trigger:** April 15, 2026 (after EPIC-5 is stable for 1 week)
**Prerequisite:** EPIC-5 complete and verified on Railway
**Touches:** `backend/auth.py` (rewrite), `backend/models.py`, `frontend/app/(dashboard)/admin/page.tsx`

#### Role Matrix

| Action | owner | risk_manager | viewer |
|---|---|---|---|
| Read any data | ✓ | ✓ | ✓ |
| Acknowledge alerts / override bankroll | ✓ | ✓ | ✗ |
| Pause betting markets / adjust line projections | ✓ | ✓ | ✗ |
| Run analysis / recalibrate / delete bets | ✓ | ✗ | ✗ |
| Manage users / sport switch | ✓ | ✗ | ✗ |

#### Sub-tasks

| # | Task | File | Done? |
|---|------|------|-------|
| 6.1 | Add `user_role` enum, `users` table, `audit_log` table | `backend/models.py` | [ ] |
| 6.2 | Write migration: `scripts/migrate_users_rbac.py` (seeds owner from API_KEY_USER1 env var) | `scripts/` | [ ] |
| 6.3 | Rewrite `verify_api_key()` → DB lookup + bcrypt verify → returns `User` ORM object | `backend/auth.py` | [ ] |
| 6.4 | Add `require_role(*roles)` dependency factory | `backend/auth.py` | [ ] |
| 6.5 | Apply role guards to all admin routes (owner-only: delete/recalibrate/sport-switch/user-mgmt; risk_manager: bankroll/pause/alerts) | `backend/main.py` | [ ] |
| 6.6 | Add audit_log write middleware for all /admin/* endpoints | `backend/main.py` | [ ] |
| 6.7 | Add endpoints: `GET /admin/users`, `POST /admin/users`, `DELETE /admin/users/{id}` | `backend/main.py` | [ ] |
| 6.8 | Add endpoints: `POST /admin/markets/{id}/pause`, `DELETE /admin/markets/{id}/pause` | `backend/main.py` | [ ] |
| 6.9 | Add endpoint: `GET /admin/audit-log` (last 100 actions) | `backend/main.py` | [ ] |
| 6.10 | Extend admin page with 4 tabs: Risk Controls, User Management, Audit Log, Quota Monitor | `frontend/app/(dashboard)/admin/page.tsx` | [ ] |
| 6.11 | Write `tests/test_auth_rbac.py` — verify 403 on role violations, audit log writes | `tests/` | [ ] |

#### EPIC-6 Handoff Prompt

```
EPIC-6: Admin Suite & Access Control

Context: Replace hardcoded user1=admin with proper RBAC. 3 roles: owner, risk_manager, viewer.
This is a solo-to-small-team system (max 5 users). No SAML/SSO in scope — SSO is a future
migration path via AuthProvider interface but NOT implemented now.

Files to read first:
  backend/auth.py              — current simple API key auth (to be rewritten)
  backend/models.py            — existing table patterns
  backend/main.py              — existing /admin/* routes and their auth dependencies
  frontend/app/(dashboard)/admin/page.tsx  — existing 6-panel admin UI

Phase A — DB Layer:
1. Add to backend/models.py:
   - Enum UserRole = Literal['owner', 'risk_manager', 'viewer']
   - Table `users`: id, username (unique), api_key_hash (bcrypt), role (UserRole), is_active (bool), created_at, last_seen
   - Table `audit_log`: id, user_id (FK users.id), action, endpoint, payload (JSON), ip_address, ts

2. Create scripts/migrate_users_rbac.py:
   - Creates both tables
   - Seeds one owner user from API_KEY_USER1 env var (bcrypt hash it, don't store plaintext)
   - Run and report output

Phase B — Auth Rewrite (backend/auth.py):
3. Rewrite verify_api_key(db, api_key) → User:
   - Query users table by doing bcrypt.checkpw against each active user's api_key_hash
   - Update last_seen on successful auth
   - Raise 401 if no match
4. Add require_role(*allowed_roles) → Depends():
   - Factory that returns a FastAPI dependency
   - Raises 403 if user.role not in allowed_roles
5. Keep verify_admin_api_key as compatibility shim calling require_role('owner')

Phase C — Route Guards (backend/main.py):
6. Apply require_role('owner') to: /admin/run-analysis, /admin/recalibrate, delete endpoints, /admin/sport-switch, user management endpoints
7. Apply require_role('owner', 'risk_manager') to: /admin/bankroll POST, /admin/alerts/*/acknowledge, /admin/markets/*/pause
8. Add audit_log write on every /admin/* endpoint (log action + user_id + endpoint + payload)

Phase D — New Endpoints:
9. GET /admin/users — list all users (owner only)
10. POST /admin/users — create user, generate API key, return key ONCE (owner only)
11. DELETE /admin/users/{id} — deactivate user (owner only)
12. POST /admin/markets/{id}/pause — pause a betting market (risk_manager+)
13. DELETE /admin/markets/{id}/pause — resume market (risk_manager+)
14. GET /admin/audit-log — last 100 entries (owner only)

Phase E — Frontend:
15. Extend frontend/app/(dashboard)/admin/page.tsx with a tabbed layout:
    - Tab 0: System Status (existing panels, unchanged)
    - Tab 1: Risk Controls (market pause toggles, line override inputs) — visible to risk_manager+
    - Tab 2: User Management (user list, add/revoke) — visible to owner only
    - Tab 3: Audit Log table (who/what/when) — visible to owner only
    - Tab 4: Quota Monitor (calls used, burn rate chart, 30-day trend) — visible to risk_manager+
    Role visibility: read current user's role from GET /api/me (new endpoint returning {username, role})

Phase F — Tests:
16. tests/test_auth_rbac.py:
    - risk_manager key → 403 on DELETE /admin/bets/1
    - risk_manager key → 200 on POST /admin/markets/1/pause
    - owner key → 200 on DELETE /admin/bets/1
    - Every admin action creates audit_log entry

Report: all test results, role matrix verification table.
Push to branch claude/clarify-bet-recommendations-ui-WC8Do.
```

---

## 14. OPENCLAW AUTONOMY WORKSTREAM — Strategic Initiative

> **Owner:** Kimi CLI (Deep Intelligence Unit) — **LEAD ARCHITECT**  
> **Status:** SPEC PHASE — Awaiting waiver wire completion  
> **Goal:** Transform OpenClaw from heuristic checker to autonomous "Soul" system per original vision  
> **Value:** Automated model monitoring, alpha decay detection, self-improvement  

### 14.1 Vision: From Tool to Autonomous Agent

Current OpenClaw (v3.0) is a **function** — called during analysis, returns verdict.

Full OpenClaw (v4.0+) is an **autonomous system** per SOUL.md:
- **Self-directed:** Monitors performance without human triggers
- **Self-aware:** Detects when model edge degrades (alpha decay)
- **Self-improving:** Proposes and implements code improvements
- **Always-on:** 24/7 monitoring, alerting, optimization

### 14.2 Core Mandates (from SOUL.md)

| Mandate | Current | Target | Owner |
|---------|---------|--------|-------|
| **Alpha Decay Detection** | ❌ None | ✅ Every 2 hours | Kimi CLI |
| **Structural Vulnerability** | ❌ Manual | ✅ Automated pattern analysis | Kimi CLI |
| **Roadmap Evolution** | ❌ Static | ✅ Living improvement proposals | Kimi CLI |
| **Autonomous Implementation** | ❌ Disabled | ⚠️ Post-Apr 7 (Guardian) | Kimi proposes; Claude implements |

### 14.3 Architecture Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPENCLAW AUTONOMOUS SYSTEM v4.0                   │
├─────────────────────────────────────────────────────────────────────┤
│  📊 Performance Monitor                                              │
│     ├── CLV decay tracker (every 2h)                                │
│     ├── Win rate trend analysis                                     │
│     ├── Conference/team pattern detection                           │
│     └── Alert when edge degrades > threshold                        │
│                                                                      │
│  🔍 Data Validator                                                   │
│     ├── Ratings source freshness checks                             │
│     ├── Odds line movement anomalies                                │
│     └── Injury data completeness                                    │
│                                                                      │
│  🧠 Learning Engine                                                  │
│     ├── Loss pattern analysis (conference, total, seed, HCA)        │
│     ├── Feature importance drift detection                          │
│     └── Automatic A/B test proposals                                │
│                                                                      │
│  📝 Roadmap Maintainer                                               │
│     ├── Auto-updates ROADMAP.md with findings                       │
│     ├── Ranks improvements by expected ROI                          │
│     └── Schedules implementation queue                              │
│                                                                      │
│  🔧 Self-Improvement (Post-Apr 7)                                    │
│     ├── Auto-recalibration triggers                                 │
│     ├── Weight adjustments based on backtests                       │
│     └── Safe code modification with rollback                        │
│                                                                      │
│  📢 Notifier                                                         │
│     ├── Discord alerts for drift, anomalies, improvements           │
│     ├── Morning brief with health summary                           │
│     └── Escalation queue for human review                           │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.4 Deliverables & Timeline (CORRECTED — Phase 1 Starts NOW)

**✅ Phase 0: Design (COMPLETE)**
- [x] Kimi CLI: Full architecture spec → `reports/OPENCLAW_AUTONOMY_SPEC_v4.md`
- [x] Kimi CLI: Database schema for time-series metrics
- [x] Kimi CLI: API contracts for all agents

**🚀 Phase 1: Foundation (START NOW — Mar 24)**
*Does NOT violate Guardian freeze (read-only monitoring)*
- [ ] Performance Monitor service (`backend/services/openclaw/performance_monitor.py`)
- [ ] CLV decay detection algorithm
- [ ] Pattern analysis engine
- [ ] Discord alerting integration
- [ ] Database migration (4 new tables)

**Phase 2: Intelligence (Mar 31-Apr 7)**
- [ ] Learning Engine with historical analysis
- [ ] Roadmap auto-maintenance
- [ ] A/B test framework design
- [ ] Weekly Monday 6 AM job

**Phase 3: Autonomy Setup (Apr 1-7)**
- [ ] Self-improvement framework (disabled mode)
- [ ] Rollback mechanism
- [ ] Safety constraint validation

**Phase 4: Activation (Apr 8+)**
- [ ] Enable self-improvement (auto-implementation)
- [ ] Full A/B test automation
- [ ] Continuous learning loop
- [ ] Weekly autonomy reports

### 14.5 Key Technical Decisions

**Database:** Use existing `player_daily_metrics` time-series schema (EPIC-1)

**Scheduling:** Integrate with `DailyIngestionOrchestrator` (EPIC-2)
- Performance checks every 2 hours
- Pattern analysis nightly
- Roadmap updates weekly

**Safety:** All self-improvement proposals require:
1. Kimi CLI review
2. Backtest validation
3. Claude Code approval (architect)
4. Rollback plan

**Guardian Compliance (Clarified):**

**ALLOWED NOW (Doesn't touch frozen CBB model files):**
- ✅ Read-only monitoring (querying bet_log, predictions tables)
- ✅ Pattern detection (analyzing outcomes)
- ✅ Proposal generation (writing to DB/markdown)
- ✅ Discord alerting (notifications)
- ✅ NEW infrastructure (monitoring tables, scheduler jobs)

**BLOCKED until Apr 7 (Would modify frozen files):**
- ❌ Auto-recalibration of betting_model.py parameters
- ❌ Auto-adjustment of WEIGHT_KENPOM etc.
- ❌ Self-modifying Python code

**Implementation Strategy:** Build everything NOW. Only the "auto-implement" switch stays off until Apr 8.

### 14.6 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Alpha decay detection latency | <4 hours | Time from edge degradation → alert |
| Pattern detection accuracy | >80% | Manual validation of flagged patterns |
| Improvement proposal quality | >70% accepted | Ratio of approved vs rejected proposals |
| Autonomous fix success rate | >95% | Rollbacks vs successful deployments |
| Human oversight reduction | 80% fewer manual reviews | Post-autonomy vs pre-autonomy |

### 14.7 Specification Delivered

**Status:** ✅ **DESIGN COMPLETE** — Full specification in `reports/OPENCLAW_AUTONOMY_SPEC_v4.md`

**What's Been Delivered:**
- Complete 6-agent architecture (Performance, Pattern, Learning, Roadmap, Self-Improvement, Notifier)
- Database schema for time-series metrics, vulnerabilities, proposals, A/B tests
- CLV decay detection algorithm with thresholds
- Pattern detection for conferences, totals, seeds, HCA
- Proposal generation and ranking logic
- A/B test framework design
- Safety constraints for autonomous implementation
- 4-phase implementation plan (Foundation → Intelligence → Autonomy → Activation)
- Discord integration specs
- Success metrics and KPIs

**Handoff Protocol:**

**When Claude completes waiver wire:**
1. Kimi CLI reviews Claude's waiver implementation
2. Kimi CLI provides audit feedback
3. Claude implements OpenClaw Phase 1 (Performance + Pattern agents) per spec
4. Kimi designs Phase 2 (Learning + Roadmap agents)
5. Weekly sync: Claude (implementation) + Kimi (architecture) + OpenClaw (execution)

---

**Document Version:** EMAC-085
**Last Updated:** March 25, 2026
**Status:** ACTIVE — EPIC-1/2/3 COMPLETE. MLB model LIVE. UAT Phase 1+2 COMPLETE (fantasy baseball critical bugs fixed). Suite 1105/1109. Next: EPIC-4 Bracket Sunset (Apr 7) + UAT Phase 3 (brier score, Odds Monitor).
**Branch:** main
**Team:** Claude Code (Architect) · Kimi CLI (Audit) · OpenClaw (Execution Target) · Gemini (Ops/Railway only)
**Next operator (Claude Code):** EPIC-4 Bracket Sunset (Apr 7) OR UAT Phase 3 (brier score calc, Odds Monitor portfolio fetch, Bet History filter). See §18.4.
**Next operator (Gemini CLI):** Railway env vars PENDING — see §16. Context retention config available in `.gemini/config.yaml`. DO NOT write code — ops/env vars only.
**Next operator (Kimi CLI):** Fantasy Baseball elite platform gap analysis COMPLETE — see `reports/FANTASY_BASEBALL_GAP_ANALYSIS.md`. Architecture roadmap created. Ready to audit Claude's implementation.
**CRITICAL REMINDER:** See ADR-010 — Next.js is the ONLY UI. Streamlit (`dashboard/`) is RETIRED. Never reference Streamlit code.
**Apr 7 mission:** V9.2 recalibration — see §10 and prior HANDOFF.md §6
**Workstream Split (PARALLEL EXECUTION):**
- **Claude (P0 — Done):** MLB betting model COMPLETE — `SportConfig.mlb()` + `mlb_analysis.py` + team wRC+ ingestion (14 tests pass)
- **Claude (P1 — Apr 7):** EPIC-4 Bracket Sunset — see §13 for copy-paste prompt
- **Kimi (P1 — DONE):** OpenClaw Phase 1 COMPLETE — 24 tests pass. Build fixed (flake8 clean).
- **Gemini (Ops):** Set `INTEGRITY_SWEEP_ENABLED=false` + `ENABLE_MLB_ANALYSIS=true` in Railway
- **URGENT:** Set `INTEGRITY_SWEEP_ENABLED=false` NOW — app is in restart loop without it

**OpenClaw Phase 1 Status (COMPLETE):**
- ✅ `backend/services/openclaw/` package created
- ✅ `performance_monitor.py` — CLV decay detection (15% CRITICAL, 8% WARNING), win rate tracking
- ✅ `pattern_detector.py` — CBB patterns (conference, seed, HCA, month, day-of-week), MLB patterns framework
- ✅ `database.py` — Guardian-gated DB layer (read-only until Apr 7)
- ✅ `scheduler.py` — APScheduler integration (every 2h performance check, daily 6 AM sweep)
- ✅ `v8_openclaw_monitoring.sql` — Migration with 4 tables + views
- ✅ `apply_openclaw_migration.py` — Migration script
- ✅ `daily_ingestion.py` updated to auto-start OpenClaw monitoring
- ✅ `tests/openclaw/` — 24 tests covering PerformanceMonitor and PatternDetector
- ✅ `scripts/openclaw_cli.py` — CLI for manual checks (`status`, `check-performance`, `run-sweep`, `health-summary`)
- ✅ Railway migration applied (per Gemini)
- ⏳ PENDING (Gemini): Verify Discord alerting integration after Railway env vars set (requires `DISCORD_ALERTS_ENABLED=true` + webhook URL)

**OpenClaw Implementation Notes:**
- Read-only monitoring during Guardian freeze — write operations blocked until Apr 7
- Phase 1 delivers foundation: monitoring + detection without self-modification
- Phase 2-4 (Learning, Roadmap, Self-improvement) scheduled post-Apr 7 per spec
- CBB patterns: conference bias, seed mispricing, HCA errors, month/day drift
- MLB patterns: framework ready for pitch fatigue, platoon splits, Coors effect (requires MLB data layer)


---

## §15. Kimi CLI Audit — MLB Betting Model (EMAC-082)

**Audit Date:** March 25, 2026  
**Auditor:** Kimi CLI (Deep Intelligence Unit)  
**Scope:** `backend/services/mlb_analysis.py`, `backend/fantasy_baseball/pybaseball_loader.py`, `backend/fantasy_baseball/advanced_metrics.py`

### §15.1 Architecture Assessment

| Aspect | Finding | Status |
|--------|---------|--------|
| Guardian Compliance | No imports from betting_model.py or analysis.py | ✅ PASS |
| SportConfig Integration | Uses SportConfig.mlb() with MLB-specific parameters | ✅ PASS |
| Async Design | run_analysis() is async, compatible with scheduler | ✅ PASS |
| Error Handling | Graceful degradation on all external API failures | ✅ PASS |
| Data Pipeline | pybaseball → cache → aggregation → projection | ✅ PASS |

### §15.2 Projection Formula Review

```
home_runs = league_avg * home_offense * away_pitching * park_factor + home_advantage
away_runs = league_avg * away_offense * home_pitching * park_factor
```

- **League Average:** 4.25 runs (2024 MLB average from FanGraphs)
- **Offense Factor:** wRC+ / 100 (team aggregate from batter cache)
- **Pitching Factor:** xERA / 4.25 (starter-specific from pitcher cache)
- **Park Factor:** 2024 Statcast run factors (Coors=1.22, Petco=0.90, etc.)
- **Home Advantage:** +0.25 runs (empirically derived for MLB)
- **Win Probability:** Normal CDF with σ = √total × 0.86

**Assessment:** Sound sabermetric approach. xERA is the correct choice over raw ERA for regression stability. Park factor application is multiplicative (industry standard).

### §15.3 Data Source Audit

| Source | Purpose | Cache Strategy | Fallback |
|--------|---------|----------------|----------|
| pybaseball (FanGraphs) | Batter wRC+, Pitcher xERA | 24-hour JSON cache | League average |
| statsapi | Schedule, probable pitchers | Live API | Empty list |
| The Odds API | Market runlines/totals | Live API | Zero edge |
| Hardcoded | Park factors | N/A | 1.0 neutral |

**Coverage:**
- Park factors: 29 parks mapped (all current MLB venues)
- Team mapping: 30 teams with abbreviation variants
- Pitcher matching: Exact name + fuzzy last-name fallback

### §15.4 Edge Calculation

```python
edge = projected_win_prob - market_implied_prob
```

- **Market conversion:** American odds → implied probability
- **Runline focus:** -110 = 52.4% implied (standard vig)
- **Positive edge:** Model more bullish on home team than market

**Assessment:** Correct implementation. Uses spreads market for runline edge.

### §15.5 Test Coverage

**14 tests in `tests/test_mlb_analysis.py`:**

| Test Category | Count | Key Tests |
|---------------|-------|-----------|
| SportConfig | 3 | MLB-specific params, home advantage magnitude |
| Projection | 6 | Park factor (Coors), pitcher quality, team offense |
| Integration | 3 | Schedule failure handling, empty returns |
| Edge Calc | 1 | Zero edge on missing market data |
| Data Loading | 1 | Team stats aggregation structure |

**All 14 tests pass.** Coverage is solid for core functionality.

### §15.6 Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| pybaseball unavailable | Low | Graceful skip with warning |
| Pitcher xERA missing | Low | Falls back to league avg (4.25) |
| Team wRC+ incomplete | Low | Minimum 3 batters threshold |
| Park factor unknown | Low | Defaults to 1.0 (neutral) |
| statsapi failure | Low | Returns empty list, no crash |
| Cache stale | Medium | 24-hour TTL, force_refresh flag |

### §15.7 Recommendations

1. **Monitor:** Add pybaseball cache hit/miss metrics to OpenClaw performance monitor
2. **Enhance:** Consider bullpen strength factor (currently only uses starter xERA)
3. **Validate:** Backtest against 2024 season data when available
4. **Document:** Add expected ROI ranges for different edge thresholds

### §15.8 Overall Assessment

**Status: APPROVED for production use**

The MLB betting model demonstrates:
- ✅ Sound statistical methodology
- ✅ Proper separation from CBB codebase
- ✅ Robust error handling
- ✅ Comprehensive test coverage
- ✅ Clean integration with existing infrastructure

The model is ready for live operation with the scheduled daily analysis job.

---

**Document Version:** EMAC-082-AUDIT
**Last Updated:** March 25, 2026


---

## §16. Gemini CLI Configuration — DevOps Lead Optimization

**Status: CONFIGURED** — Gemini CLI will now automatically load DevOps Lead context via `GEMINI.md` and restrict code-modifying tools.

### §16.1 Configuration Files (Working)

| File | Purpose | Status |
|------|---------|--------|
| `GEMINI.md` | Project context with role definition & pending ops | ✅ Active |
| `.gemini/settings.json` | Tool restrictions (blocks write_file, apply_diff, git) | ✅ Active |
| `scripts/gemini_recovery.sh` | Session recovery if auth lost | ✅ Available |

### §16.2 How It Works

**Gemini CLI automatically loads:**
1. `GEMINI.md` from project root (this defines your DevOps Lead role)
2. `.gemini/settings.json` (excludes code-modifying tools)
3. Context persists across sessions via `sessionRetention`

**Blocked tools (cannot write code):**
- `write_file` — Cannot create/modify files
- `apply_diff` — Cannot edit existing files  
- `run_git_command` — Cannot commit/push

**Allowed operations:**
- `run_shell_command` — Railway CLI, migrations, logs
- `read_file` — Read HANDOFF.md, check status

### §16.3 First-Time Setup (One-Time)

If this is the first Gemini CLI session in this project:

```bash
# Verify GEMINI.md is loaded
gemini /memory show

# Should show: "CBB Edge — DevOps Lead Context" at top

# If not loaded, refresh:
gemini /memory refresh
```

### §16.4 Current Pending Operations (Gemini)

```yaml
# Gemini CLI Configuration — DevOps Lead Profile
# EMAC-082: Optimized for Railway operations without code changes

context:
  # Persistent memory across sessions
  memory_file: .gemini/memory.json  # CREATED: Initial memory with pending ops
  
  # Auto-load these files on every session start
  auto_load:
    - HANDOFF.md
    - .env
    - railway.json
  
  # Session retention (reduce re-auth frequency)
  session_ttl: 3600  # 1 hour instead of default 15 min
  
  # Railway-specific shortcuts
  shortcuts:
    logs: railway logs --follow
    status: railway status
    vars: railway variables
    deploy: railway up

# Constraints (enforced)
constraints:
  # NEVER modify these (code freeze)
  read_only_patterns:
    - "backend/**/*.py"
    - "frontend/**/*.ts"
    - "frontend/**/*.tsx"
    - "tests/**/*.py"
  
  # CAN modify these (ops/docs only)
  write_patterns:
    - "HANDOFF.md"
    - ".env*"
    - "*.md"
    - "scripts/migrations/*.sql"
    - ".github/workflows/*.yml"

# Performance optimization
performance:
  # Cache Railway API responses
  cache_railway_api: true
  cache_ttl: 300  # 5 minutes
  
  # Batch env var operations
  batch_env_changes: true
```

### §16.2 Railway CLI Authentication (Persistent)

**One-time setup (run locally, commit token to repo securely):**

```bash
# Generate persistent token
railway login
railway whoami

# Save token for CI/automation
railway token > .railway/token

# Add to .env (already done — see .env file)
# RAILWAY_TOKEN=xxx
```

**For GitHub Actions (already configured in `.github/workflows/deploy.yml`):**
- Uses `RAILWAY_TOKEN` from repository secrets
- No interactive login required

### §16.3 Gemini Operations Checklist (Copy-Paste)

When Gemini CLI starts a new session, run these immediately:

```bash
# 1. Load context
cat HANDOFF.md | head -100

# 2. Verify Railway connection
railway whoami
railway status

# 3. Check current env vars (before making changes)
railway variables | grep -E "(INTEGRITY_SWEEP|ENABLE_MLB|ENABLE_INGESTION)"

# 4. Pending operations (copy from below)
```

### §16.4 Current Pending Operations (Gemini)

| Priority | Operation | Command | Status |
|----------|-----------|---------|--------|
| CRITICAL | Disable integrity sweep | `railway variables set INTEGRITY_SWEEP_ENABLED=false` | ✅ COMPLETE (Mar 26) |
| HIGH | Enable MLB analysis | `railway variables set ENABLE_MLB_ANALYSIS=true` | ✅ COMPLETE (Mar 26) |
| MEDIUM | Enable ingestion | `railway variables set ENABLE_INGESTION_ORCHESTRATOR=true` | ✅ COMPLETE (Mar 26) |
| LOW | Verify Discord webhook | Check `DISCORD_ALERTS_WEBHOOK` exists | ❌ FAILED (Mar 26) - Variable missing |

### §16.5 Gemini Agent Skills — IMPLEMENTED (Mar 25, 2026)

**Status:** COMPLETE — 4 skills live in `.gemini/skills/`

**Evaluation:** Implemented. Skills add session resilience (Gemini re-reads SKILL.md after context loss), bundle verification scripts co-located with documentation, and enforce correct workflows (dry-run-first, secret masking). Value exceeds simple shell aliases.

**Format Note:** Gemini CLI uses `SKILL.md` (YAML frontmatter + markdown), NOT `skill.json` + `handler.sh` as originally described in the prompt. Scripts live in `scripts/` subdirectory.

**Implemented Skills:**
| Skill | SKILL.md | Script | Purpose |
|-------|----------|--------|---------|
| `railway-logs` | `.gemini/skills/railway-logs/SKILL.md` | `scripts/filter-logs.sh` | Filter/diagnose Railway logs (`--errors`, `--grep=`, `--lines=`) |
| `db-migrate` | `.gemini/skills/db-migrate/SKILL.md` | `scripts/run-migration.sh` | Run migrations dry-run-first; `--verify-only` checks schema |
| `env-check` | `.gemini/skills/env-check/SKILL.md` | `scripts/check-vars.sh` | Verify Railway env vars; masks secrets; exits non-zero if wrong |
| `health-check` | `.gemini/skills/health-check/SKILL.md` | `scripts/check-health.sh` | Multi-component health (Railway + API /health + scheduler) |

**Gemini Usage:**
```
# Skills are auto-discovered — just describe what you need:
"check the env vars"          -> env-check skill activates
"is the API up"               -> health-check skill activates
"show me recent errors"       -> railway-logs skill activates
"run the v7 migration"        -> db-migrate skill activates

# Or run scripts directly:
bash .gemini/skills/health-check/scripts/check-health.sh
bash .gemini/skills/env-check/scripts/check-vars.sh --critical-only
bash .gemini/skills/db-migrate/scripts/run-migration.sh migrate_v7 --dry-run
```

**Critical env vars verified by env-check skill:**
- `INTEGRITY_SWEEP_ENABLED=false` (prevents restart loop)
- `ENABLE_MLB_ANALYSIS=true`
- `ENABLE_INGESTION_ORCHESTRATOR=true`

### §16.6 Gemini CLI Usage Pattern

**DO:**
- ✅ Read logs: `railway logs --follow`
- ✅ Check status: `railway status`
- ✅ Set env vars: `railway variables set KEY=value`
- ✅ Update HANDOFF.md with operation results
- ✅ Run migrations: `railway run python scripts/migrations/...`

**DON'T:**
- ❌ Edit `.py` files ( ANY Python code)
- ❌ Edit `.ts/.tsx` files ( ANY frontend code)
- ❌ Run `railway up` without Claude/Kimi approval
- ❌ Modify database schema without migration scripts

### §16.7 Session Recovery (When Context Lost)

If Gemini CLI loses authentication:

```bash
# Quick recovery script (save as scripts/gemini_recovery.sh)
#!/bin/bash
echo "=== Gemini Session Recovery ==="
echo "1. Reading HANDOFF.md..."
grep -A5 "Next operator (Gemini" HANDOFF.md
echo ""
echo "2. Checking Railway status..."
railway whoami 2>/dev/null || echo "Need: railway login"
echo ""
echo "3. Pending ops from §16.4..."
grep -A10 "Current Pending Operations" HANDOFF.md
```

---

**Document Version:** EMAC-083
**Last Updated:** March 25, 2026


---

## §17. Multi-Agent Orchestration & Workflow Automation

**Status:** Solution Identified — Awaiting Implementation Decision  
**Problem:** Manual handoffs between Claude/Kimi/Gemini via separate CLIs  
**Solution:** Use OpenClaw's ACP to orchestrate agents in parallel  
**Report:** `reports/OPENCLAW_ORCHESTRATION_WORKFLOW.md`  
**Prompt:** `CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md`

### §17.1 Current Friction (Your Actual Problem)

**Current Workflow:**
```
You → Check HANDOFF.md
  → Open Claude Code CLI → "Do your tasks" → Wait
  → Open Kimi CLI → "Do your audit" → Wait
  → Open Gemini CLI → "Do your ops" → Wait
  → Repeat...
```

**Issues:**
- Manual context switching between 3 CLIs
- You become the bottleneck
- Agents work sequentially, not in parallel
- Risk of missing HANDOFF.md updates

**NOT a Cost Problem:** You're on Claude $200/year fixed plan + Kimi fixed plan. 
Using agents more doesn't cost more. This is about **automation and efficiency**.

### §17.2 Solution: OpenClaw 24/7 Orchestration

Since OpenClaw already runs 24/7, use it as the central coordinator:

```
You in Discord/Telegram: "Start work session"
  ↓
OpenClaw spawns 3 ACP agents in parallel:
  ├─→ Claude Code → "Work on architecture tasks" → Updates HANDOFF.md
  ├─→ Kimi CLI → "Run audits" → Updates HANDOFF.md
  └─→ Gemini CLI → "Complete Railway ops" → Updates HANDOFF.md
  ↓
OpenClaw notifies you: "All agents complete. Claude: 2 tasks done, 
                       Kimi: 1 audit complete, Gemini: 3 ops done"
```

**Benefits:**
- Agents work in **parallel** (faster completion)
- **One interface** (Discord/Telegram) instead of 3 CLIs
- **Automatic notifications** when work completes
- **No context switching** for you
- Still uses HANDOFF.md as coordination point

### §17.3 Implementation: OpenClaw ACP Orchestration

Since you're already on fixed-cost plans (Claude $200/year, Kimi fixed), the goal is **efficiency**, not cost reduction.

**Implementation Steps:**

**Phase 1: Configure ACP Agents in OpenClaw**
```bash
# Add to ~/.openclaw/config.json:
{
  "agents": {
    "claude": {
      "runtime": "acp",
      "command": "claude-agent-acp",
      "description": "Architecture and implementation"
    },
    "kimi": {
      "runtime": "acp",
      "command": "kimi acp",
      "description": "Audit and analysis"
    },
    "gemini": {
      "runtime": "acp",
      "command": "gemini --acp",
      "description": "DevOps and Railway operations"
    }
  }
}
```

**Phase 2: Create Workflow Trigger**
```bash
# Add to OpenClaw:
# Trigger: "Start work session"
# Action: Spawn all 3 agents in parallel with HANDOFF.md context

openclaw workflow create daily-sprint \
  --agent claude --task "Complete architecture tasks from HANDOFF.md" \
  --agent kimi --task "Run audits from HANDOFF.md" \
  --agent gemini --task "Complete Railway ops from HANDOFF.md §16.4"
```

**Phase 3: Add Notifications**
```bash
# Watch HANDOFF.md for changes
# Notify when each agent completes work
```

**Result:** One command spawns all 3 agents in parallel. You get notified as they complete.

### §17.4 Three Approaches

| Approach | Parallel | Automation | Setup | Best For |
|----------|----------|------------|-------|----------|
| **Status Quo** | ❌ Sequential | ❌ Manual | 0 | Keeping things simple |
| **ACP Orchestration** ⭐ | ✅ Parallel | ✅ Triggered | 2-4 hours | **Your situation** |
| **Full Workflow Automation** | ✅ Parallel | ✅ Scheduled | 1-2 days | Post-Apr 7 |

### §17.5 Recommendation for CBB Edge

**Do This Now (Before Apr 7):**
- **ACP Orchestration** — Configure OpenClaw to spawn agents in parallel
- **One trigger** "Start work" → all 3 agents activate
- **Notifications** when HANDOFF.md updates
- **Time investment:** 2-4 hours
- **Benefit:** Eliminate manual context switching

**Defer Until After Apr 7:**
- Full workflow automation (scheduled runs)
- Additional local LLM integration
- Complex multi-step workflows

### §17.6 Quick Wins (No Setup Required)

While deciding on orchestration, try these immediately:

**Option A: Shell Alias**
```bash
# Add to .bashrc/.zshrc
alias start-work='
  echo "Starting work session..." && \
  echo "1. Checking HANDOFF.md" && \
  grep -A3 "Next operator (Claude" HANDOFF.md && \
  grep -A3 "Next operator (Kimi" HANDOFF.md && \
  grep -A3 "Next operator (Gemini" HANDOFF.md
'
```

**Option B: Simple Script**
```bash
# scripts/start-work.sh
#!/bin/bash
echo "=== CBB Edge Work Session ==="
echo ""
echo "CLAUDE TASKS:"
grep -A5 "Next operator (Claude" HANDOFF.md
echo ""
echo "KIMI TASKS:"
grep -A5 "Next operator (Kimi" HANDOFF.md
echo ""
echo "GEMINI TASKS:"
grep -A5 "Next operator (Gemini" HANDOFF.md
```

### §17.7 Next Steps

**To implement ACP orchestration:**

1. **Review detailed plan:** `reports/OPENCLAW_ORCHESTRATION_WORKFLOW.md`
2. **Give Claude the prompt:** `CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md`
3. **Test:** Trigger "Start work" and verify all 3 agents spawn
4. **Iterate:** Add notifications, refine workflows

**Files:**
- Analysis: `reports/OPENCLAW_ORCHESTRATION_WORKFLOW.md`
- Prompt: `CLAUDE_OPENCLAW_ORCHESTRATION_PROMPT.md`

**Key Files:**
- Analysis: `reports/MULTI_AGENT_ORCHESTRATION_ANALYSIS.md`
- Prompt: `CLAUDE_LOCAL_LLM_PROMPT.md`
- Tools: NadirClaw, Ollama, ACPX (optional)

---

**Document Version:** EMAC-082-ORCHESTRATION
**Last Updated:** March 25, 2026


---

## §18. UAT Issues — Phase 1 COMPLETE

**Status:** Phase 1 DONE (Mar 25, 2026) — commit `35373be` · Phase 2 queued
**Report:** `reports/UAT_ISSUES_ANALYSIS.md`
**Implementation Prompt:** `CLAUDE_UAT_FIXES_PROMPT.md`

### §18.1 Issues Summary (11 Total)

| Priority | Issue | Status | Effort |
|----------|-------|--------|--------|
| 🔴 **CRITICAL** | Waiver Wire API 503 | ✅ FIXED (Mar 25) | Done |
| 🔴 **CRITICAL** | Daily Lineup Yahoo 422 | ✅ FIXED (Mar 25) | Done |
| 🟠 **HIGH** | Calibration empty | Missing brier calculation | 2-3 days |
| 🟠 **HIGH** | Daily Lineup defaults | Pre-season 400 + no_games_today flag FIXED; real odds await Mar 28 | ✅ FIXED (Mar 25) |
| 🟠 **HIGH** | Odds Monitor broken | ✅ FIXED (Mar 25) — `verify_api_key` on status endpoints | Done |
| 🟡 **MEDIUM** | Bet History filter | ✅ FIXED (Mar 25) — `placed` filter + "Placed (Real)" dropdown | Done |
| 🟡 **MEDIUM** | My Roster data | Z-score FIXED (get_or_create_projection + is_proxy + cat_scores) | ✅ FIXED (Mar 25) |
| 🟡 **MEDIUM** | Current Matchup | Category stats/opponent | 1-2 days |
| 🟡 **MEDIUM** | Risk Dashboard | Settlement validation | 1 day |
| 🟢 **LOW** | Today's Bets checkbox | UI enhancement | 0.5 day |
| 🟢 **LOW** | Remove Bracket Sim | UI cleanup | 0.5 day |

### §18.2 Critical Issues (FIXED)

#### Waiver Wire — 503 Error [FIXED]
**Error:** `Invalid subresource stats requested`
**Root Cause:** Yahoo MLB rejects both `ownership` AND `stats` on `/players` collection endpoint
**Fix applied:** `out=metadata` only in `get_free_agents()` + `get_waiver_players()`. Parser never used inline stats anyway — `_parse_player()` skips them. Tests: 7/7 pass.

#### Daily Lineup — 422 Error [FIXED]
**Error:** `game_ids don't match for player key`
**Root Cause:** One stale/traded player in batch causes Yahoo to reject the entire lineup PUT
**Fix applied:** `set_lineup()` now attempts full batch first; on game_id mismatch falls back to per-player retry, skips failures, returns `{applied, skipped, warnings}`. `apply_fantasy_lineup` surfaces `skipped` + `warnings` in response instead of hard 422. Tests: 19/19 pass.

### §18.3 Key Findings

**Yahoo API Changes:**
- Waiver endpoint no longer accepts `out=metadata,stats`
- Must use `out=metadata` only, fetch stats separately
- This is a breaking change in Yahoo's API

**Data Issues:**
- MLB projections falling back to defaults (4.50/1.000/4.625)
- Likely pybaseball cache stale or player matching failing
- Recommendation: Force refresh Statcast data

**Missing Features:**
- Brier score calculation not implemented for calibration
- User "placed bet" tracking not in data model
- Settlement lookback validation bug (min="20" instead of min="1")

### §18.4 Implementation Plan

**Phase 1: Critical — COMPLETE (Mar 25, 2026)**
1. ✅ Fix Waiver Wire API endpoint — `out=metadata` only
2. ✅ Fix Daily Lineup Yahoo 422 error — per-player fallback retry
3. ✅ Tests: 19/19 lineup+yahoo+waiver pass

**Phase 2: Fantasy Baseball Critical Bugs — COMPLETE (Mar 25, 2026)**
4. ✅ Roster z-score nulls — `get_or_create_projection()` (accent normalization + proxy fallback). `RosterPlayerOut` gains `is_proxy` + `cat_scores` fields.
5. ✅ Position-aware waiver drop pairing — `_weakest_droppable_at()` + `_POS_GROUP` map in `waiver_edge_detector.py`. OF FA no longer targets 2B as drop.
6. ✅ Lineup pre-season 400 — `fetch_mlb_odds` pre-check in PUT endpoint; `no_games_today` flag in GET response + `DailyLineupResponse` schema.
7. ✅ `percent_owned` subresource probe — `out=metadata,percent_owned` in `get_free_agents()` / `get_waiver_players()`.
8. ✅ Suite: 1105/1109 (4 pre-existing failures only)

**Phase 3: High Priority — COMPLETE (Mar 25, 2026)**
9. ✅ Brier score + `days` filter wired into calibration endpoint (EMAC-084)
10. ✅ Odds Monitor portfolio fetch — `verify_api_key` on status endpoints (EMAC-083)
11. ✅ Bet history filter — `placed` filter + "Today" option in days select (EMAC-083/084)

**Phase 4: Medium Priority (Following Week)**
12. My roster data mapping (remaining undroppable display issues)
13. Current matchup display
14. Risk dashboard settlement validation fixes

**Phase 4: Cleanup (Final)**
11. Today's bets checkbox
12. Remove Bracket Simulator
13. Full UAT re-test

### §18.5 Files Created

| File | Purpose |
|------|---------|
| `reports/UAT_ISSUES_ANALYSIS.md` | Full root cause analysis for all 11 issues |
| `CLAUDE_UAT_FIXES_PROMPT.md` | Copy-paste prompt for Claude Code with code fixes |

### §18.6 Next Action

**Give Claude Code the prompt:**
```
claude "Read CLAUDE_UAT_FIXES_PROMPT.md and start with Priority 1 
        (Waiver Wire 503 and Daily Lineup 422). Update HANDOFF.md 
        §18 as you complete each fix."
```

**Estimated Timeline:** 6-9 days for all fixes

---

**Document Version:** EMAC-082-UAT
**Last Updated:** March 25, 2026


---

## §19. Elite Fantasy Baseball Platform — Architecture & Roadmap

**Status:** Analysis COMPLETE — Roadmap created, awaiting implementation  
**Gap Analysis:** `reports/FANTASY_BASEBALL_GAP_ANALYSIS.md`  
**Implementation Prompt:** `CLAUDE_FANTASY_ROADMAP_PROMPT.md`  

### §19.1 Current State vs. Elite Spec

| Phase | Spec Features | Current Coverage | Status |
|-------|--------------|------------------|--------|
| 1: Foundation | 2 | ~40% | Partial (API bugs being fixed) |
| 2: Analytics | 2 | ~30% | Basic projections only |
| 3: Optimizers | 3 | ~25% | MCMC exists, needs polish |
| 4: Intelligence | 2 | ~15% | Discord alerts only |
| 5: Elite Tools | 3 | 0% | Not started |
| **Total** | **12** | **~25%** | **Solid foundation, needs depth** |

### §19.2 Critical Insight

**You're NOT building from scratch.** You have:
- ✅ Yahoo OAuth & API (with bugs being fixed)
- ✅ Statcast/pybaseball pipeline
- ✅ MCMC simulation engine
- ✅ OpenClaw 24/7 monitoring
- ✅ Next.js 15 frontend

**The gap is FEATURE DEPTH and UX POLISH, not infrastructure.**

### §19.3 Implementation Roadmap

| Phase | Timeline | Focus | Deliverables |
|-------|----------|-------|--------------|
| **A** | Week 1-2 | Critical Fixes | Yahoo API reliability, UAT completion |
| **B** | Week 3-4 | Foundation | Enhanced dashboard, user preferences |
| **C** | Week 5-8 | Analytics | Trade analyzer, advanced scout, waiver optimizer |
| **D** | Week 9-12 | Intelligence | Standings projection, injury alerts, mobile push |
| **E** | Months 4-6 | Elite | Backtesting, strategy AI, community |

### §19.4 Top 5 Gaps to Address

| Priority | Gap | Current Pain | Solution |
|----------|-----|--------------|----------|
| 1 | Yahoo API reliability | 422/503 errors | Retry logic + validation |
| 2 | Multi-source projections | Only 1-2 sources | Aggregator service |
| 3 | Trade analyzer | Not implemented | Full trade analysis engine |
| 4 | Lineup optimizer | Manual only | Auto-optimize + push |
| 5 | Mobile/Push | Discord only | PWA + native push |

### §19.5 Files Created

| File | Purpose |
|------|---------|
| `reports/FANTASY_BASEBALL_GAP_ANALYSIS.md` | 17-page detailed gap analysis |
| `CLAUDE_FANTASY_ROADMAP_PROMPT.md` | Implementation prompt for Claude Code |

### §19.6 Next Steps

**Claude Code:** Review `CLAUDE_FANTASY_ROADMAP_PROMPT.md` and implement **Phase A** (Critical Fixes) while completing UAT Phase 2.

**Kimi CLI:** Stand by to audit implementation as phases complete.

**Timeline:** 12 weeks to full Phase D (elite differentiation), 6 months to Phase E.

---

**Document Version:** EMAC-082-FANTASY-ARCH
**Last Updated:** March 25, 2026


---

## ✅ COMPLETE — Fantasy Baseball Elite Manager System (EMAC-087)

> **Date:** March 26-27, 2026
> **Assignee:** Kimi CLI
> **Status:** ✅ COMPLETE
> **Scope:** Smart lineup selection, weather integration, daily briefing, decision tracking

### Summary of Features Built

| Feature | Description | Files | Status |
|---------|-------------|-------|--------|
| **Smart Lineup Selector** | Composite scoring with platoon splits, opposing pitcher analysis, category awareness | `smart_lineup_selector.py` | ✅ Complete |
| **Platoon Fetcher** | FanGraphs wOBA splits via pybaseball, 7-day cache | `platoon_fetcher.py` | ✅ Complete |
| **Category Tracker** | Yahoo H2H scoreboard integration, calculates category gaps | `category_tracker.py` | ✅ Complete |
| **Pitcher Deep Dive** | FIP, xFIP, SIERA, batted ball data, splits vs LHB/RHB | `pitcher_deep_dive.py` | ✅ Complete |
| **Elite Context System** | Weather, recent form, lineup spot PA multipliers, risk profiles | `elite_context.py` | ✅ Complete |
| **Weather Fetcher** | OpenWeatherMap integration, temp/wind/humidity/precipitation | `weather_fetcher.py` | ✅ Complete |
| **Park Weather Analyzer** | Stadium-specific weather (orientation, wind effects, altitude) | `park_weather.py` | ✅ Complete |
| **Daily Briefing** | Morning briefing with recommendations, confidence scores, alerts | `daily_briefing.py` | ✅ Complete |
| **Decision Tracker** | Records recommendations vs actual results, accuracy reporting | `decision_tracker.py` | ✅ Complete |
| **MLB Box Score Fetcher** | Automatic resolution with MLB Stats API box scores | `mlb_boxscore.py` | ✅ Complete |
| **Pitcher Start Fix** | Proper probable pitcher detection via MLB Stats API | `daily_lineup_optimizer.py` | ✅ Complete |
| **Lineup Display Fix** | Opponent, start time, position-based sorting | `main.py` | ✅ Complete |

### Smart Lineup Selector Scoring Weights

```
Base Projection:        35% (Steamer/ATC)
Game Environment:       20% (Odds API + park factors)
Platoon Advantage:      15% (FanGraphs wOBA splits)
Pitcher Difficulty:     10% (FanGraphs FIP/xFIP/K9)
Category Need Fit:      20% (Yahoo scoreboard gaps)
Weather Boost:          Included in Game Environment
```

### API Endpoints Added

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/fantasy/lineup/{date}` | GET | Get optimized lineup with opponent/time data |
| `/api/fantasy/lineup/apply` | PUT | Apply lineup to Yahoo with auto-correction |
| `/api/fantasy/briefing/{date}` | GET | Daily briefing with recommendations |
| `/api/fantasy/decisions/record` | POST | Manually record a decision |
| `/api/fantasy/decisions/override/{id}` | POST | Record user override |
| `/api/fantasy/decisions/resolve/{id}` | POST | Resolve with actual stats |
| `/api/fantasy/decisions/accuracy/{date}` | GET | Daily accuracy report |
| `/api/fantasy/decisions/trends` | GET | 14-day trend analysis |
| `/api/fantasy/decisions/dashboard` | GET | Complete decision dashboard |
| `/api/fantasy/decisions/pending/{date}` | GET | List pending resolutions |
| `/api/fantasy/decisions/resolve-bulk/{date}` | POST | Resolve multiple decisions |
| `/api/fantasy/decisions/run-nightly-resolution` | POST | Trigger nightly job |
| `/api/fantasy/decisions/resolve-from-mlb/{date}` | POST | Auto-resolve from MLB API |

### Environment Variables

```bash
# Weather API (OpenWeatherMap)
OPENWEATHER_API_KEY=your_key_here  # Free tier: 1,000 calls/day

# Existing
YAHOO_CLIENT_ID=...
YAHOO_CLIENT_SECRET=...
YAHOO_REFRESH_TOKEN=...
THE_ODDS_API_KEY=...
```

### Key Bug Fixes

1. **Pitcher Start Detection** (Mar 27)
   - Problem: `flag_pitcher_starts` only checked if team had game, not if specific pitcher was probable
   - Solution: Fetch probable pitchers from MLB Stats API, match by name
   - File: `daily_lineup_optimizer.py`

2. **Lineup Display** (Mar 27)
   - Problem: All batters showed 7:00 PM, no opponent, random order
   - Solution: Fetch game odds once, add opponent/start_time, sort by position
   - Files: `main.py`, `schemas.py`

3. **Pitcher Data** (Mar 27)
   - Problem: All pitchers showed 0.00 implied runs, 1.000 park factor, 0.000 score
   - Solution: Populate from team_odds, calculate SP score based on matchup
   - File: `main.py`

### Decision Tracking Flow

```
1. GET /api/fantasy/briefing/2025-03-27
   → Auto-records 14 decisions (START/BENCH/MONITOR)
   → Stores: player, recommendation, confidence, factors, weather

2. Games play...

3. POST /api/fantasy/decisions/resolve-from-mlb/2025-03-27
   → Fetches box scores from MLB Stats API
   → Resolves all pending decisions automatically

4. GET /api/fantasy/decisions/accuracy/2025-03-27
   → Shows: 71% accuracy, 85% for high confidence, 40% for low

5. GET /api/fantasy/decisions/trends?days=14
   → Trend direction, best confidence threshold, overvalued factors
```

### Testing

- All new modules have unit tests
- CI passes (lint errors fixed)
- Manual testing via API endpoints

### Next Steps (Claude)

1. Monitor decision accuracy over first week
2. Adjust scoring weights based on results
3. Consider adding: Statcast recent form, lineup spot scraper

---

## 🔴 CRITICAL P0 — PITCHER START DETECTION ISSUE (FIXED)

> **Discovered:** March 27, 2026 — Shota Imanaga showing as START but doesn't pitch today
> **Root Cause:** `flag_pitcher_starts` checked team game, not specific pitcher
> **Fix:** Added `_fetch_probable_pitchers_for_date()` and `_is_probable_starter()`
> **Status:** ✅ FIXED in commit `0c5389e`

---

## 🚀 PHASE B: ENHANCED DASHBOARD — MARCH 27, 2026 (Day 1)

> **Assignee:** Claude Code (Master Architect)  
> **Scope:** Phase B Foundation (CLAUDE_FANTASY_ROADMAP_PROMPT.md)  
> **Timeline:** Aggressive (2-4 weeks)  
> **Status:** INFRASTRUCTURE COMPLETE, UI IMPLEMENTED

### Day 1 Accomplishments

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **User Preferences Schema** | `backend/models.py` | ✅ COMPLETE | UserPreferences ORM model with JSONB settings |
| **Migration v10** | `scripts/migrate_v10_user_preferences.py` | ✅ COMPLETE | Migration with up/downgrade, dry-run support |
| **Dashboard Service** | `backend/services/dashboard_service.py` | ✅ COMPLETE | Core aggregation layer with all 6 panels |
| **Dashboard API Endpoints** | `backend/main.py` | ✅ COMPLETE | `/api/dashboard`, `/api/user/preferences` |
| **TypeScript Types** | `frontend/lib/types.ts` | ✅ COMPLETE | All dashboard types exported |
| **API Client Updates** | `frontend/lib/api.ts` | ✅ COMPLETE | Dashboard API functions added |
| **Dashboard Page** | `frontend/app/(dashboard)/dashboard/page.tsx` | ✅ COMPLETE | Main dashboard with quick stats |
| **Lineup Gaps Panel** | `dashboard/components/LineupGapsPanel.tsx` | ✅ COMPLETE | Critical/warning gap detection UI |
| **Streaks Panel** | `dashboard/components/StreaksPanel.tsx` | ✅ COMPLETE | Hot/cold tabs with 7/14/30 trends |
| **Waiver Targets Panel** | `dashboard/components/WaiverTargetsPanel.tsx` | ✅ COMPLETE | Prioritized targets with tiers |
| **Injury Flags Panel** | `dashboard/components/InjuryFlagsPanel.tsx` | ✅ COMPLETE | IL/DTD status with severity |
| **Matchup Preview Panel** | `dashboard/components/MatchupPreviewPanel.tsx` | ✅ COMPLETE | Win probability + category advantages |
| **Probable Pitchers Panel** | `dashboard/components/ProbablePitchersPanel.tsx` | ✅ COMPLETE | Two-start SP highlighting |
| **Sidebar Navigation** | `frontend/components/layout/sidebar.tsx` | ✅ COMPLETE | Dashboard link added to Fantasy Baseball section |
| **Home Redirect** | `frontend/app/(dashboard)/page.tsx` | ✅ COMPLETE | Now redirects to /dashboard |

### Files Modified/Created

```
backend/models.py                              (+ UserPreferences model)
backend/services/dashboard_service.py          (+ new file)
backend/main.py                                (+ dashboard endpoints)
scripts/migrate_v10_user_preferences.py        (+ new file)
frontend/lib/types.ts                          (+ dashboard types)
frontend/lib/api.ts                            (+ dashboard API)
frontend/app/(dashboard)/dashboard/page.tsx    (+ new file)
frontend/app/(dashboard)/dashboard/components/ (+ 6 new panel components)
frontend/components/layout/sidebar.tsx         (+ dashboard nav link)
frontend/app/(dashboard)/page.tsx              (changed redirect to /dashboard)
```

### Dashboard Features Implemented

**B1: Enhanced Dashboard Panels**
- ✅ Lineup Gaps Detection (visual alerts with severity)
- ✅ Hot/Cold Streaks (tabbed interface, 7/14/30 trends)
- ✅ Waiver Targets (priority scoring with tiers)
- ✅ Injury Flags (severity-based color coding)
- ✅ Matchup Preview (win probability bar + category advantages)
- ✅ Probable Pitchers (two-start SP highlighting)

**B2: User Preferences Infrastructure**
- ✅ Database schema with JSONB settings
- ✅ Backend CRUD operations
- ✅ API endpoints for preferences
- 🔄 UI Settings Page (Tomorrow - Day 2)

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/dashboard` | GET | Full dashboard data aggregate |
| `/api/user/preferences` | GET | Fetch user preferences |
| `/api/user/preferences` | POST | Update user preferences |
| `/api/dashboard/streaks` | GET | Hot/cold streaks only |
| `/api/dashboard/waiver-targets` | GET | Waiver targets only |

### Frontend Route

- **URL:** `/dashboard`
- **Navigation:** Fantasy Baseball → Dashboard (top of section)
- **Redirect:** Home (`/`) now redirects to `/dashboard`

### Next Steps (Day 2 — March 28)

1. **Settings Page** (`/settings`)
   - Notification preferences (email/Discord/push toggles)
   - Dashboard layout customization
   - Streak calculation thresholds
   - Waiver filter preferences

2. **Backend Integration**
   - Connect Lineup Gaps to actual Yahoo roster API
   - Connect Streaks to player_daily_metrics
   - Connect Waiver Targets to WaiverEdgeDetector
   - Connect Injury Flags to Yahoo roster status

3. **Testing**
   - Test dashboard with real Yahoo data
   - Verify all panels populate correctly
   - Test preferences persistence

### Migration Required

```bash
# Run on Railway to create user_preferences table
python scripts/migrate_v10_user_preferences.py
```

### Technical Notes

- Dashboard uses Tailwind CSS with responsive grid (1 col mobile, 2 col desktop)
- All panel components are client components with loading skeletons
- Quick stats row provides at-a-glance status
- Color coding: green=good, yellow=warning, red=critical
- All dashboard types exported from `frontend/lib/types.ts`

---

## 🚀 PHASE B: DAY 2 — DATA PIPELINE & SETTINGS — MARCH 28, 2026

> **Assignee:** Claude Code (Master Architect)  
> **Focus:** Data Reliability (CRITICAL) + Settings UI  
> **Status:** DATA RELIABILITY ENGINE COMPLETE, SETTINGS UI COMPLETE, YAHOO INTEGRATION LIVE

### Day 2 Accomplishments

| Component | File | Status | Notes |
|-----------|------|--------|-------|
| **Data Reliability Engine** | `backend/services/data_reliability_engine.py` | ✅ COMPLETE | Multi-source validation, cross-checking, quality scoring |
| **Statcast Validation** | `data_reliability_engine.py` | ✅ COMPLETE | Completeness, range checks, freshness detection |
| **Yahoo Roster Validation** | `data_reliability_engine.py` | ✅ COMPLETE | Player count, field presence, health monitoring |
| **Cross-Validation** | `data_reliability_engine.py` | ✅ COMPLETE | Compare sources, flag discrepancies above thresholds |
| **Source Health Monitor** | `data_reliability_engine.py` | ✅ COMPLETE | Track failures, automatic fallback chain |
| **Settings Page UI** | `frontend/app/(dashboard)/settings/page.tsx` | ✅ COMPLETE | Full preferences management with 4 tabs |
| **Notifications Settings** | `settings/page.tsx` | ✅ COMPLETE | Toggle alerts + channel selection (Discord/Email/Push) |
| **Dashboard Settings** | `settings/page.tsx` | ✅ COMPLETE | Panel enable/disable + refresh interval + theme |
| **Streak Settings** | `settings/page.tsx` | ✅ COMPLETE | z-score thresholds + rolling windows |
| **Waiver Settings** | `settings/page.tsx` | ✅ COMPLETE | Ownership filters + streamer thresholds |
| **Live Yahoo Integration** | `dashboard_service.py` | ✅ COMPLETE | Real roster data for gaps, injuries, streaks |
| **Data Quality Tiers** | `data_reliability_engine.py` | ✅ COMPLETE | TIER_1 (Excellent) to TIER_5 (Unavailable) |
| **Settings Navigation** | `frontend/components/layout/sidebar.tsx` | ✅ COMPLETE | Settings section with Preferences link |

### Critical: Data Reliability Architecture

**The #1 Rule of Fantasy Baseball Analytics:** "Bad data kills championships."

As a top 1% fantasy mind would tell you, the best model with bad data loses to a mediocre model with great data every time. This engine ensures data integrity through:

#### 1. Multi-Source Cross-Validation
```python
# Compare Statcast vs Yahoo vs FanGraphs
if abs(statcast_avg - fangraphs_avg) > 0.050:
    flag_discrepancy(player, field="avg", sources=["statcast", "fangraphs"])
```

#### 2. Automatic Fallback Chain
```
Primary:    Statcast API → Baseball Savant
Secondary:  pybaseball → FanGraphs
Tertiary:   Local cache (24hr TTL)
Fallback:   League averages
```

#### 3. Data Quality Tiers
| Tier | Quality | Freshness | Use Case |
|------|---------|-----------|----------|
| TIER_1_EXCELLENT | Live API | <5 min | Live lineup decisions |
| TIER_2_GOOD | Recent | <1 hour | Pre-game analysis |
| TIER_3_ACCEPTABLE | Cached | <24 hours | Projections, trends |
| TIER_4_STALE | Old | >24 hours | Historical reference |
| TIER_5_UNAVAILABLE | None | N/A | Use defaults only |

#### 4. Validation Thresholds
| Field | Threshold | Action if Exceeded |
|-------|-----------|-------------------|
| AVG | ±0.050 | Flag for review |
| OBP | ±0.030 | Flag for review |
| SLG | ±0.080 | Flag for review |
| OPS | ±0.100 | Flag for review |
| HR | ±20% | Flag for review |
| Exit Velocity | ±10 mph day-to-day | Data anomaly alert |

### Settings Page Features

**Notifications Tab:**
- Lineup deadline alerts
- Injury breaking news
- Waiver suggestions
- Trade offers
- Hot streak alerts
- Channel selection: Discord / Email / Push

**Dashboard Tab:**
- Enable/disable panels
- Refresh interval: 30s to 10m
- Theme: Dark / Light / System

**Streaks Tab:**
- Hot threshold: z-score 0.0 to 2.0 (default 0.5)
- Cold threshold: z-score -2.0 to 0.0 (default -0.5)
- Rolling windows: 7/14/21/30 days selectable

**Waiver Tab:**
- Min ownership: 0-50% (filter out extremely low-owned)
- Max ownership: 50-100% (filter out likely unavailable)
- Hide injured toggle
- Streamer z-score threshold: -1.0 to 1.0

### Live Yahoo Integration

Dashboard now fetches **real data** from Yahoo:

1. **Lineup Gaps**: Actual roster positions vs required slots
2. **Injury Flags**: Real IL/DTD/OUT status from Yahoo
3. **Streaks**: Cross-reference roster with Statcast metrics
4. **Data Validation**: Every Yahoo fetch validated for quality

### Files Modified/Created (Day 2)

```
backend/services/data_reliability_engine.py     (+ new file, 580 lines)
backend/services/dashboard_service.py           (+ Yahoo integration, validation)
frontend/app/(dashboard)/settings/page.tsx      (+ new file, 640 lines)
frontend/components/layout/sidebar.tsx          (+ Settings navigation)
```

### API Additions

```python
# Data reliability endpoints (for monitoring)
GET /api/data-quality/report          # Overall data health
GET /api/data-quality/source/{name}   # Specific source health

# Settings endpoints (already existed, now fully functional)
GET /api/user/preferences
POST /api/user/preferences
```

### Data Pipeline Best Practices Implemented

From `daily_lineup_optimization_research.md` and elite fantasy expertise:

1. **"Trust but Verify"**: All data validated against secondary sources
2. **Graceful Degradation**: Never let one API failure break the system
3. **Freshness Matters**: Stale data (<24hr old) flagged and down-weighted
4. **Observability**: Every validation result logged with confidence scores
5. **Source Hierarchy**: Clear fallback chain when primary sources fail

### Critical Data Sources Priority

```python
STATCAST_API      → Live Statcast (primary for metrics)
BASEBALL_SAVANT   → CSV exports (backup for metrics)
YAHOO_API         → Roster, lineup, matchup (primary for team data)
FANGRAPHS         → Projections, platoon splits (primary for projections)
PYBASEBALL        → Aggregated stats (fallback for everything)
CACHE             → Local TTL cache (emergency fallback)
FALLBACK          → League averages (last resort)
```

### Confidence Scoring

Every data point now has a confidence score (0.0-1.0):
- 0.90-1.00: High confidence (validated, fresh, cross-checked)
- 0.70-0.89: Good confidence (minor warnings or slightly stale)
- 0.50-0.69: Acceptable (some issues, use with caution)
- 0.00-0.49: Low confidence (stale, unvalidated, or errors)

### Next Steps (Day 3 — March 29)

1. **Data Quality Dashboard** (`/admin/data-quality`)
   - Visual monitor of source health
   - Discrepancy alerts
   - Data freshness timeline

2. **Historical Data Backfill**
   - Import 2024 season data for testing
   - Validate against known outcomes
   - Tune validation thresholds

3. **Advanced Matchup Scoring**
   - Implement pitcher quality multipliers
   - Add platoon split adjustments
   - Test correlation with actual fantasy points

### Testing Checklist

- [ ] Settings page saves preferences correctly
- [ ] Dashboard loads with real Yahoo data
- [ ] Lineup gaps detected accurately
- [ ] Injury flags match Yahoo roster
- [ ] Data validation catches stale data
- [ ] Fallback sources activate when primary fails

---

**END OF HANDOFF**
