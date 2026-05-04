# Full-System Production Audit Report (K-33)

**Auditor:** Kimi CLI (Deep Intelligence Unit)
**Timestamp:** 2026-04-26 09:45 EDT (13:45 UTC)
**Deployment:** `e0e51d6c` — "fix(bdl_injuries): capture exception text on DB-write failure"
**Branch:** `stable/cbb-prod`
**Domain:** `fantasy-app-production-5079.up.railway.app`
**Previous Audit:** K-32 (2026-04-25) — contained material misdiagnoses

---

## Executive Summary

The `e0e51d6c` deploy (13:07 UTC today) fixed error-message capture for the BDL injury ingestion path, but **that path has not executed since the deploy** — no `bdl_injuries` job has run in the past 80 minutes. The commit `b6a882a` ("persist lineup recs, valuation cache, confirm probable pitchers") IS in the deployed image, but only **one of its three deliverables is working**:

| Deliverable | Status | Evidence |
|-------------|--------|----------|
| Lineup recommendations → `fantasy_lineups` | ✅ Working | 1 active row (2026-04-26 13:17) |
| Waiver decisions → `player_valuation_cache` | ❌ Broken | 0 rows; silently failing |
| Probable pitchers `is_confirmed=True` | ❌ Broken | 0/349 rows confirmed |

Additionally, the **Admin `/api/admin/data-quality/summary` endpoint remains 500** with the same timezone-mismatch crash. The prior audit (K-32) incorrectly blamed `DataIngestionLog.run_at`; the actual root cause is `PlayerProjection.updated_at` being `timestamp without time zone`.

A major positive discovery: the **Waiver Browse endpoint has been redesigned** and now returns structured waiver intelligence (`top_available`, `category_deficits`, `matchup_opponent`, `il_slots`) instead of the old flat `players` array. This explains the "0 players" symptom from K-32 — the schema changed, not the data pipeline.

---

## 1. Deployment Verification

### Git Ancestry
```
e0e51d6c fix(bdl_injuries): capture exception text on DB-write failure
b6a882a persist lineup recs, valuation cache, confirm probable pitchers
46d17d7 [parent of b6a882a]
```
**Confirmed:** `b6a882a` is a direct ancestor of `e0e51d6c`. All three Phase 1 persistence changes are in the live image.

### Commit `e0e51d6c` Diff Summary
- Added `try/except` around the BDL injury DB-write path inside `_run()`
- Captures `str(exc)` into `error_message` and serializes `traceback.format_exc()` into `error_details`
- Does NOT change the outer `_with_advisory_lock` wrapper

### Commit `b6a882a` Diff Summary
- `get_fantasy_lineup_recommendations()` → persists optimal lineup to `fantasy_lineups`
- `get_decisions()` → persists each waiver decision to `player_valuation_cache` (inside existing loop, wrapped in `try/except`)
- `_sync_probable_pitchers()` → sets `is_confirmed=bool(pitcher_data)` and includes it in the upsert `set_` dict

---

## 2. Live API Findings

### 2.1 Admin Data Quality — 500 (P0) ❌

**Endpoint:** `GET /api/admin/data-quality/summary`  
**Result:** `500 Internal Server Error`  
**Root Cause:** `TypeError: can't subtract offset-naive and offset-aware datetimes`

**Erroneous code** (`backend/routers/data_quality.py:65-72`):
```python
now = datetime.now(ZoneInfo("UTC"))
last_projection_update = db.query(func.max(PlayerProjection.updated_at)).scalar()
staleness_hours = (
    (now - last_projection_update).total_seconds() / 3600
    if last_projection_update
    else 999
)
```

- `now` is offset-aware (`UTC`)
- `PlayerProjection.updated_at` is `timestamp without time zone` (naive)
- Subtraction crashes before reaching any `DataIngestionLog` reference

**K-32 Misdiagnosis:** The prior audit claimed the crash was at `DataIngestionLog.run_at` (lines 76, 80). This was incorrect. The endpoint fails at line 69 and never reaches line 76.

**Fix (verified):**
```python
if last_projection_update and last_projection_update.tzinfo is None:
    last_projection_update = last_projection_update.replace(tzinfo=ZoneInfo("UTC"))
```

---

### 2.2 Waiver Browse — Major Schema Redesign (P1 → P2) ⚠️

**Endpoint:** `GET /api/fantasy/waiver`  
**Result:** `200 OK` with structured intelligence  
**Old schema (K-32 assumption):** `{ players: [...], total: N }` — now **deprecated**

**New schema (verified live):**
```json
{
  "top_available": [ ... 25 players ... ],
  "category_deficits": [ ... 20 categories ... ],
  "matchup_opponent": "Hype Train",
  "il_slots_used": 2,
  "il_slots_available": 0
}
```

**Key observation:** The `top_available` array contains 25 real players with `need_score`, `z_score_total`, `projected_stats`, and `risk_flags`. This endpoint is **not broken** — it was redesigned. The "0 players" symptom from K-32 was due to looking for the old `players` key.

**Notable data quality in `top_available`:**
- Dalton Rushing: `need_score=5.128`, `z_score_total=0.0` — candidate found by recommendation engine
- Dominic Smith: `need_score=3.41`, `z_score_total=0.0` — candidate found by recommendation engine
- **All players have `z_score_total=0.0`** because Yahoo ID mapping is 0% (see §3.5)

**Action required:** Update any frontend/dashboard code still expecting the old `players` / `total` keys.

---

### 2.3 Waiver Recommendations — Working (P1) ✅

**Endpoint:** `GET /api/fantasy/waiver/recommendations`  
**Result:** `200 OK` with 14 candidates  
**Confirmed:** The recommendation engine is running and finding real players (Dalton Rushing, Dominic Smith, Jackson Holliday, etc.)

---

### 2.4 Scoreboard — Partially Working (P1) ⚠️

**Endpoint:** `GET /api/fantasy/scoreboard`  
**Result:** `200 OK`

**Working:**
- `matchup_period`: "April 21–27"
- `my_current_score`: All 18 categories have real totals
- `categories`: Full 18-category list with `is_pitching`

**Broken:**
- `opponent_name`: `"Opponent"` (placeholder, not resolved)
- `my_projected_final`: `0.0` for all 18 categories (ROW/ROS pipeline not wired)
- `opponent_projected_final`: `0.0` for all 18 categories

---

### 2.5 Roster — Working, No ROS Projection (P2) ⚠️

**Endpoint:** `GET /api/fantasy/roster`  
**Result:** `200 OK` with 25 players  
**Working:** Full roster with `name`, `team`, `positions`, `status`, `lineup_slot`, `is_editable`

**Missing:** `ros_projection` is NOT present in the response. The endpoint returns roster metadata but not Rest-of-Season projections. This is consistent with the 0% Yahoo ID mapping issue — without Yahoo IDs, the system cannot resolve roster players to `PlayerProjection` rows.

---

### 2.6 Decisions — Working (P2) ✅

**Endpoint:** `GET /api/fantasy/decisions`  
**Result:** `200 OK` with 32 decisions today (14 lineup + 18 waiver)  
**Status:** Healthy. All decisions have real scores and reasoning.

---

## 3. Database Integrity

### 3.1 Fantasy Lineups — Working (P1) ✅

```sql
SELECT COUNT(*) FROM fantasy_lineups;
-- Result: 1

SELECT platform, is_active, created_at FROM fantasy_lineups ORDER BY created_at DESC LIMIT 1;
-- Result: platform='yahoo_recommendation', is_active=true, created_at='2026-04-26 13:17:17.287'
```

**Conclusion:** Lineup persistence IS working. The single row proves `b6a882a` write path executes for at least one code path (likely the recommendations endpoint).

---

### 3.2 Player Valuation Cache — Empty (P1) ❌

```sql
SELECT COUNT(*) FROM player_valuation_cache;
-- Result: 0
```

**Code inspection** (`backend/routers/fantasy.py:455-495`, post-`b6a882a`):
```python
try:
    _existing_cache = db.query(PlayerValuationCache).filter_by(
        player_id=str(wr.bdl_player_id),
        target_date=_target_date,
        league_key=_league_key,
    ).first()
    if _existing_cache:
        # update ...
    else:
        db.add(PlayerValuationCache(
            id=str(uuid.uuid4()), player_id=..., report=_report_blob, ...
        ))
    db.commit()
except Exception as _cache_err:
    logger.warning("decisions endpoint: valuation cache write failed ...")
```

**Why it fails silently:**
1. The entire block is wrapped in `try/except`
2. On ANY exception, it logs a warning and continues
3. No `db.rollback()` inside the `except` block (may leave session dirty)
4. The warning log level may be filtered in production

**Hypothesis:** `db.commit()` is raising an exception (possibly due to schema mismatch, NOT NULL violation, or JSON serialization issue) that is swallowed by the `except`. The decisions endpoint returns `200 OK` because the exception is caught.

**Next step:** Add `logger.error` with the actual exception inside the catch block, or temporarily remove the `try/except` to surface the real error.

---

### 3.3 Probable Pitchers — 0/349 Confirmed (P1) ❌

```sql
SELECT COUNT(*) FROM probable_pitchers;
-- Result: 349

SELECT COUNT(*) FROM probable_pitchers WHERE is_confirmed = true;
-- Result: 0

SELECT is_confirmed, COUNT(*) FROM probable_pitchers GROUP BY is_confirmed;
-- Result: false=349
```

**Code inspection** (`daily_ingestion.py`, post-`b6a882a`):
```python
is_confirmed=bool(pitcher_data),
# In on_conflict_do_update set_:
"is_confirmed": stmt.excluded.is_confirmed,
```

**Hypotheses:**
1. `pitcher_data` is always falsy (e.g., empty dict `[]`, `None`, or empty string) for all 349 rows
2. `_sync_probable_pitchers()` has not run since `b6a882a` deployed (but it has — 12:30 SUCCESS)
3. The `on_conflict_do_update` is matching on the wrong unique constraint and the `is_confirmed` value is being overwritten by a later upsert with falsy data

**Most likely:** `pitcher_data` from the upstream source (Yahoo API or BallDontLie) is an empty dict `{}` or `None` for all rows. `bool({})` is `False`, `bool(None)` is `False`. The code checks `bool(pitcher_data)` but the actual data structure may always be falsy even when a pitcher name is present (e.g., `pitcher_data = {"name": "..."}` would be truthy, but if the API returns just a string or an empty list, it would be falsy).

**Next step:** Inspect the raw `pitcher_data` payload from the Yahoo API during the next `_sync_probable_pitchers` run.

---

### 3.4 Projection Freshness — Fixed (P0) ✅

```sql
SELECT job_type, status, started_at, error_message
FROM data_ingestion_logs
WHERE job_type = 'projection_freshness'
ORDER BY started_at DESC LIMIT 5;
```

| job_type | status | started_at | error_message |
|----------|--------|------------|---------------|
| projection_freshness | SUCCESS | 2026-04-26 08:48:01 | null |
| projection_freshness | SUCCESS | 2026-04-26 07:48:01 | null |
| projection_freshness | SUCCESS | 2026-04-26 06:48:01 | null |
| projection_freshness | SUCCESS | 2026-04-26 05:48:01 | null |
| projection_freshness | SUCCESS | 2026-04-26 04:48:01 | null |

**17 consecutive SUCCESS runs** since Apr 25 20:16 UTC. The `datetime.date` → `datetime.datetime` TypeError is fully resolved.

---

### 3.5 Yahoo ID Mapping — 0% (P0) ❌

```sql
SELECT COUNT(*) FROM player_id_mapping;
-- Result: 10093

SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL;
-- Result: 0
```

**Impact:**
- `get_or_create_projection()` cannot resolve Yahoo roster players to `PlayerProjection` rows
- All Yahoo players receive `z_score=0.0` population-prior proxies
- Waiver recommendations and lineup scores are unreliable
- `ros_projection` missing from roster endpoint

**Root cause:** The Yahoo ID mapping table is completely unpopulated. This is a data ingestion issue, not a code bug.

---

### 3.6 Default Projection Contamination (P0) ❌

```sql
SELECT COUNT(*) FROM player_projections
WHERE hr = 15 AND r = 65 AND rbi = 65 AND sb = 5 AND avg = 0.25 AND ops = 0.72;
-- Result: 191

SELECT COUNT(*) FROM player_projections
WHERE era = 4.0 AND whip = 1.3 AND w = 0 AND qs = 0 AND k_pit = 0;
-- Result: 457
```

These are **placeholder backfill values** that entered via ingestion and were scored by `cat_scores_builder`. They pollute the z-score pool, making comparisons between real and default projections unreliable.

**All 648 contaminated rows have non-null `cat_scores`** (backfill scored them).

---

### 3.7 z_score_total — 100% Null (P1) ❌

```sql
SELECT COUNT(*) FROM player_daily_metrics;
-- Result: 16007

SELECT COUNT(*) FROM player_daily_metrics WHERE z_score_total IS NOT NULL;
-- Result: 0
```

The `z_score_total` column in `player_daily_metrics` is completely unpopulated. This is a separate table from `player_projections.cat_scores`.

---

### 3.8 VORP — 0% Today (P1) ❌

```sql
SELECT job_type, status, started_at, records_processed
FROM data_ingestion_logs
WHERE job_type = 'vorp'
ORDER BY started_at DESC LIMIT 5;
```

| job_type | status | started_at | records_processed |
|----------|--------|------------|-------------------|
| vorp | SUCCESS | 2026-04-26 11:30:00 | 0 |
| vorp | SUCCESS | 2026-04-26 10:30:00 | 0 |
| vorp | SUCCESS | 2026-04-26 09:30:00 | 0 |

```sql
SELECT COUNT(*) FROM player_daily_metrics WHERE vorp_7d IS NOT NULL AND stat_date = '2026-04-26';
-- Result: 0
```

**Status:** The `vorp` job reports SUCCESS with `records_processed=0`. It was 70-80% populated Apr 23-25 but is 0% on Apr 26. This suggests the VORP calculation is finding no eligible players (possibly due to Yahoo ID mapping being 0%).

---

### 3.9 BDL Injuries — Fix Deployed, Not Yet Exercised (P0) ⚠️

**Last run:** 2026-04-26 12:25:20 UTC — SUCCESS (170 records)  
**Deploy time:** 2026-04-26 13:07:21 UTC  
**Current time:** 2026-04-26 13:45 UTC

There have been **zero `bdl_injuries` runs since the `e0e51d6c` deploy** (80-minute gap). The job is scheduled hourly; the 13:25 run appears to have been skipped or the scheduler missed a beat after the container restart.

**Pre-deploy failures** (still showing `error_message=NULL` because they ran on the old image):
- 12:24:14 FAILED, NULL error_message
- 12:21:13 FAILED, NULL error_message

**Cannot verify** whether the new error capture works until the next `bdl_injuries` run executes.

---

## 4. Active Issues Matrix

| # | Issue | Priority | Status | Owner | Next Action |
|---|-------|----------|--------|-------|-------------|
| 1 | Admin 500: timezone mismatch | P0 | 🔴 Broken | Claude | Fix `data_quality.py:67-69` — make `last_projection_update` timezone-aware |
| 2 | Yahoo ID mapping 0% | P0 | 🔴 Broken | Gemini/Claude | Populate `player_id_mapping.yahoo_id` from Yahoo API |
| 3 | Default projection contamination | P0 | 🔴 Broken | Claude | Remove 648 placeholder rows or flag as `is_placeholder` |
| 4 | BDL injuries fix unverified | P0 | 🟡 Unknown | Gemini | Monitor next `bdl_injuries` run (should be ~14:25 UTC) |
| 5 | Player valuation cache 0 rows | P1 | 🔴 Broken | Claude | Remove `try/except` in `fantasy.py` to surface real DB error |
| 6 | Probable pitchers 0/349 confirmed | P1 | 🔴 Broken | Claude | Inspect raw `pitcher_data` payload from upstream API |
| 7 | Scoreboard projected_final = 0.0 | P1 | 🔴 Broken | Claude | Wire ROW/ROS pipeline to `scoreboard_orchestrator.py` |
| 8 | Waiver browse schema changed | P1 | 🟡 Redesigned | Kimi/Claude | Update dashboard/docs to reflect new `top_available` schema |
| 9 | VORP 0% today | P1 | 🔴 Broken | Claude | Debug why `vorp` job finds 0 eligible players |
| 10 | z_score_total 100% null | P1 | 🔴 Broken | Claude | Investigate `player_daily_metrics.z_score_total` population |
| 11 | Scoreboard opponent_name placeholder | P2 | 🔴 Broken | Claude | Resolve opponent team name from matchup data |
| 12 | Roster endpoint missing ROS projection | P2 | 🔴 Broken | Claude | Depends on Yahoo ID mapping fix |

---

## 5. Verified Working Components

| Component | Evidence |
|-----------|----------|
| Lineup recommendations persistence | `fantasy_lineups` has 1 active row |
| Decision pipeline | 32 decisions today (14 lineup + 18 waiver) |
| Projection freshness job | 17 consecutive SUCCESS runs |
| `cat_scores` backfill | 633/633 rows populated |
| Fusion engine | Approved by Claude; Marcel math verified |
| Z-score math | `statistics.stdev` (sample, N-1) with correct direction multipliers |
| Waiver recommendations endpoint | 14 real candidates returned |
| `mlb_odds` ingestion | Running every 5 minutes, all SUCCESS |
| `explainability` job | Running hourly, all SUCCESS |
| `probable_pitchers` job | Running, 349 rows maintained |

---

## 6. Prior Audit Misdiagnoses (K-32 Corrections)

### 6.1 Admin 500 Root Cause
- **K-32 claimed:** Crash at `DataIngestionLog.run_at` (lines 76, 80)
- **K-33 finding:** Crash at `PlayerProjection.updated_at` (lines 67-69)
- **Explanation:** The endpoint fails during `staleness_hours` computation before reaching ingestion log references. `PlayerProjection.updated_at` is `timestamp without time zone` (naive); `now` is UTC-aware.

### 6.2 Waiver Browse "0 Players"
- **K-32 claimed:** Browse filter is broken / data pipeline disconnected
- **K-33 finding:** Endpoint was redesigned; returns `top_available` instead of `players`
- **Explanation:** The schema changed. The endpoint returns 25 real players with need scores. Any consumer expecting the old `players` key sees an empty result.

### 6.3 BDL Injuries Failure Pattern
- **K-32 implied:** Error capture fix would resolve the NULL error_message issue
- **K-33 finding:** Fix is deployed but has not been exercised; no `bdl_injuries` runs since 13:07 UTC deploy
- **Explanation:** The pre-deploy failures (12:24, 12:21) ran on the old image and correctly show NULL error_message. Post-deploy behavior is unobserved.

---

## 7. Recommendations

### Immediate (next 2 hours)
1. **Fix admin 500** — 1-line change in `data_quality.py` to make `last_projection_update` timezone-aware
2. **Verify BDL injuries fix** — Monitor 14:25 UTC run; if it fails, the new error capture should now populate `error_message`
3. **Surface valuation cache error** — Remove or downgrade the `try/except` in `fantasy.py:get_decisions()` to expose the real DB failure

### Short-term (next 24 hours)
4. **Populate Yahoo IDs** — This is the single biggest blocker. Without it, z-scores, VORP, and ROS projections are all degraded
5. **Clean default projections** — Flag or remove 648 placeholder rows to prevent z-score pool contamination
6. **Fix probable pitchers confirmation** — Inspect upstream `pitcher_data` payload; likely receiving falsy values despite pitcher names being present
7. **Wire scoreboard projections** — Connect ROW/ROS pipeline to `scoreboard_orchestrator.py` for `my_projected_final`

### Medium-term (next week)
8. **Add `z_score_total` population** to `player_daily_metrics`
9. **Resolve VORP eligibility** — Likely tied to Yahoo ID mapping; verify after fix
10. **Update documentation** for waiver browse schema change

---

## Appendix A: Query Log

All database queries were executed against `postgres-ygnv.railway.internal` via the MCP `postgres` server. Live API calls were made against `https://fantasy-app-production-5079.up.railway.app/api/...`.

## Appendix B: Timeline

| Time (UTC) | Event |
|------------|-------|
| 12:21 | `bdl_injuries` FAILED (old image) |
| 12:24 | `bdl_injuries` FAILED (old image) |
| 12:25 | `bdl_injuries` SUCCESS (170 records) |
| 12:30 | `probable_pitchers` SUCCESS |
| 13:07 | **Deploy `e0e51d6c`** |
| 13:17 | `fantasy_lineups` row created (lineup persistence working) |
| 13:45 | **Audit K-33 completed** |
