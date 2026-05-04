# Optimized Prompt for Claude Code — Phase 1 Remediation (Post-Deploy)

> **From:** Kimi CLI (Fresh Audit Complete)  
> **To:** Claude Code  
> **Date:** 2026-04-25  
> **Status:** Post-deployment audit reveals partial progress. Immediate remediation required.  
> **Scope:** Fix what was missed, finish what was started, deploy and verify.

---

## SITUATION

You deployed code today (commits `9860762` and `ca89a8a`). The **backfill script worked** — 260/353 numeric projection names were resolved. However, a fresh database audit at 18:06 UTC shows **critical one-line fixes were not deployed**, a **migration was never run**, and **new tables remain empty**.

**What you built (good):**
- `backend/services/balldontlie.py` — BDL client with pagination
- `backend/services/daily_ingestion.py` — updates including `_sync_bdl_injuries`
- `backend/models.py` — `IngestedInjury` model added
- `scripts/migrations/create_ingested_injuries.sql` — migration script
- `scripts/backfill_numeric_player_names.py` — backfill script (WORKED)

**What was NOT done:**
- `DataIngestionLog.run_at` → column doesn't exist. Admin endpoint still 500.
- `projection_freshness` datetime fix — code exists but DB still shows 152 failures
- Migration `create_ingested_injuries.sql` — **never run** (table doesn't exist)
- `yahoo_id` backfill — still 0/10,000
- `fantasy_lineups` — still 0 rows
- `player_valuation_cache` — still 0 rows
- `probable_pitchers.is_confirmed` — still 0/332

---

## TASK 1: Fix Admin 500 Error (5 minutes)

**File:** `backend/routers/data_quality.py`  
**Lines:** 76, 80

**Bug:** `DataIngestionLog` has **no column named `run_at`**. The actual columns are `started_at` and `completed_at`.

```python
# WRONG (line 76):
DataIngestionLog.run_at > now - timedelta(days=7)

# CORRECT:
DataIngestionLog.completed_at > now - timedelta(days=7)
```

Fix both occurrences (lines 76 and 80). Deploy. Verify with:
```bash
curl https://fantasy-app-production-5079.up.railway.app/api/admin/data-quality/summary
```

**Acceptance:** Returns 200 with JSON payload.

---

## TASK 2: Finish Projection Name Backfill (30 minutes)

**File:** `scripts/backfill_numeric_player_names.py`

**Status:** 260/353 numeric names resolved. **93 remain.**

**Root cause of remaining 93:** These players have **no entry in `player_id_mapping`** (or their `bdl_id` doesn't resolve). All 93 have `team=NULL`, `positions=NULL`, and low DQ scores (0.10–0.34).

**Action:** Extend the backfill script to handle orphans:
1. For each remaining numeric `player_name` in `player_projections`:
   - Try `GET /mlb/v1/players/{mlbam_id}` via BDL API (you already built the client)
   - If BDL returns a player, use `full_name` + `team` + `position`
   - If BDL returns 404, try MLB Stats API `https://statsapi.mlb.com/api/v1/people/{player_id}`
2. Update `player_projections` with resolved name/team/positions
3. Insert a new `player_id_mapping` row for discovered players

**Acceptance:** `SELECT COUNT(*) FROM player_projections WHERE player_name ~ '^[0-9]+$'` returns **0**.

---

## TASK 3: Run the Migration You Created (10 minutes)

**File:** `scripts/migrations/create_ingested_injuries.sql`

**Status:** Table does **not exist** in production.

**Action:**
```bash
railway run psql -f scripts/migrations/create_ingested_injuries.sql
```

Or use Railway CLI to execute the SQL.

**Acceptance:**
```sql
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' AND table_name = 'ingested_injuries';
-- returns 1 row
```

---

## TASK 4: Fix projection_freshness (15 minutes)

**File:** `backend/services/daily_ingestion.py`  
**Method:** `_check_projection_freshness` (~line 4872)

**Status:** Code already has `date` → `datetime` conversion for `latest_ensemble` and `latest_statcast`. But DB shows **152 consecutive failures** with `TypeError: unsupported operand type(s) for -: 'datetime.datetime' and 'datetime.date'`.

**Hypothesis:** The fix exists in repo code but the **deployed code is stale**. OR the `_load_persisted_ros_cache` function returns a `date` object for `ros_fetched_at` (line 4946).

**Action:**
1. Verify the deployed code matches the repo code for `_check_projection_freshness`
2. Add defensive type checking for `ros_fetched_at` at line 4952:
```python
from datetime import date, datetime, time
if isinstance(ros_fetched_at, date) and not isinstance(ros_fetched_at, datetime):
    ros_fetched_at = datetime.combine(ros_fetched_at, time(), tzinfo=ZoneInfo("America/New_York"))
elif hasattr(ros_fetched_at, "tzinfo") and ros_fetched_at.tzinfo is None:
    ros_fetched_at = ros_fetched_at.replace(tzinfo=ZoneInfo("America/New_York"))
```
3. Deploy and verify:
```sql
SELECT * FROM data_ingestion_logs 
WHERE job_type = 'projection_freshness' 
ORDER BY completed_at DESC LIMIT 5;
-- status should be SUCCESS
```

**Acceptance:** Next `projection_freshness` run shows `status = 'SUCCESS'`.

---

## TASK 5: Wire BDL Injuries to Persist (1 hour)

**Files:**
- `backend/services/daily_ingestion.py` — `_sync_bdl_injuries` method
- `backend/services/balldontlie.py` — `get_mlb_injuries` method

**Status:** You built the `IngestedInjury` model and the `_sync_bdl_injuries` method. The migration exists but was **never run**. The pipeline has never executed the BDL injury sync.

**Action:**
1. **Run the migration first** (Task 3)
2. Verify `_sync_bdl_injuries` is scheduled in the orchestrator's `start()` method
3. If not scheduled, add it:
```python
self._scheduler.add_job(
    self._sync_bdl_injuries,
    CronTrigger(hour=7, minute=0, timezone=tz),  # daily at 7 AM ET
    id="bdl_injuries",
    name="BDL Injury Sync",
    replace_existing=True,
)
```
4. Assign a new advisory lock ID in `LOCK_IDS`:
```python
"bdl_injuries": 100_036,
```
5. Deploy and trigger manually or wait for next scheduled run
6. Verify:
```sql
SELECT COUNT(*) FROM ingested_injuries;
-- should be > 0 after first run
```

**Acceptance:** `ingested_injuries` table has rows within 24 hours of deployment.

---

## TASK 6: Populate fantasy_lineups Table (30 minutes)

**File:** `backend/routers/fantasy.py` or `backend/fantasy_baseball/daily_lineup_optimizer.py`

**Status:** Table has **0 rows**. The lineup optimizer computes lineups but never persists them.

**Action:** After the lineup optimizer computes the optimal lineup, INSERT the result into `fantasy_lineups`:

```python
from backend.models import FantasyLineup

lineup = FantasyLineup(
    lineup_date=target_date,
    platform="yahoo",
    positions=json.dumps(optimal_lineup_positions),
    projected_points=projected_total,
    notes=f"Computed by {optimizer_version}"
)
db.add(lineup)
db.commit()
```

**Acceptance:** `SELECT COUNT(*) FROM fantasy_lineups` returns **> 0** after next lineup request.

---

## TASK 7: Populate player_valuation_cache Table (30 minutes)

**File:** `backend/routers/fantasy.py` — waiver endpoint, or `backend/services/decision_engine.py`

**Status:** Table has **0 rows**. MCMC/waiver computations run but don't cache.

**Action:** After waiver recommendations are computed, INSERT into `player_valuation_cache`:

```python
from backend.models import PlayerValuationCache
from uuid import uuid4

cache = PlayerValuationCache(
    id=uuid4(),
    player_id=player_id,
    player_name=player_name,
    target_date=today,
    league_key=league_key,
    report=json.dumps(valuation_report),
    computed_at=datetime.now(ZoneInfo("UTC")),
    data_as_of=datetime.now(ZoneInfo("UTC"))
)
db.add(cache)
db.commit()
```

**Acceptance:** `SELECT COUNT(*) FROM player_valuation_cache` returns **> 0** after next waiver request.

---

## TASK 8: Confirm Probable Pitchers via BDL (1 hour)

**File:** `backend/services/daily_ingestion.py` — `_sync_probable_pitchers`

**Status:** 0/332 confirmed. 119 recent rows have `bdl_player_id`.

**Action:** In the existing `_sync_probable_pitchers` method (or create `_sync_bdl_lineups`):
1. Call BDL `GET /mlb/v1/games?dates[]=YYYY-MM-DD` for today's date
2. For each game, call `GET /mlb/v1/lineups?game_ids[]={game_id}`
3. Find pitchers with `is_probable_pitcher = true`
4. Update `probable_pitchers` rows where `bdl_player_id` matches:
   - Set `is_confirmed = true`
   - Set `updated_at = now()`
5. For games where BDL has no lineup yet, leave `is_confirmed = false`

**Acceptance:** `SELECT COUNT(*) FROM probable_pitchers WHERE is_confirmed = true` returns **> 0** after next run.

---

## DEPLOYMENT CHECKLIST

After making changes:
- [ ] `py_compile` all modified `.py` files
- [ ] Run `pytest tests/` — zero regressions
- [ ] Commit with descriptive message
- [ ] Deploy to Railway: `railway up`
- [ ] Run migration if schema changed
- [ ] Verify each task's acceptance criteria via fresh DB query
- [ ] Update `HANDOFF.md` with what was done

---

## ANTI-PATTERNS TO AVOID

1. **Do NOT create new files when existing files can be extended.** The `balldontlie.py` and `daily_ingestion.py` files already exist — add to them.
2. **Do NOT skip running migrations.** A migration file in `scripts/` is useless until executed.
3. **Do NOT leave INSERT logic commented out.** If a table is empty, find where the computation happens and add the INSERT.
4. **Do NOT guess column names.** Check `information_schema.columns` if unsure.

---

## VERIFICATION COMMANDS

Run these after deployment to confirm everything worked:

```bash
# Task 1: Admin endpoint
python -c "import requests; r=requests.get('https://fantasy-app-production-5079.up.railway.app/api/admin/data-quality/summary', timeout=15); print(r.status_code, r.json()['pipeline_staleness']['status'])"

# Task 2: Numeric names
python -c "import requests; print('Run SQL: SELECT COUNT(*) FROM player_projections WHERE player_name ~ \"^[0-9]+$\"')"

# Task 3: Migration
python -c "import requests; print('Run SQL: SELECT table_name FROM information_schema.tables WHERE table_name = \"ingested_injuries\"')"

# Task 4: Freshness
python -c "import requests; print('Run SQL: SELECT status FROM data_ingestion_logs WHERE job_type = \"projection_freshness\" ORDER BY completed_at DESC LIMIT 1')"

# Task 5: Injuries
python -c "import requests; print('Run SQL: SELECT COUNT(*) FROM ingested_injuries')"

# Task 6: Lineups
python -c "import requests; print('Run SQL: SELECT COUNT(*) FROM fantasy_lineups')"

# Task 7: Valuations
python -c "import requests; print('Run SQL: SELECT COUNT(*) FROM player_valuation_cache')"

# Task 8: Pitchers
python -c "import requests; print('Run SQL: SELECT COUNT(*) FROM probable_pitchers WHERE is_confirmed = true')"
```

---

*Prompt compiled by Kimi CLI v1.17.0. All findings sourced from fresh production DB queries at 18:06 UTC. Exact file paths and line numbers verified against HEAD at commit `ca89a8a`.*
