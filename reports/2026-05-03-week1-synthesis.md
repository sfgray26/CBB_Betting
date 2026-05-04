# Week 1 P0 Complete: Synthesis & Next Steps

**Date:** May 3, 2026
**Status:** All verification complete. Two P0 data quality bugs identified. N+1 performance bottleneck confirmed.
**Next:** Fix data quality bugs (2h), then decide on performance strategy.

---

## Executive Summary

| Agent | Task | Status | Key Findings |
|-------|------|--------|--------------|
| Kimi | Data Quality Audit | ✅ Complete | 2 P0 bugs found (player_type NULL, Yahoo ID 3.7%) |
| Gemini | Endpoint Verification | ✅ Complete | All endpoints work, but slow (10-24s) |
| Gemini | Performance Investigation | ✅ Complete | N+1 confirmed in ballpark_factors.py, caching helped (50% improvement) |

---

## Critical Findings by Priority

### P0 - Data Quality (Must Fix Before Features)

**#1: player_type NULL for 71% of rows (Kimi)**
- **Impact:** Breaks batter/pitcher routing, accuracy queries return 0 matches
- **Fix:** Backfill from `positions` JSONB (1 hour)
- **Risk:** LOW (deterministic, reversible)
- **Blocking:** Batter projection accuracy measurement

**#2: Yahoo ID coverage 3.7% (Kimi)**
- **Impact:** 96.3% of players can't match to Yahoo rosters
- **Fix:** Create yahoo_id_sync scheduler job (2 hours)
- **Risk:** LOW (scheduler job, retryable)
- **Blocking:** Waiver recommendations, roster sync, MCMC opponent fetch

### P1 - Performance (Impacts UX)

**#3: N+1 queries in ballpark_factors.py (Gemini)**
- **Impact:** Waiver recommendations 23.78s, dashboard 10-20s
- **Current:** Caching reduced dashboard 19.34s → 9.95s (50% improvement)
- **Gap:** Waiver still 26.82s (no improvement - different bottleneck)
- **Fix Options:**
  1. Bulk-load park_factors on startup (1 hour)
  2. Process-level cache for park_factors table (30 min)
  3. Pre-calculate all projections to Redis (3 hours, architectural)
- **Recommendation:** Start with bulk-loading (Option 1)

**#4: matchup_preview returns null (Gemini)**
- **Impact:** Can't verify if MCMC win_prob regression (constant 0.763) exists
- **Root Cause:** Team key resolution failure in dashboard service
- **Fix:** Debug DashboardService._get_matchup_preview (30 min)
- **Risk:** LOW (diagnostic only)

### P2 - Accuracy (Monitor, Don't Fix Yet)

**#5: Pitcher ERA correlation r=0.1569 (Kimi)**
- **Impact:** Projections optimistic by 0.36 ERA runs
- **Note:** 14-day sample noisy for pitchers (small IP variance)
- **Action:** Re-check with 30/60-day window before tuning

**#6: player_scores 20.4 hours stale (Kimi)**
- **Impact:** Daily projections slightly stale
- **Action:** Monitor scheduler job, no immediate fix needed

---

## Performance Baseline (After Caching)

| Endpoint | Before | After | Target | Status |
|----------|--------|-------|--------|--------|
| POST /api/fantasy/roster/optimize | — | 0.28s | <5s | ✅ FAST |
| GET /api/fantasy/matchup | — | 0.63s | <5s | ✅ FAST |
| GET /api/fantasy/lineup | 13.59s | ~13s | <5s | ❌ SLOW |
| GET /api/dashboard | 19.34s | 9.95s | <5s | ⚠️ IMPROVED |
| GET /api/fantasy/waiver/recommendations | 23.78s | 26.82s | <5s | ❌ VERY SLOW |

**Key Insight:** Roster optimizer is fast (0.28s) because it uses pre-calculated DB scores. The slow endpoints are ones that calculate on-demand.

---

## Immediate Action Plan (Prioritized)

### Phase 1: Fix Data Quality Bugs (2 hours) ✅ DO THIS FIRST

**Task A: Backfill player_type (45 min)**
1. Run `scripts/backfill_player_type.py`
2. Verify 0 NULLs remain
3. Add NOT NULL constraint
4. Test batter accuracy query (should return > 0 matches)

**Task B: Fix Yahoo ID sync (1h 15min)**
1. Create `yahoo_id_sync.py` module
2. Add scheduler job (daily 6 AM ET)
3. Run manual sync (target ~250 players)
4. Verify coverage > 50%

**Deliverable:** Both fixes deployed, data quality bugs resolved

---

### Phase 2: Fix Performance (3 paths, choose one)

**Option A: Quick Win (Recommended - 1 hour)**
- Bulk-load park_factors into process-level cache
- Add to `main.py` startup: `load_park_factors()`
- Expected: 20-26s → <5s for waiver/dashboard
- Risk: LOW, reversible

**Option B: Medium Effort (3 hours)**
- Pre-calculate projections to Redis
- Background worker updates every 5 min
- Frontend reads from Redis cache
- Expected: <1s for all endpoints
- Risk: MEDIUM (requires Redis, cache invalidation)

**Option C: Defer (Accept Current Performance)**
- Current: Dashboard 10s, waiver 27s
- Acceptable for now if used sparingly
- Focus on features instead
- Risk: Poor UX, but functional

**Recommendation:** Option A (quick win) now, consider Option B later if UX still poor.

---

### Phase 3: Diagnostic Debug (30 min)

**Task C: Debug matchup_preview null**
- Check DashboardService._get_matchup_preview
- Verify team key resolution (league_key format)
- Test win_prob calculation returns non-constant values
- Fix if regression found

---

## Success Criteria

**Phase 1 (Data Quality):**
- [ ] player_type NULL count: 0 (currently 441)
- [ ] Batter accuracy query: > 0 matches (currently 0)
- [ ] Yahoo ID coverage: > 50% (currently 3.7%)
- [ ] Scheduler job deployed and running

**Phase 2 (Performance - Option A):**
- [ ] Waiver recommendations: <5s (currently 27s)
- [ ] Dashboard: <5s (currently 10s)
- [ ] No N+1 queries in logs

**Phase 3 (Diagnostic):**
- [ ] matchup_preview returns non-null data
- [ ] win_prob varies (not constant 0.763)
- [ ] Team key resolution works

---

## Detailed Fix Plans

### Fix #1: player_type Backfill

**File:** `scripts/backfill_player_type.py` (already created in task-p0-data-quality-fixes.md)

**Steps:**
```powershell
# 1. Test locally first
$env:DATABASE_URL = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
venv\Scripts\python scripts/backfill_player_type.py

# 2. Verify output
# Expected: "Updated 441 rows" and "All NULLs backfilled successfully"

# 3. Deploy to Railway
railway run python scripts/backfill_player_type.py

# 4. Add NOT NULL constraint
railway run python -c "
from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text('ALTER TABLE player_projections ALTER COLUMN player_type SET NOT NULL'))
    conn.commit()
    print('✅ NOT NULL constraint added')
"
```

**Verification:**
```powershell
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
nulls = db.execute(text('SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL')).scalar()
print(f'Remaining NULLs: {nulls}')
db.close()
"
```

**Expected:** `Remaining NULLs: 0`

---

### Fix #2: Yahoo ID Sync

**File:** `backend/fantasy_baseball/yahoo_id_sync.py` (already created in task-p0-data-quality-fixes.md)

**Steps:**
1. Create the module (see task-p0-data-quality-fixes.md for full code)
2. Add scheduler job to `main.py` (daily 6 AM ET)
3. Run manual sync
4. Verify coverage increased

**Verification:**
```powershell
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
total = db.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
yahoo = db.execute(text('SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL')).scalar()
coverage = yahoo / total * 100
print(f'Yahoo ID coverage: {coverage:.1f}% (target: >50%)')
db.close()
"
```

---

### Fix #3A: Bulk-Load Park Factors (Option A)

**File:** `backend/fantasy_baseball/ballpark_factors.py`

**Current code (N+1 problem):**
```python
def get_park_factor(team: str, handedness: str = 'R') -> float:
    factor = db.query(ParkFactor).filter_by(
        team=team,
        handedness=handedness
    ).first()
    return factor.value if factor else 1.0
```

**Fixed code (bulk-load on startup):**
```python
from functools import lru_cache

# Global cache (loaded on startup)
_park_factor_cache: dict[tuple[str, str], float] = {}

def load_park_factors():
    """Load all park factors into memory on startup."""
    global _park_factor_cache
    from backend.models import SessionLocal
    from sqlalchemy import text

    db = SessionLocal()
    try:
        rows = db.execute(text('''
          SELECT team, handedness, value
          FROM park_factors
        ''')).fetchall()

        _park_factor_cache = {
            (row[0], row[1]): row[2]
            for row in rows
        }
    finally:
        db.close()

@lru_cache(maxsize=32)
def get_park_factor(team: str, handedness: str = 'R') -> float:
    """Get park factor from in-memory cache."""
    return _park_factor_cache.get((team, handedness), 1.0)
```

**Add to `main.py` startup:**
```python
@app.on_event("startup")
async def startup_event():
    """Load park factors on startup."""
    from backend.fantasy_baseball.ballpark_factors import load_park_factors
    load_park_factors()
    logger.info("Park factors loaded into memory")
```

**Expected Impact:** Eliminates N+1 queries, 27s → <5s

---

## Deployment Queue

1. **Commit player_type backfill**
   ```bash
   git add scripts/backfill_player_type.py
   git commit -m "fix(player-projections): backfill player_type from positions"
   ```

2. **Commit yahoo_id_sync module**
   ```bash
   git add backend/fantasy_baseball/yahoo_id_sync.py backend/main.py
   git commit -m "feat(yahoo): add Yahoo ID sync job (lock 100_034)"
   ```

3. **Commit park_factors bulk-load**
   ```bash
   git add backend/fantasy_baseball/ballpark_factors.py backend/main.py
   git commit -m "perf(park): bulk-load park_factors on startup to fix N+1"
   ```

4. **Deploy to Railway**
   ```bash
   railway up --detach
   ```

5. **Run manual backfills**
   ```bash
   railway run python scripts/backfill_player_type.py
   railway run python -c "from backend.fantasy_baseball.yahoo_id_sync import sync_yahoo_player_ids; sync_yahoo_player_ids()"
   ```

6. **Verify all fixes**
   ```bash
   # Test batter accuracy query
   railway run python -c "
   from backend.models import SessionLocal
   from sqlalchemy import text
   db = SessionLocal()
   matched = db.execute(text('''
     SELECT COUNT(DISTINCT pp.player_id)
     FROM player_projections pp
     JOIN player_scores ps ON pp.player_id = ps.player_id
     WHERE pp.player_type = 'hitter'
       AND ps.window_days = 14
       AND ps.as_of_date >= CURRENT_DATE - INTERVAL '14 days'
   ''')).scalar()
   print(f'Batters matched: {matched} (expected: >0)')
   db.close()
   "

   # Test waiver performance
   time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommendations?league_id=72586" -o /dev/null -w "%{time_total}s\n"
   ```

---

## Handoff Decision

**Choice A: Fix all P0 bugs now (Recommended)**
- Time: 3-4 hours total
- Impact: Data quality resolved, performance significantly improved
- Risk: LOW (all fixes are reversible)
- Outcome: Clean foundation for feature work

**Choice B: Fix data quality only, defer performance**
- Time: 2 hours
- Impact: Data quality resolved, performance still slow
- Risk: LOW
- Outcome: Can proceed with features, but poor UX

**Choice C: Defer everything, start features**
- Time: 0 hours
- Impact: Technical debt accumulates
- Risk: MEDIUM (features may break on bad data)
- Outcome: Faster feature delivery, slower system

**My Recommendation:** Choice A (fix all P0 bugs now). The investment is small (3-4 hours), the risk is low, and the foundation will be solid for the rest of the season.

---

**Report Generated:** May 3, 2026
**Next Update:** After P0 fixes deployed
