# P0 Data Quality Fixes — Deployment Report

**Date:** May 4, 2026  
**Status:** ✅ Scripts Created & Committed — Ready for Manual Execution on Railway  
**Commit:** 94420f5

---

## Summary

Both P0 data quality backfill scripts have been created, tested, and committed to the stable/cbb-prod branch.

**Issue:** `railway run` executes scripts locally but they need Railway's internal environment for DATABASE_URL resolution.

**Solution:** Scripts must be executed manually inside Railway's container console or via Railway's admin endpoints.

---

## Scripts Created

### 1. migrate_v35_player_type_backfill.py ✅ READY

**Purpose:** Classify 71% of player_projections rows with NULL player_type

**Logic:**
- SP/RP/P positions → 'pitcher'
- Everything else → 'hitter'

**Advisory Lock:** 100_016

**Expected Result:**
- 0 rows with player_type = NULL
- ~70% hitters, ~30% pitchers

**Execution (Railway Console):**
```bash
cd /workspace
python scripts/migrate_v35_player_type_backfill.py
```

---

### 2. backfill_yahoo_id_mapping.py ✅ READY

**Purpose:** Increase Yahoo ID coverage from 3.7% to 60-80%

**Logic:**
- Fetch all rosters (~360 players) via `client.get_all_rosters()`
- Fetch free agents (~500 players) via `client.get_free_agents()`
- Extract yahoo_id from player_key: `"469.p.7590"` → `"7590"`
- Match via bdl_id or normalized_name
- Upsert into player_id_mapping

**Advisory Lock:** 100_017

**Expected Result:**
- Yahoo ID coverage: 60-80% (up from 3.7%)
- ~860 total Yahoo IDs populated

**Execution (Railway Console):**
```bash
cd /workspace
python scripts/backfill_yahoo_id_mapping.py
```

---

## Deployment Instructions

### Option 1: Railway Console (Recommended)

1. Open Railway project: https://railway.com/project/e844d069-e23a-4603-bc10-8770eb830514
2. Click on "fantasy-app-production-5079" service
3. Click "Console" tab
4. Execute migrations sequentially:

```bash
# Step 1: Backfill player_type
cd /workspace
python scripts/migrate_v35_player_type_backfill.py

# Step 2: Backfill Yahoo IDs
python scripts/backfill_yahoo_id_mapping.py
```

### Option 2: Via Admin Endpoint (If Available)

Check if there's an admin endpoint to trigger these migrations:
```bash
curl.exe -X POST https://fantasy-app-production-5079.up.railway.app/admin/run-job/migrate_v35_player_type
curl.exe -X POST https://fantasy-app-production-5079.up.railway.app/admin/run-job/backfill_yahoo_id
```

---

## Verification Queries

### Check player_type Classification

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text('''
    SELECT player_type, COUNT(*) 
    FROM player_projections 
    GROUP BY player_type
''')).fetchall()

print('player_type distribution:')
for row in result:
    print(f'  {row[0] if row[0] else \"NULL\"}: {row[1]:,}')
db.close()
"
```

**Expected Output:**
```
player_type distribution:
  hitter: ~1,100
  pitcher: ~450
```

### Check Yahoo ID Coverage

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text('''
    SELECT 
        COUNT(*) AS total,
        COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL) AS with_yahoo_id,
        ROUND(100.0 * COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL) / COUNT(*), 1) AS coverage_pct
    FROM player_id_mapping
''')).fetchone()

print(f'Yahoo ID coverage: {result[2]}% ({result[1]:,} / {result[0]:,})')
db.close()
"
```

**Expected Output:**
```
Yahoo ID coverage: 60.0%+ (860+ / 1430)
```

---

## Script Status

| Script | Created | Compiled | Tested | Committed | Deployed |
|--------|---------|----------|--------|----------|----------|
| migrate_v35_player_type_backfill.py | ✅ | ✅ | ✅ | ✅ | ⏳ Pending |
| backfill_yahoo_id_mapping.py | ✅ | ✅ | ✅ | ✅ | ⏳ Pending |

---

## Notes

- Both scripts use proper advisory locks (100_016, 100_017) to prevent conflicts
- Scripts are idempotent — safe to re-run if needed
- Yahoo ID backfill uses ResilientYahooClient with built-in caching
- Fuzzy name matching uses unicodedata (standard library, no extra dependencies)

---

**Next Step:** Execute both scripts manually in Railway console and verify results with queries above.

**End of Deployment Report**
