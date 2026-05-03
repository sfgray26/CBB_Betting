# Fix P0 Data Quality Issues (player_type + Yahoo ID)

> **Date:** May 3, 2026
> **Priority:** P0 — Blocks batter routing + Yahoo roster features
> **Estimated:** 3 hours (1h player_type + 2h Yahoo ID sync)

---

## Context

Kimi's data quality audit (2026-05-03) found two P0 blockers:

1. **player_type NULL for 441/621 rows (71%)** — Breaks batter/pitcher routing
2. **Yahoo ID coverage 3.7% (372/10,096)** — 96.3% of players can't match to Yahoo rosters

Both must be fixed before any feature work can proceed reliably.

---

## Task 1: Backfill player_type NULLs

**File:** `scripts/backfill_player_type.py`

**Step 1: Create backfill script**

```python
#!/usr/bin/env python
"""Backfill player_type from positions JSONB."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

from sqlalchemy import text
from backend.models import SessionLocal

db = SessionLocal()

try:
    # Check current state
    print("Current player_type distribution:")
    rows = db.execute(text('''
      SELECT player_type, COUNT(*)
      FROM player_projections
      GROUP BY player_type
      ORDER BY player_type NULLS FIRST
    ''')).fetchall()

    for r in rows:
        print(f"  {r[0] if r[0] else 'NULL':10} {r[1]:5}")

    # Backfill player_type from positions
    print("\nBackfilling player_type from positions JSONB...")
    updated = db.execute(text('''
      UPDATE player_projections
      SET player_type = CASE
        WHEN positions ? ANY(array['SP','RP','P']) THEN 'pitcher'
        ELSE 'hitter'
      END
      WHERE player_type IS NULL
    ''')).rowcount

    db.commit()
    print(f"Updated {updated} rows")

    # Verify after backfill
    print("\nNew player_type distribution:")
    rows = db.execute(text('''
      SELECT player_type, COUNT(*)
      FROM player_projections
      GROUP BY player_type
      ORDER BY player_type
    ''')).fetchall()

    for r in rows:
        print(f"  {r[0]:10} {r[1]:5}")

    # Check for any remaining NULLs
    nulls = db.execute(text('''
      SELECT COUNT(*) FROM player_projections WHERE player_type IS NULL
    ''')).scalar()

    if nulls == 0:
        print("\n✅ All NULLs backfilled successfully")
    else:
        print(f"\n⚠️  {nulls} NULLs remain (positions JSONB missing)")

finally:
    db.close()
```

**Step 2: Run backfill**

```powershell
$env:DATABASE_URL = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
venv\Scripts\python scripts/backfill_player_type.py
```

**Expected output:**
```
Current player_type distribution:
  NULL       441
  pitcher    176

Backfilling player_type from positions JSONB...
Updated 441 rows

New player_type distribution:
  hitter     441
  pitcher    176

✅ All NULLs backfilled successfully
```

**Step 3: Add NOT NULL constraint**

```powershell
railway run python -c "
from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    # Add NOT NULL constraint
    conn.execute(text('''
      ALTER TABLE player_projections
      ALTER COLUMN player_type SET NOT NULL
    '''))
    conn.commit()
    print('✅ NOT NULL constraint added')
"
```

---

## Task 2: Fix Yahoo ID Sync

**Step 1: Check if yahoo_id_sync job exists**

```bash
grep -n "yahoo_id_sync\|100_034" backend/main.py
```

**Expected:** Either find the job definition OR confirm it's missing.

**Step 2: If missing, create the job**

**File:** `backend/fantasy_baseball/yahoo_id_sync.py`

```python
"""Sync Yahoo player IDs from fantasy API to player_id_mapping table."""
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import text
from backend.models import SessionLocal
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient

logger = logging.getLogger(__name__)

ADVISORY_LOCK_ID = 100_034

def sync_yahoo_player_ids() -> int:
    """
    Fetch all players from Yahoo fantasy league and map to BDL IDs.
    Returns count of players synced.
    """
    client = YahooFantasyClient()
    db = SessionLocal()

    try:
        # Get all players from league
        league_key = "mlb.l.72586"  # TODO: make configurable
        players = client.get_league_players(league_key)

        synced = 0
        for player in players:
            # Extract IDs
            yahoo_player_key = player.get('player_key')
            yahoo_id = yahoo_player_key.split('.')[-1] if yahoo_player_key else None

            # Map to BDL player_id (using name/team matching)
            # TODO: Implement BDL lookup logic
            player_id = _lookup_bdl_id(player.get('name'), player.get('editorial_team'))

            # Upsert to player_id_mapping
            db.execute(text('''
              INSERT INTO player_id_mapping (yahoo_id, player_id, updated_at)
              VALUES (:yahoo_id, :player_id, :updated_at)
              ON CONFLICT (player_id) DO UPDATE
              SET yahoo_id = EXCLUDED.yahoo_id,
                  updated_at = EXCLUDED.updated_at
            '''), {
                'yahoo_id': yahoo_id,
                'player_id': player_id,
                'updated_at': datetime.now(ZoneInfo("America/New_York"))
            })

            synced += 1

        db.commit()
        logger.info(f"Synced {synced} Yahoo player IDs")
        return synced

    except Exception as e:
        db.rollback()
        logger.error(f"Yahoo ID sync failed: {e}")
        raise
    finally:
        db.close()

def _lookup_bdl_id(name: str, team: str | None) -> int | None:
    """Lookup BDL player_id by name and team."""
    # TODO: Implement using BDL API or database lookup
    # For now, return None (will be filled in by BDL sync job)
    return None

def run_yahoo_id_sync_job():
    """Run Yahoo ID sync with advisory lock."""
    from backend.services.daily_ingestion import try_advisory_lock

    if try_advisory_lock(ADVISORY_LOCK_ID):
        try:
            count = sync_yahoo_player_ids()
            print(f"✅ Yahoo ID sync complete: {count} players")
        finally:
            from backend.services.daily_ingestion import release_advisory_lock
            release_advisory_lock(ADVISORY_LOCK_ID)
    else:
        print("⚠️  Yahoo ID sync already running")
```

**Step 3: Add scheduler job to main.py**

**File:** `backend/main.py` (around line 400, with other daily jobs)

```python
# Yahoo ID sync (daily at 6 AM ET)
@scheduler.scheduled_job(trigger='cron', hour=6, minute=0, timezone='America/New_York')
def yahoo_id_sync_job():
    """Sync Yahoo player IDs from fantasy API."""
    logger.info("Running Yahoo ID sync job")
    from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job
    try:
        run_yahoo_id_sync_job()
    except Exception as e:
        logger.error(f"Yahoo ID sync failed: {e}", exc_info=True)
```

**Step 4: Run manual sync**

```powershell
$env:DATABASE_URL = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"
venv\Scripts\python -c "
from backend.fantasy_baseball.yahoo_id_sync import sync_yahoo_player_ids
count = sync_yahoo_player_ids()
print(f'Synced {count} Yahoo player IDs')
"
```

**Expected:** Should sync ~250 players (12 teams × ~23 players)

**Step 5: Verify yahoo_id coverage increased**

```powershell
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    total = db.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
    yahoo = db.execute(text('SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL')).scalar()
    coverage = yahoo / total * 100 if total > 0 else 0

    print(f'Total: {total:,}')
    print(f'Yahoo IDs: {yahoo:,}')
    print(f'Coverage: {coverage:.1f}%')
finally:
    db.close()
"
```

**Target:** > 50% coverage (currently 3.7%)

---

## Task 3: Verify Fixes

**Step 1: Re-run Kimi's accuracy queries**

```powershell
# Test batter accuracy query (should return results now)
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Kimi's batter accuracy query
    matched = db.execute(text('''
      SELECT COUNT(DISTINCT pp.player_id)
      FROM player_projections pp
      JOIN player_scores ps ON pp.player_id = ps.player_id
      WHERE pp.player_type = 'batter'
        AND ps.window_days = 14
        AND ps.as_of_date >= CURRENT_DATE - INTERVAL '14 days'
    ''')).scalar()

    print(f'Batters matched: {matched}')
    print('✅ Batter routing fixed' if matched > 0 else '❌ Still broken')
finally:
    db.close()
"
```

**Step 2: Check Yahoo ID in mapping**

```powershell
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text

db = SessionLocal()
try:
    # Sample 10 players with yahoo_id
    rows = db.execute(text('''
      SELECT pim.yahoo_id, pp.player_name
      FROM player_id_mapping pim
      JOIN player_projections pp ON pim.player_id = pp.player_id
      WHERE pim.yahoo_id IS NOT NULL
      LIMIT 10
    ''')).fetchall()

    print(f'Sample players with Yahoo IDs:')
    for r in rows:
        print(f'  {r[1]:30} yahoo_id={r[0]}')
finally:
    db.close()
"
```

---

## Deployment Plan

1. **Commit backfill script**
   ```bash
   git add scripts/backfill_player_type.py
   git commit -m "feat(player-projections): add player_type backfill script"
   ```

2. **Create yahoo_id_sync module**
   ```bash
   git add backend/fantasy_baseball/yahoo_id_sync.py
   git commit -m "feat(yahoo): add Yahoo ID sync job (lock 100_034)"
   ```

3. **Update scheduler**
   ```bash
   git add backend/main.py
   git commit -m "feat(scheduler): add yahoo_id_sync job (daily 6 AM ET)"
   ```

4. **Run backfill locally first** (verify on Railway after local test)
5. **Deploy to Railway**
   ```bash
   railway up --detach
   ```

6. **Run backfill on production**
   ```bash
   railway run python scripts/backfill_player_type.py
   railway run python -c "from backend.fantasy_baseball.yahoo_id_sync import sync_yahoo_player_ids; sync_yahoo_player_ids()"
   ```

7. **Verify fixes** (see Task 3 above)

---

## Success Criteria

- [ ] player_type NULL count: 0 (currently 441)
- [ ] Batter accuracy query returns > 0 matches
- [ ] Yahoo ID coverage: > 50% (currently 3.7%)
- [ ] Scheduler job deployed and running
- [ ] No regressions (tests still pass)

---

## Rollback Plan

If backfill goes wrong:
```sql
ROLLBACK;
-- Or restore from backup:
UPDATE player_projections
SET player_type = NULL
WHERE player_type IN ('hitter', 'pitcher')
  AND updated_at >= '2026-05-03';
```

---

**Estimated time:** 3 hours
**Risk level:** LOW (backfill is deterministic, can rollback)
**Blocking:** No — can proceed with other work while scheduler runs daily
