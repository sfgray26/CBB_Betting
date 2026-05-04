# Task: Implement P0 Fixes (Code Changes)

**Agent:** Claude Code (Master Architect)
**Objective:** Create all necessary code/files for P0 data quality fixes
**Timebox:** 1.5 hours
**Deliverables:** 3 files created/modified, compilation verified

---

## Mission

Create the implementation for all 3 P0 fixes:
1. player_type backfill script
2. Yahoo ID sync module + scheduler job
3. Park factors bulk-loading

DO NOT deploy or run on Railway — that's Gemini's job.

---

## Fix 1: Create player_type Backfill Script

**File:** `scripts/backfill_player_type.py`

```python
#!/usr/bin/env python
"""Backfill player_type from positions JSONB."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

from sqlalchemy import text
from backend.models import SessionLocal

def main():
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
        print(f"✅ Updated {updated} rows")

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
            print("\n✅ SUCCESS: All NULLs backfilled")
            return 0
        else:
            print(f"\n⚠️  WARNING: {nulls} NULLs remain (positions JSONB missing)")
            return 1

    except Exception as e:
        db.rollback()
        print(f"\n❌ ERROR: {e}")
        return 1
    finally:
        db.close()

if __name__ == "__main__":
    sys.exit(main())
```

**Verify:** File created at `scripts/backfill_player_type.py`

---

## Fix 2: Create Yahoo ID Sync Module

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
        league_key = "mlb.l.72586"
        league_national_key = f"{league_key}.l.{client.league_id}"

        logger.info(f"Fetching players from {league_national_key}")

        # Use league players endpoint
        players_data = client.get_league_players(league_national_key)

        if not players_data:
            logger.warning(f"No players returned from {league_national_key}")
            return 0

        synced = 0
        for player in players_data:
            # Extract Yahoo player key
            player_key = player.get('player_key', '')
            if not player_key:
                continue

            # Extract Yahoo ID (last part of key: 12345)
            yahoo_id = player_key.split('.')[-1] if player_key else None

            # Get BDL player_id from name lookup
            name = player.get('name', '')
            team = player.get('editorial_team_abbr', '')

            if not name or not yahoo_id:
                continue

            # Try to find existing BDL player_id
            player_id = _lookup_bdl_id(db, name, team)

            if player_id:
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
        logger.info(f"✅ Synced {synced} Yahoo player IDs")
        return synced

    except Exception as e:
        db.rollback()
        logger.error(f"❌ Yahoo ID sync failed: {e}", exc_info=True)
        raise
    finally:
        db.close()

def _lookup_bdl_id(db, name: str, team: str | None) -> int | None:
    """Lookup BDL player_id by name and team."""
    # Try exact name match first
    result = db.execute(text('''
      SELECT player_id FROM mlb_player_stats
      WHERE name = :name
      LIMIT 1
    '''), {'name': name}).fetchone()

    if result:
        return result[0]

    # Try name + team match
    if team:
        result = db.execute(text('''
          SELECT player_id FROM mlb_player_stats
          WHERE name = :name AND team = :team
          LIMIT 1
        '''), {'name': name, 'team': team}).fetchone()

        if result:
            return result[0]

    # Not found - return None (will be filled by BDL sync job)
    return None

def run_yahoo_id_sync_job():
    """Run Yahoo ID sync with advisory lock."""
    from backend.services.daily_ingestion import try_advisory_lock, release_advisory_lock

    if try_advisory_lock(ADVISORY_LOCK_ID):
        try:
            logger.info("🔄 Starting Yahoo ID sync job")
            count = sync_yahoo_player_ids()
            logger.info(f"✅ Yahoo ID sync complete: {count} players")
            return count
        finally:
            release_advisory_lock(ADVISORY_LOCK_ID)
    else:
        logger.warning("⚠️  Yahoo ID sync already running")
        return 0

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    count = run_yahoo_id_sync_job()
    print(f"Synced {count} Yahoo player IDs")
    sys.exit(0 if count > 0 else 1)
```

**Verify:** File created at `backend/fantasy_baseball/yahoo_id_sync.py`

---

## Fix 3: Bulk-Load Park Factors

**File:** `backend/fantasy_baseball/ballpark_factors.py`

**Find the current `get_park_factor` function (around line 50) and ADD this before it:**

```python
from functools import lru_cache
from typing import Dict, Tuple

# Global cache (loaded on startup)
_park_factor_cache: Dict[Tuple[str, str], float] = {}

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
            (row[0], row[1]): float(row[2])
            for row in rows
        }

        logger.info(f"Loaded {len(_park_factor_cache)} park factors into memory")
    finally:
        db.close()
```

**Now MODIFY the existing `get_park_factor` function to use cache:**

```python
@lru_cache(maxsize=32)
def get_park_factor(team: str, handedness: str = 'R') -> float:
    """
    Get park factor from in-memory cache.

    Args:
        team: MLB team abbreviation (e.g., 'NYY', 'BOS')
        handedness: 'R' or 'L' for batter handedness

    Returns:
        Park factor value (1.0 = neutral, >1 = hitter-friendly, <1 = pitcher-friendly)
    """
    # Try cache first
    cached = _park_factor_cache.get((team, handedness))
    if cached:
        return cached

    # Fallback: 1.0 if not found
    return 1.0
```

**Verify:** File modified, bulk-loading added

---

## Fix 4: Add Startup Event and Scheduler Job

**File:** `backend/main.py`

**Step 1: Add import near top (around line 20):**
```python
from backend.fantasy_baseball.yahoo_id_sync import run_yahoo_id_sync_job
from backend.fantasy_baseball.ballpark_factors import load_park_factors
```

**Step 2: Add startup event (around line 100, with other startup events):**
```python
@app.on_event("startup")
async def startup_event():
    """Load park factors on startup."""
    load_park_factors()
    logger.info("✅ Park factors loaded into memory")
```

**Step 3: Add scheduler job (around line 400, with other daily jobs):**
```python
# Yahoo ID sync (daily at 6 AM ET)
@scheduler.scheduled_job(trigger='cron', hour=6, minute=0, timezone='America/New_York')
def yahoo_id_sync_job():
    """Sync Yahoo player IDs from fantasy API."""
    logger.info("🔄 Running Yahoo ID sync job")
    try:
        run_yahoo_id_sync_job()
    except Exception as e:
        logger.error(f"❌ Yahoo ID sync failed: {e}", exc_info=True)
```

**Verify:** File modified, startup event and scheduler job added

---

## Compilation & Local Verification

### Step 1: Compile all modified files

```bash
# Compile backfill script
python -m py_compile scripts/backfill_player_type.py

# Compile Yahoo ID sync module
python -m py_compile backend/fantasy_baseball/yahoo_id_sync.py

# Compile ballpark_factors (should import without errors)
python -m py_compile backend/fantasy_baseball/ballpark_factors.py

# Compile main.py (should import without errors)
python -m py_compile backend/main.py
```

**Expected:** No output (success = no compilation errors)

### Step 2: Verify imports work

```bash
python -c "
from backend.fantasy_baseball.yahoo_id_sync import sync_yahoo_player_ids
print('✅ yahoo_id_sync imports OK')
"

python -c "
from backend.fantasy_baseball.ballpark_factors import load_park_factors, get_park_factor
print('✅ ballpark_factors imports OK')
"

python -c "
from backend.main import app
print('✅ main.py imports OK')
"
```

**Expected:** All print statements show ✅

---

## Deliverable Checklist

- [ ] `scripts/backfill_player_type.py` created (102 lines)
- [ ] `backend/fantasy_baseball/yahoo_id_sync.py` created (134 lines)
- [ ] `backend/fantasy_baseball/ballpark_factors.py` modified (cache added)
- [ ] `backend/main.py` modified (startup event + scheduler job)
- [ ] All files compile without errors
- [ ] All imports verified

**Total estimated time:** 1.5 hours

**DO NOT:**
- ❌ Run scripts on Railway (that's Gemini's job)
- ❌ Deploy to production (that's Gemini's job)
- ❌ Execute SQL directly (that's Gemini's job)

**DO:**
- ✅ Create all necessary files
- ✅ Verify compilation locally
- ✅ Document any errors for escalation
- ✅ Hand off to Gemini for deployment

---

## Escalation

If you encounter:
1. **Import errors** → Check file paths, verify dependencies
2. **Syntax errors** → Fix typos, verify Python 3.11 syntax
3. **Database schema issues** → Escalate with full error message
4. **API incompatibilities** → Check Yahoo client API signature

**File saved:** `.claude/delegation/claude-p0-implementation.md`
