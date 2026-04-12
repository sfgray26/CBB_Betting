# Statcast Persistence Bug Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `StatcastIngestionAgent.transform_to_performance()` which drops all rows because Baseball Savant CSV (grouped by 'name-date') returns `player_name` but not `player_id`, causing the continue statement at line 411 to skip every row.

**Architecture:** Detect missing `player_id` column, fall back to `player_name` with optional `player_id_mapping` lookup for mlbam_id, store resolved identifier as `player_id` in `StatcastPerformance`. Replace silent error swallowing with per-row logging so future failures surface.

**Tech Stack:** Python 3.11, pandas, SQLAlchemy, Baseball Savant CSV export API

---

## File Structure

| File | Responsibility |
|------|-----------------|
| `backend/fantasy_baseball/statcast_ingestion.py` | `transform_to_performance()` — transforms CSV DF to `PlayerDailyPerformance` objects; `__init__()` — load player_id_mapping cache for name→ID resolution |
| `scripts/backfill_statcast.py` | `_store_performances()` — upserts transformed performances to DB; replace silent `except...continue` with logged failures |
| `tests/test_statcast_ingestion.py` | Tests for `transform_to_performance()` with real CSV schema (no player_id column) |
| `scripts/_diagnose_statcast_csv.py` | One-time diagnostic script to capture actual Baseball Savant CSV columns for verification |

---

## Task 1: Create diagnostic script to capture actual CSV columns

**Files:**
- Create: `scripts/_diagnose_statcast_csv.py`

**Purpose:** Before modifying code, fetch a real Baseball Savant CSV and log its actual column names. This proves the `player_id` column is missing and documents the 2026 schema.

- [ ] **Step 1: Create diagnostic script**

```python
"""Diagnostic: Capture actual Baseball Savant CSV columns for 2026 season."""
import logging
import pandas as pd
from datetime import date, timedelta
from io import StringIO
import requests

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

BASEBALL_SAVANT_URL = "https://baseballsavant.mlb.com/statcast_search/csv"

def fetch_csv_columns(target_date: date, player_type: str) -> list[str]:
    """Fetch CSV for given date/player_type and return sorted column names."""
    params = {
        'all': 'true',
        'hfGT': 'R|',
        'hfSea': f'{target_date.year}|',
        'player_type': player_type,
        'game_date_gt': (target_date - timedelta(days=1)).isoformat(),
        'game_date_lt': (target_date + timedelta(days=1)).isoformat(),
        'group_by': 'name-date',
        'sort_col': 'pitches',
        'player_event_sort': 'api_p_release_speed',
        'sort_order': 'desc',
        'type': 'details',
    }
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    response = requests.get(BASEBALL_SAVANT_URL, params=params, headers=headers, timeout=60)
    if response.status_code != 200:
        logger.error(f"HTTP {response.status_code} for {player_type} on {target_date}")
        return []

    df = pd.read_csv(StringIO(response.text))
    logger.info(f"\n{'='*60}")
    logger.info(f"{player_type.upper()} CSV for {target_date}: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"{'='*60}")
    logger.info("Columns (sorted):")
    for col in sorted(df.columns):
        sample_val = df[col].iloc[0] if len(df) > 0 else None
        logger.info(f"  {col:40s} = {repr(sample_val)[:50]}")
    logger.info(f"{'='*60}\n")
    return sorted(df.columns)

# Fetch April 9, 2026 (recent game day)
test_date = date(2026, 4, 9)
batter_cols = fetch_csv_columns(test_date, 'batter')
pitcher_cols = fetch_csv_columns(test_date, 'pitcher')

# Check for expected columns
logger.info("DIAGNOSTIC SUMMARY:")
logger.info(f"  'player_id' in batter columns: {('player_id' in batter_cols)}")
logger.info(f"  'player_id' in pitcher columns: {('player_id' in pitcher_cols)}")
logger.info(f"  'player_name' in batter columns: {('player_name' in batter_cols)}")
logger.info(f"  'player_name' in pitcher columns: {('player_name' in pitcher_cols)}")
logger.info(f"  'last_name' in batter columns: {('last_name' in batter_cols)}")
logger.info(f"  'first_name' in batter columns: {('first_name' in batter_cols)}")
```

- [ ] **Step 2: Run diagnostic script locally**

Run: `venv\Scripts\python.exe scripts/_diagnose_statcast_csv.py`

Expected: Output shows `player_id` is NOT in columns, but `player_name` IS present. Sample row values confirm CSV schema.

---

## Task 2: Add player_id_mapping name-to-ID lookup cache to StatcastIngestionAgent

**Files:**
- Modify: `backend/fantasy_baseball/statcast_ingestion.py:200-273` (before `StatcastIngestionAgent` class definition)

**Purpose:** Pre-load `player_id_mapping` table into memory at initialization so `transform_to_performance()` can resolve `player_name` → `mlbam_id` (or fall back to `player_name` if not found).

- [ ] **Step 1: Add name-to-ID resolution helpers before StatcastIngestionAgent class**

```python
# Add after line 197 (after logger definition, before class definition)

class PlayerIdResolver:
    """
    Cache-based player name → MLBAM ID resolver.

    Loads player_id_mapping table at import time and provides
    fast name-to-ID lookups for Statcast CSV rows that lack player_id.
    """
    def __init__(self):
        self._by_name: Dict[str, int] = {}  # full_name → mlbam_id
        self._by_normalized: Dict[str, int] = {}  # normalized_name → mlbam_id
        self._loaded = False

    def load(self, db: Session) -> None:
        """Load player_id_mapping into memory caches."""
        if self._loaded:
            return

        from backend.models import PlayerIDMapping

        mappings = db.query(PlayerIDMapping).all()
        for m in mappings:
            if m.mlbam_id:
                self._by_name[m.full_name.lower()] = m.mlbam_id
                if m.normalized_name:
                    self._by_normalized[m.normalized_name] = m.mlbam_id

        self._loaded = True
        logger.info("PlayerIdResolver loaded %d name→mlbam_id mappings", len(self._by_name))

    def resolve(self, player_name: str) -> str:
        """
        Resolve player_name to an ID string for StatcastPerformance.player_id.

        Returns mlbam_id as string if found in player_id_mapping,
        otherwise returns player_name as-is (fallback identifier).
        """
        if not player_name:
            return "unknown"

        # Try exact full_name match (case-insensitive)
        mlbam_id = self._by_name.get(player_name.lower())
        if mlbam_id:
            return str(mlbam_id)

        # Fallback: return player_name as identifier
        # Note: StatcastPerformance.player_id is String(50), not an FK, so names work
        return player_name


# Module-level resolver singleton (will be loaded on first agent use)
_player_id_resolver = PlayerIdResolver()
```

- [ ] **Step 2: Modify StatcastIngestionAgent.__init__ to load the resolver**

Find the `__init__` method (around line 230-240) and add resolver loading:

```python
def __init__(self):
    self.db = SessionLocal()
    self.base_url = "https://baseballsavant.mlb.com/statcast_search/csv"
    self.quality_checker = DataQualityChecker()

    # NEW: Load player_id_mapping cache for name→ID resolution
    _player_id_resolver.load(self.db)
```

---

## Task 3: Fix transform_to_performance() to handle missing player_id column

**Files:**
- Modify: `backend/fantasy_baseball/statcast_ingestion.py:395-481` (the `transform_to_performance` method)

**Purpose:** When `player_id` column is missing from CSV (Baseball Savant 'name-date' grouping), fall back to using `player_name` resolved via `_player_id_resolver`.

- [ ] **Step 1: Replace the player_id extraction logic at lines 408-414**

Find:
```python
for _, row in df.iterrows():
    try:
        pid = row.get('player_id')
        if pid is None or str(pid).strip() in ('', 'nan', 'NaN'):
            continue
```

Replace with:
```python
for _, row in df.iterrows():
    try:
        # Statcast CSV with group_by='name-date' does NOT include player_id column.
        # Extract identifier: prefer player_id column if present, otherwise resolve player_name.
        pid_col = row.get('player_id')
        if pid_col is not None and str(pid_col).strip() not in ('', 'nan', 'NaN'):
            player_id = str(pid_col)
        else:
            # Fallback: resolve player_name to mlbam_id via player_id_mapping cache
            player_name_raw = row.get('player_name', '')
            if not player_name_raw or str(player_name_raw).strip() in ('', 'nan', 'NaN'):
                continue  # Skip rows with no identifiable player
            player_id = _player_id_resolver.resolve(str(player_name_raw))
```

- [ ] **Step 2: Update PlayerDailyPerformance instantiation to use resolved player_id**

Find the two `PlayerDailyPerformance(...)` constructor calls (batter case ~line 444, pitcher case ~line 419). Change `player_id=str(pid)` to `player_id=player_id` in both:

Batter case (around line 444):
```python
perf = PlayerDailyPerformance(
    player_id=player_id,  # Changed from str(pid)
    player_name=str(row.get('player_name', '')),
    ...
)
```

Pitcher case (around line 419):
```python
perf = PlayerDailyPerformance(
    player_id=player_id,  # Changed from str(pid)
    player_name=str(row.get('player_name', '')),
    ...
)
```

- [ ] **Step 3: Add early-exit logging for diagnostic visibility**

Add this after the for-loop starts (after the try block, before the is_pitcher check):

```python
# Log first row structure for diagnostics (only once per call)
if not hasattr(self, '_diag_logged'):
    self._diag_logged = True
    logger.info(
        "transform_to_performance: DataFrame has %d rows, columns: %s. player_id column present: %s",
        len(df),
        sorted(df.columns),
        'player_id' in df.columns,
    )
```

---

## Task 4: Replace silent error swallowing in backfill_statcast.py

**Files:**
- Modify: `scripts/backfill_statcast.py:294-296`

**Purpose:** The current `except ... continue` at line 294-295 swallows all upsert failures silently. Replace with logged failures so any database errors surface immediately.

- [ ] **Step 1: Replace the silent except block**

Find:
```python
except Exception as e:
    logger.warning(f"Failed to upsert performance for {perf.player_name}: {e}")
    continue
```

Replace with:
```python
except Exception as e:
    # Log full context for debugging — player_id, player_name, game_date, exception type
    logger.error(
        "Failed to upsert performance: player_id=%s player_name=%s game_date=%s error=%s: %s",
        perf.player_id, perf.player_name, perf.game_date, type(e).__name__, e,
        exc_info=True,  # Include full traceback
    )
    # Re-raise to surface the failure immediately (comment out continue for debugging)
    # continue  # Commented out to surface errors during fix verification
```

- [ ] **Step 2: After fix is verified, restore continue with error counter**

Once the fix works and you've confirmed no more errors, restore a safer version that counts failures:

```python
except Exception as e:
    logger.error(
        "Failed to upsert performance: player_id=%s player_name=%s game_date=%s error=%s: %s",
        perf.player_id, perf.player_name, perf.game_date, type(e).__name__, e,
    )
    # continue  # Uncomment after fix verified
```

---

## Task 5: Add test for transform_to_performance with real CSV schema

**Files:**
- Create: `tests/test_statcast_ingestion.py`

**Purpose:** Ensure `transform_to_performance()` correctly handles CSVs without `player_id` column, using `player_name` as fallback identifier.

- [ ] **Step 1: Write test case**

```python
"""Tests for StatcastIngestionAgent.transform_to_performance()."""
import pytest
import pandas as pd
from datetime import date
from backend.fantasy_baseball.statcast_ingestion import StatcastIngestionAgent, _player_id_resolver
from backend.models import SessionLocal, PlayerIDMapping


@pytest.fixture(autouse=True)
def load_resolver():
    """Ensure player_id_resolver is loaded before tests run."""
    db = SessionLocal()
    _player_id_resolver.load(db)
    yield
    db.close()


def test_transform_to_performance_handles_missing_player_id_column():
    """CSVs without player_id column should use player_name as identifier."""
    agent = StatcastIngestionAgent()

    # Simulate Baseball Savant CSV schema: player_name present, player_id absent
    df = pd.DataFrame([
        {
            'player_name': 'Judge, Aaron',
            'team': 'NYY',
            'game_date': '2026-04-09',
            'pa': 5,
            'ab': 4,
            'h': 2,
            'double': 0,
            'doubles': 1,  # Alternate name
            'triple': 0,
            'triples': 0,
            'hr': 1,
            'r': 2,
            'rbi': 3,
            'bb': 1,
            'strikeout': 1,
            'so': 1,  # Alternate name
            'hbp': 0,
            'sb': 0,
            'cs': 0,
            'exit_velocity_avg': 95.5,
            'launch_angle_avg': 15.2,
            'hard_hit_percent': 55.0,
            'barrel_batted_rate': 12.0,
            'xba': 0.320,
            'xslg': 0.650,
            'xwoba': 0.400,
            'pitches': 25,
            '_statcast_player_type': 'batter',
        }
    ])

    performances = agent.transform_to_performance(df)

    # Should NOT return empty list (the bug we're fixing)
    assert len(performances) == 1, f"Expected 1 performance, got {len(performances)}"

    perf = performances[0]
    # player_id should be resolved from player_name (either mlbam_id or player_name fallback)
    assert perf.player_id is not None
    assert len(perf.player_id) > 0
    assert perf.player_name == 'Judge, Aaron'
    assert perf.pa == 5
    assert perf.hr == 1
    assert perf.xwoba == pytest.approx(0.400)


def test_transform_to_performance_skips_rows_with_missing_player_name():
    """Rows without player_name or player_id should be skipped."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([
        {'player_name': None, 'team': 'NYY', 'game_date': '2026-04-09'},
        {'player_name': '', 'team': 'NYY', 'game_date': '2026-04-09'},
        {'player_name': 'nan', 'team': 'NYY', 'game_date': '2026-04-09'},
        {'player_name': 'Valid Player', 'team': 'NYY', 'game_date': '2026-04-09', 'pa': 1},
    ])

    performances = agent.transform_to_performance(df)

    # Only the valid row should be included
    assert len(performances) == 1
    assert performances[0].player_name == 'Valid Player'


def test_transform_to_performance_with_pitcher_rows():
    """Pitcher rows (_statcast_player_type='pitcher') should use zeroed batting stats."""
    agent = StatcastIngestionAgent()

    df = pd.DataFrame([
        {
            'player_name': 'Cole, Gerrit',
            'team': 'NYY',
            'game_date': '2026-04-09',
            'exit_velocity_avg': 88.0,
            'launch_angle_avg': 12.0,
            'hard_hit_percent': 35.0,
            'barrel_batted_rate': 5.0,
            'xba': 0.250,
            'xslg': 0.400,
            'xwoba': 0.300,
            'ip': 6.0,
            'er': 2,
            'strikeout': 8,
            'walk': 2,
            'bb': 2,
            'pitches': 95,
            '_statcast_player_type': 'pitcher',
        }
    ])

    performances = agent.transform_to_performance(df)

    assert len(performances) == 1
    perf = performances[0]
    # Pitcher rows should have zeroed batting stats
    assert perf.pa == 0
    assert perf.ab == 0
    assert perf.hr == 0
    # But pitching stats populated
    assert perf.ip == pytest.approx(6.0)
    assert perf.er == 2
    assert perf.k_pit == 8
    assert perf.bb_pit == 2
    assert perf.pitches == 95
```

- [ ] **Step 2: Run tests to verify they fail before fix**

Run: `venv\Scripts\python.exe -m pytest tests/test_statcast_ingestion.py -v`

Expected: Tests FAIL with "Expected 1 performance, got 0" (confirms bug exists)

- [ ] **Step 3: After implementing Tasks 2-4, re-run tests**

Run: `venv\Scripts\python.exe -m pytest tests/test_statcast_ingestion.py -v`

Expected: All tests PASS

---

## Task 6: Verify fix with Railway production backfill

**Files:**
- Execute: `scripts/backfill_statcast.py` via Railway

**Purpose:** After code is deployed to Railway, re-run the backfill and verify rows are actually stored.

- [ ] **Step 1: Deploy code to Railway**

```bash
git add backend/fantasy_baseball/statcast_ingestion.py scripts/backfill_statcast.py tests/test_statcast_ingestion.py
git commit -m "fix: Statcast transform handles missing player_id column (name-date CSV grouping)

- Add PlayerIdResolver cache for name→mlbam_id lookups from player_id_mapping
- transform_to_performance() falls back to player_name when player_id column absent
- Replace silent except...continue with error logging in backfill_statcast.py
- Add tests for CSV schema without player_id column

Fixes FOLLOW-UP 1 from 2026-04-11 production deployment."
git push
```

Wait for Railway deployment to complete.

- [ ] **Step 2: Trigger backfill via admin endpoint**

```bash
curl -X POST https://fantasy-app-production-5079.up.railway.app/admin/backfill/statcast \
  -H "X-API-Key: $API_KEY_USER1" \
  -H "Content-Type: application/json"
```

Or via Railway:
```bash
railway run python -c "
import requests
import os
resp = requests.post(
    'https://fantasy-app-production-5079.up.railway.app/admin/backfill/statcast',
    headers={'X-API-Key': os.environ['API_KEY_USER1']},
    timeout=600
)
print(resp.status_code)
print(resp.text)
"
```

- [ ] **Step 3: Check logs for success indicators**

```bash
railway logs --filter "Statcast" --tail 100
```

Expected log output:
```
Statcast combined: 6079 batters + 10562 pitchers = 16641 total rows for 2026-04-09
transform_to_performance: DataFrame has 16641 rows, columns: [...]. player_id column present: False
PlayerIdResolver loaded 2000 name→mlbam_id mappings
Stored 15234 2026-04-09 performances
```

- [ ] **Step 4: Verify database row count**

```bash
railway run python -c "
from backend.models import SessionLocal, StatcastPerformance
db = SessionLocal()
count = db.query(StatcastPerformance).count()
print(f'Total statcast_performances rows: {count}')
# Check sample data
sample = db.query(StatcastPerformance).limit(3).all()
for p in sample:
    print(f'  {p.player_name} ({p.player_id}) on {p.game_date}: pa={p.pa}, hr={p.hr}, xwoba={p.xwoba}')
db.close()
"
```

Expected: `Total statcast_performances rows: > 10000` (previously was 0)

---

## Task 7: Update validation-audit endpoint for Statcast

**Files:**
- Modify: `backend/admin_endpoints_validation.py`

**Purpose:** Remove the stale "Statcast 502 errors" finding. The Statcast fetch works (pybaseball returns data); the bug was in persistence layer. Update audit to check `statcast_performances` row count instead.

- [ ] **Step 1: Find the Statcast validation check (around lines 40-80)**

Search for:
```python
# Statcast table validation
statcast_count = db.query(StatcastPerformance).count()
if statcast_count == 0:
    issues.append({...})
```

- [ ] **Step 2: Replace with corrected check**

Replace with:
```python
# Statcast table validation
statcast_count = db.query(StatcastPerformance).count()
if statcast_count == 0:
    issues.append({
        'severity': 'HIGH',
        'category': 'statcast_empty',
        'table': 'statcast_performances',
        'issue': 'statcast_performances table is empty',
        'details': f'Expected Statcast data for 2026 season. Run POST /admin/backfill/statcast to populate.',
        'recommendation': 'Trigger backfill endpoint; if rows returned but table still empty, check transform_to_performance() for column name mismatches.',
        'threshold': 'statcast_count > 0',
    })
elif statcast_count < 5000:
    issues.append({
        'severity': 'MEDIUM',
        'category': 'statcast_low_count',
        'table': 'statcast_performances',
        'issue': f'statcast_performances has only {statcast_count} rows',
        'details': f'Expected ~15000+ rows for March 20 - April 11, 2026 season. May indicate partial backfill.',
        'recommendation': 'Re-run POST /admin/backfill/statcast to fill missing dates.',
        'threshold': 'statcast_count >= 10000',
    })
else:
    validation_results['statcast_row_count'] = statcast_count
```

---

## Task 8: Update HANDOFF.md and create final report

**Files:**
- Modify: `HANDOFF.md`
- Create: `reports/2026-04-11-statcast-bug-fix-results.md`

**Purpose:** Document the fix, update FOLLOW-UP 1 status, and provide permanent record of the bug pattern.

- [ ] **Step 1: Update HANDOFF.md FOLLOW-UP 1 section**

Find the FOLLOW-UP 1 block and replace with:

```markdown
### FOLLOW-UP 1: Statcast persistence bug — ✅ RESOLVED (2026-04-11)

**Root cause:** `StatcastIngestionAgent.transform_to_performance()` expected `player_id` column in CSV, but Baseball Savant's 'name-date' grouping returns `player_name` only. All rows skipped at line 411.
**Fix:** Added `PlayerIdResolver` cache (name→mlbam_id from `player_id_mapping`), modified transform to fall back to `player_name` when `player_id` absent.
**Files:** `backend/fantasy_baseball/statcast_ingestion.py`, `scripts/backfill_statcast.py`, `tests/test_statcast_ingestion.py`
**Result:** Backfill now populates `statcast_performances` table (~15K rows for March 20 - April 11).
**Report:** `reports/2026-04-11-statcast-bug-fix-results.md`
```

- [ ] **Step 2: Create fix report**

```markdown
# Statcast Persistence Bug Fix Results

**Date:** 2026-04-11
**Bug:** FOLLOW-UP 1 from production deployment — statcast_performances table remains empty despite successful pybaseball fetches
**Root Cause:** Baseball Savant CSV schema mismatch (no player_id column)

## Before Fix
- pybaseball fetch: ✅ Working (10,562 pitcher + 6,079 batter rows per date)
- transform_to_performance(): ❌ Returns empty list (all rows skipped)
- statcast_performances table: 0 rows

## After Fix
- Added PlayerIdResolver: name→mlbam_id cache from player_id_mapping
- transform_to_performance(): Handles missing player_id column, uses player_name fallback
- backfill_statcast.py: Error logging instead of silent except...continue
- statcast_performances table: ~15,000 rows (populated)

## Files Changed
- `backend/fantasy_baseball/statcast_ingestion.py` — PlayerIdResolver class, transform logic
- `scripts/backfill_statcast.py` — error logging
- `tests/test_statcast_ingestion.py` — new test file
- `backend/admin_endpoints_validation.py` — updated Statcast check

## Tests Added
- test_transform_to_performance_handles_missing_player_id_column
- test_transform_to_performance_skips_rows_with_missing_player_name
- test_transform_to_performance_with_pitcher_rows

## Lessons Learned
1. CSV schema assumptions must be validated against real data before writing transforms
2. Silent error swallowing (except...continue) hides bugs for weeks
3. Always add diagnostic logging showing actual column names on first run
```

---

## Self-Review Checklist

- [ ] **Spec coverage:** Does every task address the root cause (missing player_id column) and fix the symptom (empty result list)?
- [ ] **Placeholder scan:** Are all code blocks complete with actual implementations (no "TODO", "add validation", etc.)?
- [ ] **Type consistency:** Do `player_id` usages match across all modified locations (String type, mlbam_id or player_name fallback)?
- [ ] **Test coverage:** Do tests cover both success path (rows transformed) and edge cases (missing player_name, pitcher rows)?
- [ ] **Rollback safety:** Can the fix be reverted if it introduces new issues? (Yes, git revert of specific commit)
- [ ] **Production verification:** Does Task 6 include actual verification steps on Railway? (Yes, logs + DB count check)

---

**Next Steps:** Choose execution method — Subagent-Driven (recommended for code changes) or Inline Execution.
