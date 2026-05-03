# Task 3: Lineup Optimizer Performance Investigation (Gemini CLI)

**Priority:** P0 — Users cannot get optimal lineups (30s timeout)
**Assigned:** Gemini CLI (DevOps Strike Lead)
**Escalation:** Claude Code if database schema changes needed
**Timebox:** 4 hours

---

## Mission

Investigate why `/api/fantasy/roster/optimize` is timing out (>30s). Profile the code, identify bottlenecks, implement caching to target <5s response time.

---

## Problem Context

From PRODUCTION_STATUS.md:
> `/api/fantasy/roster/optimize` timing out (>30s)
> No win_prob_gain logs found (feature may not be tested)
> **Impact**: Users can't get optimal lineups
> **Priority**: P0 - Core fantasy feature

---

## Investigation Plan

### Phase 1: Reproduce and Profile (1 hour)

#### Step 1.1: Reproduce the timeout locally

```bash
# Set DATABASE_URL to production
export DATABASE_URL="postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

# Run the optimizer with Python profiling
python -m cProfile -o profile_output.out -m backend.main &
PID=$!
sleep 35
kill $PID 2>/dev/null

# Check if it's still running after 30s
ps aux | grep "backend.main" | grep -v grep

# Analyze the profile
python -c "
import pstats
p = pstats.Stats('profile_output.out')
p.sort_stats('cumulative')
p.print_stats(20)  # Top 20 functions by cumulative time
p.sort_stats('time')
p.print_stats(20)  # Top 20 functions by own time
"
```

#### Step 1.2: Check for N+1 queries

Add SQLAlchemy logging:

```python
# Create temporary script: scripts/check_optimizer_queries.py
import os
import logging
os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

# Enable SQLAlchemy query logging
logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

from backend.fantasy_baseball.daily_lineup_optimizer import optimize_lineup

# Run a single optimization
result = optimize_lineup(
    date="2026-05-02",
    yahoo_league_id="72586"
)
print(f"\nOptimization complete: {len(result['recommendations'])} recommendations")
```

Run it and count the queries:
```bash
venv\Scripts\python scripts/check_optimizer_queries.py 2>&1 | grep "SELECT" | wc -l
```

**Red flag:** > 50 queries = likely N+1 problem

#### Step 1.3: Check database indexes

```bash
railway run python -c "
from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    # Get table sizes
    tables = conn.execute(text('''
      SELECT schemaname, tablename, n_live_tup, n_dead_tup
      FROM pg_stat_user_tables
      WHERE tablename LIKE '%player%' OR tablename LIKE '%projection%'
      ORDER BY n_live_tup DESC
    ''')).fetchall()

    for t in tables:
        print(f'{t[1]:30} rows: {t[2]:7}  dead: {t[3]:7}')

    # Check indexes on key tables
    indexes = conn.execute(text('''
      SELECT
        tablename,
        indexname,
        indexdef
      FROM pg_indexes
      WHERE tablename IN ('player_projections', 'player_scores', 'statcast_performances')
      ORDER BY tablename, indexname
    ''')).fetchall()

    print('\n--- INDEXES ---')
    for i in indexes:
        print(f'{i[0]:30} {i[1]:40}')
"
```

Look for missing indexes on:
- `player_projections(yahoo_player_key)`
- `player_scores(player_id, updated_at)`
- `statcast_performances(player_id, game_date)`

---

### Phase 2: Implement Caching (2 hours)

#### Step 2.1: Add in-memory cache for player board

The `_get_player_board()` function in `mcmc_calibration.py` is likely called on every request.

**File:** `backend/fantasy_baseball/mcmc_calibration.py`

**Add cache decorator:**
```python
from functools import lru_cache
from datetime import datetime, timedelta

# Cache player board for 5 minutes
@lru_cache(maxsize=1)
def _get_player_board_cached(cache_buster: str) -> tuple[list, dict]:
    """
    Cached wrapper around _get_player_board().
    cache_buster: timestamp string to invalidate cache every 5 min
    """
    return _get_player_board()

def convert_yahoo_roster_to_mcmc_format(yahoo_roster: list[dict]) -> list[dict]:
    """
    Convert Yahoo roster to MCMC format with cached player board.
    """
    # Generate cache buster (current 5-minute window)
    now = datetime.now(ZoneInfo("America/New_York"))
    cache_buster = now.strftime("%Y%m%d%H%M")[:-1] + str(now.minute // 5 * 5)

    # Use cached board
    board, board_lookup = _get_player_board_cached(cache_buster)

    # Rest of function unchanged...
```

#### Step 2.2: Add database query cache for projections

**File:** `backend/fantasy_baseball/projections_loader.py`

**Add Redis-style cache (or in-memory if Redis not available):**
```python
from datetime import datetime, timedelta
from typing import Optional

# In-memory cache (simple dict-based)
_projection_cache: dict[str, tuple[dict, datetime]] = {}
CACHE_TTL = timedelta(minutes=10)

def get_player_projection(yahoo_player_key: str) -> Optional[dict]:
    """
    Get player projection from cache or database.
    """
    # Check cache
    if yahoo_player_key in _projection_cache:
        projection, cached_at = _projection_cache[yahoo_player_key]
        if datetime.now() - cached_at < CACHE_TTL:
            return projection  # Cache hit

    # Cache miss or expired: query DB
    projection = _query_projection_from_db(yahoo_player_key)

    if projection:
        _projection_cache[yahoo_player_key] = (projection, datetime.now())

    return projection
```

#### Step 2.3: Add caching to lineup optimizer

**File:** `backend/fantasy_baseball/daily_lineup_optimizer.py`

**Cache team implied runs calculation:**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def calculate_team_implied_runs_cache(
    mlb_team: str,
    opponent: str,
    date_str: str,
    total: float,
    spread: float
) -> float:
    """
    Cache implied runs calculation per team/game.
    """
    return calculate_team_implied_runs(mlb_team, opponent, total, spread)
```

---

### Phase 3: Benchmark and Verify (1 hour)

#### Step 3.1: Test with cache enabled

```bash
# Clear Python cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Warm up cache (first call)
time curl -X POST "http://localhost:8000/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}'

# Measure cached response time
time curl -X POST "http://localhost:8000/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}'
```

**Target:** < 5s for cached requests

#### Step 3.2: Profile after caching

```bash
python -m cProfile -o profile_cached.out -m backend.main &
PID=$!
sleep 10  # Should complete in <10s now
kill $PID 2>/dev/null

python -c "
import pstats
p = pstats.Stats('profile_cached.out')
p.sort_stats('cumulative')
p.print_stats(10)
"
```

Compare:
- Before caching: [X seconds, Y function calls]
- After caching: [X seconds, Y function calls]

**Success:** 60%+ reduction in response time

#### Step 3.3: Deploy to Railway

```bash
# Test compilation
railway run python -m py_compile backend/fantasy_baseball/mcmc_calibration.py
railway run python -m py_compile backend/fantasy_baseball/daily_lineup_optimizer.py

# Deploy
railway up --detach

# Verify production health
sleep 30  # Wait for deploy
curl https://cbb-edge-production.up.railway.app/health
```

---

## Deliverable

Create file: `reports/2026-05-03-optimizer-performance-investigation.md`

```markdown
# Lineup Optimizer Performance Investigation — May 3, 2026

## Executive Summary
- [Before: Xs timeout / After: Ys cached]
- [Performance improvement: Z%]

## Phase 1: Profiling Results

### Baseline Performance
- Response time: [Xs]
- Query count: [X queries]
- Top bottlenecks:
  1. [Function] - [X% time]
  2. [Function] - [X% time]
  3. [Function] - [X% time]

### N+1 Query Check
- Total queries: [X]
- Red flag: [YES/NO]
- Problematic functions: [List]

### Database Indexes
- player_projections indexes: [List or NONE]
- player_scores indexes: [List or NONE]
- statcast_performances indexes: [List or NONE]

## Phase 2: Caching Implementation

### Changes Made
1. [Cache added: location, TTL, strategy]
2. [Cache added: location, TTL, strategy]
3. [Cache added: location, TTL, strategy]

### Code Changes
- Files modified: [List]
- Lines changed: [X]

## Phase 3: Results

### Performance After Caching
- Response time (warm): [Xs]
- Response time (cold): [Xs]
- Target achieved: [YES/NO]

### Function Call Reduction
- Before: [X calls]
- After: [X calls]
- Reduction: [X%]

## Critical Findings
- [Any database issues (indexes, locks)]
- [Any code issues (inefficient loops, redundant calculations)]

## Recommendations
- [If <5s target not met: what's next?]
- [If met: what to optimize next?]
```

---

## Success Criteria

- [ ] Timeout reproduced and profiled
- [ ] Bottleneck identified (N+1 queries / missing indexes / expensive calculation)
- [ ] Caching implemented (board cache / projection cache / implied runs cache)
- [ ] Response time < 5s (cached)
- [ ] Deployed to Railway and verified
- [ ] Report saved to `reports/2026-05-03-optimizer-performance-investigation.md`

---

## Escalation Triggers

**Escalate to Claude Code if:**
1. Database schema changes needed (new indexes) — **requires migration**
2. Caching alone doesn't achieve <5s — **algorithm rewrite needed**
3. Deploy breaks optimizer — **rollback required**
4. Cache invalidation bugs (stale data served) — **data integrity issue**

---

## Code Review Checklist

Before deploying, verify:
- [ ] Cache TTL is appropriate (5-10 min, not hours)
- [ ] Cache invalidation works (new data refreshes cache)
- [ ] No race conditions (multiple requests don't corrupt cache)
- [ ] Memory usage acceptable (lru_cache size limited)
- [ ] Logging added (cache hits/misses visible)

---

## Reporting Format

After completion, create a GitHub issue with title:
```
[Performance] Lineup Optimizer Optimization — May 3, 2026
```

Body: Paste the contents of `reports/2026-05-03-optimizer-performance-investigation.md`

Include:
- Before/after timing comparison
- Code diff (if caching implemented)
- Railway deploy confirmation

Tag @claude-code (me) for review.
