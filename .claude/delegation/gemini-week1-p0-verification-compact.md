# Gemini CLI Week 1 P0 Tasks: Verification + Performance

**Date:** May 3, 2026
**Assigned:** Gemini CLI (DevOps Strike Lead)
**Timebox:** 6 hours total (2h verification + 4h performance)
**Escalate to:** Claude Code if backend fixes needed, DB schema changes, or deploy breaks

---

## MISSION

Two P0 blockers before we can build anything:
1. **Verify deployed features work** — 38 fantasy endpoints, unknown quality
2. **Fix optimizer timeout** — Users can't get optimal lineups (30s+)

DO NOT attempt backend code fixes. Your role: verification, profiling, reporting, deploying non-breaking changes (caching).

---

## PART 1: Endpoint Verification (2 hours)

### Step 1: Health Check

```bash
railway status
curl -s https://cbb-edge-production.up.railway.app/health | jq .
```

Expected: `{"status":"healthy","database":"connected","scheduler":"running"}`

### Step 2: Test Core Endpoints

```bash
# Test 1: Lineup endpoint
echo "=== Test 1: GET /api/fantasy/lineup ==="
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/lineup?date=2026-05-02" | head -c 500
echo ""

# Test 2: Roster optimization (PRIORITY - timing out)
echo "=== Test 2: POST /api/fantasy/roster/optimize ==="
timeout 35 curl -X POST "https://cbb-edge-production.up.railway.app/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}' \
  -w "\nResponse time: %{time_total}s\n"
echo ""

# Test 3: Matchup preview (MCMC win_prob)
echo "=== Test 3: GET /api/fantasy/matchup/preview ==="
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/matchup/preview?date=2026-05-02" | jq '.win_prob'
echo ""

# Test 4: Waiver recommendations
echo "=== Test 4: GET /api/fantasy/waiver/recommend ==="
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/waiver/recommend?league_id=72586" | head -c 500
echo ""

# Test 5: Player projections
echo "=== Test 5: GET /api/fantasy/projections ==="
time curl -s "https://cbb-edge-production.up.railway.app/api/fantasy/projections?player_type=hitter&limit=5" | jq '.[0] | {player_name, cat_scores}'
echo ""
```

### Step 3: Document Results

Create `reports/2026-05-03-endpoint-verification.md`:

```markdown
# Fantasy Endpoint Verification — May 3, 2026

## Health Check
- Status: [healthy/database_connected/scheduler_running]
- Timestamp: [ISO 8601]

## Endpoint Results

| Endpoint | Status | Response Time | Notes |
|----------|--------|---------------|-------|
| GET /api/fantasy/lineup | [✅/❌] | [Xms] | [Notes] |
| POST /api/fantasy/roster/optimize | [✅/❌/TIMEOUT] | [Xs] | [Notes] |
| GET /api/fantasy/matchup/preview | [✅/❌] | [Xms] | [win_prob varies or constant?] |
| GET /api/fantasy/waiver/recommend | [✅/❌] | [Xms] | [Notes] |
| GET /api/fantasy/projections | [✅/❌] | [Xms] | [cat_scores present?] |

## Critical Findings
- [Optimizer timeout confirmed? If yes, proceed to Part 2]
- [win_prob constant 0.763? If yes, REGRESSION]
- [Any 500 errors? If yes, BACKEND BUG]
```

---

## PART 2: Optimizer Performance (4 hours)

ONLY proceed if optimizer timeout confirmed in Part 1.

### Step 1: Profile the Timeout

```bash
# Set DATABASE_URL
export DATABASE_URL="postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

# Profile locally (30s should be enough to see bottleneck)
python -m cProfile -o profile.out -c "
from backend.main import app
import uvicorn
# Start server in background, make request, kill after 30s
" &

sleep 35
kill %1 2>/dev/null

# Analyze profile
python -c "
import pstats
p = pstats.Stats('profile.out')
p.sort_stats('cumulative')
p.print_stats(20)
" > profile_analysis.txt
```

### Step 2: Check for N+1 Queries

Create `scripts/count_optimizer_queries.py`:

```python
import os
import logging
os.environ['DATABASE_URL'] = "postgresql://postgres:oViPPSTbGvkNGzGjrYoxsLVvibJvJZAB@junction.proxy.rlwy.net:45402/railway"

logging.basicConfig()
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

from backend.fantasy_baseball.daily_lineup_optimizer import optimize_lineup

result = optimize_lineup(date="2026-05-02", yahoo_league_id="72586")
print(f"\nTotal recommendations: {len(result['recommendations'])}")
```

Run it:
```bash
venv\Scripts\python scripts/count_optimizer_queries.py 2>&1 | grep -c "SELECT"
```

**Red flag:** > 50 queries = N+1 problem

### Step 3: Check Database Indexes

```bash
railway run python -c "
from backend.models import engine
from sqlalchemy import text

with engine.connect() as conn:
    indexes = conn.execute(text('''
      SELECT tablename, indexname
      FROM pg_indexes
      WHERE tablename IN ('player_projections', 'player_scores', 'statcast_performances')
      ORDER BY tablename, indexname
    ''')).fetchall()

    for t in indexes:
        print(f'{t[0]:30} {t[1]:40}')
"
```

**Expected indexes:**
- `player_projections(yahoo_player_key)`
- `player_scores(player_id, updated_at)`
- `statcast_performances(player_id, game_date)`

### Step 4: Implement Caching

**File:** `backend/fantasy_baseball/mcmc_calibration.py`

Add after imports (around line 15):

```python
from functools import lru_cache
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# Cache player board for 5 minutes
@lru_cache(maxsize=1)
def _get_player_board_cached(cache_buster: str) -> tuple:
    """Cached wrapper around _get_player_board()."""
    return _get_player_board()
```

Modify `convert_yahoo_roster_to_mcmc_format()` (around line 180):

```python
def convert_yahoo_roster_to_mcmc_format(yahoo_roster: list) -> list:
    """Convert Yahoo roster to MCMC format with cached player board."""
    # Generate cache buster (current 5-minute window)
    now = datetime.now(ZoneInfo("America/New_York"))
    cache_buster = now.strftime("%Y%m%d%H%M")[:-1] + str(now.minute // 5 * 5)

    # Use cached board
    board, board_lookup = _get_player_board_cached(cache_buster)

    # Rest of function unchanged...
```

### Step 5: Benchmark After Caching

```bash
# Clear cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Warm cache (first call)
time curl -X POST "http://localhost:8000/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}'

# Measure cached call
time curl -X POST "http://localhost:8000/api/fantasy/roster/optimize" \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-02", "yahoo_league_id": "72586"}'
```

**Target:** < 5s for cached requests

### Step 6: Deploy to Railway

```bash
# Verify compilation
railway run python -m py_compile backend/fantasy_baseball/mcmc_calibration.py

# Deploy
railway up --detach

# Wait for deploy
sleep 30
curl -s https://cbb-edge-production.up.railway.app/health | jq .
```

### Step 7: Document Results

Create `reports/2026-05-03-optimizer-performance.md`:

```markdown
# Optimizer Performance Investigation — May 3, 2026

## Baseline Performance
- Response time: [TIMEOUT / Xs]
- Query count: [X queries]
- Top bottleneck: [function name]

## Caching Implementation
- Changes: [lru_cache added to _get_player_board, TTL=5min]
- Files modified: [mcmc_calibration.py]
- Lines changed: [~X lines]

## Results After Caching
- Response time (warm): [Xs]
- Query count reduction: [X%]
- Target <5s achieved: [YES/NO]

## Critical Findings
- [Missing indexes found? List them]
- [N+1 query problem? YES/NO]
- [Caching insufficient? What's next?]

## Recommendations
- [If <5s not met: algorithm rewrite needed]
- [If met: monitor in production]
```

---

## ESCALATION TRIGGERS

**Escalate to Claude Code immediately if:**

**Part 1:**
1. win_prob = 0.763 constant → **MCMC REGRESSION**
2. Any 500 errors → **BACKEND BUG**
3. Database connection fails → **PRODUCTION OUTAGE**

**Part 2:**
1. Missing indexes on key tables → **DB SCHEMA CHANGE NEEDED**
2. Caching alone doesn't achieve <5s → **ALGORITHM REWRITE NEEDED**
3. Deploy breaks optimizer → **ROLLBACK REQUIRED**
4. Cache invalidation bugs (stale data) → **DATA INTEGRITY ISSUE**

DO NOT attempt these fixes yourself — escalate with full findings.

---

## DELIVERABLES

1. `reports/2026-05-03-endpoint-verification.md` — Status matrix for 5 endpoints
2. `profile_analysis.txt` — Top 20 bottleneck functions
3. `reports/2026-05-03-optimizer-performance.md` — Before/after comparison
4. GitHub issue with title `[Verification] Week 1 P0 Complete — May 3, 2026`

---

## SUCCESS CRITERIA

**Part 1:**
- [ ] All 5 endpoints tested
- [ ] Response times documented
- [ ] Timeout confirmed (if present)
- [ ] Report saved

**Part 2:**
- [ ] Profiling completed
- [ ] Caching implemented
- [ ] Response time <5s (cached)
- [ ] Deployed to Railway
- [ ] Report saved

**Overall:**
- [ ] Zero backend code changes (caching only)
- [ ] Zero regressions (all still-green tests)
- [ ] Clear escalation path if blockers found
