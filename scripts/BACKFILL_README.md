# Backfill Scripts — Data Completeness Validation Gate

**CRITICAL:** These scripts must be executed before ANY feature development or UI work can proceed.

## Why This Is Necessary

MLB season opened March 2026. Today is April 8, 2026. We have ~18 days of season data that MUST be backfilled before we can trust ANY recommendations from the system.

**What happens if we skip this:**
- ❌ Scarcity Index will recommend common players as "scarce" (WRONG)
- ❌ Two-Start Detector will be BLIND to rotation changes and injuries
- ❌ Projections will be based on incomplete data (GARBAGE)
- ❌ User trust will be destroyed immediately

## Backfill Scripts

### 1. `backfill_player_id_mapping.py` — CRITICAL (FOUNDATION)

**What:** Fetches all MLB players from BallDon'tLie API and stores cross-reference mapping

**Table:** `player_id_mapping`

**Expected Records:** ~1,500 (all MLB players)

**Data Sources:** BallDon'tLie GOAT API

**Dependencies:** None

**Runtime:** 2-5 minutes

**Usage:**
```bash
python scripts/backfill_player_id_mapping.py
```

**Why It's First:** All other backfills depend on this to map player IDs between systems (BDL ↔ MLB ↔ Yahoo)

---

### 2. `backfill_positions.py` — CRITICAL

**What:** Fetches CURRENT position eligibility from Yahoo Fantasy API for all 30 teams

**Table:** `position_eligibility`

**Expected Records:** ~750 (30 teams × ~25 players with multi-eligibility)

**Data Sources:** Yahoo Fantasy API

**Dependencies:** `player_id_mapping` (to map Yahoo player keys to BDL IDs)

**Runtime:** 3-8 minutes

**Usage:**
```bash
python scripts/backfill_positions.py
```

**Note:** Yahoo API does NOT expose historical position data — only current snapshot. Ongoing daily_ingestion job will track changes over time.

---

### 3. `backfill_probable_pitchers.py` — CRITICAL

**What:** Fetches historical probable pitchers (March 20 - April 8, 2026)

**Table:** `probable_pitchers`

**Expected Records:** ~540 (18 days × 30 teams)

**Data Sources:** BallDon'tLie Games API

**Dependencies:** `player_id_mapping` (to map pitcher names to BDL IDs)

**Runtime:** 5-10 minutes

**Usage:**
```bash
python scripts/backfill_probable_pitchers.py
```

**Why It's Critical:** Two-Start Detector cannot identify opportunities without historical probable pitchers data.

---

### 4. `backfill_statcast.py` — HIGH PRIORITY

**What:** Fetches historical Statcast data from Baseball Savant (March 20 - April 8, 2026)

**Table:** `statcast_performances`

**Expected Records:** ~20,000 (18 days × ~750 players)

**Data Sources:** Baseball Savant CSV export API

**Dependencies:** None (uses MLBAM IDs directly from Baseball Savant)

**Runtime:** 10-20 minutes

**Usage:**
```bash
python scripts/backfill_statcast.py
```

**Why It's Slow:** Baseball Savant API has rate limits and we're fetching ~18 days of data.

**Note:** This is the same data source as the daily statcast ingestion job. If that job is returning 0 records, this backfill may also fail — investigate the API integration.

---

### 5. `backfill_bdl_stats.py` — MEDIUM PRIORITY

**Status:** NOT YET CREATED

**What:** Fetches historical player game stats from BDL (March 20 - April 8, 2026)

**Table:** `mlb_player_stats`

**Expected Records:** ~13,500 (18 days × 30 teams × ~25 players)

**Data Sources:** BallDon'tLie Player Stats API

**Dependencies:** Game logs must exist first

**Runtime:** TBD

---

### 6. `backfill_game_log.py` — MEDIUM PRIORITY

**Status:** NOT YET CREATED

**What:** Fetches historical game summaries from BDL (March 20 - April 8, 2026)

**Table:** `mlb_game_log`

**Expected Records:** ~270 (18 days × ~15 games)

**Data Sources:** BallDon'tLie Games API

**Dependencies:** None

**Runtime:** TBD

---

## Master Orchestration Script

### `backfill_all_data.py`

Runs all backfill scripts in the correct order with validation between each step.

**Usage:**
```bash
python scripts/backfill_all_data.py
```

**Features:**
- ✅ Executes scripts in dependency order
- ✅ Stops on critical failures
- ✅ Reports progress and elapsed time
- ✅ Provides summary of successes/failures

**Expected Runtime:** 20-30 minutes total

---

## Validation Checklist

After running backfills, verify data completeness:

### Automated Checks
```bash
# Check table counts
curl https://<your-app>.railway.app/admin/audit-tables

# Expected results:
# - player_id_mapping: 1,400+ rows
# - position_eligibility: 700+ rows
# - probable_pitchers: 500+ rows
# - statcast_performances: 15,000+ rows
```

### Manual Spot-Checks

**Player ID Mapping:**
```sql
-- Check random players have cross-reference data
SELECT * FROM player_id_mapping
WHERE full_name IN ('Mike Trout', 'Shohei Ohtani', 'Mookie Betts')
LIMIT 5;

-- Verify BDL IDs are not NULL
SELECT COUNT(*) FROM player_id_mapping WHERE bdl_player_id IS NULL;
-- Should be 0
```

**Position Eligibility:**
```sql
-- Check multi-eligibility (e.g., Bellinger CF/LF/RF)
SELECT player_name, COUNT(*) as position_count
FROM position_eligibility
GROUP BY player_name
HAVING COUNT(*) > 1
ORDER BY position_count DESC
LIMIT 10;

-- Verify CF/LF/RF breakdown
SELECT position_type, COUNT(*) as player_count
FROM position_eligibility
WHERE position_type IN ('LF', 'CF', 'RF')
GROUP BY position_type;
```

**Probable Pitchers:**
```sql
-- Check date range coverage
SELECT MIN(game_date) as earliest, MAX(game_date) as latest, COUNT(*) as total
FROM probable_pitchers;

-- Verify no gaps > 2 days
SELECT game_date, COUNT(*) as teams
FROM probable_pitchers
GROUP BY game_date
ORDER BY game_date;
```

**Statcast Performances:**
```sql
-- Check data quality
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT game_date) as days_covered,
    COUNT(DISTINCT player_id) as unique_players,
    AVG(pa) as avg_pa_per_player
FROM statcast_performances;

-- Verify xwOBA distribution (should be ~0.300 average)
SELECT 
    COUNT(*) as players,
    AVG(xwoba) as avg_xwoba,
    MIN(xwoba) as min_xwoba,
    MAX(xwoba) as max_xwoba
FROM statcast_performances
WHERE pa > 0;
```

---

## Troubleshooting

### Statcast Ingestion Returns 0 Records

**Symptom:** `statcast_performances` table is empty despite job being scheduled

**Possible Causes:**
1. **Date encoding issue** — Baseball Savant API uses strict inequality for dates (game_date_gt/lt are exclusive bounds)
2. **API changed** — Baseball Savant endpoint may have changed
3. **Off-day or weather postponements** — No games on target date
4. **Rate limiting** — Too many requests too quickly

**Debug Steps:**
```bash
# Manual test for a specific date
python -m backend.fantasy_baseball.statcast_ingestion 2026-04-06

# Check logs for API errors
railway logs --service <cbb-edge-service> | grep -i statcast

# Try different date (known game day)
python -m backend.fantasy_baseball.statcast_ingestion 2026-04-05
```

**Fix:** Update `backend/fantasy_baseball/statcast_ingestion.py` date parameters if API changed.

---

### BDL Rate Limits

**Symptom:** Backfill script fails with "rate limit exceeded" errors

**Solution:** Built-in delays between API calls. If still hitting limits, add longer delays:

```python
# In backfill scripts, add:
import time
time.sleep(2)  # 2 second delay between requests
```

---

### Yahoo Authentication Issues

**Symptom:** `backfill_positions.py` fails with authentication error

**Solution:** Verify OAuth token is valid:
```bash
python -c "from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient; c = YahooFantasyClient(); print(c.get_my_team_key())"
```

If this fails, re-authenticate via Yahoo OAuth flow.

---

## Next Steps After Backfill

1. ✅ **Verify table counts** via `/admin/audit-tables`
2. ✅ **Manual spot-checks** (see above)
3. ✅ **Create data health dashboard** — `/admin/data-health` endpoint
4. ✅ **Deploy ingestion jobs** to Railway (G-30)
5. ✅ **Deploy Redis caching** to Railway (G-29)
6. ✅ **Proceed to Phase 2.3** (Scarcity Index Computation)
7. ✅ **Proceed to Phase 3** (API Layer)

---

## Data Completeness Validation Gate

**MANDATORY REQUIREMENT:** Do NOT proceed to Phase 2.3 or Phase 3 until:

- [ ] `player_id_mapping` has 1,400+ rows with BDL ↔ MLB cross-references
- [ ] `position_eligibility` has 700+ rows with CF/LF/RF multi-eligibility
- [ ] `probable_pitchers` covers March 20 - April 8 with <2 day gaps
- [ ] `statcast_performances` has 15,000+ rows with reasonable stat distributions
- [ ] Manual spot-checks pass (5 random players have complete data)
- [ ] No critical data quality errors (NULL IDs, negative counts, etc.)

**Consequences of Violating This Gate:**
- Wrong scarcity calculations → Wrong recommendations
- Blind two-start detection → Missed opportunities
- Invalid projections → Lost user trust
- Wasted development time building on broken foundation

---

**Reference:** HANDOFF.md Directive 3 (Data Completeness Validation Gate)
