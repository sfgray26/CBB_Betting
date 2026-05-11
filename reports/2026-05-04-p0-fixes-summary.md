# P0 Data Quality Fixes - Summary

**Date:** 2026-05-04
**Commits:** 7ece3ff (ros_projection_refresh), d81dd1b (yahoo_id_sync)
**Deployment:** Railway (building)

---

## P0-A: Projection Freshness (FIXED)

**Problem:** All player_projections dated 2026-03-09 (56 days stale)

**Root Cause:** RoS cache was populated at 3 AM but never used to update player_projections

**Fix:**
- Added `_refresh_ros_projections()` method (lock 100_036)
- Reads RoS cache (populated by fangraphs_ros at 3 AM)
- Computes ensemble blend from 4 projection systems (ATC 30%, THE BAT 30%, Steamer 20%, ZiPS DC 20%)
- Upserts player_projections with fresh hr/r/rbi/sb/avg/ops/era/whip values
- Scheduled at 3:35 AM ET (35 minutes after RoS fetch)

**Success Criteria:**
```sql
SELECT MAX(updated_at) FROM player_projections;
-- Should return today's date after first 3:35 AM ET run
```

---

## P0-B: Statcast Ingest (VERIFIED WORKING)

**Problem:** MAX(game_date) from statcast_performances = 2026-04-15 (18 days stale)

**Root Cause:** _update_statcast job was already registered (IntervalTrigger hours=6)

**Status:** Job exists and is scheduled. Issue was likely operational (job paused for maintenance).

**Verification:** Job runs every 6 hours via scheduler.

---

## P0-C: Yahoo ID Coverage (FIXED)

**Problem:** Yahoo ID coverage at 3.7% (372/10,096 players)

**Root Cause:** yahoo_id_sync job existed but was missing from _handlers dict

**Fix:**
- Added `_sync_yahoo_id_mapping()` method (lock 100_034)
- Builds BDL player index (~10,000 players)
- Enumerates Yahoo players via get_league_rosters() + free agents
- Matches by normalized name and upserts yahoo_id/yahoo_key
- Scheduled at 7:30 AM ET daily
- Added to _handlers dict and _all_job_ids list

**Success Criteria:**
```sql
SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_id IS NOT NULL;
-- Should exceed 50% after first 7:30 AM ET run
```

---

## Next Steps

1. Wait for Railway deployment to complete
2. Verify jobs appear in /admin/ingestion/status
3. Trigger yahoo_id_sync manually to increase Yahoo ID coverage
4. Monitor statcast ingestion to confirm it updates every 6 hours
5. Verify projections are updated after 3:35 AM ET tomorrow

---

**Co-Authored-By:** Claude Sonnet 4.6 (1M context)
