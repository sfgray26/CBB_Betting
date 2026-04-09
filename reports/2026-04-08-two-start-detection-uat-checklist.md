# Two-Start Detection — UAT Validation Checklist

**Date:** April 8, 2026  
**Phase:** 2.2 — Two-Start Pitcher Detection  
**Status:** ✅ Logic Validation Complete | ⏳ Production Validation Pending

---

## Purpose

Full backend validation before UI consumption. Ensures the two-start detection pipeline is working correctly with real data before exposing via API.

---

## Validation Results (Local Development)

### Unit Tests: 9/10 PASS ✅

| Test ID | Test Name | Status | Notes |
|---------|-----------|--------|-------|
| UAT-01 | probable_pitchers table exists | SKIPPED | DB connection failed — validate on Railway |
| UAT-02 | TwoStartDetector initialization | ✅ PASS | Park factors loaded (30 teams) |
| UAT-03 | Returns valid TwoStartOpportunity objects | ✅ PASS | All required fields populated |
| UAT-04 | Matchup quality scores in valid range | ✅ PASS | -2.0 to +2.0, park factors 0.5-1.5 |
| UAT-05 | Streamer rating classification | ✅ PASS | EXCELLENT (≥1.0), GOOD (0-1.0), AVOID (<0) |
| UAT-06 | Acquisition method classification | ✅ PASS | ROSTERED/FREE_AGENT/WAIVER logic |
| UAT-07 | Data freshness validation | ✅ PASS | FRESH/STALE/MISSING thresholds |
| UAT-08 | IP projections realistic | ✅ PASS | 10-13 IP for 2 starts (5-6.5 per start) |
| UAT-09 | Fantasy week calculation | ✅ PASS | Increments every 7 days from Mar 28 |
| UAT-10 | End-to-end detection (mock DB) | ✅ PASS | Full pipeline validated |

**Test File:** `tests/test_two_start_detection_uat.py`

---

## Production Validation Checklist (Railway/UAT)

### Pre-Validation Requirements

- [ ] **P26 Migration Verified:** `probable_pitchers` table exists in both databases
- [ ] **Ingestion Job 100_014:** Probable pitcher sync job registered in `daily_ingestion.py`
- [ ] **MLB Stats API Access:** Verify `pybaseball` can fetch probable pitchers without errors

### Data Source Validation

- [ ] **Table Has Data:** `SELECT COUNT(*) FROM probable_pitchers` returns > 0 rows
- [ ] **Date Range Coverage:** Query has data for next 7 days (today through today+7)
- [ ] **Player Names Populated:** `pitcher_name` field NOT NULL for recent games
- [ ] **Opponent Data Complete:** `opponent`, `is_home`, `is_confirmed` populated for >90% of rows
- [ ] **Quality Scores Populated:** `quality_score` field NOT NULL (fallback to 0.0 if missing)

### Data Freshness Validation

- [ ] **Current Week Data:** Latest game_date within 48 hours for today's games
- [ ] **Confirmed Starters:** `is_confirmed=True` for >70% of games within 48h
- [ ] **Stale Data Detection:** Run `_validate_data_freshness()` — should return FRESH or STALE (not MISSING)

### Logic Validation

- [ ] **Two-Start Pitchers Found:** `detect_two_start_pitchers()` returns 0-20 opportunities (realistic range)
- [ ] **Quality Score Distribution:** Average quality scores range -1.0 to +1.0 (not all 0.0)
- [ ] **IP Projections Realistic:** All `total_ip_projection` values between 10-13
- [ ] **Streamer Rating Distribution:** Mix of EXCELLENT/GOOD/AVOID (not all one category)
- [ ] **Acquisition Methods:** Mix of ROSTERED/FREE_AGENT (not all one)

### Edge Case Validation

- [ ] **Single-Start Pitchers:** Pitchers with only 1 start in window NOT returned
- [ ] **Null Player Names:** Handled gracefully (player_name_confidence=LOW)
- [ ] **Missing Opponent Data:** Handled gracefully (quality_score=0.0)
- [ ] **Zero Quality Scores:** Classification as AVOID (not crash)
- [ ] **Negative Quality Scores:** Possible (should classify as AVOID)

### Performance Validation

- [ ] **Query Performance:** `detect_two_start_pitchers()` completes in <1 second for 7-day window
- [ ] **Database Indexes:** `_pp_date` index used in query (EXPLAIN ANALYZE)
- [ ] **Connection Pooling:** No connection leaks (run detection 5x in loop)

### Integration Validation

- [ ] **Matchup with probable_pitchers Table:** Raw SQL query returns correct columns
- [ ] **Player ID Resolution:** bdl_player_id mapping to player_name works
- [ ] **Park Factor Lookup:** All 30 teams in PARK_FACTORS dict
- [ ] **Fantasy Week Calculation:** Returns week 1-20 for 2026 season

---

## Failed Test Recovery Plan

If any production validation fails:

| Failure Type | Diagnosis | Recovery |
|--------------|------------|----------|
| Table doesn't exist | Run `scripts/migrate_v26_probable_pitchers.py` on Railway | Gemini CLI |
| No data in table | Run ingestion job 100_014 or check MLB Stats API access | Claude Code |
| Data is stale (>48h) | Check ingestion job schedule, trigger manually if needed | Claude Code |
| Logic errors | Compare mock results to production data, fix bugs | Claude Code |
| Performance issues | Add database indexes, optimize query | Claude Code |

---

## Sign-Off Criteria

**Block UI Until:** All production validation items above pass with [ ] checked.

**Minimum Viable Product (MVP):**
- probable_pitchers table has data for next 7 days
- detect_two_start_pitchers() returns at least 1 opportunity
- All data_freshness flags return FRESH or STALE (not MISSING)
- No crashes or exceptions in production logs

**Production Readiness:**
- All 10 UAT tests pass (including skipped UAT-01 on Railway)
- Data freshness <24h for games within 48h
- Performance <1s for 7-day window query

---

## Next Steps After Validation Pass

1. **Create API Endpoint:** `GET /api/fantasy/two-starts?start_date=2026-04-08&end_date=2026-04-15`
2. **Add Authentication:** Require valid API token (league admin only)
3. **Frontend Integration:** Two-Start Command Center UI (Phase 5 — Kimi CLI)
4. **Monitoring:** Add alert if data_freshness=MISSING detected

---

**Validator:** Claude Code (Principal Architect)  
**Review Required:** Kimi CLI (Weather & Park Factors Integration — K-29 active)

*Last Updated: April 8, 2026 — Session S28*
