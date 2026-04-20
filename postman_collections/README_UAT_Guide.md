# UAT Guide - CBB Edge MLB Platform

## Quick Start

1. **Open Postman** and import:
   - `CBB_Edge_UAT_Collection.json` (Collection)
   - `CBB_Edge_UAT_Environment.postman_environment.json` (Environment)

2. **Configure the Environment:**
   - Click the gear icon (top right) → "CBB Edge - UAT Environment"
   - Edit `baseUrl` → your Railway URL (e.g., `https://cbb-edge-production.up.railway.app`)
   - Edit `apiKey` → your API key from `.env` (`API_KEY_USER1`)
   - Edit `teamKey` → your Yahoo team key (find via `/admin/yahoo/roster-raw`)

3. **Select the Environment** from the dropdown (top right, next to the eye icon)

4. **Run the Health Check first:**
   - Open `0_Health_Check` folder
   - Send `Root Endpoint` → Should return API name/version
   - Send `Health Check` → Should return `status: healthy`

## Suggested UAT Order

### Phase 1: Smoke Tests (5 min)
Run these first to verify basic connectivity:
1. `0_Health_Check` → Both endpoints
2. `2_Scoreboard` → Verify scoreboard loads
3. `3_Budget` → Verify budget data loads

### Phase 2: Core Fantasy Features (20 min)
1. `1_Fantasy_Baseball` → `Roster Management` → `Get My Roster`
2. `1_Fantasy_Baseball` → `Lineup Management` → `Get Daily Lineup`
3. `1_Fantasy_Baseball` → `Matchup` → `Get Current Matchup`
4. `1_Fantasy_Baseball` → `Waiver Wire` → `Get Waiver Wire`

### Phase 3: Data Quality Validation (30 min)
1. `6_Admin_Endpoints` → `System Status` → `Ingestion Status`
   - Check for `projection_freshness.violations` (should be empty)
2. `6_Admin_Endpoints` → `Database Admin` → `Audit Tables`
   - Verify critical tables have rows (player_daily_metric, projections, etc.)
3. `6_Admin_Endpoints` → `Yahoo Admin` → `Yahoo Connection Test`
   - Should return `connected: true`

### Phase 4: Advanced Features (20 min)
1. `1_Fantasy_Baseball` → `Decisions` → `Get Decisions`
2. `1_Fantasy_Baseball` → `Player Scores` → `Get Player Scores`
3. `1_Fantasy_Baseball` → `Daily Briefing` → `Get Daily Briefing`

## Interpreting Results

### Success Indicators
- **200 OK** → Request succeeded
- **Tests passed** (bottom of response) → Response structure is valid
- **Console log** → Check for useful info (e.g., "EMPTY TABLES: ...")

### Common Issues

| Issue | Likely Cause | Action |
|-------|--------------|--------|
| 401 Unauthorized | Invalid API key | Check `apiKey` in environment |
| 503 Unavailable | Stale projections | Check `projection_freshness` violations |
| Empty arrays | Missing data | Run `/admin/audit-tables` to check |
| Null fields | Data not populated | Run backfill endpoints if needed |
| Timeout | Slow response | Note for perf investigation |

### Data Quality Red Flags

1. **Freshness violations** in `/admin/ingestion/status` → Projections may be stale
2. **Empty player lists** in roster/lineup → Yahoo sync may be broken
3. **Null scores** in scoreboard → Projection pipeline may be incomplete
4. **Zero row counts** in audit tables → Ingestion jobs may have failed

## Documenting Findings

Use `UAT_Checklist_Template.md` to track:
- ✅ Pass → Feature works as expected
- ❌ Fail → Feature broken or missing data
- ⚠️ Partial → Works but has issues

For each finding, note:
- **Endpoint** that failed
- **Expected** vs **Actual** behavior
- **Impact** on UI development (blocker vs. nice-to-have)

## Quick Tests for Specific Scenarios

### Testing Projection Freshness
```bash
GET /admin/ingestion/status
Look at: projection_freshness.violations
Should be: [] (empty)
```

### Testing Yahoo Connection
```bash
GET /admin/yahoo/test
Look at: connected
Should be: true
```

### Testing Database Health
```bash
GET /admin/audit-tables
Look for tables with row_count = 0
Critical tables: player_daily_metric, player_projection, fantasy_roster
```

### Testing Scoreboard Completeness
```bash
GET /scoreboard
Look at: categories.length
Should be: 18 (for 18-category H2H)
Each category should have: my_score, opponent_score, winner, projection
```

## Running All Tests

Postman can run the entire collection automatically:
1. Click "..." on collection → "Run collection"
2. Select "CBB Edge - UAT Environment"
3. Click "Run"

Review the results summary for failures.

## Tips for Efficient UAT

1. **Use tabs** → Keep multiple endpoints open for quick comparison
2. **Save responses** → Right-click response → "Save response" for later analysis
3. **Use tests tab** → Auto-tests catch common issues
4. **Check console** → Postman console (View → Show Console) has debug logs
5. **Document as you go** → Don't wait until the end to fill the checklist

## Post-UAT

1. Create GitHub issues for all failures
2. Label them `P1` (blocker), `P2` (important), `P3` (polish)
3. Update HANDOFF.md with UAT summary
4. Archive the Postman collection with date stamp
