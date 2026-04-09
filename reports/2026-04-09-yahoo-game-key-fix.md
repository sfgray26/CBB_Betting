# Yahoo Game Key Fix - Critical Configuration Update

**Date:** April 9, 2026  
**Status:** READY FOR IMPLEMENTATION  
**ETA:** 5 minutes

---

## Problem Identified

The `position_eligibility` sync job is failing with:
```json
{
  "yahoo:uri": "/fantasy/v2/league/72586/teams/roster?format=json",
  "description": "Invalid game key provided - 72586"
}
```

**Root Cause:** `YAHOO_GAME_ID=72586` is incorrect for 2026 MLB season.

---

## Solution

### Correct Yahoo Game Key Format for 2026 MLB

| Component | Value | Notes |
|-----------|-------|-------|
| **Game ID** | `469` | Yahoo's 2026 MLB season prefix |
| **Format** | `469.l.{league_id}` | Standard league key format |
| **Example** | `469.l.123456` | Where 123456 = your league ID |

### How to Find Your League ID

1. Log into [Yahoo Fantasy Baseball](https://baseball.fantasysports.yahoo.com)
2. Navigate to your league
3. Look at URL: `https://baseball.fantasysports.yahoo.com/b1/{league_id}`
4. The number after `/b1/` is your league ID

---

## Implementation Steps

### Step 1: Update Railway Environment Variable

```bash
# Get current value (for reference)
railway variables get YAHOO_GAME_ID

# Set correct value (replace 123456 with actual league ID)
railway variables set YAHOO_GAME_ID="469.l.123456"
```

### Step 2: Verify Update

```bash
railway variables | grep YAHOO_GAME_ID
```

Expected output:
```
YAHOO_GAME_ID=469.l.123456
```

### Step 3: Re-test Position Eligibility Job

```bash
# Trigger job manually via API
curl -X POST "https://your-app.railway.app/admin/ingestion/run-job?job_id=position_eligibility"

# OR wait for next scheduled run (8:00 AM ET)
```

### Step 4: Verify Success

Check logs for:
```
JOB START: position_eligibility
SYNC JOB ENTRY: _sync_position_eligibility - Starting...
API CLIENT INIT: YahooFantasyClient - Initializing...
JOB COMPLETE: position_eligibility - records: ~750, elapsed_ms: XXX
```

---

## Validation Checklist

- [ ] `YAHOO_GAME_ID` updated to `469.l.{league_id}` format
- [ ] Railway variables show correct value
- [ ] position_eligibility job executes without "Invalid game key" error
- [ ] ~750 records written to position_eligibility table
- [ ] Multi-eligibility players present (e.g., Bellinger CF/LF/RF)

---

## Reference

- **Yahoo Fantasy API Docs:** Game key format is `{game_id}.l.{league_id}`
- **2026 MLB Game ID:** 469 (confirmed via yahoo-fantasy-baseball-reader project)
- **League Key Format:** `469.l.XXXXXX` where XXXXXX is your Yahoo league ID

---

**Next Action Required:** Update YAHOO_GAME_ID environment variable and re-test.
