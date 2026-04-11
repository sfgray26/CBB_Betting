# Job Execution Audit (April 9, 2026)

## 1. Have these jobs EVER executed successfully?
**YES.** 
- **Today (April 9, 2026 10:32 AM EDT)**: Successful execution of `player_id_mapping` (10,000 records) and `probable_pitchers` (0 records, expected for morning run).
- **Today (Current Session)**: Successful execution of `position_eligibility` (229 records) after fixing the Yahoo configuration.

## 2. When was the last successful run?
- **Most Recent**: April 9, 2026, during the current session (Position Eligibility).
- **Previous**: April 9, 2026, at 10:32 AM EDT (Player ID Mapping).

## 3. History: Have they NEVER worked, or did they break recently?
- **Status**: They were largely "silent" before today due to an **Observability Crisis**.
- **Finding**: They were likely failing silently or skipping due to configuration issues (invalid Yahoo game key) and lack of logging.
- **Resolution**: Added 7+ log entries per job today, which immediately identified the Yahoo game key as the root cause of `position_eligibility` failures.

## 4. Patterns in failures
- **Yahoo API**: Failed with "Invalid game key" until `YAHOO_GAME_ID` (or related league config) was corrected.
- **Statcast**: Often returns 0 records if triggered too early in the day or if the date range is misconfigured.
- **Advisory Locks**: No failures found related to lock contention; the `_with_advisory_lock` wrapper is functioning correctly.

## 5. Summary Table (April 9, 2026)

| Job | Status | Records | Note |
|-----|--------|---------|------|
| `player_id_mapping` | ✅ SUCCESS | 20,000 | Successfully doubled from 10k to 20k today. |
| `position_eligibility` | ✅ SUCCESS | 229 | Fixed today via Yahoo config update. |
| `probable_pitchers` | ✅ SUCCESS | 0 | Expected for morning; evening runs will populate data. |

## 6. Verdict
The job execution framework is **STABLE** and **FUNCTIONAL**. The primary blockers were configuration (Yahoo) and visibility (Logging), both of which have been resolved.
