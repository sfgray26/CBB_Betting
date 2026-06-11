# 2026-05-20 Codex Injury Pipeline Report

## Scope

Investigate the stale MLB injury signal reported in production:

- User symptom: Geraldo Perdomo displayed as `DTD` with `ETA Apr 28 · stale · updated 21d ago`
- Goal: determine whether the injury pipeline is frozen, whether the stale label is a UI/data-selection bug, and what operational gaps exist

## Evidence Gathered

### 1. Injury jobs present in code

`backend/services/daily_ingestion.py` contains two relevant jobs:

- `bdl_injuries`
  - registered hourly
  - function: `_ingest_bdl_injuries()`
  - lock: `100_033`
- `yahoo_adp_injury`
  - registered every 4 hours
  - function: `_poll_yahoo_adp_injury()`
  - lock: `100_013`

Relevant code observations:

- `bdl_injuries` upserts into `ingested_injuries`
- `yahoo_adp_injury` writes injury/ownership fields into `player_daily_metrics.rolling_window`
- `yahoo_adp_injury` sets `adp_updated_at` in JSON payload, but there is no visible health-monitor coverage for this job

### 2. Production logs: `bdl_injuries` is healthy

Confirmed from Railway logs:

- `2026-05-20 09:01:36 -04:00` job start
- `2026-05-20 09:01:40 -04:00` `bdl_injuries: 187 injuries upserted in 4115ms`
- job completed successfully

This strongly suggests the BDL injury feed is not globally frozen.

### 3. Production logs: no recent `yahoo_adp_injury` evidence surfaced

I searched Railway logs over larger windows and found explicit `bdl_injuries` events, but did not surface corresponding `yahoo_adp_injury` events.

This does not prove the job never runs, but it is a meaningful warning sign:

- either `yahoo_adp_injury` is not running
- or its log volume is absent / filtered away / emitted through a different path
- or the production image is not executing that scheduled task as expected

### 4. Live health surface has a monitoring gap

Live endpoint:

- `https://fantasy-app-production-5079.up.railway.app/health/pipeline`

Observed payload:

- includes `bdl_injuries`
- does **not** include `yahoo_adp_injury`
- currently marks only `savant_ingestion` as stale

Implication:

- the production pipeline health endpoint cannot detect a stale or dead `yahoo_adp_injury` job today
- a 3+ week freeze in Yahoo injury freshness would not be surfaced by current health checks

### 5. Live `/health` route is not the route inspected in `backend/main.py`

Code inspection found two `/health` implementations:

- `backend/main.py` version adds `pipeline_summary`
- `backend/routers/admin.py` version returns only basic status fields

Live response matched the simpler admin-router version:

```json
{"status":"healthy","database":"connected","scheduler":"running"}
```

Implication:

- the deployed `/health` endpoint is currently too shallow for this incident
- even if pipeline sub-jobs are stale, `/health` can still report healthy

### 6. Overlay logic matches the exact stale symptom

`backend/services/injury_overlay.py`:

- freshness threshold is `180` minutes
- stale label format is `stale · updated {age}`
- overlay chooses the most recent `ingested_injuries` row per `bdl_player_id`

This means:

- a `21d ago` banner is real age from the selected injury row
- the stale banner itself is behaving as coded
- the real question is why a 21-day-old injury row was still considered the active overlay for Perdomo

### 7. Documented cleanup path does not exist for `ingested_injuries`

There is a critical mismatch between comments and implementation.

Comments claim:

- recovered injuries are removed by a separate cleanup job

Actual code:

- `_cleanup_old_metrics()` only deletes old `player_daily_metrics`
- I found no delete path for resolved/stale `ingested_injuries`

Implication:

- old BDL injury rows can persist indefinitely unless overwritten
- if BDL stops returning a player, the stale record can still be selected by overlay logic
- this is the strongest code-level explanation for Perdomo showing `updated 21d ago`

### 8. No provider webhook ingestion path exists

I searched for injury webhook receivers and found none for Yahoo or BDL injury ingestion.

Current design is poll-based:

- BDL injuries: hourly pull
- Yahoo ADP/injury: 4-hour pull

Implication:

- there is no external injury webhook to debug
- action item 3 from the request resolves to: no provider webhook exists in the current architecture

## Root Cause Assessment

### Most likely root cause of the Perdomo stale banner

The stale banner is most likely caused by **orphaned rows in `ingested_injuries`**:

1. `bdl_injuries` ingests active injuries successfully
2. overlay code reads latest row per player from `ingested_injuries`
3. comments assume resolved injuries are deleted by cleanup
4. actual cleanup implementation never deletes from `ingested_injuries`
5. old injury rows can remain and later be rendered as `stale · updated 21d ago`

This explains the exact observed symptom better than a total BDL outage, because BDL ingestion is demonstrably healthy today.

### Secondary operational gap

`yahoo_adp_injury` is under-monitored:

- not present in live `/health/pipeline`
- no explicit recent log evidence was surfaced

That is a separate reliability issue. Even if it is not the direct cause of Perdomo's stale overlay, it should be treated as a pipeline observability bug.

## What I Could Not Fully Prove

I attempted direct production DB verification for:

- latest `ingested_injuries.ingested_at`
- Perdomo rows in `ingested_injuries`
- Perdomo rows in `player_daily_metrics`

Two blockers prevented final confirmation:

- non-interactive `railway ssh` command execution was unreliable in this shell
- direct query through the public Railway Postgres hostname timed out from this environment

So I do **not** have a direct SQL row dump for Perdomo in this report.

## Required Fixes

These are backend changes and should be implemented by Claude Code under repo constraints.

### P1. Add `yahoo_adp_injury` to health monitoring

Update `backend/services/health_monitor.py`:

- include `yahoo_adp_injury` in monitored jobs
- treat it as a 4-hour job, not hourly
- make stale/failure visible in `/health/pipeline`

### P1. Implement real cleanup for resolved `ingested_injuries`

Add logic that removes or archives injuries no longer returned by the active BDL feed.

Minimum acceptable behavior:

- after each successful BDL sync, delete `ingested_injuries` rows for player/status/type combinations not present in the current feed

Without this, stale injury overlays will continue to accumulate.

### P1. Add explicit injury staleness alarm

Recommended locations:

- `backend/services/daily_ingestion.py`
- or `backend/services/health_monitor.py`

Required behavior:

- if latest BDL injury data is older than 7 days, log `WARNING`
- if latest Yahoo injury freshness is older than 7 days, log `WARNING`
- trigger existing Discord alert path if enabled

### P2. Strengthen `/health`

Current live `/health` is too shallow.

At minimum:

- ensure deployed `/health` includes pipeline summary
- or make `/health` explicitly delegate to the richer health implementation

## Recommended Verification Plan After Backend Fix

1. Trigger or wait for `bdl_injuries`
2. Trigger or wait for `yahoo_adp_injury`
3. Query:
   - latest `ingested_injuries.ingested_at`
   - latest `player_daily_metrics.rolling_window.adp_updated_at`
   - Perdomo-specific rows in both stores
4. Confirm `/health/pipeline` now includes `yahoo_adp_injury`
5. Confirm Perdomo no longer shows `stale · updated 21d ago` unless truly unresolved in source data

## Bottom Line

The injury pipeline is not fully frozen.

- `bdl_injuries` is healthy in production today
- the stale Perdomo banner is most likely caused by stale `ingested_injuries` rows that are never cleaned up
- `yahoo_adp_injury` currently has an observability gap and may also be unhealthy, but the live health endpoint would not tell us

Most urgent backend action:

- implement real `ingested_injuries` cleanup
- add `yahoo_adp_injury` to monitored health/staleness checks
