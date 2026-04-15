# HANDOFF.md — MLB Platform Operational State

> **Date:** April 15, 2026 | **Author:** Claude Code (Master Architect)
> **Status:** Infrastructure stable. Core fantasy pipeline is live. P0 observability and degraded-health semantics are now implemented in code. Decision-quality pipeline remains incomplete.
>
> **Current Focus:** Gemini must validate the new ingestion audit trail and degraded-health behavior in production before the next implementation task begins.
>
> **Full audit:** `reports/2026-04-15-comprehensive-application-audit.md`
> **Decision forensic:** `reports/2026-04-15-decision-results-investigation.md`
> **Historical context:** `HANDOFF_ARCHIVE.md`

---

## EXECUTIVE STATE

### Resolved / Verified
- **Statcast aggregation fix is live.** Zero-metric rate improved from **42.4% to 5.0%** post-aggregation fix.
- **NSB pipeline is deployed.** `z_nsb` is in scoring and explainability; `z_sb` is excluded from composite to avoid double-counting.
- **Decision-results investigation is resolved.** Low row count was not a silent crash; lineup decisions emit one row per active slot, and the larger issue was an empty waiver pool caused by free-agent identity misses.
- **FA identity fallback is deployed in code.** `daily_ingestion.py` now performs best-effort normalized-name resolution for free agents when `yahoo_key` is unavailable.
- **Scheduler is stable.** 10 core jobs are active in production.
- **Tests are green in Railway.** HANDOFF no longer treats the v27 NSB work as uncommitted.
- **P0 observability is implemented in code.** The advisory-lock boundary now writes durable `DataIngestionLog` rows for success, failure, and skipped runs.
- **P0 health semantics are implemented in code.** Empty `probable_pitchers` and empty `data_ingestion_logs` are now treated as degraded conditions by the health/validation layer rather than harmless info.

### Live Risks
1. **Live observability still needs production verification.** The logging path is implemented, but the next scheduled or manual ingestion run must confirm that `data_ingestion_logs` is actually populating in production.
2. **Environment factors are implemented but orphaned.** Weather and park modules exist, and `SmartLineupSelector` uses them at request time, but there is still no canonical environment snapshot in the DB or scoring pipeline.
3. **Probable pitchers remain empty.** This is currently an upstream/source-resilience problem, but it materially weakens two-start SP detection, matchup quality, and streaming guidance.
4. **Decision breadth remains too narrow.** The waiver path now resolves some FAs, but the candidate universe is still shallow and must be monitored after the next run.

### Blocked by Upstream / External Constraints
- **Probable pitchers:** MLB Stats API has returned 0 rows during recent sync windows.
- **Caught stealing richness:** NSB is live, but the richer CS signal still depends on source completeness outside the fantasy scorer itself.
- **OpenWeatherMap live forecasts:** meaningful live-weather integration still depends on a valid `OPENWEATHER_API_KEY` in production.

### Operator Guidance
- Treat `overall_healthy: true` as **infrastructure healthy**, not **fantasy decision-quality complete**.
- Do not reopen the `decision_results = 26` investigation unless decision volume remains suppressed after the FA fallback has had a full production run.
- Keep the focus on the fantasy pipeline. UI/design work is not the critical path.

---

## PRODUCTION SNAPSHOT (APRIL 15)

`GET /admin/pipeline-health` → **`overall_healthy: true`**

| Table | Rows | Latest Date | Status | Notes |
|-------|------|-------------|--------|-------|
| `player_rolling_stats` | **30,667** | 2026-04-13 | ✅ Healthy | Fresh enough for current scoring pipeline |
| `player_scores` | **30,580** | 2026-04-13 | ✅ Healthy | Scoring output is populating |
| `statcast_performances` | **6,971** | 2026-04-13 | ✅ Healthy | Post-fix quality materially improved |
| `simulation_results` | **10,236** | 2026-04-13 | ✅ Healthy | Simulation layer is live |
| `mlb_player_stats` | **6,801** | 2026-04-13 | ✅ Healthy | Growing via BDL |
| `probable_pitchers` | **0** | — | ⚠️ Degraded | Upstream/source resilience gap |
| `data_ingestion_logs` | **0** | — | ⚠️ Degraded | Logging is now implemented in code; next run must populate rows |

---

## RESOLVED TODAY

### 1. NSB Pipeline (P27)
- `scripts/migrate_v27_nsb.py` adds `w_caught_stealing`, `w_net_stolen_bases`, and `z_nsb`.
- `rolling_window_engine.py` computes NSB as `SB - CS` from BDL-sourced stats.
- `scoring_engine.py` uses `z_nsb` in `composite_z` and excludes `z_sb` from composite.
- `daily_ingestion.py` write paths were extended for the new columns.
- `tests/test_nsb_pipeline.py` covers the rollout.

### 2. NSB Explainability Follow-Up
- Added diagnostic endpoints for rollout/fill-rate inspection.
- `explainability_layer.py` now narrates `z_nsb` before `z_sb`.

### 3. Decision Results Investigation
**Verdict:** the low row count was a mixed signal, not a crash.

- **By design:** lineup decisions emit exactly one row per active slot.
- **Actual gap:** the waiver pool was empty because free agents lacked usable `yahoo_key` values in `player_id_mapping`.
- **Code fix:** free agents unresolved by `yahoo_key` now attempt normalized-name resolution against `player_id_mapping.normalized_name`.
- **Operational follow-up:** monitor `waiver_rows` after the next production run. If still zero, extend the name normalizer for nickname/diacritic variants.

### 4. Player ID Mapping Deduplication
**Verdict:** root cause identified, code fixed, production migration reported complete by Gemini.

- Root cause was repeated `db.merge()` inserts without a unique `bdl_id` constraint.
- Code now performs explicit select-then-upsert by `bdl_id`.
- Model and migration updates enforce `_pim_bdl_id_uc` and support `updated_at`.
- Gemini reported the production dedupe as complete and the constraint as active.

**Important note:** the exact post-migration row count needs one canonical number in future updates. Earlier HANDOFF values mixed `~2,000 expected`, `60,000 pre-fix`, and `10,000 post-migration distinct IDs`. Do not reintroduce those as simultaneous “current” states.

---

## PIPELINE WEAKNESSES THAT STILL MATTER

### 1. Observability Rollout Verification
The missing audit-trail implementation is now addressed in code. The immediate task is proving it works in the live environment.

- `DataIngestionLog` writes are now attached to the advisory-lock wrapper, which is the right control point for job-level auditing.
- Success, failure, and skipped runs now produce durable log rows in code.
- Pipeline health and validation logic now treat missing logs as degraded rather than informational.

**Required outcome:** verify production row creation and confirm the latest job run is visible without tailing Railway logs.

### 2. Environment-Factor Integration Gap
This is the largest fantasy-feature gap still blocking a championship-grade pipeline.

- `weather_fetcher.py`, `park_weather.py`, and `ballpark_factors.py` are implemented.
- `SmartLineupSelector` uses weather and park analysis at request time.
- The canonical scoring pipeline and DB do not persist an environment snapshot or join one into player scores.

**Required outcome:** create a DB-backed environment layer so weather/park factors are not trapped inside request-time ranking logic.

### 3. Probable Pitcher Source Resilience
- Current state is still `0` rows.
- This is not just cosmetic; it weakens matchup quality, two-start pitcher detection, and streamer recommendations.

**Required outcome:** implement fallback logic rather than waiting on a single source to behave perfectly.

### 4. Decision-Universe Breadth
- FA fuzzy matching is an important fix, but not the endpoint.
- Current waiver scanning remains too narrow to fully exploit reliever churn, short-horizon streamers, and category specialists.

**Required outcome:** broaden the candidate universe and add decision attribution so the engine can explain why it did or did not produce waiver actions.

---

## MLB DATA PROVIDER STRATEGY

### Policy
- **BDL is the preferred primary source** for MLB schedule, stats, injuries, and odds where it is already integrated cleanly.
- **Odds API is still allowed** when it provides tangible value: better bookmaker coverage, better market quality for a specific use case, or resilience as a fallback source.
- **Dual-source use is acceptable** if it is intentional, bounded, and documented in the relevant service.

### Current Reality
- Core MLB stats growth is already BDL-backed.
- `daily_lineup_optimizer.py` still depends on Odds API for lineup-context odds.
- This is no longer treated as a policy violation by itself; it is now a rationalization task.

### Operating Rule Going Forward
- Do not keep accidental split-provider behavior.
- If a service uses Odds API, document why that path remains superior or necessary.
- If BDL can replace that path without loss of signal or reliability, migrate it.

---

## RAILWAY MCP SERVER & DEVOPS TOOLING

**Goal:** give Gemini faster, safer Railway and DB access without code writes.

### Set Up
1. Railway MCP server added to Kimi CLI config.
2. DevOps helper scripts added in `scripts/devops/`:
   - `db_query.py`
   - `db_health.py`
   - `railway_logs_filter.py`
3. `GEMINI.md` updated with approved commands.

### Remaining Infra Follow-Up
- Enable PostgreSQL MCP once the production `DATABASE_URL` is verified.
- Reuse the same MCP config for other agents if MCP client support becomes available.

---

## NEXT STEPS (PRIORITY ORDER)

| Priority | Action | Owner | ETA | Why It Matters |
|----------|--------|-------|-----|----------------|
| P0 | Verify `data_ingestion_logs` populates on the next scheduled or manual job run | Claude + Gemini | Next run | Confirms the new audit trail is actually live in production |
| P0 | Verify `/admin/pipeline-health` and `/admin/validation-audit` now surface empty logs / probable pitchers as degraded | Claude + Gemini | Next run | Confirms false-green health reporting is gone |
| P1 | Implement probable-pitcher fallback/resilience path | Claude | 2-3 days | Unblocks two-start SP and matchup-quality features |
| P1 | Monitor post-fix waiver volume and FA fuzzy-match yield | Claude | Next production run | Confirms whether the decision engine is still underfed |
| P1 | Design and land DB-backed environment snapshots | Claude | 1 week | Moves weather/park from optional request-time logic into the canonical pipeline |
| P2 | Rationalize MLB provider usage service-by-service | Claude | 2-3 days | Prevents accidental source sprawl while preserving useful redundancy |
| P2 | Broaden waiver candidate universe and decision attribution | Claude | 3-4 days | Improves actual fantasy edge, not just pipeline completeness |

### Not The Critical Path
- UI/UX design work
- Broader non-fantasy infrastructure cleanup
- Reopening already-resolved `decision_results` forensics before the next monitored run

---

## GEMINI VALIDATION TASKS (BLOCKING BEFORE P1)

**Owner:** Gemini CLI  
**Scope:** Railway ops + read-only verification only. No code edits.  
**Goal:** prove that P0 observability and degraded-health semantics are live in production before Claude starts the next feature task.

### Validation Sequence

1. **Check whether `data_ingestion_logs` has started populating**

```bash
railway ssh python scripts/devops/db_query.py "SELECT COUNT(*) AS row_count, MAX(started_at) AS latest_started_at, MAX(completed_at) AS latest_completed_at FROM data_ingestion_logs;"
```

2. **Inspect the newest ingestion log rows**

```bash
railway ssh python scripts/devops/db_query.py "SELECT job_type, status, target_date, started_at, completed_at, records_processed, records_failed, error_message FROM data_ingestion_logs ORDER BY started_at DESC LIMIT 15;"
```

3. **Verify whether the `probable_pitchers` table is still empty**

```bash
railway ssh python scripts/devops/db_query.py "SELECT COUNT(*) AS row_count, MAX(game_date) AS latest_game_date FROM probable_pitchers;"
```

4. **Check pipeline-health endpoint from production**

Preferred path if `API_URL` and `API_KEY`/admin key env vars are available in Railway:

```bash
railway ssh python -c "import json, os, requests; base=os.getenv('API_URL') or os.getenv('NEXT_PUBLIC_API_URL') or 'https://cbb-edge-production.up.railway.app'; key=os.getenv('API_KEY') or os.getenv('ADMIN_API_KEY') or os.getenv('X_API_KEY'); headers={'X-API-Key': key} if key else {}; r=requests.get(f'{base}/admin/pipeline-health', headers=headers, timeout=30); print(r.status_code); print(json.dumps(r.json(), indent=2))"
```

5. **Check validation-audit endpoint from production**

```bash
railway ssh python -c "import json, os, requests; base=os.getenv('API_URL') or os.getenv('NEXT_PUBLIC_API_URL') or 'https://cbb-edge-production.up.railway.app'; key=os.getenv('API_KEY') or os.getenv('ADMIN_API_KEY') or os.getenv('X_API_KEY'); headers={'X-API-Key': key} if key else {}; r=requests.get(f'{base}/admin/validation-audit', headers=headers, timeout=60); print(r.status_code); data=r.json(); print(json.dumps({'critical': data.get('critical', []), 'high': data.get('high', []), 'medium': data.get('medium', []), 'low': data.get('low', []), 'info': data.get('info', [])}, indent=2))"
```

6. **Tail Railway logs for evidence of the new audit trail**

Use one or more of these after a scheduled/manual run:

```bash
railway run python scripts/devops/railway_logs_filter.py --job player_scores --lines 50
railway run python scripts/devops/railway_logs_filter.py --job decision_optimization --lines 50
railway run python scripts/devops/railway_logs_filter.py --job probable_pitchers --lines 50
```

### Success Criteria

- `data_ingestion_logs.row_count > 0`
- Recent rows show real job types with `SUCCESS`, `FAILED`, or `SKIPPED` status and non-null `started_at`
- `/admin/pipeline-health` no longer gives a misleading all-green picture if `data_ingestion_logs` or `probable_pitchers` remain empty
- `/admin/validation-audit` no longer classifies empty `data_ingestion_logs` as informational-only
- Gemini can point to at least one concrete production row proving the new audit trail works

### If Validation Fails

- If `data_ingestion_logs` is still empty after a confirmed ingestion job run: stop and report back before any P1 work begins
- If endpoint auth fails: report which env var/key was missing and provide the DB-query results instead
- If the DB rows exist but the endpoints still report incorrectly: capture the endpoint payload and escalate back to Claude

### Report Back Format

Gemini should update this HANDOFF section with:
- command(s) run
- row counts observed
- one sample `data_ingestion_logs` row
- `probable_pitchers` count
- whether `/admin/pipeline-health` and `/admin/validation-audit` now reflect degraded state correctly
- final verdict: **PASS** or **BLOCKED**

---

## GEMINI DEVOPS REPORT (APRIL 15)

### Mandatory Operations
| Task | Status | Date |
|------|--------|------|
| Disable integrity sweep | COMPLETE | April 15, 2026 |
| Enable MLB analysis model | COMPLETE | April 15, 2026 |
| Enable data ingestion orchestrator | COMPLETE | April 15, 2026 |
| Verify settings | COMPLETE | April 15, 2026 |

### Migration / Audit Notes
- Production `player_id_mapping` dedupe was reported complete.
- `_pim_bdl_id_uc` constraint was reported active.
- Statcast aggregation quality improved materially.
- Probable pitchers remained empty due to source lag.
- P0 code changes now add durable ingestion audit logging and degraded-health semantics; live production verification is still required.

**Use this section as a brief ops receipt, not as the canonical source of fantasy pipeline priorities.** The priority queue above is authoritative.

---

## ARCHITECT REVIEW QUEUE

- `/admin/diagnose-era` still needs the production `::numeric` cast fix.
- Broader timezone cleanup remains a repo-wide standard issue; do not assume all historical `datetime.utcnow()` usage is already eliminated.
- Risk-metric/null simulation path remains a known repo memory item and should be reviewed after the current pipeline blockers are stabilized.

---

**Last Updated:** April 15, 2026
