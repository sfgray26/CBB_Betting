# Decision Pipeline Diagnosis

**Date:** 2026-05-11  
**Scope:** Read-only investigation. No production code modified.

## Executive Verdict

The first empty stage is the final decision output layer: `decision_results` / `decision_explanations`, not the early BDL ingestion layer.

Best classification: **(d) filter/identity gating too aggressive in `decision_optimization`, with an API read-path bug masking the stored fallback.**

The upstream tables are not proven empty from available evidence:
- `reports/2026-05-04-comprehensive-due-diligence-audit.md` recorded production `player_scores: 77,517 rows`, `mlb_game_log: 490 rows`, and `mlb_player_stats: 13,809 rows`.
- `HANDOFF.md` currently records `mlb_player_stats` as live with `15,407 rows`.
- The code path that writes `decision_results` can still return `status: success` with 0 rows when:
  - there are no `player_scores` for exactly today's ET date, or
  - Yahoo roster resolution yields no mapped BDL IDs, so lineup optimization is intentionally skipped, or
  - waiver candidates cannot be resolved to modeled data and then filtered by `value_gain > 0.10`.

Live Railway DB/API counts could not be fetched from this sandbox: direct Postgres failed with `Permission denied` opening the Railway TCP connection, and HTTPS to the Railway app failed to connect. Treat this as a code-and-existing-evidence diagnosis, not a fresh production count certification.

## Pipeline Trace

Expected flow:

```text
mlb_game_log -> mlb_player_stats -> player_rolling_stats -> player_scores
-> player_momentum -> simulation_results -> decision_optimization
-> decision_results -> decision_explanations
```

Relevant writer functions in `backend/services/daily_ingestion.py`:

| Stage | Function | Empty behavior |
|---|---|---|
| `mlb_game_log` | `_ingest_mlb_game_log()` | BDL errors are logged; empty dates still commit success with 0 total games. |
| `mlb_box_stats` | `_ingest_mlb_box_stats()` | If no game IDs or no BDL stat rows, records `success, 0`. |
| `rolling_windows` | `_compute_rolling_windows()` | If no `mlb_player_stats` rows in the 30-day window, records `success, 0`. |
| `player_scores` | `_compute_player_scores()` | Empty rolling rows produce `success, 0` after warnings. |
| `decision_optimization` | `_run_decision_optimization()` | Empty same-day 14d scores produce `success, 0`; empty roster mapping also produces 0 lineup rows without failing. |

## Bottleneck Detail

`_run_decision_optimization()` queries only:

```sql
SELECT * FROM player_scores
WHERE as_of_date = today_et AND window_days = 14
```

If this exact-date query returns empty, it records `decision_optimization` as `success` with 0 rows. That is a false-green condition.

If score rows do exist, the next hard gate is roster resolution:

```python
roster_bdl_ids = set(yahoo_positions_by_bdl.keys())
if roster_bdl_ids:
    roster_score_rows = [s for s in score_rows if s.bdl_player_id in roster_bdl_ids]
else:
    roster_score_rows = []
```

The code explicitly does not fall back to all `player_scores`. That is correct from a product-safety standpoint, but operationally it means any Yahoo auth/mapping failure creates `0` lineup decisions while still allowing the job to finish green.

Waivers can also go to zero after multiple gates:
- Yahoo free agents must resolve to BDL IDs.
- Free agents must have both `player_scores` and `simulation_results`.
- Recommendations are filtered to `value_gain > 0.10`.
- If roster resolution produced an empty roster, `optimize_waivers()` cannot identify a drop target and returns no rows.

## API Read-Path Finding

There are two `/api/fantasy/decisions` implementations:

- `backend/routers/fantasy.py`
- `backend/main.py`

`backend/main.py` includes the fantasy router before defining its own route, so the router version is the effective handler in normal FastAPI route order.

The router handler attempts live category-aware waiver optimization before falling back to stored `decision_results`. That live branch constructs `PlayerDecisionInput` with unsupported/missing fields such as `position=...` and omits required fields like `name`, `player_type`, `eligible_positions`, and `score_0_100`. This raises per-player build errors, produces no live waiver output, and then falls back to the DB query. If `decision_results` is empty, the endpoint returns empty.

This API bug does not explain why the scheduled table is empty, but it does explain why the endpoint cannot compensate with live waiver decisions.

## Root Cause Classification

Most likely cause: **filter/identity gating too aggressive at `decision_optimization`**, not raw data ingestion failure.

Supporting evidence:
- Existing production audits show `player_scores` and `mlb_player_stats` populated before this task.
- `decision_optimization` has several legitimate 0-output paths that record `success`.
- The endpoint fallback depends entirely on persisted `decision_results` after its live branch fails.

What remains unverified due sandbox network limits:
- Current May 11 live row counts by date.
- Latest `job_runs` records for `decision_optimization`.
- Whether today's exact `player_scores` date is missing versus roster BDL mapping being empty.

## Verification SQL To Run In Railway

```sql
SELECT 'mlb_game_log' AS table_name, COUNT(*) AS n, MIN(game_date), MAX(game_date) FROM mlb_game_log
UNION ALL SELECT 'mlb_player_stats', COUNT(*), MIN(game_date), MAX(game_date) FROM mlb_player_stats
UNION ALL SELECT 'player_rolling_stats', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM player_rolling_stats
UNION ALL SELECT 'player_scores', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM player_scores
UNION ALL SELECT 'player_momentum', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM player_momentum
UNION ALL SELECT 'simulation_results', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM simulation_results
UNION ALL SELECT 'decision_results', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM decision_results
UNION ALL SELECT 'decision_explanations', COUNT(*), MIN(as_of_date), MAX(as_of_date) FROM decision_explanations;
```

```sql
SELECT job_name, status, records_processed, error_message, started_at, completed_at
FROM job_runs
WHERE job_name IN (
  'mlb_game_log', 'mlb_box_stats', 'rolling_windows', 'player_scores',
  'player_momentum', 'ros_simulation', 'decision_optimization', 'explainability'
)
ORDER BY started_at DESC
LIMIT 40;
```

```sql
SELECT COUNT(*) AS today_14d_scores
FROM player_scores
WHERE as_of_date = CURRENT_DATE AND window_days = 14;
```

```sql
SELECT COUNT(*) AS mapped_roster_candidates
FROM player_id_mapping
WHERE yahoo_key IS NOT NULL AND bdl_id IS NOT NULL;
```

## Clear Fix Candidates To Document, Not Apply

One-line observability fix:

```python
self._record_job_run("decision_optimization", "failed", 0)
```

Use that instead of `success, 0` in the `if not score_rows:` branch, or introduce a `no_input` status if the scheduler supports it. This prevents the false-green scheduler symptom.

Small product fix:

Fix the router live-waiver `PlayerDecisionInput(...)` construction in `backend/routers/fantasy.py` to match `backend/services/decision_engine.py`, or remove the live branch and rely only on persisted decisions. The current construction cannot produce valid live results.

Operational next step:

Run `decision_optimization` manually after confirming same-day `player_scores` exist. If it returns 0 with `today_14d_scores > 0`, inspect the log line:

```text
decision_optimization: roster BDL IDs empty
```

That confirms the bottleneck is Yahoo roster-to-BDL identity resolution, not upstream stat ingestion.
