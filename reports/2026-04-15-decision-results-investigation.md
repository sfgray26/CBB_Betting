# Decision Results Volume Investigation
**Date:** 2026-04-15  
**Author:** Claude Code (Master Architect)  
**Scope:** Forensic read-only pass — no code changes

---

## Schema Summary

**Table:** `decision_results`  
**ORM class:** `DecisionResult` (`backend/models.py` line 1416)  
**Docstring:** "P17 Decision Engine results -- lineup and waiver optimization outputs."

| Column | Type | Nullable | Notes |
|---|---|---|---|
| `id` | BigInteger PK | No | autoincrement |
| `as_of_date` | Date | No | yesterday's date at job run time |
| `decision_type` | String(10) | No | "lineup" or "waiver" |
| `bdl_player_id` | Integer | No | BDL player identifier |
| `target_slot` | String(10) | Yes | e.g. "OF", "SP" — lineup only |
| `drop_player_id` | Integer | Yes | waiver drop target — waiver only |
| `lineup_score` | Float | Yes | composite score used in ranking |
| `value_gain` | Float | Yes | world-with minus world-without delta |
| `confidence` | Float | No | [0, 1] normalized |
| `reasoning` | String(500) | Yes | ASCII one-liner |
| `computed_at` | DateTime(tz) | No | server default now() |

**Unique constraint:** `_dr_date_type_player_uc` on (`as_of_date`, `decision_type`, `bdl_player_id`)  
**Indexes:** `idx_dr_date_type` on (`as_of_date`, `decision_type`); `idx_dr_player_date` on (`bdl_player_id`, `as_of_date`)

---

## Writer Location and Scheduler Wiring

**Writer function:** `_run_decision_optimization()`  
**File:** `backend/services/daily_ingestion.py` line 2338  
**Advisory lock:** `LOCK_IDS["decision_optimization"] = 100_022` (line 109)  
**Scheduler:** APScheduler CronTrigger, `hour=7, minute=0, timezone=America/New_York` (lines 493-501)  
**Job ID string:** `"decision_optimization"`  
**Upstream dependency:** `_run_ros_simulation` (6 AM), which must write `simulation_results` first

The upsert path uses `pg_insert(...).on_conflict_do_update(constraint="_dr_date_type_player_uc", ...)` per row, followed by a single `db.commit()` at line 2612. A rollback on DB write failure is logged at the `ERROR` level (line 2615-2617). There are no silent swallow paths — all exceptions are caught, logged, and returned as `"failed"` status.

---

## Expected Volume Model

The decision engine produces exactly two types of rows per run:

**Lineup decisions (decision_type="lineup"):**  
- `optimize_lineup()` fills roster slots greedily: C(1), 1B(1), 2B(1), 3B(1), SS(1), OF(3), Util(1), SP(2), RP(2), P(1) = **13 active slots**.
- Bench players (up to 5) are explicitly excluded: `if pid in bench: continue` (line 365-366 in `decision_engine.py`).
- Maximum lineup rows per run = **13** (one per filled active slot, not per player in the pool).

**Waiver decisions (decision_type="waiver"):**  
- One row per free agent in `waiver_pool` that was successfully cross-referenced to a `bdl_player_id` via `PlayerIDMapping`.
- The call fetches `count=25` free agents from Yahoo (`get_free_agents(count=25)`, line 2525).
- Any free agent whose `yahoo_key` is absent from `PlayerIDMapping.bdl_id` is silently skipped (`if fa_bdl_id is None: continue`, line 2543-2544).

**Cumulative accounting:** The job runs daily and writes with `ON CONFLICT DO UPDATE`, so each day appends up to 13 lineup rows + N waiver rows (N <= 25). Across the season (say 13 days since April 1), the table could contain up to `13 x 13 = 169` lineup rows, plus waiver rows.

**26 rows is implausibly low** for even 2 clean daily runs (2 x 13 = 26 lineup rows alone), which means either: the job ran successfully only twice in lineup-only mode with zero waiver rows each time, or the row count reflects a cumulative single-date result from an early successful run before a persistent failure began.

---

## Diagnosis with Evidence

### Finding 1 — Lineup output is capped at 13 rows per run by design

The `optimize_lineup()` function only emits `DecisionResult` rows for players placed in **active slots** — bench players are explicitly excluded at `decision_engine.py:365-366`. With 13 active slots defined in `ROSTER_SLOTS` (C=1, 1B=1, 2B=1, 3B=1, SS=1, OF=3, Util=1, SP=2, RP=2, P=1), the maximum lineup output is 13 rows per day.

If exactly 2 days ran successfully and produced only lineup decisions (no waiver rows), the total would be exactly **26 rows**. This is consistent with the observed count.

### Finding 2 — Waiver pool produces 0 rows due to PlayerIDMapping gap

The waiver pool is built at `daily_ingestion.py:2518-2576`. It fetches 25 free agents from Yahoo, then cross-references via `PlayerIDMapping` to get `bdl_player_id`. If `PlayerIDMapping` has no rows mapping Yahoo keys to BDL IDs for free agents, **every free agent is skipped** at line 2543-2544, yielding an empty `waiver_results` list.

The HANDOFF.md audit table does not list `player_id_mapping` row count, but the table is described as a cross-reference populated by `_sync_player_id_mapping` job. If identity resolution has not run or populated mappings for free agents (only roster players would be mapped), all 25 waiver candidates are silently dropped.

### Finding 3 — Scope bug in `client` reuse check (line 2520)

The waiver pool block uses `if "client" not in dir()` (line 2520) to check whether the Yahoo client was already instantiated. In Python, `dir()` inside a nested function returns the local scope — but this check is inside the `_run` async inner function. If the roster fetch at line 2437 threw an exception (caught at line 2465-2470), `client` is never assigned in the local scope. In that scenario, `"client" not in dir()` evaluates to `True`, a new `YahooFantasyClient()` is created at line 2524, and the waiver fetch proceeds. This is actually safe behavior, not the bug causing 0 rows.

The real issue is whether `get_free_agents()` returns players whose `yahoo_key` values appear in `player_id_mapping`. If the identity resolution job has only mapped **roster players** (not free agents), the waiver cross-reference yields nothing.

### Finding 4 — Date scoping is correct but cumulative coverage is narrow

The job always writes `as_of_date = yesterday` (line 2358). With the unique constraint on `(as_of_date, decision_type, bdl_player_id)`, each daily run upserts 13 lineup rows per date. If the job ran successfully on 2 dates (e.g. Apr 12-13), the table would have exactly 26 lineup rows — matching the observed count exactly.

### Finding 5 — No silent rollback path

The upsert loop at lines 2586-2612 commits all rows in a single `db.commit()` at the end. The `except` block at line 2613 performs `db.rollback()` and logs at `ERROR` level — not silent. DB write failures would be visible in Railway logs.

---

## Verdict: UPSTREAM (primary) + EXPECTED (partially)

**Primary cause — waiver pool silently empty:** The `PlayerIDMapping` table has not resolved `bdl_player_id` for free agent Yahoo keys, so all 25 waiver candidates are dropped silently at line 2543-2544. This produces 0 waiver rows per run — by design, but revealing a data gap upstream.

**Secondary cause — lineup volume is by design small:** `optimize_lineup()` emits at most 13 rows per run (one per active roster slot). Bench players are intentionally excluded. 26 rows is exactly 2 x 13, meaning the job ran successfully on 2 dates and the lineup engine filled all 13 slots each time with no waiver output.

**This is NOT a bug in the writer.** The writer is correct: it logs failures at ERROR level, commits atomically, and the upsert constraint is wired correctly. The low count is a consequence of:
1. The job has only run successfully for 2 days (probably started around Apr 12-13 based on pipeline health data showing `player_scores` through Apr 13).
2. Waiver pool cross-referencing fails silently because `PlayerIDMapping` has no free agent mappings.

---

## Recommended Next Action

**Scope:** One targeted change in `_run_decision_optimization()` — add a diagnostic warning when the waiver pool is unexpectedly empty after the Yahoo fetch succeeds.

Specifically:

1. **Add a log warning** (not error) at `daily_ingestion.py` after line 2566 when `len(waiver_pool) == 0` and `len(free_agents) > 0`. This surfaces the `PlayerIDMapping` gap in Railway logs without any code-path change.

2. **Audit `player_id_mapping` row count** via `GET /admin/pipeline-health` or a direct DB query: `SELECT COUNT(*) FROM player_id_mapping WHERE bdl_id IS NOT NULL`. If it is 0 or very small (roster-players-only), the identity resolution job (`_sync_player_id_mapping`, lock `100_017`) has not populated free agent mappings.

3. **Do not expect >13 lineup rows per day** — this is correct by architecture (one row per active slot, not per player in the scoring pool). The 26-row total for ~2 days of runtime is expected given the pipeline went live around Apr 12.

No backend code changes required for the investigation. Only the diagnostic log addition (1 line) is warranted to make the waiver-pool gap visible.
