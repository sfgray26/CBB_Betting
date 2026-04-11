# Production Deployment Fix Plan (P-1 through P-4)

> **For operators:** Execute tasks in the order listed. Each task has a pre-flight check, execution command, validation query, and rollback. Do not proceed to the next task if a validation query fails — stop and analyze.

**Date:** 2026-04-11
**Author:** Claude Code (Master Architect)
**Goal:** Close the production data-quality gaps left after the Apr 10 remediation landed partially.

**Architecture:** Four independent fixes hitting the Railway production database (`Fantasy-App` service in `just-kindness`). Three run via existing admin endpoints; P-2 runs as a one-line SQL statement via Railway CLI (no code change, no deploy).

**Tech Stack:** Railway CLI, FastAPI admin endpoints, PostgreSQL (Railway managed), `pybaseball` Statcast retry logic (commit `4e11ab0`).

---

## Reality Checks (READ BEFORE EXECUTING)

These three findings reshape the original task brief. Confirmed against actual code and Apr 11 reports:

1. **P-1 target is NOT "0 NULL ops."** The Apr 10 backfill already populated 5,175 rows. Remaining 1,639 NULLs are almost certainly rows where `obp` or `slg` is itself NULL (pitcher stat-lines, 0-PA appearances). These are mathematically unbackfillable from existing columns. **Revised target: 0 NULL ops _where obp IS NOT NULL AND slg IS NOT NULL_.** The raw count will stay non-zero and that is correct.

2. **P-3 is NOT a re-run problem.** `reports/task-21-orphan-linking-results.md` shows the Apr 11 fuzzy-link run processed 366/366 candidates with 0 matches in ~7 minutes. The `2026-04-11-orphaned-players-audit.md` confirms the orphans are retired players, international prospects, and two-way players (Ohtani, Lorenzen). Re-running the fuzzy linker will produce 0% again. **Revised target: manually override Ohtani + Lorenzen (+2 linked), document the rest as known-unmatchable.**

3. **P-2 has no dedicated fix endpoint.** `/admin/diagnose-era` only reports. We execute via Railway CLI (`railway run python -c ...`) — no code, no deploy, no endpoint surface area. Fully reversible in one command.

---

## Pre-Deployment Checklist

Run all three checks. **If any check fails, stop and resolve before proceeding.**

- [ ] **Check 1: Railway CLI authenticated to correct project**

```bash
railway status
```
Expected output contains: `Project: just-kindness` and `Service: Fantasy-App`.
If wrong project: `railway link`.

- [ ] **Check 2: Production healthcheck passing**

```bash
curl -s https://fantasy-app-production.up.railway.app/health | python -m json.tool
```
Expected: `"status": "healthy"`, `"database": "connected"`, `"scheduler": "running"`.
If unhealthy: stop. Check Railway logs before proceeding.

- [ ] **Check 3: Admin API key available in local env**

```bash
echo $ADMIN_API_KEY | wc -c
```
Expected: >1 (key is set). If empty, retrieve from Railway: `railway variables | grep ADMIN_API_KEY`.
Export locally:
```bash
export ADMIN_API_KEY="<value>"
export BASE_URL="https://fantasy-app-production.up.railway.app"
```

- [ ] **Check 4: Capture baseline metrics (run validation audit now, save to file)**

```bash
curl -s "$BASE_URL/admin/validation-audit" \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  | python -m json.tool \
  > reports/2026-04-11-baseline-validation-audit.json
```
This is the "before" snapshot. We will diff it against the final run.

---

## Execution Order

| Order | Task | Risk | Expected Wall-Clock |
|-------|------|------|---------------------|
| 1 | **P-2** ERA cleanup | Very low | 2 min |
| 2 | **P-1** Ops/WHIP re-run + residual investigation | Low | 10 min |
| 3 | **P-4** Statcast retry backfill | Low | 10–20 min (API-bound) |
| 4 | **P-3** Orphan investigation + Ohtani/Lorenzen override | Medium | 45 min |
| 5 | Final validation audit + HANDOFF update | — | 10 min |

Rationale: P-2 is the smallest atomic win and de-risks the session. P-1 is mostly diagnostic. P-4 is slow but "fire and check back" (you can start it, then work on P-3 while Statcast runs). P-3 is last because it's the only task requiring judgment calls.

---

## Task P-2: Legacy Impossible ERA Cleanup

**Files touched:** None (live SQL only).

**Before state:** 1 row in `mlb_player_stats` with `era > 100` (legacy data from before the Pydantic validator landed in commit `c6fb7b3`).

**After state:** 0 rows with `era > 100 OR era < 0`.

### Steps

- [ ] **Step 1: Identify the offending row (diagnosis only)**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    rows = db.execute(text('''
        SELECT id, bdl_player_id, era, earned_runs, innings_pitched, game_id, game_date
        FROM mlb_player_stats
        WHERE era > 100 OR era < 0
        ORDER BY era DESC
    ''')).fetchall()
    print(f'Found {len(rows)} impossible ERA rows:')
    for r in rows:
        print(dict(r._mapping))
finally:
    db.close()
"
```
Expected: 1 row. **Record the `id` value.** If 0 rows, P-2 is already resolved — skip to P-1.

- [ ] **Step 2: Null-out impossible ERA values**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    r = db.execute(text('UPDATE mlb_player_stats SET era = NULL WHERE era > 100 OR era < 0'))
    print(f'Rows updated: {r.rowcount}')
    db.commit()
finally:
    db.close()
"
```
Expected: `Rows updated: 1`.

- [ ] **Step 3: Verify**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    c = db.execute(text('SELECT COUNT(*) FROM mlb_player_stats WHERE era > 100 OR era < 0')).scalar()
    print(f'Impossible ERA rows remaining: {c}')
    assert c == 0, f'FAIL: {c} remaining'
finally:
    db.close()
"
```
Expected: `Impossible ERA rows remaining: 0`.

- [ ] **Step 4: Record before/after metrics**

Append to `reports/2026-04-11-production-deployment-results.md` (create if absent):
```markdown
## P-2 Legacy ERA Cleanup
- Before: 1 row with era > 100 (id=<from Step 1>)
- After: 0 rows with era > 100
- Method: railway run python SQL exec
- Timestamp: <UTC timestamp>
```

### Rollback

If the impossible ERA needs to be restored for audit purposes:
```bash
# Rollback only needed if we discover the "impossible" value was actually legitimate.
# Since we captured the row in Step 1, restore from the recorded earned_runs/innings_pitched:
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    # Replace <id> with recorded id from Step 1
    db.execute(text('UPDATE mlb_player_stats SET era = <old_era_value> WHERE id = <id>'))
    db.commit()
finally:
    db.close()
"
```
**Note:** Rollback is unnecessary — the Pydantic validator (commit `c6fb7b3`) explicitly rejects `era > 100` as impossible, so any such row is a data bug by definition.

---

## Task P-1: OPS/WHIP Backfill Re-run + Residual Investigation

**Files touched:** None (endpoint call only).

**Endpoint:** `POST /admin/backfill-ops-whip` (`backend/admin_backfill_ops_whip.py`, no auth required).

**Hypothesis to validate:** The 1,639 remaining NULL ops are rows where `obp IS NULL` or `slg IS NULL` (pitcher rows, 0-AB appearances). If confirmed, they are unbackfillable and the fix is correct.

### Steps

- [ ] **Step 1: Baseline NULL counts + categorize**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    q = text('''
      SELECT
        COUNT(*) FILTER (WHERE ops IS NULL) AS null_ops,
        COUNT(*) FILTER (WHERE ops IS NULL AND obp IS NOT NULL AND slg IS NOT NULL) AS null_ops_backfillable,
        COUNT(*) FILTER (WHERE ops IS NULL AND (obp IS NULL OR slg IS NULL)) AS null_ops_unbackfillable,
        COUNT(*) FILTER (WHERE whip IS NULL) AS null_whip,
        COUNT(*) FILTER (WHERE whip IS NULL AND walks_allowed IS NOT NULL AND hits_allowed IS NOT NULL AND innings_pitched IS NOT NULL AND innings_pitched != '') AS null_whip_backfillable,
        COUNT(*) FILTER (WHERE whip IS NULL AND (walks_allowed IS NULL OR hits_allowed IS NULL OR innings_pitched IS NULL OR innings_pitched = '')) AS null_whip_unbackfillable
      FROM mlb_player_stats
    ''')
    row = db.execute(q).fetchone()
    print(dict(row._mapping))
finally:
    db.close()
"
```
Expected shape:
```
{'null_ops': N, 'null_ops_backfillable': X, 'null_ops_unbackfillable': Y, 'null_whip': M, ...}
```
**Record all six values.**

- [ ] **Step 2: Trigger the backfill endpoint**

```bash
curl -X POST "$BASE_URL/admin/backfill-ops-whip" \
  -H "Content-Type: application/json" \
  -w "\nHTTP %{http_code} in %{time_total}s\n" \
  | tee reports/2026-04-11-backfill-ops-whip-response.json
```
Expected response shape:
```json
{
  "status": "success",
  "ops_updated": <int>,
  "whip_updated": <int>,
  "initial_ops_null": <int>,
  "initial_whip_null": <int>,
  "final_ops_null": <int>,
  "final_whip_null": <int>,
  "total_rows": <int>
}
```
Expected HTTP code: 200. If "status": "error", stop and inspect `error` field.

- [ ] **Step 3: Re-run Step 1 diagnostic to confirm unbackfillable residuals**

Run the exact same query as Step 1. Expected:
- `null_ops_backfillable` → 0 (all rows with both obp+slg now have ops)
- `null_ops_unbackfillable` → should equal `final_ops_null` from Step 2 response
- `null_whip_backfillable` → 0
- `null_whip_unbackfillable` → should equal `final_whip_null` from Step 2 response

**If `null_ops_backfillable > 0` after Step 2, the endpoint is broken — stop and inspect `backend/admin_backfill_ops_whip.py:46-69`.**

- [ ] **Step 4: Sample the unbackfillable rows to confirm the hypothesis**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    rows = db.execute(text('''
        SELECT bdl_player_id, at_bats, hits, obp, slg, ops,
               innings_pitched, walks_allowed, hits_allowed, whip
        FROM mlb_player_stats
        WHERE ops IS NULL AND (obp IS NULL OR slg IS NULL)
        LIMIT 10
    ''')).fetchall()
    for r in rows:
        print(dict(r._mapping))
finally:
    db.close()
"
```
Expected: rows where `at_bats = 0` or `obp IS NULL` (pitcher stat lines). This confirms the hypothesis that residual NULLs are structural, not a backfill bug.

- [ ] **Step 5: Record findings**

Append to `reports/2026-04-11-production-deployment-results.md`:
```markdown
## P-1 OPS/WHIP Backfill
- Before: null_ops=<N>, null_whip=<M>
- After:  null_ops=<N'>, null_whip=<M'>
- Rows updated: ops=<X>, whip=<Y>
- Backfillable residuals remaining: ops=<0 expected>, whip=<0 expected>
- Unbackfillable residuals (structural NULLs): ops=<Y>, whip=<M'-X>
- Hypothesis confirmed: <yes/no — remaining NULLs are pitcher/0-AB rows>
```

### Rollback

The endpoint only sets fields to computed values — rollback is to set them back to NULL:
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    # WARNING: Only run this if the backfill caused data corruption.
    # Normal operation: DO NOT ROLLBACK.
    print('Rollback intentionally left as manual step — requires explicit justification')
finally:
    db.close()
"
```

---

## Task P-4: Statcast Retry Backfill

**Files touched:** None (endpoint call).

**Endpoint:** `POST /admin/backfill/statcast` (`backend/main.py:3144`, requires admin API key). Delegates to `scripts/backfill_statcast.py`, which uses the new `backend/services/retry_logic.py` (from commit `4e11ab0`) with exponential backoff for 502/503 errors.

**Expected duration:** 10–20 minutes. Can run in background while executing P-3.

**Important:** The endpoint has a `10–20 min` docstring warning. Use `--max-time 1800` on curl (30-minute timeout).

### Steps

- [ ] **Step 1: Baseline row count**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    c = db.execute(text('SELECT COUNT(*) FROM statcast_performances')).scalar()
    print(f'statcast_performances baseline: {c}')
finally:
    db.close()
"
```
Expected: `0` (empty table per HANDOFF.md).

- [ ] **Step 2: Trigger Statcast backfill**

```bash
curl -X POST "$BASE_URL/admin/backfill/statcast" \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  --max-time 1800 \
  -w "\nHTTP %{http_code} in %{time_total}s\n" \
  | tee reports/2026-04-11-statcast-backfill-response.json
```
Expected response shape (from `scripts/backfill_statcast.py`):
```json
{
  "status": "success",
  "records_processed": <int>,
  "dates_processed": <int>,
  "elapsed_ms": <int>
}
```
Expected HTTP 200. If HTTP 500 or timeout, check Railway logs for retry-logic behavior:
```bash
railway logs | grep -i "statcast\|retry" | tail -50
```

- [ ] **Step 3: Verify population**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    total = db.execute(text('SELECT COUNT(*) FROM statcast_performances')).scalar()
    distinct_dates = db.execute(text('SELECT COUNT(DISTINCT game_date) FROM statcast_performances')).scalar()
    latest = db.execute(text('SELECT MAX(game_date) FROM statcast_performances')).scalar()
    print(f'total rows: {total}')
    print(f'distinct game_dates: {distinct_dates}')
    print(f'most recent game_date: {latest}')
finally:
    db.close()
"
```
Minimum acceptable: `total > 0`.
Ideal: `distinct_dates >= 15` (covers Mar 20 – Apr 8 window from script).

- [ ] **Step 4: Record findings**

Append to `reports/2026-04-11-production-deployment-results.md`:
```markdown
## P-4 Statcast Retry Backfill
- Before: 0 rows
- After: <N> rows across <D> distinct dates, latest <YYYY-MM-DD>
- Elapsed: <ms from response>
- Retry events observed in logs: <yes/no>
- Status: <success | partial | failed>
```

### Rollback

```bash
# Rollback only if backfill corrupted existing data. Normally: do not rollback.
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    # Truncate only if explicitly needed
    r = db.execute(text('TRUNCATE statcast_performances'))
    db.commit()
    print('truncated')
finally:
    db.close()
"
```

---

## Task P-3: Orphan Investigation + Manual Overrides

**Prior evidence:** `reports/task-21-orphan-linking-results.md` (0/366 success) and `reports/2026-04-11-orphaned-players-audit.md` (Ohtani + Lorenzen are two-way players Yahoo splits into two keys; the rest are retired/minors/international prospects).

**Files touched:**
- Read: `reports/2026-04-11-orphaned-players-audit.md`
- Optional: new ad-hoc SQL script via `railway run python -c`

**Revised strategy:** **Do NOT re-run the fuzzy linker.** Investigate, apply 2 manual overrides (Ohtani, Lorenzen), document the rest.

### Steps

- [ ] **Step 1: Current orphan count (verify audit is still accurate)**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    c = db.execute(text('''
        SELECT COUNT(*) FROM position_eligibility
        WHERE bdl_player_id IS NULL
    ''')).scalar()
    print(f'Current orphans: {c}')
finally:
    db.close()
"
```
Expected: 366 (matching Apr 11 audit). If significantly different, re-audit before manual overrides.

- [ ] **Step 2: Locate BDL IDs for Ohtani and Lorenzen in player_id_mapping**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    for name_pattern in ('Ohtani', 'Lorenzen'):
        rows = db.execute(text('''
            SELECT id, full_name, mlb_id, yahoo_key, bdl_id
            FROM player_id_mapping
            WHERE full_name ILIKE :p
        '''), {'p': f'%{name_pattern}%'}).fetchall()
        print(f'--- {name_pattern} ---')
        for r in rows:
            print(dict(r._mapping))
finally:
    db.close()
"
```
**Record the BDL ID for each.** If Ohtani or Lorenzen has no row in `player_id_mapping`, the manual override is impossible — document and skip.

- [ ] **Step 3: Locate the orphaned position_eligibility rows for Ohtani and Lorenzen**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    rows = db.execute(text('''
        SELECT id, player_name, yahoo_player_key, positions, bdl_player_id
        FROM position_eligibility
        WHERE bdl_player_id IS NULL
          AND (player_name ILIKE '%Ohtani%' OR player_name ILIKE '%Lorenzen%')
        ORDER BY player_name, yahoo_player_key
    ''')).fetchall()
    for r in rows:
        print(dict(r._mapping))
finally:
    db.close()
"
```
Expected: 4 rows (2 per player — the Batter-role and Pitcher-role splits). **Record each `id`.**

- [ ] **Step 4: Apply manual BDL ID overrides**

Fill in the BDL IDs from Step 2. Do each player atomically:
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    # Replace <OHTANI_BDL_ID> and <LORENZEN_BDL_ID> with values from Step 2
    ohtani_bdl_id = <OHTANI_BDL_ID>
    lorenzen_bdl_id = <LORENZEN_BDL_ID>

    r1 = db.execute(text('''
        UPDATE position_eligibility
        SET bdl_player_id = :bid
        WHERE bdl_player_id IS NULL AND player_name ILIKE '%Ohtani%'
    '''), {'bid': ohtani_bdl_id})
    print(f'Ohtani rows updated: {r1.rowcount}')

    r2 = db.execute(text('''
        UPDATE position_eligibility
        SET bdl_player_id = :bid
        WHERE bdl_player_id IS NULL AND player_name ILIKE '%Lorenzen%'
    '''), {'bid': lorenzen_bdl_id})
    print(f'Lorenzen rows updated: {r2.rowcount}')

    db.commit()
finally:
    db.close()
"
```
Expected: 2 updates for each (one per role split). Total: 4 rows updated.

- [ ] **Step 5: Verify the overrides landed**

```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    rows = db.execute(text('''
        SELECT id, player_name, yahoo_player_key, bdl_player_id
        FROM position_eligibility
        WHERE player_name ILIKE '%Ohtani%' OR player_name ILIKE '%Lorenzen%'
        ORDER BY player_name, yahoo_player_key
    ''')).fetchall()
    for r in rows:
        print(dict(r._mapping))
    orphans_now = db.execute(text('SELECT COUNT(*) FROM position_eligibility WHERE bdl_player_id IS NULL')).scalar()
    print(f'Total orphans remaining: {orphans_now}')
finally:
    db.close()
"
```
Expected: all Ohtani/Lorenzen rows have `bdl_player_id` set; orphan count dropped by 4.

- [ ] **Step 6: Export remaining 362 orphans for future handling**

```bash
railway run python -c "
import csv
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    rows = db.execute(text('''
        SELECT id, player_name, yahoo_player_key, positions
        FROM position_eligibility
        WHERE bdl_player_id IS NULL
        ORDER BY player_name
    ''')).fetchall()
    with open('reports/2026-04-11-unmatchable-orphans.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['id', 'player_name', 'yahoo_player_key', 'positions'])
        for r in rows:
            w.writerow([r.id, r.player_name, r.yahoo_player_key, r.positions])
    print(f'exported {len(rows)} rows')
finally:
    db.close()
"
```
Expected: `exported 362` (or whatever remains after Step 4). This is the permanent record of known-unmatchable orphans — retired/international-prospect/minor-league players who genuinely lack BDL IDs.

- [ ] **Step 7: Record findings**

Append to `reports/2026-04-11-production-deployment-results.md`:
```markdown
## P-3 Orphan Investigation + Manual Overrides
- Before: 366 orphans (per Apr 11 audit)
- After: <N> orphans
- Manual overrides applied: Ohtani (2 rows), Lorenzen (2 rows) = 4 total
- Remaining 362 orphans exported to reports/2026-04-11-unmatchable-orphans.csv
- Classification of residuals: retired players, international prospects, minor-league,
  two-way players without player_id_mapping row
- Recommendation: mark as permanently unmatchable; do not re-run fuzzy linker
```

### Rollback

Manual overrides are reversible by resetting `bdl_player_id` to NULL for the 4 affected rows. Only do this if we discover the BDL IDs were wrong:
```bash
railway run python -c "
from backend.models import SessionLocal
from sqlalchemy import text
db = SessionLocal()
try:
    r = db.execute(text('''
        UPDATE position_eligibility
        SET bdl_player_id = NULL
        WHERE player_name ILIKE '%Ohtani%' OR player_name ILIKE '%Lorenzen%'
    '''))
    print(f'rows reset: {r.rowcount}')
    db.commit()
finally:
    db.close()
"
```

---

## Task: Final Validation Audit + HANDOFF Update

- [ ] **Step 1: Run final validation audit**

```bash
curl -s "$BASE_URL/admin/validation-audit" \
  -H "X-Admin-API-Key: $ADMIN_API_KEY" \
  | python -m json.tool \
  > reports/2026-04-11-final-validation-audit.json
```

- [ ] **Step 2: Diff baseline vs final**

```bash
diff reports/2026-04-11-baseline-validation-audit.json reports/2026-04-11-final-validation-audit.json
```
Expected improvements:
- `critical` count: 1 → 0 (P-2 cleared the impossible ERA)
- `high` count: 3 → 0–2 (P-1 and P-3 improvements depending on how validation defines "high")

- [ ] **Step 3: Update HANDOFF.md**

Rewrite the "REMAINING CRITICAL ISSUES" section (lines 226–290) to reflect the new state. Replace with a "2026-04-11 PRODUCTION DEPLOYMENT RESULTS" section that pulls directly from `reports/2026-04-11-production-deployment-results.md`.

Key points to document:
1. P-2: impossible ERA → NULL (complete)
2. P-1: backfillable residuals → 0; structural NULLs documented
3. P-4: Statcast populated with <N> rows (if successful)
4. P-3: 4 manual overrides applied; 362 documented as permanently unmatchable
5. Next phase unblocked: Research bundle K-34 through K-38 can now drive Tasks 4-9

- [ ] **Step 4: Commit plan + results**

```bash
git add PRODUCTION_DEPLOYMENT_PLAN.md \
        reports/2026-04-11-production-deployment-results.md \
        reports/2026-04-11-baseline-validation-audit.json \
        reports/2026-04-11-final-validation-audit.json \
        reports/2026-04-11-unmatchable-orphans.csv \
        reports/2026-04-11-backfill-ops-whip-response.json \
        reports/2026-04-11-statcast-backfill-response.json \
        HANDOFF.md
git commit -m "ops: execute production deployment P-1 through P-4 with results"
```

---

## Success Criteria Summary

| Task | Minimum Acceptable | Ideal |
|------|-------------------|-------|
| P-1  | Backfill endpoint returns success; backfillable residuals = 0 | — |
| P-2  | 0 rows with `era > 100 OR era < 0` | — |
| P-3  | 4 manual overrides landed; orphans exported to CSV | All 366 resolved (not achievable) |
| P-4  | `statcast_performances` > 0 rows | ≥15 distinct game_dates |

**Do NOT proceed to feature dev until the "Minimum Acceptable" column is green for all four tasks, or the reason for failure is documented with root cause.**

---

## What Happens After Completion

With production data stable, the platform unblocks the research bundle (K-34 through K-38) and the following feature work:
- Task 7: Derived stats (unblocked by K-34 BDL API doc)
- Task 9: VORP/z-score player valuation (unblocked by K-35, K-38)
- Tasks 4–5: Probable pitchers + statcast integration (unblocked by K-37, and P-4 landing)

Do not start any of the above until HANDOFF.md is updated.
