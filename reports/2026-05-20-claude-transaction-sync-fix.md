# Transaction Ledger De-sync Fix

**Date:** 2026-05-20
**Severity:** P1 — incorrect budget counter shown to user after add/drop
**Root cause:** Yahoo `get_transactions` API lags several minutes post-add/drop. Budget endpoint showed 0 acquisitions immediately after a confirmed add.

## What Changed

1. **`backend/models.py`** — Added `RosterAcquisition` table. Tracks each add/drop with `team_key`, player keys, `executed_at`, and `week_start` (Monday midnight ET). Indexed on `(team_key, week_start)`.

2. **`backend/routers/fantasy.py` — `add_fantasy_waiver_player`** — Added `db: Session = Depends(get_db)` parameter. On successful `client.add_drop_player()`, inserts a `RosterAcquisition` row. DB failure is logged and non-fatal (Yahoo transaction already succeeded).

3. **`backend/routers/fantasy.py` — `get_constraint_budget`** — Hoisted `week_start` / `week_end` computation out of the Yahoo try block (they were unreferenced if Yahoo threw). After Yahoo count, queries `RosterAcquisition` by `team_key` and `week_start`. Uses `max(local_count, yahoo_count)` so the counter never shows a lower value than what we know is true locally.

4. **`tests/test_fantasy_budget.py`** — Added 5 new tests covering max-wins logic, Monday midnight computation, and model instantiation with/without drop player.

## Why `max()` not `local_only`

Yahoo is authoritative for adds made before this fix was deployed, and for any adds made via the Yahoo web/app directly. `max()` ensures we always reflect the higher ground truth.

## Deployment Note (for Codex)

The `roster_acquisitions` table must exist on Railway before deploy. The local query is wrapped in try/except so if the table doesn't exist yet, the budget endpoint falls back to Yahoo count gracefully — no outage risk.

Run on Railway before deploying this commit:

```sql
CREATE TABLE IF NOT EXISTS roster_acquisitions (
    id SERIAL PRIMARY KEY,
    team_key VARCHAR(64) NOT NULL,
    player_added_key VARCHAR(32) NOT NULL,
    player_dropped_key VARCHAR(32),
    executed_at TIMESTAMPTZ NOT NULL,
    week_start TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ra_team_week ON roster_acquisitions (team_key, week_start);
```
