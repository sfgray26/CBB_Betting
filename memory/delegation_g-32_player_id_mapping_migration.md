# Delegation G-32 — Player ID Mapping Migration & Validation

> **To:** Gemini CLI (DevOps Lead)  
> **From:** Claude Code (Principal Architect)  
> **Date:** April 15, 2026  
> **Authority:** AGENTS.md — Gemini owns Railway deployment, env vars, and production DB execution.  
> **Escalation path:** If migration fails or row count remains >5K after dedupe, escalate to Claude immediately.

---

## Context

`player_id_mapping` has **~60,000 rows** when it should have **~2,000**. The root cause was identified and fixed in code by Claude on April 15, 2026:

- **Root cause:** `backend/services/daily_ingestion.py:_sync_player_id_mapping()` used `db.merge(mapping)` with `id=None`. Because there was **no unique constraint on `bdl_id`**, SQLAlchemy performed a fresh `INSERT` of ~2,000 players on every daily sync run.
- **Code fix deployed:**
  - `backend/services/daily_ingestion.py` — explicit SELECT-then-upsert by `bdl_id`
  - `backend/services/player_id_resolver.py` — aligned cache persistence with unique `bdl_id`
  - `backend/models.py` — added `UniqueConstraint("bdl_id", name="_pim_bdl_id_uc")` and `updated_at` column
  - `scripts/migrate_v14_player_id_mapping.py` — updated baseline DDL
  - `scripts/migrate_v28_player_id_mapping_fix.py` — **new migration to dedupe existing DB and enforce constraint**

See `HANDOFF.md` section `## PLAYER_ID_MAPPING DEDUPLICATION (April 15, 2026)` for full details.

---

## Your Mission

1. **Dry-run the migration**
   ```bash
   railway run python scripts/migrate_v28_player_id_mapping_fix.py --dry-run
   ```
   Review output. It should show:
   - `ALTER TABLE player_id_mapping ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;`
   - A CTE deduplication block
   - `ALTER TABLE player_id_mapping ADD CONSTRAINT _pim_bdl_id_uc UNIQUE (bdl_id);`

2. **Execute the migration**
   ```bash
   railway run python scripts/migrate_v28_player_id_mapping_fix.py
   ```
   Confirm success message: `SUCCESS: player_id_mapping deduped and constrained`

3. **Validate row count post-migration**
   Use `railway run` with a quick SQL check:
   ```bash
   railway run python -c "
   from sqlalchemy import create_engine, text
   from backend.models import SQLALCHEMY_DATABASE_URI
   engine = create_engine(SQLALCHEMY_DATABASE_URI)
   with engine.connect() as conn:
       total = conn.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
       dupes = conn.execute(text('SELECT COUNT(*) FROM (SELECT bdl_id FROM player_id_mapping WHERE bdl_id IS NOT NULL GROUP BY bdl_id HAVING COUNT(*) > 1) t')).scalar()
       print(f'Total rows: {total}')
       print(f'Duplicate bdl_ids: {dupes}')
   "
   ```
   **Expected:** Total rows ~2,000. Duplicate bdl_ids = 0.

4. **Monitor next scheduled sync for unique-constraint violations**
   After the migration, tail logs during the next `_sync_player_id_mapping` run (scheduled daily at 7:00 AM ET):
   ```bash
   railway logs --follow
   ```
   Look for:
   - `SYNC JOB SUCCESS: _sync_player_id_mapping` (good)
   - Any `IntegrityError`, `unique constraint`, or `_pim_bdl_id_uc` violation (bad — escalate immediately)

5. **Update HANDOFF.md**
   Append a new `## GEMINI DEVOPS REPORT (April 15, 2026)` subsection under `## CURRENT SESSION STATE` with:
   - Migration execution status
   - Pre-migration row count (if known from logs/DB)
   - Post-migration row count
   - Duplicate bdl_id count
   - Any log warnings or errors from the next sync run

---

## Constraints

- **Do NOT edit any `.py`, `.ts`, `.tsx`, or `.js` files.** This is a DevOps execution task only.
- If the migration fails for any reason, **stop and escalate to Claude** before retrying.
- If row count is still >> 2,000 after migration, escalate — the dedupe CTE may need tuning.

---

## Verbatim Prompt for Cold Execution

If you are reading this file cold, execute exactly:

```bash
# 1. Dry run
railway run python scripts/migrate_v28_player_id_mapping_fix.py --dry-run

# 2. Execute
railway run python scripts/migrate_v28_player_id_mapping_fix.py

# 3. Validate counts
railway run python -c "
from sqlalchemy import create_engine, text
from backend.models import SQLALCHEMY_DATABASE_URI
engine = create_engine(SQLALCHEMY_DATABASE_URI)
with engine.connect() as conn:
    total = conn.execute(text('SELECT COUNT(*) FROM player_id_mapping')).scalar()
    dupes = conn.execute(text('SELECT COUNT(*) FROM (SELECT bdl_id FROM player_id_mapping WHERE bdl_id IS NOT NULL GROUP BY bdl_id HAVING COUNT(*) > 1) t')).scalar()
    print(f'Total rows: {total}')
    print(f'Duplicate bdl_ids: {dupes}')
"

# 4. Monitor next sync (run manually or wait for 7 AM ET schedule)
railway logs --follow | grep -i "player_id_mapping\|_pim_bdl_id_uc\|integrity"
```

Then update `HANDOFF.md` with results.
