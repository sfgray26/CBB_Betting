# Fix Missing Migration for daily_availability_overrides

**Date:** 2026-06-12
**Issue:** Deploy blocker — missing migration causing 503 errors
**Goal:** Create and apply migration for `daily_availability_overrides` table

---

## Problem

The app throws 503 errors on waiver endpoints. Root cause: `daily_availability_overrides` table exists in code but missing migration to create it in the database.

## Steps

1. **Find the table definition**
   ```bash
   grep -rn "class DailyAvailabilityOverride" backend/models.py
   ```

2. **Check if migration exists**
   ```bash
   ls -la alembic/versions/ | grep availability
   ```

3. **Generate migration if missing**
   ```bash
   alembic revision --autogenerate -m "add daily_availability_overrides table"
   ```

4. **Review the migration file** — ensure it creates the table with all columns

5. **Apply migration**
   ```bash
   alembic upgrade head
   ```

6. **Verify table exists**
   ```bash
   python -c "from backend.models import DailyAvailabilityOverride; print('Table exists')"
   ```

7. **Test waiver endpoint**
   ```bash
   curl http://localhost:8000/api/fantasy/waiver-wire
   ```

## Expected Result

- Table `daily_availability_overrides` exists in Railway PostgreSQL
- Waiver endpoints return 200 (not 503)
- App deploys successfully

## Commit

```bash
git add alembic/versions/
git commit -m "fix: add missing migration for daily_availability_overrides table (503 blocker)"
```