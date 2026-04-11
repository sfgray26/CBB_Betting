# Database Write Path Analysis (April 9, 2026)

## 1. Transaction Management
- **Status**: ✅ VERIFIED
- **Findings**: Every sync job in `backend/services/daily_ingestion.py` uses a consistent `try/except/finally` pattern with `db.commit()` on success and `db.rollback()` on error.
- **Example**:
  ```python
  try:
      # ... data processing ...
      db.commit()
  except Exception as exc:
      db.rollback()
      logger.error("Job failed: %s", exc)
  finally:
      db.close()
  ```

## 2. Dry Run Modes
- **Status**: ✅ NONE FOUND
- **Findings**: A grep search for `dry_run` and `DRY_RUN` in `backend/services/daily_ingestion.py` returned zero results. The jobs are configured to perform actual writes when triggered.

## 3. Database Write Operations
- **Status**: ✅ VERIFIED
- **Findings**: The jobs use a mix of SQLAlchemy ORM methods and PostgreSQL-specific upsert logic:
  - **`PositionEligibility`**: Uses `pg_insert` with `on_conflict_do_update` via `db.execute(stmt)`. This is a robust way to handle upserts on the `_pe_yahoo_uc` constraint.
  - **`ProbablePitcherSnapshot`**: Uses `db.merge(probable)`, which handles existing records based on the primary key.
  - **`PlayerIDMapping`**: Uses `db.merge(mapping)`.
  - **`PlayerDailyMetric`**: Uses `db.add(new_metric)`.

## 4. Record Filtering & Validation
- **Status**: ⚠️ PARTIAL SUCCESS
- **Findings**:
  - `PositionEligibility`: Filters out players without `positions` or `player_key`.
  - `ProbablePitcherSnapshot`: Depends on successful name resolution via `_resolve_player_name_to_bdl_id`. If mapping fails, the record is skipped.
  - `Statcast Ingestion`: Pydantic validation is strict; rows that fail validation are logged as warnings and skipped.

## 5. Conclusion
The write path is structurally sound. Failures to see data in the database are likely due to upstream data issues (e.g., Yahoo league being empty) or resolution failures (e.g., name mapping for pitchers) rather than a failure to commit or a hidden dry-run flag.
