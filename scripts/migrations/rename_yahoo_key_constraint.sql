-- Drop duplicate yahoo_key unique constraint.
--
-- History:
--   The SQLAlchemy model defines UniqueConstraint("yahoo_key", name="_pim_yahoo_key_uc").
--   PostgreSQL also auto-created "player_id_mapping_yahoo_key_key" when the column was
--   added without an explicit name in an earlier migration.
--
-- As of 2026-04-28, BOTH constraints exist on the same column (confirmed via
-- GET /admin/db/constraints). The named constraint _pim_yahoo_key_uc is the
-- authoritative one; the auto-named duplicate is redundant.
--
-- Run via:
--   railway run psql $DATABASE_URL -f scripts/migrations/rename_yahoo_key_constraint.sql
-- Or:
--   psql $DATABASE_URL -c "ALTER TABLE player_id_mapping DROP CONSTRAINT player_id_mapping_yahoo_key_key;"

ALTER TABLE player_id_mapping
    DROP CONSTRAINT IF EXISTS player_id_mapping_yahoo_key_key;
