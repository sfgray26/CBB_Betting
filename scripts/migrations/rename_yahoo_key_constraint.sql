-- Rename auto-generated yahoo_key unique constraint to match SQLAlchemy model expectation.
--
-- The SQLAlchemy model defines:
--   UniqueConstraint("yahoo_key", name="_pim_yahoo_key_uc")
--
-- But when the table was created/migrated in production, PostgreSQL auto-named
-- the constraint "player_id_mapping_yahoo_key_key" instead of "_pim_yahoo_key_uc".
--
-- After running this migration:
--   1. Update daily_ingestion.py ON CONFLICT back to constraint="_pim_yahoo_key_uc"
--   2. Model and DB are in sync
--
-- Run via: railway run psql $DATABASE_URL -f scripts/migrations/rename_yahoo_key_constraint.sql
-- Or: psql $DATABASE_URL -c "ALTER TABLE player_id_mapping RENAME CONSTRAINT player_id_mapping_yahoo_key_key TO _pim_yahoo_key_uc;"

ALTER TABLE player_id_mapping
    RENAME CONSTRAINT player_id_mapping_yahoo_key_key TO _pim_yahoo_key_uc;
