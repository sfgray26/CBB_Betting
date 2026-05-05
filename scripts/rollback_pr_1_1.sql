-- Rollback for PR 1.1 — ONLY run in emergency
DROP TABLE IF EXISTS feature_flags CASCADE;
DROP TABLE IF EXISTS threshold_audit CASCADE;
DROP TABLE IF EXISTS threshold_config CASCADE;
