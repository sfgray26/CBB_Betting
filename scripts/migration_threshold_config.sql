-- PR 1.1: Threshold Config, Audit, and Feature Flag tables
-- Run: railway run psql -f scripts/migration_threshold_config.sql
-- Rollback: scripts/rollback_pr_1_1.sql

CREATE TABLE IF NOT EXISTS threshold_config (
    id SERIAL PRIMARY KEY,
    config_key TEXT NOT NULL,
    config_value JSONB NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    description TEXT,
    effective_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (config_key, scope)
);

CREATE INDEX IF NOT EXISTS idx_threshold_key_scope ON threshold_config(config_key, scope);

CREATE TABLE IF NOT EXISTS threshold_audit (
    id BIGSERIAL PRIMARY KEY,
    config_key TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB NOT NULL,
    changed_by TEXT,
    changed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feature_flags (
    id SERIAL PRIMARY KEY,
    flag_name TEXT NOT NULL UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rollout_pct INTEGER NOT NULL DEFAULT 0 CHECK (rollout_pct BETWEEN 0 AND 100),
    scope TEXT DEFAULT 'global',
    description TEXT
);
