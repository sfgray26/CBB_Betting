"""
PR 1.1 — Create Threshold Config System Tables

Tables:
  - threshold_config    : runtime config values (JSONB) with scoping
  - threshold_audit     : audit trail for config changes
  - feature_flags       : feature toggles with rollout percentage

Idempotent: safe to run multiple times.
"""

import os
import sys
import psycopg2
from psycopg2 import sql


def get_db_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        # Try loading from .env in repo root
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("DATABASE_URL="):
                        url = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
                        break
    if not url:
        print("ERROR: DATABASE_URL not found in environment or .env")
        sys.exit(1)
    return url


DDL = """
-- ============================================================
-- threshold_config
-- ============================================================
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

CREATE INDEX IF NOT EXISTS idx_threshold_key_scope
    ON threshold_config(config_key, scope);

-- ============================================================
-- threshold_audit
-- ============================================================
CREATE TABLE IF NOT EXISTS threshold_audit (
    id BIGSERIAL PRIMARY KEY,
    config_key TEXT NOT NULL,
    old_value JSONB,
    new_value JSONB NOT NULL,
    changed_by TEXT,
    changed_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_threshold_audit_key
    ON threshold_audit(config_key, changed_at DESC);

-- ============================================================
-- feature_flags
-- ============================================================
CREATE TABLE IF NOT EXISTS feature_flags (
    id SERIAL PRIMARY KEY,
    flag_name TEXT NOT NULL UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT FALSE,
    rollout_pct INTEGER NOT NULL DEFAULT 0
        CHECK (rollout_pct BETWEEN 0 AND 100),
    scope TEXT DEFAULT 'global',
    description TEXT
);

CREATE INDEX IF NOT EXISTS idx_feature_flags_name
    ON feature_flags(flag_name);

-- ============================================================
-- Seed baseline feature flags (idempotent)
-- ============================================================
INSERT INTO feature_flags (flag_name, enabled, rollout_pct, description)
VALUES
    ('statcast_sprint_speed_enabled', false, 0, 'Enable sprint_speed in scoring engine'),
    ('opportunity_enabled',           false, 0, 'Enable opportunity scoring modifier'),
    ('matchup_enabled',               false, 0, 'Enable matchup context boost'),
    ('market_signals_enabled',        false, 0, 'Enable market signal integration')
ON CONFLICT (flag_name) DO NOTHING;
"""


def migrate():
    db_url = get_db_url()
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    cur = conn.cursor()

    try:
        print("Applying PR 1.1 migration: threshold_config system...")
        cur.execute(DDL)
        conn.commit()

        # Verification
        cur.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name IN ('threshold_config', 'threshold_audit', 'feature_flags')
            ORDER BY table_name;
        """)
        tables = [r[0] for r in cur.fetchall()]
        print(f"  Created/verified tables: {tables}")

        cur.execute("SELECT COUNT(*) FROM feature_flags")
        flag_count = cur.fetchone()[0]
        print(f"  Feature flags seeded: {flag_count}")

        print("PR 1.1 migration applied successfully.")

    except Exception as exc:
        conn.rollback()
        print(f"ERROR: {exc}")
        sys.exit(1)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    migrate()
