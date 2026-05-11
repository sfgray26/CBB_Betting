"""
PR 1.1 admin migration — to be added to backend/routers/admin.py

Follows the exact pattern of v28/v31/v32.
Run via POST /admin/migrate/v33 after deployment.
"""

ENDPOINT_CODE = '''

@router.post("/admin/migrate/v33")
async def run_migration_v33(user: str = Depends(verify_admin_api_key), db: Session = Depends(get_db)):
    """
    PR 1.1 — Create Threshold Config System Tables.

    Creates:
    - threshold_config    : runtime config values (JSONB) with scoping
    - threshold_audit     : audit trail for config changes
    - feature_flags       : feature toggles with rollout percentage
    """
    from sqlalchemy import text

    results = {"steps": []}

    # 1. threshold_config
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS threshold_config (
                id SERIAL PRIMARY KEY,
                config_key TEXT NOT NULL,
                config_value JSONB NOT NULL,
                scope TEXT NOT NULL DEFAULT 'global',
                description TEXT,
                effective_at TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (config_key, scope)
            )
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_threshold_key_scope
            ON threshold_config(config_key, scope)
        """))
        db.commit()
        results["steps"].append({"name": "threshold_config table", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "threshold_config table", "status": f"error: {e}"})

    # 2. threshold_audit
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS threshold_audit (
                id BIGSERIAL PRIMARY KEY,
                config_key TEXT NOT NULL,
                old_value JSONB,
                new_value JSONB NOT NULL,
                changed_by TEXT,
                changed_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_threshold_audit_key
            ON threshold_audit(config_key, changed_at DESC)
        """))
        db.commit()
        results["steps"].append({"name": "threshold_audit table", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "threshold_audit table", "status": f"error: {e}"})

    # 3. feature_flags
    try:
        db.execute(text("""
            CREATE TABLE IF NOT EXISTS feature_flags (
                id SERIAL PRIMARY KEY,
                flag_name TEXT NOT NULL UNIQUE,
                enabled BOOLEAN NOT NULL DEFAULT FALSE,
                rollout_pct INTEGER NOT NULL DEFAULT 0
                    CHECK (rollout_pct BETWEEN 0 AND 100),
                scope TEXT DEFAULT 'global',
                description TEXT
            )
        """))
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_feature_flags_name
            ON feature_flags(flag_name)
        """))
        db.commit()
        results["steps"].append({"name": "feature_flags table", "status": "created"})
    except Exception as e:
        results["steps"].append({"name": "feature_flags table", "status": f"error: {e}"})

    # 4. Seed baseline feature flags
    try:
        db.execute(text("""
            INSERT INTO feature_flags (flag_name, enabled, rollout_pct, description)
            VALUES
                ('statcast_sprint_speed_enabled', false, 0, 'Enable sprint_speed in scoring engine'),
                ('opportunity_enabled',           false, 0, 'Enable opportunity scoring modifier'),
                ('matchup_enabled',               false, 0, 'Enable matchup context boost'),
                ('market_signals_enabled',        false, 0, 'Enable market signal integration')
            ON CONFLICT (flag_name) DO NOTHING
        """))
        db.commit()
        results["steps"].append({"name": "feature_flags seed", "status": "seeded"})
    except Exception as e:
        results["steps"].append({"name": "feature_flags seed", "status": f"error: {e}"})

    # Verify
    verification = {}
    try:
        for table in ['threshold_config', 'threshold_audit', 'feature_flags']:
            result = db.execute(text("""
                SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = :tbl)
            """), {"tbl": table}).fetchone()
            verification[table] = "EXISTS" if result[0] else "MISSING"

        flag_count = db.execute(text("SELECT COUNT(*) FROM feature_flags")).fetchone()
        verification["feature_flags count"] = flag_count[0]
    except Exception as e:
        verification["error"] = str(e)

    results["verification"] = verification
    return results
'''

print(ENDPOINT_CODE)
