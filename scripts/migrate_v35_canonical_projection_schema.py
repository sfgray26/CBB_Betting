#!/usr/bin/env python
"""
V35 — Canonical Projection Schema (Sprint 1a)

Creates five net-new tables that underpin the canonical projection system:

  canonical_projections  — single source of truth for all projection output
  category_impacts       — per-category z-scores/marginal-impact (1:N)
  divergence_flags       — sabermetric divergence alerts (1:N)
  player_identities      — strict cross-system identity mapping (no fuzzy)
  identity_quarantine    — staging area for unresolved player identity matches

All CREATE TABLE statements are IF NOT EXISTS — safe to re-run.
No existing tables or data are modified.

Usage
-----
    railway run python scripts/migrate_v35_canonical_projection_schema.py
    python scripts/migrate_v35_canonical_projection_schema.py [--dry-run] [--verify]
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- ── 1. canonical_projections ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS canonical_projections (
    id                          SERIAL PRIMARY KEY,
    projection_id               VARCHAR(36)  NOT NULL UNIQUE,
    player_id                   INTEGER      NOT NULL,
    player_type                 VARCHAR(10)  NOT NULL,
    source_engine               VARCHAR(25)  NOT NULL,
    projection_date             DATE         NOT NULL,
    season                      INTEGER      NOT NULL,
    source_ids                  JSONB        NOT NULL DEFAULT '[]',

    -- Playing time (nullable — batters have PA/AB, pitchers have IP)
    projected_pa                FLOAT,
    projected_ab                FLOAT,
    projected_ip                FLOAT,

    -- Projected counting stats — batters
    proj_hr                     INTEGER,
    proj_sb                     INTEGER,
    proj_r                      INTEGER,
    proj_rbi                    INTEGER,

    -- Projected counting stats — pitchers
    proj_w                      INTEGER,
    proj_sv                     INTEGER,
    proj_k                      INTEGER,

    -- Projected rate stats — batters
    proj_avg                    FLOAT,
    proj_obp                    FLOAT,
    proj_slg                    FLOAT,
    proj_ops                    FLOAT,

    -- Projected rate stats — pitchers
    proj_era                    FLOAT,
    proj_whip                   FLOAT,
    proj_k9                     FLOAT,

    -- Core advanced stats — batters
    woba                        FLOAT,
    xwoba                       FLOAT,
    wrc_plus                    FLOAT,
    iso                         FLOAT,
    bb_pct                      FLOAT,
    k_pct                       FLOAT,
    barrel_pct                  FLOAT,
    hardhit_pct                 FLOAT,
    xslg                        FLOAT,
    xba                         FLOAT,

    -- Core advanced stats — pitchers
    era                         FLOAT,
    whip                        FLOAT,
    k9                          FLOAT,
    fip                         FLOAT,
    xfip                        FLOAT,
    siera                       FLOAT,
    xera                        FLOAT,
    csw_pct                     FLOAT,
    swstr_pct                   FLOAT,
    chase_pct                   FLOAT,
    savant_pitch_quality_score  FLOAT,

    -- Confidence & explainability
    confidence_score            FLOAT,
    sample_size                 FLOAT,
    shrinkage_applied           FLOAT,
    prior_value                 FLOAT,
    posterior_value             FLOAT,
    explainability_metadata     JSONB        NOT NULL DEFAULT '{}',

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_cp_player_date      ON canonical_projections (player_id, projection_date);
CREATE INDEX IF NOT EXISTS idx_cp_season_engine    ON canonical_projections (season, source_engine);
CREATE INDEX IF NOT EXISTS idx_cp_type_confidence  ON canonical_projections (player_type, confidence_score);

COMMENT ON TABLE canonical_projections IS
    'V35: Single projection source of truth. No downstream service queries player_projections directly after population.';

-- ── 2. category_impacts ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS category_impacts (
    id                          SERIAL PRIMARY KEY,
    canonical_projection_id     INTEGER      NOT NULL
        REFERENCES canonical_projections(id) ON DELETE CASCADE,
    category                    VARCHAR(20)  NOT NULL,
    projected_value             FLOAT,
    z_score                     FLOAT,
    generic_marginal_impact     FLOAT,
    denominator_weight          FLOAT
);

CREATE INDEX IF NOT EXISTS idx_ci_projection_category  ON category_impacts (canonical_projection_id, category);
CREATE INDEX IF NOT EXISTS idx_ci_category_impact      ON category_impacts (category, generic_marginal_impact);

COMMENT ON TABLE category_impacts IS
    'V35: Context-agnostic per-category baselines (Phase A). Matchup deltas computed at runtime.';

-- ── 3. divergence_flags ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS divergence_flags (
    id                          SERIAL PRIMARY KEY,
    canonical_projection_id     INTEGER      NOT NULL
        REFERENCES canonical_projections(id) ON DELETE CASCADE,
    flag_type                   VARCHAR(50)  NOT NULL,
    severity                    VARCHAR(10)  NOT NULL,
    metric_a                    FLOAT,
    metric_b                    FLOAT,
    delta_value                 FLOAT,
    threshold_used              FLOAT,
    adjustment_applied          TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_df_projection  ON divergence_flags (canonical_projection_id);
CREATE INDEX IF NOT EXISTS idx_df_severity    ON divergence_flags (severity, flag_type);

COMMENT ON TABLE divergence_flags IS
    'V35: Sabermetric divergence alerts from SabermetricDivergenceAnalyzer. Severity: INFO | WARNING | CRITICAL.';

-- ── 4. player_identities ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS player_identities (
    id              SERIAL PRIMARY KEY,
    yahoo_guid      VARCHAR(50)  UNIQUE,
    yahoo_id        VARCHAR(20)  UNIQUE,
    mlbam_id        INTEGER      UNIQUE,
    fangraphs_id    VARCHAR(30)  UNIQUE,
    full_name       VARCHAR(150) NOT NULL,
    normalized_name VARCHAR(150) NOT NULL,
    birth_year      INTEGER,
    active          BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_pi_normalized   ON player_identities (normalized_name);
CREATE INDEX IF NOT EXISTS idx_pi_fangraphs    ON player_identities (fangraphs_id);
CREATE INDEX IF NOT EXISTS idx_pi_mlbam        ON player_identities (mlbam_id);

COMMENT ON TABLE player_identities IS
    'V35: Strict identity mapping — exact matches only. Fuzzy/ambiguous entries go to identity_quarantine.';

-- ── 5. identity_quarantine ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS identity_quarantine (
    id                  SERIAL PRIMARY KEY,
    incoming_provider   VARCHAR(30)  NOT NULL,
    incoming_raw_name   VARCHAR(150) NOT NULL,
    incoming_raw_id     VARCHAR(50),
    proposed_player_id  INTEGER,
    match_score         FLOAT,
    match_candidates    JSONB        NOT NULL DEFAULT '[]',
    status              VARCHAR(20)  NOT NULL DEFAULT 'PENDING_REVIEW',
    resolution_notes    TEXT,
    resolved_by         VARCHAR(100),
    created_at          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_iq_status_created   ON identity_quarantine (status, created_at);
CREATE INDEX IF NOT EXISTS idx_iq_provider_name    ON identity_quarantine (incoming_provider, incoming_raw_name);

COMMENT ON TABLE identity_quarantine IS
    'V35: Unresolved player identity staging. PENDING_REVIEW entries are excluded from projection and waiver pipelines.';
"""

DOWNGRADE_SQL = """
DROP TABLE IF EXISTS divergence_flags CASCADE;
DROP TABLE IF EXISTS category_impacts CASCADE;
DROP TABLE IF EXISTS canonical_projections CASCADE;
DROP TABLE IF EXISTS identity_quarantine CASCADE;
DROP TABLE IF EXISTS player_identities CASCADE;
"""

VERIFY_SQL = """
SELECT table_name, pg_relation_size(quote_ident(table_name)) AS size_bytes
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN (
      'canonical_projections',
      'category_impacts',
      'divergence_flags',
      'player_identities',
      'identity_quarantine'
  )
ORDER BY table_name;
"""


def run(dry_run: bool = False, verify_only: bool = False) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    if verify_only:
        _verify(db_url)
        return

    print("=== V35: Canonical Projection Schema migration ===")

    if dry_run:
        print("[DRY RUN] Would execute:\n")
        print(UPGRADE_SQL)
        return

    eng = create_engine(db_url)
    with eng.begin() as conn:
        for statement in UPGRADE_SQL.strip().split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            # Skip pure-comment chunks
            sql_lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
            if not any(ln.strip() for ln in sql_lines):
                continue
            print(f"  Executing: {stmt[:90]}...")
            conn.execute(text(stmt))

    print("\n=== Verification ===")
    _verify(db_url)


def _verify(db_url: str) -> None:
    expected = {
        "canonical_projections",
        "category_impacts",
        "divergence_flags",
        "player_identities",
        "identity_quarantine",
    }
    eng = create_engine(db_url)
    with eng.connect() as conn:
        rows = conn.execute(text(VERIFY_SQL)).fetchall()

    found = {r[0] for r in rows}
    missing = expected - found

    print(f"{'table':<30} {'size':>12}")
    print("-" * 44)
    for table_name, size_bytes in rows:
        print(f"{table_name:<30} {size_bytes:>12,} bytes")

    if missing:
        print(f"\nFAIL: Missing tables: {missing}")
        sys.exit(1)
    else:
        print(f"\nPASS: All {len(expected)} V35 tables present.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V35 canonical projection schema migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    parser.add_argument("--verify", action="store_true", help="Only run verification, skip migration")
    parser.add_argument("--downgrade", action="store_true", help="DROP all V35 tables (destructive!)")
    args = parser.parse_args()

    if args.downgrade:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            print("ERROR: DATABASE_URL not set", file=sys.stderr)
            sys.exit(1)
        print("=== V35 DOWNGRADE — dropping canonical projection tables ===")
        eng = create_engine(db_url)
        with eng.begin() as conn:
            conn.execute(text(DOWNGRADE_SQL))
        print("Done.")
    else:
        run(dry_run=args.dry_run, verify_only=args.verify)
