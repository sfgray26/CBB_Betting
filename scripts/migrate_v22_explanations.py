#!/usr/bin/env python
"""
P19 -- Decision Explanations Table (v22)

Creates decision_explanations for human-readable decision traces derived from
all P14-P18 signals (player_scores, player_momentum, simulation_results,
backtest_results) linked to each decision_results row.

Key design decisions:
  - Natural key: (decision_id,) -- 1:1 with decision_results
  - Named constraint _de_decision_id_uc matches SQLAlchemy ORM unique=True
    so ON CONFLICT targeting works in daily_ingestion._run_explainability()
  - factors_json stored as JSONB for richer querying than plain JSON
  - summary capped at 500 chars; narrative fields capped at 200 chars
  - computed_at uses TIMESTAMPTZ -- always timezone-aware
  - Upstream: decision_results (P17), player_scores (P14), player_momentum (P15),
              simulation_results (P16), backtest_results (P18)
  - Downstream: P20 UI display, API endpoint /admin/explanations/{decision_id}

Usage:
    python scripts/migrate_v22_explanations.py              # run upgrade
    python scripts/migrate_v22_explanations.py --downgrade  # run downgrade
    python scripts/migrate_v22_explanations.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS decision_explanations (
    id                      BIGSERIAL       PRIMARY KEY,
    decision_id             BIGINT          NOT NULL,
    bdl_player_id           INTEGER         NOT NULL,
    as_of_date              DATE            NOT NULL,
    decision_type           VARCHAR(10)     NOT NULL,

    summary                 VARCHAR(500)    NOT NULL,
    -- JSONB array: [{name, value, label, weight, narrative}, ...]
    factors_json            JSONB           NOT NULL,
    confidence_narrative    VARCHAR(200),
    risk_narrative          VARCHAR(200),
    track_record_narrative  VARCHAR(200),

    computed_at             TIMESTAMPTZ     NOT NULL DEFAULT now()
);

-- Natural key: one explanation per decision row
-- Named to match SQLAlchemy ORM unique=True so ON CONFLICT targets work.
ALTER TABLE decision_explanations
    ADD CONSTRAINT _de_decision_id_uc
    UNIQUE (decision_id);

-- Primary access pattern: lookup by decision_id
CREATE INDEX IF NOT EXISTS idx_de_decision_id
    ON decision_explanations (decision_id);

-- Bulk reads by date (daily feed queries)
CREATE INDEX IF NOT EXISTS idx_de_date
    ON decision_explanations (as_of_date);

-- Player history lookup
CREATE INDEX IF NOT EXISTS idx_de_player_date
    ON decision_explanations (bdl_player_id, as_of_date);

COMMENT ON TABLE decision_explanations IS
    'P19 Explainability Layer -- human-readable decision traces per decision_results row. '
    'Computed daily by _run_explainability() (lock 100_024, 9 AM ET). '
    'Upstream: decision_results (P17), player_scores (P14), player_momentum (P15), '
    'simulation_results (P16), backtest_results (P18). '
    'factors_json: JSONB array of [{name, value, label, weight, narrative}] ranked by abs(weight) desc. '
    'Downstream: P20 UI display, GET /admin/explanations/{decision_id}.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_de_player_date;
DROP INDEX IF EXISTS idx_de_date;
DROP INDEX IF EXISTS idx_de_decision_id;
DROP TABLE IF EXISTS decision_explanations;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating decision_explanations ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in UPGRADE_SQL.split(";"):
            statement = statement.strip()
            if not statement or statement.startswith("--"):
                continue
            try:
                conn.execute(text(statement))
            except Exception as e:
                if "already exists" in str(e).lower():
                    print("  WARNING: Skipping (already exists): {}".format(str(e)[:100]))
                else:
                    raise
        print("SUCCESS: decision_explanations created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing decision_explanations ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(DOWNGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in DOWNGRADE_SQL.split(";"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    print("  WARNING: {}".format(str(e)[:100]))
        print("SUCCESS: decision_explanations removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v22 - Decision Explanations")
    parser.add_argument("--downgrade", action="store_true", help="Rollback migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    eng = create_engine(db_url)

    if args.downgrade:
        downgrade(eng, dry_run=args.dry_run)
    else:
        upgrade(eng, dry_run=args.dry_run)
