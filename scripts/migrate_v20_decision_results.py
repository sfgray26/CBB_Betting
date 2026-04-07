#!/usr/bin/env python
"""
P17 -- Decision Engine Results Table (v20)

Creates decision_results for greedy lineup optimization and waiver intelligence
outputs derived from player_scores (P14), player_momentum (P15), and
simulation_results (P16).

Key design decisions:
  - Natural key: (as_of_date, decision_type, bdl_player_id) -- one result row
    per player per decision type per date
  - Named constraint _dr_date_type_player_uc matches SQLAlchemy ORM UniqueConstraint
    so ON CONFLICT targeting works in daily_ingestion._run_decision_optimization()
  - decision_type: "lineup" (slot assignment) | "waiver" (add/drop recommendation)
  - target_slot: normalized slot name ("OF", "SP", "Util", etc.) for lineup rows
  - drop_player_id: bdl_player_id of the waiver drop target (NULL for lineup rows)
  - computed_at uses TIMESTAMPTZ -- always timezone-aware
  - Upstream: simulation_results (P16 -- 6 AM ET), player_scores (P14), momentum (P15)
  - Downstream: P18 backtesting harness (decision quality tracking)

Usage:
    python scripts/migrate_v20_decision_results.py              # run upgrade
    python scripts/migrate_v20_decision_results.py --downgrade  # run downgrade
    python scripts/migrate_v20_decision_results.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
CREATE TABLE IF NOT EXISTS decision_results (
    id              BIGSERIAL       PRIMARY KEY,
    as_of_date      DATE            NOT NULL,
    decision_type   VARCHAR(10)     NOT NULL,
    bdl_player_id   INTEGER         NOT NULL,
    target_slot     VARCHAR(10),
    drop_player_id  INTEGER,
    lineup_score    DOUBLE PRECISION,
    value_gain      DOUBLE PRECISION,
    confidence      DOUBLE PRECISION NOT NULL,
    reasoning       VARCHAR(500),
    computed_at     TIMESTAMPTZ     NOT NULL DEFAULT now()
);

ALTER TABLE decision_results
    ADD CONSTRAINT IF NOT EXISTS _dr_date_type_player_uc
    UNIQUE (as_of_date, decision_type, bdl_player_id);

CREATE INDEX IF NOT EXISTS idx_dr_date_type
    ON decision_results (as_of_date, decision_type);

CREATE INDEX IF NOT EXISTS idx_dr_player_date
    ON decision_results (bdl_player_id, as_of_date);

COMMENT ON TABLE decision_results IS
    'P17 Decision Engine results -- lineup and waiver optimization outputs. '
    'Computed daily by _run_decision_optimization() (lock 100_022, 7 AM ET). '
    'Input: player_scores (P14) + player_momentum (P15) + simulation_results (P16). '
    'Decision types: lineup (slot assignment) | waiver (add/drop recommendation). '
    'Natural key: (as_of_date, decision_type, bdl_player_id). '
    'Downstream: P18 backtesting harness for decision quality tracking.'
"""

DOWNGRADE_SQL = """
DROP INDEX IF EXISTS idx_dr_player_date;
DROP INDEX IF EXISTS idx_dr_date_type;
DROP TABLE IF EXISTS decision_results;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Creating decision_results ===")

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
                    print(f"  WARNING: Skipping (already exists): {str(e)[:100]}")
                else:
                    raise
        print("SUCCESS: decision_results created")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Removing decision_results ===")

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
                    print(f"  WARNING: {str(e)[:100]}")
        print("SUCCESS: decision_results removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v20 - Decision Results")
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
