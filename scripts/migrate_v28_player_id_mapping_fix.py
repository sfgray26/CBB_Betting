#!/usr/bin/env python
"""
P28 -- Fix player_id_mapping duplicates and schema gaps (v28)

Addresses the root cause of 60K duplicate rows in player_id_mapping:
  1. daily_ingestion.py used db.merge() without a bdl_id unique constraint
  2. The table was missing an updated_at column (referenced by backfill script)

Changes:
  - ADD COLUMN updated_at TIMESTAMPTZ (if missing)
  - DEDUPE rows by bdl_id, keeping the richest row per player
  - ADD UNIQUE CONSTRAINT on bdl_id (_pim_bdl_id_uc)

Idempotent: uses IF NOT EXISTS / IF NOT EXISTS. Safe to run on any environment.

Usage:
    python scripts/migrate_v28_player_id_mapping_fix.py              # run upgrade
    python scripts/migrate_v28_player_id_mapping_fix.py --downgrade  # run downgrade
    python scripts/migrate_v28_player_id_mapping_fix.py --dry-run    # print SQL only
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- 1. Add updated_at if missing (backfill script already references it)
ALTER TABLE player_id_mapping
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;

-- 2. Deduplicate by bdl_id, keeping the "best" row per player.
--    Priority: has yahoo_key > has mlbam_id > lowest id (stable).
WITH ranked AS (
    SELECT id,
           ROW_NUMBER() OVER (
               PARTITION BY bdl_id
               ORDER BY
                   CASE WHEN yahoo_key IS NOT NULL THEN 0 ELSE 1 END,
                   CASE WHEN mlbam_id IS NOT NULL THEN 0 ELSE 1 END,
                   id ASC
           ) AS rn
    FROM player_id_mapping
    WHERE bdl_id IS NOT NULL
)
DELETE FROM player_id_mapping
WHERE id IN (
    SELECT id FROM ranked WHERE rn > 1
);

-- 3. Enforce uniqueness at the DB level so db.merge() or accidental inserts
--    cannot recreate the duplication problem.
ALTER TABLE player_id_mapping
    ADD CONSTRAINT _pim_bdl_id_uc UNIQUE (bdl_id);
"""

DOWNGRADE_SQL = """
ALTER TABLE player_id_mapping DROP CONSTRAINT IF EXISTS _pim_bdl_id_uc;
ALTER TABLE player_id_mapping DROP COLUMN IF EXISTS updated_at;
"""


def upgrade(engine, dry_run=False):
    print("=== UPGRADE: Fixing player_id_mapping duplicates (v28) ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in UPGRADE_SQL.split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            try:
                conn.execute(text(stmt))
                print(f"  OK: {stmt.splitlines()[0][:80]}")
            except Exception as e:
                if "already exists" in str(e).lower() or "does not exist" in str(e).lower():
                    print(f"  SKIP: {str(e)[:100]}")
                else:
                    raise
    print("SUCCESS: player_id_mapping deduped and constrained")


def downgrade(engine, dry_run=False):
    print("=== DOWNGRADE: Reverting player_id_mapping fix (v28) ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(DOWNGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in DOWNGRADE_SQL.split(";"):
            stmt = statement.strip()
            if not stmt:
                continue
            try:
                conn.execute(text(stmt))
                print(f"  OK: {stmt[:80]}")
            except Exception as e:
                if "does not exist" in str(e).lower() or "already exists" in str(e).lower():
                    print(f"  SKIP: {str(e)[:100]}")
                else:
                    raise
    print("SUCCESS: player_id_mapping constraints removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate player_id_mapping fix (v28)")
    parser.add_argument("--downgrade", action="store_true", help="Run downgrade")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL only")
    args = parser.parse_args()

    from backend.models import SQLALCHEMY_DATABASE_URI

    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    if args.downgrade:
        downgrade(engine, dry_run=args.dry_run)
    else:
        upgrade(engine, dry_run=args.dry_run)
