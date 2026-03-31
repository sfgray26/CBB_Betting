#!/usr/bin/env python
"""
ARCH-001 Phase 1 — Job Queue + Decision Audit Tables v11

Adds job_queue and execution_decisions tables for async job orchestration
and immutable audit logging of all lineup recommendations.

Usage:
    python scripts/migrate_v11_job_queue.py              # run upgrade
    python scripts/migrate_v11_job_queue.py --downgrade  # run downgrade
    python scripts/migrate_v11_job_queue.py --dry-run    # print SQL, no execute
"""

import argparse
import os
import sys

from sqlalchemy import create_engine, text

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPGRADE_SQL = """
-- Create job_queue table for async job orchestration
CREATE TABLE IF NOT EXISTS job_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 5,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    picked_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    result JSONB,
    error TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    league_key VARCHAR(100),
    team_key VARCHAR(100)
);

-- Create indexes for job_queue
CREATE INDEX IF NOT EXISTS idx_job_queue_status_priority
    ON job_queue (status, priority, created_at)
    WHERE status = 'pending';

CREATE INDEX IF NOT EXISTS idx_job_queue_league_team
    ON job_queue (league_key, team_key, created_at DESC);

-- Add comment for documentation
COMMENT ON TABLE job_queue IS 'Async job queue for heavy operations (lineup optimization, valuation). Phase 1 of API-Worker architecture.';

-- Create execution_decisions table for immutable audit logging
CREATE TABLE IF NOT EXISTS execution_decisions (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    league_key VARCHAR(100) NOT NULL,
    team_key VARCHAR(100) NOT NULL,
    target_date DATE NOT NULL,
    starters JSONB NOT NULL DEFAULT '[]',
    bench JSONB NOT NULL DEFAULT '[]',
    slot_assignments JSONB NOT NULL DEFAULT '{}',
    primary_reasoning JSONB NOT NULL DEFAULT '[]',
    category_impact JSONB NOT NULL DEFAULT '{}',
    confidence_score FLOAT,
    win_probability JSONB,
    expected_outcome_range JSONB,
    alternatives JSONB NOT NULL DEFAULT '[]',
    safety_checks JSONB NOT NULL DEFAULT '[]',
    audit JSONB NOT NULL DEFAULT '{}'
);

-- Create indexes for execution_decisions
CREATE INDEX IF NOT EXISTS idx_exec_decisions_team_date
    ON execution_decisions (league_key, team_key, target_date DESC);

CREATE INDEX IF NOT EXISTS idx_exec_decisions_request
    ON execution_decisions (request_id);

-- Add comment for documentation
COMMENT ON TABLE execution_decisions IS 'Immutable audit log of all lineup recommendations. Never deleted - forms backtesting corpus.';
"""

DOWNGRADE_SQL = """
-- Remove execution_decisions table and indexes
DROP INDEX IF EXISTS idx_exec_decisions_request;
DROP INDEX IF EXISTS idx_exec_decisions_team_date;
DROP TABLE IF EXISTS execution_decisions;

-- Remove job_queue table and indexes
DROP INDEX IF EXISTS idx_job_queue_league_team;
DROP INDEX IF EXISTS idx_job_queue_status_priority;
DROP TABLE IF EXISTS job_queue;
"""


def upgrade(engine, dry_run=False):
    """Apply the migration."""
    print("=== UPGRADE: Creating job_queue and execution_decisions tables ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(UPGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    # Use autocommit mode to avoid transaction block issues with CREATE INDEX
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        try:
            conn.execute(text(UPGRADE_SQL))
            print("SUCCESS: job_queue and execution_decisions tables created successfully")
        except Exception as e:
            # Ignore "already exists" errors
            if "already exists" in str(e).lower():
                print(f"  WARNING: Skipping (already exists): {str(e)[:80]}")
            else:
                raise


def downgrade(engine, dry_run=False):
    """Rollback the migration."""
    print("=== DOWNGRADE: Removing job_queue and execution_decisions tables ===")

    if dry_run:
        print("\n--- DRY RUN: Would execute the following SQL ---")
        print(DOWNGRADE_SQL)
        print("--- END DRY RUN ---\n")
        return

    # Use autocommit mode to avoid transaction block issues
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        for statement in DOWNGRADE_SQL.split(';'):
            statement = statement.strip()
            if statement and not statement.startswith('--'):
                try:
                    conn.execute(text(statement))
                except Exception as e:
                    print(f"  WARNING: {str(e)[:80]}")

        print("SUCCESS: job_queue and execution_decisions tables removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate v11 - Job Queue + Decision Audit Tables")
    parser.add_argument("--downgrade", action="store_true", help="Rollback migration")
    parser.add_argument("--dry-run", action="store_true", help="Print SQL without executing")
    args = parser.parse_args()

    # Get database URL from environment
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set")
        sys.exit(1)

    engine = create_engine(db_url)

    if args.downgrade:
        downgrade(engine, dry_run=args.dry_run)
    else:
        upgrade(engine, dry_run=args.dry_run)
