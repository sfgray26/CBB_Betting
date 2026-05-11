#!/usr/bin/env python3
"""
Migration: Add projected_numerator and projected_denominator to category_impacts table.

These columns support true marginal rate-stat scoring in CategoryAwareScorer.
Idempotent — safe to run multiple times.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL environment variable not set")
    sys.exit(1)

engine = create_engine(DATABASE_URL)

MIGRATIONS = [
    """ALTER TABLE category_impacts
       ADD COLUMN IF NOT EXISTS projected_numerator FLOAT""",
    """ALTER TABLE category_impacts
       ADD COLUMN IF NOT EXISTS projected_denominator FLOAT""",
    """CREATE INDEX IF NOT EXISTS idx_ci_marginal
       ON category_impacts (canonical_projection_id, category)
       WHERE projected_numerator IS NOT NULL""",
]

with engine.begin() as conn:
    for sql in MIGRATIONS:
        conn.execute(text(sql))
        print(f"  OK: {sql[:60]}...")

print("category_impacts marginal migration complete.")
