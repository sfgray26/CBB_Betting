#!/usr/bin/env python3
"""
Migration: Widen canonical_projections.player_id from INTEGER to BIGINT.

Root cause: The projection_assembly_service uses -(yahoo_id) as a fallback
player_id namespace for players that have a Yahoo ID but no MLBAM ID. Some
Yahoo IDs (e.g., 91689018531) exceed INT4 range (~2.1B), causing
NumericValueOutOfRange errors on upsert.

BIGINT is the correct type — MLBAM IDs are positive 6-digit ints; negative
Yahoo fallback IDs can reach -91B+; both fit safely in BIGINT.

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
    "ALTER TABLE canonical_projections ALTER COLUMN player_id TYPE BIGINT",
]

with engine.begin() as conn:
    for sql in MIGRATIONS:
        conn.execute(text(sql))
        print(f"  OK: {sql}")

print("canonical_projections.player_id → BIGINT migration complete.")
