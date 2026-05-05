"""
Add Baseball Savant Statcast park factor columns.

Run:
    python scripts/migration_savant_park_factors.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATABASE_URL = os.getenv("DATABASE_URL")

ALTER_COLUMNS = [
    ("venue_name", "VARCHAR(100)"),
    ("team", "VARCHAR(10)"),
    ("venue_id", "INTEGER"),
    ("rolling_years", "INTEGER"),
    ("bat_side", "VARCHAR(10)"),
    ("condition", "VARCHAR(30)"),
    ("year_range", "VARCHAR(20)"),
    ("source_url", "TEXT"),
    ("woba_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("wobacon_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("xwobacon_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("obp_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("bb_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("so_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("bacon_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("singles_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("doubles_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("triples_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("hardhit_factor", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
    ("n_pa", "INTEGER"),
]


def main() -> None:
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL is required")

    conn = psycopg2.connect(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            for column_name, column_type in ALTER_COLUMNS:
                cur.execute(
                    f"ALTER TABLE park_factors ADD COLUMN IF NOT EXISTS {column_name} {column_type}"
                )

            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_park_factors_team ON park_factors(team)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS ix_park_factors_venue_id ON park_factors(venue_id)"
            )

        conn.commit()
        print("Savant park factor migration ready: columns/indexes verified")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
