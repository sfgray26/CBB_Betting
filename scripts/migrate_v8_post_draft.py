"""
Migration v8: Post-draft time-series schema (EMAC-077 EPIC-1).

=============================================================================
HOW TO RUN
=============================================================================

  python scripts/migrate_v8_post_draft.py              # run upgrade
  python scripts/migrate_v8_post_draft.py --dry-run    # print SQL only, no changes
  python scripts/migrate_v8_post_draft.py --downgrade  # roll back all changes

Or on Railway:
  railway run python scripts/migrate_v8_post_draft.py

=============================================================================
WHAT THIS MIGRATION DOES
=============================================================================

Creates two new tables (idempotent — safe to re-run):

  player_daily_metrics   — sparse time-series of per-player analytics.
                           One row per (player_id, metric_date, sport).
                           Stores VORP, rolling z-scores, Statcast 2.0 fields,
                           and a flexible JSONB rolling_window blob.

  projection_snapshots   — delta audit trail. One row per (snapshot_date, sport).
                           Stores only changed projections as JSONB to prevent bloat.

Alters one existing table:

  predictions            — adds pricing_engine column (K-14 spec).
                           Values: 'markov' | 'gaussian' | NULL.

=============================================================================
WHAT THIS MIGRATION DOES NOT TOUCH
=============================================================================

  - No existing rows modified
  - No existing columns modified or removed
  - No CBB model files
  - All CREATE statements use IF NOT EXISTS
  - ALTER TABLE uses ADD COLUMN IF NOT EXISTS
  - Completely safe to run while Railway is live

=============================================================================
DOWNGRADE
=============================================================================

  python scripts/migrate_v8_post_draft.py --downgrade

  Drops the two new tables and removes the pricing_engine column.
  Does NOT affect any existing data in other tables.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from sqlalchemy import create_engine, inspect, text

# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

UPGRADE_SQL = [
    # ── Table 1: player_daily_metrics ───────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS player_daily_metrics (
        id              SERIAL PRIMARY KEY,
        player_id       VARCHAR(50)  NOT NULL,
        player_name     VARCHAR(100) NOT NULL,
        metric_date     DATE         NOT NULL,
        sport           VARCHAR(10)  NOT NULL
                            CHECK (sport IN ('mlb', 'cbb')),

        -- Core value metrics (sparse — NULL when not computed)
        vorp_7d         FLOAT,
        vorp_30d        FLOAT,
        z_score_total   FLOAT,
        z_score_recent  FLOAT,

        -- Statcast 2.0 (MLB only — always NULL for CBB rows)
        blast_pct       FLOAT,
        bat_speed       FLOAT,
        squared_up_pct  FLOAT,
        swing_length    FLOAT,
        stuff_plus      FLOAT,
        plv             FLOAT,

        -- Flexible rolling windows stored as JSONB
        -- Example: {"7d": {"avg": 0.310, "ops": 0.890}, "30d": {...}}
        rolling_window  JSONB        NOT NULL DEFAULT '{}',

        -- Metadata
        data_source     VARCHAR(50),
        fetched_at      TIMESTAMP    NOT NULL DEFAULT NOW(),

        UNIQUE (player_id, metric_date, sport)
    )
    """,

    # Indexes
    """
    CREATE INDEX IF NOT EXISTS idx_pdm_player_date
        ON player_daily_metrics (player_id, metric_date DESC)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_pdm_sport_date
        ON player_daily_metrics (sport, metric_date DESC)
        WHERE sport = 'mlb'
    """,

    # ── Table 2: projection_snapshots ───────────────────────────────────────
    """
    CREATE TABLE IF NOT EXISTS projection_snapshots (
        id                  SERIAL PRIMARY KEY,
        snapshot_date       DATE        NOT NULL,
        sport               VARCHAR(10) NOT NULL
                                CHECK (sport IN ('mlb', 'cbb')),

        -- Delta-compressed: only changed projections stored.
        -- Shape: {player_id: {"old": {...}, "new": {...}, "delta_reason": "..."}}
        player_changes      JSONB       NOT NULL DEFAULT '{}',

        total_players       INTEGER,
        significant_changes INTEGER,     -- rows where |delta| > threshold
        created_at          TIMESTAMP   NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_ps_date_sport
        ON projection_snapshots (snapshot_date DESC, sport)
    """,

    # ── K-14: pricing_engine column on predictions ──────────────────────────
    """
    ALTER TABLE predictions
        ADD COLUMN IF NOT EXISTS pricing_engine VARCHAR(20)
            CHECK (pricing_engine IN ('markov', 'gaussian'))
    """,
]

DOWNGRADE_SQL = [
    "ALTER TABLE predictions DROP COLUMN IF EXISTS pricing_engine",
    "DROP INDEX IF EXISTS idx_ps_date_sport",
    "DROP TABLE IF EXISTS projection_snapshots",
    "DROP INDEX IF EXISTS idx_pdm_sport_date",
    "DROP INDEX IF EXISTS idx_pdm_player_date",
    "DROP TABLE IF EXISTS player_daily_metrics",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("[ERROR] DATABASE_URL environment variable not set.")
        sys.exit(1)
    return create_engine(db_url)


def _run_statements(engine, statements: list, label: str, dry_run: bool):
    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — {label} SQL (not executed):")
        print(f"{'='*60}")
        for sql in statements:
            print(sql.strip())
            print()
        return

    with engine.begin() as conn:
        for sql in statements:
            stmt = sql.strip()
            if not stmt:
                continue
            print(f"  Running: {stmt[:80].replace(chr(10), ' ')}...")
            conn.execute(text(stmt))

    print(f"\n[OK] {label} complete.")


def _verify(engine):
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    print("\nVerification:")

    for tbl in ("player_daily_metrics", "projection_snapshots"):
        status = "EXISTS" if tbl in tables else "MISSING"
        mark = "[OK]" if tbl in tables else "[!!]"
        print(f"  {mark} {tbl}: {status}")

    # Check pricing_engine column on predictions
    if "predictions" in tables:
        cols = {c["name"] for c in inspector.get_columns("predictions")}
        has_col = "pricing_engine" in cols
        mark = "[OK]" if has_col else "[!!]"
        print(f"  {mark} predictions.pricing_engine: {'EXISTS' if has_col else 'MISSING'}")
    else:
        print("  [!!] predictions table not found — is the DB initialised?")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def upgrade(dry_run: bool = False):
    engine = _get_engine()
    print("Running migrate_v8_post_draft UPGRADE...")
    _run_statements(engine, UPGRADE_SQL, "UPGRADE", dry_run)
    if not dry_run:
        _verify(engine)
        print("""
=============================================================================
Migration v8 complete.
=============================================================================

New tables:  player_daily_metrics, projection_snapshots
New column:  predictions.pricing_engine

NEXT STEPS:
  1. Run tests:
       pytest tests/test_schema_v8.py -v

  2. Add ORM models to backend/models.py if not already present
     (PlayerDailyMetric, ProjectionSnapshot classes)

  3. Restart backend if running locally:
       uvicorn backend.main:app --reload
=============================================================================
""")


def downgrade(dry_run: bool = False):
    engine = _get_engine()
    print("Running migrate_v8_post_draft DOWNGRADE...")
    _run_statements(engine, DOWNGRADE_SQL, "DOWNGRADE", dry_run)
    if not dry_run:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        for tbl in ("player_daily_metrics", "projection_snapshots"):
            mark = "[!!] STILL EXISTS" if tbl in tables else "[OK] Removed"
            print(f"  {mark}: {tbl}")
        print("\nDowngrade complete. Schema restored to v7 state.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EMAC-077 v8 migration: post-draft time-series schema"
    )
    parser.add_argument(
        "--downgrade",
        action="store_true",
        help="Roll back all v8 changes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print SQL statements without executing",
    )
    args = parser.parse_args()

    if args.downgrade:
        downgrade(dry_run=args.dry_run)
    else:
        upgrade(dry_run=args.dry_run)
