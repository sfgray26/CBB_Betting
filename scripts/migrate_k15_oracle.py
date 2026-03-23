"""
Migration K-15: Oracle Validation columns on predictions.

=============================================================================
HOW TO RUN
=============================================================================

  python scripts/migrate_k15_oracle.py              # run upgrade
  python scripts/migrate_k15_oracle.py --dry-run    # print SQL only
  python scripts/migrate_k15_oracle.py --downgrade  # roll back

Or on Railway:
  railway run python scripts/migrate_k15_oracle.py

=============================================================================
WHAT THIS MIGRATION DOES
=============================================================================

Adds two nullable columns to the existing predictions table:

  oracle_flag    BOOLEAN  — True when our model diverges from the KenPom +
                            BartTorvik consensus beyond the time-weighted z
                            threshold.  NULL for predictions run before K-15.

  oracle_result  JSONB    — Full OracleResult snapshot: oracle_spread,
                            model_spread, divergence_points, divergence_z,
                            threshold_z, flagged, sources.  NULL pre-K-15.

=============================================================================
WHAT THIS MIGRATION DOES NOT TOUCH
=============================================================================

  - No existing rows modified
  - No existing columns modified or removed
  - No CBB model files
  - ALTER TABLE uses ADD COLUMN IF NOT EXISTS — safe to re-run
  - Completely safe to run while Railway is live

=============================================================================
DOWNGRADE
=============================================================================

  python scripts/migrate_k15_oracle.py --downgrade

  Drops both columns.  Existing oracle data is lost — run with care.
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
    """
    ALTER TABLE predictions
        ADD COLUMN IF NOT EXISTS oracle_flag BOOLEAN
    """,
    """
    ALTER TABLE predictions
        ADD COLUMN IF NOT EXISTS oracle_result JSONB
    """,
    # Partial index: only index flagged rows — keeps the index small
    """
    CREATE INDEX IF NOT EXISTS idx_predictions_oracle_flag
        ON predictions (prediction_date DESC)
        WHERE oracle_flag = TRUE
    """,
]

DOWNGRADE_SQL = [
    "DROP INDEX IF EXISTS idx_predictions_oracle_flag",
    "ALTER TABLE predictions DROP COLUMN IF EXISTS oracle_result",
    "ALTER TABLE predictions DROP COLUMN IF EXISTS oracle_flag",
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

    if "predictions" not in tables:
        print("  [!!] predictions table not found — is the DB initialised?")
        return

    cols = {c["name"] for c in inspector.get_columns("predictions")}
    for col in ("oracle_flag", "oracle_result"):
        status = "EXISTS" if col in cols else "MISSING"
        mark = "[OK]" if col in cols else "[!!]"
        print(f"  {mark} predictions.{col}: {status}")


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def upgrade(dry_run: bool = False):
    engine = _get_engine()
    print("Running migrate_k15_oracle UPGRADE...")
    _run_statements(engine, UPGRADE_SQL, "UPGRADE", dry_run)
    if not dry_run:
        _verify(engine)
        print("""
=============================================================================
Migration K-15 complete.
=============================================================================

New columns: predictions.oracle_flag, predictions.oracle_result
New index:   idx_predictions_oracle_flag (partial — flagged rows only)

NEXT STEPS:
  1. Run tests:
       pytest tests/test_oracle_validator.py -v

  2. Restart backend if running locally:
       uvicorn backend.main:app --reload

  3. Flagged predictions are available at:
       GET /admin/oracle/flagged
=============================================================================
""")


def downgrade(dry_run: bool = False):
    engine = _get_engine()
    print("Running migrate_k15_oracle DOWNGRADE...")
    _run_statements(engine, DOWNGRADE_SQL, "DOWNGRADE", dry_run)
    if not dry_run:
        _verify(engine)
        print("\nDowngrade complete. oracle_flag and oracle_result columns removed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="K-15 Oracle Validation migration: oracle_flag + oracle_result columns"
    )
    parser.add_argument(
        "--downgrade",
        action="store_true",
        help="Roll back K-15 columns",
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
