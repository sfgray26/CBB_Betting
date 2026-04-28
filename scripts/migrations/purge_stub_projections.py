"""
Purge zombie rows from player_projections.

Zombie: a row that matches BOTH the batter default pattern AND the pitcher
default pattern simultaneously, AND has team IS NULL (retired/inactive player
with Steamer-assigned placeholder values).

Audit baseline (2026-04-28): 24 zombie rows / 637 total. All have team=NULL
and yahoo_key=NULL in player_id_mapping — safe to delete.

Preservation guarantee: any row with team IS NOT NULL is untouched, even if
its stats happen to match the defaults.

Usage:
    Dry-run (default — prints count, no deletes):
        railway run python scripts/migrations/purge_stub_projections.py

    Execute (actual deletion):
        railway run python scripts/migrations/purge_stub_projections.py --execute
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set")
    sys.exit(1)

EXECUTE = "--execute" in sys.argv

ZOMBIE_WHERE = """
    hr = 15 AND r = 65 AND rbi = 65 AND sb = 5
    AND ABS(avg - 0.250) < 0.001 AND ABS(ops - 0.720) < 0.001
    AND ABS(era - 4.0) < 0.001 AND ABS(whip - 1.3) < 0.001
    AND w = 0 AND qs = 0 AND k_pit = 0
    AND team IS NULL
"""

engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    count = conn.execute(text(f"SELECT COUNT(*) FROM player_projections WHERE {ZOMBIE_WHERE}")).scalar()

    print(f"Zombie rows to delete: {count}")

    if not EXECUTE:
        print("Dry-run mode. Pass --execute to delete.")
        sys.exit(0)

    if count == 0:
        print("Nothing to delete.")
        sys.exit(0)

    # Sample before deleting for the audit trail
    samples = conn.execute(text(f"""
        SELECT id, player_name, player_id, prior_source
        FROM player_projections
        WHERE {ZOMBIE_WHERE}
        ORDER BY id
        LIMIT 10
    """)).fetchall()
    print("Sample rows being deleted:")
    for r in samples:
        print(f"  id={r[0]}  {r[1]:<30}  player_id={r[2]}  prior_source={r[3]}")

    deleted = conn.execute(text(f"DELETE FROM player_projections WHERE {ZOMBIE_WHERE}")).rowcount
    print(f"Deleted {deleted} rows.")
    print("Next step: POST /api/admin/data-quality/backfill-cat-scores?force=true")
