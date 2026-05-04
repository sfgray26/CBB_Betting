"""
backfill_scarcity_rank.py — Session O / O1
==========================================
One-shot UPDATE: set scarcity_rank for ALL position_eligibility rows where it is NULL.

Motivation (Kimi audit 2026-04-29):
  _sync_position_eligibility writes scarcity_rank on INSERT/UPDATE for actively-rostered
  players returned by Yahoo get_league_rosters(). But the full player pool (2,389 rows)
  was bulk-seeded from another source; most rows pre-date Session H and have NULL.
  Only 8–17% of rows per position currently have scarcity_rank filled.

Fix:
  Single CASE-WHEN UPDATE using the static POSITION_SCARCITY dict (same values as the
  daily sync uses). Positions not in the map get rank 99 (Util default).
  Commits immediately — no lock contention, no timeout risk.

Usage:
  # Local dev (uses DATABASE_URL from .env)
  venv/Scripts/python scripts/backfill_scarcity_rank.py

  # Production via Railway
  railway run python scripts/backfill_scarcity_rank.py

Expected output:
  Updated <N> rows with scarcity_rank (was NULL)
  Verification: C=1, SS=2, 2B=3, 3B=4, CF=5, SP=6, RP=7, LF=8, RF=9, 1B=10, DH=11, OF=12, else=99
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: load DATABASE_URL from .env when running locally
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required on Railway (env vars already injected)

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Add it to .env or Railway env vars.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# Static scarcity map (matches POSITION_SCARCITY in daily_ingestion.py)
# ---------------------------------------------------------------------------
POSITION_SCARCITY: dict[str, int] = {
    "C":   1,
    "SS":  2,
    "2B":  3,
    "3B":  4,
    "CF":  5,
    "SP":  6,
    "RP":  7,
    "LF":  8,
    "RF":  9,
    "1B":  10,
    "DH":  11,
    "OF":  12,
}

# Build a SQL CASE expression
case_branches = "\n        ".join(
    f"WHEN '{pos}' THEN {rank}" for pos, rank in POSITION_SCARCITY.items()
)

UPDATE_SQL = f"""
UPDATE position_eligibility
SET scarcity_rank = CASE primary_position
        {case_branches}
        ELSE 99
    END
WHERE scarcity_rank IS NULL
"""

VERIFY_SQL = """
SELECT primary_position,
       COUNT(*) AS total,
       COUNT(scarcity_rank) AS has_rank,
       MIN(scarcity_rank) AS min_rank
FROM position_eligibility
GROUP BY primary_position
ORDER BY MIN(scarcity_rank) NULLS LAST
"""

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
try:
    import psycopg2
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = False
cur = conn.cursor()

try:
    cur.execute(UPDATE_SQL)
    updated = cur.rowcount
    conn.commit()
    print(f"Updated {updated} rows with scarcity_rank (was NULL)")

    # Verification pass
    cur.execute(VERIFY_SQL)
    rows = cur.fetchall()
    print("\nPost-backfill coverage:")
    print(f"{'Position':<10} {'Total':>6} {'HasRank':>8} {'Pct':>6} {'MinRank':>8}")
    print("-" * 45)
    for pos, total, has_rank, min_rank in rows:
        pct = f"{100 * has_rank / total:.0f}%" if total else "—"
        print(f"{pos or '(null)':<10} {total:>6} {has_rank:>8} {pct:>6} {min_rank or '—':>8}")

except Exception as exc:
    conn.rollback()
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)
finally:
    cur.close()
    conn.close()
