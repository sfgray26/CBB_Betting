"""
dedup_mlb_player_stats.py — One-time cleanup for duplicate (bdl_player_id, game_date) rows.

Background
----------
The mlb_player_stats table has a UNIQUE constraint on (bdl_player_id, game_id), which
prevents the SAME game being inserted twice. However on 4 dates in April 2026, the BDL
API returned different game_ids for what appears to be the same game, producing 78
duplicate (player, date) pairs (156 total extra rows).

The rolling_window_engine sums ALL rows without deduplication — so affected players get
inflated AB, hits, and RBI in their rolling stats.

This script:
  1. Dry-run by default: prints the count of rows that WOULD be deleted.
  2. With --execute: deletes all but the MAX(id) row per (bdl_player_id, game_date).
  3. Prints a summary + 5 sample rows before and after.

DO NOT add a UNIQUE constraint on (bdl_player_id, game_date) — it would break legitimate
doubleheaders where the same player has two distinct game-stat rows in one day.

Usage
-----
    # Dry run (safe)
    venv/Scripts/python scripts/dedup_mlb_player_stats.py

    # Execute the cleanup
    venv/Scripts/python scripts/dedup_mlb_player_stats.py --execute
"""

import argparse
import sys
import os

# Allow running from repo root without installing the package
if __name__ == "__main__":
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _count_duplicates(db) -> int:
    from sqlalchemy import text
    result = db.execute(text("""
        SELECT COUNT(*) FROM mlb_player_stats
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM mlb_player_stats
            GROUP BY bdl_player_id, game_date
        )
    """)).scalar()
    return result or 0


def _sample_duplicates(db, limit: int = 5) -> list:
    from sqlalchemy import text
    rows = db.execute(text("""
        SELECT mps.id, mps.bdl_player_id, mps.game_date, mps.game_id,
               mps.ab, mps.hits, mps.home_runs, mps.rbi, mps.strikeouts_pit
        FROM mlb_player_stats mps
        WHERE mps.id NOT IN (
            SELECT MAX(id)
            FROM mlb_player_stats
            GROUP BY bdl_player_id, game_date
        )
        ORDER BY mps.game_date DESC, mps.bdl_player_id
        LIMIT :lim
    """), {"lim": limit}).fetchall()
    return rows


def _count_corrupt_projections(db) -> int:
    from sqlalchemy import text
    result = db.execute(text("""
        SELECT COUNT(*) FROM player_projections
        WHERE player_name ~ '^[0-9]+$' AND team IS NULL
    """)).scalar()
    return result or 0


def _sample_corrupt_projections(db, limit: int = 10) -> list:
    from sqlalchemy import text
    rows = db.execute(text("""
        SELECT player_id, player_name, team
        FROM player_projections
        WHERE player_name ~ '^[0-9]+$' AND team IS NULL
        LIMIT :lim
    """), {"lim": limit}).fetchall()
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Dedup mlb_player_stats + clean corrupt player_projections rows."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the deletes (default: dry run only).",
    )
    args = parser.parse_args()

    from backend.models import SessionLocal

    db = SessionLocal()
    try:
        # ----------------------------------------------------------------
        # Part 1: mlb_player_stats duplicates
        # ----------------------------------------------------------------
        dup_count = _count_duplicates(db)
        print(f"\n=== mlb_player_stats duplicates ===")
        print(f"Rows that would be deleted: {dup_count}")

        if dup_count > 0:
            samples = _sample_duplicates(db, limit=5)
            print(f"\nSample rows to be deleted (id | player | date | game_id | ab | hits | hr | rbi | k_pit):")
            for r in samples:
                print(
                    f"  id={r.id:6d}  bdl_id={r.bdl_player_id:5d}  "
                    f"date={r.game_date}  game_id={r.game_id}  "
                    f"ab={r.ab}  hits={r.hits}  hr={r.home_runs}  "
                    f"rbi={r.rbi}  k={r.strikeouts_pit}"
                )

        # ----------------------------------------------------------------
        # Part 2: corrupt player_projections rows
        # ----------------------------------------------------------------
        corrupt_count = _count_corrupt_projections(db)
        print(f"\n=== player_projections corrupt rows ===")
        print(f"Rows where player_name is numeric and team IS NULL: {corrupt_count}")

        if corrupt_count > 0:
            corrupt_rows = _sample_corrupt_projections(db, limit=10)
            print(f"\nCorrupt rows (player_id | player_name | team):")
            for r in corrupt_rows:
                print(f"  player_id={r.player_id}  player_name={r.player_name}  team={r.team}")

        # ----------------------------------------------------------------
        # Execute or abort
        # ----------------------------------------------------------------
        if not args.execute:
            print(f"\nDRY RUN complete. No changes made.")
            print(f"  mlb_player_stats:   would delete {dup_count} rows")
            print(f"  player_projections: would delete {corrupt_count} rows")
            print(f"\nRe-run with --execute to apply.")
            return

        from sqlalchemy import text

        # Delete stat duplicates
        if dup_count > 0:
            result = db.execute(text("""
                DELETE FROM mlb_player_stats
                WHERE id NOT IN (
                    SELECT MAX(id)
                    FROM mlb_player_stats
                    GROUP BY bdl_player_id, game_date
                )
            """))
            deleted_stats = result.rowcount
            print(f"\nEXECUTED: deleted {deleted_stats} duplicate stat rows")
        else:
            deleted_stats = 0
            print(f"\nNo duplicate stat rows to delete.")

        # Delete corrupt projection rows
        if corrupt_count > 0:
            result = db.execute(text("""
                DELETE FROM player_projections
                WHERE player_name ~ '^[0-9]+$' AND team IS NULL
            """))
            deleted_proj = result.rowcount
            print(f"EXECUTED: deleted {deleted_proj} corrupt projection rows")
        else:
            deleted_proj = 0
            print(f"No corrupt projection rows to delete.")

        db.commit()

        # Post-delete counts
        remaining_dups = _count_duplicates(db)
        remaining_corrupt = _count_corrupt_projections(db)
        print(f"\nPost-cleanup verification:")
        print(f"  mlb_player_stats duplicates remaining:      {remaining_dups}")
        print(f"  player_projections corrupt rows remaining:  {remaining_corrupt}")
        if remaining_dups == 0 and remaining_corrupt == 0:
            print(f"\nCleanup complete. Both tables are clean.")
        else:
            print(f"\nWARNING: some rows could not be cleaned — investigate manually.")

    except Exception as exc:
        db.rollback()
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
