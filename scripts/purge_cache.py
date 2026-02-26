"""
purge_cache.py — Wipe DB-cached data so the next nightly run re-fetches fresh.

Targets
-------
  team_profiles   BartTorvik / KenPom four-factor stats cached from scrapers.
                  Cleared first; always included.
  data_fetches    Scraper health-tracking logs (optional via --all flag).

In-memory caches (TeamProfileCache, InjuryService) live in the server process
and cannot be cleared here.  Restart the FastAPI server after running this
script to guarantee a fully clean state.

Usage
-----
  python scripts/purge_cache.py              # dry-run (shows counts, no delete)
  python scripts/purge_cache.py --execute    # actually delete rows
  python scripts/purge_cache.py --execute --all   # also wipe data_fetches log
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root (one level up from scripts/) is on sys.path so that
# `from backend.xxx import ...` resolves correctly when the script is run
# directly (e.g.  python scripts/purge_cache.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge DB-cached data for CBB Edge Analyzer."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete rows.  Without this flag the script runs dry.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Also wipe the data_fetches scraper-health log table.",
    )
    args = parser.parse_args()

    dry_run = not args.execute

    from backend.models import SessionLocal, TeamProfile, DataFetch

    db = SessionLocal()
    try:
        label = "[DRY RUN] " if dry_run else ""

        # ----------------------------------------------------------------
        # 1. team_profiles — always purged
        # ----------------------------------------------------------------
        profile_count: int = db.query(TeamProfile).count()
        print(
            f"{label}team_profiles: {profile_count} row(s) found"
            + (" — would delete" if dry_run else "")
        )
        if not dry_run and profile_count:
            db.query(TeamProfile).delete(synchronize_session=False)
            print(f"  Deleted {profile_count} TeamProfile row(s).")

        # ----------------------------------------------------------------
        # 2. data_fetches — only when --all is passed
        # ----------------------------------------------------------------
        fetch_count: int = db.query(DataFetch).count()
        if args.all:
            print(
                f"{label}data_fetches: {fetch_count} row(s) found"
                + (" — would delete" if dry_run else "")
            )
            if not dry_run and fetch_count:
                db.query(DataFetch).delete(synchronize_session=False)
                print(f"  Deleted {fetch_count} DataFetch row(s).")
        else:
            print(
                f"  data_fetches: {fetch_count} row(s) — skipped "
                "(pass --all to include)"
            )

        # ----------------------------------------------------------------
        # Commit or rollback
        # ----------------------------------------------------------------
        if dry_run:
            db.rollback()
            print("\nDry run complete — no rows were deleted.")
            print("Re-run with --execute to apply changes.")
        else:
            db.commit()
            print(
                f"\nCache purge complete at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC."
            )
            print(
                "REMINDER: restart the FastAPI server to also clear in-memory "
                "TeamProfileCache and InjuryService caches."
            )

    except Exception as exc:
        try:
            db.rollback()
        except Exception:
            pass  # Connection already broken; nothing to roll back
        # Print only the root cause (SQLAlchemy wraps errors, which can cause
        # the same message to appear twice in the full exception repr).
        root = exc.__cause__ or exc
        print(f"ERROR: {type(root).__name__}: {root}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
