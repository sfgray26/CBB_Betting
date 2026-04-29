"""
Backfill V31 rolling stats columns (w_runs, w_tb, w_qs, w_caught_stealing, w_net_stolen_bases).

Finds all player_rolling_stats rows where any of these columns are NULL and source
mlb_player_stats data exists for the relevant window. Recomputes using the same
rolling window engine used by the daily job (_compute_rolling_windows).

Run AFTER V31 migration has been applied (ALTER TABLE ... ADD COLUMN).
Run BEFORE backfill_v32_zscores.py (z-score backfill depends on these values).

Usage:
  python scripts/backfill_v31_rolling.py          # dry-run (default)
  python scripts/backfill_v31_rolling.py --execute  # commit changes

Run on Railway:
  railway run python scripts/backfill_v31_rolling.py --execute
"""
import argparse
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import or_

from backend.models import SessionLocal, MLBPlayerStats, PlayerRollingStats
from backend.services.rolling_window_engine import compute_all_rolling_windows

V31_COLS = ["w_runs", "w_tb", "w_qs", "w_caught_stealing", "w_net_stolen_bases"]

_NULL_FILTER = or_(*[getattr(PlayerRollingStats, col).is_(None) for col in V31_COLS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Commit changes; default is dry-run")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # Find all distinct as_of_dates that have at least one NULL V31 column
        target_dates = [
            row.as_of_date
            for row in (
                db.query(PlayerRollingStats.as_of_date)
                .filter(_NULL_FILTER)
                .distinct()
                .order_by(PlayerRollingStats.as_of_date)
                .all()
            )
        ]

        if not target_dates:
            print("No NULL V31 columns found in player_rolling_stats -- nothing to do.")
            return

        print(f"Found {len(target_dates)} as_of_date(s) needing backfill.")
        if not args.execute:
            print("DRY-RUN mode: no changes will be committed.\n")

        total_updated = 0

        for idx, as_of_date in enumerate(target_dates):
            lookback_start = as_of_date - timedelta(days=30)

            stat_rows = (
                db.query(MLBPlayerStats)
                .filter(
                    MLBPlayerStats.game_date >= lookback_start,
                    MLBPlayerStats.game_date <= as_of_date,
                )
                .all()
            )

            if not stat_rows:
                continue

            results = compute_all_rolling_windows(
                stat_rows,
                as_of_date=as_of_date,
                window_sizes=[7, 14, 30],
            )
            result_map = {(r.bdl_player_id, r.window_days): r for r in results}

            null_rows = (
                db.query(PlayerRollingStats)
                .filter(
                    PlayerRollingStats.as_of_date == as_of_date,
                    _NULL_FILTER,
                )
                .all()
            )

            date_updated = 0
            for row in null_rows:
                computed = result_map.get((row.bdl_player_id, row.window_days))
                if computed is None:
                    continue
                changed = False
                for col in V31_COLS:
                    if getattr(row, col) is None:
                        setattr(row, col, getattr(computed, col))
                        changed = True
                if changed:
                    date_updated += 1

            total_updated += date_updated

            if total_updated > 0 and total_updated % 500 < date_updated:
                print(f"  Progress: {total_updated} rows updated so far (as_of_date={as_of_date})")

            if args.execute and date_updated > 0 and (idx + 1) % 30 == 0:
                db.commit()
                print(f"  Intermediate commit at date index {idx+1}")

        if args.execute:
            db.commit()
            print(f"\nDone. Committed {total_updated} rows updated.")
        else:
            db.rollback()
            print(f"\nDry-run complete: {total_updated} rows would be updated.")
            print("Re-run with --execute to commit.")

    except Exception as exc:
        db.rollback()
        import traceback
        print(f"ERROR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
