"""
Backfill V32 Z-score columns (z_r, z_h, z_tb, z_k_b, z_ops, z_k_p, z_qs).

Finds all player_scores rows where any of these columns are NULL, then recomputes
them using the same scoring engine used by the daily job (_compute_player_scores).

Run AFTER backfill_v31_rolling.py completes (z-scores depend on w_runs/w_tb/w_qs
being populated).

Usage:
  python scripts/backfill_v32_zscores.py          # dry-run (default)
  python scripts/backfill_v32_zscores.py --execute  # commit changes

Run on Railway:
  railway run python scripts/backfill_v32_zscores.py --execute
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import or_

from backend.models import SessionLocal, PlayerRollingStats, PlayerScore
from backend.services.scoring_engine import compute_league_zscores

V32_COLS = ["z_r", "z_h", "z_tb", "z_k_b", "z_ops", "z_k_p", "z_qs"]

_NULL_FILTER = or_(*[getattr(PlayerScore, col).is_(None) for col in V32_COLS])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Commit changes; default is dry-run")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # Find all distinct (as_of_date, window_days) combos with NULL V32 columns
        combos = (
            db.query(PlayerScore.as_of_date, PlayerScore.window_days)
            .filter(_NULL_FILTER)
            .distinct()
            .order_by(PlayerScore.as_of_date, PlayerScore.window_days)
            .all()
        )

        if not combos:
            print("No NULL V32 z-score columns found in player_scores -- nothing to do.")
            return

        print(f"Found {len(combos)} (as_of_date, window_days) combo(s) needing backfill.")
        if not args.execute:
            print("DRY-RUN mode: no changes will be committed.\n")

        total_updated = 0

        for idx, (as_of_date, window_days) in enumerate(combos):
            # Load player_rolling_stats for this date + window
            rolling_rows = (
                db.query(PlayerRollingStats)
                .filter(
                    PlayerRollingStats.as_of_date == as_of_date,
                    PlayerRollingStats.window_days == window_days,
                )
                .all()
            )

            if not rolling_rows:
                continue

            score_results = compute_league_zscores(rolling_rows, as_of_date, window_days)
            result_map = {r.bdl_player_id: r for r in score_results}

            # Load player_scores rows that need updating for this combo
            null_rows = (
                db.query(PlayerScore)
                .filter(
                    PlayerScore.as_of_date == as_of_date,
                    PlayerScore.window_days == window_days,
                    _NULL_FILTER,
                )
                .all()
            )

            combo_updated = 0
            for row in null_rows:
                computed = result_map.get(row.bdl_player_id)
                if computed is None:
                    continue
                changed = False
                for col in V32_COLS:
                    if getattr(row, col) is None:
                        new_val = getattr(computed, col)
                        if new_val is not None:
                            setattr(row, col, new_val)
                            changed = True
                if changed:
                    combo_updated += 1

            total_updated += combo_updated

            if total_updated > 0 and total_updated % 500 < combo_updated:
                print(
                    f"  Progress: {total_updated} rows updated so far"
                    f" (as_of_date={as_of_date}, window={window_days}d)"
                )

            if args.execute and combo_updated > 0 and (idx + 1) % 30 == 0:
                db.commit()
                print(f"  Intermediate commit at combo index {idx+1}")

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
