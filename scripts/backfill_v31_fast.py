"""
Fast V31 rolling-stats backfill using a single VALUES-based UPDATE per date.

Problem with the original backfill_v31_rolling.py:
  SQLAlchemy ORM flushes each dirty row as a separate UPDATE statement.
  At ~0.2s/statement × 700 rows/date × 29 dates ≈ 4000s total.
  When railway run times out mid-flush the uncommitted transaction holds row
  locks and blocks all subsequent attempts.

Fix:
  For each as_of_date, build one:
    UPDATE player_rolling_stats AS t
    SET w_runs = v.w_runs, ...
    FROM (VALUES (id1, r1, ...), (id2, r2, ...)) AS v(id, ...)
    WHERE t.id = v.id
  That is ONE network round-trip per date regardless of row count.
  Commit immediately after each date so no long-running transactions.

Usage:
  python scripts/backfill_v31_fast.py           # dry-run (default)
  python scripts/backfill_v31_fast.py --execute  # commit changes

Run on Railway:
  railway run python scripts/backfill_v31_fast.py --execute
"""
import argparse
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend.models import SessionLocal, MLBPlayerStats, PlayerRollingStats
from backend.services.rolling_window_engine import compute_all_rolling_windows

V31_COLS = ["w_runs", "w_tb", "w_qs", "w_caught_stealing", "w_net_stolen_bases"]
# Filter on w_runs only: it should be non-null for every player with stats data.
# Using OR across all 5 columns causes an infinite loop because pitcher-only stats
# (w_qs) remain legitimately NULL for batters and keep re-matching the filter.
_NULL_FILTER = PlayerRollingStats.w_runs.is_(None)


def _build_values_update(updates: list[dict]) -> str | None:
    """Return a VALUES-based UPDATE SQL string for the given list of row dicts.

    Each dict must have keys: id, w_runs, w_tb, w_qs, w_caught_stealing, w_net_stolen_bases.
    Returns None if updates is empty.
    """
    if not updates:
        return None

    def _fmt(v) -> str:
        if v is None:
            return "NULL"
        return repr(float(v))

    rows = ", ".join(
        f"({int(u['id'])}, {_fmt(u['w_runs'])}, {_fmt(u['w_tb'])}, "
        f"{_fmt(u['w_qs'])}, {_fmt(u['w_caught_stealing'])}, {_fmt(u['w_net_stolen_bases'])})"
        for u in updates
    )
    # COALESCE preserves any column that was already non-null (e.g. from a partial
    # previous run). Also means pitcher-only columns (w_qs) stay NULL for batters
    # and don't flip to NULL on a re-run.
    return f"""
UPDATE player_rolling_stats AS t
SET
    w_runs              = COALESCE(t.w_runs,            v.w_runs::double precision),
    w_tb                = COALESCE(t.w_tb,              v.w_tb::double precision),
    w_qs                = COALESCE(t.w_qs,              v.w_qs::double precision),
    w_caught_stealing   = COALESCE(t.w_caught_stealing, v.w_caught_stealing::double precision),
    w_net_stolen_bases  = COALESCE(t.w_net_stolen_bases,v.w_net_stolen_bases::double precision)
FROM (VALUES {rows})
  AS v(id, w_runs, w_tb, w_qs, w_caught_stealing, w_net_stolen_bases)
WHERE t.id = v.id::integer
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute", action="store_true",
        help="Commit changes; default is dry-run",
    )
    args = parser.parse_args()

    db = SessionLocal()
    try:
        # Set a per-statement timeout so a single hung query aborts cleanly
        # instead of holding locks until the connection dies.
        db.execute(text("SET statement_timeout = '90s'"))

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
            print("No NULL V31 columns found in player_rolling_stats — nothing to do.")
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
                print(f"  [{idx+1}/{len(target_dates)}] {as_of_date}: no stat rows — skip")
                continue

            results = compute_all_rolling_windows(
                stat_rows,
                as_of_date=as_of_date,
                window_sizes=[7, 14, 30],
            )
            result_map = {(r.bdl_player_id, r.window_days): r for r in results}

            null_rows = (
                db.query(PlayerRollingStats.id,
                         PlayerRollingStats.bdl_player_id,
                         PlayerRollingStats.window_days)
                .filter(
                    PlayerRollingStats.as_of_date == as_of_date,
                    PlayerRollingStats.w_runs.is_(None),
                )
                .all()
            )

            updates = []
            for row in null_rows:
                computed = result_map.get((row.bdl_player_id, row.window_days))
                if computed is None:
                    continue
                updates.append({
                    "id": row.id,
                    "w_runs": getattr(computed, "w_runs", None),
                    "w_tb": getattr(computed, "w_tb", None),
                    "w_qs": getattr(computed, "w_qs", None),
                    "w_caught_stealing": getattr(computed, "w_caught_stealing", None),
                    "w_net_stolen_bases": getattr(computed, "w_net_stolen_bases", None),
                })

            if not updates:
                print(f"  [{idx+1}/{len(target_dates)}] {as_of_date}: 0 computable rows — skip")
                continue

            sql = _build_values_update(updates)

            if args.execute:
                db.execute(text(sql))
                db.commit()  # Commit immediately after each date — never hold open transactions
                print(f"  [{idx+1}/{len(target_dates)}] {as_of_date}: committed {len(updates)} rows")
            else:
                print(f"  [{idx+1}/{len(target_dates)}] {as_of_date}: would update {len(updates)} rows (dry-run)")

            total_updated += len(updates)

        if args.execute:
            print(f"\nDone. Total rows updated: {total_updated}")
        else:
            print(f"\nDry-run complete: {total_updated} rows would be updated.")
            print("Re-run with --execute to commit.")

    except Exception as exc:
        db.rollback()
        import traceback
        traceback.print_exc()
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
