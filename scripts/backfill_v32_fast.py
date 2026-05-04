"""
Fast V32 z-score backfill using a single VALUES-based UPDATE per (as_of_date, window_days).

Problem with backfill_v32_zscores.py:
  ORM setattr accumulates dirty rows and flushes them as individual UPDATEs at
  commit time.  At ~0.2s/statement × 2400 rows × 87 combos ≈ 41,000s.

Fix:
  For each (as_of_date, window_days) combo, build one:
    UPDATE player_scores AS t
    SET z_r = COALESCE(t.z_r, v.z_r), ...
    FROM (VALUES ...) AS v(id, z_r, z_h, ...)
    WHERE t.id = v.id
  ONE network round-trip per combo regardless of row count.
  COALESCE preserves any column already populated from a prior partial run.
  Commits immediately after each combo — no long-running transactions.

Usage:
  python scripts/backfill_v32_fast.py           # dry-run (default)
  python scripts/backfill_v32_fast.py --execute  # commit changes

Run on Railway:
  railway run python scripts/backfill_v32_fast.py --execute
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from backend.models import SessionLocal, PlayerRollingStats, PlayerScore
from backend.services.scoring_engine import compute_league_zscores

V32_COLS = ["z_r", "z_h", "z_tb", "z_k_b", "z_ops", "z_k_p", "z_qs"]

# Primary filter: z_r IS NULL (a batter stat — always non-null for processed batter rows).
# Pitcher-only rows will have z_r IS NULL permanently (correct); after processing they
# will have z_k_p / z_qs non-null, which keeps them out of a tighter combined filter.
# We process ALL z_r-null rows per combo in one VALUES UPDATE; the COALESCE clause means
# pitcher columns (z_k_p, z_qs) that compute to NULL don't overwrite existing data.
_NULL_FILTER = PlayerScore.z_r.is_(None)


def _fmt(v) -> str:
    if v is None:
        return "NULL"
    return repr(float(v))


def _build_values_update(updates: list[dict]) -> str | None:
    if not updates:
        return None

    rows = ", ".join(
        f"({int(u['id'])}, {_fmt(u['z_r'])}, {_fmt(u['z_h'])}, {_fmt(u['z_tb'])}, "
        f"{_fmt(u['z_k_b'])}, {_fmt(u['z_ops'])}, {_fmt(u['z_k_p'])}, {_fmt(u['z_qs'])})"
        for u in updates
    )
    return f"""
UPDATE player_scores AS t
SET
    z_r   = COALESCE(t.z_r,   v.z_r::double precision),
    z_h   = COALESCE(t.z_h,   v.z_h::double precision),
    z_tb  = COALESCE(t.z_tb,  v.z_tb::double precision),
    z_k_b = COALESCE(t.z_k_b, v.z_k_b::double precision),
    z_ops = COALESCE(t.z_ops, v.z_ops::double precision),
    z_k_p = COALESCE(t.z_k_p, v.z_k_p::double precision),
    z_qs  = COALESCE(t.z_qs,  v.z_qs::double precision)
FROM (VALUES {rows})
  AS v(id, z_r, z_h, z_tb, z_k_b, z_ops, z_k_p, z_qs)
WHERE t.id = v.id::integer
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true",
                        help="Commit changes; default is dry-run")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        db.execute(text("SET statement_timeout = '90s'"))

        combos = (
            db.query(PlayerScore.as_of_date, PlayerScore.window_days)
            .filter(_NULL_FILTER)
            .distinct()
            .order_by(PlayerScore.as_of_date, PlayerScore.window_days)
            .all()
        )

        if not combos:
            print("No NULL V32 z-score columns found in player_scores — nothing to do.")
            return

        print(f"Found {len(combos)} (as_of_date, window_days) combo(s) needing backfill.")
        if not args.execute:
            print("DRY-RUN mode: no changes will be committed.\n")

        total_updated = 0

        for idx, (as_of_date, window_days) in enumerate(combos):
            rolling_rows = (
                db.query(PlayerRollingStats)
                .filter(
                    PlayerRollingStats.as_of_date == as_of_date,
                    PlayerRollingStats.window_days == window_days,
                )
                .all()
            )

            if not rolling_rows:
                print(f"  [{idx+1}/{len(combos)}] {as_of_date} w={window_days}: no rolling rows — skip")
                continue

            score_results = compute_league_zscores(rolling_rows, as_of_date, window_days)
            result_map = {r.bdl_player_id: r for r in score_results}

            null_rows = (
                db.query(PlayerScore.id, PlayerScore.bdl_player_id)
                .filter(
                    PlayerScore.as_of_date == as_of_date,
                    PlayerScore.window_days == window_days,
                    _NULL_FILTER,
                )
                .all()
            )

            updates = []
            for row in null_rows:
                computed = result_map.get(row.bdl_player_id)
                if computed is None:
                    continue
                updates.append({
                    "id": row.id,
                    "z_r":   getattr(computed, "z_r",   None),
                    "z_h":   getattr(computed, "z_h",   None),
                    "z_tb":  getattr(computed, "z_tb",  None),
                    "z_k_b": getattr(computed, "z_k_b", None),
                    "z_ops": getattr(computed, "z_ops", None),
                    "z_k_p": getattr(computed, "z_k_p", None),
                    "z_qs":  getattr(computed, "z_qs",  None),
                })

            if not updates:
                print(f"  [{idx+1}/{len(combos)}] {as_of_date} w={window_days}: 0 computable rows — skip")
                continue

            sql = _build_values_update(updates)

            if args.execute:
                db.execute(text(sql))
                db.commit()  # Commit per combo — never hold open transactions
                print(f"  [{idx+1}/{len(combos)}] {as_of_date} w={window_days}: committed {len(updates)} rows")
            else:
                print(f"  [{idx+1}/{len(combos)}] {as_of_date} w={window_days}: would update {len(updates)} rows (dry-run)")

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
