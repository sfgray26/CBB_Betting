# -*- coding: utf-8 -*-
"""
Repair player_id_mapping corruption: bdl_id columns holding mlbam_id values.

Corruption pattern (documented in backend/docs/player_id_mapping_corruption_analysis.md):
  Row A (corrupted): yahoo_key set, bdl_id == mlbam_id (an MLBAM value, not a BDL id)
  Row B (clean):     yahoo_key NULL, real bdl_id, same player

Repair per player (single transaction each):
  1. Repoint FK rows (player_opportunity, player_market_signals, matchup_context)
     from the corrupted bdl value to the real bdl_id; duplicate-key collisions on
     (bdl_player_id, date) keep the real-id row and drop the corrupted-id row.
  2. Delete the corrupted row, then move its yahoo_key/yahoo_id onto the clean row
     and set mlbam_id from the corrupted bdl value. source -> 'repair_merge'.

Also resolves rows that have a yahoo_key but NULL bdl_id via live BDL name search
(--resolve-nulls), and installs a CHECK constraint blocking bdl_id == mlbam_id
(--add-constraint) so this class of corruption can never be written again.

Usage (production container -- DB hostnames only resolve there):
    railway ssh "python backend/scripts/repair_player_id_mapping.py"                 # dry run
    railway ssh "python backend/scripts/repair_player_id_mapping.py --apply"
    railway ssh "python backend/scripts/repair_player_id_mapping.py --apply --resolve-nulls"
    railway ssh "python backend/scripts/repair_player_id_mapping.py --apply --add-constraint"
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sqlalchemy import text

from backend.models import SessionLocal

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("repair_player_id_mapping")

# (table, date column) pairs holding the unique key (bdl_player_id, <date>)
FK_TABLES = [
    ("player_opportunity", "as_of_date"),
    ("player_market_signals", "as_of_date"),
    ("matchup_context", "game_date"),
]

CHECK_CONSTRAINT_NAME = "ck_pim_bdl_not_mlbam"


def find_corrupted_pairs(db) -> list[dict]:
    """Corrupted yahoo rows paired with their clean same-name sibling."""
    rows = db.execute(text("""
        SELECT c.id            AS bad_id,
               c.yahoo_key     AS yahoo_key,
               c.yahoo_id      AS yahoo_id,
               c.bdl_id        AS mlbam_value,
               c.full_name     AS full_name,
               s.id            AS good_id,
               s.bdl_id        AS good_bdl_id,
               s.mlbam_id      AS good_mlbam_id,
               s.yahoo_key     AS good_yahoo_key
        FROM player_id_mapping c
        JOIN player_id_mapping s
          ON s.normalized_name = c.normalized_name
         AND s.id <> c.id
         AND s.bdl_id IS NOT NULL
         AND (s.mlbam_id IS NULL OR s.bdl_id <> s.mlbam_id)
        WHERE c.yahoo_key IS NOT NULL
          AND c.bdl_id IS NOT NULL
          AND c.bdl_id = c.mlbam_id
        ORDER BY c.id
    """)).mappings().all()

    by_bad: dict[int, list[dict]] = {}
    for r in rows:
        by_bad.setdefault(r["bad_id"], []).append(dict(r))

    pairs, skipped = [], 0
    for bad_id, sibs in by_bad.items():
        # Prefer the sibling whose mlbam_id equals the value stuck in c.bdl_id
        exact = [s for s in sibs if s["good_mlbam_id"] == s["mlbam_value"]]
        pick = exact[0] if len(exact) >= 1 else (sibs[0] if len(sibs) == 1 else None)
        if pick is None:
            logger.warning("SKIP bad_id=%d (%s): %d ambiguous siblings",
                           bad_id, sibs[0]["full_name"], len(sibs))
            skipped += 1
            continue
        if pick["good_yahoo_key"] is not None:
            logger.warning("SKIP bad_id=%d (%s): sibling %d already has yahoo_key=%s",
                           bad_id, pick["full_name"], pick["good_id"], pick["good_yahoo_key"])
            skipped += 1
            continue
        pairs.append(pick)
    logger.info("Corrupted pairs resolvable: %d (skipped ambiguous: %d)", len(pairs), skipped)
    return pairs


def repoint_fk_rows(db, bad_bdl: int, good_bdl: int) -> None:
    for table, date_col in FK_TABLES:
        db.execute(text(f"""
            UPDATE {table} t
               SET bdl_player_id = :good
             WHERE t.bdl_player_id = :bad
               AND NOT EXISTS (
                   SELECT 1 FROM {table} t2
                    WHERE t2.bdl_player_id = :good
                      AND t2.{date_col} = t.{date_col}
               )
        """), {"good": good_bdl, "bad": bad_bdl})
        db.execute(text(f"DELETE FROM {table} WHERE bdl_player_id = :bad"), {"bad": bad_bdl})


def merge_pair(db, pair: dict) -> None:
    """Move identity from the corrupted row onto the clean row. Caller commits."""
    repoint_fk_rows(db, pair["mlbam_value"], pair["good_bdl_id"])
    db.execute(text("DELETE FROM player_id_mapping WHERE id = :bad"), {"bad": pair["bad_id"]})
    db.execute(text("""
        UPDATE player_id_mapping
           SET yahoo_key = :ykey,
               yahoo_id  = :yid,
               mlbam_id  = COALESCE(mlbam_id, :mlbam),
               source    = 'repair_merge',
               updated_at = NOW()
         WHERE id = :good
    """), {
        "ykey": pair["yahoo_key"],
        "yid": pair["yahoo_id"],
        "mlbam": pair["mlbam_value"],
        "good": pair["good_id"],
    })


def repair_corrupted(db, apply: bool) -> tuple[int, int]:
    pairs = find_corrupted_pairs(db)
    fixed = failed = 0
    for pair in pairs:
        if not apply:
            logger.info("[dry-run] would merge %s: bad_id=%d (bdl=%d) -> good_id=%d (bdl=%d)",
                        pair["full_name"], pair["bad_id"], pair["mlbam_value"],
                        pair["good_id"], pair["good_bdl_id"])
            fixed += 1
            continue
        try:
            merge_pair(db, pair)
            db.commit()
            fixed += 1
            logger.info("MERGED %s: yahoo_key=%s now -> bdl_id=%d (mlbam=%d)",
                        pair["full_name"], pair["yahoo_key"],
                        pair["good_bdl_id"], pair["mlbam_value"])
        except Exception as exc:
            db.rollback()
            failed += 1
            logger.error("FAILED merge bad_id=%d (%s): %s", pair["bad_id"], pair["full_name"], exc)
    return fixed, failed


def resolve_null_bdl_ids(db, apply: bool) -> tuple[int, int]:
    """Resolve yahoo_key rows with NULL bdl_id via live BDL name search."""
    from backend.services.bdl_mcp_client import get_bdl_resolver

    resolver = get_bdl_resolver()
    rows = db.execute(text("""
        SELECT id, full_name, mlbam_id
        FROM player_id_mapping
        WHERE yahoo_key IS NOT NULL AND bdl_id IS NULL
        ORDER BY id
    """)).mappings().all()
    logger.info("NULL bdl_id rows with yahoo_key: %d", len(rows))

    resolved = unresolved = 0
    for r in rows:
        player = resolver.get_player_by_name(r["full_name"])
        if player is None:
            unresolved += 1
            logger.info("UNRESOLVED %s (id=%d): no unambiguous BDL match", r["full_name"], r["id"])
            continue

        taken = db.execute(text(
            "SELECT id, yahoo_key FROM player_id_mapping WHERE bdl_id = :bdl"
        ), {"bdl": player.id}).mappings().first()

        if taken and taken["yahoo_key"] is not None:
            unresolved += 1
            logger.warning("CONFLICT %s: bdl_id=%d already mapped to yahoo_key=%s",
                           r["full_name"], player.id, taken["yahoo_key"])
            continue

        if not apply:
            logger.info("[dry-run] would resolve %s (id=%d) -> bdl_id=%d%s",
                        r["full_name"], r["id"], player.id,
                        f" (merge into keyless row {taken['id']})" if taken else "")
            resolved += 1
            continue

        try:
            if taken:
                # bdl_id already lives on a keyless clean row -> move this row's
                # yahoo identity onto it and delete this row.
                yident = db.execute(text(
                    "SELECT yahoo_key, yahoo_id, mlbam_id FROM player_id_mapping WHERE id = :id"
                ), {"id": r["id"]}).mappings().first()
                db.execute(text("DELETE FROM player_id_mapping WHERE id = :id"), {"id": r["id"]})
                db.execute(text("""
                    UPDATE player_id_mapping
                       SET yahoo_key = :ykey, yahoo_id = :yid,
                           mlbam_id = COALESCE(mlbam_id, :mlbam),
                           source = 'repair_merge', updated_at = NOW()
                     WHERE id = :good
                """), {"ykey": yident["yahoo_key"], "yid": yident["yahoo_id"],
                       "mlbam": yident["mlbam_id"], "good": taken["id"]})
                db.commit()
                resolved += 1
                logger.info("MERGED %s (id=%d) into keyless row %d (bdl_id=%d)",
                            r["full_name"], r["id"], taken["id"], player.id)
            else:
                db.execute(text("""
                    UPDATE player_id_mapping
                       SET bdl_id = :bdl, source = 'bdl_search',
                           resolution_confidence = 0.9, updated_at = NOW()
                     WHERE id = :id
                """), {"bdl": player.id, "id": r["id"]})
                db.commit()
                resolved += 1
                logger.info("RESOLVED %s (id=%d) -> bdl_id=%d", r["full_name"], r["id"], player.id)
        except Exception as exc:
            db.rollback()
            unresolved += 1
            logger.error("FAILED resolve for %s (id=%d): %s", r["full_name"], r["id"], exc)
    return resolved, unresolved


def add_check_constraint(db, apply: bool) -> None:
    """Block the corruption vector at the DB level: bdl_id must never equal mlbam_id."""
    exists = db.execute(text("""
        SELECT 1 FROM pg_constraint WHERE conname = :name
    """), {"name": CHECK_CONSTRAINT_NAME}).first()
    if exists:
        logger.info("Constraint %s already present", CHECK_CONSTRAINT_NAME)
        return
    remaining = db.execute(text("""
        SELECT COUNT(*) FROM player_id_mapping
        WHERE bdl_id IS NOT NULL AND mlbam_id IS NOT NULL AND bdl_id = mlbam_id
    """)).scalar()
    if remaining:
        logger.error("Cannot add constraint: %d rows still violate bdl_id = mlbam_id", remaining)
        return
    if not apply:
        logger.info("[dry-run] would add CHECK constraint %s", CHECK_CONSTRAINT_NAME)
        return
    db.execute(text(f"""
        ALTER TABLE player_id_mapping
        ADD CONSTRAINT {CHECK_CONSTRAINT_NAME}
        CHECK (bdl_id IS NULL OR mlbam_id IS NULL OR bdl_id <> mlbam_id)
    """))
    db.commit()
    logger.info("Added CHECK constraint %s", CHECK_CONSTRAINT_NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    parser.add_argument("--resolve-nulls", action="store_true",
                        help="also resolve yahoo_key rows with NULL bdl_id via BDL search")
    parser.add_argument("--add-constraint", action="store_true",
                        help="add CHECK constraint blocking bdl_id == mlbam_id")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        fixed, failed = repair_corrupted(db, args.apply)
        logger.info("Corrupted-pair repair: fixed=%d failed=%d (apply=%s)", fixed, failed, args.apply)
        if args.resolve_nulls:
            resolved, unresolved = resolve_null_bdl_ids(db, args.apply)
            logger.info("NULL bdl_id resolution: resolved=%d unresolved=%d", resolved, unresolved)
        if args.add_constraint:
            add_check_constraint(db, args.apply)
    finally:
        db.close()


if __name__ == "__main__":
    main()
