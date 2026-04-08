#!/usr/bin/env python
"""
Backfill counting stats from MLB Stats API for existing mlb_player_stats rows.

BDL /mlb/v1/stats returns NULL for ab/h/r/doubles/triples/so/sb/cs.
This script patches those NULLs using statsapi.boxscore_data() for all
historical dates in the DB.

Usage:
    python scripts/backfill_statsapi_counting_stats.py
    python scripts/backfill_statsapi_counting_stats.py --dry-run    # preview only
    python scripts/backfill_statsapi_counting_stats.py --date 2026-04-07  # single date
"""

import argparse
import json
import logging
import os
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Backfill counting stats from MLB Stats API")
    p.add_argument("--dry-run", action="store_true", help="Preview matches without writing")
    p.add_argument("--date", type=str, help="Backfill single date (YYYY-MM-DD)")
    return p.parse_args()


def get_engine():
    url = os.environ.get("DATABASE_URL")
    if not url:
        from dotenv import load_dotenv
        load_dotenv()
        url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    return create_engine(url)


def _norm(name: str) -> str:
    """Strip diacriticals and lowercase for matching."""
    s = unicodedata.normalize("NFD", name)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").strip().lower()


def backfill(engine, dry_run: bool = False, single_date: str = None):
    import statsapi

    with engine.connect() as conn:
        # Find distinct dates with NULL ab
        if single_date:
            dates = [date.fromisoformat(single_date)]
        else:
            rows = conn.execute(text(
                "SELECT DISTINCT game_date FROM mlb_player_stats "
                "WHERE ab IS NULL ORDER BY game_date"
            )).fetchall()
            dates = [r[0] for r in rows]

        if not dates:
            print("No rows needing backfill.")
            return

        print(f"Dates to backfill: {[str(d) for d in dates]}")
        total_patched = 0

        for target_date in dates:
            # Get all rows needing fill for this date
            result = conn.execute(text(
                "SELECT id, bdl_player_id, raw_payload "
                "FROM mlb_player_stats WHERE game_date = :d AND ab IS NULL"
            ), {"d": target_date}).fetchall()

            if not result:
                continue

            # Build name lookup: normalized_name -> list of (id, bdl_player_id)
            name_lookup = {}
            for row in result:
                payload = row.raw_payload if isinstance(row.raw_payload, dict) else json.loads(row.raw_payload) if row.raw_payload else {}
                player_obj = payload.get("player", {})
                full_name = _norm(player_obj.get("full_name") or "")
                if full_name:
                    name_lookup.setdefault(full_name, []).append(row.id)

            # Get statsapi schedule for this date
            date_str = target_date.strftime("%m/%d/%Y")
            try:
                games = statsapi.schedule(sportId=1, date=date_str)
            except Exception as exc:
                logger.warning("Failed to get schedule for %s: %s", date_str, exc)
                continue

            date_patched = 0

            for game in games:
                game_pk = game.get("game_id")
                if not game_pk:
                    continue
                status = game.get("status", "")
                if status not in ("Final", "Game Over", "Completed Early"):
                    logger.info("  Skipping game %s (status=%s)", game_pk, status)
                    continue

                try:
                    box = statsapi.boxscore_data(game_pk)
                except Exception as exc:
                    logger.warning("  boxscore_data(%s) failed: %s", game_pk, exc)
                    continue

                player_info = box.get("playerInfo", {})

                for side in ("away", "home"):
                    for batter in box.get(f"{side}Batters", []):
                        person_id = batter.get("personId")
                        if not person_id:
                            continue
                        info = player_info.get(f"ID{person_id}", {})
                        statsapi_name = _norm(info.get("fullName") or "")
                        if not statsapi_name:
                            continue

                        matching_ids = name_lookup.get(statsapi_name, [])
                        for row_id in matching_ids:
                            updates = _build_batter_updates(batter)
                            if updates and not dry_run:
                                set_clause = ", ".join(f"{k} = :{k}" for k in updates)
                                updates["row_id"] = row_id
                                conn.execute(text(
                                    f"UPDATE mlb_player_stats SET {set_clause} WHERE id = :row_id"
                                ), updates)
                                date_patched += 1
                            elif updates and dry_run:
                                print(f"  [DRY-RUN] Would patch row {row_id} ({statsapi_name}): {updates}")
                                date_patched += 1

                    for pitcher in box.get(f"{side}Pitchers", []):
                        person_id = pitcher.get("personId")
                        if not person_id:
                            continue
                        info = player_info.get(f"ID{person_id}", {})
                        statsapi_name = _norm(info.get("fullName") or "")
                        if not statsapi_name:
                            continue

                        matching_ids = name_lookup.get(statsapi_name, [])
                        for row_id in matching_ids:
                            updates = _build_pitcher_updates(pitcher)
                            if updates and not dry_run:
                                set_clause = ", ".join(f"{k} = :{k}" for k in updates)
                                updates["row_id"] = row_id
                                conn.execute(text(
                                    f"UPDATE mlb_player_stats SET {set_clause} WHERE id = :row_id"
                                ), updates)
                                date_patched += 1
                            elif updates and dry_run:
                                print(f"  [DRY-RUN] Would patch row {row_id} ({statsapi_name}): {updates}")
                                date_patched += 1

                time.sleep(0.15)  # Rate limit

            if not dry_run:
                conn.commit()

            print(f"  {target_date}: {date_patched} rows patched (from {len(result)} NULL-ab rows)")
            total_patched += date_patched

        print(f"\nTotal patched: {total_patched}")


def _build_batter_updates(batter: dict) -> dict:
    """Build dict of {db_column: value} for non-null counting stats from statsapi."""
    updates = {}
    field_map = {
        "ab": "ab",
        "runs": "r",
        "hits": "h",
        "doubles": "doubles",
        "triples": "triples",
        "home_runs": "hr",
        "rbi": "rbi",
        "walks": "bb",
        "strikeouts_bat": "k",
        "stolen_bases": "sb",
        "caught_stealing": "cs",
    }
    for db_col, box_key in field_map.items():
        raw = batter.get(box_key)
        if raw is not None:
            try:
                updates[db_col] = int(raw)
            except (ValueError, TypeError):
                pass
    return updates


def _build_pitcher_updates(pitcher: dict) -> dict:
    """Build dict of {db_column: value} for non-null pitching stats from statsapi."""
    updates = {}
    field_map = {
        "hits_allowed": "h",
        "earned_runs": "er",
        "walks_allowed": "bb",
        "strikeouts_pit": "k",
        "runs_allowed": "r",
    }
    for db_col, box_key in field_map.items():
        raw = pitcher.get(box_key)
        if raw is not None:
            try:
                updates[db_col] = int(raw)
            except (ValueError, TypeError):
                pass
    ip = pitcher.get("ip")
    if ip is not None:
        updates["innings_pitched"] = str(ip)
    return updates


if __name__ == "__main__":
    args = parse_args()
    eng = get_engine()
    backfill(eng, dry_run=args.dry_run, single_date=args.date)
