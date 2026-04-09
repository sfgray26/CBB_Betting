"""
Backfill Script: Position Eligibility — ALL MLB Players

Fetches position eligibility from Yahoo Fantasy API for:
  1. All rostered players (10 fantasy teams)
  2. All available players (free agents + waivers) — paginated

Creates ONE ROW PER PLAYER with boolean flags for ALL eligible positions.

Usage:
    python scripts/backfill_positions.py                 # Full run (rostered + all available)
    python scripts/backfill_positions.py --dry-run       # Preview counts only
    python scripts/backfill_positions.py --rostered-only  # Only league rosters (fast)
"""
import argparse
import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from sqlalchemy.dialects.postgresql import insert as pg_insert
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
from backend.models import SessionLocal, PositionEligibility

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Position scarcity priority — used to pick primary_position (most scarce first)
POSITION_PRIORITY = ["C", "SS", "2B", "CF", "3B", "RF", "LF", "1B", "OF", "DH", "SP", "RP", "Util"]

BATTER_POSITIONS = {"C", "1B", "2B", "3B", "SS", "LF", "CF", "RF", "OF", "DH"}

# Yahoo player status codes for the players endpoint
YAHOO_STATUS_AVAILABLE = "A"  # Free agents + waivers
YAHOO_STATUS_TAKEN = "T"  # Rostered (taken)

PAGE_SIZE = 25  # Yahoo hard limit per page


def build_position_flags(positions: list[str]) -> dict:
    """Build can_play_* boolean flags from a list of Yahoo position strings."""
    pos_set = {p.upper() for p in positions if p}
    return {
        "can_play_c": "C" in pos_set,
        "can_play_1b": "1B" in pos_set,
        "can_play_2b": "2B" in pos_set,
        "can_play_3b": "3B" in pos_set,
        "can_play_ss": "SS" in pos_set,
        "can_play_lf": "LF" in pos_set,
        "can_play_cf": "CF" in pos_set,
        "can_play_rf": "RF" in pos_set,
        "can_play_of": "OF" in pos_set or bool(pos_set & {"LF", "CF", "RF"}),
        "can_play_dh": "DH" in pos_set,
        "can_play_util": "UTIL" in pos_set or "Util" in {p for p in positions if p},
        "can_play_sp": "SP" in pos_set,
        "can_play_rp": "RP" in pos_set,
    }


def determine_primary_position(positions: list[str]) -> str:
    """Pick primary position using scarcity priority (most scarce wins)."""
    pos_upper = [p.upper() for p in positions if p]
    for priority in POSITION_PRIORITY:
        if priority.upper() in pos_upper:
            return priority
    return positions[0] if positions else "DH"


def determine_player_type(positions: list[str]) -> str:
    """Classify as batter, pitcher, or two_way."""
    pos_set = {p.upper() for p in positions if p}
    has_pitcher = bool(pos_set & {"SP", "RP", "P"})
    has_batter = bool(pos_set & BATTER_POSITIONS)
    if has_pitcher and has_batter:
        return "two_way"
    elif has_pitcher:
        return "pitcher"
    return "batter"


def _fetch_all_available_players(yahoo: YahooFantasyClient) -> list[dict]:
    """
    Paginate through ALL available players (status=A) in the league.

    Yahoo returns 25 players per page sorted by % rostered (sort=AR).
    We paginate until we get an empty page.
    Skips stats enrichment for speed — we only need position data.
    """
    all_players = []
    start = 0
    empty_pages = 0

    logger.info("Fetching all available (FA + waivers) players...")

    while True:
        try:
            # Use the Yahoo API directly to skip stats enrichment
            params = {"status": "A", "start": start, "count": PAGE_SIZE, "sort": "AR"}
            data = yahoo._get(f"league/{yahoo.league_key}/players", params=params)
            players_raw = yahoo._league_section(data, 1).get("players", {})
            batch = yahoo._parse_players_block(players_raw)
        except Exception as exc:
            logger.warning("Page at start=%d failed: %s — stopping pagination", start, exc)
            break

        if not batch:
            empty_pages += 1
            if empty_pages >= 2:  # Two consecutive empty pages = done
                break
            start += PAGE_SIZE
            continue

        empty_pages = 0
        all_players.extend(batch)
        start += PAGE_SIZE

        if len(all_players) % 100 == 0 or len(batch) < PAGE_SIZE:
            logger.info("  ... fetched %d available players so far (page start=%d)",
                        len(all_players), start)

        # Respect Yahoo rate limits — small delay between pages
        time.sleep(0.3)

        # Stop if Yahoo returned fewer than PAGE_SIZE (last page)
        if len(batch) < PAGE_SIZE:
            break

    logger.info("Total available players fetched: %d", len(all_players))
    return all_players


def _fetch_taken_players(yahoo: YahooFantasyClient) -> list[dict]:
    """
    Paginate through ALL rostered (taken) players via status=T.

    This is an alternative to get_league_rosters that goes through the
    same players endpoint, ensuring consistent data shape.
    """
    all_players = []
    start = 0
    empty_pages = 0

    logger.info("Fetching all rostered (taken) players...")

    while True:
        try:
            params = {"status": "T", "start": start, "count": PAGE_SIZE, "sort": "AR"}
            data = yahoo._get(f"league/{yahoo.league_key}/players", params=params)
            players_raw = yahoo._league_section(data, 1).get("players", {})
            batch = yahoo._parse_players_block(players_raw)
        except Exception as exc:
            logger.warning("Taken page at start=%d failed: %s — stopping", start, exc)
            break

        if not batch:
            empty_pages += 1
            if empty_pages >= 2:
                break
            start += PAGE_SIZE
            continue

        empty_pages = 0
        all_players.extend(batch)
        start += PAGE_SIZE

        if len(all_players) % 100 == 0 or len(batch) < PAGE_SIZE:
            logger.info("  ... fetched %d taken players so far", len(all_players))

        time.sleep(0.3)

        if len(batch) < PAGE_SIZE:
            break

    logger.info("Total taken players fetched: %d", len(all_players))
    return all_players


def _upsert_players(db, players: list[dict], now: datetime, dry_run: bool = False) -> tuple[int, int]:
    """
    Upsert a list of player dicts into position_eligibility.

    Returns (processed, skipped).
    """
    processed = 0
    skipped = 0
    seen_keys = set()

    for player_data in players:
        player_key = player_data.get("player_key")
        if not player_key or player_key in seen_keys:
            skipped += 1
            continue
        seen_keys.add(player_key)

        name = player_data.get("name", "Unknown")
        positions = player_data.get("positions", [])

        if not positions:
            skipped += 1
            continue

        flags = build_position_flags(positions)
        primary = determine_primary_position(positions)
        ptype = determine_player_type(positions)
        meaningful = [p for p in positions if p.upper() not in ("UTIL",)]
        multi_count = len(meaningful)

        if dry_run:
            processed += 1
            continue

        stmt = pg_insert(PositionEligibility.__table__).values(
            yahoo_player_key=player_key,
            bdl_player_id=None,
            player_name=name,
            first_name="",
            last_name="",
            primary_position=primary,
            player_type=ptype,
            multi_eligibility_count=multi_count,
            fetched_at=now,
            updated_at=now,
            **flags,
        ).on_conflict_do_update(
            constraint="_pe_yahoo_uc",
            set_={
                "player_name": name,
                "primary_position": primary,
                "player_type": ptype,
                "multi_eligibility_count": multi_count,
                "updated_at": now,
                **flags,
            },
        )
        db.execute(stmt)
        processed += 1

        # Commit in batches to avoid huge transactions
        if processed % 100 == 0:
            db.commit()

    if not dry_run:
        db.commit()

    return processed, skipped


def backfill_position_eligibility(dry_run: bool = False, rostered_only: bool = False) -> dict:
    """
    Fetch position eligibility from Yahoo Fantasy API and upsert to DB.

    Combines:
      1. All rostered (taken) players — via status=T endpoint
      2. All available (FA + waivers) players — via status=A endpoint (paginated)

    ONE ROW PER PLAYER with ALL eligible positions as boolean flags.
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Position Eligibility Backfill — ALL MLB Players")
    logger.info("  Mode: %s", "DRY-RUN" if dry_run else "LIVE")
    logger.info("  Scope: %s", "rostered only" if rostered_only else "ALL players (rostered + available)")
    logger.info("=" * 60)

    try:
        yahoo = YahooFantasyClient()
        if not yahoo.league_key:
            logger.error("No league_key configured")
            return {"status": "failed", "error": "No league_key"}

        # Phase 1: Fetch rostered (taken) players
        logger.info("")
        logger.info("--- PHASE 1: Rostered Players (status=T) ---")
        taken_players = _fetch_taken_players(yahoo)

        # Phase 2: Fetch all available players (if not rostered-only)
        available_players = []
        if not rostered_only:
            logger.info("")
            logger.info("--- PHASE 2: Available Players (status=A) ---")
            available_players = _fetch_all_available_players(yahoo)

        # Merge — taken first (they get priority on name/position data)
        all_players = taken_players + available_players
        logger.info("")
        logger.info("Total players before dedup: %d (taken=%d, available=%d)",
                     len(all_players), len(taken_players), len(available_players))

        # Deduplicate by player_key (taken players take priority)
        deduped = {}
        for p in all_players:
            pk = p.get("player_key")
            if pk and pk not in deduped:
                deduped[pk] = p
        unique_players = list(deduped.values())
        logger.info("Unique players after dedup: %d", len(unique_players))

        # Phase 3: Upsert to DB
        db = SessionLocal() if not dry_run else None
        now = datetime.now(ZoneInfo("America/New_York"))

        processed, skipped = _upsert_players(
            db, unique_players, now, dry_run=dry_run
        )

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        table_count = 0
        if not dry_run and db:
            table_count = db.query(PositionEligibility).count()
            db.close()

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("BACKFILL COMPLETE")
        logger.info("  Taken (rostered):  %d", len(taken_players))
        logger.info("  Available (FA/W):  %d", len(available_players))
        logger.info("  Unique players:    %d", len(unique_players))
        logger.info("  Upserted:          %d", processed)
        logger.info("  Skipped:           %d", skipped)
        logger.info("  Table total:       %d rows", table_count)
        logger.info("  Elapsed:           %dms", elapsed)
        logger.info("=" * 60)

        # Spot-check: breakdown by player type and top multi-eligible
        if not dry_run:
            _log_summary_stats()

        return {
            "status": "success",
            "taken_fetched": len(taken_players),
            "available_fetched": len(available_players),
            "unique_players": len(unique_players),
            "records_upserted": processed,
            "skipped": skipped,
            "table_count": table_count,
            "elapsed_ms": elapsed,
        }

    except Exception as e:
        logger.exception("Backfill failed: %s", e)
        return {
            "status": "failed",
            "error": str(e),
            "elapsed_ms": int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000),
        }


def _log_summary_stats():
    """Log breakdown by type and top multi-eligible players."""
    db = SessionLocal()

    # Type breakdown
    from sqlalchemy import func
    type_counts = db.query(
        PositionEligibility.player_type, func.count()
    ).group_by(PositionEligibility.player_type).all()
    logger.info("")
    logger.info("Player type breakdown:")
    for ptype, cnt in sorted(type_counts, key=lambda x: -x[1]):
        logger.info("  %-10s %d", ptype, cnt)

    # Position coverage
    pos_cols = ["c", "1b", "2b", "3b", "ss", "lf", "cf", "rf", "of", "dh", "sp", "rp"]
    logger.info("")
    logger.info("Position coverage (players eligible at each):")
    for pos in pos_cols:
        col = getattr(PositionEligibility, f"can_play_{pos}")
        cnt = db.query(func.count()).filter(col == True).scalar()
        logger.info("  %-4s %d", pos.upper(), cnt)

    # Top multi-eligible
    multi = db.query(PositionEligibility).filter(
        PositionEligibility.multi_eligibility_count >= 4
    ).order_by(PositionEligibility.multi_eligibility_count.desc()).limit(15).all()
    if multi:
        logger.info("")
        logger.info("Top multi-eligible players (4+ positions):")
        for p in multi:
            flags_str = []
            for pos in pos_cols:
                if getattr(p, f"can_play_{pos}", False):
                    flags_str.append(pos.upper())
            logger.info("  %-25s %s (primary=%s, count=%d)",
                        p.player_name, "/".join(flags_str), p.primary_position,
                        p.multi_eligibility_count)

    db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill position eligibility for ALL MLB players")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    parser.add_argument("--rostered-only", action="store_true", help="Only fetch league rosters (skip free agents)")
    args = parser.parse_args()

    result = backfill_position_eligibility(dry_run=args.dry_run, rostered_only=args.rostered_only)
    if result["status"] == "success":
        logger.info("Done: %d players in table", result.get("table_count", result["records_upserted"]))
    else:
        logger.error("Failed: %s", result.get("error", "unknown"))
        sys.exit(1)
