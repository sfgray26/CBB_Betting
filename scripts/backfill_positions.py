"""
Backfill Script: Position Eligibility (Current Snapshot)

Fetches CURRENT position eligibility from Yahoo Fantasy API for all rostered players.
Creates ONE ROW PER PLAYER with boolean flags for ALL eligible positions.

Note: Yahoo API only exposes current snapshot — no historical position data.
Ongoing updates tracked via _sync_position_eligibility() in daily_ingestion.py.

Usage:
    python scripts/backfill_positions.py
    python scripts/backfill_positions.py --dry-run
"""
import argparse
import logging
import os
import sys
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
PITCHER_POSITIONS = {"SP", "RP", "P"}


def build_position_flags(positions: list[str]) -> dict:
    """
    Build can_play_* boolean flags from a list of Yahoo position strings.

    One dict per player — ALL eligible positions merged into a single set of flags.
    OF is auto-set if any of LF/CF/RF is present.
    """
    pos_set = {p.upper() for p in positions if p}

    flags = {
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
    return flags


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


def backfill_position_eligibility(dry_run: bool = False) -> dict:
    """
    Fetch position eligibility from Yahoo Fantasy API and upsert to DB.

    ONE ROW PER PLAYER with ALL eligible positions as boolean flags.
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Position eligibility backfill (one row per player)")
    logger.info("=" * 60)

    try:
        yahoo = YahooFantasyClient()
        league_key = yahoo.league_key
        if not league_key:
            logger.error("No league_key configured — cannot fetch rosters")
            return {"status": "failed", "error": "No league_key"}

        logger.info("Fetching all rosters from league %s ...", league_key)
        all_players = yahoo.get_league_rosters(league_key=league_key, include_team_key=True)
        logger.info("Fetched %d players across all teams", len(all_players))

        if not all_players:
            logger.error("No players returned from Yahoo API")
            return {"status": "failed", "error": "Empty roster response"}

        db = SessionLocal()
        now = datetime.now(ZoneInfo("America/New_York"))
        created = 0
        skipped = 0

        # Deduplicate: same player may appear on multiple fantasy teams' rosters
        # (shouldn't happen in a well-formed league, but defensive)
        seen_keys = set()

        for player_data in all_players:
            player_key = player_data.get("player_key")
            if not player_key or player_key in seen_keys:
                skipped += 1
                continue
            seen_keys.add(player_key)

            name = player_data.get("name", "Unknown")
            positions = player_data.get("positions", [])

            if not positions:
                logger.debug("No positions for %s (%s) — skipping", name, player_key)
                skipped += 1
                continue

            # Build flags for ALL positions in one dict
            flags = build_position_flags(positions)
            primary = determine_primary_position(positions)
            ptype = determine_player_type(positions)

            # Count meaningful positions (exclude Util)
            meaningful = [p for p in positions if p.upper() not in ("UTIL",)]
            multi_count = len(meaningful)

            if dry_run:
                flag_str = ", ".join(k.replace("can_play_", "").upper()
                                     for k, v in flags.items() if v)
                logger.info("[DRY-RUN] %s (%s): %s | primary=%s type=%s count=%d",
                            name, player_key, flag_str, primary, ptype, multi_count)
                created += 1
                continue

            # Upsert: ON CONFLICT (yahoo_player_key) DO UPDATE
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
            created += 1

        if not dry_run:
            db.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        table_count = db.query(PositionEligibility).count() if not dry_run else 0
        db.close()

        logger.info("=" * 60)
        logger.info("Backfill complete")
        logger.info("  Players processed: %d", created)
        logger.info("  Skipped (no key/positions): %d", skipped)
        logger.info("  Table total: %d rows", table_count)
        logger.info("  Elapsed: %dms", elapsed)
        logger.info("=" * 60)

        # Spot-check: show multi-eligible players
        if not dry_run:
            db2 = SessionLocal()
            multi = db2.query(PositionEligibility).filter(
                PositionEligibility.multi_eligibility_count >= 3
            ).order_by(PositionEligibility.multi_eligibility_count.desc()).limit(10).all()
            if multi:
                logger.info("Top multi-eligible players:")
                for p in multi:
                    flags_str = []
                    for pos in ["c", "1b", "2b", "3b", "ss", "lf", "cf", "rf", "of", "dh", "sp", "rp"]:
                        if getattr(p, f"can_play_{pos}", False):
                            flags_str.append(pos.upper())
                    logger.info("  %s: %s (primary=%s, count=%d)",
                                p.player_name, "/".join(flags_str), p.primary_position, p.multi_eligibility_count)
            db2.close()

        return {
            "status": "success",
            "records_processed": created,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    result = backfill_position_eligibility(dry_run=args.dry_run)
    if result["status"] == "success":
        logger.info("Done: %d records", result["records_processed"])
    else:
        logger.error("Failed: %s", result.get("error", "unknown"))
        sys.exit(1)
