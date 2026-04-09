"""
Backfill Script: Yahoo Keys from Position Eligibility

Cross-references position_eligibility.yahoo_player_key with player_id_mapping
by matching on normalized player names. This bridges the Yahoo namespace to
the BDL namespace.

Strategy:
  1. Load all position_eligibility rows into memory (2,376 rows - small)
  2. Load all player_id_mapping rows into memory (20,000 rows - manageable)
  3. Match by normalized_name (case-insensitive, Unicode-normalized)
  4. Fuzzy match for common variations (Jr./Sr./II/III suffixes)
  5. Update player_id_mapping.yahoo_key for matches

Usage:
    python scripts/backfill_yahoo_keys.py
    python scripts/backfill_yahoo_keys.py --dry-run
"""
import argparse
import logging
import os
import sys
import unicodedata
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Try to load .env for local development; Railway uses env vars directly
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass  # Railway environment doesn't have dotenv

from sqlalchemy import text
from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def normalize_name_for_matching(name: str) -> str:
    """
    Normalize player name for fuzzy matching.

    Steps:
      1. Unicode NFKD normalization (separates accents from letters)
      2. Lowercase
      3. Remove common suffixes (Jr., Sr., II, III, IV)
      4. Remove extra whitespace
      5. Remove periods
    """
    if not name:
        return ""

    # Unicode normalization
    name = unicodedata.normalize('NFKD', name)

    # Lowercase and strip
    name = name.lower().strip()

    # Remove common suffixes
    suffixes = [" jr.", " sr.", " ii", " iii", " iv", " jr", " sr"]
    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()

    # Remove periods (for "J.R." etc)
    name = name.replace(".", "")

    # Collapse multiple spaces
    while "  " in name:
        name = name.replace("  ", " ")

    return name.strip()


def find_best_mapping_match(pe_row: PositionEligibility, mapping_rows: list[PlayerIDMapping]) -> PlayerIDMapping | None:
    """
    Find best matching player_id_mapping row for a position_eligibility row.

    Matching priority:
      1. Exact normalized_name match
      2. Fuzzy normalized_name match (after suffix removal)
      3. Partial match (first name + last initial)

    Returns None if no good match found.
    """
    pe_name_normalized = normalize_name_for_matching(pe_row.player_name)

    # Priority 1: Exact match
    for mapping in mapping_rows:
        if mapping.normalized_name == pe_name_normalized:
            return mapping

    # Priority 2: Fuzzy match (try suffix-stripped versions)
    pe_name_fuzzy = normalize_name_for_matching(pe_row.player_name)
    for mapping in mapping_rows:
        mapping_name_fuzzy = normalize_name_for_matching(mapping.full_name)
        if pe_name_fuzzy == mapping_name_fuzzy:
            return mapping

    # Priority 3: Partial match (first name + last initial)
    # e.g., "Juan S" matches "Juan Soto"
    parts = pe_name_fuzzy.split()
    if len(parts) >= 2:
        first_name = parts[0]
        last_initial = parts[-1][0] if parts[-1] else ""
        partial_pattern = f"{first_name} {last_initial}"

        for mapping in mapping_rows:
            mapping_name_fuzzy = normalize_name_for_matching(mapping.full_name)
            if mapping_name_fuzzy.startswith(partial_pattern):
                return mapping

    return None


def backfill_yahoo_keys(db_session, dry_run: bool = False) -> dict:
    """
    Backfill yahoo_key from position_eligibility into player_id_mapping.

    Args:
        db_session: SQLAlchemy session
        dry_run: If True, don't commit changes

    Returns:
        dict with status, updated_count, skipped_count, errors
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting yahoo_key backfill from position_eligibility")
    logger.info("=" * 60)

    try:
        # Load all position_eligibility rows
        logger.info("Loading position_eligibility rows...")
        pe_rows = db_session.query(PositionEligibility).all()
        logger.info(f"Loaded {len(pe_rows)} position_eligibility rows")

        # Load all player_id_mapping rows
        logger.info("Loading player_id_mapping rows...")
        mapping_rows = db_session.query(PlayerIDMapping).all()
        logger.info(f"Loaded {len(mapping_rows)} player_id_mapping rows")

        updated_count = 0
        skipped_count = 0
        errors = []

        for pe_row in pe_rows:
            try:
                # Find matching mapping row
                mapping = find_best_mapping_match(pe_row, mapping_rows)

                if not mapping:
                    logger.debug(f"No match found for {pe_row.player_name} ({pe_row.yahoo_player_key})")
                    skipped_count += 1
                    continue

                # Update yahoo_key if NULL
                if mapping.yahoo_key is None:
                    mapping.yahoo_key = pe_row.yahoo_player_key
                    mapping.last_verified = datetime.now(ZoneInfo("America/New_York")).date()
                    updated_count += 1

                    if dry_run:
                        logger.info(f"[DRY-RUN] Would update: {mapping.full_name} -> {pe_row.yahoo_player_key}")
                else:
                    logger.debug(f"Already has yahoo_key: {mapping.full_name} -> {mapping.yahoo_key}")

            except Exception as e:
                logger.error(f"Error processing {pe_row.player_name}: {e}")
                errors.append(f"{pe_row.player_name}: {e}")
                continue

        if not dry_run:
            db_session.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Yahoo key backfill complete")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify results
        yahoo_key_count = db_session.execute(text(
            "SELECT COUNT(*) FROM player_id_mapping WHERE yahoo_key IS NOT NULL"
        )).scalar()
        logger.info(f"Total player_id_mapping rows with yahoo_key: {yahoo_key_count}")

        return {
            "status": "success",
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "errors": errors,
            "elapsed_ms": elapsed,
            "yahoo_key_count": yahoo_key_count
        }

    except Exception as e:
        logger.exception("Yahoo key backfill failed")
        return {
            "status": "failed",
            "error": str(e),
            "elapsed_ms": int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = backfill_yahoo_keys(db, dry_run=args.dry_run)
        if result["status"] == "success":
            logger.info(f"Success: {result['updated_count']} rows updated")
            sys.exit(0)
        else:
            logger.error(f"Failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
