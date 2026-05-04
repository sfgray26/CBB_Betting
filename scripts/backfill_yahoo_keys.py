"""
Backfill Script: Yahoo Keys from Position Eligibility + Free Agents

Cross-references position_eligibility.yahoo_player_key AND top 500 free agents
from Yahoo API with player_id_mapping by matching on normalized player names.
This bridges the Yahoo namespace to the BDL namespace.

Strategy:
  1. Load all position_eligibility rows into memory (2,376 rows - small)
  2. Fetch top 500 free agents from Yahoo API (most relevant waiver targets)
  3. Load all player_id_mapping rows into memory (20,000 rows - manageable)
  4. Match by normalized_name (case-insensitive, Unicode-normalized)
  5. Fuzzy match for common variations (Jr./Sr./II/III suffixes)
  6. Update player_id_mapping.yahoo_key for matches

Usage:
    python scripts/backfill_yahoo_keys.py
    python scripts/backfill_yahoo_keys.py --dry-run
    python scripts/backfill_yahoo_keys.py --skip-free-agents  # Only use position_eligibility
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


def backfill_yahoo_keys(db_session, dry_run: bool = False, skip_free_agents: bool = False) -> dict:
    """
    Backfill yahoo_key from position_eligibility + free agents into player_id_mapping.

    Args:
        db_session: SQLAlchemy session
        dry_run: If True, don't commit changes
        skip_free_agents: If True, only use position_eligibility (legacy behavior)

    Returns:
        dict with status, updated_count, skipped_count, errors
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting yahoo_key backfill from position_eligibility + free agents")
    logger.info("=" * 60)

    try:
        # Load all position_eligibility rows
        logger.info("Loading position_eligibility rows...")
        pe_rows = db_session.query(PositionEligibility).all()
        logger.info(f"Loaded {len(pe_rows)} position_eligibility rows")

        # Fetch top 500 free agents from Yahoo API (unless disabled)
        free_agent_players = []
        if not skip_free_agents:
            try:
                logger.info("Fetching top 500 free agents from Yahoo API...")
                from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
                client = YahooFantasyClient()
                # Yahoo API limits count to ~500 per call
                free_agent_players = client.get_free_agents(count=500)
                logger.info(f"Fetched {len(free_agent_players)} free agents from Yahoo API")
            except Exception as exc:
                logger.warning(f"Failed to fetch free agents from Yahoo API: {exc}")
                logger.warning("Continuing with position_eligibility only...")

        # Load all player_id_mapping rows
        logger.info("Loading player_id_mapping rows...")
        mapping_rows = db_session.query(PlayerIDMapping).all()
        logger.info(f"Loaded {len(mapping_rows)} player_id_mapping rows")

        # Track yahoo_keys already assigned (in DB + during this run)
        # This prevents UniqueViolation errors if multiple BDL IDs match the same yahoo_key
        assigned_yahoo_keys: set[str] = set()
        for mapping in mapping_rows:
            if mapping.yahoo_key:
                assigned_yahoo_keys.add(mapping.yahoo_key)
        logger.info(f"Found {len(assigned_yahoo_keys)} yahoo_keys already assigned in DB")

        updated_count = 0
        skipped_count = 0
        collision_count = 0
        errors = []
        source_counts = {"position_eligibility": 0, "free_agents": 0}

        # Process position_eligibility rows
        logger.info("Processing position_eligibility rows...")
        for pe_row in pe_rows:
            try:
                # Find matching mapping row
                mapping = find_best_mapping_match(pe_row, mapping_rows)

                if not mapping:
                    logger.debug(f"No match found for {pe_row.player_name} ({pe_row.yahoo_player_key})")
                    skipped_count += 1
                    continue

                # Check for yahoo_key collision before assigning
                if pe_row.yahoo_player_key in assigned_yahoo_keys:
                    logger.warning(
                        f"Collision: yahoo_key {pe_row.yahoo_player_key} already assigned. "
                        f"Skipping {mapping.full_name} (bdl_id={mapping.bdl_id}) from position_eligibility."
                    )
                    collision_count += 1
                    skipped_count += 1
                    continue

                # Update yahoo_key if NULL
                if mapping.yahoo_key is None:
                    mapping.yahoo_key = pe_row.yahoo_player_key
                    mapping.last_verified = datetime.now(ZoneInfo("America/New_York")).date()
                    assigned_yahoo_keys.add(pe_row.yahoo_player_key)  # Track assignment
                    updated_count += 1
                    source_counts["position_eligibility"] += 1

                    if dry_run:
                        logger.info(f"[DRY-RUN] Would update: {mapping.full_name} -> {pe_row.yahoo_player_key} (position_eligibility)")
                    else:
                        logger.info(f"Updated: {mapping.full_name} -> {pe_row.yahoo_player_key} (position_eligibility)")
                else:
                    logger.debug(f"Already has yahoo_key: {mapping.full_name} -> {mapping.yahoo_key}")

            except Exception as e:
                logger.error(f"Error processing {pe_row.player_name}: {e}")
                errors.append(f"{pe_row.player_name}: {e}")
                continue

        # Process free agent rows
        if free_agent_players:
            logger.info(f"Processing {len(free_agent_players)} free agents...")
            for fa in free_agent_players:
                try:
                    fa_yahoo_key = fa.get("player_key")
                    fa_name = fa.get("name", "")

                    if not fa_yahoo_key or not fa_name:
                        logger.debug(f"Skipping FA with missing player_key or name: {fa}")
                        skipped_count += 1
                        continue

                    # Create a synthetic PositionEligibility-like object for matching
                    class FakePositionEligibility:
                        def __init__(self, name: str, yahoo_key: str):
                            self.player_name = name
                            self.yahoo_player_key = yahoo_key

                    fake_pe = FakePositionEligibility(fa_name, fa_yahoo_key)
                    mapping = find_best_mapping_match(fake_pe, mapping_rows)

                    if not mapping:
                        logger.debug(f"No match found for FA {fa_name} ({fa_yahoo_key})")
                        skipped_count += 1
                        continue

                    # Check for yahoo_key collision before assigning
                    if fa_yahoo_key in assigned_yahoo_keys:
                        logger.warning(
                            f"Collision: yahoo_key {fa_yahoo_key} already assigned. "
                            f"Skipping FA {mapping.full_name} (bdl_id={mapping.bdl_id})."
                        )
                        collision_count += 1
                        skipped_count += 1
                        continue

                    # Update yahoo_key if NULL (don't overwrite existing keys from position_eligibility)
                    if mapping.yahoo_key is None:
                        mapping.yahoo_key = fa_yahoo_key
                        mapping.last_verified = datetime.now(ZoneInfo("America/New_York")).date()
                        assigned_yahoo_keys.add(fa_yahoo_key)  # Track assignment
                        updated_count += 1
                        source_counts["free_agents"] += 1

                        if dry_run:
                            logger.info(f"[DRY-RUN] Would update: {mapping.full_name} -> {fa_yahoo_key} (free_agent)")
                        else:
                            logger.info(f"Updated: {mapping.full_name} -> {fa_yahoo_key} (free_agent)")
                    else:
                        logger.debug(f"Already has yahoo_key: {mapping.full_name} -> {mapping.yahoo_key}")

                except Exception as e:
                    logger.error(f"Error processing FA {fa.get('name', 'unknown')}: {e}")
                    errors.append(f"{fa.get('name', 'unknown')}: {e}")
                    continue

        if not dry_run:
            db_session.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Yahoo key backfill complete")
        logger.info(f"  Updated: {updated_count} (position_eligibility={source_counts['position_eligibility']}, free_agents={source_counts['free_agents']})")
        logger.info(f"  Skipped: {skipped_count} (collisions={collision_count})")
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
            "source_counts": source_counts,
            "skipped_count": skipped_count,
            "collision_count": collision_count,
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
    parser.add_argument("--skip-free-agents", action="store_true", help="Skip free agent API fetch (legacy behavior)")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        result = backfill_yahoo_keys(db, dry_run=args.dry_run, skip_free_agents=args.skip_free_agents)
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
