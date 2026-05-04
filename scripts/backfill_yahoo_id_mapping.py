#!/usr/bin/env python
"""
Backfill Yahoo IDs from Yahoo Fantasy API to player_id_mapping table.

Problem solved
--------------
Yahoo ID coverage is only 3.7% in player_id_mapping table.
This breaks Yahoo lookups for waiver recommendations and lineup decisions.

This script fetches all Yahoo players (rosters + free agents) and backfills
yahoo_id into player_id_mapping by matching on bdl_id or normalized_name.

Expected outcome: 60-80% coverage (up from 3.7%)

Usage
-----
    railway run python scripts/backfill_yahoo_id_mapping.py
    python scripts/backfill_yahoo_id_mapping.py --dry-run
"""

import argparse
import logging
import os
import sys
import time
import unicodedata
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.models import SessionLocal, PlayerIDMapping, PlayerProjection
from backend.fantasy_baseball.yahoo_client_resilient import ResilientYahooClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Yahoo player_key format: "469.p.7590" -> extract yahoo_id = "7590"
def extract_yahoo_id(player_key: str) -> Optional[str]:
    """Extract Yahoo ID from player_key."""
    if not player_key:
        return None

    parts = player_key.split('.p.')
    if len(parts) == 2:
        return parts[1]
    return None

def normalize_name(name: str) -> str:
    """Normalize name for fuzzy matching."""
    if not name:
        return ""

    # Normalize unicode (NFKD decomposition, remove combining chars)
    normalized = unicodedata.normalize("NFKD", name)

    # Remove combining characters (accents)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')

    # Lowercase and trim
    normalized = normalized.lower().strip()

    return normalized

def lookup_bdl_id_by_name(session: Session, normalized_name: str) -> Optional[int]:
    """Lookup bdl_id from player_projections using normalized_name."""
    if not normalized_name:
        return None

    # Try exact match first
    result = session.execute(
        text("""
            SELECT player_id
            FROM player_projections
            WHERE LOWER(REPLACE(player_name, ' ', '-') = :normalized_name
            LIMIT 1
        """),
        {"normalized_name": normalized_name}
    ).fetchone()

    if result and result[0]:
        # Extract bdl_id from player_id format (assuming bdl_id)
        player_id = result[0]
        if player_id and player_id.isdigit():
            return int(player_id)

    return None

def upsert_yahoo_player(
    session: Session,
    yahoo_id: str,
    player_key: str,
    player_name: str,
    bdl_id: Optional[int] = None,
    normalized_name: Optional[str] = None
) -> bool:
    """Upsert a Yahoo player into player_id_mapping."""

    try:
        # Check if row exists
        existing = session.query(PlayerIDMapping).filter(
            PlayerIDMapping.yahoo_id == yahoo_id
        ).first()

        if existing:
            # Update existing row
            existing.source = 'yahoo'
            existing.updated_at = text('NOW()')
            if bdl_id and not existing.bdl_id:
                existing.bdl_id = bdl_id
            if normalized_name and not existing.normalized_name:
                existing.normalized_name = normalized_name
            logger.debug(f"Updated existing yahoo_id={yahoo_id}")
        else:
            # Insert new row
            mapping = PlayerIDMapping(
                yahoo_id=yahoo_id,
                yahoo_key=player_key,
                bdl_id=bdl_id,
                full_name=player_name,
                normalized_name=normalized_name or normalize_name(player_name),
                source='yahoo'
            )
            session.add(mapping)
            logger.debug(f"Inserted new yahoo_id={yahoo_id}")

        return True
    except Exception as e:
        logger.error(f"Failed to upsert yahoo_id={yahoo_id}: {e}")
        return False

def fetch_and_process_rosters(client: ResilientYahooClient, session: Session, batch_size: int = 100) -> int:
    """Fetch all rosters and upsert Yahoo IDs."""

    logger.info("Fetching all rosters...")
    rosters = client.get_all_rosters()

    processed = 0
    for team_key, roster in rosters.items():
        logger.info(f"Processing roster for {team_key} ({len(roster)} players)")

        for player_dict in roster:
            player_key = player_dict.get('player_key')
            player_name = player_dict.get('name', {}).get('full', '')

            if not player_key:
                continue

            yahoo_id = extract_yahoo_id(player_key)
            if not yahoo_id:
                continue

            # Try to find bdl_id by matching name
            normalized = normalize_name(player_name)
            bdl_id = lookup_bdl_id_by_name(session, normalized)

            # Upsert
            if upsert_yahoo_player(session, yahoo_id, player_key, player_name, bdl_id, normalized):
                processed += 1

        # Commit every team
        try:
            session.commit()
            logger.info(f"Committed roster for {team_key} ({processed} total processed)")
        except Exception as e:
            logger.error(f"Failed to commit roster for {team_key}: {e}")
            session.rollback()

    return processed

def fetch_and_process_free_agents(client: ResilientYahooClient, session: Session, batch_size: int = 25) -> int:
    """Fetch free agents and upsert Yahoo IDs."""

    logger.info("Fetching free agents...")

    # Positions to paginate through
    positions = ["C", "1B", "2B", "3B", "SS", "OF", "Util", "SP", "RP", "P"]

    processed = 0
    for position in positions:
        logger.info(f"Fetching free agents for position={position}")

        start = 0
        while True:
            try:
                # Paginate free agents
                players = client.get_free_agents(
                    position=position,
                    start=start,
                    count=batch_size
                )

                if not players or len(players) == 0:
                    logger.info(f"No more free agents for {position} at start={start}")
                    break

                logger.info(f"Processing {len(players)} free agents for {position} (start={start})")

                for player_dict in players:
                    player_key = player_dict.get('player_key')
                    player_name = player_dict.get('name', {}).get('full', '')

                    if not player_key:
                        continue

                    yahoo_id = extract_yahoo_id(player_key)
                    if not yahoo_id:
                        continue

                    # Try to find bdl_id by matching name
                    normalized = normalize_name(player_name)
                    bdl_id = lookup_bdl_id_by_name(session, normalized)

                    # Upsert
                    if upsert_yahoo_player(session, yahoo_id, player_key, player_name, bdl_id, normalized):
                        processed += 1

                # Commit every batch
                try:
                    session.commit()
                    logger.info(f"Committed batch for {position} (start={start}, {processed} total processed)")
                except Exception as e:
                    logger.error(f"Failed to commit batch for {position}: {e}")
                    session.rollback()

                # Move to next page
                if len(players) < batch_size:
                    logger.info(f"Last page for {position} at start={start}")
                    break

                start += batch_size

                # Small delay to avoid rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error fetching free agents for {position} at start={start}: {e}")
                break

    return processed

def run(dry_run: bool = False) -> None:
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set", file=sys.stderr)
        sys.exit(1)

    print("=== Yahoo ID Backfill ===")

    if dry_run:
        print("[DRY RUN] Would fetch from Yahoo API and upsert player_id_mapping")
        print("This will:")
        print("  1. Fetch all rosters (~360 players)")
        print("  2. Fetch free agents (~500 players)")
        print("  3. Upsert yahoo_id into player_id_mapping")
        print("  4. Match on bdl_id or normalized_name")
        return

    # Initialize Yahoo client
    try:
        client = ResilientYahooClient()
        logger.info("Yahoo client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Yahoo client: {e}")
        sys.exit(1)

    engine = create_engine(db_url)

    # Acquire advisory lock 100_017
    print("Acquiring advisory lock 100_017...")
    with engine.begin() as conn:
        lock_result = conn.execute(
            text("SELECT pg_try_advisory_lock(100017)")
        ).scalar()

        if not lock_result:
            print("ERROR: Advisory lock 100_017 held by another process")
            sys.exit(1)

        print("Lock acquired. Starting backfill...")

        try:
            # Create session
            session = SessionLocal()

            # Fetch rosters
            roster_count = fetch_and_process_rosters(client, session)
            logger.info(f"Processed {roster_count} rostered players")

            # Fetch free agents
            fa_count = fetch_and_process_free_agents(client, session)
            logger.info(f"Processed {fa_count} free agents")

            session.close()

        except Exception as e:
            logger.error(f"Failed to fetch and process players: {e}")
            conn.execute(text("SELECT pg_advisory_unlock(100017)"))
            sys.exit(1)

    # Final lock release
    conn.execute(text("SELECT pg_advisory_unlock(100017)"))
    print("Lock released.")

    # Verify results
    print("\n=== Verification ===")
    session = SessionLocal()
    result = session.execute(
        text("""
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL) AS with_yahoo_id,
                ROUND(100.0 * COUNT(*) FILTER (WHERE yahoo_id IS NOT NULL) / COUNT(*), 1) AS coverage_pct
            FROM player_id_mapping
        """)
    ).fetchone()

    print(f"Yahoo ID coverage: {result[2]}% ({result[1]:,} / {result[0]:,})")
    session.close()

    if result[2] < 60.0:
        print(f"WARNING: Coverage ({result[2]}%) is below target (60%)")
    else:
        print(f"SUCCESS: Yahoo ID coverage now at {result[2]}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Yahoo IDs from Yahoo Fantasy API")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    args = parser.parse_args()

    run(dry_run=args.dry_run)
