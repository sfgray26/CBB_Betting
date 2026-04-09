"""
Backfill Script: Player ID Mapping (BDL ↔ MLB ↔ Yahoo)

Fetches all MLB players from BallDon'tLie API and stores cross-reference
mapping between BallDon'tLie ID, MLBAM ID, and Yahoo player IDs.

This is a ONE-TIME backfill — ongoing sync happens via daily_ingestion.py

Usage:
    python scripts/backfill_player_id_mapping.py

Expected Output:
    ~1,500 rows in player_id_mapping table (all MLB players)
"""
import logging
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from backend.services.balldontlie import BallDontLieClient
from backend.models import SessionLocal, PlayerIDMapping

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backfill_player_id_mapping() -> dict:
    """
    Fetch all MLB players from BDL and store cross-reference mapping.

    Returns:
        dict with status, records_processed, elapsed_ms
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting player ID mapping backfill")
    logger.info("=" * 60)

    try:
        # Initialize BDL client
        bdl_client = BallDontLieClient()
        db = SessionLocal()

        records_processed = 0
        records_updated = 0
        records_created = 0

        # Fetch all MLB players from BDL
        # BDL API returns players in pages — need to paginate
        logger.info("Fetching all MLB players from BDL...")

        all_players = bdl_client.get_all_mlb_players()

        logger.info(f"Total players fetched from BDL: {len(all_players)}")

        if len(all_players) == 0:
            logger.error("No players fetched from BDL — aborting backfill")
            return {
                'status': 'failed',
                'records_processed': 0,
                'error': 'No players fetched from BDL',
                'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
            }

        # Store cross-reference mappings
        logger.info("Storing player ID mappings in database...")

        for player in all_players:
            try:
                # Extract player data from BDL MLBPlayer object
                # - id: BDL player ID
                # - full_name: Player name
                # - first_name, last_name: Name components
                # - position: Position abbreviation
                # - team: MLBTeam object with abbreviation

                bdl_id = player.id  # BDL player ID
                if not bdl_id:
                    continue

                mlbam_id = None  # BDL doesn't provide mlbam_id directly
                full_name = player.full_name

                # Handle validation errors from Pydantic models
                try:
                    first_name = player.first_name
                except Exception:
                    first_name = ''

                try:
                    last_name = player.last_name
                except Exception:
                    last_name = ''

                primary_position = player.position

                # Safely extract team abbreviation with error handling
                team_abbrev = ''
                try:
                    if player.team and hasattr(player.team, 'abbreviation'):
                        team_abbrev = player.team.abbreviation
                except Exception:
                    team_abbrev = ''

                # Check if record exists
                existing = db.query(PlayerIDMapping).filter(
                    PlayerIDMapping.bdl_id == bdl_id
                ).first()

                now = datetime.now(ZoneInfo("America/New_York"))

                if existing:
                    # Update existing record
                    existing.mlbam_id = mlbam_id
                    existing.full_name = full_name
                    existing.normalized_name = full_name.lower()
                    existing.updated_at = now
                    records_updated += 1
                else:
                    # Create new record
                    mapping = PlayerIDMapping(
                        yahoo_key=None,  # Will be populated separately via Yahoo API
                        yahoo_id=None,     # Will be populated separately via Yahoo API
                        mlbam_id=mlbam_id,
                        bdl_id=bdl_id,
                        full_name=full_name,
                        normalized_name=full_name.lower(),
                        source='api',
                        resolution_confidence=1.0,  # Direct from BDL API
                        created_at=now,
                        updated_at=now
                    )
                    db.add(mapping)
                    records_created += 1

                records_processed += 1

                # Commit in batches of 100
                if records_processed % 100 == 0:
                    db.commit()
                    logger.info(f"Processed {records_processed} players...")

            except Exception as e:
                logger.warning(f"Failed to process player {full_name if full_name else 'unknown'}: {e}")
                continue

        # Final commit
        db.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("Player ID mapping backfill complete")
        logger.info(f"  Records processed: {records_processed}")
        logger.info(f"  Records created: {records_created}")
        logger.info(f"  Records updated: {records_updated}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify table count
        table_count = db.query(PlayerIDMapping).count()
        logger.info(f"Total rows in player_id_mapping table: {table_count}")

        db.close()

        return {
            'status': 'success',
            'records_processed': records_processed,
            'records_created': records_created,
            'records_updated': records_updated,
            'table_count': table_count,
            'elapsed_ms': elapsed
        }

    except Exception as e:
        logger.exception("Player ID mapping backfill failed: %s", e)
        return {
            'status': 'failed',
            'records_processed': 0,
            'error': str(e),
            'elapsed_ms': int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)
        }


if __name__ == "__main__":
    result = backfill_player_id_mapping()

    if result['status'] == 'success':
        logger.info(f"✅ Backfill successful: {result['records_processed']} records processed")
        sys.exit(0)
    else:
        logger.error(f"❌ Backfill failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)
