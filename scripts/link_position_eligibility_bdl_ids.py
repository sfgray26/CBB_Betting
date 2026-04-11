"""
Link Script: bdl_player_id from player_id_mapping to position_eligibility

Now that player_id_mapping has yahoo_key populated, we can link position_eligibility
to the BDL namespace by setting bdl_player_id.

Usage:
    python scripts/link_position_eligibility_bdl_ids.py
    python scripts/link_position_eligibility_bdl_ids.py --dry-run
"""
import argparse
import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, ".")
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required in production Railway environment

from sqlalchemy.orm import Session
from backend.models import SessionLocal, PositionEligibility, PlayerIDMapping

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def link_bdl_player_ids(db_session, dry_run: bool = False) -> dict:
    """
    Link position_eligibility to player_id_mapping via bdl_player_id.

    Args:
        db_session: SQLAlchemy session
        dry_run: If True, don't commit changes

    Returns:
        dict with status, updated_count, skipped_count
    """
    t0 = datetime.now(ZoneInfo("America/New_York"))
    logger.info("=" * 60)
    logger.info("Starting bdl_player_id linking")
    logger.info("=" * 60)

    try:
        # Load all position_eligibility rows with NULL bdl_player_id
        logger.info("Finding position_eligibility rows with NULL bdl_player_id...")
        pe_rows = db_session.query(PositionEligibility).filter(
            PositionEligibility.bdl_player_id.is_(None)
        ).all()
        logger.info(f"Found {len(pe_rows)} rows with NULL bdl_player_id")

        updated_count = 0
        skipped_count = 0
        errors = []

        for pe_row in pe_rows:
            try:
                # Find matching player_id_mapping by yahoo_key
                mapping = db_session.query(PlayerIDMapping).filter(
                    PlayerIDMapping.yahoo_key == pe_row.yahoo_player_key
                ).first()

                if not mapping:
                    logger.debug(f"No mapping found for {pe_row.yahoo_player_key}")
                    skipped_count += 1
                    continue

                if mapping.bdl_id is None:
                    logger.debug(f"Mapping has NULL bdl_id for {pe_row.yahoo_player_key}")
                    skipped_count += 1
                    continue

                # Update bdl_player_id
                pe_row.bdl_player_id = mapping.bdl_id
                updated_count += 1

                if dry_run:
                    logger.info(f"[DRY-RUN] Would update: {pe_row.player_name} -> bdl_id={mapping.bdl_id}")

            except Exception as e:
                logger.error(f"Error processing {pe_row.yahoo_player_key}: {e}")
                errors.append(f"{pe_row.yahoo_player_key}: {e}")
                continue

        if not dry_run:
            db_session.commit()

        elapsed = int((datetime.now(ZoneInfo("America/New_York")) - t0).total_seconds() * 1000)

        logger.info("=" * 60)
        logger.info("bdl_player_id linking complete")
        logger.info(f"  Updated: {updated_count}")
        logger.info(f"  Skipped: {skipped_count}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Elapsed: {elapsed}ms")
        logger.info("=" * 60)

        # Verify results
        bdl_id_count = db_session.query(PositionEligibility).filter(
            PositionEligibility.bdl_player_id.isnot(None)
        ).count()
        logger.info(f"Total position_eligibility rows with bdl_player_id: {bdl_id_count}")

        return {
            "status": "success",
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "errors": errors,
            "elapsed_ms": elapsed,
            "bdl_id_count": bdl_id_count
        }

    except Exception as e:
        logger.exception("bdl_player_id linking failed")
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
        result = link_bdl_player_ids(db, dry_run=args.dry_run)
        if result["status"] == "success":
            logger.info(f"✅ Success: {result['updated_count']} rows updated")
            sys.exit(0)
        else:
            logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
