"""
Manual sync job test with maximum verbosity for observability debugging.

This script will manually trigger one sync job (player_id_mapping) and capture
every log line to identify exactly where it fails.

Usage:
    python scripts/test_sync_job_observability.py
    railway run --service Fantasy-App -- python scripts/test_sync_job_observability.py
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure maximum verbosity logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

# Set SQLAlchemy logging to DEBUG
logging.getLogger('sqlalchemy').setLevel(logging.DEBUG)
logging.getLogger('urllib3').setLevel(logging.DEBUG)

async def test_player_id_mapping_sync():
    """
    Manually trigger the player_id_mapping sync job with maximum observability.
    This will reveal exactly where the job fails.
    """
    logger.info("=" * 80)
    logger.info("MANUAL SYNC JOB TEST: player_id_mapping")
    logger.info("Starting manual test at: %s", datetime.now(ZoneInfo("America/New_York")).isoformat())
    logger.info("=" * 80)

    try:
        # Import the ingestion scheduler
        from backend.services.daily_ingestion import DailyIngestionOrchestrator
        logger.info("✓ Successfully imported DailyIngestionOrchestrator")

        # Create scheduler instance
        logger.info("→ Creating DailyIngestionOrchestrator instance...")
        scheduler = DailyIngestionOrchestrator()
        logger.info("✓ Successfully created DailyIngestionOrchestrator")

        # Trigger the sync job manually
        logger.info("→ Triggering _sync_player_id_mapping job...")
        logger.info("-" * 80)

        result = await scheduler._sync_player_id_mapping()

        logger.info("-" * 80)
        logger.info("✓ Job execution completed")
        logger.info("Result: %s", result)

        # Analyze the result
        if result.get("status") == "success":
            logger.info("SUCCESS: Job completed successfully - processed %d records in %d ms",
                       result.get("records", 0), result.get("elapsed_ms", 0))
        elif result.get("status") == "skipped":
            logger.warning("SKIPPED: Job was skipped - check logs for reason")
        elif result.get("status") == "error":
            logger.error("FAILED: Job failed - check logs for exception details")

        # Verify database state
        logger.info("→ Verifying database state...")
        from backend.models import SessionLocal, PlayerIDMapping
        db = SessionLocal()

        try:
            total_count = db.query(PlayerIDMapping).count()
            logger.info("Total player_id_mapping records: %d", total_count)

            # Check NULL counts
            import sqlalchemy as func
            null_yahoo_key = db.query(func.count(PlayerIDMapping.id)).filter(
                PlayerIDMapping.yahoo_key.is_(None)
            ).scalar()
            null_mlbam_id = db.query(func.count(PlayerIDMapping.id)).filter(
                PlayerIDMapping.mlbam_id.is_(None)
            ).scalar()

            logger.info("NULL yahoo_key: %d / %d (%.1f%%)",
                       null_yahoo_key, total_count,
                       (null_yahoo_key / total_count * 100) if total_count > 0 else 0)
            logger.info("NULL mlbam_id: %d / %d (%.1f%%)",
                       null_mlbam_id, total_count,
                       (null_mlbam_id / total_count * 100) if total_count > 0 else 0)

            # Sample 3 most recent records
            recent = db.query(PlayerIDMapping).order_by(PlayerIDMapping.id.desc()).limit(3).all()
            logger.info("Sample of 3 most recent records:")
            for i, record in enumerate(recent, 1):
                logger.info("  %d. bdl_id=%s, full_name=%s, mlbam_id=%s, yahoo_id=%s",
                           i, record.bdl_id, record.full_name, record.mlbam_id, record.yahoo_id)

        finally:
            db.close()

    except Exception as exc:
        logger.exception("MANUAL TEST FAILED: Exception during job execution")
        logger.error("Exception type: %s", type(exc).__name__)
        logger.error("Exception message: %s", str(exc))
        import traceback
        logger.error("Stack trace:\n%s", traceback.format_exc())
        raise

    logger.info("=" * 80)
    logger.info("MANUAL SYNC JOB TEST: Complete")
    logger.info("=" * 80)

if __name__ == "__main__":
    try:
        asyncio.run(test_player_id_mapping_sync())
    except Exception as exc:
        logger.error("Test failed with exception: %s", exc)
        sys.exit(1)