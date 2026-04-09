"""
IMMEDIATE SYNC JOB EXECUTION - TRIGGER ALL JOBS NOW
Run this on Railway with: railway run --service Fantasy-App -- python scripts/run_sync_jobs_now.py
"""

import asyncio
import sys
import os
from datetime import datetime
from zoneinfo import ZoneInfo

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    logger.info("=" * 80)
    logger.info("IMMEDIATE SYNC JOB EXECUTION - Triggering all jobs NOW")
    logger.info(f"Started at: {datetime.now(ZoneInfo('America/New_York')).isoformat()}")
    logger.info("=" * 80)

    try:
        # Import the ingestion orchestrator
        from backend.services.daily_ingestion import DailyIngestionOrchestrator
        logger.info("✓ Successfully imported DailyIngestionOrchestrator")

        # Create orchestrator instance
        logger.info("→ Creating DailyIngestionOrchestrator instance...")
        orchestrator = DailyIngestionOrchestrator()
        logger.info("✓ Successfully created DailyIngestionOrchestrator")

        # Check baseline database state
        logger.info("→ Checking baseline database state...")
        from backend.models import SessionLocal, PlayerIDMapping, PositionEligibility, ProbablePitcherSnapshot
        db = SessionLocal()

        try:
            baseline_player_id = db.query(PlayerIDMapping).count()
            baseline_yahoo_id = db.query(PlayerIDMapping).filter(PlayerIDMapping.yahoo_id.isnot(None)).count()
            baseline_mlbam_id = db.query(PlayerIDMapping).filter(PlayerIDMapping.mlbam_id.isnot(None)).count()
            baseline_position = db.query(PositionEligibility).count()
            baseline_pitchers = db.query(ProbablePitcherSnapshot).count()

            logger.info(f"BASELINE player_id_mapping: {baseline_player_id} rows, {baseline_yahoo_id} non-null yahoo_id, {baseline_mlbam_id} non-null mlbam_id")
            logger.info(f"BASELINE position_eligibility: {baseline_position} rows")
            logger.info(f"BASELINE probable_pitchers: {baseline_pitchers} rows")

        finally:
            db.close()

        # JOB 1: player_id_mapping sync
        logger.info("-" * 80)
        logger.info("JOB 1: Triggering player_id_mapping sync...")
        result1 = await orchestrator._sync_player_id_mapping()
        logger.info(f"JOB 1 Result: {result1}")

        # Check database after job 1
        db = SessionLocal()
        try:
            after1_player_id = db.query(PlayerIDMapping).count()
            after1_yahoo_id = db.query(PlayerIDMapping).filter(PlayerIDMapping.yahoo_id.isnot(None)).count()
            after1_mlbam_id = db.query(PlayerIDMapping).filter(PlayerIDMapping.mlbam_id.isnot(None)).count()
            logger.info(f"AFTER JOB 1 player_id_mapping: {after1_player_id} rows, {after1_yahoo_id} non-null yahoo_id, {after1_mlbam_id} non-null mlbam_id")
            logger.info(f"DELTA: +{after1_player_id - baseline_player_id} rows, +{after1_yahoo_id - baseline_yahoo_id} yahoo_id, +{after1_mlbam_id - baseline_mlbam_id} mlbam_id")
        finally:
            db.close()

        # JOB 2: position_eligibility sync
        logger.info("-" * 80)
        logger.info("JOB 2: Triggering position_eligibility sync...")
        result2 = await orchestrator._sync_position_eligibility()
        logger.info(f"JOB 2 Result: {result2}")

        # Check database after job 2
        db = SessionLocal()
        try:
            after2_position = db.query(PositionEligibility).count()
            logger.info(f"AFTER JOB 2 position_eligibility: {after2_position} rows")
            logger.info(f"DELTA: +{after2_position - baseline_position} rows")
        finally:
            db.close()

        # JOB 3: probable_pitchers sync
        logger.info("-" * 80)
        logger.info("JOB 3: Triggering probable_pitchers sync...")
        result3 = await orchestrator._sync_probable_pitchers()
        logger.info(f"JOB 3 Result: {result3}")

        # Check database after job 3
        db = SessionLocal()
        try:
            after3_pitchers = db.query(ProbablePitcherSnapshot).count()
            logger.info(f"AFTER JOB 3 probable_pitchers: {after3_pitchers} rows")
            logger.info(f"DELTA: +{after3_pitchers - baseline_pitchers} rows")
        finally:
            db.close()

        # SUMMARY
        logger.info("=" * 80)
        logger.info("EXECUTION SUMMARY")
        logger.info(f"player_id_mapping: {baseline_player_id} → {after1_player_id} rows (+{after1_player_id - baseline_player_id})")
        logger.info(f"  yahoo_id NULL rate: {(baseline_yahoo_id/baseline_player_id*100) if baseline_player_id > 0 else 100:.1f}% → {(after1_yahoo_id/after1_player_id*100) if after1_player_id > 0 else 100:.1f}%")
        logger.info(f"  mlbam_id NULL rate: {(baseline_mlbam_id/baseline_player_id*100) if baseline_player_id > 0 else 100:.1f}% → {(after1_mlbam_id/after1_player_id*100) if after1_player_id > 0 else 100:.1f}%")
        logger.info(f"position_eligibility: {baseline_position} → {after2_position} rows (+{after2_position - baseline_position})")
        logger.info(f"probable_pitchers: {baseline_pitchers} → {after3_pitchers} rows (+{after3_pitchers - baseline_pitchers})")
        logger.info("=" * 80)

    except Exception as exc:
        logger.exception("EXECUTION FAILED")
        logger.error(f"Exception type: {type(exc).__name__}")
        logger.error(f"Exception message: {str(exc)}")
        import traceback
        logger.error(f"Stack trace:\n{traceback.format_exc()}")
        sys.exit(1)

    logger.info("IMMEDIATE SYNC JOB EXECUTION - Complete")

if __name__ == "__main__":
    asyncio.run(main())
