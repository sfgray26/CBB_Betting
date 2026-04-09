"""
Simple test endpoints for triggering sync jobs without authentication.
Add this to main.py temporarily for testing.
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy.orm import Session
from backend.models import get_db
from backend.services.daily_ingestion import DailyIngestionOrchestrator
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/sync/player-id-mapping")
async def test_sync_player_id_mapping():
    """
    Test endpoint to trigger player_id_mapping sync without auth.
    REMOVE AFTER TESTING!
    """
    try:
        logger.info("TEST: Triggering player_id_mapping sync...")
        orchestrator = DailyIngestionOrchestrator()
        result = await orchestrator._sync_player_id_mapping()
        logger.info(f"TEST: player_id_mapping completed with result: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("TEST: player_id_mapping failed")
        return {"status": "error", "error": str(e)}

@router.post("/sync/position-eligibility")
async def test_sync_position_eligibility():
    """Test endpoint to trigger position_eligibility sync without auth."""
    try:
        logger.info("TEST: Triggering position_eligibility sync...")
        orchestrator = DailyIngestionOrchestrator()
        result = await orchestrator._sync_position_eligibility()
        logger.info(f"TEST: position_eligibility completed with result: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("TEST: position_eligibility failed")
        return {"status": "error", "error": str(e)}

@router.post("/sync/probable-pitchers")
async def test_sync_probable_pitchers():
    """Test endpoint to trigger probable_pitchers sync without auth."""
    try:
        logger.info("TEST: Triggering probable_pitchers sync...")
        orchestrator = DailyIngestionOrchestrator()
        result = await orchestrator._sync_probable_pitchers()
        logger.info(f"TEST: probable_pitchers completed with result: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.exception("TEST: probable_pitchers failed")
        return {"status": "error", "error": str(e)}

@router.post("/sync/all")
async def test_sync_all():
    """Test endpoint to trigger all sync jobs without auth."""
    try:
        logger.info("TEST: Triggering ALL sync jobs...")
        orchestrator = DailyIngestionOrchestrator()

        results = {}
        results["player_id_mapping"] = await orchestrator._sync_player_id_mapping()
        results["position_eligibility"] = await orchestrator._sync_position_eligibility()
        results["probable_pitchers"] = await orchestrator._sync_probable_pitchers()

        logger.info(f"TEST: All sync jobs completed with results: {results}")
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("TEST: Sync jobs failed")
        return {"status": "error", "error": str(e)}
