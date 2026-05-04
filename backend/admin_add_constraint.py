"""One-time migration: Add missing _pim_yahoo_key_uc constraint"""

from fastapi import APIRouter
from sqlalchemy import text
from backend.models import SessionLocal, engine
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/migrate/add-yahoo-key-constraint")
async def add_yahoo_key_constraint():
    """
    One-time migration: Add _pim_yahoo_key_uc unique constraint.
    This endpoint should be removed after the constraint is added.
    """
    try:
        with engine.connect() as conn:
            # Check if constraint exists
            result = conn.execute(text(
                "SELECT COUNT(*) FROM pg_constraint WHERE conname = '_pim_yahoo_key_uc'"
            ))
            exists = result.fetchone()[0]

            if not exists:
                logger.info("Adding _pim_yahoo_key_uc constraint...")
                conn.execute(text(
                    "ALTER TABLE player_id_mapping ADD CONSTRAINT _pim_yahoo_key_uc UNIQUE (yahoo_key)"
                ))
                conn.commit()
                return {
                    "status": "success",
                    "message": "Added _pim_yahoo_key_uc constraint"
                }
            else:
                return {
                    "status": "skipped",
                    "message": "Constraint already exists"
                }
    except Exception as exc:
        logger.error("Failed to add constraint: %s", exc)
        return {"status": "error", "message": str(exc)}
