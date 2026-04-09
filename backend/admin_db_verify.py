"""
Admin endpoint to check database state without authentication.
TEMPORARY - Remove after database verification is complete.
"""

from fastapi import APIRouter
from backend.models import SessionLocal, PlayerIDMapping, PositionEligibility, ProbablePitcherSnapshot
from sqlalchemy import func
from datetime import date
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/verify-db-state")
async def verify_database_state():
    """
    Verify actual database row counts after sync job execution.
    REMOVE AFTER VERIFICATION COMPLETE!
    """
    try:
        db = SessionLocal()

        # PLAYER ID MAPPING
        player_id_total = db.query(PlayerIDMapping).count()
        player_id_with_yahoo = db.query(PlayerIDMapping).filter(PlayerIDMapping.yahoo_key.isnot(None)).count()
        player_id_with_mlbam = db.query(PlayerIDMapping).filter(PlayerIDMapping.mlbam_id.isnot(None)).count()

        # POSITION ELIGIBILITY
        position_total = db.query(PositionEligibility).count()
        position_with_cf = db.query(PositionEligibility).filter(PositionEligibility.can_play_cf == True).count()
        position_with_lf = db.query(PositionEligibility).filter(PositionEligibility.can_play_lf == True).count()
        position_with_rf = db.query(PositionEligibility).filter(PositionEligibility.can_play_rf == True).count()

        # PROBABLE PITCHERS
        pitchers_total = db.query(ProbablePitcherSnapshot).count()
        today = date.today()
        pitchers_today = db.query(ProbablePitcherSnapshot).filter(
            ProbablePitcherSnapshot.game_date >= today
        ).count()

        db.close()

        return {
            "status": "success",
            "timestamp": date.today().isoformat(),
            "tables": {
                "player_id_mapping": {
                    "total_rows": player_id_total,
                    "with_yahoo_key": player_id_with_yahoo,
                    "with_mlbam_id": player_id_with_mlbam,
                    "yahoo_key_percentage": round(player_id_with_yahoo * 100 / player_id_total, 1) if player_id_total > 0 else 0,
                    "status": "HAS_DATA" if player_id_total > 0 else "EMPTY"
                },
                "position_eligibility": {
                    "total_rows": position_total,
                    "can_play_cf": position_with_cf,
                    "can_play_lf": position_with_lf,
                    "can_play_rf": position_with_rf,
                    "status": "HAS_DATA" if position_total >= 700 else "EMPTY",
                    "meets_threshold": position_total >= 700
                },
                "probable_pitchers": {
                    "total_rows": pitchers_total,
                    "today_or_later": pitchers_today,
                    "status": "HAS_DATA" if pitchers_total > 0 else "EMPTY"
                }
            },
            "success_criteria": {
                "position_eligibility_700_plus": position_total >= 700,
                "multi_eligibility_present": (position_with_cf > 0 and position_with_lf > 0),
                "probable_pitchers_recent": pitchers_today > 0
            },
            "overall_status": "PASS" if (player_id_total > 0 and position_total >= 700) else "FAIL"
        }
    except Exception as e:
        logger.exception("Database verification failed")
        return {"status": "error", "error": str(e)}
