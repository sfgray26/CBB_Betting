"""
Admin endpoint to check database state without authentication.
TEMPORARY - Remove after database verification is complete.
"""

from fastapi import APIRouter
from backend.models import (
    SessionLocal, 
    PlayerIDMapping, 
    PositionEligibility, 
    ProbablePitcherSnapshot,
    MLBPlayerStats,
    PlayerProjection,
    MLBGameLog,
    StatcastPerformance,
    PlayerRollingStats,
    PlayerScore,
    SimulationResult,
    DecisionResult
)
from sqlalchemy import func
from datetime import date
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/verify-db-state")
async def verify_database_state():
    """
    Verify actual database row counts and recency after sync job execution.
    Expanded to include Phase 2 tables (Tasks 4-11).
    """
    try:
        db = SessionLocal()

        # Phase 1 Tables
        player_id_total = db.query(PlayerIDMapping).count()
        position_total = db.query(PositionEligibility).count()
        pitchers_total = db.query(ProbablePitcherSnapshot).count()

        # Phase 2 Tables
        stats_total = db.query(MLBPlayerStats).count()
        projections_total = db.query(PlayerProjection).count()
        logs_total = db.query(MLBGameLog).count()
        statcast_total = db.query(StatcastPerformance).count()
        rolling_total = db.query(PlayerRollingStats).count()
        score_total = db.query(PlayerScore).count()
        sim_total = db.query(SimulationResult).count()
        decision_total = db.query(DecisionResult).count()

        # Recency check
        latest_stat_date = db.query(func.max(MLBPlayerStats.game_date)).scalar()
        latest_sim_date = db.query(func.max(SimulationResult.as_of_date)).scalar()

        db.close()

        return {
            "status": "success",
            "timestamp": date.today().isoformat(),
            "latest_data_dates": {
                "mlb_player_stats": latest_stat_date.isoformat() if latest_stat_date else None,
                "simulation_results": latest_sim_date.isoformat() if latest_sim_date else None
            },
            "phase_1": {
                "player_id_mapping": player_id_total,
                "position_eligibility": position_total,
                "probable_pitchers": pitchers_total
            },
            "phase_2": {
                "mlb_player_stats": stats_total,
                "player_projections": projections_total,
                "mlb_game_logs": logs_total,
                "statcast_performances": statcast_total,
                "player_rolling_stats": rolling_total,
                "player_scores": score_total,
                "simulation_results": sim_total,
                "decision_results": decision_total
            }
        }
    except Exception as e:
        logger.exception("Database verification failed")
        return {"status": "error", "error": str(e)}
