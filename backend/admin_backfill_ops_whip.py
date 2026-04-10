"""
Task 26: Temporary admin endpoint to backfill ops and whip data
REMOVE AFTER TASK 26 COMPLETE
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.models import get_db

router = APIRouter()

@router.post("/admin/backfill-ops-whip")
def backfill_ops_whip(db: Session = Depends(get_db)):
    """
    Backfill ops and whip data for mlb_player_stats.
    POST /admin/backfill-ops-whip
    """

    result = {
        "status": "success",
        "ops_updated": 0,
        "whip_updated": 0,
        "initial_ops_null": 0,
        "initial_whip_null": 0,
        "final_ops_null": 0,
        "final_whip_null": 0,
        "total_rows": 0
    }

    try:
        # Get initial NULL counts
        result["initial_ops_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
        )).scalar()

        result["initial_whip_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
        )).scalar()

        result["total_rows"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats"
        )).scalar()

        # Backfill ops
        ops_result = db.execute(text("""
            UPDATE mlb_player_stats
            SET ops = obp + slg
            WHERE ops IS NULL
              AND obp IS NOT NULL
              AND slg IS NOT NULL
        """))
        result["ops_updated"] = ops_result.rowcount

        # Backfill whip
        whip_result = db.execute(text("""
            UPDATE mlb_player_stats
            SET whip = (bb_allowed + h_allowed)::numeric /
                      NULLIF(
                          CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
                          CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0,
                          0
                      )
            WHERE whip IS NULL
              AND bb_allowed IS NOT NULL
              AND h_allowed IS NOT NULL
              AND innings_pitched IS NOT NULL
              AND innings_pitched != ''
        """))
        result["whip_updated"] = whip_result.rowcount

        db.commit()

        # Get final NULL counts
        result["final_ops_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
        )).scalar()

        result["final_whip_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
        )).scalar()

    except Exception as e:
        db.rollback()
        result["status"] = "error"
        result["error"] = str(e)

    return result
