"""
Task 26: Temporary admin endpoint to backfill ops, whip, and caught_stealing data.
REMOVE AFTER TASK 26 COMPLETE

Endpoints:
  POST /admin/backfill-ops-whip         -- OPS + WHIP + caught_stealing in one pass
  POST /admin/backfill-caught-stealing  -- caught_stealing only (faster, targeted)
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.models import get_db

router = APIRouter()

@router.post("/admin/backfill-caught-stealing")
def backfill_caught_stealing(db: Session = Depends(get_db)):
    """
    Backfill caught_stealing = 0 for all NULL rows in mlb_player_stats.

    BDL /mlb/v1/stats does not return a caught_stealing field (cs is always null
    in BDL responses). The daily ingestion now defaults to 0 for new rows, but
    existing rows have NULL. Setting NULL -> 0 enables NSB = SB - CS calculations.

    POST /admin/backfill-caught-stealing
    """
    result = {
        "status": "success",
        "initial_null": 0,
        "rows_updated": 0,
        "final_null": 0,
        "total_rows": 0,
    }
    try:
        result["total_rows"] = db.execute(
            text("SELECT COUNT(*) FROM mlb_player_stats")
        ).scalar()
        result["initial_null"] = db.execute(
            text("SELECT COUNT(*) FROM mlb_player_stats WHERE caught_stealing IS NULL")
        ).scalar()

        update_result = db.execute(text("""
            UPDATE mlb_player_stats
            SET caught_stealing = 0
            WHERE caught_stealing IS NULL
        """))
        result["rows_updated"] = update_result.rowcount
        db.commit()

        result["final_null"] = db.execute(
            text("SELECT COUNT(*) FROM mlb_player_stats WHERE caught_stealing IS NULL")
        ).scalar()
    except Exception as e:
        db.rollback()
        result["status"] = "error"
        result["error"] = str(e)
    return result


@router.post("/admin/backfill-ops-whip")
def backfill_ops_whip(db: Session = Depends(get_db)):
    """
    Backfill ops, whip, and caught_stealing for mlb_player_stats.

    - OPS  = OBP + SLG  (computed from components already in the row)
    - WHIP = (BB_allowed + H_allowed) / IP  (skips 0-IP rows)
    - caught_stealing = 0 when NULL (BDL does not return this field)

    POST /admin/backfill-ops-whip
    """

    result = {
        "status": "success",
        "ops_updated": 0,
        "whip_updated": 0,
        "whip_skipped_zero_ip": 0,
        "caught_stealing_updated": 0,
        "initial_ops_null": 0,
        "initial_whip_null": 0,
        "initial_cs_null": 0,
        "final_ops_null": 0,
        "final_whip_null": 0,
        "final_cs_null": 0,
        "total_rows": 0,
    }

    try:
        # Get initial NULL counts
        result["initial_ops_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
        )).scalar()

        result["initial_whip_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
        )).scalar()

        result["initial_cs_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE caught_stealing IS NULL"
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
            SET whip = (walks_allowed + hits_allowed)::numeric /
                      NULLIF(
                          CAST(SPLIT_PART(innings_pitched, '.', 1) AS NUMERIC) +
                          CAST(SPLIT_PART(innings_pitched, '.', 2) AS NUMERIC) / 3.0,
                          0
                      )
            WHERE whip IS NULL
              AND walks_allowed IS NOT NULL
              AND hits_allowed IS NOT NULL
              AND innings_pitched IS NOT NULL
              AND innings_pitched != ''
              AND innings_pitched NOT IN ('0.0', '0', '0.00')
        """))
        result["whip_updated"] = whip_result.rowcount

        # Count zero-IP rows that were skipped (diagnostic only)
        result["whip_skipped_zero_ip"] = db.execute(text("""
            SELECT COUNT(*) FROM mlb_player_stats
            WHERE whip IS NULL
              AND walks_allowed IS NOT NULL
              AND hits_allowed IS NOT NULL
              AND innings_pitched IN ('0.0', '0', '0.00')
        """)).scalar()

        # Backfill caught_stealing = 0 where NULL
        # BDL /mlb/v1/stats does not return caught_stealing (cs is always null
        # in BDL responses). 0 is the correct default to enable NSB = SB - CS.
        cs_result = db.execute(text("""
            UPDATE mlb_player_stats
            SET caught_stealing = 0
            WHERE caught_stealing IS NULL
        """))
        result["caught_stealing_updated"] = cs_result.rowcount

        db.commit()

        # Get final NULL counts
        result["final_ops_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE ops IS NULL"
        )).scalar()

        result["final_whip_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE whip IS NULL"
        )).scalar()

        result["final_cs_null"] = db.execute(text(
            "SELECT COUNT(*) FROM mlb_player_stats WHERE caught_stealing IS NULL"
        )).scalar()

    except Exception as e:
        db.rollback()
        result["status"] = "error"
        result["error"] = str(e)

    return result
