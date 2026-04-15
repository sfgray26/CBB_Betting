"""
Task 26: Temporary admin endpoint to backfill ops, whip, and caught_stealing data.
REMOVE AFTER TASK 26 COMPLETE

Endpoints:
  POST /admin/backfill-ops-whip              -- OPS + WHIP + caught_stealing(=0 default)
  POST /admin/backfill-caught-stealing       -- caught_stealing=0 for NULL rows (safe default)
  POST /admin/backfill-cs-from-statcast      -- upgrade caught_stealing from Statcast truth

NSB = SB - CS requires real CS values. BDL does not return CS (cs field is always
null). pybaseball Statcast does return it via caught_stealing_2b, already ingested
into statcast_performances.cs. The `backfill-cs-from-statcast` endpoint joins
statcast_performances -> player_id_mapping (mlbam_id) -> mlb_player_stats
(bdl_player_id) on matching game_date and overwrites caught_stealing with the
Statcast value. The earlier =0 default is preserved for rows with no Statcast
match, which is correct (no CS event recorded).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.models import get_db

router = APIRouter()

@router.post("/admin/backfill-cs-from-statcast")
def backfill_cs_from_statcast(db: Session = Depends(get_db)):
    """
    Upgrade mlb_player_stats.caught_stealing using statcast_performances.cs
    as the truth source. BDL does not return CS; pybaseball does.

    Join path:
        statcast_performances.player_id (mlbam_id text)
          -> player_id_mapping.mlbam_id  (int, cast to text)
          -> mlb_player_stats.bdl_player_id = player_id_mapping.bdl_id
        on matching game_date.

    Only rows where Statcast recorded a CS event (cs > 0) are written. The
    prior =0 default is left intact for player-games with no Statcast record
    (correctly representing "no CS event").

    Returns before/after counts for auditability.
    """
    result = {
        "status": "success",
        "total_rows": 0,
        "cs_positive_before": 0,
        "cs_positive_after": 0,
        "rows_updated": 0,
        "statcast_cs_events": 0,
        "unmatched_statcast_events": 0,
    }
    try:
        result["total_rows"] = db.execute(
            text("SELECT COUNT(*) FROM mlb_player_stats")
        ).scalar()
        result["cs_positive_before"] = db.execute(
            text("SELECT COUNT(*) FROM mlb_player_stats WHERE caught_stealing > 0")
        ).scalar()
        result["statcast_cs_events"] = db.execute(
            text("SELECT COUNT(*) FROM statcast_performances WHERE cs > 0")
        ).scalar()

        update = db.execute(text("""
            UPDATE mlb_player_stats mps
            SET caught_stealing = sp.cs
            FROM statcast_performances sp
            JOIN player_id_mapping pim
              ON pim.mlbam_id IS NOT NULL
             AND pim.mlbam_id::text = sp.player_id
            WHERE mps.bdl_player_id = pim.bdl_id
              AND mps.game_date = sp.game_date
              AND sp.cs IS NOT NULL
              AND sp.cs > 0
              AND (mps.caught_stealing IS NULL OR mps.caught_stealing <> sp.cs)
        """))
        result["rows_updated"] = update.rowcount
        db.commit()

        result["cs_positive_after"] = db.execute(
            text("SELECT COUNT(*) FROM mlb_player_stats WHERE caught_stealing > 0")
        ).scalar()

        # Diagnostic: count Statcast CS events whose player we cannot map
        # back to a BDL id — reveals identity gaps, not data errors.
        result["unmatched_statcast_events"] = db.execute(text("""
            SELECT COUNT(*)
            FROM statcast_performances sp
            LEFT JOIN player_id_mapping pim
              ON pim.mlbam_id IS NOT NULL
             AND pim.mlbam_id::text = sp.player_id
            WHERE sp.cs > 0
              AND (pim.bdl_id IS NULL)
        """)).scalar()
    except Exception as e:
        db.rollback()
        result["status"] = "error"
        result["error"] = str(e)
    return result


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
