"""
Admin endpoint for ERA diagnostics.

This endpoint provides ERA investigation results via HTTP API.
Usage: GET /admin/diagnose-era
"""

from fastapi import APIRouter, HTTPException
from sqlalchemy import text
from backend.models import SessionLocal

router = APIRouter()

@router.get("/diagnose-era")
async def diagnose_era():
    """
    Admin endpoint to diagnose ERA values in mlb_player_stats table.

    Returns comprehensive ERA distribution analysis and identifies problematic values.
    """
    db = SessionLocal()

    try:
        report = {
            "query1_overall_distribution": {},
            "query2_high_era_values": [],
            "query3_low_era_values": [],
            "query4_specific_era_1726": [],
            "summary": {}
        }

        # Query 1: Overall ERA distribution
        result = db.execute(text("""
            SELECT
                COUNT(*) as total_rows,
                COUNT(era) as rows_with_era,
                COUNT(*) - COUNT(era) as rows_null_era,
                MIN(era) as min_era,
                MAX(era) as max_era,
                ROUND(AVG(era), 3) as avg_era,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY era), 3) as median_era
            FROM mlb_player_stats
        """)).fetchone()

        report["query1_overall_distribution"] = {
            "total_rows": result.total_rows,
            "rows_with_era": result.rows_with_era,
            "rows_null_era": result.rows_null_era,
            "min_era": float(result.min_era) if result.min_era else None,
            "max_era": float(result.max_era) if result.max_era else None,
            "avg_era": float(result.avg_era) if result.avg_era else None,
            "median_era": float(result.median_era) if result.median_era else None
        }

        # Query 2: Extremely high ERA values (> 50)
        high_era = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                opponent_team
            FROM mlb_player_stats
            WHERE era IS NOT NULL AND era > 50
            ORDER BY era DESC
            LIMIT 10
        """)).fetchall()

        for row in high_era:
            calculated_era = (row.earned_runs / row.innings_pitched) * 9 if row.innings_pitched and row.innings_pitched > 0 else None
            report["query2_high_era_values"].append({
                "bdl_player_id": row.bdl_player_id,
                "era": float(row.era) if row.era else None,
                "earned_runs": row.earned_runs,
                "innings_pitched": row.innings_pitched,
                "game_id": row.game_id,
                "opponent_team": row.opponent_team,
                "calculated_era": calculated_era,
                "calculation_match": abs(calculated_era - row.era) < 0.01 if calculated_era else None
            })

        # Query 3: Extremely low ERA values (< 1.0)
        low_era = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                opponent_team
            FROM mlb_player_stats
            WHERE era IS NOT NULL AND era < 1.0
            ORDER BY era ASC
            LIMIT 10
        """)).fetchall()

        for row in low_era:
            report["query3_low_era_values"].append({
                "bdl_player_id": row.bdl_player_id,
                "era": float(row.era) if row.era else None,
                "earned_runs": row.earned_runs,
                "innings_pitched": row.innings_pitched,
                "game_id": row.game_id,
                "opponent_team": row.opponent_team
            })

        # Query 4: Check for ERA = 1.726
        specific_era = db.execute(text("""
            SELECT
                bdl_player_id,
                era,
                earned_runs,
                innings_pitched,
                game_id,
                opponent_team
            FROM mlb_player_stats
            WHERE era = 1.726
        """)).fetchall()

        for row in specific_era:
            report["query4_specific_era_1726"].append({
                "bdl_player_id": row.bdl_player_id,
                "era": float(row.era) if row.era else None,
                "earned_runs": row.earned_runs,
                "innings_pitched": row.innings_pitched,
                "game_id": row.game_id,
                "opponent_team": row.opponent_team
            })

        # Query 5: Summary counts
        count_gt_100 = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 100")).scalar()
        count_gt_50 = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era > 50")).scalar()
        count_lt_1 = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era < 1")).scalar()
        count_null = db.execute(text("SELECT COUNT(*) FROM mlb_player_stats WHERE era IS NULL")).scalar()

        report["summary"] = {
            "count_gt_100": count_gt_100,
            "count_gt_50": count_gt_50,
            "count_lt_1": count_lt_1,
            "count_null": count_null,
            "recommendation": ""
        }

        # Generate recommendation
        if count_gt_100 > 0:
            report["summary"]["recommendation"] = f"ACTION REQUIRED: {count_gt_100} rows have impossible ERA (> 100)"
        elif count_gt_50 > 0:
            report["summary"]["recommendation"] = f"REVIEW RECOMMENDED: {count_gt_50} rows have very high ERA (> 50)"
        else:
            report["summary"]["recommendation"] = "NO IMPOSSIBLE ERA VALUES FOUND - All ERA values in valid range"

        return report

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"ERA diagnostic failed: {str(e)}")
    finally:
        db.close()
