"""Data quality monitoring dashboard for fantasy baseball platform."""
from datetime import datetime, timedelta
from typing import Dict, Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import (
    get_db,
    PlayerProjection,
    MLBGameLog,
    StatcastPerformance,
    DataIngestionLog,
)

router = APIRouter(prefix="/api/admin/data-quality", tags=["admin"])


@router.get("/summary")
def get_data_quality_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Aggregate health metrics across all data sources.
    
    Returns 6 key metrics with red/yellow/green status indicators:
    1. Matchup detection rate (games today)
    2. Statcast coverage (% active players with xwOBA)
    3. Null field counts (players with missing data)
    4. Pipeline staleness (hours since last projection update)
    5. Data ingestion failure rate (last 7 days)
    6. Projection coverage (% players with cat_scores)
    """
    now = datetime.now(ZoneInfo("UTC"))
    
    # Metric 1: Matchup detection rate
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    games_today = db.query(func.count(MLBGameLog.id)).filter(
        MLBGameLog.game_date == today_et
    ).scalar() or 0
    
    # Metric 2: Statcast coverage
    active_players = db.query(func.count(PlayerProjection.id)).filter(
        PlayerProjection.updated_at > now - timedelta(days=7)
    ).scalar() or 0
    
    statcast_covered = db.query(
        func.count(func.distinct(StatcastPerformance.player_id))
    ).filter(
        StatcastPerformance.game_date > today_et - timedelta(days=7)
    ).scalar() or 0
    
    statcast_coverage_pct = (
        (statcast_covered / active_players * 100) if active_players > 0 else 0.0
    )
    
    # Metric 3: Null field counts
    null_teams = db.query(func.count(PlayerProjection.id)).filter(
        PlayerProjection.team.is_(None)
    ).scalar() or 0
    
    # Metric 4: Pipeline staleness
    last_projection_update = db.query(func.max(PlayerProjection.updated_at)).scalar()
    staleness_hours = (
        (now - last_projection_update).total_seconds() / 3600
        if last_projection_update
        else 999
    )
    
    # Metric 5: Data ingestion failure rate (last 7 days)
    total_jobs = db.query(func.count(DataIngestionLog.id)).filter(
        DataIngestionLog.run_at > now - timedelta(days=7)
    ).scalar() or 1
    
    failed_jobs = db.query(func.count(DataIngestionLog.id)).filter(
        DataIngestionLog.run_at > now - timedelta(days=7),
        DataIngestionLog.status == "failed",
    ).scalar() or 0
    
    failure_rate_pct = (failed_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
    
    # Metric 6: Projection coverage
    total_projections = db.query(func.count(PlayerProjection.id)).scalar() or 0
    
    # Count projections with empty cat_scores (JSONB equality check)
    empty_cat_scores = db.query(func.count(PlayerProjection.id)).filter(
        func.jsonb_typeof(PlayerProjection.cat_scores) == "object",
        func.jsonb_array_length(func.jsonb_object_keys(PlayerProjection.cat_scores)) == 0,
    ).scalar() or 0
    
    projection_coverage_pct = (
        ((total_projections - empty_cat_scores) / total_projections * 100)
        if total_projections > 0
        else 0.0
    )
    
    return {
        "matchup_detection": {
            "games_today": games_today,
            "status": (
                "green" if games_today > 10
                else "yellow" if games_today > 0
                else "red"
            ),
        },
        "statcast_coverage": {
            "pct": round(statcast_coverage_pct, 1),
            "covered": statcast_covered,
            "total": active_players,
            "status": (
                "green" if statcast_coverage_pct > 85
                else "yellow" if statcast_coverage_pct > 60
                else "red"
            ),
        },
        "null_fields": {
            "null_teams": null_teams,
            "status": (
                "green" if null_teams < 10
                else "yellow" if null_teams < 50
                else "red"
            ),
        },
        "pipeline_staleness": {
            "hours": round(staleness_hours, 1),
            "last_update": (
                last_projection_update.isoformat() if last_projection_update else None
            ),
            "status": (
                "green" if staleness_hours < 12
                else "yellow" if staleness_hours < 36
                else "red"
            ),
        },
        "ingestion_failure_rate": {
            "pct": round(failure_rate_pct, 1),
            "failed": failed_jobs,
            "total": total_jobs,
            "status": (
                "green" if failure_rate_pct < 5
                else "yellow" if failure_rate_pct < 15
                else "red"
            ),
        },
        "projection_coverage": {
            "pct": round(projection_coverage_pct, 1),
            "with_projections": total_projections - empty_cat_scores,
            "total": total_projections,
            "status": (
                "green" if projection_coverage_pct > 90
                else "yellow" if projection_coverage_pct > 80
                else "red"
            ),
        },
        "generated_at": now.isoformat(),
    }
