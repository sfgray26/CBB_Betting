"""Data quality monitoring dashboard for fantasy baseball platform."""
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends
from sqlalchemy import func, and_, or_
from sqlalchemy.orm import Session

from backend.models import (
    get_db,
    PlayerProjection,
    MLBGameLog,
    MLBPlayerStats,
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


@router.get("/audit")
def run_data_quality_audit(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Run comprehensive data quality audit and return detailed issue list.
    
    This endpoint runs the same logic as scripts/audit_data_quality.py but
    returns JSON instead of CSV. Runs inside Railway network so DB access works.
    
    Returns:
        - issues: List of all data quality issues found
        - summary: Counts by priority/category
        - generated_at: Timestamp
    """
    all_issues = []
    
    # Priority levels
    P0, P1, P2, P3 = "P0", "P1", "P2", "P3"
    # Root cause categories
    A, B, C, D, E = "A", "B", "C", "D", "E"
    
    # === Audit 1: Null team fields ===
    null_team_players = db.query(PlayerProjection).filter(
        or_(
            PlayerProjection.team.is_(None),
            PlayerProjection.team == ""
        )
    ).limit(100).all()
    
    for player in null_team_players:
        all_issues.append({
            "issue_type": "null_team_field",
            "player_name": player.player_name,
            "player_id": player.player_id,
            "team": None,
            "pa": None,
            "expected_value": "<team_abbrev>",
            "actual_value": player.team or "NULL",
            "priority": P2,
            "category": C,
            "impact_score": 3.0,
            "description": "Player has null/empty team field"
        })
    
    # === Audit 2: Empty projections for active players ===
    week_ago = datetime.now(ZoneInfo("UTC")) - timedelta(days=7)
    
    players_with_empty_cats = db.query(PlayerProjection).filter(
        PlayerProjection.updated_at > week_ago,
        func.jsonb_typeof(PlayerProjection.cat_scores) == "object",
        func.jsonb_array_length(func.jsonb_object_keys(PlayerProjection.cat_scores)) == 0
    ).limit(100).all()
    
    for player in players_with_empty_cats:
        pa = player.sample_size or 0
        
        if pa > 50:  # Active player
            all_issues.append({
                "issue_type": "empty_projection_active_player",
                "player_name": player.player_name,
                "player_id": player.player_id,
                "team": player.team,
                "pa": pa,
                "expected_value": "{category: z_score, ...}",
                "actual_value": "{}",
                "priority": P1,
                "category": B,
                "impact_score": 7.0 + (1.0 if pa > 200 else 0.0),
                "description": f"Active player (PA={pa}) has empty cat_scores"
            })
        elif pa > 10:  # Bench/streamer
            all_issues.append({
                "issue_type": "empty_projection_bench_player",
                "player_name": player.player_name,
                "player_id": player.player_id,
                "team": player.team,
                "pa": pa,
                "expected_value": "{category: z_score, ...}",
                "actual_value": "{}",
                "priority": P2,
                "category": B,
                "impact_score": 4.0,
                "description": f"Bench player (PA={pa}) has empty cat_scores"
            })
    
    # === Audit 3: Missing Statcast data for active players ===
    today_date = date.today()
    week_ago_date = today_date - timedelta(days=7)
    
    # Get players with recent stats
    active_players = db.query(
        MLBPlayerStats.player_name,
        MLBPlayerStats.bdl_player_id,
        MLBPlayerStats.team,
        func.sum(MLBPlayerStats.pa).label("total_pa")
    ).filter(
        MLBPlayerStats.game_date > week_ago_date
    ).group_by(
        MLBPlayerStats.player_name,
        MLBPlayerStats.bdl_player_id,
        MLBPlayerStats.team
    ).having(
        func.sum(MLBPlayerStats.pa) > 50
    ).limit(200).all()
    
    for player in active_players:
        has_statcast = db.query(StatcastPerformance).filter(
            StatcastPerformance.player_id == str(player.bdl_player_id),
            StatcastPerformance.game_date > week_ago_date
        ).first()
        
        if not has_statcast:
            all_issues.append({
                "issue_type": "missing_statcast_data",
                "player_name": player.player_name,
                "player_id": str(player.bdl_player_id),
                "team": player.team,
                "pa": player.total_pa,
                "expected_value": "xwOBA, barrel%, exit_velo",
                "actual_value": "NULL",
                "priority": P2,
                "category": B,
                "impact_score": 5.0,
                "description": f"Active player (PA={player.total_pa}) missing Statcast"
            })
    
    # === Audit 4: Matchup detection issues ===
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    games_today = db.query(MLBGameLog).filter(
        MLBGameLog.game_date == today_et
    ).limit(50).all()
    
    if len(games_today) >= 5:  # Only audit if games scheduled
        for game in games_today:
            home_team = game.home_team
            
            # Check if recent player data exists
            home_players = db.query(MLBPlayerStats).filter(
                MLBPlayerStats.team == home_team,
                MLBPlayerStats.game_date > today_et - timedelta(days=3)
            ).limit(5).all()
            
            if not home_players:
                all_issues.append({
                    "issue_type": "matchup_detection_false_negative",
                    "player_name": f"{home_team} players",
                    "player_id": "N/A",
                    "team": home_team,
                    "pa": None,
                    "expected_value": f"has_game=True (vs {game.away_team})",
                    "actual_value": "no recent stats found",
                    "priority": P0,
                    "category": A,
                    "impact_score": 9.0,
                    "description": f"Game scheduled today but no recent player data"
                })
    
    # Sort by priority then impact score
    priority_order = {P0: 0, P1: 1, P2: 2, P3: 3}
    all_issues.sort(
        key=lambda x: (priority_order.get(x["priority"], 4), -x["impact_score"])
    )
    
    # Calculate summary stats
    priority_counts = {P0: 0, P1: 0, P2: 0, P3: 0}
    category_counts = {A: 0, B: 0, C: 0, D: 0, E: 0}
    
    for issue in all_issues:
        priority_counts[issue["priority"]] = priority_counts.get(issue["priority"], 0) + 1
        category_counts[issue["category"]] = category_counts.get(issue["category"], 0) + 1
    
    return {
        "issues": all_issues,
        "summary": {
            "total_issues": len(all_issues),
            "by_priority": {k: v for k, v in priority_counts.items() if v > 0},
            "by_category": {k: v for k, v in category_counts.items() if v > 0},
            "top_5": all_issues[:5] if all_issues else []
        },
        "generated_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
    }
