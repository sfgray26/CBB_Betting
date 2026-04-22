"""
Phase 2: Systematic Data Quality Audit Script

Queries production database to catalog concrete data quality issues:
- Null team fields
- Empty projections (cat_scores) for active players
- Zero z_scores with non-zero stats
- Missing Statcast data for rostered players
- Matchup detection false negatives

Outputs: CSV report with P0-P3 priority and A-E root cause categories

Usage:
    # Local dev (requires Railway tunnel or local DB)
    python scripts/audit_data_quality.py
    
    # Railway production
    railway run python scripts/audit_data_quality.py
"""

import csv
import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models import (
    SessionLocal,
    PlayerProjection,
    MLBGameLog,
    MLBPlayerStats,
    StatcastPerformance,
)
from sqlalchemy import func, and_, or_


# Priority levels
PRIORITY_P0 = "P0"  # Critical - blocks core functionality
PRIORITY_P1 = "P1"  # High - wrong decisions
PRIORITY_P2 = "P2"  # Medium - missing features
PRIORITY_P3 = "P3"  # Low - cosmetic

# Root cause categories
CATEGORY_A = "A"  # Single point of failure
CATEGORY_B = "B"  # Data source gaps
CATEGORY_C = "C"  # Integration bugs
CATEGORY_D = "D"  # Schema design
CATEGORY_E = "E"  # Error handling


def audit_null_teams(db) -> List[Dict[str, Any]]:
    """Find players with null team field."""
    issues = []
    
    null_team_players = db.query(PlayerProjection).filter(
        or_(
            PlayerProjection.team.is_(None),
            PlayerProjection.team == ""
        )
    ).all()
    
    for player in null_team_players:
        issues.append({
            "issue_type": "null_team_field",
            "player_name": player.player_name,
            "player_id": player.player_id,
            "team": None,
            "pa": None,
            "expected_value": "<team_abbrev>",
            "actual_value": player.team or "NULL",
            "priority": PRIORITY_P2,
            "category": CATEGORY_C,
            "impact_score": 3.0,
            "description": "Player has null/empty team field - breaks team-based filters"
        })
    
    return issues


def audit_empty_projections(db) -> List[Dict[str, Any]]:
    """Find active players with empty cat_scores but significant playing time."""
    issues = []
    
    # Query players updated recently (active season) with empty cat_scores
    week_ago = datetime.now(ZoneInfo("UTC")) - timedelta(days=7)
    
    players_with_empty_cats = db.query(PlayerProjection).filter(
        PlayerProjection.updated_at > week_ago,
        func.jsonb_typeof(PlayerProjection.cat_scores) == "object",
        PlayerProjection.cat_scores == text("'{}'::jsonb")
    ).all()
    
    for player in players_with_empty_cats:
        # Check if player has significant playing time (proxy for active roster)
        pa = player.sample_size or 0
        
        if pa > 50:  # Active player threshold
            issues.append({
                "issue_type": "empty_projection_active_player",
                "player_name": player.player_name,
                "player_id": player.player_id,
                "team": player.team,
                "pa": pa,
                "expected_value": "{category: z_score, ...}",
                "actual_value": "{}",
                "priority": PRIORITY_P1,
                "category": CATEGORY_B,
                "impact_score": 7.0,
                "description": f"Active player (PA={pa}) has empty cat_scores - breaks waiver scoring"
            })
        elif pa > 10:  # Bench/streamer threshold
            issues.append({
                "issue_type": "empty_projection_bench_player",
                "player_name": player.player_name,
                "player_id": player.player_id,
                "team": player.team,
                "pa": pa,
                "expected_value": "{category: z_score, ...}",
                "actual_value": "{}",
                "priority": PRIORITY_P2,
                "category": CATEGORY_B,
                "impact_score": 4.0,
                "description": f"Bench player (PA={pa}) has empty cat_scores - limits streamer pool"
            })
    
    return issues


def audit_zero_z_scores(db) -> List[Dict[str, Any]]:
    """Find players with zero z_score but non-zero stats (computation failure)."""
    issues = []
    
    week_ago = datetime.now(ZoneInfo("UTC")) - timedelta(days=7)
    
    # Query players with zero composite_z but non-zero counting stats
    zero_z_players = db.query(PlayerProjection).filter(
        PlayerProjection.updated_at > week_ago,
        PlayerProjection.woba == 0.320,  # Default value indicates no real data
        or_(
            PlayerProjection.hr > 0,
            PlayerProjection.r > 0,
            PlayerProjection.rbi > 0
        )
    ).all()
    
    for player in zero_z_players:
        issues.append({
            "issue_type": "zero_z_score_with_stats",
            "player_name": player.player_name,
            "player_id": player.player_id,
            "team": player.team,
            "pa": player.sample_size,
            "expected_value": "z_score != 0.0",
            "actual_value": f"woba={player.woba}, hr={player.hr}",
            "priority": PRIORITY_P1,
            "category": CATEGORY_D,
            "impact_score": 6.0,
            "description": "Player has stats but zero/default z_score - computation failure"
        })
    
    return issues


def audit_missing_statcast(db) -> List[Dict[str, Any]]:
    """Find active players missing Statcast data (name mismatch or qual threshold)."""
    issues = []
    
    week_ago_date = date.today() - timedelta(days=7)
    
    # Get players with recent stats (active)
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
        func.sum(MLBPlayerStats.pa) > 50  # Active player threshold
    ).all()
    
    # Check which ones have no Statcast data
    for player in active_players:
        has_statcast = db.query(StatcastPerformance).filter(
            StatcastPerformance.player_id == str(player.bdl_player_id),
            StatcastPerformance.game_date > week_ago_date
        ).first()
        
        if not has_statcast:
            issues.append({
                "issue_type": "missing_statcast_data",
                "player_name": player.player_name,
                "player_id": str(player.bdl_player_id),
                "team": player.team,
                "pa": player.total_pa,
                "expected_value": "xwOBA, barrel%, exit_velo",
                "actual_value": "NULL",
                "priority": PRIORITY_P2,
                "category": CATEGORY_B,
                "impact_score": 5.0,
                "description": f"Active player (PA={player.total_pa}) missing Statcast - likely name mismatch or qual threshold"
            })
    
    return issues


def audit_matchup_detection(db) -> List[Dict[str, Any]]:
    """Find games today where players would show 'no matchup' incorrectly."""
    issues = []
    
    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    
    # Get games scheduled for today
    games_today = db.query(MLBGameLog).filter(
        MLBGameLog.game_date == today_et
    ).all()
    
    if len(games_today) < 5:
        # Not enough games to audit (might be off-day or early season)
        return issues
    
    # Check if recent player stats exist for these teams
    for game in games_today:
        home_team = game.home_team
        away_team = game.away_team
        
        # Query recent player stats for home team
        home_players = db.query(MLBPlayerStats).filter(
            MLBPlayerStats.team == home_team,
            MLBPlayerStats.game_date > today_et - timedelta(days=3)
        ).limit(5).all()
        
        if not home_players:
            issues.append({
                "issue_type": "matchup_detection_false_negative",
                "player_name": f"{home_team} players",
                "player_id": "N/A",
                "team": home_team,
                "pa": None,
                "expected_value": f"has_game=True (vs {away_team})",
                "actual_value": "no recent stats found",
                "priority": PRIORITY_P0,
                "category": CATEGORY_A,
                "impact_score": 9.0,
                "description": f"Game scheduled today but no recent player data for {home_team}"
            })
    
    return issues


def calculate_impact_score(issue: Dict[str, Any]) -> float:
    """
    Calculate numeric impact score (0-10) based on priority and specifics.
    Higher = more urgent to fix.
    """
    priority_base = {
        PRIORITY_P0: 9.0,
        PRIORITY_P1: 7.0,
        PRIORITY_P2: 4.0,
        PRIORITY_P3: 1.0
    }
    
    base = priority_base.get(issue["priority"], 5.0)
    
    # Adjust based on PA (more active player = higher impact)
    if issue.get("pa"):
        if issue["pa"] > 200:
            base += 1.0
        elif issue["pa"] < 20:
            base -= 1.0
    
    return min(10.0, max(0.0, base))


def main():
    """Run all audit checks and output CSV report."""
    print("🔍 Phase 2: Data Quality Audit")
    print("=" * 60)
    
    db = SessionLocal()
    
    try:
        all_issues = []
        
        print("\n1️⃣  Auditing null team fields...")
        null_teams = audit_null_teams(db)
        all_issues.extend(null_teams)
        print(f"   Found {len(null_teams)} issues")
        
        print("\n2️⃣  Auditing empty projections...")
        empty_projs = audit_empty_projections(db)
        all_issues.extend(empty_projs)
        print(f"   Found {len(empty_projs)} issues")
        
        print("\n3️⃣  Auditing zero z-scores with stats...")
        zero_z = audit_zero_z_scores(db)
        all_issues.extend(zero_z)
        print(f"   Found {len(zero_z)} issues")
        
        print("\n4️⃣  Auditing missing Statcast data...")
        missing_statcast = audit_missing_statcast(db)
        all_issues.extend(missing_statcast)
        print(f"   Found {len(missing_statcast)} issues")
        
        print("\n5️⃣  Auditing matchup detection...")
        matchup_issues = audit_matchup_detection(db)
        all_issues.extend(matchup_issues)
        print(f"   Found {len(matchup_issues)} issues")
        
        # Recalculate impact scores
        for issue in all_issues:
            issue["impact_score"] = calculate_impact_score(issue)
        
        # Sort by priority then impact score
        priority_order = {PRIORITY_P0: 0, PRIORITY_P1: 1, PRIORITY_P2: 2, PRIORITY_P3: 3}
        all_issues.sort(
            key=lambda x: (priority_order.get(x["priority"], 4), -x["impact_score"])
        )
        
        # Output CSV
        output_dir = Path(__file__).parent.parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        output_file = output_dir / f"data_quality_audit_{timestamp}.csv"
        
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            if all_issues:
                fieldnames = [
                    "priority", "category", "impact_score", "issue_type",
                    "player_name", "player_id", "team", "pa",
                    "expected_value", "actual_value", "description"
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_issues)
        
        print("\n" + "=" * 60)
        print(f"✅ Audit complete: {len(all_issues)} total issues")
        print(f"📊 Report saved to: {output_file}")
        print("\nBreakdown by priority:")
        for priority in [PRIORITY_P0, PRIORITY_P1, PRIORITY_P2, PRIORITY_P3]:
            count = sum(1 for i in all_issues if i["priority"] == priority)
            if count > 0:
                print(f"   {priority}: {count} issues")
        
        print("\nBreakdown by category:")
        for category in [CATEGORY_A, CATEGORY_B, CATEGORY_C, CATEGORY_D, CATEGORY_E]:
            count = sum(1 for i in all_issues if i["category"] == category)
            if count > 0:
                cat_name = {
                    CATEGORY_A: "Single point of failure",
                    CATEGORY_B: "Data source gaps",
                    CATEGORY_C: "Integration bugs",
                    CATEGORY_D: "Schema design",
                    CATEGORY_E: "Error handling"
                }[category]
                print(f"   {category} ({cat_name}): {count} issues")
        
        # Show top 5 highest impact
        if all_issues:
            print("\n🔥 Top 5 highest impact issues:")
            for i, issue in enumerate(all_issues[:5], 1):
                print(f"   {i}. [{issue['priority']}/{issue['category']}] "
                      f"{issue['issue_type']}: {issue['player_name']} "
                      f"(impact: {issue['impact_score']:.1f})")
        
    finally:
        db.close()


if __name__ == "__main__":
    main()
