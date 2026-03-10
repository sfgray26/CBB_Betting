#!/usr/bin/env python3
"""
Emergency diagnostic: Check what games are actually in the database.
Use this to verify game dates and detect stale data.
"""

import os
import sys
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose_games():
    """Check all games in database for date issues."""
    try:
        from backend.models import SessionLocal, Prediction, Game
        from sqlalchemy import func
        
        db = SessionLocal()
        try:
            today = date.today()
            yesterday = today - timedelta(days=1)
            tomorrow = today + timedelta(days=1)
            
            print("=" * 80)
            print("🚨 EMERGENCY GAME DIAGNOSTIC")
            print("=" * 80)
            print(f"Today (UTC): {today}")
            print(f"Current time (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            # Check game dates in database
            print("📅 Game Date Distribution:")
            
            today_games = db.query(Game).filter(
                func.date(Game.game_date) == today
            ).count()
            
            yesterday_games = db.query(Game).filter(
                func.date(Game.game_date) == yesterday
            ).count()
            
            tomorrow_games = db.query(Game).filter(
                func.date(Game.game_date) == tomorrow
            ).count()
            
            total_games = db.query(Game).count()
            
            print(f"  Today ({today}):     {today_games} games")
            print(f"  Yesterday ({yesterday}): {yesterday_games} games")
            print(f"  Tomorrow ({tomorrow}):  {tomorrow_games} games")
            print(f"  Total in DB:       {total_games} games")
            print()
            
            # List all games for today
            if today_games > 0:
                print(f"🏀 Games scheduled for TODAY ({today}):")
                games_today = db.query(Game).filter(
                    func.date(Game.game_date) == today
                ).order_by(Game.game_date).all()
                
                for g in games_today:
                    print(f"  • {g.away_team} @ {g.home_team}")
                    print(f"    Time: {g.game_date.strftime('%H:%M UTC') if g.game_date else 'Unknown'}")
                    
                    # Check if this game has predictions
                    preds = db.query(Prediction).filter(Prediction.game_id == g.id).all()
                    if preds:
                        for p in preds:
                            print(f"    Prediction: {p.verdict} (Edge: {p.edge_conservative:.2%})")
                    print()
            
            # Check for suspicious games (famous matchups that might be fake)
            print("🔍 Checking for suspicious/fake games:")
            suspicious_patterns = [
                ("Duke", "UNC"),
                ("North Carolina", "Duke"),
                ("Kentucky", "Louisville"),
                ("Kansas", "Missouri"),
            ]
            
            for team1, team2 in suspicious_patterns:
                suspicious = db.query(Game).filter(
                    ((Game.home_team.ilike(f"%{team1}%")) & (Game.away_team.ilike(f"%{team2}%"))) |
                    ((Game.home_team.ilike(f"%{team2}%")) & (Game.away_team.ilike(f"%{team1}%")))
                ).all()
                
                for g in suspicious:
                    print(f"  ⚠️  FOUND: {g.away_team} @ {g.home_team}")
                    print(f"      Date: {g.game_date}")
                    print(f"      Source: {g.data_source}")
            
            print()
            
            # Check oldest games in database (stale data detection)
            print("⏰ Oldest games in database (stale data check):")
            old_games = db.query(Game).order_by(Game.game_date.asc()).limit(5).all()
            for g in old_games:
                print(f"  • {g.away_team} @ {g.home_team} — {g.game_date}")
            
            print()
            
            # Check newest games
            print("⏰ Newest games in database:")
            new_games = db.query(Game).order_by(Game.game_date.desc()).limit(5).all()
            for g in new_games:
                print(f"  • {g.away_team} @ {g.home_team} — {g.game_date}")
            
            print()
            print("=" * 80)
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_games()
