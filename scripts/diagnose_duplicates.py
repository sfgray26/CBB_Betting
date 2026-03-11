#!/usr/bin/env python3
"""
Diagnose duplicate games and predictions in the database.
"""

import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose_duplicates():
    """Check for duplicate games and predictions."""
    try:
        from backend.models import SessionLocal, Game, Prediction
        from sqlalchemy import func
        from collections import defaultdict
        
        db = SessionLocal()
        try:
            print("=" * 80)
            print("🔍 DUPLICATE DETECTION DIAGNOSTIC")
            print("=" * 80)
            
            # Check for duplicate games (same teams, same date)
            print("\n📊 GAMES: Looking for duplicates by matchup + date...")
            
            seven_days_ago = datetime.utcnow() - timedelta(days=7)
            recent_games = db.query(Game).filter(Game.game_date >= seven_days_ago).all()
            
            # Group by (date, home, away)
            game_groups = defaultdict(list)
            for g in recent_games:
                key = (g.game_date.date() if g.game_date else None, g.home_team, g.away_team)
                game_groups[key].append(g)
            
            duplicates = {k: v for k, v in game_groups.items() if len(v) > 1}
            
            if duplicates:
                print(f"\n🚨 FOUND {len(duplicates)} DUPLICATE GAME GROUPS:")
                for (date, home, away), games in sorted(duplicates.items(), key=lambda x: x[0][0] or datetime.min.date(), reverse=True):
                    print(f"\n  {date}: {away} @ {home}")
                    print(f"    Count: {len(games)} games")
                    for g in games:
                        ext_id = g.external_id or "NULL"
                        print(f"      ID={g.id}, external_id={ext_id[:20] if ext_id != 'NULL' else 'NULL'}, date={g.game_date}")
            else:
                print("\n✅ No duplicate games found")
            
            # Check predictions per game per tier
            print("\n" + "=" * 80)
            print("📊 PREDICTIONS: Count per game per tier...")
            print("=" * 80)
            
            from sqlalchemy import text
            
            result = db.execute(text("""
                SELECT 
                    g.home_team,
                    g.away_team,
                    DATE(g.game_date) as game_date,
                    p.run_tier,
                    COUNT(*) as pred_count
                FROM predictions p
                JOIN games g ON p.game_id = g.id
                WHERE g.game_date >= CURRENT_DATE - INTERVAL '7 days'
                GROUP BY g.home_team, g.away_team, DATE(g.game_date), p.run_tier
                HAVING COUNT(*) > 1
                ORDER BY game_date DESC, pred_count DESC
            """))
            
            rows = list(result)
            if rows:
                print(f"\n🚨 FOUND {len(rows)} DUPLICATE PREDICTION GROUPS:")
                for row in rows:
                    print(f"  {row.game_date}: {row.away_team} @ {row.home_team} | tier={row.run_tier} | count={row.pred_count}")
            else:
                print("\n✅ No duplicate predictions found")
            
            # Show recent game count
            print("\n" + "=" * 80)
            print("📊 RECENT GAMES SUMMARY")
            print("=" * 80)
            print(f"Total games in last 7 days: {len(recent_games)}")
            print(f"Unique matchups: {len(game_groups)}")
            print(f"Duplicate groups: {len(duplicates)}")
            
            # Sample of recent games
            print("\n📋 Sample of recent games (last 10):")
            for g in sorted(recent_games, key=lambda x: x.game_date or datetime.min, reverse=True)[:10]:
                preds = db.query(Prediction).filter(Prediction.game_id == g.id).all()
                print(f"  {g.game_date}: {g.away_team} @ {g.home_team} (ID={g.id}, preds={len(preds)})")
            
            print("\n" + "=" * 80)
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    diagnose_duplicates()
