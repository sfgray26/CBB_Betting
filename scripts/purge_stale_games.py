#!/usr/bin/env python3
"""
Emergency purge: Remove stale games (games that already happened but have today's date).

This fixes the issue where games from Feb 28 (like Duke vs UNC) are incorrectly 
labeled as today's games.
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def purge_stale_games(dry_run=True):
    """
    Remove games that:
    1. Are labeled as today's games
    2. But have game_date in the past (already happened)
    3. OR are famous matchups that shouldn't be today (Duke vs UNC)
    """
    try:
        from backend.models import SessionLocal, Game, Prediction
        from sqlalchemy import func
        
        db = SessionLocal()
        try:
            today = datetime.utcnow().date()
            six_hours_ago = datetime.utcnow() - timedelta(hours=6)
            
            print("=" * 80)
            print("🧹 EMERGENCY STALE GAME PURGE")
            print("=" * 80)
            print(f"Today: {today}")
            print(f"Cutoff (6h ago): {six_hours_ago}")
            print(f"Mode: {'DRY RUN' if dry_run else 'LIVE DELETE'}")
            print()
            
            # Find stale games (today's date but past time)
            stale_games = db.query(Game).filter(
                func.date(Game.game_date) == today,
                Game.game_date < six_hours_ago
            ).all()
            
            print(f"Found {len(stale_games)} stale games (today's date, past time):")
            for g in stale_games:
                print(f"  • {g.away_team} @ {g.home_team} at {g.game_date}")
            
            # Find suspicious Duke vs UNC games labeled as today
            suspicious = db.query(Game).filter(
                func.date(Game.game_date) == today
            ).filter(
                ((Game.home_team.ilike('%duke%')) & 
                 ((Game.away_team.ilike('%north carolina%')) | (Game.away_team.ilike('%unc %'))))
                |
                ((Game.away_team.ilike('%duke%')) & 
                 ((Game.home_team.ilike('%north carolina%')) | (Game.home_team.ilike('%unc %'))))
            ).all()
            
            print()
            print(f"Found {len(suspicious)} suspicious Duke/UNC games labeled as today:")
            for g in suspicious:
                print(f"  ⚠️ {g.away_team} @ {g.home_team} at {g.game_date}")
            
            # Combine for deletion
            to_delete = list(set(stale_games + suspicious))
            
            if not to_delete:
                print("\n✅ No stale games found!")
                return
            
            print()
            print(f"Total games to delete: {len(to_delete)}")
            
            if dry_run:
                print("\n🛑 DRY RUN — No deletions performed")
                print("Run with --live to actually delete")
                return
            
            # Perform deletion
            print("\n🗑️  DELETING...")
            deleted_count = 0
            
            for game in to_delete:
                # Delete associated predictions first
                pred_count = db.query(Prediction).filter(Prediction.game_id == game.id).delete()
                
                # Delete the game
                db.delete(game)
                deleted_count += 1
                
                print(f"  Deleted: {game.away_team} @ {game.home_team} (+ {pred_count} predictions)")
            
            db.commit()
            
            print()
            print(f"✅ Purged {deleted_count} stale games")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Purge stale games from database')
    parser.add_argument('--live', action='store_true', help='Actually delete (default is dry run)')
    args = parser.parse_args()
    
    purge_stale_games(dry_run=not args.live)
