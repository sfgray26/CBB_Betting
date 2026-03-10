#!/usr/bin/env python3
"""
Generate and send Discord notification using LIVE DATABASE data.

This ensures Discord matches the UI by using the same data source.
"""

import os
import sys
from datetime import datetime, timedelta, date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def send_discord_from_db():
    """Send Discord notification using actual database predictions."""
    try:
        from backend.models import SessionLocal, Prediction, Game
        from backend.services.discord_notifier import send_todays_bets
        
        db = SessionLocal()
        try:
            today_utc = date.today()
            now_utc = datetime.utcnow()
            
            print(f"📊 Querying database for {today_utc}...")
            
            # Get all predictions for today (including started games)
            predictions = (
                db.query(Prediction)
                .join(Game)
                .filter(Prediction.prediction_date == today_utc)
                .order_by(Game.game_date.asc())
                .all()
            )
            
            # Separate into upcoming and all
            upcoming_preds = [(p, p.game) for p in predictions if p.game.game_date > now_utc]
            all_preds = [(p, p.game) for p in predictions]
            
            # Get BET verdicts
            upcoming_bets = [(p, g) for p, g in upcoming_preds if p.verdict.startswith("Bet")]
            all_bets = [(p, g) for p, g in all_preds if p.verdict.startswith("Bet")]
            
            print(f"  Total predictions today: {len(predictions)}")
            print(f"  Upcoming games: {len(upcoming_preds)}")
            print(f"  Upcoming BETs: {len(upcoming_bets)}")
            print(f"  All BETs (incl. started): {len(all_bets)}")
            print()
            
            # Build bet_details for Discord (use ALL bets, not just upcoming)
            bet_details = []
            for pred, game in all_bets:
                fa = pred.full_analysis or {}
                calcs = fa.get("calculations", {})
                inputs = fa.get("inputs", {})
                odds = inputs.get("odds", {})
                
                bet_details.append({
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "spread": odds.get("spread"),
                    "bet_side": calcs.get("bet_side", "home"),
                    "edge_conservative": pred.edge_conservative,
                    "recommended_units": pred.recommended_units,
                    "bet_odds": calcs.get("bet_odds"),
                    "kelly_fractional": pred.kelly_fractional,
                    "projected_margin": pred.projected_margin,
                    "verdict": pred.verdict,
                    "matchup_notes": fa.get("notes", [])
                })
            
            # Build summary
            considered = [(p, g) for p, g in all_preds if p.verdict.startswith("CONSIDER")]
            
            summary = {
                "games_analyzed": len(all_preds),
                "bets_recommended": len(all_bets),
                "games_considered": len(considered),
                "duration_seconds": 0,
            }
            
            print(f"📱 Sending to Discord...")
            print(f"  Summary: {summary}")
            print(f"  Bet details count: {len(bet_details)}")
            
            if bet_details:
                send_todays_bets(bet_details, summary)
                print("✅ Discord notification sent successfully!")
            else:
                print("⚠️  No bets to send to Discord")
                
            return len(all_bets)
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 0


def main():
    print("=" * 70)
    print("🚀 DISCORD NOTIFICATION FROM LIVE DATABASE")
    print("=" * 70)
    print(f"Current UTC: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    count = send_discord_from_db()
    
    print()
    print("=" * 70)
    if count > 0:
        print(f"✅ Sent {count} bets to Discord")
    else:
        print("⚠️  No bets found to send")
    print("=" * 70)


if __name__ == "__main__":
    main()
