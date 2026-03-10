#!/usr/bin/env python3
"""
Quick admin script to manually trigger Discord notification with current database data.
Run this while waiting for Railway UI deployment.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("=" * 70)
    print("🚀 MANUAL DISCORD NOTIFICATION TRIGGER")
    print("=" * 70)
    
    try:
        from backend.models import SessionLocal, Prediction, Game
        from backend.services.discord_notifier import send_todays_bets
        from datetime import datetime, date, timedelta
        
        db = SessionLocal()
        try:
            # Get ALL predictions from last 24 hours (not just 'today')
            since = datetime.utcnow() - timedelta(hours=24)
            
            predictions = (
                db.query(Prediction, Game)
                .join(Game, Prediction.game_id == Game.id)
                .filter(Game.game_date >= since)
                .all()
            )
            
            print(f"Found {len(predictions)} predictions from last 24h")
            
            # Get BET verdicts
            bets = [(p, g) for p, g in predictions if p.verdict.startswith("Bet")]
            print(f"BET verdicts: {len(bets)}")
            
            if not bets:
                print("⚠️  No bets to send")
                return
            
            # Build bet_details
            bet_details = []
            for pred, game in bets:
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
            summary = {
                "games_analyzed": len(predictions),
                "bets_recommended": len(bets),
                "games_considered": len([(p, g) for p, g in predictions if p.verdict.startswith("CONSIDER")]),
                "duration_seconds": 0,
            }
            
            print(f"\n📱 Sending to Discord...")
            print(f"   Games: {summary['games_analyzed']}")
            print(f"   Bets: {summary['bets_recommended']}")
            
            send_todays_bets(bet_details, summary)
            
            print("\n✅ Discord notification sent!")
            
            # Also print to console
            print("\n" + "=" * 70)
            print("📋 BET SUMMARY (Console)")
            print("=" * 70)
            for i, (pred, game) in enumerate(bets[:10], 1):
                print(f"{i}. {game.away_team} @ {game.home_team}")
                print(f"   Verdict: {pred.verdict}")
                print(f"   Edge: {pred.edge_conservative:.2%}")
                print(f"   Units: {pred.recommended_units:.2f}")
                print()
            
            if len(bets) > 10:
                print(f"... and {len(bets) - 10} more")
            
        finally:
            db.close()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
