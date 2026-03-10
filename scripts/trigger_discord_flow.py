"""
Manual trigger for the Discord notification flow using FULL LOCAL JSON DATA.
This ensures all 17 bets are sent with their respective AI Scouting Reports and Sanity Checks.
"""

import os
import sys
import json
from datetime import datetime, date
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.discord_notifier import send_todays_bets

def trigger_flow():
    print("="*60)
    print("🚀 TRIGGERING FULL DISCORD NOTIFICATION FLOW")
    print("="*60)

    # Try to load the full data file
    data = None
    source = None
    
    # Try multiple possible data sources
    possible_files = [
        "full_today_data.json",
        "current_recommendations.json",
        "tmp_predictions.json",
    ]
    
    for filename in possible_files:
        filepath = os.path.join(os.getcwd(), filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                source = filename
                print(f"✅ Loaded data from: {filename}")
                break
            except Exception as e:
                print(f"⚠️  Error loading {filename}: {e}")
                continue
    
    # If no JSON file works, try database
    if data is None:
        print("⚠️  No JSON data files found. Falling back to database...")
        try:
            from backend.models import SessionLocal, Prediction, Game
            
            db = SessionLocal()
            try:
                today_utc = date.today()
                
                predictions = (
                    db.query(Prediction)
                    .join(Game)
                    .filter(Prediction.prediction_date == today_utc)
                    .all()
                )
                
                # Convert to dict format
                data = {
                    "predictions": [],
                    "total_games": len(predictions),
                    "bets_recommended": 0,
                }
                
                for p in predictions:
                    g = p.game
                    fa = p.full_analysis or {}
                    data["predictions"].append({
                        "game": {
                            "home_team": g.home_team,
                            "away_team": g.away_team,
                            "game_date": g.game_date.isoformat() if g.game_date else None,
                        },
                        "verdict": p.verdict,
                        "edge_conservative": p.edge_conservative,
                        "recommended_units": p.recommended_units,
                        "full_analysis": fa,
                    })
                    if p.verdict.startswith("Bet"):
                        data["bets_recommended"] += 1
                
                source = "database"
                print(f"✅ Loaded {len(predictions)} predictions from database")
                
            finally:
                db.close()
        except Exception as e:
            print(f"❌ Error loading from database: {e}")
            return

    if data is None:
        print("❌ Could not load data from any source")
        return

    predictions = data.get("predictions", [])
    bets = [p for p in predictions if p.get("verdict", "").startswith("Bet")]

    print(f"✅ Found {len(bets)} bets from {source}.")

    bet_details = []
    for p in bets:
        game = p.get("game", {})
        fa = p.get("full_analysis", {})
        calcs = fa.get("calculations", {})
        inputs = fa.get("inputs", {})
        odds = inputs.get("odds", {})
        
        bet_details.append({
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "spread": odds.get("spread"),
            "bet_side": calcs.get("bet_side"),
            "edge_conservative": p.get("edge_conservative"),
            "recommended_units": p.get("recommended_units"),
            "bet_odds": calcs.get("bet_odds"),
            "kelly_fractional": calcs.get("kelly_fractional"),
            "projected_margin": p.get("projected_margin"),
            "verdict": p.get("verdict"),
            "matchup_notes": fa.get("notes", [])
        })

    summary = {
        "games_analyzed": data.get("total_games", 0),
        "bets_recommended": len(bets),
        "games_considered": 5,
        "duration_seconds": 0
    }

    print(f"Calling send_todays_bets for {len(bet_details)} individual bets...")
    print("Each bet triggers a web search and GPU analysis. This will take ~2-3 minutes total.")
    
    send_todays_bets(bet_details, summary)
    
    print("\n" + "="*60)
    print(f"🏁 FULL FLOW COMPLETE. ALL {len(bets)} BETS SENT.")
    print("="*60)

if __name__ == "__main__":
    trigger_flow()
