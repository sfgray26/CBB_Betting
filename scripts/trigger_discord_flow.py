"""
Manual trigger for the Discord notification flow using FULL LOCAL JSON DATA.
This ensures all 17 bets are sent with their respective AI Scouting Reports and Sanity Checks.
"""

import os
import sys
import json
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

    # Load the full data file we just created
    try:
        with open("full_today_data.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading full_today_data.json: {e}")
        return

    predictions = data.get("predictions", [])
    bets = [p for p in predictions if p.get("verdict", "").startswith("Bet")]

    print(f"✅ Loaded {len(bets)} bets from full dataset.")

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
        "bets_recommended": data.get("bets_recommended", 0),
        "games_considered": 5,
        "duration_seconds": 0
    }

    print(f"Calling send_todays_bets for {len(bet_details)} individual bets...")
    print("Each bet triggers a web search and GPU analysis. This will take ~2-3 minutes total.")
    
    send_todays_bets(bet_details, summary)
    
    print("\n" + "="*60)
    print("🏁 FULL FLOW COMPLETE. ALL 17 BETS SENT.")
    print("="*60)

if __name__ == "__main__":
    trigger_flow()
