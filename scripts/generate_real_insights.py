"""
Generates real insights using the actual model output provided by the user.
"""

import os
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.scout import (
    generate_scouting_report, 
    generate_morning_briefing_narrative
)

def run_real_test():
    with open("tmp_today_data.json", "r") as f:
        data = json.load(f)

    print("="*60)
    print(f"📊 GENERATING INSIGHTS FOR SLATE: {data['date']}")
    print(f"Stats: {data['total_games']} Games, {data['bets_recommended']} Bets")
    print("="*60)

    # 1. Generate Scouting Reports for the sample games
    for pred in data['predictions']:
        home = pred['game']['home_team']
        away = pred['game']['away_team']
        notes = pred['full_analysis']['notes']
        verdict = pred['verdict']
        edge = pred['edge_conservative']

        print(f"\n[Scout] Analyzing {away} @ {home}...")
        insight = generate_scouting_report(home, away, notes, verdict, edge)
        print(f"RESULT: {insight}")

    # 2. Generate the Editor's Briefing
    top_bet = data['predictions'][0]
    top_info = f"{top_bet['game']['away_team']} @ {top_bet['game']['home_team']} (Edge {top_bet['edge_conservative']:.1%})"
    
    print("\n[Editor] Generating Morning Briefing Narrative...")
    briefing = generate_morning_briefing_narrative(
        n_bets=data['bets_recommended'],
        n_considered=5, # Estimated
        top_bet_info=top_info
    )
    print(f"RESULT: {briefing}")

    print("\n" + "="*60)

if __name__ == "__main__":
    run_real_test()
