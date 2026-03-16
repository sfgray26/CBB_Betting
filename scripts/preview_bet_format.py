#!/usr/bin/env python3
"""
Preview the new Discord bet message format.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.discord_notifier import _bet_embed

# Sample bet data
sample_bet = {
    "home_team": "Duke",
    "away_team": "North Carolina",
    "bet_side": "home",
    "spread": -3.5,
    "bet_odds": -110,
    "edge_conservative": 0.042,
    "recommended_units": 1.5,
    "projected_margin": 5.2,
    "kelly_fractional": 0.15,
    "verdict": "Bet 1.5u Duke -3.5 [T2]",
    "snr": 0.72,
    "integrity_verdict": "CONFIRMED (90% confidence)",
    "matchup_notes": ["Duke strong at home", "UNC on back-to-back"]
}

# Another example - away team underdog bet
sample_bet_away = {
    "home_team": "Gonzaga",
    "away_team": "Saint Mary's",
    "bet_side": "away",
    "spread": -6.5,  # Gonzaga -6.5 (home favorite)
    "bet_odds": -110,
    "edge_conservative": 0.038,
    "recommended_units": 1.0,
    "projected_margin": -2.5,
    "kelly_fractional": 0.12,
    "verdict": "Bet 1.0u Saint Mary's +6.5 [T3]",
    "snr": 0.65,
    "integrity_verdict": "CONFIRMED",
    "matchup_notes": ["Saint Mary's undervalued"]
}

print("=" * 60)
print("NEW DISCORD BET MESSAGE FORMAT - PREVIEW")
print("=" * 60)
print()

for i, bet in enumerate([sample_bet, sample_bet_away], 1):
    embed = _bet_embed(bet)
    
    print(f"--- Example {i}: {bet['home_team']} vs {bet['away_team']} ---")
    print()
    print(f"TITLE: {embed['title']}")
    print(f"DESCRIPTION: {embed['description']}")
    print()
    print("FIELDS:")
    for field in embed['fields']:
        inline = " (inline)" if field.get('inline') else ""
        print(f"  - {field['name']}: {field['value']}{inline}")
    print()
    print("-" * 60)
    print()

print("=" * 60)
print("KEY IMPROVEMENTS:")
print("=" * 60)
print("1. Title clearly shows 'BET: [Team Name]'")
print("2. Description shows 'Take [Team] [Spread]' in bold")
print("3. 'Side' field explicitly states HOME or AWAY")
print("4. 'Opponent' field shows who they're playing")
print("5. 'Betting' field combines team + spread for clarity")
print()
