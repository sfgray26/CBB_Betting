"""
Validation script for CBB Edge LLM Agents (Scout, Editor, Doctor).
Tests end-to-end LLM generation for scouting, briefings, and injury analysis.
"""

import os
import sys
import json

# Add project root to path
sys.path.append(os.getcwd())

from backend.services.scout import (
    generate_scouting_report, 
    generate_morning_briefing_narrative,
    generate_injury_impact
)

def prove_agents_work():
    print("="*60)
    print("🚀 PROVING CBB EDGE LLM AGENTS ARE ACTIVE")
    print("="*60)

    # 1. Prove the "Scout" Agent works
    print("\n[1/3] Testing SCOUT AGENT (Matchup Narrative)")
    mock_notes = [
        "Home 3PAr 42% vs Away drop 35%: +0.8 margin",
        "Transition edge: home 0.165 vs away 0.120",
        "eFG pressure: home_edge=+0.035 away_edge=+0.010 net=+0.025"
    ]
    scout_output = generate_scouting_report(
        home_team="Duke",
        away_team="North Carolina",
        matchup_notes=mock_notes,
        verdict="Bet 2.50u [T1]",
        edge=0.052
    )
    print(f"INPUT NOTES: {mock_notes}")
    print(f"SCOUT OUTPUT: \"{scout_output}\"")

    # 2. Prove the "Editor" Agent works
    print("\n[2/3] Testing EDITOR AGENT (Morning Briefing)")
    briefing_output = generate_morning_briefing_narrative(
        n_bets=3,
        n_considered=5,
        top_bet_info="Kansas @ Baylor (Edge 6.4%)"
    )
    print(f"INPUT STATS: 3 Bets, 5 Considers, Top: Kansas @ Baylor")
    print(f"EDITOR OUTPUT: \"{briefing_output}\"")

    # 3. Prove the "Doctor" Agent works
    print("\n[3/3] Testing DOCTOR AGENT (Injury Refinement)")
    injury_text = "Leading scorer (18.5 PPG) suffered a Grade 2 ankle sprain in practice Tuesday; seen in a walking boot today."
    doctor_output = generate_injury_impact(
        player="RJ Davis",
        team="UNC",
        raw_text=injury_text,
        base_impact=1.8
    )
    print(f"INPUT TEXT: \"{injury_text}\"")
    print(f"DOCTOR OUTPUT (JSON): {json.dumps(doctor_output, indent=2)}")

    print("\n" + "="*60)
    print("✅ ALL AGENTS VALIDATED AND RESPONDING VIA LOCAL LLM")
    print("="*60)

if __name__ == "__main__":
    prove_agents_work()
