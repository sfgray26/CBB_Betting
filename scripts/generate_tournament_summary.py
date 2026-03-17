#!/usr/bin/env python3
"""
Tournament Day Summary — Quick report for Discord/messaging.

Run this after simulations to get a quick summary of predictions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def generate_summary():
    """Generate tournament day summary."""
    
    # Load simulation results
    with open('outputs/tournament_2026/sim_results.json') as f:
        results = json.load(f)
    
    # Load bracket for seed info
    with open('data/bracket_2026.json') as f:
        bracket = json.load(f)
    
    team_seeds = {}
    for region in ['east', 'south', 'west', 'midwest']:
        for t in bracket.get(region, []):
            team_seeds[t['name']] = t['seed']
    
    lines = []
    lines.append("🏀 **NCAA TOURNAMENT 2026 — MODEL SUMMARY** 🏀")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} ET")
    lines.append("")
    
    # Championship favorites
    lines.append("**🏆 Championship Favorites:**")
    champ = results.get('championship', {})
    for team, prob in sorted(champ.items(), key=lambda x: -x[1])[:5]:
        seed = team_seeds.get(team, '?')
        lines.append(f"  #{seed} {team}: {prob*100:.1f}%")
    lines.append("")
    
    # Final Four probabilities
    lines.append("**🎭 Final Four Probabilities:**")
    ff = results.get('final_four', {})
    for team, prob in sorted(ff.items(), key=lambda x: -x[1])[:8]:
        seed = team_seeds.get(team, '?')
        lines.append(f"  #{seed} {team}: {prob*100:.1f}%")
    lines.append("")
    
    # Cinderella candidates
    lines.append("**🧚 Cinderella Candidates (Seed 10+):**")
    cinderellas = []
    s16 = results.get('sweet_sixteen', {})
    for team, prob in sorted(s16.items(), key=lambda x: -x[1]):
        seed = team_seeds.get(team, 0)
        if seed >= 10 and prob > 0.01:
            cinderellas.append((team, seed, prob))
    
    for team, seed, prob in cinderellas[:6]:
        lines.append(f"  #{seed} {team}: {prob*100:.1f}% to reach Sweet 16")
    lines.append("")
    
    # Upset alerts
    lines.append("**⚡ Upset Alerts (R64):**")
    lines.append(f"  All #8 vs #9 matchups: ~48% upset rate")
    lines.append(f"  All #7 vs #10 matchups: ~39% upset rate")
    lines.append(f"  All #5 vs #12 matchups: ~35% upset rate")
    lines.append("")
    
    # Model stats
    lines.append("**📊 Model Stats:**")
    lines.append(f"  Simulations: {results.get('n_sims', 'N/A'):,}")
    lines.append(f"  Avg upsets/tournament: {results.get('avg_upsets_per_tournament', 0):.1f}")
    lines.append(f"  Avg championship margin: {results.get('avg_championship_margin', 0):.1f} pts")
    lines.append("")
    
    # Key insight
    lines.append("**💡 Key Insight:**")
    lines.append(f"  Model expects {results.get('avg_upsets_per_tournament', 0):.0f} upsets per tournament.")
    lines.append("  Look for chaos in 8/9 and 7/10 matchups!")
    lines.append("")
    
    lines.append("🔗 Full analysis: [Dashboard URL]")
    lines.append("🍀 Good luck with your brackets!")
    
    return "\n".join(lines)


def main():
    try:
        summary = generate_summary()
        print(summary)
        
        # Also save to file
        output_path = Path('outputs/tournament_2026/tournament_summary.txt')
        with open(output_path, 'w') as f:
            f.write(summary)
        print(f"\n✅ Summary saved to: {output_path}")
        
        return 0
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
