#!/usr/bin/env python3
"""
Update bracket JSON with user CSV ratings for matching teams.

Usage:
    python scripts/update_bracket_from_csv.py --csv ratings.csv --bracket data/bracket_2026.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_csv_ratings(csv_path):
    """Load team ratings from CSV file."""
    ratings = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            team_name = row.get('Team Name', '').strip()
            if not team_name:
                continue
            
            try:
                ratings[team_name] = {
                    'composite_rating': float(row.get('Composite Rating', 0) or 0),
                    'kp_adj_em': float(row.get('KP AdjEM', 0) or 0),
                    'bt_adj_em': float(row.get('BT AdjEM', 0) or 0),
                    'pace': float(row.get('Pace', 68.0) or 68.0),
                    'three_pt_rate': float(row.get('3PT Rate', 0.35) or 0.35),
                    'def_efg_pct': float(row.get('Def eFG%', 0.50) or 0.50),
                    'conference': row.get('Conference', 'Unknown').strip(),
                    'tournament_exp': float(row.get('Tourney Exp', 0.70) or 0.70),
                }
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not parse row for {team_name}: {e}")
                continue
    
    return ratings


def update_bracket_with_ratings(bracket_path, ratings):
    """Update bracket JSON with CSV ratings for matching teams."""
    
    with open(bracket_path) as f:
        bracket = json.load(f)
    
    updated_count = 0
    matched_teams = []
    
    for region in ['east', 'south', 'west', 'midwest']:
        if region not in bracket:
            continue
        
        for team in bracket[region]:
            team_name = team.get('name', '')
            if team_name in ratings:
                # Update with user's ratings
                team.update(ratings[team_name])
                updated_count += 1
                matched_teams.append(f"{team_name} ({region})")
    
    return bracket, updated_count, matched_teams


def main():
    parser = argparse.ArgumentParser(description='Update bracket with CSV ratings')
    parser.add_argument('--csv', required=True, help='Path to CSV file with team ratings')
    parser.add_argument('--bracket', default='data/bracket_2026.json',
                        help='Bracket JSON to update')
    parser.add_argument('--output', default='data/bracket_2026.json',
                        help='Output bracket JSON file')
    args = parser.parse_args()
    
    print(f"Loading ratings from: {args.csv}")
    ratings = load_csv_ratings(args.csv)
    print(f"Loaded {len(ratings)} teams from CSV")
    
    print(f"\nUpdating bracket: {args.bracket}")
    bracket, updated, matched = update_bracket_with_ratings(args.bracket, ratings)
    
    print(f"\n✅ Updated {updated} teams with your ratings:")
    for team in matched:
        print(f"  - {team}")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(bracket, f, indent=2)
    
    print(f"\n💾 Saved to: {output_path}")
    
    # Now run fresh simulations
    print("\n" + "="*60)
    print("Running fresh bracket simulations with your ratings...")
    print("="*60)
    
    import subprocess
    result = subprocess.run([
        'python3', 'scripts/run_bracket_sims.py',
        '--bracket', str(output_path),
        '--quick',
        '--output', 'outputs/tournament_2026'
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
