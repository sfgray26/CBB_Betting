#!/usr/bin/env python3
"""
Import bracket ratings from user CSV file.

Usage:
    python scripts/import_bracket_from_csv.py --csv ratings.csv --output data/bracket_2026.json
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_seed(seed_val):
    """Parse seed number from value that might have text."""
    if isinstance(seed_val, int):
        return seed_val
    if isinstance(seed_val, str):
        # Extract number from "#5" or "5"
        return int(seed_val.replace("#", "").strip())
    return seed_val


def load_csv_ratings(csv_path):
    """Load team ratings from CSV file."""
    ratings = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:  # utf-8-sig handles BOM
        reader = csv.DictReader(f)
        for row in reader:
            team_name = row.get('Team Name', '').strip()
            if not team_name:
                continue
            
            try:
                ratings[team_name] = {
                    'seed': parse_seed(row.get('Seed', 0)),
                    'composite_rating': float(row.get('Composite Rating', 0) or 0),
                    'kp_adj_em': float(row.get('KP AdjEM', row.get('composite_rating', 0)) or 0),
                    'bt_adj_em': float(row.get('BT AdjEM', row.get('composite_rating', 0)) or 0),
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


def build_bracket_from_ratings(ratings, template_path):
    """Build bracket JSON from ratings, using template for structure."""
    
    # Load template for region structure
    with open(template_path) as f:
        template = json.load(f)
    
    # Group teams by seed
    seed_to_teams = {}
    for team_name, data in ratings.items():
        seed = data['seed']
        if seed not in seed_to_teams:
            seed_to_teams[seed] = []
        seed_to_teams[seed].append((team_name, data))
    
    # Build regions (4 regions of 16 teams each)
    regions = ['east', 'south', 'west', 'midwest']
    bracket = {region: [] for region in regions}
    
    # Assign teams to regions based on seed
    for seed in range(1, 17):
        teams_at_seed = seed_to_teams.get(seed, [])
        
        # Distribute across 4 regions
        for i, (team_name, data) in enumerate(teams_at_seed[:4]):
            region = regions[i % 4]
            bracket[region].append({
                'name': team_name,
                'seed': seed,
                'region': region,
                'composite_rating': data['composite_rating'],
                'kp_adj_em': data['kp_adj_em'],
                'bt_adj_em': data['bt_adj_em'],
                'pace': data['pace'],
                'three_pt_rate': data['three_pt_rate'],
                'def_efg_pct': data['def_efg_pct'],
                'conference': data['conference'],
                'tournament_exp': data['tournament_exp'],
            })
    
    # Sort each region by seed
    for region in regions:
        bracket[region] = sorted(bracket[region], key=lambda x: x['seed'])
    
    # Add metadata
    bracket['_source'] = 'User CSV Import'
    bracket['_notes'] = 'Ratings imported from user CSV file'
    
    return bracket


def main():
    parser = argparse.ArgumentParser(description='Import bracket ratings from CSV')
    parser.add_argument('--csv', required=True, help='Path to CSV file with team ratings')
    parser.add_argument('--template', default='data/bracket_template_2026.json',
                        help='Template JSON for bracket structure')
    parser.add_argument('--output', default='data/bracket_2026.json',
                        help='Output bracket JSON file')
    args = parser.parse_args()
    
    print(f"Loading ratings from: {args.csv}")
    ratings = load_csv_ratings(args.csv)
    print(f"Loaded {len(ratings)} teams from CSV")
    
    print(f"\nBuilding bracket from ratings...")
    bracket = build_bracket_from_ratings(ratings, args.template)
    
    # Count teams per region
    total = 0
    for region in ['east', 'south', 'west', 'midwest']:
        count = len(bracket.get(region, []))
        print(f"  {region}: {count} teams")
        total += count
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(bracket, f, indent=2)
    
    print(f"\n✅ Bracket saved to: {output_path}")
    print(f"Total teams: {total}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
