#!/usr/bin/env python3
"""
Pre-Tournament Checklist — Run this before March 18 tipoff!

This script verifies all systems are ready for tournament action.
"""

import json
import sys
from pathlib import Path

def check_bracket_data():
    """Verify bracket data is complete."""
    print("=" * 60)
    print("📋 BRACKET DATA CHECK")
    print("=" * 60)
    
    with open('data/bracket_2026.json') as f:
        bracket = json.load(f)
    
    regions = ['east', 'south', 'west', 'midwest']
    issues = []
    
    for region in regions:
        teams = bracket.get(region, [])
        if len(teams) != 16:
            issues.append(f"{region}: Expected 16 teams, got {len(teams)}")
        
        for team in teams:
            if not team.get('name'):
                issues.append(f"{region} seed {team.get('seed')}: Missing team name")
            if team.get('composite_rating', 0) == 0:
                issues.append(f"{region} {team.get('name')}: Missing composite rating")
    
    # Check First Four
    first_four = bracket.get('_first_four', {})
    if len(first_four) != 4:
        issues.append(f"Expected 4 First Four matchups, got {len(first_four)}")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ All 64 teams verified")
        print("✅ First Four matchups documented")
        print("✅ All ratings present")
        return True


def check_simulation_results():
    """Verify simulation results exist and look valid."""
    print("\n" + "=" * 60)
    print("🎲 SIMULATION RESULTS CHECK")
    print("=" * 60)
    
    sim_path = Path('outputs/tournament_2026/sim_results.json')
    if not sim_path.exists():
        print("❌ sim_results.json not found! Run: python scripts/run_bracket_sims.py")
        return False
    
    with open(sim_path) as f:
        results = json.load(f)
    
    champ_probs = results.get('championship', {})
    if not champ_probs:
        print("❌ No championship probabilities found")
        return False
    
    top_5 = sorted(champ_probs.items(), key=lambda x: -x[1])[:5]
    print(f"✅ Simulations found: {results.get('n_sims', 'unknown')} runs")
    print(f"✅ Top 5 championship favorites:")
    for team, prob in top_5:
        print(f"   {team}: {prob*100:.1f}%")
    
    return True


def check_discord_config():
    """Verify Discord webhook is configured."""
    print("\n" + "=" * 60)
    print("💬 DISCORD NOTIFICATION CHECK")
    print("=" * 60)
    
    env_path = Path('.env')
    if not env_path.exists():
        print("⚠️  .env file not found — Discord may not be configured")
        return False
    
    with open(env_path) as f:
        content = f.read()
    
    if 'DISCORD_WEBHOOK_URL' in content:
        print("✅ Discord webhook URL found in .env")
    else:
        print("⚠️  DISCORD_WEBHOOK_URL not found in .env")
        return False
    
    # Check scheduler script
    scheduler_path = Path('scripts/openclaw_scheduler_improved.py')
    if scheduler_path.exists():
        print("✅ Discord scheduler script found")
    else:
        print("⚠️  Discord scheduler script not found")
    
    return True


def check_ui_pages():
    """Verify dashboard pages are ready."""
    print("\n" + "=" * 60)
    print("🖥️  UI PAGES CHECK")
    print("=" * 60)
    
    pages = [
        'dashboard/pages/13_Tournament_Bracket.py',
        'dashboard/pages/14_Bracket_Visual.py'
    ]
    
    for page in pages:
        if Path(page).exists():
            print(f"✅ {page}")
        else:
            print(f"❌ {page} MISSING")
            return False
    
    return True


def generate_tournament_summary():
    """Generate a quick summary of tournament predictions."""
    print("\n" + "=" * 60)
    print("🏆 TOURNAMENT SUMMARY")
    print("=" * 60)
    
    try:
        with open('outputs/tournament_2026/sim_results.json') as f:
            results = json.load(f)
        
        print(f"\nSimulations run: {results.get('n_sims', 'N/A')}")
        print(f"Avg upsets/tournament: {results.get('avg_upsets_per_tournament', 'N/A'):.1f}")
        print(f"Avg championship margin: {results.get('avg_championship_margin', 'N/A'):.1f} pts")
        
        # Top champions
        print("\nTop 3 Championship Favorites:")
        champ = results.get('championship', {})
        for team, prob in sorted(champ.items(), key=lambda x: -x[1])[:3]:
            print(f"  🥇 {team}: {prob*100:.1f}%")
        
        # Cinderella candidates (double-digit seeds with >0.5% title chance)
        print("\nCinderella Candidates (seed 10+ with >0.5% title chance):")
        
        # Load bracket once
        with open('data/bracket_2026.json') as f:
            bracket = json.load(f)
        
        # Build team -> seed mapping
        team_seeds = {}
        for region in ['east', 'south', 'west', 'midwest']:
            for t in bracket.get(region, []):
                team_seeds[t['name']] = t['seed']
        
        # Find double-digit seeds with >0.5% title chance
        cinderellas = [(t, p, team_seeds.get(t, 99)) for t, p in champ.items() if team_seeds.get(t, 0) >= 10 and p > 0.005]
        cinderellas = sorted(cinderellas, key=lambda x: -x[1])
        
        if cinderellas:
            for team, prob, seed in cinderellas[:8]:
                print(f"  #{seed} {team}: {prob*100:.1f}%")
        else:
            print("  No double-digit seeds with >0.5% title chance")
        
    except Exception as e:
        print(f"Could not generate summary: {e}")


def main():
    print("\n" + "🎉 " * 20)
    print("  NCAA TOURNAMENT 2026 — PRE-TOURNAMENT CHECKLIST")
    print("🎉 " * 20 + "\n")
    
    checks = [
        ("Bracket Data", check_bracket_data),
        ("Simulation Results", check_simulation_results),
        ("Discord Config", check_discord_config),
        ("UI Pages", check_ui_pages),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    generate_tournament_summary()
    
    print("\n" + "=" * 60)
    print("📊 FINAL STATUS")
    print("=" * 60)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all_passed:
        print("\n🎉 ALL SYSTEMS GO! Ready for tournament action!")
        print("🍀 Good luck with your picks!")
        return 0
    else:
        print("\n⚠️  Some checks failed — review issues above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
