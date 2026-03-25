#!/usr/bin/env python
"""Diagnose the closer/saves gap in waiver recommendations."""
from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
from backend.fantasy_baseball.player_board import get_or_create_projection
from backend.services.waiver_edge_detector import WaiverEdgeDetector
from backend.services.mcmc_simulator import MCMCWeeklySimulator

client = YahooFantasyClient()

print("=== CURRENT RP SITUATION ===")
roster = client.get_roster()

rps = []
for p in roster:
    pos = p.get('positions', [])
    if 'RP' in pos:
        name = p.get('name')
        status = p.get('status') or 'Active'
        proj = get_or_create_projection(p)
        nsv = proj.get('cat_scores', {}).get('nsv', 0)
        rps.append((name, status, nsv))

print(f"Total RPs on roster: {len(rps)}")
for name, status, nsv in rps:
    marker = "✓" if status == "Active" else "INJURED"
    print(f"  {name}: {status} (projected saves: {nsv:.1f})")

healthy_rps = [r for r in rps if r[1] == 'Active']
injured_rps = [r for r in rps if r[1] != 'Active']

print(f"\nHealthy RPs: {len(healthy_rps)}")
print(f"Injured RPs: {len(injured_rps)}")

if len(healthy_rps) == 0:
    print("\n*** CRITICAL GAP: You have ZERO healthy closers! ***")
    print("This should be the #1 waiver priority!")

print("\n=== CHECKING WAIVER WIRE FOR RPs ===")
try:
    # Get all FAs and filter for RPs
    all_fas = client.get_free_agents(count=100)
    fa_rps = []
    for p in all_fas:
        if 'RP' in p.get('positions', []):
            name = p.get('name')
            proj = get_or_create_projection(p)
            nsv = proj.get('cat_scores', {}).get('nsv', 0)
            owned = p.get('percent_owned', 0)
            fa_rps.append((name, nsv, owned))
    
    # Sort by projected saves
    fa_rps.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Found {len(fa_rps)} RPs available")
    print("\nTop 10 by projected saves:")
    for name, nsv, owned in fa_rps[:10]:
        print(f"  {name}: {nsv:.1f} saves, {owned:.0f}% owned")
        
except Exception as e:
    print(f"Error: {e}")

print("\n=== WHAT THE WAIVER DETECTOR SEES ===")
sim = MCMCWeeklySimulator(n_sims=100)
detector = WaiverEdgeDetector(mcmc_simulator=sim)

# Get top moves
moves = detector.get_top_moves(roster, [], n_candidates=10)
print(f"Top 10 waiver recommendations:")
for i, m in enumerate(moves, 1):
    fa = m['add_player']
    name = fa.get('name')
    pos = ','.join(fa.get('positions', []))
    drop = m.get('drop_player_name', 'None')
    score = m.get('need_score', 0)
    print(f"  {i}. Add {name} ({pos}) - Score: {score:.2f} (Drop: {drop or 'None'})")

print("\n=== ANALYSIS ===")
print("""
THE PROBLEM:
1. The waiver detector uses category DEFICITS to score players
2. If your opponent also has weak saves, the deficit may look small
3. RP is grouped with SP/P positionally, so it doesn't recognize
   "0 healthy closers" as an emergency
4. Saves (nsv) are a BINARY category - you either get saves or you don't
   Z-scores don't capture this well

THE FIX NEEDED:
1. Special case: If healthy_RP_count < 2, prioritize closers regardless
2. Check for closers with nsv > 5 as high priority adds
3. Surface "Closer Needed" as a roster construction alert
4. Position group for RP should be separate from SP for roster construction
""")
