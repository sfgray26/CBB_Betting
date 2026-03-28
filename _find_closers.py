#!/usr/bin/env python
"""Manually find available closers on the waiver wire."""
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
from backend.fantasy_baseball.player_board import get_or_create_projection

# Top projected closers for 2026 - manual list since data is broken
TOP_CLOSERS = {
    "Edwin Diaz": 32,
    "Raisel Iglesias": 30,
    "Emmanuel Clase": 28,
    "Ryan Pressly": 26,
    "Camilo Doval": 25,
    "Andres Munoz": 24,
    "Kirby Yates": 22,
    "Devin Williams": 24,
    "Ryan Helsley": 26,
    "Clay Holmes": 23,
    "Tanner Scott": 21,
    "Pete Fairbanks": 20,
    "Kenley Jansen": 19,
    "Jose Leclerc": 18,
    "Paul Sewald": 17,
    "David Bednar": 16,
    "Alexis Diaz": 15,
    "Jason Adam": 15,  # Your injured guy
    "A.J. Puk": 14,
    "Jhoan Duran": 20,
    "Evan Phillips": 14,
    "Seranthony Dominguez": 13,
    "Robert Suarez": 12,
    "Cade Smith": 8,  # Cleveland setup, might get saves
    "Gregory Soto": 10,
}

client = YahooFantasyClient()

print("=== CHECKING FOR AVAILABLE CLOSERS ===")
print("(Comparing top closer list to your roster)")
print()

# Get your roster
roster = client.get_roster()
my_player_names = {p.get('name', '').lower() for p in roster}

print("Your current RPs:")
for p in roster:
    if 'RP' in p.get('positions', []):
        name = p.get('name')
        status = p.get('status') or 'Active'
        proj_saves = TOP_CLOSERS.get(name, 0)
        print(f"  {name}: {status} (~{proj_saves} saves projected)")

print()
print("Checking waiver wire for closers with 10+ projected saves...")
print()

# Check free agents
try:
    # Get RPs with lower count to avoid API error
    fa_rps = client.get_free_agents(position='RP', count=50)
    
    available_closers = []
    for p in fa_rps:
        name = p.get('name')
        if name in TOP_CLOSERS:
            proj_saves = TOP_CLOSERS[name]
            owned = p.get('percent_owned', 0)
            available_closers.append((name, proj_saves, owned))
    
    # Sort by projected saves
    available_closers.sort(key=lambda x: x[1], reverse=True)
    
    print("AVAILABLE CLOSERS (10+ saves projected):")
    print("-" * 60)
    found_any = False
    for name, saves, owned in available_closers:
        if saves >= 10:
            found_any = True
            priority = "MUST ADD" if saves >= 20 else "Good option" if saves >= 15 else "Streaming"
            print(f"  {name:25} | ~{saves:2d} saves | {owned:4.1f}% owned | {priority}")
    
    if not found_any:
        print("  No top-tier closers found available.")
        print("  Check again tomorrow - closers get dropped frequently.")
    
    print()
    print("Also check for:")
    print("  - Cade Smith (CLE) - Setup but high-leverage, could get saves")
    print("  - Any new closers named this week (watch bullpen news)")
    
except Exception as e:
    print(f"API Error (Yahoo limits): {e}")
    print()
    print("FALLBACK: Top closers likely available in your league:")
    print("  (Check Yahoo app manually for these names)")
    for name, saves in sorted(TOP_CLOSERS.items(), key=lambda x: x[1], reverse=True)[:15]:
        if name.lower() not in my_player_names and saves >= 12:
            print(f"  - {name} (~{saves} saves)")

print()
print("=== RECOMMENDATION ===")
print("""
IMMEDIATE ACTIONS:
1. Check your Yahoo app for these names in Free Agents
2. Prioritize anyone with 15+ projected saves
3. Target: Clase, Pressly, Doval, Munoz, Yates level if available
4. Acceptable: Anyone with 10+ saves and clear closer role

DO NOT WAIT for the waiver wire UI fix - saves are too important
in H2H leagues. Check manually now.

Your current situation:
- Edwin Diaz: Elite closer (~32 saves) - KEEP
- Jason Adam: Hurt, 15 saves when healthy - MOVE TO IL, add replacement

GOAL: Have 2-3 reliable closers by Opening Day.
""")
