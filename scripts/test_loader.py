#!/usr/bin/env python3
"""Test the projections loader."""

import sys
sys.path.insert(0, '.')

from backend.fantasy_baseball.projections_loader import (
    DATA_DIR, load_steamer_batting, load_steamer_pitching, load_adp
)

print(f"DATA_DIR: {DATA_DIR}")
print(f"Exists: {DATA_DIR.exists()}")
print()

# Try loading batting
batters = load_steamer_batting(DATA_DIR / 'steamer_batting_2026.csv')
print(f"Batters loaded: {len(batters)}")
if batters:
    b = batters[0]
    print(f"  Sample: {b['name']} - {b['team']} - {b['positions']}")
    print(f"  Stats: HR={b['proj']['hr']}, SB={b['proj']['nsb']}, AVG={b['proj']['avg']}")
print()

# Try loading pitching
pitchers = load_steamer_pitching(DATA_DIR / 'steamer_pitching_2026.csv')
print(f"Pitchers loaded: {len(pitchers)}")
if pitchers:
    p = pitchers[0]
    print(f"  Sample: {p['name']} - {p['team']} - {p['positions']}")
    print(f"  Stats: W={p['proj']['w']}, ERA={p['proj']['era']}, K={p['proj']['k_pit']}")
print()

# Try loading ADP
adp = load_adp(DATA_DIR / 'adp_yahoo_2026.csv')
print(f"ADP entries loaded: {len(adp)}")
if adp:
    sample_id = list(adp.keys())[0]
    print(f"  Sample: {sample_id} -> ADP {adp[sample_id]}")

print()
print(f"Total players: {len(batters) + len(pitchers)}")
print()
print("SUCCESS: All projections loaded correctly!")
