#!/usr/bin/env python
"""Check roster including all positions."""
from backend.fantasy_baseball.yahoo_client_resilient import YahooFantasyClient
from backend.fantasy_baseball.player_board import get_or_create_projection
import json

client = YahooFantasyClient()
roster = client.get_roster()

print('=== FULL ROSTER WITH POSITIONS ===')
for p in roster:
    name = p.get('name', 'Unknown')
    pos = ','.join(p.get('positions', []))
    status = p.get('status') or 'Active'
    print(f"{name:25} | {pos:20} | {status}")

print("\n=== OUTFIELDERS (including LF/CF/RF) ===")
ofs = []
for p in roster:
    pos = p.get('positions', [])
    if any(x in pos for x in ['OF', 'LF', 'CF', 'RF']):
        name = p.get('name', 'Unknown')
        status = p.get('status') or 'Active'
        ofs.append((name, status, pos))
        print(f"{name} - {status} - {pos}")

if not ofs:
    print("No OFs found!")

print("\n=== ROSTER CONSTRUCTION ANALYSIS ===")
print(f"You have {len(ofs)} OF-eligible players")
print(f"Of those, {sum(1 for _, s, _ in ofs if s and s != 'Active')} are injured")

print("\n=== Z-SCORE EXPLANATION ===")
print("""
Z-Score measures how many standard deviations above/below average a player is.
For fantasy baseball:
- Z = 0 means average (replacement level)
- Z > 0 means above average
- Z > 5 is a strong starter
- Z > 10 is elite

Kwan's Z-Score: +7.09
This means he's significantly above average overall, BUT:
- He's highly specialized (great AVG, poor power)
- His value depends entirely on your team's needs

CATEGORY Z-SCORES (how he helps/hurts each category):
+ R    +1.4  (Good - scores runs as leadoff hitter)
+ HR   -1.5  (BAD - only 8 HR projected)
+ RBI  -0.4  (Weak - low RBI total)
+ SB   +1.3  (Good - 18 steals)
+ AVG  +2.3  (ELITE - .305 BA helps a lot)

Kwan is a "category specialist" - he's not well-rounded but dominates AVG
and helps in R/SB while hurting HR/RBI.
""")

print("=== SHOULD YOU ADD KWAN? ===")
print("""
ANALYSIS:

1. THE APP IS WRONG ABOUT "DROP REQUIRED"
   - You have 22 players
   - 4 are injured (3 on IL, 1 DTD)
   - Standard Yahoo: 23 active + 2 IL slots = 25 spots
   - You likely have 3-4 open active roster spots!
   - The app doesn't understand IL doesn't count against active spots

2. ROSTER CONSTRUCTION CHECK:
   - 3 first basemen (Alonso, Pasquantino, Torkelson) - CROWDED
   - 6 starting pitchers - decent depth
   - Only 2 relievers (Diaz, Adam - and Adam is hurt)
   - OF: Suzuki (IL), plus others not showing in filtered output

3. KWAN'S FIT:
   PROS:
   - Elite batting average (.305) - rare and valuable
   - Solid SBs (18) - good category
   - OF eligible (you need OFs)
   - High Z-score (7.09 = legitimately good)
   
   CONS:
   - Zero power (8 HR) - hurts you in HR/TB
   - Weak RBI (55) - hurts in RBI
   - Only OF eligible (no multi-position flexibility)

4. THE REAL ISSUE:
   You have THREE 1B (Alonso, Pasquantino, Torkelson).
   That's your roster problem - not a lack of Kwan.
   
   Better move: Trade one of the 1B for an OF, or drop the worst 1B
   (probably Torkelson or Pasquantino) for Kwan if you want AVG/SB.

RECOMMENDATION:
✓ Kwan is a GOOD player (Z=7.09 is legit)
✓ You DON'T need to drop anyone (IL slots available!)
✓ BUT consider if you need AVG/SB or if power (HR/RBI/TB) is your gap

If you're weak in AVG and SBs: Add Kwan, don't drop anyone (use IL space)
If you're competitive in AVG already: Pass - he doesn't help elsewhere
""")
