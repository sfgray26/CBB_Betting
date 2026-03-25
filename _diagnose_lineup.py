#!/usr/bin/env python
"""Diagnose what's wrong with the daily lineup optimizer."""
from backend.fantasy_baseball.yahoo_client import YahooFantasyClient
from backend.fantasy_baseball.daily_lineup_optimizer import get_lineup_optimizer
from backend.fantasy_baseball.player_board import get_or_create_projection
from datetime import date, timedelta

client = YahooFantasyClient()
roster = client.get_roster()

print("=" * 70)
print("DAILY LINEUP OPTIMIZER DIAGNOSIS")
print("=" * 70)

# Get today's lineup report
today = date.today().isoformat()
optimizer = get_lineup_optimizer()

# Get projections
projections = [get_or_create_projection(p) for p in roster]

# Build report
report = optimizer.build_daily_report(
    game_date=today,
    roster=roster,
    projections=projections,
)

print(f"\n1. MLB GAMES TODAY: {report.get('games_count', 0)} games found")
if report.get('no_games_today'):
    print("   WARNING: No games scheduled!")
else:
    print("   Games found:")
    for g in report.get('games', [])[:5]:
        print(f"     {g['away']} @ {g['home']} - Total: {g.get('total', 'N/A')}")

print("\n2. STACK CANDIDATES (high implied runs):")
stacks = report.get('stack_candidates', [])
if stacks:
    for team, runs in stacks[:5]:
        print(f"     {team}: {runs} implied runs")
else:
    print("   None found (no games or no odds)")

print("\n3. BATTER RANKINGS (Current 'Optimization'):")
batters = report.get('batter_rankings', [])
print(f"   Total batters ranked: {len(batters)}")
print("\n   Current logic: START top 9, BENCH rest")
print("   " + "-" * 60)

for i, b in enumerate(batters[:12], 1):
    status = "START" if i <= 9 else "BENCH"
    name = b.get('name', 'Unknown')
    pos = ','.join(b.get('positions', [])[:2])  # First 2 positions
    score = b.get('score', 0)
    reason = b.get('reason', '')
    print(f"   {i:2}. {status:5} | {name:20} | {pos:10} | Score: {score:.2f}")
    print(f"       Reason: {reason[:50]}")

print("\n4. PROBLEMS IDENTIFIED:")
print("   " + "-" * 60)

# Check for position issues
positions_filled = {'C': 0, '1B': 0, '2B': 0, '3B': 0, 'SS': 0, 'OF': 0, 'Util': 0}
for i, b in enumerate(batters[:9], 1):  # Top 9 "starters"
    pos_list = b.get('positions', [])
    for pos in pos_list:
        if pos in positions_filled:
            positions_filled[pos] += 1
        if pos in ['LF', 'CF', 'RF']:
            positions_filled['OF'] += 1

print("   Position coverage in 'top 9':")
for pos, count in positions_filled.items():
    if pos == 'OF':
        needed = 3
    elif pos == 'Util':
        needed = 0  # Util is extra
    else:
        needed = 1
    status = "OK" if count >= needed else "MISSING"
    print(f"     {pos}: {count} (need {needed}) [{status}]")

# Check for off-day players
print("\n   Off-day check (teams playing today):")
playing_teams = set()
for g in report.get('games', []):
    playing_teams.add(g['home'])
    playing_teams.add(g['away'])

if playing_teams:
    off_day_starters = []
    for i, b in enumerate(batters[:9], 1):
        if b.get('team') not in playing_teams:
            off_day_starters.append(b.get('name'))

    if off_day_starters:
        print(f"   CRITICAL: These 'starters' have NO GAME today:")
        for name in off_day_starters:
            print(f"       - {name}")
    else:
        print("   OK: All top 9 have games today")
else:
    print("   Cannot check - no game data available")

print("\n5. WHAT THE OPTIMIZER SHOULD DO:")
print("   " + "-" * 60)
print("""
   CURRENT (Broken):
   1. Rank all batters by lineup_score
   2. START top 9, BENCH rest
   
   PROBLEMS:
   - Ignores position requirements (might bench your only Catcher!)
   - Ignores off-days (might start player with no game)
   - Ignores IL/DTD status
   - Doesn't fill specific slots (C/1B/2B/3B/SS/OF/Util)
   
   PROPER ALGORITHM:
   1. Filter to active players on teams playing today
   2. Fill mandatory positions first:
      - C: Must have 1 (often only 1 on roster)
      - SS: Must have 1
   3. Fill other positions with best available eligible:
      - 1B: Best 1B-eligible
      - 2B: Best 2B-eligible (Semien/Castro)
      - 3B: Best 3B-eligible (Castro can flex here)
      - OF1/OF2/OF3: Best 3 OFs (Soto, Nimmo, Buxton/Frelick)
   4. Fill Util with best remaining batter
   5. Use multi-position eligibility (Castro: 2B/3B/LF/RF) to optimize
   
   KEY OPTIMIZATIONS MISSING:
   - Park factor stacking (play hitters at Coors, bench at Petco)
   - Opponent pitcher quality (bench vs deGrom, start vs weak SP)
   - Handedness splits (lefties vs RHP)
   - Recent hot/cold streaks
   - Weather (wind at Wrigley)
""")

print("\n6. YOUR SPECIFIC ROSTER ANALYSIS:")
print("   " + "-" * 60)

# Group by position
by_pos = {}
for p in roster:
    name = p.get('name')
    pos = p.get('positions', [])
    status = p.get('status') or 'Active'
    
    for pos_code in pos:
        if pos_code not in ('Util', 'IL'):
            by_pos.setdefault(pos_code, []).append((name, status))

print("   Position depth:")
for pos in ['C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF']:
    if pos in by_pos:
        players = by_pos[pos]
        active = [n for n, s in players if s == 'Active']
        hurt = [n for n, s in players if s != 'Active']
        print(f"     {pos}: {len(active)} active", end="")
        if hurt:
            print(f" (+ {len(hurt)} injured: {', '.join(hurt)})")
        else:
            print()

print("\n   Roster spots to fill daily:")
print("     C:  Yainer Diaz (only option)")
print("     1B: Alonso (start) vs Pasquantino/Torkelson (bench)")
print("     2B: Semien (new pickup, should start) or Castro (flex)")
print("     3B: Chapman (start) or Castro (flex)")
print("     SS: Perdomo (only option)")
print("     OF: Soto, Nimmo, Buxton (start 3)")
print("     Util: Castro (if not at 2B/3B) or Frelick or Crow-Armstrong")
print("\n     Key decision: Castro's position flexibility maximizes value")
print("     - If Semien at 2B, Castro can play 3B or OF")
print("     - This allows better overall lineup")

print("\n" + "=" * 70)
print("CONCLUSION: Optimizer needs complete rewrite with position constraints")
print("=" * 70)
