import json
from collections import Counter

def load(path):
    return json.load(open(path, encoding='utf-8'))

TS = "20260421_190158"

# Deep dive into specific changes
print("=" * 70)
print("FRESH DEEP DIVE — April 21, 2026")
print("=" * 70)

# 1. ROSTER deep dive
roster = load(f'postman_collections/responses/api_fantasy_roster_{TS}.json')
print("\n[1] ROSTER STRUCTURE CHANGE")
print("-" * 40)
players = roster.get('players', [])
print(f"Total players: {len(players)}")
if players:
    print(f"Keys per player: {sorted(players[0].keys())}")
    
    # Check for None names
    none_names = [p for p in players if p.get('player_name') is None]
    print(f"Players with None name: {len(none_names)}")
    
    # Check season_stats content
    p = players[0]
    print(f"\n  First player: {p.get('player_name')}")
    print(f"  Yahoo key: {p.get('yahoo_player_key')}")
    print(f"  BDL ID: {p.get('bdl_player_id')}")
    print(f"  MLBAM ID: {p.get('mlbam_id')}")
    print(f"  Season stats: {p.get('season_stats')}")
    print(f"  Rolling 14d: {p.get('rolling_14d')}")
    print(f"  Rolling 7d: {p.get('rolling_7d')}")
    print(f"  ROS projection: {p.get('ros_projection')}")
    print(f"  ROW projection: {p.get('row_projection')}")
    print(f"  Ownership %: {p.get('ownership_pct')}")
    print(f"  Game context: {p.get('game_context')}")
    print(f"  Injury status: {p.get('injury_status')}")
    print(f"  Injury timeline: {p.get('injury_return_timeline')}")
    
    # Count rolling data availability
    for window in ['rolling_7d', 'rolling_14d', 'rolling_15d', 'rolling_30d']:
        has_data = sum(1 for p in players if p.get(window))
        print(f"  {window}: {has_data}/{len(players)}")

# 2. WAIVER deep dive — what's new
waiver = load(f'postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_{TS}.json')
print("\n[2] WAIVER — NEW vs OLD COMPARISON")
print("-" * 40)
players = waiver.get('top_available', [])
print(f"Total players: {len(players)}")

# Check category_contributions
print("\n  category_contributions detail:")
for p in players:
    cc = p.get('category_contributions', {})
    if cc:
        print(f"    {p['name']}: {cc}")

# Check hot_cold
print("\n  hot_cold detail:")
for p in players:
    hc = p.get('hot_cold')
    if hc:
        print(f"    {p['name']}: {hc}")

# Check projected_saves
print("\n  projected_saves detail:")
for p in players:
    ps = p.get('projected_saves')
    if ps:
        print(f"    {p['name']}: {ps}")

# Check stats field keys
print("\n  stats field keys (all players):")
all_keys = Counter()
for p in players:
    for k in p.get('stats', {}).keys():
        all_keys[k] += 1
for k, count in all_keys.most_common():
    print(f"    '{k}': {count}/{len(players)}")

# Check for stat ID 38
if '38' in all_keys:
    print(f"\n  [BUG] Stat ID '38' still present: {all_keys['38']}/{len(players)} players")
else:
    print(f"\n  [FIXED] Stat ID '38' not present")

# Check K_P values
print("\n  K_P values for SPs:")
for p in players:
    if p.get('position') == 'SP':
        stats = p.get('stats', {})
        print(f"    {p['name']}: IP={stats.get('IP')}, K_P={stats.get('K_P')}, K_9={stats.get('K_9')}")

# 3. WAIVER RECOMMENDATIONS deep dive
wrecs = load(f'postman_collections/responses/api_fantasy_waiver_recommendations_{TS}.json')
print("\n[3] WAIVER RECOMMENDATIONS")
print("-" * 40)
print(f"Keys: {list(wrecs.keys())}")
print(f"Opponent: {wrecs.get('matchup_opponent')}")
print(f"Category deficits: {wrecs.get('category_deficits')}")
recs = wrecs.get('recommendations', [])
print(f"Recommendations: {len(recs)}")
for i, r in enumerate(recs):
    print(f"\n  Rec {i+1}:")
    print(f"    action: {r.get('action')}")
    add = r.get('add_player', {})
    print(f"    add: {add.get('name')} ({add.get('team')}, {add.get('position')})")
    print(f"    drop: {r.get('drop_player_name')} ({r.get('drop_player_position')})")
    print(f"    need_score: {r.get('need_score')}")
    print(f"    confidence: {r.get('confidence')}")
    print(f"    category_targets: {r.get('category_targets')}")
    print(f"    win_prob_before: {r.get('win_prob_before')}")
    print(f"    win_prob_after: {r.get('win_prob_after')}")
    print(f"    win_prob_gain: {r.get('win_prob_gain')}")
    print(f"    category_win_probs keys: {list(r.get('category_win_probs', {}).keys())}")
    print(f"    mcmc_enabled: {r.get('mcmc_enabled')}")

# 4. LINEUP deep dive
lineup = load(f'postman_collections/responses/api_fantasy_lineup_2026_04_20_{TS}.json')
print("\n[4] LINEUP — SCHEDULE NOW WORKING")
print("-" * 40)
print(f"games_count: {lineup.get('games_count')}")
print(f"no_games_today: {lineup.get('no_games_today')}")
batters = lineup.get('batters', [])
pitchers = lineup.get('pitchers', [])
print(f"Batters: {len(batters)}")
for b in batters[:5]:
    print(f"  {b.get('name')}: pos={b.get('position')} status={b.get('status')} has_game={b.get('has_game')} opponent={b.get('opponent')} lineup_score={b.get('lineup_score')}")

print(f"\nPitchers: {len(pitchers)}")
for p in pitchers[:5]:
    print(f"  {p.get('name')}: type={p.get('pitcher_type')} status={p.get('status')} opponent={p.get('opponent')} sp_score={p.get('sp_score')}")

# 5. MATCHUP deep dive
matchup = load(f'postman_collections/responses/api_fantasy_matchup_{TS}.json')
print("\n[5] MATCHUP — REAL DATA NOW")
print("-" * 40)
my_stats = matchup.get('my_team', {}).get('stats', {})
opp_stats = matchup.get('opponent', {}).get('stats', {})
print(f"My team: {matchup.get('my_team', {}).get('team_name')}")
print(f"Opponent: {matchup.get('opponent', {}).get('team_name')}")
for k in sorted(set(my_stats.keys()) | set(opp_stats.keys())):
    my_v = my_stats.get(k, 'N/A')
    opp_v = opp_stats.get(k, 'N/A')
    my_type = type(my_v).__name__
    opp_type = type(opp_v).__name__
    match = "[OK]" if my_type == opp_type else "[MISMATCH]"
    print(f"  {k}: my={my_v!r} ({my_type}) vs opp={opp_v!r} ({opp_type}) {match}")

# 6. DECISIONS deep dive
decisions = load(f'postman_collections/responses/api_fantasy_decisions_{TS}.json')
print("\n[6] DECISIONS")
print("-" * 40)
print(f"as_of_date: {decisions.get('as_of_date')}")
decs = decisions.get('decisions', [])
lineup_decs = [d for d in decs if d['decision']['decision_type'] == 'lineup']
waiver_decs = [d for d in decs if d['decision']['decision_type'] == 'waiver']
print(f"Lineup: {len(lineup_decs)}, Waiver: {len(waiver_decs)}")

# Drop targets
drops = Counter(d['decision'].get('drop_player_name') for d in waiver_decs)
print(f"Drop target distribution: {dict(drops)}")

# Check for confidence_narrative quality
print("\n  Sample decision explanations:")
for d in lineup_decs[:2]:
    print(f"    {d['decision']['player_name']}: {d['explanation'].get('summary')}")
    print(f"      confidence: {d['explanation'].get('confidence_narrative')}")
    print(f"      factors: {len(d['explanation'].get('factors', []))}")

# 7. DRAFT BOARD
draft = load(f'postman_collections/responses/api_fantasy_draft_board_limit_200_{TS}.json')
print("\n[7] DRAFT BOARD")
print("-" * 40)
players = draft.get('players', [])
ages = [p.get('age', 0) for p in players]
print(f"Total: {len(players)}")
print(f"Age=0: {sum(1 for a in ages if a == 0)}/{len(players)}")
print(f"Age>0: {sum(1 for a in ages if a > 0)}")

# 8. BRIEFING
briefing = load(f'postman_collections/responses/api_fantasy_briefing_2026_04_20_{TS}.json')
print("\n[8] BRIEFING")
print("-" * 40)
print(f"Categories: {len(briefing.get('categories', []))}")
print(f"Starters: {len(briefing.get('starters', []))}")
print(f"Bench: {len(briefing.get('bench', []))}")
print(f"Category names: {[c['category'] for c in briefing.get('categories', [])]}")

print("\n" + "=" * 70)
print("END")
print("=" * 70)
