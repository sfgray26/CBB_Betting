import json
from collections import Counter

def load(path):
    return json.load(open(path, encoding='utf-8'))

TS = "20260421_210933"
files = {
    'draft_board': f'postman_collections/responses/api_fantasy_draft_board_limit_200_{TS}.json',
    'decisions': f'postman_collections/responses/api_fantasy_decisions_{TS}.json',
    'waiver': f'postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_{TS}.json',
    'lineup': f'postman_collections/responses/api_fantasy_lineup_2026_04_20_{TS}.json',
    'briefing': f'postman_collections/responses/api_fantasy_briefing_2026_04_20_{TS}.json',
    'matchup': f'postman_collections/responses/api_fantasy_matchup_{TS}.json',
    'pipeline': f'postman_collections/responses/admin_pipeline_health_{TS}.json',
    'roster': f'postman_collections/responses/api_fantasy_roster_{TS}.json',
    'waiver_recs': f'postman_collections/responses/api_fantasy_waiver_recommendations_{TS}.json',
}

data = {k: load(v) for k, v in files.items()}

print("=" * 70)
print("LATEST PROBE ANALYSIS -- April 21, 2026 21:09 UTC")
print("=" * 70)

# 1. ROSTER
print("\n[1] ROSTER")
print("-" * 40)
roster = data['roster']
print(f"Status: 200, size={len(json.dumps(roster))} bytes")
players = roster.get('players', [])
print(f"Players: {len(players)}")
if players:
    print(f"First player: {players[0].get('player_name')}")
    print(f"Keys: {sorted(players[0].keys())}")
    
    for window in ['rolling_7d', 'rolling_14d', 'rolling_15d', 'rolling_30d']:
        has_data = sum(1 for p in players if p.get(window))
        print(f"  {window}: {has_data}/{len(players)}")
    
    has_ros = sum(1 for p in players if p.get('ros_projection'))
    has_row = sum(1 for p in players if p.get('row_projection'))
    has_game_ctx = sum(1 for p in players if p.get('game_context'))
    has_injury = sum(1 for p in players if p.get('injury_status'))
    has_bdl = sum(1 for p in players if p.get('bdl_player_id'))
    has_mlbam = sum(1 for p in players if p.get('mlbam_id'))
    print(f"  ros_projection: {has_ros}/{len(players)}")
    print(f"  row_projection: {has_row}/{len(players)}")
    print(f"  game_context: {has_game_ctx}/{len(players)}")
    print(f"  injury_status: {has_injury}/{len(players)}")
    print(f"  bdl_player_id: {has_bdl}/{len(players)}")
    print(f"  mlbam_id: {has_mlbam}/{len(players)}")
    
    # Show first player's season_stats
    print(f"\n  First player season_stats:")
    ss = players[0].get('season_stats', {})
    vals = ss.get('values', {})
    for k, v in sorted(vals.items()):
        if v is not None:
            print(f"    {k}: {v}")

# 2. WAIVER -- NOW 503
print("\n[2] WAIVER")
print("-" * 40)
waiver = data['waiver']
print(f"Status: {waiver.get('detail', 'Unknown')[:100]}")

# 3. WAIVER RECS -- NOW 503
print("\n[3] WAIVER RECOMMENDATIONS")
print("-" * 40)
wrecs = data['waiver_recs']
print(f"Status: {wrecs.get('detail', 'Unknown')[:100]}")

# 4. LINEUP
print("\n[4] LINEUP")
print("-" * 40)
lineup = data['lineup']
print(f"games_count: {lineup.get('games_count')}")
print(f"no_games_today: {lineup.get('no_games_today')}")
batters = lineup.get('batters', [])
pitchers = lineup.get('pitchers', [])
print(f"Batters: {len(batters)}, Pitchers: {len(pitchers)}")
if batters:
    has_game = sum(1 for b in batters if b.get('has_game'))
    print(f"  has_game=True: {has_game}/{len(batters)}")
    statuses = Counter(b.get('status') for b in batters)
    print(f"  statuses: {dict(statuses)}")
if pitchers:
    statuses = Counter(p.get('status') for p in pitchers)
    print(f"  pitcher statuses: {dict(statuses)}")

# 5. MATCHUP
print("\n[5] MATCHUP")
print("-" * 40)
matchup = data['matchup']
my_stats = matchup.get('my_team', {}).get('stats', {})
opp_stats = matchup.get('opponent', {}).get('stats', {})
print(f"My team: {matchup.get('my_team', {}).get('team_name')}")
print(f"Opponent: {matchup.get('opponent', {}).get('team_name')}")
for k in sorted(set(my_stats.keys()) | set(opp_stats.keys())):
    my_v = my_stats.get(k, 'N/A')
    opp_v = opp_stats.get(k, 'N/A')
    print(f"  {k}: my={my_v!r} opp={opp_v!r}")

# 6. BRIEFING
print("\n[6] BRIEFING")
print("-" * 40)
briefing = data['briefing']
print(f"Categories: {len(briefing.get('categories', []))}")
print(f"Starters: {len(briefing.get('starters', []))}")
print(f"Bench: {len(briefing.get('bench', []))}")
cats = briefing.get('categories', [])
if cats:
    print(f"Names: {[c['category'] for c in cats]}")

# 7. DECISIONS
print("\n[7] DECISIONS")
print("-" * 40)
decisions = data['decisions']
print(f"as_of_date: {decisions.get('as_of_date')}")
decs = decisions.get('decisions', [])
lineup_decs = [d for d in decs if d['decision']['decision_type'] == 'lineup']
waiver_decs = [d for d in decs if d['decision']['decision_type'] == 'waiver']
print(f"Lineup: {len(lineup_decs)}, Waiver: {len(waiver_decs)}")

if waiver_decs:
    drops = Counter(d['decision'].get('drop_player_name') for d in waiver_decs)
    print(f"Drop targets: {dict(drops)}")

# Check impossible projections
impossible = []
for d in decs:
    for factor in d.get('explanation', {}).get('factors', []):
        narrative = factor.get('narrative', '')
        if '0.00 ERA' in narrative or '0.00 WHIP' in narrative:
            impossible.append((d['decision']['player_name'], narrative))
print(f"Impossible projections (0.00 ERA/WHIP): {len(impossible)}")
for name, narrative in impossible[:5]:
    print(f"  {name}: {narrative}")

# 8. DRAFT BOARD
print("\n[8] DRAFT BOARD")
print("-" * 40)
draft = data['draft_board']
players = draft.get('players', [])
ages = [p.get('age', 0) for p in players]
print(f"Total: {len(players)}")
print(f"Age=0: {sum(1 for a in ages if a == 0)}/{len(players)}")

# 9. PIPELINE
print("\n[9] PIPELINE")
print("-" * 40)
pipeline = data['pipeline']
print(f"Overall: {pipeline.get('overall_healthy')}")
for t in pipeline.get('tables', []):
    print(f"  {t['name']}: rows={t['row_count']}, latest={t['latest_date']}")

print("\n" + "=" * 70)
print("END")
print("=" * 70)
