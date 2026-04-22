import json
from collections import Counter
from datetime import datetime

def load(path):
    return json.load(open(path, encoding='utf-8'))

TS = "20260421_190158"
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
print("FRESH PROBE ANALYSIS — April 21, 2026")
print("=" * 70)

# 1. ROSTER — what's fixed vs what's not
print("\n[1] ROSTER ENDPOINT")
print("-" * 40)
roster = data['roster']
print(f"Status: 200, size={len(json.dumps(roster))} bytes")
print(f"Team key: {roster.get('team_key')}")
print(f"Player count: {roster.get('count')}")
players = roster.get('players', [])
if players:
    print(f"First player: {players[0].get('name')}")
    print(f"Keys: {list(players[0].keys())}")
    
    # Check proxy players
    proxies = [p for p in players if p.get('is_proxy')]
    print(f"Proxy players: {len(proxies)}")
    for p in proxies:
        print(f"  {p.get('name')}: z_score={p.get('z_score')}, cat_scores={p.get('cat_scores')}")
    
    # Check season_stats
    has_season = sum(1 for p in players if p.get('season_stats'))
    print(f"Players with season_stats: {has_season}/{len(players)}")
    
    # Check rolling_14d
    has_rolling = sum(1 for p in players if p.get('rolling_14d'))
    print(f"Players with rolling_14d: {has_rolling}/{len(players)}")
    
    # Check freshness
    print(f"Freshness: {roster.get('freshness')}")

# 2. WAIVER RECOMMENDATIONS — was 503, now 200
print("\n[2] WAIVER RECOMMENDATIONS")
print("-" * 40)
wrecs = data['waiver_recs']
print(f"Status: 200, size={len(json.dumps(wrecs))} bytes")
print(f"Keys: {list(wrecs.keys())}")
recs = wrecs.get('recommendations', [])
print(f"Recommendations: {len(recs)}")
if recs:
    for i, r in enumerate(recs[:3]):
        print(f"  Rec {i+1}: {r.get('action')} — {r.get('add_player',{}).get('name','?')} for {r.get('drop_player_name','?')}")
        print(f"    need_score={r.get('need_score')}, confidence={r.get('confidence')}")
        print(f"    win_prob_before={r.get('win_prob_before')}, win_prob_after={r.get('win_prob_after')}")
        print(f"    category_targets={r.get('category_targets')}")

# 3. WAIVER — check if hollow data is fixed
print("\n[3] WAIVER ENDPOINT")
print("-" * 40)
waiver = data['waiver']
players = waiver.get('top_available', [])
print(f"Players: {len(players)}")
if players:
    # Check key fields that were previously empty
    fields = ['category_contributions', 'owned_pct', 'starts_this_week', 
              'hot_cold', 'projected_saves', 'projected_points', 'statcast_signals']
    for f in fields:
        non_empty = sum(1 for p in players if p.get(f) not in [None, {}, [], 0, 0.0, ''])
        print(f"  {f}: non_empty={non_empty}/{len(players)}")
    
    # Show sample player
    p = players[0]
    print(f"\n  Sample player ({p.get('name')}):")
    for k, v in p.items():
        if isinstance(v, dict):
            print(f"    {k}: dict with keys {list(v.keys())}")
        elif isinstance(v, list):
            print(f"    {k}: list len={len(v)}")
        else:
            print(f"    {k}: {v!r}")

# 4. LINEUP — check if schedule blindness is fixed
print("\n[4] LINEUP ENDPOINT")
print("-" * 40)
lineup = data['lineup']
print(f"Date: {lineup.get('date')}")
print(f"Games count: {lineup.get('games_count')}")
print(f"No games: {lineup.get('no_games_today')}")
batters = lineup.get('batters', [])
pitchers = lineup.get('pitchers', [])
print(f"Batters: {len(batters)}, Pitchers: {len(pitchers)}")
if batters:
    positions = Counter(b.get('position') for b in batters)
    print(f"  Batter positions: {dict(positions)}")
    statuses = Counter(b.get('status') for b in batters)
    print(f"  Batter statuses: {dict(statuses)}")
    has_games = sum(1 for b in batters if b.get('has_game'))
    print(f"  has_game=True: {has_games}/{len(batters)}")
if pitchers:
    statuses = Counter(p.get('status') for p in pitchers)
    print(f"  Pitcher statuses: {dict(statuses)}")

# 5. MATCHUP
print("\n[5] MATCHUP ENDPOINT")
print("-" * 40)
matchup = data['matchup']
my_stats = matchup.get('my_team', {}).get('stats', {})
opp_stats = matchup.get('opponent', {}).get('stats', {})
print(f"My keys ({len(my_stats)}): {sorted(my_stats.keys())}")
print(f"Opp keys ({len(opp_stats)}): {sorted(opp_stats.keys())}")
for k in sorted(set(my_stats.keys()) | set(opp_stats.keys())):
    my_v = my_stats.get(k, 'MISSING')
    opp_v = opp_stats.get(k, 'MISSING')
    print(f"  {k}: my={my_v!r} opp={opp_v!r}")

# 6. BRIEFING
print("\n[6] BRIEFING ENDPOINT")
print("-" * 40)
briefing = data['briefing']
print(f"Categories: {len(briefing.get('categories', []))}")
print(f"Starters: {len(briefing.get('starters', []))}")
print(f"Bench: {len(briefing.get('bench', []))}")
cats = briefing.get('categories', [])
if cats:
    print(f"Category names: {[c['category'] for c in cats]}")

# 7. DECISIONS
print("\n[7] DECISIONS ENDPOINT")
print("-" * 40)
decisions = data['decisions']
print(f"as_of_date: {decisions.get('as_of_date')}")
print(f"count: {decisions.get('count')}")
decs = decisions.get('decisions', [])
lineup = [d for d in decs if d['decision']['decision_type'] == 'lineup']
waiver = [d for d in decs if d['decision']['decision_type'] == 'waiver']
print(f"Lineup: {len(lineup)}, Waiver: {len(waiver)}")

# Check universal drop bug
if waiver:
    drops = Counter(d['decision'].get('drop_player_name') for d in waiver)
    print(f"Drop targets: {dict(drops)}")

# Check impossible projections
impossible = []
for d in decs:
    for factor in d.get('explanation', {}).get('factors', []):
        narrative = factor.get('narrative', '')
        if '0.00 ERA' in narrative or '0.00 WHIP' in narrative or '91.2 HR' in narrative or '204.4 RBI' in narrative:
            impossible.append((d['decision']['player_name'], narrative))
print(f"Impossible projections: {len(impossible)}")
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
print("\n[9] PIPELINE HEALTH")
print("-" * 40)
pipeline = data['pipeline']
print(f"Overall: {pipeline.get('overall_healthy')}")
for t in pipeline.get('tables', []):
    print(f"  {t['name']}: rows={t['row_count']}, latest={t['latest_date']}")

print("\n" + "=" * 70)
print("END")
print("=" * 70)
