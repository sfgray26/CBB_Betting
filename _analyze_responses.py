import json
from datetime import datetime

# Load all response files
files = {
    'draft_board': 'postman_collections/responses/api_fantasy_draft_board_limit_200_20260420_181810.json',
    'decisions': 'postman_collections/responses/api_fantasy_decisions_20260420_181810.json',
    'waiver': 'postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_20260420_181810.json',
    'lineup': 'postman_collections/responses/api_fantasy_lineup_2026_04_20_20260420_181810.json',
    'briefing': 'postman_collections/responses/api_fantasy_briefing_2026_04_20_20260420_181810.json',
    'matchup': 'postman_collections/responses/api_fantasy_matchup_20260420_181810.json',
    'pipeline': 'postman_collections/responses/admin_pipeline_health_20260420_181810.json',
    'scheduler': 'postman_collections/responses/admin_scheduler_status_20260420_181810.json',
    'roster': 'postman_collections/responses/api_fantasy_roster_20260420_181810.json',
    'waiver_recs': 'postman_collections/responses/api_fantasy_waiver_recommendations_20260420_181810.json',
}

data = {}
for name, path in files.items():
    try:
        data[name] = json.load(open(path))
    except Exception as e:
        data[name] = {"_error": str(e)}

print("=" * 70)
print("ANALYSIS 1: WAIVER ENDPOINT - Field Completeness")
print("=" * 70)

waiver = data.get('waiver', {})
players = waiver.get('top_available', [])
print(f"Total players: {len(players)}")

if players:
    # Check which fields are present and populated
    schema_fields = [
        'player_id', 'name', 'team', 'position', 'need_score',
        'category_contributions', 'owned_pct', 'starts_this_week',
        'statcast_signals', 'projected_saves', 'projected_points',
        'hot_cold', 'status', 'injury_note', 'injury_status', 'stats',
        'statcast_stats'
    ]
    
    for field in schema_fields:
        present_count = sum(1 for p in players if field in p)
        null_count = sum(1 for p in players if field in p and p[field] is None)
        empty_dict = sum(1 for p in players if field in p and p[field] == {})
        empty_list = sum(1 for p in players if field in p and p[field] == [])
        zero_count = sum(1 for p in players if field in p and p[field] == 0)
        zero_float = sum(1 for p in players if field in p and p[field] == 0.0)
        
        print(f"  {field}: present={present_count}/{len(players)}", end="")
        if null_count:
            print(f", null={null_count}", end="")
        if empty_dict:
            print(f", empty_dict={empty_dict}", end="")
        if empty_list:
            print(f", empty_list={empty_list}", end="")
        if zero_count:
            print(f", zero={zero_count}", end="")
        if zero_float:
            print(f", zero_float={zero_float}", end="")
        print()
    
    # Check category_contributions in detail
    print("\n  category_contributions detail:")
    non_empty_cc = [p for p in players if p.get('category_contributions')]
    print(f"    Non-empty: {len(non_empty_cc)}/{len(players)}")
    if non_empty_cc:
        print(f"    Sample keys: {list(non_empty_cc[0]['category_contributions'].keys())}")
    
    # Check stats field
    print("\n  stats field detail:")
    for i, p in enumerate(players[:5]):
        stats = p.get('stats', {})
        print(f"    Player {i+1} ({p.get('name', '?')}): {stats}")
    
    # Check for pitchers with wins in batter stats (schema pollution)
    print("\n  Schema pollution check (batters with pitcher stats):")
    for p in players:
        if p.get('position') in ['1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF', 'C']:
            stats = p.get('stats', {})
            if 'IP' in stats or 'W' in stats or 'ERA' in stats:
                print(f"    {p.get('name')} ({p.get('position')}): has pitcher stats: {stats}")
                break
    
    # Check for pitchers with batter stats
    print("\n  Schema pollution check (pitchers with batter stats):")
    for p in players:
        if p.get('position') in ['SP', 'RP', 'P']:
            stats = p.get('stats', {})
            if 'AVG' in stats or 'HR' in stats or 'RBI' in stats:
                print(f"    {p.get('name')} ({p.get('position')}): has batter stats: {stats}")
                break

print("\n" + "=" * 70)
print("ANALYSIS 2: LINEUP ENDPOINT")
print("=" * 70)

lineup = data.get('lineup', {})
print(f"Date: {lineup.get('date')}")
print(f"Games count: {lineup.get('games_count')}")
print(f"No games today: {lineup.get('no_games_today')}")
print(f"Batters: {len(lineup.get('batters', []))}")
print(f"Pitchers: {len(lineup.get('pitchers', []))}")
print(f"Warnings: {len(lineup.get('lineup_warnings', []))}")

batters = lineup.get('batters', [])
if batters:
    # Check field uniformity
    print("\n  Batter field analysis:")
    fields_to_check = ['position', 'implied_runs', 'park_factor', 'lineup_score', 
                       'start_time', 'opponent', 'status', 'has_game', 'injury_status']
    for field in fields_to_check:
        values = [b.get(field) for b in batters]
        unique = set(str(v) for v in values)
        nulls = sum(1 for v in values if v is None)
        print(f"    {field}: unique_values={len(unique)}, nulls={nulls}")
        if len(unique) <= 5:
            print(f"      Values: {unique}")

pitchers = lineup.get('pitchers', [])
if pitchers:
    print("\n  Pitcher field analysis:")
    fields_to_check = ['pitcher_type', 'opponent', 'opponent_implied_runs', 'park_factor',
                       'sp_score', 'start_time', 'status', 'is_confirmed', 'injury_status']
    for field in fields_to_check:
        values = [p.get(field) for p in pitchers]
        unique = set(str(v) for v in values)
        nulls = sum(1 for v in values if v is None)
        print(f"    {field}: unique_values={len(unique)}, nulls={nulls}")
        if len(unique) <= 5:
            print(f"      Values: {unique}")

print("\n" + "=" * 70)
print("ANALYSIS 3: MATCHUP ENDPOINT")
print("=" * 70)

matchup = data.get('matchup', {})
my_team = matchup.get('my_team', {})
opp_team = matchup.get('opponent', {})
my_stats = my_team.get('stats', {})
opp_stats = opp_team.get('stats', {})

print(f"Week: {matchup.get('week')}")
print(f"My team: {my_team.get('team_name')} ({my_team.get('team_key')})")
print(f"Opponent: {opp_team.get('team_name')} ({opp_team.get('team_key')})")

print("\n  My stats keys:")
for k, v in sorted(my_stats.items()):
    print(f"    {k}: {v} (type: {type(v).__name__})")

print("\n  Opponent stats keys:")
for k, v in sorted(opp_stats.items()):
    print(f"    {k}: {v} (type: {type(v).__name__})")

# Check missing keys
all_keys = set(my_stats.keys()) | set(opp_stats.keys())
print(f"\n  Total unique keys: {len(all_keys)}")
my_missing = all_keys - set(my_stats.keys())
opp_missing = all_keys - set(opp_stats.keys())
if my_missing:
    print(f"  Missing from my_team: {my_missing}")
if opp_missing:
    print(f"  Missing from opponent: {opp_missing}")

print("\n" + "=" * 70)
print("ANALYSIS 4: BRIEFING ENDPOINT")
print("=" * 70)

briefing = data.get('briefing', {})
print(f"Date: {briefing.get('date')}")
print(f"Generated: {briefing.get('generated_at')}")
print(f"Strategy: {briefing.get('strategy')}")
print(f"Risk profile: {briefing.get('risk_profile')}")
print(f"Overall confidence: {briefing.get('overall_confidence')}")
print(f"Categories: {len(briefing.get('categories', []))}")
print(f"Starters: {len(briefing.get('starters', []))}")
print(f"Bench: {len(briefing.get('bench', []))}")
print(f"Monitor: {len(briefing.get('monitor', []))}")
print(f"Alerts: {len(briefing.get('alerts', []))}")

if briefing.get('categories'):
    print("\n  Categories:")
    for c in briefing['categories']:
        print(f"    {c['category']}: me={c['current']} opp={c['opponent']} status={c['status']} urgency={c['urgency']}")

if briefing.get('starters'):
    print("\n  Starter sample:")
    for s in briefing['starters'][:3]:
        print(f"    {s['name']}: {s['recommendation']} (confidence={s['confidence']}, rating={s['rating']})")
        print(f"      Factors: {s['factors']}")

print("\n" + "=" * 70)
print("ANALYSIS 5: DRAFT BOARD - Sample Player Deep Dive")
print("=" * 70)

draft = data.get('draft_board', {})
players = draft.get('players', [])
print(f"Total players: {len(players)}")

if players:
    # Check first player structure
    p = players[0]
    print(f"\n  First player ({p.get('name')}):")
    for k, v in p.items():
        if isinstance(v, dict):
            print(f"    {k}: {type(v).__name__} with keys {list(v.keys())}")
        elif isinstance(v, list):
            print(f"    {k}: list[{len(v)}]")
        else:
            print(f"    {k}: {v} ({type(v).__name__})")
    
    # Check for proxy indicators
    print("\n  Checking for proxy/placeholder indicators:")
    ages = [p.get('age') for p in players]
    print(f"    Age distribution: min={min(ages)}, max={max(ages)}, zero_count={sum(1 for a in ages if a == 0)}")
    
    keeper_rounds = [p.get('keeper_round') for p in players if p.get('keeper_round') is not None]
    print(f"    Players with keeper_round: {len(keeper_rounds)}")
    
    # Check for missing z_park_adjusted or z_risk_adjusted
    missing_park = sum(1 for p in players if p.get('z_park_adjusted') is None)
    missing_risk = sum(1 for p in players if p.get('z_risk_adjusted') is None)
    print(f"    Missing z_park_adjusted: {missing_park}")
    print(f"    Missing z_risk_adjusted: {missing_risk}")

print("\n" + "=" * 70)
print("ANALYSIS 6: DECISIONS ENDPOINT")
print("=" * 70)

decisions = data.get('decisions', {})
decs = decisions.get('decisions', [])
print(f"Total decisions: {len(decs)}")

if decs:
    lineup_decs = [d for d in decs if d['decision']['decision_type'] == 'lineup']
    waiver_decs = [d for d in decs if d['decision']['decision_type'] == 'waiver']
    print(f"  Lineup: {len(lineup_decs)}")
    print(f"  Waiver: {len(waiver_decs)}")
    
    # Check waiver drop targets
    if waiver_decs:
        drop_targets = [d['decision'].get('drop_player_name') for d in waiver_decs]
        from collections import Counter
        drop_counts = Counter(drop_targets)
        print(f"\n  Waiver drop targets:")
        for name, count in drop_counts.most_common():
            print(f"    {name}: {count}/{len(waiver_decs)}")
    
    # Check for impossible projections
    print("\n  Checking for impossible projection narratives:")
    impossible = []
    for d in decs:
        for factor in d.get('explanation', {}).get('factors', []):
            narrative = factor.get('narrative', '')
            if '0.00 ERA' in narrative or '0.00 WHIP' in narrative or '91.2 HR' in narrative or '204.4 RBI' in narrative:
                impossible.append((d['decision']['player_name'], narrative))
    
    if impossible:
        print(f"    Found {len(impossible)} impossible projections:")
        for name, narrative in impossible[:10]:
            print(f"      {name}: {narrative}")
    else:
        print("    None found")

print("\n" + "=" * 70)
print("ANALYSIS 7: ERROR RESPONSES")
print("=" * 70)

for name in ['roster', 'waiver_recs', 'player_scores']:
    d = data.get(name, {})
    if '_error' in d:
        print(f"  {name}: {d['_error']}")
    else:
        print(f"  {name}: {json.dumps(d)[:200]}")

print("\n" + "=" * 70)
print("ANALYSIS 8: PIPELINE HEALTH")
print("=" * 70)

pipeline = data.get('pipeline', {})
print(f"Overall healthy: {pipeline.get('overall_healthy')}")
for t in pipeline.get('tables', []):
    print(f"  {t['name']}: rows={t['row_count']}, latest={t['latest_date']}, healthy={t['healthy']}")
