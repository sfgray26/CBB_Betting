import json
from collections import Counter

# Load all response files
files = {
    'draft_board': 'postman_collections/responses/api_fantasy_draft_board_limit_200_20260420_181810.json',
    'decisions': 'postman_collections/responses/api_fantasy_decisions_20260420_181810.json',
    'waiver': 'postman_collections/responses/api_fantasy_waiver_position_ALL_player_type_ALL_20260420_181810.json',
    'lineup': 'postman_collections/responses/api_fantasy_lineup_2026_04_20_20260420_181810.json',
    'briefing': 'postman_collections/responses/api_fantasy_briefing_2026_04_20_20260420_181810.json',
    'matchup': 'postman_collections/responses/api_fantasy_matchup_20260420_181810.json',
}

data = {}
for name, path in files.items():
    data[name] = json.load(open(path))

print("=" * 70)
print("DEEP DIVE 1: WAIVER STATS FIELD - Schema Pollution & Stat ID Bug")
print("=" * 70)

waiver = data['waiver']
players = waiver['top_available']

# Analyze all unique stat keys across all players
all_stat_keys = Counter()
for p in players:
    for k in p.get('stats', {}).keys():
        all_stat_keys[k] += 1

print("All stat keys found in waiver endpoint:")
for k, count in all_stat_keys.most_common():
    print(f"  '{k}': {count}/25 players")

# Check the mysterious '38' key
print("\nPlayers with '38' key:")
for p in players:
    stats = p.get('stats', {})
    if '38' in stats:
        print(f"  {p['name']} ({p['position']}): '38' = {stats['38']}")

# Check stat value types
print("\nStat value types:")
for k in all_stat_keys:
    types = Counter()
    for p in players:
        v = p.get('stats', {}).get(k)
        types[type(v).__name__] += 1
    print(f"  {k}: {dict(types)}")

# Check NSB format inconsistency
print("\nNSB format analysis:")
nsb_values = [p.get('stats', {}).get('NSB') for p in players if 'NSB' in p.get('stats', {})]
print(f"  NSB values: {nsb_values}")

print("\n" + "=" * 70)
print("DEEP DIVE 2: DRAFT BOARD - Age=0 Crisis")
print("=" * 70)

draft = data['draft_board']
players = draft['players']

ages = [p.get('age', 0) for p in players]
print(f"Age statistics:")
print(f"  Total players: {len(players)}")
print(f"  Age=0: {sum(1 for a in ages if a == 0)} ({sum(1 for a in ages if a == 0)/len(ages)*100:.1f}%)")
print(f"  Age>0: {sum(1 for a in ages if a > 0)}")
print(f"  Min age (non-zero): {min((a for a in ages if a > 0), default='N/A')}")
print(f"  Max age: {max(ages)}")

# Show some age=0 players
print("\n  Sample age=0 players:")
for p in players:
    if p.get('age') == 0:
        print(f"    {p['name']} (team={p['team']}, rank={p['rank']})")
        if p['rank'] > 10:
            break

# Check proj completeness
print("\nProjection completeness:")
proj_keys = set()
for p in players:
    proj_keys.update(p.get('proj', {}).keys())
print(f"  Projection keys: {sorted(proj_keys)}")

missing_proj = []
for p in players:
    proj = p.get('proj', {})
    missing = [k for k in proj_keys if k not in proj]
    if missing:
        missing_proj.append((p['name'], missing))

if missing_proj:
    print(f"  Players with missing projection keys: {len(missing_proj)}")
    for name, missing in missing_proj[:5]:
        print(f"    {name}: missing {missing}")

print("\n" + "=" * 70)
print("DEEP DIVE 3: BRIEFING - Category Name Mismatch")
print("=" * 70)

briefing = data['briefing']
cats = briefing.get('categories', [])
print("Category names in briefing:")
for c in cats:
    print(f"  {c['category']}")

# Compare to v2 canonical categories
v2_canonical = ['R', 'H', 'HR_B', 'RBI', 'K_B', 'TB', 'NSB', 'AVG', 'OPS',
                'W', 'L', 'HR_P', 'K_P', 'QS', 'ERA', 'WHIP', 'K_9', 'NSV']
briefing_cats = [c['category'] for c in cats]
print(f"\nV2 canonical: {v2_canonical}")
print(f"Briefing has: {briefing_cats}")

missing_from_briefing = set(v2_canonical) - set(briefing_cats)
extra_in_briefing = set(briefing_cats) - set(v2_canonical)
print(f"\nMissing from briefing: {sorted(missing_from_briefing)}")
print(f"Extra in briefing (legacy?): {sorted(extra_in_briefing)}")

print("\n" + "=" * 70)
print("DEEP DIVE 4: MATCHUP - Type Inconsistencies")
print("=" * 70)

matchup = data['matchup']
my_stats = matchup['my_team']['stats']
opp_stats = matchup['opponent']['stats']

print("My team stat types:")
for k, v in sorted(my_stats.items()):
    print(f"  {k}: {v!r} ({type(v).__name__})")

print("\nOpponent stat types:")
for k, v in sorted(opp_stats.items()):
    print(f"  {k}: {v!r} ({type(v).__name__})")

# Check type consistency between teams
print("\nType consistency between teams:")
all_keys = set(my_stats.keys()) | set(opp_stats.keys())
for k in sorted(all_keys):
    my_type = type(my_stats.get(k)).__name__ if k in my_stats else 'MISSING'
    opp_type = type(opp_stats.get(k)).__name__ if k in opp_stats else 'MISSING'
    if my_type != opp_type:
        print(f"  {k}: my={my_type}, opp={opp_type}  <-- MISMATCH")

print("\n" + "=" * 70)
print("DEEP DIVE 5: LINEUP - Uniformity Crisis")
print("=" * 70)

lineup = data['lineup']
batters = lineup['batters']
pitchers = lineup['pitchers']

print("Batter field values (all 14 batters):")
for field in ['position', 'implied_runs', 'park_factor', 'lineup_score', 'opponent', 'status', 'has_game']:
    values = [b.get(field) for b in batters]
    unique = sorted(set(str(v) for v in values))
    print(f"  {field}: {unique}")

print("\nPitcher field values (all 9 pitchers):")
for field in ['opponent', 'opponent_implied_runs', 'park_factor', 'is_confirmed']:
    values = [p.get(field) for p in pitchers]
    unique = sorted(set(str(v) for v in values))
    print(f"  {field}: {unique}")

print("\nLineup warnings (first 10):")
for w in lineup['lineup_warnings'][:10]:
    print(f"  {w}")

print("\n" + "=" * 70)
print("DEEP DIVE 6: DECISIONS - Narrative Quality")
print("=" * 70)

decisions = data['decisions']['decisions']

# Count unique narrative patterns
narrative_patterns = Counter()
for d in decisions:
    for factor in d.get('explanation', {}).get('factors', []):
        narrative = factor.get('narrative', '')
        # Extract pattern (remove player-specific numbers)
        import re
        pattern = re.sub(r'\d+\.\d+|\d+', 'X', narrative)
        narrative_patterns[pattern] += 1

print("Most common narrative patterns:")
for pattern, count in narrative_patterns.most_common(15):
    print(f"  ({count}x) {pattern}")

# Check for duplicated ERA narrative
print("\nERA narrative duplication check:")
era_narratives = []
for d in decisions:
    for factor in d.get('explanation', {}).get('factors', []):
        if 'ERA' in factor.get('narrative', ''):
            era_narratives.append((d['decision']['player_name'], factor['narrative']))

for name, narrative in era_narratives:
    print(f"  {name}: {narrative}")

print("\n" + "=" * 70)
print("DEEP DIVE 7: WAIVER - Two-Start Pitchers")
print("=" * 70)

waiver = data['waiver']
two_start = waiver.get('two_start_pitchers', [])
print(f"Two-start pitchers: {len(two_start)}")
if two_start:
    for p in two_start:
        print(f"  {p['name']}")
else:
    print("  (empty list)")

print("\n" + "=" * 70)
print("DEEP DIVE 8: WAIVER - Pagination & Metadata")
print("=" * 70)

print(f"Week end: {waiver.get('week_end')}")
print(f"Matchup opponent: {waiver.get('matchup_opponent')}")
print(f"Category deficits: {waiver.get('category_deficits')}")
print(f"Pagination: {waiver.get('pagination')}")
print(f"Urgent alert: {waiver.get('urgent_alert')}")
print(f"Closer alert: {waiver.get('closer_alert')}")
print(f"IL slots used: {waiver.get('il_slots_used')}")
print(f"IL slots available: {waiver.get('il_slots_available')}")
print(f"FAAB balance: {waiver.get('faab_balance')}")
