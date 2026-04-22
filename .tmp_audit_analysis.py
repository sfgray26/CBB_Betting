import json

with open('postman_collections/responses/api_fantasy_roster_20260422_145543.json') as f:
    roster = json.load(f)

print('=== PLAYERS WITH NULL ROLLING WINDOWS ===')
for p in roster['players']:
    nulls = []
    for field in ['rolling_7d', 'rolling_14d', 'rolling_30d']:
        if p.get(field) is None:
            nulls.append(field)
    if nulls:
        print(f"  {p['player_name']}: {', '.join(nulls)}")

print('\n=== PLAYERS WITH NULL BDL/MLBAM IDs ===')
for p in roster['players']:
    if p.get('bdl_player_id') is None or p.get('mlbam_id') is None:
        print(f"  {p['player_name']}: bdl={p.get('bdl_player_id')}, mlbam={p.get('mlbam_id')}")

print('\n=== PLAYERS WITH INJURY STATUS ===')
for p in roster['players']:
    if p.get('injury_status'):
        print(f"  {p['player_name']}: {p['injury_status']}")

print('\n=== ROS/ROW PROJECTION STATUS ===')
for p in roster['players']:
    ros = p.get('ros_projection')
    row = p.get('row_projection')
    if ros is not None or row is not None:
        print(f"  {p['player_name']}: ros={ros is not None}, row={row is not None}")

with open('postman_collections/responses/api_fantasy_waiver_recommendations_20260422_145543.json') as f:
    recs = json.load(f)

print('\n=== WAIVER REC DETAIL ===')
print('matchup_opponent:', recs['matchup_opponent'])
print('category_deficits count:', len(recs['category_deficits']))
for r in recs['recommendations']:
    ap = r['add_player']
    print(f"Add: {ap['name']} need_score={ap['need_score']} cat_keys={list(ap['category_contributions'].keys())}")

with open('postman_collections/responses/api_fantasy_decisions_20260422_145543.json') as f:
    decisions = json.load(f)

print('\n=== DECISIONS AS_OF_DATE ===')
as_of_dates = set()
for d in decisions.get('decisions', []):
    as_of_dates.add(d['decision']['as_of_date'])
print('Unique as_of_dates:', as_of_dates)

dt_counts = {}
for d in decisions.get('decisions', []):
    dt = d['decision']['decision_type']
    dt_counts[dt] = dt_counts.get(dt, 0) + 1
print('Decision type counts:', dt_counts)
