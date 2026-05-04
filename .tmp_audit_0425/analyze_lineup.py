import json
with open('.tmp_audit_0425/api_fantasy_lineup.json') as f:
    d = json.load(f)
print('starters:', len(d.get('starters', [])))
print('bench:', len(d.get('bench', [])))
print('unrostered:', len(d.get('unrostered', [])))
print('total_score:', d.get('total_lineup_score'))
for s in d.get('starters', []):
    print(f"  {s['player_name']}: {s['assigned_slot']} score={s['lineup_score']}")
