import json
with open('.tmp_audit_0425/api_fantasy_waiver_recommendations.json') as f:
    d = json.load(f)

print('opponent:', d.get('matchup_opponent'))
print('recommendations count:', len(d.get('recommendations', [])))

for rec in d.get('recommendations', [])[:5]:
    add = rec.get('add_player', {})
    print(f"  Add: {add.get('name')} need_score={add.get('need_score')} drop={rec.get('drop_player_name')} gain={rec.get('win_prob_gain')} mcmc={rec.get('mcmc_enabled')}")
