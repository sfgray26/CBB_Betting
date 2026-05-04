import json
with open('.tmp_audit_0425/api_fantasy_waiver.json') as f:
    d = json.load(f)
print('matchup_opponent:', d.get('matchup_opponent'))
print('category_deficits count:', len(d.get('category_deficits', [])))

fas = d.get('top_available', [])
print('FA count:', len(fas))

scores = {}
for fa in fas:
    s = fa.get('need_score', 0)
    if s == 0:
        scores['zero'] = scores.get('zero', 0) + 1
    elif s > 0:
        scores['positive'] = scores.get('positive', 0) + 1
    else:
        scores['negative'] = scores.get('negative', 0) + 1
print('need_score distribution:', scores)

sorted_fas = sorted(fas, key=lambda x: x.get('need_score', 0), reverse=True)
for fa in sorted_fas[:8]:
    cc = list(fa.get('category_contributions', {}).keys())
    print(f"  {fa['name']}: need_score={fa['need_score']} pos={fa['position']} contributions={cc}")
