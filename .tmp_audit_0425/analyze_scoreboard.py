import json
with open('.tmp_audit_0425/api_fantasy_scoreboard.json') as f:
    d = json.load(f)
print('opponent_name:', d.get('opponent_name'))
print('overall_win_probability:', d.get('overall_win_probability'))
print('categories_won/lost/tied:', d.get('categories_won'), d.get('categories_lost'), d.get('categories_tied'))

rows = d.get('rows', [])
print('row count:', len(rows))

non_zero_cats = []
for r in rows:
    if r.get('my_current', 0) != 0 or r.get('opp_current', 0) != 0:
        non_zero_cats.append(r['category'])
print('categories with non-zero current:', non_zero_cats)

for r in rows[:6]:
    print(f"  {r['category']}: my={r['my_current']} opp={r['opp_current']} proj_margin={r['projected_margin']} status={r['status']}")
