import json
with open('.tmp_audit_0425/api_fantasy_scoreboard.json') as f:
    d = json.load(f)
rows = d.get('rows', [])
for r in rows:
    print(f"{r['category']:>6}: my={r['my_current']:>6} opp={r['opp_current']:>6} proj={r['my_projected_final']:>6} status={r['status']}")
