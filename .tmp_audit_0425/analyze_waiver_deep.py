import json
with open('.tmp_audit_0425/api_fantasy_waiver.json') as f:
    d = json.load(f)

fas = d.get('top_available', [])
zero_fas = [fa for fa in fas if fa.get('need_score', 0) == 0]
print(f'Zero-score FAs: {len(zero_fas)}')

empty_cc = [fa for fa in zero_fas if not fa.get('category_contributions')]
print(f'  With empty category_contributions: {len(empty_cc)}')

from collections import Counter
pos_counts = Counter(fa.get('position', 'unknown') for fa in zero_fas)
print(f'  Positions: {dict(pos_counts)}')

for fa in zero_fas[:5]:
    print(f"    {fa['name']} ({fa['position']}) stats keys: {list(fa.get('stats', {}).keys())}")
