import json
with open('.tmp_audit_0425/admin_audit_tables.json') as f:
    d = json.load(f)
for tbl in d.get('table_counts', []):
    print(f"{tbl['table']}: {tbl['count']} rows")
