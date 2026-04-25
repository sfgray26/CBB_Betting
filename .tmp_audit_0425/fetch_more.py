import requests, json
url = 'https://fantasy-app-production-5079.up.railway.app'
key = 'j01F3n2sSzbhi-jNAEULNkgzFqRXgOl2FuIDgKRoyfg'
headers = {'X-API-Key': key}

# Lineup
r = requests.get(f'{url}/api/fantasy/lineup/2026-04-25', headers=headers, timeout=20)
with open('.tmp_audit_0425/api_fantasy_lineup.json', 'w') as f:
    f.write(r.text)
print(f'lineup: {r.status_code}')

# Audit tables
r = requests.get(f'{url}/admin/audit-tables', headers=headers, timeout=20)
with open('.tmp_audit_0425/admin_audit_tables.json', 'w') as f:
    f.write(r.text)
print(f'audit_tables: {r.status_code}')
