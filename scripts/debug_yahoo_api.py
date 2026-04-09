"""
Debug Yahoo API authentication and response structure.
Run with: railway run -- python scripts/debug_yahoo_api.py
"""

import requests
import os
import json

print("=" * 70)
print("YAHOO API DEBUG - April 9, 2026")
print("=" * 70)
print()

access_token = os.getenv('YAHOO_ACCESS_TOKEN')
refresh_token = os.getenv('YAHOO_REFRESH_TOKEN')
client_id = os.getenv('YAHOO_CLIENT_ID')
client_secret = os.getenv('YAHOO_CLIENT_SECRET')

print(f"Access token (first 20 chars): {access_token[:20]}...")
print(f"Refresh token (first 20 chars): {refresh_token[:20]}...")
print(f"Client ID (first 10 chars): {client_id[:10]}...")
print()

headers = {'Authorization': f'Bearer {access_token}'}

# TEST 1: User endpoint
print("TEST 1: User Endpoint")
print("-" * 70)
r = requests.get('https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1',
                headers=headers, params={'format': 'json'})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"✅ User authenticated: {data.get('users', [{}])[0].get('profile', {}).get('guid', 'N/A')}")
else:
    print(f"❌ User endpoint FAILED")
    print(f"Response: {r.text[:200]}")
print()

# TEST 2: League endpoint
print("TEST 2: League Endpoint (469.l.72586)")
print("-" * 70)
r = requests.get('https://fantasysports.yahooapis.com/fantasy/v2/league/469.l.72586',
                headers=headers, params={'format': 'json'})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"✅ League accessible")
    print(f"League name: {data.get('fantasy_content', {}).get('league', {}).get('name', 'N/A')}")
    print(f"Has teams: {'team' in str(data).lower()}")
else:
    print(f"❌ League endpoint FAILED")
    print(f"Response: {r.text[:200]}")
print()

# TEST 3: League rosters endpoint
print("TEST 3: League Rosters Endpoint")
print("-" * 70)
r = requests.get('https://fantasysports.yahooapis.com/fantasy/v2/league/469.l.72586/teams/roster',
                headers=headers, params={'format': 'json'})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"✅ Rosters endpoint accessible")

    # Count teams
    fantasy_content = data.get('fantasy_content', {})
    league_data = fantasy_content.get('league', {})
    teams_data = league_data.get('teams', {})

    if isinstance(teams_data, dict):
        team_count = len(teams_data.get('team', []))
    else:
        team_count = "unknown"

    print(f"Teams found: {team_count}")

    # Look for players in response
    response_str = json.dumps(data)
    has_players = 'player' in response_str.lower()
    print(f"Response contains 'player': {has_players}")

    # Sample first 500 chars
    print(f"Response preview: {response_str[:500]}")
else:
    print(f"❌ Rosters endpoint FAILED")
    print(f"Response: {r.text[:200]}")
print()

# TEST 4: Token refresh if needed
if "401" in str([r.status_code]):
    print("TEST 4: Token Refresh (401 detected)")
    print("-" * 70)
    r = requests.post('https://api.login.yahoo.com/oauth2/get_token', data={
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    })
    print(f"Refresh status: {r.status_code}")
    if r.status_code == 200:
        new_token = r.json().get('access_token')
        print(f"✅ New access token received: {new_token[:20]}...")
        print(f"Update command: railway variables set YAHOO_ACCESS_TOKEN=\"{new_token}\"")
    else:
        print(f"❌ Token refresh FAILED")
        print(f"Response: {r.text[:200]}")

print()
print("=" * 70)
