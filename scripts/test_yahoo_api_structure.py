"""
Test Yahoo API response structure to debug parsing issues.
"""
import requests
import json
import os

access_token = os.getenv('YAHOO_ACCESS_TOKEN')

if not access_token:
    print("ERROR: YAHOO_ACCESS_TOKEN not found in environment")
    exit(1)

headers = {'Authorization': f'Bearer {access_token}'}

print("=" * 70)
print("TESTING YAHOO API ROSTER ENDPOINT STRUCTURE")
print("=" * 70)
print()

print(f"Access token (first 20 chars): {access_token[:20]}...")
print()

# Test rosters endpoint
url = 'https://fantasysports.yahooapis.com/fantasy/v2/league/469.l.72586/teams/roster'
print(f"URL: {url}")
print()

response = requests.get(url, headers=headers, params={'format': 'json'}, timeout=10)
print(f"Status Code: {response.status_code}")
print()

if response.status_code == 200:
    data = response.json()
    print("[SUCCESS] Response received successfully")
    print()

    # Navigate the structure
    fantasy_content = data.get('fantasy_content', {})
    print(f"fantasy_content type: {type(fantasy_content).__name__}")
    print(f"fantasy_content keys: {list(fantasy_content.keys())}")
    print()

    league_value = fantasy_content.get('league')
    print(f"league value type: {type(league_value).__name__}")

    # Yahoo API returns 'league' as a list containing the actual league data
    if isinstance(league_value, list) and len(league_value) > 0:
        league_data = league_value[0]
        print(f"league is a list with {len(league_value)} elements")
        print(f"Using first element as league_data")
    elif isinstance(league_value, dict):
        league_data = league_value
        print(f"league is a dict")
    else:
        league_data = {}
        print(f"league has unexpected type")

    print(f"league_data keys: {list(league_data.keys()) if isinstance(league_data, dict) else 'N/A'}")
    print()

    teams_data = league_data.get('teams', {}) if isinstance(league_data, dict) else {}
    print(f"teams type: {type(teams_data).__name__}")
    print(f"teams keys: {list(teams_data.keys()) if isinstance(teams_data, dict) else 'N/A (not a dict)'}")
    print()

    if isinstance(teams_data, dict):
        team_list = teams_data.get('team', [])
        print(f"team value type: {type(team_list).__name__}")

        if isinstance(team_list, list):
            print(f"Number of teams: {len(team_list)}")
            print()

            if len(team_list) > 0:
                first_team = team_list[0]
                print(f"First team type: {type(first_team).__name__}")

                if isinstance(first_team, list):
                    print(f"First team is a list with {len(first_team)} elements")
                    print()

                    # Look for roster in the team
                    for i, item in enumerate(first_team):
                        print(f"Item {i} type: {type(item).__name__}", end="")
                        if isinstance(item, dict):
                            print(f", keys: {list(item.keys())}")
                        else:
                            print()
                elif isinstance(first_team, dict):
                    print(f"First team is a dict with keys: {list(first_team.keys())}")

                    # Look for roster
                    if 'roster' in first_team:
                        roster = first_team['roster']
                        print(f"Found roster! Type: {type(roster).__name__}")
                        if isinstance(roster, dict):
                            print(f"Roster keys (first 10): {list(roster.keys())[:10]}")

                            # Show a sample player entry
                            for key in list(roster.keys())[:5]:
                                if key.isdigit():
                                    print(f"Sample player under key '{key}':")
                                    print(f"  Type: {type(roster[key]).__name__}")
                                    if isinstance(roster[key], dict):
                                        print(f"  Keys: {list(roster[key].keys())}")
                                    break

    print()
    print("=" * 70)
    print("FULL RESPONSE (first 2000 chars)")
    print("=" * 70)
    print(json.dumps(data, indent=2)[:2000])
else:
    print(f"[ERROR] Request failed")
    print(f"Response: {response.text[:500]}")
