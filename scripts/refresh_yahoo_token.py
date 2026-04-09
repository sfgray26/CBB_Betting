"""
Refresh Yahoo access token and display the full token for Railway update.
"""

import requests
import os

client_id = os.getenv('YAHOO_CLIENT_ID')
client_secret = os.getenv('YAHOO_CLIENT_SECRET')
refresh_token = os.getenv('YAHOO_REFRESH_TOKEN')

print("Refreshing Yahoo access token...")
print()

response = requests.post('https://api.login.yahoo.com/oauth2/get_token', data={
    'client_id': client_id,
    'client_secret': client_secret,
    'refresh_token': refresh_token,
    'grant_type': 'refresh_token'
}, timeout=10)

if response.status_code == 200:
    tokens = response.json()
    access_token = tokens.get('access_token')

    print("✅ Token refresh successful!")
    print()
    print("FULL ACCESS TOKEN (copy this):")
    print(access_token)
    print()
    print("Railway update command:")
    print(f'railway variables set YAHOO_ACCESS_TOKEN="{access_token}"')
    print()
    print("Token expires in:", tokens.get('expires_in', 'unknown'), "seconds")
else:
    print(f"❌ Token refresh failed: {response.status_code}")
    print(f"Error: {response.text}")
