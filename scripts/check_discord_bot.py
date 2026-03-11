#!/usr/bin/env python3
"""
Discord Bot Diagnostics

Checks if the bot token is valid and can access the server/channels.
"""

import os
import sys
import requests

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    print("ERROR: DISCORD_BOT_TOKEN not found in environment")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bot {TOKEN}",
    "Content-Type": "application/json",
}

print("=" * 60)
print("Discord Bot Diagnostics")
print("=" * 60)
print()

# Test 1: Check token validity (get bot user info)
print("Test 1: Validating token...")
resp = requests.get("https://discord.com/api/v10/users/@me", headers=HEADERS, timeout=10)
if resp.status_code == 200:
    bot_data = resp.json()
    print(f"  [OK] Token is valid")
    print(f"  Bot Name: {bot_data.get('username')}#{bot_data.get('discriminator', '0')}")
    print(f"  Bot ID: {bot_data.get('id')}")
else:
    print(f"  [FAILED] Token invalid: {resp.status_code}")
    print(f"  Response: {resp.text[:200]}")
    print()
    print("=" * 60)
    print("TROUBLESHOOTING:")
    print("=" * 60)
    print("1. Go to https://discord.com/developers/applications")
    print("2. Select your bot application")
    print("3. Go to 'Bot' section")
    print("4. Click 'Reset Token' to generate a new one")
    print("5. Copy the new token to your .env file")
    print()
    print("If token is correct but still failing:")
    print("- Bot may have been removed from the server")
    print("- Use OAuth2 URL Generator to re-add the bot")
    sys.exit(1)

print()

# Test 2: List guilds (servers) the bot is in
print("Test 2: Checking server membership...")
resp = requests.get("https://discord.com/api/v10/users/@me/guilds", headers=HEADERS, timeout=10)
if resp.status_code == 200:
    guilds = resp.json()
    print(f"  [OK] Bot is in {len(guilds)} server(s):")
    for guild in guilds:
        print(f"    - {guild.get('name')} (ID: {guild.get('id')})")
else:
    print(f"  [FAILED] Could not list guilds: {resp.status_code}")

print()

# Test 3: Check specific channels
CHANNELS_TO_TEST = [
    ("cbb-bets", os.getenv("DISCORD_CHANNEL_CBB_BETS")),
    ("general", os.getenv("DISCORD_CHANNEL_GENERAL")),
]

print("Test 3: Checking channel access...")
for name, channel_id in CHANNELS_TO_TEST:
    if not channel_id:
        print(f"  [SKIPPED] {name}: No channel ID configured")
        continue
    
    url = f"https://discord.com/api/v10/channels/{channel_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    
    if resp.status_code == 200:
        channel_data = resp.json()
        print(f"  [OK] #{name}: Accessible")
        print(f"       Name: {channel_data.get('name')}")
        print(f"       Type: {channel_data.get('type')}")
    elif resp.status_code == 403:
        print(f"  [FAILED] #{name}: Forbidden (no permission)")
        print(f"       The bot needs to be added to the server or given channel access")
    elif resp.status_code == 404:
        print(f"  [FAILED] #{name}: Channel not found")
        print(f"       Check the channel ID is correct")
    else:
        print(f"  [FAILED] #{name}: HTTP {resp.status_code}")
        print(f"       {resp.text[:100]}")

print()

# Test 4: Check bot permissions
print("Test 4: Checking bot permissions...")
print("  Required permissions for this bot:")
print("    - Send Messages")
print("    - Embed Links")
print("    - Attach Files (optional)")
print("    - Mention @everyone, @here, and All Roles (for @admin)")
print()
print("  To check/fix permissions:")
print("    1. Go to your Discord server")
print("    2. Right-click bot name → Roles")
print("    3. Ensure bot role has 'Send Messages' in all channels")

print()
print("=" * 60)
print("Summary")
print("=" * 60)
print("If all tests show [OK] but messages still fail, the token may")
print("have been regenerated. Generate a new one in the Discord")
print("Developer Portal and update your .env file.")
