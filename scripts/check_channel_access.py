#!/usr/bin/env python3
"""
Check which Discord channels the bot can actually access.

This helps diagnose permission issues.
"""

import os
import sys
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    print("ERROR: DISCORD_BOT_TOKEN not found")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bot {TOKEN}",
    "Content-Type": "application/json",
}

# Test both working (bets) and new channels
CHANNELS = [
    ("cbb-bets (WORKING)", os.getenv("DISCORD_CHANNEL_CBB_BETS")),
    ("cbb-morning-brief", os.getenv("DISCORD_CHANNEL_CBB_BRIEF")),
    ("cbb-alerts", os.getenv("DISCORD_CHANNEL_CBB_ALERTS")),
    ("cbb-tournament", os.getenv("DISCORD_CHANNEL_CBB_TOURNAMENT")),
    ("fantasy-lineups", os.getenv("DISCORD_CHANNEL_FANTASY_LINEUPS")),
    ("fantasy-waivers", os.getenv("DISCORD_CHANNEL_FANTASY_WAIVERS")),
    ("fantasy-news", os.getenv("DISCORD_CHANNEL_FANTASY_NEWS")),
    ("general", os.getenv("DISCORD_CHANNEL_GENERAL")),
]

print("=" * 70)
print("Discord Channel Access Check")
print("=" * 70)
print()

for name, channel_id in CHANNELS:
    if not channel_id:
        print(f"[NOT CONFIGURED] {name}")
        continue
    
    url = f"https://discord.com/api/v10/channels/{channel_id}"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    
    if resp.status_code == 200:
        data = resp.json()
        guild_id = data.get('guild_id', 'unknown')
        channel_name = data.get('name', 'unknown')
        print(f"[OK - ACCESSIBLE] {name}")
        print(f"    Discord Name: #{channel_name}")
        print(f"    Channel ID: {channel_id}")
        print(f"    Server ID: {guild_id}")
    elif resp.status_code == 403:
        print(f"[FORBIDDEN - NO ACCESS] {name}")
        print(f"    Channel ID: {channel_id}")
        print(f"    The bot cannot see this channel.")
        print(f"    FIX: Add the bot to this channel or check permissions.")
    elif resp.status_code == 404:
        print(f"[NOT FOUND] {name}")
        print(f"    Channel ID: {channel_id}")
        print(f"    This channel ID doesn't exist.")
        print(f"    FIX: Double-check the channel ID in Discord.")
    elif resp.status_code == 401:
        print(f"[UNAUTHORIZED] {name}")
        print(f"    Token is invalid or expired.")
        print(f"    If one channel works but others don't, this is weird.")
    else:
        print(f"[ERROR {resp.status_code}] {name}")
        print(f"    Response: {resp.text[:100]}")
    
    print()

print("=" * 70)
print("TROUBLESHOOTING STEPS:")
print("=" * 70)
print()
print("If you see [FORBIDDEN] for new channels but [OK] for cbb-bets:")
print()
print("1. In Discord, go to each new channel")
print("2. Click the gear icon (Channel Settings)")
print("3. Go to 'Permissions' tab")
print("4. Click '+' to add a role/member")
print("5. Add your bot")
print("6. Grant these permissions:")
print("   - View Channel")
print("   - Send Messages")
print("   - Embed Links")
print("   - Mention @everyone, @here, and All Roles")
print()
print("ALTERNATIVE - Quick fix:")
print("1. In Discord server settings → Roles")
print("2. Find your bot's role")
print("3. Enable 'Administrator' permission (nuclear option)")
print("   OR enable these server-wide:")
print("   - View Channels")
print("   - Send Messages")
print("   - Embed Links")
