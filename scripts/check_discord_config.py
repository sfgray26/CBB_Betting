#!/usr/bin/env python3
"""
Check Discord configuration and test notification capability.
"""

import os
import sys

sys.path.insert(0, '/root/.openclaw/workspace/CBB_Betting')

print("=" * 60)
print("🔍 DISCORD CONFIGURATION CHECK")
print("=" * 60)

# Check environment variables
print("\n1. Environment Variables:")
discord_token = os.getenv("DISCORD_BOT_TOKEN")
discord_channel = os.getenv("DISCORD_CHANNEL_ID", "1477436117426110615")

if discord_token:
    print(f"   ✅ DISCORD_BOT_TOKEN: {discord_token[:10]}... (set)")
else:
    print("   ❌ DISCORD_BOT_TOKEN: NOT SET")
    print("      → Discord notifications will be skipped")
    
print(f"   ℹ️  DISCORD_CHANNEL_ID: {discord_channel}")

# Try to import and check discord_notifier
print("\n2. Discord Notifier Module:")
try:
    from backend.services.discord_notifier import _bot_token, _post, send_todays_bets
    token = _bot_token()
    if token:
        print(f"   ✅ Token detected by module: {token[:10]}...")
    else:
        print("   ⚠️  No token detected by module")
        print("      → Check if DISCORD_BOT_TOKEN is set in Railway environment")
except Exception as e:
    print(f"   ❌ Import error: {e}")

# Check Railway-specific variables
print("\n3. Railway Environment Check:")
railway_vars = {k: v for k, v in os.environ.items() if 'DISCORD' in k or 'RAILWAY' in k}
if railway_vars:
    for k, v in railway_vars.items():
        if 'TOKEN' in k or 'KEY' in k or 'SECRET' in k:
            print(f"   {k}: {'[SET]' if v else '[NOT SET]'}")
        else:
            print(f"   {k}: {v}")
else:
    print("   No Railway/Discord variables found in current environment")

print("\n" + "=" * 60)
print("📋 SUMMARY")
print("=" * 60)

if not discord_token:
    print("""
❌ DISCORD_BOT_TOKEN is not set!

To fix:
1. Go to Railway dashboard → your project → Variables
2. Add: DISCORD_BOT_TOKEN = your_discord_bot_token
3. Redeploy the service

Get your token from: https://discord.com/developers/applications
→ Your App → Bot → Reset Token (or copy existing)
    """)
else:
    print("""
✅ DISCORD_BOT_TOKEN is set!

Next steps:
1. The app should send notifications when analysis completes
2. Check Railway logs: railway logs
3. Look for: "Sending Discord embed for X @ Y"
    """)

print("=" * 60)
