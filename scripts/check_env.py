#!/usr/bin/env python3
"""
Diagnostic script to check Discord environment variables.
"""

import os
import sys

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load from project root
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        print(f"Loading .env from: {env_path}")
        load_dotenv(env_path)
        print("[OK] .env file loaded")
    else:
        print(f"[MISSING] .env file not found at: {env_path}")
except ImportError:
    print("[WARNING] python-dotenv not installed, checking raw environment")

print()
print("=" * 60)
print("Discord Environment Variables Check")
print("=" * 60)
print()

# Check all expected variables
vars_to_check = [
    "DISCORD_BOT_TOKEN",
    "DISCORD_CHANNEL_CBB_BETS",
    "DISCORD_CHANNEL_CBB_BRIEF",
    "DISCORD_CHANNEL_CBB_ALERTS",
    "DISCORD_CHANNEL_CBB_TOURNAMENT",
    "DISCORD_CHANNEL_FANTASY_LINEUPS",
    "DISCORD_CHANNEL_FANTASY_WAIVERS",
    "DISCORD_CHANNEL_FANTASY_NEWS",
    "DISCORD_CHANNEL_FANTASY_DRAFT",
    "DISCORD_CHANNEL_OPENCLAW_BRIEFS",
    "DISCORD_CHANNEL_OPENCLAW_ESCALATIONS",
    "DISCORD_CHANNEL_OPENCLAW_HEALTH",
    "DISCORD_CHANNEL_SYSTEM_ERRORS",
    "DISCORD_CHANNEL_SYSTEM_LOGS",
    "DISCORD_CHANNEL_DATA_ALERTS",
    "DISCORD_CHANNEL_GENERAL",
    "DISCORD_CHANNEL_ADMIN_COMMANDS",
]

total = len(vars_to_check)
configured = 0

for var in vars_to_check:
    value = os.getenv(var)
    if value:
        # Show first/last 5 chars of token for security
        if "TOKEN" in var and len(value) > 20:
            display = f"{value[:5]}...{value[-5:]} (len={len(value)})"
        else:
            display = value
        print(f"✅ {var:40} = {display}")
        configured += 1
    else:
        print(f"❌ {var:40} NOT SET")

print()
print("=" * 60)
print(f"Summary: {configured}/{total} variables configured")
print("=" * 60)

if configured == 0:
    print()
    print("[WARNING] No variables found! Possible causes:")
    print("   1. .env file not in project root")
    print("   2. .env file has syntax errors")
    print("   3. Variables not exported (need 'export VAR=value' on Unix)")
    print("   4. Running from wrong directory")
    print()
    print("Current working directory:", os.getcwd())
    sys.exit(1)
elif configured < total:
    print()
    print(f"[WARNING] Missing {total - configured} variables")
    sys.exit(1)
else:
    print()
    print("[OK] All variables configured!")
    sys.exit(0)
