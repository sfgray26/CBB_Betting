#!/usr/bin/env python
"""Diagnose Discord configuration issues."""
import os
import requests

# Load .env manually
if os.path.exists('.env'):
    with open('.env', 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                # Remove quotes if present
                val = val.strip('"\'')
                os.environ.setdefault(key, val)

print("=" * 60)
print("DISCORD CONFIGURATION DIAGNOSTIC")
print("=" * 60)

# Check token
token = os.environ.get('DISCORD_BOT_TOKEN')
if not token:
    print("ERROR: DISCORD_BOT_TOKEN not found in .env")
    exit(1)
print(f"Token: {token[:15]}...{token[-5:]}")

# Check OpenClaw config
print("\n--- OpenClaw Config ---")
config_path = '.openclaw/config.yaml'
if os.path.exists(config_path):
    with open(config_path, encoding='utf-8') as f:
        content = f.read()
        if 'discord_enabled: true' in content:
            print("OpenClaw Discord: ENABLED")
        elif 'discord_enabled: false' in content:
            print("OpenClaw Discord: DISABLED (set to true in .openclaw/config.yaml)")
        else:
            print("OpenClaw Discord: UNKNOWN setting")
else:
    print("OpenClaw config: NOT FOUND")

# Test token validity
print("\n--- Token Validation ---")
resp = requests.get(
    'https://discord.com/api/v10/users/@me',
    headers={'Authorization': f'Bot {token}'},
    timeout=30
)
print(f"API Response: HTTP {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    print(f"  Bot Name: {data['username']}#{data.get('discriminator', '0')}")
    print(f"  Bot ID: {data['id']}")
    print(f"  Verified: {data.get('verified', False)}")
    print("  Token is VALID")
elif resp.status_code == 401:
    print("  ERROR: Token is INVALID or EXPIRED")
    print("  Solution: Generate new bot token at https://discord.com/developers/applications")
else:
    print(f"  Unexpected response: {resp.text[:200]}")

# Test channel access
print("\n--- Channel Access Test ---")
channels = {k: v for k, v in os.environ.items() if k.startswith('DISCORD_CHANNEL_')}
if not channels:
    print("WARNING: No DISCORD_CHANNEL_* variables set")
else:
    for name, cid in sorted(channels.items()):
        if not cid:
            print(f"  {name}: EMPTY")
            continue
        resp = requests.get(
            f'https://discord.com/api/v10/channels/{cid}',
            headers={'Authorization': f'Bot {token}'},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            print(f"  {name}: OK ({data.get('name', 'unknown')})")
        elif resp.status_code == 401:
            print(f"  {name}: AUTH FAILED (401) - Bot may have been removed from server")
        elif resp.status_code == 403:
            print(f"  {name}: FORBIDDEN (403) - Bot lacks permissions")
        elif resp.status_code == 404:
            print(f"  {name}: NOT FOUND (404) - Channel doesn't exist")
        else:
            print(f"  {name}: ERROR {resp.status_code}")

print("\n" + "=" * 60)
