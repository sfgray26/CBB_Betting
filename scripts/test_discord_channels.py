#!/usr/bin/env python3
"""
Test script for Discord multi-channel setup.

Sends a test message to each configured channel to verify routing works.
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)

# Try to load .env file
try:
    from dotenv import load_dotenv
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"Loaded .env from: {env_path}")
except ImportError:
    pass  # python-dotenv not installed, assume env vars are set

from backend.services.discord_notifier import (
    send_to_channel,
    route_notification,
    CHANNEL_MAP,
    _get_channel_id,
    _bot_token,
)


def test_channel_config():
    """Print channel configuration status."""
    print("=" * 60)
    print("Discord Channel Configuration Test")
    print("=" * 60)
    print()
    
    # Check bot token
    token = _bot_token()
    if token:
        print(f"[OK] DISCORD_BOT_TOKEN: Configured (len={len(token)})")
    else:
        print("[MISSING] DISCORD_BOT_TOKEN: NOT SET - All notifications will be skipped")
        print("  (Uncomment DISCORD_BOT_TOKEN in .env file to enable)")
    
    print()
    
    print()
    print("Channel Configuration:")
    print("-" * 60)
    
    all_configured = True
    for channel_name, env_var in CHANNEL_MAP.items():
        channel_id = _get_channel_id(channel_name)
        status = "[OK]" if channel_id else "[MISSING]"
        id_display = channel_id if channel_id else "NOT SET"
        print(f"{status} {channel_name:25} {env_var:35} = {id_display}")
        if not channel_id:
            all_configured = False
    
    print()
    return all_configured


def send_test_messages(dry_run=False):
    """Send test messages to all channels."""
    print("=" * 60)
    print("Sending Test Messages")
    print("=" * 60)
    print()
    
    if dry_run:
        print("[DRY RUN MODE] No actual messages sent")
        print()
    
    results = []
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    # Test CBB channels
    test_cases = [
        ("cbb-bets", "[BET] Test bet notification", {"color": 0x2ECC71}),
        ("cbb-morning-brief", "[BRIEF] Test morning brief", {"color": 0x3498DB}),
        ("cbb-alerts", "[ALERT] Test line movement alert", {"color": 0xF1C40F}),
        ("cbb-tournament", "🏆 Test tournament update", {"color": 0x9B59B6}),
        
        ("fantasy-lineups", "[LINEUP] Test lineup recommendation", {"color": 0x3498DB}),
        ("fantasy-waivers", "[WAIVER] Test waiver suggestion", {"color": 0x2ECC71}),
        ("fantasy-news", "📰 Test injury alert", {"color": 0xE67E22}),
        ("fantasy-draft", "📝 Test draft pick", {"color": 0x1ABC9C}),
        
        ("openclaw-briefs", "🧠 Test research brief", {"color": 0x3498DB}),
        ("openclaw-escalations", "[ESCALATION] Test high-stakes escalation", {"color": 0xE74C3C}),
        ("openclaw-health", "🛰️ Test health report", {"color": 0x2ECC71}),
        
        ("system-errors", "[ERROR] Test system error", {"color": 0xE74C3C}),
        ("system-logs", "📝 Test routine log entry"),
        ("data-alerts", "[DATA] Test data degradation", {"color": 0xE67E22}),
        
        ("general", "[GENERAL] Test general message"),
    ]
    
    for channel_name, description, *embed_opts in test_cases:
        print(f"Testing #{channel_name}...", end=" ")
        
        channel_id = _get_channel_id(channel_name)
        if not channel_id:
            print("SKIPPED (not configured)")
            results.append((channel_name, "SKIPPED", "Not configured"))
            continue
        
        if dry_run:
            print(f"DRY RUN (would send to {channel_id})")
            results.append((channel_name, "DRY RUN", channel_id))
            continue
        
        # Build test embed
        embed = {
            "title": f"Test: {description}",
            "description": f"This is a test message sent at {timestamp}",
            "color": embed_opts[0].get("color", 0x95A5A6) if embed_opts else 0x95A5A6,
            "fields": [
                {"name": "Channel", "value": f"#{channel_name}", "inline": True},
                {"name": "Status", "value": "Working", "inline": True},
            ],
            "footer": {"text": "CBB Edge Channel Test"},
        }
        
        try:
            success = send_to_channel(channel_name, embed=embed)
            if success:
                print("[SENT]")
                results.append((channel_name, "SUCCESS", channel_id))
            else:
                print("[FAILED]")
                results.append((channel_name, "FAILED", "API error or no token"))
        except Exception as e:
            print(f"[ERROR]: {e}")
            results.append((channel_name, "ERROR", str(e)))
    
    print()
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Discord channel configuration")
    parser.add_argument("--dry-run", "-d", action="store_true", 
                        help="Show what would be sent without actually sending")
    parser.add_argument("--config-only", "-c", action="store_true",
                        help="Only show configuration, don't send messages")
    args = parser.parse_args()
    
    # Test configuration
    config_ok = test_channel_config()
    
    if args.config_only:
        sys.exit(0 if config_ok else 1)
    
    # Send test messages
    results = send_test_messages(dry_run=args.dry_run)
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    success_count = sum(1 for _, status, _ in results if status == "SUCCESS")
    failed_count = sum(1 for _, status, _ in results if status in ("FAILED", "ERROR"))
    skipped_count = sum(1 for _, status, _ in results if status == "SKIPPED")
    dry_count = sum(1 for _, status, _ in results if status == "DRY RUN")
    
    print(f"Success:   {success_count}")
    print(f"Failed:    {failed_count}")
    print(f"Skipped:   {skipped_count}")
    if args.dry_run:
        print(f"Dry Run:   {dry_count}")
    
    print()
    
    if failed_count > 0:
        print("Failed channels:")
        for channel, status, detail in results:
            if status in ("FAILED", "ERROR"):
                print(f"  - #{channel}: {detail}")
        sys.exit(1)
    elif success_count > 0:
        print("All test messages sent successfully!")
        sys.exit(0)
    elif dry_count > 0:
        print("Dry run complete - no messages actually sent")
        sys.exit(0)
    else:
        print("[WARNING] No channels configured or all skipped")
        sys.exit(1)


if __name__ == "__main__":
    main()
