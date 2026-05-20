#!/usr/bin/env python3
"""
Test UAT Discord integration using existing discord_notifier service

This script tests that UAT notifications will work with your existing
Hermes Discord setup.

Usage:
    python scripts/test_uat_discord.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_discord_connection():
    """Test connection to Discord using existing service"""
    print("="*60)
    print("Testing UAT Discord Integration")
    print("="*60)
    print()
    
    # Check environment variables
    print("🔍 Checking Discord configuration...")
    print("-"*60)
    
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    system_logs_channel = os.getenv("DISCORD_CHANNEL_SYSTEM_LOGS")
    general_channel = os.getenv("DISCORD_CHANNEL_GENERAL")
    
    if bot_token:
        print(f"✅ DISCORD_BOT_TOKEN: Set ({len(bot_token)} chars)")
    else:
        print("❌ DISCORD_BOT_TOKEN: Not set")
        return False
    
    if system_logs_channel:
        print(f"✅ DISCORD_CHANNEL_SYSTEM_LOGS: {system_logs_channel}")
    else:
        print("⚠️  DISCORD_CHANNEL_SYSTEM_LOGS: Not set")
    
    if general_channel:
        print(f"✅ DISCORD_CHANNEL_GENERAL: {general_channel}")
    else:
        print("⚠️  DISCORD_CHANNEL_GENERAL: Not set")
    
    print()
    
    # Test importing the discord service
    print("🔍 Testing discord_notifier import...")
    print("-"*60)
    
    try:
        from backend.services.discord_notifier import send_to_channel, _bot_token
        print("✅ discord_notifier imported successfully")
        
        # Check if bot token is accessible
        token = _bot_token()
        if token:
            print(f"✅ Bot token accessible ({len(token)} chars)")
        else:
            print("❌ Bot token not accessible")
            return False
            
    except ImportError as e:
        print(f"❌ Failed to import discord_notifier: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    print()
    
    # Send test notification
    print("📢 Sending test UAT notification...")
    print("-"*60)
    
    try:
        # Build test embed (same format as real UAT)
        embed = {
            "title": "🧪 Test UAT Report - CBB Edge",
            "description": "This is a test notification to verify UAT Discord integration",
            "color": 0x3498DB,  # Blue for test
            "fields": [
                {"name": "🎽 Elite FM Score", "value": "Test: 8.5/10", "inline": True},
                {"name": "📊 Quant Score", "value": "Test: 8.2/10", "inline": True},
                {"name": "⚠️ Critical Issues", "value": "0", "inline": True},
                {"name": "📁 Full Report", "value": "Test notification only", "inline": False}
            ],
            "footer": {"text": "CBB Edge UAT Automation - Test"}
        }
        
        # Try system-logs first, then general
        print("   Attempting to send to #system-logs...")
        success = send_to_channel("system-logs", embed=embed)
        
        if not success:
            print("   Falling back to #general...")
            success = send_to_channel("general", embed=embed)
        
        if success:
            print("✅ Test notification sent successfully!")
            print()
            print("🎉 Discord integration is working correctly!")
            print("   UAT reports will be sent to your Discord channels.")
            return True
        else:
            print("❌ Failed to send notification")
            print()
            print("💡 Troubleshooting:")
            print("   1. Verify bot is in the server")
            print("   2. Check bot has permission to send messages")
            print("   3. Verify channel IDs are correct")
            return False
            
    except Exception as e:
        print(f"❌ Error sending notification: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    # Load .env file if present
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"📁 Loading environment from: {env_file}")
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ.setdefault(key, value.strip('"\''))
    
    print()
    
    # Run test
    success = test_discord_connection()
    
    print()
    print("="*60)
    if success:
        print("✅ TEST PASSED - Discord integration ready")
        print("="*60)
        return 0
    else:
        print("❌ TEST FAILED - Check configuration")
        print("="*60)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
