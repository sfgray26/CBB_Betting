#!/usr/bin/env python3
"""
Test script for UAT notifications (Discord & WhatsApp)

This script tests your notification configuration without running the full UAT suite.
Run this after setting up Discord or WhatsApp to verify everything works.

Usage:
    python3 scripts/test_notifications.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def test_discord_webhook():
    """Test Discord webhook notification"""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        print("❌ DISCORD_WEBHOOK_URL not set")
        return False
    
    try:
        import aiohttp
        
        payload = {
            "embeds": [{
                "title": "🔔 Test Notification - CBB Edge UAT",
                "description": "This is a test notification to verify your Discord webhook is working.",
                "color": 3447003,  # Blue
                "fields": [
                    {"name": "🎽 Elite FM Score", "value": "Test: 8.5/10", "inline": True},
                    {"name": "📊 Quant Score", "value": "Test: 8.2/10", "inline": True},
                    {"name": "Status", "value": "✅ Webhook working!", "inline": False}
                ],
                "footer": {"text": "CBB Edge UAT Automation - Test Message"}
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(webhook_url, json=payload) as response:
                if response.status == 204:
                    print("✅ Discord webhook test successful!")
                    print(f"   Webhook: {webhook_url[:50]}...")
                    return True
                else:
                    print(f"❌ Discord webhook failed: HTTP {response.status}")
                    print(f"   Response: {await response.text()}")
                    return False
                    
    except ImportError:
        print("⚠️  aiohttp not installed. Install with: pip install aiohttp")
        return False
    except Exception as e:
        print(f"❌ Discord webhook error: {e}")
        return False


async def test_discord_bot():
    """Test Discord bot notification"""
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    
    if not bot_token:
        print("❌ DISCORD_BOT_TOKEN not set")
        return False
    
    if not channel_id:
        print("❌ DISCORD_CHANNEL_ID not set")
        return False
    
    try:
        import aiohttp
        
        embed = {
            "title": "🔔 Test Notification - CBB Edge UAT",
            "description": "This is a test notification to verify your Discord bot is working.",
            "color": 3447003,
            "fields": [
                {"name": "🎽 Elite FM Score", "value": "Test: 8.5/10", "inline": True},
                {"name": "📊 Quant Score", "value": "Test: 8.2/10", "inline": True},
                {"name": "Status", "value": "✅ Bot working!", "inline": False}
            ],
            "footer": {"text": "CBB Edge UAT Automation - Test Message"}
        }
        
        payload = {
            "content": "🧪 Test notification from CBB Edge UAT",
            "embeds": [embed]
        }
        
        headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    print("✅ Discord bot test successful!")
                    print(f"   Channel ID: {channel_id}")
                    return True
                else:
                    print(f"❌ Discord bot failed: HTTP {response.status}")
                    print(f"   Response: {await response.text()}")
                    return False
                    
    except ImportError:
        print("⚠️  aiohttp not installed. Install with: pip install aiohttp")
        return False
    except Exception as e:
        print(f"❌ Discord bot error: {e}")
        return False


async def test_whatsapp():
    """Test WhatsApp notification via Twilio"""
    webhook_url = os.getenv("WHATSAPP_WEBHOOK_URL")
    
    if not webhook_url:
        print("❌ WHATSAPP_WEBHOOK_URL not set")
        return False
    
    try:
        import aiohttp
        
        message = """🔔 *CBB Edge UAT Test*

This is a test notification.

🎽 Elite FM Score: Test 8.5/10
📊 Quant Score: Test 8.2/10

✅ WhatsApp integration working!"""
        
        # Twilio format
        if "twilio" in webhook_url.lower():
            account_sid = os.getenv("TWILIO_ACCOUNT_SID")
            auth_token = os.getenv("TWILIO_AUTH_TOKEN")
            from_number = os.getenv("WHATSAPP_FROM_NUMBER")
            to_number = os.getenv("WHATSAPP_TO_NUMBER")
            
            if not all([account_sid, auth_token, from_number, to_number]):
                print("❌ Missing Twilio credentials")
                return False
            
            payload = {
                "To": to_number,
                "From": from_number,
                "Body": message
            }
            
            auth = aiohttp.BasicAuth(account_sid, auth_token)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, data=payload, auth=auth) as response:
                    if response.status in [200, 201]:
                        print("✅ WhatsApp (Twilio) test successful!")
                        print(f"   To: {to_number}")
                        return True
                    else:
                        print(f"❌ WhatsApp failed: HTTP {response.status}")
                        print(f"   Response: {await response.text()}")
                        return False
        
        # Generic webhook format
        else:
            to_number = os.getenv("WHATSAPP_TO_NUMBER")
            
            payload = {
                "message": message,
                "phone": to_number,
                "priority": "normal"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload) as response:
                    if response.status in [200, 201, 202]:
                        print("✅ WhatsApp webhook test successful!")
                        return True
                    else:
                        print(f"❌ WhatsApp failed: HTTP {response.status}")
                        return False
                        
    except ImportError:
        print("⚠️  aiohttp not installed. Install with: pip install aiohttp")
        return False
    except Exception as e:
        print(f"❌ WhatsApp error: {e}")
        return False


async def main():
    """Main test function"""
    print("="*60)
    print("CBB Edge UAT - Notification Test")
    print("="*60)
    print()
    
    # Check if .env file exists and load it
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"📁 Loading environment from: {env_file}")
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ.setdefault(key, value.strip('"\''))
    else:
        print("⚠️  No .env file found. Using existing environment variables.")
    
    print()
    
    # Track results
    results = {
        "discord_webhook": False,
        "discord_bot": False,
        "whatsapp": False
    }
    
    # Test Discord Webhook
    print("Testing Discord Webhook...")
    print("-" * 40)
    results["discord_webhook"] = await test_discord_webhook()
    print()
    
    # Test Discord Bot
    print("Testing Discord Bot...")
    print("-" * 40)
    results["discord_bot"] = await test_discord_bot()
    print()
    
    # Test WhatsApp
    print("Testing WhatsApp...")
    print("-" * 40)
    results["whatsapp"] = await test_whatsapp()
    print()
    
    # Summary
    print("="*60)
    print("Test Summary")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    print(f"\nPassed: {passed_tests}/{total_tests}")
    
    for platform, passed in results.items():
        status = "✅" if passed else "❌"
        configured = "Yes" if any([
            os.getenv("DISCORD_WEBHOOK_URL") and platform == "discord_webhook",
            os.getenv("DISCORD_BOT_TOKEN") and platform == "discord_bot",
            os.getenv("WHATSAPP_WEBHOOK_URL") and platform == "whatsapp"
        ]) else "No"
        print(f"  {status} {platform.replace('_', ' ').title()}: {'Pass' if passed else 'Fail'} (Configured: {configured})")
    
    print()
    
    if passed_tests == 0:
        print("⚠️  No notification platforms configured or all tests failed.")
        print("   See docs/UAT_NOTIFICATION_SETUP.md for setup instructions.")
        return 1
    elif passed_tests < total_tests:
        print("⚠️  Some notification platforms not working.")
        print("   Check the error messages above.")
        return 0  # Partial success is OK
    else:
        print("🎉 All configured notification platforms working!")
        print("   You're ready to run UAT with notifications.")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
